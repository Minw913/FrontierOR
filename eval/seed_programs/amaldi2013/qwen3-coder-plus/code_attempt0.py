import argparse
import json
import numpy as np
from scipy.spatial.distance import pdist, squareform
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def compute_normal_and_offset(points_subset):
    """
    Compute a hyperplane that fits a subset of points optimally.
    Returns the unit-norm normal vector w and scalar offset w0.
    """
    if len(points_subset) == 0:
        return np.array([0.0] * len(points_subset[0])), 0.0
    if len(points_subset) == 1:
        # For a single point, any hyperplane passing through it works
        center = points_subset[0]
        w = np.zeros(len(center))
        w[0] = 1.0  # Standard basis vector
        w0 = np.dot(w, center)
        return w, w0
        
    points_array = np.array(points_subset)
    # Get centroid
    centroid = np.mean(points_array, axis=0)
    
    # Center the points
    centered_points = points_array - centroid
    
    # Apply SVD
    _, _, vh = np.linalg.svd(centered_points)
    
    # Take the direction that corresponds to the smallest singular value 
    # (the direction orthogonal to the best fit subspace)
    w = vh[-1]
    
    # Normalize
    norm_w = np.linalg.norm(w)
    if norm_w > 0:
        w /= norm_w
    else:
        w = np.zeros_like(w)
        w[0] = 1.0  # Default direction if all zero
    
    # Calculate offset w0 so that the hyperplane passes through the centroid
    w0 = np.dot(w, centroid)
    
    return w, w0


def solve_hyperplane_covering(instance_data, time_limit, log_path=None):
    n = instance_data["n"]
    d = instance_data["d"]
    epsilon = instance_data["epsilon"]
    K = instance_data["upper_bound_K"]
    points = np.array(instance_data["points"])
    big_M = instance_data["big_M"]

    # Initialize logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    # Create model
    model = gp.Model("hyperplane_covering")
    model.setParam('TimeLimit', time_limit)
    
    # Variables
    # binary variable: y[k] = 1 if hyperplane k is used
    y = model.addVars(K, vtype=GRB.BINARY, name="y")
    
    # binary variable: x[i,k] = 1 if point i is assigned to hyperplane k
    x = model.addVars(n, K, vtype=GRB.BINARY, name="x")
    
    # continuous variables for hyperplane parameters
    w = model.addVars(d, K, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="w")
    w0 = model.addVars(K, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="w0")
    
    model.update()

    # Constraints
    # (1) Every point must be assigned to at least one hyperplane
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, k] for k in range(K)) >= 1, name=f"assign_point_{i}")

    # (2) Point i can only be assigned to hyperplane k if hyperplane k is selected
    for i in range(n):
        for k in range(K):
            model.addConstr(x[i, k] <= y[k], name=f"use_hyperplane_{i}_{k}")

    # (3) Distance constraint: distance from point i to hyperplane k is at most epsilon if assigned
    # |w_k · p_i - w0_k| ≤ epsilon if x[i,k] = 1
    # We linearize this using a new variable z_ik representing |w_k · p_i - w0_k|
    z = {}
    for i in range(n):
        for k in range(K):
            z_var = model.addVar(lb=0.0, ub=big_M, name=f"z_{i}_{k}")
            z[i, k] = z_var
            # Add constraints equivalent to z_ik >= |dot(w_k, p_i) - w0_k| when x_ik = 1
            pt = points[i]
            model.addGenConstrAbs(z[i, k], w[0, k]*pt[0] + sum(w[j, k]*pt[j] for j in range(1, d)) - w0[k])

    # Now add the constraint z_ik <= epsilon if x_ik = 1
    for i in range(n):
        for k in range(K):
            model.addConstr(z[i, k] <= epsilon + (1 - x[i, k]) * big_M, name=f"epsilon_distance_{i}_{k}")

    # (4) Unit norm constraint on w_k (this is non-linear - we handle differently)
    # Since this makes the problem non-convex, we can try different relaxations.
    # One way is to normalize during post-processing or use a MILP relaxation.
    # Here we will add bounds and rely on solver heuristics, then fix later
    
    # Relaxation: add bounds to prevent numerical issues
    BIG_VAL = 5.0
    for k in range(K):
        for j in range(d):
            model.addConstr(w[j, k] >= -BIG_VAL, name=f"w_lb_{j}_{k}")
            model.addConstr(w[j, k] <= BIG_VAL, name=f"w_ub_{j}_{k}")

    # Objective: minimize number of hyperplanes used
    model.setObjective(gp.quicksum(y[k] for k in range(K)), GRB.MINIMIZE)
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    # Solve
    model.optimize(callback)

    # Extract solution
    objective_value = int(model.ObjVal) if model.SolCount > 0 else float('inf')

    if model.SolCount == 0:
        # Return infeasible solution
        return {
            "objective_value": -1,
            "hyperplanes": [],
            "point_assignments": [[] for _ in range(n)]
        }

    # Build the actual hyperplanes based on the assignments
    point_assignments = [[] for _ in range(n)]
    for i in range(n):
        for k in range(K):
            if x[i, k].x > 0.5:
                point_assignments[i].append(len(point_assignments[i]))  # This is a placeholder index; we adjust below
    
    # Rebuild point assignements and hyperplanes properly
    used_hyperplanes = []
    hyperplane_map = {}  # Maps old k to new index in results
    for k in range(K):
        if y[k].x > 0.5:
            # Collect points assigned to this hyperplane
            points_assigned_to_this = [i for i in range(n) if x[i, k].x > 0.5]
            
            if len(points_assigned_to_this) == 0:
                continue
                
            # Compute an optimal hyperplane for these points
            points_subset = [points[i] for i in points_assigned_to_this]
            w_opt, w0_opt = compute_normal_and_offset(points_subset)
            
            hp_info = {
                "w": w_opt.tolist(),
                "w0": w0_opt,
                "assigned_points": points_assigned_to_this
            }
            
            used_idx = len(used_hyperplanes)
            hyperplane_map[k] = used_idx
            used_hyperplanes.append(hp_info)

    # Update point assignments to reflect correct hyperplane indices
    final_point_assignments = [[] for _ in range(n)]
    for i in range(n):
        for k in range(K):
            if x[i, k].x > 0.5 and k in hyperplane_map:
                final_point_assignments[i].append(hyperplane_map[k])

    result = {
        "objective_value": len(used_hyperplanes),
        "hyperplanes": used_hyperplanes,
        "point_assignments": final_point_assignments
    }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Solve
    solution = solve_hyperplane_covering(instance_data, args.time_limit, args.log_path)

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()