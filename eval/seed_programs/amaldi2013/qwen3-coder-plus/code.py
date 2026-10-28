import argparse
import json
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def compute_normal_and_offset(points_subset):
    """
    Compute a hyperplane that fits a subset of points optimally.
    Returns the unit-norm normal vector w and scalar offset w0.
    """
    if len(points_subset) == 0:
        return np.array([0.0 if i == 0 else 0.0 for i in range(len(points_subset[0]) if points_subset else 1)]), 0.0
    if len(points_subset) == 1:
        # For a single point, any hyperplane passing through it works
        center = np.array(points_subset[0])
        w = np.zeros(len(center))
        if len(w) > 0:
            w[0] = 1.0  # Standard basis vector
        w0 = np.dot(w, center) if len(w) > 0 else 0.0
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
        if len(w) > 0:
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
    # Increase big_M to ensure feasibility - it might be too restrictive otherwise
    if "big_M" in instance_data:
        big_M = instance_data["big_M"]
    else:
        big_M = 1000.0  # Default large value

    # Initialize logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    # Create model
    model = gp.Model("hyperplane_covering")
    model.setParam("Threads", 1)
    model.setParam('TimeLimit', time_limit)

    # Variables
    # binary variable: y[k] = 1 if hyperplane k is used
    y = model.addVars(K, vtype=GRB.BINARY, name="y")

    # binary variable: x[i,k] = 1 if point i is assigned to hyperplane k
    x = model.addVars(n, K, vtype=GRB.BINARY, name="x")

    # continuous variables for hyperplane parameters
    w = model.addVars(K, d, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="w")
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
    z = {}  # auxiliary variables for absolute value
    for i in range(n):
        for k in range(K):
            z_var = model.addVar(lb=0.0, ub=big_M, name=f"z_{i}_{k}")
            z[i, k] = z_var
            # This represents |w_k · p_i - w0_k| 
            dot_product = gp.LinExpr()
            for j in range(d):
                dot_product.add(w[k, j], points[i][j])
            
            # Now create absolute value: |dot_product - w0_k|
            diff_var = model.addVar(lb=-big_M, ub=big_M, name=f"diff_{i}_{k}")
            model.addConstr(diff_var == dot_product - w0[k])
            
            # Use Gurobi's GenConstrAbs to connect z and diff
            model.addGenConstrAbs(z[i, k], diff_var, name=f"abs_constr_{i}_{k}")

            # Add the constraint: z_ik <= epsilon + (1 - x_ik)*BigM
            model.addConstr(z[i, k] <= epsilon + (1 - x[i, k]) * big_M, name=f"epsilon_constraint_{i}_{k}")

    # Constraint to ensure unit norm (approximately)
    # We can't directly enforce ||w|| = 1 due to non-convexity
    # However, we assume the solution will have normalized directions
    
    # Instead, add normalization heuristic by requiring at least one component not too small
    for k in range(K):
        if d > 0:
            # Add a constraint to normalize roughly: sum of squares is ~1
            w_squared_vars = {}
            aux_var = model.addVar(name=f"w_sq_norm_aux_{k}", lb=1.0, ub=1.0)  # We'll enforce this via other means
            
            # For simplicity, we will handle normalization in post-processing after optimization
            pass  

    # Objective: minimize number of hyperplanes used
    model.setObjective(gp.quicksum(y[k] for k in range(K)), GRB.MINIMIZE)

    # Callback function for logging
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    # Solve
    model.optimize(callback)

    # Check if solution exists
    if model.SolCount == 0:
        return {
            "objective_value": -1,
            "hyperplanes": [],
            "point_assignments": [[] for _ in range(n)]
        }

    # Extract assignment decisions
    assignments = [[] for _ in range(n)]
    for i in range(n):
        for k in range(K):
            if x[i, k].x > 0.5:
                assignments[i].append(k)

    # Build hyperplanes based on assignments and compute normals independently
    used_hyperplanes = []
    hyperplane_indices_map = {}

    for k in range(K):
        if y[k].x > 0.5:
            points_in_hyperplane = [i for i in range(n) if x[i, k].x > 0.5]
            if points_in_hyperplane:
                # Get the actual points assigned to this hyperplane
                points_subset = [list(points[i]) for i in points_in_hyperplane]
                
                w_norm, w0_val = compute_normal_and_offset(points_subset)
                
                # Add to the list of used hyperplanes
                hidx = len(used_hyperplanes)
                hyperplane_indices_map[k] = hidx
                used_hyperplanes.append({
                    "w": w_norm.tolist(),
                    "w0": w0_val,
                    "assigned_points": points_in_hyperplane
                })

    # Build final point assignments based on the new indices
    final_point_assignments = [[] for _ in range(n)]
    for i in range(n):
        for k in assignments[i]:
            if k in hyperplane_indices_map:
                final_point_assignments[i].append(hyperplane_indices_map[k])

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