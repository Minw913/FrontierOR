import json
import argparse
import time
import numpy as np
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    d = instance['d']
    epsilon = instance['epsilon']
    big_M = instance['big_M']
    K_max = instance['upper_bound_K']
    points = np.array(instance['points'], dtype=np.float64)

    if n == 0:
        solution = {
            "objective_value": 0.0,
            "hyperplanes": [],
            "point_assignments": []
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        return

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()
    best_obj = float('inf')
    best_solution = None

    # Compute pairwise distances for clustering heuristic
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist_matrix[i, j] = np.linalg.norm(points[i] - points[j])
            dist_matrix[j, i] = dist_matrix[i, j]

    # Greedy clustering: group points that can be covered by one hyperplane
    # Two points can be on same hyperplane if their distance <= 2*epsilon in projection
    # But for small epsilon we approximate by direct distance threshold
    max_dist_for_cluster = 2.0 * epsilon * 1.5  # conservative

    # Simple greedy set cover like approach for initial solution
    uncovered = set(range(n))
    initial_hyperplanes = []
    initial_assignments = [[] for _ in range(n)]
    hyperplane_id = 0

    while uncovered and hyperplane_id < K_max:
        # Find largest group of points that are close
        best_count = 0
        best_center = None
        best_group = []

        for candidate in list(uncovered):
            group = [candidate]
            for other in uncovered:
                if other != candidate and dist_matrix[candidate][other] <= max_dist_for_cluster:
                    group.append(other)
            if len(group) > best_count:
                best_count = len(group)
                best_group = group
                best_center = candidate

        if best_count == 0:
            # Assign singletons
            for p in list(uncovered):
                initial_hyperplanes.append({
                    "w": [float(1.0 if i == 0 else 0.0) for i in range(d)],
                    "w0": -float(points[p][0]),
                    "assigned_points": [int(p)]
                })
                initial_assignments[p].append(hyperplane_id)
                hyperplane_id += 1
                uncovered.remove(p)
            break

        # Create a hyperplane roughly orthogonal to principal direction or simple
        if len(best_group) == 1:
            p0 = points[best_group[0]]
            w = np.zeros(d, dtype=np.float64)
            w[0] = 1.0
            w0 = -p0[0]
        else:
            # Use the centroid and a normal from first two points or zero
            group_pts = points[best_group]
            centroid = group_pts.mean(axis=0)
            if len(best_group) >= 2:
                vec = group_pts[1] - group_pts[0]
                norm = np.linalg.norm(vec)
                if norm > 1e-8:
                    w = vec / norm
                else:
                    w = np.zeros(d, dtype=np.float64)
                    w[0] = 1.0
            else:
                w = np.zeros(d, dtype=np.float64)
                w[0] = 1.0
            w0 = -np.dot(w, centroid)

        initial_hyperplanes.append({
            "w": [float(x) for x in w.tolist()],
            "w0": float(w0),
            "assigned_points": [int(x) for x in best_group]
        })
        for p_idx in best_group:
            initial_assignments[p_idx].append(hyperplane_id)
            if p_idx in uncovered:
                uncovered.remove(p_idx)
        hyperplane_id += 1

    current_obj = len(initial_hyperplanes)
    if current_obj < best_obj:
        best_obj = current_obj
        best_solution = {
            "objective_value": float(current_obj),
            "hyperplanes": initial_hyperplanes,
            "point_assignments": [[int(h) for h in lst] for lst in initial_assignments]
        }
        if logger:
            logger.log(float(current_obj))

    # Try to improve with MIP if time permits and instance is small
    elapsed = time.time() - start_time
    if n <= 30 and d <= 5 and K_max <= 15 and elapsed < args.time_limit * 0.7:
        try:
            # Build MIP model for exact cover with hyperplane fitting
            m = Model("HyperplaneCover")
            m.setParam('TimeLimit', max(1, args.time_limit - int(elapsed) - 5))
            m.setParam('Threads', 1)
            m.setParam('MIPGap', 0.01)
            m.setParam('LogToConsole', 0)

            # Variables
            use = m.addVars(K_max, vtype=GRB.BINARY, name="use")
            assign = m.addVars(n, K_max, vtype=GRB.BINARY, name="assign")
            w = m.addVars(K_max, d, lb=-1.0, ub=1.0, name="w")
            w0 = m.addVars(K_max, lb=-big_M*2, ub=big_M*2, name="w0")
            # Auxiliary for |w·x + w0|
            abs_dist = m.addVars(n, K_max, name="abs_dist")

            # Objective
            m.setObjective(quicksum(use[k] for k in range(K_max)), GRB.MINIMIZE)

            # Constraints
            # Each point assigned to at least one hyperplane
            for i in range(n):
                m.addConstr(quicksum(assign[i, k] for k in range(K_max)) >= 1)

            # Assignment only if used
            for i in range(n):
                for k in range(K_max):
                    m.addConstr(assign[i, k] <= use[k])

            # Norm of w approximately 1 (we relax unit norm to [-1,1] box and penalize weakly)
            for k in range(K_max):
                m.addConstr(quicksum(w[k, j] * w[k, j] for j in range(d)) >= 0.5)

            # Distance constraints
            for i in range(n):
                for k in range(K_max):
                    dot = quicksum(w[k, j] * points[i][j] for j in range(d))
                    m.addConstr(dot + w0[k] <= abs_dist[i, k] + big_M * (1 - assign[i, k]))
                    m.addConstr(- (dot + w0[k]) <= abs_dist[i, k] + big_M * (1 - assign[i, k]))
                    m.addConstr(abs_dist[i, k] <= epsilon + big_M * (1 - assign[i, k]))

            # Symmetry breaking: use hyperplanes in order
            for k in range(1, K_max):
                m.addConstr(use[k] <= use[k-1])

            m.optimize()

            if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
                obj_val = m.objVal
                if obj_val < best_obj - 1e-6:
                    # Extract solution
                    hyperplanes = []
                    point_assignments = [[] for _ in range(n)]
                    for k in range(K_max):
                        if use[k].X > 0.5:
                            w_vec = [w[k, j].X for j in range(d)]
                            norm = np.linalg.norm(w_vec)
                            if norm > 1e-6:
                                w_vec = [x / norm for x in w_vec]
                            else:
                                w_vec = [1.0 if j == 0 else 0.0 for j in range(d)]
                            w0_val = w0[k].X
                            assigned = []
                            for i in range(n):
                                if assign[i, k].X > 0.5:
                                    assigned.append(i)
                                    point_assignments[i].append(len(hyperplanes))
                            hyperplanes.append({
                                "w": [float(x) for x in w_vec],
                                "w0": float(w0_val),
                                "assigned_points": [int(x) for x in assigned]
                            })
                    best_solution = {
                        "objective_value": float(len(hyperplanes)),
                        "hyperplanes": hyperplanes,
                        "point_assignments": [[int(h) for h in lst] for lst in point_assignments]
                    }
                    best_obj = len(hyperplanes)
                    if logger:
                        logger.log(float(best_obj))
        except Exception:
            # MIP failed, keep heuristic
            pass

    # Final fallback: ensure we have a feasible solution
    if best_solution is None:
        # Assign each point to its own hyperplane
        hyperplanes = []
        point_assignments = [[] for _ in range(n)]
        for i in range(n):
            w = [1.0 if j == 0 else 0.0 for j in range(d)]
            w0 = -float(points[i][0])
            hyperplanes.append({
                "w": w,
                "w0": w0,
                "assigned_points": [i]
            })
            point_assignments[i].append(i)
        best_solution = {
            "objective_value": float(n),
            "hyperplanes": hyperplanes,
            "point_assignments": point_assignments
        }

    # Write output
    with open(args.solution_path, 'w') as f:
        json.dump(best_solution, f, indent=2)


if __name__ == "__main__":
    main()