import argparse
import json
import time
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    p = instance['p']
    cost_matrices = instance['cost_matrices']
    reference_point = instance['reference_point']
    search_direction_lambda = instance['search_direction_lambda']

    start_time = time.time()

    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model("multi_criteria_assignment")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', 0)

    # Decision variables
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    t = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="t")

    # Assignment constraints: each task assigned to exactly one agent
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1, name=f"task_{i}")

    # Assignment constraints: each agent assigned exactly one task
    for j in range(n):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n)) == 1, name=f"agent_{j}")

    # Worst-case deviation constraints for each criterion
    for k in range(p):
        total_cost_k = gp.quicksum(
            cost_matrices[k][i][j] * x[i, j]
            for i in range(n) for j in range(n)
        )
        model.addConstr(
            search_direction_lambda[k] * (total_cost_k - reference_point[k]) <= t,
            name=f"criterion_{k}"
        )

    # Objective: minimize worst-case weighted deviation
    model.setObjective(t, GRB.MINIMIZE)

    # Callback to log incumbent solutions
    best_obj = [float('inf')]

    def incumbent_callback(model_cb, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model_cb.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0] - 1e-9:
                best_obj[0] = obj_val
                if logger:
                    try:
                        x_vals = model_cb.cbGetSolution(x)
                        assignment = [0] * n
                        assignment_matrix = [[0] * n for _ in range(n)]
                        for i in range(n):
                            for j in range(n):
                                if x_vals[i, j] > 0.5:
                                    assignment[i] = j
                                    assignment_matrix[i][j] = 1
                        sol_dict = {
                            "objective_value": obj_val,
                            "assignment": assignment,
                            "assignment_matrix": assignment_matrix
                        }
                        logger.log_solution(obj_val, sol_dict)
                    except Exception:
                        logger.log(obj_val)

    model.optimize(incumbent_callback)

    elapsed = time.time() - start_time

    # Extract solution
    solution = None

    if model.SolCount > 0:
        obj_val = t.X
        assignment = [0] * n
        assignment_matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if x[i, j].X > 0.5:
                    assignment[i] = j
                    assignment_matrix[i][j] = 1

        solution = {
            "objective_value": float(obj_val),
            "assignment": assignment,
            "assignment_matrix": assignment_matrix
        }

        if logger:
            logger.log_solution(float(obj_val), solution)
    else:
        # Fallback: solve LP relaxation and round, or use Hungarian for each criterion
        # Try a simple greedy/heuristic fallback
        # Use the assignment that minimizes the first criterion as a starting point
        # Then try to improve via local search

        # First, try solving individual assignment problems and pick the best
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        best_fallback_obj = float('inf')
        best_fallback_assignment = list(range(n))

        # Try each criterion's optimal assignment
        for k in range(p):
            cost_arr = np.array(cost_matrices[k], dtype=float)
            row_ind, col_ind = linear_sum_assignment(cost_arr)
            assignment_k = [0] * n
            for i, j in zip(row_ind, col_ind):
                assignment_k[i] = j

            # Evaluate this assignment on the objective
            obj_k = max(
                search_direction_lambda[c] * (
                    sum(cost_matrices[c][i][assignment_k[i]] for i in range(n)) - reference_point[c]
                )
                for c in range(p)
            )

            if obj_k < best_fallback_obj:
                best_fallback_obj = obj_k
                best_fallback_assignment = assignment_k[:]

        # Also try a combined cost assignment
        combined_cost = np.zeros((n, n))
        for k in range(p):
            for i in range(n):
                for j in range(n):
                    combined_cost[i][j] += search_direction_lambda[k] * cost_matrices[k][i][j]

        row_ind, col_ind = linear_sum_assignment(combined_cost)
        assignment_combined = [0] * n
        for i, j in zip(row_ind, col_ind):
            assignment_combined[i] = j

        obj_combined = max(
            search_direction_lambda[c] * (
                sum(cost_matrices[c][i][assignment_combined[i]] for i in range(n)) - reference_point[c]
            )
            for c in range(p)
        )

        if obj_combined < best_fallback_obj:
            best_fallback_obj = obj_combined
            best_fallback_assignment = assignment_combined[:]

        # Local search improvement
        remaining_time = args.time_limit - (time.time() - start_time)
        if remaining_time > 1:
            improved = True
            while improved and (time.time() - start_time) < args.time_limit - 0.5:
                improved = False
                for i in range(n):
                    for j in range(n):
                        if best_fallback_assignment[i] == j:
                            continue
                        # Try swapping: task i gets agent j, find task that has agent j
                        i2 = -1
                        for ii in range(n):
                            if best_fallback_assignment[ii] == j:
                                i2 = ii
                                break
                        if i2 == -1:
                            continue
                        # Swap: task i -> agent j, task i2 -> old agent of task i
                        old_j = best_fallback_assignment[i]
                        best_fallback_assignment[i] = j
                        best_fallback_assignment[i2] = old_j

                        obj_new = max(
                            search_direction_lambda[c] * (
                                sum(cost_matrices[c][ii][best_fallback_assignment[ii]] for ii in range(n)) - reference_point[c]
                            )
                            for c in range(p)
                        )

                        if obj_new < best_fallback_obj - 1e-9:
                            best_fallback_obj = obj_new
                            improved = True
                            if logger:
                                assignment_matrix_fb = [[0] * n for _ in range(n)]
                                for ii in range(n):
                                    assignment_matrix_fb[ii][best_fallback_assignment[ii]] = 1
                                sol_dict = {
                                    "objective_value": float(best_fallback_obj),
                                    "assignment": best_fallback_assignment[:],
                                    "assignment_matrix": assignment_matrix_fb
                                }
                                logger.log_solution(float(best_fallback_obj), sol_dict)
                        else:
                            # Revert
                            best_fallback_assignment[i] = old_j
                            best_fallback_assignment[i2] = j

        assignment_matrix_fb = [[0] * n for _ in range(n)]
        for i in range(n):
            assignment_matrix_fb[i][best_fallback_assignment[i]] = 1

        solution = {
            "objective_value": float(best_fallback_obj),
            "assignment": best_fallback_assignment,
            "assignment_matrix": assignment_matrix_fb
        }

        if logger:
            logger.log_solution(float(best_fallback_obj), solution)

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()