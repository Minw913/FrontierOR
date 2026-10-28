import argparse
import json
import math
import os
import random
import time

from solution_logger import SolutionLogger


def evaluate_assignment(assignment, A, B, C):
    total_a = 0
    total_b = 0
    total_c = 0
    for i, j in enumerate(assignment):
        total_a += A[i][j]
        total_b += B[i][j]
        total_c += C[i][j]
    return total_a * total_b + total_c, total_a, total_b, total_c


def greedy_assignment(cost):
    """Fallback assignment construction when SciPy is unavailable."""
    n = len(cost)
    assignment = [-1] * n
    free_rows = set(range(n))
    free_cols = set(range(n))

    while free_rows:
        best = None
        for i in free_rows:
            row_best_col = min(free_cols, key=lambda j: cost[i][j])
            candidate = (cost[i][row_best_col], i, row_best_col)
            if best is None or candidate < best:
                best = candidate

        _, i, j = best
        assignment[i] = j
        free_rows.remove(i)
        free_cols.remove(j)

    return assignment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    n = int(instance["n"])
    A = instance["A"]
    B = instance["B"]
    C = instance["C"]

    if n == 0:
        solution = {"objective_value": 0.0, "assignment": []}
        if logger:
            logger.log_solution(0.0, solution)
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(solution, f)
        return

    best_assignment = list(range(n))
    best_obj, best_a, best_b, best_c = evaluate_assignment(
        best_assignment, A, B, C
    )

    def make_solution(obj, assignment):
        return {
            "objective_value": float(obj),
            "assignment": [int(x) for x in assignment],
        }

    if logger:
        logger.log_solution(
            float(best_obj), make_solution(best_obj, best_assignment)
        )

    def register(assignment):
        nonlocal best_assignment, best_obj, best_a, best_b, best_c
        obj, total_a, total_b, total_c = evaluate_assignment(
            assignment, A, B, C
        )
        if obj < best_obj:
            best_obj = obj
            best_a = total_a
            best_b = total_b
            best_c = total_c
            best_assignment = list(assignment)
            if logger:
                logger.log_solution(
                    float(obj), make_solution(obj, best_assignment)
                )
            return True
        return False

    # Reserve most of the available time for the global Gurobi search.
    heuristic_budget = min(
        8.0,
        max(0.0, 0.20 * max(0, args.time_limit))
    )
    heuristic_deadline = min(deadline, time.monotonic() + heuristic_budget)

    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        A_np = np.asarray(A, dtype=np.float64)
        B_np = np.asarray(B, dtype=np.float64)
        C_np = np.asarray(C, dtype=np.float64)

        def solve_linear(cost):
            rows, cols = linear_sum_assignment(cost)
            assignment = [-1] * n
            for i, j in zip(rows.tolist(), cols.tolist()):
                assignment[i] = j
            return assignment

        scale_a = max(1.0, float(np.max(np.abs(A_np))))
        scale_b = max(1.0, float(np.max(np.abs(B_np))))
        scale_c = max(1.0, float(np.max(np.abs(C_np))))

        normalized_a = A_np / scale_a
        normalized_b = B_np / scale_b
        normalized_c = C_np / scale_c

        candidate_costs = [
            normalized_c,
            normalized_a,
            -normalized_a,
            normalized_b,
            -normalized_b,
            normalized_a + normalized_b,
            normalized_a - normalized_b,
            -normalized_a + normalized_b,
            -normalized_a - normalized_b,
            normalized_a + normalized_b + normalized_c,
            normalized_a - normalized_b + normalized_c,
            -normalized_a + normalized_b + normalized_c,
            -normalized_a - normalized_b + normalized_c,
        ]

        rng = random.Random(0)
        for _ in range(6):
            wa = rng.uniform(-2.0, 2.0)
            wb = rng.uniform(-2.0, 2.0)
            wc = rng.uniform(0.0, 2.0)
            candidate_costs.append(
                wa * normalized_a + wb * normalized_b + wc * normalized_c
            )

    except Exception:
        np = None
        linear_sum_assignment = None

        def solve_linear(cost):
            return greedy_assignment(cost)

        candidate_costs = [C, A, B]

    def pair_swap_local_search(initial_assignment):
        assignment = list(initial_assignment)
        obj, total_a, total_b, total_c = evaluate_assignment(
            assignment, A, B, C
        )
        register(assignment)

        max_iterations = min(100, max(5, 2 * n))
        for _ in range(max_iterations):
            if time.monotonic() >= heuristic_deadline:
                break

            best_swap = None
            best_swap_obj = obj
            best_values = None

            for i in range(n - 1):
                if (i & 7) == 0 and time.monotonic() >= heuristic_deadline:
                    break

                pi = assignment[i]
                for k in range(i + 1, n):
                    pk = assignment[k]

                    delta_a = (
                        A[i][pk] + A[k][pi] - A[i][pi] - A[k][pk]
                    )
                    delta_b = (
                        B[i][pk] + B[k][pi] - B[i][pi] - B[k][pk]
                    )
                    delta_c = (
                        C[i][pk] + C[k][pi] - C[i][pi] - C[k][pk]
                    )

                    new_a = total_a + delta_a
                    new_b = total_b + delta_b
                    new_c = total_c + delta_c
                    new_obj = new_a * new_b + new_c

                    if new_obj < best_swap_obj:
                        best_swap_obj = new_obj
                        best_swap = (i, k)
                        best_values = (new_a, new_b, new_c)

            if best_swap is None:
                break

            i, k = best_swap
            assignment[i], assignment[k] = assignment[k], assignment[i]
            total_a, total_b, total_c = best_values
            obj = best_swap_obj
            register(assignment)

        return assignment, total_a, total_b, total_c

    # Improve the identity assignment first.
    if time.monotonic() < heuristic_deadline:
        pair_swap_local_search(best_assignment)

    # Generate assignment solutions from several scalarizations, then improve
    # them under the true nonlinear objective.
    seen = {tuple(best_assignment)}
    for cost in candidate_costs:
        if time.monotonic() >= heuristic_deadline:
            break

        try:
            assignment = solve_linear(cost)
        except Exception:
            continue

        key = tuple(assignment)
        if key in seen:
            continue
        seen.add(key)

        assignment, total_a, total_b, _ = pair_swap_local_search(assignment)

        # Repeatedly minimize the first-order objective
        # b*A + a*B + C around the current assignment.
        for _ in range(3):
            if time.monotonic() >= heuristic_deadline:
                break

            if np is not None:
                gradient_cost = total_b * A_np + total_a * B_np + C_np
            else:
                gradient_cost = [
                    [
                        total_b * A[i][j] + total_a * B[i][j] + C[i][j]
                        for j in range(n)
                    ]
                    for i in range(n)
                ]

            try:
                next_assignment = solve_linear(gradient_cost)
            except Exception:
                break

            key = tuple(next_assignment)
            if key in seen:
                break
            seen.add(key)

            assignment, total_a, total_b, _ = pair_swap_local_search(
                next_assignment
            )

    # Globally optimize the compact nonconvex MIQP with Gurobi.
    remaining = deadline - time.monotonic()
    if remaining > 0.05:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            model = gp.Model("multiplicative_assignment")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.NonConvex = 2
            model.Params.TimeLimit = max(0.01, remaining)

            x = model.addVars(n, n, vtype=GRB.BINARY, name="x")

            model.addConstrs(
                (gp.quicksum(x[i, j] for j in range(n)) == 1
                 for i in range(n)),
                name="entity"
            )
            model.addConstrs(
                (gp.quicksum(x[i, j] for i in range(n)) == 1
                 for j in range(n)),
                name="position"
            )

            # Safe aggregate bounds. The row-wise bounds are inexpensive and
            # valid for every permutation assignment.
            a_lb = sum(min(A[i]) for i in range(n))
            a_ub = sum(max(A[i]) for i in range(n))
            b_lb = sum(min(B[i]) for i in range(n))
            b_ub = sum(max(B[i]) for i in range(n))

            aggregate_a = model.addVar(
                lb=float(a_lb), ub=float(a_ub), name="aggregate_a"
            )
            aggregate_b = model.addVar(
                lb=float(b_lb), ub=float(b_ub), name="aggregate_b"
            )

            a_expr = gp.quicksum(
                A[i][j] * x[i, j] for i in range(n) for j in range(n)
            )
            b_expr = gp.quicksum(
                B[i][j] * x[i, j] for i in range(n) for j in range(n)
            )
            c_expr = gp.quicksum(
                C[i][j] * x[i, j] for i in range(n) for j in range(n)
            )

            model.addConstr(aggregate_a == a_expr)
            model.addConstr(aggregate_b == b_expr)
            model.setObjective(
                aggregate_a * aggregate_b + c_expr, GRB.MINIMIZE
            )

            # Warm-start from the best heuristic assignment.
            for i in range(n):
                assigned = best_assignment[i]
                for j in range(n):
                    x[i, j].Start = 1.0 if j == assigned else 0.0
            aggregate_a.Start = float(best_a)
            aggregate_b.Start = float(best_b)

            flat_x = [x[i, j] for i in range(n) for j in range(n)]

            def incumbent_callback(cb_model, where):
                if where != GRB.Callback.MIPSOL:
                    return
                try:
                    values = cb_model.cbGetSolution(flat_x)
                    assignment = [-1] * n
                    offset = 0
                    for i in range(n):
                        row_values = values[offset:offset + n]
                        assignment[i] = max(
                            range(n), key=lambda j: row_values[j]
                        )
                        offset += n
                    register(assignment)
                except Exception:
                    # Logging or extraction failure must not terminate the
                    # optimization run.
                    return

            model.optimize(incumbent_callback)

            if model.SolCount > 0:
                assignment = []
                for i in range(n):
                    assignment.append(
                        max(range(n), key=lambda j: x[i, j].X)
                    )
                register(assignment)

        except Exception:
            # The heuristic incumbent remains a valid output if Gurobi cannot
            # initialize or the nonlinear model encounters an environment
            # limitation.
            pass

    final_solution = make_solution(best_obj, best_assignment)

    solution_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if solution_dir:
        os.makedirs(solution_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f)


if __name__ == "__main__":
    main()