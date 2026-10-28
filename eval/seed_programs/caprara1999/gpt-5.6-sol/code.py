import argparse
import json
import os
import time

from solution_logger import SolutionLogger


def make_solution(objective, selected):
    return {
        "objective_value": float(objective),
        "selected_items": [int(v) for v in selected],
    }


def evaluate_solution(selected, diagonal, edges):
    value = 0
    for i, chosen in enumerate(selected):
        if chosen:
            value += diagonal[i]
    for i, j, coefficient in edges:
        if selected[i] and selected[j]:
            value += coefficient
    return value


def construct_solution(order, weights, capacity):
    selected = [0] * len(weights)
    total_weight = 0
    for i in order:
        weight = weights[i]
        if total_weight + weight <= capacity:
            selected[i] = 1
            total_weight += weight
    return selected, total_weight


def write_solution(path, solution):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(solution, output_file, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as instance_file:
        data = json.load(instance_file)

    n = int(data["n"])
    capacity = int(data["capacity"])
    weights = [int(v) for v in data["weights"]]
    matrix = data["profit_matrix"]

    diagonal = [int(matrix[i][i]) for i in range(n)]
    row_interaction_profit = [0] * n
    edges = []

    # The coefficient of x_i*x_j is P[i][j] + P[j][i]. For a symmetric
    # matrix this is twice the supplied off-diagonal value.
    for i in range(n):
        row_i = matrix[i]
        for j in range(i + 1, n):
            coefficient = int(row_i[j]) + int(matrix[j][i])
            if coefficient > 0:
                edges.append((i, j, coefficient))
                row_interaction_profit[i] += coefficient
                row_interaction_profit[j] += coefficient

    del matrix
    data = None

    best_selected = [0] * n
    best_objective = 0
    best_weight = 0

    if logger:
        logger.log_solution(
            best_objective,
            make_solution(best_objective, best_selected),
        )

    # Produce several deterministic primal solutions before starting Gurobi.
    # Potential-profit orderings help identify groups with valuable synergies,
    # while individual-profit and low-weight orders provide diversification.
    orderings = [
        sorted(
            range(n),
            key=lambda i: (
                diagonal[i] / weights[i],
                diagonal[i],
                -weights[i],
            ),
            reverse=True,
        ),
        sorted(
            range(n),
            key=lambda i: (
                (diagonal[i] + row_interaction_profit[i]) / weights[i],
                diagonal[i] + row_interaction_profit[i],
                -weights[i],
            ),
            reverse=True,
        ),
        sorted(
            range(n),
            key=lambda i: (
                (2 * diagonal[i] + row_interaction_profit[i]) / weights[i],
                2 * diagonal[i] + row_interaction_profit[i],
                -weights[i],
            ),
            reverse=True,
        ),
        sorted(
            range(n),
            key=lambda i: (
                diagonal[i] + row_interaction_profit[i],
                -weights[i],
            ),
            reverse=True,
        ),
        sorted(
            range(n),
            key=lambda i: (
                -weights[i],
                diagonal[i] + row_interaction_profit[i],
            ),
            reverse=True,
        ),
    ]

    for order in orderings:
        if time.monotonic() >= deadline:
            break
        candidate, candidate_weight = construct_solution(order, weights, capacity)
        candidate_objective = evaluate_solution(candidate, diagonal, edges)

        if (
            candidate_objective > best_objective
            or (
                candidate_objective == best_objective
                and candidate_weight > best_weight
            )
        ):
            strictly_improved = candidate_objective > best_objective
            best_selected = candidate
            best_objective = candidate_objective
            best_weight = candidate_weight

            if strictly_improved and logger:
                logger.log_solution(
                    best_objective,
                    make_solution(best_objective, best_selected),
                )

    remaining_time = deadline - time.monotonic()

    if remaining_time > 0.01:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            model = gp.Model("quadratic_knapsack")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1

            x = model.addVars(n, vtype=GRB.BINARY, name="x")
            m = len(edges)

            # Continuous auxiliaries are sufficient: with nonnegative objective
            # coefficients, maximization drives y_ij to min(x_i, x_j).
            y = model.addVars(m, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y")

            model.addConstr(
                gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity,
                name="capacity",
            )

            model.addConstrs(
                (y[k] <= x[edges[k][0]] for k in range(m)),
                name="pair_left",
            )
            model.addConstrs(
                (y[k] <= x[edges[k][1]] for k in range(m)),
                name="pair_right",
            )

            objective = gp.LinExpr()
            for i in range(n):
                if diagonal[i] != 0:
                    objective += diagonal[i] * x[i]
            for k, (_, _, coefficient) in enumerate(edges):
                objective += coefficient * y[k]

            model.setObjective(objective, GRB.MAXIMIZE)

            # Supply the best deterministic solution as a MIP start.
            model.setAttr("Start", [x[i] for i in range(n)], best_selected)

            model.update()

            remaining_time = deadline - time.monotonic()
            if remaining_time > 0:
                model.Params.TimeLimit = max(0.001, remaining_time)

                x_variables = [x[i] for i in range(n)]
                incumbent = {
                    "objective": best_objective,
                    "selected": best_selected[:],
                    "weight": best_weight,
                }

                def incumbent_callback(cb_model, where):
                    if where != GRB.Callback.MIPSOL:
                        return
                    try:
                        values = cb_model.cbGetSolution(x_variables)
                        selected = [1 if value >= 0.5 else 0 for value in values]
                        total_weight = sum(
                            weights[i] for i in range(n) if selected[i]
                        )
                        if total_weight > capacity:
                            return

                        objective_value = evaluate_solution(
                            selected, diagonal, edges
                        )

                        if objective_value > incumbent["objective"]:
                            incumbent["objective"] = objective_value
                            incumbent["selected"] = selected
                            incumbent["weight"] = total_weight

                            if logger:
                                logger.log_solution(
                                    objective_value,
                                    make_solution(objective_value, selected),
                                )
                    except Exception:
                        # Logging or callback extraction must not interrupt the
                        # optimization; the final incumbent is queried below.
                        return

                model.optimize(incumbent_callback)

                if incumbent["objective"] > best_objective:
                    best_objective = incumbent["objective"]
                    best_selected = incumbent["selected"]
                    best_weight = incumbent["weight"]

                if model.SolCount > 0:
                    solver_selected = [
                        1 if x[i].X >= 0.5 else 0 for i in range(n)
                    ]
                    solver_weight = sum(
                        weights[i] for i in range(n) if solver_selected[i]
                    )

                    if solver_weight <= capacity:
                        solver_objective = evaluate_solution(
                            solver_selected, diagonal, edges
                        )
                        if solver_objective > best_objective:
                            best_objective = solver_objective
                            best_selected = solver_selected
                            best_weight = solver_weight
                            if logger:
                                logger.log_solution(
                                    best_objective,
                                    make_solution(
                                        best_objective, best_selected
                                    ),
                                )

        except Exception:
            # If Gurobi is unavailable, cannot initialize, or runs out of
            # resources, return the best feasible heuristic solution.
            pass

    final_solution = make_solution(best_objective, best_selected)
    write_solution(args.solution_path, final_solution)


if __name__ == "__main__":
    main()