import argparse
import json
import os
import time

import numpy as np

from solution_logger import SolutionLogger


def build_solution(costs, facilities):
    facilities = sorted(int(f) for f in facilities)
    facility_array = np.asarray(facilities, dtype=np.int64)

    selected_costs = costs[:, facility_array]
    nearest_positions = np.argmin(selected_costs, axis=1)
    assigned_facilities = facility_array[nearest_positions]
    assignment_costs = costs[np.arange(costs.shape[0]), assigned_facilities]
    objective = int(np.sum(assignment_costs, dtype=np.int64))

    assignments = [
        {
            "customer": int(i),
            "facility": int(assigned_facilities[i]),
            "cost": int(assignment_costs[i]),
        }
        for i in range(costs.shape[0])
    ]

    return {
        "objective_value": objective,
        "opened_facilities": facilities,
        "assignments": assignments,
    }


def greedy_facilities(costs, p, deadline):
    """Greedy-add construction, returning a feasible set even if interrupted."""
    n = costs.shape[0]
    selected = np.zeros(n, dtype=bool)

    column_totals = np.sum(costs, axis=0, dtype=np.int64)
    first = int(np.argmin(column_totals))
    facilities = [first]
    selected[first] = True
    best_cost = costs[:, first].copy()

    while len(facilities) < p and time.monotonic() < deadline:
        candidates = np.flatnonzero(~selected)
        if candidates.size == 0:
            break

        candidate_costs = costs[:, candidates]
        improvements = np.maximum(
            best_cost[:, None] - candidate_costs, 0
        ).sum(axis=0, dtype=np.int64)

        chosen = int(candidates[int(np.argmax(improvements))])
        facilities.append(chosen)
        selected[chosen] = True
        best_cost = np.minimum(best_cost, costs[:, chosen])

    # Complete the solution quickly if the heuristic deadline was reached.
    if len(facilities) < p:
        remaining = np.flatnonzero(~selected)
        order = remaining[np.argsort(column_totals[remaining], kind="stable")]
        for value in order[: p - len(facilities)]:
            facilities.append(int(value))

    return sorted(facilities)


def local_search(costs, facilities, deadline, incumbent_callback):
    """Best-improvement one-swap local search."""
    n = costs.shape[0]
    facilities = sorted(int(f) for f in facilities)
    current_solution = build_solution(costs, facilities)
    current_objective = current_solution["objective_value"]

    rounds = 0
    while time.monotonic() < deadline and rounds < 30:
        rounds += 1
        facility_array = np.asarray(facilities, dtype=np.int64)
        open_mask = np.zeros(n, dtype=bool)
        open_mask[facility_array] = True
        closed = np.flatnonzero(~open_mask)

        if closed.size == 0:
            break

        open_costs = costs[:, facility_array]
        assigned_position = np.argmin(open_costs, axis=1)
        best_cost = open_costs[np.arange(n), assigned_position]

        if len(facilities) > 1:
            second_cost = np.partition(open_costs, 1, axis=1)[:, 1]
        else:
            second_cost = np.full(n, np.iinfo(np.int64).max // 4, dtype=np.int64)

        best_swap_objective = current_objective
        best_remove_position = None
        best_add = None

        interrupted = False
        for add in closed:
            if time.monotonic() >= deadline:
                interrupted = True
                break

            add = int(add)
            add_cost = costs[:, add]

            for remove_position in range(len(facilities)):
                without_removed = np.where(
                    assigned_position == remove_position,
                    second_cost,
                    best_cost,
                )
                new_objective = int(
                    np.minimum(without_removed, add_cost).sum(dtype=np.int64)
                )

                if new_objective < best_swap_objective:
                    best_swap_objective = new_objective
                    best_remove_position = remove_position
                    best_add = add

        if best_remove_position is None:
            break

        facilities[best_remove_position] = best_add
        facilities.sort()
        current_solution = build_solution(costs, facilities)
        current_objective = current_solution["objective_value"]
        incumbent_callback(current_solution)

        if interrupted:
            break

    return current_solution


def solve_with_gurobi(costs, p, initial_solution, deadline, incumbent_callback):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return initial_solution

    n = costs.shape[0]
    if time.monotonic() >= deadline - 0.05:
        return initial_solution

    try:
        model = gp.Model("p_median")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1

        y = model.addVars(n, vtype=GRB.BINARY, name="open")
        x = model.addVars(
            n,
            n,
            lb=0.0,
            ub=1.0,
            vtype=GRB.CONTINUOUS,
            name="assign",
        )

        model.addConstr(gp.quicksum(y[j] for j in range(n)) == p, name="facility_count")

        model.addConstrs(
            (
                gp.quicksum(x[i, j] for j in range(n)) == 1
                for i in range(n)
            ),
            name="serve_once",
        )

        model.addConstrs(
            (
                x[i, j] <= y[j]
                for i in range(n)
                for j in range(n)
                if i != j
            ),
            name="open_link",
        )

        # An open facility must serve itself, as required by the problem.
        model.addConstrs(
            (x[i, i] == y[i] for i in range(n)),
            name="self_service",
        )

        objective = gp.LinExpr()
        for i in range(n):
            objective.addTerms(
                [int(v) for v in costs[i, :]],
                [x[i, j] for j in range(n)],
            )
        model.setObjective(objective, GRB.MINIMIZE)

        initial_open = set(initial_solution["opened_facilities"])
        for j in range(n):
            y[j].Start = 1.0 if j in initial_open else 0.0

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return initial_solution

        model.Params.TimeLimit = max(0.01, remaining)

        callback_state = {
            "best_solution": initial_solution,
            "best_objective": initial_solution["objective_value"],
        }

        def mip_callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return

            try:
                values = cb_model.cbGetSolution(y)
                ranked = sorted(
                    range(n),
                    key=lambda j: (-float(values[j]), j),
                )
                facilities = sorted(ranked[:p])
                solution = build_solution(costs, facilities)

                if solution["objective_value"] < callback_state["best_objective"]:
                    callback_state["best_objective"] = solution["objective_value"]
                    callback_state["best_solution"] = solution
                    incumbent_callback(solution)
            except Exception:
                # Logging or extracting an incumbent must not abort optimization.
                pass

        model.optimize(mip_callback)

        best_solution = callback_state["best_solution"]

        if model.SolCount > 0:
            values = [float(y[j].X) for j in range(n)]
            ranked = sorted(range(n), key=lambda j: (-values[j], j))
            facilities = sorted(ranked[:p])
            final_mip_solution = build_solution(costs, facilities)

            if final_mip_solution["objective_value"] < best_solution["objective_value"]:
                best_solution = final_mip_solution
                incumbent_callback(best_solution)

        return best_solution

    except Exception:
        # Return the best heuristic solution if Gurobi cannot build or solve
        # the model (for example, due to license or model-size restrictions).
        return initial_solution


def main():
    parser = argparse.ArgumentParser(description="Solve the discrete p-median problem.")
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    time_limit = max(0, args.time_limit)
    deadline = start_time + time_limit

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as instance_file:
        instance = json.load(instance_file)

    n = int(instance["n"])
    p = int(instance["p"])
    costs = np.asarray(instance["cost_matrix"], dtype=np.int64)

    if costs.shape != (n, n):
        raise ValueError("cost_matrix must have shape [n][n]")
    if not (0 < p < n):
        raise ValueError("p must be strictly between 0 and n")

    best_solution = build_solution(costs, list(range(p)))
    best_objective = best_solution["objective_value"]

    if logger:
        logger.log_solution(best_objective, best_solution)

    def accept_incumbent(solution):
        nonlocal best_solution, best_objective
        objective_value = int(solution["objective_value"])
        if objective_value < best_objective:
            best_solution = solution
            best_objective = objective_value
            if logger:
                logger.log_solution(objective_value, solution)

    # Reserve most of the runtime for exact MIP search.
    available = max(0.0, deadline - time.monotonic())
    heuristic_budget = min(8.0, 0.20 * available)
    heuristic_deadline = min(deadline, time.monotonic() + heuristic_budget)

    if time.monotonic() < heuristic_deadline:
        greedy_set = greedy_facilities(costs, p, heuristic_deadline)
        greedy_solution = build_solution(costs, greedy_set)
        accept_incumbent(greedy_solution)

    if time.monotonic() < heuristic_deadline:
        local_solution = local_search(
            costs,
            best_solution["opened_facilities"],
            heuristic_deadline,
            accept_incumbent,
        )
        accept_incumbent(local_solution)

    if time.monotonic() < deadline:
        mip_solution = solve_with_gurobi(
            costs,
            p,
            best_solution,
            deadline,
            accept_incumbent,
        )
        accept_incumbent(mip_solution)

    solution_directory = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(solution_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as solution_file:
        json.dump(best_solution, solution_file, separators=(",", ":"))


if __name__ == "__main__":
    main()