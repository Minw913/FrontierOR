import argparse
import json
import math
import os
import time

import numpy as np

from solution_logger import SolutionLogger


def compute_cost_matrix(coordinates):
    """
    Compute rounded Euclidean distances from coordinates.

    Rounding uses floor(distance + 0.5), i.e. conventional rounding to the
    nearest nonnegative integer rather than NumPy's ties-to-even rounding.
    The calculation is blocked to limit temporary memory usage.
    """
    coords = np.asarray(coordinates, dtype=np.float64)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("node_coordinates must have shape [n][2]")

    n = coords.shape[0]
    costs = np.empty((n, n), dtype=np.int64)

    # Keep temporary arrays at a moderate size.
    target_elements = 2_000_000
    block_size = max(1, min(n, target_elements // max(1, n)))

    all_x = coords[:, 0]
    all_y = coords[:, 1]

    for start in range(0, n, block_size):
        end = min(n, start + block_size)

        dx = coords[start:end, 0, None] - all_x[None, :]
        dy = coords[start:end, 1, None] - all_y[None, :]

        np.square(dx, out=dx)
        np.square(dy, out=dy)
        dx += dy
        np.sqrt(dx, out=dx)
        dx += 0.5
        np.floor(dx, out=dx)

        costs[start:end, :] = dx.astype(np.int64)

    np.fill_diagonal(costs, 0)
    return costs


def load_cost_matrix(instance):
    n = int(instance["n"])
    matrix_data = instance.get("cost_matrix")

    # Some large instances intentionally omit the quadratic matrix and store
    # it as JSON null. In that case, derive it from node_coordinates.
    if matrix_data is None:
        coordinates = instance.get("node_coordinates")
        if coordinates is None:
            raise ValueError(
                "Instance has neither a usable cost_matrix nor node_coordinates"
            )
        costs = compute_cost_matrix(coordinates)
    else:
        try:
            costs = np.asarray(matrix_data, dtype=np.int64)
        except (TypeError, ValueError):
            coordinates = instance.get("node_coordinates")
            if coordinates is None:
                raise
            costs = compute_cost_matrix(coordinates)

    if costs.shape != (n, n):
        raise ValueError(
            f"cost_matrix must have shape [{n}][{n}], got {costs.shape}"
        )

    return costs


def build_solution(costs, facilities):
    n = costs.shape[0]
    facilities = sorted(int(facility) for facility in facilities)
    facility_array = np.asarray(facilities, dtype=np.int64)

    selected_costs = costs[:, facility_array]
    nearest_positions = np.argmin(selected_costs, axis=1)
    assigned_facilities = facility_array[nearest_positions]
    assignment_costs = costs[np.arange(n), assigned_facilities]

    objective_value = int(
        np.sum(assignment_costs, dtype=np.int64)
    )

    assignments = [
        {
            "customer": int(customer),
            "facility": int(assigned_facilities[customer]),
            "cost": int(assignment_costs[customer]),
        }
        for customer in range(n)
    ]

    return {
        "objective_value": objective_value,
        "opened_facilities": facilities,
        "assignments": assignments,
    }


def greedy_facilities(costs, p, deadline):
    """Construct a facility set using greedy marginal improvement."""
    n = costs.shape[0]
    selected = np.zeros(n, dtype=bool)
    column_totals = np.sum(costs, axis=0, dtype=np.int64)

    first = int(np.argmin(column_totals))
    facilities = [first]
    selected[first] = True
    current_cost = costs[:, first].copy()

    # Bound temporary matrices to roughly 32 MB.
    block_width = max(1, min(n, 4_000_000 // max(1, n)))

    while len(facilities) < p and time.monotonic() < deadline:
        candidates = np.flatnonzero(~selected)
        if candidates.size == 0:
            break

        best_improvement = -1
        best_candidate = None

        for start in range(0, candidates.size, block_width):
            if time.monotonic() >= deadline:
                break

            block = candidates[start:start + block_width]
            difference = current_cost[:, None] - costs[:, block]
            np.maximum(difference, 0, out=difference)
            improvements = difference.sum(axis=0, dtype=np.int64)

            position = int(np.argmax(improvements))
            improvement = int(improvements[position])
            candidate = int(block[position])

            if (
                improvement > best_improvement
                or (
                    improvement == best_improvement
                    and (best_candidate is None or candidate < best_candidate)
                )
            ):
                best_improvement = improvement
                best_candidate = candidate

        if best_candidate is None:
            break

        facilities.append(best_candidate)
        selected[best_candidate] = True
        np.minimum(
            current_cost,
            costs[:, best_candidate],
            out=current_cost,
        )

    # Always return exactly p facilities, even if the time budget expired.
    if len(facilities) < p:
        remaining = np.flatnonzero(~selected)
        order = remaining[
            np.argsort(column_totals[remaining], kind="stable")
        ]
        needed = p - len(facilities)
        facilities.extend(int(value) for value in order[:needed])

    return sorted(facilities)


def local_search(costs, initial_facilities, deadline, incumbent_callback):
    """Perform best-improvement facility swaps."""
    n = costs.shape[0]
    facilities = sorted(int(f) for f in initial_facilities)
    current_solution = build_solution(costs, facilities)
    current_objective = current_solution["objective_value"]

    max_rounds = 30

    for _ in range(max_rounds):
        if time.monotonic() >= deadline:
            break

        facility_array = np.asarray(facilities, dtype=np.int64)
        open_mask = np.zeros(n, dtype=bool)
        open_mask[facility_array] = True
        closed_facilities = np.flatnonzero(~open_mask)

        if closed_facilities.size == 0:
            break

        open_costs = costs[:, facility_array]
        assigned_positions = np.argmin(open_costs, axis=1)
        customer_indices = np.arange(n)
        best_costs = open_costs[customer_indices, assigned_positions]

        if len(facilities) > 1:
            second_costs = np.partition(open_costs, 1, axis=1)[:, 1]
        else:
            second_costs = np.full(
                n,
                np.iinfo(np.int64).max // 4,
                dtype=np.int64,
            )

        best_swap_objective = current_objective
        best_remove_position = None
        best_add_facility = None
        interrupted = False

        for add_facility_raw in closed_facilities:
            if time.monotonic() >= deadline:
                interrupted = True
                break

            add_facility = int(add_facility_raw)
            add_costs = costs[:, add_facility]

            for remove_position in range(len(facilities)):
                if remove_position % 32 == 0 and time.monotonic() >= deadline:
                    interrupted = True
                    break

                without_removed = np.where(
                    assigned_positions == remove_position,
                    second_costs,
                    best_costs,
                )
                candidate_objective = int(
                    np.minimum(without_removed, add_costs).sum(
                        dtype=np.int64
                    )
                )

                if candidate_objective < best_swap_objective:
                    best_swap_objective = candidate_objective
                    best_remove_position = remove_position
                    best_add_facility = add_facility

            if interrupted:
                break

        if best_remove_position is None:
            break

        facilities[best_remove_position] = best_add_facility
        facilities.sort()

        current_solution = build_solution(costs, facilities)
        current_objective = current_solution["objective_value"]
        incumbent_callback(current_solution)

        if interrupted:
            break

    return current_solution


def solve_with_gurobi(
    costs,
    p,
    initial_solution,
    deadline,
    incumbent_callback,
):
    """
    Solve smaller instances exactly with the standard p-median MIP.

    Large instances are left to the scalable heuristics because constructing
    an O(n^2) Gurobi model can consume the entire runtime and excessive memory.
    """
    n = costs.shape[0]

    if n > 300 or time.monotonic() >= deadline - 0.1:
        return initial_solution

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
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

        model.addConstr(
            gp.quicksum(y[j] for j in range(n)) == p,
            name="facility_count",
        )

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
            ),
            name="open_link",
        )

        model.setObjective(
            gp.quicksum(
                int(costs[i, j]) * x[i, j]
                for i in range(n)
                for j in range(n)
                if costs[i, j] != 0
            ),
            GRB.MINIMIZE,
        )

        initial_open = set(initial_solution["opened_facilities"])

        for j in range(n):
            y[j].Start = 1.0 if j in initial_open else 0.0

        initial_assignments = {
            assignment["customer"]: assignment["facility"]
            for assignment in initial_solution["assignments"]
        }

        for i in range(n):
            assigned = initial_assignments[i]
            for j in range(n):
                x[i, j].Start = 1.0 if j == assigned else 0.0

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
                open_values = cb_model.cbGetSolution(y)
                ranked = sorted(
                    range(n),
                    key=lambda facility: (
                        -float(open_values[facility]),
                        facility,
                    ),
                )
                facilities = sorted(ranked[:p])
                solution = build_solution(costs, facilities)

                if (
                    solution["objective_value"]
                    < callback_state["best_objective"]
                ):
                    callback_state["best_objective"] = solution[
                        "objective_value"
                    ]
                    callback_state["best_solution"] = solution
                    incumbent_callback(solution)
            except Exception:
                # A callback extraction/logging failure should not stop Gurobi.
                pass

        model.optimize(mip_callback)
        best_solution = callback_state["best_solution"]

        if model.SolCount > 0:
            open_values = [float(y[j].X) for j in range(n)]
            ranked = sorted(
                range(n),
                key=lambda facility: (
                    -open_values[facility],
                    facility,
                ),
            )
            mip_facilities = sorted(ranked[:p])
            mip_solution = build_solution(costs, mip_facilities)

            if (
                mip_solution["objective_value"]
                < best_solution["objective_value"]
            ):
                best_solution = mip_solution
                incumbent_callback(best_solution)

        return best_solution

    except Exception:
        return initial_solution


def main():
    parser = argparse.ArgumentParser(
        description="Solve the discrete p-median problem."
    )
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path
        else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as file:
        instance = json.load(file)

    n = int(instance["n"])
    p = int(instance["p"])

    if not (0 < p < n):
        raise ValueError("p must be strictly between zero and n")

    costs = load_cost_matrix(instance)

    # This baseline can be produced immediately after loading the instance.
    best_solution = build_solution(costs, range(p))
    best_objective = best_solution["objective_value"]

    if logger:
        logger.log_solution(best_objective, best_solution)

    def accept_incumbent(solution):
        nonlocal best_solution, best_objective

        objective = int(solution["objective_value"])
        if objective < best_objective:
            best_solution = solution
            best_objective = objective

            if logger:
                logger.log_solution(objective, solution)

    available = max(0.0, deadline - time.monotonic())

    # Spend a bounded fraction on construction/local search and leave the
    # remainder for Gurobi on instances where an exact model is practical.
    if n <= 300:
        heuristic_budget = min(8.0, 0.25 * available)
    else:
        heuristic_budget = available

    heuristic_deadline = min(
        deadline,
        time.monotonic() + heuristic_budget,
    )

    if time.monotonic() < heuristic_deadline:
        greedy_set = greedy_facilities(
            costs,
            p,
            heuristic_deadline,
        )
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

    output_directory = os.path.dirname(
        os.path.abspath(args.solution_path)
    )
    os.makedirs(output_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as file:
        json.dump(best_solution, file, separators=(",", ":"))


if __name__ == "__main__":
    main()