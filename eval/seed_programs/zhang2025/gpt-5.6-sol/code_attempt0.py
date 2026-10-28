import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def make_solution(selected_duties, costs):
    selected = sorted(int(j) for j in selected_duties)
    objective = float(math.fsum(float(costs[j]) for j in selected))
    return {
        "objective_value": objective,
        "selected_duties": selected,
    }


def write_solution(path, solution):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(solution, f, separators=(",", ":"))


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

    num_trips = int(instance["parameters"]["num_trips_M"])
    duties = instance["duties"]
    costs = [float(c) for c in instance["costs"]]

    if len(duties) != len(costs):
        raise ValueError("The duties and costs arrays have different lengths")

    # Remove empty duties and retain only the cheapest representative of duties
    # having identical trip incidence.
    representative = {}
    normalized_by_original = {}

    for original_idx, duty in enumerate(duties):
        normalized = tuple(sorted(set(int(t) for t in duty)))
        if not normalized:
            continue
        if normalized[0] < 0 or normalized[-1] >= num_trips:
            raise ValueError(f"Duty {original_idx} contains an invalid trip_id")

        normalized_by_original[original_idx] = normalized
        previous = representative.get(normalized)
        if previous is None or costs[original_idx] < costs[previous]:
            representative[normalized] = original_idx

    kept_original = list(representative.values())
    kept_sets = [normalized_by_original[j] for j in kept_original]
    kept_costs = [costs[j] for j in kept_original]
    num_columns = len(kept_original)

    incidence = [[] for _ in range(num_trips)]
    singleton_column = [None] * num_trips

    for col, trip_set in enumerate(kept_sets):
        for trip in trip_set:
            incidence[trip].append(col)
        if len(trip_set) == 1:
            trip = trip_set[0]
            old = singleton_column[trip]
            if old is None or kept_costs[col] < kept_costs[old]:
                singleton_column[trip] = col

    for trip, columns in enumerate(incidence):
        if not columns:
            raise RuntimeError(f"Instance is infeasible: trip {trip} has no covering duty")

    # A guaranteed feasible starting solution exists if every trip has a
    # singleton duty. Improve it by replacing disjoint singleton duties with
    # profitable multi-trip duties.
    warm_columns = None
    warm_solution = None

    if all(col is not None for col in singleton_column):
        occupied = [False] * num_trips
        chosen_multi = []

        # Only spend time sorting improvements when some budget remains.
        if deadline - time.monotonic() > 0.1:
            improvements = []
            for col, trip_set in enumerate(kept_sets):
                if len(trip_set) <= 1:
                    continue
                singleton_cost = math.fsum(
                    kept_costs[singleton_column[t]] for t in trip_set
                )
                saving = singleton_cost - kept_costs[col]
                if saving > 1e-12:
                    # Saving first; then prefer duties covering more trips.
                    improvements.append((saving, len(trip_set), col))

            improvements.sort(reverse=True)
            for _, _, col in improvements:
                trip_set = kept_sets[col]
                if all(not occupied[t] for t in trip_set):
                    chosen_multi.append(col)
                    for t in trip_set:
                        occupied[t] = True

        warm_columns = list(chosen_multi)
        for trip in range(num_trips):
            if not occupied[trip]:
                warm_columns.append(singleton_column[trip])

        warm_original = [kept_original[col] for col in warm_columns]
        warm_solution = make_solution(warm_original, costs)
        if logger:
            logger.log_solution(warm_solution["objective_value"], warm_solution)

    # If preprocessing consumed the whole budget and a feasible solution is
    # already available, return it without invoking the solver.
    if time.monotonic() >= deadline and warm_solution is not None:
        write_solution(args.solution_path, warm_solution)
        return

    model = gp.Model("driver_duty_set_partitioning")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    remaining = max(0.01, deadline - time.monotonic())
    model.Params.TimeLimit = remaining

    x = model.addVars(
        num_columns,
        vtype=GRB.BINARY,
        obj={j: kept_costs[j] for j in range(num_columns)},
        name="select",
    )

    for trip in range(num_trips):
        model.addConstr(
            gp.quicksum(x[col] for col in incidence[trip]) == 1,
            name=f"cover_{trip}",
        )

    model.ModelSense = GRB.MINIMIZE

    if warm_columns is not None:
        warm_set = set(warm_columns)
        for col in range(num_columns):
            x[col].Start = 1.0 if col in warm_set else 0.0

    best_solution = warm_solution
    best_objective = (
        warm_solution["objective_value"] if warm_solution is not None else math.inf
    )

    model._variables = [x[j] for j in range(num_columns)]
    model._kept_original = kept_original
    model._all_costs = costs
    model._logger = logger
    model._best_objective = best_objective
    model._best_solution = best_solution

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return

        values = cb_model.cbGetSolution(cb_model._variables)
        selected_original = [
            cb_model._kept_original[col]
            for col, value in enumerate(values)
            if value > 0.5
        ]
        solution = make_solution(selected_original, cb_model._all_costs)
        objective = solution["objective_value"]

        if objective < cb_model._best_objective - 1e-9:
            cb_model._best_objective = objective
            cb_model._best_solution = solution
            if cb_model._logger:
                cb_model._logger.log_solution(objective, solution)

    model.optimize(incumbent_callback)

    # Read the solver's final incumbent explicitly. This also handles cases
    # where a solver-provided incumbent was not considered an improvement over
    # the manually constructed warm start.
    if model.SolCount > 0:
        selected_original = [
            kept_original[col]
            for col in range(num_columns)
            if x[col].X > 0.5
        ]
        final_incumbent = make_solution(selected_original, costs)

        if (
            model._best_solution is None
            or final_incumbent["objective_value"] < model._best_objective - 1e-9
        ):
            model._best_objective = final_incumbent["objective_value"]
            model._best_solution = final_incumbent
            if logger:
                logger.log_solution(
                    final_incumbent["objective_value"], final_incumbent
                )

    best_solution = model._best_solution
    if best_solution is None:
        raise RuntimeError(
            "No feasible solution was found within the time limit, and no "
            "singleton-based feasible solution was available."
        )

    write_solution(args.solution_path, best_solution)


if __name__ == "__main__":
    main()