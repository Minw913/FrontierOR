import argparse
import json
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def compute_temporal_bounds(num_activities, horizon, constraints):
    """Propagate lower and upper bounds through difference constraints."""
    lb = [0] * num_activities
    ub = [horizon] * num_activities
    lb[0] = 0
    ub[0] = 0

    # Longest-path propagation for lower bounds.
    for _ in range(num_activities):
        changed = False
        for i, j, delta in constraints:
            candidate = lb[i] + delta
            if candidate > lb[j]:
                lb[j] = candidate
                changed = True
        if not changed:
            break

    # A further possible relaxation indicates a positive cycle.
    for i, j, delta in constraints:
        if lb[j] < lb[i] + delta:
            return None, None

    # Reverse propagation for upper bounds.
    for _ in range(num_activities):
        changed = False
        for i, j, delta in constraints:
            candidate = ub[j] - delta
            if candidate < ub[i]:
                ub[i] = candidate
                changed = True
        if not changed:
            break

    for i, j, delta in constraints:
        if ub[i] > ub[j] - delta:
            return None, None

    if lb[0] != 0 or ub[0] != 0:
        return None, None

    for i in range(num_activities):
        lb[i] = max(lb[i], 0)
        ub[i] = min(ub[i], horizon)
        if lb[i] > ub[i]:
            return None, None

    return lb, ub


def make_solution(start_values, end_activity):
    starts = {str(i): int(start_values[i]) for i in range(len(start_values))}
    objective = int(start_values[end_activity])
    return {
        "objective_value": objective,
        "start_times": starts,
    }


def main():
    args = parse_args()
    wall_start = time.monotonic()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = int(data["n"])
    num_activities = int(data.get("num_activities", n + 2))
    num_resources = int(data["num_resources"])
    horizon = int(data["d_bar"])
    end_activity = n + 1

    if num_activities != n + 2:
        end_activity = num_activities - 1

    activities_data = data["activities"]
    resources_data = data["resources"]

    activity_ids = list(range(num_activities))
    real_activities = list(range(1, end_activity))

    processing_times = [0] * num_activities
    demands = {}

    for i in activity_ids:
        entry = activities_data.get(str(i), {})
        processing_times[i] = int(entry.get("processing_time", 0))
        raw_demands = entry.get("resource_demands", {})
        demands[i] = {str(k): int(v) for k, v in raw_demands.items()}

    resource_keys = list(resources_data.keys())
    if len(resource_keys) != num_resources:
        resource_keys = sorted(resources_data.keys(), key=lambda x: int(x))

    capacities = {}
    period_prefix = {}

    for resource_key in resource_keys:
        resource = resources_data[resource_key]
        capacities[resource_key] = int(resource["capacity"])

        present = [0] * (horizon + 1)
        for period in set(int(p) for p in resource.get("periods", [])):
            if 1 <= period <= horizon:
                present[period] = 1

        prefix = [0] * (horizon + 1)
        running = 0
        for period in range(1, horizon + 1):
            running += present[period]
            prefix[period] = running
        period_prefix[resource_key] = prefix

    temporal_constraints = [
        (int(c["i"]), int(c["j"]), int(c["delta"]))
        for c in data.get("temporal_constraints", [])
    ]

    lb, ub = compute_temporal_bounds(
        num_activities, horizon, temporal_constraints
    )
    if lb is None:
        raise RuntimeError("The temporal constraints are infeasible.")

    def consumption(i, resource_key, start):
        demand = demands[i].get(resource_key, 0)
        if demand == 0 or processing_times[i] == 0:
            return 0
        finish = min(horizon, start + processing_times[i])
        if finish <= start:
            return 0
        prefix = period_prefix[resource_key]
        overlap = prefix[finish] - prefix[min(start, horizon)]
        return demand * overlap

    def candidate_is_feasible(start_values):
        if len(start_values) != num_activities:
            return False
        if start_values[0] != 0:
            return False

        for i in activity_ids:
            if start_values[i] < 0 or start_values[i] > horizon:
                return False

        for i, j, delta in temporal_constraints:
            if start_values[j] - start_values[i] < delta:
                return False

        for resource_key in resource_keys:
            total = 0
            capacity = capacities[resource_key]
            for i in real_activities:
                total += consumption(i, resource_key, start_values[i])
                if total > capacity:
                    return False
        return True

    fallback_solution = None
    best_logged_objective = float("inf")

    # Temporal lower and upper schedules are inexpensive initial candidates.
    for candidate in (ub[:], lb[:]):
        if candidate_is_feasible(candidate):
            solution = make_solution(candidate, end_activity)
            if (
                fallback_solution is None
                or solution["objective_value"] < fallback_solution["objective_value"]
            ):
                fallback_solution = solution

    if fallback_solution is not None:
        best_logged_objective = fallback_solution["objective_value"]
        if logger:
            logger.log_solution(
                fallback_solution["objective_value"], fallback_solution
            )

    model = gp.Model("partially_renewable_project_scheduling")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    remaining_time = max(0.0, float(args.time_limit) - (time.monotonic() - wall_start))
    model.Params.TimeLimit = remaining_time

    start_vars = {
        i: model.addVar(
            lb=float(lb[i]),
            ub=float(ub[i]),
            vtype=GRB.INTEGER,
            name=f"s_{i}",
        )
        for i in activity_ids
    }
    start_vars[0].LB = 0.0
    start_vars[0].UB = 0.0

    x_vars = {}
    starts_by_activity = {}
    resource_terms = {k: [] for k in resource_keys}

    for i in real_activities:
        allowed_starts = []

        for t in range(lb[i], ub[i] + 1):
            coefficients = {}
            individually_feasible = True

            for resource_key in resource_keys:
                coefficient = consumption(i, resource_key, t)
                if coefficient > capacities[resource_key]:
                    individually_feasible = False
                    break
                if coefficient:
                    coefficients[resource_key] = coefficient

            if not individually_feasible:
                continue

            var = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{t}")
            x_vars[(i, t)] = var
            allowed_starts.append(t)

            for resource_key, coefficient in coefficients.items():
                resource_terms[resource_key].append((coefficient, var))

        if not allowed_starts:
            raise RuntimeError(
                f"Activity {i} has no individually resource-feasible start time."
            )

        starts_by_activity[i] = allowed_starts

    model.update()

    for i in real_activities:
        starts = starts_by_activity[i]
        model.addConstr(
            gp.quicksum(x_vars[(i, t)] for t in starts) == 1,
            name=f"choose_start_{i}",
        )
        model.addConstr(
            start_vars[i]
            == gp.quicksum(t * x_vars[(i, t)] for t in starts),
            name=f"link_start_{i}",
        )

    for index, (i, j, delta) in enumerate(temporal_constraints):
        model.addConstr(
            start_vars[j] - start_vars[i] >= delta,
            name=f"temporal_{index}",
        )

    for resource_key in resource_keys:
        terms = resource_terms[resource_key]
        if terms:
            coefficients = [coefficient for coefficient, _ in terms]
            variables = [var for _, var in terms]
            expression = gp.LinExpr(coefficients, variables)
            model.addConstr(
                expression <= capacities[resource_key],
                name=f"resource_{resource_key}",
            )

    model.setObjective(start_vars[end_activity], GRB.MINIMIZE)

    # Supply a complete MIP start when one of the temporal schedules uses
    # available time-indexed variables. It may still violate aggregate
    # capacities; Gurobi can use it as a repair start.
    warm_candidate = None
    for candidate in (ub, lb):
        if all(
            candidate[i] in starts_by_activity[i]
            for i in real_activities
        ):
            warm_candidate = candidate
            if candidate_is_feasible(candidate):
                break

    if warm_candidate is not None:
        for i in activity_ids:
            start_vars[i].Start = warm_candidate[i]
        for i in real_activities:
            chosen = warm_candidate[i]
            for t in starts_by_activity[i]:
                x_vars[(i, t)].Start = 1.0 if t == chosen else 0.0

    callback_best = [best_logged_objective]

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return

        try:
            values = [
                int(round(cb_model.cbGetSolution(start_vars[i])))
                for i in activity_ids
            ]
            solution = make_solution(values, end_activity)
            objective = solution["objective_value"]

            if objective < callback_best[0]:
                callback_best[0] = objective
                if logger:
                    logger.log_solution(objective, solution)
        except Exception:
            # Logging must not terminate the optimization.
            pass

    if logger:
        model.optimize(incumbent_callback)
    else:
        model.optimize()

    final_solution = None

    if model.SolCount > 0:
        values = [
            int(round(start_vars[i].X))
            for i in activity_ids
        ]
        final_solution = make_solution(values, end_activity)
    elif fallback_solution is not None:
        final_solution = fallback_solution
    else:
        raise RuntimeError(
            "No feasible solution was found within the supplied time limit."
        )

    if final_solution["objective_value"] < callback_best[0]:
        callback_best[0] = final_solution["objective_value"]
        if logger:
            logger.log_solution(
                final_solution["objective_value"], final_solution
            )

    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, indent=2)


if __name__ == "__main__":
    main()