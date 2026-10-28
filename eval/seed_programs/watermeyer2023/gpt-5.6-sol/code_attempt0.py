import argparse
import json
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def write_solution(path, solution):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(solution, f, separators=(",", ":"))


def propagate_bounds(num_nodes, deadline, arcs):
    """Propagate integer lower and upper bounds through difference constraints."""
    lower = [0] * num_nodes
    upper = [deadline] * num_nodes

    # The dummy start is fixed at zero.
    lower[0] = 0
    upper[0] = 0

    positive_cycle = False

    # Longest-path propagation for lower bounds.
    for iteration in range(num_nodes):
        changed = False
        for u, v, delta in arcs:
            value = lower[u] + delta
            if value > lower[v]:
                lower[v] = value
                changed = True
        if not changed:
            break
        if iteration == num_nodes - 1:
            positive_cycle = True

    # Reverse propagation for upper bounds:
    # S[v] >= S[u] + d  =>  S[u] <= S[v] - d.
    for iteration in range(num_nodes):
        changed = False
        for u, v, delta in arcs:
            value = upper[v] - delta
            if value < upper[u]:
                upper[u] = value
                changed = True
        if not changed:
            break
        if iteration == num_nodes - 1:
            positive_cycle = True

    if positive_cycle:
        return None, None

    if lower[0] != 0 or upper[0] != 0:
        return None, None

    for i in range(num_nodes):
        lower[i] = max(lower[i], 0)
        upper[i] = min(upper[i], deadline)
        if lower[i] > upper[i]:
            return None, None

    return lower, upper


def build_resource_profiles(instance, lower, upper):
    """
    For each activity, split its start-time domain into contiguous intervals
    having identical resource-consumption vectors.
    """
    num_nodes = instance["num_nodes"]
    deadline = instance["deadline"]
    processing_times = instance["processing_times"]
    resources = instance.get("resources", [])
    num_resources = len(resources)

    prefixes = []
    for resource in resources:
        marked = [0] * (deadline + 1)
        for period in resource.get("time_periods", []):
            if 1 <= period <= deadline:
                marked[period] = 1

        prefix = [0] * (deadline + 1)
        for t in range(1, deadline + 1):
            prefix[t] = prefix[t - 1] + marked[t]
        prefixes.append(prefix)

    profiles = []
    for i in range(num_nodes):
        duration = int(processing_times[i])
        intervals = []
        interval_start = lower[i]
        previous_vector = None

        for start in range(lower[i], upper[i] + 1):
            vector = []
            end = min(deadline, start + duration)
            left = min(deadline, max(0, start))

            for r, resource in enumerate(resources):
                demand = int(resource["demands"][i])
                counted_periods = prefixes[r][end] - prefixes[r][left]
                vector.append(demand * counted_periods)

            vector = tuple(vector)

            if previous_vector is None:
                previous_vector = vector
                interval_start = start
            elif vector != previous_vector:
                intervals.append(
                    {
                        "lower": interval_start,
                        "upper": start - 1,
                        "consumption": previous_vector,
                        "var": None,
                    }
                )
                interval_start = start
                previous_vector = vector

        intervals.append(
            {
                "lower": interval_start,
                "upper": upper[i],
                "consumption": previous_vector
                if previous_vector is not None
                else tuple([0] * num_resources),
                "var": None,
            }
        )
        profiles.append(intervals)

    return profiles


def candidate_is_feasible(starts, instance, arcs, lower, upper, profiles):
    num_nodes = instance["num_nodes"]
    resources = instance.get("resources", [])

    if len(starts) != num_nodes or starts[0] != 0:
        return False

    for i, value in enumerate(starts):
        if value < lower[i] or value > upper[i]:
            return False

    for u, v, delta in arcs:
        if starts[v] < starts[u] + delta:
            return False

    totals = [0] * len(resources)
    for i, start in enumerate(starts):
        selected = None
        for interval in profiles[i]:
            if interval["lower"] <= start <= interval["upper"]:
                selected = interval
                break
        if selected is None:
            return False
        for r, amount in enumerate(selected["consumption"]):
            totals[r] += amount

    for r, resource in enumerate(resources):
        if totals[r] > int(resource["capacity"]):
            return False

    return True


def set_mip_start(start_vars, profiles, starts):
    for i, value in enumerate(starts):
        start_vars[i].Start = value
        if len(profiles[i]) > 1:
            for interval in profiles[i]:
                interval["var"].Start = (
                    1.0
                    if interval["lower"] <= value <= interval["upper"]
                    else 0.0
                )


def main():
    wall_start = time.monotonic()

    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    num_nodes = int(instance["num_nodes"])
    num_real = int(instance["num_real_activities"])
    end_node = num_real + 1
    deadline = int(instance["deadline"])
    resources = instance.get("resources", [])

    if end_node != num_nodes - 1:
        raise ValueError("The dummy end node must be num_real_activities + 1.")

    arcs = []
    for constraint in instance.get("temporal_constraints", []):
        # Maximum lags are already represented in the input as reverse arcs
        # with negative deltas, so every entry is handled uniformly.
        arcs.append(
            (
                int(constraint["from"]),
                int(constraint["to"]),
                int(constraint["delta"]),
            )
        )

    lower, upper = propagate_bounds(num_nodes, deadline, arcs)
    if lower is None:
        raise RuntimeError("The temporal constraints are infeasible.")

    profiles = build_resource_profiles(instance, lower, upper)

    best_solution = None
    best_objective = None

    # The propagated earliest and latest schedules both satisfy all temporal
    # constraints. They are inexpensive possible incumbents.
    initial_candidates = [list(lower)]
    if upper != lower:
        initial_candidates.append(list(upper))

    for starts in initial_candidates:
        if candidate_is_feasible(starts, instance, arcs, lower, upper, profiles):
            objective = int(starts[end_node])
            if best_objective is None or objective < best_objective:
                best_objective = objective
                best_solution = {
                    "objective_value": objective,
                    "start_times": [int(v) for v in starts],
                }
                if logger:
                    logger.log_solution(objective, best_solution)

    # A resource-feasible earliest schedule attains the temporal lower bound
    # on the end node and is therefore globally optimal.
    if (
        best_solution is not None
        and best_objective == lower[end_node]
        and best_solution["start_times"] == lower
    ):
        write_solution(args.solution_path, best_solution)
        return

    model = gp.Model("partially_renewable_project_scheduling")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.MIPFocus = 1

    start_vars = [
        model.addVar(
            vtype=GRB.INTEGER,
            lb=lower[i],
            ub=upper[i],
            name=f"start_{i}",
        )
        for i in range(num_nodes)
    ]

    model.addConstr(start_vars[0] == 0, name="dummy_start")

    resource_expressions = [gp.LinExpr() for _ in resources]

    for i in range(num_nodes):
        activity_profiles = profiles[i]

        if len(activity_profiles) == 1:
            consumption = activity_profiles[0]["consumption"]
            for r, amount in enumerate(consumption):
                if amount:
                    resource_expressions[r].addConstant(amount)
            continue

        selection_vars = []
        lower_expr = gp.LinExpr()
        upper_expr = gp.LinExpr()

        for k, interval in enumerate(activity_profiles):
            y = model.addVar(vtype=GRB.BINARY, name=f"profile_{i}_{k}")
            interval["var"] = y
            selection_vars.append(y)
            lower_expr += interval["lower"] * y
            upper_expr += interval["upper"] * y

            for r, amount in enumerate(interval["consumption"]):
                if amount:
                    resource_expressions[r] += amount * y

        model.addConstr(gp.quicksum(selection_vars) == 1, name=f"choose_profile_{i}")
        model.addConstr(start_vars[i] >= lower_expr, name=f"profile_lb_{i}")
        model.addConstr(start_vars[i] <= upper_expr, name=f"profile_ub_{i}")

    for index, (u, v, delta) in enumerate(arcs):
        model.addConstr(
            start_vars[v] >= start_vars[u] + delta,
            name=f"temporal_{index}",
        )

    for r, resource in enumerate(resources):
        model.addConstr(
            resource_expressions[r] <= int(resource["capacity"]),
            name=f"resource_{r}",
        )

    model.setObjective(start_vars[end_node], GRB.MINIMIZE)

    if best_solution is not None:
        set_mip_start(start_vars, profiles, best_solution["start_times"])

    remaining = max(0.0, float(args.time_limit) - (time.monotonic() - wall_start))
    model.Params.TimeLimit = remaining

    incumbent_state = {
        "objective": best_objective,
        "solution": best_solution,
    }

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            values = cb_model.cbGetSolution(start_vars)
            starts = [int(round(value)) for value in values]
            objective = int(starts[end_node])

            current = incumbent_state["objective"]
            if current is not None and objective >= current:
                return

            solution = {
                "objective_value": objective,
                "start_times": starts,
            }
            incumbent_state["objective"] = objective
            incumbent_state["solution"] = solution

            if logger:
                logger.log_solution(objective, solution)
        except Exception:
            # Logging must never interrupt the optimizer.
            pass

    model.optimize(incumbent_callback)

    if model.SolCount > 0:
        starts = [int(round(var.X)) for var in start_vars]
        objective = int(starts[end_node])
        final_solution = {
            "objective_value": objective,
            "start_times": starts,
        }

        if (
            incumbent_state["objective"] is None
            or objective < incumbent_state["objective"]
        ):
            incumbent_state["objective"] = objective
            incumbent_state["solution"] = final_solution
            if logger:
                logger.log_solution(objective, final_solution)

    if incumbent_state["solution"] is None:
        raise RuntimeError(
            "No feasible solution was found within the supplied time limit."
        )

    write_solution(args.solution_path, incumbent_state["solution"])


if __name__ == "__main__":
    main()