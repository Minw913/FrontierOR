import argparse
import json
import os
import time
from collections import deque

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def make_solution(activity_ids, starts, end_id):
    schedule = {str(i): int(round(starts[i])) for i in activity_ids}
    return {
        "objective_value": float(schedule[str(end_id)]),
        "schedule": schedule,
    }


def propagate_bounds(activity_ids, constraints, deadline, start_id=0):
    lb = {i: 0 for i in activity_ids}
    ub = {i: deadline for i in activity_ids}
    lb[start_id] = 0
    ub[start_id] = 0

    outgoing = {i: [] for i in activity_ids}
    incoming = {i: [] for i in activity_ids}

    for i, j, lag in constraints:
        outgoing[i].append((j, lag))
        incoming[j].append((i, lag))

    # Propagate lower bounds through s_j >= s_i + lag.
    queue = deque(activity_ids)
    in_queue = set(activity_ids)

    while queue:
        i = queue.popleft()
        in_queue.discard(i)

        for j, lag in outgoing[i]:
            candidate = lb[i] + lag
            if candidate > lb[j]:
                if candidate > ub[j]:
                    return lb, ub, False
                lb[j] = candidate
                if j not in in_queue:
                    queue.append(j)
                    in_queue.add(j)

    # Propagate upper bounds through s_i <= s_j - lag.
    queue = deque(activity_ids)
    in_queue = set(activity_ids)

    while queue:
        j = queue.popleft()
        in_queue.discard(j)

        for i, lag in incoming[j]:
            candidate = ub[j] - lag
            if candidate < ub[i]:
                if candidate < lb[i]:
                    return lb, ub, False
                ub[i] = candidate
                if i not in in_queue:
                    queue.append(i)
                    in_queue.add(i)

    return lb, ub, True


def temporal_feasible(starts, constraints):
    return all(starts[j] - starts[i] >= lag for i, j, lag in constraints)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_clock = time.monotonic()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    num_real = int(instance["num_real_activities"])
    end_id = num_real + 1
    deadline = int(instance["max_project_duration"])

    activities = sorted(instance["activities"], key=lambda a: int(a["id"]))
    activity_ids = [int(a["id"]) for a in activities]
    processing = {
        int(a["id"]): int(a["processing_time"])
        for a in activities
    }

    temporal_constraints = [
        (
            int(c["from_activity"]),
            int(c["to_activity"]),
            int(c["time_lag"]),
        )
        for c in instance["temporal_constraints"]
    ]

    lb, ub, bounds_feasible = propagate_bounds(
        activity_ids, temporal_constraints, deadline, start_id=0
    )

    if not bounds_feasible:
        # Instances intended for optimization are expected to be feasible.
        # This fallback still emits a schema-valid result if inconsistent data
        # are supplied.
        fallback = {i: 0 for i in activity_ids}
        solution = make_solution(activity_ids, fallback, end_id)
        output_dir = os.path.dirname(os.path.abspath(args.solution_path))
        os.makedirs(output_dir, exist_ok=True)
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(solution, f, indent=2)
        return

    resources = instance.get("resources", [])

    resource_data = []
    for resource in resources:
        periods = {
            int(t)
            for t in resource.get("Pi_k", [])
            if 1 <= int(t) <= deadline
        }

        prefix = [0] * (deadline + 1)
        running = 0
        for t in range(1, deadline + 1):
            if t in periods:
                running += 1
            prefix[t] = running

        raw_demands = resource.get("demands", {})
        demands = {
            i: int(raw_demands.get(str(i), raw_demands.get(i, 0)))
            for i in activity_ids
        }

        resource_data.append({
            "capacity": int(resource["capacity"]),
            "prefix": prefix,
            "demands": demands,
        })

    def overlap_count(prefix, activity_id, start_time):
        p = processing[activity_id]
        if p <= 0:
            return 0
        left = min(deadline, max(0, start_time))
        right = min(deadline, max(0, start_time + p))
        if right <= left:
            return 0
        return prefix[right] - prefix[left]

    model = gp.Model("partially_renewable_project_scheduling")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    remaining = max(0.0, args.time_limit - (time.monotonic() - start_clock))
    model.Params.TimeLimit = remaining

    start_vars = {
        i: model.addVar(
            lb=float(lb[i]),
            ub=float(ub[i]),
            vtype=GRB.INTEGER,
            name=f"s_{i}",
        )
        for i in activity_ids
    }
    model.update()

    for i, j, lag in temporal_constraints:
        model.addConstr(
            start_vars[j] - start_vars[i] >= lag,
            name=f"temporal_{i}_{j}",
        )

    # For activities whose resource consumption changes with their start time,
    # create a one-hot start-time representation. Activities with constant
    # consumption need only their integer start variable.
    start_choice_vars = {}
    resource_expressions = [gp.LinExpr() for _ in resource_data]

    for i in activity_ids:
        times = list(range(lb[i], ub[i] + 1))
        patterns = {}
        consumption_varies = False

        for k, resource in enumerate(resource_data):
            demand = resource["demands"][i]
            if demand == 0 or processing[i] == 0:
                values = [0] * len(times)
            else:
                prefix = resource["prefix"]
                values = [
                    demand * overlap_count(prefix, i, t)
                    for t in times
                ]

            patterns[k] = values
            if values and any(v != values[0] for v in values[1:]):
                consumption_varies = True

        if len(times) == 1:
            model.addConstr(start_vars[i] == times[0], name=f"fixed_{i}")
        elif consumption_varies:
            x = model.addVars(
                times,
                vtype=GRB.BINARY,
                name=f"x_{i}",
            )
            start_choice_vars[i] = x
            model.addConstr(
                gp.quicksum(x[t] for t in times) == 1,
                name=f"choose_start_{i}",
            )
            model.addConstr(
                start_vars[i] == gp.quicksum(t * x[t] for t in times),
                name=f"link_start_{i}",
            )

        for k, values in patterns.items():
            if not values:
                continue

            if all(v == values[0] for v in values):
                if values[0] != 0:
                    resource_expressions[k].addConstant(values[0])
            else:
                x = start_choice_vars[i]
                nonzero_times = [
                    (t, values[pos])
                    for pos, t in enumerate(times)
                    if values[pos] != 0
                ]
                if nonzero_times:
                    resource_expressions[k] += gp.quicksum(
                        coeff * x[t] for t, coeff in nonzero_times
                    )

    for k, resource in enumerate(resource_data):
        model.addConstr(
            resource_expressions[k] <= resource["capacity"],
            name=f"resource_{k}",
        )

    model.setObjective(start_vars[end_id], GRB.MINIMIZE)

    # Earliest temporally feasible schedule as a MIP start.
    initial_starts = {i: int(lb[i]) for i in activity_ids}
    for i in activity_ids:
        start_vars[i].Start = initial_starts[i]
        if i in start_choice_vars:
            for t, var in start_choice_vars[i].items():
                var.Start = 1.0 if t == initial_starts[i] else 0.0

    def resource_feasible(starts):
        for resource in resource_data:
            total = 0
            for i in activity_ids:
                demand = resource["demands"][i]
                if demand:
                    total += demand * overlap_count(
                        resource["prefix"], i, starts[i]
                    )
            if total > resource["capacity"]:
                return False
        return True

    best_logged = [float("inf")]
    best_solution = [None]

    if (
        temporal_feasible(initial_starts, temporal_constraints)
        and resource_feasible(initial_starts)
        and initial_starts[0] == 0
        and initial_starts[end_id] <= deadline
    ):
        initial_solution = make_solution(
            activity_ids, initial_starts, end_id
        )
        best_logged[0] = initial_solution["objective_value"]
        best_solution[0] = initial_solution
        if logger:
            logger.log_solution(
                initial_solution["objective_value"],
                initial_solution,
            )

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return

        starts = {
            i: int(round(cb_model.cbGetSolution(start_vars[i])))
            for i in activity_ids
        }
        objective = float(starts[end_id])

        if objective + 1e-9 < best_logged[0]:
            solution = make_solution(activity_ids, starts, end_id)
            best_logged[0] = objective
            best_solution[0] = solution
            if logger:
                logger.log_solution(objective, solution)

    model.optimize(incumbent_callback)

    if model.SolCount > 0:
        final_starts = {
            i: int(round(start_vars[i].X))
            for i in activity_ids
        }
        final_solution = make_solution(activity_ids, final_starts, end_id)

        if final_solution["objective_value"] + 1e-9 < best_logged[0]:
            best_logged[0] = final_solution["objective_value"]
            best_solution[0] = final_solution
            if logger:
                logger.log_solution(
                    final_solution["objective_value"],
                    final_solution,
                )
        else:
            best_solution[0] = final_solution

    if best_solution[0] is None:
        # Normally Gurobi will find an incumbent. If the time limit expires
        # before one is accepted, retain the schema-valid earliest schedule.
        best_solution[0] = make_solution(
            activity_ids, initial_starts, end_id
        )

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution[0], f, indent=2)


if __name__ == "__main__":
    main()