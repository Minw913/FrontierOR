import argparse
import json
import math
import os
import random
import time
from collections import deque

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


def topological_order(ids, successors, indegree):
    degree = dict(indegree)
    queue = deque(sorted(i for i in ids if degree[i] == 0))
    order = []

    while queue:
        i = queue.popleft()
        order.append(i)
        for j in successors[i]:
            degree[j] -= 1
            if degree[j] == 0:
                queue.append(j)

    if len(order) != len(ids):
        raise ValueError("The precedence graph contains a directed cycle.")
    return order


def random_topological_order(ids, successors, indegree, rng):
    degree = dict(indegree)
    available = [i for i in ids if degree[i] == 0]
    order = []

    while available:
        pos = rng.randrange(len(available))
        i = available.pop(pos)
        order.append(i)
        for j in successors[i]:
            degree[j] -= 1
            if degree[j] == 0:
                available.append(j)

    return order


def compute_windows(ids, order, duration, predecessors, successors, deadline,
                    start_activity):
    earliest = {i: 0 for i in ids}
    earliest[start_activity] = 0

    for i in order:
        if i == start_activity:
            earliest[i] = 0
        elif predecessors[i]:
            earliest[i] = max(
                earliest[p] + duration[p] for p in predecessors[i]
            )

    latest = {i: deadline - duration[i] for i in ids}
    for i in reversed(order):
        if successors[i]:
            latest[i] = min(
                latest[j] - duration[i] for j in successors[i]
            )

    latest[start_activity] = 0
    earliest[start_activity] = 0

    for i in ids:
        if earliest[i] > latest[i]:
            raise ValueError(
                f"Instance is infeasible under the deadline: activity {i} "
                f"has window [{earliest[i]}, {latest[i]}]."
            )

    return earliest, latest


def resource_profile(schedule, ids, duration, requirements, num_resources,
                     deadline):
    loads = [[0] * max(0, deadline) for _ in range(num_resources)]

    for i in ids:
        d = duration[i]
        if d <= 0:
            continue
        start = schedule[i]
        finish = min(deadline, start + d)
        for k, req in enumerate(requirements[i]):
            if req == 0:
                continue
            for t in range(max(0, start), finish):
                loads[k][t] += req

    amounts = [max(row) if row else 0 for row in loads]
    return loads, amounts


def make_solution(schedule, amounts, costs):
    objective = float(sum(costs[k] * amounts[k] for k in range(len(costs))))
    return {
        "objective_value": objective,
        "start_times": {str(i): int(schedule[i]) for i in schedule},
        "resource_amounts": {
            str(k): int(amounts[k]) for k in range(len(amounts))
        },
    }


def validate_schedule(schedule, ids, duration, precedences, earliest, latest):
    for i in ids:
        s = schedule[i]
        if s < earliest[i] or s > latest[i]:
            return False
    for i, j in precedences:
        if schedule[j] < schedule[i] + duration[i]:
            return False
    return True


def main():
    args = parse_args()
    wall_start = time.monotonic()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    activities = data["activities"]
    num_resources = int(data["num_resource_types"])
    deadline = int(data["max_project_duration"])
    costs = [int(x) for x in data["resource_costs"]]

    ids = [int(a["id"]) for a in activities]
    id_set = set(ids)
    duration = {int(a["id"]): int(a["duration"]) for a in activities}
    requirements = {
        int(a["id"]): [int(x) for x in a["resource_requirements"]]
        for a in activities
    }

    precedences = [
        (int(edge[0]), int(edge[1]))
        for edge in data["precedence_relations"]
    ]

    predecessors = {i: [] for i in ids}
    successors = {i: [] for i in ids}
    indegree = {i: 0 for i in ids}

    for i, j in precedences:
        if i not in id_set or j not in id_set:
            raise ValueError("Precedence relation references an unknown activity.")
        successors[i].append(j)
        predecessors[j].append(i)
        indegree[j] += 1

    order = topological_order(ids, successors, indegree)

    dummy_candidates = [
        i for i in ids
        if duration[i] == 0 and all(r == 0 for r in requirements[i])
    ]
    source_candidates = [
        i for i in dummy_candidates if len(predecessors[i]) == 0
    ]
    sink_candidates = [
        i for i in dummy_candidates if len(successors[i]) == 0
    ]

    if source_candidates:
        start_activity = source_candidates[0]
    else:
        graph_sources = [i for i in ids if not predecessors[i]]
        if not graph_sources:
            raise ValueError("No project start activity could be identified.")
        start_activity = graph_sources[0]

    if sink_candidates:
        end_activity = sink_candidates[0]
    else:
        graph_sinks = [i for i in ids if not successors[i]]
        if not graph_sinks:
            raise ValueError("No project end activity could be identified.")
        end_activity = graph_sinks[0]

    earliest, latest = compute_windows(
        ids, order, duration, predecessors, successors, deadline,
        start_activity
    )

    rng = random.Random(0)
    best_solution = None
    best_objective = float("inf")

    def consider_schedule(schedule):
        nonlocal best_solution, best_objective
        if not validate_schedule(
            schedule, ids, duration, precedences, earliest, latest
        ):
            return
        _, amounts = resource_profile(
            schedule, ids, duration, requirements, num_resources, deadline
        )
        solution = make_solution(schedule, amounts, costs)
        objective = solution["objective_value"]
        if objective < best_objective - 1e-9:
            best_objective = objective
            best_solution = solution
            if logger:
                logger.log_solution(objective, solution)

    # Earliest-start and latest-start baseline schedules.
    consider_schedule(dict(earliest))
    consider_schedule(dict(latest))

    # Greedy randomized construction heuristics.
    heuristic_deadline = wall_start + min(
        max(0.0, args.time_limit * 0.15), 2.0
    )
    passes = 10

    for pass_index in range(passes):
        if time.monotonic() >= heuristic_deadline:
            break

        topo = order if pass_index == 0 else random_topological_order(
            ids, successors, indegree, rng
        )
        schedule = {}
        loads = [[0] * max(0, deadline) for _ in range(num_resources)]
        current_peaks = [0] * num_resources

        for i in topo:
            if i == start_activity:
                schedule[i] = 0
                continue

            low = earliest[i]
            if predecessors[i]:
                low = max(
                    low,
                    max(schedule[p] + duration[p] for p in predecessors[i])
                )
            high = latest[i]

            if low > high:
                schedule = None
                break

            d = duration[i]
            reqs = requirements[i]

            if d <= 0 or all(r == 0 for r in reqs):
                schedule[i] = low
                continue

            width = high - low + 1
            if width <= 80:
                candidates = list(range(low, high + 1))
            else:
                candidates = {low, high, (low + high) // 2}
                for _ in range(50):
                    candidates.add(rng.randint(low, high))
                candidates = list(candidates)

            rng.shuffle(candidates)
            chosen = low
            chosen_score = None

            for candidate in candidates:
                score = 0
                finish = candidate + d
                for k in range(num_resources):
                    req = reqs[k]
                    if req == 0:
                        peak = current_peaks[k]
                    else:
                        interval_peak = 0
                        for t in range(candidate, finish):
                            interval_peak = max(interval_peak, loads[k][t])
                        peak = max(current_peaks[k], interval_peak + req)
                    score += costs[k] * peak

                tie_break = candidate if pass_index % 2 == 0 else -candidate
                key = (score, tie_break)
                if chosen_score is None or key < chosen_score:
                    chosen_score = key
                    chosen = candidate

            schedule[i] = chosen
            finish = chosen + d
            for k, req in enumerate(reqs):
                if req == 0:
                    continue
                for t in range(chosen, finish):
                    loads[k][t] += req
                    if loads[k][t] > current_peaks[k]:
                        current_peaks[k] = loads[k][t]

        if schedule is not None:
            consider_schedule(schedule)

    if best_solution is None:
        raise RuntimeError("No feasible schedule was constructed.")

    elapsed = time.monotonic() - wall_start
    remaining = args.time_limit - elapsed

    # Activities requiring time-indexed variables.
    indexed_activities = [
        i for i in ids
        if duration[i] > 0 and any(r != 0 for r in requirements[i])
    ]

    variable_count = sum(
        latest[i] - earliest[i] + 1 for i in indexed_activities
    )
    estimated_nonzeros = sum(
        (latest[i] - earliest[i] + 1)
        * duration[i]
        * sum(1 for r in requirements[i] if r != 0)
        for i in indexed_activities
    )

    # Avoid spending the entire limit constructing an excessively large model.
    build_mip = (
        remaining > 0.15
        and variable_count <= 600000
        and estimated_nonzeros <= 12000000
    )

    if build_mip:
        try:
            model = gp.Model("resource_investment_scheduling")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1

            start_vars = {}
            for i in ids:
                start_vars[i] = model.addVar(
                    lb=earliest[i],
                    ub=latest[i],
                    vtype=GRB.INTEGER,
                    name=f"s_{i}",
                )

            x_vars = {}
            activity_x = {}
            for i in indexed_activities:
                activity_x[i] = []
                for t in range(earliest[i], latest[i] + 1):
                    var = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{t}")
                    x_vars[(i, t)] = var
                    activity_x[i].append((t, var))

            resource_vars = {}
            for k in range(num_resources):
                upper = sum(max(0, requirements[i][k]) for i in ids)
                individual_lb = max(
                    [requirements[i][k] for i in ids] + [0]
                )
                total_energy = sum(
                    duration[i] * requirements[i][k] for i in ids
                )
                energy_lb = (
                    math.ceil(total_energy / deadline) if deadline > 0 else 0
                )
                resource_vars[k] = model.addVar(
                    lb=max(individual_lb, energy_lb),
                    ub=max(upper, individual_lb, energy_lb),
                    vtype=GRB.INTEGER,
                    name=f"R_{k}",
                )

            model.update()

            for i in indexed_activities:
                model.addConstr(
                    gp.quicksum(var for _, var in activity_x[i]) == 1,
                    name=f"one_start_{i}",
                )
                model.addConstr(
                    start_vars[i]
                    == gp.quicksum(t * var for t, var in activity_x[i]),
                    name=f"start_link_{i}",
                )

            model.addConstr(start_vars[start_activity] == 0, name="project_start")
            model.addConstr(
                start_vars[end_activity] <= deadline,
                name="project_deadline",
            )

            for i, j in precedences:
                model.addConstr(
                    start_vars[j] >= start_vars[i] + duration[i],
                    name=f"prec_{i}_{j}",
                )

            for k in range(num_resources):
                for tau in range(deadline):
                    terms = []
                    for i in indexed_activities:
                        req = requirements[i][k]
                        if req == 0:
                            continue
                        first = max(earliest[i], tau - duration[i] + 1)
                        last = min(latest[i], tau)
                        if first <= last:
                            terms.extend(
                                req * x_vars[(i, t)]
                                for t in range(first, last + 1)
                            )
                    if terms:
                        model.addConstr(
                            gp.quicksum(terms) <= resource_vars[k],
                            name=f"capacity_{k}_{tau}",
                        )

            model.setObjective(
                gp.quicksum(
                    costs[k] * resource_vars[k] for k in range(num_resources)
                ),
                GRB.MINIMIZE,
            )

            incumbent_starts = {
                int(i): int(t)
                for i, t in best_solution["start_times"].items()
            }
            incumbent_amounts = {
                int(k): int(v)
                for k, v in best_solution["resource_amounts"].items()
            }

            for i in ids:
                start_vars[i].Start = incumbent_starts[i]
            for i in indexed_activities:
                selected = incumbent_starts[i]
                for t, var in activity_x[i]:
                    var.Start = 1.0 if t == selected else 0.0
            for k in range(num_resources):
                resource_vars[k].Start = incumbent_amounts[k]

            callback_state = {
                "best_objective": best_objective,
                "best_solution": best_solution,
            }

            def incumbent_callback(cb_model, where):
                if where != GRB.Callback.MIPSOL:
                    return
                try:
                    values = cb_model.cbGetSolution(
                        [start_vars[i] for i in ids]
                    )
                    schedule = {
                        i: int(round(values[pos]))
                        for pos, i in enumerate(ids)
                    }

                    if not validate_schedule(
                        schedule, ids, duration, precedences,
                        earliest, latest
                    ):
                        return

                    _, amounts = resource_profile(
                        schedule, ids, duration, requirements,
                        num_resources, deadline
                    )
                    solution = make_solution(schedule, amounts, costs)
                    objective = solution["objective_value"]

                    if objective < callback_state["best_objective"] - 1e-9:
                        callback_state["best_objective"] = objective
                        callback_state["best_solution"] = solution
                        if logger:
                            logger.log_solution(objective, solution)
                except Exception:
                    # Never interrupt optimization because of callback-side
                    # extraction or logging errors.
                    return

            remaining = args.time_limit - (time.monotonic() - wall_start)
            if remaining > 0.05:
                model.Params.TimeLimit = max(0.01, remaining)
                model.optimize(incumbent_callback)

                if callback_state["best_objective"] < best_objective - 1e-9:
                    best_objective = callback_state["best_objective"]
                    best_solution = callback_state["best_solution"]

                if model.SolCount > 0:
                    schedule = {
                        i: int(round(start_vars[i].X)) for i in ids
                    }
                    consider_schedule(schedule)

        except gp.GurobiError:
            # The heuristic incumbent remains available if model construction
            # or optimization cannot be completed.
            pass

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution, f, indent=2)


if __name__ == "__main__":
    main()