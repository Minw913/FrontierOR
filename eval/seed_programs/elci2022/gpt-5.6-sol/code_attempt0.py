import argparse
import json
import math
import random
import time
from collections import defaultdict

from solution_logger import SolutionLogger


EPS = 1e-9


def build_solution(instance, assignment, schedule):
    probabilities = [
        float(scenario["probability"]) for scenario in instance["scenarios"]
    ]

    expected_makespan = 0.0
    for s in range(instance["num_scenarios"]):
        makespan = 0
        for j in range(instance["n"]):
            makespan = max(makespan, int(schedule[str(s)][str(j)]["finish"]))
        expected_makespan += probabilities[s] * makespan

    return {
        "objective_value": float(expected_makespan),
        "assignment": {str(j): int(assignment[j]) for j in range(instance["n"])},
        "schedule": schedule,
    }


def earliest_feasible_start(release, duration, demand, capacity, placed):
    """
    Find an earliest feasible integer start time against a list of already
    placed intervals (start, finish, demand). The capacity profile changes
    only at interval endpoints.
    """
    release = int(release)
    duration = int(duration)
    demand = int(demand)

    if duration <= 0 or demand <= 0:
        return release

    t = release
    while True:
        finish = t + duration
        points = {t, finish}

        for st, fi, _ in placed:
            if t < st < finish:
                points.add(st)
            if t < fi < finish:
                points.add(fi)

        points = sorted(points)
        violating_time = None
        jump_to = None

        for k in range(len(points) - 1):
            a = points[k]
            b = points[k + 1]
            if b <= a:
                continue

            active = []
            usage = demand
            for st, fi, req in placed:
                if st <= a < fi:
                    usage += req
                    active.append(fi)

            if usage > capacity:
                violating_time = a
                if active:
                    jump_to = min(active)
                else:
                    # This can only occur if the job itself exceeds capacity.
                    jump_to = t + 1
                break

        if violating_time is None:
            return t

        if jump_to is None or jump_to <= t:
            t += 1
        else:
            t = int(jump_to)


def schedule_facility_jobs(instance, scenario_index, facility, jobs, mode):
    release_times = instance["release_times"]
    demands = instance["capacity_requirements"][facility]
    capacity = instance["facility_capacity_limits"][facility]
    processing = instance["scenarios"][scenario_index]["processing_times"][facility]

    if mode == 0:
        ordered = sorted(
            jobs,
            key=lambda j: (
                release_times[j],
                -(demands[j] * processing[j]),
                -processing[j],
                j,
            ),
        )
    elif mode == 1:
        ordered = sorted(
            jobs,
            key=lambda j: (
                release_times[j],
                -demands[j],
                -processing[j],
                j,
            ),
        )
    elif mode == 2:
        ordered = sorted(
            jobs,
            key=lambda j: (
                release_times[j],
                -processing[j],
                -(demands[j] * processing[j]),
                j,
            ),
        )
    else:
        ordered = sorted(
            jobs,
            key=lambda j: (
                -(demands[j] * processing[j]),
                release_times[j],
                j,
            ),
        )

    placed = []
    result = {}
    facility_finish = 0

    for j in ordered:
        p = int(processing[j])
        q = int(demands[j])
        r = int(release_times[j])
        start = earliest_feasible_start(r, p, q, capacity, placed)
        finish = start + p

        result[j] = (start, finish)
        if p > 0 and q > 0:
            placed.append((start, finish, q))
        facility_finish = max(facility_finish, finish)

    return result, facility_finish


def heuristic_schedule(instance, assignment, use_all_modes=True):
    n = instance["n"]
    m = instance["m"]
    num_scenarios = instance["num_scenarios"]

    jobs_by_facility = [[] for _ in range(m)]
    for j in range(n):
        jobs_by_facility[assignment[j]].append(j)

    schedule = {}

    for s in range(num_scenarios):
        scenario_schedule = {}
        for i in range(m):
            jobs = jobs_by_facility[i]
            if not jobs:
                continue

            modes = range(4) if use_all_modes else range(2)
            best_result = None
            best_finish = math.inf

            for mode in modes:
                result, facility_finish = schedule_facility_jobs(
                    instance, s, i, jobs, mode
                )
                if facility_finish < best_finish:
                    best_finish = facility_finish
                    best_result = result

            for j, (start, finish) in best_result.items():
                scenario_schedule[str(j)] = {
                    "facility": int(i),
                    "start": int(start),
                    "finish": int(finish),
                }

        # Every job must occur, including the n=0 edge case.
        schedule[str(s)] = scenario_schedule

    return build_solution(instance, assignment, schedule)


def feasible_facilities(instance):
    n = instance["n"]
    m = instance["m"]
    capacities = instance["facility_capacity_limits"]
    requirements = instance["capacity_requirements"]

    feasible = []
    for j in range(n):
        options = [
            i for i in range(m)
            if requirements[i][j] <= capacities[i]
        ]
        if not options:
            raise ValueError(
                f"Job {j} cannot be assigned to any facility because its "
                f"capacity requirement exceeds every facility capacity."
            )
        feasible.append(options)
    return feasible


def construct_greedy_assignment(instance, feasible, variant, rng):
    n = instance["n"]
    m = instance["m"]
    scenarios = instance["scenarios"]
    probabilities = [float(s["probability"]) for s in scenarios]
    capacities = instance["facility_capacity_limits"]
    requirements = instance["capacity_requirements"]
    releases = instance["release_times"]
    num_scenarios = instance["num_scenarios"]

    expected_p = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            expected_p[i][j] = sum(
                probabilities[s] *
                scenarios[s]["processing_times"][i][j]
                for s in range(num_scenarios)
            )

    hardness = []
    for j in range(n):
        min_ep = min(expected_p[i][j] for i in feasible[j])
        max_area = max(
            expected_p[i][j] * requirements[i][j] / max(1, capacities[i])
            for i in feasible[j]
        )
        hardness.append(
            (
                len(feasible[j]),
                -releases[j],
                -max_area,
                -min_ep,
                j,
            )
        )

    if variant == 0:
        order = [x[-1] for x in sorted(hardness)]
    elif variant == 1:
        order = sorted(
            range(n),
            key=lambda j: (
                len(feasible[j]),
                -min(
                    expected_p[i][j] *
                    requirements[i][j] / max(1, capacities[i])
                    for i in feasible[j]
                ),
                -releases[j],
                j,
            ),
        )
    else:
        order = list(range(n))
        rng.shuffle(order)
        order.sort(key=lambda j: len(feasible[j]))

    loads = [[0.0] * num_scenarios for _ in range(m)]
    assignment = [-1] * n

    for j in order:
        best_i = None
        best_score = math.inf

        for i in feasible[j]:
            scenario_scores = []
            for s in range(num_scenarios):
                p = scenarios[s]["processing_times"][i][j]
                q = requirements[i][j]
                if variant % 2 == 0:
                    increment = p * q / max(1, capacities[i])
                    if 2 * q > capacities[i]:
                        increment = max(increment, float(p))
                else:
                    increment = float(p) if 2 * q > capacities[i] else (
                        p * q / max(1, capacities[i])
                    )

                projected_i = loads[i][s] + increment
                projected_max = projected_i
                for k in range(m):
                    if k != i:
                        projected_max = max(projected_max, loads[k][s])
                projected_max = max(projected_max, releases[j] + p)
                scenario_scores.append(projected_max)

            score = sum(
                probabilities[s] * scenario_scores[s]
                for s in range(num_scenarios)
            )
            score += rng.random() * (1e-6 if variant < 2 else 0.02 * max(1.0, score))

            if score < best_score:
                best_score = score
                best_i = i

        assignment[j] = best_i
        for s in range(num_scenarios):
            p = scenarios[s]["processing_times"][best_i][j]
            q = requirements[best_i][j]
            increment = p * q / max(1, capacities[best_i])
            if 2 * q > capacities[best_i]:
                increment = max(increment, float(p))
            loads[best_i][s] += increment

    return assignment


def estimate_time_indexed_size(instance, feasible):
    n = instance["n"]
    m = instance["m"]
    num_scenarios = instance["num_scenarios"]
    max_release = max(instance["release_times"], default=0)

    horizons = []
    y_count = 0
    usage_count = 0

    for s in range(num_scenarios):
        processing = instance["scenarios"][s]["processing_times"]
        horizon = max_release
        for j in range(n):
            horizon += max(processing[i][j] for i in feasible[j])
        horizons.append(int(horizon))

        usage_count += m * max(0, int(horizon))
        for j in range(n):
            r = max(0, int(instance["release_times"][j]))
            for i in feasible[j]:
                p = int(processing[i][j])
                y_count += max(0, int(horizon) - p - r + 1)

    return horizons, y_count, usage_count


def solve_time_indexed(
    instance,
    feasible,
    horizons,
    initial_solution,
    deadline,
    logger,
    best_box,
):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return

    remaining = deadline - time.monotonic()
    if remaining <= 0.05:
        return

    n = instance["n"]
    m = instance["m"]
    num_scenarios = instance["num_scenarios"]
    capacities = instance["facility_capacity_limits"]
    requirements = instance["capacity_requirements"]
    releases = instance["release_times"]
    probabilities = [
        float(s["probability"]) for s in instance["scenarios"]
    ]

    try:
        model = gp.Model("stochastic_cumulative_scheduling")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(0.01, deadline - time.monotonic())

        x = {}
        for j in range(n):
            for i in feasible[j]:
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

        y = {}
        y_list = []
        y_meta = []
        starts_at = [
            [defaultdict(list) for _ in range(m)]
            for _ in range(num_scenarios)
        ]
        finishes_at = [
            [defaultdict(list) for _ in range(m)]
            for _ in range(num_scenarios)
        ]

        for s in range(num_scenarios):
            processing = instance["scenarios"][s]["processing_times"]
            horizon = horizons[s]
            for j in range(n):
                r = max(0, int(releases[j]))
                for i in feasible[j]:
                    p = int(processing[i][j])
                    latest = horizon - p
                    for t in range(r, latest + 1):
                        var = model.addVar(
                            vtype=GRB.BINARY,
                            name=f"y_{s}_{i}_{j}_{t}",
                        )
                        y[s, i, j, t] = var
                        y_list.append(var)
                        y_meta.append((s, i, j, t, p))
                        starts_at[s][i][t].append(
                            (int(requirements[i][j]), var)
                        )
                        finishes_at[s][i][t + p].append(
                            (int(requirements[i][j]), var)
                        )

        cmax = {
            s: model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"C_{s}")
            for s in range(num_scenarios)
        }

        model.update()

        for j in range(n):
            model.addConstr(
                gp.quicksum(x[i, j] for i in feasible[j]) == 1,
                name=f"assign_{j}",
            )

        for s in range(num_scenarios):
            processing = instance["scenarios"][s]["processing_times"]
            horizon = horizons[s]

            for j in range(n):
                for i in feasible[j]:
                    p = int(processing[i][j])
                    r = max(0, int(releases[j]))
                    model.addConstr(
                        gp.quicksum(
                            y[s, i, j, t]
                            for t in range(r, horizon - p + 1)
                        ) == x[i, j],
                        name=f"link_{s}_{i}_{j}",
                    )

                completion_expr = gp.LinExpr()
                for i in feasible[j]:
                    p = int(processing[i][j])
                    r = max(0, int(releases[j]))
                    for t in range(r, horizon - p + 1):
                        completion_expr += (t + p) * y[s, i, j, t]
                model.addConstr(
                    cmax[s] >= completion_expr,
                    name=f"completion_{s}_{j}",
                )

            for i in range(m):
                previous_usage = None
                for t in range(max(0, horizon)):
                    usage = model.addVar(
                        lb=0.0,
                        ub=float(capacities[i]),
                        vtype=GRB.CONTINUOUS,
                        name=f"usage_{s}_{i}_{t}",
                    )

                    expr = gp.LinExpr()
                    if previous_usage is not None:
                        expr += previous_usage
                    for q, var in starts_at[s][i].get(t, ()):
                        expr += q * var
                    for q, var in finishes_at[s][i].get(t, ()):
                        expr -= q * var

                    model.addConstr(usage == expr)
                    previous_usage = usage

        model.setObjective(
            gp.quicksum(probabilities[s] * cmax[s] for s in range(num_scenarios)),
            GRB.MINIMIZE,
        )

        # Warm start from the heuristic incumbent.
        initial_assignment = {
            int(j): int(i)
            for j, i in initial_solution["assignment"].items()
        }
        selected_starts = {}
        for s_key, scenario_schedule in initial_solution["schedule"].items():
            s = int(s_key)
            for j_key, item in scenario_schedule.items():
                j = int(j_key)
                selected_starts[(s, int(item["facility"]), j, int(item["start"]))] = 1

        for (i, j), var in x.items():
            var.Start = 1.0 if initial_assignment[j] == i else 0.0
        for key, var in y.items():
            var.Start = 1.0 if key in selected_starts else 0.0
        for s in range(num_scenarios):
            scenario_makespan = max(
                (
                    item["finish"]
                    for item in initial_solution["schedule"][str(s)].values()
                ),
                default=0,
            )
            cmax[s].Start = float(scenario_makespan)

        model.update()

        callback_state = {
            "last_callback_objective": math.inf,
        }

        def extract_callback_solution(cb_model):
            x_values = cb_model.cbGetSolution(list(x.values()))
            assignment = [-1] * n
            for ((i, j), value) in zip(x.keys(), x_values):
                if value > 0.5:
                    assignment[j] = i

            y_values = cb_model.cbGetSolution(y_list)
            schedule = {str(s): {} for s in range(num_scenarios)}
            for value, (s, i, j, t, p) in zip(y_values, y_meta):
                if value > 0.5:
                    schedule[str(s)][str(j)] = {
                        "facility": int(i),
                        "start": int(t),
                        "finish": int(t + p),
                    }

            if any(i < 0 for i in assignment):
                return None
            if any(len(schedule[str(s)]) != n for s in range(num_scenarios)):
                return None
            return build_solution(instance, assignment, schedule)

        def callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return
            try:
                obj = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
                if obj >= callback_state["last_callback_objective"] - 1e-8:
                    return
                callback_state["last_callback_objective"] = obj

                solution = extract_callback_solution(cb_model)
                if solution is None:
                    if logger:
                        logger.log(obj)
                    return

                actual_obj = solution["objective_value"]
                if actual_obj < best_box["objective"] - 1e-8:
                    best_box["objective"] = actual_obj
                    best_box["solution"] = solution
                    if logger:
                        logger.log_solution(actual_obj, solution)
            except Exception:
                # Logging/extraction errors must not interrupt optimization.
                pass

        model.optimize(callback)

        if model.SolCount > 0:
            assignment = [-1] * n
            for (i, j), var in x.items():
                if var.X > 0.5:
                    assignment[j] = i

            schedule = {str(s): {} for s in range(num_scenarios)}
            for var, (s, i, j, t, p) in zip(y_list, y_meta):
                if var.X > 0.5:
                    schedule[str(s)][str(j)] = {
                        "facility": int(i),
                        "start": int(t),
                        "finish": int(t + p),
                    }

            if (
                all(i >= 0 for i in assignment)
                and all(len(schedule[str(s)]) == n for s in range(num_scenarios))
            ):
                solution = build_solution(instance, assignment, schedule)
                if solution["objective_value"] < best_box["objective"] - 1e-8:
                    best_box["objective"] = solution["objective_value"]
                    best_box["solution"] = solution
                    if logger:
                        logger.log_solution(
                            solution["objective_value"], solution
                        )

    except Exception:
        # Preserve and return the heuristic incumbent if model construction,
        # licensing, or optimization fails.
        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    n = int(instance["n"])
    m = int(instance["m"])
    num_scenarios = int(instance["num_scenarios"])

    if len(instance["scenarios"]) != num_scenarios:
        raise ValueError("num_scenarios does not match the scenarios list length.")

    feasible = feasible_facilities(instance)
    rng = random.Random(0)

    best_box = {
        "objective": math.inf,
        "solution": None,
    }

    # Handle the empty-job case directly.
    if n == 0:
        empty_solution = {
            "objective_value": 0.0,
            "assignment": {},
            "schedule": {str(s): {} for s in range(num_scenarios)},
        }
        best_box["objective"] = 0.0
        best_box["solution"] = empty_solution
        if logger:
            logger.log_solution(0.0, empty_solution)
    else:
        # Generate several structurally different greedy assignments.
        candidate_count = min(10, max(3, 2 + int(math.sqrt(n))))
        assignments_seen = set()

        for variant in range(candidate_count):
            if time.monotonic() >= deadline:
                break

            assignment = construct_greedy_assignment(
                instance, feasible, variant, rng
            )
            key = tuple(assignment)
            if key in assignments_seen:
                continue
            assignments_seen.add(key)

            solution = heuristic_schedule(
                instance,
                assignment,
                use_all_modes=(variant < 3),
            )
            objective = solution["objective_value"]

            if objective < best_box["objective"] - 1e-8:
                best_box["objective"] = objective
                best_box["solution"] = solution
                if logger:
                    logger.log_solution(objective, solution)

        # There is always at least one feasible assignment after the
        # feasible-facility check, even under an extremely small time limit.
        if best_box["solution"] is None:
            assignment = [options[0] for options in feasible]
            solution = heuristic_schedule(instance, assignment, use_all_modes=False)
            best_box["objective"] = solution["objective_value"]
            best_box["solution"] = solution
            if logger:
                logger.log_solution(solution["objective_value"], solution)

        horizons, y_count, usage_count = estimate_time_indexed_size(
            instance, feasible
        )

        total_estimated_vars = y_count + usage_count + n * m
        dynamic_limit = min(
            500000,
            max(40000, 20000 + 7000 * max(1, args.time_limit)),
        )

        use_mip = (
            total_estimated_vars <= dynamic_limit
            and max(horizons, default=0) <= 200000
            and deadline - time.monotonic() > 0.1
        )

        # For large instances, use the available time for assignment moves.
        if not use_mip:
            assignment = [
                int(best_box["solution"]["assignment"][str(j)])
                for j in range(n)
            ]

            improved = True
            while improved and time.monotonic() < deadline:
                improved = False
                job_order = list(range(n))
                rng.shuffle(job_order)

                for j in job_order:
                    if time.monotonic() >= deadline:
                        break
                    old_i = assignment[j]

                    alternatives = list(feasible[j])
                    rng.shuffle(alternatives)
                    for new_i in alternatives:
                        if new_i == old_i or time.monotonic() >= deadline:
                            continue

                        trial = assignment[:]
                        trial[j] = new_i
                        solution = heuristic_schedule(
                            instance, trial, use_all_modes=False
                        )
                        objective = solution["objective_value"]

                        if objective < best_box["objective"] - 1e-8:
                            assignment = trial
                            best_box["objective"] = objective
                            best_box["solution"] = solution
                            improved = True
                            if logger:
                                logger.log_solution(objective, solution)
                            break

                    if improved:
                        break
        else:
            solve_time_indexed(
                instance=instance,
                feasible=feasible,
                horizons=horizons,
                initial_solution=best_box["solution"],
                deadline=deadline,
                logger=logger,
                best_box=best_box,
            )

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_box["solution"], f, indent=2)


if __name__ == "__main__":
    main()