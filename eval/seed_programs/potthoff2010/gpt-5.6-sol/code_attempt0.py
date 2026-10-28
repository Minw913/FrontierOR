import argparse
import heapq
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


SOURCE = "__SOURCE__"
SINK = "__SINK__"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def directed_edges(items, time_key):
    edges = {}
    for item in items:
        a = item["from_station"]
        b = item["to_station"]
        t = int(item[time_key])
        edges[(a, b)] = min(edges.get((a, b), math.inf), t)
        if item.get("bidirectional", False):
            edges[(b, a)] = min(edges.get((b, a), math.inf), t)
    return edges


def all_pairs_shortest(station_ids, edge_times):
    adjacency = {s: [] for s in station_ids}
    for (a, b), travel_time in edge_times.items():
        if a in adjacency and b in adjacency:
            adjacency[a].append((b, travel_time))

    distances = {}
    for source in station_ids:
        dist = {source: 0}
        queue = [(0, source)]
        while queue:
            current, node = heapq.heappop(queue)
            if current != dist.get(node):
                continue
            for nxt, weight in adjacency[node]:
                candidate = current + weight
                if candidate < dist.get(nxt, math.inf):
                    dist[nxt] = candidate
                    heapq.heappush(queue, (candidate, nxt))
        distances[source] = dist
    return distances


def task_required_stations(task, lines):
    line = lines.get(task.get("line_id"))
    departure = task["departure_station"]
    arrival = task["arrival_station"]

    if not line or departure == arrival:
        return {departure, arrival}

    stations = line.get("stations", [])
    try:
        i = stations.index(departure)
        j = stations.index(arrival)
    except ValueError:
        return {departure, arrival}

    lo, hi = sorted((i, j))
    return set(stations[lo:hi + 1])


def derive_duty_start(duty, task_by_id, rescheduling_time):
    if duty.get("duty_type") == "reserve":
        return (
            duty.get("standby_station", duty["crew_base"]),
            max(rescheduling_time, int(duty.get("start_time_minutes", rescheduling_time))),
        )

    latest_task = None
    latest_arrival = -1

    detail_by_id = {
        detail["task_id"]: detail for detail in duty.get("task_details", [])
    }

    for task_id in duty.get("pre_disruption_tasks", []):
        task = task_by_id.get(task_id)
        if task is not None:
            arrival = int(task["arrival_time_minutes"])
            station = task["arrival_station"]
        else:
            detail = detail_by_id.get(task_id)
            if detail is None:
                continue
            hh, mm = map(int, detail["arrival_time"].split(":"))
            arrival = 60 * hh + mm
            station = detail["arrival_station"]

        if arrival > latest_arrival:
            latest_arrival = arrival
            latest_task = station

    if latest_task is not None:
        return latest_task, max(rescheduling_time, latest_arrival)

    return duty["crew_base"], max(
        rescheduling_time, int(duty.get("start_time_minutes", rescheduling_time))
    )


def main():
    args = parse_args()
    start_wall = time.monotonic()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    variant = instance.get("released_problem_variant")
    if variant != "relief_point_completion_v1":
        raise ValueError(f"Unsupported released_problem_variant: {variant!r}")

    parameters = instance["parameters"]
    network = instance["network"]
    stations = {s["id"]: s for s in network["stations"]}
    station_ids = list(stations)
    relief_points = {s["id"] for s in network["stations"] if s["is_relief_point"]}

    task_by_id = {task["task_id"]: task for task in instance["tasks"]}
    lines = {line["line_id"]: line for line in instance.get("train_lines", [])}
    duties = instance.get("active_duties", []) + instance.get("reserve_duties", [])
    duty_by_id = {d["duty_id"]: d for d in duties}

    rescheduling_time = int(instance["disruption"]["rescheduling_time_minutes"])
    pre_disruption = {
        task_id
        for duty in duties
        for task_id in duty.get("pre_disruption_tasks", [])
    }

    task_ids = {
        task["task_id"]
        for task in instance["tasks"]
        if int(task["departure_time_minutes"]) >= rescheduling_time
    }
    for duty in duties:
        task_ids.update(duty.get("tasks", []))
    task_ids.update(instance["disruption"].get("canceled_tasks", []))
    task_ids.difference_update(pre_disruption)
    task_ids.intersection_update(task_by_id)
    task_ids = sorted(
        task_ids,
        key=lambda tid: (
            int(task_by_id[tid]["departure_time_minutes"]),
            int(task_by_id[tid]["arrival_time_minutes"]),
            tid,
        ),
    )

    forced_canceled = set(instance["disruption"].get("canceled_tasks", []))
    forced_canceled.intersection_update(task_ids)

    rail_edges = directed_edges(
        network.get("route_segments", []), "travel_time_minutes"
    )
    rail_distances = all_pairs_shortest(station_ids, rail_edges)
    taxi_edges = directed_edges(
        network.get("taxi_connections", []), "travel_time_minutes"
    )

    def connection(from_station, to_station, available_minutes):
        """Return (travel time, uses taxi), preferring a cost-free rail move."""
        if from_station == to_station:
            return 0, False

        rail_time = rail_distances.get(from_station, {}).get(to_station, math.inf)
        if rail_time <= available_minutes:
            return int(rail_time), False

        taxi_time = taxi_edges.get((from_station, to_station), math.inf)
        if taxi_time <= available_minutes:
            return int(taxi_time), True

        return None

    min_connection = int(parameters.get("min_connection_time_minutes", 0))
    max_overtime = int(parameters.get("max_overtime_minutes", 0))
    max_duration = int(parameters.get("max_duty_duration_hours", 24)) * 60

    model = gp.Model("relief_point_completion_v1")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    cancel = {
        tid: model.addVar(vtype=GRB.BINARY, name=f"cancel[{tid}]")
        for tid in task_ids
    }

    x = {}
    changed = {}
    arc_vars = {}
    arc_info = {}
    duty_task_ids = {}
    duty_original = {}
    duty_start_data = {}
    duty_horizon = {}

    for duty in duties:
        did = duty["duty_id"]
        knowledge = set(duty.get("route_knowledge", []))
        licenses = set(duty.get("rolling_stock_licenses", []))
        original = [tid for tid in duty.get("tasks", []) if tid in task_ids]
        duty_original[did] = original

        start_station, ready_time = derive_duty_start(
            duty, task_by_id, rescheduling_time
        )
        horizon = min(
            int(duty["end_time_minutes"]) + max_overtime,
            int(duty["start_time_minutes"]) + max_duration,
        )
        duty_start_data[did] = (start_station, ready_time)
        duty_horizon[did] = horizon

        feasible_tasks = []
        for tid in task_ids:
            task = task_by_id[tid]
            if task["rolling_stock_type"] not in licenses:
                continue
            if not task_required_stations(task, lines).issubset(knowledge):
                continue
            if int(task["departure_time_minutes"]) < ready_time:
                continue
            if int(task["arrival_time_minutes"]) > horizon:
                continue
            feasible_tasks.append(tid)

        duty_task_ids[did] = feasible_tasks
        changed[did] = model.addVar(vtype=GRB.BINARY, name=f"changed[{did}]")

        for tid in feasible_tasks:
            x[(did, tid)] = model.addVar(
                vtype=GRB.BINARY, name=f"x[{did},{tid}]"
            )

        candidate_arcs = []

        # Source-to-task arcs.
        for tid in feasible_tasks:
            task = task_by_id[tid]
            gap = int(task["departure_time_minutes"]) - ready_time
            mode = connection(start_station, task["departure_station"], gap)
            if mode is not None:
                travel, taxi = mode
                candidate_arcs.append(
                    (
                        SOURCE,
                        tid,
                        {
                            "taxi": taxi,
                            "travel": travel,
                            "release": False,
                            "return_station": None,
                            "return_time": None,
                        },
                    )
                )

        # Task-to-task arcs.
        for i, first in enumerate(feasible_tasks):
            t1 = task_by_id[first]
            for second in feasible_tasks[i + 1:]:
                t2 = task_by_id[second]
                departure2 = int(t2["departure_time_minutes"])
                arrival1 = int(t1["arrival_time_minutes"])
                if departure2 < arrival1:
                    continue

                extra_connection = (
                    min_connection
                    if t1["rolling_stock_type"] != t2["rolling_stock_type"]
                    else 0
                )
                available = departure2 - arrival1 - extra_connection
                if available < 0:
                    continue

                mode = connection(
                    t1["arrival_station"], t2["departure_station"], available
                )
                if mode is not None:
                    travel, taxi = mode
                    candidate_arcs.append(
                        (
                            first,
                            second,
                            {
                                "taxi": taxi,
                                "travel": travel,
                                "release": False,
                                "return_station": None,
                                "return_time": None,
                            },
                        )
                    )

        def make_end_info(location, current_time):
            base = duty["crew_base"]
            remaining = horizon - current_time
            mode = connection(location, base, remaining)
            if mode is not None:
                travel, taxi = mode
                return {
                    "taxi": taxi,
                    "travel": travel,
                    "release": False,
                    "return_station": base,
                    "return_time": current_time + travel,
                }
            if location in relief_points:
                return {
                    "taxi": False,
                    "travel": 0,
                    "release": True,
                    "return_station": location,
                    "return_time": current_time,
                }
            return None

        empty_end = make_end_info(start_station, ready_time)
        if empty_end is not None:
            candidate_arcs.append((SOURCE, SINK, empty_end))

        for tid in feasible_tasks:
            task = task_by_id[tid]
            end_info = make_end_info(
                task["arrival_station"], int(task["arrival_time_minutes"])
            )
            if end_info is not None:
                candidate_arcs.append((tid, SINK, end_info))

        for u, v, info in candidate_arcs:
            key = (did, u, v)
            arc_vars[key] = model.addVar(
                vtype=GRB.BINARY, name=f"arc[{did},{u},{v}]"
            )
            arc_info[key] = info

        source_out = [
            var
            for (d, u, _), var in arc_vars.items()
            if d == did and u == SOURCE
        ]
        if not source_out:
            raise RuntimeError(f"Duty {did} has no feasible completion")
        model.addConstr(gp.quicksum(source_out) == 1, name=f"source[{did}]")

        for tid in feasible_tasks:
            incoming = [
                var
                for (d, _, v), var in arc_vars.items()
                if d == did and v == tid
            ]
            outgoing = [
                var
                for (d, u, _), var in arc_vars.items()
                if d == did and u == tid
            ]
            model.addConstr(
                gp.quicksum(incoming) == x[(did, tid)],
                name=f"inflow[{did},{tid}]",
            )
            model.addConstr(
                gp.quicksum(outgoing) == x[(did, tid)],
                name=f"outflow[{did},{tid}]",
            )

        sink_in = [
            var
            for (d, _, v), var in arc_vars.items()
            if d == did and v == SINK
        ]
        model.addConstr(gp.quicksum(sink_in) == 1, name=f"sink[{did}]")

        original_set = set(original)
        differences = []

        for tid in feasible_tasks:
            var = x[(did, tid)]
            if tid in original_set:
                model.addConstr(changed[did] >= 1 - var)
                differences.append(1 - var)
            else:
                model.addConstr(changed[did] >= var)
                differences.append(var)

        for tid in original:
            if tid not in feasible_tasks:
                model.addConstr(changed[did] == 1)
                differences.append(gp.LinExpr(1.0))

        original_pairs = set(zip(original, original[1:]))
        for key, var in arc_vars.items():
            d, u, v = key
            if d != did:
                continue
            info = arc_info[key]
            if info["taxi"] or info["release"]:
                model.addConstr(changed[did] >= var)
                differences.append(var)
            if u not in (SOURCE, SINK) and v not in (SOURCE, SINK):
                if (u, v) not in original_pairs:
                    model.addConstr(changed[did] >= var)
                    differences.append(var)

        if differences:
            model.addConstr(changed[did] <= gp.quicksum(differences))
        else:
            model.addConstr(changed[did] == 0)

    for tid in task_ids:
        covering = [
            x[(did, tid)]
            for did in duty_by_id
            if (did, tid) in x
        ]
        model.addConstr(
            cancel[tid] + gp.quicksum(covering) == 1,
            name=f"coverage[{tid}]",
        )
        if tid in forced_canceled:
            model.addConstr(cancel[tid] == 1, name=f"forced_cancel[{tid}]")

    cost_changed = float(parameters["cost_duty_changed"])
    cost_reassigned = float(parameters["cost_task_reassigned"])
    cost_transfer = float(parameters["cost_new_transfer"])
    cost_taxi = float(parameters["cost_taxi"])

    objective = gp.LinExpr()

    for tid in task_ids:
        task = task_by_id[tid]
        penalty = (
            float(parameters["cost_cancel_A_to_B"])
            if task["task_type"] == "A_to_B"
            else float(parameters["cost_cancel_A_to_A"])
        )
        objective += penalty * cancel[tid]

    for duty in duties:
        did = duty["duty_id"]
        original_set = set(duty_original[did])
        original_pairs = set(zip(duty_original[did], duty_original[did][1:]))

        objective += cost_changed * changed[did]

        for tid in duty_task_ids[did]:
            if tid not in original_set:
                objective += cost_reassigned * x[(did, tid)]

        for key, var in arc_vars.items():
            d, u, v = key
            if d != did:
                continue
            if arc_info[key]["taxi"]:
                objective += cost_taxi * var
            if (
                u not in (SOURCE, SINK)
                and v not in (SOURCE, SINK)
                and (u, v) not in original_pairs
            ):
                objective += cost_transfer * var

    model.setObjective(objective, GRB.MINIMIZE)

    def build_solution(value_getter):
        canceled_tasks = [
            tid for tid in task_ids if value_getter(cancel[tid]) > 0.5
        ]
        assignments = {}

        for duty in duties:
            did = duty["duty_id"]
            successor = {}
            selected_keys = []

            for key, var in arc_vars.items():
                d, u, v = key
                if d == did and value_getter(var) > 0.5:
                    successor[u] = v
                    selected_keys.append(key)

            ordered_tasks = []
            node = successor.get(SOURCE, SINK)
            visited = set()
            while node != SINK and node not in visited:
                visited.add(node)
                ordered_tasks.append(node)
                node = successor.get(node, SINK)

            end_key = None
            for key in selected_keys:
                if key[2] == SINK:
                    end_key = key
                    break
            if end_key is None:
                raise RuntimeError(f"No selected end arc for duty {did}")

            end_info = arc_info[end_key]
            uses_taxi = any(arc_info[key]["taxi"] for key in selected_keys)
            taxi_count = sum(
                1 for key in selected_keys if arc_info[key]["taxi"]
            )

            original = duty_original[did]
            original_set = set(original)
            original_pairs = set(zip(original, original[1:]))
            reassigned_count = sum(
                1 for tid in ordered_tasks if tid not in original_set
            )
            new_transfer_count = sum(
                1
                for key in selected_keys
                if key[1] not in (SOURCE, SINK)
                and key[2] not in (SOURCE, SINK)
                and (key[1], key[2]) not in original_pairs
            )

            is_changed = value_getter(changed[did]) > 0.5
            completion_cost = (
                (cost_changed if is_changed else 0.0)
                + cost_reassigned * reassigned_count
                + cost_transfer * new_transfer_count
                + cost_taxi * taxi_count
            )

            assignments[did] = {
                "completion_index": 1 if is_changed else 0,
                "cost": float(completion_cost),
                "covered_tasks": ordered_tasks,
                "uses_taxi": bool(uses_taxi),
                "release_at_relief_point": bool(end_info["release"]),
                "return_station": end_info["return_station"],
                "return_time_minutes": int(end_info["return_time"]),
            }

        cancellation_cost = 0.0
        for tid in canceled_tasks:
            task = task_by_id[tid]
            cancellation_cost += (
                float(parameters["cost_cancel_A_to_B"])
                if task["task_type"] == "A_to_B"
                else float(parameters["cost_cancel_A_to_A"])
            )

        total = cancellation_cost + sum(
            assignment["cost"] for assignment in assignments.values()
        )

        return {
            "released_problem_variant": "relief_point_completion_v1",
            "objective_value": float(total),
            "canceled_tasks": canceled_tasks,
            "duty_assignments": assignments,
        }

    best_logged = [math.inf]

    def callback(cb_model, where):
        if where != GRB.Callback.MIPSOL or logger is None:
            return
        try:
            incumbent = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if incumbent + 1e-7 >= best_logged[0]:
                return
            solution = build_solution(
                lambda variable: cb_model.cbGetSolution(variable)
            )
            best_logged[0] = solution["objective_value"]
            logger.log_solution(solution["objective_value"], solution)
        except Exception:
            # Logging must not invalidate an otherwise feasible optimization run.
            pass

    elapsed = time.monotonic() - start_wall
    model.Params.TimeLimit = max(0.01, args.time_limit - elapsed)
    model.optimize(callback)

    if model.SolCount <= 0:
        raise RuntimeError("No feasible solution was found within the time limit")

    final_solution = build_solution(lambda variable: variable.X)

    if logger and final_solution["objective_value"] + 1e-7 < best_logged[0]:
        logger.log_solution(final_solution["objective_value"], final_solution)

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, indent=2, sort_keys=False)


if __name__ == "__main__":
    main()