import argparse
import bisect
import json
import math
import os
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def normalized_key(value):
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def recursively_find_number(data, aliases, default=None):
    aliases = {normalized_key(x) for x in aliases}

    def visit(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if normalized_key(key) in aliases and isinstance(value, (int, float)) and not isinstance(value, bool):
                    return float(value)
            for value in obj.values():
                answer = visit(value)
                if answer is not None:
                    return answer
        elif isinstance(obj, list):
            for value in obj:
                answer = visit(value)
                if answer is not None:
                    return answer
        return None

    result = visit(data)
    return default if result is None else result


def get_window_value(records, window_index, field, default=0.0):
    if not records:
        return float(default)

    exact = None
    fallback = None
    for record in records:
        if not isinstance(record, dict):
            continue
        if fallback is None and field in record:
            fallback = record[field]
        if int(record.get("time_window_index", -1)) == int(window_index):
            exact = record.get(field, default)
            break

    value = exact if exact is not None else fallback
    return float(default if value is None else value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_clock = time.monotonic()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as file:
        instance = json.load(file)

    trips_list = instance.get("potential_trips", [])
    trips = {int(t["id"]): t for t in trips_list}
    trip_ids = list(trips.keys())

    patterns = []
    for direction in instance.get("directions", []):
        pattern_id = str(direction["pattern_id"])
        if pattern_id not in patterns:
            patterns.append(pattern_id)
    for trip in trips_list:
        pattern_id = str(trip["pattern_id"])
        if pattern_id not in patterns:
            patterns.append(pattern_id)

    trips_by_pattern = defaultdict(list)
    for trip_id, trip in trips.items():
        trips_by_pattern[str(trip["pattern_id"])].append(trip_id)

    for pattern_id in patterns:
        trips_by_pattern[pattern_id].sort(
            key=lambda i: (
                float(trips[i]["main_stop_arrival_time_minutes"]),
                float(trips[i]["departure_time_minutes"]),
                i,
            )
        )

    windows = {
        int(w["index"]): w
        for w in instance.get("time_windows", [])
    }

    stopping_times = instance.get("stopping_times", {})
    pull_times = instance.get("pull_in_out_times", {})

    alpha = float(instance.get("objective_function", {}).get("alpha", 1.0))

    vehicle_cost = recursively_find_number(
        instance,
        [
            "per_vehicle_deployment_cost",
            "vehicle_deployment_cost",
            "fixed_vehicle_cost",
            "vehicle_fixed_cost",
            "cost_per_vehicle",
            "vehicle_cost",
            "fleet_cost",
        ],
        default=1000.0,
    )

    depot_identifier = str(instance.get("topology", {}).get("depot", "O"))

    def depot_min_stop(window_index):
        possible = [
            instance.get("depot_stopping_times"),
            stopping_times.get(depot_identifier),
            instance.get("depot_min_stopping_times"),
        ]
        for records in possible:
            if isinstance(records, list):
                return get_window_value(
                    records,
                    window_index,
                    "min_stopping_time_minutes",
                    default=0.0,
                )

        return recursively_find_number(
            instance,
            [
                "minimum_depot_stopping_time_minutes",
                "min_depot_stopping_time_minutes",
                "depot_min_stopping_time_minutes",
            ],
            default=0.0,
        )

    def terminal_min_stop(terminal, window_index):
        return get_window_value(
            stopping_times.get(str(terminal), []),
            window_index,
            "min_stopping_time_minutes",
            default=0.0,
        )

    def terminal_max_stop(terminal, window_index):
        return get_window_value(
            stopping_times.get(str(terminal), []),
            window_index,
            "max_stopping_time_minutes",
            default=1.0e12,
        )

    def pull_out(terminal, window_index):
        return get_window_value(
            pull_times.get(str(terminal), []),
            window_index,
            "pull_out_time_minutes",
            default=0.0,
        )

    def pull_in(terminal, window_index):
        return get_window_value(
            pull_times.get(str(terminal), []),
            window_index,
            "pull_in_time_minutes",
            default=0.0,
        )

    def trip_window(trip_id):
        return int(trips[trip_id]["time_window_index"])

    def trip_pull_out(trip_id):
        trip = trips[trip_id]
        return pull_out(trip["start_terminal"], trip_window(trip_id))

    def trip_pull_in(trip_id):
        trip = trips[trip_id]
        return pull_in(trip["end_terminal"], trip_window(trip_id))

    model = gp.Model("integrated_timetable_vehicle_scheduling")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    y = {
        i: model.addVar(vtype=GRB.BINARY, name=f"select_{i}")
        for i in trip_ids
    }

    # Timetable arcs use None as source or sink internally.
    tt_vars = {}
    tt_cost = {}
    tt_out = defaultdict(list)
    tt_in = defaultdict(list)

    initial_windows = instance.get("initial_trip_time_windows", {})
    final_windows = instance.get("final_trip_time_windows", {})

    for pattern_id in patterns:
        ids = trips_by_pattern[pattern_id]
        initial_window = int(initial_windows.get(pattern_id, -1))
        final_window = int(final_windows.get(pattern_id, -1))

        for i in ids:
            if trip_window(i) == initial_window:
                key = (pattern_id, None, i)
                var = model.addVar(vtype=GRB.BINARY, name=f"tt_source_{pattern_id}_{i}")
                tt_vars[key] = var
                tt_cost[key] = 0.0
                tt_in[(pattern_id, i)].append(var)

            if trip_window(i) == final_window:
                key = (pattern_id, i, None)
                var = model.addVar(vtype=GRB.BINARY, name=f"tt_sink_{pattern_id}_{i}")
                tt_vars[key] = var
                tt_cost[key] = 0.0
                tt_out[(pattern_id, i)].append(var)

        main_times = [float(trips[i]["main_stop_arrival_time_minutes"]) for i in ids]

        for pos, i in enumerate(ids):
            earlier_window = windows[trip_window(i)]
            minimum_seconds = float(earlier_window["min_headway_seconds"])
            maximum_seconds = float(earlier_window["max_headway_seconds"])
            ideal_seconds = float(earlier_window["ideal_headway_seconds"])

            earliest_minutes = main_times[pos] + minimum_seconds / 60.0
            latest_minutes = main_times[pos] + maximum_seconds / 60.0

            begin = bisect.bisect_left(main_times, earliest_minutes, lo=pos + 1)
            end = bisect.bisect_right(main_times, latest_minutes, lo=begin)

            for next_pos in range(begin, end):
                j = ids[next_pos]
                actual_seconds = (
                    float(trips[j]["main_stop_arrival_time_minutes"])
                    - float(trips[i]["main_stop_arrival_time_minutes"])
                ) * 60.0

                if actual_seconds + 1e-7 < minimum_seconds:
                    continue
                if actual_seconds - 1e-7 > maximum_seconds:
                    continue

                penalty = (actual_seconds - ideal_seconds) ** 2
                key = (pattern_id, i, j)
                var = model.addVar(vtype=GRB.BINARY, name=f"tt_{pattern_id}_{i}_{j}")
                tt_vars[key] = var
                tt_cost[key] = penalty
                tt_out[(pattern_id, i)].append(var)
                tt_in[(pattern_id, j)].append(var)

    model.update()

    for pattern_id in patterns:
        ids = trips_by_pattern[pattern_id]
        source_vars = [
            var
            for (p, u, v), var in tt_vars.items()
            if p == pattern_id and u is None and v is not None
        ]
        sink_vars = [
            var
            for (p, u, v), var in tt_vars.items()
            if p == pattern_id and u is not None and v is None
        ]

        model.addConstr(gp.quicksum(source_vars) == 1, name=f"tt_source_flow_{pattern_id}")
        model.addConstr(gp.quicksum(sink_vars) == 1, name=f"tt_sink_flow_{pattern_id}")

        for i in ids:
            incoming = gp.quicksum(tt_in[(pattern_id, i)])
            outgoing = gp.quicksum(tt_out[(pattern_id, i)])
            model.addConstr(incoming == y[i], name=f"tt_in_select_{pattern_id}_{i}")
            model.addConstr(outgoing == y[i], name=f"tt_out_select_{pattern_id}_{i}")

    # Vehicle compatibility arcs.
    connection_vars = {}
    connection_cost = {}
    connection_type = {}
    vehicle_out = defaultdict(list)
    vehicle_in = defaultdict(list)

    by_departure = sorted(
        trip_ids,
        key=lambda i: (float(trips[i]["departure_time_minutes"]), i)
    )
    departure_values = [float(trips[i]["departure_time_minutes"]) for i in by_departure]

    for i in trip_ids:
        first = trips[i]
        first_window = trip_window(i)
        arrival = float(first["arrival_time_minutes"])
        begin = bisect.bisect_left(departure_values, arrival - 1e-9)

        for position in range(begin, len(by_departure)):
            j = by_departure[position]
            if i == j:
                continue

            second = trips[j]
            departure = float(second["departure_time_minutes"])
            gap = departure - arrival
            if gap < -1e-7:
                continue

            same_terminal = str(first["end_terminal"]) == str(second["start_terminal"])

            if same_terminal:
                minimum = terminal_min_stop(first["end_terminal"], first_window)
                maximum = terminal_max_stop(first["end_terminal"], first_window)
                if gap + 1e-7 < minimum or gap - 1e-7 > maximum:
                    continue
                cost = max(0.0, gap - minimum)
                kind = "in_line"
            else:
                required = (
                    pull_in(first["end_terminal"], first_window)
                    + depot_min_stop(first_window)
                    + pull_out(second["start_terminal"], trip_window(j))
                )
                if gap + 1e-7 < required:
                    continue
                cost = (
                    pull_in(first["end_terminal"], first_window)
                    + pull_out(second["start_terminal"], trip_window(j))
                )
                kind = "out_line"

            key = (i, j)
            var = model.addVar(vtype=GRB.BINARY, name=f"connect_{i}_{j}")
            connection_vars[key] = var
            connection_cost[key] = cost
            connection_type[key] = kind
            vehicle_out[i].append(var)
            vehicle_in[j].append(var)

    start_vars = {
        i: model.addVar(vtype=GRB.BINARY, name=f"pullout_{i}")
        for i in trip_ids
    }
    end_vars = {
        i: model.addVar(vtype=GRB.BINARY, name=f"pullin_{i}")
        for i in trip_ids
    }

    model.update()

    for i in trip_ids:
        model.addConstr(
            start_vars[i] + gp.quicksum(vehicle_in[i]) == y[i],
            name=f"vehicle_predecessor_{i}",
        )
        model.addConstr(
            end_vars[i] + gp.quicksum(vehicle_out[i]) == y[i],
            name=f"vehicle_successor_{i}",
        )

    timetable_expression = gp.quicksum(
        tt_cost[key] * var for key, var in tt_vars.items()
    )

    fleet_expression = vehicle_cost * gp.quicksum(start_vars.values())
    terminal_and_deadhead_expression = gp.quicksum(
        connection_cost[key] * var for key, var in connection_vars.items()
    )
    pullout_expression = gp.quicksum(
        trip_pull_out(i) * start_vars[i] for i in trip_ids
    )
    pullin_expression = gp.quicksum(
        trip_pull_in(i) * end_vars[i] for i in trip_ids
    )

    vehicle_expression = (
        fleet_expression
        + terminal_and_deadhead_expression
        + pullout_expression
        + pullin_expression
    )

    model.setObjective(
        timetable_expression + alpha * vehicle_expression,
        GRB.MINIMIZE,
    )

    def construct_solution(y_values, tt_values, connection_values,
                           start_values, end_values, objective_value):
        selected = sorted(i for i in trip_ids if y_values.get(i, 0.0) > 0.5)

        tt_arcs_used = {}
        for pattern_id in patterns:
            source_marker = f"source_{pattern_id}"
            sink_marker = f"sink_{pattern_id}"

            outgoing = {}
            source_next = None

            for (p, u, v), value in tt_values.items():
                if p != pattern_id or value <= 0.5:
                    continue
                if u is None:
                    source_next = v
                else:
                    outgoing[u] = v

            ordered_arcs = []
            if source_next is not None:
                ordered_arcs.append([source_marker, str(source_next)])
                current = source_next
                visited = set()

                while current is not None and current not in visited:
                    visited.add(current)
                    next_node = outgoing.get(current)
                    if next_node is None:
                        ordered_arcs.append([str(current), sink_marker])
                        break
                    ordered_arcs.append([str(current), str(next_node)])
                    current = next_node

            tt_arcs_used[pattern_id] = ordered_arcs

        vs_flows = {}
        num_vehicles = int(round(sum(
            1.0 for i in selected if start_values.get(i, 0.0) > 0.5
        )))

        for i in selected:
            vs_flows[f"trip_{i}_in-->trip_{i}_out"] = 1

            if start_values.get(i, 0.0) > 0.5:
                vs_flows[f"depot_out-->trip_{i}_in"] = 1

            if end_values.get(i, 0.0) > 0.5:
                vs_flows[f"trip_{i}_out-->depot_in"] = 1

        for (i, j), value in connection_values.items():
            if value > 0.5:
                vs_flows[f"trip_{i}_out-->trip_{j}_in"] = 1

        vs_flows["depot_in-->depot_out"] = num_vehicles

        return {
            "objective_value": float(objective_value),
            "selected_trips": selected,
            "num_vehicles": num_vehicles,
            "tt_arcs_used": tt_arcs_used,
            "vs_flows": vs_flows,
        }

    # Build a guaranteed feasible MIP start: shortest timetable path per pattern,
    # followed by a greedy compatible vehicle matching.
    initial_y = {i: 0.0 for i in trip_ids}
    initial_tt = {key: 0.0 for key in tt_vars}
    initial_connections = {key: 0.0 for key in connection_vars}
    initial_starts = {i: 0.0 for i in trip_ids}
    initial_ends = {i: 0.0 for i in trip_ids}

    initial_feasible = True

    for pattern_id in patterns:
        ids = trips_by_pattern[pattern_id]
        distance = {i: math.inf for i in ids}
        predecessor = {}
        source_key_for = {}

        for key in tt_vars:
            p, u, v = key
            if p == pattern_id and u is None and v is not None:
                distance[v] = 0.0
                predecessor[v] = None
                source_key_for[v] = key

        for i in ids:
            if not math.isfinite(distance[i]):
                continue
            for key, var in tt_vars.items():
                p, u, v = key
                if p != pattern_id or u != i or v is None:
                    continue
                candidate = distance[i] + tt_cost[key]
                if candidate + 1e-9 < distance[v]:
                    distance[v] = candidate
                    predecessor[v] = i

        best_final = None
        best_cost = math.inf
        best_sink_key = None
        for key in tt_vars:
            p, u, v = key
            if p == pattern_id and u is not None and v is None:
                if distance.get(u, math.inf) < best_cost:
                    best_cost = distance[u]
                    best_final = u
                    best_sink_key = key

        if best_final is None:
            initial_feasible = False
            break

        chain = []
        current = best_final
        while current is not None:
            chain.append(current)
            current = predecessor.get(current)
        chain.reverse()

        first = chain[0]
        source_key = (pattern_id, None, first)
        if source_key not in initial_tt:
            initial_feasible = False
            break

        initial_tt[source_key] = 1.0
        initial_tt[best_sink_key] = 1.0

        for i in chain:
            initial_y[i] = 1.0
        for i, j in zip(chain, chain[1:]):
            initial_tt[(pattern_id, i, j)] = 1.0

    if initial_feasible:
        selected_initial = [i for i in trip_ids if initial_y[i] > 0.5]
        chosen_predecessor = set()
        chosen_successor = set()

        candidate_connections = []
        for (i, j), cost in connection_cost.items():
            if initial_y[i] <= 0.5 or initial_y[j] <= 0.5:
                continue
            saving = (
                vehicle_cost
                + trip_pull_in(i)
                + trip_pull_out(j)
                - cost
            )
            if saving > 1e-9:
                candidate_connections.append((saving, i, j))

        candidate_connections.sort(reverse=True)

        for saving, i, j in candidate_connections:
            if i in chosen_successor or j in chosen_predecessor:
                continue
            initial_connections[(i, j)] = 1.0
            chosen_successor.add(i)
            chosen_predecessor.add(j)

        for i in selected_initial:
            initial_starts[i] = 0.0 if i in chosen_predecessor else 1.0
            initial_ends[i] = 0.0 if i in chosen_successor else 1.0

        initial_objective = 0.0
        for key, value in initial_tt.items():
            initial_objective += tt_cost[key] * value

        initial_vehicle_cost = vehicle_cost * sum(initial_starts.values())
        initial_vehicle_cost += sum(
            connection_cost[key] * value
            for key, value in initial_connections.items()
        )
        initial_vehicle_cost += sum(
            trip_pull_out(i) * initial_starts[i] for i in trip_ids
        )
        initial_vehicle_cost += sum(
            trip_pull_in(i) * initial_ends[i] for i in trip_ids
        )
        initial_objective += alpha * initial_vehicle_cost

        for i in trip_ids:
            y[i].Start = initial_y[i]
            start_vars[i].Start = initial_starts[i]
            end_vars[i].Start = initial_ends[i]
        for key, var in tt_vars.items():
            var.Start = initial_tt[key]
        for key, var in connection_vars.items():
            var.Start = initial_connections[key]

        initial_solution = construct_solution(
            initial_y,
            initial_tt,
            initial_connections,
            initial_starts,
            initial_ends,
            initial_objective,
        )
        if logger:
            logger.log_solution(initial_objective, initial_solution)

    remaining_time = max(0.01, float(args.time_limit) - (time.monotonic() - start_clock))
    model.Params.TimeLimit = remaining_time

    best_logged_objective = [initial_solution["objective_value"] if initial_feasible else math.inf]

    y_items = list(y.items())
    tt_items = list(tt_vars.items())
    connection_items = list(connection_vars.items())
    start_items = list(start_vars.items())
    end_items = list(end_vars.items())

    def callback(cb_model, where):
        if where != GRB.Callback.MIPSOL or logger is None:
            return
        try:
            objective_value = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
            if objective_value >= best_logged_objective[0] - 1e-7:
                return

            y_raw = cb_model.cbGetSolution([var for _, var in y_items])
            tt_raw = cb_model.cbGetSolution([var for _, var in tt_items])
            connection_raw = cb_model.cbGetSolution([var for _, var in connection_items])
            start_raw = cb_model.cbGetSolution([var for _, var in start_items])
            end_raw = cb_model.cbGetSolution([var for _, var in end_items])

            solution = construct_solution(
                {key: value for (key, _), value in zip(y_items, y_raw)},
                {key: value for (key, _), value in zip(tt_items, tt_raw)},
                {key: value for (key, _), value in zip(connection_items, connection_raw)},
                {key: value for (key, _), value in zip(start_items, start_raw)},
                {key: value for (key, _), value in zip(end_items, end_raw)},
                objective_value,
            )
            logger.log_solution(objective_value, solution)
            best_logged_objective[0] = objective_value
        except Exception:
            # Logging must never interrupt optimization.
            pass

    model.optimize(callback if logger else None)

    if model.SolCount > 0:
        final_y = {key: var.X for key, var in y_items}
        final_tt = {key: var.X for key, var in tt_items}
        final_connections = {key: var.X for key, var in connection_items}
        final_starts = {key: var.X for key, var in start_items}
        final_ends = {key: var.X for key, var in end_items}
        final_objective = float(model.ObjVal)

        final_solution = construct_solution(
            final_y,
            final_tt,
            final_connections,
            final_starts,
            final_ends,
            final_objective,
        )
    elif initial_feasible:
        final_solution = initial_solution
        final_objective = initial_solution["objective_value"]
    else:
        raise RuntimeError("No feasible integrated timetable and vehicle schedule was found.")

    if logger and final_objective < best_logged_objective[0] - 1e-7:
        logger.log_solution(final_objective, final_solution)

    solution_directory = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(solution_directory, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as file:
        json.dump(final_solution, file, indent=2)


if __name__ == "__main__":
    main()