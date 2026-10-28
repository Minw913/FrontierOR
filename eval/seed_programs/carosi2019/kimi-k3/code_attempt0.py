import argparse
import bisect
import json
import math
import os
import re
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


TT_SOURCE = "TT_source"
TT_SINK = "TT_sink"
DEFAULT_DEPLOYMENT_COST = 1_000_000.0


def _as_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", "").replace("_", ""))
        except ValueError:
            return None
    return None


def extract_deployment_cost(instance):
    """Find a per-vehicle fixed cost if supplied outside the terse schema."""
    named_values = []
    stack = [instance]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                key_l = str(key).lower().replace("-", "_").replace(" ", "_")
                is_candidate = (
                    ("deployment" in key_l)
                    or ("vehicle" in key_l and "cost" in key_l)
                    or ("fleet" in key_l and "cost" in key_l)
                )
                number = _as_number(value)
                if is_candidate and number is not None:
                    named_values.append(abs(number))
                if isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(current, list):
            for value in current:
                if isinstance(value, (dict, list)):
                    stack.append(value)

    objective = instance.get("objective_function", {})
    text_parts = [
        str(objective.get("type", "")),
        str(objective.get("description", "")),
    ]
    for field in ("vs_cost_components", "tt_cost_components"):
        values = objective.get(field, [])
        if isinstance(values, list):
            text_parts.extend(str(v) for v in values)
        else:
            text_parts.append(str(values))
    text = " ".join(text_parts).lower()

    num = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    patterns = [
        rf"\b({num})\s*(?:\*|x|×)?\s*(?:per[-_\s]*vehicle|number\s+of\s+vehicles|num_vehicles|vehicles|vehicle\s+units|fleet(?:\s+size)?)\b",
        rf"\b(?:per[-_\s]*vehicle|number\s+of\s+vehicles|num_vehicles|vehicles|vehicle\s+units|fleet(?:\s+size)?)\s*(?:\*|x|×)?\s*({num})\b",
        rf"\b(?:deployment|fixed(?:\s+vehicle)?)\s*cost[^0-9]{0,30}({num})",
    ]
    regex_values = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            value = _as_number(match.group(1))
            if value is not None:
                regex_values.append(abs(value))

    candidates = named_values + regex_values
    if candidates:
        return float(max(candidates))
    return DEFAULT_DEPLOYMENT_COST


def window_indexed(table):
    result = {}
    for key, entries in (table or {}).items():
        if not isinstance(entries, list):
            continue
        indexed = {}
        for position, entry in enumerate(entries):
            if isinstance(entry, dict):
                index = entry.get("time_window_index", position)
                try:
                    index = int(index)
                except (TypeError, ValueError):
                    index = position
                indexed[index] = entry
        result[str(key)] = indexed
    return result


def get_entry(indexed_table, key, window_index):
    try:
        window_index = int(window_index)
    except (TypeError, ValueError):
        pass
    return indexed_table.get(str(key), {}).get(window_index, {})


def get_time_window_map(windows):
    result = {}
    for position, window in enumerate(windows or []):
        try:
            index = int(window.get("index", position))
        except (TypeError, ValueError):
            index = position
        result[index] = window
    return result


def normalize_trips(instance):
    trips_by_id = {}
    trips_by_pattern = {}
    pattern_ids = []

    directions = instance.get("directions", []) or []
    for direction in directions:
        if isinstance(direction, dict):
            pattern_id = direction.get("pattern_id")
            if pattern_id is not None and pattern_id not in pattern_ids:
                pattern_ids.append(pattern_id)

    potential = instance.get("potential_trips", []) or []
    for raw in potential:
        trip_id = raw.get("id")
        if trip_id in trips_by_id:
            raise ValueError(f"Duplicate potential trip id {trip_id!r}")
        pattern_id = raw.get("pattern_id")
        trip = {
            "id": trip_id,
            "pattern_id": pattern_id,
            "direction": raw.get("direction"),
            "start_terminal": str(raw.get("start_terminal")),
            "end_terminal": str(raw.get("end_terminal")),
            "departure": float(raw.get("departure_time_minutes")),
            "arrival": float(raw.get("arrival_time_minutes")),
            "main_stop": float(raw.get("main_stop_arrival_time_minutes")),
            "time_window": raw.get("time_window_index"),
        }
        trips_by_id[trip_id] = trip
        trips_by_pattern.setdefault(pattern_id, []).append(trip)
        if pattern_id is not None and pattern_id not in pattern_ids:
            pattern_ids.append(pattern_id)

    return trips_by_id, trips_by_pattern, pattern_ids


def target_allows_window(target, window_index):
    if target is None:
        return True
    if isinstance(target, (list, tuple, set)):
        allowed = set()
        for value in target:
            try:
                allowed.add(int(value))
            except (TypeError, ValueError):
                allowed.add(value)
        try:
            return int(window_index) in allowed
        except (TypeError, ValueError):
            return window_index in allowed
    try:
        return int(window_index) == int(target)
    except (TypeError, ValueError):
        return window_index == target


def build_timetable_candidates(instance, trips_by_pattern, pattern_ids):
    windows = get_time_window_map(instance.get("time_windows", []))
    initial_map = instance.get("initial_trip_time_windows", {}) or {}
    final_map = instance.get("final_trip_time_windows", {}) or {}

    edge_lists = {}
    edge_costs = {}
    adjacency = {}
    incoming_adjacency = {}
    source_ids = {}
    sink_ids = {}

    for pattern in pattern_ids:
        trips = sorted(
            trips_by_pattern.get(pattern, []),
            key=lambda t: (t["main_stop"], str(t["id"])),
        )
        edges = []
        costs = {}
        adj = {t["id"]: [] for t in trips}
        inc = {t["id"]: [] for t in trips}

        initial_target = initial_map.get(pattern)
        final_target = final_map.get(pattern)
        src = {
            t["id"]
            for t in trips
            if target_allows_window(initial_target, t["time_window"])
        }
        dst = {
            t["id"]
            for t in trips
            if target_allows_window(final_target, t["time_window"])
        }

        for first_pos, first in enumerate(trips):
            window_index = first["time_window"]
            try:
                window_index = int(window_index)
            except (TypeError, ValueError):
                pass
            window = windows.get(window_index)
            if not isinstance(window, dict):
                continue
            min_headway = float(window.get("min_headway_seconds", 0))
            max_headway = float(window.get("max_headway_seconds", 0))
            ideal_headway = float(window.get("ideal_headway_seconds", 0))

            for second in trips[first_pos + 1:]:
                headway_seconds = (second["main_stop"] - first["main_stop"]) * 60.0
                if headway_seconds > max_headway:
                    break
                if headway_seconds < min_headway:
                    continue
                deviation = headway_seconds - ideal_headway
                penalty = deviation * deviation
                first_id = first["id"]
                second_id = second["id"]
                edges.append((first_id, second_id, penalty))
                costs[(first_id, second_id)] = penalty
                adj[first_id].append(second_id)
                inc[second_id].append(first_id)

        edge_lists[pattern] = edges
        edge_costs[pattern] = costs
        adjacency[pattern] = adj
        incoming_adjacency[pattern] = inc
        source_ids[pattern] = src
        sink_ids[pattern] = dst

    return {
        "edge_lists": edge_lists,
        "edge_costs": edge_costs,
        "adjacency": adjacency,
        "incoming_adjacency": incoming_adjacency,
        "source_ids": source_ids,
        "sink_ids": sink_ids,
    }


def choose_timetable_path(pattern, trips, tt_data):
    ordered = sorted(
        trips,
        key=lambda t: (t["main_stop"], str(t["id"])),
    )
    adjacency = tt_data["adjacency"][pattern]
    costs = tt_data["edge_costs"][pattern]
    sink_ids = tt_data["sink_ids"][pattern]
    source_ids = tt_data["source_ids"][pattern]

    best_value = {}
    next_node = {}
    for trip in reversed(ordered):
        trip_id = trip["id"]
        value = math.inf
        successor = None
        if trip_id in sink_ids:
            value = 0.0
            successor = None
        for candidate in adjacency.get(trip_id, []):
            candidate_value = costs[(trip_id, candidate)] + best_value.get(candidate, math.inf)
            if candidate_value < value:
                value = candidate_value
                successor = candidate
        best_value[trip_id] = value
        next_node[trip_id] = successor

    starts = [trip["id"] for trip in ordered if trip["id"] in source_ids]
    if not starts:
        return None
    start = min(starts, key=lambda trip_id: best_value.get(trip_id, math.inf))
    if best_value.get(start, math.inf) == math.inf:
        return None

    path = []
    current = start
    while True:
        path.append(current)
        successor = next_node.get(current)
        if successor is None:
            break
        current = successor
    return path


def build_vehicle_candidates(instance, trips, stop_lookup, pull_lookup, depot_lookup, deadline):
    sorted_trips = sorted(
        trips,
        key=lambda t: (t["departure"], str(t["id"])),
    )
    departures = [trip["departure"] for trip in sorted_trips]
    edges = []
    costs = {}

    for first in sorted_trips:
        if deadline is not None and time.monotonic() > deadline:
            break
        first_id = first["id"]
        start_position = bisect.bisect_left(departures, first["arrival"])

        for pos in range(start_position, len(sorted_trips)):
            second = sorted_trips[pos]
            second_id = second["id"]
            gap = second["departure"] - first["arrival"]

            if gap < -1e-9:
                continue
            if gap <= 1e-9 and str(second_id) <= str(first_id):
                continue

            if first["end_terminal"] == second["start_terminal"]:
                entry = get_entry(
                    stop_lookup,
                    second["start_terminal"],
                    first["time_window"],
                )
                min_stop = float(entry.get("min_stopping_time_minutes", 0.0))
                max_stop = float(entry.get("max_stopping_time_minutes", 1e18))
                if gap < min_stop - 1e-9 or gap > max_stop + 1e-9:
                    continue
                cost = max(0.0, gap - min_stop)
            else:
                pull_in_entry = get_entry(
                    pull_lookup,
                    first["end_terminal"],
                    first["time_window"],
                )
                pull_out_entry = get_entry(
                    pull_lookup,
                    second["start_terminal"],
                    second["time_window"],
                )
                depot_entry = get_entry(depot_lookup, instance.get("topology", {}).get("depot", ""), first["time_window"])
                pull_in_time = float(pull_in_entry.get("pull_in_time_minutes", 0.0))
                depot_min = float(depot_entry.get("min_stopping_time_minutes", 0.0))
                pull_out_time = float(pull_out_entry.get("pull_out_time_minutes", 0.0))
                required_gap = pull_in_time + depot_min + pull_out_time
                if gap < required_gap - 1e-9:
                    continue
                cost = pull_in_time + pull_out_time

            edges.append((first_id, second_id, cost))
            costs[(first_id, second_id)] = cost

    return edges, costs


def greedy_vehicle_connections(trip_ids, candidate_edges):
    ordered_edges = sorted(candidate_edges, key=lambda edge: edge[2])
    parent = {trip_id: trip_id for trip_id in trip_ids}
    incoming = {trip_id: 0 for trip_id in trip_ids}
    outgoing = {trip_id: 0 for trip_id in trip_ids}
    accepted = []

    def find(node):
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    for first, second, cost in ordered_edges:
        if outgoing.get(first, 1) != 0 or incoming.get(second, 1) != 0:
            continue
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            continue
        accepted.append((first, second, cost))
        outgoing[first] = 1
        incoming[second] = 1
        parent[first_root] = second_root

    return accepted


def operation_node(trip_id, phase):
    return str((trip_id, phase))


def build_solution(
    trips_by_id,
    pattern_paths,
    timetable_costs,
    vehicle_edges,
    vehicle_costs,
    pull_lookup,
    deployment_cost,
    alpha,
    num_vehicles=None,
    max_runtime_check=None,
):
    selected = sorted(trips_by_id.keys())
    selected_set = set(selected)

    tt_arcs_used = {}
    timetable_total = 0.0
    for pattern, path in pattern_paths.items():
        arcs = []
        if path:
            previous = TT_SOURCE
            for trip_id in path:
                arcs.append([previous, str(trip_id)])
                previous = str(trip_id)
            arcs.append([previous, TT_SINK])
            for first, second in zip(path, path[1:]):
                timetable_total += timetable_costs[pattern][(first, second)]
        tt_arcs_used[pattern] = arcs

    incoming_trip = {second for first, second, _ in vehicle_edges}
    outgoing_trip = {first for first, second, _ in vehicle_edges}

    vehicle_total = 0.0
    vs_flows = {}
    for trip_id in selected:
        trip = trips_by_id[trip_id]
        start_node = operation_node(trip_id, "start")
        end_node = operation_node(trip_id, "end")
        vs_flows[start_node + "-->" + end_node] = 1

        if trip_id not in incoming_trip:
            pull_out_entry = get_entry(
                pull_lookup,
                trip["start_terminal"],
                trip["time_window"],
            )
            cost = float(pull_out_entry.get("pull_out_time_minutes", 0.0))
            vehicle_total += cost
            vs_flows["VS_O_minus-->" + start_node] = 1

        if trip_id not in outgoing_trip:
            pull_in_entry = get_entry(
                pull_lookup,
                trip["end_terminal"],
                trip["time_window"],
            )
            cost = float(pull_in_entry.get("pull_in_time_minutes", 0.0))
            vehicle_total += cost
            vs_flows[end_node + "-->VS_O_plus"] = 1

    for first, second, cost in vehicle_edges:
        first_end = operation_node(first, "end")
        second_start = operation_node(second, "start")
        vs_flows[first_end + "-->" + second_start] = 1
        vehicle_total += cost

    if num_vehicles is None:
        num_vehicles = len(selected) - len(vehicle_edges)
    num_vehicles = int(round(num_vehicles))
    vehicle_total += deployment_cost * num_vehicles
    vs_flows["VS_O_plus-->VS_O_minus"] = num_vehicles

    objective = alpha * vehicle_total + timetable_total
    return objective, {
        "objective_value": float(objective),
        "selected_trips": selected,
        "num_vehicles": num_vehicles,
        "tt_arcs_used": tt_arcs_used,
        "vs_flows": vs_flows,
    }


def solve(instance, deadline, logger):
    start_time = time.monotonic()
    alpha = float(instance.get("objective_function", {}).get("alpha", 1.0))
    deployment_cost = extract_deployment_cost(instance)
    trips_by_id, trips_by_pattern, pattern_ids = normalize_trips(instance)
    all_trips = list(trips_by_id.values())

    stop_lookup = window_indexed(instance.get("stopping_times", {}))
    pull_lookup = window_indexed(instance.get("pull_in_out_times", {}))
    depot_name = instance.get("topology", {}).get("depot", "")
    depot_lookup = {
        depot_name: stop_lookup.get(depot_name, {}),
    }

    tt_data = build_timetable_candidates(instance, trips_by_pattern, pattern_ids)
    pattern_paths = {}
    for pattern in pattern_ids:
        path = choose_timetable_path(pattern, trips_by_pattern.get(pattern, []), tt_data)
        if path is None:
            raise ValueError(f"No feasible timetable chain found for pattern {pattern!r}")
        pattern_paths[pattern] = path

    soft_deadline = None
    if deadline is not None:
        total_remaining = max(0.0, deadline - start_time)
        reserve = min(1.0, max(0.05, 0.05 * total_remaining))
        soft_deadline = deadline - reserve if total_remaining > reserve else deadline

    vehicle_candidates, vehicle_cost_map = build_vehicle_candidates(
        instance,
        all_trips,
        stop_lookup,
        pull_lookup,
        depot_lookup,
        soft_deadline,
    )

    if deadline is None or time.monotonic() < deadline:
        fallback_vehicle_edges = greedy_vehicle_connections(list(trips_by_id), vehicle_candidates)
    else:
        fallback_vehicle_edges = []

    heuristic_objective, heuristic_solution = build_solution(
        trips_by_id,
        pattern_paths,
        tt_data["edge_costs"],
        fallback_vehicle_edges,
        vehicle_cost_map,
        pull_lookup,
        deployment_cost,
        alpha,
    )

    best_objective = heuristic_objective
    best_solution = heuristic_solution
    if logger:
        logger.log_solution(float(best_objective), best_solution)

    if deadline is not None and time.monotonic() >= deadline - 0.01:
        return best_objective, best_solution

    model = gp.Model("IntegratedTimetableVehicleScheduling")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    trip_ids = list(trips_by_id)
    selected_var = model.addVars(trip_ids, vtype=GRB.BINARY, name="selected_trip")

    tt_source_vars = {}
    tt_sink_vars = {}
    tt_arc_vars = {}
    tt_out_variables = {}
    tt_in_variables = {}

    for pattern in pattern_ids:
        tt_source_vars[pattern] = {}
        tt_sink_vars[pattern] = {}
        tt_arc_vars[pattern] = {}
        tt_out_variables[pattern] = {trip_id: [] for trip_id in trips_by_pattern.get(pattern, [])}
        tt_in_variables[pattern] = {trip_id: [] for trip_id in trips_by_pattern.get(pattern, [])}

        for edge in tt_data["edge_lists"][pattern]:
            first, second, cost = edge
            variable = model.addVar(
                vtype=GRB.BINARY,
                name=f"tt_arc[{pattern},{first},{second}]",
            )
            tt_arc_vars[pattern][(first, second)] = variable
            tt_out_variables[pattern][first].append((second, variable))
            tt_in_variables[pattern][second].append((first, variable))

        trip_ids_pattern = set(t["id"] for t in trips_by_pattern.get(pattern, []))
        for trip_id in tt_data["source_ids"][pattern].intersection(trip_ids_pattern):
            variable = model.addVar(
                vtype=GRB.BINARY,
                name=f"tt_source[{pattern},{trip_id}]",
            )
            tt_source_vars[pattern][trip_id] = variable
            tt_in_variables[pattern][trip_id].append((TT_SOURCE, variable))

        for trip_id in tt_data["sink_ids"][pattern].intersection(trip_ids_pattern):
            variable = model.addVar(
                vtype=GRB.BINARY,
                name=f"tt_sink[{pattern},{trip_id}]",
            )
            tt_sink_vars[pattern][trip_id] = variable
            tt_out_variables[pattern][trip_id].append((TT_SINK, variable))

    vehicle_start_vars = model.addVars(trip_ids, vtype=GRB.BINARY, name="vs_depot_pull_out")
    vehicle_end_vars = model.addVars(trip_ids, vtype=GRB.BINARY, name="vs_depot_pull_in")
    vehicle_connection_vars = {}
    vehicle_in_variables = {trip_id: [] for trip_id in trip_ids}
    vehicle_out_variables = {trip_id: [] for trip_id in trip_ids}

    for first, second, cost in vehicle_candidates:
        variable = model.addVar(
            vtype=GRB.BINARY,
            name=f"vs_connection[{first},{second}]",
        )
        vehicle_connection_vars[(first, second)] = variable
        vehicle_out_variables[first].append((second, variable))
        vehicle_in_variables[second].append((first, variable))

    num_vehicles_var = model.addVar(
        lb=0,
        ub=max(0, len(trip_ids)),
        vtype=GRB.INTEGER,
        name="num_vehicles",
    )

    for pattern in pattern_ids:
        if tt_source_vars[pattern]:
            model.addConstr(
                gp.quicksum(tt_source_vars[pattern].values()) == 1,
                name=f"one_initial_trip[{pattern}]",
            )
            model.addConstr(
                gp.quicksum(tt_sink_vars[pattern].values()) == 1,
                name=f"one_final_trip[{pattern}]",
            )

    for trip_id, trip in trips_by_id.items():
        pattern = trip["pattern_id"]
        incoming_terms = [variable for _, variable in tt_in_variables[pattern].get(trip_id, [])]
        outgoing_terms = [variable for _, variable in tt_out_variables[pattern].get(trip_id, [])]
        model.addConstr(
            selected_var[trip_id] == gp.quicksum(incoming_terms),
            name=f"tt_incoming[{trip_id}]",
        )
        model.addConstr(
            selected_var[trip_id] == gp.quicksum(outgoing_terms),
            name=f"tt_outgoing[{trip_id}]",
        )

        vehicle_incoming_terms = [variable for _, variable in vehicle_in_variables.get(trip_id, [])]
        vehicle_outgoing_terms = [variable for _, variable in vehicle_out_variables.get(trip_id, [])]
        model.addConstr(
            vehicle_start_vars[trip_id] + gp.quicksum(vehicle_incoming_terms)
            == selected_var[trip_id],
            name=f"vs_incoming[{trip_id}]",
        )
        model.addConstr(
            vehicle_end_vars[trip_id] + gp.quicksum(vehicle_outgoing_terms)
            == selected_var[trip_id],
            name=f"vs_outgoing[{trip_id}]",
        )

    model.addConstr(
        gp.quicksum(vehicle_start_vars.values()) == num_vehicles_var,
        name="depot_departure_flow",
    )
    model.addConstr(
        gp.quicksum(vehicle_end_vars.values()) == num_vehicles_var,
        name="depot_return_flow",
    )

    timetable_expression = gp.quicksum(
        cost * variable
        for pattern in pattern_ids
        for (first, second), variable in tt_arc_vars[pattern].items()
    )
    fixed_cost = min(deployment_cost, 1e12)
    vehicle_expression = fixed_cost * num_vehicles_var
    for first, second, cost in vehicle_candidates:
        vehicle_expression += cost * vehicle_connection_vars[(first, second)]

    for trip_id, trip in trips_by_id.items():
        pull_out_entry = get_entry(pull_lookup, trip["start_terminal"], trip["time_window"])
        pull_in_entry = get_entry(pull_lookup, trip["end_terminal"], trip["time_window"])
        vehicle_expression += float(pull_out_entry.get("pull_out_time_minutes", 0.0)) * vehicle_start_vars[trip_id]
        vehicle_expression += float(pull_in_entry.get("pull_in_time_minutes", 0.0)) * vehicle_end_vars[trip_id]

    model.setObjective(
        alpha * vehicle_expression + timetable_expression,
        GRB.MINIMIZE,
    )

    # Complete feasible MIP start from the deterministic fallback.
    incoming_target = {second for first, second, _ in fallback_vehicle_edges}
    outgoing_source = {first for first, second, _ in fallback_vehicle_edges}
    fallback_vehicle_count = len(trip_ids) - len(fallback_vehicle_edges)
    num_vehicles_var.Start = float(fallback_vehicle_count)

    for pattern, path in pattern_paths.items():
        for trip_id in path:
            selected_var[trip_id].Start = 1.0
        if path:
            tt_source_vars[pattern][path[0]].Start = 1.0
            tt_sink_vars[pattern][path[-1]].Start = 1.0
            for first, second in zip(path, path[1:]):
                tt_arc_vars[pattern][(first, second)].Start = 1.0

    for trip_id in trip_ids:
        if trip_id in incoming_target:
            vehicle_start_vars[trip_id].Start = 0.0
        else:
            vehicle_start_vars[trip_id].Start = 1.0
        if trip_id in outgoing_source:
            vehicle_end_vars[trip_id].Start = 0.0
        else:
            vehicle_end_vars[trip_id].Start = 1.0

    for first, second, _ in fallback_vehicle_edges:
        variable = vehicle_connection_vars.get((first, second))
        if variable is not None:
            variable.Start = 1.0

    callback_vars = list(model.getVars())

    def decode_solution(get_value):
        decoded_paths = {}
        for pattern in pattern_ids:
            candidates = []
            for trip_id, variable in tt_source_vars[pattern].items():
                value = get_value(variable)
                if value > 0.5:
                    candidates.append((value, str(trip_id), trip_id))
            if not candidates:
                decoded_paths[pattern] = []
                continue
            candidates.sort(reverse=True)
            current = candidates[0][2]
            path = [current]
            while True:
                successors = []
                for second, variable in tt_out_variables[pattern].get(current, []):
                    if second == TT_SINK:
                        continue
                    value = get_value(variable)
                    if value > 0.5:
                        successors.append((value, str(second), second))
                if not successors:
                    break
                successors.sort(reverse=True)
                current = successors[0][2]
                path.append(current)
            decoded_paths[pattern] = path

        accepted_connections = []
        for (first, second), variable in vehicle_connection_vars.items():
            if get_value(variable) > 0.5:
                cost = vehicle_cost_map[(first, second)]
                accepted_connections.append((first, second, cost))

        vehicles_value = get_value(num_vehicles_var)
        objective, solution = build_solution(
            trips_by_id,
            decoded_paths,
            tt_data["edge_costs"],
            accepted_connections,
            vehicle_cost_map,
            pull_lookup,
            deployment_cost,
            alpha,
            num_vehicles=int(round(vehicles_value)),
        )
        return objective, solution

    state = {
        "objective": float(best_objective),
        "solution": best_solution,
    }

    def incumbent_callback(callback_model, where):
        if where != GRB.Callback.MIPSOL:
            return
        if deadline is not None and time.monotonic() >= deadline:
            callback_model.terminate()
            return
        values = callback_model.cbGetSolution(callback_vars)
        value_by_id = {id(variable): values[index] for index, variable in enumerate(callback_vars)}

        def get_value(variable):
            return value_by_id.get(id(variable), 0.0)

        objective, solution = decode_solution(get_value)
        if objective < state["objective"] - 1e-7:
            state["objective"] = float(objective)
            state["solution"] = solution
            if logger:
                logger.log_solution(float(objective), solution)

    remaining = None
    if deadline is not None:
        remaining = max(0.0, deadline - time.monotonic())
        if remaining > 0:
            model.Params.TimeLimit = max(0.0, remaining)

    model.optimize(incumbent_callback)

    if model.SolCount > 0:
        def final_value(variable):
            return variable.X

        objective, solution = decode_solution(final_value)
        if objective < state["objective"] - 1e-7:
            state["objective"] = float(objective)
            state["solution"] = solution
            if logger:
                logger.log_solution(float(objective), solution)

    return state["objective"], state["solution"]


def write_solution(path, solution):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(solution, output_file, indent=2)
        output_file.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start = time.monotonic()
    deadline = start + max(0.0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as instance_file:
        instance = json.load(instance_file)

    _, solution = solve(instance, deadline, logger)
    write_solution(args.solution_path, solution)

    close = getattr(logger, "close", None)
    if callable(close):
        close()


if __name__ == "__main__":
    main()