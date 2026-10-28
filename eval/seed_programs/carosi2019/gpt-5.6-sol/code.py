import argparse
import json
import os
import re
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def indexed_table(raw_table):
    """Convert {location: [{time_window_index: ..., ...}, ...]} to indexed maps."""
    result = {}
    for location, rows in (raw_table or {}).items():
        result[location] = {
            int(row["time_window_index"]): row
            for row in rows
        }
    return result


def lookup_row(table, location, window_index):
    rows = table.get(location, {})
    if window_index in rows:
        return rows[window_index]
    if not rows:
        return {}
    # Defensive fallback for incomplete data: use the nearest available window.
    nearest = min(rows, key=lambda w: abs(w - window_index))
    return rows[nearest]


def recursively_find_numeric(data, candidate_keys):
    candidates = {k.lower() for k in candidate_keys}

    def visit(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if str(key).lower() in candidates and isinstance(value, (int, float)):
                    return float(value)
            for value in obj.values():
                found = visit(value)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for value in obj:
                found = visit(value)
                if found is not None:
                    return found
        return None

    return visit(data)


def vehicle_deployment_cost(instance):
    value = recursively_find_numeric(
        instance,
        {
            "vehicle_deployment_cost",
            "per_vehicle_deployment_cost",
            "fixed_vehicle_cost",
            "vehicle_fixed_cost",
            "vehicle_cost",
            "fleet_cost",
            "bus_cost",
            "deployment_cost",
            "large_vehicle_cost",
            "cost_per_vehicle",
        },
    )
    if value is not None:
        return value

    # Some generators put the value only in a textual objective description.
    description = str(instance.get("objective_function", {}).get("description", ""))
    patterns = [
        r"(?:per[- ]vehicle|vehicle deployment|fixed vehicle|fleet)\s*cost"
        r"\s*(?:is|=|:)\s*([-+]?\d+(?:\.\d+)?)",
        r"\bM\s*=\s*([-+]?\d+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, description, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))

    # Last-resort default. Valid benchmark instances are expected to provide it.
    return 10000.0


def make_tt_penalty(instance, earlier_trip, later_trip, windows):
    window = windows[int(earlier_trip["time_window_index"])]
    actual_seconds = (
        float(later_trip["main_stop_arrival_time_minutes"])
        - float(earlier_trip["main_stop_arrival_time_minutes"])
    ) * 60.0
    ideal_seconds = float(window["ideal_headway_seconds"])

    description = str(instance.get("penalty_function", {}).get("description", "")).lower()
    # Normally the schema's headway quantities imply seconds. Support an
    # explicitly minute-based formula if an instance describes one.
    minute_based = "minute" in description and "second" not in description
    if minute_based:
        deviation = actual_seconds / 60.0 - ideal_seconds / 60.0
    else:
        deviation = actual_seconds - ideal_seconds
    return deviation * deviation


def tuple_node(trip_id, side):
    return f"({trip_id}, '{side}')"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    wall_start = time.monotonic()

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    trips = list(instance.get("potential_trips", []))
    trips_by_id = {int(t["id"]): t for t in trips}
    trip_ids = list(trips_by_id)
    trip_ids_sorted = sorted(trip_ids)

    windows = {
        int(row["index"]): row
        for row in instance.get("time_windows", [])
    }
    stopping = indexed_table(instance.get("stopping_times", {}))
    pull_times = indexed_table(instance.get("pull_in_out_times", {}))
    depot = instance.get("topology", {}).get("depot", "O")

    pattern_ids = [str(d["pattern_id"]) for d in instance.get("directions", [])]
    # Include any pattern present in trips but omitted from the directions array.
    for t in trips:
        pid = str(t["pattern_id"])
        if pid not in pattern_ids:
            pattern_ids.append(pid)

    pattern_trips = defaultdict(list)
    for trip_id in trip_ids:
        pattern_trips[str(trips_by_id[trip_id]["pattern_id"])].append(trip_id)

    for pid in pattern_ids:
        pattern_trips[pid].sort(
            key=lambda i: (
                float(trips_by_id[i]["main_stop_arrival_time_minutes"]),
                float(trips_by_id[i]["departure_time_minutes"]),
                i,
            )
        )

    initial_windows = {
        str(k): int(v)
        for k, v in instance.get("initial_trip_time_windows", {}).items()
    }
    final_windows = {
        str(k): int(v)
        for k, v in instance.get("final_trip_time_windows", {}).items()
    }

    def stop_min(terminal, window_index):
        return float(
            lookup_row(stopping, terminal, int(window_index)).get(
                "min_stopping_time_minutes", 0.0
            )
        )

    def stop_max(terminal, window_index):
        row = lookup_row(stopping, terminal, int(window_index))
        return float(row.get("max_stopping_time_minutes", float("inf")))

    def pull_out(terminal, window_index):
        return float(
            lookup_row(pull_times, terminal, int(window_index)).get(
                "pull_out_time_minutes", 0.0
            )
        )

    def pull_in(terminal, window_index):
        return float(
            lookup_row(pull_times, terminal, int(window_index)).get(
                "pull_in_time_minutes", 0.0
            )
        )

    model = gp.Model("integrated_timetable_vehicle_schedule")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    x = {
        i: model.addVar(vtype=GRB.BINARY, name=f"x[{i}]")
        for i in trip_ids
    }

    # Timetable network arcs.
    tt_source_vars = {}
    tt_sink_vars = {}
    tt_pair_vars = {}
    tt_incoming = defaultdict(list)
    tt_outgoing = defaultdict(list)
    tt_objective_terms = []

    for pid in pattern_ids:
        ids = pattern_trips[pid]
        initial_window = initial_windows.get(pid)
        final_window = final_windows.get(pid)

        for i in ids:
            trip = trips_by_id[i]
            wi = int(trip["time_window_index"])

            if initial_window is not None and wi == initial_window:
                var = model.addVar(vtype=GRB.BINARY, name=f"tt_src[{pid},{i}]")
                tt_source_vars[(pid, i)] = var
                tt_incoming[i].append(var)

            if final_window is not None and wi == final_window:
                var = model.addVar(vtype=GRB.BINARY, name=f"tt_sink[{pid},{i}]")
                tt_sink_vars[(pid, i)] = var
                tt_outgoing[i].append(var)

        # Feasible headway arcs. Lists are sorted by main-stop time, so scanning
        # can stop once the maximum headway is exceeded.
        for pos, i in enumerate(ids):
            ti = trips_by_id[i]
            wi = int(ti["time_window_index"])
            window = windows[wi]
            min_h = float(window["min_headway_seconds"])
            max_h = float(window["max_headway_seconds"])
            time_i = float(ti["main_stop_arrival_time_minutes"])

            for j in ids[pos + 1:]:
                tj = trips_by_id[j]
                delta_seconds = (
                    float(tj["main_stop_arrival_time_minutes"]) - time_i
                ) * 60.0
                if delta_seconds > max_h + 1e-9:
                    break
                if delta_seconds + 1e-9 < min_h:
                    continue

                var = model.addVar(
                    vtype=GRB.BINARY, name=f"tt_arc[{pid},{i},{j}]"
                )
                tt_pair_vars[(pid, i, j)] = var
                tt_outgoing[i].append(var)
                tt_incoming[j].append(var)
                penalty = make_tt_penalty(instance, ti, tj, windows)
                if penalty:
                    tt_objective_terms.append(penalty * var)

    # Require one source-to-sink chain for every service pattern.
    for pid in pattern_ids:
        source_vars = [
            var for (p, _), var in tt_source_vars.items() if p == pid
        ]
        sink_vars = [
            var for (p, _), var in tt_sink_vars.items() if p == pid
        ]
        model.addConstr(gp.quicksum(source_vars) == 1, name=f"tt_source_flow[{pid}]")
        model.addConstr(gp.quicksum(sink_vars) == 1, name=f"tt_sink_flow[{pid}]")

        for i in pattern_trips[pid]:
            model.addConstr(
                gp.quicksum(tt_incoming[i]) == x[i],
                name=f"tt_in[{i}]",
            )
            model.addConstr(
                gp.quicksum(tt_outgoing[i]) == x[i],
                name=f"tt_out[{i}]",
            )

    # Vehicle path-cover formulation.
    vehicle_start = {
        i: model.addVar(vtype=GRB.BINARY, name=f"vs_start[{i}]")
        for i in trip_ids
    }
    vehicle_end = {
        i: model.addVar(vtype=GRB.BINARY, name=f"vs_end[{i}]")
        for i in trip_ids
    }

    vs_link_vars = {}
    vs_link_cost = {}
    vs_predecessors = defaultdict(list)
    vs_successors = defaultdict(list)

    ids_by_departure = sorted(
        trip_ids,
        key=lambda i: (
            float(trips_by_id[i]["departure_time_minutes"]),
            i,
        ),
    )

    # Generate all feasible in-line and out-line successor arcs.
    for i in trip_ids:
        ti = trips_by_id[i]
        arrival_i = float(ti["arrival_time_minutes"])
        end_terminal = str(ti["end_terminal"])
        wi = int(ti["time_window_index"])

        for j in ids_by_departure:
            if i == j:
                continue
            tj = trips_by_id[j]
            departure_j = float(tj["departure_time_minutes"])
            if departure_j + 1e-9 < arrival_i:
                continue

            gap = departure_j - arrival_i
            start_terminal = str(tj["start_terminal"])
            wj = int(tj["time_window_index"])

            feasible = False
            cost = 0.0

            if start_terminal == end_terminal:
                minimum = stop_min(end_terminal, wi)
                maximum = stop_max(end_terminal, wi)
                if gap + 1e-9 >= minimum and gap <= maximum + 1e-9:
                    feasible = True
                    cost = max(0.0, gap - minimum)
            else:
                depot_minimum = stop_min(depot, wi)
                required = (
                    pull_in(end_terminal, wi)
                    + depot_minimum
                    + pull_out(start_terminal, wj)
                )
                if gap + 1e-9 >= required:
                    feasible = True
                    cost = (
                        pull_in(end_terminal, wi)
                        + pull_out(start_terminal, wj)
                    )

            if feasible:
                var = model.addVar(
                    vtype=GRB.BINARY, name=f"vs_link[{i},{j}]"
                )
                vs_link_vars[(i, j)] = var
                vs_link_cost[(i, j)] = cost
                vs_successors[i].append(var)
                vs_predecessors[j].append(var)

    fleet = model.addVar(
        vtype=GRB.INTEGER,
        lb=0,
        ub=len(trip_ids),
        name="fleet",
    )

    for i in trip_ids:
        model.addConstr(
            vehicle_start[i] + gp.quicksum(vs_predecessors[i]) == x[i],
            name=f"vs_cover_in[{i}]",
        )
        model.addConstr(
            vehicle_end[i] + gp.quicksum(vs_successors[i]) == x[i],
            name=f"vs_cover_out[{i}]",
        )

    model.addConstr(
        fleet == gp.quicksum(vehicle_start.values()),
        name="fleet_departures",
    )
    model.addConstr(
        fleet == gp.quicksum(vehicle_end.values()),
        name="fleet_returns",
    )

    deployment_cost = vehicle_deployment_cost(instance)
    alpha = float(instance.get("objective_function", {}).get("alpha", 1.0))
    objective_description = str(
        instance.get("objective_function", {}).get("description", "")
    ).lower().replace(" ", "")
    timetable_weight = (
        1.0 - alpha
        if "(1-alpha)" in objective_description or "1-alpha" in objective_description
        else 1.0
    )

    vehicle_terms = [deployment_cost * fleet]

    for i in trip_ids:
        trip = trips_by_id[i]
        wi = int(trip["time_window_index"])
        vehicle_terms.append(
            pull_out(str(trip["start_terminal"]), wi) * vehicle_start[i]
        )
        vehicle_terms.append(
            pull_in(str(trip["end_terminal"]), wi) * vehicle_end[i]
        )

    for arc, var in vs_link_vars.items():
        cost = vs_link_cost[arc]
        if cost:
            vehicle_terms.append(cost * var)

    model.setObjective(
        alpha * gp.quicksum(vehicle_terms)
        + timetable_weight * gp.quicksum(tt_objective_terms),
        GRB.MINIMIZE,
    )

    model.update()

    def value_from(source, var):
        if callable(source):
            return float(source(var))
        return float(var.X)

    def build_solution(value_source, objective_value):
        selected = [
            i for i in trip_ids_sorted
            if value_from(value_source, x[i]) > 0.5
        ]

        tt_arcs_used = {}
        for pid in pattern_ids:
            chosen_source = None
            for (p, i), var in tt_source_vars.items():
                if p == pid and value_from(value_source, var) > 0.5:
                    chosen_source = i
                    break

            arcs = []
            if chosen_source is not None:
                arcs.append(["source", str(chosen_source)])
                current = chosen_source
                visited = {current}

                while True:
                    sink_var = tt_sink_vars.get((pid, current))
                    if sink_var is not None and value_from(value_source, sink_var) > 0.5:
                        arcs.append([str(current), "sink"])
                        break

                    next_trip = None
                    for (p, i, j), var in tt_pair_vars.items():
                        if (
                            p == pid
                            and i == current
                            and value_from(value_source, var) > 0.5
                        ):
                            next_trip = j
                            break

                    if next_trip is None or next_trip in visited:
                        break
                    arcs.append([str(current), str(next_trip)])
                    current = next_trip
                    visited.add(current)

            tt_arcs_used[pid] = arcs

        fleet_value = int(round(value_from(value_source, fleet)))
        flows = {"VS_O_plus-->VS_O_minus": fleet_value}

        for i in selected:
            start_node = tuple_node(i, "start")
            end_node = tuple_node(i, "end")
            flows[f"{start_node}-->{end_node}"] = 1

            if value_from(value_source, vehicle_start[i]) > 0.5:
                flows[f"VS_O_minus-->{start_node}"] = 1
            if value_from(value_source, vehicle_end[i]) > 0.5:
                flows[f"{end_node}-->VS_O_plus"] = 1

        for (i, j), var in vs_link_vars.items():
            if value_from(value_source, var) > 0.5:
                flows[
                    f"{tuple_node(i, 'end')}-->{tuple_node(j, 'start')}"
                ] = 1

        return {
            "objective_value": float(objective_value),
            "selected_trips": selected,
            "num_vehicles": fleet_value,
            "tt_arcs_used": tt_arcs_used,
            "vs_flows": flows,
        }

    best_logged = [float("inf")]

    def callback(cb_model, where):
        if where != GRB.Callback.MIPSOL or logger is None:
            return
        try:
            objective = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
            if objective >= best_logged[0] - 1e-7:
                return

            def cb_value(var):
                return cb_model.cbGetSolution(var)

            solution = build_solution(cb_value, objective)
            logger.log_solution(objective, solution)
            best_logged[0] = objective
        except Exception:
            # Logging must not interfere with the optimization itself.
            pass

    remaining = max(0.01, float(args.time_limit) - (time.monotonic() - wall_start))
    model.Params.TimeLimit = remaining
    model.optimize(callback)

    if model.SolCount <= 0:
        # Valid benchmark instances should be feasible. Still emit a schema-valid
        # result rather than leaving the requested solution path absent.
        solution = {
            "objective_value": 1e100,
            "selected_trips": [],
            "num_vehicles": 0,
            "tt_arcs_used": {pid: [] for pid in pattern_ids},
            "vs_flows": {"VS_O_plus-->VS_O_minus": 0},
        }
    else:
        solution = build_solution(None, model.ObjVal)
        if logger is not None and model.ObjVal < best_logged[0] - 1e-7:
            logger.log_solution(float(model.ObjVal), solution)
            best_logged[0] = float(model.ObjVal)

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()