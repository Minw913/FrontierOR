import argparse
import bisect
import heapq
import json
import math
import os
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


EPS = 1e-9


def get_travel_time(data, origin, destination):
    if origin == destination:
        return 0
    return int(data["travel_times"][origin][destination])


def type_allows_piece(duty_type, duration):
    return (
        duration >= duty_type["piece_length_min"]
        and duration <= duty_type["piece_length_max"]
    )


def duty_time_feasible(duty_type, duty_start, duty_end, work_time):
    if duty_type.get("duty_length_max") is not None:
        if duty_end - duty_start > duty_type["duty_length_max"]:
            return False
    if duty_type.get("work_time_max") is not None:
        if work_time > duty_type["work_time_max"]:
            return False
    if duty_type.get("start_time_min") is not None:
        if duty_start < duty_type["start_time_min"]:
            return False
    if duty_type.get("start_time_max") is not None:
        if duty_start > duty_type["start_time_max"]:
            return False
    if duty_type.get("end_time_max") is not None:
        if duty_end > duty_type["end_time_max"]:
            return False
    return True


def single_piece_duty_feasible(piece, duty_type, sign_on, sign_off):
    if duty_type["num_pieces"] != 1:
        return False
    if not type_allows_piece(duty_type, piece["duration"]):
        return False

    duty_start = piece["start"] - sign_on
    duty_end = piece["end"] + sign_off
    return duty_time_feasible(
        duty_type,
        duty_start,
        duty_end,
        piece["duration"],
    )


def two_piece_duty_feasible(first, second, duty_type, sign_on, sign_off):
    if duty_type["num_pieces"] != 2:
        return False
    if first["depot"] != second["depot"]:
        return False
    if not type_allows_piece(duty_type, first["duration"]):
        return False
    if not type_allows_piece(duty_type, second["duration"]):
        return False
    if second["start"] < first["end"]:
        return False

    minimum_break = duty_type.get("break_length_min")
    if minimum_break is not None:
        if second["start"] - first["end"] < minimum_break:
            return False

    duty_start = first["start"] - sign_on
    duty_end = second["end"] + sign_off
    work_time = first["duration"] + second["duration"]

    return duty_time_feasible(
        duty_type,
        duty_start,
        duty_end,
        work_time,
    )


def build_successors(data, trips, depot_name, max_successors):
    n = len(trips)
    successors = [[] for _ in range(n)]

    for i in range(n):
        a = trips[i]
        to_next = data["travel_times"][a["end_location"]]
        to_depot = get_travel_time(data, a["end_location"], depot_name)

        candidates = []
        for j in range(i + 1, n):
            b = trips[j]
            gap = b["start_time"] - a["end_time"]
            if gap < 0:
                continue

            direct = int(to_next[b["start_location"]])
            if direct > gap:
                continue

            via_depot = (
                to_depot
                + get_travel_time(data, depot_name, b["start_location"])
            )

            # If a depot return is possible, the connection is treated as a
            # long arc and therefore belongs between two depot excursions.
            if gap >= via_depot:
                continue

            candidates.append((b["start_time"], direct, j))

        candidates.sort()
        successors[i] = [j for _, _, j in candidates[:max_successors]]

    return successors


def generate_pieces(data, deadline):
    trips = data["trips"]
    depots = data["depots"]
    duty_types = data["duty_types"]

    piece_min = min(t["piece_length_min"] for t in duty_types.values())
    piece_max = max(t["piece_length_max"] for t in duty_types.values())

    n = len(trips)
    depot_count = max(1, len(depots))

    if n <= 100:
        max_successors = 14
        max_chains_per_start = 45
    elif n <= 250:
        max_successors = 10
        max_chains_per_start = 28
    else:
        max_successors = 7
        max_chains_per_start = 16

    global_cap = max(3000, min(30000, n * depot_count * max_chains_per_start))
    pieces = []
    seen = set()

    for depot in depots:
        depot_name = depot["name"]
        successors = build_successors(
            data, trips, depot_name, max_successors=max_successors
        )

        for first_idx in range(n):
            if time.monotonic() >= deadline or len(pieces) >= global_cap:
                break

            first = trips[first_idx]
            pullout = get_travel_time(
                data, depot_name, first["start_location"]
            )
            excursion_start = first["start_time"] - pullout
            generated_for_start = 0

            stack = [(first_idx, (first_idx,))]
            local_seen = set()

            while stack:
                if time.monotonic() >= deadline:
                    break
                last_idx, chain = stack.pop()
                if chain in local_seen:
                    continue
                local_seen.add(chain)

                last = trips[last_idx]
                pullin = get_travel_time(
                    data, last["end_location"], depot_name
                )
                excursion_end = last["end_time"] + pullin
                duration = excursion_end - excursion_start

                if duration > piece_max:
                    continue

                if duration >= piece_min:
                    ids = tuple(trips[k]["trip_id"] for k in chain)
                    key = (depot_name, ids)
                    if key not in seen:
                        seen.add(key)
                        pieces.append(
                            {
                                "depot": depot_name,
                                "indices": chain,
                                "trips": ids,
                                "start": excursion_start,
                                "end": excursion_end,
                                "duration": duration,
                            }
                        )
                        generated_for_start += 1
                        if len(pieces) >= global_cap:
                            break

                if generated_for_start >= max_chains_per_start:
                    continue

                next_nodes = successors[last_idx]
                # Reversed so the earliest successors are explored first.
                for nxt in reversed(next_nodes):
                    stack.append((nxt, chain + (nxt,)))

        if time.monotonic() >= deadline or len(pieces) >= global_cap:
            break

    return pieces


def generate_pairs(
    pieces,
    duty_types,
    sign_on,
    sign_off,
    deadline,
):
    two_types = [
        (name, rules)
        for name, rules in duty_types.items()
        if rules["num_pieces"] == 2
    ]

    by_depot = defaultdict(list)
    for p_idx, piece in enumerate(pieces):
        by_depot[piece["depot"]].append(p_idx)

    pairs = []
    pair_seen = set()

    # Limit pair arcs per first piece and duty type to control model size.
    if len(pieces) <= 4000:
        pair_limit = 24
        scan_limit = 180
    elif len(pieces) <= 10000:
        pair_limit = 14
        scan_limit = 100
    else:
        pair_limit = 8
        scan_limit = 60

    for depot_name, indices in by_depot.items():
        indices.sort(key=lambda i: (pieces[i]["start"], pieces[i]["end"]))
        starts = [pieces[i]["start"] for i in indices]

        for first_idx in indices:
            if time.monotonic() >= deadline:
                return pairs

            first = pieces[first_idx]

            for type_name, rules in two_types:
                if not type_allows_piece(rules, first["duration"]):
                    continue

                min_break = rules.get("break_length_min") or 0
                earliest = first["end"] + min_break
                pos = bisect.bisect_left(starts, earliest)

                accepted = 0
                scanned = 0
                used_second_first_trips = set()

                while (
                    pos < len(indices)
                    and accepted < pair_limit
                    and scanned < scan_limit
                ):
                    second_idx = indices[pos]
                    pos += 1
                    scanned += 1

                    if second_idx == first_idx:
                        continue
                    second = pieces[second_idx]

                    # Chronological separation normally implies disjointness,
                    # but retain an explicit check for zero-duration edge cases.
                    if set(first["trips"]).intersection(second["trips"]):
                        continue

                    if not two_piece_duty_feasible(
                        first, second, rules, sign_on, sign_off
                    ):
                        continue

                    # Encourage diversity instead of adding many columns whose
                    # second pieces begin with the same trip.
                    second_first_trip = second["trips"][0]
                    if (
                        second_first_trip in used_second_first_trips
                        and accepted >= max(3, pair_limit // 2)
                    ):
                        continue

                    key = (first_idx, second_idx, type_name)
                    if key not in pair_seen:
                        pair_seen.add(key)
                        pairs.append(key)
                        accepted += 1
                        used_second_first_trips.add(second_first_trip)

    return pairs


def color_vehicle_excursions(selected_piece_indices, pieces, depot_names):
    schedules = {name: [] for name in depot_names}

    by_depot = defaultdict(list)
    for p_idx in selected_piece_indices:
        by_depot[pieces[p_idx]["depot"]].append(p_idx)

    for depot_name in depot_names:
        intervals = by_depot.get(depot_name, [])
        intervals.sort(
            key=lambda i: (
                pieces[i]["start"],
                pieces[i]["end"],
                pieces[i]["trips"],
            )
        )

        # Heap entries are (available_time, vehicle_index).
        available = []
        vehicle_chains = []

        for p_idx in intervals:
            piece = pieces[p_idx]

            if available and available[0][0] <= piece["start"]:
                _, vehicle_idx = heapq.heappop(available)
            else:
                vehicle_idx = len(vehicle_chains)
                vehicle_chains.append([])

            vehicle_chains[vehicle_idx].extend(piece["trips"])
            heapq.heappush(
                available,
                (piece["end"], vehicle_idx),
            )

        schedules[depot_name] = vehicle_chains

    return schedules


def build_solution_from_selection(
    pieces,
    selected_singles,
    selected_pairs,
    depot_names,
):
    crew_schedule = []
    selected_piece_indices = set()

    for p_idx, type_name in selected_singles:
        piece = pieces[p_idx]
        selected_piece_indices.add(p_idx)
        crew_schedule.append(
            {
                "depot": piece["depot"],
                "type": type_name,
                "trips": list(piece["trips"]),
            }
        )

    for first_idx, second_idx, type_name in selected_pairs:
        first = pieces[first_idx]
        second = pieces[second_idx]
        selected_piece_indices.add(first_idx)
        selected_piece_indices.add(second_idx)
        crew_schedule.append(
            {
                "depot": first["depot"],
                "type": type_name,
                "trips": list(first["trips"]) + list(second["trips"]),
            }
        )

    vehicle_schedule = color_vehicle_excursions(
        selected_piece_indices, pieces, depot_names
    )
    number_of_vehicles = sum(len(v) for v in vehicle_schedule.values())
    objective = number_of_vehicles + len(crew_schedule)

    return {
        "objective_value": int(objective),
        "vehicle_schedule": vehicle_schedule,
        "crew_schedule": crew_schedule,
    }


def emergency_solution(data):
    trips = data["trips"]
    depots = data["depots"]
    depot_names = [d["name"] for d in depots]
    one_piece_types = [
        name
        for name, rules in data["duty_types"].items()
        if rules["num_pieces"] == 1
    ]
    fallback_type = (
        one_piece_types[0]
        if one_piece_types
        else next(iter(data["duty_types"].keys()))
    )

    vehicle_schedule = {name: [] for name in depot_names}
    crew_schedule = []

    for trip in trips:
        best_depot = min(
            depot_names,
            key=lambda d: (
                get_travel_time(data, d, trip["start_location"])
                + get_travel_time(data, trip["end_location"], d)
            ),
        )
        vehicle_schedule[best_depot].append([trip["trip_id"]])
        crew_schedule.append(
            {
                "depot": best_depot,
                "type": fallback_type,
                "trips": [trip["trip_id"]],
            }
        )

    objective = sum(len(v) for v in vehicle_schedule.values()) + len(
        crew_schedule
    )
    return {
        "objective_value": int(objective),
        "vehicle_schedule": vehicle_schedule,
        "crew_schedule": crew_schedule,
    }


def solve(data, time_limit, logger):
    start_clock = time.monotonic()
    deadline = start_clock + max(1, time_limit)

    trips = data["trips"]
    trip_ids = [t["trip_id"] for t in trips]
    depot_names = [d["name"] for d in data["depots"]]
    duty_types = data["duty_types"]
    crew_parameters = data["crew_parameters"]

    sign_on = int(crew_parameters["sign_on_time_depot_minutes"])
    sign_off = int(crew_parameters["sign_off_time_depot_minutes"])

    # Reserve most of the runtime for Gurobi.
    generation_deadline = min(
        deadline - 0.5,
        start_clock + max(0.5, min(20.0, time_limit * 0.22)),
    )

    pieces = generate_pieces(data, generation_deadline)

    if not pieces:
        return emergency_solution(data)

    piece_covering_trip = defaultdict(list)
    for p_idx, piece in enumerate(pieces):
        for trip_id in piece["trips"]:
            piece_covering_trip[trip_id].append(p_idx)

    if any(not piece_covering_trip[trip_id] for trip_id in trip_ids):
        return emergency_solution(data)

    pair_generation_deadline = min(
        deadline - 0.3,
        start_clock + max(0.8, min(35.0, time_limit * 0.35)),
    )
    pairs = generate_pairs(
        pieces,
        duty_types,
        sign_on,
        sign_off,
        pair_generation_deadline,
    )

    remaining = deadline - time.monotonic()
    if remaining <= 0.2:
        return emergency_solution(data)

    model = gp.Model("integrated_vehicle_crew_scheduling")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.1, remaining)
    model.Params.Heuristics = 0.2

    y = model.addVars(len(pieces), vtype=GRB.BINARY, name="piece")

    single_vars = {}
    one_piece_types = [
        (name, rules)
        for name, rules in duty_types.items()
        if rules["num_pieces"] == 1
    ]

    for p_idx, piece in enumerate(pieces):
        for type_name, rules in one_piece_types:
            if single_piece_duty_feasible(
                piece, rules, sign_on, sign_off
            ):
                single_vars[(p_idx, type_name)] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"single_{p_idx}_{type_name}",
                )

    pair_vars = {}
    for pair_idx, (first_idx, second_idx, type_name) in enumerate(pairs):
        pair_vars[(first_idx, second_idx, type_name)] = model.addVar(
            vtype=GRB.BINARY,
            name=f"pair_{pair_idx}",
        )

    vehicle_vars = {
        depot: model.addVar(
            vtype=GRB.INTEGER,
            lb=0,
            name=f"vehicles_{depot}",
        )
        for depot in depot_names
    }

    model.update()

    incident_duties = defaultdict(list)
    for (p_idx, type_name), var in single_vars.items():
        incident_duties[p_idx].append(var)

    for (first_idx, second_idx, type_name), var in pair_vars.items():
        incident_duties[first_idx].append(var)
        incident_duties[second_idx].append(var)

    for p_idx in range(len(pieces)):
        if incident_duties[p_idx]:
            model.addConstr(
                y[p_idx] == gp.quicksum(incident_duties[p_idx]),
                name=f"piece_duty_{p_idx}",
            )
        else:
            model.addConstr(y[p_idx] == 0, name=f"disable_piece_{p_idx}")

    for trip_id in trip_ids:
        model.addConstr(
            gp.quicksum(y[p] for p in piece_covering_trip[trip_id]) == 1,
            name=f"cover_trip_{trip_id}",
        )

    pieces_by_depot = defaultdict(list)
    for p_idx, piece in enumerate(pieces):
        pieces_by_depot[piece["depot"]].append(p_idx)

    # The minimum number of vehicles for depot-to-depot excursions equals
    # the maximum number of overlapping excursion intervals.
    for depot_name in depot_names:
        depot_pieces = pieces_by_depot.get(depot_name, [])
        event_times = sorted({pieces[p]["start"] for p in depot_pieces})

        for event_no, event_time in enumerate(event_times):
            active = [
                p
                for p in depot_pieces
                if pieces[p]["start"] <= event_time < pieces[p]["end"]
            ]
            if active:
                model.addConstr(
                    vehicle_vars[depot_name]
                    >= gp.quicksum(y[p] for p in active),
                    name=f"fleet_{depot_name}_{event_no}",
                )

    driver_cost = gp.quicksum(single_vars.values()) + gp.quicksum(
        pair_vars.values()
    )
    vehicle_cost = gp.quicksum(vehicle_vars.values())
    model.setObjective(driver_cost + vehicle_cost, GRB.MINIMIZE)

    best_logged = [math.inf]

    def incumbent_callback(cb_model, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            try:
                value = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
                rounded = int(round(value))
                if rounded < best_logged[0]:
                    best_logged[0] = rounded
                    logger.log(rounded)
            except Exception:
                pass

    try:
        model.optimize(incumbent_callback)
    except gp.GurobiError:
        return emergency_solution(data)

    if model.SolCount <= 0:
        return emergency_solution(data)

    selected_singles = [
        (p_idx, type_name)
        for (p_idx, type_name), var in single_vars.items()
        if var.X > 0.5
    ]
    selected_pairs = [
        (first_idx, second_idx, type_name)
        for (first_idx, second_idx, type_name), var in pair_vars.items()
        if var.X > 0.5
    ]

    solution = build_solution_from_selection(
        pieces,
        selected_singles,
        selected_pairs,
        depot_names,
    )

    if logger:
        logger.log_solution(solution["objective_value"], solution)

    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path
        else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    solution = solve(data, args.time_limit, logger)

    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(output_directory, exist_ok=True)

    temporary_path = args.solution_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump(solution, f, separators=(",", ":"))
    os.replace(temporary_path, args.solution_path)


if __name__ == "__main__":
    main()