import argparse
import json
import math
import os
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


TOL = 1e-7


def lookup_numeric(mapping, value):
    """Look up a JSON dictionary whose keys encode numeric values."""
    candidates = [
        str(value),
        f"{value:g}",
        f"{float(value):.1f}",
        f"{float(value):.2f}",
        f"{float(value):.3f}",
    ]
    for key in candidates:
        if key in mapping:
            return float(mapping[key])

    target = float(value)
    for key, result in mapping.items():
        try:
            if abs(float(key) - target) <= 1e-8:
                return float(result)
        except (TypeError, ValueError):
            pass
    raise KeyError(f"No entry for numeric key {value}")


def distance_between(instance, origin, destination):
    key = f"{origin}_{destination}"
    distances = instance["distances_nautical_miles"]
    if key in distances:
        return float(distances[key])
    raise KeyError(f"Missing distance for ordered port pair {origin} -> {destination}")


def clamp(value, lower, upper):
    return min(max(value, lower), upper)


def intervals_overlap(a_start, a_end, b_start, b_end):
    return a_start < b_end - TOL and b_start < a_end - TOL


def spatial_overlap(a_pos, a_len, b_pos, b_len):
    return a_pos < b_pos + b_len - TOL and b_pos < a_pos + a_len - TOL


def prepare_data(instance):
    ports = {p["id"]: p for p in instance["ports"]}
    speeds = [float(v) for v in instance["speeds_knots"]]

    ships = []
    for original_ship in instance["ships"]:
        ship = dict(original_ship)
        visits = sorted(
            original_ship["port_visits"],
            key=lambda v: int(v["port_call_index"]),
        )
        ship["_visits"] = visits
        ship["_index_by_call"] = {
            int(v["port_call_index"]): i for i, v in enumerate(visits)
        }
        ships.append(ship)

    external_by_port = defaultdict(list)
    for port_group in instance.get("external_ships", []):
        port_id = port_group["port_id"]
        for external in port_group.get("ships", []):
            record = dict(external)
            record["_port_id"] = port_id
            record["_start"] = float(record["start_time_hours"])
            record["_handling"] = float(record["handling_time_hours"])
            record["_end"] = float(
                record.get(
                    "end_time_hours",
                    record["_start"] + record["_handling"],
                )
            )
            external_by_port[port_id].append(record)

    return ports, speeds, ships, external_by_port


def calculate_objective(instance, ships, solution):
    costs = instance["cost_parameters"]
    fuel_price = float(costs["fuel_price_usd_per_tonne"])
    handling_rate = float(costs["handling_cost_usd_per_hour"])
    delay_rate = float(costs["delay_cost_usd_per_hour"])
    waiting_rate = float(costs["waiting_cost_usd_per_hour"])
    lft_rate = float(costs["lft_penalty_usd"])

    ship_data_by_id = {int(s["id"]): s for s in ships}
    total = 0.0

    for ship_solution in solution["ships"]:
        ship = ship_data_by_id[int(ship_solution["ship_id"])]
        visits_by_call = {
            int(v["port_call_index"]): v for v in ship_solution["port_visits"]
        }

        for visit_solution in ship_solution["port_visits"]:
            waiting = max(
                0.0,
                float(visit_solution["start_time"])
                - float(visit_solution["arrival_time"]),
            )
            total += waiting_rate * waiting
            total += handling_rate * float(visit_solution["handling_time"])
            total += delay_rate * float(visit_solution["delay"])
            total += lft_rate * float(visit_solution["lft_violation"])

        for speed_solution in ship_solution["speed_selections"]:
            from_call = int(speed_solution["from_call"])
            to_call = int(speed_solution["to_call"])
            speed = float(speed_solution["speed_knots"])

            from_visit = ship["_visits"][ship["_index_by_call"][from_call]]
            to_visit = ship["_visits"][ship["_index_by_call"][to_call]]
            distance = distance_between(
                instance,
                from_visit["port_id"],
                to_visit["port_id"],
            )
            consumption = lookup_numeric(
                ship["fuel_consumption_per_nm_by_speed"], speed
            )
            total += fuel_price * consumption * distance

    return float(total)


def construct_heuristic(instance, ports, speeds, ships, external_by_port):
    """
    Construct a guaranteed conflict-free schedule by processing ships and
    visits sequentially. Positions are kept at their ideal values where
    possible, and a visit is delayed until all spatially conflicting existing
    occupations have ended.
    """
    beta = float(instance["handling_time_deviation_factor_beta_per_meter"])
    occupations = defaultdict(list)
    for port_id, external_ships in external_by_port.items():
        for ext in external_ships:
            occupations[port_id].append(
                {
                    "position": float(ext["berthing_position"]),
                    "length": float(ext["length_meters"]),
                    "start": ext["_start"],
                    "end": ext["_end"],
                }
            )

    solution = {"objective_value": 0.0, "ships": []}

    for ship in ships:
        ship_length = float(ship["length_meters"])
        fuel_map = ship["fuel_consumption_per_nm_by_speed"]

        # A fuel-efficient baseline speed. The exact MIP subsequently improves it.
        selected_speeds = [
            min(speeds, key=lambda s: lookup_numeric(fuel_map, s))
            for _ in range(max(0, len(ship["_visits"]) - 1))
        ]

        ship_output = {
            "ship_id": int(ship["id"]),
            "port_visits": [],
            "speed_selections": [],
        }

        previous_end = None
        for visit_index, visit in enumerate(ship["_visits"]):
            port_id = visit["port_id"]
            quay_length = float(ports[port_id]["quay_length_meters"])
            max_position = quay_length - ship_length
            if max_position < -TOL:
                raise ValueError(
                    f"Ship {ship['id']} is longer than the quay at {port_id}"
                )

            ideal = float(visit["ideal_berthing_position"])
            position = clamp(ideal, 0.0, max(0.0, max_position))
            deviation = abs(position - ideal)
            minimum_handling = float(visit["min_handling_time_hours"])
            handling = minimum_handling * (1.0 + beta * deviation)

            if visit_index == 0:
                arrival = max(
                    0.0, float(visit["earliest_start_time_hours"])
                )
            else:
                previous_visit = ship["_visits"][visit_index - 1]
                speed = selected_speeds[visit_index - 1]
                distance = distance_between(
                    instance, previous_visit["port_id"], port_id
                )
                travel = distance * lookup_numeric(
                    instance["travel_time_hours_per_nm_by_speed"], speed
                )
                arrival = previous_end + travel

            start = max(
                arrival,
                float(visit["earliest_start_time_hours"]),
                0.0,
            )

            # Push this call forward until it has no time-space collision.
            while True:
                new_start = start
                for occupied in occupations[port_id]:
                    if not spatial_overlap(
                        position,
                        ship_length,
                        occupied["position"],
                        occupied["length"],
                    ):
                        continue
                    if intervals_overlap(
                        start,
                        start + handling,
                        occupied["start"],
                        occupied["end"],
                    ):
                        new_start = max(new_start, occupied["end"])
                if new_start <= start + TOL:
                    break
                start = new_start

            if visit_index == 0:
                # The first-call arrival is unconstrained except for arrival <= start.
                arrival = start

            end = start + handling
            expected = float(visit["expected_finish_time_hours"])
            latest = float(visit["latest_finish_time_hours"])
            delay = max(0.0, end - expected)
            violation = max(0.0, end - latest)

            ship_output["port_visits"].append(
                {
                    "port_call_index": int(visit["port_call_index"]),
                    "port_id": port_id,
                    "berthing_position": float(position),
                    "start_time": float(start),
                    "handling_time": float(handling),
                    "arrival_time": float(arrival),
                    "delay": float(delay),
                    "lft_violation": float(violation),
                    "position_deviation": float(deviation),
                }
            )

            occupations[port_id].append(
                {
                    "position": position,
                    "length": ship_length,
                    "start": start,
                    "end": end,
                }
            )
            previous_end = end

        for leg_index, speed in enumerate(selected_speeds):
            from_visit = ship["_visits"][leg_index]
            to_visit = ship["_visits"][leg_index + 1]
            ship_output["speed_selections"].append(
                {
                    "from_call": int(from_visit["port_call_index"]),
                    "to_call": int(to_visit["port_call_index"]),
                    "speed_knots": float(speed),
                }
            )

        solution["ships"].append(ship_output)

    solution["objective_value"] = calculate_objective(
        instance, ships, solution
    )
    return solution


def build_model(instance, ports, speeds, ships, external_by_port, heuristic):
    model = gp.Model("integrated_berth_speed_optimization")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    beta = float(instance["handling_time_deviation_factor_beta_per_meter"])
    costs = instance["cost_parameters"]
    fuel_price = float(costs["fuel_price_usd_per_tonne"])
    handling_rate = float(costs["handling_cost_usd_per_hour"])
    delay_rate = float(costs["delay_cost_usd_per_hour"])
    waiting_rate = float(costs["waiting_cost_usd_per_hour"])
    lft_rate = float(costs["lft_penalty_usd"])

    variables = {}
    visits_at_port = defaultdict(list)
    objective = gp.LinExpr()

    for ship_index, ship in enumerate(ships):
        ship_length = float(ship["length_meters"])
        ship_vars = {
            "visits": [],
            "speed_vars": [],
        }
        variables[ship_index] = ship_vars

        for visit_index, visit in enumerate(ship["_visits"]):
            port_id = visit["port_id"]
            call = int(visit["port_call_index"])
            quay = float(ports[port_id]["quay_length_meters"])
            upper_position = quay - ship_length
            if upper_position < -TOL:
                raise ValueError(
                    f"Ship {ship['id']} is longer than quay at {port_id}"
                )

            prefix = f"s{ship_index}_c{call}"
            position = model.addVar(
                lb=0.0,
                ub=max(0.0, upper_position),
                name=f"x_{prefix}",
            )
            start = model.addVar(lb=0.0, name=f"start_{prefix}")
            arrival = model.addVar(lb=0.0, name=f"arrival_{prefix}")
            deviation = model.addVar(lb=0.0, name=f"dev_{prefix}")
            handling = model.addVar(lb=0.0, name=f"handling_{prefix}")
            waiting = model.addVar(lb=0.0, name=f"waiting_{prefix}")
            delay = model.addVar(lb=0.0, name=f"delay_{prefix}")
            violation = model.addVar(lb=0.0, name=f"lft_{prefix}")
            side = model.addVar(vtype=GRB.BINARY, name=f"side_{prefix}")

            ideal = float(visit["ideal_berthing_position"])
            minimum_handling = float(visit["min_handling_time_hours"])

            model.addGenConstrIndicator(
                side, True, deviation == position - ideal
            )
            model.addGenConstrIndicator(
                side, False, deviation == ideal - position
            )
            model.addConstr(
                handling
                == minimum_handling * (1.0 + beta * deviation),
                name=f"handling_definition_{prefix}",
            )
            model.addConstr(
                start >= float(visit["earliest_start_time_hours"]),
                name=f"earliest_{prefix}",
            )
            model.addConstr(arrival <= start, name=f"arrival_before_start_{prefix}")
            model.addConstr(waiting == start - arrival, name=f"waiting_{prefix}")
            model.addConstr(
                delay
                >= start
                + handling
                - float(visit["expected_finish_time_hours"]),
                name=f"delay_definition_{prefix}",
            )
            model.addConstr(
                violation
                >= start
                + handling
                - float(visit["latest_finish_time_hours"]),
                name=f"lft_definition_{prefix}",
            )

            visit_vars = {
                "position": position,
                "start": start,
                "arrival": arrival,
                "deviation": deviation,
                "handling": handling,
                "waiting": waiting,
                "delay": delay,
                "violation": violation,
                "side": side,
                "ship_index": ship_index,
                "visit_index": visit_index,
                "ship_length": ship_length,
                "port_id": port_id,
            }
            ship_vars["visits"].append(visit_vars)
            visits_at_port[port_id].append(visit_vars)

            objective += waiting_rate * waiting
            objective += handling_rate * handling
            objective += delay_rate * delay
            objective += lft_rate * violation

        for leg_index in range(len(ship["_visits"]) - 1):
            from_visit = ship["_visits"][leg_index]
            to_visit = ship["_visits"][leg_index + 1]
            distance = distance_between(
                instance, from_visit["port_id"], to_visit["port_id"]
            )

            leg_vars = []
            for speed_index, speed in enumerate(speeds):
                z = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"speed_s{ship_index}_l{leg_index}_k{speed_index}",
                )
                leg_vars.append(z)
                fuel_per_nm = lookup_numeric(
                    ship["fuel_consumption_per_nm_by_speed"], speed
                )
                objective += fuel_price * fuel_per_nm * distance * z

            model.addConstr(
                gp.quicksum(leg_vars) == 1,
                name=f"one_speed_s{ship_index}_l{leg_index}",
            )
            ship_vars["speed_vars"].append(leg_vars)

            previous = ship_vars["visits"][leg_index]
            current = ship_vars["visits"][leg_index + 1]
            travel_time = gp.quicksum(
                distance
                * lookup_numeric(
                    instance["travel_time_hours_per_nm_by_speed"], speed
                )
                * leg_vars[k]
                for k, speed in enumerate(speeds)
            )
            model.addConstr(
                current["arrival"]
                == previous["start"] + previous["handling"] + travel_time,
                name=f"arrival_link_s{ship_index}_l{leg_index}",
            )

    # Pairwise non-overlap between optimized port calls.
    disjunction_groups = []
    for port_id, port_visits in visits_at_port.items():
        for i in range(len(port_visits)):
            a = port_visits[i]
            for j in range(i + 1, len(port_visits)):
                b = port_visits[j]
                binaries = [
                    model.addVar(vtype=GRB.BINARY, name=f"sep_{port_id}_{i}_{j}_{k}")
                    for k in range(4)
                ]
                model.addConstr(gp.quicksum(binaries) == 1)

                model.addGenConstrIndicator(
                    binaries[0],
                    True,
                    a["position"] + a["ship_length"] <= b["position"],
                )
                model.addGenConstrIndicator(
                    binaries[1],
                    True,
                    b["position"] + b["ship_length"] <= a["position"],
                )
                model.addGenConstrIndicator(
                    binaries[2],
                    True,
                    a["start"] + a["handling"] <= b["start"],
                )
                model.addGenConstrIndicator(
                    binaries[3],
                    True,
                    b["start"] + b["handling"] <= a["start"],
                )
                disjunction_groups.append(binaries)

    # Non-overlap between optimized and fixed external ships.
    for port_id, port_visits in visits_at_port.items():
        for i, visit_vars in enumerate(port_visits):
            for e, ext in enumerate(external_by_port.get(port_id, [])):
                ext_pos = float(ext["berthing_position"])
                ext_len = float(ext["length_meters"])
                ext_start = ext["_start"]
                ext_end = ext["_end"]

                binaries = [
                    model.addVar(
                        vtype=GRB.BINARY,
                        name=f"extsep_{port_id}_{i}_{e}_{k}",
                    )
                    for k in range(4)
                ]
                model.addConstr(gp.quicksum(binaries) == 1)

                model.addGenConstrIndicator(
                    binaries[0],
                    True,
                    visit_vars["position"] + visit_vars["ship_length"] <= ext_pos,
                )
                model.addGenConstrIndicator(
                    binaries[1],
                    True,
                    ext_pos + ext_len <= visit_vars["position"],
                )
                model.addGenConstrIndicator(
                    binaries[2],
                    True,
                    visit_vars["start"] + visit_vars["handling"] <= ext_start,
                )
                model.addGenConstrIndicator(
                    binaries[3],
                    True,
                    ext_end <= visit_vars["start"],
                )
                disjunction_groups.append(binaries)

    model.setObjective(objective, GRB.MINIMIZE)

    # Supply the constructive schedule as a partial MIP start.
    heuristic_by_ship = {
        int(s["ship_id"]): s for s in heuristic["ships"]
    }
    for ship_index, ship in enumerate(ships):
        hs = heuristic_by_ship[int(ship["id"])]
        h_visits = {
            int(v["port_call_index"]): v for v in hs["port_visits"]
        }
        h_speeds = {
            (int(v["from_call"]), int(v["to_call"])): float(v["speed_knots"])
            for v in hs["speed_selections"]
        }

        for visit_index, visit in enumerate(ship["_visits"]):
            call = int(visit["port_call_index"])
            hv = h_visits[call]
            vv = variables[ship_index]["visits"][visit_index]

            vv["position"].Start = float(hv["berthing_position"])
            vv["start"].Start = float(hv["start_time"])
            vv["arrival"].Start = float(hv["arrival_time"])
            vv["deviation"].Start = float(hv["position_deviation"])
            vv["handling"].Start = float(hv["handling_time"])
            vv["waiting"].Start = max(
                0.0, float(hv["start_time"]) - float(hv["arrival_time"])
            )
            vv["delay"].Start = float(hv["delay"])
            vv["violation"].Start = float(hv["lft_violation"])
            vv["side"].Start = (
                1.0
                if float(hv["berthing_position"])
                >= float(visit["ideal_berthing_position"]) - TOL
                else 0.0
            )

        for leg_index, leg_vars in enumerate(
            variables[ship_index]["speed_vars"]
        ):
            from_call = int(ship["_visits"][leg_index]["port_call_index"])
            to_call = int(ship["_visits"][leg_index + 1]["port_call_index"])
            chosen = h_speeds[(from_call, to_call)]
            for k, speed in enumerate(speeds):
                leg_vars[k].Start = 1.0 if abs(speed - chosen) <= 1e-8 else 0.0

    model.update()
    return model, variables


def extract_solution(instance, ships, speeds, variables, getter):
    solution = {"objective_value": 0.0, "ships": []}

    for ship_index, ship in enumerate(ships):
        ship_output = {
            "ship_id": int(ship["id"]),
            "port_visits": [],
            "speed_selections": [],
        }

        for visit_index, visit in enumerate(ship["_visits"]):
            vv = variables[ship_index]["visits"][visit_index]
            position = float(getter(vv["position"]))
            start = float(getter(vv["start"]))
            arrival = float(getter(vv["arrival"]))
            handling = float(getter(vv["handling"]))
            deviation = float(getter(vv["deviation"]))
            delay = max(0.0, float(getter(vv["delay"])))
            violation = max(0.0, float(getter(vv["violation"])))

            ship_output["port_visits"].append(
                {
                    "port_call_index": int(visit["port_call_index"]),
                    "port_id": visit["port_id"],
                    "berthing_position": position,
                    "start_time": start,
                    "handling_time": handling,
                    "arrival_time": arrival,
                    "delay": delay,
                    "lft_violation": violation,
                    "position_deviation": deviation,
                }
            )

        for leg_index, leg_vars in enumerate(
            variables[ship_index]["speed_vars"]
        ):
            values = [float(getter(z)) for z in leg_vars]
            chosen_index = max(range(len(values)), key=values.__getitem__)
            ship_output["speed_selections"].append(
                {
                    "from_call": int(
                        ship["_visits"][leg_index]["port_call_index"]
                    ),
                    "to_call": int(
                        ship["_visits"][leg_index + 1]["port_call_index"]
                    ),
                    "speed_knots": float(speeds[chosen_index]),
                }
            )

        solution["ships"].append(ship_output)

    solution["objective_value"] = calculate_objective(
        instance, ships, solution
    )
    return solution


def write_solution(path, solution):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(solution, output_file, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_clock = time.monotonic()
    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path
        else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as input_file:
        instance = json.load(input_file)

    ports, speeds, ships, external_by_port = prepare_data(instance)
    heuristic = construct_heuristic(
        instance, ports, speeds, ships, external_by_port
    )
    best_solution = heuristic
    best_objective = float(heuristic["objective_value"])

    if logger:
        logger.log_solution(best_objective, best_solution)

    try:
        model, variables = build_model(
            instance,
            ports,
            speeds,
            ships,
            external_by_port,
            heuristic,
        )

        elapsed = time.monotonic() - start_clock
        model.Params.TimeLimit = max(0.01, float(args.time_limit) - elapsed)

        callback_state = {"best_logged": best_objective}

        def incumbent_callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return
            try:
                candidate = extract_solution(
                    instance,
                    ships,
                    speeds,
                    variables,
                    lambda var: cb_model.cbGetSolution(var),
                )
                objective = float(candidate["objective_value"])
                if objective < callback_state["best_logged"] - 1e-6:
                    callback_state["best_logged"] = objective
                    if logger:
                        logger.log_solution(objective, candidate)
            except Exception:
                # Logging must never interrupt the optimization search.
                pass

        model.optimize(incumbent_callback)

        if model.SolCount > 0:
            candidate = extract_solution(
                instance,
                ships,
                speeds,
                variables,
                lambda var: var.X,
            )
            candidate_objective = float(candidate["objective_value"])
            if candidate_objective < best_objective - 1e-6:
                best_solution = candidate
                best_objective = candidate_objective
                if logger and candidate_objective < callback_state["best_logged"] - 1e-6:
                    logger.log_solution(candidate_objective, candidate)

    except (gp.GurobiError, ValueError, KeyError):
        # The constructive schedule remains available if model construction,
        # licensing, or optimization fails.
        pass

    best_solution["objective_value"] = float(best_objective)
    write_solution(args.solution_path, best_solution)


if __name__ == "__main__":
    main()