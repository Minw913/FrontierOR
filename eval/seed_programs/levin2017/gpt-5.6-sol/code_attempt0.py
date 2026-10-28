import argparse
import json
import os
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def clean_value(value, tolerance=1e-8):
    value = float(value)
    if abs(value) <= tolerance:
        return 0.0
    if value < 0.0 and value >= -tolerance:
        return 0.0
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    network = instance["network"]
    time_data = instance["time_parameters"]
    demand_data = instance["demand"]
    fleet_data = instance["fleet"]

    T = int(time_data["time_horizon_T"])
    dt_seconds = float(time_data["time_step_duration_seconds"])
    decision_times = range(T)
    state_times = range(T + 1)

    nodes = network["nodes"]
    links_raw = network["links"]

    centroids = [str(n["id"]) for n in nodes if n["type"] == "centroid"]
    junctions = [str(n["id"]) for n in nodes if n["type"] == "junction"]
    centroid_set = set(centroids)
    junction_set = set(junctions)

    links = []
    link_by_id = {}
    outgoing_links = defaultdict(list)
    incoming_links = defaultdict(list)

    for raw in links_raw:
        link = {
            "id": raw["id"],
            "from": str(raw["from"]),
            "to": str(raw["to"]),
            "type": raw["type"],
            "fftt": max(1, int(raw.get("free_flow_travel_time_steps", 1))),
            "cwtt": max(1, int(raw.get("congested_travel_time_steps", 1))),
            "capacity": float(raw.get("capacity_vph", 0.0)) * dt_seconds / 3600.0,
            "jam": float(raw.get("jam_density_vehicles", 0.0)),
        }
        links.append(link)
        link_by_id[link["id"]] = link
        outgoing_links[link["from"]].append(link["id"])
        incoming_links[link["to"]].append(link["id"])

    road_links = [l["id"] for l in links if l["type"] == "road"]
    connector_links = [l["id"] for l in links if l["type"] == "centroid_connector"]

    outgoing_connectors = [
        l["id"]
        for l in links
        if l["type"] == "centroid_connector"
        and l["from"] in centroid_set
        and l["to"] in junction_set
    ]
    incoming_connectors = [
        l["id"]
        for l in links
        if l["type"] == "centroid_connector"
        and l["from"] in junction_set
        and l["to"] in centroid_set
    ]

    outgoing_connector_set = set(outgoing_connectors)
    incoming_connector_set = set(incoming_connectors)
    road_link_set = set(road_links)

    centroid_out_connectors = defaultdict(list)
    centroid_in_connectors = defaultdict(list)
    for lid in outgoing_connectors:
        centroid_out_connectors[link_by_id[lid]["from"]].append(lid)
    for lid in incoming_connectors:
        centroid_in_connectors[link_by_id[lid]["to"]].append(lid)

    # A turn is made at a junction from a link ending at that junction
    # to a link beginning at that junction.
    turns = []
    outgoing_turns = defaultdict(list)
    incoming_turns = defaultdict(list)

    for j in junctions:
        for in_lid in incoming_links[j]:
            in_link = link_by_id[in_lid]
            if in_link["type"] != "road" and in_lid not in outgoing_connector_set:
                continue
            for out_lid in outgoing_links[j]:
                out_link = link_by_id[out_lid]
                if out_link["type"] != "road" and out_lid not in incoming_connector_set:
                    continue
                turn = (in_lid, out_lid)
                turns.append(turn)
                outgoing_turns[in_lid].append(turn)
                incoming_turns[out_lid].append(turn)

    demand = defaultdict(float)
    active_od = set()
    for od in demand_data.get("od_pairs", []):
        origin = str(od["origin"])
        destination = str(od["destination"])
        active_od.add((origin, destination))
        for departure_time in od.get("departure_times", []):
            t = int(departure_time)
            if 0 <= t < T:
                demand[(origin, destination, t)] += 1.0

    # Retain OD entries even if their traveler list is empty.
    for od in demand_data.get("od_pairs", []):
        active_od.add((str(od["origin"]), str(od["destination"])))

    initial_distribution_raw = fleet_data.get("initial_distribution", {})
    initial_parked = {
        c: float(initial_distribution_raw.get(c, initial_distribution_raw.get(str(c), 0.0)))
        for c in centroids
    }
    initial_fleet = sum(initial_parked.values())

    model = gp.Model("SAV_dynamic_traffic_assignment")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.Method = 0

    # Cumulative link counts.
    N_U = {}
    N_D = {}
    for link in links:
        lid = link["id"]
        terminal_centroid = link["to"] if lid in incoming_connector_set else None
        for d in centroids:
            forbidden = terminal_centroid is not None and d != terminal_centroid
            ub = 0.0 if forbidden else GRB.INFINITY
            for t in state_times:
                if t == 0:
                    N_U[(lid, d, t)] = model.addVar(lb=0.0, ub=0.0)
                    N_D[(lid, d, t)] = model.addVar(lb=0.0, ub=0.0)
                else:
                    N_U[(lid, d, t)] = model.addVar(lb=0.0, ub=ub)
                    N_D[(lid, d, t)] = model.addVar(lb=0.0, ub=ub)

    # Junction turning flows.
    y = {}
    for in_lid, out_lid in turns:
        out_link = link_by_id[out_lid]
        terminal_centroid = out_link["to"] if out_lid in incoming_connector_set else None
        for d in centroids:
            ub = 0.0 if terminal_centroid is not None and d != terminal_centroid else GRB.INFINITY
            for t in decision_times:
                y[(in_lid, out_lid, d, t)] = model.addVar(lb=0.0, ub=ub)

    # Departures from parked fleets into outgoing centroid connectors.
    y_centroid = {}
    for c in centroids:
        for lid in centroid_out_connectors[c]:
            for d in centroids:
                for t in decision_times:
                    y_centroid[(c, lid, d, t)] = model.addVar(lb=0.0)

    # Parked fleet.
    p = {}
    for c in centroids:
        for t in state_times:
            if t == 0:
                p[(c, t)] = model.addVar(
                    lb=initial_parked[c], ub=initial_parked[c]
                )
            else:
                p[(c, t)] = model.addVar(lb=0.0)

    # Traveler pickup and waiting variables are needed only for OD pairs
    # appearing in the input. Missing OD pairs are identically zero.
    e = {}
    omega = {}
    for o, d in sorted(active_od):
        for t in decision_times:
            e[(o, d, t)] = model.addVar(lb=0.0)
        for t in state_times:
            if t == 0 or t == T:
                omega[(o, d, t)] = model.addVar(lb=0.0, ub=0.0)
            else:
                omega[(o, d, t)] = model.addVar(lb=0.0)

    # Upstream cumulative-count dynamics.
    for link in links:
        lid = link["id"]
        for d in centroids:
            for t in decision_times:
                if lid in outgoing_connector_set:
                    c = link["from"]
                    entry_flow = y_centroid[(c, lid, d, t)]
                else:
                    entry_flow = gp.quicksum(
                        y[(in_lid, lid, d, t)]
                        for in_lid, _ in incoming_turns[lid]
                    )

                model.addConstr(
                    N_U[(lid, d, t + 1)]
                    == N_U[(lid, d, t)] + entry_flow
                )

    # Downstream cumulative-count dynamics.
    for link in links:
        lid = link["id"]
        for d in centroids:
            for t in decision_times:
                if lid in incoming_connector_set:
                    # Incoming centroid connectors have one-step free-flow travel.
                    model.addConstr(
                        N_D[(lid, d, t + 1)] == N_U[(lid, d, t)]
                    )
                else:
                    exit_flow = gp.quicksum(
                        y[(lid, out_lid, d, t)]
                        for _, out_lid in outgoing_turns[lid]
                    )
                    model.addConstr(
                        N_D[(lid, d, t + 1)]
                        == N_D[(lid, d, t)] + exit_flow
                    )

    # Sending constraints for roads and outgoing centroid connectors.
    for lid in road_links + outgoing_connectors:
        link = link_by_id[lid]
        tau = link["fftt"]

        for d in centroids:
            for t in decision_times:
                exit_flow = gp.quicksum(
                    y[(lid, out_lid, d, t)]
                    for _, out_lid in outgoing_turns[lid]
                )

                if t < tau:
                    model.addConstr(exit_flow == 0.0)
                else:
                    delayed_index = t - tau + 1
                    model.addConstr(
                        exit_flow
                        <= N_U[(lid, d, delayed_index)] - N_D[(lid, d, t)]
                    )

    # Road-link flow capacity and receiving constraints.
    for lid in road_links:
        link = link_by_id[lid]
        capacity = link["capacity"]
        jam = link["jam"]
        cwtt = link["cwtt"]

        for t in decision_times:
            total_exit = gp.quicksum(
                y[(lid, out_lid, d, t)]
                for _, out_lid in outgoing_turns[lid]
                for d in centroids
            )
            model.addConstr(total_exit <= capacity)

            total_entry = gp.quicksum(
                y[(in_lid, lid, d, t)]
                for in_lid, _ in incoming_turns[lid]
                for d in centroids
            )
            model.addConstr(total_entry <= capacity)

            # Standard LTM receiving bound:
            # N_D(t-cwtt+1) + jam occupancy - N_U(t).
            delayed_index = t - cwtt + 1
            if delayed_index >= 0:
                delayed_downstream = gp.quicksum(
                    N_D[(lid, d, delayed_index)] for d in centroids
                )
            else:
                delayed_downstream = 0.0

            current_upstream = gp.quicksum(
                N_U[(lid, d, t)] for d in centroids
            )
            model.addConstr(
                total_entry <= delayed_downstream + jam - current_upstream
            )

    # Parked-fleet conservation and departure availability.
    for c in centroids:
        for t in decision_times:
            arrivals = gp.quicksum(
                N_U[(lid, c, t)] - N_D[(lid, c, t)]
                for lid in centroid_in_connectors[c]
            )
            departures = gp.quicksum(
                y_centroid[(c, lid, d, t)]
                for lid in centroid_out_connectors[c]
                for d in centroids
            )

            model.addConstr(
                p[(c, t + 1)] == p[(c, t)] + arrivals - departures
            )
            model.addConstr(departures <= p[(c, t)])

    model.addConstr(
        gp.quicksum(p[(c, T)] for c in centroids) == initial_fleet
    )

    # Traveler waiting dynamics and coupling to vehicle departures.
    for o, d in active_od:
        for t in decision_times:
            model.addConstr(
                omega[(o, d, t + 1)]
                == omega[(o, d, t)] + demand[(o, d, t)] - e[(o, d, t)]
            )
            model.addConstr(e[(o, d, t)] <= omega[(o, d, t)])

            destination_departures = gp.quicksum(
                y_centroid[(o, lid, d, t)]
                for lid in centroid_out_connectors[o]
            )
            model.addConstr(e[(o, d, t)] <= destination_departures)

    vehicle_time = gp.quicksum(
        N_U[(link["id"], d, t)] - N_D[(link["id"], d, t)]
        for link in links
        for d in centroids
        for t in state_times
    )
    waiting_time = gp.quicksum(
        omega[(o, d, t)]
        for o, d in active_od
        for t in state_times
    )

    model.setObjective(vehicle_time + waiting_time, GRB.MINIMIZE)

    elapsed = time.monotonic() - start_time
    model.Params.TimeLimit = max(0.01, float(args.time_limit) - elapsed)
    model.optimize()

    if model.SolCount <= 0:
        raise RuntimeError(
            f"No feasible solution was found within the time limit "
            f"(Gurobi status {model.Status})."
        )

    solution = {
        "objective_value": clean_value(model.ObjVal),
        "y": {},
        "y_centroid": {},
        "N_U": {},
        "N_D": {},
        "p": {},
        "e": {},
        "omega": {},
    }

    for in_lid, out_lid in turns:
        in_link = link_by_id[in_lid]
        out_link = link_by_id[out_lid]
        from_node = in_link["from"]
        via_node = in_link["to"]
        to_node = out_link["to"]

        for d in centroids:
            for t in decision_times:
                key = f"{from_node}|{via_node}|{to_node}|{d}|{t}"
                solution["y"][key] = clean_value(y[(in_lid, out_lid, d, t)].X)

    for c in centroids:
        for lid in centroid_out_connectors[c]:
            junction = link_by_id[lid]["to"]
            for d in centroids:
                for t in decision_times:
                    key = f"{c}|{junction}|{d}|{t}"
                    solution["y_centroid"][key] = clean_value(
                        y_centroid[(c, lid, d, t)].X
                    )

    for link in links:
        lid = link["id"]
        from_node = link["from"]
        to_node = link["to"]
        for d in centroids:
            for t in state_times:
                key = f"{from_node}|{to_node}|{d}|{t}"
                solution["N_U"][key] = clean_value(N_U[(lid, d, t)].X)
                solution["N_D"][key] = clean_value(N_D[(lid, d, t)].X)

    for c in centroids:
        for t in state_times:
            solution["p"][f"{c}|{t}"] = clean_value(p[(c, t)].X)

    # Emit all centroid pairs. OD pairs absent from the model are zero.
    for o in centroids:
        for d in centroids:
            is_active = (o, d) in active_od

            for t in decision_times:
                value = e[(o, d, t)].X if is_active else 0.0
                solution["e"][f"{o}|{d}|{t}"] = clean_value(value)

            for t in state_times:
                value = omega[(o, d, t)].X if is_active else 0.0
                solution["omega"][f"{o}|{d}|{t}"] = clean_value(value)

    if logger:
        logger.log_solution(solution["objective_value"], solution)

    solution_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if solution_dir:
        os.makedirs(solution_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(solution, f, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()