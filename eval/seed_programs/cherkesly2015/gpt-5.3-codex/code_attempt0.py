import argparse
import json
import time
from copy import deepcopy

from solution_logger import SolutionLogger


def parse_instance(path):
    with open(path, "r") as f:
        data = json.load(f)

    nodes = {n["node_id"]: n for n in data["nodes"]}

    origin = None
    destination = None
    req_to_pickup = {}
    req_to_delivery = {}

    for n in data["nodes"]:
        t = n["node_type"]
        if t == "depot_origin":
            origin = n["node_id"]
        elif t == "depot_destination":
            destination = n["node_id"]
        elif t == "pickup":
            req_to_pickup[n["request_id"]] = n["node_id"]
        elif t == "delivery":
            req_to_delivery[n["request_id"]] = n["node_id"]

    arc_cost = {}
    arc_time = {}
    for a in data["arcs"]:
        i, j = a["from"], a["to"]
        arc_cost[(i, j)] = float(a["cost"])
        arc_time[(i, j)] = float(a["travel_time"])

    return {
        "raw": data,
        "nodes": nodes,
        "origin": origin,
        "destination": destination,
        "req_to_pickup": req_to_pickup,
        "req_to_delivery": req_to_delivery,
        "arc_cost": arc_cost,
        "arc_time": arc_time,
        "Q": data["vehicle_capacity"],
    }


def build_sequence(origin, destination, pickups, pickup_to_delivery):
    # LIFO-safe pattern: pickups then deliveries in reverse order
    deliveries = [pickup_to_delivery[p] for p in reversed(pickups)]
    return [origin] + pickups + deliveries + [destination]


def evaluate_sequence(seq, inst, pickup_to_delivery):
    nodes = inst["nodes"]
    arc_cost = inst["arc_cost"]
    arc_time = inst["arc_time"]
    Q = inst["Q"]

    times = {}
    loads = {}

    start0 = max(float(nodes[seq[0]]["tw_early"]), 0.0)
    if start0 > float(nodes[seq[0]]["tw_late"]) + 1e-9:
        return None

    times[seq[0]] = start0
    load0 = float(nodes[seq[0]]["load"])
    if load0 < -1e-9 or load0 > Q + 1e-9:
        return None
    loads[seq[0]] = load0

    total_cost = 0.0

    for idx in range(len(seq) - 1):
        i, j = seq[idx], seq[idx + 1]
        if (i, j) not in arc_cost:
            return None

        total_cost += arc_cost[(i, j)]

        depart_i = times[i] + float(nodes[i]["service_time"])
        arr_j = depart_i + arc_time[(i, j)]
        start_j = max(float(nodes[j]["tw_early"]), arr_j)

        if start_j > float(nodes[j]["tw_late"]) + 1e-9:
            return None

        times[j] = start_j

        load_j = loads[i] + float(nodes[j]["load"])
        if load_j < -1e-9 or load_j > Q + 1e-9:
            return None
        loads[j] = load_j

    # Check pickup-delivery precedence inequality from statement
    pos = {n: k for k, n in enumerate(seq)}
    for p, d in pickup_to_delivery.items():
        if p in pos and d in pos:
            if pos[p] > pos[d]:
                return None
            direct_t = arc_time.get((p, d), 0.0)
            lhs = times[p] + float(nodes[p]["service_time"]) + direct_t
            if lhs > times[d] + 1e-9:
                return None

    return {
        "sequence": seq,
        "cost": total_cost,
        "times": times,
        "loads": loads,
    }


def make_route(pickups, inst, pickup_to_delivery):
    seq = build_sequence(inst["origin"], inst["destination"], pickups, pickup_to_delivery)
    ev = evaluate_sequence(seq, inst, pickup_to_delivery)
    if ev is None:
        return None
    return {
        "pickups": pickups[:],
        "sequence": ev["sequence"],
        "cost": ev["cost"],
        "times": ev["times"],
        "loads": ev["loads"],
    }


def total_cost(routes):
    return float(sum(r["cost"] for r in routes))


def solve_heuristic(inst, time_limit, logger=None):
    start = time.time()
    deadline = start + max(1, time_limit)

    req_to_pick = inst["req_to_pickup"]
    req_to_del = inst["req_to_delivery"]
    pickup_to_delivery = {}
    for r, p in req_to_pick.items():
        if r in req_to_del:
            pickup_to_delivery[p] = req_to_del[r]

    pickups = list(pickup_to_delivery.keys())
    nodes = inst["nodes"]

    # Sort requests for constructive phase (time-window urgency first)
    pickups.sort(key=lambda p: (nodes[p]["tw_late"], -nodes[p]["load"]))

    routes = []
    unscheduled = []

    # Greedy insertion in LIFO pattern
    for p in pickups:
        if time.time() > deadline:
            unscheduled.append(p)
            continue

        best_choice = None
        best_delta = float("inf")

        # Try inserting into existing routes
        for ridx, r in enumerate(routes):
            cur_pickups = r["pickups"]
            for pos in range(len(cur_pickups) + 1):
                cand = cur_pickups[:pos] + [p] + cur_pickups[pos:]
                nr = make_route(cand, inst, pickup_to_delivery)
                if nr is None:
                    continue
                delta = nr["cost"] - r["cost"]
                if delta < best_delta:
                    best_delta = delta
                    best_choice = ("insert", ridx, nr)

        # Try creating new route
        nr_single = make_route([p], inst, pickup_to_delivery)
        if nr_single is not None and nr_single["cost"] < best_delta:
            best_choice = ("new", None, nr_single)

        if best_choice is None:
            unscheduled.append(p)
        else:
            if best_choice[0] == "insert":
                routes[best_choice[1]] = best_choice[2]
            else:
                routes.append(best_choice[2])

    # If anything unscheduled, try forcing singleton at end
    for p in unscheduled:
        if time.time() > deadline:
            break
        nr_single = make_route([p], inst, pickup_to_delivery)
        if nr_single is not None:
            routes.append(nr_single)

    if not routes:
        return None

    best_obj = total_cost(routes)
    if logger:
        logger.log(best_obj)

    # Local search: merge routes (big impact on fixed-cost objective)
    improved = True
    while improved and time.time() < deadline:
        improved = False

        best_move = None
        best_delta = -1e-9

        m = len(routes)
        for i in range(m):
            if time.time() > deadline:
                break
            for j in range(i + 1, m):
                if time.time() > deadline:
                    break
                A = routes[i]["pickups"]
                B = routes[j]["pickups"]

                for cand_pickups in (A + B, B + A):
                    nr = make_route(cand_pickups, inst, pickup_to_delivery)
                    if nr is None:
                        continue
                    delta = nr["cost"] - (routes[i]["cost"] + routes[j]["cost"])
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (i, j, nr)

        if best_move is not None:
            i, j, nr = best_move
            # remove higher index first
            for idx in sorted([i, j], reverse=True):
                routes.pop(idx)
            routes.append(nr)
            new_obj = total_cost(routes)
            if new_obj + 1e-9 < best_obj:
                best_obj = new_obj
                if logger:
                    logger.log(best_obj)
            improved = True

    return routes


def build_output(routes):
    out_routes = []
    for vid, r in enumerate(routes, start=1):
        details = []
        for nid in r["sequence"]:
            details.append(
                {
                    "node_id": int(nid),
                    "arrival_time": float(r["times"][nid]),
                    "load_after": float(r["loads"][nid]),
                }
            )
        out_routes.append(
            {
                "vehicle_id": int(vid),
                "route_sequence": [int(x) for x in r["sequence"]],
                "route_details": details,
            }
        )

    return {
        "objective_value": float(sum(r["cost"] for r in routes)),
        "routes": out_routes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    inst = parse_instance(args.instance_path)
    routes = solve_heuristic(inst, args.time_limit, logger=logger)

    if routes is None:
        sol = {"objective_value": 1e18, "routes": []}
    else:
        sol = build_output(routes)

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()