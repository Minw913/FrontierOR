import argparse
import json
import math
import random
import time
from collections import defaultdict
from copy import deepcopy

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    return parser.parse_args()


def get_time_window(node, default_tw):
    if "time_window" in node and isinstance(node["time_window"], list) and len(node["time_window"]) == 2:
        return float(node["time_window"][0]), float(node["time_window"][1])

    # Robust fallbacks for potential alternative field names
    e_keys = ["tw_early", "earliest", "ready_time", "start_time", "a"]
    l_keys = ["tw_late", "latest", "due_time", "end_time", "b"]

    e_val = None
    l_val = None
    for k in e_keys:
        if k in node:
            e_val = float(node[k])
            break
    for k in l_keys:
        if k in node:
            l_val = float(node[k])
            break

    if e_val is not None and l_val is not None:
        return e_val, l_val

    return float(default_tw[0]), float(default_tw[1])


def build_data(instance):
    params = instance["parameters"]
    fleet = instance["fleet"]

    planning_horizon = params["planning_horizon"]
    h_start, h_end = float(planning_horizon[0]), float(planning_horizon[1])

    default_service = float(params.get("service_time", 0.0))
    max_route_duration = float(params["max_route_duration"])

    depot_id = int(instance["depot"]["id"])

    # Customers
    customers = {}
    truck_customers = []
    trailer_customers = []

    for c in instance.get("truck_customers", []):
        cid = int(c["id"])
        customers[cid] = {
            "id": cid,
            "type": "truck",
            "supply": float(c["supply"]),
            "service_time": float(c.get("service_time", default_service)),
            "tw": get_time_window(c, (h_start, h_end)),
        }
        truck_customers.append(cid)

    for c in instance.get("trailer_customers", []):
        cid = int(c["id"])
        customers[cid] = {
            "id": cid,
            "type": "trailer",
            "supply": float(c["supply"]),
            "service_time": float(c.get("service_time", default_service)),
            "tw": get_time_window(c, (h_start, h_end)),
        }
        trailer_customers.append(cid)

    # Fleet type availability
    truck_type_avail = {}
    for t in fleet.get("truck_types", []):
        truck_type_avail[int(t["id"])] = int(t["num_available"])

    trailer_type_avail = {}
    for tr in fleet.get("trailer_types", []):
        trailer_type_avail[int(tr["id"])] = int(tr["num_available"])

    # Vehicle classes
    classes = []
    for vc in fleet.get("vehicle_classes", []):
        c = {
            "class_id": int(vc["class_id"]),
            "truck_type_id": int(vc["truck_type_id"]),
            "trailer_type_id": int(vc["trailer_type_id"]),
            "truck_capacity": float(vc["truck_capacity"]),
            "trailer_capacity": float(vc["trailer_capacity"]),
            "total_capacity": float(vc["total_capacity"]),
            "truck_fixed_cost": float(vc["truck_fixed_cost"]),
            "truck_cost_factor": float(vc["truck_cost_factor"]),
            "trailer_towing_cost_multiplier": float(vc["trailer_towing_cost_multiplier"]),
            "num_available": int(vc["num_available"]),
            "description": vc.get("description", ""),
        }
        c["is_solo"] = (c["trailer_type_id"] == 0)
        classes.append(c)

    distance_matrix = instance["distance_matrix"]
    travel_time_matrix = instance["travel_time_matrix"]

    transshipment_ids = [int(t["id"]) for t in instance.get("transshipment_locations", [])]
    n_orig = len(distance_matrix)

    data = {
        "params": {
            "h_start": h_start,
            "h_end": h_end,
            "max_route_duration": max_route_duration,
            "coupling_cost": float(params.get("coupling_cost", 0.0)),
            "decoupling_cost": float(params.get("decoupling_cost", 0.0)),
            "transfer_cost": float(params.get("transfer_cost", 0.0)),
        },
        "depot_id": depot_id,
        "customers": customers,
        "truck_customers": truck_customers,
        "trailer_customers": trailer_customers,
        "classes": classes,
        "truck_type_avail": truck_type_avail,
        "trailer_type_avail": trailer_type_avail,
        "dist": distance_matrix,
        "tt": travel_time_matrix,
        "transshipment_ids": transshipment_ids,
        "n_orig": n_orig,
    }
    return data


def class_can_serve_customer(cls, cust):
    # Truck customers must be served by solo truck (no trailer attached).
    if cust["type"] == "truck" and not cls["is_solo"]:
        return False
    return True


def evaluate_route_sequence(seq, cls, data):
    """
    Feasibility and timing for a route:
    depot -> seq... -> depot
    Uses conservative earliest-start schedule at planning horizon start.
    """
    customers = data["customers"]
    tt = data["tt"]
    depot = data["depot_id"]
    h_start = data["params"]["h_start"]
    h_end = data["params"]["h_end"]
    max_dur = data["params"]["max_route_duration"]

    total_load = 0.0
    for cid in seq:
        c = customers[cid]
        if not class_can_serve_customer(cls, c):
            return False, None
        total_load += c["supply"]

    if total_load - cls["total_capacity"] > 1e-9:
        return False, None

    arrivals = []
    curr = h_start
    prev = depot

    # Arrival at start depot
    arrivals.append(curr)

    for cid in seq:
        c = customers[cid]
        curr += float(tt[prev][cid])  # arrival
        arr = curr
        tw_e, tw_l = c["tw"]
        if curr < tw_e:
            curr = tw_e
        if curr > tw_l + 1e-9:
            return False, None
        arrivals.append(arr)
        curr += c["service_time"]
        prev = cid

    end_arr = curr + float(tt[prev][depot])
    if end_arr > h_end + 1e-9:
        return False, None
    if end_arr - h_start > max_dur + 1e-9:
        return False, None

    arrivals.append(end_arr)

    return True, {
        "arrivals": arrivals,  # [depot_start, cust1_arr, ..., depot_end_arr]
        "end_time": end_arr,
        "load": total_load,
    }


def route_distance(seq, data):
    dist = data["dist"]
    depot = data["depot_id"]
    if not seq:
        return 0.0
    total = float(dist[depot][seq[0]])
    for i in range(len(seq) - 1):
        total += float(dist[seq[i]][seq[i + 1]])
    total += float(dist[seq[-1]][depot])
    return total


def route_cost(seq, cls, data):
    d = route_distance(seq, data)
    factor = cls["truck_cost_factor"] + (0.0 if cls["is_solo"] else cls["trailer_towing_cost_multiplier"])
    return cls["truck_fixed_cost"] + factor * d


def route_var_cost_only(seq, cls, data):
    d = route_distance(seq, data)
    factor = cls["truck_cost_factor"] + (0.0 if cls["is_solo"] else cls["trailer_towing_cost_multiplier"])
    return factor * d


def can_open_new_vehicle(cls, used_class, used_truck_type, used_trailer_type, data):
    cid = cls["class_id"]
    ttid = cls["truck_type_id"]
    trid = cls["trailer_type_id"]

    if used_class[cid] >= cls["num_available"]:
        return False

    if ttid in data["truck_type_avail"]:
        if used_truck_type[ttid] >= data["truck_type_avail"][ttid]:
            return False

    if trid != 0 and trid in data["trailer_type_avail"]:
        if used_trailer_type[trid] >= data["trailer_type_avail"][trid]:
            return False

    return True


def open_vehicle(cls, used_class, used_truck_type, used_trailer_type):
    used_class[cls["class_id"]] += 1
    used_truck_type[cls["truck_type_id"]] += 1
    if cls["trailer_type_id"] != 0:
        used_trailer_type[cls["trailer_type_id"]] += 1


def best_insertion_for_customer(route, cust_id, data):
    cls = route["class"]
    old_seq = route["seq"]

    best = None
    old_var_cost = route_var_cost_only(old_seq, cls, data)

    for pos in range(len(old_seq) + 1):
        new_seq = old_seq[:pos] + [cust_id] + old_seq[pos:]
        feas, info = evaluate_route_sequence(new_seq, cls, data)
        if not feas:
            continue
        new_var = route_var_cost_only(new_seq, cls, data)
        delta = new_var - old_var_cost
        if (best is None) or (delta < best["delta"] - 1e-12):
            best = {"delta": delta, "pos": pos, "new_seq": new_seq, "eval": info}
    return best


def create_new_route_option(cust_id, candidate_classes, used_class, used_truck_type, used_trailer_type, data):
    best = None
    cust = data["customers"][cust_id]

    for cls in candidate_classes:
        if not class_can_serve_customer(cls, cust):
            continue
        if not can_open_new_vehicle(cls, used_class, used_truck_type, used_trailer_type, data):
            continue

        seq = [cust_id]
        feas, info = evaluate_route_sequence(seq, cls, data)
        if not feas:
            continue
        cost = route_cost(seq, cls, data)
        if (best is None) or (cost < best["cost"] - 1e-12):
            best = {"class": cls, "seq": seq, "eval": info, "cost": cost}
    return best


def build_customer_order(customer_ids, data, rng):
    # Priority: truck customers first by caller; here sort by demand desc + tighter latest time.
    arr = []
    for cid in customer_ids:
        c = data["customers"][cid]
        e, l = c["tw"]
        arr.append((cid, c["supply"], l - e, l))

    # Base deterministic sort
    arr.sort(key=lambda x: (-x[1], x[2], x[3]))

    # Small randomization
    blocks = []
    i = 0
    while i < len(arr):
        j = min(i + 5, len(arr))
        block = arr[i:j]
        rng.shuffle(block)
        blocks.extend(block)
        i = j

    return [x[0] for x in blocks]


def construct_solution(data, rng, time_deadline):
    classes = data["classes"]
    customers = data["customers"]

    # Split classes
    solo_classes = [c for c in classes if c["is_solo"]]
    all_classes = classes[:]

    # Prefer cheaper classes first for opening new routes
    def class_rank(c):
        cap = max(1.0, c["total_capacity"])
        per_dist = c["truck_cost_factor"] + (0.0 if c["is_solo"] else c["trailer_towing_cost_multiplier"])
        return (c["truck_fixed_cost"] / cap, per_dist, -cap)

    solo_classes.sort(key=class_rank)
    all_classes.sort(key=class_rank)

    routes = []
    used_class = defaultdict(int)
    used_truck_type = defaultdict(int)
    used_trailer_type = defaultdict(int)

    # Phase 1: truck customers (restricted) first
    phase_customers = [
        build_customer_order(data["truck_customers"], data, rng),
        build_customer_order(data["trailer_customers"], data, rng),
    ]
    phase_classes = [solo_classes, all_classes]

    for phase_idx in range(2):
        cust_order = phase_customers[phase_idx]
        candidate_for_new = phase_classes[phase_idx]

        for cid in cust_order:
            if time.time() >= time_deadline:
                return None  # interrupted construction

            c = customers[cid]
            best_existing = None

            # Try inserting into existing routes first (vehicle-saving behavior)
            for r_idx, r in enumerate(routes):
                if not class_can_serve_customer(r["class"], c):
                    continue
                ins = best_insertion_for_customer(r, cid, data)
                if ins is None:
                    continue
                if (best_existing is None) or (ins["delta"] < best_existing["delta"] - 1e-12):
                    best_existing = {"route_idx": r_idx, **ins}

            # Try creating new route
            best_new = create_new_route_option(
                cid, candidate_for_new, used_class, used_truck_type, used_trailer_type, data
            )

            # Decision rule: prefer existing if feasible, unless new route is significantly cheaper
            choose_new = False
            if best_existing is None and best_new is not None:
                choose_new = True
            elif best_existing is not None and best_new is None:
                choose_new = False
            elif best_existing is None and best_new is None:
                return None  # infeasible in this construction
            else:
                # Both exist. Keep routes compact: choose insertion unless very costly.
                if best_new["cost"] + 1e-9 < best_existing["delta"]:
                    choose_new = True
                else:
                    choose_new = False

            if choose_new:
                cls = best_new["class"]
                open_vehicle(cls, used_class, used_truck_type, used_trailer_type)
                routes.append(
                    {
                        "class": cls,
                        "seq": [cid],
                        "eval": best_new["eval"],
                    }
                )
            else:
                r = routes[best_existing["route_idx"]]
                r["seq"] = best_existing["new_seq"]
                r["eval"] = best_existing["eval"]

    # Final full feasibility validation
    served = set()
    for r in routes:
        feas, info = evaluate_route_sequence(r["seq"], r["class"], data)
        if not feas:
            return None
        r["eval"] = info
        for cid in r["seq"]:
            if cid in served:
                return None
            served.add(cid)

    all_customers_set = set(data["truck_customers"]) | set(data["trailer_customers"])
    if served != all_customers_set:
        return None

    # Check fleet avail consistency
    check_class = defaultdict(int)
    check_truck = defaultdict(int)
    check_trailer = defaultdict(int)
    for r in routes:
        cls = r["class"]
        check_class[cls["class_id"]] += 1
        check_truck[cls["truck_type_id"]] += 1
        if cls["trailer_type_id"] != 0:
            check_trailer[cls["trailer_type_id"]] += 1

    for cls in classes:
        if check_class[cls["class_id"]] > cls["num_available"]:
            return None
    for tid, n in check_truck.items():
        if tid in data["truck_type_avail"] and n > data["truck_type_avail"][tid]:
            return None
    for trid, n in check_trailer.items():
        if trid in data["trailer_type_avail"] and n > data["trailer_type_avail"][trid]:
            return None

    return routes


def routes_to_solution(routes, data):
    total_obj = 0.0
    output_routes = []

    # Vehicle index per class
    class_counter = defaultdict(int)

    n_orig = data["n_orig"]
    p_len = len(data["transshipment_ids"])
    next_extra_id = n_orig + 2 * p_len  # starts after prescribed expanded parking IDs

    for r in routes:
        cls = r["class"]
        seq = r["seq"]

        total_obj += route_cost(seq, cls, data)

        vehicle_class = cls["class_id"]
        vehicle_index = class_counter[vehicle_class]
        class_counter[vehicle_class] += 1

        physical = [data["depot_id"]] + seq + [data["depot_id"]]

        # Ensure eval available
        feas, info = evaluate_route_sequence(seq, cls, data)
        if not feas:
            # Shouldn't happen for accepted routes; fallback empty arrivals
            arrivals = [0.0 for _ in physical]
        else:
            arrivals = info["arrivals"]

        # Build route_sequence with unique stop identifiers if repeated physical node appears
        seen = defaultdict(int)
        route_sequence = []
        arrival_times = {}

        for i, pid in enumerate(physical):
            if seen[pid] == 0:
                sid = int(pid)
            else:
                sid = int(next_extra_id)
                next_extra_id += 1
            seen[pid] += 1
            route_sequence.append(sid)
            arrival_times[str(sid)] = float(arrivals[i])

        out_r = {
            "vehicle_class": int(vehicle_class),
            "vehicle_index": int(vehicle_index),
            "customers_served": [int(c) for c in seq],
            "route_sequence": route_sequence,
            "route_sequence_physical": [int(x) for x in physical],
            "decouple_locations": [],
            "couple_locations": [],
            "arrival_times": arrival_times,
        }
        output_routes.append(out_r)

    solution = {
        "objective_value": float(total_obj),
        "routes": output_routes,
        "expanded_node_convention": (
            "Let n_orig be the number of physical locations and P be the instance parking-location IDs in their published list order. "
            "Physical nodes retain IDs 0..n_orig-1. The decouple copy of P[i] has ID n_orig+i, and the couple copy of P[i] has "
            "ID n_orig+len(P)+i. Both copies map back to physical location P[i]."
        ),
    }
    return solution


def main():
    args = parse_args()
    start_time = time.time()
    deadline = start_time + args.time_limit

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    data = build_data(instance)

    # Multi-start randomized construction
    best_routes = None
    best_obj = math.inf
    iter_idx = 0

    # Guarantee at least one attempt
    while True:
        now = time.time()
        if now >= deadline - 0.02:
            break

        rng = random.Random(1000 + iter_idx)
        candidate = construct_solution(data, rng, deadline)

        if candidate is not None:
            sol = routes_to_solution(candidate, data)
            obj = sol["objective_value"]
            if obj < best_obj - 1e-9:
                best_obj = obj
                best_routes = candidate
                if logger:
                    logger.log_solution(best_obj, sol)

        iter_idx += 1

        # If already found a solution and little time left, stop
        if best_routes is not None and time.time() >= deadline - 0.2:
            break

    if best_routes is None:
        # Fallback (may be infeasible if no feasible construction found)
        final_solution = {
            "objective_value": float(1e18),
            "routes": [],
            "expanded_node_convention": (
                "Let n_orig be the number of physical locations and P be the instance parking-location IDs in their published list order. "
                "Physical nodes retain IDs 0..n_orig-1. The decouple copy of P[i] has ID n_orig+i, and the couple copy of P[i] has "
                "ID n_orig+len(P)+i. Both copies map back to physical location P[i]."
            ),
        }
    else:
        final_solution = routes_to_solution(best_routes, data)

    with open(args.solution_path, "w") as f:
        json.dump(final_solution, f, indent=2)


if __name__ == "__main__":
    main()