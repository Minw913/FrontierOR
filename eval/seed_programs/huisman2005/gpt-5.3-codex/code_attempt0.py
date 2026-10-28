import argparse
import json
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    return parser.parse_args()


def remaining_time(start_time, time_limit):
    return max(0.0, time_limit - (time.time() - start_time))


def build_compatibility_edges(trips, travel_times):
    n = len(trips)
    edges_out = [[] for _ in range(n)]
    edges_in = [[] for _ in range(n)]

    starts = [t["start_time"] for t in trips]
    ends = [t["end_time"] for t in trips]
    s_loc = [t["start_location"] for t in trips]
    e_loc = [t["end_location"] for t in trips]

    for i in range(n):
        ei = ends[i]
        eli = e_loc[i]
        for j in range(i + 1, n):
            tt = travel_times[eli][s_loc[j]]
            if ei + tt <= starts[j]:
                edges_out[i].append(j)
                edges_in[j].append(i)
    return edges_out, edges_in


def solve_vehicle_min_path_cover_mip(trips, travel_times, time_budget):
    n = len(trips)
    edges_out, edges_in = build_compatibility_edges(trips, travel_times)

    m = gp.Model("vehicle_path_cover")
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    if time_budget > 0:
        m.Params.TimeLimit = max(1.0, time_budget)

    x = {}
    for i in range(n):
        for j in edges_out[i]:
            x[i, j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    m.update()

    for i in range(n):
        if edges_out[i]:
            m.addConstr(gp.quicksum(x[i, j] for j in edges_out[i]) <= 1)
    for j in range(n):
        if edges_in[j]:
            m.addConstr(gp.quicksum(x[i, j] for i in edges_in[j]) <= 1)

    m.setObjective(gp.quicksum(x.values()), GRB.MAXIMIZE)
    m.optimize()

    succ = {i: None for i in range(n)}
    pred = {i: None for i in range(n)}

    if m.SolCount > 0:
        for (i, j), var in x.items():
            if var.X > 0.5:
                succ[i] = j
                pred[j] = i
        chains = []
        for i in range(n):
            if pred[i] is None:
                chain = []
                cur = i
                while cur is not None:
                    chain.append(cur)
                    cur = succ[cur]
                chains.append(chain)
        # Safety: cover all trips even if numerical oddities
        seen = set(v for c in chains for v in c)
        for i in range(n):
            if i not in seen:
                chains.append([i])
        return chains

    return None


def greedy_vehicle_chains(trips, travel_times):
    # Fallback fast heuristic
    chains = []  # each element: list of trip indices
    chain_last_trip = []  # last trip index per chain

    for j, tj in enumerate(trips):
        best_chain = None
        best_ready_time = None
        for cidx, last_i in enumerate(chain_last_trip):
            ti = trips[last_i]
            ready = ti["end_time"] + travel_times[ti["end_location"]][tj["start_location"]]
            if ready <= tj["start_time"]:
                if best_chain is None or ready > best_ready_time:
                    best_chain = cidx
                    best_ready_time = ready
        if best_chain is None:
            chains.append([j])
            chain_last_trip.append(j)
        else:
            chains[best_chain].append(j)
            chain_last_trip[best_chain] = j
    return chains


def assign_chain_to_depot(chain, trips, depots, travel_times):
    first_trip = trips[chain[0]]
    last_trip = trips[chain[-1]]

    best_depot_name = None
    best_cost = float("inf")
    for d in depots:
        dn = d["name"]
        c = travel_times[dn][first_trip["start_location"]] + travel_times[last_trip["end_location"]][dn]
        if c < best_cost:
            best_cost = c
            best_depot_name = dn
    return best_depot_name


def compute_boundary_extra_time(depot_name, loc, crew_params, travel_times, is_start):
    if loc == depot_name:
        return crew_params["sign_on_time_depot_minutes"] if is_start else crew_params["sign_off_time_depot_minutes"]
    extra = crew_params["extra_time_non_depot_relief_minutes"]
    if not crew_params["extra_time_includes_deadhead_to_depot"]:
        if is_start:
            extra += travel_times[depot_name][loc]
        else:
            extra += travel_times[loc][depot_name]
    return extra


def build_duty_feasibility_checker(trips, travel_times, duty_types, crew_params, relief_locations):
    relief_set = set(relief_locations)

    starts = [t["start_time"] for t in trips]
    ends = [t["end_time"] for t in trips]
    s_loc = [t["start_location"] for t in trips]
    e_loc = [t["end_location"] for t in trips]

    duty_type_items = list(duty_types.items())

    def time_ok(v, lo, hi):
        if lo is not None and v < lo:
            return False
        if hi is not None and v > hi:
            return False
        return True

    def check_one_piece(a, b, depot_name, dt):
        dur = ends[b] - starts[a]
        if dur < dt["piece_length_min"] or dur > dt["piece_length_max"]:
            return False
        st = starts[a]
        en = ends[b]
        if not time_ok(st, dt.get("start_time_min"), dt.get("start_time_max")):
            return False
        if dt.get("end_time_max") is not None and en > dt["end_time_max"]:
            return False
        work_time = dur
        if dt.get("work_time_max") is not None and work_time > dt["work_time_max"]:
            return False

        start_extra = compute_boundary_extra_time(depot_name, s_loc[a], crew_params, travel_times, True)
        end_extra = compute_boundary_extra_time(depot_name, e_loc[b], crew_params, travel_times, False)
        duty_span = (en - st) + start_extra + end_extra
        if dt.get("duty_length_max") is not None and duty_span > dt["duty_length_max"]:
            return False
        return True

    def check_two_piece(a, b, depot_name, dt):
        st = starts[a]
        en = ends[b]
        if not time_ok(st, dt.get("start_time_min"), dt.get("start_time_max")):
            return False
        if dt.get("end_time_max") is not None and en > dt["end_time_max"]:
            return False

        start_extra = compute_boundary_extra_time(depot_name, s_loc[a], crew_params, travel_times, True)
        end_extra = compute_boundary_extra_time(depot_name, e_loc[b], crew_params, travel_times, False)
        duty_span = (en - st) + start_extra + end_extra
        if dt.get("duty_length_max") is not None and duty_span > dt["duty_length_max"]:
            return False

        bmin = dt.get("break_length_min")
        bmin = 0 if bmin is None else bmin

        for k in range(a, b):
            p1 = ends[k] - starts[a]
            p2 = ends[b] - starts[k + 1]
            if p1 < dt["piece_length_min"] or p1 > dt["piece_length_max"]:
                continue
            if p2 < dt["piece_length_min"] or p2 > dt["piece_length_max"]:
                continue

            transfer = travel_times[e_loc[k]][s_loc[k + 1]]
            brk = starts[k + 1] - ends[k] - transfer
            if brk < bmin:
                continue

            work_time = p1 + p2
            if dt.get("work_time_max") is not None and work_time > dt["work_time_max"]:
                continue
            return True
        return False

    def feasible_type_for_subseq(chain, i, j, depot_name):
        # chain positions i..j inclusive
        a = chain[i]
        b = chain[j]

        # Soft relief filtering (avoid hard infeasibility): if boundary is relief/depot, preferred.
        boundary_good = (s_loc[a] in relief_set or s_loc[a] == depot_name) and (e_loc[b] in relief_set or e_loc[b] == depot_name)

        best_type = None

        # Prefer naturally matching types
        for tname, dt in duty_type_items:
            np = dt["num_pieces"]
            if np == 1:
                if check_one_piece(a, b, depot_name, dt):
                    if boundary_good or best_type is None:
                        best_type = tname
                        if boundary_good:
                            return best_type
            elif np == 2:
                if check_two_piece(a, b, depot_name, dt):
                    if boundary_good or best_type is None:
                        best_type = tname
                        if boundary_good:
                            return best_type

        return best_type

    return feasible_type_for_subseq


def partition_chain_into_duties(chain, depot_name, feasible_type_for_subseq, duty_types, time_guard_end):
    m = len(chain)
    dp = [10**9] * (m + 1)
    nxt = [-1] * (m + 1)
    tsel = [None] * (m + 1)
    dp[m] = 0

    for i in range(m - 1, -1, -1):
        if time.time() >= time_guard_end:
            break
        for j in range(i, m):
            typ = feasible_type_for_subseq(chain, i, j, depot_name)
            if typ is None:
                continue
            val = 1 + dp[j + 1]
            if val < dp[i]:
                dp[i] = val
                nxt[i] = j + 1
                tsel[i] = typ

    duties = []
    if dp[0] >= 10**9:
        # fallback: one trip per duty
        one_piece_types = [k for k, v in duty_types.items() if v["num_pieces"] == 1]
        default_type = one_piece_types[0] if one_piece_types else list(duty_types.keys())[0]
        for tid in chain:
            duties.append({"depot": depot_name, "type": default_type, "trips": [tid]})
        return duties

    i = 0
    while i < m:
        j = nxt[i]
        if j == -1:
            # emergency fallback remainder
            one_piece_types = [k for k, v in duty_types.items() if v["num_pieces"] == 1]
            default_type = one_piece_types[0] if one_piece_types else list(duty_types.keys())[0]
            for p in range(i, m):
                duties.append({"depot": depot_name, "type": default_type, "trips": [chain[p]]})
            break
        duties.append({"depot": depot_name, "type": tsel[i], "trips": chain[i:j]})
        i = j
    return duties


def main():
    args = parse_args()
    start = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    best_obj = float("inf")

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    trips = inst["trips"]
    depots = inst["depots"]
    travel_times = inst["travel_times"]
    duty_types = inst["duty_types"]
    crew_params = inst["crew_parameters"]
    relief_locations = inst["relief_locations"]

    # -------- Vehicle scheduling --------
    rt = remaining_time(start, args.time_limit)
    chains = None
    if rt > 2:
        mip_budget = max(1.0, rt * 0.6)
        chains = solve_vehicle_min_path_cover_mip(trips, travel_times, mip_budget)

    if chains is None:
        chains = greedy_vehicle_chains(trips, travel_times)

    # Assign depot per chain
    chain_depot = []
    for ch in chains:
        dname = assign_chain_to_depot(ch, trips, depots, travel_times)
        chain_depot.append(dname)

    # Build vehicle schedule output map
    vehicle_schedule = {d["name"]: [] for d in depots}
    for ch, dn in zip(chains, chain_depot):
        vehicle_schedule[dn].append([int(trips[idx]["trip_id"]) for idx in ch])

    # Trivial crew incumbent: one duty per trip (for logging baseline)
    trivial_duties = []
    one_piece_types = [k for k, v in duty_types.items() if v["num_pieces"] == 1]
    default_type = one_piece_types[0] if one_piece_types else list(duty_types.keys())[0]
    for ch, dn in zip(chains, chain_depot):
        for idx in ch:
            trivial_duties.append({"depot": dn, "type": default_type, "trips": [int(trips[idx]["trip_id"])]})
    trivial_obj = len(chains) + len(trivial_duties)
    if trivial_obj < best_obj:
        best_obj = trivial_obj
        if logger:
            logger.log(int(best_obj))

    # -------- Crew scheduling (improve) --------
    feasible_checker = build_duty_feasibility_checker(
        trips, travel_times, duty_types, crew_params, relief_locations
    )

    crew_schedule = []
    guard_end = start + args.time_limit - 0.2
    for ch, dn in zip(chains, chain_depot):
        if time.time() >= guard_end:
            # fallback remainder quickly
            for idx in ch:
                crew_schedule.append({"depot": dn, "type": default_type, "trips": [int(trips[idx]["trip_id"])]})
            continue
        duties = partition_chain_into_duties(ch, dn, feasible_checker, duty_types, guard_end)
        for d in duties:
            d["trips"] = [int(trips[idx]["trip_id"]) for idx in d["trips"]]
        crew_schedule.extend(duties)

    objective_value = int(len(chains) + len(crew_schedule))
    if objective_value < best_obj and logger:
        logger.log(objective_value)
        best_obj = objective_value

    solution = {
        "objective_value": objective_value,
        "vehicle_schedule": vehicle_schedule,
        "crew_schedule": crew_schedule,
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()