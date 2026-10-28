import argparse
import json
import time
import math
import heapq
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def dijkstra_all_pairs(stations, edges):
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((v, w))
    dist_all = {}
    for s in stations:
        dist = {x: math.inf for x in stations}
        dist[s] = 0
        pq = [(0, s)]
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist[u]:
                continue
            for v, w in adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        dist_all[s] = dist
    return dist_all


def build_task_traversed_stations(task, line_map):
    line_id = task["line_id"]
    dep = task["departure_station"]
    arr = task["arrival_station"]
    direction = task.get("direction", None)

    if line_id not in line_map:
        return [dep, arr]
    seq = line_map[line_id]["stations"]
    if dep not in seq or arr not in seq:
        return [dep, arr]

    i = seq.index(dep)
    j = seq.index(arr)
    if direction == "forward":
        if i <= j:
            return seq[i : j + 1]
        return [dep, arr]
    elif direction == "backward":
        if i >= j:
            return list(reversed(seq[j : i + 1]))
        return [dep, arr]
    else:
        if i <= j:
            return seq[i : j + 1]
        else:
            return list(reversed(seq[j : i + 1]))


def get_initial_station(duty, task_map):
    if "current_station" in duty and duty["current_station"]:
        return duty["current_station"]
    if duty.get("duty_type") == "reserve":
        return duty.get("standby_station", duty["crew_base"])
    pre = duty.get("pre_disruption_tasks", [])
    if pre:
        last = pre[-1]
        if last in task_map:
            return task_map[last]["arrival_station"]
    # fallback
    td = duty.get("task_details", [])
    if td:
        # try last detail arriving before disruption if available
        return td[-1]["arrival_station"]
    return duty["crew_base"]


def compute_horizon(duty, params):
    end_base = duty["end_time_minutes"]
    max_ot = params.get("max_overtime_minutes", 0)
    h1 = end_base + max_ot
    max_dur_h = params.get("max_duty_duration_hours", None)
    if max_dur_h is None:
        return h1
    h2 = duty["start_time_minutes"] + int(round(60 * max_dur_h))
    return min(h1, h2)


def simulate_sequence(
    duty,
    seq,
    duty_ctx,
    task_map,
    rail_dist,
    taxi_time,
    params,
    station_is_relief,
):
    start_station = duty_ctx["start_station"]
    start_time = duty_ctx["start_time"]
    horizon = duty_ctx["horizon"]
    crew_base = duty["crew_base"]

    min_conn = params.get("min_connection_time_minutes", 0)
    c_duty_changed = params["cost_duty_changed"]
    c_task_re = params["cost_task_reassigned"]
    c_new_transfer = params["cost_new_transfer"]
    c_taxi = params["cost_taxi"]

    orig_tasks = duty_ctx["orig_tasks"]
    orig_set = duty_ctx["orig_set"]
    orig_pairs = duty_ctx["orig_pairs"]

    time_now = start_time
    st_now = start_station
    prev_stock = None
    taxi_count = 0

    # license checks and timing
    for idx, tid in enumerate(seq):
        if tid not in task_map:
            return None
        t = task_map[tid]
        dep_st = t["departure_station"]
        arr_st = t["arrival_station"]
        dep_t = t["departure_time_minutes"]
        arr_t = t["arrival_time_minutes"]
        stock = t["rolling_stock_type"]

        # op eligibility prechecked usually, but keep safe
        if tid not in duty_ctx["operable_tasks_set"]:
            return None

        gap = min_conn if (prev_stock is not None and prev_stock != stock) else 0

        # choose reposition mode: prefer non-taxi if feasible
        feasible_non_taxi = False
        travel_non_taxi = math.inf
        if st_now == dep_st:
            travel_non_taxi = 0
            feasible_non_taxi = (time_now + gap <= dep_t)
        else:
            d = rail_dist.get(st_now, {}).get(dep_st, math.inf)
            if math.isfinite(d):
                travel_non_taxi = d
                feasible_non_taxi = (time_now + d + gap <= dep_t)

        chosen_taxi = False
        if feasible_non_taxi:
            pass
        else:
            tx = taxi_time.get((st_now, dep_st), math.inf)
            if math.isfinite(tx) and (time_now + tx + gap <= dep_t):
                chosen_taxi = True
                taxi_count += 1
            else:
                return None

        if chosen_taxi:
            pass  # arrival for operation is at dep_t with waiting as needed
        else:
            pass

        if arr_t > horizon:
            return None

        st_now = arr_st
        time_now = arr_t
        prev_stock = stock

    # return to base if possible; otherwise release fallback if allowed
    release_at_relief = False
    uses_taxi_return = 0

    if st_now == crew_base:
        return_time = time_now
        return_station = crew_base
    else:
        rail_back = rail_dist.get(st_now, {}).get(crew_base, math.inf)
        can_rail = math.isfinite(rail_back) and (time_now + rail_back <= horizon)

        if can_rail:
            return_time = int(time_now + rail_back)
            return_station = crew_base
        else:
            tx = taxi_time.get((st_now, crew_base), math.inf)
            can_taxi = math.isfinite(tx) and (time_now + tx <= horizon)
            if can_taxi:
                uses_taxi_return = 1
                return_time = int(time_now + tx)
                return_station = crew_base
            else:
                # emergency release only at relief point
                if station_is_relief.get(st_now, False):
                    release_at_relief = True
                    return_station = st_now
                    return_time = time_now
                else:
                    return None

    taxi_total = taxi_count + uses_taxi_return
    unchanged = (seq == orig_tasks)

    if unchanged:
        cost = 0.0
    else:
        new_tasks = sum(1 for tid in seq if tid not in orig_set)
        new_transfers = 0
        for i in range(len(seq) - 1):
            if (seq[i], seq[i + 1]) not in orig_pairs:
                new_transfers += 1
        cost = float(
            c_duty_changed
            + c_task_re * new_tasks
            + c_new_transfer * new_transfers
            + c_taxi * taxi_total
        )

    return {
        "covered_tasks": list(seq),
        "uses_taxi": bool(taxi_total > 0),
        "release_at_relief_point": bool(release_at_relief),
        "return_station": return_station,
        "return_time_minutes": int(return_time),
        "cost": float(cost),
        "_taxi_count": taxi_total,
    }


def generate_candidates_for_duty(
    duty,
    duty_ctx,
    operable_tasks,
    task_map,
    rail_dist,
    taxi_time,
    params,
    station_is_relief,
    cancel_penalty,
    time_deadline,
):
    # Candidate uniqueness by sequence tuple
    seen = set()
    cands = []

    def try_add(seq):
        key = tuple(seq)
        if key in seen:
            return
        cand = simulate_sequence(
            duty, seq, duty_ctx, task_map, rail_dist, taxi_time, params, station_is_relief
        )
        if cand is None:
            return
        seen.add(key)
        cands.append(cand)

    # always try empty
    try_add([])

    # unchanged original
    try_add(duty_ctx["orig_tasks"])

    # single-task candidates
    for tid in operable_tasks[:40]:
        if time.time() > time_deadline:
            break
        try_add([tid])

    # beam search for multi-task sequences
    max_depth = 6
    beam_width = 30
    branch_per_state = 5

    start_state = {
        "seq": [],
        "time": duty_ctx["start_time"],
        "station": duty_ctx["start_station"],
        "prev_stock": None,
        "taxi_count": 0,
        "score": 0.0,
    }
    beam = [start_state]

    min_conn = params.get("min_connection_time_minutes", 0)

    for _ in range(max_depth):
        if time.time() > time_deadline:
            break
        next_beam = []
        for st in beam:
            if time.time() > time_deadline:
                break
            used = set(st["seq"])
            feasible_next = []
            for tid in operable_tasks:
                if tid in used:
                    continue
                t = task_map[tid]
                dep_st = t["departure_station"]
                dep_t = t["departure_time_minutes"]
                arr_t = t["arrival_time_minutes"]
                stock = t["rolling_stock_type"]

                if arr_t > duty_ctx["horizon"]:
                    continue

                gap = min_conn if (st["prev_stock"] is not None and st["prev_stock"] != stock) else 0

                ok = False
                add_taxi = 0
                # non-taxi
                if st["station"] == dep_st:
                    if st["time"] + gap <= dep_t:
                        ok = True
                        add_taxi = 0
                else:
                    d = rail_dist.get(st["station"], {}).get(dep_st, math.inf)
                    if math.isfinite(d) and st["time"] + d + gap <= dep_t:
                        ok = True
                        add_taxi = 0
                    else:
                        tx = taxi_time.get((st["station"], dep_st), math.inf)
                        if math.isfinite(tx) and st["time"] + tx + gap <= dep_t:
                            ok = True
                            add_taxi = 1

                if ok:
                    feasible_next.append((dep_t, add_taxi, tid))

            feasible_next.sort(key=lambda x: (x[0], x[1]))
            for _, add_taxi, tid in feasible_next[:branch_per_state]:
                t = task_map[tid]
                new_seq = st["seq"] + [tid]
                gain = cancel_penalty[tid]
                # rough marginal costs
                if len(st["seq"]) == 0:
                    gain -= params["cost_duty_changed"]
                if tid not in duty_ctx["orig_set"]:
                    gain -= params["cost_task_reassigned"]
                if add_taxi:
                    gain -= params["cost_taxi"]
                if len(new_seq) >= 2:
                    p = (new_seq[-2], new_seq[-1])
                    if p not in duty_ctx["orig_pairs"]:
                        gain -= params["cost_new_transfer"]

                ns = {
                    "seq": new_seq,
                    "time": t["arrival_time_minutes"],
                    "station": t["arrival_station"],
                    "prev_stock": t["rolling_stock_type"],
                    "taxi_count": st["taxi_count"] + add_taxi,
                    "score": st["score"] + gain,
                }
                next_beam.append(ns)
                try_add(new_seq)

        if not next_beam:
            break
        next_beam.sort(key=lambda s: s["score"], reverse=True)
        beam = next_beam[:beam_width]

    # If too many candidates, keep diverse by value proxy
    if len(cands) > 80:
        def cand_value(c):
            cov = sum(cancel_penalty[tid] for tid in c["covered_tasks"])
            return cov - c["cost"]
        cands.sort(key=cand_value, reverse=True)
        # always keep empty if present
        kept = cands[:80]
        if not any(len(k["covered_tasks"]) == 0 for k in kept):
            for c in cands:
                if len(c["covered_tasks"]) == 0:
                    kept[-1] = c
                    break
        cands = kept

    # final stable ordering: shorter index for lower cost then more coverage
    cands.sort(key=lambda c: (c["cost"], -len(c["covered_tasks"])))
    return cands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    t0 = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    variant = inst.get("released_problem_variant", "relief_point_completion_v1")
    net = inst["network"]
    params = inst["parameters"]
    tasks = inst["tasks"]
    disruption = inst["disruption"]

    station_ids = [s["id"] for s in net["stations"]]
    station_is_relief = {s["id"]: bool(s.get("is_relief_point", False)) for s in net["stations"]}

    # Build travel graphs
    rail_edges = []
    for seg in net.get("route_segments", []):
        u, v, w = seg["from_station"], seg["to_station"], seg["travel_time_minutes"]
        rail_edges.append((u, v, w))
        if seg.get("bidirectional", False):
            rail_edges.append((v, u, w))
    rail_dist = dijkstra_all_pairs(station_ids, rail_edges)

    taxi_time = {}
    for tx in net.get("taxi_connections", []):
        u, v, w = tx["from_station"], tx["to_station"], tx["travel_time_minutes"]
        taxi_time[(u, v)] = min(taxi_time.get((u, v), math.inf), w)
        if tx.get("bidirectional", False):
            taxi_time[(v, u)] = min(taxi_time.get((v, u), math.inf), w)

    line_map = {ln["line_id"]: ln for ln in inst.get("train_lines", [])}
    task_map = {t["task_id"]: t for t in tasks}
    task_ids = [t["task_id"] for t in tasks]

    # penalties per task
    cancel_penalty = {}
    for t in tasks:
        cancel_penalty[t["task_id"]] = (
            params["cost_cancel_A_to_B"] if t["task_type"] == "A_to_B" else params["cost_cancel_A_to_A"]
        )

    # Precompute traversed stations for route knowledge checks
    task_traversed = {}
    for t in tasks:
        task_traversed[t["task_id"]] = build_task_traversed_stations(t, line_map)

    duties = []
    duties.extend(inst.get("active_duties", []))
    duties.extend(inst.get("reserve_duties", []))

    # Prepare duty contexts
    duty_ctx = {}
    for d in duties:
        did = d["duty_id"]
        start_station = get_initial_station(d, task_map)
        rs_time = disruption["rescheduling_time_minutes"]
        if d.get("duty_type") == "reserve":
            start_time = max(rs_time, d.get("start_time_minutes", rs_time))
        else:
            start_time = rs_time
        horizon = compute_horizon(d, params)

        orig_tasks = list(d.get("tasks", []))
        orig_set = set(orig_tasks)
        orig_pairs = set((orig_tasks[i], orig_tasks[i + 1]) for i in range(len(orig_tasks) - 1))

        # operable tasks by license + route
        route_k = set(d.get("route_knowledge", []))
        stock_k = set(d.get("rolling_stock_licenses", []))
        operable = []
        for t in tasks:
            tid = t["task_id"]
            if t["rolling_stock_type"] not in stock_k:
                continue
            traversed = task_traversed[tid]
            if all(st in route_k for st in traversed):
                if t["departure_time_minutes"] >= start_time:
                    operable.append(tid)

        operable.sort(key=lambda tid: task_map[tid]["departure_time_minutes"])

        duty_ctx[did] = {
            "start_station": start_station,
            "start_time": start_time,
            "horizon": horizon,
            "orig_tasks": orig_tasks,
            "orig_set": orig_set,
            "orig_pairs": orig_pairs,
            "operable_tasks": operable,
            "operable_tasks_set": set(operable),
        }

    # Candidate generation with time awareness
    all_candidates = {}
    gen_deadline = t0 + max(1, int(args.time_limit * 0.55))
    for d in duties:
        if time.time() > gen_deadline:
            break
        did = d["duty_id"]
        cands = generate_candidates_for_duty(
            d,
            duty_ctx[did],
            duty_ctx[did]["operable_tasks"],
            task_map,
            rail_dist,
            taxi_time,
            params,
            station_is_relief,
            cancel_penalty,
            gen_deadline,
        )
        if not cands:
            # emergency fallback (try empty again)
            c0 = simulate_sequence(
                d, [], duty_ctx[did], task_map, rail_dist, taxi_time, params, station_is_relief
            )
            if c0 is None:
                # last-resort dummy, likely infeasible in checker but prevents crash
                c0 = {
                    "covered_tasks": [],
                    "uses_taxi": False,
                    "release_at_relief_point": True,
                    "return_station": duty_ctx[did]["start_station"],
                    "return_time_minutes": duty_ctx[did]["start_time"],
                    "cost": float(params["cost_duty_changed"]),
                    "_taxi_count": 0,
                }
            cands = [c0]
        all_candidates[did] = cands

    # For any duty not processed due to time, give minimal candidate
    for d in duties:
        did = d["duty_id"]
        if did not in all_candidates:
            c0 = simulate_sequence(
                d, [], duty_ctx[did], task_map, rail_dist, taxi_time, params, station_is_relief
            )
            if c0 is None:
                c0 = {
                    "covered_tasks": [],
                    "uses_taxi": False,
                    "release_at_relief_point": True,
                    "return_station": duty_ctx[did]["start_station"],
                    "return_time_minutes": duty_ctx[did]["start_time"],
                    "cost": float(params["cost_duty_changed"]),
                    "_taxi_count": 0,
                }
            all_candidates[did] = [c0]

    # Build MIP
    model = gp.Model("crew_reschedule")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    remain = max(1, args.time_limit - int(time.time() - t0))
    model.Params.TimeLimit = remain
    model.Params.OutputFlag = 0

    x = {}  # (did, ci)
    for d in duties:
        did = d["duty_id"]
        for ci, c in enumerate(all_candidates[did]):
            x[(did, ci)] = model.addVar(vtype=GRB.BINARY, name=f"x_{did}_{ci}")

    z = {tid: model.addVar(vtype=GRB.BINARY, name=f"z_{tid}") for tid in task_ids}

    # one completion per duty
    for d in duties:
        did = d["duty_id"]
        model.addConstr(gp.quicksum(x[(did, ci)] for ci in range(len(all_candidates[did]))) == 1,
                        name=f"one_comp_{did}")

    # coverage or cancel
    cover_vars = defaultdict(list)
    for d in duties:
        did = d["duty_id"]
        for ci, c in enumerate(all_candidates[did]):
            for tid in c["covered_tasks"]:
                cover_vars[tid].append(x[(did, ci)])

    for tid in task_ids:
        if cover_vars[tid]:
            model.addConstr(gp.quicksum(cover_vars[tid]) + z[tid] >= 1, name=f"cover_{tid}")
        else:
            model.addConstr(z[tid] == 1, name=f"force_cancel_{tid}")

    obj = gp.quicksum(
        all_candidates[did][ci]["cost"] * x[(did, ci)]
        for d in duties
        for did in [d["duty_id"]]
        for ci in range(len(all_candidates[did]))
    ) + gp.quicksum(cancel_penalty[tid] * z[tid] for tid in task_ids)
    model.setObjective(obj, GRB.MINIMIZE)

    best_cb = [math.inf]

    def cb(m, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            val = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if val + 1e-9 < best_cb[0]:
                best_cb[0] = val
                logger.log(float(val))

    model.optimize(cb)

    # Extract solution
    have_sol = model.SolCount > 0
    selected = {}

    if have_sol:
        for d in duties:
            did = d["duty_id"]
            chosen = 0
            bestv = -1.0
            for ci in range(len(all_candidates[did])):
                v = x[(did, ci)].X
                if v > bestv:
                    bestv = v
                    chosen = ci
            selected[did] = chosen
    else:
        # fallback: choose cheapest candidate per duty
        for d in duties:
            did = d["duty_id"]
            cands = all_candidates[did]
            best_ci = min(range(len(cands)), key=lambda i: cands[i]["cost"])
            selected[did] = best_ci

    # Determine canceled tasks
    covered = set()
    for d in duties:
        did = d["duty_id"]
        c = all_candidates[did][selected[did]]
        covered.update(c["covered_tasks"])

    canceled = []
    if have_sol:
        for tid in task_ids:
            if z[tid].X > 0.5:
                canceled.append(tid)
        # Ensure consistency with coverage constraints (robust fallback)
        for tid in task_ids:
            if tid not in covered and tid not in canceled:
                canceled.append(tid)
    else:
        canceled = [tid for tid in task_ids if tid not in covered]

    canceled_set = set(canceled)

    duty_assignments = {}
    total_comp_cost = 0.0
    for d in duties:
        did = d["duty_id"]
        ci = selected[did]
        c = all_candidates[did][ci]
        total_comp_cost += c["cost"]
        duty_assignments[did] = {
            "completion_index": int(ci),
            "cost": float(c["cost"]),
            "covered_tasks": list(c["covered_tasks"]),
            "uses_taxi": bool(c["uses_taxi"]),
            "release_at_relief_point": bool(c["release_at_relief_point"]),
            "return_station": c["return_station"],
            "return_time_minutes": int(c["return_time_minutes"]),
        }

    cancel_cost = sum(cancel_penalty[tid] for tid in canceled_set)
    objective_value = float(total_comp_cost + cancel_cost)

    solution = {
        "released_problem_variant": "relief_point_completion_v1",
        "objective_value": objective_value,
        "canceled_tasks": canceled,
        "duty_assignments": duty_assignments,
    }

    if logger:
        logger.log_solution(objective_value, solution)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()