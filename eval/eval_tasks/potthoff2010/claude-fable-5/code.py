#!/usr/bin/env python3
"""
Crew rescheduling solver for released variant 'relief_point_completion_v1'.

Approach:
 1. Read instance, derive per-duty context (current station, ready time, horizon).
 2. Enumerate a finite pool of feasible completions per duty via bounded DFS
    over the task network (direct connections at relief points, at most one
    taxi move per completion, base return or emergency relief-point release).
 3. Greedy assignment for a quick feasible incumbent.
 4. Set-partitioning MIP (Gurobi): each duty picks exactly one completion,
    each unfinished task is covered or canceled. Incumbents are logged.
"""

import argparse
import json
import time
import bisect
from collections import defaultdict

INF = float("inf")


def parse_hhmm(s):
    if s is None:
        return None
    try:
        p = str(s).split(":")
        return int(p[0]) * 60 + int(p[1])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t_start = time.time()
    hard_deadline = t_start + max(5, args.time_limit) - 2.0

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    with open(args.instance_path) as f:
        inst = json.load(f)

    P = inst.get("parameters", {})
    C_CH = float(P.get("cost_duty_changed", 0))
    C_RE = float(P.get("cost_task_reassigned", 0))
    C_TR = float(P.get("cost_new_transfer", 0))
    C_TAXI = float(P.get("cost_taxi", 0))
    C_AB = float(P.get("cost_cancel_A_to_B", 0))
    C_AA = float(P.get("cost_cancel_A_to_A", 0))
    MINCONN = int(P.get("min_connection_time_minutes", 0) or 0)
    MAX_OT = int(P.get("max_overtime_minutes", 0) or 0)
    MAX_DUR = int(float(P.get("max_duty_duration_hours", 24)) * 60)

    disr = inst.get("disruption", {}) or {}
    resched = disr.get("rescheduling_time_minutes")
    if resched is None:
        resched = parse_hhmm(disr.get("rescheduling_time")) or 0
    disr_cancel = set(disr.get("canceled_tasks") or [])

    task_by_id = {t["task_id"]: t for t in inst.get("tasks", [])}
    lines = {l["line_id"]: l for l in inst.get("train_lines", [])}
    net = inst.get("network", {})
    stations = {s["id"]: s for s in net.get("stations", [])}
    is_relief = {sid: bool(s.get("is_relief_point", False)) for sid, s in stations.items()}

    taxi_adj = defaultdict(list)
    taxi_tt = {}
    for tc in net.get("taxi_connections", []) or []:
        a, b, tt = tc["from_station"], tc["to_station"], int(tc["travel_time_minutes"])
        pairs = [(a, b)]
        if tc.get("bidirectional", True):
            pairs.append((b, a))
        for (u, v) in pairs:
            if (u, v) not in taxi_tt or tt < taxi_tt[(u, v)]:
                taxi_tt[(u, v)] = tt
    for (u, v), tt in taxi_tt.items():
        taxi_adj[u].append((v, tt))

    # ---- task universe -------------------------------------------------
    U = set()
    for tid, t in task_by_id.items():
        dep = t.get("departure_time_minutes")
        if dep is None:
            dep = parse_hhmm(t.get("departure_time")) or 0
            t["departure_time_minutes"] = dep
        if t.get("arrival_time_minutes") is None:
            t["arrival_time_minutes"] = parse_hhmm(t.get("arrival_time")) or dep
        if dep >= resched and tid not in disr_cancel:
            U.add(tid)
    U_list = sorted(U)

    def cancel_pen(tid):
        t = task_by_id[tid]
        return C_AB if t.get("task_type") == "A_to_B" else C_AA

    pen = {tid: cancel_pen(tid) for tid in U_list}

    forced_cancel_list = sorted(
        tid for tid in disr_cancel
        if tid in task_by_id and task_by_id[tid]["departure_time_minutes"] >= resched
    )
    forced_cost = sum(cancel_pen(tid) for tid in forced_cancel_list)

    tasks_by_station = defaultdict(list)
    for tid in U_list:
        t = task_by_id[tid]
        tasks_by_station[t["departure_station"]].append((t["departure_time_minutes"], tid))
    for st in tasks_by_station:
        tasks_by_station[st].sort()

    # route-knowledge path stations per task
    path_cache = {}
    for tid in U_list:
        t = task_by_id[tid]
        dep, arr = t["departure_station"], t["arrival_station"]
        line = lines.get(t.get("line_id"))
        ps = {dep, arr}
        if line:
            sts = line.get("stations") or []
            if dep == arr:
                if dep in sts:
                    ps = set(sts)
            elif dep in sts and arr in sts:
                i, j = sts.index(dep), sts.index(arr)
                if i > j:
                    i, j = j, i
                ps = set(sts[i:j + 1])
        path_cache[tid] = ps

    # ---- duty contexts -------------------------------------------------
    duties_raw = list(inst.get("active_duties", []) or []) + list(inst.get("reserve_duties", []) or [])
    duty_ids = []
    CTX = {}
    for d in duties_raw:
        did = d["duty_id"]
        duty_ids.append(did)
        base = d.get("crew_base")
        smin = d.get("start_time_minutes")
        if smin is None:
            smin = parse_hhmm(d.get("start_time")) or 0
        emin = d.get("end_time_minutes")
        if emin is None:
            emin = parse_hhmm(d.get("end_time")) or smin
        max_return = min(emin + MAX_OT, smin + MAX_DUR)

        details = {}
        for td in d.get("task_details", []) or []:
            details[td.get("task_id")] = td

        start_station = None
        ready = max(resched, 0)
        if d.get("duty_type") == "reserve" or d.get("standby_station"):
            start_station = d.get("standby_station") or base
            ready = max(resched, smin)
        else:
            pre = d.get("pre_disruption_tasks") or []
            best_arr, best_st = None, None
            for tid in pre:
                td = details.get(tid) or task_by_id.get(tid)
                if not td:
                    continue
                am = td.get("arrival_time_minutes")
                if am is None:
                    am = parse_hhmm(td.get("arrival_time"))
                if am is None:
                    continue
                if best_arr is None or am > best_arr:
                    best_arr, best_st = am, td.get("arrival_station")
            if best_st is not None:
                start_station = best_st
                ready = max(resched, best_arr)
            else:
                start_station = base
                ready = max(resched, smin) if smin >= resched else resched
        if d.get("current_station"):
            start_station = d["current_station"]
        if start_station is None:
            start_station = base

        orig = list(d.get("tasks") or [])
        orig_set = set(orig)
        orig_pairs = set(zip(orig, orig[1:]))

        rk = set(d.get("route_knowledge") or [])
        lic = set(d.get("rolling_stock_licenses") or [])
        allowed = set()
        for tid in U_list:
            t = task_by_id[tid]
            if t.get("rolling_stock_type") not in lic:
                continue
            if not path_cache[tid] <= rk:
                continue
            allowed.add(tid)

        obs = defaultdict(list)
        for tid in orig:
            if tid in allowed:
                t = task_by_id[tid]
                obs[t["departure_station"]].append((t["departure_time_minutes"], tid))
        for st in obs:
            obs[st].sort()

        CTX[did] = {
            "id": did, "base": base, "start_min": smin, "end_min": emin,
            "max_return": max_return, "start_station": start_station, "ready": ready,
            "orig": orig, "orig_set": orig_set, "orig_pairs": orig_pairs,
            "allowed": allowed, "orig_by_station": dict(obs),
            "maxlen": max(8, len(orig) + 4),
        }

    def comp_cost(ctx, seq, taxi_used):
        if not taxi_used and list(seq) == ctx["orig"]:
            return 0.0
        c = C_CH
        oset = ctx["orig_set"]
        opairs = ctx["orig_pairs"]
        for t in seq:
            if t not in oset:
                c += C_RE
        for a, b in zip(seq, seq[1:]):
            if (a, b) not in opairs:
                c += C_TR
        if taxi_used:
            c += C_TAXI
        return float(c)

    def pool_add(pool, seq, cost, taxi, rel, st, tm):
        cur = pool.get(seq)
        if cur is None or cost < cur[0] - 1e-9 or (abs(cost - cur[0]) < 1e-9 and tm < cur[4]):
            pool[seq] = (cost, taxi, rel, st, tm)

    def validate_chain(ctx, seq):
        """Validate an explicit task sequence; return (station,time,taxi_used) or None."""
        station, tme = ctx["start_station"], ctx["ready"]
        taxi_used, last = False, None
        for tid in seq:
            if tid not in ctx["allowed"]:
                return None
            t = task_by_id[tid]
            dep = t["departure_time_minutes"]
            if t["departure_station"] != station:
                if taxi_used:
                    return None
                tt = taxi_tt.get((station, t["departure_station"]))
                if tt is None or dep < tme + tt:
                    return None
                if last is not None and not is_relief.get(station, False):
                    return None
                taxi_used = True
            else:
                gap = 0
                if last is not None:
                    if task_by_id[last]["rolling_stock_type"] != t["rolling_stock_type"]:
                        gap = MINCONN
                    if (last, tid) not in ctx["orig_pairs"] and not is_relief.get(station, False):
                        return None
                if dep < tme + gap:
                    return None
            if t["arrival_time_minutes"] > ctx["max_return"]:
                return None
            station, tme, last = t["arrival_station"], t["arrival_time_minutes"], tid
        return (station, tme, taxi_used)

    KDIR, KTAXI, WINDOW = 6, 3, 240
    nd = max(1, len(duty_ids))
    cap = max(300, min(5000, 400000 // nd))
    node_budget = max(4000, min(60000, 2500000 // nd))

    def enumerate_duty(ctx, tdeadline):
        base, max_return = ctx["base"], ctx["max_return"]
        allowed = ctx["allowed"]
        orig_set, orig_pairs = ctx["orig_set"], ctx["orig_pairs"]
        base_pool, relief_pool = {}, {}

        def close(station, tme, seq, taxi_used):
            if tme > max_return:
                return
            if station == base:
                pool_add(base_pool, seq, comp_cost(ctx, seq, taxi_used),
                         taxi_used, False, station, tme)
            else:
                if not taxi_used:
                    tt = taxi_tt.get((station, base))
                    if tt is not None and tme + tt <= max_return:
                        pool_add(base_pool, seq, comp_cost(ctx, seq, True),
                                 True, False, base, tme + tt)
                if is_relief.get(station, False) or not seq:
                    pool_add(relief_pool, seq, comp_cost(ctx, seq, taxi_used),
                             taxi_used, True, station, tme)

        stack = [(ctx["start_station"], ctx["ready"], (), False, None)]
        nodes = 0
        while stack:
            if nodes >= node_budget or len(base_pool) >= cap or time.time() > tdeadline:
                break
            station, tme, seq, taxi_used, last = stack.pop()
            nodes += 1
            close(station, tme, seq, taxi_used)
            if len(seq) >= ctx["maxlen"]:
                continue
            relief_here = is_relief.get(station, False)
            cand = []
            # original tasks departing here (no window limit)
            for dep, tid in ctx["orig_by_station"].get(station, []):
                if tid in seq:
                    continue
                t = task_by_id[tid]
                gap = 0
                if last is not None:
                    if task_by_id[last]["rolling_stock_type"] != t["rolling_stock_type"]:
                        gap = MINCONN
                    if (last, tid) not in orig_pairs and not relief_here:
                        continue
                if dep < tme + gap or t["arrival_time_minutes"] > max_return:
                    continue
                cand.append((1, dep, tid, taxi_used))
            # other tasks departing here (windowed, capped)
            lst = tasks_by_station.get(station, [])
            i = bisect.bisect_left(lst, (tme, ""))
            cnt = 0
            for dep, tid in lst[i:]:
                if dep > tme + WINDOW or cnt >= KDIR:
                    break
                if tid in orig_set or tid in seq or tid not in allowed:
                    continue
                t = task_by_id[tid]
                gap = 0
                if last is not None:
                    if task_by_id[last]["rolling_stock_type"] != t["rolling_stock_type"]:
                        gap = MINCONN
                    if (last, tid) not in orig_pairs and not relief_here:
                        continue
                if dep < tme + gap or t["arrival_time_minutes"] > max_return:
                    continue
                cand.append((0, dep, tid, taxi_used))
                cnt += 1
            # taxi repositioning (at most one per completion)
            if not taxi_used and (last is None or relief_here):
                for to, tt in taxi_adj.get(station, []):
                    lst2 = tasks_by_station.get(to, [])
                    j = bisect.bisect_left(lst2, (tme + tt, ""))
                    cnt2 = 0
                    for dep, tid in lst2[j:]:
                        if dep > tme + tt + WINDOW or cnt2 >= KTAXI:
                            break
                        if tid in seq or tid not in allowed:
                            continue
                        t = task_by_id[tid]
                        if t["arrival_time_minutes"] > max_return:
                            continue
                        cand.append((1 if tid in orig_set else 0, dep, tid, True))
                        cnt2 += 1
            # push so that original / earliest tasks are explored first
            cand.sort(key=lambda z: (z[0], -z[1]))
            for _, dep, tid, txu in cand:
                t = task_by_id[tid]
                stack.append((t["arrival_station"], t["arrival_time_minutes"],
                              seq + (tid,), txu, tid))

        # explicitly try the unchanged original chain
        if ctx["orig"]:
            r = validate_chain(ctx, ctx["orig"])
            if r:
                close(r[0], r[1], tuple(ctx["orig"]), r[2])
        return base_pool, relief_pool

    # ---- enumeration ----------------------------------------------------
    enum_end = min(hard_deadline, t_start + 0.45 * max(5, args.time_limit))
    cands = {}
    remaining = list(duty_ids)
    for idx, did in enumerate(remaining):
        left = max(1, len(remaining) - idx)
        tdl = min(hard_deadline, time.time() + max(0.2, (enum_end - time.time()) / left))
        ctx = CTX[did]
        bp, rp = enumerate_duty(ctx, tdl)
        pool = bp if bp else rp
        if not pool:
            pool = {(): (comp_cost(ctx, (), False), False,
                         ctx["start_station"] != ctx["base"],
                         ctx["start_station"], ctx["ready"])}
        lst = []
        for seq, (cost, taxi, rel, st, tm) in pool.items():
            lst.append({"seq": list(seq), "sset": frozenset(seq), "cost": cost,
                        "taxi": bool(taxi), "release": bool(rel),
                        "ret_station": st, "ret_time": int(tm)})
        lst.sort(key=lambda c: (c["cost"], -len(c["seq"])))
        cands[did] = lst

    # ---- helpers ---------------------------------------------------------
    def build_solution(assign):
        duty_assignments = {}
        covered = set()
        total = 0.0
        for did in duty_ids:
            k = assign[did]
            c = cands[did][k]
            duty_assignments[did] = {
                "completion_index": int(k),
                "cost": float(c["cost"]),
                "covered_tasks": list(c["seq"]),
                "uses_taxi": bool(c["taxi"]),
                "release_at_relief_point": bool(c["release"]),
                "return_station": c["ret_station"],
                "return_time_minutes": int(c["ret_time"]),
            }
            total += c["cost"]
            covered.update(c["seq"])
        canceled = [t for t in U_list if t not in covered]
        total += sum(pen[t] for t in canceled)
        total += forced_cost
        sol = {
            "released_problem_variant": "relief_point_completion_v1",
            "objective_value": float(total),
            "canceled_tasks": canceled + forced_cancel_list,
            "duty_assignments": duty_assignments,
        }
        return total, sol

    best = [INF, None, None]  # obj, sol dict, assign

    def record(assign):
        tot, sol = build_solution(assign)
        if tot < best[0] - 1e-9:
            best[0], best[1], best[2] = tot, sol, dict(assign)
            if logger:
                try:
                    logger.log_solution(tot, sol)
                except Exception:
                    pass
            return True
        return False

    # initial: cheapest completion per duty
    assign = {did: 0 for did in duty_ids}
    record(assign)

    # ---- greedy improvement ---------------------------------------------
    covercnt = defaultdict(int)
    for did in duty_ids:
        for t in cands[did][assign[did]]["sset"]:
            covercnt[t] += 1
    passes, changed = 0, True
    gdl = min(hard_deadline, enum_end + 0.15 * max(5, args.time_limit))
    while changed and passes < 8 and time.time() < gdl:
        changed = False
        passes += 1
        for did in duty_ids:
            if time.time() > gdl:
                break
            cur = assign[did]
            for t in cands[did][cur]["sset"]:
                covercnt[t] -= 1
            bestk = cur
            bestv = cands[did][cur]["cost"] - sum(
                pen[t] for t in cands[did][cur]["sset"] if covercnt[t] == 0)
            for k, c in enumerate(cands[did]):
                if k == cur:
                    continue
                v = c["cost"] - sum(pen[t] for t in c["sset"] if covercnt[t] == 0)
                if v < bestv - 1e-9:
                    bestv, bestk = v, k
            if bestk != cur:
                changed = True
            assign[did] = bestk
            for t in cands[did][bestk]["sset"]:
                covercnt[t] += 1
        record(assign)
    record(assign)

    # ---- MIP -------------------------------------------------------------
    try:
        import gurobipy as gp
        from gurobipy import GRB

        mip_time = hard_deadline - time.time()
        if mip_time > 3 and duty_ids:
            m = gp.Model("crew")
            m.Params.OutputFlag = 0
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.TimeLimit = mip_time

            idx_list = []
            for did in duty_ids:
                for k in range(len(cands[did])):
                    idx_list.append((did, k))
            x = m.addVars(len(idx_list), vtype=GRB.BINARY, name="x")
            y = m.addVars(len(U_list), vtype=GRB.BINARY, name="y")
            ypos = {t: i for i, t in enumerate(U_list)}

            by_duty = defaultdict(list)
            cover_map = defaultdict(list)
            for i, (did, k) in enumerate(idx_list):
                by_duty[did].append(i)
                for t in cands[did][k]["sset"]:
                    cover_map[t].append(i)
            for did in duty_ids:
                m.addConstr(gp.quicksum(x[i] for i in by_duty[did]) == 1)
            for t in U_list:
                m.addConstr(gp.quicksum(x[i] for i in cover_map[t]) + y[ypos[t]] >= 1)

            obj = gp.quicksum(cands[did][k]["cost"] * x[i]
                              for i, (did, k) in enumerate(idx_list))
            obj += gp.quicksum(pen[t] * y[ypos[t]] for t in U_list)
            m.setObjective(obj + forced_cost, GRB.MINIMIZE)

            # warm start from greedy
            wa = best[2] if best[2] is not None else assign
            covered_ws = set()
            for i, (did, k) in enumerate(idx_list):
                x[i].Start = 1.0 if wa.get(did) == k else 0.0
            for did in duty_ids:
                covered_ws.update(cands[did][wa[did]]["sset"])
            for t in U_list:
                y[ypos[t]].Start = 0.0 if t in covered_ws else 1.0

            xlist = [x[i] for i in range(len(idx_list))]

            def cb(model, where):
                if where == GRB.Callback.MIPSOL:
                    try:
                        vals = model.cbGetSolution(xlist)
                        a = {}
                        for i, v in enumerate(vals):
                            if v > 0.5:
                                did, k = idx_list[i]
                                a[did] = k
                        if len(a) == len(duty_ids):
                            record(a)
                    except Exception:
                        pass

            m.optimize(cb)
            if m.SolCount > 0:
                a = {}
                for i, (did, k) in enumerate(idx_list):
                    if x[i].X > 0.5:
                        a[did] = k
                if len(a) == len(duty_ids):
                    record(a)
    except Exception:
        pass

    if best[1] is None:
        # absolute fallback
        _, sol = build_solution({did: 0 for did in duty_ids})
        best[1] = sol

    with open(args.solution_path, "w") as f:
        json.dump(best[1], f, indent=1)


if __name__ == "__main__":
    main()