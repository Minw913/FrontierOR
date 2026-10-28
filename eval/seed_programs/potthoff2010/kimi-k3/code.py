import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.setrecursionlimit(1000000)


def parse_time_str(s):
    try:
        parts = str(s).split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, required=True)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t0 = time.time()
    time_limit = max(1, int(args.time_limit))
    solver_buffer = min(2.0, max(0.2, 0.1 * time_limit))

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    variant = inst.get("released_problem_variant", "relief_point_completion_v1")
    P = inst.get("parameters", {}) or {}
    dis = inst.get("disruption", {}) or {}

    resched = dis.get("rescheduling_time_minutes")
    if resched is None:
        resched = parse_time_str(dis.get("rescheduling_time", "0:00"))

    c_duty = P.get("cost_duty_changed", 0) or 0
    c_reassign = P.get("cost_task_reassigned", 0) or 0
    c_transfer = P.get("cost_new_transfer", 0) or 0
    c_taxi = P.get("cost_taxi", 0) or 0
    c_can_B = P.get("cost_cancel_A_to_B", 0) or 0
    c_can_A = P.get("cost_cancel_A_to_A", 0) or 0
    min_conn = P.get("min_connection_time_minutes", 0) or 0
    max_over = P.get("max_overtime_minutes", 0) or 0
    max_dur_h = P.get("max_duty_duration_hours", 24) or 24

    tasks = {}
    for t in inst.get("tasks", []) or []:
        tasks[t["task_id"]] = t

    def dep_min(t):
        v = t.get("departure_time_minutes")
        return v if v is not None else parse_time_str(t.get("departure_time", "0:00"))

    def arr_min(t):
        v = t.get("arrival_time_minutes")
        return v if v is not None else parse_time_str(t.get("arrival_time", "0:00"))

    net = inst.get("network", {}) or {}
    taxi_time = {}
    for tc in net.get("taxi_connections", []) or []:
        a = tc["from_station"]; b = tc["to_station"]; tt = tc["travel_time_minutes"]
        k = (a, b)
        if k not in taxi_time or tt < taxi_time[k]:
            taxi_time[k] = tt
        if tc.get("bidirectional", False):
            k = (b, a)
            if k not in taxi_time or tt < taxi_time[k]:
                taxi_time[k] = tt

    relief = {}
    for s in net.get("stations", []) or []:
        relief[s["id"]] = bool(s.get("is_relief_point", False))

    lines = {}
    for ln in inst.get("train_lines", []) or []:
        lines[ln["line_id"]] = ln

    def stations_needed(t, strict):
        dep = t["departure_station"]
        arr = t["arrival_station"]
        if dep == arr or t.get("task_type") == "A_to_A":
            return {dep}
        if not strict:
            return {dep, arr}
        ln = lines.get(t.get("line_id"))
        if ln is None:
            return {dep, arr}
        st = ln.get("stations", [])
        try:
            i = st.index(dep)
            j = st.index(arr)
        except ValueError:
            return {dep, arr}
        lo, hi = (i, j) if i <= j else (j, i)
        return set(st[lo:hi + 1])

    raw_duties = []
    for d in inst.get("active_duties", []) or []:
        raw_duties.append((d, False))
    for d in inst.get("reserve_duties", []) or []:
        raw_duties.append((d, True))

    # Adaptive route-knowledge strictness: if original duties violate the
    # strict "all stations along the line segment" rule, fall back to
    # endpoint-only checks.
    strict = True
    for d, _ in raw_duties:
        know = set(d.get("route_knowledge", []) or [])
        bad = False
        for tid in d.get("tasks", []) or []:
            t = tasks.get(tid)
            if t is not None and not stations_needed(t, True).issubset(know):
                bad = True
                break
        if bad:
            strict = False
            break
    need = {tid: stations_needed(t, strict) for tid, t in tasks.items()}

    def pre_info(duty, tid):
        t = tasks.get(tid)
        if t is not None:
            return (t["arrival_station"], arr_min(t), t.get("rolling_stock_type"))
        for td in duty.get("task_details", []) or []:
            if td.get("task_id") == tid:
                return (td.get("arrival_station"),
                        parse_time_str(td.get("arrival_time", "0:00")), None)
        return None

    duties = []
    for d, is_res in raw_duties:
        base = d.get("crew_base")
        st_min = d.get("start_time_minutes")
        if st_min is None:
            st_min = parse_time_str(d.get("start_time", "0:00"))
        en_min = d.get("end_time_minutes")
        if en_min is None:
            en_min = parse_time_str(d.get("end_time", "0:00"))
        pre = list(d.get("pre_disruption_tasks", []) or [])
        pre_ordered = []
        for tid in pre:
            info = pre_info(d, tid)
            if info is not None:
                pre_ordered.append((info[1], tid, info[0], info[2]))
        pre_ordered.sort()
        pre_tids = [x[1] for x in pre_ordered]
        pre_last = pre_tids[-1] if pre_tids else None
        pre_arr = pre_ordered[-1][0] if pre_ordered else None
        pre_rs = pre_ordered[-1][3] if pre_ordered else None
        if is_res:
            s0 = d.get("standby_station", base)
            tau0 = max(resched, st_min)
        else:
            if pre_last is not None:
                s0 = pre_ordered[-1][2]
                tau0 = max(resched, pre_arr)
            else:
                s0 = base
                tau0 = max(resched, st_min)
                pre_tids = []
                pre_last = None
                pre_arr = None
                pre_rs = None
        R = min(en_min + max_over, st_min + int(round(max_dur_h * 60)))
        orig = list(d.get("tasks", []) or [])
        full_seq = pre_tids + orig
        duties.append({
            "duty_id": d.get("duty_id", "duty_%d" % len(duties)),
            "base": base, "st": st_min, "en": en_min,
            "s0": s0, "tau0": tau0,
            "pre_arr": pre_arr, "pre_rs": pre_rs, "pre_last": pre_last,
            "pre_tids": pre_tids, "R": R,
            "orig": orig, "orig_set": set(orig),
            "orig_full_pairs": set(zip(full_seq[:-1], full_seq[1:])),
            "know": set(d.get("route_knowledge", []) or []),
            "lic": set(d.get("rolling_stock_licenses", []) or []),
        })

    dis_canceled = set(dis.get("canceled_tasks", []) or [])
    universe = set()
    for tid, t in tasks.items():
        if dep_min(t) >= resched and tid not in dis_canceled:
            universe.add(tid)

    def cancel_cost(tid):
        return c_can_B if tasks[tid].get("task_type") == "A_to_B" else c_can_A

    # Per-duty eligible tasks (licenses, route knowledge, horizon reachability)
    for du in duties:
        el = {}
        for tid in universe:
            t = tasks[tid]
            if t.get("rolling_stock_type") not in du["lic"]:
                continue
            if not need[tid].issubset(du["know"]):
                continue
            if dep_min(t) < du["tau0"]:
                continue
            if arr_min(t) > du["R"]:
                continue
            el[tid] = t
        du["elig"] = el

    # Build connection DAG per duty: start arcs, task->task arcs, end arcs.
    for du in duties:
        s0 = du["s0"]; tau0 = du["tau0"]; R = du["R"]; base = du["base"]
        ofp = du["orig_full_pairs"]
        pre_last = du["pre_last"]
        start_arcs = []
        for tid, t in du["elig"].items():
            dep = dep_min(t); dst = t["departure_station"]
            gap = min_conn if (du["pre_rs"] is not None
                               and du["pre_rs"] != t.get("rolling_stock_type")) else 0
            ready = tau0
            if du["pre_arr"] is not None and du["pre_arr"] + gap > ready:
                ready = du["pre_arr"] + gap
            nt = pre_last is not None and (pre_last, tid) not in ofp
            if dst == s0:
                if dep >= ready:
                    start_arcs.append({"task": tid, "taxi": False, "newtrans": nt})
            else:
                tt = taxi_time.get((s0, dst))
                if tt is not None and dep >= ready + tt:
                    start_arcs.append({"task": tid, "taxi": True, "newtrans": nt})
        items = sorted(du["elig"].items(), key=lambda kv: (dep_min(kv[1]), kv[0]))
        n_it = len(items)
        cap = 80 if n_it > 120 else n_it
        task_arcs = []
        for ii in range(n_it):
            tid_i, ti = items[ii]
            ai = ti["arrival_station"]; arri = arr_min(ti); rsi = ti.get("rolling_stock_type")
            succ = []
            for jj in range(n_it):
                if ii == jj:
                    continue
                tid_j, tj = items[jj]
                depj = dep_min(tj)
                if depj < arri:
                    continue
                gap = min_conn if rsi != tj.get("rolling_stock_type") else 0
                dj = tj["departure_station"]
                if ai == dj:
                    if depj >= arri + gap:
                        succ.append({"i": tid_i, "j": tid_j, "taxi": False,
                                     "newtrans": (tid_i, tid_j) not in ofp})
                else:
                    tt = taxi_time.get((ai, dj))
                    if tt is not None and depj >= arri + (tt if tt > gap else gap):
                        succ.append({"i": tid_i, "j": tid_j, "taxi": True,
                                     "newtrans": (tid_i, tid_j) not in ofp})
                if len(succ) >= cap:
                    break
            task_arcs.extend(succ)
        end_arcs = []
        for tid, t in du["elig"].items():
            a = t["arrival_station"]; arr = arr_min(t)
            base_ok = False
            if a == base:
                if arr <= R:
                    end_arcs.append({"task": tid, "taxi": False, "release": False,
                                     "ret_station": base, "ret_time": arr})
                    base_ok = True
            else:
                tt = taxi_time.get((a, base))
                if tt is not None and arr + tt <= R:
                    end_arcs.append({"task": tid, "taxi": True, "release": False,
                                     "ret_station": base, "ret_time": arr + tt})
                    base_ok = True
            if not base_ok and a != base and relief.get(a, False) and arr <= R:
                end_arcs.append({"task": tid, "taxi": False, "release": True,
                                 "ret_station": a, "ret_time": arr})
        start_end = None
        if s0 == base:
            if tau0 <= R:
                start_end = {"task": None, "taxi": False, "release": False,
                             "ret_station": base, "ret_time": tau0}
        else:
            tt = taxi_time.get((s0, base))
            if tt is not None and tau0 + tt <= R:
                start_end = {"task": None, "taxi": True, "release": False,
                             "ret_station": base, "ret_time": tau0 + tt}
            elif relief.get(s0, False) and tau0 <= R:
                start_end = {"task": None, "taxi": False, "release": True,
                             "ret_station": s0, "ret_time": tau0}
        du["start_arcs"] = start_arcs
        du["task_arcs"] = task_arcs
        du["end_arcs"] = end_arcs
        du["start_end"] = start_end

    # Reachability, emergency relief-point release pruning (base return is
    # mandatory whenever the frozen network/horizon permits it), fallbacks.
    for du in duties:
        out = defaultdict(list)
        for a in du["task_arcs"]:
            out[a["i"]].append(a)
        reach = set()
        stack = [a["task"] for a in du["start_arcs"]]
        while stack:
            u = stack.pop()
            if u in reach:
                continue
            reach.add(u)
            for a in out.get(u, []):
                if a["j"] not in reach:
                    stack.append(a["j"])
        base_possible = du["start_end"] is not None and not du["start_end"]["release"]
        if not base_possible:
            for a in du["end_arcs"]:
                if not a["release"] and a["task"] in reach:
                    base_possible = True
                    break
        if base_possible:
            du["end_arcs"] = [a for a in du["end_arcs"] if not a["release"]]
            if du["start_end"] is not None and du["start_end"]["release"]:
                du["start_end"] = None
        du["task_arcs"] = [a for a in du["task_arcs"] if a["i"] in reach]
        du["end_arcs"] = [a for a in du["end_arcs"] if a["task"] in reach]
        if du["start_end"] is None and not du["end_arcs"]:
            du["start_end"] = {"task": None, "taxi": False,
                               "release": du["s0"] != du["base"],
                               "ret_station": du["s0"], "ret_time": du["tau0"]}
        out2 = defaultdict(list)
        for a in du["task_arcs"]:
            out2[a["i"]].append(a)
        for k in out2:
            out2[k].sort(key=lambda a: (dep_min(tasks[a["j"]]), a["j"]))
        du["out_arcs"] = dict(out2)
        eb = defaultdict(list)
        for a in du["end_arcs"]:
            eb[a["task"]].append(a)
        du["end_by_task"] = dict(eb)
        du["starts_sorted"] = sorted(du["start_arcs"],
                                     key=lambda a: (dep_min(tasks[a["task"]]), a["task"]))

    def duty_cost(du, seq, taxi_cnt):
        if seq == du["orig"] and taxi_cnt == 0:
            return 0.0
        n_re = 0
        for x in seq:
            if x not in du["orig_set"]:
                n_re += 1
        full = du["pre_tids"] + list(seq)
        ofp = du["orig_full_pairs"]
        n_nt = 0
        for k in range(len(full) - 1):
            if (full[k], full[k + 1]) not in ofp:
                n_nt += 1
        return float(c_duty + c_reassign * n_re + c_transfer * n_nt + c_taxi * taxi_cnt)

    def trivial_entries():
        entries = {}
        for du in duties:
            se = du["start_end"]
            if se is not None:
                seq = []
                taxi_cnt = 1 if se["taxi"] else 0
                entries[du["duty_id"]] = {
                    "covered_tasks": seq, "taxi_cnt": taxi_cnt,
                    "release": se["release"], "ret_station": se["ret_station"],
                    "ret_time": se["ret_time"],
                    "cost": duty_cost(du, seq, taxi_cnt),
                }
            else:
                entries[du["duty_id"]] = {
                    "covered_tasks": [], "taxi_cnt": 0,
                    "release": du["s0"] != du["base"], "ret_station": du["s0"],
                    "ret_time": du["tau0"],
                    "cost": duty_cost(du, [], 0),
                }
        return entries

    def completion_index_of(du, seq, budget):
        # Best-effort rank of the selected completion in a canonical DFS
        # enumeration of the duty's feasible completions (0 if not found).
        try:
            target = tuple(seq)
            end_nodes = set(du["end_by_task"].keys())
            count = [0]
            steps = [0]
            max_steps = 20 * budget + 5000
            found = [None]
            if du["start_end"] is not None:
                if len(target) == 0:
                    return 0
                count[0] = 1
            out = du["out_arcs"]

            def dfs(u, path):
                if found[0] is not None or count[0] >= budget or steps[0] > max_steps:
                    return
                steps[0] += 1
                if u in end_nodes:
                    if tuple(path) == target:
                        found[0] = count[0]
                        return
                    count[0] += 1
                for a in out.get(u, []):
                    path.append(a["j"])
                    dfs(a["j"], path)
                    path.pop()
                    if found[0] is not None or count[0] >= budget or steps[0] > max_steps:
                        return

            for a in du["starts_sorted"]:
                dfs(a["task"], [a["task"]])
                if found[0] is not None or count[0] >= budget or steps[0] > max_steps:
                    break
            return found[0] if found[0] is not None else 0
        except Exception:
            return 0

    def finalize(entries, idx_budget):
        covered = set()
        total = 0.0
        for du in duties:
            e = entries[du["duty_id"]]
            covered.update(e["covered_tasks"])
            total += e["cost"]
        canceled = sorted(tid for tid in universe if tid not in covered)
        for tid in canceled:
            total += cancel_cost(tid)
        dass = {}
        for du in duties:
            e = entries[du["duty_id"]]
            if time.time() - t0 > time_limit - 0.3:
                ci = 0
            else:
                ci = completion_index_of(du, e["covered_tasks"], idx_budget)
            dass[du["duty_id"]] = {
                "completion_index": int(ci),
                "cost": float(e["cost"]),
                "covered_tasks": list(e["covered_tasks"]),
                "uses_taxi": bool(e["taxi_cnt"] > 0),
                "release_at_relief_point": bool(e["release"]),
                "return_station": e["ret_station"],
                "return_time_minutes": int(round(e["ret_time"])),
            }
        sol = {
            "released_problem_variant": variant,
            "objective_value": float(total),
            "canceled_tasks": canceled,
            "duty_assignments": dass,
        }
        return sol, float(total)

    def extract_entries(getval):
        entries = {}
        for du in duties:
            seq = []
            taxi_cnt = 0
            release = False
            ret_station = du["s0"]
            ret_time = du["tau0"]
            se = du["start_end"]
            used_se = se is not None and getval(se["gid"]) > 0.5
            if used_se:
                release = se["release"]
                ret_station = se["ret_station"]
                ret_time = se["ret_time"]
                if se["taxi"]:
                    taxi_cnt += 1
            else:
                cur = None
                for a in du["start_arcs"]:
                    if getval(a["gid"]) > 0.5:
                        cur = a["task"]
                        if a["taxi"]:
                            taxi_cnt += 1
                        break
                guard = 0
                while cur is not None:
                    guard += 1
                    if guard > len(du["elig"]) + 2:
                        break
                    seq.append(cur)
                    nxt = None
                    for a in du["out_arcs"].get(cur, []):
                        if getval(a["gid"]) > 0.5:
                            nxt = a["j"]
                            if a["taxi"]:
                                taxi_cnt += 1
                            break
                    if nxt is None:
                        for a in du["end_by_task"].get(cur, []):
                            if getval(a["gid"]) > 0.5:
                                release = a["release"]
                                ret_station = a["ret_station"]
                                ret_time = a["ret_time"]
                                if a["taxi"]:
                                    taxi_cnt += 1
                                break
                    cur = nxt
            entries[du["duty_id"]] = {
                "covered_tasks": seq, "taxi_cnt": taxi_cnt,
                "release": release, "ret_station": ret_station,
                "ret_time": ret_time,
                "cost": duty_cost(du, seq, taxi_cnt),
            }
        return entries

    entries0 = trivial_entries()
    sol0, obj0 = finalize(entries0, idx_budget=2000)
    holder = {"sol": sol0, "val": obj0}
    if logger is not None:
        try:
            logger.log_solution(obj0, sol0)
        except Exception:
            pass

    try:
        import gurobipy as gp
        from gurobipy import GRB

        m = gp.Model("relief_point_completion_v1")
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.NumericFocus = 0
        m.Params.Threads = 1
        m.Params.OutputFlag = 0
        m.Params.LogToConsole = 0

        all_vars = []
        for du in duties:
            nS = len(du["start_arcs"])
            nT = len(du["task_arcs"])
            nE = len(du["end_arcs"])
            sv = m.addVars(nS, vtype=GRB.BINARY, name="s")
            tv = m.addVars(nT, vtype=GRB.BINARY, name="a")
            ev = m.addVars(nE, vtype=GRB.BINARY, name="e")
            for k in range(nS):
                du["start_arcs"][k]["var"] = sv[k]
                du["start_arcs"][k]["gid"] = len(all_vars)
                all_vars.append(sv[k])
            for k in range(nT):
                du["task_arcs"][k]["var"] = tv[k]
                du["task_arcs"][k]["gid"] = len(all_vars)
                all_vars.append(tv[k])
            for k in range(nE):
                du["end_arcs"][k]["var"] = ev[k]
                du["end_arcs"][k]["gid"] = len(all_vars)
                all_vars.append(ev[k])
            if du["start_end"] is not None:
                v = m.addVar(vtype=GRB.BINARY, name="se")
                du["start_end"]["var"] = v
                du["start_end"]["gid"] = len(all_vars)
                all_vars.append(v)
            if du["elig"]:
                du["x"] = m.addVars(list(du["elig"].keys()), vtype=GRB.BINARY, name="x")
            else:
                du["x"] = {}
            du["y"] = m.addVar(vtype=GRB.BINARY, name="y")
        cancel = m.addVars(list(universe), vtype=GRB.BINARY, name="c")
        m.update()

        obj = gp.LinExpr()
        for du in duties:
            y = du["y"]
            starts = [a["var"] for a in du["start_arcs"]]
            if du["start_end"] is not None:
                starts.append(du["start_end"]["var"])
            m.addConstr(gp.quicksum(starts) == 1)

            in_a = defaultdict(list)
            out_a = defaultdict(list)
            for a in du["start_arcs"]:
                in_a[a["task"]].append(a["var"])
            for a in du["task_arcs"]:
                out_a[a["i"]].append(a["var"])
                in_a[a["j"]].append(a["var"])
            ends = []
            for a in du["end_arcs"]:
                out_a[a["task"]].append(a["var"])
                ends.append(a["var"])
            if du["start_end"] is not None:
                ends.append(du["start_end"]["var"])
            for tid in du["elig"]:
                xv = du["x"][tid]
                m.addConstr(xv == gp.quicksum(in_a.get(tid, [])))
                m.addConstr(xv == gp.quicksum(out_a.get(tid, [])))
            m.addConstr(gp.quicksum(ends) == 1)

            for tid in du["elig"]:
                if tid in du["orig_set"]:
                    m.addConstr(du["x"][tid] + y >= 1)
                else:
                    m.addConstr(du["x"][tid] <= y)
            for tid in du["orig_set"]:
                if tid not in du["elig"]:
                    m.addConstr(y >= 1)

            nt_vars = [a["var"] for a in du["start_arcs"] if a.get("newtrans")]
            nt_vars += [a["var"] for a in du["task_arcs"] if a.get("newtrans")]
            tx_vars = [a["var"] for a in du["start_arcs"] if a.get("taxi")]
            tx_vars += [a["var"] for a in du["task_arcs"] if a.get("taxi")]
            tx_vars += [a["var"] for a in du["end_arcs"] if a.get("taxi")]
            if du["start_end"] is not None and du["start_end"].get("taxi"):
                tx_vars.append(du["start_end"]["var"])
            if nt_vars:
                m.addConstr(gp.quicksum(nt_vars) <= len(nt_vars) * y)
            if tx_vars:
                m.addConstr(gp.quicksum(tx_vars) <= len(tx_vars) * y)

            obj += c_duty * y
            rv = [du["x"][tid] for tid in du["elig"] if tid not in du["orig_set"]]
            if rv:
                obj += c_reassign * gp.quicksum(rv)
            if nt_vars:
                obj += c_transfer * gp.quicksum(nt_vars)
            if tx_vars:
                obj += c_taxi * gp.quicksum(tx_vars)

        for tid in universe:
            xs = [du["x"][tid] for du in duties if tid in du["elig"]]
            if xs:
                m.addConstr(gp.quicksum(xs) + cancel[tid] >= 1)
            else:
                m.addConstr(cancel[tid] >= 1)
            obj += cancel_cost(tid) * cancel[tid]

        m.setObjective(obj, GRB.MINIMIZE)

        # MIP start: trivial "keep empty completions, cancel everything".
        try:
            for du in duties:
                if du["start_end"] is not None:
                    du["start_end"]["var"].Start = 1.0
                du["y"].Start = 1.0 if du["orig"] else 0.0
            for tid in universe:
                cancel[tid].Start = 1.0
        except Exception:
            pass

        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    if time.time() - t0 > time_limit - solver_buffer:
                        model.terminate()
                        return
                    vals = model.cbGetSolution(all_vars)
                    entries = extract_entries(lambda g: vals[g])
                    sol, tot = finalize(entries, idx_budget=500)
                    if tot < holder["val"] - 1e-6:
                        holder["val"] = tot
                        holder["sol"] = sol
                        if logger is not None:
                            logger.log_solution(tot, sol)
                except Exception:
                    pass

        m.Params.TimeLimit = max(0.5, float(time_limit) - (time.time() - t0) - solver_buffer)
        m.optimize(callback)

        try:
            if m.SolCount > 0:
                vals = m.getAttr("X", all_vars)
                entries = extract_entries(lambda g: vals[g])
                sol, tot = finalize(entries, idx_budget=20000)
                if tot < holder["val"] - 1e-6:
                    holder["val"] = tot
                    holder["sol"] = sol
                    if logger is not None:
                        logger.log_solution(tot, sol)
                elif tot <= holder["val"] + 1e-6:
                    holder["sol"] = sol
        except Exception:
            pass
    except Exception:
        pass

    solution = holder["sol"]
    outdir = os.path.dirname(os.path.abspath(args.solution_path))
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()