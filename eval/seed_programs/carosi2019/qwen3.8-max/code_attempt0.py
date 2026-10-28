import argparse
import json
import os
import time
import bisect
from collections import defaultdict

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except Exception:
    GUROBI_AVAILABLE = False

SOURCE = -1
SINK = -2
EPS = 1e-6
DEFAULT_MAX_STOP = 10_000_000.0
DEFAULT_MAX_HEADWAY_SEC = 1_000_000_000.0


def select_timetable_positions(lo, hi, cap, target, mains):
    if hi < lo:
        return []
    count = hi - lo + 1
    if cap is None or count <= cap:
        return list(range(lo, hi + 1))

    chosen = {lo, hi}
    center = bisect.bisect_left(mains, target, lo, hi + 1)
    center = max(lo, min(hi, center))

    remaining = cap - len(chosen)
    if remaining > 0:
        half = remaining // 2
        start = max(lo, center - half)
        end = min(hi, start + remaining - 1)
        start = max(lo, end - remaining + 1)
        for p in range(start, end + 1):
            chosen.add(p)

    if len(chosen) < cap:
        need = cap - len(chosen)
        denom = max(1, need * 3 - 1)
        for t in range(need * 3):
            if len(chosen) >= cap:
                break
            p = lo + int((count - 1) * (t / denom))
            chosen.add(p)

    return sorted(chosen)[:cap]


def choose_positions_range(lo, hi, cap):
    if hi < lo:
        return []
    count = hi - lo + 1
    if cap is None or count <= cap:
        return list(range(lo, hi + 1))

    chosen = set()
    k_first = max(1, cap // 3)
    for p in range(lo, min(hi + 1, lo + k_first)):
        chosen.add(p)

    k_last = max(1, cap // 3)
    for p in range(max(lo, hi - k_last + 1), hi + 1):
        chosen.add(p)

    if len(chosen) < cap:
        rem = cap - len(chosen)
        if rem == 1:
            chosen.add((lo + hi) // 2)
        else:
            for t in range(rem):
                p = lo + int((count - 1) * t / (rem - 1))
                chosen.add(p)
                if len(chosen) >= cap:
                    break

    return sorted(chosen)[:cap]


def get_by_pattern(d, p):
    if not isinstance(d, dict):
        return None
    if p in d:
        return d[p]
    sp = str(p)
    if sp in d:
        return d[sp]
    try:
        ip = int(p)
        if ip in d:
            return d[ip]
    except Exception:
        pass
    return None


def find_deployment_cost(instance):
    candidates = []

    def add(v):
        try:
            fv = float(v)
            if fv >= 0:
                candidates.append(fv)
        except Exception:
            pass

    exact_keys = {
        "per_vehicle_deployment_cost",
        "vehicle_deployment_cost",
        "per_vehicle_cost",
        "vehicle_cost",
        "fleet_cost",
        "deployment_cost",
        "bus_cost",
        "fixed_vehicle_cost",
    }

    obj = instance.get("objective_function") if isinstance(instance.get("objective_function"), dict) else {}
    for d in (instance, obj):
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            if str(k).lower() in exact_keys and isinstance(v, (int, float)) and not isinstance(v, bool):
                add(v)

    def rec(o, depth=0):
        if depth > 4:
            return
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "potential_trips":
                    continue
                kl = str(k).lower()
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    if ("cost" in kl and any(s in kl for s in ("vehicle", "fleet", "deployment", "bus"))) or kl in exact_keys:
                        add(v)
                elif isinstance(v, (dict, list)):
                    rec(v, depth + 1)
        elif isinstance(o, list):
            for item in o[:10]:
                rec(item, depth + 1)

    rec({k: v for k, v in instance.items() if k != "potential_trips"})

    positive = [c for c in candidates if c > 0]
    if positive:
        return max(positive)
    return 1_000_000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("positional", nargs="*", help=argparse.SUPPRESS)
    parser.add_argument("--instance-path", default=None)
    parser.add_argument("--solution-path", default=None)
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--log-path", default=None)
    args, _ = parser.parse_known_args()

    pos = args.positional or []
    if args.instance_path is None and len(pos) >= 1:
        args.instance_path = pos[0]
    if args.solution_path is None and len(pos) >= 2:
        args.solution_path = pos[1]
    if args.time_limit is None and len(pos) >= 3:
        try:
            args.time_limit = float(pos[2])
        except Exception:
            args.time_limit = 60.0
    if args.time_limit is None:
        args.time_limit = 60.0

    if not args.instance_path or not args.solution_path:
        parser.error("instance-path and solution-path are required")

    start_time = time.time()
    time_limit = max(1.0, float(args.time_limit))

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    try:
        with open(args.instance_path, "r") as f:
            instance = json.load(f)
    except Exception:
        instance = {}

    topology = instance.get("topology", {}) or {}
    depot = str(topology.get("depot", "O"))

    objfun = instance.get("objective_function", {}) or {}
    try:
        alpha = float(objfun.get("alpha", 1.0))
    except Exception:
        alpha = 1.0

    per_vehicle = find_deployment_cost(instance)

    tw = {}
    for w in instance.get("time_windows", []) or []:
        if not isinstance(w, dict):
            continue
        try:
            idx = int(w.get("index", w.get("time_window_index", 0)))
        except Exception:
            continue
        try:
            ideal = float(w.get("ideal_headway_seconds", 0.0))
        except Exception:
            ideal = 0.0
        try:
            min_h = float(w.get("min_headway_seconds", 0.0))
        except Exception:
            min_h = 0.0
        try:
            max_h = float(w.get("max_headway_seconds", DEFAULT_MAX_HEADWAY_SEC))
        except Exception:
            max_h = DEFAULT_MAX_HEADWAY_SEC
        tw[idx] = {"ideal": ideal, "min": min_h, "max": max_h}

    def get_tw(i):
        return tw.get(i, {"ideal": 0.0, "min": 0.0, "max": DEFAULT_MAX_HEADWAY_SEC})

    stop_map = {}
    stopping = instance.get("stopping_times", {}) or {}
    if isinstance(stopping, dict):
        for term, recs in stopping.items():
            if isinstance(recs, dict):
                recs = list(recs.values())
            if not isinstance(recs, list):
                continue
            for r in recs:
                if not isinstance(r, dict):
                    continue
                try:
                    widx = int(r.get("time_window_index", 0))
                except Exception:
                    continue
                try:
                    mn = float(r.get("min_stopping_time_minutes", 0.0))
                except Exception:
                    mn = 0.0
                try:
                    mx = float(r.get("max_stopping_time_minutes", DEFAULT_MAX_STOP))
                except Exception:
                    mx = DEFAULT_MAX_STOP
                stop_map[(str(term), widx)] = (mn, mx)

    pull_map = {}
    pulling = instance.get("pull_in_out_times", {}) or {}
    if isinstance(pulling, dict):
        for term, recs in pulling.items():
            if isinstance(recs, dict):
                recs = list(recs.values())
            if not isinstance(recs, list):
                continue
            for r in recs:
                if not isinstance(r, dict):
                    continue
                try:
                    widx = int(r.get("time_window_index", 0))
                except Exception:
                    continue
                try:
                    out_t = float(r.get("pull_out_time_minutes", 0.0))
                except Exception:
                    out_t = 0.0
                try:
                    in_t = float(r.get("pull_in_time_minutes", 0.0))
                except Exception:
                    in_t = 0.0
                pull_map[(str(term), widx)] = (out_t, in_t)

    patterns = []
    for d in instance.get("directions", []) or []:
        if not isinstance(d, dict):
            continue
        pid = d.get("pattern_id", d.get("id"))
        if pid is not None and pid not in patterns:
            patterns.append(pid)

    raw_trips = instance.get("potential_trips", []) or []
    trips = []
    for idx, t in enumerate(raw_trips):
        if not isinstance(t, dict):
            continue
        pid = t.get("pattern_id", t.get("direction", 0))
        if pid is None:
            pid = 0
        if pid not in patterns:
            patterns.append(pid)

        init_flag = False
        final_flag = False
        for k, v in t.items():
            if isinstance(v, bool):
                kl = str(k).lower()
                if v and ("initial" in kl or "first" in kl):
                    init_flag = True
                if v and ("final" in kl or "last" in kl):
                    final_flag = True

        try:
            trip_id = int(t.get("id", idx))
        except Exception:
            trip_id = idx

        try:
            dep_t = float(t.get("departure_time_minutes", 0.0))
        except Exception:
            dep_t = 0.0
        try:
            arr_t = float(t.get("arrival_time_minutes", dep_t))
        except Exception:
            arr_t = dep_t
        try:
            main_t = float(t.get("main_stop_arrival_time_minutes", dep_t))
        except Exception:
            main_t = dep_t
        try:
            widx = int(t.get("time_window_index", 0))
        except Exception:
            widx = 0

        trips.append({
            "idx": len(trips),
            "id": trip_id,
            "pattern": pid,
            "start": t.get("start_terminal", depot),
            "end": t.get("end_terminal", depot),
            "dep": dep_t,
            "arr": arr_t,
            "main": main_t,
            "win": widx,
            "initial_flag": init_flag,
            "final_flag": final_flag,
        })

    N = len(trips)

    if N == 0:
        empty_sol = {
            "objective_value": 0.0,
            "selected_trips": [],
            "num_vehicles": 0,
            "tt_arcs_used": {str(p): [] for p in patterns},
            "vs_flows": {"VS_O_plus-->VS_O_minus": 0.0},
        }
        if logger is not None:
            try:
                logger.log_solution(0.0, empty_sol)
            except Exception:
                pass
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.solution_path)), exist_ok=True)
        except Exception:
            pass
        with open(args.solution_path, "w") as f:
            json.dump(empty_sol, f)
        return

    pattern_trips = defaultdict(list)
    for i, t in enumerate(trips):
        pattern_trips[t["pattern"]].append(i)

    active_patterns = [p for p in patterns if pattern_trips.get(p)]

    term_names = {depot}
    for t in trips:
        term_names.add(str(t["start"]))
        term_names.add(str(t["end"]))
    term_id = {name: i for i, name in enumerate(sorted(term_names))}

    start_tid = [term_id[str(t["start"])] for t in trips]
    end_tid = [term_id[str(t["end"])] for t in trips]
    dep = [t["dep"] for t in trips]
    arr = [t["arr"] for t in trips]
    main_stop = [t["main"] for t in trips]
    wins = [t["win"] for t in trips]
    ids = [t["id"] for t in trips]

    initial_wins = instance.get("initial_trip_time_windows", {}) or {}
    final_wins = instance.get("final_trip_time_windows", {}) or {}
    any_init_flag = any(t["initial_flag"] for t in trips)
    any_final_flag = any(t["final_flag"] for t in trips)

    first_sets = {}
    final_sets = {}
    for p in active_patterns:
        idxs = pattern_trips[p]

        s = set()
        init_val = get_by_pattern(initial_wins, p)
        if init_val is not None:
            try:
                w = int(init_val)
                s = {i for i in idxs if trips[i]["win"] == w}
            except Exception:
                s = set()
        if not s and any_init_flag:
            s = {i for i in idxs if trips[i]["initial_flag"]}
        if not s:
            s = set(idxs)
        first_sets[p] = s

        s = set()
        final_val = get_by_pattern(final_wins, p)
        if final_val is not None:
            try:
                w = int(final_val)
                s = {i for i in idxs if trips[i]["win"] == w}
            except Exception:
                s = set()
        if not s and any_final_flag:
            s = {i for i in idxs if trips[i]["final_flag"]}
        if not s:
            s = set(idxs)
        final_sets[p] = s

    stop_min_end = [0.0] * N
    stop_max_end = [DEFAULT_MAX_STOP] * N
    pull_out_start = [0.0] * N
    pull_in_end = [0.0] * N
    depot_min_prev = [0.0] * N

    for i, t in enumerate(trips):
        mn, mx = stop_map.get((str(t["end"]), t["win"]), (0.0, DEFAULT_MAX_STOP))
        stop_min_end[i] = mn
        stop_max_end[i] = mx

        out_t, in_t = pull_map.get((str(t["start"]), t["win"]), (0.0, 0.0))
        pull_out_start[i] = out_t

        out_t2, in_t2 = pull_map.get((str(t["end"]), t["win"]), (0.0, 0.0))
        pull_in_end[i] = in_t2

        dmin, _ = stop_map.get((str(depot), t["win"]), (0.0, DEFAULT_MAX_STOP))
        depot_min_prev[i] = dmin

    def pair_cost(i, j):
        gap = dep[j] - arr[i]
        if gap < -EPS:
            return None
        if gap < 0.0:
            gap = 0.0

        if end_tid[i] == start_tid[j]:
            if gap + EPS >= stop_min_end[i] and gap <= stop_max_end[i] + EPS:
                return max(0.0, gap - stop_min_end[i])
            return None

        req = pull_in_end[i] + depot_min_prev[i] + pull_out_start[j]
        if gap + EPS >= req:
            return pull_in_end[i] + pull_out_start[j]
        return None

    if N <= 300:
        tt_cap = None
    else:
        tt_cap = max(50, min(500, int(150000 // max(1, N))))
    if time_limit < 30 and tt_cap is not None:
        tt_cap = max(30, tt_cap // 2)

    tt_arcs = []
    tt_in = defaultdict(list)
    tt_out = defaultdict(list)
    source_arcs = defaultdict(list)
    sink_arcs = defaultdict(list)
    tt_key = {}
    pattern_order = {}

    for p in active_patterns:
        order = sorted(pattern_trips[p], key=lambda i: (main_stop[i], dep[i], ids[i]))
        pattern_order[p] = order
        mains = [main_stop[i] for i in order]

        for i in order:
            if i in first_sets[p]:
                k = len(tt_arcs)
                tt_arcs.append((SOURCE, i, 0.0))
                tt_in[i].append(k)
                source_arcs[p].append(k)
                tt_key[(p, SOURCE, i)] = k
            if i in final_sets[p]:
                k = len(tt_arcs)
                tt_arcs.append((i, SINK, 0.0))
                tt_out[i].append(k)
                sink_arcs[p].append(k)
                tt_key[(p, i, SINK)] = k

        for pos, i in enumerate(order):
            info = get_tw(wins[i])
            min_h = info["min"]
            max_h = info["max"]
            ideal = info["ideal"]
            if max_h < min_h:
                max_h = min_h

            main_i = mains[pos]
            lo = bisect.bisect_left(mains, main_i + min_h / 60.0 - 1e-9, pos + 1)
            hi = bisect.bisect_right(mains, main_i + max_h / 60.0 + 1e-9, pos + 1) - 1
            if hi < lo:
                continue

            count = hi - lo + 1
            if tt_cap is None or count <= tt_cap:
                positions = range(lo, hi + 1)
            else:
                target = main_i + ideal / 60.0
                positions = select_timetable_positions(lo, hi, tt_cap, target, mains)

            for posj in positions:
                j = order[posj]
                actual = (mains[posj] - main_i) * 60.0
                if actual + EPS < min_h or actual > max_h + EPS:
                    continue
                cost = (actual - ideal) ** 2
                k = len(tt_arcs)
                tt_arcs.append((i, j, cost))
                tt_out[i].append(k)
                tt_in[j].append(k)
                tt_key[(p, i, j)] = k

    def dp_path(p, allowed=None):
        order = pattern_order.get(p, [])
        if not order:
            return []

        dist = {}
        pred = {}
        for k in source_arcs.get(p, []):
            v = tt_arcs[k][1]
            if allowed is None or v in allowed:
                if v not in dist or 0.0 < dist[v]:
                    dist[v] = 0.0
                    pred[v] = None

        best_last = None
        best_cost = float("inf")

        for i in order:
            if allowed is not None and i not in allowed:
                continue
            if i not in dist:
                continue
            d = dist[i]
            for k in tt_out.get(i, []):
                _, v, c = tt_arcs[k]
                if v == SINK:
                    if d + c < best_cost:
                        best_cost = d + c
                        best_last = i
                else:
                    if allowed is not None and v not in allowed:
                        continue
                    nd = d + c
                    if v not in dist or nd < dist[v]:
                        dist[v] = nd
                        pred[v] = i

        if best_last is None:
            return []

        path = []
        cur = best_last
        seen = set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            path.append(cur)
            cur = pred.get(cur)
        path.reverse()
        return path

    def fallback_path(p):
        order = pattern_order.get(p, [])
        if not order:
            return []
        srcs = {tt_arcs[k][1] for k in source_arcs.get(p, [])}
        sinks = {tt_arcs[k][0] for k in sink_arcs.get(p, [])}
        inter = srcs & sinks
        if inter:
            return [min(inter, key=lambda i: (main_stop[i], ids[i]))]
        if srcs:
            return [min(srcs, key=lambda i: (main_stop[i], ids[i]))]
        if sinks:
            return [min(sinks, key=lambda i: (main_stop[i], ids[i]))]
        return [order[0]]

    heuristic_paths = {}
    dp_ok = True
    for p in active_patterns:
        path = dp_path(p)
        if not path:
            dp_ok = False
            path = fallback_path(p)
        heuristic_paths[p] = path

    env = None
    if GUROBI_AVAILABLE:
        try:
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.setParam("Seed", 0)
            env.setParam("Threads", 1)
            env.setParam("NumericFocus", 0)
            env.setParam("MIPGap", 1e-4)
            env.start()
        except Exception:
            env = None

    def generate_comp_arcs(indices, cap=None, time_budget=None):
        if not indices:
            return [], {}
        order = sorted(indices, key=lambda i: (dep[i], ids[i]))
        deps = [dep[i] for i in order]
        arcs = []
        key = {}
        st = time.time()

        for pos_i, i in enumerate(order):
            if time_budget is not None and time.time() - st > time_budget:
                break
            start_pos = bisect.bisect_left(deps, arr[i] - EPS, pos_i + 1)
            if start_pos >= len(order):
                continue

            if cap is None:
                positions = range(start_pos, len(order))
            else:
                positions = choose_positions_range(start_pos, len(order) - 1, cap)

            for pos_j in positions:
                j = order[pos_j]
                if j == i:
                    continue
                c = pair_cost(i, j)
                if c is not None:
                    key[(i, j)] = len(arcs)
                    arcs.append((i, j, c))

        return arcs, key

    def solve_vehicle_lp(selected, time_budget, cap=None):
        if env is None or not selected:
            return None
        selected = list(selected)
        gen_budget = max(1.0, min(20.0, time_budget * 0.4)) if time_budget else None
        arcs, _ = generate_comp_arcs(selected, cap=cap, time_budget=gen_budget)

        try:
            m = gp.Model(env=env)
            m.Params.OutputFlag = 0
            m.Params.Seed = 0
            m.Params.Threads = 1
            m.Params.NumericFocus = 0
            if time_budget is not None and time_budget > 0:
                m.Params.TimeLimit = max(0.5, float(time_budget))

            v_start = m.addVars(selected, lb=0.0, ub=1.0, name="v_start")
            v_end = m.addVars(selected, lb=0.0, ub=1.0, name="v_end")
            v_comp = m.addVars(len(arcs), lb=0.0, ub=1.0, name="v_comp")
            v_circ = m.addVar(lb=0.0, ub=float(len(selected)), name="v_circ")

            pred = defaultdict(list)
            succ = defaultdict(list)
            for k, (i, j, _) in enumerate(arcs):
                pred[j].append(k)
                succ[i].append(k)

            m.addConstrs(
                (v_start[j] + gp.quicksum(v_comp[k] for k in pred[j]) == 1.0 for j in selected),
                name="start_cov"
            )
            m.addConstrs(
                (v_end[i] + gp.quicksum(v_comp[k] for k in succ[i]) == 1.0 for i in selected),
                name="end_cov"
            )
            m.addConstr(v_circ == gp.quicksum(v_start[i] for i in selected), name="circ_start")
            m.addConstr(v_circ == gp.quicksum(v_end[i] for i in selected), name="circ_end")

            secondary = (
                gp.quicksum(pull_out_start[i] * v_start[i] for i in selected)
                + gp.quicksum(pull_in_end[i] * v_end[i] for i in selected)
                + gp.quicksum(arcs[k][2] * v_comp[k] for k in range(len(arcs)))
            )

            if alpha > 0:
                obj = alpha * (per_vehicle * v_circ + secondary)
            else:
                obj = v_circ + 1e-6 * secondary
            m.setObjective(obj, GRB.MINIMIZE)
            m.optimize()

            if m.SolCount == 0:
                return None

            start = {}
            end = {}
            comp = {}
            comp_cost = {}

            for i in selected:
                if v_start[i].X > 0.5:
                    start[i] = 1.0
                if v_end[i].X > 0.5:
                    end[i] = 1.0

            for k, (i, j, c) in enumerate(arcs):
                if v_comp[k].X > 0.5:
                    comp[(i, j)] = 1.0
                    comp_cost[(i, j)] = c

            ok = True
            for j in selected:
                inc = 1 if j in start else 0
                for k in pred[j]:
                    if v_comp[k].X > 0.5:
                        inc += 1
                if inc != 1:
                    ok = False
                    break

            if ok:
                for i in selected:
                    out = 1 if i in end else 0
                    for k in succ[i]:
                        if v_comp[k].X > 0.5:
                            out += 1
                    if out != 1:
                        ok = False
                        break

            if not ok or not start:
                return None

            return {
                "start": start,
                "end": end,
                "comp": comp,
                "circ": float(len(start)),
                "comp_cost": comp_cost,
            }
        except Exception:
            return None

    def greedy_vehicle(selected, time_budget=1.0):
        selected = sorted(selected, key=lambda i: (dep[i], ids[i]))
        vehicles = []
        start = {}
        end = {}
        comp = {}
        comp_cost = {}
        st = time.time()

        for idx, j in enumerate(selected):
            if time.time() - st > time_budget and idx > 20:
                for rem in selected[idx:]:
                    start[rem] = 1.0
                    vehicles.append(rem)
                break

            best_v = -1
            best_c = None
            best_last = None
            best_arr = -1e100

            for vi, last in enumerate(vehicles):
                c = pair_cost(last, j)
                if c is None:
                    continue
                if best_c is None or c < best_c - EPS or (abs(c - best_c) <= EPS and arr[last] > best_arr):
                    best_v = vi
                    best_c = c
                    best_last = last
                    best_arr = arr[last]

            if best_v >= 0:
                comp[(best_last, j)] = 1.0
                comp_cost[(best_last, j)] = best_c
                vehicles[best_v] = j
            else:
                start[j] = 1.0
                vehicles.append(j)

        for last in vehicles:
            end[last] = 1.0

        return {
            "start": start,
            "end": end,
            "comp": comp,
            "circ": float(len(vehicles)),
            "comp_cost": comp_cost,
        }

    def schedule_vehicles(selected, time_left, prefer_lp=True):
        selected = set(selected)
        if not selected:
            return {"start": {}, "end": {}, "comp": {}, "circ": 0.0, "comp_cost": {}}

        k = len(selected)
        if GUROBI_AVAILABLE and env is not None and prefer_lp and time_left > 4:
            if k <= 300 or k * (k - 1) // 2 <= 150000:
                cap = None
            else:
                cap = max(30, min(200, int(150000 // max(1, k))))
            budget = max(1.0, min(60.0, time_left - 2.0))
            flows = solve_vehicle_lp(list(selected), budget, cap)
            if flows is not None:
                return flows

        budget = max(0.2, min(10.0, time_left - 1.0))
        return greedy_vehicle(list(selected), budget)

    def compute_objective(paths, veh):
        tt = 0.0
        for _, path in paths.items():
            for a, b in zip(path, path[1:]):
                actual = (main_stop[b] - main_stop[a]) * 60.0
                ideal = get_tw(wins[a])["ideal"]
                tt += (actual - ideal) ** 2

        vc = per_vehicle * float(veh.get("circ", 0.0))
        for i, val in veh.get("start", {}).items():
            vc += pull_out_start[i] * float(val)
        for i, val in veh.get("end", {}).items():
            vc += pull_in_end[i] * float(val)

        comp_cost = veh.get("comp_cost", {})
        for (i, j), val in veh.get("comp", {}).items():
            c = comp_cost.get((i, j))
            if c is None:
                c = pair_cost(i, j)
                if c is None:
                    c = 0.0
            vc += c * float(val)

        return tt + alpha * vc

    def trip_node(i, kind):
        return f"({ids[i]}, '{kind}')"

    def build_solution(paths, veh, obj):
        selected = set()
        for path in paths.values():
            selected.update(path)

        vs_flows = {}
        circ_key = "VS_O_plus-->VS_O_minus"

        start_sum = sum(1 for i in selected if float(veh.get("start", {}).get(i, 0.0)) > 0.5)
        end_sum = sum(1 for i in selected if float(veh.get("end", {}).get(i, 0.0)) > 0.5)
        num = int(round(float(veh.get("circ", start_sum))))
        if num <= 0 and selected:
            num = max(1, start_sum, end_sum)
        if start_sum > 0 and num != start_sum:
            num = start_sum

        vs_flows[circ_key] = float(num)

        for i in selected:
            vs_flows[f"{trip_node(i, 'start')}-->{trip_node(i, 'end')}"] = 1.0

        for i, val in veh.get("start", {}).items():
            if i in selected and float(val) > 0.5:
                vs_flows[f"VS_O_minus-->{trip_node(i, 'start')}"] = 1.0

        for i, val in veh.get("end", {}).items():
            if i in selected and float(val) > 0.5:
                vs_flows[f"{trip_node(i, 'end')}-->VS_O_plus"] = 1.0

        for (i, j), val in veh.get("comp", {}).items():
            if i in selected and j in selected and float(val) > 0.5:
                vs_flows[f"{trip_node(i, 'end')}-->{trip_node(j, 'start')}"] = 1.0

        tt_arcs_used = {}
        for p in patterns:
            path = paths.get(p, [])
            arcs = []
            if path:
                arcs.append(["source", str(ids[path[0]])])
                for a, b in zip(path, path[1:]):
                    arcs.append([str(ids[a]), str(ids[b])])
                arcs.append([str(ids[path[-1]]), "sink"])
            tt_arcs_used[str(p)] = arcs

        selected_ids = sorted(int(ids[i]) for i in selected)

        return {
            "objective_value": float(obj),
            "selected_trips": selected_ids,
            "num_vehicles": int(num),
            "tt_arcs_used": tt_arcs_used,
            "vs_flows": vs_flows,
        }

    best_obj = float("inf")
    best_sol = None
    best_internal = None

    def consider(paths, veh):
        nonlocal best_obj, best_sol, best_internal
        obj = compute_objective(paths, veh)
        sol = build_solution(paths, veh, obj)
        if obj < best_obj - 1e-6:
            best_obj = obj
            best_sol = sol
            best_internal = (paths, veh)
            if logger is not None:
                try:
                    logger.log_solution(obj, sol)
                except Exception:
                    pass

    selected_heuristic = set()
    for path in heuristic_paths.values():
        selected_heuristic.update(path)

    time_left = time_limit - (time.time() - start_time)
    veh_heuristic = schedule_vehicles(selected_heuristic, max(1.0, time_left - 1.0), prefer_lp=time_left > 8)
    consider(heuristic_paths, veh_heuristic)

    def run_mip(time_left):
        if env is None or not active_patterns or not dp_ok:
            return None

        for p in active_patterns:
            if not source_arcs.get(p) or not sink_arcs.get(p):
                return None

        remaining_var_budget = 300000 - len(tt_arcs) - 3 * N
        if remaining_var_budget < 10000:
            return None

        comp_cap = max(20, min(200, int(remaining_var_budget // max(1, N))))
        gen_start = time.time()
        gen_budget = min(30.0, max(5.0, time_left * 0.3))
        comp_arcs, comp_key = generate_comp_arcs(list(range(N)), cap=comp_cap, time_budget=gen_budget)
        time_left -= (time.time() - gen_start)

        if time_left < 8:
            return None

        total_vars = N + len(tt_arcs) + len(comp_arcs) + 2 * N + 1
        if total_vars > 400000:
            return None

        try:
            m = gp.Model(env=env)
            m.Params.OutputFlag = 0
            m.Params.Seed = 0
            m.Params.Threads = 1
            m.Params.NumericFocus = 0
            m.Params.MIPGap = 1e-4
            m.Params.TimeLimit = max(1.0, time_left - 2.0)

            x = m.addVars(N, vtype=GRB.BINARY, name="x")
            y = m.addVars(len(tt_arcs), lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y")
            v_comp = m.addVars(len(comp_arcs), lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="v_comp")
            v_start = m.addVars(N, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="v_start")
            v_end = m.addVars(N, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="v_end")
            v_circ = m.addVar(lb=0.0, ub=float(N), vtype=GRB.CONTINUOUS, name="v_circ")

            m.addConstrs(
                (gp.quicksum(y[k] for k in tt_in[i]) == x[i] for i in range(N)),
                name="tt_in"
            )
            m.addConstrs(
                (gp.quicksum(y[k] for k in tt_out[i]) == x[i] for i in range(N)),
                name="tt_out"
            )

            for p in active_patterns:
                m.addConstr(gp.quicksum(y[k] for k in source_arcs[p]) == 1.0)
                m.addConstr(gp.quicksum(y[k] for k in sink_arcs[p]) == 1.0)

            pred = defaultdict(list)
            succ = defaultdict(list)
            for k, (i, j, _) in enumerate(comp_arcs):
                pred[j].append(k)
                succ[i].append(k)

            m.addConstrs(
                (v_start[j] + gp.quicksum(v_comp[k] for k in pred[j]) == x[j] for j in range(N)),
                name="veh_start"
            )
            m.addConstrs(
                (v_end[i] + gp.quicksum(v_comp[k] for k in succ[i]) == x[i] for i in range(N)),
                name="veh_end"
            )
            m.addConstr(v_circ == gp.quicksum(v_start[i] for i in range(N)), name="veh_circ_start")
            m.addConstr(v_circ == gp.quicksum(v_end[i] for i in range(N)), name="veh_circ_end")

            tt_cost = [arc[2] for arc in tt_arcs]
            comp_cost = [arc[2] for arc in comp_arcs]

            obj = gp.quicksum(tt_cost[k] * y[k] for k in range(len(tt_arcs)))
            veh_obj = (
                per_vehicle * v_circ
                + gp.quicksum(pull_out_start[i] * v_start[i] for i in range(N))
                + gp.quicksum(pull_in_end[i] * v_end[i] for i in range(N))
                + gp.quicksum(comp_cost[k] * v_comp[k] for k in range(len(comp_arcs)))
            )

            if alpha > 0:
                obj += alpha * veh_obj
            else:
                obj += 1e-8 * v_circ

            m.setObjective(obj, GRB.MINIMIZE)

            if best_internal is not None:
                try:
                    b_paths, _ = best_internal
                    selected_set = set()
                    for path in b_paths.values():
                        selected_set.update(path)

                    for i in range(N):
                        x[i].Start = 1.0 if i in selected_set else 0.0

                    for p, path in b_paths.items():
                        if not path:
                            continue
                        k = tt_key.get((p, SOURCE, path[0]))
                        if k is not None:
                            y[k].Start = 1.0
                        for a, b in zip(path, path[1:]):
                            k = tt_key.get((p, a, b))
                            if k is not None:
                                y[k].Start = 1.0
                        k = tt_key.get((p, path[-1], SINK))
                        if k is not None:
                            y[k].Start = 1.0
                except Exception:
                    pass

            last_cb_obj = [float("inf")]

            def cb(model, where):
                if where == GRB.Callback.MIPSOL:
                    try:
                        o = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                        if o < last_cb_obj[0] - 1e-6:
                            last_cb_obj[0] = o
                            if logger is not None:
                                logger.log(float(o))
                    except Exception:
                        pass

            if logger is not None:
                m.optimize(cb)
            else:
                m.optimize()

            if m.SolCount == 0:
                return None

            selected_x = set()
            for i in range(N):
                try:
                    if x[i].X > 0.5:
                        selected_x.add(i)
                except Exception:
                    pass

            paths = {}
            for p in active_patterns:
                path = []
                best_k = None
                best_val = -1.0

                for k in source_arcs.get(p, []):
                    try:
                        val = y[k].X
                    except Exception:
                        val = 0.0
                    if val > best_val:
                        best_val = val
                        best_k = k

                if best_k is not None and best_val > 0.2:
                    cur = tt_arcs[best_k][1]
                    path = [cur]
                    seen = {cur}

                    while True:
                        best_out = None
                        bv = -1.0
                        for k in tt_out.get(cur, []):
                            try:
                                val = y[k].X
                            except Exception:
                                val = 0.0
                            if val > bv:
                                bv = val
                                best_out = k

                        if best_out is None or bv < 0.2:
                            break

                        v = tt_arcs[best_out][1]
                        if v == SINK:
                            break
                        if v in seen:
                            break

                        path.append(v)
                        seen.add(v)
                        if len(path) > len(pattern_trips[p]) + 3:
                            break

                if (not path) or path[0] not in first_sets[p] or path[-1] not in final_sets[p]:
                    allowed = selected_x.intersection(set(pattern_trips[p]))
                    alt = dp_path(p, allowed)
                    if alt:
                        path = alt

                if (not path) or path[-1] not in final_sets[p]:
                    alt = dp_path(p)
                    if alt:
                        path = alt

                if not path:
                    path = heuristic_paths.get(p, [])

                paths[p] = path

            return paths
        except Exception:
            return None

    time_left = time_limit - (time.time() - start_time)
    if GUROBI_AVAILABLE and env is not None and time_left > 15:
        mip_paths = run_mip(time_left - 3.0)
        if mip_paths:
            selected_mip = set()
            for path in mip_paths.values():
                selected_mip.update(path)
            time_left = time_limit - (time.time() - start_time)
            veh_mip = schedule_vehicles(selected_mip, max(1.0, time_left - 1.0), prefer_lp=time_left > 8)
            consider(mip_paths, veh_mip)

    if best_sol is None:
        if selected_heuristic:
            veh_last = greedy_vehicle(list(selected_heuristic), 0.5)
            consider(heuristic_paths, veh_last)
        else:
            empty_paths = {p: [] for p in patterns}
            empty_veh = {"start": {}, "end": {}, "comp": {}, "circ": 0.0, "comp_cost": {}}
            consider(empty_paths, empty_veh)

    if best_sol is None:
        best_sol = build_solution(
            {p: [] for p in patterns},
            {"start": {}, "end": {}, "comp": {}, "circ": 0.0, "comp_cost": {}},
            0.0
        )

    try:
        os.makedirs(os.path.dirname(os.path.abspath(args.solution_path)), exist_ok=True)
    except Exception:
        pass

    with open(args.solution_path, "w") as f:
        json.dump(best_sol, f)


if __name__ == "__main__":
    main()