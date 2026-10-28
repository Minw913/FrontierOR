#!/usr/bin/env python3
import argparse
import json
import time
import heapq
import sys
from collections import defaultdict

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except Exception:
    GUROBI_AVAILABLE = False

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


class Option:
    __slots__ = ("p", "h", "start", "end", "penalty", "cap_list", "lo_idx", "hi_idx", "key")

    def __init__(self, p, h, start, end, penalty, cap_list, lo_idx, hi_idx, key=0.0):
        self.p = p
        self.h = h
        self.start = start
        self.end = end
        self.penalty = penalty
        self.cap_list = cap_list
        self.lo_idx = lo_idx
        self.hi_idx = hi_idx
        self.key = key


class Vessel:
    __slots__ = (
        "id", "load", "unload", "r", "pen", "fs", "fe", "es", "ee",
        "profiles", "unload_cost", "load_est", "berth_cost", "berth_order", "options"
    )

    def __init__(self):
        self.options = []


class FlowEdge:
    __slots__ = ("to", "rev", "cap", "cost")

    def __init__(self, to, rev, cap, cost):
        self.to = to
        self.rev = rev
        self.cap = cap
        self.cost = cost


class Schedule:
    def __init__(self, num_berths, H, base, cap):
        self.B = num_berths
        self.H = H
        self.base = base
        self.cap = cap
        self.cons_usage = [0.0] * H
        self.intervals = [[] for _ in range(num_berths)]
        self.min_start = [None] * num_berths
        self.max_end = [None] * num_berths
        self.assigned = {}

    def feasible(self, i, opt, b, allow_qc=False):
        if b < 0 or b >= self.B:
            return False
        s, e = opt.start, opt.end

        for ss, ee, j in self.intervals[b]:
            if not (e < ss or s > ee):
                return False

        if self.min_start[b] is not None:
            ns = min(self.min_start[b], s)
            ne = max(self.max_end[b], e)
            if ne - ns > self.H - 1 + 1e-9:
                return False

        if not allow_qc:
            for cidx, d in opt.cap_list:
                if self.cons_usage[cidx] + d > self.cap[cidx] + 1e-6:
                    return False
        return True

    def add(self, i, opt_idx, opt, b):
        for cidx, d in opt.cap_list:
            self.cons_usage[cidx] += d
        self.intervals[b].append((opt.start, opt.end, i))
        if self.min_start[b] is None or opt.start < self.min_start[b]:
            self.min_start[b] = opt.start
        if self.max_end[b] is None or opt.end > self.max_end[b]:
            self.max_end[b] = opt.end
        self.assigned[i] = (opt_idx, b, opt)

    def remove(self, i):
        opt_idx, b, opt = self.assigned[i]
        for cidx, d in opt.cap_list:
            self.cons_usage[cidx] -= d
        self.intervals[b] = [x for x in self.intervals[b] if x[2] != i]
        if self.intervals[b]:
            self.min_start[b] = min(x[0] for x in self.intervals[b])
            self.max_end[b] = max(x[1] for x in self.intervals[b])
        else:
            self.min_start[b] = None
            self.max_end[b] = None
        del self.assigned[i]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.time()
    time_limit = max(0, int(args.time_limit))

    logger = None
    if args.log_path and SolutionLogger is not None:
        try:
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    V = int(inst.get("num_vessels", 0))
    B = int(inst.get("num_berths", 0))
    num_QCs = int(inst.get("num_QCs", 0))
    S = int(inst.get("num_subblocks", 0))
    H = int(inst.get("H", 0))
    E = int(inst.get("E", 0))
    T = H + E
    co = float(inst.get("penalty_cost_coefficient_co", 0.0))

    if V == 0:
        sol = {
            "objective_value": 0.0,
            "vessels": [],
            "berths": [{"berth": b, "rho": 0, "varsigma": 0} for b in range(B)],
            "qc_usage": {}
        }
        if logger:
            try:
                logger.log_solution(0.0, sol)
            except Exception:
                pass
        with open(args.solution_path, "w") as f:
            json.dump(sol, f, indent=2)
        return

    if H <= 0:
        H = 1
    if B <= 0:
        B = 1
    T = H + E

    D_L = inst.get("loading_route_lengths_D_L", [[0.0] * B for _ in range(S)])
    D_U = inst.get("average_unloading_route_lengths_D_U", [0.0] * B)

    all_window_vals = []
    for vd in inst.get("vessels", []):
        all_window_vals.extend(vd.get("feasible_time_window", [1, 1]))
        all_window_vals.extend(vd.get("expected_time_window", [1, 1]))

    if all_window_vals:
        min_time_val = min(all_window_vals)
        max_time_val = max(all_window_vals)
        if min_time_val == 0 and max_time_val <= max(0, T - 1):
            base = 0
        else:
            base = 1
    else:
        base = 1

    raw_cap = inst.get("QC_capacity_per_timestep", [])
    cap = []
    for idx in range(H):
        cv = raw_cap[idx] if idx < len(raw_cap) else num_QCs
        if num_QCs > 0:
            cv = min(cv, num_QCs)
        cap.append(float(cv))

    berth_prefix = []
    for b in range(B):
        vals = []
        for s in range(S):
            try:
                vals.append(float(D_L[s][b]))
            except Exception:
                vals.append(0.0)
        vals.sort()
        pref = [0.0]
        for x in vals:
            pref.append(pref[-1] + x)
        berth_prefix.append(pref)

    vessels = []
    for idx, vd in enumerate(inst.get("vessels", [])):
        v = Vessel()
        v.id = vd.get("vessel_id", idx)
        v.load = int(vd.get("loading_containers", 0))
        v.unload = int(vd.get("unloading_containers", 0))
        v.r = int(vd.get("reserved_subblocks", 0))
        v.pen = float(vd.get("penalty_cost_coefficient", 0.0))
        fw = vd.get("feasible_time_window", [base, base + H - 1])
        ew = vd.get("expected_time_window", fw)
        v.fs = int(fw[0])
        v.fe = int(fw[1])
        v.es = int(ew[0])
        v.ee = int(ew[1])

        profiles = []
        for prof in vd.get("qc_profiles", []):
            h = int(prof.get("handling_time", 1))
            if h <= 0:
                h = 1
            qc = [int(x) for x in prof.get("qc_per_timestep", [])]
            if len(qc) < h:
                qc += [0] * (h - len(qc))
            elif len(qc) > h:
                qc = qc[:h]
            profiles.append({"handling_time": h, "qc_per_timestep": qc})
        if not profiles:
            profiles = [{"handling_time": 1, "qc_per_timestep": [0]}]
        v.profiles = profiles

        v.unload_cost = []
        for b in range(B):
            du = float(D_U[b]) if b < len(D_U) else 0.0
            v.unload_cost.append(co * du * float(v.unload))

        v.load_est = []
        for b in range(B):
            if v.r > 0 and v.load > 0 and S > 0:
                take = min(v.r, S)
                dist_sum = berth_prefix[b][take] if take < len(berth_prefix[b]) else berth_prefix[b][-1]
                v.load_est.append(co * float(v.load) / float(v.r) * dist_sum)
            else:
                v.load_est.append(0.0)

        v.berth_cost = [v.unload_cost[b] + v.load_est[b] for b in range(B)]
        v.berth_order = sorted(range(B), key=lambda x: v.berth_cost[x])
        vessels.append(v)

    def elapsed():
        return time.time() - start_time

    def remaining():
        return max(0.0, float(time_limit) - elapsed())

    def tau_to_cidx(tau):
        if tau < base or tau >= base + T:
            return None
        if tau >= base + H:
            c = tau - H
        else:
            c = tau
        if base <= c < base + H:
            return c - base
        return None

    # Option generation caps, chosen to keep MILP tractable on one core.
    if time_limit <= 5:
        max_x_vars = 0
    elif time_limit <= 20:
        max_x_vars = 15000
    elif time_limit <= 60:
        max_x_vars = 40000
    elif time_limit <= 180:
        max_x_vars = 80000
    else:
        max_x_vars = 120000

    if max_x_vars > 0:
        total_options_cap = max(V, max_x_vars // max(1, B))
    else:
        total_options_cap = max(V, 20000 // max(1, B))
    per_vessel_cap = max(10, total_options_cap // max(1, V))
    per_vessel_cap = min(per_vessel_cap, 2000)

    for i, v in enumerate(vessels):
        opts = []
        min_berth_cost = min(v.berth_cost) if v.berth_cost else 0.0

        for p_idx, prof in enumerate(v.profiles):
            h = prof["handling_time"]
            qc = prof["qc_per_timestep"]

            earliest = max(v.fs, base)
            latest_first = min(v.fe - h + 1, base + H - 1)
            if latest_first >= earliest:
                start_range = range(earliest, latest_first + 1)
            else:
                latest_full = min(v.fe - h + 1, base + T - 1)
                if latest_full >= earliest:
                    start_range = range(earliest, latest_full + 1)
                else:
                    start_range = range(0, -1)

            for t in start_range:
                e = t + h - 1
                if e > base + T - 1:
                    continue

                penalty = v.pen * (max(0, v.es - t) + max(0, e - v.ee))

                deltas = {}
                feasible_cap = True
                for off, q in enumerate(qc):
                    if q == 0:
                        continue
                    tau = t + off
                    cidx = tau_to_cidx(tau)
                    if cidx is None:
                        feasible_cap = False
                        break
                    deltas[cidx] = deltas.get(cidx, 0) + int(q)
                if not feasible_cap:
                    continue

                if any(deltas[c] > cap[c] + 1e-9 for c in deltas):
                    continue

                lo = max(base, e - H + 1)
                hi = min(t, base + H - 1)
                if lo > hi:
                    continue
                lo_idx = lo - base
                hi_idx = hi - base

                cap_list = tuple((c, deltas[c]) for c in sorted(deltas))
                key = penalty + min_berth_cost
                opts.append(Option(p_idx, h, t, e, penalty, cap_list, lo_idx, hi_idx, key))

        if not opts:
            prof = v.profiles[0]
            h = prof["handling_time"]
            qc = prof["qc_per_timestep"]
            t = max(base, min(v.fs, base + T - h))
            if t < base:
                t = base
            if t + h - 1 > base + T - 1:
                t = max(base, base + T - h)
            e = t + h - 1
            penalty = v.pen * (max(0, v.es - t) + max(0, e - v.ee))
            deltas = {}
            for off, q in enumerate(qc):
                if q == 0:
                    continue
                cidx = tau_to_cidx(t + off)
                if cidx is not None:
                    deltas[cidx] = deltas.get(cidx, 0) + int(q)
            cap_list = tuple((c, deltas[c]) for c in sorted(deltas))
            lo = max(base, e - H + 1)
            hi = min(t, base + H - 1)
            if lo > hi:
                lo, hi = base, base + H - 1
            opts.append(Option(0, h, t, e, penalty, cap_list, lo - base, hi - base, penalty + min_berth_cost))

        opts.sort(key=lambda x: x.key)
        if len(opts) > per_vessel_cap:
            opts = opts[:per_vessel_cap]
        v.options = opts

    best = {"obj": float("inf"), "sol": None, "assignments": None}

    def consider(sol, obj, assignments):
        if obj is None:
            return
        obj = float(obj)
        if obj < best["obj"] - 1e-9:
            best["obj"] = obj
            best["sol"] = sol
            best["assignments"] = assignments
            if logger:
                try:
                    logger.log_solution(obj, sol)
                except Exception:
                    pass

    def assign_subblocks(berth_of):
        assignments = [[] for _ in range(V)]
        used = [False] * S

        active = [i for i in range(V) if vessels[i].r > 0 and vessels[i].load > 0 and berth_of[i] is not None]
        zero = [i for i in range(V) if vessels[i].r > 0 and vessels[i].load == 0]

        def flow_assign(active_list):
            A = len(active_list)
            if A == 0 or S == 0:
                return
            maxf = sum(vessels[i].r for i in active_list)
            if maxf <= 0:
                return

            N = 2 + S + A
            source = 0
            sink = N - 1
            graph = [[] for _ in range(N)]

            def add_edge(fr, to, cap_, cost_):
                f = FlowEdge(to, len(graph[to]), cap_, cost_)
                r = FlowEdge(fr, len(graph[fr]), 0, -cost_)
                graph[fr].append(f)
                graph[to].append(r)

            for s in range(S):
                add_edge(source, 1 + s, 1, 0.0)

            factors = {}
            for i in active_list:
                vv = vessels[i]
                factors[i] = co * float(vv.load) / float(vv.r) if vv.r > 0 else 0.0

            for s in range(S):
                sn = 1 + s
                for ai, i in enumerate(active_list):
                    b = berth_of[i]
                    try:
                        dist = float(D_L[s][b])
                    except Exception:
                        dist = 0.0
                    cost = factors[i] * dist
                    add_edge(sn, 1 + S + ai, 1, cost)

            for ai, i in enumerate(active_list):
                add_edge(1 + S + ai, sink, vessels[i].r, 0.0)

            potential = [0.0] * N
            flow = 0
            INF = float("inf")

            while flow < maxf:
                dist = [INF] * N
                prevv = [-1] * N
                preve = [-1] * N
                dist[source] = 0.0
                pq = [(0.0, source)]

                while pq:
                    d, node = heapq.heappop(pq)
                    if d > dist[node] + 1e-12:
                        continue
                    for ei, edge in enumerate(graph[node]):
                        if edge.cap <= 0:
                            continue
                        rc = edge.cost + potential[node] - potential[edge.to]
                        if rc < 0.0:
                            rc = 0.0
                        nd = d + rc
                        if nd < dist[edge.to] - 1e-12:
                            dist[edge.to] = nd
                            prevv[edge.to] = node
                            preve[edge.to] = ei
                            heapq.heappush(pq, (nd, edge.to))

                if dist[sink] >= INF / 2:
                    break

                for n in range(N):
                    if dist[n] < INF / 2:
                        potential[n] += dist[n]

                addf = maxf - flow
                node = sink
                while node != source:
                    pv = prevv[node]
                    if pv < 0:
                        addf = 0
                        break
                    edge = graph[pv][preve[node]]
                    if edge.cap < addf:
                        addf = edge.cap
                    node = pv
                if addf <= 0:
                    break

                path_cost = 0.0
                node = sink
                while node != source:
                    pv = prevv[node]
                    edge = graph[pv][preve[node]]
                    path_cost += edge.cost
                    edge.cap -= addf
                    graph[node][edge.rev].cap += addf
                    node = pv

                flow += addf

            for s in range(S):
                sn = 1 + s
                for edge in graph[sn]:
                    if 1 + S <= edge.to < 1 + S + A and edge.cap <= 0.5:
                        ai = edge.to - (1 + S)
                        i = active_list[ai]
                        assignments[i].append(s)
                        used[s] = True

        def greedy_active(active_list):
            if not active_list or S == 0:
                return
            rem = {i: vessels[i].r for i in active_list}
            total = sum(rem.values())
            if total <= 0:
                return

            edges = len(active_list) * S
            if edges <= 200000:
                pairs = []
                for i in active_list:
                    vv = vessels[i]
                    b = berth_of[i]
                    factor = co * float(vv.load) / float(vv.r) if vv.r > 0 else 0.0
                    for s in range(S):
                        try:
                            dist = float(D_L[s][b])
                        except Exception:
                            dist = 0.0
                        pairs.append((factor * dist, s, i))
                pairs.sort(key=lambda x: x[0])
                for _, s, i in pairs:
                    if total <= 0:
                        break
                    if not used[s] and rem.get(i, 0) > 0:
                        assignments[i].append(s)
                        used[s] = True
                        rem[i] -= 1
                        total -= 1
            else:
                factors = {}
                berths = {}
                for i in active_list:
                    vv = vessels[i]
                    factors[i] = co * float(vv.load) / float(vv.r) if vv.r > 0 else 0.0
                    berths[i] = berth_of[i]
                for s in range(S):
                    if total <= 0:
                        break
                    if used[s]:
                        continue
                    best_i = None
                    best_cost = float("inf")
                    for i in active_list:
                        if rem.get(i, 0) <= 0:
                            continue
                        try:
                            dist = float(D_L[s][berths[i]])
                        except Exception:
                            dist = 0.0
                        c = factors[i] * dist
                        if c < best_cost - 1e-12:
                            best_cost = c
                            best_i = i
                    if best_i is not None:
                        assignments[best_i].append(s)
                        used[s] = True
                        rem[best_i] -= 1
                        total -= 1

        active_demand = sum(vessels[i].r for i in active)
        if active:
            use_flow = (
                active_demand <= S
                and active_demand <= 500
                and len(active) * S <= 80000
                and remaining() > 3.0
            )
            if use_flow:
                try:
                    flow_assign(active)
                except Exception:
                    greedy_active(active)
            else:
                greedy_active(active)

        free = [s for s in range(S) if not used[s]]
        fp = 0
        for i in zero:
            need = vessels[i].r
            while need > 0 and fp < len(free):
                assignments[i].append(free[fp])
                used[free[fp]] = True
                fp += 1
                need -= 1

        for i in range(V):
            r = vessels[i].r
            if r <= 0:
                assignments[i] = []
                continue
            if len(assignments[i]) < r:
                b = berth_of[i] if berth_of[i] is not None else 0
                if S > 0:
                    order = sorted(range(S), key=lambda s: float(D_L[s][b]) if b < B else 0.0)
                    while len(assignments[i]) < r:
                        assignments[i].append(order[len(assignments[i]) % S])
                else:
                    assignments[i].extend([0] * (r - len(assignments[i])))
            elif len(assignments[i]) > r:
                assignments[i] = assignments[i][:r]

        return assignments

    def build_solution_from_assignments(assignments_in):
        assignments = []
        berth_of = []
        for i, v in enumerate(vessels):
            if i < len(assignments_in) and assignments_in[i] is not None:
                k, b = assignments_in[i]
                if not v.options:
                    k = 0
                else:
                    k = max(0, min(int(k), len(v.options) - 1))
                b = max(0, min(int(b), B - 1))
                assignments_in[i] = (k, b)
                berth_of.append(b)
            else:
                if v.options:
                    k = 0
                    b = v.berth_order[0] if v.berth_order else 0
                else:
                    k = 0
                    b = 0
                assignments_in[i] = (k, b)
                berth_of.append(b)

        subblocks = assign_subblocks(berth_of)
        assignments = subblocks
        qc_usage = [0.0] * H
        min_start = [None] * B
        max_end = [None] * B
        obj = 0.0
        vessel_entries = []

        for i, v in enumerate(vessels):
            k, b = assignments_in[i]
            if not v.options:
                vessel_entries.append({
                    "vessel_id": v.id,
                    "berth": b,
                    "start_time": base,
                    "end_time": base,
                    "start": base,
                    "end": base,
                    "rho": base,
                    "varsigma": base,
                    "subblocks": []
                })
                continue

            opt = v.options[k]
            obj += opt.penalty
            obj += v.unload_cost[b]

            if v.r > 0 and v.load > 0 and subblocks[i]:
                factor = co * float(v.load) / float(v.r)
                for s in subblocks[i]:
                    if 0 <= s < S and b < B:
                        try:
                            obj += factor * float(D_L[s][b])
                        except Exception:
                            pass

            for cidx, d in opt.cap_list:
                if 0 <= cidx < H:
                    qc_usage[cidx] += float(d)

            if min_start[b] is None or opt.start < min_start[b]:
                min_start[b] = opt.start
            if max_end[b] is None or opt.end > max_end[b]:
                max_end[b] = opt.end

            vessel_entries.append({
                "vessel_id": v.id,
                "berth": b,
                "start_time": opt.start,
                "end_time": opt.end,
                "start": opt.start,
                "end": opt.end,
                "rho": opt.start,
                "varsigma": opt.end,
                "handling_time": opt.h,
                "qc_profile": opt.p,
                "subblocks": subblocks[i]
            })

        berth_entries = []
        for b in range(B):
            rho = min_start[b] if min_start[b] is not None else 0
            varsigma = max_end[b] if max_end[b] is not None else 0
            berth_entries.append({"berth": b, "rho": rho, "varsigma": varsigma})

        sol = {
            "objective_value": float(obj),
            "vessels": vessel_entries,
            "berths": berth_entries,
            "qc_usage": {str(t): qc_usage[t] for t in range(H)}
        }
        return obj, sol, assignments_in

    def safe_consider(assignments_in):
        try:
            obj, sol, out_assignments = build_solution_from_assignments(assignments_in)
            consider(sol, obj, out_assignments)
            return out_assignments
        except Exception:
            return None

    def greedy_initial(order_key=None):
        sched = Schedule(B, H, base, cap)
        if order_key is None:
            order = sorted(range(V), key=lambda i: (min((o.key for o in vessels[i].options), default=0.0), i))
        else:
            order = sorted(range(V), key=order_key)

        for i in order:
            v = vessels[i]
            if not v.options:
                continue

            placed = False
            if V > 500:
                candidate_count = min(len(v.options), 80)
            elif remaining() < 10.0:
                candidate_count = min(len(v.options), 200)
            else:
                candidate_count = min(len(v.options), 800)

            for opt_idx in range(candidate_count):
                opt = v.options[opt_idx]
                for b in v.berth_order:
                    if sched.feasible(i, opt, b):
                        sched.add(i, opt_idx, opt, b)
                        placed = True
                        break
                if placed:
                    break

            if not placed:
                for opt_idx in range(candidate_count):
                    opt = v.options[opt_idx]
                    for b in v.berth_order:
                        if sched.feasible(i, opt, b, allow_qc=True):
                            sched.add(i, opt_idx, opt, b)
                            placed = True
                            break
                    if placed:
                        break

            if not placed:
                best_b = min(range(B), key=lambda b: (len(sched.intervals[b]), b))

                def overlap_score(k):
                    opt = v.options[k]
                    cnt = 0
                    span = 0
                    if sched.min_start[best_b] is not None:
                        ns = min(sched.min_start[best_b], opt.start)
                        ne = max(sched.max_end[best_b], opt.end)
                        span = max(0, ne - ns - (H - 1))
                    for ss, ee, _ in sched.intervals[best_b]:
                        if not (opt.end < ss or opt.start > ee):
                            cnt += 1
                    return (cnt, span, opt.key)

                opt_idx = min(range(len(v.options)), key=overlap_score)
                sched.add(i, opt_idx, v.options[opt_idx], best_b)

        out = [None] * V
        for i in range(V):
            if i in sched.assigned:
                opt_idx, b, _ = sched.assigned[i]
                out[i] = (opt_idx, b)
            else:
                out[i] = (0, vessels[i].berth_order[0] if vessels[i].berth_order else 0)
        return out

    def local_search_once(assignments_in):
        if assignments_in is None:
            return None

        sched = Schedule(B, H, base, cap)
        for i in range(V):
            if i >= len(assignments_in) or assignments_in[i] is None:
                continue
            k, b = assignments_in[i]
            v = vessels[i]
            if not v.options:
                continue
            k = max(0, min(int(k), len(v.options) - 1))
            b = max(0, min(int(b), B - 1))
            sched.add(i, k, v.options[k], b)

        improved = False
        order = sorted(range(V), key=lambda i: vessels[i].pen, reverse=True)

        if V > 200:
            scan_limit = 80
        elif remaining() < 10.0:
            scan_limit = 200
        else:
            scan_limit = 500

        for i in order:
            v = vessels[i]
            if not v.options or i not in sched.assigned:
                continue

            old_k, old_b, old_opt = sched.assigned[i]
            old_cost = old_opt.penalty + v.berth_cost[old_b]
            sched.remove(i)

            best_choice = None
            best_cost = old_cost - 1e-9
            limit = min(len(v.options), scan_limit)

            for k in range(limit):
                opt = v.options[k]
                for b in v.berth_order:
                    if sched.feasible(i, opt, b):
                        cost = opt.penalty + v.berth_cost[b]
                        if cost < best_cost:
                            best_cost = cost
                            best_choice = (k, b, opt)

            if best_choice is not None:
                sched.add(i, best_choice[0], best_choice[2], best_choice[1])
                improved = True
            else:
                sched.add(i, old_k, old_opt, old_b)

        if not improved:
            return None

        out = [None] * V
        for i in range(V):
            if i in sched.assigned:
                k, b, _ = sched.assigned[i]
                out[i] = (k, b)
            else:
                out[i] = assignments_in[i]
        return out

    def solve_with_gurobi():
        if not GUROBI_AVAILABLE or max_x_vars <= 0:
            return None

        rem = remaining()
        if rem < 3.0:
            return None

        occupancy_entries = 0
        for v in vessels:
            for opt in v.options:
                occupancy_entries += B * max(1, opt.end - opt.start + 1)

        if occupancy_entries > 2000000 or T * B > 200000:
            return None

        env = None
        m = None
        try:
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.setParam("Threads", 1)
            env.start()

            m = gp.Model(env=env)
            m.setParam("TimeLimit", max(0.5, rem - 1.0))
            m.setParam("MIPGap", 0.05)
            m.modelSense = GRB.MINIMIZE

            x = {}
            for i, v in enumerate(vessels):
                for k, opt in enumerate(v.options):
                    for b in range(B):
                        x[(i, k, b)] = m.addVar(
                            vtype=GRB.BINARY,
                            obj=opt.penalty + v.berth_cost[b]
                        )
            m.update()

            for i, v in enumerate(vessels):
                vars_i = [x[(i, k, b)] for k in range(len(v.options)) for b in range(B)]
                if vars_i:
                    m.addConstr(gp.quicksum(vars_i) == 1.0, name="assign_%d" % i)

            qc_expr = [gp.LinExpr() for _ in range(H)]
            for (i, k, b), var in x.items():
                opt = vessels[i].options[k]
                for cidx, d in opt.cap_list:
                    if 0 <= cidx < H:
                        qc_expr[cidx] += var * float(d)

            for cidx in range(H):
                m.addConstr(qc_expr[cidx] <= cap[cidx], name="qc_%d" % cidx)

            occ = defaultdict(list)
            for (i, k, b), var in x.items():
                opt = vessels[i].options[k]
                for t in range(opt.start, opt.end + 1):
                    occ[(b, t)].append(var)

            for (b, t), vars_bt in occ.items():
                if len(vars_bt) > 1:
                    m.addConstr(gp.quicksum(vars_bt) <= 1.0, name="berth_%d_%d" % (b, t))

            if len(x) <= 80000:
                rho = [m.addVar(lb=base, ub=base + T - 1, vtype=GRB.CONTINUOUS) for _ in range(B)]
                varsigma = [m.addVar(lb=base, ub=base + T - 1, vtype=GRB.CONTINUOUS) for _ in range(B)]
                m.update()

                M = T + 5
                for b in range(B):
                    for i, v in enumerate(vessels):
                        for k, opt in enumerate(v.options):
                            var = x[(i, k, b)]
                            m.addConstr(rho[b] <= opt.start + M * (1 - var))
                            m.addConstr(varsigma[b] >= opt.end - M * (1 - var))
                    m.addConstr(varsigma[b] - rho[b] <= max(0, H - 1))

            m.optimize()

            if m.SolCount > 0:
                assignments = [None] * V
                for (i, k, b), var in x.items():
                    try:
                        if var.X > 0.5:
                            assignments[i] = (k, b)
                    except Exception:
                        pass

                for i in range(V):
                    if assignments[i] is None:
                        assignments[i] = (0, vessels[i].berth_order[0] if vessels[i].berth_order else 0)
                return assignments

        except Exception:
            return None
        finally:
            try:
                if m is not None:
                    m.dispose()
            except Exception:
                pass
            try:
                if env is not None:
                    env.dispose()
            except Exception:
                pass

        return None

    try:
        safe_consider(greedy_initial())
    except Exception:
        pass

    if remaining() > 1.0:
        alt_keys = [
            lambda i: (vessels[i].fs, i),
            lambda i: (-vessels[i].pen, i),
            lambda i: (-(vessels[i].load + vessels[i].unload), i),
        ]
        for key in alt_keys:
            if remaining() <= 0.5:
                break
            try:
                safe_consider(greedy_initial(key))
            except Exception:
                pass

    if best["assignments"] is not None and V <= 300 and remaining() > 1.0:
        try:
            improved = local_search_once(best["assignments"])
            if improved is not None:
                safe_consider(improved)
        except Exception:
            pass

    if remaining() > 3.0:
        try:
            milp_assignments = solve_with_gurobi()
            if milp_assignments is not None:
                safe_consider(milp_assignments)
                if V <= 300 and remaining() > 1.0:
                    try:
                        improved = local_search_once(milp_assignments)
                        if improved is not None:
                            safe_consider(improved)
                    except Exception:
                        pass
        except Exception:
            pass

    if best["sol"] is None:
        fallback = [(0, vessels[i].berth_order[0] if vessels[i].berth_order else 0) for i in range(V)]
        safe_consider(fallback)

    if best["sol"] is None:
        vessel_entries = []
        for i, v in enumerate(vessels):
            vessel_entries.append({
                "vessel_id": v.id,
                "berth": 0,
                "start_time": base,
                "end_time": base,
                "start": base,
                "end": base,
                "rho": base,
                "varsigma": base,
                "subblocks": []
            })
        sol = {
            "objective_value": 0.0,
            "vessels": vessel_entries,
            "berths": [{"berth": b, "rho": 0, "varsigma": 0} for b in range(B)],
            "qc_usage": {}
        }
    else:
        sol = best["sol"]

    if logger:
        try:
            logger.log_solution(float(best["obj"]) if best["obj"] != float("inf") else 0.0, sol)
        except Exception:
            pass

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()