import argparse
import json
import time
from collections import deque

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    t0 = time.time()
    time_limit = float(args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # -------------------------- read instance --------------------------
    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    K = int(inst["num_resource_types"])
    T = int(inst["max_project_duration"])
    acts = inst["activities"]
    precs = [[int(a), int(b)] for a, b in inst["precedence_relations"]]
    costs = [float(c) for c in inst["resource_costs"]]

    durations = {}
    reqs = {}
    ids = []
    for a in acts:
        i = int(a["id"])
        ids.append(i)
        durations[i] = int(a["duration"])
        reqs[i] = [int(v) for v in a["resource_requirements"]]

    preds = {i: [] for i in ids}
    succs = {i: [] for i in ids}
    for i, j in precs:
        preds[j].append(i)
        succs[i].append(j)

    # Identify fictitious start / end activities
    fict = [i for i in ids if durations[i] == 0 and all(v == 0 for v in reqs[i])]
    start_id = None
    end_id = None
    for i in fict:
        if len(preds[i]) == 0:
            start_id = i
        if len(succs[i]) == 0:
            end_id = i
    if start_id is None and fict:
        start_id = fict[0]
    if end_id is None and fict:
        end_id = fict[-1]

    # Topological order
    indeg = {i: len(preds[i]) for i in ids}
    q = deque([i for i in ids if indeg[i] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    # Earliest start times (longest paths from project start)
    ES = {i: 0 for i in ids}
    for u in topo:
        base = ES[u] + durations[u]
        for v in succs[u]:
            if base > ES[v]:
                ES[v] = base

    # Latest start times (backward pass from deadline)
    LS = {i: T for i in ids}
    if end_id is not None:
        LS[end_id] = T
    for u in reversed(topo):
        du = durations[u]
        best = LS[u]
        for v in succs[u]:
            val = LS[v] - du
            if val < best:
                best = val
        LS[u] = best
    for i in ids:
        if LS[i] < ES[i]:
            LS[i] = ES[i]

    real_ids = [i for i in ids if durations[i] > 0]

    # -------------------------- helpers --------------------------
    def compute_peaks(st):
        diff = [[0] * (T + 2) for _ in range(K)]
        for i in real_ids:
            s = st[i]
            d = durations[i]
            r = reqs[i]
            for k in range(K):
                rk = r[k]
                if rk:
                    diff[k][s] += rk
                    e = s + d
                    diff[k][min(e, T + 1)] -= rk
        peaks = [0] * K
        for k in range(K):
            cur = 0
            pk = 0
            dk = diff[k]
            for t in range(T + 1):
                cur += dk[t]
                if cur > pk:
                    pk = cur
            peaks[k] = pk
        return peaks

    def objective_of(peaks):
        return sum(costs[k] * peaks[k] for k in range(K))

    def build_solution(st):
        peaks = compute_peaks(st)
        obj = float(objective_of(peaks))
        sol = {
            "objective_value": obj,
            "start_times": {str(i): int(st[i]) for i in ids},
            "resource_amounts": {str(k): int(peaks[k]) for k in range(K)},
        }
        return obj, sol, peaks

    # -------------------------- heuristic --------------------------
    def local_search(st, budget):
        end_time = time.time() + budget
        usage = [[0] * (T + 1) for _ in range(K)]
        for i in real_ids:
            s = st[i]
            d = durations[i]
            r = reqs[i]
            for k in range(K):
                rk = r[k]
                if rk:
                    uk = usage[k]
                    for t in range(s, s + d):
                        uk[t] += rk
        peaks = [max(usage[k]) for k in range(K)] if T > 0 else [0] * K
        cur_obj = objective_of(peaks)
        improved = True
        while improved and time.time() < end_time:
            improved = False
            for i in real_ids:
                if time.time() >= end_time:
                    break
                d = durations[i]
                r = reqs[i]
                lo = ES[i]
                for p in preds[i]:
                    v = st[p] + durations[p]
                    if v > lo:
                        lo = v
                hi = LS[i]
                for sv in succs[i]:
                    vv = st[sv] - d
                    if vv < hi:
                        hi = vv
                if hi < lo:
                    continue
                s_old = st[i]
                for k in range(K):
                    rk = r[k]
                    if rk:
                        uk = usage[k]
                        for t in range(s_old, s_old + d):
                            uk[t] -= rk
                base_peaks = [max(usage[k]) for k in range(K)]
                best_s = s_old
                best_obj = cur_obj
                cnt = 0
                for s in range(lo, hi + 1):
                    cnt += 1
                    if (cnt & 1023) == 0 and time.time() >= end_time:
                        break
                    if s == s_old:
                        continue
                    tot = 0.0
                    for k in range(K):
                        pk = base_peaks[k]
                        rk = r[k]
                        if rk:
                            seg = max(usage[k][s:s + d]) + rk
                            if seg > pk:
                                pk = seg
                        tot += costs[k] * pk
                    if tot < best_obj - 1e-9:
                        best_obj = tot
                        best_s = s
                for k in range(K):
                    rk = r[k]
                    if rk:
                        uk = usage[k]
                        for t in range(best_s, best_s + d):
                            uk[t] += rk
                st[i] = best_s
                if best_s != s_old:
                    improved = True
                    peaks = [max(usage[k]) for k in range(K)]
                    cur_obj = objective_of(peaks)
        return st, cur_obj

    st_es = {i: ES[i] for i in ids}
    st_ls = {i: LS[i] for i in ids}
    obj_es = objective_of(compute_peaks(st_es))
    obj_ls = objective_of(compute_peaks(st_ls))
    st0 = dict(st_ls) if obj_ls < obj_es else dict(st_es)

    heur_budget = min(0.15 * time_limit, 20.0)
    heur_budget = max(heur_budget, min(2.0, 0.5 * time_limit))
    best_st, _ = local_search(st0, heur_budget)
    best_obj, best_sol, best_peaks = build_solution(best_st)

    if logger:
        try:
            logger.log_solution(best_obj, best_sol)
        except Exception:
            pass

    # -------------------------- MIP (Gurobi) --------------------------
    if time_limit - (time.time() - t0) > 3.0:
        try:
            model = gp.Model("resource_investment")
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.LogToConsole = 0

            idx = []
            for i in ids:
                for t in range(ES[i], LS[i] + 1):
                    idx.append((i, t))
            x = model.addVars(idx, vtype=GRB.BINARY, name="x")
            R = model.addVars(K, lb=0.0, vtype=GRB.CONTINUOUS, name="R")

            model.setObjective(
                gp.quicksum(costs[k] * R[k] for k in range(K)), GRB.MINIMIZE
            )

            # Each activity starts exactly once within its window
            for i in ids:
                model.addConstr(
                    gp.quicksum(x[i, t] for t in range(ES[i], LS[i] + 1)) == 1.0
                )

            # Precedence relations: s_j >= s_i + d_i
            for i, j in precs:
                di = durations[i]
                model.addConstr(
                    gp.quicksum(t * x[j, t] for t in range(ES[j], LS[j] + 1))
                    >= gp.quicksum(
                        (t + di) * x[i, t] for t in range(ES[i], LS[i] + 1)
                    )
                )

            # Resource capacity constraints for every period and resource
            res_terms = {}
            for i in real_ids:
                d = durations[i]
                r = reqs[i]
                es = ES[i]
                ls = LS[i]
                for k in range(K):
                    rk = r[k]
                    if rk == 0:
                        continue
                    for tau in range(es, ls + 1):
                        var = x[i, tau]
                        for t in range(tau, tau + d):
                            key = (k, t)
                            if key in res_terms:
                                res_terms[key].append((rk, var))
                            else:
                                res_terms[key] = [(rk, var)]
            for (k, t), terms in res_terms.items():
                model.addConstr(gp.quicksum(c * v for c, v in terms) <= R[k])

            # Simple valid lower bounds on capacities
            if T > 0:
                for k in range(K):
                    mx = 0
                    work = 0
                    for i in real_ids:
                        rk = reqs[i][k]
                        if rk > mx:
                            mx = rk
                        work += rk * durations[i]
                    if mx > 0:
                        model.addConstr(R[k] >= mx)
                    if work > 0:
                        model.addConstr(R[k] >= work / float(T))

            # MIP warm start from heuristic
            for i in ids:
                si = best_st[i]
                for t in range(ES[i], LS[i] + 1):
                    x[i, t].Start = 1.0 if t == si else 0.0
            for k in range(K):
                R[k].Start = float(best_peaks[k])

            keys = idx
            varlist = [x[key] for key in keys]
            cb_state = {"obj": best_obj}

            def _callback(m, where):
                if where == GRB.Callback.MIPSOL:
                    try:
                        objbst = m.cbGet(GRB.Callback.MIPSOL_OBJBST)
                        if objbst < cb_state["obj"] - 1e-6:
                            vals = m.cbGetSolution(varlist)
                            vd = dict(zip(keys, vals))
                            st = {}
                            for i in ids:
                                s = ES[i]
                                for t in range(ES[i], LS[i] + 1):
                                    if vd[(i, t)] > 0.5:
                                        s = t
                                        break
                                st[i] = s
                            obj, sol, _ = build_solution(st)
                            if obj < cb_state["obj"] - 1e-6:
                                cb_state["obj"] = obj
                                if logger:
                                    logger.log_solution(obj, sol)
                    except Exception:
                        pass

            remaining = time_limit - (time.time() - t0) - 0.5
            model.Params.TimeLimit = max(1.0, remaining)
            model.optimize(_callback)

            if model.SolCount > 0:
                vals = model.getAttr("X", varlist)
                vd = dict(zip(keys, vals))
                st = {}
                for i in ids:
                    s = ES[i]
                    for t in range(ES[i], LS[i] + 1):
                        if vd[(i, t)] > 0.5:
                            s = t
                            break
                    st[i] = s
                obj, sol, peaks = build_solution(st)
                if obj < best_obj - 1e-9:
                    best_obj = obj
                    best_sol = sol
                    best_peaks = peaks
                    best_st = st
        except Exception:
            pass

    # -------------------------- output --------------------------
    if logger:
        try:
            logger.log_solution(best_obj, best_sol)
        except Exception:
            pass

    best_sol["objective_value"] = float(best_obj)
    with open(args.solution_path, "w") as f:
        json.dump(best_sol, f, indent=2)


if __name__ == "__main__":
    main()