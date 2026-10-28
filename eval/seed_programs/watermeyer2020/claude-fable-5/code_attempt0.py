import argparse
import json
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t0 = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    N = data["num_activities"]
    dbar = data["d_bar"]

    # processing times and demands
    p = [0] * N
    dem = [dict() for _ in range(N)]  # dem[j][resource_key] = per-period demand
    for k, v in data.get("activities", {}).items():
        j = int(k)
        if 0 <= j < N:
            p[j] = int(v.get("processing_time", 0))
            for rk, dv in v.get("resource_demands", {}).items():
                dv = int(dv)
                if dv > 0:
                    dem[j][rk] = dv

    cons = data.get("temporal_constraints", [])

    # ---- Longest path distance matrix (Floyd-Warshall) ----
    NEG = -(10 ** 9)
    dist = np.full((N, N), NEG, dtype=np.int64)
    np.fill_diagonal(dist, 0)
    for c in cons:
        i, j, dl = int(c["i"]), int(c["j"]), int(c["delta"])
        if dl > dist[i, j]:
            dist[i, j] = dl
    # ensure the max-duration arc end -> start with -dbar
    if -dbar > dist[N - 1, 0]:
        dist[N - 1, 0] = -dbar
    for k in range(N):
        cand = dist[:, k:k + 1] + dist[k:k + 1, :]
        np.maximum(dist, cand, out=dist)

    HALF = NEG // 2
    ES = [0] * N
    LS = [dbar] * N
    for j in range(N):
        d0j = dist[0, j]
        dj0 = dist[j, 0]
        es = int(d0j) if d0j > HALF else 0
        es = max(0, es)
        ls = int(-dj0) if dj0 > HALF else dbar
        ls = min(dbar, ls)
        if ls < es:
            ls = es  # keep model well-formed; infeasibility handled by solver
        ES[j] = es
        LS[j] = ls
    ES[0] = 0
    LS[0] = 0

    # ---- Resource period indicators (prefix sums) ----
    maxp = max(p) if p else 0
    H = dbar + maxp + 2
    res_pref = {}
    res_cap = {}
    for rk, rv in data.get("resources", {}).items():
        ind = np.zeros(H + 1, dtype=np.int64)
        for per in rv.get("periods", []):
            per = int(per)
            if 1 <= per <= H:
                ind[per] = 1
        res_pref[rk] = np.cumsum(ind)  # pref[t] = # designated periods in 1..t
        res_cap[rk] = int(rv["capacity"])

    def overlap(rk, t, pj):
        # periods t+1 .. t+pj intersected with resource's period set
        if pj <= 0:
            return 0
        pref = res_pref[rk]
        hi = min(t + pj, H)
        lo = min(t, H)
        return int(pref[hi] - pref[lo])

    # ---- Check ES schedule resource feasibility (for pre-logging / MIP start) ----
    es_feasible = True
    for rk, cap in res_cap.items():
        tot = 0
        for j in range(N):
            d = dem[j].get(rk, 0)
            if d > 0:
                tot += d * overlap(rk, ES[j], p[j])
        if tot > cap:
            es_feasible = False
            break

    best_solution = None
    best_obj = None
    if es_feasible:
        best_obj = ES[N - 1]
        best_solution = {
            "objective_value": int(best_obj),
            "start_times": {str(j): int(ES[j]) for j in range(N)},
        }
        if logger:
            logger.log_solution(int(best_obj), best_solution)

    # ---- Build MIP ----
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model("rcpsp_pi", env=env)
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1

    x = {}  # (j,t) -> var
    var_list = []
    key_list = []
    for j in range(N):
        for t in range(ES[j], LS[j] + 1):
            v = m.addVar(vtype=GRB.BINARY, name=f"x_{j}_{t}")
            x[(j, t)] = v
            var_list.append(v)
            key_list.append((j, t))

    # assignment constraints
    S = {}
    for j in range(N):
        vars_j = [x[(j, t)] for t in range(ES[j], LS[j] + 1)]
        m.addConstr(gp.quicksum(vars_j) == 1)
        S[j] = gp.quicksum(t * x[(j, t)] for t in range(ES[j], LS[j] + 1))

    # temporal constraints (aggregated)
    seen = set()
    for c in cons:
        i, j, dl = int(c["i"]), int(c["j"]), int(c["delta"])
        key = (i, j)
        if key in seen:
            pass
        seen.add(key)
        # skip if trivially implied by windows
        if ES[j] - LS[i] >= dl:
            continue
        m.addConstr(S[j] - S[i] >= dl)
    if ES[0] - LS[N - 1] < -dbar:
        m.addConstr(S[0] - S[N - 1] >= -dbar)

    # resource constraints
    for rk, cap in res_cap.items():
        terms = []
        for j in range(N):
            d = dem[j].get(rk, 0)
            if d <= 0 or p[j] <= 0:
                continue
            for t in range(ES[j], LS[j] + 1):
                ov = overlap(rk, t, p[j])
                if ov > 0:
                    terms.append(d * ov * x[(j, t)])
        if terms:
            m.addConstr(gp.quicksum(terms) <= cap)

    m.setObjective(S[N - 1], GRB.MINIMIZE)

    # MIP start: ES schedule (Gurobi discards it if infeasible)
    for (j, t), v in x.items():
        v.Start = 1.0 if t == ES[j] else 0.0

    # time limit
    elapsed = time.time() - t0
    tl = max(1.0, args.time_limit - elapsed - 1.0)
    m.Params.TimeLimit = tl

    m._vars = var_list
    m._keys = key_list
    m._N = N
    m._best = best_obj

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            objv = int(round(obj))
            if model._best is not None and objv >= model._best:
                return
            vals = model.cbGetSolution(model._vars)
            starts = {}
            for (jj, tt), val in zip(model._keys, vals):
                if val > 0.5:
                    starts[str(jj)] = int(tt)
            # fill any missing (shouldn't happen)
            for jj in range(model._N):
                starts.setdefault(str(jj), 0)
            sol = {"objective_value": objv, "start_times": starts}
            model._best = objv
            model._incumbent = sol
            if logger:
                logger.log_solution(objv, sol)

    m._incumbent = None
    m.optimize(cb)

    # ---- Extract final solution ----
    if m.SolCount > 0:
        objv = int(round(m.ObjVal))
        starts = {}
        for (j, t), v in x.items():
            if v.X > 0.5:
                starts[str(j)] = int(t)
        for j in range(N):
            starts.setdefault(str(j), int(ES[j]))
        sol = {"objective_value": objv, "start_times": starts}
        if best_solution is None or objv < best_solution["objective_value"]:
            best_solution = sol
            if logger and (best_obj is None or objv < best_obj):
                logger.log_solution(objv, sol)
    if best_solution is None:
        # fallback: temporally-feasible ES schedule
        best_solution = {
            "objective_value": int(ES[N - 1]),
            "start_times": {str(j): int(ES[j]) for j in range(N)},
        }

    with open(args.solution_path, "w") as f:
        json.dump(best_solution, f)


if __name__ == "__main__":
    main()