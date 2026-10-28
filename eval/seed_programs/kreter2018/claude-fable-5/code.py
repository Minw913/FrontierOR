import argparse
import json
import time
from collections import deque

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def main():
    t_start = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    K = inst["num_resource_types"]
    T = inst["max_project_duration"]
    costs = inst["resource_costs"]
    acts = inst["activities"]
    ids = [a["id"] for a in acts]
    dur = {a["id"]: a["duration"] for a in acts}
    req = {a["id"]: a["resource_requirements"] for a in acts}

    preds = {i: [] for i in ids}
    succs = {i: [] for i in ids}
    for i, j in inst["precedence_relations"]:
        succs[i].append(j)
        preds[j].append(i)

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

    # Earliest / latest start times (CPM)
    ES = {i: 0 for i in ids}
    for u in topo:
        for v in succs[u]:
            if ES[u] + dur[u] > ES[v]:
                ES[v] = ES[u] + dur[u]
    LS = {i: T - dur[i] for i in ids}
    for u in reversed(topo):
        for v in succs[u]:
            if LS[v] - dur[u] < LS[u]:
                LS[u] = LS[v] - dur[u]

    # Fix project start activity at time 0 (unique source)
    sources = [i for i in ids if not preds[i]]
    if len(sources) == 1:
        LS[sources[0]] = 0

    for i in ids:
        if LS[i] < ES[i]:
            LS[i] = ES[i]  # safety (shouldn't happen on feasible instances)

    # ---------- Heuristic: earliest-start schedule ----------
    def usage_of(start):
        peak = [[0] * max(T, 1) for _ in range(K)]
        for j in ids:
            d = dur[j]
            if d == 0:
                continue
            for k in range(K):
                r = req[j][k]
                if r:
                    for t in range(start[j], min(start[j] + d, T)):
                        peak[k][t] += r
        return [max(peak[k]) if peak[k] else 0 for k in range(K)]

    h_start = {i: ES[i] for i in ids}
    hR = usage_of(h_start)
    h_obj = sum(costs[k] * hR[k] for k in range(K))
    best_sol = {
        "objective_value": float(h_obj),
        "start_times": {str(i): int(h_start[i]) for i in ids},
        "resource_amounts": {str(k): int(hR[k]) for k in range(K)},
    }
    if logger:
        logger.log_solution(best_sol["objective_value"], best_sol)

    # ---------- MIP model (time-indexed) ----------
    model = gp.Model("racp")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    remaining = args.time_limit - (time.time() - t_start) - 2.0
    model.Params.TimeLimit = max(1.0, remaining)

    # Start variables x[j,t]
    x = {}
    y = {}  # cumulative: y[j,t] = 1 if activity j started at or before t
    for j in ids:
        for t in range(ES[j], LS[j] + 1):
            x[j, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{t}")
            y[j, t] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"y_{j}_{t}")

    # Resource amount variables
    Rmin = [0] * K
    Rmax = [0] * K
    for k in range(K):
        for j in ids:
            if dur[j] > 0:
                if req[j][k] > Rmin[k]:
                    Rmin[k] = req[j][k]
                Rmax[k] += req[j][k]
    R = [model.addVar(lb=Rmin[k], ub=max(Rmin[k], Rmax[k]), vtype=GRB.INTEGER, name=f"R_{k}")
         for k in range(K)]

    model.setObjective(gp.quicksum(costs[k] * R[k] for k in range(K)), GRB.MINIMIZE)

    # Cumulative linking constraints
    for j in ids:
        model.addConstr(y[j, ES[j]] == x[j, ES[j]])
        for t in range(ES[j] + 1, LS[j] + 1):
            model.addConstr(y[j, t] == y[j, t - 1] + x[j, t])
        model.addConstr(y[j, LS[j]] == 1)

    # Disaggregated precedence: y[j,t] <= y[i, t - d_i]
    for i, j in inst["precedence_relations"]:
        di = dur[i]
        for t in range(ES[j], LS[j] + 1):
            ti = t - di
            if ti > LS[i]:
                continue  # trivially satisfied
            if ti < ES[i]:
                model.addConstr(x[j, t] == 0)
            else:
                model.addConstr(y[j, t] <= y[i, ti])

    # Resource capacity constraints
    usage = [[gp.LinExpr() for _ in range(T)] for _ in range(K)]
    active_kt = [[False] * T for _ in range(K)]
    for j in ids:
        d = dur[j]
        if d == 0:
            continue
        rks = [k for k in range(K) if req[j][k] > 0]
        if not rks:
            continue
        for tau in range(ES[j], LS[j] + 1):
            for t in range(tau, min(tau + d, T)):
                for k in rks:
                    usage[k][t].add(x[j, tau], req[j][k])
                    active_kt[k][t] = True
    for k in range(K):
        for t in range(T):
            if active_kt[k][t]:
                model.addConstr(usage[k][t] <= R[k])

    # Warm start from heuristic
    for (j, t), var in x.items():
        var.Start = 1.0 if t == h_start[j] else 0.0
    for k in range(K):
        R[k].Start = max(hR[k], Rmin[k])

    # Callback for incumbent logging
    x_keys = list(x.keys())
    x_vars = [x[key] for key in x_keys]
    state = {"best": best_sol["objective_value"]}

    def cb(m, where):
        if where == GRB.Callback.MIPSOL:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < state["best"] - 1e-6:
                xv = m.cbGetSolution(x_vars)
                rv = m.cbGetSolution(R)
                starts = {}
                bestval = {}
                for (jt, v) in zip(x_keys, xv):
                    j, t = jt
                    if j not in bestval or v > bestval[j]:
                        bestval[j] = v
                        starts[j] = t
                ramts = [int(round(rv[k])) for k in range(K)]
                sol = {
                    "objective_value": float(sum(costs[k] * ramts[k] for k in range(K))),
                    "start_times": {str(i): int(starts[i]) for i in ids},
                    "resource_amounts": {str(k): int(ramts[k]) for k in range(K)},
                }
                state["best"] = obj
                best_sol.update(sol)
                if logger:
                    logger.log_solution(sol["objective_value"], sol)

    model.optimize(cb)

    # Extract final best from model if available
    if model.SolCount > 0:
        obj = model.ObjVal
        if obj < best_sol["objective_value"] - 1e-6 or True:
            starts = {}
            for j in ids:
                bt, bv = ES[j], -1.0
                for t in range(ES[j], LS[j] + 1):
                    v = x[j, t].X
                    if v > bv:
                        bv = v
                        bt = t
                starts[j] = bt
            ramts = [int(round(R[k].X)) for k in range(K)]
            cand_obj = float(sum(costs[k] * ramts[k] for k in range(K)))
            if cand_obj <= best_sol["objective_value"] + 1e-6:
                best_sol = {
                    "objective_value": cand_obj,
                    "start_times": {str(i): int(starts[i]) for i in ids},
                    "resource_amounts": {str(k): int(ramts[k]) for k in range(K)},
                }

    with open(args.solution_path, "w") as f:
        json.dump(best_sol, f, indent=2)


if __name__ == "__main__":
    main()