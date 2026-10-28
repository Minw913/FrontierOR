import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None

PATTERNS = {
    "y_{t}": "setup in one-based period t",
    "z_{i}_{k}_{t}": "visit node i by vehicle k in period t",
    "x_{i}_{j}_{k}_{t}": "edge traversal",
    "p_{t}_{omega}": "scenario production",
    "I_{i}_{t}_{omega}": "scenario inventory",
    "e_{i}_{t}_{omega}": "scenario emergency quantity",
    "q_{i}_{k}_{t}_{omega}": "scenario delivery quantity",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()
    t0 = time.time()

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path, "r") as fh:
        inst = json.load(fh)

    n = int(inst["n"])
    T = int(inst["T"])
    m = int(inst["m"])
    Q = float(inst["Q"])
    C = float(inst["C"])
    f = float(inst["f"])
    u = float(inst["u"])
    c = inst["transportation_costs"]
    h = [float(v) for v in inst["h"]]
    L = [float(v) for v in inst["L"]]
    I0 = [float(v) for v in inst["I0"]]
    sigma = [float(v) for v in inst["sigma"]]

    D = inst.get("demand_scenarios")
    probs = inst.get("scenario_probabilities")
    if not D:
        D = [inst["d_bar"]]
        probs = [1.0]
    nS = len(D)
    if not probs or len(probs) != nS:
        probs = [1.0 / nS] * nS
    tot = sum(probs)
    if tot > 0:
        probs = [pw / tot for pw in probs]

    # future demand fut[w][i][t] = sum_{t'>=t} d  (i: 1..n, t: 1..T ; stored 0-based lists)
    fut = []
    for w in range(nS):
        fw = []
        for i in range(n):
            row = [float(x) for x in D[w][i]]
            acc = [0.0] * (T + 1)
            for t in range(T - 1, -1, -1):
                acc[t] = acc[t + 1] + row[t]
            fw.append(acc)
        fut.append(fw)

    # ---------------- Trivial fallback solution (no production, no routing) --------------
    def trivial_solution():
        mv = {}
        obj = 0.0
        for w in range(nS):
            pw = probs[w]
            for t in range(1, T + 1):
                if I0[0] > 1e-9:
                    mv["I_0_%d_%d" % (t, w + 1)] = float(I0[0])
                obj += pw * h[0] * I0[0]
            for i in range(1, n + 1):
                inv = float(I0[i])
                for t in range(1, T + 1):
                    d = float(D[w][i - 1][t - 1])
                    served = min(inv, d)
                    inv -= served
                    ee = d - served
                    if inv > 1e-9:
                        mv["I_%d_%d_%d" % (i, t, w + 1)] = inv
                    if ee > 1e-9:
                        mv["e_%d_%d_%d" % (i, t, w + 1)] = ee
                    obj += pw * (h[i] * inv + sigma[i - 1] * ee)
        sol = {
            "objective_value": obj,
            "model_variables": mv,
            "y": {},
            "z": {},
            "x": {},
            "model_variable_name_patterns": dict(PATTERNS),
        }
        return obj, sol

    triv_obj, triv_sol = trivial_solution()
    best_obj = triv_obj
    best_sol = triv_sol
    if logger:
        try:
            logger.log_solution(triv_obj, triv_sol)
        except Exception:
            pass

    # ---------------- Build MIP ----------------
    mdl = gp.Model("sprp")
    mdl.Params.OutputFlag = 0
    mdl.Params.Seed = 0
    mdl.Params.MIPGap = 1e-4
    mdl.Params.NumericFocus = 0
    mdl.Params.Threads = 1
    mdl.Params.LazyConstraints = 1
    mdl.Params.MIPFocus = 1

    Ts = range(1, T + 1)
    Ks = range(1, m + 1)
    Ws = range(1, nS + 1)

    yv = {t: mdl.addVar(vtype=GRB.BINARY, obj=f, name="y_%d" % t) for t in Ts}

    zv = {}
    for t in Ts:
        for k in Ks:
            for i in range(0, n + 1):
                zv[(i, k, t)] = mdl.addVar(vtype=GRB.BINARY, name="z_%d_%d_%d" % (i, k, t))

    xv = {}
    for t in Ts:
        for k in Ks:
            for i in range(0, n + 1):
                for j in range(i + 1, n + 1):
                    ub = 2.0 if i == 0 else 1.0
                    xv[(i, j, k, t)] = mdl.addVar(
                        vtype=GRB.INTEGER, lb=0.0, ub=ub, obj=float(c[i][j]),
                        name="x_%d_%d_%d_%d" % (i, j, k, t))

    pv = {}
    Iv = {}
    ev = {}
    qv = {}
    for w in Ws:
        pw = probs[w - 1]
        for t in Ts:
            pv[(t, w)] = mdl.addVar(lb=0.0, ub=C, obj=pw * u, name="p_%d_%d" % (t, w))
            Iv[(0, t, w)] = mdl.addVar(lb=0.0, ub=L[0], obj=pw * h[0], name="I_0_%d_%d" % (t, w))
            for i in range(1, n + 1):
                d = float(D[w - 1][i - 1][t - 1])
                ubI = max(0.0, L[i] - d)
                Iv[(i, t, w)] = mdl.addVar(lb=0.0, ub=ubI, obj=pw * h[i],
                                           name="I_%d_%d_%d" % (i, t, w))
                ev[(i, t, w)] = mdl.addVar(lb=0.0, obj=pw * sigma[i - 1],
                                           name="e_%d_%d_%d" % (i, t, w))
                Mq = min(Q, L[i], fut[w - 1][i - 1][t - 1])
                if Mq > 1e-9:
                    for k in Ks:
                        qv[(i, k, t, w)] = mdl.addVar(lb=0.0, ub=Mq,
                                                      name="q_%d_%d_%d_%d" % (i, k, t, w))

    mdl.ModelSense = GRB.MINIMIZE
    mdl.update()

    # Constraints
    for w in Ws:
        for t in Ts:
            mdl.addConstr(pv[(t, w)] <= C * yv[t])
            # plant balance
            ship = gp.quicksum(qv[(i, k, t, w)] for i in range(1, n + 1) for k in Ks
                               if (i, k, t, w) in qv)
            prev = I0[0] if t == 1 else Iv[(0, t - 1, w)]
            mdl.addConstr(prev + pv[(t, w)] - ship - Iv[(0, t, w)] == 0)
            # customer balance
            for i in range(1, n + 1):
                d = float(D[w - 1][i - 1][t - 1])
                deliv = gp.quicksum(qv[(i, k, t, w)] for k in Ks if (i, k, t, w) in qv)
                prev_i = I0[i] if t == 1 else Iv[(i, t - 1, w)]
                mdl.addConstr(prev_i + deliv + ev[(i, t, w)] - Iv[(i, t, w)] == d)
            # vehicle capacity + delivery-visit link
            for k in Ks:
                load = gp.quicksum(qv[(i, k, t, w)] for i in range(1, n + 1)
                                   if (i, k, t, w) in qv)
                mdl.addConstr(load <= Q * zv[(0, k, t)])
                for i in range(1, n + 1):
                    if (i, k, t, w) in qv:
                        Mq = qv[(i, k, t, w)].UB
                        mdl.addConstr(qv[(i, k, t, w)] <= Mq * zv[(i, k, t)])

    for t in Ts:
        for i in range(1, n + 1):
            mdl.addConstr(gp.quicksum(zv[(i, k, t)] for k in Ks) <= 1)
        for k in Ks:
            # degree constraints
            mdl.addConstr(gp.quicksum(xv[(0, j, k, t)] for j in range(1, n + 1))
                          == 2 * zv[(0, k, t)])
            for i in range(1, n + 1):
                expr = gp.quicksum(xv[(j, i, k, t)] for j in range(0, i)) + \
                       gp.quicksum(xv[(i, j, k, t)] for j in range(i + 1, n + 1))
                mdl.addConstr(expr == 2 * zv[(i, k, t)])
                mdl.addConstr(zv[(i, k, t)] <= zv[(0, k, t)])
                mdl.addConstr(xv[(0, i, k, t)] <= 2 * zv[(i, k, t)])
                for j in range(i + 1, n + 1):
                    mdl.addConstr(xv[(i, j, k, t)] <= zv[(i, k, t)])
                    mdl.addConstr(xv[(i, j, k, t)] <= zv[(j, k, t)])
        # vehicle symmetry breaking
        for k in range(1, m):
            mdl.addConstr(zv[(0, k + 1, t)] <= zv[(0, k, t)])

    # ---------------- warm start from trivial solution ----------------
    for v in mdl.getVars():
        v.Start = 0.0
    for name, val in triv_sol["model_variables"].items():
        parts = name.split("_")
        if parts[0] == "I":
            key = (int(parts[1]), int(parts[2]), int(parts[3]))
            if key in Iv:
                Iv[key].Start = val
        elif parts[0] == "e":
            key = (int(parts[1]), int(parts[2]), int(parts[3]))
            if key in ev:
                ev[key].Start = val

    # ---------------- solution extraction helpers ----------------
    def build_solution(getv):
        mv = {}
        yd = {}
        zd = {}
        xd = {}
        obj = 0.0
        for t in Ts:
            v = int(round(getv(yv[t])))
            if v:
                mv["y_%d" % t] = v
                yd[str(t)] = v
                obj += f * v
        for (i, k, t), var in zv.items():
            v = int(round(getv(var)))
            if v:
                key = "%d_%d_%d" % (i, k, t)
                mv["z_" + key] = v
                zd[key] = v
        for (i, j, k, t), var in xv.items():
            v = int(round(getv(var)))
            if v:
                key = "%d_%d_%d_%d" % (i, j, k, t)
                mv["x_" + key] = v
                xd[key] = v
                obj += float(c[i][j]) * v
        for (t, w), var in pv.items():
            v = max(0.0, float(getv(var)))
            if v > 1e-7:
                mv["p_%d_%d" % (t, w)] = v
                obj += probs[w - 1] * u * v
        for (i, t, w), var in Iv.items():
            v = max(0.0, float(getv(var)))
            if v > 1e-7:
                mv["I_%d_%d_%d" % (i, t, w)] = v
                obj += probs[w - 1] * h[i] * v
        for (i, t, w), var in ev.items():
            v = max(0.0, float(getv(var)))
            if v > 1e-7:
                mv["e_%d_%d_%d" % (i, t, w)] = v
                obj += probs[w - 1] * sigma[i - 1] * v
        for (i, k, t, w), var in qv.items():
            v = max(0.0, float(getv(var)))
            if v > 1e-7:
                mv["q_%d_%d_%d_%d" % (i, k, t, w)] = v
        sol = {
            "objective_value": obj,
            "model_variables": mv,
            "y": yd,
            "z": zd,
            "x": xd,
            "model_variable_name_patterns": dict(PATTERNS),
        }
        return obj, sol

    all_vars = mdl.getVars()
    xkeys = list(xv.keys())
    xlist = [xv[key] for key in xkeys]
    zkeys = list(zv.keys())
    zlist = [zv[key] for key in zkeys]

    # precompute edge lists per (k,t)
    edges_kt = {}
    for (i, j, k, t) in xkeys:
        edges_kt.setdefault((k, t), []).append((i, j))

    state = {"best": best_obj, "best_sol": best_sol, "seen": set()}

    def uf_find(parent, a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def callback(model, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            xval = model.cbGetSolution(xlist)
            zval = model.cbGetSolution(zlist)
            xmap = dict(zip(xkeys, xval))
            zmap = dict(zip(zkeys, zval))
            comps_found = []
            local_seen = set()
            for t in Ts:
                for k in Ks:
                    parent = list(range(n + 1))
                    used = False
                    for (i, j) in edges_kt[(k, t)]:
                        if xmap[(i, j, k, t)] > 0.5:
                            used = True
                            ri = uf_find(parent, i)
                            rj = uf_find(parent, j)
                            if ri != rj:
                                parent[ri] = rj
                    if not used:
                        continue
                    r0 = uf_find(parent, 0)
                    comps = {}
                    for i in range(1, n + 1):
                        if zmap[(i, k, t)] > 0.5:
                            r = uf_find(parent, i)
                            if r != r0:
                                comps.setdefault(r, []).append(i)
                    for S in comps.values():
                        if len(S) >= 2:
                            fs = frozenset(S)
                            if fs not in local_seen:
                                local_seen.add(fs)
                                comps_found.append(S)
            if comps_found:
                for S in comps_found:
                    fs = frozenset(S)
                    if fs in state["seen"]:
                        continue
                    state["seen"].add(fs)
                    Slist = sorted(S)
                    j_star = Slist[0]
                    for t2 in Ts:
                        for k2 in Ks:
                            lhs = gp.quicksum(xv[(a, b, k2, t2)]
                                              for ai, a in enumerate(Slist)
                                              for b in Slist[ai + 1:])
                            rhs = gp.quicksum(zv[(i, k2, t2)] for i in Slist) \
                                - zv[(j_star, k2, t2)]
                            model.cbLazy(lhs <= rhs)
                return
            # feasible incumbent (no subtour violations)
            obj_cb = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_cb < state["best"] - 1e-6:
                vals = model.cbGetSolution(all_vars)
                valmap = dict(zip(all_vars, vals))
                objr, sol = build_solution(lambda v: valmap.get(v, 0.0))
                if objr < state["best"] - 1e-9:
                    state["best"] = objr
                    state["best_sol"] = sol
                    if logger:
                        try:
                            logger.log_solution(objr, sol)
                        except Exception:
                            pass
        except gp.GurobiError:
            pass

    elapsed = time.time() - t0
    remaining = args.time_limit - elapsed - 3.0
    mdl.Params.TimeLimit = max(1.0, remaining)

    try:
        mdl.optimize(callback)
    except Exception:
        pass

    # final extraction
    try:
        if mdl.SolCount > 0:
            objr, sol = build_solution(lambda v: v.X)
            if objr < state["best"] - 1e-9:
                state["best"] = objr
                state["best_sol"] = sol
                if logger:
                    try:
                        logger.log_solution(objr, sol)
                    except Exception:
                        pass
    except Exception:
        pass

    with open(args.solution_path, "w") as fh:
        json.dump(state["best_sol"], fh)


if __name__ == "__main__":
    main()