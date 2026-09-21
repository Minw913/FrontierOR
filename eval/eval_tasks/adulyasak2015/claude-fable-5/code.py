import argparse
import json
import time
from itertools import combinations
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def build_sol_dict(obj, y_on, z_on, x_on):
    """Build solution dictionary following the required schema.
    Periods and vehicles are 1-based; nodes 0..n with 0 = plant."""
    yd = {str(t): 1 for t in y_on}
    zd = {"{}_{}_{}".format(i, k, t): 1 for (i, k, t) in z_on}
    xd = {"{}_{}_{}_{}".format(i, j, k, t): v for (i, j, k, t, v) in x_on}
    return {
        "objective_value": float(obj),
        "model_variables": {"y": yd, "z": zd, "x": xd},
        "y": yd,
        "z": zd,
        "x": xd,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()
    t_start = time.time()

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path, "r") as fh:
        data = json.load(fh)

    n = int(data["n"])
    T = int(data["T"])
    m = int(data["m"])
    Q = float(data["Q"])
    C = float(data["C"])
    f = float(data["f"])
    u = float(data["u"])
    c = data["transportation_costs"]
    h = data["h"]
    L = data["L"]
    I0 = data["I0"]
    sigma = data["sigma"]

    S = int(data.get("n_scenarios") or 1)
    probs = data.get("scenario_probabilities") or [1.0]
    scen = data.get("demand_scenarios")
    if not scen:
        scen = [data["d_bar"]]
        S = 1
        probs = [1.0]
    S = min(S, len(scen), len(probs))

    # demand d[s][i][t], i in 1..n, t in 1..T
    d = [[[0.0] * (T + 1) for _ in range(n + 1)] for _ in range(S)]
    for s in range(S):
        for i in range(1, n + 1):
            for t in range(1, T + 1):
                d[s][i][t] = float(scen[s][i - 1][t - 1])

    # future demand suffix sums
    Fut = [[[0.0] * (T + 2) for _ in range(n + 1)] for _ in range(S)]
    for s in range(S):
        for i in range(1, n + 1):
            for t in range(T, 0, -1):
                Fut[s][i][t] = Fut[s][i][t + 1] + d[s][i][t]

    # ---------------- Trivial feasible fallback: do nothing ----------------
    trivial_obj = 0.0
    for s in range(S):
        cost = h[0] * I0[0] * T
        for i in range(1, n + 1):
            inv = float(I0[i])
            for t in range(1, T + 1):
                dd = d[s][i][t]
                served = min(inv, dd)
                unmet = dd - served
                inv -= served
                cost += h[i] * inv + sigma[i - 1] * unmet
        trivial_obj += probs[s] * cost
    trivial_sol = build_sol_dict(trivial_obj, [], [], [])
    if logger:
        logger.log_solution(trivial_obj, trivial_sol)

    best_sol = trivial_sol
    best_obj = trivial_obj

    # ---------------- Build MIP ----------------
    try:
        mod = gp.Model("sprp")
        mod.Params.OutputFlag = 0
        mod.Params.Seed = 0
        mod.Params.MIPGap = 1e-4
        mod.Params.NumericFocus = 0
        mod.Params.Threads = 1
        mod.Params.LazyConstraints = 1

        periods = range(1, T + 1)
        vehicles = range(m)
        custs = range(1, n + 1)

        # First-stage variables
        y = {}
        for t in periods:
            y[t] = mod.addVar(vtype=GRB.BINARY, obj=f, name="y_%d" % t)

        z = {}
        for t in periods:
            for k in vehicles:
                for i in range(n + 1):
                    z[i, k, t] = mod.addVar(vtype=GRB.BINARY,
                                            name="z_%d_%d_%d" % (i, k, t))

        x = {}
        for t in periods:
            for k in vehicles:
                for i in range(n + 1):
                    for j in range(i + 1, n + 1):
                        ub = 2 if i == 0 else 1
                        x[i, j, k, t] = mod.addVar(
                            vtype=GRB.INTEGER, lb=0, ub=ub,
                            obj=float(c[i][j]),
                            name="x_%d_%d_%d_%d" % (i, j, k, t))

        # Second-stage variables
        p = {}
        Iv = {}
        q = {}
        ev = {}
        qub = {}
        for s in range(S):
            pr = probs[s]
            for t in periods:
                p[t, s] = mod.addVar(lb=0.0, ub=C, obj=pr * u)
                Iv[0, t, s] = mod.addVar(lb=0.0, ub=float(L[0]), obj=pr * h[0])
                for i in custs:
                    ubI = max(0.0, float(L[i]) - d[s][i][t])
                    Iv[i, t, s] = mod.addVar(lb=0.0, ub=ubI, obj=pr * h[i])
                    ev[i, t, s] = mod.addVar(lb=0.0, ub=d[s][i][t],
                                             obj=pr * sigma[i - 1])
                    qb = min(Q, float(L[i]), Fut[s][i][t])
                    qub[i, t, s] = max(0.0, qb)
                    for k in vehicles:
                        q[i, k, t, s] = mod.addVar(lb=0.0, ub=qub[i, t, s])

        mod.ModelSense = GRB.MINIMIZE

        # ---------- First-stage constraints ----------
        for t in periods:
            for i in custs:
                mod.addConstr(gp.quicksum(z[i, k, t] for k in vehicles) <= 1)
            for k in vehicles:
                # depot degree
                mod.addConstr(
                    gp.quicksum(x[0, j, k, t] for j in custs) == 2 * z[0, k, t])
                for i in custs:
                    mod.addConstr(z[i, k, t] <= z[0, k, t])
                    mod.addConstr(
                        gp.quicksum(x[min(i, j), max(i, j), k, t]
                                    for j in range(n + 1) if j != i)
                        == 2 * z[i, k, t])
                # link edges to visits
                for i in range(n + 1):
                    for j in range(i + 1, n + 1):
                        if i == 0:
                            mod.addConstr(x[i, j, k, t] <= 2 * z[j, k, t])
                        else:
                            mod.addConstr(x[i, j, k, t] <= z[i, k, t])
                            mod.addConstr(x[i, j, k, t] <= z[j, k, t])
            # vehicle symmetry breaking
            for k in range(m - 1):
                mod.addConstr(z[0, k, t] >= z[0, k + 1, t])

        # ---------- Second-stage constraints ----------
        for s in range(S):
            for t in periods:
                Iprev0 = Iv[0, t - 1, s] if t > 1 else float(I0[0])
                mod.addConstr(
                    Iprev0 + p[t, s]
                    == gp.quicksum(q[i, k, t, s] for i in custs for k in vehicles)
                    + Iv[0, t, s])
                mod.addConstr(p[t, s] <= C * y[t])
                for k in vehicles:
                    mod.addConstr(
                        gp.quicksum(q[i, k, t, s] for i in custs)
                        <= Q * z[0, k, t])
                for i in custs:
                    Iprev = Iv[i, t - 1, s] if t > 1 else float(I0[i])
                    tot_q = gp.quicksum(q[i, k, t, s] for k in vehicles)
                    mod.addConstr(Iprev + tot_q + ev[i, t, s]
                                  == d[s][i][t] + Iv[i, t, s])
                    mod.addConstr(Iprev + tot_q <= float(L[i]))
                    for k in vehicles:
                        mod.addConstr(q[i, k, t, s]
                                      <= qub[i, t, s] * z[i, k, t])

        # MIP start: do-nothing solution
        for t in periods:
            y[t].Start = 0.0
        for key in z:
            z[key].Start = 0.0
        for key in x:
            x[key].Start = 0.0

        # ---------- Callback (lazy subtour elimination + incumbent logging) ----------
        xkeys = list(x.keys())
        xvarlist = [x[key] for key in xkeys]
        zkeys = list(z.keys())
        zvarlist = [z[key] for key in zkeys]
        ytkeys = list(periods)
        yvarlist = [y[t] for t in ytkeys]

        state = {"best": best_obj, "bestsol": None}

        def cb(model, where):
            if where != GRB.Callback.MIPSOL:
                return
            xv = model.cbGetSolution(xvarlist)
            edges_by_kt = defaultdict(list)
            for val, key in zip(xv, xkeys):
                if val > 0.5:
                    i, j, k, t = key
                    edges_by_kt[(k, t)].append((i, j))
            violated = False
            for (k, t), elist in edges_by_kt.items():
                parent = {}

                def find(a):
                    while parent[a] != a:
                        parent[a] = parent[parent[a]]
                        a = parent[a]
                    return a

                for (i, j) in elist:
                    parent.setdefault(i, i)
                    parent.setdefault(j, j)
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj
                comps = defaultdict(set)
                for node in parent:
                    comps[find(node)].add(node)
                for comp in comps.values():
                    if 0 in comp:
                        continue
                    violated = True
                    Sset = sorted(comp)
                    lhs = gp.quicksum(x[i, j, k, t]
                                      for i, j in combinations(Sset, 2))
                    zsum = gp.quicksum(z[i, k, t] for i in Sset)
                    lim = Sset if len(Sset) <= 8 else Sset[:1]
                    for eidx in lim:
                        model.cbLazy(lhs <= zsum - z[eidx, k, t])
            if not violated:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj < state["best"] - 1e-6:
                    state["best"] = obj
                    yv = model.cbGetSolution(yvarlist)
                    zv = model.cbGetSolution(zvarlist)
                    y_on = [t for t, v in zip(ytkeys, yv) if v > 0.5]
                    z_on = [(i, k + 1, t)
                            for (i, k, t), v in zip(zkeys, zv) if v > 0.5]
                    x_on = [(i, j, k + 1, t, int(round(v)))
                            for (i, j, k, t), v in zip(xkeys, xv) if v > 0.5]
                    sol = build_sol_dict(obj, y_on, z_on, x_on)
                    state["bestsol"] = sol
                    if logger:
                        logger.log_solution(obj, sol)

        remaining = args.time_limit - (time.time() - t_start) - 3.0
        mod.Params.TimeLimit = max(1.0, remaining)

        mod.optimize(cb)

        if mod.SolCount > 0:
            obj = mod.ObjVal
            if obj < best_obj - 1e-9 or state["bestsol"] is not None:
                y_on = [t for t in ytkeys if y[t].X > 0.5]
                z_on = [(i, k + 1, t) for (i, k, t) in zkeys
                        if z[i, k, t].X > 0.5]
                x_on = [(i, j, k + 1, t, int(round(x[i, j, k, t].X)))
                        for (i, j, k, t) in xkeys if x[i, j, k, t].X > 0.5]
                sol = build_sol_dict(obj, y_on, z_on, x_on)
                if obj < best_obj - 1e-9:
                    best_obj = obj
                    best_sol = sol
                    if obj < state["best"] - 1e-6 and logger:
                        logger.log_solution(obj, sol)
                elif state["bestsol"] is not None and state["best"] < best_obj - 1e-9:
                    best_obj = state["best"]
                    best_sol = state["bestsol"]
        else:
            if state["bestsol"] is not None and state["best"] < best_obj - 1e-9:
                best_obj = state["best"]
                best_sol = state["bestsol"]
    except Exception:
        # Keep whatever best solution we have (at least the trivial one)
        pass

    with open(args.solution_path, "w") as fh:
        json.dump(best_sol, fh)


if __name__ == "__main__":
    main()