import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_solution(hc, sc, p, xkeys, xvals, ykeys, yvals, zkeys, zvals):
    """Build the output solution dict and recompute its objective consistently."""
    prod = {}
    obj = 0.0
    for (i, k, t, dd), v in zip(xkeys, xvals):
        if v is None or v < 1e-7:
            v = 0.0
        else:
            v = float(round(v, 6))
        prod["x_%d_%d_%d_%d" % (i, k, t, dd)] = v
        obj += hc[i] * (dd - t) * v
    sud = {}
    for (i, j, k, t), v in zip(ykeys, yvals):
        iv = int(round(v)) if v is not None else 0
        if iv < 0:
            iv = 0
        sud["y_%d_%d_%d_%d" % (i, j, k, t)] = iv
        obj += sc[i][j][k] * iv
    carry = {}
    for (i, k, t), v in zip(zkeys, zvals):
        if t < p:
            iv = int(round(v)) if v is not None else 0
            carry["z_%d_%d_%d" % (i, k, t)] = iv
    sol = {
        "objective_value": obj,
        "production": prod,
        "setups": sud,
        "carryover": carry,
    }
    return obj, sol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t_start = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    n = data["dimensions"]["n"]
    m = data["dimensions"]["m"]
    p = data["dimensions"]["p"]
    dem = data["demands"]                 # [n][p]
    hc = data["inventory_costs"]          # [n]
    pt = data["processing_time"]          # scalar
    st = data["setup_times"]              # [n][n][m]
    sc = data["setup_costs"]              # [n][n][m]
    cap = data["machine_capacities"]      # [m][p]
    maxprod = data["max_production"]      # [n][p]
    maxsu = data["max_setups_per_item"]   # [n][m][p]
    ml = data["minimum_lot_sizes"]        # [n]
    elig = data["machine_eligibility"]    # [n][m]

    # remaining demand of item i from period t onward
    rem = [[0] * (p + 1) for _ in range(n)]
    for i in range(n):
        for t in range(p - 1, -1, -1):
            rem[i][t] = rem[i][t + 1] + dem[i][t]

    model = gp.Model("clsd_parallel")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0
    model.Params.MIPFocus = 1

    # ---------------- Variables ----------------
    # Production split by demand period (facility-location style)
    xvars = {}
    for i in range(n):
        for k in range(m):
            if not elig[i][k]:
                continue
            for t in range(p):
                for dd in range(t, p):
                    if dem[i][dd] > 0:
                        xvars[(i, k, t, dd)] = model.addVar(
                            lb=0.0, ub=dem[i][dd],
                            obj=hc[i] * (dd - t),
                            vtype=GRB.CONTINUOUS,
                            name="x_%d_%d_%d_%d" % (i, k, t, dd))

    # Setup state at the beginning of each period; t = 0..p (t = p is end state)
    zvars = {}
    for k in range(m):
        for t in range(p + 1):
            for i in range(n):
                if elig[i][k]:
                    zvars[(i, k, t)] = model.addVar(
                        vtype=GRB.BINARY, name="z_%d_%d_%d" % (i, k, t))

    # Changeover arcs (integer counts) + connectivity flow
    yvars = {}
    fvars = {}
    gvars = {}
    arcs = {}
    Ktot = {}
    for k in range(m):
        items_k = [i for i in range(n) if elig[i][k]]
        for t in range(p):
            alist = []
            K = 0
            for i in items_k:
                if maxsu[i][k][t] > 0:
                    K += maxsu[i][k][t]
            Ktot[(k, t)] = K
            if K > 0:
                for i in items_k:
                    for j in items_k:
                        if i == j:
                            continue
                        if maxsu[j][k][t] > 0:
                            alist.append((i, j))
                            yvars[(i, j, k, t)] = model.addVar(
                                lb=0, ub=maxsu[j][k][t],
                                obj=sc[i][j][k],
                                vtype=GRB.INTEGER,
                                name="y_%d_%d_%d_%d" % (i, j, k, t))
                            fvars[(i, j, k, t)] = model.addVar(
                                lb=0.0, ub=K, vtype=GRB.CONTINUOUS,
                                name="f_%d_%d_%d_%d" % (i, j, k, t))
                for i in items_k:
                    gvars[(i, k, t)] = model.addVar(
                        lb=0.0, ub=K, vtype=GRB.CONTINUOUS,
                        name="g_%d_%d_%d" % (i, k, t))
            arcs[(k, t)] = alist

    model.ModelSense = GRB.MINIMIZE
    model.update()

    # ---------------- Constraints ----------------
    # One configuration per machine per (start-of-)period state
    for k in range(m):
        for t in range(p + 1):
            vs = [zvars[(i, k, t)] for i in range(n) if elig[i][k]]
            if vs:
                model.addConstr(gp.quicksum(vs) == 1, "cfg_%d_%d" % (k, t))

    # Precompute in/out arc lists
    in_arcs = {}
    out_arcs = {}
    for (i, j, k, t) in yvars:
        in_arcs.setdefault((j, k, t), []).append(i)
        out_arcs.setdefault((i, k, t), []).append(j)

    Xexpr = {}
    for k in range(m):
        items_k = [i for i in range(n) if elig[i][k]]
        for t in range(p):
            K = Ktot[(k, t)]
            for i in items_k:
                # aggregated production quantity
                xl = [xvars[(i, k, t, dd)] for dd in range(t, p)
                      if (i, k, t, dd) in xvars]
                X = gp.quicksum(xl) if xl else gp.LinExpr(0.0)
                Xexpr[(i, k, t)] = (X, xl)

                yin = gp.quicksum(yvars[(jj, i, k, t)]
                                  for jj in in_arcs.get((i, k, t), []))
                yout = gp.quicksum(yvars[(i, jj, k, t)]
                                   for jj in out_arcs.get((i, k, t), []))

                # setup-state / degree balance (Eulerian trail condition)
                model.addConstr(
                    zvars[(i, k, t)] + yin == zvars[(i, k, t + 1)] + yout,
                    "flowz_%d_%d_%d" % (i, k, t))

                # max setups into item i
                if in_arcs.get((i, k, t)):
                    model.addConstr(yin <= maxsu[i][k][t],
                                    "msu_%d_%d_%d" % (i, k, t))

                # production allowed only if configured (carryover or setup)
                if xl:
                    cap_units = cap[k][t] // pt if pt > 0 else rem[i][t]
                    M = min(maxprod[i][t], cap_units, rem[i][t])
                    M = max(M, 0)
                    model.addConstr(
                        X <= M * (zvars[(i, k, t)] + yin),
                        "link_%d_%d_%d" % (i, k, t))
                    # minimum lot size per setup (non-triangular safeguard)
                    if ml[i] > 0:
                        model.addConstr(X >= ml[i] * yin,
                                        "mlot_%d_%d_%d" % (i, k, t))
                else:
                    # no production possible; still ensure no wasted min-lot issue:
                    if ml[i] > 0 and in_arcs.get((i, k, t)):
                        model.addConstr(yin == 0, "nolot_%d_%d_%d" % (i, k, t))

                # connectivity flow balance
                if K > 0:
                    fin = gp.quicksum(fvars[(jj, i, k, t)]
                                      for jj in in_arcs.get((i, k, t), []))
                    fout = gp.quicksum(fvars[(i, jj, k, t)]
                                       for jj in out_arcs.get((i, k, t), []))
                    model.addConstr(
                        gvars[(i, k, t)] + fin - fout == yin,
                        "fbal_%d_%d_%d" % (i, k, t))
                    model.addConstr(gvars[(i, k, t)] <= K * zvars[(i, k, t)],
                                    "gsrc_%d_%d_%d" % (i, k, t))

            # flow capacity on arcs
            for (i, j) in arcs[(k, t)]:
                model.addConstr(
                    fvars[(i, j, k, t)] <= K * yvars[(i, j, k, t)],
                    "fcap_%d_%d_%d_%d" % (i, j, k, t))

            # machine capacity
            prod_time = gp.quicksum(pt * Xexpr[(i, k, t)][0] for i in items_k)
            su_time = gp.quicksum(st[i][j][k] * yvars[(i, j, k, t)]
                                  for (i, j) in arcs[(k, t)])
            model.addConstr(prod_time + su_time <= cap[k][t],
                            "cap_%d_%d" % (k, t))

    # demand satisfaction
    for i in range(n):
        for dd in range(p):
            if dem[i][dd] > 0:
                vs = [xvars[(i, k, t, dd)] for k in range(m)
                      for t in range(dd + 1) if (i, k, t, dd) in xvars]
                model.addConstr(gp.quicksum(vs) == dem[i][dd],
                                "dem_%d_%d" % (i, dd))

    # per-period max production
    for i in range(n):
        for t in range(p):
            vs = [Xexpr[(i, k, t)][0] for k in range(m) if (i, k, t) in Xexpr]
            if vs:
                model.addConstr(gp.quicksum(vs) <= maxprod[i][t],
                                "mp_%d_%d" % (i, t))

    # ---------------- Callback for incumbent logging ----------------
    xkeys = list(xvars.keys())
    xlist = [xvars[key] for key in xkeys]
    ykeys = list(yvars.keys())
    ylist = [yvars[key] for key in ykeys]
    zkeys = [key for key in zvars.keys() if key[2] < p]
    zlist = [zvars[key] for key in zkeys]

    model._best = float("inf")
    model._logger = logger

    def cb(mdl, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            obj = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj >= mdl._best - 1e-9:
                return
            mdl._best = obj
            if mdl._logger is None:
                return
            xv = mdl.cbGetSolution(xlist)
            yv = mdl.cbGetSolution(ylist)
            zv = mdl.cbGetSolution(zlist)
            robj, sol = build_solution(hc, sc, p, xkeys, xv, ykeys, yv,
                                       zkeys, zv)
            mdl._logger.log_solution(robj, sol)
        except Exception:
            pass

    elapsed = time.time() - t_start
    tl = max(1.0, args.time_limit - elapsed - 3.0)
    model.Params.TimeLimit = tl

    if logger is not None:
        model.optimize(cb)
    else:
        model.optimize()

    # ---------------- Write final solution ----------------
    if model.SolCount > 0:
        xv = [v.X for v in xlist]
        yv = [v.X for v in ylist]
        zv = [v.X for v in zlist]
        robj, sol = build_solution(hc, sc, p, xkeys, xv, ykeys, yv, zkeys, zv)
        if logger is not None:
            try:
                logger.log_solution(robj, sol)
            except Exception:
                pass
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
    else:
        # no feasible solution found within the time limit; emit empty skeleton
        sol = {"objective_value": float("inf"),
               "production": {}, "setups": {}, "carryover": {}}
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)


if __name__ == "__main__":
    main()