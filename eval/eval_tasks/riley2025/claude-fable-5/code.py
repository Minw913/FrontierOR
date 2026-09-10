import argparse
import json
import math
import time

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t_start = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if (args.log_path and SolutionLogger) else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    I = data["num_demand_sites"]
    J = data["num_production_sites"]
    S = data["num_scenarios"]
    T = data["num_time_periods"]
    M = data["num_module_sizes"]
    K = data["units_per_module_size"]
    umax = float(data["u_max"])

    module_sizes = sorted(data["module_sizes"], key=lambda a: a["id"])
    cap = [float(ms["capacity"]) for ms in module_sizes]
    default_cost = [float(ms["capital_cost_per_unit"]) for ms in module_sizes]

    # per-unit capital costs
    ucost = {}
    for m in range(M):
        for k in range(K):
            ucost[(m, k)] = default_cost[m]
    for mu in data.get("module_units", []):
        m = mu["module_size_id"]
        k = mu["unit_id"]
        ucost[(m, k)] = float(mu["capital_cost"])

    # order units within each size by ascending cost (cheapest bought first)
    order = {m: sorted(range(K), key=lambda k: (ucost[(m, k)], k)) for m in range(M)}

    pc = data["production_costs"]  # [I][J]
    pen = data["penalty_costs"]    # [I][T]

    scenarios = sorted(data["scenarios"], key=lambda a: a["id"])
    prob = []
    dem = []  # dem[s][i][t]
    sp = data.get("scenario_probabilities", None)
    for s in range(S):
        sc = scenarios[s]
        p = sc.get("probability", None)
        if p is None and sp is not None:
            p = sp[s]
        prob.append(float(p))
        dem.append(sc["demands"])

    # ------------------------------------------------------------------
    # Key structural observation:
    # Relocation costs are only defined between production sites; moves
    # to/from the depot are free and instantaneous.  Hence any period-to-
    # period reconfiguration can be routed through the depot at zero cost,
    # and site-to-site paid relocations are never needed.  The model then
    # only requires: sum_j z[j,m,t,s] <= (# purchased units of size m).
    # Depot stocks and depot relocation flows are derived afterwards.
    # ------------------------------------------------------------------

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model("modular_supply_chain", env=env)
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    # First-stage: purchase binaries (keyed by actual unit_id)
    yvar = {}
    for m in range(M):
        for k in range(K):
            yvar[(m, k)] = model.addVar(vtype=GRB.BINARY,
                                        obj=ucost[(m, k)],
                                        name=f"y_{m}_{k}")
    # symmetry breaking: cheapest units first within each size
    for m in range(M):
        seq = order[m]
        for a in range(K - 1):
            model.addConstr(yvar[(m, seq[a])] >= yvar[(m, seq[a + 1])])

    # z upper bounds (units above umax capacity at one site are useless)
    zub = []
    for m in range(M):
        if cap[m] > 1e-12:
            zub.append(min(K, int(math.ceil(umax / cap[m]))))
        else:
            zub.append(K)

    # Second-stage variables
    xvar = {}   # (i,j,t,s) served fraction
    uvar = {}   # (i,t,s) unmet fraction
    fixed_u = {}  # zero-demand entries fixed to 1 (cost 0)
    zvar = {}   # (j,m,t,s) integer units at site

    for s in range(S):
        ps = prob[s]
        for t in range(T):
            for j in range(J):
                for m in range(M):
                    zvar[(j, m, t, s)] = model.addVar(vtype=GRB.INTEGER, lb=0,
                                                      ub=zub[m], obj=0.0,
                                                      name=f"z_{j}_{m}_{t}_{s}")
            for i in range(I):
                d = float(dem[s][i][t])
                if d <= 1e-12:
                    fixed_u[(i, t, s)] = 1.0
                    continue
                uvar[(i, t, s)] = model.addVar(lb=0.0, ub=1.0,
                                               obj=ps * pen[i][t] * d,
                                               name=f"u_{i}_{t}_{s}")
                for j in range(J):
                    xvar[(i, j, t, s)] = model.addVar(lb=0.0, ub=1.0,
                                                      obj=ps * pc[i][j] * d,
                                                      name=f"x_{i}_{j}_{t}_{s}")

    # Constraints
    for s in range(S):
        for t in range(T):
            # demand balance
            for i in range(I):
                if (i, t, s) in uvar:
                    model.addConstr(
                        gp.quicksum(xvar[(i, j, t, s)] for j in range(J))
                        + uvar[(i, t, s)] == 1.0)
            # capacity per site
            for j in range(J):
                prod = gp.LinExpr()
                for i in range(I):
                    if (i, j, t, s) in xvar:
                        prod.addTerms(float(dem[s][i][t]), xvar[(i, j, t, s)])
                capex = gp.quicksum(cap[m] * zvar[(j, m, t, s)] for m in range(M))
                model.addConstr(prod <= capex)
                model.addConstr(prod <= umax)
        # units at sites limited by purchases
        for t in range(T):
            for m in range(M):
                model.addConstr(
                    gp.quicksum(zvar[(j, m, t, s)] for j in range(J))
                    <= gp.quicksum(yvar[(m, k)] for k in range(K)))

    model.ModelSense = GRB.MINIMIZE

    # ---------------- solution construction helpers -------------------
    def build_solution(yvals, xvals, uvals, zvals):
        sol = {}
        yout = {}
        N = [0] * M
        capital = 0.0
        for m in range(M):
            for k in range(K):
                b = 1 if yvals.get((m, k), 0.0) > 0.5 else 0
                yout[f"y_{m}_{k}"] = b
                if b:
                    sol[f"y_{m}_{k}"] = 1
                    N[m] += 1
                    capital += ucost[(m, k)]
        op = 0.0
        for (i, j, t, s), v in xvals.items():
            if v > 1e-9:
                v = min(1.0, v)
                sol[f"x_{i}_{j}_{t}_{s}"] = v
                op += prob[s] * pc[i][j] * float(dem[s][i][t]) * v
        for (i, t, s), v in uvals.items():
            if v > 1e-9:
                v = min(1.0, v)
                sol[f"u_{i}_{t}_{s}"] = v
                op += prob[s] * pen[i][t] * float(dem[s][i][t]) * v
        for (i, t, s), v in fixed_u.items():
            sol[f"u_{i}_{t}_{s}"] = 1.0
        # units at sites + derived depot stocks / free depot relocations
        for s in range(S):
            for m in range(M):
                prev = [0] * J
                for t in range(T):
                    cur = [int(round(zvals.get((j, m, t, s), 0.0))) for j in range(J)]
                    for j in range(J):
                        dlt = cur[j] - prev[j]
                        if dlt > 0:
                            sol[f"r_{J}_{j}_{m}_{t}_{s}"] = dlt   # depot -> site j
                        elif dlt < 0:
                            sol[f"r_{j}_{J}_{m}_{t}_{s}"] = -dlt  # site j -> depot
                        if cur[j] > 0:
                            sol[f"z_{j}_{m}_{t}_{s}"] = cur[j]
                    depot = N[m] - sum(cur)
                    if depot > 0:
                        sol[f"zd_{m}_{t}_{s}"] = depot
                    prev = cur
        obj = capital + op
        return obj, {"objective_value": obj, "model_variables": sol, "y": yout}

    # incumbent logging callback
    ykeys = list(yvar.keys())
    xkeys = list(xvar.keys())
    ukeys = list(uvar.keys())
    zkeys = list(zvar.keys())
    ylist = [yvar[k] for k in ykeys]
    xlist = [xvar[k] for k in xkeys]
    ulist = [uvar[k] for k in ukeys]
    zlist = [zvar[k] for k in zkeys]

    model._best = float("inf")

    def cb(mdl, where):
        if where == GRB.Callback.MIPSOL:
            obj_rep = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_rep < mdl._best - 1e-9:
                try:
                    yv = dict(zip(ykeys, mdl.cbGetSolution(ylist))) if ylist else {}
                    xv = dict(zip(xkeys, mdl.cbGetSolution(xlist))) if xlist else {}
                    uv = dict(zip(ukeys, mdl.cbGetSolution(ulist))) if ulist else {}
                    zv = dict(zip(zkeys, mdl.cbGetSolution(zlist))) if zlist else {}
                    obj, soldict = build_solution(yv, xv, uv, zv)
                    mdl._best = obj_rep
                    if logger:
                        logger.log_solution(obj, soldict)
                except Exception:
                    mdl._best = obj_rep
                    if logger:
                        logger.log(obj_rep)

    elapsed = time.time() - t_start
    tl = max(1.0, float(args.time_limit) - elapsed - 3.0)
    model.Params.TimeLimit = tl

    try:
        model.optimize(cb)
    except Exception:
        pass

    # ---------------- extract final solution --------------------------
    if model.SolCount > 0:
        yv = {k: yvar[k].X for k in ykeys}
        xv = {k: xvar[k].X for k in xkeys}
        uv = {k: uvar[k].X for k in ukeys}
        zv = {k: zvar[k].X for k in zkeys}
        obj, soldict = build_solution(yv, xv, uv, zv)
    else:
        # trivial fallback: buy nothing, everything unmet
        yv = {}
        xv = {}
        uv = {(i, t, s): 1.0 for s in range(S) for t in range(T) for i in range(I)
              if float(dem[s][i][t]) > 1e-12}
        zv = {}
        obj, soldict = build_solution(yv, xv, uv, zv)

    if logger:
        logger.log_solution(obj, soldict)

    with open(args.solution_path, "w") as f:
        json.dump(soldict, f)


if __name__ == "__main__":
    main()