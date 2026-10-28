import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser(
        description="Two-stage stochastic capacitated facility location solver"
    )
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    t0 = time.time()
    TL = float(args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # ------------------------- load instance -------------------------
    with open(args.instance_path, "r") as fh:
        inst = json.load(fh)

    F = int(inst["num_facilities"])
    C = int(inst["num_customers"])
    scen = inst["scenarios"]
    S = int(inst.get("num_scenarios", scen.get("num_scenarios",
                                                 len(scen["probabilities"]))))
    fixed = [float(v) for v in inst["facilities"]["fixed_costs"]]
    cap = [float(v) for v in inst["facilities"]["capacities"]]
    routing = [[float(v) for v in row] for row in inst["routing_costs"]]
    probs = [float(v) for v in scen["probabilities"]]
    demands = [[float(v) for v in row] for row in scen["demands"]]

    D_max = max(sum(row) for row in demands)

    # ------------- recourse transportation LP (reused, RHS updated) -------------
    sub = gp.Model("recourse")
    sub.Params.OutputFlag = 0
    sub.Params.Method = 1          # dual simplex: warm-starts on RHS changes
    sub.Params.Presolve = 0        # preserve basis / duals across resolves
    sub.Params.Threads = 1
    sub.Params.Seed = 0
    sub.Params.NumericFocus = 0

    xs = sub.addVars(F, C, lb=0.0,
                     obj=[routing[i][j] for i in range(F) for j in range(C)],
                     name="x")
    dem = [sub.addConstr(xs.sum("*", j) >= demands[0][j],
                         name="dem_%d" % j) for j in range(C)]
    sup = [sub.addConstr(xs.sum(i, "*") <= cap[i],
                         name="sup_%d" % i) for i in range(F)]
    sub.update()

    def full_eval(yvals, deadline=None):
        """Exact expected cost of a facility vector.
        Returns (objective or None, list of (s, alphas, betas) dual info)."""
        if sum(cap[i] for i in range(F) if yvals[i]) < D_max - 1e-9:
            return None, []
        sub.setAttr("RHS", sup, [cap[i] * yvals[i] for i in range(F)])
        total = sum(fixed[i] for i in range(F) if yvals[i])
        duals = []
        complete = True
        for s in range(S):
            if deadline is not None and time.time() > deadline:
                complete = False
                break
            sub.setAttr("RHS", dem, demands[s])
            sub.optimize()
            if sub.Status != GRB.OPTIMAL:
                return None, duals
            total += probs[s] * sub.ObjVal
            duals.append((s, sub.getAttr("Pi", dem), sub.getAttr("Pi", sup)))
        return (total if complete else None), duals

    # ---------------------- initial feasible candidates ----------------------
    def gen_greedy(key):
        order = sorted(range(F), key=key)
        g = [0] * F
        tot = 0.0
        for i in order:
            if tot >= D_max:
                break
            g[i] = 1
            tot += cap[i]
        return g

    cands = [
        gen_greedy(lambda i: fixed[i] / cap[i] if cap[i] > 0 else float("inf")),
        gen_greedy(lambda i: (fixed[i], -cap[i])),
        [1] * F,
    ]
    seen, uniq = set(), []
    for cand in cands:
        t = tuple(cand)
        if t not in seen:
            seen.add(t)
            uniq.append(cand)

    best_obj, best_y, init_duals = None, None, []
    for idx, cand in enumerate(uniq):
        if idx > 0 and (time.time() - t0) > 0.30 * TL:
            break
        dl = None if idx == 0 else t0 + TL - 1.0
        obj, duals = full_eval(cand, deadline=dl)
        init_duals.extend(duals)
        if obj is not None and (best_obj is None or obj < best_obj):
            best_obj, best_y = obj, list(cand)

    if best_y is None:  # pathological fallback (schema guarantees feasibility)
        best_y = [1] * F
        minc = [min(routing[i][j] for i in range(F)) for j in range(C)]
        best_obj = sum(fixed) + sum(
            probs[s] * sum(demands[s][j] * minc[j] for j in range(C))
            for s in range(S))

    if logger:
        logger.log_solution(float(best_obj),
                            {"objective_value": float(best_obj),
                             "y": [int(v) for v in best_y]})

    # ---------------------- Benders (L-shaped) master ----------------------
    remaining = TL - (time.time() - t0)
    if remaining > 2.0:
        master = gp.Model("benders_master")
        master.Params.Seed = 0
        master.Params.MIPGap = 1e-4
        master.Params.NumericFocus = 0
        master.Params.Threads = 1
        master.Params.OutputFlag = 0
        master.Params.LazyConstraints = 1
        master.Params.PreCrush = 1

        yv = master.addVars(F, vtype=GRB.BINARY, obj=fixed, name="y")
        th = master.addVars(S, lb=0.0, obj=probs, name="theta")
        master.addConstr(
            gp.quicksum(cap[i] * yv[i] for i in range(F)) >= D_max,
            name="aggregate_capacity")
        master.update()
        yv_list = [yv[i] for i in range(F)]

        def build_cut(s, alphas, betas):
            const = 0.0
            ds = demands[s]
            for j in range(C):
                const += ds[j] * alphas[j]
            coefs = [cap[i] * betas[i] for i in range(F)]
            expr = gp.LinExpr()
            expr.addConstant(const)
            expr.addTerms(coefs, yv_list)
            return th[s] >= expr

        for (s, a, b) in init_duals:
            master.addConstr(build_cut(s, a, b))

        state = {"best_obj": best_obj, "best_y": best_y}

        def cb(model, where):
            if where != GRB.Callback.MIPSOL:
                return
            if time.time() - t0 > TL - 1.0:
                model.terminate()
                return
            ysol = model.cbGetSolution(yv)
            thsol = model.cbGetSolution(th)
            yvals = [1 if ysol[i] > 0.5 else 0 for i in range(F)]
            sub.setAttr("RHS", sup, [cap[i] * yvals[i] for i in range(F)])
            true_obj = sum(fixed[i] for i in range(F) if yvals[i])
            for s in range(S):
                if (s & 15) == 0 and time.time() - t0 > TL - 0.5:
                    model.terminate()
                    return
                sub.setAttr("RHS", dem, demands[s])
                sub.optimize()
                if sub.Status != GRB.OPTIMAL:
                    return
                q = sub.ObjVal
                true_obj += probs[s] * q
                if q > thsol[s] + 1e-6 * max(1.0, abs(q)):
                    alphas = sub.getAttr("Pi", dem)
                    betas = sub.getAttr("Pi", sup)
                    ds = demands[s]
                    const = 0.0
                    for j in range(C):
                        const += ds[j] * alphas[j]
                    coefs = [cap[i] * betas[i] for i in range(F)]
                    expr = gp.LinExpr()
                    expr.addConstant(const)
                    expr.addTerms(coefs, yv_list)
                    model.cbLazy(th[s] >= expr)
            if true_obj < state["best_obj"] - 1e-7 * max(1.0, abs(state["best_obj"])):
                state["best_obj"] = true_obj
                state["best_y"] = list(yvals)
                if logger:
                    logger.log_solution(float(true_obj),
                                        {"objective_value": float(true_obj),
                                         "y": [int(v) for v in yvals]})

        master.Params.TimeLimit = max(remaining - 1.0, 0.05)
        try:
            master.optimize(cb)
        except gp.GurobiError:
            pass

        # Every incumbent was exactly evaluated inside the callback.
        best_obj = state["best_obj"]
        best_y = state["best_y"]

    # --------------------------- write solution ---------------------------
    out = {"objective_value": float(best_obj),
           "y": [int(v) for v in best_y]}
    with open(args.solution_path, "w") as fh:
        json.dump(out, fh)


if __name__ == "__main__":
    main()