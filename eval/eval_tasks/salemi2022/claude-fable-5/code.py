import json
import argparse
import time
import sys

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def load_instance(path):
    with open(path, "r") as f:
        return json.load(f)


def build_solution_dict(inst, x_vals, p_vals, pbar_vals):
    """Project raw variable values onto the required solution schema.
    y, y_bar, q are recomputed consistently from the rounded x."""
    G = inst["n_generators"]
    S = inst["n_scenarios"]
    T = inst["T"]
    gens = inst["generators"]

    x = [[1 if x_vals[g][t] > 0.5 else 0 for t in range(T)] for g in range(G)]
    y = [[0] * T for _ in range(G)]
    ybar = [[0] * T for _ in range(G)]
    q = [[0.0] * T for _ in range(G)]

    for g in range(G):
        K = gens[g]["startup_costs_K"]
        prev = 0
        for t in range(T):
            cur = x[g][t]
            if cur == 1 and prev == 0:
                y[g][t] = 1
                # count consecutive off periods before t (capped at t)
                d = 0
                i = t - 1
                while i >= 0 and x[g][i] == 0:
                    d += 1
                    i -= 1
                # startup cost constraints only exist for k = 1..t
                if t >= 1 and d >= 1:
                    kk = min(d, t)
                    q[g][t] = float(K[kk - 1])
                else:
                    q[g][t] = 0.0
            elif cur == 0 and prev == 1:
                ybar[g][t] = 1
            prev = cur

    p = [[[0.0] * T for _ in range(S)] for _ in range(G)]
    pbar = [[[0.0] * T for _ in range(S)] for _ in range(G)]
    for g in range(G):
        for s in range(S):
            for t in range(T):
                if x[g][t] == 1:
                    pv = max(0.0, float(p_vals[g][s][t]))
                    pbv = max(0.0, float(pbar_vals[g][s][t]))
                    if pbv < pv:
                        pbv = pv
                    p[g][s][t] = pv
                    pbar[g][s][t] = pbv
                else:
                    p[g][s][t] = 0.0
                    pbar[g][s][t] = 0.0

    # recompute objective from projected solution
    obj = 0.0
    for g in range(G):
        cf = gens[g]["c_f"]
        cg = gens[g]["c_g"]
        for t in range(T):
            obj += cf * x[g][t] + q[g][t]
        prod = 0.0
        for s in range(S):
            for t in range(T):
                prod += p[g][s][t]
        obj += cg * prod / S

    sol = {
        "objective_value": obj,
        "x": x,
        "y": y,
        "y_bar": ybar,
        "q": q,
        "p": p,
        "p_bar": pbar,
    }
    return sol, obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    inst = load_instance(args.instance_path)
    G = inst["n_generators"]
    S = inst["n_scenarios"]
    T = inst["T"]
    gens = inst["generators"]
    D = inst["demand_scenarios"]
    R = inst["reserve_scenarios"]

    model = gp.Model("suc")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    # Variables
    x = model.addVars(G, T, vtype=GRB.BINARY, name="x")
    y = model.addVars(G, T, vtype=GRB.BINARY, name="y")
    yb = model.addVars(G, T, vtype=GRB.BINARY, name="yb")
    q = model.addVars(G, T, lb=0.0, name="q")
    p = model.addVars(G, S, T, lb=0.0, name="p")
    pb = model.addVars(G, S, T, lb=0.0, name="pb")

    for g in range(G):
        gen = gens[g]
        M = gen["M"]
        m = gen["m"]
        L = gen["L"]
        ell = gen["ell"]
        RU = gen["RU"]
        SU = gen["SU"]
        RD = gen["RD"]
        SD = gen["SD"]
        K = gen["startup_costs_K"]

        for t in range(T):
            # logical linking (x[-1] = 0)
            if t == 0:
                model.addConstr(y[g, t] - yb[g, t] == x[g, t])
            else:
                model.addConstr(y[g, t] - yb[g, t] == x[g, t] - x[g, t - 1])

            # minimum up time
            lo = max(0, t - L + 1)
            model.addConstr(gp.quicksum(y[g, i] for i in range(lo, t + 1)) <= x[g, t])

            # minimum down time
            lo = max(0, t - ell + 1)
            model.addConstr(gp.quicksum(yb[g, i] for i in range(lo, t + 1)) <= 1 - x[g, t])

            # startup cost
            for k in range(1, t + 1):
                model.addConstr(
                    q[g, t]
                    >= K[k - 1]
                    * (x[g, t] - gp.quicksum(x[g, i] for i in range(t - k, t)))
                )

        for s in range(S):
            for t in range(T):
                model.addConstr(p[g, s, t] >= m * x[g, t])
                model.addConstr(p[g, s, t] <= pb[g, s, t])
                model.addConstr(pb[g, s, t] <= M * x[g, t])

                # ramp up (p[-1]=0, x[-1]=0)
                if t == 0:
                    model.addConstr(p[g, s, t] <= SU * y[g, t])
                else:
                    model.addConstr(
                        p[g, s, t] - p[g, s, t - 1]
                        <= RU * x[g, t - 1] + SU * y[g, t]
                    )
                    # ramp down
                    model.addConstr(
                        p[g, s, t - 1] - p[g, s, t]
                        <= RD * x[g, t] + SD * yb[g, t]
                    )

    # demand and reserve
    for s in range(S):
        for t in range(T):
            model.addConstr(gp.quicksum(p[g, s, t] for g in range(G)) >= D[s][t])
            model.addConstr(
                gp.quicksum(pb[g, s, t] for g in range(G)) >= D[s][t] + R[s][t]
            )

    # objective
    obj = gp.LinExpr()
    for g in range(G):
        cf = gens[g]["c_f"]
        cg = gens[g]["c_g"]
        for t in range(T):
            obj += cf * x[g, t] + q[g, t]
        for s in range(S):
            for t in range(T):
                obj += (cg / S) * p[g, s, t]
    model.setObjective(obj, GRB.MINIMIZE)

    # variable lists for fast callback extraction
    x_list = [[x[g, t] for t in range(T)] for g in range(G)]
    p_list = [[[p[g, s, t] for t in range(T)] for s in range(S)] for g in range(G)]
    pb_list = [[[pb[g, s, t] for t in range(T)] for s in range(S)] for g in range(G)]

    best = {"obj": float("inf"), "sol": None}

    def callback(mdl, where):
        if where == GRB.Callback.MIPSOL:
            try:
                xv = [mdl.cbGetSolution(x_list[g]) for g in range(G)]
                pv = [
                    [mdl.cbGetSolution(p_list[g][s]) for s in range(S)]
                    for g in range(G)
                ]
                pbv = [
                    [mdl.cbGetSolution(pb_list[g][s]) for s in range(S)]
                    for g in range(G)
                ]
                sol, objval = build_solution_dict(inst, xv, pv, pbv)
                if objval < best["obj"] - 1e-9:
                    best["obj"] = objval
                    best["sol"] = sol
                    if logger:
                        logger.log_solution(objval, sol)
            except Exception:
                pass

    elapsed = time.time() - start_time
    remaining = max(1.0, args.time_limit - elapsed - 3.0)
    model.Params.TimeLimit = remaining

    model.optimize(callback)

    sol = None
    if model.SolCount > 0:
        xv = [[x[g, t].X for t in range(T)] for g in range(G)]
        pv = [[[p[g, s, t].X for t in range(T)] for s in range(S)] for g in range(G)]
        pbv = [[[pb[g, s, t].X for t in range(T)] for s in range(S)] for g in range(G)]
        sol, objval = build_solution_dict(inst, xv, pv, pbv)
        if objval < best["obj"] - 1e-9:
            best["obj"] = objval
            best["sol"] = sol
            if logger:
                logger.log_solution(objval, sol)
        sol = best["sol"] if best["sol"] is not None else sol
    elif best["sol"] is not None:
        sol = best["sol"]
    else:
        # Fallback: fix all generators on, solve LP for feasible dispatch
        try:
            for g in range(G):
                for t in range(T):
                    x[g, t].LB = 1.0
                    x[g, t].UB = 1.0
                    y[g, t].LB = 1.0 if t == 0 else 0.0
                    y[g, t].UB = 1.0 if t == 0 else 0.0
                    yb[g, t].LB = 0.0
                    yb[g, t].UB = 0.0
            model.Params.TimeLimit = max(
                1.0, args.time_limit - (time.time() - start_time) - 1.0
            )
            model.optimize()
            if model.SolCount > 0:
                xv = [[1.0] * T for _ in range(G)]
                pv = [
                    [[p[g, s, t].X for t in range(T)] for s in range(S)]
                    for g in range(G)
                ]
                pbv = [
                    [[pb[g, s, t].X for t in range(T)] for s in range(S)]
                    for g in range(G)
                ]
                sol, objval = build_solution_dict(inst, xv, pv, pbv)
                if logger:
                    logger.log_solution(objval, sol)
        except Exception:
            sol = None
        if sol is None:
            # last resort: naive all-on at max capacity
            xv = [[1.0] * T for _ in range(G)]
            pv = [
                [[float(gens[g]["M"]) for _ in range(T)] for _ in range(S)]
                for g in range(G)
            ]
            sol, objval = build_solution_dict(inst, xv, pv, pv)
            if logger:
                logger.log_solution(objval, sol)

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()