import argparse
import json
import math

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=600)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    N = int(data["N"])
    S = int(data["S"])
    k = int(data["k"])
    beta = float(data["beta"])
    gamma = float(data["gamma"])
    mu_bar = float(data["mu_bar"])
    mu = [float(m) for m in data["mu"]]
    p_s = float(data["p_s"])
    scenarios = data["scenarios"]

    model = gp.Model("cvar_portfolio")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1, args.time_limit - 5)

    x = model.addVars(N, lb=0.0, ub=1.0, name="x")
    z = model.addVars(N, vtype=GRB.BINARY, name="z")
    a = model.addVar(lb=-GRB.INFINITY, name="a")
    v = model.addVar(lb=0.0, name="v")
    u = model.addVars(S, lb=0.0, name="u")

    model.addConstr(gp.quicksum(x[i] for i in range(N)) == 1.0)
    model.addConstr(gp.quicksum(mu[i] * x[i] for i in range(N)) >= mu_bar)
    for i in range(N):
        model.addConstr(x[i] <= z[i])
    model.addConstr(gp.quicksum(z[i] for i in range(N)) <= k)

    for s in range(S):
        row = scenarios[s]
        model.addConstr(u[s] >= -gp.quicksum(row[i] * x[i] for i in range(N)) - a)

    model.addConstr(v >= (1.0 / (1.0 - beta)) * gp.quicksum(p_s * u[s] for s in range(S)))

    obj = (1.0 / (2.0 * gamma)) * gp.quicksum(x[i] * x[i] for i in range(N)) + a + v
    model.setObjective(obj, GRB.MINIMIZE)

    def callback(m, where):
        if where == GRB.Callback.MIPSOL:
            objv = m.cbGetSolution(m._obj_expr) if False else m.cbGet(GRB.Callback.MIPSOL_OBJ)
            xv = m.cbGetSolution([x[i] for i in range(N)])
            zv = m.cbGetSolution([z[i] for i in range(N)])
            av = m.cbGetSolution(a)
            vv = m.cbGetSolution(v)
            sol = {
                "objective_value": float(objv),
                "x": [max(0.0, float(w)) for w in xv],
                "z": [int(round(w)) for w in zv],
                "a": float(av),
                "v": float(vv),
            }
            if logger:
                logger.log_solution(float(objv), sol)

    model.optimize(callback)

    if model.SolCount > 0:
        xv = [max(0.0, x[i].X) for i in range(N)]
        zv = [int(round(z[i].X)) for i in range(N)]
        solution = {
            "objective_value": float(model.ObjVal),
            "x": xv,
            "z": zv,
            "a": float(a.X),
            "v": float(v.X),
        }
        if logger:
            logger.log_solution(float(model.ObjVal), solution)
    else:
        # Fallback heuristic: all wealth in the asset with the highest expected return
        best_i = max(range(N), key=lambda i: mu[i])
        xv = [0.0] * N
        xv[best_i] = 1.0
        zv = [0] * N
        zv[best_i] = 1
        losses = sorted(-scenarios[s][best_i] for s in range(S))
        # VaR at beta level
        idx = min(S - 1, int(math.ceil(beta * S)) - 1)
        av = losses[idx]
        tail = sum(max(l - av, 0.0) for l in losses)
        vv = (1.0 / (1.0 - beta)) * p_s * tail
        objv = (1.0 / (2.0 * gamma)) + av + vv
        solution = {
            "objective_value": float(objv),
            "x": xv,
            "z": zv,
            "a": float(av),
            "v": float(vv),
        }
        if logger:
            logger.log_solution(float(objv), solution)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()