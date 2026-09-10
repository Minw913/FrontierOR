import argparse
import json
import time

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

    t_start = time.time()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    N = data["num_nodes"]
    K = data["num_commodities"]
    S = data["num_scenarios"]
    E = data["num_edges"]
    gamma = float(data["gamma"])
    c0 = int(data["c_0"])
    edges = data["edges"]
    fcost = data["construction_costs"]
    ccost = data["flow_costs"]
    caps = data["capacities"]
    demands = data["demands"]  # [S][K][N]

    prob = 1.0 / S

    # incidence lists
    out_edges = [[] for _ in range(N)]
    in_edges = [[] for _ in range(N)]
    for e, (u, v) in enumerate(edges):
        out_edges[u].append(e)
        in_edges[v].append(e)

    m = gp.Model("stochastic_network_design")
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.OutputFlag = 0
    m.Params.MIPFocus = 1

    # Variables
    z = m.addVars(E, vtype=GRB.BINARY, name="z")
    x = m.addVars(S, K, E, lb=0.0, name="x")
    y = m.addVars(S, E, lb=0.0, name="y")

    # y definition and capacity/linking constraints
    for s in range(S):
        for e in range(E):
            m.addConstr(gp.quicksum(x[s, k, e] for k in range(K)) == y[s, e])
            m.addConstr(y[s, e] <= caps[e] * z[e])

    # Flow conservation
    for s in range(S):
        dem_s = demands[s]
        for k in range(K):
            dem_sk = dem_s[k]
            for n in range(N):
                m.addConstr(
                    gp.quicksum(x[s, k, e] for e in out_edges[n])
                    - gp.quicksum(x[s, k, e] for e in in_edges[n])
                    == dem_sk[n]
                )

    # Budget on number of activated edges
    m.addConstr(gp.quicksum(z[e] for e in range(E)) <= c0)

    # Objective
    obj = gp.QuadExpr()
    for e in range(E):
        obj += fcost[e] * z[e]
    for s in range(S):
        for e in range(E):
            for k in range(K):
                obj += prob * ccost[e] * x[s, k, e]
            obj += (prob / (2.0 * gamma)) * y[s, e] * y[s, e]
    m.setObjective(obj, GRB.MINIMIZE)

    # ordered var lists for fast callback extraction
    z_vars = [z[e] for e in range(E)]
    x_keys = [(s, k, e) for s in range(S) for k in range(K) for e in range(E)]
    x_vars = [x[key] for key in x_keys]
    y_keys = [(s, e) for s in range(S) for e in range(E)]
    y_vars = [y[key] for key in y_keys]

    def build_solution_dict(zv, xv, yv, objv):
        z_out = {}
        for e in range(E):
            z_out[str(e)] = 1 if zv[e] > 0.5 else 0
        x_out = {}
        for idx, (s, k, e) in enumerate(x_keys):
            val = xv[idx]
            if val > 1e-9:
                x_out[f"{s},{k},{e}"] = val
        y_out = {}
        for idx, (s, e) in enumerate(y_keys):
            val = yv[idx]
            if val > 1e-9:
                y_out[f"{s},{e}"] = val
        z_sparse = {str(e): 1 for e in range(E) if zv[e] > 0.5}
        return {
            "objective_value": objv,
            "model_variables": {"z": z_sparse, "x": x_out, "y": y_out},
            "z": z_out,
        }

    best = {"sol": None, "obj": float("inf")}

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            objv = model.cbGetSolution(model.getVarByName("dummy_obj")) if False else None
            objv = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if objv < best["obj"] - 1e-9:
                zv = model.cbGetSolution(z_vars)
                xv = model.cbGetSolution(x_vars)
                yv = model.cbGetSolution(y_vars)
                sol = build_solution_dict(zv, xv, yv, objv)
                best["obj"] = objv
                best["sol"] = sol
                if logger:
                    logger.log_solution(objv, sol)

    elapsed = time.time() - t_start
    remaining = max(1.0, args.time_limit - elapsed - 2.0)
    m.Params.TimeLimit = remaining

    m.optimize(callback)

    # Extract final solution
    final_sol = None
    if m.SolCount > 0:
        zv = [z[e].X for e in range(E)]
        xv = [v.X for v in x_vars]
        yv = [v.X for v in y_vars]
        objv = m.ObjVal
        final_sol = build_solution_dict(zv, xv, yv, objv)
        if objv < best["obj"] - 1e-9:
            if logger:
                logger.log_solution(objv, final_sol)
        elif best["sol"] is not None and best["obj"] < objv:
            final_sol = best["sol"]
    elif best["sol"] is not None:
        final_sol = best["sol"]

    if final_sol is None:
        # No feasible solution found; emit an empty witness.
        final_sol = {
            "objective_value": 1e18,
            "model_variables": {"z": {}, "x": {}, "y": {}},
            "z": {str(e): 0 for e in range(E)},
        }

    with open(args.solution_path, "w") as f:
        json.dump(final_sol, f)


if __name__ == "__main__":
    main()