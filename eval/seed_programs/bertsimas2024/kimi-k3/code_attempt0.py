import argparse
import json

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger

PATTERNS = {
    "z[e]": "binary design variable for edge e",
    "x[k,r,e]": "commodity k, scenario r, edge e flow",
    "tf[r,e]": "total scenario flow on edge e",
}


def solve_instance(data, time_limit, logger):
    N = int(data["num_nodes"])
    K = int(data["num_commodities"])
    R = int(data["num_scenarios"])
    E = int(data["num_edges"])
    gamma = float(data["gamma"])
    c0 = int(data["c_0"])
    edges = data["edges"]
    ccost = [float(v) for v in data["construction_costs"]]
    fcost = [float(v) for v in data["flow_costs"]]
    cap = [float(v) for v in data["capacities"]]
    demands = data["demands"]

    # Incident edge lists for flow conservation.
    out_edges = [[] for _ in range(N)]
    in_edges = [[] for _ in range(N)]
    for e, (u, v) in enumerate(edges):
        out_edges[u].append(e)
        in_edges[v].append(e)

    m = gp.Model("stochastic_multicommodity_design")
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = max(1, int(time_limit))

    z = m.addVars(E, vtype=GRB.BINARY, name="z")
    x = m.addVars(K, R, E, lb=0.0, name="x")
    tf = m.addVars(R, E, lb=0.0, name="tf")

    # Flow conservation: net outflow - inflow = net supply.
    for k in range(K):
        for r in range(R):
            d = demands[r][k]
            for n in range(N):
                m.addConstr(
                    gp.quicksum(x[k, r, e] for e in out_edges[n])
                    - gp.quicksum(x[k, r, e] for e in in_edges[n])
                    == d[n]
                )

    # Total per-scenario edge flow and activation/capacity coupling.
    for r in range(R):
        for e in range(E):
            m.addConstr(tf[r, e] == gp.quicksum(x[k, r, e] for k in range(K)))
            m.addConstr(tf[r, e] <= cap[e] * z[e])

    # Construction budget.
    m.addConstr(gp.quicksum(z[e] for e in range(E)) <= c0)

    quad_coef = 1.0 / (2.0 * gamma * R)
    obj = (
        gp.quicksum(ccost[e] * z[e] for e in range(E))
        + (1.0 / R)
        * gp.quicksum(
            fcost[e] * x[k, r, e]
            for k in range(K)
            for r in range(R)
            for e in range(E)
        )
        + quad_coef
        * gp.quicksum(tf[r, e] * tf[r, e] for r in range(R) for e in range(E))
    )
    m.setObjective(obj, GRB.MINIMIZE)

    z_list = [z[e] for e in range(E)]
    x_list = [x[k, r, e] for k in range(K) for r in range(R) for e in range(E)]
    tf_list = [tf[r, e] for r in range(R) for e in range(E)]

    def build_solution(zv, xv, tfv):
        construction = 0.0
        zdict = {}
        mvars = {}
        for e in range(E):
            active = 1 if zv[e] > 0.5 else 0
            zdict[str(e)] = active
            if active:
                construction += ccost[e]
                mvars["z[%d]" % e] = 1.0
        transport = 0.0
        idx = 0
        for k in range(K):
            for r in range(R):
                for e in range(E):
                    v = xv[idx]
                    idx += 1
                    if v > 1e-6:
                        transport += fcost[e] * v
                        mvars["x[%d,%d,%d]" % (k, r, e)] = v
        transport /= R
        quad_sum = 0.0
        idx = 0
        for r in range(R):
            for e in range(E):
                v = tfv[idx]
                idx += 1
                if v > 1e-6:
                    quad_sum += v * v
                    mvars["tf[%d,%d]" % (r, e)] = v
        obj_val = construction + transport + quad_sum / (2.0 * gamma * R)
        sol = {
            "objective_value": obj_val,
            "model_variables": mvars,
            "z": zdict,
            "model_variable_name_patterns": dict(PATTERNS),
        }
        return obj_val, sol

    if logger is not None:

        def cb(model, where):
            if where == GRB.Callback.MIPSOL:
                zv = model.cbGetSolution(z_list)
                xv = model.cbGetSolution(x_list)
                tfv = model.cbGetSolution(tf_list)
                obj_val, sol = build_solution(zv, xv, tfv)
                logger.log_solution(obj_val, sol)

        m.optimize(cb)
    else:
        m.optimize()

    if m.SolCount > 0:
        zv = [v.X for v in z_list]
        xv = [v.X for v in x_list]
        tfv = [v.X for v in tf_list]
        _, sol = build_solution(zv, xv, tfv)
        return sol

    # Fallback (should not occur for feasible instances): empty witness.
    return {
        "objective_value": 0.0,
        "model_variables": {},
        "z": {str(e): 0 for e in range(E)},
        "model_variable_name_patterns": dict(PATTERNS),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    logger = (
        SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    )

    sol = solve_instance(data, args.time_limit, logger)

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()