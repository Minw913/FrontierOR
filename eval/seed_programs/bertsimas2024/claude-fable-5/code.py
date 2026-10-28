import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def build_solution_dict(E, xkeys, tfkeys, zvals, xvals, fcon, fcost, gamma, R):
    """Build a self-consistent sparse witness from raw variable values."""
    prob = 1.0 / R
    mv = {}
    zd = {}
    obj = 0.0

    zbin = [1 if v > 0.5 else 0 for v in zvals]
    for e in range(E):
        zd[str(e)] = zbin[e]
        if zbin[e] == 1:
            mv["z[%d]" % e] = 1
            obj += fcon[e]

    # recompute total flows from commodity flows for internal consistency
    tf_acc = {}  # (r, e) -> total flow
    for idx, (k, r, e) in enumerate(xkeys):
        v = xvals[idx]
        if v > 1e-9:
            mv["x[%d,%d,%d]" % (k, r, e)] = v
            obj += prob * fcost[e] * v
            key = (r, e)
            tf_acc[key] = tf_acc.get(key, 0.0) + v

    inv2g = 1.0 / (2.0 * gamma)
    for (r, e), v in tf_acc.items():
        if v > 1e-9:
            mv["tf[%d,%d]" % (r, e)] = v
            obj += prob * inv2g * v * v

    sol = {
        "objective_value": obj,
        "model_variables": mv,
        "z": zd,
    }
    return sol, obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=600)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    N = data["num_nodes"]
    K = data["num_commodities"]
    R = data["num_scenarios"]
    E = data["num_edges"]
    gamma = float(data["gamma"])
    c0 = int(data["c_0"])
    edges = data["edges"]
    fcon = data["construction_costs"]
    fcost = data["flow_costs"]
    caps = data["capacities"]
    demands = data["demands"]  # [R][K][N]

    prob = 1.0 / R
    inv2g = 1.0 / (2.0 * gamma)

    # adjacency lists
    out_edges = [[] for _ in range(N)]
    in_edges = [[] for _ in range(N)]
    for e in range(E):
        u, v = edges[e]
        out_edges[u].append(e)
        in_edges[v].append(e)

    m = gp.Model("scnd")
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.OutputFlag = 0

    # variables
    z = m.addVars(E, vtype=GRB.BINARY, name="z")
    x = m.addVars(K, R, E, lb=0.0, name="x")
    tf = m.addVars(R, E, lb=0.0, name="tf")

    # tf definition and capacity linking
    for r in range(R):
        for e in range(E):
            m.addConstr(tf[r, e] == gp.quicksum(x[k, r, e] for k in range(K)))
            m.addConstr(tf[r, e] <= caps[e] * z[e])

    # flow conservation
    for r in range(R):
        dem_r = demands[r]
        for k in range(K):
            dem = dem_r[k]
            for n in range(N):
                m.addConstr(
                    gp.quicksum(x[k, r, e] for e in out_edges[n])
                    - gp.quicksum(x[k, r, e] for e in in_edges[n])
                    == dem[n]
                )

    # cardinality on activated edges
    m.addConstr(gp.quicksum(z[e] for e in range(E)) <= c0)

    # objective
    obj = gp.quicksum(fcon[e] * z[e] for e in range(E))
    obj += prob * gp.quicksum(
        fcost[e] * x[k, r, e] for k in range(K) for r in range(R) for e in range(E)
    )
    obj += prob * inv2g * gp.quicksum(
        tf[r, e] * tf[r, e] for r in range(R) for e in range(E)
    )
    m.setObjective(obj, GRB.MINIMIZE)

    # ordered variable lists for fast callback extraction
    zlist = [z[e] for e in range(E)]
    xkeys = [(k, r, e) for k in range(K) for r in range(R) for e in range(E)]
    xlist = [x[k, r, e] for (k, r, e) in xkeys]
    tfkeys = [(r, e) for r in range(R) for e in range(E)]

    elapsed = time.time() - start_time
    remaining = max(1.0, args.time_limit - elapsed - 3.0)
    m.Params.TimeLimit = remaining

    best = {"obj": float("inf"), "sol": None}

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                zvals = model.cbGetSolution(zlist)
                xvals = model.cbGetSolution(xlist)
                sol, sobj = build_solution_dict(
                    E, xkeys, tfkeys, zvals, xvals, fcon, fcost, gamma, R
                )
                if sobj < best["obj"] - 1e-12:
                    best["obj"] = sobj
                    best["sol"] = sol
                    if logger:
                        logger.log_solution(sobj, sol)
            except Exception:
                try:
                    if logger:
                        logger.log(model.cbGet(GRB.Callback.MIPSOL_OBJ))
                except Exception:
                    pass

    try:
        m.optimize(callback)
    except gp.GurobiError:
        pass

    # extract final solution
    final_sol = None
    if m.SolCount > 0:
        zvals = [z[e].X for e in range(E)]
        xvals = [v.X for v in xlist]
        sol, sobj = build_solution_dict(
            E, xkeys, tfkeys, zvals, xvals, fcon, fcost, gamma, R
        )
        if sobj < best["obj"] - 1e-12:
            best["obj"] = sobj
            best["sol"] = sol
            if logger:
                logger.log_solution(sobj, sol)
        final_sol = best["sol"] if best["sol"] is not None else sol
    elif best["sol"] is not None:
        final_sol = best["sol"]

    if final_sol is None:
        # no feasible solution found: emit an empty witness
        final_sol = {
            "objective_value": float("inf"),
            "model_variables": {},
            "z": {str(e): 0 for e in range(E)},
        }

    with open(args.solution_path, "w") as f:
        json.dump(final_sol, f)


if __name__ == "__main__":
    main()