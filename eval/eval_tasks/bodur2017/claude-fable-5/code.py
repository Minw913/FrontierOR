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

    nF = data["num_facilities"]
    nC = data["num_customers"]
    nS = data["num_scenarios"]

    facilities = data["facilities"]
    open_cost = [0.0] * nF
    cap = [0.0] * nF
    for fac in facilities:
        i = fac["id"]
        open_cost[i] = float(fac["opening_cost"])
        cap[i] = float(fac["capacity"])

    tcost = data["transportation_costs"]

    scenarios = data["scenarios"]
    prob = [0.0] * nS
    dem = [[0.0] * nC for _ in range(nS)]
    for sc in scenarios:
        s = sc["id"]
        prob[s] = float(sc["probability"])
        d = sc["demands"]
        for c in range(nC):
            dem[s][c] = float(d[c])

    max_total_demand = max(sum(dem[s]) for s in range(nS))

    m = gp.Model("stochastic_cflp")
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    # Reserve small buffer for output writing
    m.Params.TimeLimit = max(1.0, args.time_limit - 5.0)

    x = m.addVars(nF, vtype=GRB.BINARY, name="x")
    # y[s, f, c]
    y = m.addVars(nS, nF, nC, lb=0.0, vtype=GRB.CONTINUOUS, name="y")

    # Demand satisfaction
    for s in range(nS):
        for c in range(nC):
            m.addConstr(gp.quicksum(y[s, f, c] for f in range(nF)) >= dem[s][c])

    # Capacity / linking
    for s in range(nS):
        for f in range(nF):
            m.addConstr(gp.quicksum(y[s, f, c] for c in range(nC)) <= cap[f] * x[f])

    # Aggregate capacity feasibility
    m.addConstr(gp.quicksum(cap[f] * x[f] for f in range(nF)) >= max_total_demand)

    obj = gp.quicksum(open_cost[f] * x[f] for f in range(nF))
    obj += gp.quicksum(
        prob[s] * tcost[f][c] * y[s, f, c]
        for s in range(nS) for f in range(nF) for c in range(nC)
    )
    m.setObjective(obj, GRB.MINIMIZE)

    def build_solution(obj_val, xvals, yvals):
        open_facs = [f for f in range(nF) if xvals[f] > 0.5]
        x_dict = {str(f): (1 if xvals[f] > 0.5 else 0) for f in range(nF)}
        y_dict = {}
        for s in range(nS):
            sd = {}
            for f in range(nF):
                fd = {}
                for c in range(nC):
                    v = yvals[(s, f, c)]
                    if v < 1e-9:
                        v = 0.0
                    fd[str(c)] = v
                sd[str(f)] = fd
            y_dict[str(s)] = sd
        return {
            "objective_value": obj_val,
            "open_facilities": open_facs,
            "x": x_dict,
            "y": y_dict,
        }

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                try:
                    xvals = model.cbGetSolution([x[f] for f in range(nF)])
                    yflat = model.cbGetSolution(
                        [y[s, f, c] for s in range(nS) for f in range(nF) for c in range(nC)]
                    )
                    yvals = {}
                    idx = 0
                    for s in range(nS):
                        for f in range(nF):
                            for c in range(nC):
                                yvals[(s, f, c)] = yflat[idx]
                                idx += 1
                    sol = build_solution(obj_val, xvals, yvals)
                    logger.log_solution(obj_val, sol)
                except Exception:
                    try:
                        logger.log(obj_val)
                    except Exception:
                        pass

    m.optimize(callback)

    if m.SolCount > 0:
        obj_val = m.ObjVal
        xvals = [x[f].X for f in range(nF)]
        yvals = {(s, f, c): y[s, f, c].X
                 for s in range(nS) for f in range(nF) for c in range(nC)}
        solution = build_solution(obj_val, xvals, yvals)
    else:
        # Fallback: open all facilities, greedy allocation by cheapest cost
        xvals = [1.0] * nF
        yvals = {}
        total_cost = sum(open_cost)
        for s in range(nS):
            rem = list(cap)
            for f in range(nF):
                for c in range(nC):
                    yvals[(s, f, c)] = 0.0
            for c in range(nC):
                need = dem[s][c]
                order = sorted(range(nF), key=lambda f: tcost[f][c])
                for f in order:
                    if need <= 1e-12:
                        break
                    q = min(need, rem[f])
                    if q > 0:
                        yvals[(s, f, c)] += q
                        rem[f] -= q
                        need -= q
                        total_cost += prob[s] * tcost[f][c] * q
        solution = build_solution(total_cost, xvals, yvals)
        if logger:
            logger.log_solution(total_cost, solution)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()