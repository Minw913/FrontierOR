import argparse
import json
import sys

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    num_trips = data["parameters"]["num_trips_M"]
    duties = data["duties"]
    costs = data["costs"]
    num_duties = len(duties)

    # Build trip -> duties covering map
    cover = [[] for _ in range(num_trips)]
    for j, duty in enumerate(duties):
        for t in duty:
            cover[t].append(j)

    model = gp.Model("set_partitioning")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1, args.time_limit - 5)

    x = model.addVars(num_duties, vtype=GRB.BINARY)
    model.setObjective(gp.quicksum(costs[j] * x[j] for j in range(num_duties)), GRB.MINIMIZE)

    for t in range(num_trips):
        if not cover[t]:
            # Infeasible trip: no duty covers it (shouldn't happen for valid instances)
            model.addConstr(gp.quicksum() == 1)
        else:
            model.addConstr(gp.quicksum(x[j] for j in cover[t]) == 1)

    best = {"obj": None, "sel": None}

    def callback(m, where):
        if where == GRB.Callback.MIPSOL:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if best["obj"] is None or obj < best["obj"] - 1e-9:
                vals = m.cbGetSolution([x[j] for j in range(num_duties)])
                sel = [j for j in range(num_duties) if vals[j] > 0.5]
                best["obj"] = obj
                best["sel"] = sel
                if logger:
                    logger.log_solution(obj, {
                        "objective_value": obj,
                        "selected_duties": sel,
                    })

    model.optimize(callback)

    sel = None
    obj = None
    if model.SolCount > 0:
        sel = [j for j in range(num_duties) if x[j].X > 0.5]
        obj = sum(costs[j] for j in sel)
    elif best["sel"] is not None:
        sel = best["sel"]
        obj = best["obj"]

    if sel is None:
        # No feasible solution found; output empty (invalid but graceful)
        solution = {"objective_value": float("inf"), "selected_duties": []}
    else:
        solution = {"objective_value": obj, "selected_duties": sel}
        if logger and (best["obj"] is None or obj < best["obj"] - 1e-9):
            logger.log_solution(obj, solution)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()