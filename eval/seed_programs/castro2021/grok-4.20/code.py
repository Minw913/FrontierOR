import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    m = instance['m']
    supplies = instance['supplies']
    demands = instance['demands']
    linear_costs = instance['linear_costs']
    quadratic_costs = instance.get('quadratic_costs', None)
    cost_type = instance['cost_type']
    arc_capacity = instance['arc_capacity']
    relative_demand_slack = instance['relative_demand_slack']

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model = gp.Model("Transportation")
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1
    model.Params.LogToConsole = 0
    model.Params.MIPGap = 1e-4
    model.Params.MIPGapAbs = 1e-4

    x = {}
    for i in range(n):
        for j in range(m):
            x[i, j] = model.addVar(lb=0.0, ub=arc_capacity, name=f"x_{i}_{j}")

    # slack variables for suppliers
    s = {}
    for i in range(n):
        s[i] = model.addVar(lb=0.0, ub=supplies[i], name=f"s_{i}")

    # Customer demand constraints (exactly)
    for j in range(m):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n)) == demands[j], name=f"demand_{j}")

    # Supplier capacity constraints
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(m)) + s[i] == supplies[i], name=f"supply_{i}")

    # Objective
    obj = 0.0
    for i in range(n):
        for j in range(m):
            lc = linear_costs[i][j]
            flow = x[i, j]
            if cost_type == 'quadratic' and quadratic_costs is not None:
                qc = quadratic_costs[i][j]
                obj += lc * flow + qc * flow * flow
            else:
                obj += lc * flow

    model.setObjective(obj, GRB.MINIMIZE)

    # Callback for logging incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(float(obj_val))

    # Optimize
    start_time = time.time()
    model.optimize(callback)

    # Extract solution
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        objective_value = float(model.objVal)
        flows = {}
        for i in range(n):
            for j in range(m):
                val = x[i, j].X
                if val > 1e-6:  # only report positive flows
                    flows[f"x_{i}_{j}"] = float(val)

        solution = {
            "objective_value": objective_value,
            "flows": flows
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # If infeasible or other status, write minimal output
        solution = {
            "objective_value": 0.0,
            "flows": {}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()