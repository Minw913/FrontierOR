import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    n = data['n']
    m = data['m']
    profits = data['profits']
    pairwise_profits = data['pairwise_profits']
    weights = data['weights']
    capacities = data['capacities']

    model = gp.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', 1)
    model.setParam('NonConvex', 2)

    x = model.addVars(n, m, vtype=GRB.BINARY, name="x")

    linear_expr = gp.quicksum(profits[i] * x[i, k] for i in range(n) for k in range(m))
    quad_expr = gp.quicksum(
        pairwise_profits[i][j] * x[i, k] * x[j, k]
        for k in range(m)
        for i in range(n)
        for j in range(i + 1, n)
    )
    model.setObjective(linear_expr + quad_expr, GRB.MAXIMIZE)

    for k in range(m):
        model.addConstr(gp.quicksum(weights[i] * x[i, k] for i in range(n)) <= capacities[k], name=f"cap_{k}")

    for i in range(n):
        model.addConstr(gp.quicksum(x[i, k] for k in range(m)) <= 1, name=f"assign_{i}")

    model.setParam('TimeLimit', args.time_limit)

    logger = None
    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="maximize")

    if logger:
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj_val)
        model.optimize(callback)
    else:
        model.optimize()

    if model.SolCount > 0:
        objective_value = model.ObjVal
        assignment = []
        for i in range(n):
            for k in range(m):
                if x[i, k].X > 0.5:
                    assignment.append([i, k])
    else:
        objective_value = 0.0
        assignment = []

    solution = {
        "objective_value": objective_value,
        "assignment": assignment
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()