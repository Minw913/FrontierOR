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
    capacity = data['capacity']
    weights = data['weights']
    profit_matrix = data['profit_matrix']
    
    logger = None
    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="maximize")
    
    count_nonzero = 0
    E = []
    for i in range(n):
        for j in range(i+1, n):
            if profit_matrix[i][j] > 0:
                count_nonzero += 1
                E.append((i, j))
    
    use_linearization = count_nonzero <= 10000
    
    model = gp.Model()
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    
    if not use_linearization:
        model.setParam('NonConvex', 2)
    
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    
    if use_linearization:
        y = model.addVars(E, vtype=GRB.BINARY, name="y")
        obj = gp.quicksum(profit_matrix[i][i] * x[i] for i in range(n))
        obj += gp.quicksum(profit_matrix[i][j] * y[i, j] for (i, j) in E)
        model.setObjective(obj, GRB.MAXIMIZE)
        model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity, "knapsack")
        for (i, j) in E:
            model.addConstr(y[i, j] <= x[i], f"lin1_{i}_{j}")
            model.addConstr(y[i, j] <= x[j], f"lin2_{i}_{j}")
            model.addConstr(y[i, j] >= x[i] + x[j] - 1, f"lin3_{i}_{j}")
    else:
        obj = gp.quicksum(profit_matrix[i][i] * x[i] for i in range(n))
        for i in range(n):
            for j in range(i+1, n):
                obj += profit_matrix[i][j] * x[i] * x[j]
        model.setObjective(obj, GRB.MAXIMIZE)
        model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity, "knapsack")
    
    if logger:
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                objval = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                x_vals = [model.cbGetSolution(x[i]) for i in range(n)]
                sol_list = [int(round(val)) for val in x_vals]
                solution_dict = {
                    "objective_value": float(objval),
                    "solution": sol_list
                }
                logger.log_solution(float(objval), solution_dict)
        model.optimize(callback)
    else:
        model.optimize()
    
    if model.SolCount > 0:
        sol_x = [int(round(x[i].X)) for i in range(n)]
        obj_val = model.ObjVal
    else:
        sol_x = [0] * n
        obj_val = 0.0
    
    output = {
        "objective_value": float(obj_val),
        "solution": sol_x
    }
    with open(args.solution_path, 'w') as f:
        json.dump(output, f)

if __name__ == '__main__':
    main()