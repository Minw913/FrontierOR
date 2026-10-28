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

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['num_facilities']
    m = instance['num_customers']
    f = instance['opening_costs']
    c = instance['allocation_costs']

    model = gp.Model("qUFL")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', 0)

    y = model.addVars(n, vtype=GRB.BINARY, name="y")
    x = model.addVars(n, m, lb=0.0, ub=1.0, name="x")

    for j in range(m):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n)) == 1)

    for i in range(n):
        for j in range(m):
            model.addConstr(x[i, j] <= y[i])

    obj = gp.LinExpr()
    for i in range(n):
        obj += f[i] * y[i]
    
    for i in range(n):
        for j in range(m):
            obj += c[i][j] * x[i, j] * x[i, j]

    model.setObjective(obj, GRB.MINIMIZE)

    all_vars = [y[i] for i in range(n)] + [x[i, j] for i in range(n) for j in range(m)]
    model._vars = all_vars
    model._logger = logger
    model._f = f
    model._c = c
    model._n = n
    model._m = m

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            vals = model.cbGetSolution(model._vars)
            n = model._n
            m = model._m
            f = model._f
            c = model._c
            
            y_val = [int(round(v)) for v in vals[:n]]
            x_val = []
            idx = n
            true_obj = float(sum(f[i] * y_val[i] for i in range(n)))
            for i in range(n):
                row = []
                for j in range(m):
                    xij = float(max(0.0, vals[idx]))
                    row.append(xij)
                    if xij > 1e-6:
                        true_obj += c[i][j] * (xij ** 2)
                    idx += 1
                x_val.append(row)
                
            sol_dict = {
                "objective_value": true_obj,
                "variant": "quadratic",
                "y": y_val,
                "x": x_val
            }
            if model._logger:
                model._logger.log_solution(true_obj, sol_dict)

    model.optimize(callback)

    if model.SolCount > 0:
        y_val = [int(round(y[i].X)) for i in range(n)]
        x_val = [[float(max(0.0, x[i, j].X)) for j in range(m)] for i in range(n)]
        
        true_obj = float(sum(f[i] * y_val[i] for i in range(n)))
        for i in range(n):
            for j in range(m):
                if x_val[i][j] > 1e-6:
                    true_obj += c[i][j] * (x_val[i][j] ** 2)
    else:
        y_val = [1] * n
        x_val = [[0.0] * m for _ in range(n)]
        true_obj = float(sum(f))
        for j in range(m):
            sum_inv_c = sum(1.0 / c[i][j] for i in range(n))
            for i in range(n):
                x_val[i][j] = float((1.0 / c[i][j]) / sum_inv_c)
                true_obj += c[i][j] * (x_val[i][j] ** 2)

    solution = {
        "objective_value": true_obj,
        "variant": "quadratic",
        "y": y_val,
        "x": x_val
    }
    
    if logger and model.SolCount == 0:
        logger.log_solution(true_obj, solution)

    with open(args.solution_path, 'w') as f_out:
        json.dump(solution, f_out)

if __name__ == "__main__":
    main()