import argparse
import json
import gurobipy as gp
from solution_logger import SolutionLogger

def get_fallback_solution(n, capacity, weights, profit_matrix):
    selected_items = [0] * n
    best_profit = -1
    best_i = -1
    for i in range(n):
        if weights[i] <= capacity:
            if profit_matrix[i][i] > best_profit:
                best_profit = profit_matrix[i][i]
                best_i = i
    if best_i != -1:
        selected_items[best_i] = 1
        objective_value = best_profit
    else:
        objective_value = 0.0
    return float(objective_value), selected_items

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path) as f:
        data = json.load(f)

    n = data['n']
    capacity = data['capacity']
    weights = data['weights']
    profit_matrix = data['profit_matrix']

    try:
        model = gp.Model("quad_knapsack")
        model.setParam('OutputFlag', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', args.time_limit)

        x = model.addVars(n, vtype=gp.GRB.BINARY, name="x")

        obj = gp.QuadExpr()

        for i in range(n):
            diag = profit_matrix[i][i]
            if diag != 0:
                obj += diag * x[i]

        for i in range(n):
            for j in range(i+1, n):
                off_val = profit_matrix[i][j]
                if off_val != 0:
                    obj += 2 * off_val * x[i] * x[j]

        model.setObjective(obj, gp.GRB.MAXIMIZE)

        model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity, "knapsack")

        if logger:
            def gurobi_callback(model, where):
                if where == gp.GRB.Callback.MIPSOL:
                    obj_val = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
                    logger.log(obj_val)
            model.optimize(gurobi_callback)
        else:
            model.optimize()

        if model.SolCount > 0:
            selected_items = [1 if round(x[i].X) == 1 else 0 for i in range(n)]
            objective_value = model.ObjVal
        else:
            objective_value, selected_items = get_fallback_solution(n, capacity, weights, profit_matrix)

    except Exception as e:
        objective_value, selected_items = get_fallback_solution(n, capacity, weights, profit_matrix)

    solution = {
        "objective_value": float(objective_value),
        "selected_items": selected_items
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()