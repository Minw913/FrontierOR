import json
import argparse
import time
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
        instance = json.load(f)

    n = instance['n']
    capacity = instance['capacity']
    weights = instance['weights']
    profit_matrix = instance['profit_matrix']

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    start_time = time.time()

    model = gp.Model("QuadraticKnapsack")
    model.Params.Threads = 1
    model.Params.TimeLimit = args.time_limit
    model.Params.MIPGap = 0.0
    model.setParam('LogToConsole', 0)

    x = [model.addVar(vtype=GRB.BINARY, name=f"x_{i}") for i in range(n)]

    # Linearize quadratic objective: maximize x^T P x
    # Since P is symmetric, obj = sum_i P[i][i]*x_i + 2*sum_{i<j} P[i][j]*x_i*x_j
    obj = gp.LinExpr()
    for i in range(n):
        obj += profit_matrix[i][i] * x[i]
        for j in range(i+1, n):
            if profit_matrix[i][j] != 0:
                obj += 2 * profit_matrix[i][j] * x[i] * x[j]

    model.setObjective(obj, GRB.MAXIMIZE)

    # Knapsack constraint
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity, "capacity")

    # Callback to log incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(float(obj_val))

    model.optimize(callback)

    # Extract solution
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        selected = [1 if x[i].X > 0.5 else 0 for i in range(n)]
        objective_value = model.objVal
    else:
        # Fallback: select nothing
        selected = [0] * n
        objective_value = 0.0

    solution = {
        "objective_value": float(objective_value),
        "selected_items": selected
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()