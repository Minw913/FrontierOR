import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB


def compute_profit(sol, profit_matrix, n):
    """Compute the true total profit for a given binary selection."""
    profit = 0.0
    for i in range(n):
        if sol[i]:
            profit += profit_matrix[i][i]
            for j in range(i + 1, n):
                if sol[j]:
                    profit += profit_matrix[i][j]
    return profit


def greedy_solution(n, weights, capacity, profit_matrix):
    """Construct a feasible greedy solution based on marginal profit / weight ratio."""
    selected = [0] * n
    remaining = set(range(n))
    current_weight = 0
    marginal = [profit_matrix[i][i] for i in range(n)]

    while True:
        best_item = -1
        best_ratio = -1.0
        for i in remaining:
            if marginal[i] > 0 and current_weight + weights[i] <= capacity:
                if weights[i] > 0:
                    ratio = marginal[i] / weights[i]
                else:
                    ratio = float('inf')
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_item = i
        if best_item == -1:
            break

        selected[best_item] = 1
        current_weight += weights[best_item]
        remaining.remove(best_item)
        for j in remaining:
            marginal[j] += profit_matrix[j][best_item]

    profit = compute_profit(selected, profit_matrix, n)
    return selected, profit


def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    capacity = instance['capacity']
    weights = instance['weights']
    profit_matrix = instance['profit_matrix']

    start_time = time.time()

    if n == 0:
        output = {"objective_value": 0.0, "solution": []}
        if logger:
            logger.log_solution(0.0, output)
        with open(args.solution_path, 'w') as f:
            json.dump(output, f, indent=2)
        return

    # Heuristic initial solution
    best_sol, best_obj = greedy_solution(n, weights, capacity, profit_matrix)
    if logger:
        logger.log_solution(float(best_obj), {"objective_value": float(best_obj), "solution": best_sol})

    # If time limit is very small, just return the greedy solution
    if args.time_limit <= 1:
        output = {"objective_value": float(best_obj), "solution": best_sol}
        with open(args.solution_path, 'w') as f:
            json.dump(output, f, indent=2)
        return

    # Count non-zero off-diagonal entries to decide on formulation
    nonzero_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if profit_matrix[i][j] > 0:
                nonzero_pairs += 1

    # Use linearized MILP if the number of interaction terms is moderate, else use MIQP
    use_milp = (nonzero_pairs <= 500000)

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    model = gp.Model(env=env)

    x = model.addVars(n, vtype=GRB.BINARY, name='x')

    if use_milp:
        y = {}
        for i in range(n):
            for j in range(i + 1, n):
                if profit_matrix[i][j] > 0:
                    y[(i, j)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f'y_{i}_{j}')

        model.update()
        for (i, j), var in y.items():
            model.addConstr(var <= x[i])
            model.addConstr(var <= x[j])
            model.addConstr(var >= x[i] + x[j] - 1.0)

        obj = gp.LinExpr()
        for i in range(n):
            obj += profit_matrix[i][i] * x[i]
        for (i, j), var in y.items():
            obj += profit_matrix[i][j] * var
        model.setObjective(obj, GRB.MAXIMIZE)
    else:
        obj = gp.QuadExpr()
        for i in range(n):
            obj += profit_matrix[i][i] * x[i]
        for i in range(n):
            for j in range(i + 1, n):
                if profit_matrix[i][j] > 0:
                    obj += profit_matrix[i][j] * x[i] * x[j]
        model.setObjective(obj, GRB.MAXIMIZE)
        model.Params.NonConvex = 2

    # Knapsack capacity constraint
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity)

    # Gurobi parameters as required
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    elapsed = time.time() - start_time
    remaining_time = max(1.0, args.time_limit - elapsed - 0.5)
    model.Params.TimeLimit = remaining_time

    # Provide initial solution as a warm start
    try:
        for i in range(n):
            x[i].Start = best_sol[i]
        if use_milp:
            for (i, j), var in y.items():
                var.Start = 1.0 if (best_sol[i] and best_sol[j]) else 0.0
    except Exception:
        pass

    # Callback to log improved incumbents during the Gurobi search
    best_obj_gurobi = [best_obj]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_gurobi = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_gurobi > best_obj_gurobi[0]:
                x_vals = model.cbGet(GRB.Callback.MIPSOL_X)
                sol = [int(round(x_vals[i])) for i in range(n)]
                profit = compute_profit(sol, profit_matrix, n)
                if profit > best_obj_gurobi[0]:
                    best_obj_gurobi[0] = profit
                    if logger:
                        logger.log_solution(float(profit), {"objective_value": float(profit), "solution": sol})

    model.optimize(callback)

    # Extract best solution found by Gurobi, if any
    if model.SolCount > 0:
        x_vals = model.getAttr('X', x)
        sol = [int(round(x_vals[i])) for i in range(n)]
        total_weight = sum(weights[i] for i in range(n) if sol[i])
        if total_weight > capacity:
            # Fallback to the heuristic solution if something went wrong
            sol = best_sol
            profit = best_obj
        else:
            profit = compute_profit(sol, profit_matrix, n)
            if profit > best_obj:
                best_obj = profit
                best_sol = sol
    else:
        sol = best_sol
        profit = best_obj

    # Final logging of the best solution
    if logger:
        logger.log_solution(float(best_obj), {"objective_value": float(best_obj), "solution": best_sol})

    output = {
        "objective_value": float(best_obj),
        "solution": best_sol
    }
    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == '__main__':
    solve()