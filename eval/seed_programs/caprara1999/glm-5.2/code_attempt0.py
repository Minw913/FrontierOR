import argparse
import json
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None


def compute_objective(selected, profit_matrix, n):
    """Compute total profit for a given selection."""
    obj = 0.0
    for i in range(n):
        if selected[i]:
            obj += profit_matrix[i][i]
            for j in range(i + 1, n):
                if selected[j]:
                    obj += 2 * profit_matrix[i][j]
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Load instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    n = instance['n']
    capacity = instance['capacity']
    weights = instance['weights']
    profit_matrix = instance['profit_matrix']

    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path and SolutionLogger else None

    # Gurobi environment and model
    env = gp.Env(empty=True)
    env.setParam('Seed', 0)
    env.start()
    model = gp.Model(env=env)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', 0)
    model.setParam('NonConvex', 2)  # Required for non-convex MIQP with binary variables

    # Decision variables
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    x_vars = [x[i] for i in range(n)]

    # Objective: maximize sum_i P_ii x_i + 2 * sum_{i<j} P_ij x_i x_j
    obj = gp.QuadExpr()
    for i in range(n):
        if profit_matrix[i][i] != 0:
            obj.addTerms(profit_matrix[i][i], x[i], x[i])
        for j in range(i + 1, n):
            if profit_matrix[i][j] != 0:
                # profit_matrix is symmetric, so both P_ij and P_ji appear in the full sum.
                # Their combined contribution is 2 * P_ij.
                obj.addTerms(2 * profit_matrix[i][j], x[i], x[j])
    model.setObjective(obj, GRB.MAXIMIZE)

    # Capacity constraint
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity, name="capacity")

    # Greedy initial solution (by individual profit per weight)
    items = list(range(n))
    items.sort(key=lambda i: profit_matrix[i][i] / weights[i] if weights[i] > 0 else 0.0, reverse=True)
    selected_greedy = [0] * n
    cur_weight = 0
    for i in items:
        if cur_weight + weights[i] <= capacity:
            selected_greedy[i] = 1
            cur_weight += weights[i]
    greedy_obj = compute_objective(selected_greedy, profit_matrix, n)

    # Provide the greedy solution as a MIP start
    for i in range(n):
        x[i].Start = selected_greedy[i]
    if logger:
        solution_dict = {
            "objective_value": float(greedy_obj),
            "selected_items": selected_greedy
        }
        logger.log_solution(float(greedy_obj), solution_dict)

    # Callback to log every incumbent found during the search
    class IncumbentCallback:
        def __init__(self, logger, x_vars, n):
            self.logger = logger
            self.x_vars = x_vars
            self.n = n

        def __call__(self, model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                sol = model.cbGetSolution(self.x_vars)
                selected = [int(round(sol[i])) for i in range(self.n)]
                solution_dict = {
                    "objective_value": float(obj_val),
                    "selected_items": selected
                }
                if self.logger:
                    self.logger.log_solution(float(obj_val), solution_dict)

    callback = IncumbentCallback(logger, x_vars, n)
    model.setCallback(callback)

    # Optimize
    model.optimize()

    # Retrieve best solution
    if model.SolCount > 0:
        selected = [int(round(x[i].Xn)) for i in range(n)]
        obj_val = model.ObjVal
    else:
        selected = [0] * n
        obj_val = 0.0

    # Write output
    output = {
        "objective_value": float(obj_val),
        "selected_items": selected
    }
    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()