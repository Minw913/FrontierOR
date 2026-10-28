import json
import argparse
import time
import sys
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger


def solve(instance_path, solution_path, time_limit, log_path=None):
    # Read instance
    with open(instance_path, 'r') as f:
        instance = json.load(f)

    W = float(instance['problem_parameters']['knapsack_capacity'])
    Gamma = float(instance['problem_parameters']['uncertainty_budget'])
    items = instance['items']
    n = len(items)

    # Extract item data
    w = [float(item['weight']) for item in items]
    p = [float(item['nominal_profit']) for item in items]
    d = [float(item['max_degradation']) for item in items]
    c = [float(item['outsource_penalty']) for item in items]
    r = [float(item['repair_capacity']) for item in items]

    # Initialize logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    # Create Gurobi model
    model = Model()
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)

    # Decision variables
    # First-stage binary: x_i = 1 if item i is selected
    x = [model.addVar(vtype=GRB.BINARY, name=f'x_{i}') for i in range(n)]

    # Dual variables from the reformulation
    alpha = [model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'alpha_{i}') for i in range(n)]
    beta = [model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'beta_{i}') for i in range(n)]
    gamma_var = [model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'gamma_{i}') for i in range(n)]
    theta = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='theta')

    model.update()

    # Objective: min Σ x_i(c_i - p_i) - Σ c_i*α_i + Σ γ_i + Γ*θ
    obj = quicksum((c[i] - p[i]) * x[i] for i in range(n)) \
          - quicksum(c[i] * alpha[i] for i in range(n)) \
          + quicksum(gamma_var[i] for i in range(n)) \
          + Gamma * theta
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraint 1: Capacity in dual space: Σ w_i*α_i + Σ r_i*β_i ≤ W
    model.addConstr(
        quicksum(w[i] * alpha[i] + r[i] * beta[i] for i in range(n)) <= W,
        name='dual_capacity'
    )

    for i in range(n):
        # Constraint 2: α_i ≤ x_i (coupling binary and continuous)
        model.addConstr(alpha[i] <= x[i], name=f'alpha_le_x_{i}')

        # Constraint 3: α_i ≥ β_i (from z_i ≤ y_i in primal)
        model.addConstr(alpha[i] >= beta[i], name=f'alpha_ge_beta_{i}')

        # Constraint 4: γ_i + θ ≥ d_i*(α_i - β_i)
        model.addConstr(
            gamma_var[i] + theta >= d[i] * (alpha[i] - beta[i]),
            name=f'gamma_theta_{i}'
        )

    model.update()

    # Store for callback access
    model._x_vars = x
    model._n = n
    model._logger = logger
    model._logged_solutions = set()

    # Callback to log incumbent solutions
    def incumbent_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            x_vals = model.cbGetSolution(model._x_vars)
            x_dict = {str(i): int(round(x_vals[i])) for i in range(model._n)}
            solution = {
                'objective_value': obj_val,
                'x': x_dict
            }
            # Avoid duplicate logging
            sol_key = tuple(x_dict[str(i)] for i in range(model._n))
            if sol_key not in model._logged_solutions:
                model._logged_solutions.add(sol_key)
                if model._logger:
                    model._logger.log_solution(obj_val, solution)

    # Optimize
    model.optimize(incumbent_callback)

    # Extract final solution
    if model.SolCount > 0:
        obj_val = model.ObjVal
        x_vals = {str(i): int(round(x[i].X)) for i in range(n)}
        solution = {
            'objective_value': obj_val,
            'x': x_vals
        }
    else:
        # Fallback: select nothing (always feasible with objective 0)
        solution = {
            'objective_value': 0.0,
            'x': {str(i): 0 for i in range(n)}
        }

    # Log final solution if not already logged via callback
    if logger:
        sol_key = tuple(solution['x'][str(i)] for i in range(n))
        if sol_key not in model._logged_solutions:
            logger.log_solution(solution['objective_value'], solution)

    # Write solution
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Robust Knapsack with LP-Relaxed Recourse')
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON instance file')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path to write the solution JSON')
    parser.add_argument('--time_limit', type=int, default=3600,
                        help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, default=None,
                        help='Path to JSONL log file for intermediate solutions')
    args = parser.parse_args()

    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)