import json
import argparse
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB


def compute_objective(x, factor_loadings, idiosyncratic_variance, gamma, kappa, mu):
    """Compute the objective value for a given allocation."""
    n = len(x)
    r = len(factor_loadings[0]) if n > 0 else 0

    # Factor part: 0.5 * ||L^T x||^2
    factor_part = 0.0
    for j in range(r):
        val = sum(factor_loadings[i][j] * x[i] for i in range(n))
        factor_part += 0.5 * val * val

    # Diagonal part: 0.5 * sum_i (d_i + 1/gamma) * x_i^2
    diag_part = 0.0
    for i in range(n):
        diag_part += 0.5 * (idiosyncratic_variance[i] + 1.0 / gamma) * x[i] * x[i]

    # Linear part: -kappa * mu'x
    linear_part = -kappa * sum(mu[i] * x[i] for i in range(n))

    return factor_part + diag_part + linear_part


def main():
    parser = argparse.ArgumentParser(description="Portfolio Optimization Solver")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    k = instance['k']
    r = instance['r']
    gamma = instance['gamma']
    kappa = instance['kappa']
    mu = instance['mu']
    factor_loadings = instance['factor_loadings']  # shape [n][r]
    idiosyncratic_variance = instance['idiosyncratic_variance']
    constraints = instance['constraints']

    has_min_return = constraints.get('has_min_return_constraint', False)
    r_bar = constraints.get('r_bar', None)
    has_min_investment = constraints.get('has_min_investment_constraint', False)
    l_min = constraints.get('l_min_investment', None)
    u_max = constraints.get('u_max_investment', None)

    # Build Gurobi model
    model = gp.Model("portfolio")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', 0)

    # Decision variables
    x = model.addVars(n, lb=0.0, ub=GRB.INFINITY, name="x")
    z = model.addVars(n, vtype=GRB.BINARY, name="z")

    # Budget constraint: sum of allocations = 1
    model.addConstr(gp.quicksum(x[i] for i in range(n)) == 1.0, "budget")

    # Cardinality constraint: at most k securities selected
    model.addConstr(gp.quicksum(z[i] for i in range(n)) <= k, "cardinality")

    # Linking constraints between x and z
    for i in range(n):
        ub = u_max[i] if (u_max is not None and u_max[i] is not None) else 1.0
        lb = l_min[i] if (l_min is not None and l_min[i] is not None) else 0.0
        model.addConstr(x[i] <= ub * z[i], f"upper_link_{i}")
        model.addConstr(x[i] >= lb * z[i], f"lower_link_{i}")

    # Minimum return constraint (if applicable)
    if has_min_return and r_bar is not None:
        model.addConstr(
            gp.quicksum(mu[i] * x[i] for i in range(n)) >= r_bar,
            "min_return"
        )

    # Auxiliary variables for factor structure: y_j = sum_i L[i][j] * x[i]
    y = model.addVars(r, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y")
    for j in range(r):
        model.addConstr(
            y[j] == gp.quicksum(factor_loadings[i][j] * x[i] for i in range(n)),
            f"factor_{j}"
        )

    # Build objective: 0.5*||L^T x||^2 + 0.5*sum(d_i + 1/gamma)*x_i^2 - kappa*mu'x
    obj = gp.QuadExpr()

    # Factor variance component: 0.5 * sum_j y_j^2
    for j in range(r):
        obj += 0.5 * y[j] * y[j]

    # Diagonal (idiosyncratic + ridge) component
    for i in range(n):
        diag_val = idiosyncratic_variance[i] + 1.0 / gamma
        obj += 0.5 * diag_val * x[i] * x[i]

    # Expected return component (subtracted)
    for i in range(n):
        obj += -kappa * mu[i] * x[i]

    model.setObjective(obj, GRB.MINIMIZE)

    # Update model before optimization
    model.update()

    # Callback to log incumbent solutions
    best_obj = [float('inf')]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0] - 1e-10:
                best_obj[0] = obj_val
                x_val = [model.cbGetSolution(x[i]) for i in range(n)]
                solution = {"objective_value": obj_val, "x": x_val}
                if logger:
                    logger.log_solution(obj_val, solution)

    # Optimize
    model.optimize(callback)

    # Extract solution
    if model.SolCount > 0:
        x_val = [x[i].X for i in range(n)]
        obj_val = model.ObjVal
    else:
        # Fallback heuristic: equal weight on top-k by expected return
        x_val = [0.0] * n
        indices = sorted(range(n), key=lambda i: -mu[i])[:k]
        for i in indices:
            x_val[i] = 1.0 / k
        obj_val = compute_objective(x_val, factor_loadings, idiosyncratic_variance,
                                    gamma, kappa, mu)

    # Final solution
    solution = {
        "objective_value": obj_val,
        "x": x_val
    }

    if logger:
        logger.log_solution(obj_val, solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()