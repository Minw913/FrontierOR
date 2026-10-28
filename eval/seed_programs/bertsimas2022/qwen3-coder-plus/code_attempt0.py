import argparse
import json
import numpy as np
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger


def solve_portfolio_optimization(instance_path, solution_path, time_limit, log_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)

    # Extract data
    n = instance['n']
    k = instance['k']
    r = instance['r']
    gamma = instance['gamma']
    kappa = instance['kappa']
    mu = np.array(instance['mu'])
    factor_loadings = np.array(instance['factor_loadings'])  # shape [n, r]
    idiosyncratic_variance = np.array(instance['idiosyncratic_variance'])
    
    constraints = instance['constraints']
    has_min_return_constraint = constraints['has_min_return_constraint']
    r_bar = constraints['r_bar']
    has_min_investment_constraint = constraints['has_min_investment_constraint']
    l_min_investment = constraints['l_min_investment']
    u_max_investment = constraints['u_max_investment']
    
    # Compute full covariance matrix: F @ F.T + diag(omega_i^2)
    cov_matrix = np.dot(factor_loadings, factor_loadings.T) + np.diag(idiosyncratic_variance)

    # Create Gurobi model
    model = Model("PortfolioOptimization")
    model.setParam('TimeLimit', time_limit)
    model.setParam('NonConvex', 2)  # To allow quadratic terms
    
    # Decision variables
    x = model.addVars(n, lb=0.0, ub=1.0, name="x")  # Allocation fractions
    z = model.addVars(n, vtype=GRB.BINARY, name="z")  # Indicator if security i is selected
    
    # Constraint: Sum of allocations equals 1
    model.addConstr(quicksum(x[i] for i in range(n)) == 1.0, name="budget")
    
    # Cardinality constraint: At most k securities with positive allocation
    model.addConstr(quicksum(z[i] for i in range(n)) <= k, name="cardinality")
    
    # Link x and z variables: x[i] > 0 only if z[i] = 1
    for i in range(n):
        model.addConstr(x[i] <= z[i], name=f"xz_link_{i}")
        
    # Min investment constraints if applicable
    if has_min_investment_constraint and l_min_investment is not None:
        for i in range(n):
            model.addConstr(x[i] >= l_min_investment[i] * z[i], name=f"min_invest_{i}")
    
    # Max investment constraints if applicable
    if u_max_investment is not None:
        for i in range(n):
            model.addConstr(x[i] <= u_max_investment[i] * z[i], name=f"max_invest_{i}")
    
    # Min return constraint if applicable
    if has_min_return_constraint and r_bar is not None:
        model.addConstr(quicksum(mu[i] * x[i] for i in range(n)) >= r_bar, name="min_return")
    
    # Set up objective function
    # Objective = 0.5 * sum(i,j) x[i] * cov_matrix[i, j] * x[j] (variance term)
    #          + 0.5/gamma * sum(i) x[i]^2 (ridge regularization / concentration penalty)
    #          - kappa * sum(i) mu[i] * x[i] (return term to minimize if kappa=1, otherwise constrained)
    
    # Variance term: 0.5 * x.T * cov_matrix * x
    variance_terms = []
    for i in range(n):
        for j in range(n):
            variance_terms.append(0.5 * cov_matrix[i, j] * x[i] * x[j])
    
    # Concentration penalty: 0.5/gamma * sum(x[i]^2)
    reg_penalty_terms = []
    for i in range(n):
        reg_penalty_terms.append(0.5 / gamma * x[i] * x[i])
    
    # Return term: -kappa * sum(mu[i] * x[i])
    return_terms = []
    if kappa == 1:  # Include return in objective
        for i in range(n):
            return_terms.append(-mu[i] * x[i])  # Note negative because we're minimizing
    
    total_objective = quicksum(variance_terms) + quicksum(reg_penalty_terms) + quicksum(return_terms)
    
    model.setObjective(total_objective, GRB.MINIMIZE)
    
    # Set callback for logging improved solutions
    if log_path:
        logger = SolutionLogger(log_path, sense="minimize")
        
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj_val)
        model.optimize(callback)
    else:
        model.optimize()
    
    # Extract solution
    x_vals = [x[i].X if x[i].X > 1e-8 else 0.0 for i in range(n)]
    objective_val = model.ObjVal if model.SolCount > 0 else float('inf')
    
    solution = {
        "objective_value": objective_val,
        "x": x_vals
    }

    with open(solution_path, 'w') as f:
        json.dump(solution, f)


def main():
    parser = argparse.ArgumentParser(description='Solve Portfolio Optimization Problem')
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON file containing the problem instance.')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path where the final solution JSON file must be written.')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum runtime in seconds.')
    parser.add_argument('--log_path', type=str, default=None,
                        help='Path to a JSONL file for logging intermediate solutions (optional).')

    args = parser.parse_args()
    
    solve_portfolio_optimization(
        instance_path=args.instance_path,
        solution_path=args.solution_path,
        time_limit=args.time_limit,
        log_path=args.log_path
    )


if __name__ == '__main__':
    main()