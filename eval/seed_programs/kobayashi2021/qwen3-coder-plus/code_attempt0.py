import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from solution_logger import SolutionLogger


def solve_portfolio_optimization(instance_data, time_limit, log_path):
    # Parse input data
    N = instance_data["N"]
    S = instance_data["S"]
    k = instance_data["k"]
    beta = instance_data["beta"]
    gamma = instance_data["gamma"]
    mu_bar = instance_data["mu_bar"]
    mu = np.array(instance_data["mu"])
    p_s = instance_data["p_s"]
    scenarios = np.array(instance_data["scenarios"])

    # Initialize the logger if a log path is provided
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    # Create Gurobi model
    model = gp.Model("PortfolioOptimization")
    
    # Set time limit
    model.setParam('TimeLimit', time_limit)
    
    # Variables
    x = model.addVars(N, lb=0.0, name="weights")  # Portfolio weights
    z = model.addVars(N, vtype=GRB.BINARY, name="inclusion")  # Inclusion decisions
    a = model.addVar(name="VaR_threshold", lb=-GRB.INFINITY)  # VaR threshold
    v = model.addVar(lb=0.0, name="CVaR_contribution")  # Tail-risk contribution
    
    # Constraints
    
    # Budget constraint: sum of weights must equal 1
    model.addConstr(gp.quicksum(x[i] for i in range(N)) == 1.0, name="budget")
    
    # Expected return constraint: sum(weight * expected_return) >= mu_bar
    model.addConstr(gp.quicksum(mu[i] * x[i] for i in range(N)) >= mu_bar, 
                   name="expected_return")
    
    # Cardinality constraint: at most k assets can be included
    model.addConstr(gp.quicksum(z[i] for i in range(N)) <= k, name="cardinality")
    
    # Relationship between inclusion and weight: if z[i] == 0, then x[i] == 0
    for i in range(N):
        model.addConstr(x[i] <= z[i], name=f"inclusion_weight_{i}")
    
    # Constraint defining v: it must be >= the CVaR calculation
    # CVaR = VaR + 1/(1-beta) * Sum_s p_s * max{-(return_s' * x) - VaR, 0}
    # We reformulate by introducing variables q_s for each scenario such that
    # q_s >= -(scenario_s * x) - VaR   and     q_s >= 0  (meaning q_s >= max{-(scenario_s * x) - VaR, 0})
    # Then v >= 1/(1-beta) * Sum_s (p_s * q_s)
    
    q = model.addVars(S, lb=0.0,
                      name="excess_loss")  # These capture the excess loss for each scenario
    
    for s in range(S):
        scenario_returns = scenarios[s]
        # Compute portfolio returns (negative because we want losses)
        portfolio_return = gp.quicksum(scenario_returns[i] * x[i] for i in range(N))
        
        # Define the excess loss variables: q >= (-portfolio_return) - a  ->  q >= (a - portfolio_return)
        # and q >= 0 (already set with lower bound)
        model.addConstr(q[s] >= a - portfolio_return, name=f"excess_loss_scenario_{s}_1")
    
    # Link v to the scenario-excess-losses
    cvar_expression = gp.quicksum(p_s * q[s] for s in range(S)) / (1.0 - beta)
    model.addConstr(v >= cvar_expression, name="cvar_constraint")

    # Objective function: minimize (regularization term) + a + v
    reg_term = 1.0 / (2 * gamma) * gp.quicksum(x[i] * x[i] for i in range(N))
    model.setObjective(reg_term + a + v, GRB.MINIMIZE)
    
    # Optimize
    model.optimize()
    
    # Extract the solution
    if model.SolCount > 0:  # Check if a solution was found
        solution_weights = [x[i].X for i in range(N)]
        solution_inclusion = [int(z[i].X) for i in range(N)]
        solution_a = a.X
        solution_v = v.X
        obj_value = model.ObjVal
        
        if obj_value == float("inf") or obj_value == float("-inf"):
            raise ValueError("The model did not converge to a valid finite objective value.")
            
        result = {
            "objective_value": obj_value,
            "x": solution_weights,
            "z": solution_inclusion,
            "a": solution_a,
            "v": solution_v
        }
    else:
        raise ValueError("Model did not find any feasible solution within the given time limit.")
        
    return result


def main():
    parser = argparse.ArgumentParser(description="Solve Portfolio Optimization Problem with Cardinality and Risk Control")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON file containing the problem instance.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path where the final solution JSON file must be written.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None,
                        help="Path to a JSONL file for logging intermediate solutions (optional).")
    
    args = parser.parse_args()
    
    # Load the instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Solve the problem
    solution = solve_portfolio_optimization(
        instance_data=instance_data,
        time_limit=args.time_limit,
        log_path=args.log_path
    )
    
    # Write the solution to the specified output path
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()