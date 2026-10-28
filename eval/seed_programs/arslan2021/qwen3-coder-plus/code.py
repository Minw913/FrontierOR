import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time


def solve_instance(instance_data, time_limit):
    # Extract problem parameters
    knapsack_capacity = instance_data["problem_parameters"]["knapsack_capacity"]
    uncertainty_budget = instance_data["problem_parameters"]["uncertainty_budget"]
    
    items = instance_data["items"]
    n_items = len(items)
    
    # Extract item properties
    weights = [item["weight"] for item in items]
    nominal_profits = [item["nominal_profit"] for item in items]
    max_degradations = [item["max_degradation"] for item in items]
    outsource_penalties = [item["outsource_penalty"] for item in items]
    repair_capacities = [item["repair_capacity"] for item in items]
    
    # Create model
    model = gp.Model("AdversarialKnapsack")
    
    # Set Gurobi parameters
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    
    # First stage variables: x[i] = 1 if item i is selected, 0 otherwise
    x = model.addVars(n_items, vtype=GRB.BINARY, name="x")
    
    # Second stage variables: y[i], z[i] - continuous variables representing fractions of production and repair
    y = model.addVars(n_items, lb=0, ub=1, name="y")  # Production fraction
    z = model.addVars(n_items, lb=0, ub=1, name="z")  # Repair fraction
    
    # Uncertainty variables: d[i] - degradation factors
    d = model.addVars(n_items, lb=0, ub=1, name="d")
    
    # Constraints
    # 1. Repair fraction cannot exceed production fraction
    model.addConstrs((z[i] <= y[i] for i in range(n_items)), name="repair_not_exceed_production")
    
    # 2. Capacity constraint: total capacity used by production and repair
    model.addConstr(
        gp.quicksum(weights[i] * y[i] + repair_capacities[i] * z[i] for i in range(n_items)) 
        <= knapsack_capacity,
        name="capacity_constraint"
    )
    
    # 3. Item can only be produced if selected
    model.addConstrs((y[i] <= x[i] for i in range(n_items)), name="production_if_selected")
    
    # 4. Item can only be repaired if selected
    model.addConstrs((z[i] <= x[i] for i in range(n_items)), name="repair_if_selected")
    
    # 5. Budget constraint on total degradation
    model.addConstr(gp.quicksum(d[i] for i in range(n_items)) <= uncertainty_budget, name="budget_constraint")
    
    # Objective: minimize the robust objective (first-stage cost + worst-case second-stage cost)
    # First stage cost: sum of (outsource_penalty[i] - nominal_profit[i]) for selected items
    first_stage_cost = gp.quicksum((outsource_penalties[i] - nominal_profits[i]) * x[i] for i in range(n_items))
    
    # The second stage cost under adversarial choice of d is:
    # sum over i of [max_degradation[i] * d[i] * y[i] - outsource_penalty[i] * (1 - y[i]) - max_degradation[i] * d[i] * z[i]]
    # Simplified: sum over i of [max_degradation[i] * d[i] * (y[i] - z[i]) + outsource_penalty[i] * y[i]] - sum of outsource_penalty[i]
    # For robust optimization we need to consider the worst case which means maximizing this expression w.r.t. d
    # To handle this robustly, we reformulate using duality or robust optimization techniques
    # Here we use the fact that the adversary maximizes the second stage cost given our decisions
    
    # We'll reformulate the robust counterpart using strong duality
    # The inner maximization problem (adversary's problem) is:
    # max sum_i [max_degradation[i] * d[i] * (y[i] - z[i])] 
    # s.t. sum_i d[i] <= uncertainty_budget, 0 <= d[i] <= 1
    
    # Using dual of the inner problem, we introduce auxiliary variable theta and constraints
    # max sum_i [max_degradation[i] * d[i] * (y[i] - z[i])] <= theta
    # For the robust counterpart we need to ensure this holds at optimality
    
    # Let's use the reformulation where we model the worst case explicitly
    # We'll introduce a variable for the second stage cost and constrain it appropriately
    worst_case_second_stage = model.addVar(name="worst_case_second_stage")
    
    # The adversary's problem: maximize sum_i [max_degradation[i] * d[i] * (y[i] - z[i])]
    # This is linear in d and can be reformulated using duality
    # We introduce lambda >= 0 (dual variable for budget constraint) and mu[i] >= 0 (for upper bound d[i] <= 1)
    # Dual objective: lambda * uncertainty_budget + sum_i mu[i]
    # s.t. lambda + mu[i] >= max_degradation[i] * (y[i] - z[i])
    # Since we want to maximize the original (minimize the negative), we handle it in the robust constraint
    
    # Introduce dual variables
    lambda_dual = model.addVar(lb=0, name="lambda_dual")
    mu = model.addVars(n_items, lb=0, name="mu")
    
    # Dual constraints: lambda + mu[i] >= max_degradation[i] * (y[i] - z[i])
    model.addConstrs(
        (lambda_dual + mu[i] >= max_degradations[i] * (y[i] - z[i]) for i in range(n_items)),
        name="dual_constraints"
    )
    
    # Worst case cost = lambda * uncertainty_budget + sum(mu[i]) + sum(outsource_penalty[i] * y[i])
    model.addConstr(
        worst_case_second_stage >= 
        lambda_dual * uncertainty_budget + gp.quicksum(mu[i] for i in range(n_items)) +
        gp.quicksum(outsource_penalties[i] * y[i] for i in range(n_items)) -
        gp.quicksum(nominal_profits[i] * y[i] for i in range(n_items)),
        name="worst_case_constraint"
    )
    
    # The overall objective is first_stage_cost + worst_case_second_stage
    model.setObjective(first_stage_cost + worst_case_second_stage, GRB.MINIMIZE)
    
    # Optimize the model
    start_time = time.time()
    model.optimize()
    end_time = time.time()
    
    # Check if we have a feasible solution
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        # Extract solution
        x_values = {i: int(x[i].X) for i in range(n_items)}
        
        # Prepare solution dict
        solution_dict = {
            "objective_value": model.ObjVal,
            "x": {str(i): x_values[i] for i in range(n_items)}
        }
        
        return solution_dict
    else:
        raise Exception(f"Model status: {model.status}")


def main():
    parser = argparse.ArgumentParser(description='Solve Adversarial Knapsack Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to save the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log file for intermediate solutions')
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
    
    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Solve the instance
    solution = solve_instance(instance_data, args.time_limit)
    
    # Log the final solution if logger is available
    if logger:
        logger.log_solution(solution["objective_value"], solution)
    
    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()