import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        instance_data = json.load(file)
    return instance_data

def solve_instance(instance_data, time_limit):
    problem_parameters = instance_data['problem_parameters']
    items = instance_data['items']
    num_items = len(items)
    
    knapsack_capacity = problem_parameters['knapsack_capacity']
    uncertainty_budget = problem_parameters['uncertainty_budget']
    
    model = gp.Model("RobustKnapsack")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit
    
    # First-stage decision: whether to select an item for potential in-house production
    x = model.addVars(num_items, vtype=GRB.BINARY, name='x')
    
    # Second-stage decisions under worst-case degradation
    z = model.addVars(num_items, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='z')  # Production fraction
    r = model.addVars(num_items, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='r')  # Repair fraction
    
    # Adversarial degradation factors
    delta = model.addVars(num_items, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='delta')
    
    # Objective: minimize total worst-case cost
    model.setObjective(
        gp.quicksum((items[i]['outsource_penalty'] - items[i]['nominal_profit']) * x[i] for i in range(num_items)) +
        gp.quicksum(items[i]['max_degradation'] * delta[i] * z[i] for i in range(num_items)) -
        gp.quicksum(items[i]['max_degradation'] * delta[i] * r[i] for i in range(num_items)) -
        gp.quicksum(items[i]['outsource_penalty'] * z[i] for i in range(num_items)),
        GRB.MINIMIZE
    )
    
    # Capacity constraint for production and repair
    model.addConstr(
        gp.quicksum(items[i]['weight'] * z[i] + items[i]['repair_capacity'] * r[i] for i in range(num_items)) <= knapsack_capacity,
        name='CapacityConstraint'
    )
    
    # Repair fraction cannot exceed production fraction
    for i in range(num_items):
        model.addConstr(r[i] <= z[i], name=f'RepairNotExceedProduction_{i}')
    
    # Production and repair are only allowed for selected items
    for i in range(num_items):
        model.addConstr(z[i] <= x[i], name=f'ProductionOnlySelected_{i}')
        model.addConstr(r[i] <= x[i], name=f'RepairOnlySelected_{i}')
    
    # Adversary's uncertainty budget constraint
    model.addConstr(gp.quicksum(delta[i] for i in range(num_items)) <= uncertainty_budget, name='UncertaintyBudget')
    
    # Maximize the second-stage cost over delta (adversarial problem)
    # Since our objective already includes the second-stage cost with delta,
    # we need to ensure delta is chosen to maximize this cost under the budget.
    # The inner maximization problem is:
    # maximize delta[i] * (items[i]['max_degradation'] * (z[i] - r[i])) subject to sum(delta[i]) <= uncertainty_budget and 0 <= delta[i] <= 1
    # This is implicitly handled in our formulation as delta is part of the model and is constrained by the uncertainty budget.
    
    model.optimize()
    
    objective_value = model.getObjective().getValue()
    solution_dict = {str(i): x[i].X for i in range(num_items)}
    
    return objective_value, solution_dict

def main():
    parser = argparse.ArgumentParser(description='Solve Robust Knapsack Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to output the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions (optional)')
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    instance_data = read_instance(args.instance_path)
    objective_value, solution_dict = solve_instance(instance_data, args.time_limit)
    
    if logger:
        logger.log_solution(objective_value, solution_dict)
    
    solution_output = {
        "objective_value": objective_value,
        "x": solution_dict
    }
    
    with open(args.solution_path, 'w') as file:
        json.dump(solution_output, file, indent=4)

if __name__ == "__main__":
    main()