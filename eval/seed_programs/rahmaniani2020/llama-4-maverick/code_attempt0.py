import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance_data(instance_path):
    with open(instance_path, 'r') as file:
        instance_data = json.load(file)
    return instance_data

def solve(instance_data, time_limit, log_path=None):
    num_facilities = instance_data['num_facilities']
    num_customers = instance_data['num_customers']
    num_scenarios = instance_data['num_scenarios']
    fixed_costs = instance_data['facilities']['fixed_costs']
    capacities = instance_data['facilities']['capacities']
    routing_costs = instance_data['routing_costs']
    probabilities = instance_data['scenarios']['probabilities']
    demands = instance_data['scenarios']['demands']

    # Create a new Gurobi model
    model = gp.Model()
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = False

    # Facility opening variables
    y = model.addVars(num_facilities, vtype=GRB.BINARY, name='y')

    # Flow variables for each scenario
    flow = model.addVars(num_scenarios, num_facilities, num_customers, vtype=GRB.CONTINUOUS, name='flow')

    # Objective function: minimize total fixed costs + expected routing costs
    model.setObjective(
        gp.quicksum(fixed_costs[i] * y[i] for i in range(num_facilities)) +
        gp.quicksum(probabilities[s] * gp.quicksum(routing_costs[i][j] * flow[s, i, j] for i in range(num_facilities) for j in range(num_customers)) for s in range(num_scenarios)),
        GRB.MINIMIZE
    )

    # Constraints
    for s in range(num_scenarios):
        # Demand satisfaction for each customer in each scenario
        for j in range(num_customers):
            model.addConstr(gp.quicksum(flow[s, i, j] for i in range(num_facilities)) >= demands[s][j], f'demand_s{s}_c{j}')
        
        # Capacity constraints for each facility in each scenario
        for i in range(num_facilities):
            model.addConstr(flow.sum(s, i, '*') <= y[i] * capacities[i], f'capacity_s{s}_f{i}')

    # Total capacity of opened facilities must be at least as large as the maximum total demand across all scenarios
    max_total_demand = max(sum(demands[s]) for s in range(num_scenarios))
    model.addConstr(gp.quicksum(capacities[i] * y[i] for i in range(num_facilities)) >= max_total_demand, 'total_capacity')

    # Initialize logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    # Optimize the model
    model.optimize()

    # Log the final solution if it's feasible and better than the previous best
    if model.SolCount > 0:
        objective_value = model.getObjective().getValue()
        y_values = [int(y[i].x) for i in range(num_facilities)]
        if logger:
            logger.log(objective_value)
        return objective_value, y_values
    else:
        raise Exception("No feasible solution found.")

def write_solution(solution_path, objective_value, y_values):
    solution_data = {
        'objective_value': objective_value,
        'y': y_values
    }
    with open(solution_path, 'w') as file:
        json.dump(solution_data, file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    instance_data = read_instance_data(args.instance_path)
    objective_value, y_values = solve(instance_data, args.time_limit, args.log_path)
    write_solution(args.solution_path, objective_value, y_values)

if __name__ == "__main__":
    main()