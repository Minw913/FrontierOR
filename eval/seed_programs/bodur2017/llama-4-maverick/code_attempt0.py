import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        return json.load(file)

def solve(instance):
    num_facilities = instance['num_facilities']
    num_customers = instance['num_customers']
    num_scenarios = instance['num_scenarios']
    facilities = instance['facilities']
    customers = instance['customers']
    scenarios = instance['scenarios']
    transportation_costs = instance['transportation_costs']

    model = gp.Model()
    model.modelSense = GRB.MINIMIZE

    # Facility opening variables
    x = model.addVars(num_facilities, vtype=GRB.BINARY, name='x')

    # Allocation variables
    y = model.addVars(num_scenarios, num_facilities, num_customers, vtype=GRB.CONTINUOUS, name='y')

    # Objective function: minimize total expected cost
    total_opening_cost = gp.quicksum(facility['opening_cost'] * x[facility['id']] for facility in facilities)
    total_expected_transportation_cost = gp.quicksum(scenario['probability'] * 
                                                     gp.quicksum(transportation_costs[facility][customer] * y[scenario['id'], facility, customer] 
                                                                 for facility in range(num_facilities) for customer in range(num_customers)) 
                                                     for scenario in scenarios)
    model.setObjective(total_opening_cost + total_expected_transportation_cost)

    # Constraints: 
    # 1. Demand satisfaction
    for scenario in scenarios:
        for customer in range(num_customers):
            model.addConstr(gp.quicksum(y[scenario['id'], facility, customer] for facility in range(num_facilities)) >= scenario['demands'][customer])

    # 2. Capacity constraints
    for scenario in scenarios:
        for facility in range(num_facilities):
            model.addConstr(gp.quicksum(y[scenario['id'], facility, customer] for customer in range(num_customers)) <= facilities[facility]['capacity'] * x[facility])

    # 3. Total capacity of open facilities must be at least as large as the maximum total demand across all scenarios
    max_total_demand = max(sum(scenario['demands']) for scenario in scenarios)
    total_capacity = gp.quicksum(facility['capacity'] * x[facility['id']] for facility in facilities)
    model.addConstr(total_capacity >= max_total_demand)

    return model, x, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    model, x, y = solve(instance)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model.Params.TimeLimit = args.time_limit
    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        solution = {
            'objective_value': model.objVal,
            'open_facilities': [i for i in range(instance['num_facilities']) if x[i].x > 0.5],
            'x': {str(i): x[i].x for i in range(instance['num_facilities'])},
            'y': {str(scenario['id']): {str(facility): {str(customer): y[scenario['id'], facility, customer].x for customer in range(instance['num_customers'])} for facility in range(instance['num_facilities'])} for scenario in instance['scenarios']}
        }
        with open(args.solution_path, 'w') as file:
            json.dump(solution, file, indent=4)
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()