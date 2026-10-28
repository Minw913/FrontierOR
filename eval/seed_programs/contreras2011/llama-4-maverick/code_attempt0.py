import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve_hub_location(instance, time_limit, logger):
    n = instance['n']
    collection_cost_chi = instance['cost_parameters']['collection_cost_chi']
    transfer_cost_alpha = instance['cost_parameters']['transfer_cost_alpha']
    distribution_cost_delta = instance['cost_parameters']['distribution_cost_delta']
    flow_matrix = np.array(instance['flow_matrix'])
    distance_matrix = np.array(instance['distance_matrix'])
    outgoing_flow = np.array(instance['outgoing_flow'])
    setup_costs = np.array(instance['setup_costs'])
    capacities = np.array(instance['capacities'])
    total_flow_D = instance['total_flow_D']

    model = gp.Model('Hub Location Problem')
    model.Params.LogToConsole = 0  # Disable Gurobi console output

    # Decision Variables
    z = model.addVars(n, vtype=GRB.BINARY, name='z')  # Hub selection
    y = model.addVars(n, n, vtype=GRB.BINARY, name='y')  # Assignment of nodes to hubs
    x = model.addVars(n, n, n, n, vtype=GRB.CONTINUOUS, name='x')  # Flow routing

    # Objective Function
    model.setObjective(
        gp.quicksum(setup_costs[i] * z[i] for i in range(n)) +
        gp.quicksum(
            flow_matrix[i, j] * (
                collection_cost_chi * distance_matrix[i, k] * y[i, k] +
                transfer_cost_alpha * distance_matrix[k, m] * y[i, k] * y[j, m] +
                distribution_cost_delta * distance_matrix[m, j] * y[j, m]
            ) for i in range(n) for j in range(n) for k in range(n) for m in range(n)
        ),
        GRB.MINIMIZE
    )

    # Constraints
    model.addConstrs((y[i, k] <= z[k] for i in range(n) for k in range(n)), name='Assign_to_Open_Hub')
    model.addConstrs((gp.quicksum(y[i, k] for k in range(n)) == 1 for i in range(n)), name='Assign_Each_Node_To_One_Hub')
    model.addConstrs((y[k, k] == z[k] for k in range(n)), name='Hub_Assigned_To_Itself')
    model.addConstrs((x[i, j, k, m] <= y[i, k] for i in range(n) for j in range(n) for k in range(n) for m in range(n)), name='Routing_Consistent_With_Origin_Assignment')
    model.addConstrs((x[i, j, k, m] <= y[j, m] for i in range(n) for j in range(n) for k in range(n) for m in range(n)), name='Routing_Consistent_With_Destination_Assignment')
    model.addConstrs((gp.quicksum(x[i, j, k, m] for k in range(n) for m in range(n)) == 1 for i in range(n) for j in range(n)), name='Flow_Routed_Through_Exactly_One_Hub_Pair')
    model.addConstr((gp.quicksum(outgoing_flow[i] * y[i, k] for i in range(n) for k in range(n)) <= gp.quicksum(capacities[k] * z[k] for k in range(n))), name='Total_Capacity_Sufficient')
    model.addConstrs((gp.quicksum(flow_matrix[j, i] for j in range(n) if i != j) * y[i, k] <= capacities[k] * z[k] for i in range(n) for k in range(n)), name='Hub_Capacity_Constraint')

    # Simplify the x variable by adding a constraint that forces x to be consistent with y
    model.addConstrs((gp.quicksum(x[i, j, k, m] for m in range(n)) == y[i, k] for i in range(n) for j in range(n) for k in range(n)), name='Simplify_x_1')
    model.addConstrs((gp.quicksum(x[i, j, k, m] for k in range(n)) == y[j, m] for i in range(n) for j in range(n) for m in range(n)), name='Simplify_x_2')

    # Time Limit
    model.Params.TimeLimit = time_limit

    # Optimize
    model.optimize()
    if logger:
        logger.log(model.getObjective().getValue())

    # Extract Solution
    objective_value = model.getObjective().getValue()
    hubs = [i for i in range(n) if z[i].x > 0.5]
    assignment = [int(np.argmax([y[i, k].x for k in range(n)])) for i in range(n)]

    return objective_value, hubs, assignment

def main():
    parser = argparse.ArgumentParser(description='Solve Hub Location Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions (optional)')
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    objective_value, hubs, assignment = solve_hub_location(instance, args.time_limit, logger)

    solution = {
        'objective_value': objective_value,
        'hubs': hubs,
        'assignment': assignment
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == '__main__':
    main()