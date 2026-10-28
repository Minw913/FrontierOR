import argparse
import json
import math
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from solution_logger import SolutionLogger

def solve_hub_location_problem(instance_data, time_limit, log_path):
    n = instance_data["n"]
    chi = instance_data["cost_parameters"]["collection_cost_chi"]
    alpha = instance_data["cost_parameters"]["transfer_cost_alpha"]
    delta = instance_data["cost_parameters"]["distribution_cost_delta"]
    flow_matrix = np.array(instance_data["flow_matrix"])
    distance_matrix = np.array(instance_data["distance_matrix"])
    setup_costs = np.array(instance_data["setup_costs"])
    capacities = np.array(instance_data["capacities"])
    outgoing_flows = np.array(instance_data["outgoing_flow"])
    total_flow = instance_data["total_flow_D"]

    # Create model
    model = gp.Model("hub_location")
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPGap', 1e-8)  # Allow very small optimality gaps
    
    # Decision variables
    # y[k] = 1 if hub k is opened, 0 otherwise
    y = model.addVars(n, vtype=GRB.BINARY, name="y")

    # x[i,k] = 1 if node i is assigned to hub k
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")

    # z[i,j,k,m] = flow from i to j via hubs k and m (used in the linearization)
    z = model.addVars(n, n, n, n, vtype=GRB.CONTINUOUS, lb=0, name="z")

    # Objective function: minimize total cost
    # Cost = Sum of setup costs + routing costs
    setup_cost = gp.quicksum(setup_costs[k] * y[k] for k in range(n))
    
    # Routing cost involves sum over all origins i, destinations j, and hub pairs k,m
    routing_cost = gp.quicksum(
        flow_matrix[i][j] * (
            chi * distance_matrix[i][k] * x[i, k] +
            delta * distance_matrix[m][j] * x[j, m] +
            alpha * distance_matrix[k][m] * z[i, j, k, m]
        )
        for i in range(n)
        for j in range(n)
        for k in range(n)
        for m in range(n)
    )

    model.setObjective(setup_cost + routing_cost, GRB.MINIMIZE)

    # Additional constraints to link z with x variables correctly
    # z[i, j, k, m] can only be > 0 if x[i, k] = 1 AND x[j, m] = 1
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for m in range(n):
                    # z[i,j,k,m] <= x[i,k]
                    model.addConstr(z[i, j, k, m] <= x[i, k])
                    
                    # z[i,j,k,m] <= x[j,m]
                    model.addConstr(z[i, j, k, m] <= x[j, m])
                    
                    # z[i,j,k,m] >= x[i,k] + x[j,m] - 1
                    model.addConstr(z[i, j, k, m] >= x[i, k] + x[j, m] - 1)

    # Each node must be assigned to exactly one hub
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, k] for k in range(n)) == 1)

    # A node can only be assigned to an open hub
    for i in range(n):
        for k in range(n):
            model.addConstr(x[i, k] <= y[k])

    # Each hub is assigned to itself
    for k in range(n):
        model.addConstr(x[k, k] <= y[k])  # If k is assigned to k, then k must be open

    # Capacity constraint: Total incoming flow to hub k <= capacity of hub k
    for k in range(n):
        total_incoming = gp.quicksum(flow_matrix[i][j] * x[j, k] for i in range(n) for j in range(n))
        model.addConstr(total_incoming <= capacities[k] * y[k])

    # Capacity constraint: Total capacity of opened hubs must be at least total flow
    model.addConstr(gp.quicksum(capacities[k] * y[k] for k in range(n)) >= total_flow)

    # Set callback to handle logging of intermediate solutions
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    # Optimize the model
    model.optimize(mycallback)

    # Extract solution values
    hubs_open = [i for i in range(n) if y[i].X > 0.99]  # Check if the variable is approximately 1
    assignment_result = [None] * n
    for i in range(n):
        for k in range(n):
            if x[i, k].X > 0.99:
                assignment_result[i] = k
                break
            
    # Calculate the actual objective value for correctness check
    actual_obj_value = model.ObjVal if model.SolCount > 0 else float('inf')

    # Prepare result
    result = {
        "objective_value": actual_obj_value,
        "hubs": hubs_open,
        "assignment": assignment_result
    }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    
    args = parser.parse_args()
    
    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Solve the problem
    solution = solve_hub_location_problem(instance_data, args.time_limit, args.log_path)

    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()