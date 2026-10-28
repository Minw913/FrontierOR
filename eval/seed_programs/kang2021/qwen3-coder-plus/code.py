import argparse
import json
import math
from itertools import product
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_instance(instance_data, time_limit, log_path):
    # Parse instance data
    num_customers = instance_data["num_customers"]
    num_nodes = instance_data["num_nodes"]
    depot_index = instance_data["depot_index"]
    customer_indices = instance_data["customer_indices"]
    
    truck_travel_times = instance_data["truck_travel_times"]
    drone_battery_consumption = instance_data["drone_battery_consumption"]
    drone_delivery_time = instance_data["drone_delivery_time"]
    drone_battery_capacities = instance_data["drone_battery_capacities"]
    num_drones = instance_data["parameters"]["num_drones"]

    # Create Gurobi model
    model = gp.Model("DroneDelivery")
    model.setParam('TimeLimit', time_limit)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)

    # Define sets
    customers = customer_indices
    nodes = list(range(num_nodes))
    # Virtual sink node is num_nodes (same as depot physically)
    all_nodes = nodes + [num_nodes]
    
    # Variables
    # x[i,j] = 1 if truck travels from i to j
    x = {}
    for i in all_nodes:
        for j in all_nodes:
            if i != j:
                # Only allow return to virtual sink from actual depot
                if j == num_nodes:
                    if i != depot_index:
                        continue
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    
    # y[d, k, i] = 1 if drone d serves customer k from node i
    y = {}
    for d in range(num_drones):
        for k in customers:
            for i in nodes:
                y[d, k, i] = model.addVar(vtype=GRB.BINARY, name=f"y_{d}_{k}_{i}")
    
    # w[i] = waiting time at node i
    w = {}
    for i in nodes:
        w[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"w_{i}")

    # z[i] = 1 if node i is visited by truck
    z = {}
    for i in nodes:
        if i != depot_index:
            z[i] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}")
        else:
            # Depot is always visited
            z[i] = model.addVar(vtype=GRB.BINARY, lb=1, ub=1, name=f"z_{i}")

    # Constraints
    # Each customer is served exactly once
    for k in customers:
        model.addConstr(gp.quicksum(y[d, k, i] for d in range(num_drones) for i in nodes) == 1)

    # Link truck visit variables with arc variables
    for i in nodes:
        if i != depot_index:
            model.addConstr(z[i] == gp.quicksum(x[j, i] for j in all_nodes if j != i and (j, i) in x))
        # For depot, we don't need to link since it's always visited

    # Flow conservation for truck
    for i in nodes:
        if i == depot_index:
            # Outflow from depot
            model.addConstr(gp.quicksum(x[depot_index, j] for j in nodes if depot_index != j and (depot_index, j) in x) == 1)
            # Inflow to virtual sink
            model.addConstr(gp.quicksum(x[j, num_nodes] for j in nodes if j != num_nodes and (j, num_nodes) in x) == 1)
        else:
            inflow = gp.quicksum(x[j, i] for j in all_nodes if j != i and (j, i) in x)
            outflow = gp.quicksum(x[i, j] for j in all_nodes if i != j and (i, j) in x)
            model.addConstr(inflow == z[i])
            model.addConstr(outflow == z[i])

    # Subtour elimination constraints (MTZ-like formulation)
    u = {}  # Position variables for subtour elimination
    for i in nodes:
        if i != depot_index:
            u[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=1, ub=len(customer_indices), name=f"u_{i}")
    
    for i in nodes:
        for j in nodes:
            if i != depot_index and j != depot_index and i != j and (i, j) in x:
                # Prevent subtours not containing depot
                model.addConstr(u[i] - u[j] + len(customer_indices) * x[i, j] <= len(customer_indices) - 1)

    # Battery capacity constraints for each drone
    for d in range(num_drones):
        model.addConstr(gp.quicksum(
            drone_battery_consumption[d][i][k] * y[d, k, i] 
            for k in customers for i in nodes
            if (d, k, i) in y
        ) <= drone_battery_capacities[d])

    # Waiting time constraints
    for i in nodes:
        for d in range(num_drones):
            # Waiting time at node i must be at least the total delivery time of drone d from i
            total_time_for_drone_d_at_i = gp.quicksum(
                drone_delivery_time[d][i][k] * y[d, k, i] for k in customers
                if (d, k, i) in y
            )
            model.addConstr(w[i] >= total_time_for_drone_d_at_i)

    # Objective function: minimize total completion time
    truck_travel_time = gp.quicksum(
        truck_travel_times[i][j] * x[i, j] 
        for i in all_nodes for j in all_nodes 
        if i != j and (i, j) in x
    )
    waiting_time = gp.quicksum(w[i] for i in nodes)
    model.setObjective(truck_travel_time + waiting_time, GRB.MINIMIZE)

    # Callback to log intermediate solutions
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                # Get current solution values
                truck_route = []
                drone_assignments = []
                waiting_times = {}

                for i in all_nodes:
                    for j in all_nodes:
                        if (i, j) in x and model.cbGetSolution(x[i, j]) > 0.5:
                            truck_route.append([i, j])

                for d in range(num_drones):
                    for k in customers:
                        for i in nodes:
                            if model.cbGetSolution(y[d, k, i]) > 0.5:
                                drone_assignments.append({
                                    "drone": d,
                                    "dispatch_node": i,
                                    "delivery_node": k
                                })

                for i in nodes:
                    val = model.cbGetSolution(w[i])
                    if val > 1e-6:  # Only include non-negligible waiting times
                        waiting_times[str(i)] = val
                
                solution_dict = {
                    "objective_value": obj,
                    "truck_route": truck_route,
                    "drone_assignments": drone_assignments,
                    "waiting_times": waiting_times
                }
                
                logger.log_solution(obj, solution_dict)

    # Optimize the model
    model.optimize(callback)

    # Extract solution
    if model.SolCount > 0:
        # Build solution
        solution = {
            "objective_value": model.objVal,
            "truck_route": [],
            "drone_assignments": [],
            "waiting_times": {}
        }

        # Extract truck route
        for i in all_nodes:
            for j in all_nodes:
                if (i, j) in x and x[i, j].x > 0.5:
                    solution["truck_route"].append([i, j])

        # Extract drone assignments
        for d in range(num_drones):
            for k in customers:
                for i in nodes:
                    if y[d, k, i].x > 0.5:
                        solution["drone_assignments"].append({
                            "drone": d,
                            "dispatch_node": i,
                            "delivery_node": k
                        })

        # Extract waiting times
        for i in nodes:
            if w[i].x > 1e-6:  # Only include non-negligible waiting times
                solution["waiting_times"][str(i)] = w[i].x

        return solution
    else:
        # No feasible solution found
        return {
            "objective_value": float('inf'),
            "truck_route": [],
            "drone_assignments": [],
            "waiting_times": {}
        }


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

    # Solve the instance
    solution = solve_instance(instance_data, args.time_limit, args.log_path)

    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()