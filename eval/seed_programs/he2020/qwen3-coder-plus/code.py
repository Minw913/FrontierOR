import argparse
import json
import math
import time
from typing import Dict, List, Tuple
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_vrptw_ic(instance_data: dict, time_limit: int):
    """
    Solve the Vehicle Routing Problem with Time Windows and Inconvenience Costs (VRPTW-IC)
    using Gurobi optimization.
    
    Args:
        instance_data: Dictionary containing the problem instance data
        time_limit: Maximum allowed computation time in seconds
    
    Returns:
        A tuple (objective_value, routes, schedule) where:
        - objective_value: Total cost of the solution
        - routes: List of routes, each being a list of location indices
        - schedule: Dictionary mapping customer IDs to their service start times
    """
    # Extract instance data
    num_customers = instance_data['num_customers']
    num_vehicles = instance_data['num_vehicles']
    vehicle_capacity = instance_data['vehicle_capacity']
    planning_horizon = instance_data['planning_horizon']
    depot_start = instance_data['depot']
    depot_end = instance_data['depot_end']
    customers = instance_data['customers']
    distance_matrix = instance_data['distance_matrix']
    inconvenience_params = instance_data['inconvenience_cost_function']
    
    rho = inconvenience_params['rho']
    
    # Create node list: depot_start (0), customers (1 to n), depot_end (n+1)
    nodes = [depot_start['id']] + [c['id'] for c in customers] + [depot_end['id']]
    n_nodes = len(nodes)
    depot_idx = 0
    end_depot_idx = n_nodes - 1
    
    # Customer indices in the node list
    customer_indices = list(range(1, n_nodes - 1))
    
    # Prepare customer data with node indices
    customer_data = {}
    for i, cust in enumerate(customers):
        node_idx = i + 1  # Node index in our model (1-based after depot_start)
        customer_data[node_idx] = {
            'id': cust['id'],
            'demand': cust['demand'],
            'time_window': cust['time_window'],
            'service_time': cust['service_time'],
            'preferred_time': cust['preferred_time'],
            'idx': node_idx
        }
    
    # Build model
    model = gp.Model("VRPTW_IC")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    
    # Decision variables
    # x[i,j,k]: binary, whether vehicle k travels from i to j
    x = {}
    for k in range(num_vehicles):
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    x[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")
    
    # t[i, k]: continuous, arrival time of vehicle k at node i
    t = {}
    for k in range(num_vehicles):
        for i in range(n_nodes):
            t[i, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"t_{i}_{k}")
    
    # s[i]: continuous, service start time at customer i
    s = {}
    for i in customer_indices:
        s[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"s_{i}")
    
    # q[i, k]: continuous, load of vehicle k when leaving node i
    q = {}
    for k in range(num_vehicles):
        for i in range(n_nodes):
            q[i, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"q_{i}_{k}")
    
    # Inconvenience cost variables
    inconveniences = {}
    for i in customer_indices:
        inconveniences[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"inconvenience_{i}")
    
    model.update()
    
    # Objective function: minimize total routing cost + inconvenience costs
    routing_cost_expr = gp.LinExpr()
    inconvenience_cost_expr = gp.LinExpr()
    
    for k in range(num_vehicles):
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    routing_cost_expr.addTerms(distance_matrix[i][j], x[i, j, k])
    
    for i in customer_indices:
        inconvenience_cost_expr.add(inconveniences[i])
    
    model.setObjective(routing_cost_expr + inconvenience_cost_expr, GRB.MINIMIZE)
    
    # Constraints
    # 1. Each customer is visited exactly once
    for i in customer_indices:
        model.addConstr(
            gp.quicksum(x[i, j, k] for k in range(num_vehicles) 
                       for j in range(n_nodes) if i != j and j != i) == 1,
            name=f"visit_once_{i}"
        )
    
    # 2. Flow conservation
    for k in range(num_vehicles):
        for i in range(n_nodes):
            if i == depot_idx:  # Start depot
                model.addConstr(
                    gp.quicksum(x[i, j, k] for j in range(1, n_nodes)) -
                    gp.quicksum(x[j, i, k] for j in range(1, n_nodes)) == 1,
                    name=f"flow_conservation_out_{k}_{i}"
                )
            elif i == end_depot_idx:  # End depot
                model.addConstr(
                    gp.quicksum(x[j, i, k] for j in range(n_nodes-1)) -
                    gp.quicksum(x[i, j, k] for j in range(n_nodes-1)) == 1,
                    name=f"flow_conservation_in_{k}_{i}"
                )
            else:  # Customer
                model.addConstr(
                    gp.quicksum(x[j, i, k] for j in range(n_nodes) if j != i) ==
                    gp.quicksum(x[i, j, k] for j in range(n_nodes) if j != i),
                    name=f"flow_conservation_customer_{k}_{i}"
                )
    
    # 3. Capacity constraints
    for k in range(num_vehicles):
        model.addConstr(
            gp.quicksum(customer_data[i]['demand'] * 
                       gp.quicksum(x[i, j, k] for j in range(n_nodes) if i != j)
                       for i in customer_indices) <= vehicle_capacity,
            name=f"capacity_{k}"
        )
    
    # 4. Time window constraints
    for i in customer_indices:
        cust_info = customer_data[i]
        model.addConstr(s[i] >= cust_info['time_window'][0], name=f"early_time_{i}")
        model.addConstr(s[i] <= cust_info['time_window'][1], name=f"late_time_{i}")
    
    # 5. Time consistency constraints
    for k in range(num_vehicles):
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    if i in customer_indices and j in customer_indices:
                        # Customer to customer
                        model.addConstr(
                            t[i, k] + customer_data[i]['service_time'] + distance_matrix[i][j] - 
                            (1 - x[i, j, k]) * planning_horizon <= t[j, k],
                            name=f"time_consistency_c2c_{i}_{j}_{k}"
                        )
                    elif i == depot_idx and j in customer_indices:
                        # Depot to customer
                        model.addConstr(
                            depot_start['time_window'][0] + distance_matrix[i][j] - 
                            (1 - x[i, j, k]) * planning_horizon <= t[j, k],
                            name=f"time_consistency_d2c_{i}_{j}_{k}"
                        )
                    elif i in customer_indices and j == end_depot_idx:
                        # Customer to end depot
                        model.addConstr(
                            s[i] + customer_data[i]['service_time'] + distance_matrix[i][j] - 
                            (1 - x[i, j, k]) * planning_horizon <= t[j, k],
                            name=f"time_consistency_c2d_{i}_{j}_{k}"
                        )
    
    # 6. Service start time relationship with arrival time
    for i in customer_indices:
        model.addConstr(s[i] >= t[i, 0], name=f"service_after_arrival_{i}_0")
        for k in range(1, num_vehicles):
            # If customer i is not served by vehicle k, this constraint doesn't matter
            # We'll use big-M to handle this properly
            is_served_by_k = gp.quicksum(x[i, j, k] for j in range(n_nodes) if i != j)
            model.addConstr(s[i] >= t[i, k] - planning_horizon * (1 - is_served_by_k), 
                           name=f"service_after_arrival_{i}_{k}")
        
        # Also ensure that s[i] >= arrival time at i for whichever vehicle serves it
        for k in range(num_vehicles):
            model.addConstr(s[i] >= t[i, k] - planning_horizon * (1 - gp.quicksum(
                x[j, i, k] for j in range(n_nodes) if j != i)), 
                           name=f"service_after_arrival_link_{i}_{k}")
    
    # 7. Load constraints
    for k in range(num_vehicles):
        # At depot start, load is 0
        model.addConstr(q[depot_idx, k] == 0, name=f"load_depot_start_{k}")
        
        # Flow of load
        for i in customer_indices:
            model.addConstr(
                q[i, k] >= (customer_data[i]['demand'] * 
                           gp.quicksum(x[j, i, k] for j in range(n_nodes) if j != i)),
                name=f"load_flow_min_{i}_{k}"
            )
            
            # Update load after serving customer i
            for j in range(n_nodes):
                if i != j:
                    model.addConstr(
                        q[i, k] + customer_data[i]['demand'] * x[i, j, k] <= q[j, k] + 
                        vehicle_capacity * (1 - x[i, j, k]),
                        name=f"load_update_{i}_{j}_{k}"
                    )
    
    # 8. Inconvenience cost definition
    for i in customer_indices:
        preferred_time = customer_data[i]['preferred_time']
        # This is a quadratic term, so we linearize it using piecewise linear approximation
        # For now, we'll use a direct quadratic approach which Gurobi supports
        model.addQConstr(
            inconveniences[i] >= rho * (s[i] - preferred_time) * (s[i] - preferred_time),
            name=f"inconvenience_def_{i}"
        )
    
    # Solve the model
    model.optimize()
    
    # Extract solution
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        routes = []
        schedule = {}
        
        # Build routes and schedules
        for k in range(num_vehicles):
            route = [depot_idx]
            current_node = depot_idx
            
            while True:
                next_node = None
                for j in range(n_nodes):
                    if current_node != j and x[current_node, j, k].x > 0.9:
                        next_node = j
                        break
                
                if next_node is None or next_node == end_depot_idx:
                    route.append(end_depot_idx)
                    break
                else:
                    route.append(next_node)
                    # Record service start time for customers
                    if next_node in customer_indices:
                        schedule[nodes[next_node]] = round(s[next_node].x, 2)
                    current_node = next_node
            
            # Only add non-trivial routes (with at least one customer)
            if len(route) > 2:
                routes.append([nodes[i] for i in route])
        
        objective_value = model.objVal
        
        return objective_value, routes, schedule
    
    else:
        # Return empty solution if no feasible solution found
        return float('inf'), [], {}


def main():
    parser = argparse.ArgumentParser(description='Solve VRPTW-IC problem')
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path to save the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, default=None,
                        help='Path to log intermediate solutions')
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Solve the problem
    start_time = time.time()
    try:
        objective_value, routes, schedule = solve_vrptw_ic(instance_data, args.time_limit)
        
        # Log the final solution if logger is available
        if logger and objective_value < float('inf'):
            solution_dict = {
                "objective_value": objective_value,
                "routes": routes,
                "schedule": schedule
            }
            logger.log_solution(objective_value, solution_dict)
        
        # Write solution
        solution = {
            "objective_value": objective_value,
            "routes": routes,
            "schedule": schedule
        }
        
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    
    except Exception as e:
        print(f"Error occurred: {e}")
        # Write an empty solution in case of error
        solution = {
            "objective_value": float('inf'),
            "routes": [],
            "schedule": {}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()