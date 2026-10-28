import argparse
import json
import math
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from solution_logger import SolutionLogger

def solve_vrptw_instance(instance_path, time_limit, log_path):
    # Load instance
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    # Prepare data
    num_customers = instance["num_customers"]
    vehicle_capacity = instance["vehicle_capacity"]
    scheduling_horizon = instance["scheduling_horizon"]
    depot_info = instance["depot"]
    customers_info = instance["customers"]
    dist_matrix = instance["distance_matrix"]
    
    # Numbering convention: depot=0, customers=1..n, return depot=n+1
    n = num_customers
    # Distance matrix between (0 to n) -> (0 to n)
    # For consistency with problem statement, use original indexing
    # Return depot is implicitly handled
    
    all_nodes = list(range(n + 1))  # 0 (depot) and 1..n (customers)
    customer_nodes = list(range(1, n + 1))
    N = len(all_nodes)  # = n+1 nodes
    
    # Compute time windows and service times
    time_windows_start = [depot_info['time_window'][0]]
    time_windows_end = [depot_info['time_window'][1]]
    service_times = [depot_info['service_time']]
    
    demands = [0]  # Depot demand = 0
    customer_demand_map = {}
    
    for c in customers_info:
        time_windows_start.append(c['time_window'][0])
        time_windows_end.append(c['time_window'][1])
        service_times.append(c['service_time'])
        demands.append(c['demand'])
        customer_demand_map[c['id']] = c['demand']
    
    # Create model
    model = gp.Model("VRPTW_with_Split_Deliveries")
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0  # Disable default console logging
    
    # Determine max number of vehicles (upper bound)
    total_demand = sum(demands)
    min_vehicles_needed = math.ceil(total_demand / vehicle_capacity)
    max_vehicles = instance.get("num_vehicles_upper_bound", min_vehicles_needed + int(math.sqrt(n)))
    K = max_vehicles  # max number of vehicles
    
    # Precompute feasible arcs based on time windows
    feasible_arcs = []
    for i in all_nodes:
        for j in all_nodes:
            if i != j:
                if (time_windows_start[i] + service_times[i] + dist_matrix[i][j] <= time_windows_end[j]):
                    feasible_arcs.append((i, j))
    
    # Decision Variables
    x = {}  # x[k,i,j] = 1 if vehicle k uses arc (i,j)
    s = {}  # s[k,i] = start time when vehicle k serves customer i
    q = {}  # q[k,i,j] = flow of demand on arc (i,j) by vehicle k
    
    for k in range(K):
        for (i, j) in feasible_arcs:
            x[k, i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{k}_{i}_{j}")
            
        for i in all_nodes:
            s[k, i] = model.addVar(vtype=GRB.CONTINUOUS, lb=time_windows_start[i], ub=time_windows_end[i], 
                                  name=f"s_{k}_{i}")
        
        for (i, j) in feasible_arcs:
            cap_ub = min(demands[j] if j > 0 else vehicle_capacity, vehicle_capacity)
            q[k, i, j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=cap_ub,
                                     name=f"q_{k}_{i}_{j}")
    
    # Objective: minimize total travel distance
    obj_expr = gp.LinExpr()
    for k in range(K):
        for (i, j) in feasible_arcs:
            obj_expr += dist_matrix[i][j] * x[k, i, j]
    
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    # Constraints
    # 1. Each customer is served in full by one or more vehicles
    for i in customer_nodes:
        model.addConstr(gp.quicksum(q[k, j, i] for k in range(K) for (j, l) in feasible_arcs if l == i) >= demands[i],
                      name=f"demand_fulfillment_{i}")
    
    # 2. Flow conservation for each customer and vehicle
    for k in range(K):
        for i in customer_nodes:
            model.addConstr(
                gp.quicksum(x[k, j, i] for (j, l) in feasible_arcs if l == i) == \
                gp.quicksum(x[k, i, j] for (l, j) in feasible_arcs if l == i),
                name=f"flow_conservation_{k}_{i}"
            )
    
    # 3. A vehicle serves a customer if and only if it traverses an arc to that customer
    for k in range(K):
        for i in customer_nodes:
            model.addConstr(
                gp.quicksum(x[k, j, i] for (j, l) in feasible_arcs if l == i) <= 1,
                name=f"visit_at_most_once_{k}_{i}"
            )
    
    # 4. Start from depot constraints - at most one arc out of depot per vehicle
    for k in range(K):
        model.addConstr(gp.quicksum(x[k, 0, j] for (i, j) in feasible_arcs if i == 0) <= 1, 
                       name=f"out_depot_{k}")
    
    # 5. Visit time constraint
    big_M = max(time_windows_end) + max(dist_matrix[i][j] for i in all_nodes for j in all_nodes)
    for k in range(K):
        for (i, j) in feasible_arcs:
            model.addConstr(s[k, j] >= s[k, i] + service_times[i] + dist_matrix[i][j] - big_M * (1 - x[k, i, j]),
                           name=f"time_flow_{k}_{i}_{j}")
    
    # 6. Flow capacity constraint (capacity leaving node)
    for k in range(K):
        for i in all_nodes:
            model.addConstr(
                gp.quicksum(q[k, i, j] for (l, j) in feasible_arcs if l == i) <= vehicle_capacity,
                name=f"vehicle_capacity_{k}_{i}"
            )
    
    # 7. Relationship between flow and x variables
    for k in range(K):
        for (i, j) in feasible_arcs:
            model.addConstr(q[k, i, j] <= vehicle_capacity * x[k, i, j],
                           name=f"flow_to_arc_{k}_{i}_{j}")
    
    # Callback for logging incumbent solutions
    solution_logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    best_obj = float('inf')
    
    def mycallback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj:
                best_obj = obj_val
                if solution_logger:
                    solution_logger.log(obj_val)
    
    # Optimize
    model.optimize(mycallback)
    
    # Extract solution
    if model.SolCount == 0:
        raise Exception("No solution found!")
    
    routes = []
    deliveries_map = {c['id']: {"demand": c['demand'], "total_delivered": 0} for c in customers_info}
    num_vehicles_used = 0
    
    for k in range(K):
        # Extract route for vehicle k
        route_edges = [(i, j) for (i, j) in feasible_arcs if x[k, i, j].X > 0.9]
        if not route_edges:
            continue
        
        # Build ordered path
        path = [0]  # start at depot
        current = 0
        visited = {0}
        
        while True:
            next_node_found = False
            for (i, j) in route_edges:
                if i == current and j not in visited:
                    path.append(j)
                    current = j
                    visited.add(j)
                    next_node_found = True
                    break
            
            if current == 0 and len(path) > 1:  # returned to depot
                break
            if not next_node_found:
                if current in customer_nodes and current not in path[:-1]:  # unvisited customer node in cycle
                    break
                break
        
        # Only process if route serves at least one customer
        if len(path) > 2 or (len(path) == 2 and path[1] != 0):  # has customers
            route_deliveries = {}
            
            for i in range(1, len(path)):  # Skip first depot entry
                node = path[i]
                if node in customer_nodes and node not in route_deliveries:
                    # Identify the total quantity delivered from this vehicle to node
                    tot_q = sum(q[k, prev_node, node].X for prev_node in range(N) if (prev_node, node) in feasible_arcs and q[k, prev_node, node].X > 1e-3)
                    route_deliveries[str(node)] = round(tot_q, 3)
                    
                    # Update global deliveries
                    cid = customers_info[node - 1]['id']  # assuming original customer ID ordering
                    deliveries_map[cid]["total_delivered"] += round(tot_q, 3)
            
            routes.append({
                "vehicle": k,
                "route": path,
                "deliveries": route_deliveries
            })
            num_vehicles_used += 1
    
    result = {
        "objective_value": model.ObjVal,
        "num_vehicles_used": num_vehicles_used,
        "routes": routes,
        "deliveries": deliveries_map
    }
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Solve VRPTW with Split Deliveries.')
    parser.add_argument('--instance_path', required=True, help='Path to input instance JSON file')
    parser.add_argument('--solution_path', required=True, help='Path to output solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', help='Path to log file for intermediate solutions')
    
    args = parser.parse_args()
    
    # Solve the instance
    solution = solve_vrptw_instance(args.instance_path, args.time_limit, args.log_path)
    
    # Save solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)
    
    print(f"Solution saved to {args.solution_path}")

if __name__ == "__main__":
    main()