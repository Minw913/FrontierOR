import argparse
import json
import math
from typing import Dict, List, Tuple
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def solve_vrptw(instance_data, time_limit, logger=None):
    num_requests = instance_data['num_requests']
    num_vehicles = instance_data['num_vehicles']
    vehicle_capacity = instance_data['vehicle_capacity']
    
    # Build node map
    n = len(instance_data['nodes'])
    
    # Create distance and time matrices
    distances = [[0.0 for _ in range(n)] for _ in range(n)]
    times = [[0.0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            node_i = instance_data['nodes'][i]
            node_j = instance_data['nodes'][j]
            dist = euclidean_distance(node_i['x'], node_i['y'], node_j['x'], node_j['y'])
            distances[i][j] = dist
            times[i][j] = dist  # Assuming unit speed, time equals distance

    # Indices for depots
    depot_origin_idx = next(i for i in range(n) if instance_data['nodes'][i]['type'] == 'depot_origin')
    depot_dest_idx = next(i for i in range(n) if instance_data['nodes'][i]['type'] == 'depot_destination')

    # Identify pickup and delivery nodes
    pickup_indices = [i for i in range(n) if instance_data['nodes'][i]['type'] == 'pickup']
    delivery_indices = [i for i in range(n) if instance_data['nodes'][i]['type'] == 'delivery']
    
    # Create mapping from request_id to pickup and delivery nodes
    req_to_pu_del = {}
    for req in instance_data['requests']:
        req_id = req['request_id']
        pu = req['pickup_node']
        dl = req['delivery_node']
        req_to_pu_del[req_id] = (pu, dl)

    # Get time windows and demands for nodes
    time_windows = [instance_data['nodes'][i]['time_window'] for i in range(n)]
    demands = [instance_data['nodes'][i]['demand'] for i in range(n)]
    service_times = [instance_data['nodes'][i]['service_time'] for i in range(n)]

    # Create model
    model = gp.Model("vrptw")
    model.setParam("Threads", 1)
    model.setParam(GRB.Param.TimeLimit, time_limit)

    # Variables: whether vehicle k travels from node i to node j
    x = {}  # Binary variables for vehicle routing
    for k in range(num_vehicles):
        for i in range(n):
            for j in range(n):
                if i != j:  # No self loops
                    x[k, i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{k}_{i}_{j}")

    # Arrival times at nodes (per vehicle)
    arrival_time = {}
    for k in range(num_vehicles):
        for i in range(n):
            # Upper bound on max time based on constraint
            arrival_time[k, i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, 
                                             ub=instance_data['planning_horizon'],
                                             name=f"arrival_time_{k}_{i}")

    # Load at each node per vehicle
    load = {}
    for k in range(num_vehicles):
        for i in range(n):
            load[k, i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,ub=vehicle_capacity,
                                      name=f"load_{k}_{i}")

    # Set objective - minimize total routing cost
    obj_expr = gp.LinExpr()
    for k in range(num_vehicles):
        for i in range(n):
            for j in range(n):
                if i != j:
                    obj_expr += distances[i][j] * x[k, i, j]

    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Each request must be served exactly once
    for pickup_idx in pickup_indices:
        # Find corresponding delivery node
        # We need to ensure pickup is followed by delivery and served by the same vehicle
        req_id_for_node = None
        for req in instance_data['requests']:
            if req['pickup_node'] == pickup_idx:
                req_id_for_node = req['request_id']
                break
        assert req_id_for_node is not None, f"No request found for pickup node {pickup_idx}"
        
        delivery_idx = req_to_pu_del[req_id_for_node][1]

        # Each pickup must be part of exactly one route
        pickup_sum = gp.LinExpr()
        for k in range(num_vehicles):
            for i in range(n):
                if i != pickup_idx:
                    pickup_sum += x[k, i, pickup_idx]
        model.addConstr(pickup_sum == 1, name=f"pickup_served_{pickup_idx}")

        # Each delivery must be part of exactly one route
        del_sum = gp.LinExpr()
        for k in range(num_vehicles):
            for i in range(n):
                if i != delivery_idx:
                    del_sum += x[k, i, delivery_idx]
        model.addConstr(del_sum == 1, name=f"delivery_served_{delivery_idx}")

        # Precedence: pickup before delivery on the same route
        # If there's route going to delivery node, find which vehicle - then check that
        # it has pickup in the route with some earlier visit sequence (hard to formulate directly)
        
        # Add constraints ensuring pickup and delivery are served by the same vehicle
        for k in range(num_vehicles):
            # If vehicle k serves the delivery, then it must have served the pickup
            model.addConstr(
                sum(x[k, i, delivery_idx] for i in range(n) if i != delivery_idx) <=
                sum(x[k, i, pickup_idx] for i in range(n) if i != pickup_idx),
                name=f"same_vehicle_{req_id_for_node}_{k}"
            )

    # Flow conservation: each used vehicle leaves and returns to depot exactly once
    for k in range(num_vehicles):
        # The number of nodes the vehicle leaves and enters should balance
        model.addConstr(sum(x[k, depot_origin_idx, j] for j in range(n) if j != depot_origin_idx) ==
                       sum(x[k, i, depot_dest_idx] for i in range(n) if i != depot_dest_idx),
                       name=f"flow_cons_depots_{k}")
        
        # Ensure vehicle starts and ends properly
        departure_flow = sum(x[k, depot_origin_idx, j] for j in range(n) if j not in [depot_origin_idx, depot_dest_idx])
        arrival_flow = sum(x[k, i, depot_dest_idx] for i in range(n) if i not in [depot_origin_idx, depot_dest_idx])
        model.addConstr(departure_flow == arrival_flow, name=f"balanced_vehicles_{k}")
        
        # Additional flow conservation for all nodes except depots
        for i in range(n):
            if i not in [depot_origin_idx, depot_dest_idx]:
                model.addConstr(
                    sum(x[k, i, j] for j in range(n) if i != j) ==  # leaving i  
                    sum(x[k, j, i] for j in range(n) if i != j),    # entering i
                    name=f"flow_cons_{k}_{i}")

    # Time windows: arrival + service must be within time window if node is served
    for k in range(num_vehicles):
        for j in range(n):
            if j not in [depot_origin_idx, depot_dest_idx]:  # For non-depot nodes
                # If any vehicle visits j, arrival + service happens within TW
                for i in range(n):
                    if i != j:
                        # Add time window compliance constraints
                        # Arrival_time[k, j] should respect time window of node j
                        early_time, late_time = time_windows[j]
                        
                        # If x[k, i, j] = 1, arrival_time[k, j] >= early_time
                        model.addConstr(arrival_time[k, j] >= early_time - 
                                       (1 - sum(x[k, i_prime, j] for i_prime in range(n) if i_prime != j)) * instance_data['planning_horizon'] * 2,
                                       name=f"time_early_{k}_{j}")
                        
                        model.addConstr(arrival_time[k, j] <= late_time +
                                       (1 - sum(x[k, i_prime, j] for i_prime in range(n) if i_prime != j)) * instance_data['planning_horizon'] * 2,
                                       name=f"time_late_{k}_{j}")
                        
                        # Time transitions: if x[k, i, j] = 1, arrival[k, j] = arrival[k, i] + time[i,j] + service[i]
                        for i_other in range(n):
                            if i_other != j:
                                model.addConstr(
                                    arrival_time[k, j] >= (arrival_time[k, i_other] + service_times[i_other] + 
                                         times[i_other][j]) - (1 - x[k, i_other, j]) * instance_data['planning_horizon'] * 2,
                                    name=f"time_transition_{k}_{i_other}_{j}"
                                )

    # Capacity constraints: load never exceeds vehicle capacity
    for k in range(num_vehicles):
        for i in range(n):
            model.addConstr(load[k, i] <= vehicle_capacity, name=f"capacity_{k}_{i}")

    # Load dynamics: after visiting node i, update the load based on demand
    for k in range(num_vehicles):
        # Flow for loads
        for i in range(n):
            for j in range(n):
                if i != j:
                    # If vehicle drives from i to j, then we add load from i to the load at j
                    model.addConstr(load[k, j] >= (load[k, i] + demands[i]) - 
                                  (1 - x[k, i, j]) * vehicle_capacity * 2,
                                  name=f"load_transfer_{k}_{i}_{j}")
        
        # Initial load at depot origin should be 0
        model.addConstr(load[k, depot_origin_idx] == 0, name=f"init_load_{k}")

    # Callback to log best solution found so far
    def solution_log_callback(model, where):
        if where == GRB.Callback.MIPSOL and logger:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            logger.log(obj_val)

    model.optimize(solution_log_callback)

    # Extract solution
    routes = []
    obj_value = 0
    
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        obj_value = model.objVal
        
        for k in range(num_vehicles):
            route = [depot_origin_idx]  # Start at origin depot
            current = depot_origin_idx
            
            # Follow the route
            while True:
                found_next = False
                for j in range(n):
                    if j not in route and abs(model.getVarByName(f"x_{k}_{current}_{j}").X - 1) < 1e-5:
                        route.append(j)
                        current = j
                        found_next = True
                        break
                
                if not found_next or current == depot_dest_idx:
                    break
            
            if len(route) > 1:  # If this vehicle was actually used
                routes.append({
                    "vehicle": k,
                    "nodes": route
                })
    
    result = {
        "objective_value": obj_value,
        "routes": routes
    }
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    
    args = parser.parse_args()
    
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    solution = solve_vrptw(instance_data, args.time_limit, logger)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()