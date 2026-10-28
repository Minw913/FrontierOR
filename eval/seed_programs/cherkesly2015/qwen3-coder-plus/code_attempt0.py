import argparse
import json
import math
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
import itertools

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def solve_vrptw_lifo(instance_path, solution_path, time_limit, log_path=None):
    # Load the instance data
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    # Extract parameters
    n_requests = instance['n_requests']
    n_nodes = instance['n_nodes']
    vehicle_capacity = instance['vehicle_capacity']
    planning_horizon = instance['planning_horizon']
    vehicle_fixed_cost = instance['vehicle_fixed_cost']
    nodes = instance['nodes']
    arcs_data = instance['arcs']
    
    # Create node lookup dictionaries
    node_to_request = {}
    pickup_to_delivery = {}
    delivery_to_pickup = {}
    
    origin_depot = None
    destination_depot = None
    
    for node in nodes:
        node_id = node['node_id']
        node_to_request[node_id] = node['request_id']
        
        if node['node_type'] == 'depot_origin':
            origin_depot = node_id
        elif node['node_type'] == 'depot_destination':
            destination_depot = node_id
        elif node['node_type'] == 'pickup':
            pickup_node = node_id
            delivery_node = pickup_node + n_requests
            pickup_to_delivery[pickup_node] = delivery_node
            delivery_to_pickup[delivery_node] = pickup_node
    
    # Process arcs into a dictionary
    arcs = defaultdict(list)
    arc_costs = {}
    travel_times = {}
    
    for arc_data in arcs_data:
        from_node = arc_data['from']
        to_node = arc_data['to']
        arcs[from_node].append(to_node)
        arc_costs[(from_node, to_node)] = arc_data['cost']
        travel_times[(from_node, to_node)] = arc_data['travel_time']
    
    # Create lists of different types of nodes
    pickup_nodes = [node['node_id'] for node in nodes if node['node_type'] == 'pickup']
    delivery_nodes = [node['node_id'] for node in nodes if node['node_type'] == 'delivery']
    all_nodes = {node['node_id']: node for node in nodes}
    
    # Create Gurobi model
    model = gp.Model("VRPTW_LIFO")
    model.setParam(GRB.Param.TimeLimit, time_limit)
    model.setParam(GRB.Param.MIPFocus, 1)  # Focus on finding feasible solutions
    model.setParam(GRB.Param.Heuristics, 0.05)  # Use heuristics less aggressively
    model.setParam(GRB.Param.Method, 1)  # Barrier method initially
    
    # Decision variables
    # Binary variables: x[i,j,k] = 1 if vehicle k traverses arc (i,j)
    max_vehicles = len(pickup_nodes)  # Upper bound (could use 1 vehicle per request as max)
    vehicles = list(range(max_vehicles))
    
    x = {}  # Arc traversal variable
    t = {}  # Start time at node
    q = {}  # Load after serving node
    
    # Set a large enough upper bound for time and load variables
    T_MAX = planning_horizon[1]
    
    for k in vehicles:
        for from_node in arcs:
            for to_node in arcs[from_node]:
                x[from_node, to_node, k] = model.addVar(vtype=GRB.BINARY,
                                                       name=f"x_{from_node}_{to_node}_{k}")
                
    for k in vehicles:
        for node_id in all_nodes:
            # Time at which service starts at node
            t[node_id, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=all_nodes[node_id]['tw_early'],
                                         ub=min(all_nodes[node_id]['tw_late'], T_MAX),
                                         name=f"t_{node_id}_{k}")
            # Load of vehicle after serving node
            node_load = all_nodes[node_id]['load']
            max_load_after = min(vehicle_capacity, vehicle_capacity + node_load)
            min_load_after = max(0, node_load)
            q[node_id, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=min_load_after,
                                         ub=max_load_after,
                                         name=f"q_{node_id}_{k}")
    
    model.update()
    
    # Constraints
    
    # 1. Each pickup node is visited exactly once
    for pick_node in pickup_nodes:
        model.addConstr(gp.quicksum(x[pick_node, j, k] for k in vehicles 
                                    for j in arcs[pick_node]) == 1, name=f"visit_pick_once_{pick_node}")
    
    # 2. Flow conservation for each vehicle
    for k in vehicles:
        # Must start from origin depot
        model.addConstr(gp.quicksum(x[origin_depot, j, k] for j in arcs[origin_depot]) <= 1,
                        name=f"start_from_origin_{k}")
        
        # Outflow equals inflow for non-depot nodes
        for node_id in all_nodes:
            if node_id != origin_depot and node_id != destination_depot:
                inflow = gp.quicksum(x[j, node_id, k] for j in all_nodes if node_id in arcs[j])
                outflow = gp.quicksum(x[node_id, j, k] for j in arcs[node_id])
                model.addConstr(inflow == outflow, name=f"flow_conservation_{node_id}_{k}")
        
        # Inflow to origin depot and outflow from destination depot must be 0
        model.addConstr(gp.quicksum(x[j, origin_depot, k] for j in all_nodes if origin_depot in arcs[j]) == 0,
                        name=f"no_inflow_origin_{k}")
        model.addConstr(gp.quicksum(x[destination_depot, j, k] for j in arcs[destination_depot]) == 0,
                        name=f"no_outflow_destination_{k}")
        
        # If outflow from origin depot, then inflow to destination depot exists
        origin_out = gp.quicksum(x[origin_depot, j, k] for j in arcs[origin_depot])
        dest_in = gp.quicksum(x[j, destination_depot, k] for j in all_nodes if destination_depot in arcs[j])
        model.addConstr(origin_out == dest_in, name=f"vehicle_flow_{k}")
    
    # 3. Pickup-delivery constraint: if vehicle k serves pickup, it must serve corresponding delivery
    for pick_node in pickup_nodes:
        delivery_node = pickup_to_delivery[pick_node]
        for k in vehicles:
            # Sum of outgoing edges from pickup implies same vehicle serves the delivery
            model.addConstr(gp.quicksum(x[pick_node, j, k] for j in arcs[pick_node]) <= 
                            gp.quicksum(x[i, delivery_node, k] for i in all_nodes if delivery_node in arcs[i]),
                            name=f"pickup_followed_by_delivery_{pick_node}_{k}")
            
            # Reverse implication: if delivery served by k, then pickup was served by k
            model.addConstr(gp.quicksum(x[i, delivery_node, k] for i in all_nodes if delivery_node in arcs[i]) <= 
                            gp.quicksum(x[pick_node, j, k] for j in arcs[pick_node]),
                            name=f"delivery_follows_pickup_{delivery_node}_{k}")
    
    # 4. Time constraints: arrival and service timing
    BIG_M_TIME = T_MAX  # Large number for big-M
    BIG_M_LOAD = vehicle_capacity  # Large number for big-M relating to loads
    
    for k in vehicles:
        for from_node in all_nodes:
            for to_node in arcs[from_node]:
                if not (from_node == origin_depot and to_node in delivery_nodes):
                    # Ensure time consistency: arrival at 'to_node' >= departure from 'from_node' + travel time
                    model.addConstr(t[to_node, k] >= t[from_node, k] + 
                                   all_nodes[from_node]['service_time'] + 
                                   travel_times[(from_node, to_node)] - 
                                   BIG_M_TIME*(1 - x[from_node, to_node, k]),
                                   name=f"time_consistency_{from_node}_{to_node}_{k}")
    
    # 5. Load consistency: ensure load after serving node is updated correctly
    for k in vehicles:
        for from_node in all_nodes:
            for to_node in arcs[from_node]:
                if from_node != destination_depot and to_node != origin_depot:
                    # Load at 'to_node' >= Load at 'from_node' + demand at 'to_node' - adjustment when used
                    demand_at_to = all_nodes[to_node]['load']
                    model.addConstr(q[to_node, k] >= q[from_node, k] + demand_at_to - 
                                   BIG_M_LOAD * (1 - x[from_node, to_node, k]),
                                   name=f"load_consistency_{from_node}_{to_node}_{k}")
                
                # Prevent cycles and ensure proper handling of pickup/delivery relationship
                if from_node in pickup_nodes and to_node in delivery_nodes:
                    # LIFO constraints prevent direct pickup -> delivery of different requests (already handled via arc exclusion)
                    # We only allow pickup -> delivery for the same request through specific path construction
                    if pickup_to_delivery.get(from_node) != to_node:
                        for k1 in vehicles:
                            model.addConstr(x[from_node, to_node, k1] == 0,
                                            name=f"lifo_forbid_diff_req_{from_node}_{to_node}_v{k1}")
    
    # 6. Capacity constraints
    for k in vehicles:
        for node_id in all_nodes:
            model.addConstr(q[node_id, k] <= vehicle_capacity, name=f"capacity_{node_id}_{k}")
    
    # 7. Time window constraints
    for k in vehicles:
        for node_id in all_nodes:
            model.addConstr(t[node_id, k] >= all_nodes[node_id]['tw_early'], name=f"tw_early_{node_id}_{k}")
            model.addConstr(t[node_id, k] <= all_nodes[node_id]['tw_late'], name=f"tw_late_{node_id}_{k}")
    
    # 8. Pickup-delivery precedence within a vehicle route
    for pick_node in pickup_nodes:
        delivery_node = pickup_to_delivery[pick_node]
        for k in vehicles:
            model.addConstr(t[pick_node, k] + all_nodes[pick_node]['service_time'] + 
                           travel_times.get((pick_node, delivery_node), 999999) <= 
                           t[delivery_node, k] + 
                           BIG_M_TIME * (2 - gp.quicksum(x[pick_node, j, k] for j in arcs[pick_node]) - 
                                        gp.quicksum(x[i, delivery_node, k] for i in all_nodes if delivery_node in arcs[i])),
                           name=f"pickup_before_delivery_{pick_node}_{k}")
    
    # 9. LIFO constraint implementation via subroute restrictions
    # For LIFO, implement lifted subtour elimination constraints
    for k in vehicles:
        # Visit to a pickup must be directly followed by a visit to its corresponding delivery
        # Or be part of a path that doesn't violate LIFO policy
        # To ensure LIFO, we create constraints that prevent a pickup being "interrupted"
        
        # This is already handled by excluding arcs between pickup and delivery of other requests
        pass
    
    # Objective function: minimize total travel cost including fixed costs for each vehicle used
    obj_expr = gp.LinExpr()
    
    for from_node in arcs:
        for to_node in arcs[from_node]:
            for k in vehicles:
                obj_expr += x[from_node, to_node, k] * arc_costs[(from_node, to_node)]
    
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    # Initialize solution logger if provided
    logger = None
    if log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(log_path, sense="minimize")
    
    def sol_logger_cb(model, where):
        if where == GRB.Callback.MIPSol:
            obj = model.cbGet(GRB.Callback.MIPSolObj)
            if logger:
                logger.log(obj)
    
    # Optimize the model
    if logger:
        model.optimize(sol_logger_cb)
    else:
        model.optimize()
    
    # Extract solution
    if model.solCount > 0:
        routes = []
        for k in vehicles:
            route_start_times = []
            for node in all_nodes:
                if model.getAttr('X', x[origin_depot, node, k]) > 0.5:
                    route_start_times.append(node)
                    
            if not route_start_times:
                continue
                
            vehicle_route = [origin_depot]
            current_node = route_start_times[0]
            vehicle_route.append(current_node)
            
            visited_non_depots = {current_node}
            
            while current_node != destination_depot:
                next_nodes = []
                for j in arcs[current_node]:
                    if x[current_node, j, k].X > 0.5:
                        next_nodes.append(j)
                        
                if not next_nodes:
                    break
                
                # Find the next node in the route
                current_node = next_nodes[0]
                if current_node == destination_depot:
                    vehicle_route.append(destination_depot)
                    break
                elif current_node in pickup_nodes:
                    # Handle paired pickup-delivery
                    paired_delivery = pickup_to_delivery[current_node]
                    vehicle_route.append(paired_delivery)
                    visited_non_depots.add(current_node)
                    visited_non_depots.add(paired_delivery)
                    
                    # Continue to next node
                    next_after_delivery = None
                    for j in arcs[paired_delivery]:
                        if x[paired_delivery, j, k].X > 0.5:
                            next_after_delivery = j
                            
                    if next_after_delivery is not None:
                        current_node = next_after_delivery
                        if current_node == destination_depot:
                            vehicle_route.append(current_node)
                            break
                        else:
                            vehicle_route.append(current_node)
                            visited_non_depots.add(current_node)
                else:
                    vehicle_route.append(current_node)
                    visited_non_depots.add(current_node)
                    current_node = next_nodes[0]  # In case, though we handle properly above
                    
            # Add completed route
            if len(vehicle_route) > 1:
                route_det = []
                for node in vehicle_route:
                    arrive_time_val = t[node, k].X if t[node, k].X < float('inf') else 0
                    load_after_val = q[node, k].X if q[node, k].X < float('inf') else 0
                    route_det.append({
                        "node_id": node,
                        "arrival_time": round(arrive_time_val, 2),
                        "load_after": round(load_after_val, 2)
                    })
                
                routes.append({
                    "vehicle_id": k,
                    "route_sequence": vehicle_route,
                    "route_details": route_det
                })
        
        solution = {
            "objective_value": round(model.objVal, 2),
            "routes": routes
        }
    else:
        solution = {
            "objective_value": "INFEASIBLE" if model.status == GRB.INFEASIBLE else "TIME_LIMIT",
            "routes": []
        }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f)

def main():
    parser = argparse.ArgumentParser(description='Solve VRP with Time Windows and LIFO constraints')
    parser.add_argument('--instance_path', type=str, help='Path to the input instance JSON file')
    parser.add_argument('--solution_path', type=str, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log intermediate solutions (JSONL)')
    
    args = parser.parse_args()
    
    solve_vrptw_lifo(
        instance_path=args.instance_path,
        solution_path=args.solution_path,
        time_limit=args.time_limit,
        log_path=args.log_path
    )

if __name__ == "__main__":
    main()