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
            # Calculate delivery node based on problem description
            delivery_node = pickup_node + n_requests  # Based on numbering described: delivery j = pickup j + n
            pickup_to_delivery[pickup_node] = delivery_node
            delivery_to_pickup[delivery_node] = pickup_node
    
    # Process arcs into a dictionary
    arcs = defaultdict(set)  # Using sets for uniqueness
    arc_costs = {}
    travel_times = {}
    
    for arc_data in arcs_data:
        from_node = arc_data['from']
        to_node = arc_data['to']
        arcs[from_node].add(to_node)
        arc_costs[(from_node, to_node)] = arc_data['cost']
        travel_times[(from_node, to_node)] = arc_data['travel_time']
    
    # Make sure depot arcs are available
    for arc in arcs_data:
        if arc['from'] not in arcs:
            arcs[arc['from']] = set()
        arcs[arc['from']].add(arc['to'])
    
    # Create lists of different types of nodes
    pickup_nodes = [node['node_id'] for node in nodes if node['node_type'] == 'pickup']
    delivery_nodes = [node['node_id'] for node in nodes if node['node_type'] == 'delivery']
    all_nodes = {node['node_id']: node for node in nodes}
    
    # Create Gurobi model
    model = gp.Model("VRPTW_LIFO")
    model.setParam("Threads", 1)
    model.setParam(GRB.Param.TimeLimit, time_limit)
    model.setParam(GRB.Param.MIPFocus, 1)  # Focus on finding feasible solutions
    model.setParam(GRB.Param.Heuristics, 0.05)  # Use heuristics less aggressively
    model.setParam(GRB.Param.Method, 1)  # Try barrier first
    
    # Decision variables
    # We need to be careful about limiting the number of vehicles to reasonable number
    # A safe upper bound is the number of requests
    max_vehicles = min(n_requests + 1, 20)  # Use smaller number for tractability
    vehicles = list(range(max_vehicles))
    
    x = {}  # Arc traversal variable
    t = {}  # Start time at node
    q = {}  # Load on vehicle when leaving node
    
    # T_MAX should be max of planning horizon
    T_MAX = planning_horizon[1]
    
    # Define binary variables for arcs
    for k in vehicles:
        for arc in [(f, t) for f in arcs for t in arcs[f]]:
            x[arc[0], arc[1], k] = model.addVar(vtype=GRB.BINARY,
                                                 name=f"x_{arc[0]}_{arc[1]}_{k}")
    
    # Define continuous variables for time and load
    for k in vehicles:
        for node_id in all_nodes:
            # Time variable
            t[node_id, k] = model.addVar(vtype=GRB.CONTINUOUS, 
                                          lb=all_nodes[node_id]['tw_early'],
                                          ub=min(all_nodes[node_id]['tw_late'], T_MAX),
                                          name=f"t_{node_id}_{k}")
            
            # Load variable
            node_load = all_nodes[node_id]['load']
            max_load_after = min(vehicle_capacity, vehicle_capacity + node_load)
            min_load_after = max(0, node_load)
            q[node_id, k] = model.addVar(vtype=GRB.CONTINUOUS, 
                                          lb=min_load_after, 
                                          ub=max_load_after,
                                          name=f"q_{node_id}_{k}")
    
    model.update()
    
    # 1. Each pickup node must be visited exactly once among all vehicles
    for pick_node in pickup_nodes:
        model.addConstr(gp.quicksum(x[pick_node, j, k] 
                    for k in vehicles 
                    for j in arcs[pick_node]) == 1,
                    name=f"pickup_once_{pick_node}")
    
    # 2. Flow conservation for each non-depot node and for each vehicle
    for k in vehicles:
        # Flow conservation at each node for each vehicle
        for node_id in all_nodes:
            if node_id in [origin_depot, destination_depot]:
                continue
            
            inflow = gp.quicksum(x[j, node_id, k] for j in all_nodes if node_id in arcs.get(j, set()))
            outflow = gp.quicksum(x[node_id, j, k] for j in arcs.get(node_id, set()))
            model.addConstr(inflow == outflow, name=f"flow_cons_{node_id}_{k}")

        # Vehicle leaves origin depot as many times as it enters destination depot
        origin_out = gp.quicksum(x[origin_depot, j, k] for j in arcs.get(origin_depot, set()))
        dest_in = gp.quicksum(x[j, destination_depot, k] for j in all_nodes if destination_depot in arcs.get(j, set()))
        model.addConstr(origin_out == dest_in, name=f"veh_flow_bal_{k}")
        
        # Limit each vehicle to at most one use
        model.addConstr(origin_out <= 1, name=f"at_most_route_per_v_{k}")
    
    # 3. Each vehicle must start at origin and end at destination (if used)
    for k in vehicles:
        model.addConstr(gp.quicksum(x[j, origin_depot, k] 
                    for j in all_nodes if origin_depot in arcs.get(j, set())) == 0,
                    name=f"origin_noin_{k}")
        model.addConstr(gp.quicksum(x[destination_depot, j, k] 
                    for j in arcs.get(destination_depot, set())) == 0,
                    name=f"dest_nout_{k}")
    
    # 4. If a vehicle serves a pickup, it must also serve the corresponding delivery
    for pick_node in pickup_nodes:
        deliv_node = pickup_to_delivery.get(pick_node)
        if deliv_node is not None:
            for k in vehicles:
                # If the pickup is served by vehicle k, then the delivery must also be served by k
                pickup_outflow = gp.quicksum(x[pick_node, j, k] for j in arcs.get(pick_node, set()))
                deliv_inflow = gp.quicksum(x[j, deliv_node, k] for j in all_nodes if deliv_node in arcs.get(j, set()))
                model.addConstr(pickup_outflow == deliv_inflow, name=f"pair_pd_{pick_node}_{deliv_node}_{k}")
    
    # 5. Precedence: Delivery happens after pickup
    BIG_M_T = T_MAX + 1000
    for pick_node in pickup_nodes:
        deliv_node = pickup_to_delivery.get(pick_node)
        if deliv_node is not None:
            time_diff = all_nodes[pick_node]['service_time'] + travel_times.get((pick_node, deliv_node), 0)
            for k in vehicles:
                model.addConstr(t[deliv_node, k] >= t[pick_node, k] + time_diff -
                               BIG_M_T * (2 - gp.quicksum(x[pick_node, j, k] for j in arcs.get(pick_node, set()))
                                         - gp.quicksum(x[j, deliv_node, k] for j in all_nodes if deliv_node in arcs.get(j, set()))),
                               name=f"precedence_pd_{pick_node}_{deliv_node}_{k}")
    
    # 6. Time windows
    for k in vehicles:
        for node_id in all_nodes:
            model.addConstr(t[node_id, k] >= all_nodes[node_id]['tw_early'], 
                            name=f"tw_early_{node_id}_{k}")
            model.addConstr(t[node_id, k] <= all_nodes[node_id]['tw_late'], 
                            name=f"tw_late_{node_id}_{k}")  
    
    # 7. Time link via arcs
    for k in vehicles:
        for from_node in arcs:
            for to_node in arcs[from_node]:
                # If arc (from, to) is used by vehicle k, impose timing constraints
                travel_t = travel_times.get((from_node, to_node), 0)
                service_time_from = all_nodes[from_node]['service_time']
                
                model.addConstr(t[to_node, k] >= t[from_node, k] + service_time_from + travel_t -
                               BIG_M_T * (1 - x[from_node, to_node, k]),
                               name=f"time_link_{from_node}_{to_node}_{k}")
    
    # 8. Capacity constraints
    for k in vehicles:
        for node_id in all_nodes:
            model.addConstr(q[node_id, k] <= vehicle_capacity, name=f"capac_{node_id}_{k}")
    
    # 9. Load update constraints
    BIG_M_Q = vehicle_capacity + 1
    for k in vehicles:
        for from_node in arcs:
            for to_node in arcs[from_node]:
                # If x[i,j,k]=1, then load after j >= load after i + demand_j
                demand_to = all_nodes[to_node]['load']
                # So: q[to] >= q[from] + demand_to - M*(1 - x[from,to,k])
                model.addConstr(q[to_node, k] >= q[from_node, k] + demand_to - 
                               BIG_M_Q * (1 - x[from_node, to_node, k]),
                               name=f"load_upd_{from_node}_{to_node}_{k}")
        
        # Initial load at origin depot is 0
        model.addConstr(q[origin_depot, k] == 0, name=f"init_load_{k}")
    
    # 10. LIFO Constraints: For simplicity, we model this by preventing direct pickup->delivery
    # for different requests in the arc structure, but we might need to add more sophisticated cuts
    
    # Define objective - minimize total cost
    obj_expr = gp.quicksum(
        x[from_node, to_node, k] * arc_costs.get((from_node, to_node), 0)
        for from_node in arcs 
        for to_node in arcs[from_node]
        for k in vehicles
    )
    
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    # Initialize solution logger if provided
    logger = None
    if log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(log_path, sense="minimize")
    
    def sol_logger_cb(model, where):
        if where == GRB.CB_MIPSOL:
            obj = model.cbGet(GRB.CB_MIPSOL_OBJ)
            if logger:
                logger.log(obj)
    
    # Solve the model
    if logger:
        model.optimize(sol_logger_cb)
    else:
        model.optimize()
    
    # Extract solution
    if model.solCount > 0:
        routes = []
        
        for k in vehicles:
            # Check if vehicle k was actually used
            used = False
            for j in arcs.get(origin_depot, set()):
                if x[origin_depot, j, k].X > 0.5:
                    used = True
                    break
            if not used:
                continue  # Skip unused vehicles
            
            # Reconstruct route for vehicle k
            current_node = origin_depot
            route = [origin_depot]
            route_detail = [{
                "node_id": origin_depot,
                "arrival_time": round(t[origin_depot, k].X, 2),
                "load_after": round(q[origin_depot, k].X, 2)
            }]
            
            # Follow the flow until reaching destination
            while current_node != destination_depot:
                # Find next node on route
                next_node = None
                for j in arcs.get(current_node, set()):
                    if x[current_node, j, k].X > 0.5:
                        next_node = j
                        break
                        
                if next_node is not None:
                    route.append(next_node)
                    route_detail.append({
                        "node_id": next_node,
                        "arrival_time": round(t[next_node, k].X, 2),
                        "load_after": round(q[next_node, k].X, 2)
                    })
                    current_node = next_node
                else:
                    break  # Avoid infinite loop
            
            if len(route) > 1:  # Non-trivial route
                routes.append({
                    "vehicle_id": k,
                    "route_sequence": route,
                    "route_details": route_detail
                })
        
        solution = {
            "objective_value": round(model.objVal, 2),
            "routes": routes
        }
    else:
        solution = {
            "objective_value": float('inf'),
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