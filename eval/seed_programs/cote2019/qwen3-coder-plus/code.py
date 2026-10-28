import argparse
import json
import math
from typing import List, Dict, Tuple
from collections import defaultdict
import itertools
import random
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def calculate_expected_area_weight(customer_data):
    """Calculate expected area and weight for a customer."""
    total_exp_area = 0
    total_exp_weight = 0
    
    for item in customer_data['items']:
        for realization in item['realizations']:
            prob = realization['probability']
            area = realization['height'] * realization['width']
            total_exp_area += prob * area
            total_exp_weight += prob * realization['weight']
    
    return total_exp_area, total_exp_weight


def check_2d_packing_feasibility(route, customer_data_map, vehicle_H, vehicle_W, vehicle_Q, all_items_list):
    """Check if route is packable for all demand scenarios"""
    if not route:
        return True
        
    # Find which items in all_items_list belong to this route
    route_items_dict = {}
    route_customer_data = []
    
    for cust_id in route:
        customer_data = customer_data_map.get(cust_id)
        if not customer_data:  
            # Find customer data by matching id
            customer_data = next(c for c in all_items_list if c['id'] == cust_id)
        route_customer_data.append(customer_data)
        route_items_dict[cust_id] = customer_data['items']
    
    # Extract the list of stochastic components involved in this route
    stochastic_items_info = []
    for customer in route_customer_data:
        if customer.get('is_stochastic', False):
            for item in customer['items']:
                if item.get('is_stochastic', False):
                    stochastic_items_info.append({
                        'customer_id': customer['id'],
                        'item_id': item['item_id'],
                        'realizations': item['realizations']
                    })
    
    # If the list of stochastic items is too large, return True to not over-constrain
    # because checking all possible scenarios is computationally expensive
    if len(stochastic_items_info) > 6:  # arbitrary small number for tractability
        return True
    
    # If no stochastic items, just use the deterministic values
    if not stochastic_items_info:
        items_for_packing = []
        for customer in route_customer_data:
            for item in customer['items']:
                r = item['realizations'][0]  # Only one for deterministic
                items_for_packing.append({
                    'height': r['height'],
                    'width': r['width'],
                    'weight': r['weight'],
                    'customer_id': customer['id']
                })
        return is_packable(items_for_packing, vehicle_H, vehicle_W, vehicle_Q)
    
    # Generate all possible realizations for this route
    realization_lists = [s['realizations'] for s in stochastic_items_info]
    if len(realization_lists) == 0:
        return True
        
    # Calculate total possible scenarios
    num_scenarios = 1
    for rl in realization_lists:
        num_scenarios *= len(rl)
        
    if num_scenarios > 1000:  # Too many to enumerate
        return True
    
    # Enumerate scenarios
    all_combinations = list(itertools.product(*realization_lists))
    for combo in all_combinations:
        items_for_packing = []
        combo_idx = 0
        for customer in route_customer_data:
            for item in customer['items']:
                if item.get('is_stochastic', False):
                    r_data = combo[combo_idx]
                    combo_idx += 1
                else:
                    r_data = item['realizations'][0]  # Deterministic
                    
                items_for_packing.append({
                    'height': r_data['height'],
                    'width': r_data['width'],
                    'weight': r_data['weight'],
                    'customer_id': customer['id']
                })
        
        if not is_packable(items_for_packing, vehicle_H, vehicle_W, vehicle_Q):
            return False
    
    return True


def is_packable(items, H, W, Q):
    """Simplified 2D packing feasibility check with weight and LIFO requirements"""
    # Verify weight constraint
    total_weight = sum(item['weight'] for item in items)
    if total_weight > Q:
        return False

    # Group by customers in order of route
    customer_sequence = []
    customer_items = defaultdict(list)
    for item in items:
        customer_id = item['customer_id']
        if customer_id not in [c for c, _ in customer_sequence]:
            customer_sequence.append((customer_id, []))
            customer_idx = len(customer_sequence)-1
        else:
            customer_idx = [i for i,(c,_) in enumerate(customer_sequence) if c==customer_id][0]
        customer_sequence[customer_idx][1].append(item)

    # Check if we can pack items using shelf-based packing with LIFO
    current_height = 0
    for customer_id, customer_item_list in customer_sequence:
        # Pack all items for this customer at vertical position current_height
        max_h_this_customer = 0
        current_width = 0
        for item in customer_item_list:
            # Try to place this item
            if current_width + item['width'] <= W:
                # Fits in current horizontal position
                current_width += item['width']
                max_h_this_customer = max(max_h_this_customer, item['height'])
            else:
                # Move to next row
                current_height += max_h_this_customer
                current_width = item['width']
                max_h_this_customer = item['height']
        
        if current_height + max_h_this_customer > H:
            return False
        current_height += max_h_this_customer

    return True


def solve_with_gurobi(instance, logger):
    """Create and solve MIP model for VRPSD with packing constraints."""
    # Constants
    n_customers = instance['parameters']['n_customers']
    K = instance['vehicle']['K']
    vehicle_H = instance['vehicle']['H']
    vehicle_W = instance['vehicle']['W']
    vehicle_Q = instance['vehicle']['Q']
    recourse_cost = instance['parameters']['recourse_cost_cf']
    distance_matrix = instance['distance_matrix']
    
    customer_ids = [c['id'] for c in instance['customers']]
    
    # Create map from customer id to customer data
    customer_data_map = {c['id']: c for c in instance['customers']}
    
    # Precompute expected areas/weights
    expected_values = {}
    for c in instance['customers']:
        exp_area, exp_weight = calculate_expected_area_weight(c)
        expected_values[c['id']] = {'area': exp_area, 'weight': exp_weight}
    
    m = gp.Model("VRPSD_Packing")
    m.setParam("Threads", 1)
    m.setParam(GRB.Param.LogToConsole, 0)
    
    # Decision variables: x[i][j] = whether we go from i to j (0 = depot)
    x = {}
    for i in range(n_customers + 1):  # 0 is depot
        for j in range(n_customers + 1):
            if i != j:
                x[i, j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    
    # MTZ formulation for subtour elimination
    u = {}
    for i in customer_ids:
        u[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=1, ub=len(customer_ids), name=f"u_{i}")
    
    # Constraints: each customer must be entered and left exactly once
    for j in customer_ids:
        m.addConstr(gp.quicksum(x[i, j] for i in range(n_customers + 1) if i != j) == 1, f"enter_{j}")
        m.addConstr(gp.quicksum(x[j, i] for i in range(n_customers + 1) if i != j) == 1, f"leave_{j}")
    
    # Exactly K vehicles leave and return to depot
    m.addConstr(gp.quicksum(x[0, j] for j in customer_ids) == K, "veh_departures")
    m.addConstr(gp.quicksum(x[i, 0] for i in customer_ids) == K, "veh_returns")
    
    # MTZ subtour elimination (adapted for multiple vehicles)
    for i in customer_ids:
        for j in customer_ids:
            if i != j:
                m.addConstr(u[i] - u[j] + 1 <= (len(customer_ids)+1) * (1 - x[i,j]), f"mtz_{i}_{j}")
    
    # Capacity constraints based on expected values (upper bound estimates)
    max_capacity_by_area = math.ceil(vehicle_H * vehicle_W / (1 if all(expected_values[cid]['area'] <= 1 for cid in expected_values) else min(v for v in [expected_values[cid]['area'] for cid in expected_values] if v > 0)))
    max_capacity_by_weight = math.ceil(vehicle_Q / (1 if all(expected_values[cid]['weight'] <= 1 for cid in expected_values) else min(v for v in [expected_values[cid]['weight'] for cid in expected_values] if v > 0)))
    estimated_max_customers_per_route = min(max_capacity_by_area, max_capacity_by_weight, len(customer_ids)//K + 1)

    # Subtour elimination with expected capacity
    all_customers_set = set(customer_ids)
    for i in customer_ids:
        for S_size in range(estimated_max_customers_per_route+1, len(customer_ids)//2+2):
            for S in itertools.combinations([c for c in customer_ids if c != i], S_size-1):
                S_set = set(S + (i,))
                # Max demand estimate for this subset - both area and weight
                total_exp_area = sum(expected_values[cid]['area'] for cid in S_set)
                total_exp_weight = sum(expected_values[cid]['weight'] for cid in S_set)
                
                max_vehicle_capacity_area = math.ceil(max(total_exp_area / (vehicle_H * vehicle_W), 1))
                max_vehicle_capacity_weight = math.ceil(max(total_exp_weight / vehicle_Q, 1))
                
                min_vehicles_needed = max(max_vehicle_capacity_area, max_vehicle_capacity_weight)
                
                # Sum of edges inside subset S
                inner_edges = gp.quicksum(x[p, q] for p in S_set for q in S_set if p != q)
                
                # Each vehicle can do at most |S|-1 edges within S
                m.addConstr(inner_edges <= len(S_set) - min_vehicles_needed)
                    
    # Objective function
    total_routing_cost = gp.quicksum(distance_matrix[i][j] * x[i, j] 
                                   for i in range(n_customers + 1) 
                                   for j in range(n_customers + 1) 
                                   if i != j)
    
    m.setObjective(total_routing_cost, GRB.MINIMIZE)
    
    # Set additional parameters
    time_remaining = 70  # Leave 10s for post-processing
    m.setParam(GRB.Param.TimeLimit, time_remaining)
    
    m.optimize()
    
    if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and m.solCount > 0:
        # Extract the routes
        active_edges = [(i, j) for (i, j) in x if x[i, j].x > 0.99]
        
        # Build route structure
        routes = extract_routes_from_active_edges(active_edges, K)
        
        # Validate each route against actual packing
        total_routing_cost = 0
        total_penalty = 0
        valid_routes = []
        
        for route in routes:
            if len(route) == 0:
                continue
                
            # Add routing cost
            prev_node = 0  # depot
            route_cost = 0
            for cust_id in route:
                route_cost += distance_matrix[prev_node][cust_id]
                prev_node = cust_id
            route_cost += distance_matrix[prev_node][0]  # back to depot
            total_routing_cost += route_cost
            
            # Add penalty cost if route is not packable
            is_packable = check_2d_packing_feasibility(route, customer_data_map, vehicle_H, vehicle_W, vehicle_Q, instance['customers'])
            if not is_packable:
                total_penalty += recourse_cost * len(route)
            else:
                valid_routes.append(route)
                
        routes = valid_routes
        objective_value = total_routing_cost + total_penalty
        
        return routes, objective_value
    else:
        return [], float('inf')


def extract_routes_from_active_edges(active_edges, K):
    """Extract ordered routes from active edges."""
    routes = []
    
    # Create a mapping from origin to destination
    next_nodes = {}
    for i, j in active_edges:
        next_nodes[i] = j
    
    # Find starting points - nodes going from depot
    starts = [next_nodes[0]]
    for _ in range(1, K):  # There will be K routes starting from depot
        if len(starts) >= K:
            break
        current = next_nodes.get(starts[-1])
        if current and current == 0:  # End of route back to depot
            starts.append(next_nodes[0])  # Start new route
        
    # Now build each route
    used_nodes = set()
    routes = []
    
    for start_node in starts:
        if start_node == 0:  # if this is empty route, skip
            continue
            
        current = start_node
        route = []
        while current != 0 and current not in used_nodes and len(route) < 500: # Prevent infinite loops
            route.append(current)
            used_nodes.add(current)
            if current in next_nodes:
                current = next_nodes[current]
            else:
                break # No more connections
        
        if route:
            routes.append(route)
            
        # Limit number of routes extracted
        if len(routes) >= K:
            break
    
    # Make sure we have K routes
    while len(routes) < K:
        routes.append([])
    
    return routes


def nearest_neighbor_routes(instance):
    """Basic initial solution using nearest neighbor clustering."""
    n_customers = instance['parameters']['n_customers']
    K = instance['vehicle']['K']
    distance_matrix = instance['distance_matrix']
    customer_ids = [c['id'] for c in instance['customers']]
    
    # Simple greedy assignment - assign closest customers to each route in turn
    unassigned = [c['id'] for c in instance['customers']]
    
    routes = [[] for _ in range(K)]
    current_route = 0  # Round-robin assignment
    
    # Assign customers in distance-to-depot order to different vehicles
    depot_dist = [(distance_matrix[0][i], i) for i in range(1, n_customers+1)]
    depot_dist.sort()  # Sort by distance to depot
    
    for _, customer_id in depot_dist:
        routes[current_route % K].append(customer_id)
        current_route += 1
    
    # Calculate cost 
    customer_data_map = {c['id']: c for c in instance['customers']}
    H, W, Q = instance['vehicle']['H'], instance['vehicle']['W'], instance['vehicle']['Q']
    recourse_cost = instance['parameters']['recourse_cost_cf']
    
    total_cost = 0
    actual_routes = []
    
    for route in routes:
        # Check if route satisfies capacity constraints
        route_ok = check_2d_packing_feasibility(route, customer_data_map, H, W, Q, instance['customers'])
        
        # Calculate routing cost
        route_cost = 0
        prev_node = 0  # depot
        for cust_id in route:
            route_cost += distance_matrix[prev_node][cust_id]
            prev_node = cust_id
        route_cost += distance_matrix[prev_node][0]  # back to depot
        
        if not route_ok:
            route_cost += recourse_cost * len(route)
        
        total_cost += route_cost
        
        if route:  # only keep non-empty routes
            actual_routes.append(route)
        
    return actual_routes, total_cost


def main():
    parser = argparse.ArgumentParser(description='VRPSD with 2D packing optimization')
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense='minimize') if args.log_path else None
    
    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    # Run greedy initial solution
    init_routes, init_cost = nearest_neighbor_routes(instance)
    best_routes = init_routes
    best_obj = init_cost
    
    if logger:
        logger.log(init_cost)
    
    # Run optimization if needed
    try:
        solved_routes, solved_obj = solve_with_gurobi(instance, logger)
        if solved_obj < best_obj:
            best_routes = solved_routes
            best_obj = solved_obj
            if logger:
                logger.log(solved_obj)
    except Exception as e:
        print(f"Optimization failed: {e}, proceeding with initial solution")
        
    # Save solution
    solution = {
        "objective_value": float(best_obj),
        "routes": best_routes
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()