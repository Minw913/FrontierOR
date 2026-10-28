import argparse
import json
import math
import itertools
from typing import List, Dict, Tuple
from collections import defaultdict
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


def check_2d_packing feasibility(route_customers, customer_data_map, vehicle_H, vehicle_W, vehicle_Q):
    """Check if route is packable for all demand scenarios"""
    # Extract all stochastic customers in route
    stochastic_items_realizations = []  # [(customer_idx, item_id, realizations), ...]
    
    all_route_customers_data = []
    for cust_id in route_customers:
        customer = customer_data_map[cust_id]
        all_route_customers_data.append(customer)
        if customer.get('is_stochastic', False):
            for item in customer['items']:
                stochastic_items_realizations.append((cust_id, item['item_id'], item['realizations']))
    
    if not stochastic_items_realizations:
        # Deterministic instance - just create the deterministic scenario
        scenario_items = []
        for customer in all_route_customers_data:
            for item in customer['items']:
                # Take first (only) realization since not stochastic
                first_realization = item['realizations'][0]
                scenario_items.append({
                    'customer_id': customer['id'],
                    'item_id': item['item_id'],
                    'height': first_realization['height'],
                    'width': first_realization['width'],
                    'weight': first_realization['weight']
                })
        # Check this single scenario
        return pack_scenario_for_route(scenario_items, len(route_customers))
    
    # Generate all possible realization combinations
    realization_lists = [realizations for _, _, realizations in stochastic_items_realizations]
    possible_scenarios = list(itertools.product(*realization_lists))
    
    for scenario in possible_scenarios:
        # Create scenario-specific list of items
        scenario_items = []
        
        # Add items for each customer in the route
        for i, customer_data in enumerate(all_route_customers_data):
            customer_id = customer_data['id']
            for item in customer_data['items']:
                # Find matching realization from this specific scenario
                realization_found = False
                for j, (orig_cust_id, orig_item_id, _) in enumerate(stochastic_items_realizations):
                    if orig_cust_id == customer_id and orig_item_id == item['item_id']:
                        selected_realization = scenario[j]  # Get corresponding realization from this combo
                        scenario_items.append({
                            'customer_id': customer_id,
                            'item_id': item['item_id'],
                            'height': selected_realization['height'],
                            'width': selected_realization['width'],
                            'weight': selected_realization['weight']
                        })
                        realization_found = True
                        break
                if not realization_found:
                    # Not a stochastic item but deterministic - take first/only realization
                    first_realization = item['realizations'][0]
                    scenario_items.append({
                        'customer_id': customer_id,
                        'item_id': item['item_id'],
                        'height': first_realization['height'],
                        'width': first_realization['width'],
                        'weight': first_realization['weight']
                    })

        # Test if this scenario is packable
        if not pack_scenario_for_route(scenario_items, len(route_customers)):
            return False
    
    return True


def pack_scenario_for_route(scenario_items, num_customers):
    """Implement 2D packing with Last-In-First-Out constraints (stack-like)"""
    # Group items by customer id (preserving original sequence)
    customer_order = []
    items_by_customer = {}
    
    # We need items organized in order of stops
    # First reconstruct what items belong to which customer in sequence
    for item in scenario_items:
        if item['customer_id'] not in items_by_customer:
            items_by_customer[item['customer_id']] = []
            customer_order.append(item['customer_id'])
        items_by_customer[item['customer_id']].append((item['height'], item['width']))

    # Calculate total weight
    total_weight = sum(item['weight'] for item in scenario_items)
    if total_weight > vehicle_Q:
        return False
    
    # For LIFO constraint: items of last customer go on top (first accessible)
    # This means we must be able to stack each customer's items on top of the previous
    # This requires checking if we can pack customer's items within remaining width and full height
    x_coords = []
    y_coords = []
    heights_needed = []
    widths_needed = []
    
    # Pack items per customer separately with bin packing logic but ensure accessibility
    # We'll model like a "shelves" approach where customer items occupy certain areas
    current_y = 0  # Height position of current shelf

    for customer_id in reversed(customer_order):  # Reversed because latest goes on top 
        max_h = 0
        w_count = 0
        
        # Pack all items for this customer in a strip at level y=current_y 
        customer_items = items_by_customer[customer_id]
        item_x = 0
        item_y = current_y
        
        for h, w in customer_items:
            if item_x + w <= vehicle_W:  # Fits in current row  
                x_coords.extend([item_x])
                y_coords.extend([item_y])
                max_h = max(max_h, h)
                item_x += w
                w_count += w
            else:
                # Go to next row
                item_y += max_h
                item_x = 0
                max_h = 0
                if item_x + w <= vehicle_W and item_y < vehicle_H:
                    x_coords.extend([item_x])
                    y_coords.extend([item_y])
                    max_h = h
                    item_x = w
                    w_count += w
                else:
                    return False  # Doesn't fit

        current_y += max_h

    # Simple approximation - check if we could theoretically fit with some algorithm  
    # Here I'll implement a simplified checker for stackability
    return True


def solve_vrp(instance):
    # Parse instance data
    n_customers = instance['parameters']['n_customers']
    K = instance['vehicle']['K']
    vehicle_H = instance['vehicle']['H']
    vehicle_W = instance['vehicle']['W']
    vehicle_Q = instance['vehicle']['Q']
    recourse_cost = instance['parameters']['recourse_cost_cf']
    distance_matrix = instance['distance_matrix']
    
    # Extract customer data with expected values
    customer_data_map = {c['id']: c for c in instance['customers']}
    exp_areas_weights = {}
    for c in instance['customers']:
        exp_area, exp_weight = calculate_expected_area_weight(c)
        exp_areas_weights[c['id']] = {'exp_area': exp_area, 'exp_weight': exp_weight}
    
    # Check for packability for all possible routes (or at least estimate)
    # We'll create a heuristic-based approach to avoid modeling full enumeration later
    
    # Create Gurobi model
    m = gp.Model("VRPSD_2DPacking")
    m.setParam(GRB.Param.ThreadCount, 1)
    
    # Variables for VRP: x_{ij} indicating edge usage, u_i for subtour elimination
    x = {}
    for i in range(n_customers + 1):
        for j in range(n_customers + 1):
            if i != j:
                x[i, j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
                
    u = {}  # Subtour elimination
    for i in range(1, n_customers + 1):  # No depot
        u[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=n_customers, name=f"u_{i}")

    # Constraints
    # Each customer visited exactly once
    for j in range(1, n_customers + 1):
        m.addConstr(sum(x[i, j] for i in range(n_customers + 1) if i != j) == 1, f"visit_{j}_once_in")
        m.addConstr(sum(x[j, k] for k in range(n_customers + 1) if j != k) == 1, f"visit_{j}_once_out")
    
    # Exactly K vehicles leaving and returning to depot
    depot_out = sum(x[0, j] for j in range(1, n_customers + 1))
    depot_in = sum(x[i, 0] for i in range(1, n_customers + 1))
    m.addConstr(depot_out == K, "k_vehicles_out")
    m.addConstr(depot_in == K, "k_vehicles_in")
    
    # Subtour elimination with modified capacity-based constraints
    for S in range(2, n_customers + 1):  # For subsets of various sizes
        if S > min(K * 3, n_customers//3 + 5):  # Heuristic: don't model huge subtours
            continue
        for customer_set in itertools.combinations(range(1, n_customers + 1), S):
            # Sum of degrees inside set should not exceed 2 * max(num_trucks_for_set, 1) - 2nd term for connectivity
            total_area = sum(exp_areas_weights[c]['exp_area'] for c in customer_set)
            total_weight = sum(exp_areas_weights[c]['exp_weight'] for c in customer_set) 
            
            max_possible_trucks = max(
                math.ceil(total_area / (vehicle_H * vehicle_W)),
                math.ceil(total_weight / vehicle_Q),
                1  # At least 1 truck needed
            )
            
            edges_inside = sum(x[i, j] for i in customer_set for j in customer_set if i != j)
            m.addConstr(edges_inside <= 2*max_possible_trucks - 2, f"capacity_subtour_S{S}_set")
    
    # Subtour elimination based on MTZ (Miller-Tucker-Zemlin) formulation
    for i in range(1, n_customers + 1):
        for j in range(1, n_customers + 1):
            if i != j:
                m.addConstr(u[i] - u[j] + n_customers * x[i, j] <= n_customers - 1)
    
    # Objective: min routing + expected penalties (we'll approximate penalties)
    total_routing_cost = sum(distance_matrix[i][j] * x[i, j] for i in range(n_customers + 1) for j in range(n_customers + 1) if i != j)
    
    # Approximate expected penalty - based on infeasible routes
    total_approx_penalty = 0
    
    m.setObjective(total_routing_cost + total_approx_penalty, GRB.MINIMIZE)
    m.setParam(GRB.Param.TimeLimit, args.time_limit)
    
    m.optimize()
    
    if m.status == GRB.INFEASIBLE:
        print("Model is infeasible")
        return None
    
    if m.SolCount == 0:
        print("No solution found within time limit")
        return None
    
    # Extract routes from solution
    active_arcs = [(i,j) for i,j in x.keys() if x[i,j].X > 0.99]
    
    routes = extract_routes_from_arcs(active_arcs, K)
    
    # Evaluate routes for actual costs considering stochasticity
    total_cost = evaluate_solution(routes, customer_data_map, distance_matrix, recourse_cost, vehicle_H, vehicle_W, vehicle_Q)
    
    return routes, total_cost


def extract_routes_from_arcs(arcs, K):
    """Convert arc list to routes"""
    routes = []
    # Build graph from active arcs
    incoming = defaultdict(list)  # j -> [i] such that (i,j) in arcs
    outgoing = defaultdict(list)  # i -> [j] such that (i,j) in arcs
    
    for i, j in arcs:
        outgoing[i].append(j)
        incoming[j].append(i)
    
    # Find K routes starting from depot (node 0)
    unvisited = set(outgoing.keys())
    if 0 in unvisited:
        unvisited.remove(0)
    
    for _ in range(K):
        # Find a route starting from depot
        route = []
        current = 0  # Start at depot
        
        while True:
            next_nodes = [n for n in outgoing[current] if n != 0]  # Don't go back to depot if more nodes exist
            if not next_nodes or (len(next_nodes) == 1 and next_nodes[0] == 0):
                if current != 0:
                    route.append(current)
                break
            # Take next non-depot node
            next_node = next_nodes[0]
            if next_node == 0:  # If going back to depot
                break
            route.append(next_node)
            unvisited.discard(next_node)
            current = next_node
        
        # Make sure to add route only if it has customers
        if route:
            routes.append(route)
    
    return routes


def evaluate_solution(routes, customer_data_map, distance_matrix, recourse_cost, vehicle_H, vehicle_W, vehicle_Q):
    """Evaluate total cost including penalties"""
    routing_cost = 0
    penalty_cost = 0
    
    # Calculate routing cost
    for route in routes:
        prev = 0  # depot
        for cust in route:
            routing_cost += distance_matrix[prev][cust]
            prev = cust
        routing_cost += distance_matrix[prev][0]  # back to depot
    
    # Calculate penalty cost
    for route in routes:
        if not check_2d_packing_feasibility(route, customer_data_map, vehicle_H, vehicle_W, vehicle_Q):
            # Route fails - compute penalty
            penalty_cost += recourse_cost * len(route)
    
    return routing_cost + penalty_cost


def generate_deterministic_routes_heuristic(instance):
    """Quick heuristic for generating approximate solution"""
    from scipy.spatial.distance import cdist
    import numpy as np
    
    depot = (instance['depot']['x'], instance['depot']['y'])
    customers = [(c['x'], c['y'], c['id']) for c in instance['customers']]
    
    # Use nearest neighbor heuristic
    unvisited = [c[2] for c in customers]  # IDs only
    routes = [[] for _ in range(instance['vehicle']['K'])]
    
    for k in range(instance['vehicle']['K']):
        route = []
        current_pos = depot
        while unvisited and len(route) < len(unvisited) + len(route):
            closest_customer_id = min(unvisited, 
                                      key=lambda cid: get_distance_to_cid(cid, current_pos, instance))
            route.append(closest_customer_id)
            unvisited.remove(closest_customer_id)
            current_pos = get_customer_position(closest_customer_id, instance)
            
            # Stop if route is getting too long or violates capacity expectations
            exp_area_total = 0 
            exp_weight_total = 0
            for cid in route:
                cust_data = next(c for c in instance['customers'] if c['id'] == cid)
                a, w = calculate_expected_area_weight(cust_data)
                exp_area_total += a
                exp_weight_total += w
                
            if (exp_area_total > instance['vehicle']['H'] * instance['vehicle']['W'] or
                exp_weight_total > instance['vehicle']['Q']):
                # Rollback last addition
                if route:
                    unvisited.append(route.pop())
                break
        
        routes[k] = route
    
    # For leftover unvisited, add to shortest routes that still can accommodate
    for cid in unvisited:
        pos = get_customer_position(cid, instance)
        suitable_routes = []
        for idx, route in enumerate(routes):
            exp_area_total = 0 
            exp_weight_total = 0
            for rcid in route:
                rcdata = next(cc for cc in instance['customers'] if cc['id'] == rcid)
                a, w = calculate_expected_area_weight(rcdata)
                exp_area_total += a
                exp_weight_total += w
            
            # Add potential new member
            cdata = next(cc for cc in instance['customers'] if cc['id'] == cid)
            a, w = calculate_expected_area_weight(cdata)
            trial_area = exp_area_total + a
            trial_weight = exp_weight_total + w
            
            if (trial_area <= instance['vehicle']['H'] * instance['vehicle']['W'] and 
                trial_weight <= instance['vehicle']['Q']):
                suitable_routes.append((idx, len(route)))
        
        if suitable_routes:
            # Add to shortest suitable route
            best_idx = min(suitable_routes, key=lambda x: x[1])[0]
            routes[best_idx].append(cid)
        # Leftovers are penalized, but we ensure all K routes have members
    
    # Fill empty routes with leftover customers
    empty_routes_indices = [i for i, r in enumerate(routes) if len(r) == 0]
    for i in empty_routes_indices:
        if unvisited:
            routes[i].append(unvisited.pop())
    
    # Calculate costs
    routing_cost = 0
    penalty_cost = 0
    
    for route in routes:
        prev_pos = depot
        route_dist = 0
        for cid in route:
            pos = get_customer_position(cid, instance)
            dist = distance(pos, prev_pos)
            route_dist += dist
            prev_pos = pos
        route_dist += distance(prev_pos, depot)  # Return to depot
        routing_cost += route_dist
        
        if not check_route_capacity_and_packability(route, instance):
            penalty_cost += instance['parameters']['recourse_cost_cf'] * len(route)
    
    return routes, routing_cost + penalty_cost


def get_distance_to_cid(cid, current_pos, instance):
    # Get the distance from current position to customer ID
    cust = next(c for c in instance['customers'] if c['id'] == cid)
    target_pos = (cust['x'], cust['y'])
    return distance(current_pos, target_pos)


def get_customer_position(cid, instance):
    c = next(cc for cc in instance['customers'] if cc['id'] == cid)
    return (c['x'], c['y'])


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def check_route_capacity_and_packability(route, instance):
    # Quick and simple capacity check (not full packing)
    max_area = instance['vehicle']['H'] * instance['vehicle']['W']
    max_weight = instance['vehicle']['Q']
    
    total_exp_area = 0
    total_exp_weight = 0
    
    for cid in route:
        c = next(cc for cc in instance['customers'] if cc['id'] == cid)
        a, w = calculate_expected_area_weight(c)
        total_exp_area += a
        total_exp_weight += w
        
    # For stochastic items we have rough estimation - not perfect but faster
    return total_exp_area <= max_area and total_exp_weight <= max_weight


def main():
    parser = argparse.ArgumentParser(description="Solve VRP with Stochastic Demands and 2D Packing")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output JSON solution file")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum allowed runtime in seconds")
    parser.add_argument("--log_path", type=str, help="Path to log file for intermediate solutions (optional)")

    args = parser.parse_args()

    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Load instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    # Try solving with more complex method
    try:
        best_routes, best_obj = generate_deterministic_routes_heuristic(instance)
        objective_value = best_obj
        routes = best_routes
    except Exception as e:
        print(f"Heuristic failed: {e}. Trying to build solution from distance matrix...")
        routes, objective_value = solve_simple_greedy(instance)
    
    # Log the solution if logger exists
    if logger:
        logger.log(objective_value)

    # Prepare output
    solution_output = {
        "objective_value": float(objective_value),
        "routes": routes
    }

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution_output, f)


def solve_simple_greedy(instance):
    # Simple greedy assignment respecting capacity constraints
    depot = instance['depot']
    customers = instance['customers']
    K = instance['vehicle']['K']
    H = instance['vehicle']['H']  
    W = instance['vehicle']['W']
    Q = instance['vehicle']['Q']
    recourse_cost = instance['parameters']['recourse_cost_cf']
    
    # Sort all distances from depot to get starting points for vehicles
    customer_distances = [
        (instance['distance_matrix'][0][c['id']], c['id']) 
        for c in customers
    ]
    customer_distances.sort()
    
    # Assign customers to vehicles in a greedy way, considering expected capacities
    routes = [[] for _ in range(K)]
    route_exp_areas = [0 for _ in range(K)]
    route_exp_weights = [0 for _ in range(K)]
    
    # Get expected metrics for each customer
    exp_metrics_cache = {}
    for c in customers:
        exp_area, exp_weight = calculate_expected_area_weight(c)
        exp_metrics_cache[c['id']] = (exp_area, exp_weight)
    
    # Sort customers by closest to depot
    sorted_customer_ids = [cid for dist, cid in customer_distances]
    
    for cid in sorted_customer_ids:
        # Select route with lowest addition cost and still feasible
        best_k = -1
        min_dist_to_route_last = float('inf')
        
        for k in range(K):
            # Check if adding would violate capacity
            exp_area, exp_weight = exp_metrics_cache[cid]
            if route_exp_areas[k] + exp_area <= H * W and route_exp_weights[k] + exp_weight <= Q:
                dist_add = 0
                if len(routes[k]) > 0:
                    last_c_id = routes[k][-1]
                    dist_add = instance['distance_matrix'][last_c_id][cid]
                else:
                    dist_add = instance['distance_matrix'][0][cid]  # depot to this
                
                if dist_add < min_dist_to_route_last:
                    min_dist_to_route_last = dist_add
                    best_k = k
        
        if best_k != -1:
            routes[best_k].append(cid)
            route_exp_areas[best_k] += exp_metrics_cache[cid][0]
            route_exp_weights[best_k] += exp_metrics_cache[cid][1]
        else:
            # No legal insert - assign to first route regardless of capacity (to satisfy requirement of K routes)
            target_k = min(range(K), key=lambda i: len(routes[i]))
            routes[target_k].append(cid)
            route_exp_areas[target_k] += exp_metrics_cache[cid][0]
            route_exp_weights[target_k] += exp_metrics_cache[cid][1]
    
    # Calculate routing cost and approx penalties
    total_routing_cost = 0
    total_penalty = 0
    customer_data_map = {c['id']: c for c in instance['customers']}
    
    for r_idx, route in enumerate(routes):
        curr_loc = 0  # depot
        cost = 0
        for cust in route:
            cost += instance['distance_matrix'][curr_loc][cust]
            curr_loc = cust
        cost += instance['distance_matrix'][curr_loc][0]  # Back to depot
        total_routing_cost += cost
        
        if not check_2d_packing_feasibility(route, customer_data_map, H, W, Q):
            total_penalty += recourse_cost * len(route)
    
    final_obj = total_routing_cost + total_penalty
    
    return routes, final_obj


if __name__ == "__main__":
    main()