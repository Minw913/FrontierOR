import argparse
import json
import math
import random
import copy
from solution_logger import SolutionLogger


def calculate_distance(x1, y1, x2, y2):
    return round(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))


def split_customer_requests(customers):
    """Split each customer into two nodes: delivery and pickup"""
    expanded_nodes = []
    node_to_customer_map = {}
    
    # Add depot first
    # Node ID 0 remains depot
    
    for cust_idx, cust in enumerate(customers):
        # Delivery node (linehaul) - has original id but represents delivery only
        delivery_node = {
            "id": cust["id"],  # 1-indexed for deliveries
            "original_id": cust["id"],
            "x": cust["x"],
            "y": cust["y"],
            "demand": cust["delivery_demand"],  # This goes from depot to customer
            "type": "linehaul"
        }
        
        # Pickup node (backhaul) - offset to distinguish from deliveries
        pickup_node = {
            "id": len(customers) + cust["id"],  # n+1 indexed for pickups 
            "original_id": cust["id"],
            "x": cust["x"],
            "y": cust["y"],
            "demand": cust["pickup_demand"],  # This goes from customer to depot
            "type": "backhaul"
        }
        
        expanded_nodes.append(delivery_node)
        expanded_nodes.append(pickup_node)
        node_to_customer_map[delivery_node['id']] = cust_idx
        node_to_customer_map[pickup_node['id']] = cust_idx
    
    return expanded_nodes, node_to_customer_map


def find_nearest_neighbor_path(nodes, start_node_idx, used_nodes, vehicle_capacity, node_id_to_idx, distances):
    """Nearest neighbor heuristic for finding initial routes"""
    path = [start_node_idx]
    current_load = nodes[start_node_idx]['demand']
    current_pos_idx = start_node_idx
    used_nodes.add(start_node_idx)
    
    while True:
        closest_node = None
        min_dist = float('inf')
        
        for i, node in enumerate(nodes):
            if i not in used_nodes and nodes[i]['type'] == nodes[current_pos_idx]['type']:
                # Only consider same type (linehaul or backhaul) and respect capacity
                if current_load + node['demand'] <= vehicle_capacity:
                    if distances[nodes[current_pos_idx]['id']][nodes[i]['id']] < min_dist:
                        min_dist = distances[nodes[current_pos_idx]['id']][nodes[i]['id']]
                        closest_node = i
        
        if closest_node is not None:
            path.append(closest_node)
            used_nodes.add(closest_node)
            current_pos_idx = closest_node
            current_load += nodes[closest_node]['demand']
        else:
            break
    
    return path


def construct_initial_solution_customers_based(input_data):
    """Construct initial solution respecting problem constraints"""
    distance_matrix = input_data["distance_matrix"]
    vehicle_capacity = input_data["vehicle_capacity"]
    customers = input_data["customers"]
    depot = input_data["depot"]
    
    # First try to serve delivery and pickup as much as possible together
    all_routes = []
    
    # Create expanded nodes (but we need to track them properly)
    used_nodes = set()
    
    # Create copies for processing
    delivery_nodes = []
    pickup_nodes = []
    
    for i, c in enumerate(customers):
        delivery_nodes.append(("delivery", i, c["delivery_demand"]))
        pickup_nodes.append(("pickup", i, c["pickup_demand"]))
    
    # Process each node type separately but smartly
    unvisited = list(range(len(customers)))
    
    # Phase 1: Try to make full trips (delivery + return for pickup) when possible
    visited_customers = set()
    
    while len(visited_customers) < len(customers):
        # Form route with as many customers as possible respecting capacity
        route = [0]  # Start at depot
        route_details = [{
            "id": 0,
            "role": "depot",
            "customer_id": 0,
            "delivery_quantity": 0,
            "pickup_quantity": 0
        }]
        
        load_on_vehicle = 0  # Initially no goods loaded
        
        # First pass: make delivery stops respecting capacity
        remaining_customers = [i for i in range(len(customers)) if i not in visited_customers]
        random.shuffle(remaining_customers)
        
        # Try to do multiple deliveries respecting load capacity
        deliveries_made = []
        temp_load_on_vehicle = 0
        
        for cust_idx in remaining_customers:
            if temp_load_on_vehicle + customers[cust_idx]["delivery_demand"] <= vehicle_capacity:
                deliveries_made.append(cust_idx)
                temp_load_on_vehicle += customers[cust_idx]["delivery_demand"]
        
        # Add these delivery stops to our route
        for cust_idx in deliveries_made:
            route.append(cust_idx + 1)  # Node numbering starts at 1
            route_details.append({
                "id": cust_idx + 1,
                "role": "linehaul",
                "customer_id": cust_idx + 1,  # Original customer id
                "delivery_quantity": customers[cust_idx]["delivery_demand"],
                "pickup_quantity": 0
            })
            visited_customers.add(cust_idx)
        
        # Return to depot - now vehicle has no cargo since it delivered everything
        # Now go pick up the goods and return to depot
        route.append(0)  # Back to depot after deliveries
        route_details.append({
            "id": 0,
            "role": "depot",
            "customer_id": 0,
            "delivery_quantity": 0,
            "pickup_quantity": 0
        })
        
        # Now add the pickups for those customers we just served
        # Add pickup stops after returning to depot
        pickup_route = [0]  # Start from depot again
        temp_pickup_details = [{
            "id": 0,
            "role": "depot",
            "customer_id": 0,
            "delivery_quantity": 0,
            "pickup_quantity": 0
        }]
        
        # Go pick up the goods and bring them back
        current_load = 0
        for cust_idx in deliveries_made:
            pickup_qty = customers[cust_idx]["pickup_demand"]
            if current_load + pickup_qty <= vehicle_capacity:
                pickup_node_id = len(customers) + cust_idx + 1  # Pickup nodes are offset by n
                pickup_route.append(pickup_node_id)
                temp_pickup_details.append({
                    "id": pickup_node_id,
                    "role": "backhaul",
                    "customer_id": cust_idx + 1,  # Original customer id
                    "delivery_quantity": 0,
                    "pickup_quantity": pickup_qty
                })
                current_load += pickup_qty
        
        # Return to depot
        pickup_route.append(0)
        temp_pickup_details.append({
            "id": 0,
            "role": "depot",
            "customer_id": 0,
            "delivery_quantity": 0,
            "pickup_quantity": 0
        })
        
        if len(pickup_route) > 2:  # Only add if there were actual pickups
            route.extend(pickup_route[1:])  # Skip first 0 since already there
            route_details.extend(temp_pickup_details[1:])
        
        all_routes.append(route)
    
    return all_routes, route_details


def get_distance_between_nodes(node1_id, node2_id, dist_matrix, num_customers):
    """Get distance between two nodes using the matrix"""
    return dist_matrix[node1_id][node2_id]


def compute_route_distance(route, distance_matrix):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i]][route[i+1]]
    return total_distance


def evaluate_solution(routes, distance_matrix):
    total_distance = sum(compute_route_distance(route, distance_matrix) for route in routes)
    return total_distance


def local_search_optimization(input_data, initial_routes):
    """Basic improvement via local searches"""
    distance_matrix = input_data["distance_matrix"]
    vehicle_capacity = input_data["vehicle_capacity"]
    customers = input_data["customers"]
    
    routes = copy.deepcopy(initial_routes)
    
    improved = True
    best_total_distance = evaluate_solution(routes, distance_matrix)
    
    # Try moves like relocate, swap, 2-opt
    iterations = 0
    max_iterations = 50  # Limit for practical purposes
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        for route_idx, route in enumerate(routes):
            # Try relocate: move a node from one position to another within same route or different route
            temp_routes = copy.deepcopy(routes)
            
            # Only work on non-depot nodes
            non_depot_nodes = [(i, node) for i, node in enumerate(route) if node != 0]
            
            if len(non_depot_nodes) <= 1:
                continue
                
            for pos_idx, orig_pos in enumerate(non_depot_nodes):
                node_to_move = orig_pos[1]
                
                # Try moving to different position in same route
                for new_pos in range(1, len(route) - 1):  # Don't put at beginning/end
                    if new_pos == orig_pos[0]:
                        continue
                        
                    new_route = route[:]
                    new_route.pop(orig_pos[0])  # Remove from old position
                    new_route.insert(new_pos, node_to_move)  # Insert at new position
                    
                    temp_routes = copy.deepcopy(routes)
                    temp_routes[route_idx] = new_route
                    
                    # Check if valid and better
                    try:
                        if is_valid_solution(temp_routes, input_data):
                            new_distance = evaluate_solution(temp_routes, distance_matrix)
                            if new_distance < best_total_distance:
                                routes = temp_routes
                                best_total_distance = new_distance
                                improved = True
                                break
                    except:
                        continue
            
            if improved:
                break
            
            # Try swap within same route
            for i in range(1, len(route)-1):  # Exclude depot
                for j in range(i+1, len(route)-1):  # Exclude depot
                    if route[i] != 0 and route[j] != 0:
                        new_route = route[:]
                        new_route[i], new_route[j] = new_route[j], new_route[i]
                        
                        temp_routes = copy.deepcopy(routes)
                        temp_routes[route_idx] = new_route
                        
                        try:
                            if is_valid_solution(temp_routes, input_data):
                                new_distance = evaluate_solution(temp_routes, distance_matrix)
                                if new_distance < best_total_distance:
                                    routes = temp_routes
                                    best_total_distance = new_distance
                                    improved = True
                                    break
                        except:
                            continue
                        
                if improved:
                    break
            if improved:
                break
    
    return routes, best_total_distance


def is_valid_solution(routes, input_data):
    """Check if the solution satisfies all constraints"""
    # Check capacity constraint and proper flow
    vehicle_capacity = input_data["vehicle_capacity"]
    depot_id = input_data["depot"]["id"]
    customers = {c["id"]: c for c in input_data["customers"]}  # Map customer id to customer
    num_customers = input_data["num_customers"]
    
    # Track whether each customer's delivery and pickup are completed
    delivery_completeness = {c["id"]: False for c in input_data["customers"]}
    pickup_completeness = {c["id"]: False for c in input_data["customers"]}
    
    for route in routes:
        current_load = 0
        if route[0] != depot_id:
            return False
        
        # Track loads separately
        delivery_load_for_vehicle = 0  # Goods going out
        pickup_load_from_vehicle = 0   # Goods coming back
        
        segment_before_returning_to_depot = True  # We assume initially we're making deliveries
        
        for node_ptr in range(1, len(route)):
            node_id = route[node_ptr]
            
            if node_id == depot_id:
                # When we return to depot, check that all deliveries are done
                # Now reset for pickups
                segment_before_returning_to_depot = False
                if delivery_load_for_vehicle > 0:
                    # This means this vehicle did deliveries but didn't have enough capacity for pickups
                    # Actually what matters is the flow constraint
                    pass  # For now
                else:
                    # Must be returning from pickups
                    pass
            elif 1 <= node_id <= num_customers:  # This is a delivery stop
                if segment_before_returning_to_depot:
                    # Making a delivery
                    customer_id = node_id
                    delivery_qty = customers[customer_id]["delivery_demand"]
                    
                    # Capacity check: when delivering, we start with total delivery load
                    if current_load + delivery_qty > vehicle_capacity:
                        return False
                    
                    current_load += delivery_qty
                    delivery_load_for_vehicle += delivery_qty
                    delivery_completeness[customer_id] = True
                else:
                    # After returning to depot, if visiting same ID again
                    # Should not be allowed since we said they can be separate
                    return False  # Each part done once per customer
                    
            elif num_customers + 1 <= node_id <= 2 * num_customers:  # This is a pickup stop
                if not segment_before_returning_to_depot:
                    # Making a pickup, we start empty after delivery phase ended
                    customer_id = node_id - num_customers
                    pickup_qty = customers[customer_id]["pickup_demand"]
                    
                    # We should be back to depot and start picking up
                    if current_load + pickup_qty > vehicle_capacity:
                        return False
                    
                    current_load += pickup_qty
                    pickup_load_from_vehicle += pickup_qty
                    pickup_completeness[customer_id] = True
                else:
                    # In delivery phase, shouldn't pick up
                    return False
            
            elif node_id == depot_id:
                # Check if capacity satisfied at this point
                if current_load > vehicle_capacity:
                    return False
    
    # Check all customers had both delivery and pickup done
    for cust_id in delivery_completeness:
        if not delivery_completeness[cust_id] or not pickup_completeness[cust_id]:
            return False
    
    return True


def convert_routes_to_final_format(routes, input_data):
    """Convert internal representation to final format"""
    num_customers = input_data["num_customers"]
    customers = {c["id"]: c for c in input_data["customers"]}
    
    final_routes = []
    detailed_routes = []
    
    for route in routes:
        final_route = []
        detailed_route = []
        
        for node_id in route:
            if node_id == 0:
                final_route.append(0)
                detailed_route.append({
                    "id": 0,
                    "role": "depot",
                    "customer_id": 0,
                    "delivery_quantity": 0,
                    "pickup_quantity": 0
                })
            elif 1 <= node_id <= num_customers:
                # Delivery at customer node_id
                cust_id = node_id
                final_route.append(node_id)
                detailed_route.append({
                    "id": cust_id,
                    "role": "linehaul",
                    "customer_id": cust_id,
                    "delivery_quantity": customers[cust_id]["delivery_demand"],
                    "pickup_quantity": 0
                })
            elif num_customers + 1 <= node_id <= 2 * num_customers:
                # Pickup at customer node_id - num_customers
                cust_id = node_id - num_customers
                expanded_node_id = node_id
                final_route.append(expanded_node_id)
                detailed_route.append({
                    "id": expanded_node_id,
                    "role": "backhaul",
                    "customer_id": cust_id,
                    "delivery_quantity": 0,
                    "pickup_quantity": customers[cust_id]["pickup_demand"]
                })
        
        final_routes.append(final_route)
        detailed_routes.append(detailed_route)
    
    return final_routes, detailed_routes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False)
    
    args = parser.parse_args()
    
    with open(args.instance_path, 'r') as f:
        input_data = json.load(f)
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Our strategy: Create individual routes for each delivery/pickup and then optimize
    all_routes = []
    distance_matrix = input_data["distance_matrix"]
    
    # For each customer, create separate trips for delivery and pickup
    for i, customer in enumerate(input_data["customers"]):
        customer_id = customer["id"]
        
        # Route for delivery (take goods from depot to customer)
        if customer["delivery_demand"] > 0:
            delivery_route = [0, customer_id, 0]
            all_routes.append(delivery_route)
        
        # Route for pickup (go to customer get goods, bring to depot)
        if customer["pickup_demand"] > 0:
            pickup_node_id = input_data["num_customers"] + customer_id  # Expanded node ID
            pickup_route = [0, pickup_node_id, 0]
            all_routes.append(pickup_route)
    
    # Calculate initial total distance
    total_distance = evaluate_solution(all_routes, distance_matrix)
    print(f"Starting solution distance: {total_distance}")

    if logger:
        logger.log(total_distance)
    
    # Perform local search improvements
    improved_routes, new_distance = local_search_optimization(input_data, all_routes)
    
    best_routes = improved_routes if new_distance < total_distance else all_routes
    best_distance = min(new_distance, total_distance)
    
    print(f"Improved solution distance: {best_distance}")
    
    final_routes, final_routes_detail = convert_routes_to_final_format(best_routes, input_data)
    
    result = {
        "objective_value": best_distance,
        "routes": final_routes,
        "routes_detailed": final_routes_detail
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(result, f)
    
    print("Solution saved.")


if __name__ == "__main__":
    main()