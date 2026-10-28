import argparse
import json
import time
from solution_logger import SolutionLogger

def check_capacity(route, q, C):
    """
    Check if a route satisfies the vehicle capacity constraint at every point.
    The route is a list of expanded nodes (including depot at start and end).
    q: list of demands for expanded nodes (0-indexed, index 0 to 2n).
    C: vehicle capacity.
    """
    # Calculate initial load: sum of delivery demands in the route
    initial_load = 0
    for node in route:
        if 1 <= node <= len(q)//2:  # delivery node (1 to n)
            initial_load += -q[node]  # since delivery demand is stored as negative
    
    current_load = initial_load
    # Skip the starting depot (index0) and check after each subsequent node
    for i in range(1, len(route) - 1):  # exclude the last depot
        node = route[i]
        current_load += q[node]
        if current_load < 0 or current_load > C:
            return False
    return True

def compute_route_distance(route, dist_matrix):
    """Compute the total distance of a route given the expanded distance matrix."""
    total = 0
    for i in range(len(route) - 1):
        from_node = route[i]
        to_node = route[i+1]
        total += dist_matrix[from_node][to_node]
    return total

def main():
    parser = argparse.ArgumentParser(description='Vehicle Routing with Split Deliveries and Pickups')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the input JSON instance file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the output JSON solution file')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log intermediate solutions (JSONL format)')
    args = parser.parse_args()

    # Initialize solution logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start_time = time.time()

    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['num_customers']
    C = instance['vehicle_capacity']
    customers_list = instance['customers']
    original_matrix = instance['distance_matrix']  # (n+1) x (n+1) matrix

    # Build expanded distance matrix (2n+1) x (2n+1)
    expanded_distance = [[0] * (2*n+1) for _ in range(2*n+1)]
    for i in range(2*n+1):
        if i == 0:
            orig_i = 0
        elif i <= n:
            orig_i = i
        else:
            orig_i = i - n

        for j in range(2*n+1):
            if j == 0:
                orig_j = 0
            elif j <= n:
                orig_j = j
            else:
                orig_j = j - n
            expanded_distance[i][j] = original_matrix[orig_i][orig_j]

    # Build demand array q for expanded nodes (0 to 2n)
    q = [0] * (2*n+1)
    for i in range(1, n+1):  # delivery nodes: 1 to n
        # Find customer corresponding to delivery node i (customer at index i-1 in the list)
        customer = customers_list[i-1]
        q[i] = -customer['delivery_demand']  # negative for delivery
    for i in range(n+1, 2*n+1):  # pickup nodes: n+1 to 2n
        customer = customers_list[i - n - 1]  # index = (i - n - 1)
        q[i] = customer['pickup_demand']

    # Insertion heuristic
    routes = []  # list of routes, each is [0, ... , 0]
    unvisited = list(range(1, 2*n+1))  # all expanded nodes (1 to 2n)

    # Sort unvisited nodes by distance from depot (descending) to prioritize farther nodes
    unvisited.sort(key=lambda node: expanded_distance[0][node], reverse=True)

    # Process each unvisited node
    while unvisited and (time.time() - start_time < args.time_limit):
        node = unvisited.pop(0)
        best_increase = float('inf')
        best_route_idx = None
        best_pos = None

        # Try inserting into existing routes
        for idx, route in enumerate(routes):
            for pos in range(1, len(route)):  # positions: after the start depot and before the end depot
                new_route = route[:pos] + [node] + route[pos:]
                if check_capacity(new_route, q, C):
                    new_dist = compute_route_distance(new_route, expanded_distance)
                    old_dist = compute_route_distance(route, expanded_distance)
                    increase = new_dist - old_dist
                    if increase < best_increase:
                        best_increase = increase
                        best_route_idx = idx
                        best_pos = pos

        # Consider new route
        new_route0 = [0, node, 0]
        if check_capacity(new_route0, q, C):
            new_route0_dist = compute_route_distance(new_route0, expanded_distance)
        else:
            new_route0_dist = float('inf')  # mark infeasible

        # Decision: insert into existing route or create new route
        if best_route_idx is None and new_route0_dist == float('inf'):
            # Fallback: create new route even if it exceeds capacity? (shouldn't happen for feasible instances)
            routes.append(new_route0)
        elif best_route_idx is None:
            routes.append(new_route0)
        else:
            if new_route0_dist < best_increase:
                routes.append(new_route0)
            else:
                # Insert into existing route
                routes[best_route_idx] = routes[best_route_idx][:best_pos] + [node] + routes[best_route_idx][best_pos:]

    # Assign any remaining unvisited nodes (if time runs out) to new routes
    while unvisited:
        node = unvisited.pop(0)
        routes.append([0, node, 0])

    # Compute initial total distance
    total_distance = sum(compute_route_distance(route, expanded_distance) for route in routes)
    if logger:
        logger.log(total_distance)

    # Local search: 2-opt (first improvement per route)
    if time.time() - start_time < args.time_limit:
        improved = True
        while improved and (time.time() - start_time < args.time_limit):
            improved = False
            for idx in range(len(routes)):
                route = routes[idx]
                if len(route) <= 4:  # [0, node, 0] or [0, node1, node2, 0] -> no 2-opt possible
                    continue
                current_dist = compute_route_distance(route, expanded_distance)
                new_route = route  # start with current route
                # Try all 2-opt swaps (reverse segment between i and j)
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        # Create new route by reversing segment [i, j]
                        candidate = route[:i] + list(reversed(route[i:j+1])) + route[j+1:]
                        if check_capacity(candidate, q, C):
                            candidate_dist = compute_route_distance(candidate, expanded_distance)
                            if candidate_dist < current_dist:
                                new_route = candidate
                                current_dist = candidate_dist
                                improved = True
                                break  # break inner loop
                    if improved:
                        break  # break outer loop to restart the while
                if improved:
                    routes[idx] = new_route
                    break  # break route loop to restart the while

        # Recompute total distance after 2-opt
        total_distance = sum(compute_route_distance(route, expanded_distance) for route in routes)
        if logger:
            logger.log(total_distance)

    # Build detailed routes for output
    routes_detailed = []
    for route in routes:
        detailed_route = []
        for node in route:
            if node == 0:
                detailed_route.append({
                    "role": "depot",
                    "customer_id": None,
                    "delivery": 0,
                    "pickup": 0
                })
            elif 1 <= node <= n:
                # Delivery node: corresponds to customer at index (node-1) in the list
                customer = customers_list[node-1]
                detailed_route.append({
                    "role": "linehaul",
                    "customer_id": customer['id'],
                    "delivery": customer['delivery_demand'],
                    "pickup": 0
                })
            else:  # pickup node: n+1 to 2n
                idx = node - n - 1
                customer = customers_list[idx]
                detailed_route.append({
                    "role": "backhaul",
                    "customer_id": customer['id'],
                    "delivery": 0,
                    "pickup": customer['pickup_demand']
                })
        routes_detailed.append(detailed_route)

    # Write solution
    solution = {
        "objective_value": total_distance,
        "routes": routes,
        "routes_detailed": routes_detailed
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()