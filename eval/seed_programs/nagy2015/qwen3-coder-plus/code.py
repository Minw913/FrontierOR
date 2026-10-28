import argparse
import json
import math
import random
import copy
from solution_logger import SolutionLogger


def calculate_distance(x1, y1, x2, y2):
    return round(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))


def compute_route_distance(route, distance_matrix):
    if not route or len(route) < 2:
        return 0
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i]][route[i+1]]
    return total_distance


def evaluate_solution(routes, distance_matrix):
    total_distance = sum(compute_route_distance(route, distance_matrix) for route in routes)
    return total_distance


class HeuristicVRPSolver:
    def __init__(self, input_data):
        self.num_customers = input_data["num_customers"]
        self.distance_matrix = input_data["distance_matrix"]
        self.vehicle_capacity = input_data["vehicle_capacity"]
        self.customers = input_data["customers"]
        self.depot = input_data["depot"]
        
        # Create a mapping from customer ID to customer object for easy access
        self.customer_map = {c["id"]: c for c in self.customers}
        
    def construct_initial_solution(self):
        """
        Construct an initial valid solution respecting the capacity constraints.
        Each customer's pickup and delivery can be satisfied separately.
        """
        # We'll represent delivery nodes as 1 to n, and pickup nodes as n+1 to 2n
        delivery_nodes = []
        pickup_nodes = []
        
        for i in range(self.num_customers):
            customer_id = i + 1
            delivery_demand = self.customer_map[customer_id]["delivery_demand"]
            pickup_demand = self.customer_map[customer_id]["pickup_demand"]
            
            if delivery_demand > 0:
                delivery_nodes.append((customer_id, delivery_demand))
            if pickup_demand > 0:
                pickup_nodes.append((self.num_customers + customer_id, pickup_demand))

        routes = []

        # Schedule all delivery visits first
        routes.extend(self.build_routes_from_nodes(delivery_nodes, delivery=True))

        # Schedule all pickup visits
        routes.extend(self.build_routes_from_nodes(pickup_nodes, delivery=False))

        return routes

    def build_routes_from_nodes(self, nodes_with_demands, delivery=True):
        """
        Schedule a list of (node_id, demand) pairs into routes respecting capacity constraints.
        """
        routes = []
        unassigned = nodes_with_demands[:]
        
        while unassigned:
            route = [0]  # start from depot
            current_load = 0
            i = 0  
            
            while i < len(unassigned):
                node_id, demand = unassigned[i]
                if current_load + demand <= self.vehicle_capacity:
                    # Map the actual node id according to delivery/pickup
                    actual_node_id = node_id if delivery else node_id
                    route.append(actual_node_id)
                    current_load += demand
                    del unassigned[i]  # remove assigned item
                else:
                    i += 1  # try next item
            
            if len(route) > 1:
                route.append(0)  # return to depot
                routes.append(route)
        
        return routes

    def two_opt(self, route, distance_matrix, max_improvement_passes=10):
        """Perform 2-opt local search improvement on the route"""
        best_route = route[:]
        best_distance = compute_route_distance(route, distance_matrix)
        
        for _ in range(max_improvement_passes):
            improved = False
            for i in range(1, len(route) - 2):  # Skip depot nodes at start/end
                for j in range(i + 1, len(route) - 1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    
                    new_distance = compute_route_distance(new_route, distance_matrix)
                    
                    if new_distance < best_distance:
                        best_distance = new_distance
                        best_route = new_route[:]
                        route = best_route  # Update the route to continue improvement
                        improved = True
                        # No need to continue 2opt if we make changes, restart with updated route
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return best_route

    def local_search_optimization(self, routes, distance_matrix):
        """Apply local search methods to improve the routes"""
        routes = [route[:] for route in routes]  # Make a copy
        best_distance = evaluate_solution(routes, self.distance_matrix)
        
        no_improvement_count = 0
        max_no_improvement = 3  # Stop after 3 rounds without improvement
        
        while no_improvement_count < max_no_improvement:
            improved = False
            for i in range(len(routes)):
                new_route = self.two_opt(routes[i], distance_matrix)
                if new_route != routes[i]:
                    old_route = routes[i]
                    routes[i] = new_route
                    
                    new_distance = evaluate_solution(routes, self.distance_matrix)
                    if new_distance < best_distance:
                        best_distance = new_distance
                        improved = True
                        if 'logger' in vars(globals().get('__builtins__', {})) or 'logger' in globals():
                            if 'logger' in globals() and logger:
                                logger.log(best_distance)
                    else:
                        routes[i] = old_route  # Revert change if not improvement
            
            if not improved:
                no_improvement_count += 1
            else:
                no_improvement_count = 0  # Reset counter on improvement
                
        return routes, best_distance
    
    def get_node_details(self, node_id):
        """Translate node ID (expanded form) back to customer info"""
        if node_id == 0:  # Depot
            return 0, "depot", 0, 0
        elif 1 <= node_id <= self.num_customers:  # Delivery node
            customer_id = node_id
            customer = self.customer_map[customer_id]
            return customer_id, "linehaul", customer["delivery_demand"], 0
        elif self.num_customers + 1 <= node_id <= 2 * self.num_customers:  # Pickup node
            customer_id = node_id - self.num_customers
            customer = self.customer_map[customer_id]
            return customer_id, "backhaul", 0, customer["pickup_demand"]
        else:
            raise ValueError(f"Invalid node_id: {node_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False)
    
    args = parser.parse_args()
    
    # Initialize logger globally so it can be accessed inside methods
    global logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        input_data = json.load(f)
    
    # Initialize heuristic solver
    solver = HeuristicVRPSolver(input_data)
    
    # Build initial solution
    all_routes = solver.construct_initial_solution()
    
    # Calculate initial total distance
    total_distance = evaluate_solution(all_routes, input_data["distance_matrix"])
    
    if logger:
        logger.log(total_distance)
    
    # Further optimize routes using local search
    optimized_routes, final_distance = solver.local_search_optimization(all_routes, input_data["distance_matrix"])
    
    # Prepare the final solution
    result = {
        "objective_value": float(final_distance),
        "routes": optimized_routes,
        "routes_detailed": []
    }
    
    for route in optimized_routes:
        detailed_route = []
        for node_id in route:
            customer_id, role, delivery_qty, pickup_qty = solver.get_node_details(node_id)
            stop_info = {
                "id": node_id,
                "role": role,
                "customer_id": customer_id,
                "delivery_quantity": delivery_qty,
                "pickup_quantity": pickup_qty
            }
            detailed_route.append(stop_info)
        
        result["routes_detailed"].append(detailed_route)
    
    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()