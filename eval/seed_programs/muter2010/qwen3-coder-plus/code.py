import argparse
import json
import math
import random
import itertools
from typing import List, Dict, Tuple
from datetime import datetime
from solution_logger import SolutionLogger


def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class VRPTWInstance:
    def __init__(self, data):
        self.num_customers = data["num_customers"]
        self.depot = data["depot"]
        self.customers = {c["id"]: c for c in data["customers"]}
        self.customer_list = data["customers"]
        self.vehicle_capacity = data["vehicle_capacity"]
        self.nodes = [data["depot"]] + data["customers"]  # depot is index 0
        self.node_dict = {node["id"]: i for i, node in enumerate(self.nodes)}
        
        # Compute distance matrix
        n = len(self.nodes)
        self.distance_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        self.id_to_idx = {node["id"]: idx for idx, node in enumerate(self.nodes)}
        self.idx_to_id = {idx: node["id"] for idx, node in enumerate(self.nodes)}
        
        for i, node1 in enumerate(self.nodes):
            for j, node2 in enumerate(self.nodes):
                self.distance_matrix[i][j] = calculate_distance(
                    node1["x"], node1["y"],
                    node2["x"], node2["y"]
                )

    def get_distance(self, id1, id2):
        """Get distance between two nodes by their IDs."""
        i, j = self.id_to_idx[id1], self.id_to_idx[id2]
        return self.distance_matrix[i][j]


class RouteValidator:
    def __init__(self, instance: VRPTWInstance):
        self.instance = instance
        self.depot = instance.depot
        self.customers = instance.customers
        self.capacity = instance.vehicle_capacity

    def validate_route(self, customer_ids: List[int]) -> bool:
        """Check if a route is feasible regarding capacity and time windows."""
        if not customer_ids:
            return True

        # Capacity check
        total_demand = sum(self.customers[cid]["demand"] for cid in customer_ids)
        if total_demand > self.capacity:
            return False

        # Time window check
        current_time = self.depot["ready_time"]
        current_loc = self.depot["id"]

        for curr_cid in customer_ids:
            # Travel from current location to next customer
            travel_time = self.instance.get_distance(current_loc, curr_cid)
            arrival_time = current_time + travel_time
            customer_info = self.customers[curr_cid]

            # Check if arrival is after due date
            if arrival_time > customer_info["due_date"]:
                return False

            # Update current time: arrival time + wait time if needed + service time
            service_start_time = max(arrival_time, customer_info["ready_time"])
            current_time = service_start_time + customer_info["service_time"]
            current_loc = curr_cid

        # Return to depot
        travel_time_back = self.instance.get_distance(current_loc, self.depot["id"])
        arrival_at_depot = current_time + travel_time_back
        if arrival_at_depot > self.depot["due_date"]:
            return False

        return True

    def compute_route_duration(self, customer_ids: List[int]) -> float:
        """Compute the total travel duration for a route."""
        if not customer_ids:
            return 0.0

        total_time = self.instance.get_distance(self.depot["id"], customer_ids[0])
        
        # Add travel time between consecutive customers
        for i in range(len(customer_ids) - 1):
            total_time += self.instance.get_distance(customer_ids[i], customer_ids[i+1])
        
        # Add travel time back to depot
        total_time += self.instance.get_distance(customer_ids[-1], self.depot["id"])
        
        return total_time


class LocalSearchOptimizer:
    def __init__(self, instance: VRPTWInstance, validator: RouteValidator):
        self.instance = instance
        self.validator = validator
        self.depot_id = instance.depot["id"]
        self.routes = []

    def initial_solution_greedy(self):
        """Create an initial solution using a greedy strategy."""
        unassigned = set(c["id"] for c in self.instance.customer_list)
        all_routes = []

        while unassigned:
            route = []
            current_load = 0
            current_time = self.instance.depot["ready_time"]
            current_location = self.depot_id

            remaining_customers = list(unassigned)

            while remaining_customers:
                next_customer = None
                best_score = float('inf')

                for cid in remaining_customers:
                    cust = self.instance.customers[cid]

                    # Check capacity constraint
                    if current_load + cust["demand"] > self.instance.vehicle_capacity:
                        continue

                    # Check time feasibility
                    travel_time = self.instance.get_distance(current_location, cid)
                    arrival_time = current_time + travel_time
                    
                    # Adjust arrival time to account for ready_time
                    actual_start_service = max(arrival_time, cust["ready_time"])
                    
                    # Compute what will happen after service: departure time
                    next_departure = actual_start_service + cust["service_time"]
                    
                    # Need to also check if we can return to depot in time
                    post_service_travel = self.instance.get_distance(cid, self.depot_id)
                    final_arrival = next_departure + post_service_travel
                    
                    if final_arrival <= self.instance.depot["due_date"]:
                        # Use the travel time as score heuristic (greedy closest first)
                        if travel_time < best_score:
                            best_score = travel_time
                            next_customer = cid

                if next_customer is not None:
                    # Add customer to route if valid
                    new_route = route + [next_customer]
                    if self.validator.validate_route(new_route):
                        route.append(next_customer)
                        unassigned.remove(next_customer)
                        
                        # Update current state
                        current_location = next_customer
                        customer_info = self.instance.customers[next_customer]
                        travel_time = self.instance.get_distance(
                            self.depot_id if len(route) == 1 else route[-2],
                            next_customer
                        )
                        arrival_time = current_time + travel_time
                        service_start_time = max(arrival_time, customer_info["ready_time"])
                        current_time = service_start_time + customer_info["service_time"]
                        current_load += customer_info["demand"]
                        
                        remaining_customers = [cid for cid in remaining_customers if cid != next_customer]
                    else:
                        break  # This should not happen with validation checks above
                else:
                    break  # No more valid moves

            all_routes.append(route)

            # Exit if too many routes or stuck in infinite loop
            if len(all_routes) > len(self.instance.customer_list):
                break

        return all_routes

    def two_opt_improvement(self, route: List[int]) -> List[int]:
        """Apply 2-opt improvement to a single route."""
        best_route = route[:]
        best_distance = self._compute_single_route_distance(route)
        improved = True

        while improved:
            improved = False
            n = len(best_route)

            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Reverse subsequence between i and j
                    new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]

                    if self.validator.validate_route(new_route):
                        new_distance = self._compute_single_route_distance(new_route)
                        if new_distance < best_distance:
                            best_route = new_route[:]
                            best_distance = new_distance
                            improved = True
        return best_route

    def relocate_customer(self, routes: List[List[int]]) -> List[List[int]]:
        """Move a single customer from one route to another."""
        current_routes = [r[:] for r in routes]
        n_routes = len(current_routes)

        for i in range(n_routes):
            for j, customer in enumerate(current_routes[i]):
                # Try moving customer to each position in other routes or within same route
                source_route = current_routes[i][:j] + current_routes[i][j+1:]  # Remove customer
                
                # Check if source route remains valid after removal
                if source_route and not self.validator.validate_route(source_route):
                    continue

                # Try inserting customer into different locations of other routes
                for k in range(n_routes):
                    if i == k:  # Same route case handled differently
                        continue
                    target_route = current_routes[k]

                    # Try all possible insertion positions
                    for pos in range(len(target_route) + 1):
                        new_target_route = target_route[:pos] + [customer] + target_route[pos:]
                        if self.validator.validate_route(new_target_route):
                            # Form new solution
                            new_routes = []
                            for idx in range(n_routes):
                                if idx == i:
                                    new_routes.append(source_route[:])
                                elif idx == k:
                                    new_routes.append(new_target_route[:])
                                else:
                                    new_routes.append(current_routes[idx][:])

                            # Check if it improves total distance
                            old_dist = sum(self._compute_single_route_distance(r) for r in current_routes)
                            new_dist = sum(self._compute_single_route_distance(r) for r in new_routes)
                            
                            if new_dist < old_dist:
                                return self.filter_empty_routes(new_routes)
        
        return self.filter_empty_routes(current_routes)

    def filter_empty_routes(self, routes: List[List[int]]) -> List[List[int]]:
        """Remove empty routes."""
        return [route for route in routes if route]

    def _compute_single_route_distance(self, customer_ids: List[int]) -> float:
        """Compute total distance of a single route."""
        if not customer_ids:
            return 0.0

        total_dist = self.instance.get_distance(self.depot_id, customer_ids[0])
        
        for i in range(len(customer_ids) - 1):
            total_dist += self.instance.get_distance(customer_ids[i], customer_ids[i+1])
        
        total_dist += self.instance.get_distance(customer_ids[-1], self.depot_id)
        
        return total_dist

    def optimize(self) -> List[List[int]]:
        """Run local search optimization process."""
        routes = self.initial_solution_greedy()
        current_best_distance = sum(self._compute_single_route_distance(r) for r in routes)
        
        # Iterative improvement
        prev_best = float('inf')
        iterations = 5  # Limit number of improvement cycles
        
        while abs(prev_best - current_best_distance) > 1e-6 and iterations > 0:
            prev_best = current_best_distance
            
            # Apply 2-opt improvement to each route
            improved_routes = []
            for route in routes:
                improved_route = self.two_opt_improvement(route)
                improved_routes.append(improved_route)
            
            routes = self.filter_empty_routes(improved_routes)
            
            # Try moving customers between routes
            routes = self.relocate_customer(routes)
            
            current_best_distance = sum(self._compute_single_route_distance(r) for r in routes)
            iterations -= 1

        return self.filter_empty_routes(routes)


def solve_instance(instance_data: Dict, timeout: int):
    instance = VRPTWInstance(instance_data)
    validator = RouteValidator(instance)
    optimizer = LocalSearchOptimizer(instance, validator)
    
    start_time = datetime.now()
    
    best_routes = optimizer.optimize()
    best_distance = sum(optimizer._compute_single_route_distance(r) for r in best_routes)

    # Convert back to required solution format
    result_routes = []
    for route in best_routes:
        if route:  # Only add non-empty routes
            path = [instance.depot["id"]] + route + [instance.depot["id"]]
            result_routes.append({
                "customer_ids": route,
                "path": path
            })

    return {
        "objective_value": best_distance,
        "routes": result_routes
    }


def main():
    parser = argparse.ArgumentParser(description="VRPTW Solver")
    parser.add_argument("--instance_path", required=True, help="Path to input instance JSON file")
    parser.add_argument("--solution_path", required=True, help="Path to output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Max runtime in seconds")
    parser.add_argument("--log_path", required=False, help="Path to logging file")

    args = parser.parse_args()

    # Initialize logger if provided
    logger = None
    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    try:
        # Solve the instance
        solution = solve_instance(instance_data, args.time_limit)
        
        # Log final best objective if logger exists
        if logger:
            logger.log(solution["objective_value"])

        # Write solution to file
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()