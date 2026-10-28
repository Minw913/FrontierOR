import argparse
import json
import math
import random
import time
from itertools import combinations
from typing import List, Dict, Tuple, Optional, Any

from solution_logger import SolutionLogger

class PackingAlgorithm:
    def __init__(self, H: int, W: int):
        self.H = H
        self.W = W
        self.rectangles = []  # list of maximal rectangles (x, y, w, h)
        self.initialize()

    def initialize(self):
        # Start with the entire packing area as a maximal rectangle
        self.rectangles = [(0, 0, self.W, self.H)]

    def pack_item(self, width: int, height: int) -> Optional[Tuple[int, int]]:
        best_score = float('inf')
        best_rect = None
        best_index = None

        for idx, rect in enumerate(self.rectangles):
            x, y, w, h = rect
            if w >= width and h >= height:
                # Evaluate this rectangle: we use the top-left corner
                score = y  # prioritize lower y (closer to the front of the vehicle)
                if score < best_score:
                    best_score = score
                    best_rect = (x, y)
                    best_index = idx

        if best_rect is None:
            return None

        x, y = best_rect
        # Remove the chosen maximal rectangle and split the remaining space
        del self.rectangles[best_index]

        # Split the remaining space into two new rectangles (horizontal and vertical)
        if w > width:
            self.rectangles.append((x + width, y, w - width, height))
        if h > height:
            self.rectangles.append((x, y + height, width, h - height))
        if w > width and h > height:
            self.rectangles.append((x + width, y + height, w - width, h - height))

        # Merge overlapping or adjacent rectangles? (optional, for efficiency)

        return (x, y)

    def reset(self):
        self.initialize()

def check_packing_feasibility(items: List[Tuple[int, int]], H: int, W: int) -> bool:
    """Check if a list of items (each as (width, height)) can be packed into the vehicle using the maximal rectangles algorithm."""
    packer = PackingAlgorithm(H, W)
    # Sort items by area descending to improve packing
    sorted_items = sorted(items, key=lambda item: item[0]*item[1], reverse=True)
    for w, h in sorted_items:
        pos = packer.pack_item(w, h)
        if pos is None:
            return False
    return True

class Customer:
    def __init__(self, id: int, is_stochastic: bool, items: List[Dict]):
        self.id = id
        self.is_stochastic = is_stochastic
        self.items = items
        self.expected_weight = 0.0
        self.expected_area = 0.0
        self.compute_expected_values()

    def compute_expected_values(self):
        total_weight = 0.0
        total_area = 0.0
        for item in self.items:
            if item['is_stochastic']:
                for realization in item['realizations']:
                    prob = realization['probability']
                    h = realization['height']
                    w = realization['width']
                    weight = realization['weight']
                    total_weight += prob * weight
                    total_area += prob * (h * w)
            else:
                # deterministic item: there should be exactly one realization
                realization = item['realizations'][0]
                h = realization['height']
                w = realization['width']
                weight = realization['weight']
                total_weight += weight
                total_area += (h * w)
        self.expected_weight = total_weight
        self.expected_area = total_area

def read_instance(instance_path: str) -> Dict:
    with open(instance_path, 'r') as f:
        data = json.load(f)
    return data

def parse_instance(data: Dict) -> Tuple:
    depot = data['depot']
    vehicle_params = data['vehicle']
    H = vehicle_params['H']
    W = vehicle_params['W']
    Q = vehicle_params['Q']
    K = vehicle_params['K']
    recourse_cost_cf = data['parameters']['recourse_cost_cf']

    distance_matrix = data['distance_matrix']
    customers_data = data['customers']

    # Create list of Customer objects
    customers = []
    for cust_data in customers_data:
        customer = Customer(
            id=cust_data['id'],
            is_stochastic=cust_data['is_stochastic'],
            items=cust_data['items']
        )
        customers.append(customer)

    return depot, H, W, Q, K, recourse_cost_cf, distance_matrix, customers

class Route:
    def __init__(self, customers: List[Customer], indices: List[int], distance_matrix: List[List[int]]):
        self.customers = customers
        self.indices = indices  # customer IDs in the order of the route (excluding depot)
        self.distance_matrix = distance_matrix
        self.cost = self.compute_route_cost()
        self.feasible = None
        self.packing_infeasible_scenarios = 0.0  # probability mass where packing is infeasible

    def compute_route_cost(self) -> int:
        if not self.indices:
            return 0
        total_cost = 0
        # start from depot (index 0) to first customer
        total_cost += self.distance_matrix[0][self.indices[0]]
        for i in range(len(self.indices) - 1):
            total_cost += self.distance_matrix[self.indices[i]][self.indices[i+1]]
        total_cost += self.distance_matrix[self.indices[-1]][0]
        return total_cost

    def get_customer_objects(self) -> List[Customer]:
        return [cust for cust in self.customers if cust.id in self.indices]

    def expected_penalty_cost(self, recourse_cost_cf: int) -> float:
        # Expected number of unserved customers = (number of customers in route) * probability_of_infeasibility
        return len(self.indices) * self.packing_infeasible_scenarios * recourse_cost_cf

def check_route_packing_feasibility(route: Route, H: int, W: int, recourse_cost_cf: int, deterministic: bool = False, num_samples: int = 10) -> None:
    """Check packing feasibility for a route. For deterministic, check once. For stochastic, sample scenarios.
    Sets the route's packing_infeasible_scenarios to the probability of infeasibility."""
    customers = route.get_customer_objects()
    if deterministic:
        # For deterministic route, we can check directly
        items = []
        for customer in customers:
            for item in customer.items:
                # deterministic item: only one realization
                realization = item['realizations'][0]
                items.append((realization['width'], realization['height']))
        feasible = check_packing_feasibility(items, H, W)
        route.packing_infeasible_scenarios = 0.0 if feasible else 1.0
    else:
        # Stochastic: we need to sample scenarios
        # First, we need to generate a list of all stochastic items and their possible realizations
        all_items = []
        for customer in customers:
            for item in customer.items:
                if item['is_stochastic']:
                    realizations = item['realizations']
                    all_items.append(realizations)
                else:
                    # deterministic item: only one realization
                    all_items.append([item['realizations'][0]])

        # If there are no stochastic items, then we are effectively deterministic
        if not any(len(item_list) > 1 for item_list in all_items):
            items = []
            for realizations in all_items:
                r = realizations[0]
                items.append((r['width'], r['height']))
            feasible = check_packing_feasibility(items, H, W)
            route.packing_infeasible_scenarios = 0.0 if feasible else 1.0
            return

        # We will sample a number of scenarios
        infeasible_count = 0
        for sample in range(num_samples):
            items = []
            for realizations in all_items:
                # Sample one realization per item according to their probabilities
                r = random.choices(realizations, weights=[r['probability'] for r in realizations], k=1)[0]
                items.append((r['width'], r['height']))
            feasible = check_packing_feasibility(items, H, W)
            if not feasible:
                infeasible_count += 1

        # Estimate the probability of infeasibility
        route.packing_infeasible_scenarios = infeasible_count / num_samples

def savings_heuristic(depot_idx: int, customers: List[Customer], distance_matrix: List[List[int]], K: int, Q: int, H: int, W: int) -> List[List[int]]:
    n = len(customers)
    # Precompute savings for each pair of customers
    savings = []
    for i in range(n):
        for j in range(i+1, n):
            c_i = customers[i].id
            c_j = customers[j].id
            s = distance_matrix[depot_idx][c_i] + distance_matrix[depot_idx][c_j] - distance_matrix[c_i][c_j]
            savings.append((s, i, j))

    savings.sort(key=lambda x: x[0], reverse=True)

    # Initialize routes: each customer as a separate route
    routes = [[i] for i in range(n)]
    total_weight_per_route = [customers[i].expected_weight for i in range(n)]
    total_area_per_route = [customers[i].expected_area for i in range(n)]

    # Merge routes
    for s, i, j in savings:
        route_i = None
        route_j = None
        for r in routes:
            if i in r:
                route_i = r
            if j in r:
                route_j = r

        if route_i is None or route_j is None or route_i == route_j:
            continue

        # Check if both routes are not the same
        # We consider merging: either attach j at the end of i or i at the end of j?
        # But we don't know the order. We consider two possibilities: route_i + route_j or route_j + route_i?
        # Alternatively, we can only merge if one of the endpoints is adjacent to the depot.

        # Simple: if the last element of route_i is i and the first element of route_j is j, then we can merge (i, j)
        # But we don't track endpoints. Instead, we only merge if both are endpoints adjacent to the depot.

        # Instead, we allow merging if the combined capacity is within the vehicle's limits.
        combined_weight = total_weight_per_route[routes.index(route_i)] + total_weight_per_route[routes.index(route_j)]
        combined_area = total_area_per_route[routes.index(route_i)] + total_area_per_route[routes.index(route_j)]

        if combined_weight > Q or combined_area > (H * W):
            continue

        # Now merge the two routes: we simply concatenate them arbitrarily? We can choose the order that minimizes the added distance?
        # We have two options: 
        #   Option1: [ ... i] and [j ...] -> ... i, j ...
        #   Option2: [ ... i] and [ ... j] -> ... i, reverse of j's route ... (but we don't track order)

        # Since we don't track the order of the route, we assume the current routes are unordered. We can just merge as two sets.
        # Actually, for the savings method, we are building chains. We need to track the endpoints.

        # We'll simplify: merge the two lists arbitrarily. We'll always add route_j at the end of route_i.
        new_route = route_i + route_j
        # Update the routes
        routes.remove(route_i)
        routes.remove(route_j)
        routes.append(new_route)
        total_weight_per_route.remove(combined_weight - combined_weight)  # remove the old ones? 
        total_weight_per_route.remove(combined_weight - combined_weight)
        total_weight_per_route.append(combined_weight)
        total_area_per_route.remove(combined_area - combined_area)
        total_area_per_route.remove(combined_area - combined_area)
        total_area_per_route.append(combined_area)

        # If we have K routes, break
        if len(routes) <= K:
            break

    # Now we have a set of routes (each is a list of indices in the customers list). We need to convert to customer IDs and form the route as a sequence.
    # Also, we haven't determined the order within a route. We can apply a TSP on each route.
    # Instead, we'll use the order in the list for now and then apply 2-opt per route.

    # Convert to customer IDs
    final_routes = []
    for route in routes:
        cust_ids = [customers[i].id for i in route]
        # We will form the order by starting at the customer closest to the depot? Or we can use the order in the list for now.
        final_routes.append(cust_ids)

    # If we have more than K routes, we have to merge until we have K
    while len(final_routes) > K:
        # Merge the two smallest routes? Or the two closest?
        # Find the two routes that are closest by their endpoints?
        min_distance = float('inf')
        merge_i, merge_j = -1, -1
        for i in range(len(final_routes)):
            for j in range(i+1, len(final_routes)):
                # Distance between last customer of i and first customer of j? But we don't have endpoints.
                # Instead, compute centroid of each route?
                # Alternatively, compute the distance between any customer in i and any in j and take the minimum.
                for cust_i in final_routes[i]:
                    for cust_j in final_routes[j]:
                        d = distance_matrix[cust_i][cust_j]
                        if d < min_distance:
                            min_distance = d
                            merge_i = i
                            merge_j = j

        # Merge route_j into route_i
        if merge_i != -1 and merge_j != -1:
            route_i = final_routes[merge_i]
            route_j = final_routes[merge_j]
            combined_route = route_i + route_j
            del final_routes[merge_j]
            del final_routes[merge_i]
            final_routes.append(combined_route)
        else:
            break

    return final_routes

def two_opt(route: List[int], dist_matrix: List[List[int]]) -> Tuple[List[int], int]:
    best = route
    improved = True
    best_cost = compute_route_cost(route, dist_matrix)
    n = len(route)
    while improved:
        improved = False
        for i in range(1, n-1):
            for j in range(i+1, n):
                if j - i == 1: continue
                new_route = route[:i] + route[i:j][::-1] + route[j:]
                new_cost = compute_route_cost(new_route, dist_matrix)
                if new_cost < best_cost:
                    best = new_route
                    best_cost = new_cost
                    improved = True
        route = best
    return best, best_cost

def compute_route_cost(route: List[int], dist_matrix: List[List[int]]) -> int:
    if not route:
        return 0
    total_cost = dist_matrix[0][route[0]]
    for i in range(len(route)-1):
        total_cost += dist_matrix[route[i]][route[i+1]]
    total_cost += dist_matrix[route[-1]][0]
    return total_cost

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    # Read instance
    data = read_instance(args.instance_path)
    depot, H, W, Q, K, recourse_cost_cf, distance_matrix, customers = parse_instance(data)

    # Initialize the solution logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Build mapping from customer id to Customer object
    cust_id_to_obj = {cust.id: cust for cust in customers}
    all_customer_ids = [cust.id for cust in customers]

    # Step 1: Build initial solution using savings heuristic
    routes = savings_heuristic(depot['id'], customers, distance_matrix, K, Q, H, W)

    # Apply 2-opt to each route to improve the routing cost
    improved_routes = []
    total_routing_cost = 0
    for route in routes:
        if len(route) > 1:
            new_route, cost = two_opt(route, distance_matrix)
            improved_routes.append(new_route)
            total_routing_cost += cost
        else:
            improved_routes.append(route)
            total_routing_cost += compute_route_cost(route, distance_matrix)
    routes = improved_routes

    # Step 2: Check packing feasibility for each route and compute expected penalty
    total_penalty = 0.0
    for route_cust_ids in routes:
        cust_objs = [cust_id_to_obj[cid] for cid in route_cust_ids]
        route_obj = Route(customers, route_cust_ids, distance_matrix)
        # Check if the route has any stochastic customers
        has_stochastic = any(cust.is_stochastic for cust in cust_objs)
        check_route_packing_feasibility(route_obj, H, W, recourse_cost_cf, deterministic=not has_stochastic)
        total_penalty += route_obj.expected_penalty_cost(recourse_cost_cf)

    total_cost = total_routing_cost + total_penalty

    # Log initial solution
    if logger:
        logger.log(total_cost)

    # Step 3: Local search to improve the solution
    # We will try to relocate a customer from one route to another, and exchange two customers between routes.
    # We must respect the capacity constraints (expected weight and area) and check packing feasibility after each move.

    # We'll do a simple first-improvement local search until time runs out.

    best_routes = routes
    best_total_cost = total_cost

    def compute_total_penalty(routes: List[List[int]]) -> float:
        penalty = 0.0
        for route_cust_ids in routes:
            cust_objs = [cust_id_to_obj[cid] for cid in route_cust_ids]
            route_obj = Route(customers, route_cust_ids, distance_matrix)
            has_stochastic = any(cust.is_stochastic for cust in cust_objs)
            check_route_packing_feasibility(route_obj, H, W, recourse_cost_cf, deterministic=not has_stochastic)
            penalty += route_obj.expected_penalty_cost(recourse_cost_cf)
        return penalty

    def compute_total_routing_cost(routes: List[List[int]]) -> int:
        total = 0
        for route in routes:
            total += compute_route_cost(route, distance_matrix)
        return total

    def is_capacity_feasible(route: List[int]) -> bool:
        total_weight = 0.0
        total_area = 0.0
        for cid in route:
            cust = cust_id_to_obj[cid]
            total_weight += cust.expected_weight
            total_area += cust.expected_area
        return total_weight <= Q and total_area <= (H * W)

    def get_route_expected_weight(route: List[int]) -> float:
        return sum(cust_id_to_obj[cid].expected_weight for cid in route)

    def get_route_expected_area(route: List[int]) -> float:
        return sum(cust_id_to_obj[cid].expected_area for cid in route)

    # Local search operations
    improved = True
    while improved and time.time() - start_time < args.time_limit:
        improved = False
        # Try relocate: move a customer from one route to another
        for i in range(len(best_routes)):
            for j in range(len(best_routes)):
                if i == j:
                    continue
                for idx, cust_id in enumerate(best_routes[i]):
                    # Try remove cust_id from route i and insert into route j at every possible position
                    new_route_i = best_routes[i][:idx] + best_routes[i][idx+1:]
                    # Check if route i without this customer is still feasible? (it will be, but we can check capacity if needed)
                    # Then try inserting cust_id into route j at every position
                    for pos in range(len(best_routes[j])+1):
                        new_route_j = best_routes[j][:pos] + [cust_id] + best_routes[j][pos:]
                        # Check capacity for new_route_j
                        if not is_capacity_feasible(new_route_j):
                            continue

                        # Also, check capacity for new_route_i? (it might be empty, then skip)
                        if new_route_i and not is_capacity_feasible(new_route_i):
                            continue

                        # Create new set of routes
                        new_routes = best_routes.copy()
                        new_routes[i] = new_route_i
                        new_routes[j] = new_route_j
                        # Remove route i if it becomes empty
                        if not new_route_i:
                            del new_routes[i]

                        # Check if we have exactly K routes? But we might have less if we deleted an empty route.
                        if len(new_routes) != K:
                            # We must have exactly K routes. We can split a route if we have less, but that's complex.
                            # Alternatively, if we have less than K, we can leave one route empty? But we must use K vehicles.
                            # Actually the problem requires all K vehicles must be used. So we cannot have fewer than K routes.
                            # Therefore, we must not delete a route if it becomes empty? Instead, we should not move the last customer in a route?
                            # So we skip if it would empty a route and reduce the number of routes.
                            continue

                        # Compute new routing cost
                        new_routing_cost = compute_total_routing_cost(new_routes)
                        # Compute new penalty: we only need to check the two modified routes (i and j)
                        # But note: if we removed a route, then we have to check the new routes. We'll recompute all for simplicity? 
                        # Alternatively, we can check only the two changed routes and then update the total penalty.
                        new_penalty = compute_total_penalty(new_routes)
                        new_total_cost = new_routing_cost + new_penalty

                        if new_total_cost < best_total_cost:
                            best_routes = new_routes
                            best_total_cost = new_total_cost
                            improved = True
                            if logger:
                                logger.log(best_total_cost)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        # If not improved by relocate, try exchange
        if not improved:
            for i in range(len(best_routes)):
                for j in range(i+1, len(best_routes)):
                    for idx_i, cust_i in enumerate(best_routes[i]):
                        for idx_j, cust_j in enumerate(best_routes[j]):
                            # Exchange cust_i (from route i) and cust_j (from route j)
                            new_route_i = best_routes[i][:idx_i] + [cust_j] + best_routes[i][idx_i+1:]
                            new_route_j = best_routes[j][:idx_j] + [cust_i] + best_routes[j][idx_j+1:]

                            # Check capacity for both routes
                            if not is_capacity_feasible(new_route_i) or not is_capacity_feasible(new_route_j):
                                continue

                            new_routes = best_routes.copy()
                            new_routes[i] = new_route_i
                            new_routes[j] = new_route_j

                            new_routing_cost = compute_total_routing_cost(new_routes)
                            new_penalty = compute_total_penalty(new_routes)
                            new_total_cost = new_routing_cost + new_penalty

                            if new_total_cost < best_total_cost:
                                best_routes = new_routes
                                best_total_cost = new_total_cost
                                improved = True
                                if logger:
                                    logger.log(best_total_cost)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

    # Output the solution
    solution = {
        "objective_value": best_total_cost,
        "routes": best_routes
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()