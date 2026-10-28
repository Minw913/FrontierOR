import argparse
import json
import time
import sys
from solution_logger import SolutionLogger

def compute_route_cost(customer_list, distance_matrix, depot_id):
    if not customer_list:
        return 0
    total_cost = distance_matrix[depot_id][customer_list[0]['id']]
    for i in range(len(customer_list) - 1):
        from_id = customer_list[i]['id']
        to_id = customer_list[i+1]['id']
        total_cost += distance_matrix[from_id][to_id]
    total_cost += distance_matrix[customer_list[-1]['id']][depot_id]
    return total_cost

class RobustRoute:
    def __init__(self, route_id, Gamma, capacity):
        self.id = route_id
        self.Gamma = Gamma
        self.capacity = capacity
        self.customers = []   # list of customer dicts
    
    def add_customer(self, customer):
        self.customers.append(customer)
    
    def worst_case_demand(self):
        total_nominal = sum(c['nominal_demand'] for c in self.customers)
        deviations = [c['demand_deviation'] for c in self.customers]
        sorted_dev = sorted(deviations, reverse=True)
        k = min(len(deviations), self.Gamma)
        return total_nominal + sum(sorted_dev[:k])
    
    def is_feasible_after_adding(self, customer):
        temp_customers = self.customers + [customer]
        total_nominal = sum(c['nominal_demand'] for c in temp_customers)
        deviations = [c['demand_deviation'] for c in temp_customers]
        sorted_dev = sorted(deviations, reverse=True)
        k = min(len(deviations), self.Gamma)
        worst_case = total_nominal + sum(sorted_dev[:k])
        return worst_case <= self.capacity

def insertion_heuristic(instance, depot_id, distance_matrix):
    num_vehicles = instance['num_vehicles']
    capacity = instance['vehicle_capacity']
    Gamma = instance['uncertainty_set']['Gamma']
    customers = instance['customers']
    
    routes = [RobustRoute(i, Gamma, capacity) for i in range(num_vehicles)]
    
    customers_sorted = sorted(customers, key=lambda c: c['nominal_demand'] + c['demand_deviation'], reverse=True)
    
    for customer in customers_sorted:
        best_increase = float('inf')
        best_route = None
        best_position = None
        
        feasible_found = False
        for route in routes:
            if not route.is_feasible_after_adding(customer):
                continue
            feasible_found = True
            current_customers = route.customers
            current_route_cost = 0.0
            if current_customers:
                current_route_cost = compute_route_cost(current_customers, distance_matrix, depot_id)
            
            for pos in range(0, len(current_customers) + 1):
                new_customers = current_customers[:] 
                new_customers.insert(pos, customer)
                new_route_cost = compute_route_cost(new_customers, distance_matrix, depot_id)
                increase = new_route_cost - current_route_cost
                
                if increase < best_increase:
                    best_increase = increase
                    best_route = route
                    best_position = pos
        
        if not feasible_found:
            for route in routes:
                current_customers = route.customers
                current_route_cost = 0.0
                if current_customers:
                    current_route_cost = compute_route_cost(current_customers, distance_matrix, depot_id)
                
                for pos in range(0, len(current_customers) + 1):
                    new_customers = current_customers[:] 
                    new_customers.insert(pos, customer)
                    new_route_cost = compute_route_cost(new_customers, distance_matrix, depot_id)
                    increase = new_route_cost - current_route_cost
                    
                    if increase < best_increase:
                        best_increase = increase
                        best_route = route
                        best_position = pos
        
        if best_route is not None:
            best_route.customers.insert(best_position, customer)
    
    return routes

def local_search_2opt(routes, depot_id, distance_matrix, time_limit, start_time):
    for route in routes:
        if time.time() - start_time > time_limit:
            return
        current_customers = route.customers
        if len(current_customers) <= 1:
            continue
        current_route_cost = compute_route_cost(current_customers, distance_matrix, depot_id)
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > time_limit:
                break
            for i in range(0, len(current_customers)-1):
                if improved:
                    break
                for j in range(i+2, len(current_customers)):
                    new_customers = current_customers[:]
                    new_customers[i+1:j+1] = reversed(new_customers[i+1:j+1])
                    new_route_cost = compute_route_cost(new_customers, distance_matrix, depot_id)
                    if new_route_cost < current_route_cost:
                        current_customers = new_customers
                        current_route_cost = new_route_cost
                        improved = True
                        break
            if improved:
                continue
        route.customers = current_customers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    start_time = time.time()
    time_limit = args.time_limit
    
    depot_id = instance['depot']['id']
    distance_matrix = instance['distance_matrix']
    
    routes = insertion_heuristic(instance, depot_id, distance_matrix)
    
    if time.time() - start_time < time_limit:
        local_search_2opt(routes, depot_id, distance_matrix, time_limit, start_time)
    
    total_cost = 0.0
    for route in routes:
        route_cost = compute_route_cost(route.customers, distance_matrix, depot_id)
        total_cost += route_cost
    total_cost = float(total_cost)
    
    routes_output = []
    for route in routes:
        customer_ids = [c['id'] for c in route.customers]
        route_cost = compute_route_cost(route.customers, distance_matrix, depot_id)
        worst_case_demand = route.worst_case_demand()
        routes_output.append({
            "customers": customer_ids,
            "cost": int(route_cost),
            "worst_case_demand": int(worst_case_demand)
        })
    
    if logger:
        logger.log(total_cost)
    
    solution = {
        "objective_value": total_cost,
        "routes": routes_output
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()