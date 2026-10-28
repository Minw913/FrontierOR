import argparse
import json
import math
import random
import copy
from solution_logger import SolutionLogger


def dist(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


class VRPSolver:
    def __init__(self, instance_data):
        self.num_customers = instance_data['num_customers']
        self.depot = instance_data['depot']
        self.vehicle_capacity = instance_data['vehicle_capacity']
        self.num_vehicles = instance_data['num_vehicles']
        self.width = instance_data.get('endogenous_time_window_width', 10.0)
        
        self.customers = instance_data['customers']
        self.scenarios = instance_data['scenarios']
        self.dist_matrix = instance_data['distance_matrix']
        
        # Pre-process customer data for faster access
        self.customer_locations = {c['id']: (c['x'], c['y']) for c in self.customers}
        self.customer_exo_windows = {c['id']: c['exogenous_time_window'] for c in self.customers}
        
        # Use customer-specific endogenous width if provided, otherwise default
        self.customer_endo_widths = {c['id']: c.get('endogenous_time_window_width', self.width) for c in self.customers}
        
        self.start_depot_id = 0
        self.end_depot_id = len(self.dist_matrix) - 1

    def calculate_serving_costs(self):
        """
        Calculate the cost of serving each customer in each scenario based on their demands.
        """
        serving_costs_per_scenario = {}
        for scenario in self.scenarios:
            scenario_id = scenario['scenario_id']
            demands = scenario['demands']
            
            # Calculate the serving cost of each customer (travel from depot to customer and back to depot)
            serving_costs = {}
            for idx, customer in enumerate(self.customers):
                c_id = customer['id']
                serving_cost = self.dist_matrix[self.start_depot_id][c_id] + \
                               self.dist_matrix[c_id][self.end_depot_id]
                serving_costs[c_id] = serving_cost
                
            serving_costs_per_scenario[scenario_id] = serving_costs
            
        return serving_costs_per_scenario

    def initial_solution_heuristic(self):
        """
        Generate an initial feasible solution using a greedy heuristic approach.
        """
        # Step 1: Set up endogenous time windows using a simple strategy
        
        # For each customer, place the endogenous window in the center of the exogenous window if possible
        customer_to_time_window = {}
        for customer in self.customers:
            c_id = customer['id']
            exo_start, exo_end = customer['exogenous_time_window']
            width = self.customer_endo_widths[c_id]
            
            center = (exo_start + exo_end) / 2.0
            start = max(exo_start, min(center - width/2.0, exo_end - width))
            
            customer_to_time_window[c_id] = [start, start + width]

        # Step 2: Construct routes for each scenario using the time windows above
        routes_by_scenario = {}
        total_expected_cost = 0.0
        
        for scenario in self.scenarios:
            scenario_id = scenario['scenario_id']
            demands = scenario['demands']
            
            # Create customer-demand pairs
            customer_demands = [(i, demands[i-1]) for i in range(1, self.num_customers + 1)]  # Adjust for customer IDs 1-based
            customer_demands.sort(key=lambda x: x[1], reverse=True)  # Sort by descending demand
            
            # Route construction
            unvisited = set([customer_id for customer_id, _ in customer_demands])
            current_routes = []
            
            while unvisited:
                route = [self.start_depot_id]
                current_load = 0
                current_time = 0  # Start at time 0 from depot
                
                # Try to assign customers to this route
                remaining_customers = sorted([(cid, demands[cid-1]) for cid in unvisited], 
                                           key=lambda x: demands[x[0]-1], reverse=True)
                
                i = 0
                while i < len(remaining_customers):
                    customer_id, demand = remaining_customers[i]
                    
                    # Check if we can add this customer to current route
                    last_loc = route[-1] if route else self.start_depot_id
                    
                    arrival_time = current_time + self.dist_matrix[last_loc][customer_id]
                    exo_start, exo_end = self.customer_exo_windows[customer_id]
                    en_duration = self.customer_endo_widths[customer_id]
                    en_start, en_end = customer_to_time_window[customer_id]
                    
                    # Can visit only if load allows, exo window allows, and en window allows,
                    # and we can serve the customer within its endogenous window
                    if current_load + demand <= self.vehicle_capacity and \
                       arrival_time >= exo_start and arrival_time <= exo_end - en_duration and \
                       arrival_time >= en_start and arrival_time <= en_end:
                        
                        route.append(customer_id)
                        current_load += demand
                        current_time = arrival_time
                        
                        unvisited.remove(customer_id)
                        i = 0  # Restart scan since order changed
                    else:
                        i += 1
                
                # Head back to depot
                if len(route) > 1:
                    last_customer_id = route[-1]
                    final_arrival_time = current_time + self.dist_matrix[last_customer_id][self.end_depot_id]
                    
                    route.append(self.end_depot_id)
                    
                    # Calculate total route cost
                    total_route_cost = 0.0
                    for i in range(len(route) - 1):
                        total_route_cost += self.dist_matrix[route[i]][route[i+1]]
                    
                    # Record arrival times for each customer
                    arrival_times = {}
                    curr_time = 0
                    for i, node in enumerate(route):
                        if i == 0:  # Starting at depot
                            arrival_times[str(node)] = curr_time
                        else:  # Moving to next node
                            prev_node = route[i-1]
                            travel_time = self.dist_matrix[prev_node][node]
                            curr_time += travel_time
                            
                            if i < len(route) - 1:  # Not the last depot
                                arrival_times[str(node)] = curr_time
                            else:  # Return to the last depot
                                arrival_times[str(node)] = curr_time
                    
                    current_routes.append({
                        'route': route,
                        'arrival_times': arrival_times,
                        'cost': total_route_cost
                    })
                        
                else:
                    break  # No progress made, break to avoid infinite loop
            
            routes_by_scenario[scenario_id] = current_routes
            scenario_prob = scenario['probability']
            for r in current_routes:
                total_expected_cost += scenario_prob * r['cost']
        
        return {
            'time_windows': customer_to_time_window,
            'routes': routes_by_scenario,
            'objective_value': total_expected_cost
        }

    def solve_with_improvement_heuristic(self, time_limit=60):
        best_solution = self.initial_solution_heuristic()
        logger = getattr(self, 'logger', None)
        if logger and best_solution:
            logger.log(best_solution['objective_value'])
            
        # Implement a basic improvement heuristic (local search style)
        current_solution = copy.deepcopy(best_solution)
        current_cost = best_solution['objective_value']
        
        # For improvement, we'll try swapping the positions of endogenous windows for different customers
        improved = True
        iteration_count = 0
        import time
        start_time = time.time()
        
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            customer_ids = list(current_solution['time_windows'].keys())
            
            # Try different adjustments to time windows
            for i in range(len(customer_ids)):
                original_time_window = current_solution['time_windows'][customer_ids[i]]
                
                # Try shifting the time window slightly
                for offset_diff in [-5, -2, -1, 1, 2, 5]:  # Different time offsets to try
                    new_solution = copy.deepcopy(current_solution)
                    
                    # Apply offset
                    orig_start, orig_end = original_time_window
                    new_start = max(self.customer_exo_windows[customer_ids[i]][0],
                                    min(orig_start + offset_diff, 
                                        self.customer_exo_windows[customer_ids[i]][1] - self.customer_endo_widths[customer_ids[i]]))
                    new_end = new_start + self.customer_endo_widths[customer_ids[i]]
                    
                    new_solution['time_windows'][customer_ids[i]] = [new_start, new_end]
                    
                    # Reroute this scenario with the new time windows
                    scenario_modified = False
                    for s_id in new_solution['routes']:
                        if self.reroute_with_new_time_windows(new_solution, s_id):
                            scenario_modified = True
                            
                    if scenario_modified:
                        rerouted_cost = self.compute_total_cost(new_solution)
                        
                        if rerouted_cost < current_cost:
                            current_solution = new_solution
                            current_cost = rerouted_cost
                            if logger:
                                logger.log(current_cost)
                            improved = True
                            best_solution = new_solution
                            break  # Break because we have improvement, restart from beginning
                
                if improved:
                    break  # Restart outer loop after improvement
            
            iteration_count += 1
            if iteration_count > 50:  # Limit iterations to avoid running too long
                break
                
        return best_solution

    def reroute_with_new_time_windows(self, solution, scenario_id):
        """
        Reconstruct routes for the given scenario considering the new time window assignment.
        """
        scenario_data = next(s for s in self.scenarios if s['scenario_id'] == scenario_id)
        demands = scenario_data['demands']
        
        unvisited = set(range(1, self.num_customers + 1))  # Customer IDs are 1-based
        new_routes = []
        
        while unvisited:
            route = [self.start_depot_id]
            current_load = 0
            current_time = 0
            route_change = False
            
            # Get unvisited customers respecting demands
            cust_demands = [(cid, demands[cid-1]) for cid in unvisited]
            cust_sorted = sorted(cust_demands, key=lambda x: x[1], reverse=True)
            
            i = 0
            attempts = 0
            max_attempts = len(cust_sorted)
            
            while i < len(cust_sorted) and attempts < max_attempts:
                customer_id, demand = cust_sorted[i]
                
                # Check constraints: capacity, time windows
                last_loc = route[-1]
                arrival_time_at_customer = current_time + self.dist_matrix[last_loc][customer_id]
                
                exo_start, exo_end = self.customer_exo_windows[customer_id]
                en_start, en_end = solution['time_windows'][customer_id]
                
                # Can visit only if within limits
                fits_time_constraint = (
                    current_time + self.dist_matrix[last_loc][customer_id] >= en_start and
                    current_time + self.dist_matrix[last_loc][customer_id] <= en_end and
                    current_load + demand <= self.vehicle_capacity and
                    arrival_time_at_customer >= self.customer_exo_windows[customer_id][0] and
                    arrival_time_at_customer + 1 <= self.customer_exo_windows[customer_id][1]
                )
                
                if fits_time_constraint:
                    route.append(customer_id)
                    current_load += demand
                    current_time = current_time + self.dist_matrix[last_loc][customer_id]
                    unvisited.remove(customer_id)
                    
                    cust_sorted = [(cid, demands[cid-1]) for cid in unvisited]
                    cust_sorted = sorted(cust_sorted, key=lambda x: x[1], reverse=True)  # Reorder
                    
                    i = 0
                    attempts = 0  # Reset attempts counter
                    route_change = True
                else:
                    i += 1
                    attempts += 1
            
            # Go back to depot
            if len(route) > 1:
                last_customer = route[-1]
                depot_travel_cost = self.dist_matrix[last_customer][self.end_depot_id]
                final_time = current_time + depot_travel_cost
                
                route.append(self.end_depot_id)
                
                # Calculate route cost and arrival times
                total_route_cost = 0.0
                for j in range(len(route) - 1):
                    total_route_cost += self.dist_matrix[route[j]][route[j+1]]
                
                arrival_times_map = {}
                temp_time = 0
                for k in range(len(route)):
                    if k == 0:
                        arrival_times_map[str(route[k])] = temp_time
                    else:
                        temp_time += self.dist_matrix[route[k-1]][route[k]]
                        arrival_times_map[str(route[k])] = temp_time
                
                new_routes.append({
                    'route': route,
                    'arrival_times': arrival_times_map,
                    'cost': total_route_cost
                })
            elif len(unvisited) > 0:  # Could not visit anyone, there's an issue
                print(f"Warning: Could not visit some customers in scenario {scenario_id}")
                return False
                
        # Update solution with new routes
        solution['routes'][scenario_id] = new_routes
        return route_change

    def compute_total_cost(self, solution):
        """
        Recompute the objective value of the solution including probabilities.
        """
        total_cost = 0.0
        
        for scenario in self.scenarios:
            scenario_id = scenario['scenario_id']
            probability = scenario['probability']
            routes = solution['routes'][scenario_id]
            
            scenario_cost = 0.0
            for route_obj in routes:
                scenario_cost += route_obj['cost']
                
            total_cost += probability * scenario_cost
            
        return total_cost


def main():
    parser = argparse.ArgumentParser(description='Solve Stochastic VRP with time windows.')
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path to output the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, required=False,
                        help='Path to log intermediate solutions')
    
    args = parser.parse_args()

    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Initialize solution logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Create solver instance
    solver = VRPSolver(instance_data)
    # Assign logger to solver to allow logging within methods
    solver.logger = logger

    # Solve the problem
    solution = solver.solve_with_improvement_heuristic(time_limit=args.time_limit)
    
    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)
    

if __name__ == '__main__':
    main()