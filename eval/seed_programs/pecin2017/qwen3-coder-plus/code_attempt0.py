import argparse
import json
import math
import random
import time
from typing import List, Dict, Tuple
from solution_logger import SolutionLogger


def calculate_distance(p1: Dict, p2: Dict) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1['x'] - p2['x']) ** 2 + (p1['y'] - p2['y']) ** 2)


class VRPTW:
    def __init__(self, instance):
        self.problem_params = instance['problem_parameters']
        self.nodes = {node['id']: node for node in instance['nodes']}
        
        # Extract important information
        self.n_nodes = self.problem_params['n_nodes']
        self.n_customers = self.problem_params['n_customers']
        self.vehicle_capacity = self.problem_params['vehicle_capacity']
        self.planning_horizon = self.problem_params['planning_horizon']
        
        # Identify depot nodes - the source and sink might have same ID or different
        self.depot_source_id = None
        self.depot_sink_id = None
        
        depot_nodes = []
        for node in instance['nodes']:
            if node['type'] == 'depot_source':
                self.depot_source_id = node['id']
                depot_nodes.append(node)
            elif node['type'] == 'depot_sink':
                self.depot_sink_id = node['id']
                depot_nodes.append(node)
        
        # Make sure we assign proper depot IDs
        if len(depot_nodes) >= 1:
            self.depot_source_id = depot_nodes[0]['id']
            # If only one depot is specified, make them the same
            if len(depot_nodes) == 1:
                self.depot_sink_id = self.depot_nodes[0]['id']
        
        # Extract customer IDs
        self.customer_ids = [node['id'] for node in instance['nodes'] if node['type'] == 'customer']
        
        # Calculate travel times and costs - Euclidean distance based on problem description
        self.travel_times = {}
        self.travel_costs = {}
        
        for i in self.nodes:
            for j in self.nodes:
                dist = calculate_distance(self.nodes[i], self.nodes[j])
                self.travel_times[i, j] = dist  # Time equals distance
                self.travel_costs[i, j] = dist  # Cost equals distance
                
        # Precompute admissible arcs
        self.admissible_arcs = self._compute_admissible_arcs()
    
    def _is_valid_connection(self, from_node_id: int, to_node_id: int) -> bool:
        """Check if direct connection is allowed between two nodes."""
        from_node = self.nodes[from_node_id]
        to_node = self.nodes[to_node_id]
        
        # Capacity check
        if from_node['demand'] + to_node['demand'] > self.vehicle_capacity:
            return False
            
        # Time window check: a vehicle starting at from_node at earliest possible time
        arrival_time = from_node['time_window_open'] + from_node['service_time'] + self.travel_times[from_node_id, to_node_id]
        if arrival_time > to_node['time_window_close']:
            return False
            
        return True
    
    def _compute_admissible_arcs(self):
        """Precompute which arcs are allowed based on capacity and time constraints."""
        arcs = set()
        for i in self.nodes:
            for j in self.nodes:
                if i != j and self._is_valid_connection(i, j):
                    arcs.add((i, j))
        return arcs


def check_route_feasibility(vrptw: VRPTW, route: List[int]) -> Tuple[bool, float]:
    """Check if a route is feasible and return its cost if feasible."""
    if not route:
        return True, 0.0
    
    total_demand = 0
    current_time = 0.0
    total_cost = 0.0
    
    # Start from depot
    prev_node_id = vrptw.depot_source_id
    prev_node = vrptw.nodes[prev_node_id]
    current_time = 0.0  # Start at time 0
    
    for node_id in route:
        node = vrptw.nodes[node_id]
        
        # Check capacity
        total_demand += node['demand']
        if total_demand > vrptw.vehicle_capacity:
            return False, float('inf')
        
        # Travel to node
        travel_time = vrptw.travel_times[prev_node_id, node_id]
        arrival_time = current_time + travel_time
        
        # Check if service can start within time window
        service_start_time = max(arrival_time, node['time_window_open'])
        if service_start_time > node['time_window_close']:
            return False, float('inf')
        
        # Update time: start service + service duration
        current_time = service_start_time + node['service_time']
        total_cost += vrptw.travel_costs[prev_node_id, node_id]
        prev_node_id = node_id
    
    # Return to depot
    travel_time_back = vrptw.travel_times[prev_node_id, vrptw.depot_sink_id]
    return_arrival = current_time + travel_time_back
    
    if return_arrival > vrptw.planning_horizon:
        return False, float('inf')
    
    total_cost += vrptw.travel_costs[prev_node_id, vrptw.depot_sink_id]
    
    return True, total_cost


class VRPTWSolver:
    def __init__(self, vrptw: VRPTW, time_limit: int, logger=None):
        self.vrptw = vrptw
        self.time_limit = time_limit
        self.logger = logger
        self.best_solution = None
        self.best_objective = float('inf')
        self.start_time = time.time()
    
    def solve(self):
        """Main solve method - uses heuristics to find solution."""
        print(f"Solving VRPTW with {len(self.vrptw.customer_ids)} customers and time limit {self.time_limit}")
        
        # Create initial solution using greedy heuristic
        print("Creating initial greedy solution...")
        greedy_solution = self.greedy_savings_init()
        if greedy_solution and self.evaluate_solution(greedy_solution)[0] < self.best_objective:
            self.update_best_solution(greedy_solution)
        
        # Run local search improvments
        print("Running local search improvements...")
        improved_solution = self.local_search_improve(self.best_solution)
        if improved_solution and self.evaluate_solution(improved_solution)[0] < self.best_objective:
            self.update_best_solution(improved_solution)
            
        return self.best_solution
    
    def greedy_savings_init(self):
        """Create initial solution using Clarke-Wright Savings heuristic."""
        n_customers = len(self.vrptw.customer_ids)
        customer_list = list(self.vrptw.customer_ids)
        
        # Calculate savings: c_0i + c_j0 - c_ij for every pair of customers (i,j)
        savings = []
        depot_id = self.vrptw.depot_source_id
        for i_idx in range(len(customer_list)):
            i = customer_list[i_idx]
            for j_idx in range(i_idx+1, len(customer_list)):
                j = customer_list[j_idx]
                
                saving = (self.vrptw.travel_costs[depot_id, i] + 
                         self.vrptw.travel_costs[j, depot_id] - 
                         self.vrptw.travel_costs[i, j])
                         
                if (i, j) in self.vrptw.admissible_arcs:
                    savings.append((saving, i, j))
        
        # Sort savings in descending order
        savings.sort(reverse=True)
        
        # Each customer initially in their own route
        routes = [[cust] for cust in customer_list]
        customer_to_route = {cust: idx for idx, cust in enumerate(customer_list)}
        
        # Process savings to merge routes
        for saving, i, j in savings:
            if time.time() - self.start_time > self.time_limit * 0.9:  # Use 90% of time
                break
                
            route_i_idx = customer_to_route.get(i)
            route_j_idx = customer_to_route.get(j)
            
            if route_i_idx is None or route_j_idx is None:
                continue
                
            route_i = routes[route_i_idx]
            route_j = routes[route_j_idx]
            
            if route_i_idx == route_j_idx or len(route_i) + len(route_j) > 100:  # Avoid overly long routes
                continue
            
            # Try merging: route_i -> route_j (or reverse orders)
            candidates = [
                route_i + route_j,     # concat i after j
            ]
            
            if route_i != route_j:
                candidates.append(route_j + route_i)  # concat j after i
                
                # Try inserting j route at beginning of i route  
                candidates.append(route_j + route_i)
                
                # Try inserting i at end of j
                candidates.append(route_j + route_i)
                
                # More merge possibilities
                candidates.append(route_i + route_j)
                
            best_candidate = None
            best_feasible = False
            for candidate in candidates:
                feasible, _ = check_route_feasibility(self.vrptw, candidate)
                if feasible:
                    if not best_candidate or len(best_candidate) < len(candidate): # Prefer shorter routes
                        best_candidate = candidate
                        best_feasible = True
            
            if best_feasible and route_i_idx != route_j_idx:
                # Merge the routes
                merged_route = best_candidate
                # Update routes and tracking
                max_idx = max(route_i_idx, route_j_idx)
                min_idx = min(route_i_idx, route_j_idx)
                
                # Replace both routes with the merged route in the smaller index
                routes[min_idx] = merged_route
                routes[max_idx] = []  # Empty the higher index route
                
                # Update customer_to_route mapping for all customers in the merged route
                for cust in merged_route:
                    customer_to_route[cust] = min_idx

                # Clear out empty routes
                routes = [r for r in routes if r]
                
                # Renumber mapping
                for idx, route in enumerate(routes):
                    for cust in route:
                        customer_to_route[cust] = idx
        
        # Clean up remaining individual routes
        final_routes = []
        for route in routes:
            if route:  # Only add non-empty routes
                final_routes.append(route)
        
        return final_routes
    
    def local_search_improve(self, initial_solution):
        """Apply local search improvements to the solution."""
        if not initial_solution:
            return initial_solution
            
        solution = [r[:] for r in initial_solution]  # Deep copy
        improved = True
        iteration = 0
        
        while improved and time.time() - self.start_time < self.time_limit * 0.8:
            improved = False
            iteration += 1
            print(f"Iteration {iteration}, current best cost: {self.best_objective}")
            
            # 2-opt within route
            for r_idx in range(len(solution)):
                route = solution[r_idx]
                if len(route) < 2:
                    continue
                    
                # Try swapping two edges within route
                for i in range(len(route) - 1):
                    for j in range(i + 2, len(route)):
                        # Swap edges at positions (i, i+1) and (j, j+1)
                        # This reverses the segment from i+1 to j
                        new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                        
                        feasible, cost = check_route_feasibility(self.vrptw, new_route)
                        if feasible and cost < check_route_feasibility(self.vrptw, route)[1]:
                            solution[r_idx] = new_route
                            total_cost, all_routes = self.evaluate_solution(solution)
                            if total_cost < self.best_objective:
                                self.update_best_solution(solution)
                                improved = True
                            
            # Or-opt (moving 1,2,3 consecutive nodes between routes)
            for i_r in range(len(solution)):
                for j_r in range(len(solution)):
                    if i_r == j_r:
                        continue
                        
                    route_i = solution[i_r][:]  # Copy of route i
                    route_j = solution[j_r][:]  # Copy of route j
                    
                    # Try moving segments of various lengths
                    for seg_len in range(1, min(3, len(route_i)) + 1):
                        for seg_pos in range(len(route_i) - seg_len + 1):
                            if time.time() - self.start_time > self.time_limit * 0.8:
                                break
                                
                            # Extract segment to move
                            seg = route_i[seg_pos:seg_pos + seg_len]
                            new_route_i = route_i[:seg_pos] + route_i[seg_pos + seg_len:]
                            
                            # Check if route_i (after removal) is still valid
                            feasible_i, _ = check_route_feasibility(self.vrptw, new_route_i)
                            if not feasible_i:
                                continue
                            
                            # Try inserting segment at different positions in route_j
                            for ins_pos in range(len(route_j) + 1):
                                new_route_j = route_j[:ins_pos] + seg + route_j[ins_pos:]
                                feasible_j, _ = check_route_feasibility(self.vrptw, new_route_j)
                                
                                if feasible_j:
                                    new_solution = [r[:] for r in solution]
                                    new_solution[i_r] = new_route_i
                                    new_solution[j_r] = new_route_j
                                    
                                    total_cost, all_routes = self.evaluate_solution(new_solution)
                                    if total_cost < self.best_objective:
                                        solution = new_solution
                                        self.update_best_solution(solution)
                                        improved = True
        
        return solution
    
    def evaluate_solution(self, routes):
        """Evaluate the objective value of a solution (total cost)."""
        total_cost = 0
        route_details = []
        
        for route in routes:
            if not route:
                continue
                
            full_route = [self.vrptw.depot_source_id] + route + [self.vrptw.depot_sink_id]
            
            for i in range(len(full_route)-1):
                from_node = full_route[i]
                to_node = full_route[j] = i + 1
                total_cost += self.vrptw.travel_costs[from_node, to_node]
        
        return total_cost, routes
    
    def update_best_solution(self, solution):
        """Update the best solution found and log it."""
        obj_val, routes = self.evaluate_solution(solution)
        if obj_val < self.best_objective:
            self.best_objective = obj_val
            self.best_solution = [r[:] for r in solution if r]  # Copy only non-empty routes
            
            if self.logger:
                self.logger.log(obj_val)
            print(f"New best solution found with cost {obj_val}")


def main():
    parser = argparse.ArgumentParser(description='Solve Vehicle Routing Problem with Time Windows')
    parser.add_argument('--instance_path', required=True, help='Path to instance JSON file')
    parser.add_argument('--solution_path', required=True, help='Path to output solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', required=False, help='Path to log file (optional)')
    
    args = parser.parse_args()
    
    # Load instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    # Setup logger if path provided
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Initialize VRPTW object
    vrptw = VRPTW(instance)
    
    # Initialize solver and solve
    solver = VRPTWSolver(vrptw, args.time_limit, logger)
    solution_routes = solver.solve()
    
    # Format solution as required
    if solution_routes:
        formatted_routes = []
        for route in solution_routes:
            formatted_routes.append([vrptw.depot_source_id] + route + [vrptw.depot_sink_id])
        
        obj_val, _ = solver.evaluate_solution(solution_routes)
    else:
        formatted_routes = [[vrptw.depot_source_id, vrptw.depot_sink_id]] if hasattr(vrptw, 'depot_source_id') and hasattr(vrptw, 'depot_sink_id') else [[]]
        obj_val = 0.0
        
        for route in instance['nodes']:
            if route['type'] == 'customer':
                # This would mean we couldn't solve, let's return an error case instead
                print("Error: Could not create valid solution")
                formatted_routes = [[]]
                obj_val = float('inf')
                break
    
    solution_output = {
        "objective_value": obj_val,
        "routes": formatted_routes
    }
    
    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution_output, f, indent=2)


if __name__ == '__main__':
    main()