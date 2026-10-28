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
        
        # Identify depot nodes
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
        
        # Ensure depot IDs are assigned properly
        if len(depot_nodes) >= 1:
            if not self.depot_source_id:
                self.depot_source_id = depot_nodes[0]['id']
            if not self.depot_sink_id:
                self.depot_sink_id = depot_nodes[-1]['id']  # Assume last is sink if there's multiple
        
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
        
        # Capacity check (for edge capacity constraint)
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
    
    # Start from depot - need to handle time window of depot
    prev_node_id = vrptw.depot_source_id
    prev_node = vrptw.nodes[prev_node_id] 
    # Start time is when we leave the depot, which can be time 0 (earliest window open for depot)
    current_time = max(prev_node['time_window_open'], 0.0)  # Depot starts at 0
    
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
        
        # Update time: arrival time + wait time + service time
        current_time = service_start_time + node['service_time']
        total_cost += vrptw.travel_costs[prev_node_id, node_id]
        prev_node_id = node_id
    
    # Return to depot
    travel_time_back = vrptw.travel_times[prev_node_id, vrptw.depot_sink_id]
    return_arrival_time_at_depot = current_time + travel_time_back
    
    if return_arrival_time_at_depot > vrptw.planning_horizon:
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
        if self.best_objective == float('inf') or (greedy_solution and self.evaluate_solution(greedy_solution)[0] < self.best_objective):
            self.update_best_solution(greedy_solution)
        
        # Run local search improvments
        print("Running local search improvements...")
        improved_solution = self.local_search_improve(self.best_solution) if self.best_solution else greedy_solution
        if improved_solution and self.evaluate_solution(improved_solution)[0] < self.best_objective:
            self.update_best_solution(improved_solution)
            
        return self.best_solution
    
    def greedy_savings_init(self):
        """Create initial solution using Clarke-Wright Savings heuristic."""
        n_customers = len(self.vrptw.customer_ids)
        customer_list = list(self.vrptw.customer_ids)
        
        if not customer_list:
            return []

        # Each customer initially in their own route as [[customer_id]]
        routes = [[cust] for cust in customer_list]
        customer_to_route = {cust: idx for idx, cust in enumerate(customer_list)}
        
        # Calculate savings: c_0i + c_i0 - c_ij for every pair of customers (i,j)
        savings = []
        depot_id = self.vrptw.depot_source_id
        for i_idx in range(len(customer_list)):
            i = customer_list[i_idx]
            for j_idx in range(i_idx+1, len(customer_list)):
                j = customer_list[j_idx]
                
                # Only consider valid arcs
                if (i, j) in self.vrptw.admissible_arcs or (j, i) in self.vrptw.admissible_arcs:
                    saving = (self.vrptw.travel_costs[depot_id, i] + 
                             self.vrptw.travel_costs[j, depot_id] - 
                             self.vrptw.travel_costs[i, j])
                    savings.append((saving, i, j))
        
        # Sort savings in descending order
        savings.sort(reverse=True)
        
        # Process savings to merge routes
        for saving, i, j in savings:
            if time.time() - self.start_time > self.time_limit * 0.7:  # Use 70% of time for init
                break
                
            route_i_idx = customer_to_route.get(i)
            route_j_idx = customer_to_route.get(j)
            
            if route_i_idx is None or route_j_idx is None or route_i_idx == route_j_idx:
                continue
            
            route_i = routes[route_i_idx]
            route_j = routes[route_j_idx]
            
            # Try forward merge: route_i + route_j
            test_route_forward = route_i + route_j
            feasible_forward, _ = check_route_feasibility(self.vrptw, test_route_forward)
            
            # Try backward merge: route_j + route_i
            test_route_backward = route_j + route_i
            feasible_backward, _ = check_route_feasibility(self.vrptw, test_route_backward)
            
            if feasible_forward or feasible_backward:
                if feasible_forward and feasible_backward:
                    # Choose the one with better (shorter) length as heuristic
                    chosen_route = test_route_forward if len(test_route_forward) <= len(test_route_backward) else test_route_backward
                elif feasible_forward:
                    chosen_route = test_route_forward
                else:
                    chosen_route = test_route_backward
                
                # Perform the merge
                max_idx = max(route_i_idx, route_j_idx)
                min_idx = min(route_i_idx, route_j_idx)
                
                # Replace both routes with the merged route in the smaller index
                if min_idx < len(routes):
                    routes[min_idx] = chosen_route
                
                # Clear the other route
                if max_idx < len(routes):
                    routes[max_idx] = []  # This will be removed later
                
                # Update customer_to_route mapping for all customers in the merged route
                for cust in chosen_route:
                    customer_to_route[cust] = min_idx
        
        # Clean up empty routes
        routes = [r for r in routes if r]
        
        return routes
    
    def local_search_improve(self, initial_solution):
        """Apply local search improvements to the solution."""
        if not initial_solution:
            return initial_solution
            
        solution = [r[:] for r in initial_solution]  # Deep copy
        improved = True
        pass_count = 0
        
        # Continue improving until no improvement for a few passes or timeout
        while improved and pass_count < 3 and time.time() - self.start_time < self.time_limit * 0.9:
            improved = False
            pass_count += 1
            print(f"Local search pass {pass_count}, current best cost: {self.best_objective}")
            
            # 2-opt improvement: try swapping within routes
            for r_idx in range(len(solution)):
                route = solution[r_idx]
                if len(route) < 2:
                    continue
                    
                best_local_update = None
                best_delta_cost = 0  # negative if beneficial
                
                for i in range(len(route)):
                    for j in range(i + 2, len(route)):
                        # Reverse segment from i to j
                        new_route = route[:]
                        new_route[i:j+1] = reversed(new_route[i:j+1])
                        
                        _, old_cost = check_route_feasibility(self.vrptw, route)
                        feasible_new, new_cost = check_route_feasibility(self.vrptw, new_route)
                        
                        if feasible_new and new_cost < old_cost:
                            if new_cost - old_cost < best_delta_cost:  # Best improvement found so far
                                best_delta_cost = new_cost - old_cost
                                best_local_update = (r_idx, new_route)
                
                if best_local_update and best_delta_cost < 0:
                    r_idx, updated_route = best_local_update
                    solution[r_idx] = updated_route
                    improved = True
                    total_cost, _ = self.evaluate_solution(solution)
                    if total_cost < self.best_objective:
                        self.update_best_solution(solution)
                        print(f"2-opt improved: {self.best_objective}")
            
            # Relocate: try moving customers from one route to another
            for src_route_idx in range(len(solution)):
                src_route = solution[src_route_idx][:]
                for dest_route_idx in range(len(solution)):
                    if src_route_idx == dest_route_idx or not src_route:
                        continue
                    
                    for pos_in_src in range(len(src_route)):
                        customer_to_move = src_route[pos_in_src]
                        candidate_src_route = src_route[:pos_in_src] + src_route[pos_in_src+1:]
                        
                        # Check if source route remains valid after removal
                        feasible_src, _ = check_route_feasibility(self.vrptw, candidate_src_route)
                        
                        if not feasible_src:
                            continue
                        
                        dest_route = solution[dest_route_idx][:]
                        
                        # Try inserting the customer into different positions in destination
                        for pos_in_dest in range(len(dest_route) + 1):
                            candidate_dest_route = dest_route[:pos_in_dest] + [customer_to_move] + dest_route[pos_in_dest:]
                            feasible_dest, _ = check_route_feasibility(self.vrptw, candidate_dest_route)
                            
                            if feasible_dest:
                                # Apply the move temporarily
                                temp_solution = [r[:] for r in solution]
                                temp_solution[src_route_idx] = candidate_src_route
                                temp_solution[dest_route_idx] = candidate_dest_route
                                
                                total_old_cost, _ = self.evaluate_solution(solution)
                                total_new_cost, _ = self.evaluate_solution(temp_solution)
                                
                                if total_new_cost < total_old_cost:
                                    solution = temp_solution
                                    improved = True
                                    self.update_best_solution(solution)
                                    print(f"Relocate improved: {self.best_objective}")
                                    # Break out of nested loops and restart since topology changed
                                    break
                        if improved:
                            break
                if time.time() - self.start_time > self.time_limit * 0.9:
                    break

        return solution
    
    def evaluate_solution(self, routes):
        """Evaluate the objective value of a solution (total cost)."""
        total_cost = 0.0
        
        for route in routes:
            if not route:
                continue
                
            # Add depot at beginning and end
            full_route = [self.vrptw.depot_source_id] + route + [self.vrptw.depot_sink_id]
            
            for i in range(len(full_route)-1):
                from_node = full_route[i]
                to_node = full_route[i+1]
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
    if solution_routes is not None:
        formatted_routes = []
        for route in solution_routes:
            if route:  # Only format non-empty routes
                formatted_routes.append([vrptw.depot_source_id] + route + [vrptw.depot_sink_id])
        
        if formatted_routes:
            obj_val, _ = solver.evaluate_solution(solution_routes)
        else:
            obj_val = 0.0
    else:
        # Handle case where no solution was found (shouldn't happen if depot exists)
        formatted_routes = [[vrptw.depot_source_id, vrptw.depot_sink_id]] if vrptw.depot_source_id and vrptw.depot_sink_id else []
        obj_val = 0.0
    
    solution_output = {
        "objective_value": float(obj_val),
        "routes": formatted_routes
    }
    
    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution_output, f, indent=2)


if __name__ == '__main__':
    main()