import argparse
import json
import math
import time
import random
from collections import defaultdict
import heapq
import itertools


def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def compute_travel_time(departure_time, distance, max_speeds, speeds):
    """
    Compute the travel time for a given arc considering the time-dependent travel speeds.
    
    Args:
        departure_time: The time when the vehicle starts traversing the arc.
        distance: Total travel distance of the arc.
        max_speeds: [total_vertices][total_vertices] matrix of maximum speeds per arc.
        speeds: [total_vertices][total_vertices][H] matrix of time-dependent speeds per arc.
        
    Returns:
        The total time needed to cross the arc.
    """
    time_periods = len(speeds[0][0])
    period_start_times = [i * 60 for i in range(time_periods + 1)]  # Assuming periodic partitions
    
    period_index = 0
    for i in range(len(period_start_times) - 1):
        if period_start_times[i] <= departure_time < period_start_times[i + 1]:
            period_index = i
            break
    
    current_time = departure_time
    remaining_dist = distance
    
    while remaining_dist > 0:
        start_of_current_period = period_start_times[period_index]
        end_of_current_period = period_start_times[period_index + 1]
        
        # Speed during the current period for this arc
        current_speed = speeds[period_index]
        if current_speed <= 0:
            return float('inf')
        
        # Time to traverse remaining distance if staying in current period
        potential_arrival_time = current_time + remaining_dist / current_speed
        
        if potential_arrival_time <= end_of_current_period:
            # Entire remaining distance is covered in this period
            travel_time = potential_arrival_time - current_time
            current_time += travel_time
            remaining_dist = 0
        else:
            # Only part of the distance is covered in this period
            time_in_current_period = end_of_current_period - current_time
            dist_covered = current_speed * time_in_current_period
            
            remaining_dist -= dist_covered
            current_time = end_of_current_period
            
            if period_index < len(period_start_times) - 2:
                period_index += 1
            else:
                # Reached the end of the planning horizon
                return float('inf')

    return current_time - departure_time


def compute_total_travel_time(tour, distances, max_speeds_matrix, speeds_cube, b_h, period_boundaries):
    """
    Calculate the total travel time for a given tour, considering time-dependent speeds.
    
    Args:
        tour: List of vertices in the tour (starting and ending at depot).
        distances: Distance matrix.
        max_speeds_matrix: [n][n] matrix of maximum speeds.
        speeds_cube: [n][n][H] matrix of speeds per period per arc.
        b_h: Congestion factors per period.
        period_boundaries: List of period boundaries.
    
    Returns:
        Total travel time for the tour, or infinity if it's invalid.
    """
    n = len(tour)
    if n < 2:
        return 0.0

    total_time = 0.0
    current_time = 0.0

    for i in range(n - 1):
        from_node = tour[i]
        to_node = tour[i + 1]
        
        # Get speeds for this arc over each period
        max_speed = max_speeds_matrix[from_node][to_node]
        d_ijh = []  # This would typically come from delta_ijh values
        
        # Speed for arc (i,j) in period h is: delta_ijh * b_h * u_ij
        # Since we don't explicitly have delta_ijh, we'll use the speeds_cube directly
        speeds_for_arc = []
        for h in range(len(b_h)):
            start_h = period_boundaries[h]
            end_h = period_boundaries[h+1]
            
            # Calculate delta factor if it's not provided, based on provided v_ijh values
            v_ijh = speeds_cube[from_node][to_node][h]
            expected_speed = v_ijh
            speeds_for_arc.append(expected_speed)
        
        # Use the first matrix to get speeds
        speeds_over_periods = speeds_cube[from_node][to_node]
        
        travel_time = compute_travel_time(current_time, distances[from_node][to_node], 
                                         max_speeds_matrix[from_node][to_node],
                                         speeds_over_periods)
                                        
        if travel_time == float('inf'):
            return float('inf')
        
        total_time += travel_time
        current_time += travel_time

    return total_time


class TDSPTSolver:
    def __init__(self, instance_data):
        self.instance = instance_data
        self.n_customers = instance_data['parameters']['n_customers']
        self.total_vertices = instance_data['parameters']['total_vertices']
        self.H = instance_data['parameters']['H']
        self.b_h = instance_data['parameters']['b_h']
        self.time_horizon_T = instance_data['parameters']['time_horizon_T']
        self.period_boundaries = instance_data['parameters']['period_boundaries']
        
        self.vertices = instance_data['vertices']
        self.depot_idx = instance_data['depot']['index']  # Should be 0
        
        self.distances = instance_data['arcs']['distances']
        self.max_speeds = instance_data['arcs']['max_speeds_u_ij']
        self.v_ijh = instance_data['arcs']['speeds_v_ijh']
        
        self.all_nodes = set(range(self.total_vertices))
        
        # Validate that depot is at index 0
        assert self.depot_idx == 0, f"Expected depot at index 0, but got {self.depot_idx}"

    def construct_greedy_solution(self):
        """
        Construct an initial solution using greedy heuristic based on earliest arrival.
        """
        unvisited = set(range(1, self.total_vertices))  # All except depot
        current_node = 0  # Start at depot
        tour = [current_node]
        current_time = 0.0

        while unvisited:
            next_node = min(unvisited, 
                           key=lambda node: self.calculate_earliest_arrival(current_time, current_node, node))
            
            travel_time = compute_travel_time(current_time, 
                                            self.distances[current_node][next_node],
                                            self.max_speeds,
                                            self.v_ijh[current_node][next_node])
            current_time += travel_time
            current_node = next_node
            tour.append(current_node)
            unvisited.remove(current_node)

        # Return to depot
        travel_time = compute_travel_time(current_time,
                                        self.distances[current_node][0],
                                        self.max_speeds,
                                        self.v_ijh[current_node][0])
        current_time += travel_time
        tour.append(0)

        total_time = compute_total_travel_time(tour, self.distances, self.max_speeds, 
                                             self.v_ijh, self.b_h, self.period_boundaries)
        return tour, total_time

    def calculate_earliest_arrival(self, current_time, from_node, to_node):
        """Helper to estimate arrival time based on current period's speed."""
        return compute_travel_time(current_time,
                                   self.distances[from_node][to_node],
                                   self.max_speeds,
                                   self.v_ijh[from_node][to_node])

    def perturb_tour(self, tour):
        """
        Generate a neighbor solution by performing a simple swap or insertion.
        """
        # We exclude first and last (depot visits) when considering swaps
        tour_length = len(tour) - 1  # Exclude last depot visit
        idx1, idx2 = random.sample(range(1, tour_length), 2)  # Only consider internal nodes
        
        new_tour = tour[:]
        # Reverse segment between idx1 and idx2
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        new_tour[idx1:idx2+1] = reversed(new_tour[idx1:idx2+1])
        
        # Ensure still starts and ends at depot
        total_time = compute_total_travel_time(new_tour, self.distances, self.max_speeds, 
                                             self.v_ijh, self.b_h, self.period_boundaries)
        return new_tour, total_time

    def solve_with_local_search(self, time_limit=300):
        """
        Solve using iterative improvement local search methods.
        """
        start_time = time.time()
        
        # Initialize with greedy solution
        best_tour, best_obj = self.construct_greedy_solution()
        
        # Local search for improvements
        no_improvement_count = 0
        max_no_improvement = 50
        
        iterations = 0
        while time.time() - start_time < time_limit and no_improvement_count < max_no_improvement:
            iterations += 1
            candidate_tour, candidate_obj = self.perturb_tour(best_tour)
            
            if candidate_obj < best_obj:
                best_tour = candidate_tour
                best_obj = candidate_obj
                no_improvement_count = 0  # Reset counter on improvement
            else:
                no_improvement_count += 1
                
            if iterations % 100 == 0:  # Occasionally perform larger perturbation
                random.shuffle(best_tour[1:-1])  # Shuffle middle section
                new_time = compute_total_travel_time(best_tour, self.distances, self.max_speeds, 
                                                   self.v_ijh, self.b_h, self.period_boundaries)
                if new_time < best_obj:
                    best_obj = new_time
                else:
                    # Revert if no good
                    _, original_time = self.construct_greedy_solution()
                    if original_time < best_obj:
                        best_tour, best_obj = self.construct_greedy_solution()
                
        return best_tour, best_obj


def solve_tdsp(instance_data, args, logger=None):
    """
    Main solving function for time-dependent traveling salesman problem.
    """
    solver = TDSPTSolver(instance_data)
    
    # Log initial solution if available before running the main algorithm
    initial_solution, initial_cost = solver.construct_greedy_solution()
    if logger:
        logger.log(initial_cost)
    
    # Run the improvement algorithm
    tour, obj_val = solver.solve_with_local_search(time_limit=args.time_limit)
    
    # Log final solution
    if logger and obj_val < initial_cost:
        logger.log(obj_val)
    
    # Ensure we return a valid result with proper format
    solution = {
        "objective_value": obj_val,
        "tour": tour
    }
    
    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Load the instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Initialize logger if path provided
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None
        
    # Solve the instance
    start_time = time.time()
    try:
        solution = solve_tdsp(instance_data, args, logger)
        solve_time = time.time() - start_time
        
        # Write the solution
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        
        print(f"Solved in {solve_time:.2f}s")
        
    except Exception as e:
        print(f"Error solving instance: {str(e)}")
        raise


if __name__ == '__main__':
    main()