import json
import argparse
import time
import random
import math
import sys

from solution_logger import SolutionLogger


def compute_travel_time(i, j, departure_time, distances, speeds_v_ijh, period_boundaries, H):
    """Compute travel time from i to j departing at departure_time using time-dependent speeds."""
    dist = distances[i][j]
    if dist == 0:
        return 0.0
    
    remaining = dist
    current_time = departure_time
    
    # Find current period
    period = H - 1  # default to last period
    for h in range(H):
        if current_time < period_boundaries[h + 1]:
            period = h
            break
    
    max_iter = H * 2 + 10  # safety limit
    for _ in range(max_iter):
        speed = speeds_v_ijh[i][j][period]
        if speed <= 1e-15:
            # Speed is essentially zero, can't traverse
            return float('inf')
        
        tentative_arrival = current_time + remaining / speed
        
        if period >= H - 1 or tentative_arrival <= period_boundaries[period + 1]:
            # Traversal completes within this period
            return tentative_arrival - departure_time
        else:
            # Travel until end of this period
            time_in_period = period_boundaries[period + 1] - current_time
            distance_covered = time_in_period * speed
            remaining -= distance_covered
            current_time = period_boundaries[period + 1]
            period += 1
            if remaining <= 1e-12:
                return current_time - departure_time
    
    return float('inf')


def evaluate_tour(tour, distances, speeds_v_ijh, period_boundaries, H):
    """Evaluate total tour duration."""
    current_time = 0.0
    for idx in range(len(tour) - 1):
        i = tour[idx]
        j = tour[idx + 1]
        tt = compute_travel_time(i, j, current_time, distances, speeds_v_ijh, period_boundaries, H)
        if tt == float('inf'):
            return float('inf')
        current_time += tt
    return current_time


def evaluate_tour_with_times(tour, distances, speeds_v_ijh, period_boundaries, H):
    """Evaluate total tour duration and return arrival times at each stop."""
    current_time = 0.0
    times = [0.0]
    for idx in range(len(tour) - 1):
        i = tour[idx]
        j = tour[idx + 1]
        tt = compute_travel_time(i, j, current_time, distances, speeds_v_ijh, period_boundaries, H)
        if tt == float('inf'):
            return float('inf'), times
        current_time += tt
        times.append(current_time)
    return current_time, times


def nearest_neighbor_tour(n_vertices, distances, speeds_v_ijh, period_boundaries, H, start=0):
    """Construct a tour using nearest neighbor heuristic with time-dependent travel times."""
    visited = [False] * n_vertices
    visited[start] = True
    tour = [start]
    current_time = 0.0
    
    for _ in range(n_vertices - 1):
        current = tour[-1]
        best_next = -1
        best_tt = float('inf')
        for j in range(n_vertices):
            if not visited[j]:
                tt = compute_travel_time(current, j, current_time, distances, speeds_v_ijh, period_boundaries, H)
                if tt < best_tt:
                    best_tt = tt
                    best_next = j
        if best_next == -1:
            break
        visited[best_next] = True
        tour.append(best_next)
        current_time += best_tt
    
    tour.append(start)
    return tour


def two_opt_improve(tour, distances, speeds_v_ijh, period_boundaries, H, time_limit_abs, best_obj):
    """Perform 2-opt local search improvement."""
    n = len(tour) - 1  # tour length without the returning depot
    improved = True
    
    while improved:
        if time.time() > time_limit_abs:
            break
        improved = False
        for i in range(1, n - 1):
            if time.time() > time_limit_abs:
                break
            for j in range(i + 1, n):
                if time.time() > time_limit_abs:
                    break
                # Reverse segment from i to j
                new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
                new_obj = evaluate_tour(new_tour, distances, speeds_v_ijh, period_boundaries, H)
                if new_obj < best_obj - 1e-9:
                    tour = new_tour
                    best_obj = new_obj
                    improved = True
                    break
            if improved:
                break
    
    return tour, best_obj


def or_opt_improve(tour, distances, speeds_v_ijh, period_boundaries, H, time_limit_abs, best_obj):
    """Perform Or-opt local search (relocate segments of size 1, 2, 3)."""
    n = len(tour) - 1  # number of locations (including depot at position 0)
    improved = True
    
    while improved:
        if time.time() > time_limit_abs:
            break
        improved = False
        for seg_len in [1, 2, 3]:
            if improved:
                break
            for i in range(1, n - seg_len + 1):
                if time.time() > time_limit_abs:
                    break
                if improved:
                    break
                # Extract segment tour[i:i+seg_len]
                segment = tour[i:i+seg_len]
                remaining = tour[:i] + tour[i+seg_len:]
                
                for j in range(1, len(remaining)):
                    if j == i:
                        continue  # same position
                    new_tour = remaining[:j] + segment + remaining[j:]
                    new_obj = evaluate_tour(new_tour, distances, speeds_v_ijh, period_boundaries, H)
                    if new_obj < best_obj - 1e-9:
                        tour = new_tour
                        best_obj = new_obj
                        improved = True
                        break
    
    return tour, best_obj


def partial_evaluate_tour(tour, idx_from, distances, speeds_v_ijh, period_boundaries, H, arrival_time_at_idx):
    """Evaluate tour from a given index onwards, given the arrival time at that index."""
    current_time = arrival_time_at_idx
    for k in range(idx_from, len(tour) - 1):
        i = tour[k]
        j = tour[k + 1]
        tt = compute_travel_time(i, j, current_time, distances, speeds_v_ijh, period_boundaries, H)
        if tt == float('inf'):
            return float('inf')
        current_time += tt
    return current_time


def two_opt_improve_fast(tour, distances, speeds_v_ijh, period_boundaries, H, time_limit_abs, best_obj):
    """Faster 2-opt with partial re-evaluation."""
    n = len(tour) - 1
    improved = True
    
    # Precompute cumulative arrival times
    obj, times = evaluate_tour_with_times(tour, distances, speeds_v_ijh, period_boundaries, H)
    best_obj = obj
    
    while improved:
        if time.time() > time_limit_abs:
            break
        improved = False
        for i in range(1, n - 1):
            if time.time() > time_limit_abs:
                break
            for j in range(i + 1, n):
                if time.time() > time_limit_abs:
                    break
                # Reverse segment from i to j
                new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
                # Re-evaluate from position i-1 (since the edge from i-1 to i changes)
                # The arrival time at position i-1 is times[i-1] (departure time)
                # But now the edge from tour[i-1] to new_tour[i] is different
                # We need to re-evaluate from position i onward
                # Actually we need to start from the last unchanged position
                new_obj = partial_evaluate_tour(new_tour, i-1, distances, speeds_v_ijh, period_boundaries, H, times[i-1] if i > 0 else 0.0)
                # But wait, times[i-1] is the arrival time at position i-1, which is departure from i-1
                # Actually times[0] = 0 (depot departure), times[k] = arrival at position k
                # The departure from position k is times[k]
                # Edge from position i-1 to i: departs at times[i-1]
                # After reversal, position i-1 is unchanged, so departure from i-1 is still times[i-1]
                # But the next vertex changed, so we recompute from i-1
                
                # Compute from i-1 to end with new_tour
                departure = times[i-1] if i-1 >= 0 else 0.0
                current_time = departure
                for k in range(i-1, len(new_tour) - 1):
                    a = new_tour[k]
                    b = new_tour[k + 1]
                    tt = compute_travel_time(a, b, current_time, distances, speeds_v_ijh, period_boundaries, H)
                    if tt == float('inf'):
                        current_time = float('inf')
                        break
                    current_time += tt
                new_obj = current_time
                
                if new_obj < best_obj - 1e-9:
                    tour = new_tour
                    best_obj = new_obj
                    # Recompute times
                    obj, times = evaluate_tour_with_times(tour, distances, speeds_v_ijh, period_boundaries, H)
                    improved = True
                    break
            if improved:
                break
    
    return tour, best_obj


def perturbation_double_bridge(tour):
    """Double-bridge perturbation for iterated local search."""
    # tour: [0, ..., 0] with depot at start and end
    n = len(tour) - 1  # positions 0 to n-1 are the actual vertices
    if n < 8:
        # For small tours, just do a random segment reversal
        if n <= 2:
            return tour[:]
        i = random.randint(1, n - 2)
        j = random.randint(i + 1, n - 1)
        new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
        return new_tour
    
    # Pick 3 random cut points (excluding 0 and n)
    cuts = sorted(random.sample(range(1, n), 3))
    a, b, c = cuts
    
    # Split into 4 segments
    seg1 = tour[:a]
    seg2 = tour[a:b]
    seg3 = tour[b:c]
    seg4 = tour[c:n]
    
    # Reconnect in double-bridge fashion: seg1 + seg3 + seg2 + seg4 + [depot]
    new_tour = seg1 + seg3 + seg2 + seg4 + [tour[0]]
    return new_tour


def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    start_time = time.time()
    time_limit_abs = start_time + args.time_limit - 1  # reserve 1 second for output
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    params = instance['parameters']
    n_customers = params['n_customers']
    total_vertices = params['total_vertices']
    H = params['H']
    period_boundaries = params['period_boundaries']
    
    distances = instance['arcs']['distances']
    speeds_v_ijh = instance['arcs']['speeds_v_ijh']
    
    # Build initial solution using nearest neighbor
    best_tour = nearest_neighbor_tour(total_vertices, distances, speeds_v_ijh, period_boundaries, H, start=0)
    best_obj = evaluate_tour(best_tour, distances, speeds_v_ijh, period_boundaries, H)
    
    solution_dict = {
        "objective_value": best_obj,
        "tour": best_tour
    }
    if logger:
        logger.log_solution(best_obj, solution_dict)
    
    # Try multiple starting points with nearest neighbor
    for start_customer in range(1, min(total_vertices, 20)):
        if time.time() > time_limit_abs:
            break
        # Build NN tour starting from different customers but always depot first
        visited = [False] * total_vertices
        visited[0] = True
        visited[start_customer] = True
        tour = [0, start_customer]
        current_time = compute_travel_time(0, start_customer, 0.0, distances, speeds_v_ijh, period_boundaries, H)
        
        for _ in range(total_vertices - 2):
            current = tour[-1]
            best_next = -1
            best_tt = float('inf')
            for j in range(total_vertices):
                if not visited[j]:
                    tt = compute_travel_time(current, j, current_time, distances, speeds_v_ijh, period_boundaries, H)
                    if tt < best_tt:
                        best_tt = tt
                        best_next = j
            if best_next == -1:
                break
            visited[best_next] = True
            tour.append(best_next)
            current_time += best_tt
        
        tour.append(0)
        obj = evaluate_tour(tour, distances, speeds_v_ijh, period_boundaries, H)
        if obj < best_obj - 1e-9:
            best_tour = tour
            best_obj = obj
            solution_dict = {
                "objective_value": best_obj,
                "tour": best_tour
            }
            if logger:
                logger.log_solution(best_obj, solution_dict)
    
    # Local search improvement
    if time.time() < time_limit_abs:
        tour, obj = two_opt_improve_fast(best_tour[:], distances, speeds_v_ijh, period_boundaries, H, time_limit_abs, best_obj)
        if obj < best_obj - 1e-9:
            best_tour = tour
            best_obj = obj
            solution_dict = {
                "objective_value": best_obj,
                "tour": best_tour
            }
            if logger:
                logger.log_solution(best_obj, solution_dict)
    
    # Or-opt improvement
    if time.time() < time_limit_abs:
        tour, obj = or_opt_improve(best_tour[:], distances, speeds_v_ijh, period_boundaries, H, time_limit_abs, best_obj)
        if obj < best_obj - 1e-9:
            best_tour = tour
            best_obj = obj
            solution_dict = {
                "objective_value": best_obj,
                "tour": best_tour
            }
            if logger:
                logger.log_solution(best_obj, solution_dict)
    
    # Iterated Local Search with double-bridge perturbation
    ils_best_tour = best_tour[:]
    ils_best_obj = best_obj
    no_improve_count = 0
    
    while time.time() < time_limit_abs:
        # Perturbation
        perturbed_tour = perturbation_double_bridge(ils_best_tour)
        perturbed_obj = evaluate_tour(perturbed_tour, distances, speeds_v_ijh, period_boundaries, H)
        
        if time.time() > time_limit_abs:
            break
        
        # Local search on perturbed solution
        improved_tour, improved_obj = two_opt_improve_fast(
            perturbed_tour, distances, speeds_v_ijh, period_boundaries, H, 
            min(time_limit_abs, time.time() + max(1.0, (time_limit_abs - time.time()) * 0.3)),
            perturbed_obj
        )
        
        if time.time() < time_limit_abs:
            improved_tour, improved_obj = or_opt_improve(
                improved_tour, distances, speeds_v_ijh, period_boundaries, H,
                min(time_limit_abs, time.time() + max(0.5, (time_limit_abs - time.time()) * 0.2)),
                improved_obj
            )
        
        # Acceptance criterion
        if improved_obj < ils_best_obj - 1e-9:
            ils_best_tour = improved_tour
            ils_best_obj = improved_obj
            no_improve_count = 0
            
            if ils_best_obj < best_obj - 1e-9:
                best_tour = ils_best_tour[:]
                best_obj = ils_best_obj
                solution_dict = {
                    "objective_value": best_obj,
                    "tour": best_tour
                }
                if logger:
                    logger.log_solution(best_obj, solution_dict)
        else:
            no_improve_count += 1
            # Accept worse solutions occasionally to escape local optima
            if no_improve_count > 10:
                # Restart from best known
                ils_best_tour = best_tour[:]
                ils_best_obj = best_obj
                no_improve_count = 0
    
    # Final 2-opt pass if time remains
    if time.time() < time_limit_abs:
        tour, obj = two_opt_improve_fast(best_tour[:], distances, speeds_v_ijh, period_boundaries, H, time_limit_abs, best_obj)
        if obj < best_obj - 1e-9:
            best_tour = tour
            best_obj = obj
            solution_dict = {
                "objective_value": best_obj,
                "tour": best_tour
            }
            if logger:
                logger.log_solution(best_obj, solution_dict)
    
    # Write final solution
    solution = {
        "objective_value": best_obj,
        "tour": best_tour
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    solve()