import json
import argparse
import time
import random
import math
import sys

def load_instance(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def compute_travel_time(i, j, departure_time, distances, speeds, period_boundaries, H):
    """Compute travel time from vertex i to vertex j when departing at departure_time."""
    remaining_dist = distances[i][j]
    if remaining_dist < 1e-12:
        return 0.0
    
    current_time = departure_time
    
    # Find the period containing current_time
    h = 0
    while h < H - 1 and current_time >= period_boundaries[h + 1] - 1e-12:
        h += 1
    
    while h < H and remaining_dist > 1e-12:
        speed = speeds[i][j][h]
        if speed < 1e-12:
            return float('inf')
        
        time_needed = remaining_dist / speed
        tentative_arrival = current_time + time_needed
        
        if tentative_arrival <= period_boundaries[h + 1] + 1e-10:
            return tentative_arrival - departure_time
        else:
            time_in_period = period_boundaries[h + 1] - current_time
            if time_in_period < 0:
                time_in_period = 0.0
            dist_covered = speed * time_in_period
            remaining_dist -= dist_covered
            current_time = period_boundaries[h + 1]
            h += 1
    
    if remaining_dist > 1e-12:
        return float('inf')
    return current_time - departure_time

def evaluate_tour(tour, distances, speeds, period_boundaries, H):
    """Evaluate total tour duration."""
    current_time = 0.0
    for idx in range(len(tour) - 1):
        i = tour[idx]
        j = tour[idx + 1]
        tt = compute_travel_time(i, j, current_time, distances, speeds, period_boundaries, H)
        if tt >= float('inf') / 2:
            return float('inf')
        current_time += tt
    return current_time

def compute_arrival_times(tour, distances, speeds, period_boundaries, H):
    """Compute arrival time at each position in the tour."""
    n_pos = len(tour)
    times = [0.0] * n_pos
    for idx in range(n_pos - 1):
        i = tour[idx]
        j = tour[idx + 1]
        tt = compute_travel_time(i, j, times[idx], distances, speeds, period_boundaries, H)
        times[idx + 1] = times[idx] + tt
    return times

def evaluate_from_position(tour, pos, time_at_pos, distances, speeds, period_boundaries, H):
    """Evaluate tour from position pos to end, returning final arrival time."""
    current_time = time_at_pos
    for idx in range(pos, len(tour) - 1):
        i = tour[idx]
        j = tour[idx + 1]
        tt = compute_travel_time(i, j, current_time, distances, speeds, period_boundaries, H)
        if tt >= float('inf') / 2:
            return float('inf')
        current_time += tt
    return current_time

def nearest_neighbor(n, distances, speeds, period_boundaries, H):
    """Nearest neighbor construction based on actual travel time."""
    visited = [False] * (n + 1)
    tour = [0]
    visited[0] = True
    current = 0
    current_time = 0.0
    
    for _ in range(n):
        best_next = -1
        best_tt = float('inf')
        for j in range(1, n + 1):
            if not visited[j]:
                tt = compute_travel_time(current, j, current_time, distances, speeds, period_boundaries, H)
                if tt < best_tt:
                    best_tt = tt
                    best_next = j
        if best_next == -1:
            break
        tour.append(best_next)
        visited[best_next] = True
        current_time += best_tt
        current = best_next
    
    tour.append(0)
    return tour

def greedy_insertion(n, distances, speeds, period_boundaries, H):
    """Cheapest insertion heuristic."""
    if n == 0:
        return [0, 0]
    
    # Start with depot and nearest customer
    best_first = 1
    best_tt = float('inf')
    for j in range(1, n + 1):
        tt = compute_travel_time(0, j, 0.0, distances, speeds, period_boundaries, H)
        if tt < best_tt:
            best_tt = tt
            best_first = j
    
    tour = [0, best_first, 0]
    inserted = {best_first}
    
    while len(inserted) < n:
        best_cost_increase = float('inf')
        best_customer = -1
        best_pos = -1
        
        for c in range(1, n + 1):
            if c in inserted:
                continue
            for pos in range(1, len(tour)):
                # Insert c at position pos
                new_tour = tour[:pos] + [c] + tour[pos:]
                new_obj = evaluate_tour(new_tour, distances, speeds, period_boundaries, H)
                old_obj = evaluate_tour(tour, distances, speeds, period_boundaries, H)
                cost_increase = new_obj - old_obj
                if cost_increase < best_cost_increase:
                    best_cost_increase = cost_increase
                    best_customer = c
                    best_pos = pos
        
        if best_customer == -1:
            break
        tour = tour[:best_pos] + [best_customer] + tour[best_pos:]
        inserted.add(best_customer)
    
    return tour

def two_opt_first_improvement(tour, distances, speeds, period_boundaries, H, time_limit, start_time):
    """2-opt with first improvement strategy for directed TSP."""
    n = len(tour) - 2
    if n <= 1:
        return tour, evaluate_tour(tour, distances, speeds, period_boundaries, H)
    
    best_obj = evaluate_tour(tour, distances, speeds, period_boundaries, H)
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit * 0.80:
            break
        improved = False
        arrival_times = compute_arrival_times(tour, distances, speeds, period_boundaries, H)
        
        for i in range(1, n + 1):
            if time.time() - start_time > time_limit * 0.80:
                return tour, best_obj
            for j in range(i + 2, n + 2):
                # Reverse segment tour[i:j]
                new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
                # Evaluate from position i-1 (last unchanged position)
                eval_start = max(0, i - 1)
                new_obj = evaluate_from_position(new_tour, eval_start, arrival_times[eval_start], 
                                                distances, speeds, period_boundaries, H)
                if new_obj < best_obj - 1e-10:
                    tour = new_tour
                    best_obj = new_obj
                    improved = True
                    break
            if improved:
                break
    
    return tour, best_obj

def or_opt_improvement(tour, distances, speeds, period_boundaries, H, time_limit, start_time):
    """Or-opt: relocate segments of size 1, 2, 3."""
    n = len(tour) - 2
    if n <= 1:
        return tour, evaluate_tour(tour, distances, speeds, period_boundaries, H)
    
    best_obj = evaluate_tour(tour, distances, speeds, period_boundaries, H)
    
    for seg_len in [1, 2, 3]:
        improved = True
        while improved:
            if time.time() - start_time > time_limit * 0.80:
                return tour, best_obj
            improved = False
            
            for i in range(1, n + 2 - seg_len):
                if time.time() - start_time > time_limit * 0.80:
                    return tour, best_obj
                
                segment = tour[i:i + seg_len]
                remaining = tour[:i] + tour[i + seg_len:]
                
                for j in range(1, len(remaining)):
                    new_tour = remaining[:j] + segment + remaining[j:]
                    new_obj = evaluate_tour(new_tour, distances, speeds, period_boundaries, H)
                    if new_obj < best_obj - 1e-10:
                        tour = new_tour
                        best_obj = new_obj
                        improved = True
                        break
                if improved:
                    break
    
    return tour, best_obj

def solve_dp(n, distances, speeds, period_boundaries, H, time_limit, start_time):
    """Held-Karp DP for small instances."""
    if n == 0:
        return [0, 0], 0.0
    
    full_mask = (1 << n) - 1
    INF = float('inf')
    
    dp = [[INF] * (n + 1) for _ in range(1 << n)]
    parent = [[-1] * (n + 1) for _ in range(1 << n)]
    
    # Base: depot to each customer
    for j in range(1, n + 1):
        tt = compute_travel_time(0, j, 0.0, distances, speeds, period_boundaries, H)
        mask = 1 << (j - 1)
        dp[mask][j] = tt
    
    # Fill DP table
    for mask in range(1, 1 << n):
        if time.time() - start_time > time_limit * 0.75:
            return None, INF
        
        for j in range(1, n + 1):
            if not (mask & (1 << (j - 1))):
                continue
            if dp[mask][j] >= INF:
                continue
            
            current_time = dp[mask][j]
            
            for k in range(1, n + 1):
                if mask & (1 << (k - 1)):
                    continue
                new_mask = mask | (1 << (k - 1))
                tt = compute_travel_time(j, k, current_time, distances, speeds, period_boundaries, H)
                if tt >= INF:
                    continue
                new_time = current_time + tt
                if new_time < dp[new_mask][k] - 1e-12:
                    dp[new_mask][k] = new_time
                    parent[new_mask][k] = j
    
    # Find best last customer
    best_obj = INF
    best_last = -1
    for j in range(1, n + 1):
        if dp[full_mask][j] >= INF:
            continue
        tt = compute_travel_time(j, 0, dp[full_mask][j], distances, speeds, period_boundaries, H)
        total = dp[full_mask][j] + tt
        if total < best_obj:
            best_obj = total
            best_last = j
    
    if best_last == -1:
        return None, INF
    
    # Reconstruct tour
    path = []
    mask = full_mask
    current = best_last
    while current != -1:
        path.append(current)
        prev = parent[mask][current]
        mask = mask ^ (1 << (current - 1))
        current = prev
    path.reverse()
    
    tour = [0] + path + [0]
    return tour, best_obj

def simulated_annealing(tour, distances, speeds, period_boundaries, H, time_limit, start_time, logger):
    """Simulated annealing with mixed neighborhood operators."""
    n = len(tour) - 2
    if n <= 1:
        return tour, evaluate_tour(tour, distances, speeds, period_boundaries, H)
    
    best_tour = tour[:]
    best_obj = evaluate_tour(tour, distances, speeds, period_boundaries, H)
    current_tour = tour[:]
    current_obj = best_obj
    
    # Adaptive temperature
    T = max(best_obj * 0.05, 0.1)
    
    # Adjust cooling based on instance size
    if n <= 30:
        cooling = 0.9995
    elif n <= 80:
        cooling = 0.9997
    else:
        cooling = 0.99985
    
    no_improve_count = 0
    max_no_improve = 5000 + n * 50
    iteration = 0
    
    while time.time() - start_time < time_limit * 0.90:
        iteration += 1
        
        # Generate neighbor
        r = random.random()
        if r < 0.35:
            # 2-opt move
            i = random.randint(1, n)
            j = random.randint(i + 1, min(i + max(3, n // 3), n + 1))
            new_tour = current_tour[:i] + current_tour[i:j][::-1] + current_tour[j:]
        elif r < 0.60:
            # Relocate single node
            i = random.randint(1, n)
            node = current_tour[i]
            remaining = current_tour[:i] + current_tour[i + 1:]
            j = random.randint(1, len(remaining) - 1)
            new_tour = remaining[:j] + [node] + remaining[j:]
        elif r < 0.80:
            # Swap two nodes
            i = random.randint(1, n)
            j = random.randint(1, n)
            if i == j:
                continue
            new_tour = current_tour[:]
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        else:
            # Relocate segment of size 2 or 3
            seg_len = random.choice([2, 3])
            if n < seg_len + 1:
                continue
            i = random.randint(1, n - seg_len + 1)
            segment = current_tour[i:i + seg_len]
            remaining = current_tour[:i] + current_tour[i + seg_len:]
            j = random.randint(1, len(remaining) - 1)
            new_tour = remaining[:j] + segment + remaining[j:]
        
        new_obj = evaluate_tour(new_tour, distances, speeds, period_boundaries, H)
        if new_obj >= float('inf') / 2:
            continue
        
        delta = new_obj - current_obj
        if delta < 0 or (T > 1e-10 and random.random() < math.exp(-delta / T)):
            current_tour = new_tour
            current_obj = new_obj
            
            if current_obj < best_obj - 1e-12:
                best_obj = current_obj
                best_tour = current_tour[:]
                no_improve_count = 0
                if logger:
                    solution_dict = {"objective_value": best_obj, "tour": best_tour}
                    logger.log_solution(best_obj, solution_dict)
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        T *= cooling
        
        # Reheat if stuck
        if no_improve_count > max_no_improve:
            T = max(best_obj * 0.08, 0.5)
            no_improve_count = 0
            current_tour = best_tour[:]
            current_obj = best_obj
    
    return best_tour, best_obj

def iterated_local_search(tour, distances, speeds, period_boundaries, H, time_limit, start_time, logger):
    """Iterated Local Search with perturbation and local search."""
    n = len(tour) - 2
    if n <= 1:
        return tour, evaluate_tour(tour, distances, speeds, period_boundaries, H)
    
    best_tour = tour[:]
    best_obj = evaluate_tour(tour, distances, speeds, period_boundaries, H)
    current_tour = tour[:]
    current_obj = best_obj
    
    while time.time() - start_time < time_limit * 0.85:
        # Perturbation: random 2-opt or swap
        perturbed = current_tour[:]
        num_perturb = random.randint(1, max(1, n // 10))
        for _ in range(num_perturb):
            r = random.random()
            if r < 0.5:
                i = random.randint(1, n)
                j = random.randint(i + 1, min(i + max(2, n // 5), n + 1))
                perturbed = perturbed[:i] + perturbed[i:j][::-1] + perturbed[j:]
            else:
                i = random.randint(1, n)
                j = random.randint(1, n)
                if i != j:
                    perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
        
        # Local search on perturbed solution
        improved_tour, improved_obj = two_opt_first_improvement(
            perturbed, distances, speeds, period_boundaries, H, time_limit, start_time)
        
        # Acceptance criterion
        if improved_obj < current_obj:
            current_tour = improved_tour
            current_obj = improved_obj
            if current_obj < best_obj - 1e-12:
                best_obj = current_obj
                best_tour = current_tour[:]
                if logger:
                    logger.log_solution(best_obj, {"objective_value": best_obj, "tour": best_tour})
        else:
            # Accept with small probability (record-to-record)
            threshold = best_obj * 1.02
            if improved_obj < threshold:
                current_tour = improved_tour
                current_obj = improved_obj
    
    return best_tour, best_obj

def main():
    parser = argparse.ArgumentParser(description='Time-Dependent TSP Solver')
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    start_time = time.time()
    random.seed(42)
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Load instance
    data = load_instance(args.instance_path)
    
    n_customers = data['parameters']['n_customers']
    total_vertices = data['parameters']['total_vertices']
    H = data['parameters']['H']
    period_boundaries = data['parameters']['period_boundaries']
    
    distances = data['arcs']['distances']
    speeds = data['arcs']['speeds_v_ijh']
    
    best_tour = None
    best_obj = float('inf')
    
    if n_customers == 0:
        best_tour = [0, 0]
        best_obj = 0.0
        if logger:
            logger.log_solution(best_obj, {"objective_value": best_obj, "tour": best_tour})
    
    elif n_customers <= 17:
        # Try exact DP
        tour, obj = solve_dp(n_customers, distances, speeds, period_boundaries, H, 
                            args.time_limit, start_time)
        if tour is not None and obj < float('inf'):
            best_tour = tour
            best_obj = obj
            if logger:
                logger.log_solution(best_obj, {"objective_value": best_obj, "tour": best_tour})
        else:
            # Fallback to heuristic if DP timed out
            tour = nearest_neighbor(n_customers, distances, speeds, period_boundaries, H)
            best_obj = evaluate_tour(tour, distances, speeds, period_boundaries, H)
            best_tour = tour
            if logger:
                logger.log_solution(best_obj, {"objective_value": best_obj, "tour": best_tour})
            
            # Try to improve with local search and SA
            remaining = args.time_limit - (time.time() - start_time)
            if remaining > 2:
                tour2, obj2 = two_opt_first_improvement(best_tour, distances, speeds, 
                                                       period_boundaries, H, args.time_limit, start_time)
                if obj2 < best_obj:
                    best_obj = obj2
                    best_tour = tour2
                    if logger:
                        logger.log_solution(best_obj, {"objective_value": best_obj, "tour": best_tour})
            
            remaining = args.time_limit - (time.time() - start_time)
            if remaining > 1:
                tour3, obj3 = simulated_annealing(best_tour, distances, speeds, period_boundaries, H,
                                                 args.time_limit, start_time, logger)
                if obj3 < best_obj:
                    best_obj = obj3
                    best_tour = tour3
    
    else:
        # Heuristic approach for larger instances
        # Construction phase
        tour_nn = nearest_neighbor(n_customers, distances, speeds, period_boundaries, H)
        obj_nn = evaluate_tour(tour_nn, distances, speeds, period_boundaries, H)
        best_tour = tour_nn
        best_obj = obj_nn
        if logger:
            logger.log_solution(best_obj, {"objective_value": best_obj, "tour": best_tour})
        
        # Also try random tours and keep the best
        for _ in range(min(5, max(1, 10 - n_customers // 20))):
            if time.time() - start_time > args.time_limit * 0.1:
                break
            customers = list(range(1, n_customers + 1))
            random.shuffle(customers)
            rand_tour = [0] + customers + [0]
            rand_obj = evaluate_tour(rand_tour, distances, speeds, period_boundaries, H)
            if rand_obj < best_obj:
                best_obj = rand_obj
                best_tour = rand_tour
                if logger:
                    logger.log_solution(best_obj, {"objective_value": best_obj, "tour": best_tour})
        
        # Local search: 2-opt
        remaining = args.time_limit - (time.time() - start_time)
        if remaining > 3:
            tour2, obj2 = two_opt_first_improvement(best_tour, distances, speeds, 
                                                   period_boundaries, H, args.time_limit, start_time)
            if obj2 < best_obj:
                best_obj = obj2
                best_tour = tour2
                if logger:
                    logger.log_solution(best_obj, {"objective_value": best_obj, "tour": best_tour})
        
        # Or-opt
        remaining = args.time_limit - (time.time() - start_time)
        if remaining > 3:
            tour3, obj3 = or_opt_improvement(best_tour, distances, speeds, 
                                            period_boundaries, H, args.time_limit, start_time)
            if obj3 < best_obj:
                best_obj = obj3
                best_tour = tour3
                if logger:
                    logger.log_solution(best_obj, {"objective_value": best_obj, "tour": best_tour})
        
        # Metaheuristic phase: choose based on remaining time and instance size
        remaining = args.time_limit - (time.time() - start_time)
        if remaining > 2:
            if n_customers <= 60:
                # Use SA for smaller instances
                tour4, obj4 = simulated_annealing(best_tour, distances, speeds, period_boundaries, H,
                                                 args.time_limit, start_time, logger)
                if obj4 < best_obj:
                    best_obj = obj4
                    best_tour = tour4
            else:
                # Use SA with faster evaluation for larger instances
                tour4, obj4 = simulated_annealing(best_tour, distances, speeds, period_boundaries, H,
                                                 args.time_limit, start_time, logger)
                if obj4 < best_obj:
                    best_obj = obj4
                    best_tour = tour4
        
        # If still have time, try ILS
        remaining = args.time_limit - (time.time() - start_time)
        if remaining > 5 and n_customers <= 100:
            tour5, obj5 = iterated_local_search(best_tour, distances, speeds, period_boundaries, H,
                                               args.time_limit, start_time, logger)
            if obj5 < best_obj:
                best_obj = obj5
                best_tour = tour5
    
    # Ensure we have a valid solution
    if best_tour is None:
        best_tour = [0] + list(range(1, n_customers + 1)) + [0]
        best_obj = evaluate_tour(best_tour, distances, speeds, period_boundaries, H)
    
    # Write solution
    solution = {
        "objective_value": best_obj,
        "tour": best_tour
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    from solution_logger import SolutionLogger
    main()