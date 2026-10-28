import argparse
import json
import time
import math
import random
from solution_logger import SolutionLogger


def solve(instance, time_limit, logger=None):
    start_time = time.time()
    
    num_nodes = instance["num_nodes"]
    D0 = instance["distance_limitation_d0"]
    depot = instance["depot_vertex"]
    scores = instance["scores"]
    dist = instance["distance_matrix"]
    
    # Best solution tracking
    best_score = 0
    best_tour = [depot]
    
    if logger:
        logger.log(best_score)
    
    def tour_distance(tour):
        """Calculate total distance of a tour (cycle)."""
        if len(tour) <= 1:
            return 0
        total = 0
        for i in range(len(tour)):
            total += dist[tour[i]][tour[(i + 1) % len(tour)]]
        return total
    
    def tour_score(tour):
        """Calculate total score of visited nodes."""
        return sum(scores[v] for v in tour)
    
    def update_best(tour):
        nonlocal best_score, best_tour
        s = tour_score(tour)
        d = tour_distance(tour)
        if d <= D0 and s > best_score:
            best_score = s
            best_tour = list(tour)
            if logger:
                logger.log(best_score)
            return True
        return False
    
    # Greedy construction: iteratively insert the node with best score/distance ratio
    def greedy_construct():
        tour = [depot]
        visited = {depot}
        current_dist = 0
        
        while True:
            best_ratio = -1
            best_node = -1
            best_pos = -1
            best_delta = float('inf')
            
            for node in range(num_nodes):
                if node in visited:
                    continue
                if scores[node] == 0:
                    continue
                
                # Find best insertion position
                for pos in range(len(tour)):
                    i = tour[pos]
                    j = tour[(pos + 1) % len(tour)]
                    delta = dist[i][node] + dist[node][j] - dist[i][j]
                    
                    if current_dist + delta <= D0:
                        ratio = scores[node] / max(delta, 1)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_node = node
                            best_pos = pos
                            best_delta = delta
            
            if best_node == -1:
                break
            
            tour.insert(best_pos + 1, best_node)
            visited.add(best_node)
            current_dist += best_delta
        
        return tour
    
    # Initial greedy solution
    tour = greedy_construct()
    update_best(tour)
    
    # 2-opt improvement
    def two_opt(tour):
        if len(tour) <= 3:
            return tour
        improved = True
        while improved:
            improved = False
            for i in range(len(tour)):
                for j in range(i + 2, len(tour)):
                    if i == 0 and j == len(tour) - 1:
                        continue
                    # Reverse segment between i+1 and j
                    a, b = tour[i], tour[(i + 1) % len(tour)]
                    c, d = tour[j], tour[(j + 1) % len(tour)]
                    
                    old_dist = dist[a][b] + dist[c][d]
                    new_dist = dist[a][c] + dist[b][d]
                    
                    if new_dist < old_dist:
                        # Reverse the segment
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        improved = True
                        
                    if time.time() - start_time > time_limit - 1:
                        return tour
        return tour
    
    tour = two_opt(tour)
    update_best(tour)
    
    # Or-opt: try moving single nodes or sequences
    def or_opt(tour):
        if len(tour) <= 2:
            return tour
        improved = True
        while improved:
            improved = False
            for seg_len in [1, 2, 3]:
                for i in range(len(tour)):
                    if time.time() - start_time > time_limit - 1:
                        return tour
                    
                    if i + seg_len > len(tour):
                        continue
                    
                    # Try removing segment starting at i of length seg_len
                    # and inserting it elsewhere
                    # Skip if depot would be removed
                    segment_indices = list(range(i, min(i + seg_len, len(tour))))
                    segment = [tour[idx] for idx in segment_indices]
                    if depot in segment:
                        continue
                    
                    remaining = [tour[k] for k in range(len(tour)) if k not in segment_indices]
                    
                    # Cost of removing segment
                    prev_i = tour[(i - 1) % len(tour)]
                    next_seg = tour[(i + seg_len) % len(tour)]
                    
                    old_cost = dist[prev_i][segment[0]]
                    for s in range(len(segment) - 1):
                        old_cost += dist[segment[s]][segment[s+1]]
                    old_cost += dist[segment[-1]][next_seg]
                    
                    new_link = dist[prev_i][next_seg]
                    removal_saving = old_cost - new_link
                    
                    best_insert_cost = float('inf')
                    best_insert_pos = -1
                    
                    for j in range(len(remaining)):
                        a = remaining[j]
                        b = remaining[(j + 1) % len(remaining)]
                        insert_cost = dist[a][segment[0]]
                        for s in range(len(segment) - 1):
                            insert_cost += dist[segment[s]][segment[s+1]]
                        insert_cost += dist[segment[-1]][b] - dist[a][b]
                        
                        if insert_cost < best_insert_cost:
                            best_insert_cost = insert_cost
                            best_insert_pos = j
                    
                    if best_insert_cost < removal_saving - 1e-9 and best_insert_pos >= 0:
                        # Perform the move
                        new_tour = remaining[:best_insert_pos + 1] + segment + remaining[best_insert_pos + 1:]
                        if tour_distance(new_tour) <= D0:
                            tour[:] = new_tour
                            improved = True
                            break
                if improved:
                    break
        return tour
    
    tour = or_opt(tour)
    update_best(tour)
    
    # Try adding more nodes after optimization freed up distance
    def try_add_nodes(tour):
        current_dist = tour_distance(tour)
        visited = set(tour)
        improved = True
        while improved:
            improved = False
            best_ratio = -1
            best_node = -1
            best_pos = -1
            best_delta = float('inf')
            
            for node in range(num_nodes):
                if node in visited:
                    continue
                if scores[node] == 0:
                    continue
                
                for pos in range(len(tour)):
                    i = tour[pos]
                    j = tour[(pos + 1) % len(tour)]
                    delta = dist[i][node] + dist[node][j] - dist[i][j]
                    
                    if current_dist + delta <= D0:
                        ratio = scores[node] / max(delta, 1)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_node = node
                            best_pos = pos
                            best_delta = delta
            
            if best_node >= 0:
                tour.insert(best_pos + 1, best_node)
                visited.add(best_node)
                current_dist += best_delta
                improved = True
        
        return tour
    
    tour = try_add_nodes(tour)
    update_best(tour)
    
    # Node swap: try replacing a visited node with an unvisited one
    def node_swap(tour):
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > time_limit - 1:
                return tour
            visited = set(tour)
            for idx in range(len(tour)):
                if tour[idx] == depot:
                    continue
                node_out = tour[idx]
                prev_node = tour[(idx - 1) % len(tour)]
                next_node = tour[(idx + 1) % len(tour)]
                
                # Cost savings from removing node_out
                old_cost = dist[prev_node][node_out] + dist[node_out][next_node]
                new_cost_remove = dist[prev_node][next_node]
                saving = old_cost - new_cost_remove
                
                for node_in in range(num_nodes):
                    if node_in in visited:
                        continue
                    if scores[node_in] <= scores[node_out]:
                        continue
                    
                    # Try inserting node_in at position idx
                    cost_add = dist[prev_node][node_in] + dist[node_in][next_node] - dist[prev_node][next_node]
                    
                    current_d = tour_distance(tour)
                    new_d = current_d - saving + cost_add + new_cost_remove - dist[prev_node][next_node]
                    # Simplified: new distance = current - old_cost + dist[prev][new] + dist[new][next]
                    # But let's just compute directly
                    
                    # Actually: new_d = current_d - (dist[prev][out] + dist[out][next]) + (dist[prev][in] + dist[in][next])
                    actual_new_d = current_d - dist[prev_node][node_out] - dist[node_out][next_node] + dist[prev_node][node_in] + dist[node_in][next_node]
                    
                    if actual_new_d <= D0:
                        tour[idx] = node_in
                        improved = True
                        break
                
                if improved:
                    break
        return tour
    
    tour = node_swap(tour)
    update_best(tour)
    
    # Perturbation-based local search (ILS)
    def perturb(tour, strength=3):
        """Remove random nodes and re-insert them plus try new nodes."""
        if len(tour) <= 2:
            return list(tour)
        
        new_tour = list(tour)
        non_depot = [i for i, v in enumerate(new_tour) if v != depot]
        
        if len(non_depot) == 0:
            return new_tour
        
        # Remove some random nodes
        num_remove = min(strength, len(non_depot))
        to_remove = random.sample(non_depot, num_remove)
        to_remove.sort(reverse=True)
        removed_nodes = []
        for idx in to_remove:
            removed_nodes.append(new_tour[idx])
            new_tour.pop(idx)
        
        # Re-insert removed nodes at best positions
        for node in removed_nodes:
            best_delta = float('inf')
            best_pos = -1
            current_d = tour_distance(new_tour)
            
            for pos in range(len(new_tour)):
                i = new_tour[pos]
                j = new_tour[(pos + 1) % len(new_tour)]
                delta = dist[i][node] + dist[node][j] - dist[i][j]
                if current_d + delta <= D0 and delta < best_delta:
                    best_delta = delta
                    best_pos = pos
            
            if best_pos >= 0:
                new_tour.insert(best_pos + 1, node)
        
        return new_tour
    
    # ILS loop
    current_tour = list(best_tour)
    current_score = best_score
    
    iterations = 0
    no_improve_count = 0
    
    while time.time() - start_time < time_limit - 2:
        iterations += 1
        
        # Determine perturbation strength
        strength = random.randint(2, min(max(3, len(current_tour) // 3), 8))
        
        new_tour = perturb(current_tour, strength)
        
        # Local search on perturbed solution
        new_tour = two_opt(new_tour)
        if time.time() - start_time > time_limit - 1:
            update_best(new_tour)
            break
        
        new_tour = try_add_nodes(new_tour)
        new_tour = or_opt(new_tour)
        if time.time() - start_time > time_limit - 1:
            update_best(new_tour)
            break
        
        new_tour = try_add_nodes(new_tour)
        new_tour = node_swap(new_tour)
        new_tour = try_add_nodes(new_tour)
        
        new_score = tour_score(new_tour)
        new_dist = tour_distance(new_tour)
        
        if new_dist <= D0:
            if new_score > current_score:
                current_tour = new_tour
                current_score = new_score
                no_improve_count = 0
            else:
                # Accept with some probability (simulated annealing style)
                if random.random() < 0.3:
                    current_tour = new_tour
                    current_score = new_score
                no_improve_count += 1
            
            update_best(new_tour)
        else:
            no_improve_count += 1
        
        # Restart from best if stuck
        if no_improve_count > 50:
            current_tour = list(best_tour)
            current_score = best_score
            no_improve_count = 0
    
    # Build solution
    tour = best_tour
    visited = list(tour)
    edges = []
    for i in range(len(tour)):
        edges.append([tour[i], tour[(i + 1) % len(tour)]])
    
    return {
        "objective_value": float(best_score),
        "visited_vertices": visited,
        "tour_edges": edges
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    solution = solve(instance, args.time_limit, logger)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()