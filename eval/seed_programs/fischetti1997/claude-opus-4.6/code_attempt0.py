import argparse
import json
import math
import time
import random
import sys
from solution_logger import SolutionLogger


def solve(instance_path, solution_path, time_limit, log_path):
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    n = instance['problem_size']['n']
    m = instance['problem_size']['m']
    
    nodes = instance['nodes']
    clusters = instance['clusters']
    
    # Build coordinate arrays
    node_x = {}
    node_y = {}
    node_cluster = {}
    for node in nodes:
        node_x[node['id']] = node['x']
        node_y[node['id']] = node['y']
        node_cluster[node['id']] = node['cluster']
    
    # Parse distance matrix
    dist_values = instance['distance_matrix_upper_triangular']['values']
    
    # Build node id list sorted
    node_ids = sorted(node_x.keys())
    node_index = {nid: idx for idx, nid in enumerate(node_ids)}
    
    def get_dist(i, j):
        """Get distance between node ids i and j"""
        ii = node_index[i]
        jj = node_index[j]
        if ii == jj:
            return 0
        if ii > jj:
            ii, jj = jj, ii
        # Upper triangular index: row ii, col jj (jj > ii)
        # Index = ii * n - ii*(ii+1)/2 + (jj - ii - 1)
        idx = ii * n - ii * (ii + 1) // 2 + (jj - ii - 1)
        return dist_values[idx]
    
    # Cluster data
    cluster_nodes = {}
    for cl in clusters:
        cluster_nodes[cl['id']] = cl['node_ids']
    
    cluster_ids = sorted(cluster_nodes.keys())
    
    # For small instances, we can try more thorough approaches
    # For large instances, use heuristics
    
    # Precompute inter-cluster distance cache
    # For each pair of clusters, precompute distances between all node pairs
    # This could be large, so we'll be selective
    
    def compute_tour_cost(tour):
        """Compute cost of a tour (list of node ids)"""
        cost = 0
        for i in range(len(tour)):
            cost += get_dist(tour[i], tour[(i + 1) % len(tour)])
        return cost
    
    def nearest_neighbor_tour(selected_nodes, start_idx=0):
        """Build a tour using nearest neighbor heuristic"""
        remaining = set(range(len(selected_nodes)))
        tour = [start_idx]
        remaining.remove(start_idx)
        while remaining:
            last = tour[-1]
            best_next = None
            best_dist = float('inf')
            for r in remaining:
                d = get_dist(selected_nodes[last], selected_nodes[r])
                if d < best_dist:
                    best_dist = d
                    best_next = r
            tour.append(best_next)
            remaining.remove(best_next)
        return [selected_nodes[i] for i in tour]
    
    def two_opt(tour):
        """Improve tour with 2-opt"""
        improved = True
        best_cost = compute_tour_cost(tour)
        n_tour = len(tour)
        while improved:
            improved = False
            for i in range(n_tour - 1):
                for j in range(i + 2, n_tour):
                    if j == n_tour - 1 and i == 0:
                        continue
                    # Cost of current edges
                    d1 = get_dist(tour[i], tour[i + 1])
                    d2 = get_dist(tour[j], tour[(j + 1) % n_tour])
                    # Cost of new edges
                    d3 = get_dist(tour[i], tour[j])
                    d4 = get_dist(tour[i + 1], tour[(j + 1) % n_tour])
                    if d3 + d4 < d1 + d2:
                        tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                        best_cost = best_cost - d1 - d2 + d3 + d4
                        improved = True
            if time.time() - start_time > time_limit - 1:
                break
        return tour, best_cost
    
    def three_opt_segment(tour, max_iterations=1):
        """Simple or-opt (moving segments of 1, 2, or 3 nodes)"""
        n_tour = len(tour)
        improved = True
        best_cost = compute_tour_cost(tour)
        iteration = 0
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            for seg_len in [1, 2, 3]:
                for i in range(n_tour):
                    if time.time() - start_time > time_limit - 1:
                        return tour, best_cost
                    # Remove segment starting at i of length seg_len
                    # Try inserting it at every other position
                    prev_i = (i - 1) % n_tour
                    end_i = (i + seg_len - 1) % n_tour
                    next_end = (end_i + 1) % n_tour
                    
                    # Cost removed
                    cost_removed = get_dist(tour[prev_i], tour[i]) + get_dist(tour[end_i], tour[next_end])
                    # Cost of reconnecting after removal
                    cost_reconnect = get_dist(tour[prev_i], tour[next_end])
                    
                    saving_from_removal = cost_removed - cost_reconnect
                    
                    # Try inserting segment between j and j+1
                    for j in range(n_tour):
                        if j == prev_i or j == i or j == end_i:
                            continue
                        next_j = (j + 1) % n_tour
                        if next_j == i:
                            continue
                        
                        cost_insert_old = get_dist(tour[j], tour[next_j])
                        cost_insert_new = get_dist(tour[j], tour[i]) + get_dist(tour[end_i], tour[next_j])
                        
                        cost_increase = cost_insert_new - cost_insert_old
                        
                        if saving_from_removal > cost_increase + 0.5:
                            # Perform the move
                            # Extract segment
                            segment = []
                            for k in range(seg_len):
                                segment.append(tour[(i + k) % n_tour])
                            
                            # Remove segment from tour
                            new_tour = []
                            k = next_end
                            for _ in range(n_tour - seg_len):
                                new_tour.append(tour[k % n_tour])
                                k = (k + 1) % n_tour
                            
                            # Find j in new_tour and insert segment after it
                            try:
                                j_idx = new_tour.index(tour[j])
                            except ValueError:
                                continue
                            
                            final_tour = new_tour[:j_idx + 1] + segment + new_tour[j_idx + 1:]
                            new_cost = compute_tour_cost(final_tour)
                            if new_cost < best_cost - 0.5:
                                tour = final_tour
                                best_cost = new_cost
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return tour, best_cost
    
    # Strategy: 
    # 1. For each cluster, pick a representative (initially closest to centroid or random)
    # 2. Build a TSP tour on selected nodes
    # 3. Optimize: alternate between improving the tour and changing node selections
    
    # Compute cluster centroids
    cluster_centroids = {}
    for cid in cluster_ids:
        xs = [node_x[nid] for nid in cluster_nodes[cid]]
        ys = [node_y[nid] for nid in cluster_nodes[cid]]
        cluster_centroids[cid] = (sum(xs) / len(xs), sum(ys) / len(ys))
    
    # Initial selection: node closest to centroid
    def select_closest_to_centroid():
        selected = {}
        for cid in cluster_ids:
            cx, cy = cluster_centroids[cid]
            best_node = None
            best_dist = float('inf')
            for nid in cluster_nodes[cid]:
                d = math.sqrt((node_x[nid] - cx) ** 2 + (node_y[nid] - cy) ** 2)
                if d < best_dist:
                    best_dist = d
                    best_node = nid
            selected[cid] = best_node
        return selected
    
    selected = select_closest_to_centroid()
    selected_list = [selected[cid] for cid in cluster_ids]
    
    # Build initial tour using nearest neighbor
    tour = nearest_neighbor_tour(selected_list)
    tour, tour_cost = two_opt(tour)
    
    best_tour = tour[:]
    best_cost = tour_cost
    
    if logger:
        logger.log(best_cost)
    
    # Node selection optimization: for each position in tour, try all nodes in that cluster
    def optimize_node_selection(tour):
        improved = True
        current_cost = compute_tour_cost(tour)
        while improved:
            improved = False
            for i in range(len(tour)):
                if time.time() - start_time > time_limit - 2:
                    return tour, current_cost
                current_node = tour[i]
                cid = node_cluster[current_node]
                prev_node = tour[(i - 1) % len(tour)]
                next_node = tour[(i + 1) % len(tour)]
                
                current_edge_cost = get_dist(prev_node, current_node) + get_dist(current_node, next_node)
                
                best_replacement = current_node
                best_edge_cost = current_edge_cost
                
                for candidate in cluster_nodes[cid]:
                    if candidate == current_node:
                        continue
                    new_edge_cost = get_dist(prev_node, candidate) + get_dist(candidate, next_node)
                    if new_edge_cost < best_edge_cost:
                        best_edge_cost = new_edge_cost
                        best_replacement = candidate
                
                if best_replacement != current_node:
                    current_cost = current_cost - current_edge_cost + best_edge_cost
                    tour[i] = best_replacement
                    improved = True
        return tour, current_cost
    
    tour, tour_cost = optimize_node_selection(tour)
    if tour_cost < best_cost:
        best_tour = tour[:]
        best_cost = tour_cost
        if logger:
            logger.log(best_cost)
    
    # Iterate: 2-opt then node selection
    for iteration in range(100):
        if time.time() - start_time > time_limit - 3:
            break
        
        tour, tour_cost = two_opt(tour)
        if tour_cost < best_cost:
            best_tour = tour[:]
            best_cost = tour_cost
            if logger:
                logger.log(best_cost)
        
        if time.time() - start_time > time_limit - 3:
            break
        
        old_cost = tour_cost
        tour, tour_cost = optimize_node_selection(tour)
        if tour_cost < best_cost:
            best_tour = tour[:]
            best_cost = tour_cost
            if logger:
                logger.log(best_cost)
        
        if tour_cost >= old_cost - 0.5:
            break
    
    # Or-opt improvement
    if time.time() - start_time < time_limit - 5:
        tour, tour_cost = three_opt_segment(best_tour[:], max_iterations=3)
        if tour_cost < best_cost:
            best_tour = tour[:]
            best_cost = tour_cost
            if logger:
                logger.log(best_cost)
        
        # Another round of node selection + 2-opt
        tour, tour_cost = optimize_node_selection(best_tour[:])
        if tour_cost < best_cost:
            best_tour = tour[:]
            best_cost = tour_cost
            if logger:
                logger.log(best_cost)
        
        if time.time() - start_time < time_limit - 3:
            tour, tour_cost = two_opt(best_tour[:])
            if tour_cost < best_cost:
                best_tour = tour[:]
                best_cost = tour_cost
                if logger:
                    logger.log(best_cost)
    
    # Perturbation-based restarts (double-bridge + re-optimize)
    def double_bridge(tour):
        n_t = len(tour)
        if n_t < 8:
            return tour[:]
        positions = sorted(random.sample(range(1, n_t), 3))
        a, b, c = positions
        new_tour = tour[:a] + tour[c:] + tour[b:c] + tour[a:b]
        return new_tour
    
    restart_count = 0
    while time.time() - start_time < time_limit - 4:
        restart_count += 1
        
        # Perturb the best tour
        if m >= 8:
            perturbed = double_bridge(best_tour[:])
        else:
            # For small tours, random shuffle
            perturbed = best_tour[:]
            random.shuffle(perturbed)
        
        # Re-optimize
        perturbed, p_cost = two_opt(perturbed)
        if time.time() - start_time > time_limit - 3:
            if p_cost < best_cost:
                best_tour = perturbed[:]
                best_cost = p_cost
                if logger:
                    logger.log(best_cost)
            break
        
        perturbed, p_cost = optimize_node_selection(perturbed)
        if time.time() - start_time > time_limit - 3:
            if p_cost < best_cost:
                best_tour = perturbed[:]
                best_cost = p_cost
                if logger:
                    logger.log(best_cost)
            break
        
        perturbed, p_cost = two_opt(perturbed)
        if time.time() - start_time > time_limit - 3:
            if p_cost < best_cost:
                best_tour = perturbed[:]
                best_cost = p_cost
                if logger:
                    logger.log(best_cost)
            break
        
        perturbed, p_cost = optimize_node_selection(perturbed)
        
        if p_cost < best_cost:
            best_tour = perturbed[:]
            best_cost = p_cost
            if logger:
                logger.log(best_cost)
    
    # Try multiple random starting tours if we still have time
    while time.time() - start_time < time_limit - 4:
        # Random node selection
        rand_selected = []
        for cid in cluster_ids:
            rand_selected.append(random.choice(cluster_nodes[cid]))
        
        # Random starting NN
        rtour = nearest_neighbor_tour(rand_selected, random.randint(0, m - 1))
        rtour, rcost = two_opt(rtour)
        if time.time() - start_time > time_limit - 2:
            if rcost < best_cost:
                best_tour = rtour[:]
                best_cost = rcost
                if logger:
                    logger.log(best_cost)
            break
        
        rtour, rcost = optimize_node_selection(rtour)
        if time.time() - start_time > time_limit - 2:
            if rcost < best_cost:
                best_tour = rtour[:]
                best_cost = rcost
                if logger:
                    logger.log(best_cost)
            break
        
        rtour, rcost = two_opt(rtour)
        rtour, rcost = optimize_node_selection(rtour)
        
        if rcost < best_cost:
            best_tour = rtour[:]
            best_cost = rcost
            if logger:
                logger.log(best_cost)
    
    # Final verification
    final_cost = compute_tour_cost(best_tour)
    
    # Verify one node per cluster
    visited_clusters = set()
    for nid in best_tour:
        cid = node_cluster[nid]
        assert cid not in visited_clusters, f"Cluster {cid} visited twice"
        visited_clusters.add(cid)
    assert len(visited_clusters) == m, f"Not all clusters visited: {len(visited_clusters)} vs {m}"
    
    solution = {
        "objective_value": float(final_cost),
        "tour": best_tour,
        "visited_nodes": best_tour[:]
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)