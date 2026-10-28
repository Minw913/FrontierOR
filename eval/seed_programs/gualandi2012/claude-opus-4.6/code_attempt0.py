import argparse
import json
import time
import sys
from collections import defaultdict

def solve_graph_coloring(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    start_time = time.time()
    
    # Read instance
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    graph = instance['graph']
    num_vertices = graph['num_vertices']
    num_edges = graph['num_edges']
    edges = graph['edges']
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Build adjacency list
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Handle trivial cases
    if num_edges == 0:
        coloring = {str(v): 0 for v in range(num_vertices)}
        color_classes = {"0": list(range(num_vertices))}
        obj = 1 if num_vertices > 0 else 0
        if logger:
            logger.log(obj)
        solution = {
            "objective_value": obj,
            "coloring": coloring,
            "color_classes": color_classes
        }
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
        return
    
    # Greedy coloring with DSATUR (degree of saturation) heuristic
    def dsatur_coloring():
        colors = [-1] * num_vertices
        saturation = [0] * num_vertices
        degree = [len(adj[v]) for v in range(num_vertices)]
        colored_count = 0
        # Track which colors are adjacent to each vertex
        adj_colors = [set() for _ in range(num_vertices)]
        
        num_colors_used = 0
        
        while colored_count < num_vertices:
            # Pick uncolored vertex with max saturation, break ties by degree
            best_v = -1
            best_sat = -1
            best_deg = -1
            for v in range(num_vertices):
                if colors[v] == -1:
                    if (saturation[v] > best_sat or 
                        (saturation[v] == best_sat and degree[v] > best_deg)):
                        best_v = v
                        best_sat = saturation[v]
                        best_deg = degree[v]
            
            # Assign smallest available color
            c = 0
            while c in adj_colors[best_v]:
                c += 1
            colors[best_v] = c
            if c + 1 > num_colors_used:
                num_colors_used = c + 1
            colored_count += 1
            
            # Update saturation of neighbors
            for u in adj[best_v]:
                if colors[u] == -1:
                    if c not in adj_colors[u]:
                        adj_colors[u].add(c)
                        saturation[u] += 1
        
        return colors, num_colors_used
    
    best_colors, best_num_colors = dsatur_coloring()
    if logger:
        logger.log(best_num_colors)
    
    # Try to improve with iterated greedy using different vertex orderings
    def greedy_color_with_order(order):
        colors = [-1] * num_vertices
        num_colors_used = 0
        for v in order:
            used = set()
            for u in adj[v]:
                if colors[u] != -1:
                    used.add(colors[u])
            c = 0
            while c in used:
                c += 1
            colors[v] = c
            if c + 1 > num_colors_used:
                num_colors_used = c + 1
        return colors, num_colors_used
    
    # Largest-first ordering
    order_lf = sorted(range(num_vertices), key=lambda v: len(adj[v]), reverse=True)
    cols, nc = greedy_color_with_order(order_lf)
    if nc < best_num_colors:
        best_colors, best_num_colors = cols, nc
        if logger:
            logger.log(best_num_colors)
    
    # Try random permutations for improvement
    import random
    random.seed(42)
    
    def time_remaining():
        return time_limit - (time.time() - start_time)
    
    # Attempt color reduction via reassignment
    def try_reduce_colors(colors, num_colors):
        """Try to reduce the number of colors by reassigning vertices."""
        improved = True
        current_colors = list(colors)
        current_num = num_colors
        
        while improved and time_remaining() > 1:
            improved = False
            # Find the color class with fewest vertices
            color_counts = defaultdict(list)
            for v in range(num_vertices):
                color_counts[current_colors[v]].append(v)
            
            # Sort color classes by size (smallest first)
            sorted_classes = sorted(color_counts.items(), key=lambda x: len(x[1]))
            
            for target_color, vertices_in_class in sorted_classes:
                if time_remaining() < 1:
                    break
                # Try to reassign all vertices in this class to other colors
                can_eliminate = True
                new_assignments = {}
                for v in vertices_in_class:
                    neighbor_colors = set()
                    for u in adj[v]:
                        if u in new_assignments:
                            neighbor_colors.add(new_assignments[u])
                        else:
                            neighbor_colors.add(current_colors[u])
                    
                    found = False
                    for c in range(current_num):
                        if c != target_color and c not in neighbor_colors:
                            new_assignments[v] = c
                            found = True
                            break
                    if not found:
                        can_eliminate = False
                        break
                
                if can_eliminate:
                    for v, c in new_assignments.items():
                        current_colors[v] = c
                    # Renumber colors
                    used_colors = set(current_colors)
                    color_map = {}
                    idx = 0
                    for c in sorted(used_colors):
                        color_map[c] = idx
                        idx += 1
                    current_colors = [color_map[c] for c in current_colors]
                    current_num = len(used_colors)
                    improved = True
                    break
        
        return current_colors, current_num
    
    best_colors, best_num_colors = try_reduce_colors(best_colors, best_num_colors)
    if logger:
        logger.log(best_num_colors)
    
    # Tabu search / local search to try to reduce colors further
    def try_k_coloring(k, max_iter=100000):
        """Try to find a valid k-coloring using tabu search."""
        if k >= best_num_colors:
            return None
        
        # Start from current best coloring, merge two smallest color classes
        colors = [c % k for c in best_colors]
        
        # Count conflicts
        def count_conflicts(colors):
            count = 0
            for u, v in edges:
                if colors[u] == colors[v]:
                    count += 1
            return count
        
        # Build conflict count per vertex
        conflicts = count_conflicts(colors)
        if conflicts == 0:
            return colors
        
        # Vertex conflict count
        vertex_conflicts = [0] * num_vertices
        for u, v in edges:
            if colors[u] == colors[v]:
                vertex_conflicts[u] += 1
                vertex_conflicts[v] += 1
        
        # Tabu tenure
        tabu = [[0] * k for _ in range(num_vertices)]
        iteration = 0
        best_conflicts = conflicts
        
        while conflicts > 0 and iteration < max_iter and time_remaining() > 1:
            iteration += 1
            
            # Find best move: change color of a conflicting vertex
            best_move_delta = float('inf')
            best_move_v = -1
            best_move_c = -1
            
            # Only consider vertices with conflicts
            conflict_vertices = [v for v in range(num_vertices) if vertex_conflicts[v] > 0]
            
            # Sample if too many
            if len(conflict_vertices) > 50:
                sampled = random.sample(conflict_vertices, 50)
            else:
                sampled = conflict_vertices
            
            for v in sampled:
                old_c = colors[v]
                for new_c in range(k):
                    if new_c == old_c:
                        continue
                    
                    # Calculate delta
                    delta = 0
                    for u in adj[v]:
                        if colors[u] == old_c:
                            delta -= 1  # resolved conflict
                        if colors[u] == new_c:
                            delta += 1  # new conflict
                    
                    # Check tabu
                    if tabu[v][new_c] > iteration:
                        # Aspiration: allow if it improves best
                        if conflicts + delta >= best_conflicts:
                            continue
                    
                    if delta < best_move_delta:
                        best_move_delta = delta
                        best_move_v = v
                        best_move_c = new_c
            
            if best_move_v == -1:
                break
            
            # Apply move
            v = best_move_v
            old_c = colors[v]
            new_c = best_move_c
            
            for u in adj[v]:
                if colors[u] == old_c:
                    vertex_conflicts[u] -= 1
                    vertex_conflicts[v] -= 1
                    conflicts -= 1
                if colors[u] == new_c:
                    vertex_conflicts[u] += 1
                    vertex_conflicts[v] += 1
                    conflicts += 1
            
            colors[v] = new_c
            tabu[v][old_c] = iteration + random.randint(1, 10 + int(0.6 * len(conflict_vertices)))
            
            if conflicts < best_conflicts:
                best_conflicts = conflicts
            
            if conflicts == 0:
                return colors
        
        return None
    
    # Try to reduce number of colors one by one
    target_k = best_num_colors - 1
    while target_k >= 1 and time_remaining() > 2:
        max_iter = min(500000, int(time_remaining() * 10000))
        result = try_k_coloring(target_k, max_iter=max_iter)
        if result is not None:
            best_colors = result
            best_num_colors = target_k
            if logger:
                logger.log(best_num_colors)
            target_k -= 1
        else:
            break
    
    # Also try random restarts for tabu search
    while time_remaining() > 2 and target_k >= 1:
        # Random restart with different initial coloring
        max_iter = min(200000, int(time_remaining() * 5000))
        result = try_k_coloring(target_k, max_iter=max_iter)
        if result is not None:
            best_colors = result
            best_num_colors = target_k
            if logger:
                logger.log(best_num_colors)
            target_k -= 1
        else:
            if time_remaining() < 5:
                break
    
    # Build output
    coloring = {str(v): best_colors[v] for v in range(num_vertices)}
    color_classes = defaultdict(list)
    for v in range(num_vertices):
        color_classes[str(best_colors[v])].append(v)
    
    solution = {
        "objective_value": best_num_colors,
        "coloring": coloring,
        "color_classes": dict(color_classes)
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve_graph_coloring(args.instance_path, args.solution_path, args.time_limit, args.log_path)