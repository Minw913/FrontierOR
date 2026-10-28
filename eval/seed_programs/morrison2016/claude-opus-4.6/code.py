import argparse
import json
import time
import sys
from collections import defaultdict

def solve_graph_coloring(instance_path, solution_path, time_limit, log_path=None):
    from solution_logger import SolutionLogger
    
    start_time = time.time()
    
    # Read instance
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    graph = instance['graph']
    num_vertices = graph['num_vertices']
    num_edges = graph['num_edges']
    edges = graph['edges']
    bounds = instance.get('bounds', {})
    
    clique_lb = bounds.get('clique_lower_bound', 1)
    greedy_ub = bounds.get('greedy_upper_bound', num_vertices)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Build adjacency list and adjacency set
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Handle trivial cases
    if num_edges == 0:
        # No edges, 1 color suffices (if there are vertices)
        if num_vertices == 0:
            result = {
                "objective_value": 0,
                "selected_sets": [],
                "coloring": {}
            }
        else:
            result = {
                "objective_value": 1,
                "selected_sets": [list(range(num_vertices))],
                "coloring": {str(v): 0 for v in range(num_vertices)}
            }
            if logger:
                logger.log(1)
        with open(solution_path, 'w') as f:
            json.dump(result, f)
        return
    
    # Check if graph is bipartite
    def is_bipartite():
        color_map = [-1] * num_vertices
        for start in range(num_vertices):
            if color_map[start] != -1:
                continue
            queue = [start]
            color_map[start] = 0
            while queue:
                node = queue.pop(0)
                for neighbor in adj[node]:
                    if color_map[neighbor] == -1:
                        color_map[neighbor] = 1 - color_map[node]
                        queue.append(neighbor)
                    elif color_map[neighbor] == color_map[node]:
                        return False, None
        return True, color_map
    
    bip, bip_colors = is_bipartite()
    if bip and clique_lb <= 2:
        # Check if we need 1 or 2 colors
        if num_edges == 0:
            num_colors = 1
        else:
            num_colors = 2
        if logger:
            logger.log(num_colors)
        if num_colors == 1:
            result = {
                "objective_value": 1,
                "selected_sets": [list(range(num_vertices))],
                "coloring": {str(v): 0 for v in range(num_vertices)}
            }
        else:
            sets = [[], []]
            coloring = {}
            for v in range(num_vertices):
                c = bip_colors[v]
                sets[c].append(v)
                coloring[str(v)] = c
            result = {
                "objective_value": 2,
                "selected_sets": [sorted(s) for s in sets],
                "coloring": coloring
            }
        with open(solution_path, 'w') as f:
            json.dump(result, f)
        return
    
    # DSatur greedy coloring
    def dsatur_coloring():
        colors = [-1] * num_vertices
        saturation = [0] * num_vertices
        degree = [len(adj[v]) for v in range(num_vertices)]
        colored_neighbors = [set() for _ in range(num_vertices)]
        num_colors_used = 0
        
        for _ in range(num_vertices):
            # Pick vertex with highest saturation, break ties by degree
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
            
            # Find smallest available color
            neighbor_colors = colored_neighbors[best_v]
            c = 0
            while c in neighbor_colors:
                c += 1
            
            colors[best_v] = c
            if c + 1 > num_colors_used:
                num_colors_used = c + 1
            
            for nb in adj[best_v]:
                if colors[nb] == -1:
                    if c not in colored_neighbors[nb]:
                        colored_neighbors[nb].add(c)
                        saturation[nb] += 1
        
        return colors, num_colors_used
    
    best_colors, best_num_colors = dsatur_coloring()
    best_num_colors = min(best_num_colors, greedy_ub)
    
    if logger:
        logger.log(best_num_colors)
    
    # If DSatur matches lower bound, we're optimal
    if best_num_colors <= clique_lb:
        pass  # Already optimal
    else:
        # Try to improve with more sophisticated methods
        # Use a greedy independent set extraction approach (iterated)
        def greedy_coloring_lis(order):
            """Color using given vertex order"""
            colors = [-1] * num_vertices
            num_used = 0
            for v in order:
                neighbor_colors = set()
                for nb in adj[v]:
                    if colors[nb] != -1:
                        neighbor_colors.add(colors[nb])
                c = 0
                while c in neighbor_colors:
                    c += 1
                colors[v] = c
                if c + 1 > num_used:
                    num_used = c + 1
            return colors, num_used
        
        # Try largest-first ordering
        order_lf = sorted(range(num_vertices), key=lambda v: len(adj[v]), reverse=True)
        c_lf, n_lf = greedy_coloring_lis(order_lf)
        if n_lf < best_num_colors:
            best_colors = c_lf
            best_num_colors = n_lf
            if logger:
                logger.log(best_num_colors)
        
        # Try smallest-last ordering
        def smallest_last_order():
            remaining = set(range(num_vertices))
            rem_degree = {v: len(adj[v]) for v in range(num_vertices)}
            order = []
            for _ in range(num_vertices):
                # Find vertex with smallest degree in remaining
                min_v = min(remaining, key=lambda v: rem_degree[v])
                order.append(min_v)
                remaining.remove(min_v)
                for nb in adj[min_v]:
                    if nb in remaining:
                        rem_degree[nb] -= 1
            order.reverse()
            return order
        
        if time.time() - start_time < time_limit * 0.3:
            order_sl = smallest_last_order()
            c_sl, n_sl = greedy_coloring_lis(order_sl)
            if n_sl < best_num_colors:
                best_colors = c_sl
                best_num_colors = n_sl
                if logger:
                    logger.log(best_num_colors)
        
        # Try random orderings with DSatur-like approach
        import random
        random.seed(42)
        
        # Iterated greedy with random perturbation
        while time.time() - start_time < time_limit * 0.5 and best_num_colors > clique_lb:
            order = list(range(num_vertices))
            random.shuffle(order)
            c_r, n_r = greedy_coloring_lis(order)
            if n_r < best_num_colors:
                best_colors = c_r
                best_num_colors = n_r
                if logger:
                    logger.log(best_num_colors)
        
        # Try TabuCol-like local search to reduce number of colors
        def tabu_search(target_k, max_iter=100000):
            """Try to find a proper coloring with target_k colors using tabu search"""
            elapsed = time.time() - start_time
            if elapsed >= time_limit * 0.95:
                return None
            
            # Initialize with a possibly improper coloring
            colors_local = [0] * num_vertices
            # Start from best coloring, merge two smallest color classes
            color_classes = defaultdict(list)
            for v in range(num_vertices):
                color_classes[best_colors[v]].append(v)
            
            # Reassign colors 0..target_k-1
            sorted_classes = sorted(color_classes.keys())
            mapping = {}
            for i, old_c in enumerate(sorted_classes):
                if i < target_k:
                    mapping[old_c] = i
                else:
                    mapping[old_c] = i % target_k
            
            for v in range(num_vertices):
                colors_local[v] = mapping[best_colors[v]]
            
            # Count conflicts
            def count_conflicts():
                count = 0
                for u, v in edges:
                    if colors_local[u] == colors_local[v]:
                        count += 1
                return count
            
            # Vertex conflict count
            vertex_conflicts = [0] * num_vertices
            for u, v in edges:
                if colors_local[u] == colors_local[v]:
                    vertex_conflicts[u] += 1
                    vertex_conflicts[v] += 1
            
            total_conflicts = sum(vertex_conflicts) // 2
            
            if total_conflicts == 0:
                return colors_local
            
            # Color-vertex conflict matrix: gamma[v][c] = number of neighbors of v with color c
            gamma = [[0] * target_k for _ in range(num_vertices)]
            for v in range(num_vertices):
                for nb in adj[v]:
                    gamma[v][colors_local[nb]] += 1
            
            tabu_tenure = [[-1] * target_k for _ in range(num_vertices)]
            tabu_base = int(0.6 * total_conflicts) + 5
            
            iteration = 0
            while iteration < max_iter and total_conflicts > 0:
                if time.time() - start_time >= time_limit * 0.95:
                    return None
                
                # Find best non-tabu move
                best_delta = float('inf')
                best_moves = []
                
                # Only consider conflicting vertices
                for v in range(num_vertices):
                    if vertex_conflicts[v] == 0:
                        continue
                    current_c = colors_local[v]
                    for c in range(target_k):
                        if c == current_c:
                            continue
                        # Delta = new conflicts for v with color c - current conflicts for v
                        delta = gamma[v][c] - gamma[v][current_c]
                        
                        # Check tabu
                        is_tabu = tabu_tenure[v][c] > iteration
                        
                        # Aspiration: accept if it leads to new best
                        if is_tabu and total_conflicts + delta >= total_conflicts:
                            # Only skip if truly tabu and not aspiration
                            if total_conflicts + delta > 0:
                                continue
                        
                        if delta < best_delta:
                            best_delta = delta
                            best_moves = [(v, c)]
                        elif delta == best_delta:
                            best_moves.append((v, c))
                
                if not best_moves:
                    # All moves are tabu, pick random conflicting vertex
                    conf_vertices = [v for v in range(num_vertices) if vertex_conflicts[v] > 0]
                    if not conf_vertices:
                        break
                    v = random.choice(conf_vertices)
                    current_c = colors_local[v]
                    c = random.randint(0, target_k - 1)
                    while c == current_c:
                        c = random.randint(0, target_k - 1)
                    best_moves = [(v, c)]
                    best_delta = gamma[v][c] - gamma[v][current_c]
                
                # Pick a random best move
                v, new_c = random.choice(best_moves)
                old_c = colors_local[v]
                
                # Apply move
                colors_local[v] = new_c
                tabu_tenure[v][old_c] = iteration + random.randint(1, tabu_base + 1)
                
                # Update gamma and vertex_conflicts
                for nb in adj[v]:
                    gamma[nb][old_c] -= 1
                    gamma[nb][new_c] += 1
                    
                    nb_c = colors_local[nb]
                    # Effect on neighbor's conflicts
                    if nb_c == old_c:
                        # Was a conflict, now removed
                        vertex_conflicts[nb] -= 1
                        vertex_conflicts[v] -= 1
                        total_conflicts -= 1
                    if nb_c == new_c:
                        # New conflict created
                        vertex_conflicts[nb] += 1
                        vertex_conflicts[v] += 1
                        total_conflicts += 1
                
                iteration += 1
            
            if total_conflicts == 0:
                return colors_local
            return None
        
        # Try to reduce colors one at a time
        target_k = best_num_colors - 1
        while target_k >= clique_lb and time.time() - start_time < time_limit * 0.95:
            remaining_time = time_limit - (time.time() - start_time)
            if remaining_time < 0.5:
                break
            
            # Scale max iterations based on problem size and remaining time
            max_iter = min(500000, max(10000, int(remaining_time * 5000)))
            
            result_colors = tabu_search(target_k, max_iter=max_iter)
            if result_colors is not None:
                best_colors = result_colors
                best_num_colors = target_k
                if logger:
                    logger.log(best_num_colors)
                target_k -= 1
            else:
                break
    
    # Build solution
    color_classes = defaultdict(list)
    coloring = {}
    for v in range(num_vertices):
        c = best_colors[v]
        color_classes[c].append(v)
        coloring[str(v)] = c
    
    # Renumber colors to be 0..k-1
    unique_colors = sorted(color_classes.keys())
    remap = {old: new for new, old in enumerate(unique_colors)}
    
    selected_sets = []
    new_coloring = {}
    for new_c, old_c in enumerate(unique_colors):
        selected_sets.append(sorted(color_classes[old_c]))
    for v in range(num_vertices):
        new_coloring[str(v)] = remap[best_colors[v]]
    
    result = {
        "objective_value": best_num_colors,
        "selected_sets": selected_sets,
        "coloring": new_coloring
    }
    
    with open(solution_path, 'w') as f:
        json.dump(result, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve_graph_coloring(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()