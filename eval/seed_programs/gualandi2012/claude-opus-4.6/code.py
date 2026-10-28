import argparse
import json
import time
import sys
from collections import defaultdict
import random

def solve_graph_coloring(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    graph = instance['graph']
    num_vertices = graph['num_vertices']
    num_edges = graph['num_edges']
    edges = graph['edges']
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    if num_vertices == 0:
        solution = {"objective_value": 0, "coloring": {}, "color_classes": {}}
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
        return
    
    if num_edges == 0:
        coloring = {str(v): 0 for v in range(num_vertices)}
        color_classes = {"0": list(range(num_vertices))}
        if logger:
            logger.log(1)
        solution = {"objective_value": 1, "coloring": coloring, "color_classes": color_classes}
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
        return
    
    def validate_coloring(colors):
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        return True
    
    def normalize_coloring(colors):
        used = sorted(set(colors))
        mapping = {c: i for i, c in enumerate(used)}
        return [mapping[c] for c in colors], len(used)
    
    def time_remaining():
        return time_limit - (time.time() - start_time)
    
    # DSATUR heuristic
    def dsatur_coloring():
        colors = [-1] * num_vertices
        adj_colors = [set() for _ in range(num_vertices)]
        degree = [len(adj[v]) for v in range(num_vertices)]
        saturation = [0] * num_vertices
        num_colors_used = 0
        
        uncolored = set(range(num_vertices))
        
        for _ in range(num_vertices):
            best_v = max(uncolored, key=lambda v: (saturation[v], degree[v]))
            
            c = 0
            while c in adj_colors[best_v]:
                c += 1
            colors[best_v] = c
            if c + 1 > num_colors_used:
                num_colors_used = c + 1
            uncolored.remove(best_v)
            
            for u in adj[best_v]:
                if colors[u] == -1:
                    if c not in adj_colors[u]:
                        adj_colors[u].add(c)
                        saturation[u] += 1
        
        return colors, num_colors_used
    
    best_colors, best_num_colors = dsatur_coloring()
    assert validate_coloring(best_colors), "DSATUR produced invalid coloring"
    if logger:
        logger.log(best_num_colors)
    
    # Greedy with largest-first ordering
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
    
    order_lf = sorted(range(num_vertices), key=lambda v: len(adj[v]), reverse=True)
    cols, nc = greedy_color_with_order(order_lf)
    if nc < best_num_colors and validate_coloring(cols):
        best_colors, best_num_colors = cols, nc
        if logger:
            logger.log(best_num_colors)
    
    # Try random orderings
    random.seed(42)
    for _ in range(50):
        if time_remaining() < 2:
            break
        order = list(range(num_vertices))
        random.shuffle(order)
        cols, nc = greedy_color_with_order(order)
        if nc < best_num_colors and validate_coloring(cols):
            best_colors, best_num_colors = cols, nc
            if logger:
                logger.log(best_num_colors)
    
    # Try to reduce colors by reassigning smallest color class
    def try_reduce_colors(colors, num_colors):
        current_colors = list(colors)
        current_num = num_colors
        improved = True
        
        while improved and time_remaining() > 1:
            improved = False
            color_counts = defaultdict(list)
            for v in range(num_vertices):
                color_counts[current_colors[v]].append(v)
            
            sorted_classes = sorted(color_counts.items(), key=lambda x: len(x[1]))
            
            for target_color, vertices_in_class in sorted_classes:
                if time_remaining() < 1:
                    break
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
                    test_colors = list(current_colors)
                    for v, c in new_assignments.items():
                        test_colors[v] = c
                    if validate_coloring(test_colors):
                        test_colors, test_num = normalize_coloring(test_colors)
                        current_colors = test_colors
                        current_num = test_num
                        improved = True
                        break
        
        return current_colors, current_num
    
    new_colors, new_num = try_reduce_colors(best_colors, best_num_colors)
    if new_num < best_num_colors and validate_coloring(new_colors):
        best_colors, best_num_colors = new_colors, new_num
        if logger:
            logger.log(best_num_colors)
    
    # Tabu search to find k-coloring
    def try_k_coloring(k, max_iter=100000):
        if k >= best_num_colors:
            return None
        
        # Initialize by mapping current best coloring to k colors
        colors = [best_colors[v] % k for v in range(num_vertices)]
        
        conflicts = 0
        for u, v in edges:
            if colors[u] == colors[v]:
                conflicts += 1
        
        if conflicts == 0:
            return colors
        
        # Recompute vertex_conflicts from scratch
        vertex_conflicts = [0] * num_vertices
        for u, v in edges:
            if colors[u] == colors[v]:
                vertex_conflicts[u] += 1
                vertex_conflicts[v] += 1
        
        tabu = [[0] * k for _ in range(num_vertices)]
        best_conflicts = conflicts
        
        for iteration in range(1, max_iter + 1):
            if time_remaining() < 1:
                break
            
            # Periodically recompute conflicts to avoid drift
            if iteration % 10000 == 0:
                conflicts = 0
                vertex_conflicts = [0] * num_vertices
                for u, v in edges:
                    if colors[u] == colors[v]:
                        conflicts += 1
                        vertex_conflicts[u] += 1
                        vertex_conflicts[v] += 1
            
            conflict_vertices = [v for v in range(num_vertices) if vertex_conflicts[v] > 0]
            if not conflict_vertices:
                break
            
            if len(conflict_vertices) > 50:
                sampled = random.sample(conflict_vertices, 50)
            else:
                sampled = conflict_vertices
            
            best_move_delta = float('inf')
            best_move_v = -1
            best_move_c = -1
            
            for v in sampled:
                old_c = colors[v]
                for new_c in range(k):
                    if new_c == old_c:
                        continue
                    delta = 0
                    for u in adj[v]:
                        if colors[u] == old_c:
                            delta -= 1
                        if colors[u] == new_c:
                            delta += 1
                    
                    if tabu[v][new_c] > iteration:
                        if conflicts + delta >= best_conflicts:
                            continue
                    
                    if delta < best_move_delta:
                        best_move_delta = delta
                        best_move_v = v
                        best_move_c = new_c
            
            if best_move_v == -1:
                v = random.choice(conflict_vertices)
                old_c = colors[v]
                candidates = [c for c in range(k) if c != old_c]
                new_c = random.choice(candidates)
                best_move_v = v
                best_move_c = new_c
                best_move_delta = 0
                for u in adj[v]:
                    if colors[u] == old_c:
                        best_move_delta -= 1
                    if colors[u] == new_c:
                        best_move_delta += 1
            
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
            tabu_tenure = random.randint(1, max(10, int(0.6 * len(conflict_vertices))))
            tabu[v][old_c] = iteration + tabu_tenure
            
            if conflicts < best_conflicts:
                best_conflicts = conflicts
            
            if conflicts == 0:
                # Final verification
                if validate_coloring(colors):
                    return colors
                else:
                    # Recompute
                    conflicts = 0
                    vertex_conflicts = [0] * num_vertices
                    for uu, vv in edges:
                        if colors[uu] == colors[vv]:
                            conflicts += 1
                            vertex_conflicts[uu] += 1
                            vertex_conflicts[vv] += 1
                    if conflicts == 0:
                        return colors
        
        return None
    
    target_k = best_num_colors - 1
    while target_k >= 1 and time_remaining() > 2:
        max_iter = min(1000000, max(10000, int(time_remaining() * 5000)))
        result = try_k_coloring(target_k, max_iter=max_iter)
        if result is not None and validate_coloring(result):
            result, target_k_actual = normalize_coloring(result)
            best_colors = result
            best_num_colors = target_k_actual
            if logger:
                logger.log(best_num_colors)
            target_k = best_num_colors - 1
        else:
            break
    
    # Final strict validation
    assert validate_coloring(best_colors), "Final coloring is invalid!"
    best_colors, best_num_colors = normalize_coloring(best_colors)
    
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