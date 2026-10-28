import argparse
import json
import time
import random
from collections import defaultdict

def solve_graph_coloring(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    start_time = time.time()
    
    # Read instance
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    graph = instance['graph']
    num_nodes = graph['num_nodes']
    nodes = graph['nodes']
    edges = graph['edges']
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Build adjacency list
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Handle trivial cases
    if num_nodes == 0:
        solution = {
            "objective_value": 0,
            "vertex_colors": {},
            "color_classes": []
        }
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
        return
    
    if len(edges) == 0:
        vertex_colors = {str(n): 0 for n in nodes}
        solution = {
            "objective_value": 1,
            "vertex_colors": vertex_colors,
            "color_classes": [nodes[:]]
        }
        if logger:
            logger.log(1)
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
        return
    
    # Greedy coloring with DSATUR (degree of saturation) heuristic
    def dsatur_coloring():
        color = {}
        saturation = {n: 0 for n in nodes}
        degree = {n: len(adj[n]) for n in nodes}
        neighbor_colors = {n: set() for n in nodes}
        
        uncolored = set(nodes)
        
        while uncolored:
            # Pick vertex with highest saturation, break ties by degree
            best = max(uncolored, key=lambda n: (saturation[n], degree[n]))
            
            # Find smallest available color
            used = neighbor_colors[best]
            c = 0
            while c in used:
                c += 1
            
            color[best] = c
            uncolored.remove(best)
            
            # Update saturation of neighbors
            for nb in adj[best]:
                if nb in uncolored:
                    if c not in neighbor_colors[nb]:
                        neighbor_colors[nb].add(c)
                        saturation[nb] += 1
            
        return color
    
    # Greedy coloring with a given vertex ordering
    def greedy_coloring(order):
        color = {}
        for v in order:
            used = set()
            for nb in adj[v]:
                if nb in color:
                    used.add(color[nb])
            c = 0
            while c in used:
                c += 1
            color[v] = c
        return color
    
    # Get initial solution with DSATUR
    best_coloring = dsatur_coloring()
    best_num_colors = max(best_coloring.values()) + 1
    
    if logger:
        logger.log(best_num_colors)
    
    # Try greedy with largest-first ordering
    order_lf = sorted(nodes, key=lambda n: len(adj[n]), reverse=True)
    col_lf = greedy_coloring(order_lf)
    nc_lf = max(col_lf.values()) + 1
    if nc_lf < best_num_colors:
        best_num_colors = nc_lf
        best_coloring = col_lf
        if logger:
            logger.log(best_num_colors)
    
    # Try random greedy colorings
    for _ in range(min(100, max(1, int((time_limit * 0.05) / max(num_nodes, 1) * 10000)))):
        if time.time() - start_time > time_limit * 0.1:
            break
        order = nodes[:]
        random.shuffle(order)
        col = greedy_coloring(order)
        nc = max(col.values()) + 1
        if nc < best_num_colors:
            best_num_colors = nc
            best_coloring = col
            if logger:
                logger.log(best_num_colors)
    
    # Now try to improve using local search (TabuCol-like approach)
    # Try to color with k-1 colors
    def try_color_with_k(k, max_iter, deadline):
        """Try to find a proper k-coloring using tabu search."""
        if k <= 0:
            return None
        
        # Initialize with a random assignment of k colors
        color = {}
        for v in nodes:
            color[v] = random.randint(0, k - 1)
        
        # Count conflicts
        conflicts = 0
        node_conflicts = defaultdict(int)  # number of conflicts per node
        for u, v in edges:
            if color[u] == color[v]:
                conflicts += 1
                node_conflicts[u] += 1
                node_conflicts[v] += 1
        
        if conflicts == 0:
            return color
        
        # Tabu tenure
        tabu_tenure = max(10, num_nodes // 10)
        tabu = {}  # (node, color) -> iteration when it becomes non-tabu
        
        best_conflicts = conflicts
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                return None
            
            if conflicts == 0:
                return color
            
            # Find conflicting nodes
            conflict_nodes = [v for v in nodes if node_conflicts[v] > 0]
            if not conflict_nodes:
                return color
            
            # Pick a random conflicting node
            v = random.choice(conflict_nodes)
            old_c = color[v]
            
            # Evaluate all possible color changes for v
            best_delta = float('inf')
            best_color = old_c
            best_moves = []
            
            for c in range(k):
                if c == old_c:
                    continue
                
                # Calculate delta in conflicts
                delta = 0
                for nb in adj[v]:
                    if color[nb] == old_c:
                        delta -= 1  # removing a conflict
                    if color[nb] == c:
                        delta += 1  # adding a conflict
                
                # Check tabu
                is_tabu = tabu.get((v, c), 0) > iteration
                
                # Aspiration: accept if it leads to new best
                if is_tabu and conflicts + delta >= best_conflicts:
                    continue
                
                if delta < best_delta:
                    best_delta = delta
                    best_moves = [(v, c)]
                elif delta == best_delta:
                    best_moves.append((v, c))
            
            if not best_moves:
                continue
            
            v, new_c = random.choice(best_moves)
            old_c = color[v]
            
            # Apply move
            for nb in adj[v]:
                if color[nb] == old_c:
                    node_conflicts[nb] -= 1
                    node_conflicts[v] -= 1
                    conflicts -= 1
                if color[nb] == new_c:
                    node_conflicts[nb] += 1
                    node_conflicts[v] += 1
                    conflicts += 1
            
            color[v] = new_c
            tabu[(v, old_c)] = iteration + tabu_tenure + random.randint(0, 5)
            
            if conflicts < best_conflicts:
                best_conflicts = conflicts
            
            if conflicts == 0:
                return color
        
        return None
    
    # Iteratively try to reduce number of colors
    deadline = start_time + time_limit * 0.95
    k = best_num_colors - 1
    
    while k >= 1 and time.time() < deadline:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        
        max_iter = min(100000, max(10000, num_nodes * 1000))
        
        # Try multiple restarts
        found = False
        attempts = 0
        while time.time() < deadline:
            attempts += 1
            result = try_color_with_k(k, max_iter, min(time.time() + remaining / 3, deadline))
            if result is not None:
                best_coloring = result
                best_num_colors = k
                if logger:
                    logger.log(best_num_colors)
                found = True
                break
            if attempts >= 5:
                break
            remaining = deadline - time.time()
            if remaining <= 0:
                break
        
        if not found:
            break
        k -= 1
    
    # Build output
    vertex_colors = {str(v): best_coloring[v] for v in nodes}
    
    color_classes_dict = defaultdict(list)
    for v in nodes:
        color_classes_dict[best_coloring[v]].append(v)
    
    # Remap colors to be contiguous from 0
    unique_colors = sorted(color_classes_dict.keys())
    remap = {c: i for i, c in enumerate(unique_colors)}
    
    vertex_colors = {str(v): remap[best_coloring[v]] for v in nodes}
    color_classes = [[] for _ in range(len(unique_colors))]
    for v in nodes:
        color_classes[remap[best_coloring[v]]].append(v)
    
    solution = {
        "objective_value": best_num_colors,
        "vertex_colors": vertex_colors,
        "color_classes": color_classes
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
    
    solve_graph_coloring(args.instance_path, args.solution_path, args.time_limit, args.log_path)