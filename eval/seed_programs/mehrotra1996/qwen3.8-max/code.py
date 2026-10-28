import json
import argparse
import time
import sys
from collections import defaultdict
import random

def dsatur_coloring(nodes, edges):
    """DSATUR greedy coloring algorithm."""
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    color = {}
    saturation = defaultdict(set)
    uncolored = set(nodes)
    degree = {v: len(adj[v]) for v in nodes}
    
    while uncolored:
        best_v = max(uncolored, key=lambda v: (len(saturation[v]), degree[v]))
        
        used_colors = saturation[best_v]
        c = 0
        while c in used_colors:
            c += 1
        
        color[best_v] = c
        uncolored.remove(best_v)
        
        for neighbor in adj[best_v]:
            if neighbor in uncolored:
                saturation[neighbor].add(c)
    
    return color

def greedy_coloring_with_order(nodes, edges, order):
    """Greedy coloring with a given vertex order."""
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    color = {}
    for v in order:
        neighbor_colors = set()
        for u in adj[v]:
            if u in color:
                neighbor_colors.add(color[u])
        c = 0
        while c in neighbor_colors:
            c += 1
        color[v] = c
    
    return color

def local_search_improve(color, adj, nodes, num_colors, time_limit):
    """Try to reduce number of colors using local search."""
    start_time = time.time()
    best_color = dict(color)
    best_num_colors = num_colors
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try to remove the highest color
        for target_color in range(best_num_colors - 1, -1, -1):
            if time.time() - start_time >= time_limit:
                break
            
            vertices_with_color = [v for v in nodes if best_color.get(v) == target_color]
            if not vertices_with_color:
                continue
            
            can_remove = True
            new_color = dict(best_color)
            
            # Sort vertices by number of constraints (neighbors with other colors)
            vertices_with_color.sort(key=lambda v: len(adj[v]))
            
            for v in vertices_with_color:
                neighbor_colors = set()
                for u in adj[v]:
                    if u in new_color:
                        neighbor_colors.add(new_color[u])
                
                found = False
                for c in range(best_num_colors):
                    if c != target_color and c not in neighbor_colors:
                        new_color[v] = c
                        found = True
                        break
                
                if not found:
                    can_remove = False
                    break
            
            if can_remove:
                best_color = new_color
                used = set(best_color.values())
                color_map = {}
                next_c = 0
                for c in sorted(used):
                    color_map[c] = next_c
                    next_c += 1
                best_color = {v: color_map[c] for v, c in best_color.items()}
                best_num_colors = next_c
                improved = True
                break
        
        if not improved:
            break
    
    return best_color, best_num_colors

def kempe_chain_search(color, adj, nodes, num_colors, time_limit):
    """Try Kempe chain moves to reduce colors."""
    start_time = time.time()
    best_color = dict(color)
    best_num_colors = num_colors
    
    random.seed(42)
    
    while time.time() - start_time < time_limit:
        improved = False
        
        for target_color in range(best_num_colors - 1, -1, -1):
            if time.time() - start_time >= time_limit:
                break
            
            vertices_with_color = [v for v in nodes if best_color.get(v) == target_color]
            if not vertices_with_color:
                continue
            
            # Try Kempe chain for each vertex with target color
            for v in vertices_with_color:
                if time.time() - start_time >= time_limit:
                    break
                
                # Try to find a color c such that we can swap v to c via Kempe chain
                neighbor_colors = set()
                for u in adj[v]:
                    if u in best_color:
                        neighbor_colors.add(best_color[u])
                
                # Direct assignment first
                for c in range(best_num_colors):
                    if c != target_color and c not in neighbor_colors:
                        best_color[v] = c
                        # Renumber
                        used = set(best_color.values())
                        color_map = {}
                        next_c = 0
                        for cc in sorted(used):
                            color_map[cc] = next_c
                            next_c += 1
                        best_color = {vv: color_map[cc] for vv, cc in best_color.items()}
                        best_num_colors = next_c
                        improved = True
                        break
                
                if improved:
                    break
                
                # Kempe chain: try to swap colors in a component
                for c in range(best_num_colors):
                    if c == target_color:
                        continue
                    # BFS from v in subgraph of colors {target_color, c}
                    visited = set()
                    queue = [v]
                    visited.add(v)
                    component = [v]
                    
                    while queue:
                        curr = queue.pop(0)
                        for u in adj[curr]:
                            if u not in visited and u in best_color:
                                if best_color[u] == target_color or best_color[u] == c:
                                    visited.add(u)
                                    queue.append(u)
                                    component.append(u)
                    
                    # Check if we can swap target_color and c in this component
                    # v has target_color, after swap it gets c
                    # Check that no neighbor of v outside component has color c
                    can_swap = True
                    for u in adj[v]:
                        if u not in visited and best_color.get(u) == c:
                            can_swap = False
                            break
                    
                    if can_swap:
                        for u in component:
                            if best_color[u] == target_color:
                                best_color[u] = c
                            elif best_color[u] == c:
                                best_color[u] = target_color
                        
                        used = set(best_color.values())
                        color_map = {}
                        next_c = 0
                        for cc in sorted(used):
                            color_map[cc] = next_c
                            next_c += 1
                        best_color = {vv: color_map[cc] for vv, cc in best_color.items()}
                        best_num_colors = next_c
                        improved = True
                        break
                
                if improved:
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return best_color, best_num_colors

def find_clique_lower_bound(nodes, edges, adj, time_limit):
    """Find a lower bound using greedy clique finding."""
    start_time = time.time()
    best_clique_size = 1
    
    # Try greedy clique from each vertex (limited)
    sorted_nodes = sorted(nodes, key=lambda v: -len(adj[v]))
    
    for start in sorted_nodes[:min(50, len(sorted_nodes))]:
        if time.time() - start_time > time_limit:
            break
        
        clique = [start]
        candidates = [v for v in adj[start]]
        
        while candidates:
            # Find vertex connected to all in clique
            valid = []
            for v in candidates:
                if all(v in adj[u] for u in clique):
                    valid.append(v)
            
            if not valid:
                break
            
            # Pick vertex with most connections to remaining candidates
            next_v = max(valid, key=lambda v: len(adj[v] & set(candidates)))
            clique.append(next_v)
            candidates = [v for v in candidates if v in adj[next_v]]
        
        best_clique_size = max(best_clique_size, len(clique))
    
    return best_clique_size

def solve_with_gurobi(nodes, edges, num_nodes, time_limit, upper_bound):
    """Solve graph coloring exactly using Gurobi."""
    import gurobipy as gp
    from gurobipy import GRB
    
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    K = upper_bound
    
    try:
        model = gp.Model("graph_coloring")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = time_limit
        model.Params.OutputFlag = 0
        
        x = {}
        y = {}
        
        for v in nodes:
            for k in range(K):
                x[v, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}_{k}")
        
        for k in range(K):
            y[k] = model.addVar(vtype=GRB.BINARY, name=f"y_{k}")
        
        model.update()
        
        for v in nodes:
            model.addConstr(gp.quicksum(x[v, k] for k in range(K)) == 1)
        
        for u, v in edges:
            for k in range(K):
                model.addConstr(x[u, k] + x[v, k] <= 1)
        
        for v in nodes:
            for k in range(K):
                model.addConstr(y[k] >= x[v, k])
        
        for k in range(K - 1):
            model.addConstr(y[k] >= y[k + 1])
        
        model.setObjective(gp.quicksum(y[k] for k in range(K)), GRB.MINIMIZE)
        
        model.optimize()
        
        if model.SolCount > 0:
            color = {}
            for v in nodes:
                for k in range(K):
                    if x[v, k].X > 0.5:
                        color[v] = k
                        break
            
            num_colors = int(round(sum(y[k].X for k in range(K))))
            is_optimal = (model.Status == GRB.OPTIMAL)
            return color, num_colors, is_optimal
    except Exception as e:
        pass
    
    return None, None, False

def build_solution(color_dict, nodes, num_c):
    vertex_colors = {str(v): color_dict[v] for v in nodes}
    color_classes = [[] for _ in range(num_c)]
    for v in nodes:
        color_classes[color_dict[v]].append(v)
    return {
        "objective_value": num_c,
        "vertex_colors": vertex_colors,
        "color_classes": color_classes
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    start_time = time.time()
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    graph = instance["graph"]
    num_nodes = graph["num_nodes"]
    num_edges = graph["num_edges"]
    nodes = graph["nodes"]
    edges = [tuple(e) for e in graph["edges"]]
    
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Handle trivial cases
    if num_nodes == 0:
        solution = {"objective_value": 0, "vertex_colors": {}, "color_classes": []}
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        if logger:
            logger.log_solution(0, solution)
        return
    
    if num_edges == 0:
        vertex_colors = {str(v): 0 for v in nodes}
        solution = {"objective_value": 1, "vertex_colors": vertex_colors, "color_classes": [list(nodes)]}
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        if logger:
            logger.log_solution(1, solution)
        return
    
    # Step 1: DSATUR
    color = dsatur_coloring(nodes, edges)
    num_colors = max(color.values()) + 1
    
    best_solution = build_solution(color, nodes, num_colors)
    if logger:
        logger.log_solution(num_colors, best_solution)
    
    # Try different orderings for potentially better greedy
    elapsed = time.time() - start_time
    if elapsed < args.time_limit * 0.2:
        # Try Welsh-Powell (descending degree order)
        degree_order = sorted(nodes, key=lambda v: -len(adj[v]))
        color2 = greedy_coloring_with_order(nodes, edges, degree_order)
        num_colors2 = max(color2.values()) + 1
        if num_colors2 < num_colors:
            num_colors = num_colors2
            color = color2
            best_solution = build_solution(color, nodes, num_colors)
            if logger:
                logger.log_solution(num_colors, best_solution)
        
        # Try smallest-last ordering
        remaining = set(nodes)
        order = []
        temp_adj = defaultdict(set)
        for u, v in edges:
            temp_adj[u].add(v)
            temp_adj[v].add(u)
        
        while remaining:
            v = min(remaining, key=lambda x: len(temp_adj[x] & remaining))
            order.append(v)
            remaining.remove(v)
        
        order.reverse()
        color3 = greedy_coloring_with_order(nodes, edges, order)
        num_colors3 = max(color3.values()) + 1
        if num_colors3 < num_colors:
            num_colors = num_colors3
            color = color3
            best_solution = build_solution(color, nodes, num_colors)
            if logger:
                logger.log_solution(num_colors, best_solution)
    
    # Step 2: Local search
    elapsed = time.time() - start_time
    remaining_time = args.time_limit - elapsed
    if remaining_time > 1:
        improved_color, improved_num = local_search_improve(
            color, adj, nodes, num_colors, min(remaining_time * 0.3, 10)
        )
        if improved_num < num_colors:
            num_colors = improved_num
            color = improved_color
            best_solution = build_solution(color, nodes, num_colors)
            if logger:
                logger.log_solution(num_colors, best_solution)
    
    # Step 3: Kempe chain search
    elapsed = time.time() - start_time
    remaining_time = args.time_limit - elapsed
    if remaining_time > 2:
        improved_color, improved_num = kempe_chain_search(
            color, adj, nodes, num_colors, min(remaining_time * 0.3, 15)
        )
        if improved_num < num_colors:
            num_colors = improved_num
            color = improved_color
            best_solution = build_solution(color, nodes, num_colors)
            if logger:
                logger.log_solution(num_colors, best_solution)
    
    # Step 4: Gurobi exact solve for small instances
    elapsed = time.time() - start_time
    remaining_time = args.time_limit - elapsed
    
    if remaining_time > 3 and num_nodes <= 300:
        gurobi_color, gurobi_num, is_optimal = solve_with_gurobi(
            nodes, edges, num_nodes, remaining_time - 1, num_colors
        )
        if gurobi_color is not None and gurobi_num < num_colors:
            num_colors = gurobi_num
            color = gurobi_color
            best_solution = build_solution(color, nodes, num_colors)
            if logger:
                logger.log_solution(num_colors, best_solution)
    
    # Write final solution
    with open(args.solution_path, 'w') as f:
        json.dump(best_solution, f)

if __name__ == "__main__":
    main()