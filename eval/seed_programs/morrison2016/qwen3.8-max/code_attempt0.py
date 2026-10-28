import json
import argparse
import time
from collections import defaultdict


def dsatur_coloring(num_vertices, adj):
    """DSATUR heuristic for graph coloring."""
    color = [-1] * num_vertices
    saturation = [0] * num_vertices
    degree = [len(adj[v]) for v in range(num_vertices)]
    
    num_colors = 0
    
    for _ in range(num_vertices):
        best_v = -1
        best_sat = -1
        best_deg = -1
        for v in range(num_vertices):
            if color[v] == -1:
                if (saturation[v] > best_sat or 
                    (saturation[v] == best_sat and degree[v] > best_deg)):
                    best_v = v
                    best_sat = saturation[v]
                    best_deg = degree[v]
        
        neighbor_colors = set()
        for u in adj[best_v]:
            if color[u] != -1:
                neighbor_colors.add(color[u])
        
        c = 0
        while c in neighbor_colors:
            c += 1
        
        color[best_v] = c
        if c >= num_colors:
            num_colors = c + 1
        
        for u in adj[best_v]:
            if color[u] == -1:
                colors_in_neigh = set()
                for w in adj[u]:
                    if color[w] != -1:
                        colors_in_neigh.add(color[w])
                saturation[u] = len(colors_in_neigh)
    
    return color, num_colors


def try_reduce_colors(coloring, num_colors, adj, num_vertices):
    """Try to reduce the number of colors by eliminating one color class."""
    color_classes = defaultdict(list)
    for v in range(num_vertices):
        color_classes[coloring[v]].append(v)
    
    # Try eliminating each color class, starting with smallest
    sorted_colors = sorted(color_classes.keys(), key=lambda c: len(color_classes[c]))
    
    for target_color in sorted_colors:
        new_coloring = list(coloring)
        success = True
        
        # Sort vertices by degree (higher degree first) for better recoloring
        vertices_to_recolor = sorted(color_classes[target_color], 
                                      key=lambda v: -len(adj[v]))
        
        for v in vertices_to_recolor:
            neighbor_colors = set()
            for u in adj[v]:
                if new_coloring[u] != target_color:
                    neighbor_colors.add(new_coloring[u])
            
            found = False
            for c in range(num_colors):
                if c != target_color and c not in neighbor_colors:
                    new_coloring[v] = c
                    found = True
                    break
            
            if not found:
                success = False
                break
        
        if success:
            used_colors = sorted(set(new_coloring))
            color_map = {c: i for i, c in enumerate(used_colors)}
            final_coloring = [color_map[c] for c in new_coloring]
            return final_coloring, len(used_colors)
    
    return list(coloring), num_colors


def local_search_improve(coloring, num_colors, adj, num_vertices, time_limit):
    """Try to reduce colors using iterative elimination with backtracking."""
    start = time.time()
    best_coloring = list(coloring)
    best_num_colors = num_colors
    
    while time.time() - start < time_limit:
        improved = False
        color_classes = defaultdict(list)
        for v in range(num_vertices):
            color_classes[best_coloring[v]].append(v)
        
        sorted_colors = sorted(color_classes.keys(), key=lambda c: len(color_classes[c]))
        
        for target_color in sorted_colors:
            if time.time() - start >= time_limit:
                break
            
            new_coloring = list(best_coloring)
            vertices_to_recolor = list(color_classes[target_color])
            
            # Try different orderings
            success = try_recolor_class(new_coloring, vertices_to_recolor, 
                                        target_color, best_num_colors, adj)
            
            if success:
                used_colors = sorted(set(new_coloring))
                color_map = {c: i for i, c in enumerate(used_colors)}
                best_coloring = [color_map[c] for c in new_coloring]
                best_num_colors = len(used_colors)
                improved = True
                break
        
        if not improved:
            break
    
    return best_coloring, best_num_colors


def try_recolor_class(coloring, vertices, target_color, num_colors, adj):
    """Try to recolor all vertices in a color class using backtracking."""
    if len(vertices) == 0:
        return True
    
    v = vertices[0]
    remaining = vertices[1:]
    
    neighbor_colors = set()
    for u in adj[v]:
        if coloring[u] != target_color:
            neighbor_colors.add(coloring[u])
    
    original_color = coloring[v]
    
    for c in range(num_colors):
        if c != target_color and c not in neighbor_colors:
            coloring[v] = c
            if try_recolor_class(coloring, remaining, target_color, num_colors, adj):
                return True
    
    coloring[v] = original_color
    return False


def format_solution(coloring, num_colors, num_vertices):
    """Format the solution according to the output schema."""
    color_classes = [[] for _ in range(num_colors)]
    for v in range(num_vertices):
        color_classes[coloring[v]].append(v)
    
    for cc in color_classes:
        cc.sort()
    
    coloring_dict = {}
    for v in range(num_vertices):
        coloring_dict[str(v)] = coloring[v]
    
    return {
        "objective_value": num_colors,
        "selected_sets": color_classes,
        "coloring": coloring_dict
    }


def solve_with_gurobi(num_vertices, edges, lower_bound, upper_bound, 
                      time_limit, logger, initial_coloring, initial_num_colors):
    """Use Gurobi MIP to solve graph coloring."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None, None
    
    K = upper_bound
    
    model = gp.Model("graph_coloring")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)
    
    # Variables
    x = {}
    for v in range(num_vertices):
        for k in range(K):
            x[v, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}_{k}")
    
    y = {}
    for k in range(K):
        y[k] = model.addVar(vtype=GRB.BINARY, name=f"y_{k}")
    
    model.update()
    
    # Objective: minimize number of colors used
    model.setObjective(gp.quicksum(y[k] for k in range(K)), GRB.MINIMIZE)
    
    # Each vertex gets exactly one color
    for v in range(num_vertices):
        model.addConstr(gp.quicksum(x[v, k] for k in range(K)) == 1, name=f"assign_{v}")
    
    # Adjacent vertices get different colors
    for u, v in edges:
        for k in range(K):
            model.addConstr(x[u, k] + x[v, k] <= 1, name=f"edge_{u}_{v}_{k}")
    
    # Color k can only be used if y[k] = 1
    for v in range(num_vertices):
        for k in range(K):
            model.addConstr(x[v, k] <= y[k], name=f"use_{v}_{k}")
    
    # Symmetry breaking: colors used in order
    for k in range(K - 1):
        model.addConstr(y[k] >= y[k + 1], name=f"sym_{k}")
    
    # Lower bound
    if lower_bound > 1:
        model.addConstr(gp.quicksum(y[k] for k in range(K)) >= lower_bound)
    
    # Warm start from heuristic solution
    if initial_coloring is not None:
        for v in range(num_vertices):
            for k in range(K):
                x[v, k].Start = 1.0 if initial_coloring[v] == k else 0.0
        for k in range(K):
            y[k].Start = 1.0 if k < initial_num_colors else 0.0
    
    best_obj = [float('inf')]
    best_coloring = [None]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_obj[0]:
                best_obj[0] = obj
                coloring = [0] * num_vertices
                for v in range(num_vertices):
                    for k in range(K):
                        if model.cbGetSolution(x[v, k]) > 0.5:
                            coloring[v] = k
                            break
                best_coloring[0] = coloring
                
                if logger:
                    sol_dict = format_solution(coloring, int(round(obj)), num_vertices)
                    logger.log_solution(int(round(obj)), sol_dict)
    
    model.optimize(callback)
    
    if best_coloring[0] is not None:
        return int(round(best_obj[0])), best_coloring[0]
    
    return None, None


def main():
    parser = argparse.ArgumentParser(description='Graph Coloring Solver')
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    start_time = time.time()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Load instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    num_vertices = instance['graph']['num_vertices']
    num_edges = instance['graph']['num_edges']
    edges = [(e[0], e[1]) for e in instance['graph']['edges']]
    lower_bound = instance['bounds']['clique_lower_bound']
    upper_bound = instance['bounds']['greedy_upper_bound']
    
    # Build adjacency
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Handle trivial cases
    if num_vertices == 0:
        solution = {"objective_value": 0, "selected_sets": [], "coloring": {}}
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        return
    
    if num_edges == 0:
        coloring = [0] * num_vertices
        solution = format_solution(coloring, 1, num_vertices)
        if logger:
            logger.log_solution(1, solution)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        return
    
    # DSATUR coloring
    best_coloring, best_num_colors = dsatur_coloring(num_vertices, adj)
    
    # Try to reduce colors iteratively
    improved = True
    while improved:
        new_coloring, new_num_colors = try_reduce_colors(
            best_coloring, best_num_colors, adj, num_vertices)
        if new_num_colors < best_num_colors:
            best_coloring = new_coloring
            best_num_colors = new_num_colors
        else:
            improved = False
    
    # Log initial solution
    sol_dict = format_solution(best_coloring, best_num_colors, num_vertices)
    if logger:
        logger.log_solution(best_num_colors, sol_dict)
    
    # If already at lower bound, we're optimal
    if best_num_colors == lower_bound:
        solution = format_solution(best_coloring, best_num_colors, num_vertices)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        return
    
    # Try local search improvement if time permits
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed
    
    if remaining > 2 and best_num_colors > lower_bound:
        ls_time = min(remaining * 0.2, 5.0)  # Use at most 20% of time or 5s for local search
        ls_coloring, ls_num_colors = local_search_improve(
            best_coloring, best_num_colors, adj, num_vertices, ls_time)
        if ls_num_colors < best_num_colors:
            best_coloring = ls_coloring
            best_num_colors = ls_num_colors
            sol_dict = format_solution(best_coloring, best_num_colors, num_vertices)
            if logger:
                logger.log_solution(best_num_colors, sol_dict)
    
    # Try MIP if time permits and problem is manageable
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed
    K = min(best_num_colors, upper_bound)
    
    if remaining > 2 and best_num_colors > lower_bound:
        # Check problem size for MIP feasibility
        num_vars = num_vertices * K + K
        num_constrs = num_vertices + len(edges) * K + num_vertices * K + K
        
        if num_vars <= 500000 and num_constrs <= 5000000:
            mip_time = remaining - 1
            if mip_time > 1:
                mip_obj, mip_coloring = solve_with_gurobi(
                    num_vertices, edges, lower_bound, K,
                    mip_time, logger, best_coloring, best_num_colors
                )
                if mip_obj is not None and mip_obj < best_num_colors:
                    best_coloring = mip_coloring
                    best_num_colors = mip_obj
    
    # Format and write output
    solution = format_solution(best_coloring, best_num_colors, num_vertices)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()