import json
import argparse
import time
import random
import sys
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Graph Coloring Solver")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    return parser.parse_args()

def load_instance(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data['graph']

def dsatur_coloring(n, adj_list):
    """DSATUR greedy coloring heuristic."""
    color = [-1] * n
    saturation = [0] * n
    degree = [len(adj_list[v]) for v in range(n)]
    neighbor_colors = [set() for _ in range(n)]
    uncolored = set(range(n))
    num_colors = 0
    
    while uncolored:
        best_v = max(uncolored, key=lambda v: (saturation[v], degree[v]))
        used_colors = neighbor_colors[best_v]
        c = 0
        while c in used_colors:
            c += 1
        color[best_v] = c
        if c >= num_colors:
            num_colors = c + 1
        uncolored.remove(best_v)
        for u in adj_list[best_v]:
            if color[u] == -1:
                neighbor_colors[u].add(c)
                saturation[u] = len(neighbor_colors[u])
    
    return color, num_colors

def greedy_coloring_with_order(n, adj_list, order):
    """Greedy coloring with a given vertex order."""
    color = [-1] * n
    num_colors = 0
    for v in order:
        used = set()
        for u in adj_list[v]:
            if color[u] != -1:
                used.add(color[u])
        c = 0
        while c in used:
            c += 1
        color[v] = c
        if c >= num_colors:
            num_colors = c + 1
    return color, num_colors

def find_clique_lower_bound(n, adj_list):
    """Find a clique to get a lower bound on chromatic number."""
    # Greedy clique finding
    best_clique_size = 1
    adj_set = [set(adj_list[v]) for v in range(n)]
    
    # Try starting from each vertex
    for start in range(min(n, 50)):
        clique = [start]
        candidates = [v for v in adj_set[start]]
        
        while candidates:
            # Pick candidate with most connections to current clique
            best_cand = max(candidates, key=lambda v: sum(1 for c in clique if v in adj_set[c]))
            # Check if it connects to all in clique
            if all(best_cand in adj_set[c] for c in clique):
                clique.append(best_cand)
                candidates = [v for v in candidates if v != best_cand and v in adj_set[best_cand]]
            else:
                candidates.remove(best_cand)
        
        best_clique_size = max(best_clique_size, len(clique))
    
    return best_clique_size

def try_reduce_one_color(n, adj_list, color, num_colors, target, max_attempts=200):
    """Try to reduce coloring from num_colors to target colors."""
    vertices_to_recolor = [v for v in range(n) if color[v] >= target]
    
    if not vertices_to_recolor:
        return color[:], True
    
    for attempt in range(max_attempts):
        new_color = color[:]
        # Mark vertices to recolor
        for v in vertices_to_recolor:
            new_color[v] = -1
        
        # Sort by most constrained (fewest available colors)
        def get_available(v):
            avail = set(range(target))
            for u in adj_list[v]:
                if new_color[u] != -1 and new_color[u] < target:
                    avail.discard(new_color[u])
            return avail
        
        # Order vertices by number of available colors (ascending)
        order = sorted(vertices_to_recolor, key=lambda v: len(get_available(v)))
        
        success = True
        for v in order:
            avail = get_available(v)
            if avail:
                # Choose color that minimizes future conflicts
                if attempt == 0:
                    new_color[v] = min(avail)
                else:
                    new_color[v] = random.choice(list(avail))
            else:
                success = False
                break
        
        if success:
            return new_color, True
    
    return color, False

def local_search_improve(n, adj_list, color, num_colors, time_limit, start_time):
    """Iteratively try to reduce the number of colors."""
    best_color = color[:]
    best_num = num_colors
    
    while best_num > 1:
        if time.time() - start_time > time_limit * 0.85:
            break
        
        target = best_num - 1
        new_color, success = try_reduce_one_color(n, adj_list, best_color, best_num, target, max_attempts=300)
        
        if success:
            best_color = new_color
            best_num = target
        else:
            break
    
    return best_color, best_num

def solve_with_gurobi(n, adj_list, upper_bound, time_limit, start_time):
    """Try to solve exactly with Gurobi for small instances."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None, None
    
    remaining = time_limit - (time.time() - start_time)
    if remaining <= 1:
        return None, None
    
    K = upper_bound
    
    model = gp.Model("graph_coloring")
    model.setParam('OutputFlag', 0)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', max(1, remaining * 0.9))
    
    # Variables
    x = {}
    for v in range(n):
        for k in range(K):
            x[v, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}_{k}")
    
    y = {}
    for k in range(K):
        y[k] = model.addVar(vtype=GRB.BINARY, name=f"y_{k}")
    
    model.update()
    
    # Objective
    model.setObjective(gp.quicksum(y[k] for k in range(K)), GRB.MINIMIZE)
    
    # Each vertex gets exactly one color
    for v in range(n):
        model.addConstr(gp.quicksum(x[v, k] for k in range(K)) == 1, name=f"assign_{v}")
    
    # Adjacent vertices get different colors
    edges = set()
    for v in range(n):
        for u in adj_list[v]:
            if v < u:
                edges.add((v, u))
    
    for (u, v) in edges:
        for k in range(K):
            model.addConstr(x[u, k] + x[v, k] <= 1, name=f"edge_{u}_{v}_{k}")
    
    # Link x and y
    for v in range(n):
        for k in range(K):
            model.addConstr(x[v, k] <= y[k], name=f"link_{v}_{k}")
    
    # Symmetry breaking
    for k in range(1, K):
        model.addConstr(y[k] <= y[k-1], name=f"sym_{k}")
    
    model.optimize()
    
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
        num_colors = int(round(model.objVal))
        color = [0] * n
        for v in range(n):
            for k in range(K):
                if x[v, k].X > 0.5:
                    color[v] = k
                    break
        return color, num_colors
    
    return None, None

def build_solution(n, color, num_colors):
    """Build the solution dictionary."""
    coloring = {}
    color_classes = defaultdict(list)
    
    for v in range(n):
        coloring[str(v)] = color[v]
        color_classes[str(color[v])].append(v)
    
    solution = {
        "objective_value": num_colors,
        "coloring": coloring,
        "color_classes": dict(color_classes)
    }
    return solution

def main():
    args = parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    start_time = time.time()
    random.seed(42)
    
    # Load instance
    graph = load_instance(args.instance_path)
    n = graph['n']
    m = graph['m']
    adj_list = graph['adjacency_list']
    
    # Strategy 1: DSATUR coloring
    color_dsatur, num_dsatur = dsatur_coloring(n, adj_list)
    
    best_color = color_dsatur[:]
    best_num = num_dsatur
    
    # Log initial solution
    solution = build_solution(n, best_color, best_num)
    if logger:
        logger.log_solution(best_num, solution)
    
    # Strategy 2: Try multiple random orderings
    if time.time() - start_time < args.time_limit * 0.3:
        for _ in range(min(50, max(1, int(args.time_limit * 2)))):
            if time.time() - start_time > args.time_limit * 0.3:
                break
            order = list(range(n))
            random.shuffle(order)
            # Sort by degree (descending) with some randomness
            order.sort(key=lambda v: -len(adj_list[v]) + random.randint(-2, 2))
            color_rand, num_rand = greedy_coloring_with_order(n, adj_list, order)
            if num_rand < best_num:
                best_color = color_rand[:]
                best_num = num_rand
                solution = build_solution(n, best_color, best_num)
                if logger:
                    logger.log_solution(best_num, solution)
    
    # Strategy 3: Local search to reduce colors
    if time.time() - start_time < args.time_limit * 0.6:
        improved_color, improved_num = local_search_improve(
            n, adj_list, best_color, best_num, args.time_limit, start_time
        )
        if improved_num < best_num:
            best_color = improved_color[:]
            best_num = improved_num
            solution = build_solution(n, best_color, best_num)
            if logger:
                logger.log_solution(best_num, solution)
    
    # Strategy 4: Try Gurobi for small instances
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed
    
    # Use Gurobi if instance is small enough and we have time
    use_gurobi = False
    if n <= 80 and remaining > 5:
        use_gurobi = True
    elif n <= 150 and best_num <= 15 and remaining > 10:
        use_gurobi = True
    elif n <= 50 and remaining > 3:
        use_gurobi = True
    
    if use_gurobi:
        gurobi_color, gurobi_num = solve_with_gurobi(
            n, adj_list, best_num, args.time_limit, start_time
        )
        if gurobi_color is not None and gurobi_num < best_num:
            best_color = gurobi_color[:]
            best_num = gurobi_num
            solution = build_solution(n, best_color, best_num)
            if logger:
                logger.log_solution(best_num, solution)
    
    # Strategy 5: Additional random restarts with local search if time permits
    if time.time() - start_time < args.time_limit * 0.8:
        for _ in range(20):
            if time.time() - start_time > args.time_limit * 0.8:
                break
            
            # Random greedy
            order = list(range(n))
            random.shuffle(order)
            color_r, num_r = greedy_coloring_with_order(n, adj_list, order)
            
            if num_r < best_num:
                best_color = color_r[:]
                best_num = num_r
                solution = build_solution(n, best_color, best_num)
                if logger:
                    logger.log_solution(best_num, solution)
            
            # Try to reduce
            if num_r <= best_num + 2:
                imp_color, imp_num = local_search_improve(
                    n, adj_list, color_r, num_r, args.time_limit, start_time
                )
                if imp_num < best_num:
                    best_color = imp_color[:]
                    best_num = imp_num
                    solution = build_solution(n, best_color, best_num)
                    if logger:
                        logger.log_solution(best_num, solution)
    
    # Final solution
    solution = build_solution(n, best_color, best_num)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()