import json
import argparse
import time
import sys
import random
from collections import defaultdict
import heapq

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        data = json.load(f)
    return data

def build_adjacency(num_nodes, edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def dsatur_coloring(num_nodes, edges):
    """DSATUR greedy coloring algorithm."""
    adj = build_adjacency(num_nodes, edges)
    
    color = [-1] * num_nodes
    saturation = [0] * num_nodes
    degree = [len(adj[v]) for v in range(num_nodes)]
    uncolored = set(range(num_nodes))
    
    # Use a heap for efficiency: (-saturation, -degree, vertex)
    heap = [(-saturation[v], -degree[v], v) for v in range(num_nodes)]
    heapq.heapify(heap)
    
    while uncolored:
        # Pop vertices until we find one that's still uncolored with correct priority
        while heap:
            neg_sat, neg_deg, v = heapq.heappop(heap)
            if v in uncolored and -neg_sat == saturation[v]:
                break
        else:
            break
        
        # Find the smallest color not used by neighbors
        neighbor_colors = set()
        for u in adj[v]:
            if color[u] != -1:
                neighbor_colors.add(color[u])
        
        c = 0
        while c in neighbor_colors:
            c += 1
        
        color[v] = c
        uncolored.remove(v)
        
        # Update saturation of uncolored neighbors
        for u in adj[v]:
            if color[u] == -1:
                old_sat = saturation[u]
                saturation[u] = len(set(color[w] for w in adj[u] if color[w] != -1))
                if saturation[u] != old_sat:
                    heapq.heappush(heap, (-saturation[u], -degree[u], u))
    
    num_colors = max(color) + 1 if color else 0
    return color, num_colors

def welsh_powell_coloring(num_nodes, edges):
    """Welsh-Powell greedy coloring (order by decreasing degree)."""
    adj = build_adjacency(num_nodes, edges)
    
    # Sort vertices by degree (descending)
    vertices = sorted(range(num_nodes), key=lambda v: -len(adj[v]))
    
    color = [-1] * num_nodes
    for v in vertices:
        neighbor_colors = set(color[u] for u in adj[v] if color[u] != -1)
        c = 0
        while c in neighbor_colors:
            c += 1
        color[v] = c
    
    num_colors = max(color) + 1 if color else 0
    return color, num_colors

def color_elimination(num_nodes, edges, coloring, num_colors, adj, time_limit, start_time, logger=None):
    """Try to eliminate colors one by one, starting from the highest."""
    current_coloring = coloring[:]
    current_num_colors = num_colors
    
    improved = True
    while improved and time.time() - start_time < time_limit - 1:
        improved = False
        
        # Try to eliminate each color from highest to lowest
        for target_color in range(current_num_colors - 1, 0, -1):
            if time.time() - start_time >= time_limit - 1:
                break
            
            # Get vertices with target_color
            vertices_with_color = [v for v in range(num_nodes) if current_coloring[v] == target_color]
            
            if not vertices_with_color:
                continue
            
            # Try to reassign each vertex to a lower color
            can_eliminate = True
            new_assignments = {}
            temp_coloring = current_coloring[:]
            
            # Sort vertices by number of available colors (most constrained first)
            vertex_options = []
            for v in vertices_with_color:
                neighbor_colors = set(temp_coloring[u] for u in adj[v] if temp_coloring[u] != target_color)
                available = [c for c in range(target_color) if c not in neighbor_colors]
                vertex_options.append((len(available), v, available))
            
            vertex_options.sort()
            
            for _, v, available in vertex_options:
                # Recompute available colors considering previous assignments
                neighbor_colors = set(temp_coloring[u] for u in adj[v] if temp_coloring[u] != target_color)
                available = [c for c in range(target_color) if c not in neighbor_colors]
                
                if not available:
                    can_eliminate = False
                    break
                
                # Choose color that appears least among neighbors' neighbors (heuristic)
                best_c = min(available, key=lambda c: sum(1 for u in adj[v] if temp_coloring[u] == c))
                temp_coloring[v] = best_c
            
            if can_eliminate:
                current_coloring = temp_coloring
                current_num_colors = target_color
                improved = True
                
                if logger:
                    solution_dict = {
                        "objective_value": float(current_num_colors),
                        "coloring": {str(v): current_coloring[v] for v in range(num_nodes)}
                    }
                    logger.log_solution(float(current_num_colors), solution_dict)
    
    return current_coloring, current_num_colors

def local_search_color_reduction(num_nodes, edges, coloring, num_colors, adj, time_limit, start_time, logger=None):
    """More aggressive local search with randomization."""
    current_coloring = coloring[:]
    current_num_colors = num_colors
    
    random.seed(42)
    iteration = 0
    
    while time.time() - start_time < time_limit - 1:
        iteration += 1
        used_colors = sorted(set(current_coloring))
        
        if len(used_colors) <= 1:
            break
        
        # Try to eliminate the highest color
        target = used_colors[-1]
        vertices_with_target = [v for v in range(num_nodes) if current_coloring[v] == target]
        
        # Multiple random orderings
        num_attempts = min(20, max(1, int((time_limit - (time.time() - start_time)) * 2)))
        
        success = False
        for attempt in range(num_attempts):
            if time.time() - start_time >= time_limit - 1:
                break
            
            shuffled = vertices_with_target[:]
            if attempt > 0:
                random.shuffle(shuffled)
            
            temp_coloring = current_coloring[:]
            feasible = True
            
            for v in shuffled:
                neighbor_colors = set(temp_coloring[u] for u in adj[v])
                available = [c for c in range(target) if c not in neighbor_colors]
                
                if not available:
                    feasible = False
                    break
                
                # Pick color that minimizes saturation increase for neighbors
                best_c = min(available, key=lambda c: sum(
                    1 for u in adj[v] if temp_coloring[u] == -1 or temp_coloring[u] == target
                ))
                temp_coloring[v] = best_c
            
            if feasible:
                current_coloring = temp_coloring
                current_num_colors = target
                success = True
                
                if logger:
                    solution_dict = {
                        "objective_value": float(current_num_colors),
                        "coloring": {str(v): current_coloring[v] for v in range(num_nodes)}
                    }
                    logger.log_solution(float(current_num_colors), solution_dict)
                break
        
        if not success:
            # Try eliminating a different color
            tried_all = True
            for target in reversed(used_colors[1:]):
                if time.time() - start_time >= time_limit - 1:
                    break
                
                vertices_with_target = [v for v in range(num_nodes) if current_coloring[v] == target]
                
                for attempt in range(5):
                    shuffled = vertices_with_target[:]
                    if attempt > 0:
                        random.shuffle(shuffled)
                    
                    temp_coloring = current_coloring[:]
                    feasible = True
                    
                    for v in shuffled:
                        neighbor_colors = set(temp_coloring[u] for u in adj[v])
                        # Can use any color except target
                        available = [c for c in range(current_num_colors) if c != target and c not in neighbor_colors]
                        
                        if not available:
                            feasible = False
                            break
                        
                        temp_coloring[v] = available[0]
                    
                    if feasible:
                        current_coloring = temp_coloring
                        # Recompute num_colors
                        current_num_colors = max(current_coloring) + 1
                        
                        if logger:
                            solution_dict = {
                                "objective_value": float(current_num_colors),
                                "coloring": {str(v): current_coloring[v] for v in range(num_nodes)}
                            }
                            logger.log_solution(float(current_num_colors), solution_dict)
                        
                        tried_all = False
                        break
                
                if not tried_all:
                    break
            
            if tried_all:
                break
    
    return current_coloring, current_num_colors

def solve_with_gurobi(num_nodes, edges, adj, time_limit, start_time, upper_bound, logger=None):
    """Try to solve exactly with Gurobi."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None, None
    
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time < 3:
        return None, None
    
    K = upper_bound
    
    # For very large instances, skip MIP
    if num_nodes > 1000 or num_nodes * K > 100000:
        return None, None
    
    try:
        model = gp.Model("graph_coloring")
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', remaining_time - 1)
        model.setParam('OutputFlag', 0)
        
        # Variables
        x = {}
        for v in range(num_nodes):
            for c in range(K):
                x[v, c] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}_{c}")
        
        y = {}
        for c in range(K):
            y[c] = model.addVar(vtype=GRB.BINARY, name=f"y_{c}")
        
        model.update()
        
        # Objective
        model.setObjective(gp.quicksum(y[c] for c in range(K)), GRB.MINIMIZE)
        
        # Each vertex gets exactly one color
        for v in range(num_nodes):
            model.addConstr(gp.quicksum(x[v, c] for c in range(K)) == 1)
        
        # Adjacent vertices can't share color
        for u, v in edges:
            for c in range(K):
                model.addConstr(x[u, c] + x[v, c] <= 1)
        
        # Linking constraints
        for v in range(num_nodes):
            for c in range(K):
                model.addConstr(x[v, c] <= y[c])
        
        # Symmetry breaking
        for c in range(1, K):
            model.addConstr(y[c] <= y[c-1])
        
        model.optimize()
        
        if model.status == GRB.OPTIMAL or (model.status == GRB.TIME_LIMIT and model.SolCount > 0):
            obj = int(round(model.ObjVal))
            coloring_result = [0] * num_nodes
            for v in range(num_nodes):
                for c in range(K):
                    if x[v, c].X > 0.5:
                        coloring_result[v] = c
                        break
            return coloring_result, obj
    except Exception as e:
        pass
    
    return None, None

def solve_with_gurobi_clique(num_nodes, edges, adj, time_limit, start_time, upper_bound, logger=None):
    """MIP formulation using clique-based lower bound for better performance."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None, None
    
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time < 3:
        return None, None
    
    K = upper_bound
    
    if num_nodes > 500 or num_nodes * K > 50000:
        return None, None
    
    try:
        model = gp.Model("graph_coloring_clique")
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', remaining_time - 1)
        model.setParam('OutputFlag', 0)
        
        # Variables: x[v,c] = 1 if vertex v gets color c
        x = {}
        for v in range(num_nodes):
            for c in range(K):
                x[v, c] = model.addVar(vtype=GRB.BINARY)
        
        y = {}
        for c in range(K):
            y[c] = model.addVar(vtype=GRB.BINARY)
        
        model.update()
        
        model.setObjective(gp.quicksum(y[c] for c in range(K)), GRB.MINIMIZE)
        
        for v in range(num_nodes):
            model.addConstr(gp.quicksum(x[v, c] for c in range(K)) == 1)
        
        for u, v in edges:
            for c in range(K):
                model.addConstr(x[u, c] + x[v, c] <= 1)
        
        for v in range(num_nodes):
            for c in range(K):
                model.addConstr(x[v, c] <= y[c])
        
        for c in range(1, K):
            model.addConstr(y[c] <= y[c-1])
        
        model.optimize()
        
        if model.status == GRB.OPTIMAL or (model.status == GRB.TIME_LIMIT and model.SolCount > 0):
            obj = int(round(model.ObjVal))
            coloring_result = [0] * num_nodes
            for v in range(num_nodes):
                for c in range(K):
                    if x[v, c].X > 0.5:
                        coloring_result[v] = c
                        break
            return coloring_result, obj
    except:
        pass
    
    return None, None

def find_max_clique_greedy(num_nodes, adj):
    """Find a greedy maximal clique for lower bound."""
    # Simple greedy clique finding
    best_clique = []
    
    for start in range(num_nodes):
        clique = [start]
        candidates = sorted(adj[start], key=lambda v: -len(adj[v]))
        
        for v in candidates:
            if all(v in adj[u] for u in clique):
                clique.append(v)
        
        if len(clique) > len(best_clique):
            best_clique = clique
    
    return best_clique

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    start_time = time.time()
    
    # Read instance
    data = read_instance(args.instance_path)
    num_nodes = data['num_nodes']
    num_edges = data['num_edges']
    edges = data['edges']
    
    adj = build_adjacency(num_nodes, edges)
    
    # Handle trivial cases
    if num_nodes == 0:
        solution_dict = {"objective_value": 0.0, "coloring": {}}
        with open(args.solution_path, 'w') as f:
            json.dump(solution_dict, f)
        return
    
    if num_edges == 0:
        coloring = [0] * num_nodes
        solution_dict = {
            "objective_value": 1.0,
            "coloring": {str(v): 0 for v in range(num_nodes)}
        }
        if logger:
            logger.log_solution(1.0, solution_dict)
        with open(args.solution_path, 'w') as f:
            json.dump(solution_dict, f)
        return
    
    # Step 1: Get initial solution with DSATUR
    coloring, num_colors = dsatur_coloring(num_nodes, edges)
    
    # Also try Welsh-Powell and take the better one
    wp_coloring, wp_num_colors = welsh_powell_coloring(num_nodes, edges)
    if wp_num_colors < num_colors:
        coloring = wp_coloring
        num_colors = wp_num_colors
    
    solution_dict = {
        "objective_value": float(num_colors),
        "coloring": {str(v): coloring[v] for v in range(num_nodes)}
    }
    
    if logger:
        logger.log_solution(float(num_colors), solution_dict)
    
    # Step 2: Color elimination heuristic
    if time.time() - start_time < args.time_limit - 2:
        coloring, num_colors = color_elimination(
            num_nodes, edges, coloring, num_colors, adj, args.time_limit, start_time, logger
        )
        
        solution_dict = {
            "objective_value": float(num_colors),
            "coloring": {str(v): coloring[v] for v in range(num_nodes)}
        }
    
    # Step 3: Try Gurobi MIP for small/medium instances
    if time.time() - start_time < args.time_limit - 3:
        gurobi_coloring, gurobi_obj = solve_with_gurobi(
            num_nodes, edges, adj, args.time_limit, start_time, num_colors, logger
        )
        if gurobi_coloring is not None and gurobi_obj < num_colors:
            coloring = gurobi_coloring
            num_colors = gurobi_obj
            
            solution_dict = {
                "objective_value": float(num_colors),
                "coloring": {str(v): coloring[v] for v in range(num_nodes)}
            }
            
            if logger:
                logger.log_solution(float(num_colors), solution_dict)
    
    # Step 4: Local search for further improvement
    if time.time() - start_time < args.time_limit - 1:
        coloring, num_colors = local_search_color_reduction(
            num_nodes, edges, coloring, num_colors, adj, args.time_limit, start_time, logger
        )
        
        solution_dict = {
            "objective_value": float(num_colors),
            "coloring": {str(v): coloring[v] for v in range(num_nodes)}
        }
    
    # Write final solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f)

if __name__ == '__main__':
    main()