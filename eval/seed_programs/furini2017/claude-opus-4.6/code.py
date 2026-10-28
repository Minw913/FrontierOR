import argparse
import json
import time
import sys
from solution_logger import SolutionLogger


def solve_graph_coloring(instance_path, solution_path, time_limit, log_path):
    start_time = time.time()
    
    # Read instance
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    graph = instance['graph']
    n = graph['n']
    m = graph['m']
    edges = graph['edges']
    adj = graph['adjacency_list']
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Handle trivial cases
    if n == 0:
        solution = {
            "objective_value": 0,
            "coloring": {},
            "color_classes": {}
        }
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
        return
    
    if m == 0:
        coloring = {str(i): 0 for i in range(n)}
        color_classes = {"0": list(range(n))}
        solution = {
            "objective_value": 1,
            "coloring": coloring,
            "color_classes": color_classes
        }
        if logger:
            logger.log(1)
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
        return
    
    # Convert adjacency list to sets for faster lookup
    adj_sets = [set(neighbors) for neighbors in adj]
    
    # Compute a greedy coloring using DSATUR heuristic
    def dsatur_coloring():
        colors = [-1] * n
        saturation = [0] * n
        degrees = [len(adj[i]) for i in range(n)]
        colored_neighbors = [set() for _ in range(n)]
        
        # Pick vertex with highest degree first
        first = max(range(n), key=lambda v: degrees[v])
        colors[first] = 0
        for nb in adj[first]:
            saturation[nb] += 1
            colored_neighbors[nb].add(0)
        
        num_colors = 1
        
        for step in range(1, n):
            # Pick uncolored vertex with max saturation, break ties by degree
            best_v = -1
            best_sat = -1
            best_deg = -1
            for v in range(n):
                if colors[v] == -1:
                    if (saturation[v] > best_sat or 
                        (saturation[v] == best_sat and degrees[v] > best_deg)):
                        best_v = v
                        best_sat = saturation[v]
                        best_deg = degrees[v]
            
            # Find smallest color not used by neighbors
            neighbor_colors = colored_neighbors[best_v]
            c = 0
            while c in neighbor_colors:
                c += 1
            
            colors[best_v] = c
            if c >= num_colors:
                num_colors = c + 1
            
            for nb in adj[best_v]:
                if c not in colored_neighbors[nb]:
                    colored_neighbors[nb].add(c)
                    saturation[nb] += 1
        
        return colors, num_colors
    
    best_colors, best_num_colors = dsatur_coloring()
    if logger:
        logger.log(best_num_colors)
    
    # Compute clique lower bound
    def greedy_clique():
        """Find a clique using greedy approach."""
        # Sort vertices by degree descending
        vertices_by_degree = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
        clique = []
        clique_set = set()
        for v in vertices_by_degree:
            if all(v in adj_sets[u] for u in clique):
                clique.append(v)
                clique_set.add(v)
        return clique
    
    clique = greedy_clique()
    lower_bound = len(clique)
    
    # Try to improve lower bound with multiple random restarts
    import random
    random.seed(42)
    
    def greedy_clique_random():
        vertices = list(range(n))
        random.shuffle(vertices)
        # Sort by degree but with some randomness
        clique = []
        clique_set = set()
        for v in vertices:
            if all(v in adj_sets[u] for u in clique):
                clique.append(v)
                clique_set.add(v)
        return clique
    
    for _ in range(100):
        if time.time() - start_time > time_limit * 0.1:
            break
        c = greedy_clique_random()
        if len(c) > lower_bound:
            lower_bound = len(c)
            clique = c
    
    if best_num_colors == lower_bound:
        # Already optimal
        coloring = {str(i): best_colors[i] for i in range(n)}
        color_classes = {}
        for i in range(n):
            c = str(best_colors[i])
            if c not in color_classes:
                color_classes[c] = []
            color_classes[c].append(i)
        solution = {
            "objective_value": best_num_colors,
            "coloring": coloring,
            "color_classes": color_classes
        }
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
        return
    
    # Try to improve with local search (tabu coloring)
    def try_coloring_with_k(k, max_iter=100000):
        """Try to find a proper k-coloring using tabu search."""
        nonlocal best_colors, best_num_colors
        
        if k >= best_num_colors:
            return False
        
        # Initialize with a greedy assignment (may have conflicts)
        colors = [0] * n
        # Use DSATUR-like but limited to k colors
        for i in range(n):
            colors[i] = random.randint(0, k - 1)
        
        # Count conflicts
        conflicts = 0
        vertex_conflicts = [0] * n
        for u, v in edges:
            if colors[u] == colors[v]:
                conflicts += 1
                vertex_conflicts[u] += 1
                vertex_conflicts[v] += 1
        
        if conflicts == 0:
            best_colors = colors[:]
            best_num_colors = k
            if logger:
                logger.log(k)
            return True
        
        # Tabu tenure
        tabu_tenure = 10
        tabu = [[0] * k for _ in range(n)]
        
        best_conflicts = conflicts
        iteration = 0
        
        while iteration < max_iter and time.time() - start_time < time_limit - 0.5:
            iteration += 1
            
            # Find best non-tabu move
            best_delta = float('inf')
            best_move = None
            
            # Consider vertices with conflicts
            conflict_vertices = [v for v in range(n) if vertex_conflicts[v] > 0]
            if not conflict_vertices:
                break
            
            # Sample if too many
            if len(conflict_vertices) > 50:
                sample = random.sample(conflict_vertices, 50)
            else:
                sample = conflict_vertices
            
            for v in sample:
                old_c = colors[v]
                for new_c in range(k):
                    if new_c == old_c:
                        continue
                    
                    # Compute delta
                    delta = 0
                    for nb in adj[v]:
                        if colors[nb] == old_c:
                            delta -= 1
                        if colors[nb] == new_c:
                            delta += 1
                    
                    # Check tabu
                    if tabu[v][new_c] > iteration:
                        # Aspiration: allow if it improves best
                        if conflicts + delta < best_conflicts:
                            pass
                        else:
                            continue
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (v, new_c)
            
            if best_move is None:
                # Pick random conflict vertex and random color
                v = random.choice(conflict_vertices)
                new_c = random.randint(0, k - 1)
                while new_c == colors[v]:
                    new_c = random.randint(0, k - 1)
                best_move = (v, new_c)
                best_delta = 0
                for nb in adj[v]:
                    if colors[nb] == colors[v]:
                        best_delta -= 1
                    if colors[nb] == new_c:
                        best_delta += 1
            
            v, new_c = best_move
            old_c = colors[v]
            
            # Apply move
            for nb in adj[v]:
                if colors[nb] == old_c:
                    vertex_conflicts[nb] -= 1
                    vertex_conflicts[v] -= 1
                if colors[nb] == new_c:
                    vertex_conflicts[nb] += 1
                    vertex_conflicts[v] += 1
            
            conflicts += best_delta
            colors[v] = new_c
            tabu[v][old_c] = iteration + tabu_tenure + random.randint(0, 5)
            
            if conflicts < best_conflicts:
                best_conflicts = conflicts
            
            if conflicts == 0:
                best_colors = colors[:]
                best_num_colors = k
                if logger:
                    logger.log(k)
                return True
        
        return False
    
    # Try reducing colors one by one
    for k in range(best_num_colors - 1, lower_bound - 1, -1):
        if time.time() - start_time > time_limit - 1.0:
            break
        
        remaining_time = time_limit - (time.time() - start_time) - 0.5
        if remaining_time <= 0:
            break
        
        # Adaptive iteration count based on remaining time
        max_iter = max(10000, int(remaining_time * 50000))
        
        success = try_coloring_with_k(k, max_iter=max_iter)
        if not success:
            break
        
        if k == lower_bound:
            break
    
    # For small instances, try exact approach with Gurobi
    if n <= 80 and best_num_colors > lower_bound and time.time() - start_time < time_limit - 2.0:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            remaining_time = time_limit - (time.time() - start_time) - 0.5
            if remaining_time > 1.0:
                K = best_num_colors  # Upper bound
                
                model = gp.Model("graph_coloring")
                model.setParam('TimeLimit', remaining_time)
                model.setParam('OutputFlag', 0)
                model.setParam('Threads', 1)
                
                # x[v, c] = 1 if vertex v gets color c
                x = {}
                for v in range(n):
                    for c in range(K):
                        x[v, c] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}_{c}")
                
                # y[c] = 1 if color c is used
                y = {}
                for c in range(K):
                    y[c] = model.addVar(vtype=GRB.BINARY, name=f"y_{c}")
                
                model.update()
                
                # Each vertex gets exactly one color
                for v in range(n):
                    model.addConstr(gp.quicksum(x[v, c] for c in range(K)) == 1)
                
                # Adjacent vertices get different colors
                for u, v_node in edges:
                    for c in range(K):
                        model.addConstr(x[u, c] + x[v_node, c] <= y[c])
                
                # Symmetry breaking: use colors in order
                for c in range(1, K):
                    model.addConstr(y[c] <= y[c - 1])
                
                # Link x and y
                for v in range(n):
                    for c in range(K):
                        model.addConstr(x[v, c] <= y[c])
                
                # Fix clique vertices to distinct colors
                for i, v in enumerate(clique):
                    if i < K:
                        x[v, i].LB = 1.0
                
                model.setObjective(gp.quicksum(y[c] for c in range(K)), GRB.MINIMIZE)
                
                # Set initial solution
                for v in range(n):
                    for c in range(K):
                        x[v, c].Start = 1.0 if best_colors[v] == c else 0.0
                for c in range(K):
                    y[c].Start = 1.0 if any(best_colors[v] == c for v in range(n)) else 0.0
                
                def callback(model, where):
                    nonlocal best_num_colors, best_colors
                    if where == GRB.Callback.MIPSOL:
                        obj = int(round(model.cbGet(GRB.Callback.MIPSOL_OBJBND)))
                        obj_val = int(round(model.cbGet(GRB.Callback.MIPSOL_OBJ)))
                        if obj_val < best_num_colors:
                            best_num_colors = obj_val
                            new_colors = [0] * n
                            for v in range(n):
                                for c in range(K):
                                    if model.cbGetSolution(x[v, c]) > 0.5:
                                        new_colors[v] = c
                                        break
                            best_colors = new_colors
                            if logger:
                                logger.log(best_num_colors)
                
                model.optimize(callback)
                
                if model.SolCount > 0:
                    obj_val = int(round(model.ObjVal))
                    if obj_val < best_num_colors:
                        best_num_colors = obj_val
                        for v in range(n):
                            for c in range(K):
                                if x[v, c].X > 0.5:
                                    best_colors[v] = c
                                    break
                        if logger:
                            logger.log(best_num_colors)
        except ImportError:
            pass
        except Exception:
            pass
    
    # Remap colors to be contiguous starting from 0
    used_colors = set(best_colors)
    color_map = {}
    idx = 0
    for c in sorted(used_colors):
        color_map[c] = idx
        idx += 1
    
    final_colors = [color_map[best_colors[i]] for i in range(n)]
    num_colors = len(used_colors)
    
    coloring = {str(i): final_colors[i] for i in range(n)}
    color_classes = {}
    for i in range(n):
        c = str(final_colors[i])
        if c not in color_classes:
            color_classes[c] = []
        color_classes[c].append(i)
    
    solution = {
        "objective_value": num_colors,
        "coloring": coloring,
        "color_classes": color_classes
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


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