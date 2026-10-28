import argparse
import json
import time
import sys
from solution_logger import SolutionLogger


def solve_max_clique(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    
    # Initialize logger
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    # Read instance
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    graph = instance['graph']
    num_vertices = graph['num_vertices']
    num_edges = graph['num_edges']
    adj_list = graph['adjacency_list']
    
    # Build adjacency sets
    adj = {}
    for v_str, neighbors in adj_list.items():
        v = int(v_str)
        adj[v] = set(neighbors)
    
    # Ensure all vertices are in adj
    for v in range(num_vertices):
        if v not in adj:
            adj[v] = set()
    
    # Best clique found so far
    best_clique = []
    
    def update_best(clique):
        nonlocal best_clique
        if len(clique) > len(best_clique):
            best_clique = list(clique)
            if logger:
                logger.log(len(best_clique))
    
    def time_remaining():
        return time_limit - (time.time() - start_time)
    
    def is_time_up():
        return time.time() - start_time >= time_limit
    
    # Greedy initial solution
    def greedy_clique(vertices=None):
        if vertices is None:
            vertices = list(range(num_vertices))
        # Sort by degree descending within subgraph
        vset = set(vertices)
        deg = {v: len(adj[v] & vset) for v in vertices}
        sorted_v = sorted(vertices, key=lambda v: deg[v], reverse=True)
        
        clique = []
        clique_set = set()
        for v in sorted_v:
            if all(v in adj[u] for u in clique):
                clique.append(v)
                clique_set.add(v)
        return clique
    
    initial = greedy_clique()
    update_best(initial)
    
    # For small graphs, try exact BnB
    # For larger graphs, use heuristics + Gurobi
    
    # Branch and Bound for Maximum Clique (Bron-Kerbosch style with pruning)
    # Using greedy coloring for upper bound
    
    def greedy_color(vertices, adj_local):
        """Greedy coloring to get upper bound on clique size."""
        color = {}
        for v in vertices:
            used = set()
            for u in adj_local.get(v, set()):
                if u in color:
                    used.add(color[u])
            c = 0
            while c in used:
                c += 1
            color[v] = c
        num_colors = max(color.values()) + 1 if color else 0
        return num_colors, color
    
    def bnb_clique():
        """Branch and bound for maximum clique."""
        nonlocal best_clique
        
        # Order vertices by degree (ascending for BnB efficiency)
        vertices = sorted(range(num_vertices), key=lambda v: len(adj[v]))
        
        # Build ordering-based sets
        # We use a Tomita/MCQ style approach
        
        node_count = [0]
        
        def expand(clique, candidates):
            if is_time_up():
                return
            
            node_count[0] += 1
            
            if not candidates:
                update_best(clique)
                return
            
            # Color candidates for upper bound
            # Order candidates by degree in subgraph
            cand_set = set(candidates)
            cand_sorted = sorted(candidates, key=lambda v: len(adj[v] & cand_set))
            
            # Greedy coloring
            color = {}
            color_classes = []  # list of lists
            for v in cand_sorted:
                placed = False
                for i, cc in enumerate(color_classes):
                    # Check if v is non-adjacent to all in cc
                    if not any(u in adj[v] for u in cc):
                        cc.append(v)
                        color[v] = i
                        placed = True
                        break
                if not placed:
                    color_classes.append([v])
                    color[v] = len(color_classes) - 1
            
            num_colors = len(color_classes)
            
            if len(clique) + num_colors <= len(best_clique):
                return  # Prune
            
            # Process vertices in order of color (highest color first = most constrained)
            # Actually, process from highest color to lowest for better pruning
            ordered = sorted(candidates, key=lambda v: color[v], reverse=True)
            
            for v in ordered:
                if is_time_up():
                    return
                
                if len(clique) + color[v] + 1 <= len(best_clique):
                    return  # Prune: remaining vertices can't improve
                
                new_candidates = [u for u in candidates if u in adj[v] and u != v]
                expand(clique + [v], new_candidates)
                candidates = [u for u in candidates if u != v]
        
        expand([], vertices)
    
    # For very large graphs, try Gurobi ILP approach
    def solve_with_gurobi():
        nonlocal best_clique
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError:
            return
        
        remaining_time = time_remaining()
        if remaining_time <= 1:
            return
        
        # Build complement edge set (for constraints)
        # For each pair not connected, add constraint x_i + x_j <= 1
        
        # For large graphs, this can be huge. Let's check feasibility.
        complement_edges = []
        all_vertices = list(range(num_vertices))
        
        # Number of complement edges = C(n,2) - num_edges
        num_complement = num_vertices * (num_vertices - 1) // 2 - num_edges
        
        if num_vertices > 3000 or num_complement > 500000:
            # Too large for direct ILP, skip
            return
        
        try:
            model = gp.Model("max_clique")
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', max(remaining_time - 1, 1))
            model.setParam('Threads', 1)
            
            # Variables
            x = model.addVars(num_vertices, vtype=GRB.BINARY, name="x")
            
            # Objective
            model.setObjective(gp.quicksum(x[i] for i in range(num_vertices)), GRB.MAXIMIZE)
            
            # Constraints: for non-edges, x_i + x_j <= 1
            edge_set = set()
            for v in range(num_vertices):
                for u in adj[v]:
                    if u > v:
                        edge_set.add((v, u))
            
            for i in range(num_vertices):
                for j in range(i + 1, num_vertices):
                    if (i, j) not in edge_set:
                        model.addConstr(x[i] + x[j] <= 1)
            
            # Warm start with best known clique
            for v in range(num_vertices):
                x[v].Start = 1 if v in set(best_clique) else 0
            
            # Callback for logging
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = int(round(model.cbGet(GRB.Callback.MIPSOL_OBJ)))
                    if obj > len(best_clique):
                        vals = model.cbGetSolution(model._x)
                        clique = [v for v in range(num_vertices) if vals[v] > 0.5]
                        update_best(clique)
            
            model._x = x
            model.optimize(callback)
            
            if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
                if model.SolCount > 0:
                    clique = [v for v in range(num_vertices) if x[v].X > 0.5]
                    update_best(clique)
        except Exception as e:
            pass
    
    # Decision: use BnB for small, Gurobi for medium, heuristics for large
    if num_vertices <= 500:
        # Try exact BnB first
        bnb_clique()
        if not is_time_up() and time_remaining() > 2:
            solve_with_gurobi()
    elif num_vertices <= 3000:
        solve_with_gurobi()
        if not is_time_up() and time_remaining() > 2:
            bnb_clique()
    else:
        # Large graph - use heuristics
        # Multiple random greedy attempts
        import random
        
        while not is_time_up():
            # Randomized greedy
            vertices = list(range(num_vertices))
            random.shuffle(vertices)
            
            clique = []
            clique_set = set()
            for v in vertices:
                if all(v in adj[u] for u in clique):
                    clique.append(v)
                    clique_set.add(v)
            update_best(clique)
            
            # Local search: try to swap vertices
            clique_set = set(best_clique)
            improved = True
            while improved and not is_time_up():
                improved = False
                # Try adding a vertex
                for v in range(num_vertices):
                    if v not in clique_set:
                        if all(v in adj[u] for u in clique_set):
                            clique_set.add(v)
                            improved = True
                            update_best(list(clique_set))
                
                if not improved:
                    # Try swap: remove one, add two
                    for v in list(clique_set):
                        if is_time_up():
                            break
                        test_set = clique_set - {v}
                        # Find vertices adjacent to all in test_set but not in clique_set
                        candidates = []
                        for u in range(num_vertices):
                            if u not in test_set and all(u in adj[w] for w in test_set):
                                candidates.append(u)
                        # Try pairs from candidates
                        for i in range(len(candidates)):
                            if is_time_up():
                                break
                            for j in range(i + 1, len(candidates)):
                                if candidates[j] in adj[candidates[i]]:
                                    new_set = test_set | {candidates[i], candidates[j]}
                                    if len(new_set) > len(clique_set):
                                        clique_set = new_set
                                        improved = True
                                        update_best(list(clique_set))
                                        break
                            if improved:
                                break
                        if improved:
                            break
    
    # Write solution
    solution = {
        "objective_value": len(best_clique),
        "clique_vertices": sorted(best_clique)
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve_max_clique(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()