import argparse
import json
import time
import sys
from collections import defaultdict
from solution_logger import SolutionLogger


def solve(instance, time_limit, logger=None):
    start_time = time.time()
    
    graph = instance["graph"]
    partition = instance["partition"]
    
    num_vertices = graph["num_vertices"]
    edges = graph["edges"]
    num_clusters = partition["num_clusters"]
    clusters = partition["clusters"]
    
    # Build adjacency list
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Build edge set for quick lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
    
    def are_adjacent(u, v):
        return (min(u, v), max(u, v)) in edge_set
    
    # For small instances, try exact approach with Gurobi
    # For larger instances, use heuristic
    
    total_vertices_in_clusters = sum(len(c) for c in clusters)
    total_combinations = 1
    for c in clusters:
        total_combinations *= len(c)
        if total_combinations > 1e8:
            break
    
    # Heuristic approach: greedy + local search
    # First, try a greedy approach
    
    def compute_chromatic_number(selected):
        """Compute chromatic number of induced subgraph using greedy coloring with ordering."""
        n = len(selected)
        if n == 0:
            return 0, {}
        
        # Build induced subgraph adjacency
        induced_adj = defaultdict(set)
        for i in range(n):
            for j in range(i + 1, n):
                if are_adjacent(selected[i], selected[j]):
                    induced_adj[selected[i]].add(selected[j])
                    induced_adj[selected[j]].add(selected[i])
        
        # Use DSatur (saturation degree ordering) for better coloring
        colors = {}
        saturation = {v: set() for v in selected}
        degree = {v: len(induced_adj[v]) for v in selected}
        uncolored = set(selected)
        
        while uncolored:
            # Pick vertex with highest saturation, break ties by degree
            best = max(uncolored, key=lambda v: (len(saturation[v]), degree[v]))
            # Assign smallest available color
            used = saturation[best]
            c = 0
            while c in used:
                c += 1
            colors[best] = c
            uncolored.remove(best)
            for nb in induced_adj[best]:
                if nb in uncolored:
                    saturation[nb].add(c)
        
        num_colors = max(colors.values()) + 1 if colors else 0
        return num_colors, colors
    
    def compute_clique_number(selected):
        """Compute a lower bound on chromatic number via max clique in induced subgraph."""
        # For perfect graphs, chromatic number = clique number
        # Find max clique using greedy approach
        n = len(selected)
        if n == 0:
            return 0
        
        induced_adj = defaultdict(set)
        for i in range(n):
            for j in range(i + 1, n):
                if are_adjacent(selected[i], selected[j]):
                    induced_adj[selected[i]].add(selected[j])
                    induced_adj[selected[j]].add(selected[i])
        
        # Greedy max clique
        best_clique = []
        vertices_sorted = sorted(selected, key=lambda v: len(induced_adj[v]), reverse=True)
        
        for start in vertices_sorted[:min(len(vertices_sorted), 20)]:
            clique = [start]
            candidates = list(induced_adj[start])
            for v in sorted(candidates, key=lambda x: len(induced_adj[x]), reverse=True):
                if all(v in induced_adj[c] for c in clique):
                    clique.append(v)
            if len(clique) > len(best_clique):
                best_clique = clique
        
        return len(best_clique)
    
    # Try to use Gurobi for exact solution if instance is manageable
    use_gurobi = True
    
    if use_gurobi:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            remaining_time = time_limit - (time.time() - start_time)
            if remaining_time < 1:
                use_gurobi = False
            else:
                # First, find a good heuristic solution to use as warm start
                # Greedy: pick vertices to minimize conflicts
                best_selected = []
                best_obj = num_clusters + 1
                best_colors = {}
                
                # Try several random-ish greedy starts
                import random
                random.seed(42)
                
                # Simple greedy: pick first vertex from each cluster
                selected = [c[0] for c in clusters]
                obj, colors = compute_chromatic_number(selected)
                if obj < best_obj:
                    best_obj = obj
                    best_selected = selected[:]
                    best_colors = colors.copy()
                    if logger:
                        logger.log(best_obj)
                
                # Try: for each cluster, pick vertex with minimum degree in the graph
                selected = [min(c, key=lambda v: len(adj[v])) for c in clusters]
                obj, colors = compute_chromatic_number(selected)
                if obj < best_obj:
                    best_obj = obj
                    best_selected = selected[:]
                    best_colors = colors.copy()
                    if logger:
                        logger.log(best_obj)
                
                # Now set up Gurobi model
                # Variables: x[i][v] = 1 if vertex v is selected from cluster i
                # y[v][c] = 1 if selected vertex v gets color c
                # w[c] = 1 if color c is used
                # Minimize sum of w[c]
                
                # Upper bound on colors needed
                K = best_obj  # Use heuristic solution as upper bound
                
                m = gp.Model("graph_coloring")
                m.setParam("OutputFlag", 0)
                m.setParam("TimeLimit", max(1, remaining_time - 1))
                m.setParam("Threads", 1)
                
                # x[i][j] = 1 if j-th vertex in cluster i is selected
                x = {}
                vertex_to_cluster = {}
                for i, cluster in enumerate(clusters):
                    for v in cluster:
                        x[i, v] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{v}")
                        vertex_to_cluster[v] = i
                
                # w[c] = 1 if color c is used
                w = {}
                for c in range(K):
                    w[c] = m.addVar(vtype=GRB.BINARY, name=f"w_{c}")
                
                # y[v][c] = 1 if vertex v gets color c
                y = {}
                all_vertices = []
                for cluster in clusters:
                    for v in cluster:
                        all_vertices.append(v)
                        for c in range(K):
                            y[v, c] = m.addVar(vtype=GRB.BINARY, name=f"y_{v}_{c}")
                
                m.update()
                
                # Objective: minimize number of colors
                m.setObjective(gp.quicksum(w[c] for c in range(K)), GRB.MINIMIZE)
                
                # Constraint: exactly one vertex per cluster
                for i, cluster in enumerate(clusters):
                    m.addConstr(gp.quicksum(x[i, v] for v in cluster) == 1)
                
                # Constraint: each selected vertex gets exactly one color
                for i, cluster in enumerate(clusters):
                    for v in cluster:
                        m.addConstr(gp.quicksum(y[v, c] for c in range(K)) == x[i, v])
                
                # Constraint: y[v,c] <= w[c]
                for v in all_vertices:
                    for c in range(K):
                        m.addConstr(y[v, c] <= w[c])
                
                # Constraint: y[v,c] <= x[cluster(v), v]
                for v in all_vertices:
                    ci = vertex_to_cluster[v]
                    for c in range(K):
                        m.addConstr(y[v, c] <= x[ci, v])
                
                # Constraint: adjacent vertices in different clusters can't have same color
                # For each edge (u,v) where u and v are in different clusters:
                # y[u,c] + y[v,c] <= 1 for all colors c
                for u_edge, v_edge in edges:
                    cu = vertex_to_cluster.get(u_edge)
                    cv = vertex_to_cluster.get(v_edge)
                    if cu is not None and cv is not None and cu != cv:
                        for c in range(K):
                            m.addConstr(y[u_edge, c] + y[v_edge, c] <= 1)
                
                # Symmetry breaking: w[c] >= w[c+1]
                for c in range(K - 1):
                    m.addConstr(w[c] >= w[c + 1])
                
                # Warm start with heuristic solution
                for i, cluster in enumerate(clusters):
                    for v in cluster:
                        if v == best_selected[i]:
                            x[i, v].Start = 1.0
                        else:
                            x[i, v].Start = 0.0
                
                for v in all_vertices:
                    for c in range(K):
                        y[v, c].Start = 0.0
                
                for i, cluster in enumerate(clusters):
                    v = best_selected[i]
                    color = best_colors[v]
                    if color < K:
                        y[v, color].Start = 1.0
                
                used_colors_set = set(best_colors.values())
                for c in range(K):
                    w[c].Start = 1.0 if c in used_colors_set else 0.0
                
                # Callback to log improvements
                def callback(model, where):
                    if where == GRB.Callback.MIPSOL:
                        obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                        if logger:
                            logger.log(int(round(obj_val)))
                
                m.optimize(callback)
                
                if m.SolCount > 0:
                    obj_val = int(round(m.ObjVal))
                    selected_vertices = {}
                    coloring = {}
                    
                    for i, cluster in enumerate(clusters):
                        for v in cluster:
                            if x[i, v].X > 0.5:
                                selected_vertices[i] = v
                                for c in range(K):
                                    if y[v, c].X > 0.5:
                                        coloring[v] = c
                                        break
                                break
                    
                    return {
                        "objective_value": obj_val,
                        "selected_vertices": {str(k): v for k, v in selected_vertices.items()},
                        "coloring": {str(k): v for k, v in coloring.items()}
                    }
                else:
                    # Fall back to heuristic
                    selected_vertices = {str(i): best_selected[i] for i in range(num_clusters)}
                    coloring = {str(v): c for v, c in best_colors.items()}
                    return {
                        "objective_value": best_obj,
                        "selected_vertices": selected_vertices,
                        "coloring": coloring
                    }
                    
        except Exception as e:
            use_gurobi = False
    
    if not use_gurobi:
        # Pure heuristic fallback
        import random
        random.seed(42)
        
        selected = [c[0] for c in clusters]
        best_obj, best_colors = compute_chromatic_number(selected)
        best_selected = selected[:]
        
        if logger:
            logger.log(best_obj)
        
        # Local search: try swapping vertices in clusters
        for iteration in range(1000):
            if time.time() - start_time > time_limit - 1:
                break
            
            i = random.randint(0, num_clusters - 1)
            if len(clusters[i]) <= 1:
                continue
            
            old_v = selected[i]
            new_v = random.choice(clusters[i])
            if new_v == old_v:
                continue
            
            selected[i] = new_v
            obj, colors = compute_chromatic_number(selected)
            
            if obj < best_obj:
                best_obj = obj
                best_selected = selected[:]
                best_colors = colors.copy()
                if logger:
                    logger.log(best_obj)
            elif obj > best_obj + 1:
                selected[i] = old_v  # revert bad moves
        
        selected_vertices = {str(i): best_selected[i] for i in range(num_clusters)}
        coloring = {str(v): c for v, c in best_colors.items()}
        return {
            "objective_value": best_obj,
            "selected_vertices": selected_vertices,
            "coloring": coloring
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    with open(args.instance_path, "r") as f:
        instance = json.load(f)
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    solution = solve(instance, args.time_limit, logger)
    
    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()