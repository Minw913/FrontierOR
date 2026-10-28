import argparse
import json
import time
import sys
from solution_logger import SolutionLogger


def solve_max_clique(instance, time_limit, logger=None):
    start_time = time.time()
    
    num_vertices = instance["num_vertices"]
    edges = instance["edges"]
    adjacency_list = instance["adjacency_list"]
    
    # Convert adjacency list keys to int and values to sets for fast lookup
    adj = {}
    for v in range(num_vertices):
        key = str(v)
        if key in adjacency_list:
            adj[v] = set(adjacency_list[key])
        else:
            adj[v] = set()
    
    # Edge set for fast lookup
    edge_set = set()
    for e in edges:
        edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
    
    def is_edge(u, v):
        if u == v:
            return False
        return (min(u, v), max(u, v)) in edge_set
    
    # Start with greedy clique as initial best
    best_clique = list(instance.get("greedy_clique", []))
    best_size = len(best_clique)
    
    if logger and best_size > 0:
        logger.log(best_size)
    
    # Verify greedy clique
    valid = True
    for i in range(len(best_clique)):
        for j in range(i + 1, len(best_clique)):
            if not is_edge(best_clique[i], best_clique[j]):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        best_clique = []
        best_size = 0
    
    # For small graphs, try exact branch and bound
    # For larger graphs, use heuristics + Gurobi
    
    def remaining_time():
        return time_limit - (time.time() - start_time)
    
    # Branch and Bound with coloring bound (Tomita-style)
    # Order vertices by degree (descending)
    vertices_by_degree = sorted(range(num_vertices), key=lambda v: len(adj[v]), reverse=True)
    
    # Color-based upper bound
    def greedy_color(vertices, adj_sets):
        """Greedy coloring of subgraph induced by vertices. Returns color assignment and number of colors."""
        color = {}
        num_colors = 0
        # Order by degree in subgraph (descending)
        for v in vertices:
            used = set()
            for u in adj_sets.get(v, set()):
                if u in color:
                    used.add(color[u])
            c = 0
            while c in used:
                c += 1
            color[v] = c
            if c + 1 > num_colors:
                num_colors = c + 1
        return color, num_colors
    
    # Use networkx-free Bron-Kerbosch with pivoting and pruning
    def bronkerbosch_max_clique():
        nonlocal best_clique, best_size
        
        # We'll implement a iterative/recursive BK with pivot and coloring bound
        # Stack-based to allow time checking
        
        call_count = [0]
        
        def bk(R, P, X):
            nonlocal best_clique, best_size
            
            call_count[0] += 1
            if call_count[0] % 10000 == 0:
                if remaining_time() < 0.5:
                    return True  # timeout signal
            
            if not P and not X:
                if len(R) > best_size:
                    best_clique = list(R)
                    best_size = len(R)
                    if logger:
                        logger.log(best_size)
                return False
            
            # Pruning: even if we add all of P, can we beat best?
            if len(R) + len(P) <= best_size:
                return False
            
            # Coloring bound on P
            if len(P) > 10:
                # Order P vertices and do greedy coloring
                P_list = sorted(P, key=lambda v: -len(adj[v] & P))
                color_count = 0
                color_assign = {}
                for v in P_list:
                    used_colors = set()
                    for u in adj[v]:
                        if u in color_assign:
                            used_colors.add(color_assign[u])
                    c = 0
                    while c in used_colors:
                        c += 1
                    color_assign[v] = c
                    if c + 1 > color_count:
                        color_count = c + 1
                
                if len(R) + color_count <= best_size:
                    return False
            
            # Choose pivot: vertex in P ∪ X with max connections to P
            pivot = None
            max_conn = -1
            for u in P | X:
                conn = len(adj[u] & P)
                if conn > max_conn:
                    max_conn = conn
                    pivot = u
            
            # Branch on vertices in P \ N(pivot)
            candidates = P - adj[pivot]
            # Order candidates by degree in P (descending) for better pruning
            candidates = sorted(candidates, key=lambda v: len(adj[v] & P), reverse=True)
            
            for v in candidates:
                new_P = P & adj[v]
                new_X = X & adj[v]
                R.add(v)
                timeout = bk(R, new_P, new_X)
                R.remove(v)
                if timeout:
                    return True
                P.remove(v)
                X.add(v)
            
            return False
        
        P = set(range(num_vertices))
        R = set()
        X = set()
        bk(R, P, X)
    
    # For very large graphs, try Gurobi ILP approach
    def solve_with_gurobi():
        nonlocal best_clique, best_size
        
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError:
            return
        
        rt = remaining_time()
        if rt < 1:
            return
        
        model = gp.Model("max_clique")
        model.setParam("TimeLimit", max(1, rt - 1))
        model.setParam("Threads", 1)
        model.setParam("OutputFlag", 0)
        
        # Variables: x[v] = 1 if vertex v is in the clique
        x = {}
        for v in range(num_vertices):
            x[v] = model.addVar(vtype=gp.GRB.BINARY, name=f"x_{v}")
        
        model.setObjective(gp.quicksum(x[v] for v in range(num_vertices)), gp.GRB.MAXIMIZE)
        
        # For every pair of non-adjacent vertices, at most one can be selected
        # This can be very many constraints for sparse graphs
        # For dense graphs, complement is sparse, so fewer constraints
        
        # Build complement edge list
        complement_edges = []
        for u in range(num_vertices):
            for v in range(u + 1, num_vertices):
                if not is_edge(u, v):
                    complement_edges.append((u, v))
        
        for (u, v) in complement_edges:
            model.addConstr(x[u] + x[v] <= 1)
        
        # Warm start with best known clique
        if best_size > 0:
            for v in range(num_vertices):
                x[v].Start = 1.0 if v in set(best_clique) else 0.0
        
        def callback(model, where):
            nonlocal best_clique, best_size
            if where == gp.GRB.Callback.MIPSOL:
                obj = int(round(model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)))
                if obj > best_size:
                    vals = model.cbGetSolution(list(x.values()))
                    clique = [v for v in range(num_vertices) if vals[v] > 0.5]
                    best_clique = clique
                    best_size = obj
                    if logger:
                        logger.log(best_size)
        
        model.optimize(callback)
        
        if model.SolCount > 0:
            clique = [v for v in range(num_vertices) if x[v].X > 0.5]
            if len(clique) > best_size:
                best_clique = clique
                best_size = len(clique)
                if logger:
                    logger.log(best_size)
    
    # Decision: for small/medium graphs use BK, for larger use Gurobi
    # Also consider density - dense complement means many constraints for Gurobi
    
    actual_density = instance.get("actual_density", 0.5)
    
    if num_vertices <= 300:
        # Try branch and bound first
        bronkerbosch_max_clique()
        
        # If time remains and we didn't finish, try Gurobi
        if remaining_time() > 2:
            solve_with_gurobi()
    else:
        # For larger graphs, go directly to Gurobi if complement is not too dense
        num_complement_edges = num_vertices * (num_vertices - 1) // 2 - len(edges)
        
        if num_complement_edges < 500000:
            solve_with_gurobi()
        else:
            # Try BK with time limit
            bronkerbosch_max_clique()
            if remaining_time() > 2:
                solve_with_gurobi()
    
    return best_clique, best_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    clique, size = solve_max_clique(instance, args.time_limit, logger)
    
    solution = {
        "objective_value": size,
        "clique_vertices": clique
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()