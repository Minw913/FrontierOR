import json
import argparse
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path) as f:
        instance = json.load(f)
        
    vertices = instance['vertices']
    edges = instance['edges']
    k = instance['interdiction_budget_k']
    
    # Normalize edges to sorted tuples for consistent hashing
    edges = [tuple(sorted(e)) for e in edges]
    
    start_time = time.time()
    
    def get_max_clique(current_edges):
        """Tomita's max clique algorithm with greedy coloring bound."""
        adj = {v: set() for v in vertices}
        for u, v in current_edges:
            if u in adj and v in adj:
                adj[u].add(v)
                adj[v].add(u)
                
        best_clique = []
        best_size = 0
        timed_out = False
        
        def color_sort(P):
            """Greedy coloring of set P, returns ordered list and color map."""
            colors = {}
            color_sets = []
            # Order vertices by degree within P (descending)
            P_list = sorted(list(P), key=lambda v: len(P & adj[v]), reverse=True)
            for v in P_list:
                c = 0
                while c < len(color_sets):
                    # Check if v can be added to color class c
                    if all(u not in adj[v] for u in color_sets[c]):
                        color_sets[c].append(v)
                        break
                    c += 1
                else:
                    color_sets.append([v])
                colors[v] = c
            return P_list, colors
            
        def branch(R, P, X):
            nonlocal best_clique, best_size, timed_out
            if time.time() - start_time > args.time_limit:
                timed_out = True
                return
            if not P and not X:
                if len(R) > best_size:
                    best_clique = list(R)
                    best_size = len(R)
                return
            P_list, colors = color_sort(P)
            # Iterate from the highest color (rightmost) to prune earlier
            for i in range(len(P_list)-1, -1, -1):
                v = P_list[i]
                if len(R) + colors[v] + 1 <= best_size:
                    return
                new_P = P & adj[v]
                new_X = X & adj[v]
                branch(R + [v], new_P, new_X)
                P.remove(v)
                X.add(v)
                
        branch([], set(vertices), set())
        return best_clique, timed_out

    # Find original max clique
    original_clique, _ = get_max_clique(edges)
    omega_high = len(original_clique)
    omega_low = 1
    
    best_omega = omega_high
    best_removed = []
    
    if logger:
        logger.log_solution(best_omega, {"objective_value": best_omega, "interdicted_edges": []})
        
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        # If Gurobi is not available, fallback to original graph
        solution = {
            "objective_value": best_omega,
            "interdicted_edges": [list(e) for e in best_removed]
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        return
    
    while omega_low <= omega_high:
        if time.time() - start_time > args.time_limit:
            break
            
        mid = (omega_low + omega_high) // 2
        
        m = gp.Model("hitting")
        m.setParam('OutputFlag', 0)
        m.setParam('Seed', 0)
        m.setParam('MIPGap', 1e-4)
        m.setParam('NumericFocus', 0)
        m.setParam('Threads', 1)
        m.setParam('TimeLimit', max(1, args.time_limit - (time.time() - start_time)))
        
        x = {e: m.addVar(vtype=GRB.BINARY) for e in edges}
        m.addConstr(gp.quicksum(x.values()) <= k)
        
        feasible = False
        removed = []
        
        while True:
            if time.time() - start_time > args.time_limit:
                break
            m.optimize()
            if m.status == GRB.INFEASIBLE:
                break
            if m.status == GRB.TIME_LIMIT and m.SolCount == 0:
                break
            removed = [e for e in edges if x[e].X > 0.5]
            current_edges = [e for e in edges if x[e].X < 0.5]
            clique, timed_out = get_max_clique(current_edges)
            if timed_out:
                break
            if len(clique) <= mid:
                feasible = True
                break
            else:
                clique_set = set(clique)
                clique_edges = [e for e in edges if e[0] in clique_set and e[1] in clique_set]
                if not clique_edges:
                    feasible = True
                    break
                m.addConstr(gp.quicksum(x[e] for e in clique_edges) >= 1)
                
        if feasible:
            best_omega = mid
            best_removed = removed
            if logger:
                logger.log_solution(best_omega, {"objective_value": best_omega, "interdicted_edges": [list(e) for e in best_removed]})
            omega_high = mid - 1
        else:
            omega_low = mid + 1
            
    solution = {
        "objective_value": best_omega,
        "interdicted_edges": [list(e) for e in best_removed]
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()