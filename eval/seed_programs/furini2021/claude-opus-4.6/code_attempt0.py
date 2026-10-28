import argparse
import json
import time
import itertools
from collections import defaultdict

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    vertices = instance['vertices']
    edges = instance['edges']
    budget = instance['interdiction_budget_k']
    num_vertices = instance['num_vertices']
    num_edges = instance['num_edges']

    start_time = time.time()
    deadline = start_time + args.time_limit - 1  # leave 1 second buffer

    # Build adjacency structures
    adj = defaultdict(set)
    edge_set = set()
    for e in edges:
        u, v = e[0], e[1]
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))

    vertex_set = set(vertices)

    # Function to find maximum clique using Bron-Kerbosch with pivoting
    def find_max_clique(adj_local, vertices_local):
        """Find maximum clique in graph defined by adj_local over vertices_local"""
        best = [0]
        best_clique = [[]]
        
        def bron_kerbosch(R, P, X):
            if time.time() > deadline:
                return
            if not P and not X:
                if len(R) > best[0]:
                    best[0] = len(R)
                    best_clique[0] = list(R)
                return
            # Pruning: if |R| + |P| <= best, can't improve
            if len(R) + len(P) <= best[0]:
                return
            # Choose pivot that maximizes |P ∩ N(u)|
            pivot = max(P | X, key=lambda u: len(P & adj_local.get(u, set())))
            candidates = P - adj_local.get(pivot, set())
            for v in list(candidates):
                neighbors_v = adj_local.get(v, set())
                bron_kerbosch(R | {v}, P & neighbors_v, X & neighbors_v)
                P = P - {v}
                X = X | {v}
        
        bron_kerbosch(set(), set(vertices_local), set())
        return best[0], best_clique[0]

    # Function to find all maximal cliques (with size limit)
    def find_all_cliques_of_size(adj_local, vertices_local, target_size):
        """Find all cliques of given size"""
        cliques = []
        
        def bron_kerbosch(R, P, X):
            if time.time() > deadline:
                return
            if len(R) == target_size:
                cliques.append(frozenset(R))
                return
            if not P:
                return
            if len(R) + len(P) < target_size:
                return
            # Choose pivot
            pivot = max(P | X, key=lambda u: len(P & adj_local.get(u, set()))) if (P | X) else None
            if pivot is None:
                return
            candidates = P - adj_local.get(pivot, set())
            for v in list(candidates):
                neighbors_v = adj_local.get(v, set())
                bron_kerbosch(R | {v}, P & neighbors_v, X & neighbors_v)
                P = P - {v}
                X = X | {v}
        
        bron_kerbosch(set(), set(vertices_local), set())
        return cliques

    # Find initial maximum clique
    clique_size, max_clique = find_max_clique(adj, vertex_set)
    
    if budget == 0 or clique_size <= 1:
        result = {
            "objective_value": clique_size,
            "interdicted_edges": []
        }
        if logger:
            logger.log(clique_size)
        with open(args.solution_path, 'w') as f:
            json.dump(result, f)
        return

    best_obj = clique_size
    best_interdicted = []
    if logger:
        logger.log(best_obj)

    # Strategy: iterative approach
    # Try to reduce the clique number by removing edges from large cliques
    
    # For small instances, try MIP formulation
    # For larger instances, use heuristic approach
    
    def compute_clique_number(removed_edges):
        """Compute clique number after removing given edges"""
        adj_new = defaultdict(set)
        removed_set = set()
        for e in removed_edges:
            removed_set.add((min(e[0], e[1]), max(e[0], e[1])))
        for e in edges:
            u, v = e[0], e[1]
            key = (min(u, v), max(u, v))
            if key not in removed_set:
                adj_new[u].add(v)
                adj_new[v].add(u)
        cn, _ = find_max_clique(adj_new, vertex_set)
        return cn

    # Try using Gurobi for the MIP formulation
    try:
        import gurobipy as gp
        from gurobipy import GRB
        
        # Binary search on the target clique number
        # For each target t, check if we can remove <= budget edges to make clique number <= t
        # We need to find all cliques of size > t and ensure each has at least one edge removed
        
        # Iterative constraint generation approach
        def solve_with_target(target_t):
            """Try to find edge removal that makes clique number <= target_t"""
            if time.time() > deadline:
                return None, None
            
            # Create edge variables
            edge_list = []
            edge_index = {}
            for i, e in enumerate(edges):
                u, v = e[0], e[1]
                key = (min(u, v), max(u, v))
                edge_list.append(key)
                edge_index[key] = i
            
            model = gp.Model("clique_interdiction")
            model.setParam('OutputFlag', 0)
            remaining_time = deadline - time.time()
            if remaining_time <= 0:
                return None, None
            model.setParam('TimeLimit', max(1, remaining_time))
            model.setParam('Threads', 1)
            
            # x[i] = 1 if edge i is removed
            x = model.addVars(num_edges, vtype=GRB.BINARY, name="x")
            
            # Budget constraint
            model.addConstr(gp.quicksum(x[i] for i in range(num_edges)) <= budget)
            
            # Dummy objective (feasibility)
            model.setObjective(0, GRB.MINIMIZE)
            
            # Iterative: find max clique in current solution, add constraint
            iteration = 0
            max_iterations = 500
            
            # Start by adding constraints for known cliques
            # For each clique of size > target_t, at least one edge must be removed
            
            # Find cliques using current adjacency
            current_adj = defaultdict(set)
            for e in edges:
                u, v = e[0], e[1]
                current_adj[u].add(v)
                current_adj[v].add(u)
            
            while iteration < max_iterations and time.time() < deadline:
                iteration += 1
                model.optimize()
                
                if model.Status == GRB.INFEASIBLE:
                    return None, None
                
                if model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
                    if model.SolCount == 0:
                        return None, None
                
                if model.SolCount == 0:
                    return None, None
                
                # Get current solution
                removed = []
                for i in range(num_edges):
                    if x[i].X > 0.5:
                        removed.append(list(edge_list[i]))
                
                # Build residual graph
                removed_set = set()
                for e in removed:
                    removed_set.add((min(e[0], e[1]), max(e[0], e[1])))
                
                res_adj = defaultdict(set)
                for e in edges:
                    u, v = e[0], e[1]
                    key = (min(u, v), max(u, v))
                    if key not in removed_set:
                        res_adj[u].add(v)
                        res_adj[v].add(u)
                
                # Find max clique in residual
                cn, clq = find_max_clique(res_adj, vertex_set)
                
                if cn <= target_t:
                    return removed, cn
                
                # Add constraint: at least one edge in this clique must be removed
                clq_edges = []
                for i in range(len(clq)):
                    for j in range(i+1, len(clq)):
                        u, v = clq[i], clq[j]
                        key = (min(u, v), max(u, v))
                        if key in edge_index:
                            clq_edges.append(edge_index[key])
                
                if clq_edges:
                    # Need to remove enough edges so that the clique of size cn 
                    # is broken down to size target_t
                    # A clique of size cn has cn*(cn-1)/2 edges
                    # A clique of size target_t has target_t*(target_t-1)/2 edges
                    # We need to remove at least cn*(cn-1)/2 - target_t*(target_t-1)/2 edges? No.
                    # Actually, to ensure no clique of size target_t+1 exists among these vertices,
                    # for every subset of size target_t+1 of the clique, at least one edge must be removed
                    
                    # Simpler: for the clique vertices, add constraints for all (target_t+1)-subsets
                    if len(clq) <= 20:  # manageable
                        for sub in itertools.combinations(clq, target_t + 1):
                            sub_edges = []
                            for i2 in range(len(sub)):
                                for j2 in range(i2+1, len(sub)):
                                    u2, v2 = sub[i2], sub[j2]
                                    key2 = (min(u2, v2), max(u2, v2))
                                    if key2 in edge_index:
                                        sub_edges.append(edge_index[key2])
                            if sub_edges:
                                model.addConstr(gp.quicksum(x[ei] for ei in sub_edges) >= 1)
                    else:
                        # Just add: sum of removed edges in clique >= some amount
                        # For a clique of size s, to reduce to target_t, 
                        # we need that no (target_t+1)-subset is complete
                        # At minimum, we need to remove >= s - target_t edges
                        # (each removed edge can "cover" at most one vertex being excluded)
                        # Actually that's not quite right. Let's use a simpler bound:
                        # sum of x over clique edges >= 1
                        model.addConstr(gp.quicksum(x[ei] for ei in clq_edges) >= 1)
                
                if time.time() > deadline:
                    break
            
            return None, None
        
        # Binary search for the minimum achievable clique number
        lo, hi = 1, clique_size
        
        # First try target = clique_size - 1 as a quick win
        best_found_obj = clique_size
        best_found_edges = []
        
        while lo < hi and time.time() < deadline:
            mid = (lo + hi) // 2
            removed, cn = solve_with_target(mid)
            if removed is not None and cn is not None and cn <= mid:
                hi = mid
                best_found_obj = cn
                best_found_edges = removed
                if logger:
                    if cn < best_obj:
                        best_obj = cn
                        best_interdicted = removed
                        logger.log(best_obj)
            else:
                lo = mid + 1
            
            if time.time() > deadline:
                break
        
        if best_found_obj < best_obj:
            best_obj = best_found_obj
            best_interdicted = best_found_edges
            if logger:
                logger.log(best_obj)
                
    except Exception as e:
        # Fallback: greedy heuristic
        pass

    # Greedy heuristic as fallback/improvement
    if time.time() < deadline:
        # Try greedy: repeatedly find max clique, remove edge that appears in most cliques
        current_adj = defaultdict(set)
        for e in edges:
            u, v = e[0], e[1]
            current_adj[u].add(v)
            current_adj[v].add(u)
        
        greedy_removed = []
        for b in range(budget):
            if time.time() > deadline:
                break
            cn, clq = find_max_clique(current_adj, vertex_set)
            if cn <= 1:
                break
            
            # Find edge in clique whose removal might help most
            # Heuristic: remove edge between two highest-degree vertices in the clique
            best_edge = None
            best_score = -1
            for i in range(len(clq)):
                for j in range(i+1, len(clq)):
                    u, v = clq[i], clq[j]
                    # Score: number of common neighbors in clique
                    common = len(current_adj[u] & current_adj[v] & set(clq))
                    if common > best_score:
                        best_score = common
                        best_edge = (u, v)
            
            if best_edge:
                u, v = best_edge
                current_adj[u].discard(v)
                current_adj[v].discard(u)
                greedy_removed.append([min(u,v), max(u,v)])
        
        if time.time() < deadline:
            cn_greedy, _ = find_max_clique(current_adj, vertex_set)
            if cn_greedy < best_obj:
                best_obj = cn_greedy
                best_interdicted = greedy_removed
                if logger:
                    logger.log(best_obj)

    result = {
        "objective_value": best_obj,
        "interdicted_edges": best_interdicted
    }

    with open(args.solution_path, 'w') as f:
        json.dump(result, f)

if __name__ == '__main__':
    solve()