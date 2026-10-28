import argparse
import json
import time
import sys
from itertools import combinations
from solution_logger import SolutionLogger

def parse_args():
    parser = argparse.ArgumentParser(description="Clique Interdiction Problem Solver")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    return parser.parse_args()

def load_instance(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def solve_with_gurobi(instance, time_limit, logger):
    """
    MIP formulation for clique interdiction.
    
    Binary variables:
    - x_e: 1 if edge e is interdicted (removed)
    - y_v: 1 if vertex v is in the maximum clique
    - z_s: 1 if clique s is selected (for enumeration approach)
    
    We use a Benders-style approach:
    - Master problem: choose which edges to remove
    - Subproblem: find max clique in remaining graph
    
    Actually, let's use a direct formulation:
    - x_e in {0,1}: edge e is removed
    - y_v in {0,1}: vertex v is in the adversary's clique
    
    The adversary maximizes sum(y_v) subject to: for every pair (u,v) in the clique,
    edge (u,v) must exist and not be removed.
    
    This is a bilevel problem. We'll use a branch-and-cut approach with Gurobi.
    
    Alternative: enumerate all maximal cliques, then formulate as a covering problem.
    """
    import gurobipy as gp
    from gurobipy import GRB
    
    n = instance['num_vertices']
    vertices = instance['vertices']
    edges = instance['edges']
    k = instance['interdiction_budget_k']
    
    # Build adjacency
    vertex_to_idx = {v: i for i, v in enumerate(vertices)}
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
    
    # For small instances, enumerate all maximal cliques
    # Then formulate: minimize t such that every clique of size > t has at least one edge removed
    
    # First, let's try a direct MIP formulation
    # Variables: x_e (edge removed), y_v (vertex in clique)
    # The interdictor minimizes, the adversary maximizes
    # This is bilevel, so we use the dualization or enumeration approach
    
    # Approach: Enumerate all maximal cliques. For each clique C of size s,
    # the interdictor must remove at least one edge from C to destroy it.
    # We want to minimize the max clique size remaining.
    
    # Binary search on the answer t:
    # For a given t, check if we can remove at most k edges such that no clique of size > t remains.
    # This means: for every clique of size > t, at least one edge must be removed.
    
    # Let's enumerate all maximal cliques using Bron-Kerbosch
    adj = [set() for _ in range(n)]
    for u, v in edges:
        ui, vi = vertex_to_idx[u], vertex_to_idx[v]
        adj[ui].add(vi)
        adj[vi].add(ui)
    
    def bron_kerbosch(R, P, X, cliques):
        if not P and not X:
            cliques.append(frozenset(R))
            return
        # Choose pivot
        pivot_candidates = P | X
        if not pivot_candidates:
            return
        pivot = max(pivot_candidates, key=lambda v: len(P & adj[v]))
        for v in list(P - adj[pivot]):
            bron_kerbosch(R | {v}, P & adj[v], X & adj[v], cliques)
            P = P - {v}
            X = X | {v}
    
    maximal_cliques = []
    all_vertices_set = set(range(n))
    bron_kerbosch(set(), all_vertices_set, set(), maximal_cliques)
    
    # Sort cliques by size descending
    maximal_cliques.sort(key=lambda c: len(c), reverse=True)
    
    if not maximal_cliques:
        # No edges, clique number is 1
        solution = {
            "objective_value": 1,
            "interdicted_edges": []
        }
        if logger:
            logger.log_solution(1, solution)
        return solution
    
    max_clique_size = len(maximal_cliques[0])
    
    # If budget is large enough to remove all edges
    if k >= len(edges):
        solution = {
            "objective_value": 1,
            "interdicted_edges": edges[:k]
        }
        if logger:
            logger.log_solution(1, solution)
        return solution
    
    # Binary search on the target clique size t
    # For target t: we need to ensure no clique of size > t remains
    # i.e., for every maximal clique of size > t, remove at least one edge from it
    
    # Map edges to indices
    edge_to_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_to_idx[(min(u, v), max(u, v))] = i
    
    def get_edges_in_clique(clique):
        """Get edge indices within a clique."""
        clique_list = list(clique)
        clique_edges = []
        for i in range(len(clique_list)):
            for j in range(i+1, len(clique_list)):
                u, v = clique_list[i], clique_list[j]
                key = (min(u, v), max(u, v))
                if key in edge_to_idx:
                    clique_edges.append(edge_to_idx[key])
        return clique_edges
    
    # For each maximal clique, precompute its edges
    clique_edges_list = []
    for c in maximal_cliques:
        ce = get_edges_in_clique(c)
        clique_edges_list.append(ce)
    
    def try_target_t(t, remaining_time):
        """
        Check if we can remove at most k edges so that no clique of size > t remains.
        Uses a set cover / hitting set formulation.
        """
        # Cliques that need to be hit (size > t)
        relevant = []
        for i, c in enumerate(maximal_cliques):
            if len(c) > t:
                relevant.append(i)
        
        if not relevant:
            return True, []
        
        # Formulate as ILP: minimize number of edges removed (or just feasibility)
        # x_e in {0,1}, for each relevant clique: sum of x_e for edges in clique >= 1
        # sum x_e <= k
        
        try:
            model = gp.Model("hitting_set")
            model.setParam('OutputFlag', 0)
            model.setParam('Seed', 0)
            model.setParam('Threads', 1)
            model.setParam('TimeLimit', remaining_time)
            
            x = model.addVars(len(edges), vtype=GRB.BINARY, name="x")
            
            # Constraint: total removed <= k
            model.addConstr(gp.quicksum(x[e] for e in range(len(edges))) <= k)
            
            # For each relevant clique, at least one edge must be removed
            for ci in relevant:
                ce = clique_edges_list[ci]
                model.addConstr(gp.quicksum(x[e] for e in ce) >= 1)
            
            model.setObjective(gp.quicksum(x[e] for e in range(len(edges))), GRB.MINIMIZE)
            model.optimize()
            
            if model.Status == GRB.OPTIMAL or (model.Status == GRB.TIME_LIMIT and model.SolCount > 0):
                removed = [e for e in range(len(edges)) if x[e].X > 0.5]
                if len(removed) <= k:
                    return True, removed
            return False, []
        except Exception as e:
            return False, []
    
    # Also need a function to find max clique in remaining graph for verification
    def max_clique_after_removal(removed_edge_indices):
        """Find max clique size after removing specified edges."""
        removed_set = set(removed_edge_indices)
        # Build new adjacency
        new_adj = [set() for _ in range(n)]
        for i, (u, v) in enumerate(edges):
            if i not in removed_set:
                ui, vi = vertex_to_idx[u], vertex_to_idx[v]
                new_adj[ui].add(vi)
                new_adj[vi].add(ui)
        
        # Find max clique using Bron-Kerbosch with pruning
        best = [0]
        
        def bk(R, P, X):
            if best[0] >= len(R) + len(P):
                return
            if not P and not X:
                best[0] = max(best[0], len(R))
                return
            pivot_candidates = P | X
            if not pivot_candidates:
                best[0] = max(best[0], len(R))
                return
            pivot = max(pivot_candidates, key=lambda v: len(P & new_adj[v]))
            for v in list(P - new_adj[pivot]):
                bk(R | {v}, P & new_adj[v], X & new_adj[v])
                P = P - {v}
                X = X | {v}
        
        bk(set(), all_vertices_set, set())
        return best[0]
    
    start_time = time.time()
    
    # Log initial solution (remove no edges)
    initial_obj = max_clique_size
    initial_solution = {
        "objective_value": initial_obj,
        "interdicted_edges": []
    }
    if logger:
        logger.log_solution(initial_obj, initial_solution)
    
    # Binary search on target t
    best_obj = max_clique_size
    best_removed = []
    
    lo, hi = 1, max_clique_size
    
    while lo <= hi:
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed - 1
        if remaining <= 0:
            break
        
        mid = (lo + hi) // 2
        feasible, removed = try_target_t(mid, remaining)
        
        if feasible:
            # Verify
            actual_clique = max_clique_after_removal(removed)
            if actual_clique <= mid:
                best_obj = actual_clique
                best_removed = removed
                solution = {
                    "objective_value": best_obj,
                    "interdicted_edges": [edges[i] for i in best_removed]
                }
                if logger:
                    logger.log_solution(best_obj, solution)
                hi = mid - 1
            else:
                # The hitting set was feasible but didn't reduce enough
                # This shouldn't happen if formulation is correct, but be safe
                lo = mid + 1
        else:
            lo = mid + 1
    
    # If binary search didn't improve, try a more aggressive approach
    if best_obj == max_clique_size and k > 0:
        # Try removing edges from the largest clique
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed - 1
        if remaining > 0:
            # Greedy: remove edges to minimize max clique
            # Try targeting the largest clique
            for target in range(max_clique_size - 1, 0, -1):
                elapsed = time.time() - start_time
                remaining = time_limit - elapsed - 1
                if remaining <= 0:
                    break
                feasible, removed = try_target_t(target, remaining)
                if feasible:
                    actual_clique = max_clique_after_removal(removed)
                    if actual_clique < best_obj:
                        best_obj = actual_clique
                        best_removed = removed
                        solution = {
                            "objective_value": best_obj,
                            "interdicted_edges": [edges[i] for i in best_removed]
                        }
                        if logger:
                            logger.log_solution(best_obj, solution)
                    break
    
    # Final verification
    if best_removed:
        final_clique = max_clique_after_removal(best_removed)
        best_obj = final_clique
    
    solution = {
        "objective_value": best_obj,
        "interdicted_edges": [edges[i] for i in best_removed]
    }
    
    return solution


def solve_large_instance(instance, time_limit, logger):
    """
    For larger instances, use a heuristic approach combined with Gurobi.
    """
    import gurobipy as gp
    from gurobipy import GRB
    
    n = instance['num_vertices']
    vertices = instance['vertices']
    edges = instance['edges']
    k = instance['interdiction_budget_k']
    
    vertex_to_idx = {v: i for i, v in enumerate(vertices)}
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        ui, vi = vertex_to_idx[u], vertex_to_idx[v]
        adj[ui].add(vi)
        adj[vi].add(ui)
    
    # Find max clique using greedy + local search
    def find_max_clique(adj_sets, all_verts):
        """Find a large clique greedily, then try to verify it's maximum."""
        # Greedy: repeatedly add vertex with most connections to current clique
        best_clique = []
        
        # Try multiple starting vertices
        sorted_verts = sorted(all_verts, key=lambda v: len(adj_sets[v]), reverse=True)
        
        for start in sorted_verts[:min(20, len(sorted_verts))]:
            clique = {start}
            candidates = adj_sets[start] & all_verts
            while candidates:
                # Pick vertex with most connections to current clique members
                best_v = max(candidates, key=lambda v: len(adj_sets[v] & clique))
                clique.add(best_v)
                candidates = candidates & adj_sets[best_v]
            if len(clique) > len(best_clique):
                best_clique = list(clique)
        
        return best_clique
    
    all_verts = set(range(n))
    
    # Get initial max clique estimate
    initial_clique = find_max_clique(adj, all_verts)
    initial_obj = len(initial_clique)
    
    solution = {
        "objective_value": initial_obj,
        "interdicted_edges": []
    }
    if logger:
        logger.log_solution(initial_obj, solution)
    
    # Use Gurobi with a direct formulation
    # For moderate sizes, use the bilevel approach linearized
    
    start_time = time.time()
    
    # Try the hitting set approach with enumerated cliques if feasible
    # Otherwise use a direct MIP
    
    # Direct MIP approach:
    # x_e: edge e is removed
    # y_v: vertex v is in adversary's clique
    # Bilevel: min max sum(y_v) s.t. y_u + y_v - 1 <= 1 - x_{uv} for all edges (u,v)
    #          sum(x_e) <= k
    #          sum(y_v) is maximized by adversary
    
    # Linearize: for the adversary's problem, we can use the LP dual
    # But this gets complex. Let's use a simpler approach.
    
    # Approach: Binary search on objective value t
    # For a given t, check feasibility via MIP
    
    edge_to_idx = {}
    for i, (u, v) in enumerate(edges):
        ui, vi = vertex_to_idx[u], vertex_to_idx[v]
        edge_to_idx[(min(ui, vi), max(ui, vi))] = i
    
    def check_feasibility(t, remaining_time):
        """
        Check if we can remove at most k edges so that no clique of size > t exists.
        Use a MIP with clique constraints generated lazily.
        """
        try:
            model = gp.Model("interdiction")
            model.setParam('OutputFlag', 0)
            model.setParam('Seed', 0)
            model.setParam('Threads', 1)
            model.setParam('TimeLimit', remaining_time)
            
            x = model.addVars(len(edges), vtype=GRB.BINARY, name="x")
            y = model.addVars(n, vtype=GRB.BINARY, name="y")
            
            # Budget constraint
            model.addConstr(gp.quicksum(x[e] for e in range(len(edges))) <= k)
            
            # Clique size constraint: sum(y) <= t
            model.addConstr(gp.quicksum(y[v] for v in range(n)) <= t)
            
            # Edge constraints: if both u,v in clique, edge must exist (not removed)
            # y_u + y_v - 1 <= 1 - x_e  =>  y_u + y_v + x_e <= 2
            for i, (u, v) in enumerate(edges):
                ui, vi = vertex_to_idx[u], vertex_to_idx[v]
                model.addConstr(y[ui] + y[vi] + x[i] <= 2)
            
            # The adversary wants to maximize sum(y), interdictor wants to minimize it
            # We want: for ALL possible cliques of size > t, at least one edge is removed
            # Equivalently: max sum(y) s.t. constraints <= t
            # This is: interdictor chooses x, then adversary finds max clique
            # We need: max_clique(remaining graph) <= t
            
            # This formulation checks: does there exist x (|x| <= k) such that
            # the max clique in remaining graph <= t?
            # The constraint sum(y) <= t with the edge constraints means:
            # any clique in the remaining graph has size <= t
            # But we need this for the ADVERSARY's optimal response
            
            # Actually, the correct formulation is:
            # The interdictor picks x. Then the adversary picks y to maximize sum(y)
            # subject to y being a clique in the remaining graph.
            # We want max_y sum(y) <= t for the chosen x.
            
            # This means: for ALL y that form a clique in remaining graph, sum(y) <= t
            # Which is: max_y { sum(y) : y_u + y_v <= 1 + (1-x_e) for all edges, y binary } <= t
            
            # By strong duality or direct reasoning:
            # We need: there exists x with sum(x) <= k such that
            # max { sum(y) : y_u + y_v <= 2 - x_e for all edges (u,v), y binary } <= t
            
            # The inner max is the max clique problem. We can't directly put it in one MIP easily.
            # But we can use the following: the constraint y_u + y_v + x_e <= 2 ensures
            # that if x_e = 1 (edge removed), then u and v can't both be in the clique.
            # And sum(y) <= t means the clique size is at most t.
            
            # But this formulation finds x and y simultaneously, which is wrong.
            # We need: for the chosen x, ALL cliques have size <= t.
            
            # Correct approach: minimize t such that for all cliques C with |C| > t,
            # at least one edge of C is removed.
            
            # This is the hitting set formulation. Let's just use that.
            model.dispose()
            return None
        except:
            return None
    
    # Since the direct MIP is tricky for bilevel, let's use the enumeration approach
    # but with a limit on the number of cliques
    
    # For large instances, use a different strategy:
    # 1. Find large cliques
    # 2. Greedily remove edges to break them
    
    # Greedy approach
    current_adj = [set(a) for a in adj]
    removed_indices = []
    
    def get_max_clique_greedy(adj_sets):
        """Find a large clique."""
        best = []
        sorted_v = sorted(range(n), key=lambda v: len(adj_sets[v]), reverse=True)
        for start in sorted_v[:min(50, n)]:
            clique = {start}
            cands = adj_sets[start].copy()
            while cands:
                # Pick vertex most connected to clique
                v = max(cands, key=lambda u: len(adj_sets[u] & clique))
                clique.add(v)
                cands = cands & adj_sets[v]
            if len(clique) > len(best):
                best = list(clique)
        return best
    
    current_clique = get_max_clique_greedy(current_adj)
    current_obj = len(current_clique)
    
    if current_obj < initial_obj:
        initial_obj = current_obj
    
    # Iteratively remove edges from the current max clique
    best_obj = initial_obj
    best_removed = []
    
    for iteration in range(min(k, len(edges))):
        elapsed = time.time() - start_time
        if elapsed >= time_limit - 1:
            break
        
        # Find current max clique
        current_clique = get_max_clique_greedy(current_adj)
        if len(current_clique) <= 1:
            break
        
        # Try removing each edge in the current clique and see which removal
        # leads to the smallest max clique
        best_next_obj = len(current_clique)
        best_edge_to_remove = None
        
        clique_edges = []
        for i in range(len(current_clique)):
            for j in range(i+1, len(current_clique)):
                u, v = current_clique[i], current_clique[j]
                key = (min(u, v), max(u, v))
                if key in edge_to_idx:
                    eidx = edge_to_idx[key]
                    if eidx not in removed_indices:
                        clique_edges.append(eidx)
        
        # For efficiency, just remove a random/strategic edge from the clique
        # Strategy: remove the edge whose endpoints have the most connections within the clique
        if clique_edges:
            # Score each edge by how "central" it is
            best_score = -1
            best_eidx = clique_edges[0]
            for eidx in clique_edges:
                u, v = edges[eidx]
                ui, vi = vertex_to_idx[u], vertex_to_idx[v]
                # Score: number of common neighbors in clique
                score = len((current_adj[ui] & current_adj[vi]) & set(current_clique))
                if score > best_score:
                    best_score = score
                    best_eidx = eidx
            
            # Remove this edge
            u, v = edges[best_eidx]
            ui, vi = vertex_to_idx[u], vertex_to_idx[v]
            current_adj[ui].discard(vi)
            current_adj[vi].discard(ui)
            removed_indices.append(best_eidx)
            
            # Check new max clique
            new_clique = get_max_clique_greedy(current_adj)
            new_obj = len(new_clique)
            
            if new_obj < best_obj:
                best_obj = new_obj
                best_removed = list(removed_indices)
                solution = {
                    "objective_value": best_obj,
                    "interdicted_edges": [edges[i] for i in best_removed]
                }
                if logger:
                    logger.log_solution(best_obj, solution)
    
    # Final check
    if removed_indices:
        final_clique = get_max_clique_greedy(current_adj)
        if len(final_clique) < best_obj:
            best_obj = len(final_clique)
            best_removed = list(removed_indices)
    
    solution = {
        "objective_value": best_obj,
        "interdicted_edges": [edges[i] for i in best_removed]
    }
    
    return solution


def solve(instance, time_limit, logger):
    """Main solver that dispatches based on instance size."""
    n = instance['num_vertices']
    m = instance['num_edges']
    k = instance['interdiction_budget_k']
    
    start_time = time.time()
    
    # For small instances, use exact approach
    # For larger ones, use heuristic
    
    if n <= 100 and m <= 5000:
        return solve_with_gurobi(instance, time_limit, logger)
    else:
        return solve_large_instance(instance, time_limit, logger)


def main():
    args = parse_args()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    instance = load_instance(args.instance_path)
    
    solution = solve(instance, args.time_limit, logger)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()