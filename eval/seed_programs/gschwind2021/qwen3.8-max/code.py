import json
import argparse
import time
import sys
import random
from collections import defaultdict, deque
from itertools import combinations

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    return parser.parse_args()

def is_s_plex(group_set, adj, s):
    """Check if a group is an s-plex."""
    n = len(group_set)
    if n <= s:
        return True
    for v in group_set:
        neighbors_in_group = len(adj[v] & group_set)
        if neighbors_in_group < n - s:
            return False
    return True

def is_connected(group_set, adj):
    """Check if the induced subgraph of group is connected."""
    if len(group_set) <= 1:
        return True
    start = next(iter(group_set))
    visited = {start}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v in group_set and v not in visited:
                visited.add(v)
                queue.append(v)
    return len(visited) == len(group_set)

def can_add_vertex(group_set, v, adj, s, connectivity_required):
    """Check if adding v to group_set maintains s-plex property (and connectivity)."""
    new_group = group_set | {v}
    if not is_s_plex(new_group, adj, s):
        return False
    if connectivity_required and not is_connected(new_group, adj):
        return False
    return True

def greedy_build_s_plex(adj, vertices, s, connectivity_required, seed_order=None, time_left=None):
    """Greedily build a large s-plex."""
    if seed_order is None:
        order = list(vertices)
    else:
        order = seed_order
    
    if not order:
        return set()
    
    group = {order[0]}
    
    for v in order[1:]:
        if time_left and time_left() <= 0:
            break
        if v in group:
            continue
        if can_add_vertex(group, v, adj, s, connectivity_required):
            group.add(v)
    
    return group

def find_s_plex_from_vertex(adj, start, s, connectivity_required, vertices_sorted):
    """Build an s-plex starting from a specific vertex."""
    group = {start}
    
    # Sort candidates by degree (descending) for greedy
    candidates = [v for v in vertices_sorted if v != start]
    
    for v in candidates:
        if v in group:
            continue
        if can_add_vertex(group, v, adj, s, connectivity_required):
            group.add(v)
    
    return group

def bron_kerbosch(adj, vertices, R, P, X, cliques, max_cliques=5000, time_left=None):
    """Find maximal cliques using Bron-Kerbosch with pivoting."""
    if time_left and time_left() <= 0:
        return
    if len(cliques) >= max_cliques:
        return
    if not P and not X:
        if len(R) >= 2:
            cliques.append(set(R))
        return
    
    # Choose pivot
    union_px = P | X
    if union_px:
        pivot = max(union_px, key=lambda v: len(adj[v] & P))
    else:
        pivot = None
    
    candidates = P - (adj[pivot] if pivot is not None else set())
    
    for v in list(candidates):
        if time_left and time_left() <= 0:
            return
        if len(cliques) >= max_cliques:
            return
        bron_kerbosch(adj, vertices, R | {v}, P & adj[v], X & adj[v], cliques, max_cliques, time_left)
        P = P - {v}
        X = X | {v}

def extend_clique_to_s_plex(clique, adj, vertices, s, connectivity_required):
    """Try to extend a clique to a larger s-plex."""
    group = set(clique)
    
    # Try adding vertices not in the clique
    candidates = sorted([v for v in vertices if v not in group], 
                       key=lambda v: len(adj[v] & group), reverse=True)
    
    for v in candidates:
        if can_add_vertex(group, v, adj, s, connectivity_required):
            group.add(v)
    
    return group

def generate_candidate_s_plexes(adj, vertices, s, connectivity_required, start_time, time_limit):
    """Generate a set of candidate s-plexes."""
    def time_left():
        return time_limit - (time.time() - start_time)
    
    candidates = []
    candidate_set = set()  # To avoid duplicates
    
    vertices_sorted = sorted(vertices, key=lambda v: len(adj[v]), reverse=True)
    
    # Strategy 1: Single vertices (always valid)
    for v in vertices:
        key = frozenset([v])
        if key not in candidate_set:
            candidate_set.add(key)
            candidates.append({v})
    
    # Strategy 2: Greedy from each vertex
    for v in vertices_sorted:
        if time_left() <= 1:
            break
        group = find_s_plex_from_vertex(adj, v, s, connectivity_required, vertices_sorted)
        key = frozenset(group)
        if key not in candidate_set:
            candidate_set.add(key)
            candidates.append(group)
    
    # Strategy 3: Find cliques and extend them
    if time_left() > 2 and s >= 1:
        cliques = []
        bron_kerbosch(adj, vertices, set(), set(vertices), set(), cliques, max_cliques=2000, time_left=time_left)
        
        for clique in cliques:
            if time_left() <= 1:
                break
            extended = extend_clique_to_s_plex(clique, adj, vertices, s, connectivity_required)
            key = frozenset(extended)
            if key not in candidate_set:
                candidate_set.add(key)
                candidates.append(extended)
    
    # Strategy 4: Random orderings
    if time_left() > 2:
        random.seed(42)
        for _ in range(min(20, len(vertices))):
            if time_left() <= 1:
                break
            shuffled = list(vertices)
            random.shuffle(shuffled)
            group = greedy_build_s_plex(adj, vertices, s, connectivity_required, shuffled, time_left)
            key = frozenset(group)
            if key not in candidate_set:
                candidate_set.add(key)
                candidates.append(group)
    
    # Strategy 5: Pairs and small groups
    if time_left() > 1:
        for u in vertices:
            if time_left() <= 0.5:
                break
            for v in adj[u]:
                if v > u:  # avoid duplicates
                    pair = {u, v}
                    if is_s_plex(pair, adj, s):
                        if not connectivity_required or is_connected(pair, adj):
                            key = frozenset(pair)
                            if key not in candidate_set:
                                candidate_set.add(key)
                                candidates.append(pair)
    
    return candidates

def greedy_set_cover(vertices, candidates, partitioning):
    """Greedy set cover/partition."""
    uncovered = set(vertices)
    selected = []
    
    while uncovered:
        best_group = None
        best_coverage = -1
        
        for group in candidates:
            coverage = len(group & uncovered)
            if coverage > best_coverage:
                best_coverage = coverage
                best_group = group
        
        if best_group is None or best_coverage == 0:
            # Add remaining vertices as singletons
            for v in uncovered:
                selected.append({v})
            break
        
        selected.append(best_group)
        uncovered -= best_group
    
    return selected

def solve_with_gurobi(vertices, candidates, partitioning, start_time, time_limit):
    """Solve set cover/partition ILP with Gurobi."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None
    
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time <= 1:
        return None
    
    n = len(vertices)
    m = len(candidates)
    
    vertex_to_idx = {v: i for i, v in enumerate(vertices)}
    
    try:
        model = gp.Model("s_plex_decomposition")
        model.setParam('OutputFlag', 0)
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', max(1, remaining_time - 2))
        
        # Decision variables
        x = model.addVars(m, vtype=GRB.BINARY, name="x")
        
        # Objective: minimize number of groups
        model.setObjective(gp.quicksum(x[j] for j in range(m)), GRB.MINIMIZE)
        
        # Constraints
        for i, v in enumerate(vertices):
            covering_groups = [j for j in range(m) if v in candidates[j]]
            if partitioning:
                model.addConstr(
                    gp.quicksum(x[j] for j in covering_groups) == 1,
                    name=f"partition_{v}"
                )
            else:
                model.addConstr(
                    gp.quicksum(x[j] for j in covering_groups) >= 1,
                    name=f"cover_{v}"
                )
        
        model.optimize()
        
        if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
            if model.SolCount > 0:
                selected = []
                for j in range(m):
                    if x[j].X > 0.5:
                        selected.append(candidates[j])
                return selected
        
        return None
    except Exception as e:
        return None

def local_search_improve(selected, adj, vertices, s, connectivity_required, partitioning, start_time, time_limit):
    """Try to improve solution by merging groups."""
    def time_left():
        return time_limit - (time.time() - start_time)
    
    improved = True
    while improved and time_left() > 1:
        improved = False
        # Try merging pairs of groups
        for i in range(len(selected)):
            if time_left() <= 0.5:
                break
            for j in range(i + 1, len(selected)):
                if time_left() <= 0.5:
                    break
                merged = selected[i] | selected[j]
                if is_s_plex(merged, adj, s):
                    if not connectivity_required or is_connected(merged, adj):
                        selected[i] = merged
                        selected.pop(j)
                        improved = True
                        break
            if improved:
                break
    
    return selected

def main():
    args = parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    start_time = time.time()
    
    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    graph = instance["graph"]
    settings = instance["problem_settings"]
    
    vertices = graph["vertices"]
    edges = graph["edges"]
    s = settings["parameter_s"]
    decomposition_type = settings["decomposition_type"]
    connectivity_required = settings["connectivity_required"]
    partitioning = (decomposition_type == "partitioning")
    
    # Build adjacency
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Ensure all vertices are in adj
    for v in vertices:
        if v not in adj:
            adj[v] = set()
    
    def time_left():
        return args.time_limit - (time.time() - start_time)
    
    # Phase 1: Generate candidate s-plexes
    candidates = generate_candidate_s_plexes(adj, vertices, s, connectivity_required, start_time, args.time_limit)
    
    # Phase 2: Get initial greedy solution
    greedy_solution = greedy_set_cover(vertices, candidates, partitioning)
    
    # Verify and fix greedy solution
    covered = set()
    for g in greedy_solution:
        covered |= g
    
    for v in vertices:
        if v not in covered:
            greedy_solution.append({v})
    
    # Log initial solution
    best_solution = greedy_solution
    best_obj = len(best_solution)
    
    solution_dict = {
        "objective_value": float(best_obj),
        "solution": [sorted(list(g)) for g in best_solution]
    }
    
    if logger:
        logger.log_solution(best_obj, solution_dict)
    
    # Phase 3: Try ILP
    if time_left() > 3:
        ilp_solution = solve_with_gurobi(vertices, candidates, partitioning, start_time, args.time_limit)
        
        if ilp_solution is not None:
            # Verify ILP solution
            covered = set()
            valid = True
            for g in ilp_solution:
                if not is_s_plex(g, adj, s):
                    valid = False
                    break
                if connectivity_required and not is_connected(g, adj):
                    valid = False
                    break
                covered |= g
            
            if valid:
                # Check coverage
                all_covered = all(v in covered for v in vertices)
                if all_covered:
                    if len(ilp_solution) < best_obj:
                        best_solution = ilp_solution
                        best_obj = len(best_solution)
                        solution_dict = {
                            "objective_value": float(best_obj),
                            "solution": [sorted(list(g)) for g in best_solution]
                        }
                        if logger:
                            logger.log_solution(best_obj, solution_dict)
    
    # Phase 4: Local search improvement
    if time_left() > 2:
        improved_solution = local_search_improve(
            list(best_solution), adj, vertices, s, connectivity_required, 
            partitioning, start_time, args.time_limit
        )
        
        # Verify
        covered = set()
        valid = True
        for g in improved_solution:
            if not is_s_plex(g, adj, s):
                valid = False
                break
            if connectivity_required and not is_connected(g, adj):
                valid = False
                break
            covered |= g
        
        if valid and all(v in covered for v in vertices):
            if len(improved_solution) < best_obj:
                best_solution = improved_solution
                best_obj = len(best_solution)
                solution_dict = {
                    "objective_value": float(best_obj),
                    "solution": [sorted(list(g)) for g in best_solution]
                }
                if logger:
                    logger.log_solution(best_obj, solution_dict)
    
    # Phase 5: Additional greedy improvement - try building larger s-plexes from scratch
    if time_left() > 2:
        # Try a different greedy: repeatedly find largest s-plex
        remaining = set(vertices)
        new_solution = []
        
        while remaining and time_left() > 1:
            # Find the best s-plex from remaining vertices
            best_group = None
            best_size = 0
            
            for v in sorted(remaining, key=lambda x: len(adj[x] & remaining), reverse=True)[:min(20, len(remaining))]:
                group = {v}
                # Greedily add vertices
                for u in sorted(remaining - {v}, key=lambda x: len(adj[x] & group), reverse=True):
                    if can_add_vertex(group, u, adj, s, connectivity_required):
                        group.add(u)
                
                if len(group) > best_size:
                    best_size = len(group)
                    best_group = group
            
            if best_group and best_size > 0:
                new_solution.append(best_group)
                remaining -= best_group
            else:
                # Add singletons for remaining
                for v in remaining:
                    new_solution.append({v})
                remaining = set()
        
        if remaining:
            for v in remaining:
                new_solution.append({v})
        
        # Verify
        covered = set()
        valid = True
        for g in new_solution:
            if not is_s_plex(g, adj, s):
                valid = False
                break
            if connectivity_required and not is_connected(g, adj):
                valid = False
                break
            covered |= g
        
        if valid and all(v in covered for v in vertices) and len(new_solution) < best_obj:
            best_solution = new_solution
            best_obj = len(best_solution)
            solution_dict = {
                "objective_value": float(best_obj),
                "solution": [sorted(list(g)) for g in best_solution]
            }
            if logger:
                logger.log_solution(best_obj, solution_dict)
    
    # Final verification and fix
    covered = set()
    for g in best_solution:
        covered |= g
    
    for v in vertices:
        if v not in covered:
            best_solution.append({v})
            best_obj = len(best_solution)
    
    # Write final solution
    solution_dict = {
        "objective_value": float(best_obj),
        "solution": [sorted(list(g)) for g in best_solution]
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f, indent=2)
    
    if logger:
        logger.log_solution(best_obj, solution_dict)

if __name__ == "__main__":
    main()