import argparse
import json
import time
import sys
from collections import defaultdict, deque
from itertools import combinations

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

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    graph = instance['graph']
    settings = instance['problem_settings']

    num_vertices = graph['num_vertices']
    vertices = graph['vertices']
    edges = graph['edges']

    relaxed_clique_type = settings['relaxed_clique_type']
    decomposition_type = settings['decomposition_type']
    connectivity_required = settings['connectivity_required']

    # Get parameter
    param_s = settings.get('parameter_s', None)
    param_k = settings.get('parameter_k', None)
    param_gamma = settings.get('parameter_gamma', None)

    # Build adjacency structures
    adj = defaultdict(set)
    edge_set = set()
    for e in edges:
        u, v = e[0], e[1]
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))

    vertex_set = set(vertices)

    def is_connected(group):
        if len(group) <= 1:
            return True
        group_set = set(group)
        start = group[0]
        visited = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if neighbor in group_set and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return len(visited) == len(group_set)

    def is_valid_splex(group, s):
        """Check if group is a valid s-plex."""
        if len(group) <= 1:
            return True
        group_set = set(group)
        n = len(group)
        for v in group:
            neighbors_in_group = len(adj[v] & group_set)
            if neighbors_in_group < n - s:
                return False
        return True

    def is_valid_relaxed_clique(group):
        """Check if a group is a valid relaxed clique of the specified type."""
        if len(group) <= 1:
            return True
        group_set = set(group)
        n = len(group)

        if relaxed_clique_type == 's-plex':
            s = param_s
            for v in group:
                neighbors_in_group = len(adj[v] & group_set)
                if neighbors_in_group < n - s:
                    return False
            return True

        elif relaxed_clique_type == 'k-core':
            k = param_k if param_k is not None else param_s
            for v in group:
                neighbors_in_group = len(adj[v] & group_set)
                if neighbors_in_group < k:
                    return False
            return True

        elif relaxed_clique_type == 's-clique':
            s = param_s
            # BFS from each vertex in original graph, check distances
            for v in group:
                dist = bfs_distances_original(v, group_set, s)
                for u in group:
                    if u != v and dist.get(u, float('inf')) > s:
                        return False
            return True

        elif relaxed_clique_type == 's-club':
            s = param_s
            # BFS within induced subgraph
            for v in group:
                dist = bfs_distances_induced(v, group_set, s)
                for u in group:
                    if u != v and dist.get(u, float('inf')) > s:
                        return False
            return True

        elif relaxed_clique_type == 'gamma-quasi-clique':
            gamma = param_gamma if param_gamma is not None else float(param_s)
            edge_count = 0
            for u, v in combinations(group, 2):
                if (min(u, v), max(u, v)) in edge_set:
                    edge_count += 1
            max_edges = n * (n - 1) / 2
            if max_edges == 0:
                return True
            return edge_count / max_edges >= gamma

        elif relaxed_clique_type == 's-defective-clique':
            s = param_s
            edge_count = 0
            for u, v in combinations(group, 2):
                if (min(u, v), max(u, v)) in edge_set:
                    edge_count += 1
            max_edges = n * (n - 1) // 2
            return edge_count >= max_edges - s

        elif relaxed_clique_type == 'k-block':
            k = param_k if param_k is not None else param_s
            return vertex_connectivity(group, group_set) >= k

        elif relaxed_clique_type == 's-bundle':
            s = param_s
            return vertex_connectivity(group, group_set) >= n - s

        return True

    def bfs_distances_original(start, target_set, max_dist):
        """BFS in original graph, return distances to vertices in target_set."""
        dist = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            d = dist[node]
            if d >= max_dist:
                continue
            for neighbor in adj[node]:
                if neighbor not in dist:
                    dist[neighbor] = d + 1
                    queue.append(neighbor)
        return dist

    def bfs_distances_induced(start, group_set, max_dist):
        """BFS within induced subgraph."""
        dist = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            d = dist[node]
            if d >= max_dist:
                continue
            for neighbor in adj[node]:
                if neighbor in group_set and neighbor not in dist:
                    dist[neighbor] = d + 1
                    queue.append(neighbor)
        return dist

    def vertex_connectivity(group, group_set):
        """Compute vertex connectivity of induced subgraph using max-flow."""
        if len(group) <= 1:
            return len(group) - 1 if len(group) == 1 else 0
        # Simple approach: try removing vertices
        n = len(group)
        if n <= 2:
            # Check if edge exists
            u, v = group[0], group[1]
            if v in adj[u]:
                return 1
            return 0
        # For small groups, use brute force
        min_conn = n - 1
        for size in range(0, n - 1):
            found_disconnect = False
            for removed in combinations(group, size):
                removed_set = set(removed)
                remaining = [v for v in group if v not in removed_set]
                if len(remaining) <= 1:
                    continue
                remaining_set = set(remaining)
                # Check connectivity of remaining
                visited = {remaining[0]}
                queue = deque([remaining[0]])
                while queue:
                    node = queue.popleft()
                    for neighbor in adj[node]:
                        if neighbor in remaining_set and neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                if len(visited) < len(remaining):
                    return size
            if time.time() - start_time > args.time_limit * 0.8:
                return size  # conservative
        return n - 1

    def check_group(group):
        """Check if group is valid (relaxed clique + connectivity if required)."""
        if not is_valid_relaxed_clique(group):
            return False
        if connectivity_required and not is_connected(group):
            return False
        return True

    # For s-plex specifically, use a greedy approach
    # General greedy: try to form large groups

    is_partitioning = (decomposition_type == 'partitioning')

    # Start with trivial solution: each vertex in its own group
    best_solution = [[v] for v in vertices]
    best_obj = len(best_solution)
    if logger:
        logger.log(best_obj)

    # Greedy approach: try to merge vertices into larger groups
    # For s-plex: every vertex must have at least |group| - s neighbors in group

    if relaxed_clique_type == 's-plex':
        s = param_s

        # Greedy approach for s-plex partitioning
        # Sort vertices by degree (descending) to form large cliques first
        remaining = set(vertices)
        groups = []

        # Try greedy expansion
        vertex_list = sorted(vertices, key=lambda v: len(adj[v] & remaining), reverse=True)

        def can_add_to_splex(group_set, group_list, new_vertex, s):
            """Check if adding new_vertex to group maintains s-plex property."""
            new_size = len(group_set) + 1
            # Check new_vertex has enough neighbors
            neighbors_in_group = len(adj[new_vertex] & group_set)
            if neighbors_in_group < new_size - s:
                return False
            # Check existing members still have enough neighbors
            for v in group_list:
                neighbors_count = len(adj[v] & group_set) - (1 if new_vertex in adj[v] else 0)
                # After adding new_vertex, v's neighbor count in new group
                new_neighbors = (len(adj[v] & group_set)) + (1 if new_vertex in adj[v] else 0)
                if new_neighbors < new_size - s:
                    return False
            return True

        def greedy_splex_partition(vertex_order, remaining_set):
            remaining = set(remaining_set)
            groups = []
            
            while remaining:
                # Pick seed vertex
                seed = max(remaining, key=lambda v: len(adj[v] & remaining))
                group = [seed]
                group_set = {seed}
                remaining.remove(seed)

                # Try to add neighbors
                candidates = sorted(
                    [v for v in remaining if v in adj[seed]],
                    key=lambda v: len(adj[v] & remaining),
                    reverse=True
                )

                for cand in candidates:
                    if cand not in remaining:
                        continue
                    if can_add_to_splex(group_set, group, cand, s):
                        # Also check connectivity if required
                        if connectivity_required:
                            if not (adj[cand] & group_set):
                                continue
                        group.append(cand)
                        group_set.add(cand)
                        remaining.discard(cand)

                # Second pass: try non-neighbors too
                candidates2 = sorted(
                    [v for v in remaining],
                    key=lambda v: len(adj[v] & group_set),
                    reverse=True
                )
                for cand in candidates2:
                    if cand not in remaining:
                        continue
                    if can_add_to_splex(group_set, group, cand, s):
                        if connectivity_required:
                            if not (adj[cand] & group_set):
                                continue
                        group.append(cand)
                        group_set.add(cand)
                        remaining.discard(cand)

                groups.append(group)
            return groups

        groups = greedy_splex_partition(vertex_list, set(vertices))

        if len(groups) < best_obj:
            best_solution = groups
            best_obj = len(groups)
            if logger:
                logger.log(best_obj)

        # Try multiple random orderings for improvement
        import random
        random.seed(42)

        iteration = 0
        while time.time() - start_time < args.time_limit * 0.7:
            iteration += 1
            # Shuffle vertices
            shuffled = list(vertices)
            random.shuffle(shuffled)
            
            remaining = set(vertices)
            groups = []
            
            while remaining:
                # Pick seed - use different strategies
                if iteration % 3 == 0:
                    seed = min(remaining, key=lambda v: len(adj[v] & remaining))
                elif iteration % 3 == 1:
                    seed = max(remaining, key=lambda v: len(adj[v] & remaining))
                else:
                    seed = random.choice(list(remaining))
                
                group = [seed]
                group_set = {seed}
                remaining.remove(seed)

                # Candidates: prioritize by neighbors in group
                changed = True
                while changed:
                    changed = False
                    candidates = sorted(
                        [v for v in remaining],
                        key=lambda v: len(adj[v] & group_set),
                        reverse=True
                    )
                    for cand in candidates:
                        if cand not in remaining:
                            continue
                        if can_add_to_splex(group_set, group, cand, s):
                            if connectivity_required:
                                if not (adj[cand] & group_set):
                                    continue
                            group.append(cand)
                            group_set.add(cand)
                            remaining.discard(cand)
                            changed = True

                groups.append(group)

            if len(groups) < best_obj:
                best_solution = groups
                best_obj = len(groups)
                if logger:
                    logger.log(best_obj)

        # Try local search: merge small groups
        elapsed = time.time() - start_time
        if elapsed < args.time_limit * 0.9:
            improved = True
            while improved and time.time() - start_time < args.time_limit * 0.9:
                improved = False
                # Sort groups by size
                best_solution.sort(key=len)
                
                for i in range(len(best_solution)):
                    if time.time() - start_time > args.time_limit * 0.9:
                        break
                    if len(best_solution[i]) == 0:
                        continue
                    
                    for j in range(i + 1, len(best_solution)):
                        if len(best_solution[j]) == 0:
                            continue
                        
                        # Try to merge group i into group j
                        merged = best_solution[j] + best_solution[i]
                        if check_group(merged):
                            best_solution[j] = merged
                            best_solution[i] = []
                            improved = True
                            break
                
                # Clean up empty groups
                best_solution = [g for g in best_solution if len(g) > 0]
                
                if improved:
                    best_obj = len(best_solution)
                    if logger:
                        logger.log(best_obj)

        # Try to move singleton vertices into existing groups
        if time.time() - start_time < args.time_limit * 0.95:
            improved = True
            while improved and time.time() - start_time < args.time_limit * 0.95:
                improved = False
                singletons = [i for i, g in enumerate(best_solution) if len(g) == 1]
                
                for si in singletons:
                    if time.time() - start_time > args.time_limit * 0.95:
                        break
                    if len(best_solution[si]) == 0:
                        continue
                    v = best_solution[si][0]
                    
                    # Try to add v to another group
                    best_target = -1
                    best_target_size = 0
                    
                    for j in range(len(best_solution)):
                        if j == si or len(best_solution[j]) == 0:
                            continue
                        
                        group_set = set(best_solution[j])
                        
                        # Quick check for s-plex
                        new_size = len(group_set) + 1
                        neighbors_in = len(adj[v] & group_set)
                        if neighbors_in < new_size - s:
                            continue
                        
                        # Check all existing members
                        valid = True
                        for u in best_solution[j]:
                            u_neighbors = len(adj[u] & group_set) + (1 if v in adj[u] else 0)
                            if u_neighbors < new_size - s:
                                valid = False
                                break
                        
                        if not valid:
                            continue
                        
                        if connectivity_required and not (adj[v] & group_set):
                            continue
                        
                        if len(best_solution[j]) > best_target_size:
                            best_target = j
                            best_target_size = len(best_solution[j])
                    
                    if best_target >= 0:
                        best_solution[best_target].append(v)
                        best_solution[si] = []
                        improved = True
                
                best_solution = [g for g in best_solution if len(g) > 0]
                
                if improved:
                    best_obj = len(best_solution)
                    if logger:
                        logger.log(best_obj)

    else:
        # General case for other relaxed clique types
        # Use simple greedy
        remaining = set(vertices)
        groups = []
        
        while remaining and time.time() - start_time < args.time_limit * 0.8:
            seed = max(remaining, key=lambda v: len(adj[v] & remaining))
            group = [seed]
            group_set = {seed}
            remaining.remove(seed)
            
            candidates = sorted(remaining, key=lambda v: len(adj[v] & group_set), reverse=True)
            
            for cand in candidates:
                if cand not in remaining:
                    continue
                if time.time() - start_time > args.time_limit * 0.8:
                    break
                test_group = group + [cand]
                test_set = group_set | {cand}
                
                if check_group(test_group):
                    group.append(cand)
                    group_set.add(cand)
                    remaining.discard(cand)
            
            groups.append(group)
        
        # Add remaining as singletons
        for v in remaining:
            groups.append([v])
        
        if len(groups) < best_obj:
            best_solution = groups
            best_obj = len(groups)
            if logger:
                logger.log(best_obj)

    # Verify solution
    all_vertices_covered = set()
    for g in best_solution:
        for v in g:
            all_vertices_covered.add(v)
    
    assert all_vertices_covered == set(vertices), "Not all vertices covered!"

    if is_partitioning:
        # Check no duplicates
        total = sum(len(g) for g in best_solution)
        assert total == len(vertices), f"Partitioning violated: {total} vs {len(vertices)}"

    output = {
        "objective_value": float(best_obj),
        "solution": best_solution
    }

    with open(args.solution_path, 'w') as f:
        json.dump(output, f)

if __name__ == '__main__':
    main()