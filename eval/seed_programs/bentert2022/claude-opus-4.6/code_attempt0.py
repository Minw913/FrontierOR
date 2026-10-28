import argparse
import json
import time
import heapq
from collections import defaultdict

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    start_time = time.time()
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    n = instance['n']
    vertices = instance['vertices']
    
    coords = {}
    for v in vertices:
        coords[v['id']] = (v['x'], v['y'])
    
    ids = sorted(coords.keys())
    id_to_idx = {vid: i for i, vid in enumerate(ids)}
    
    def dist_sq(i, j):
        xi, yi = coords[ids[i]], coords[ids[j]]  # wrong, coords uses vid
        return -1  # placeholder
    
    # Precompute squared distances
    # For efficiency with large n, compute on the fly
    xs = [coords[ids[i]][0] for i in range(n)]
    ys = [coords[ids[i]][1] for i in range(n)]
    
    def weight(i, j):
        dx = xs[i] - xs[j]
        dy = ys[i] - ys[j]
        return dx * dx + dy * dy
    
    def compute_objective(adj_list):
        """Given adjacency list (idx -> list of (neighbor_idx, weight)), compute objective."""
        total = 0.0
        vertex_costs = {}
        for i in range(n):
            max_w = 0.0
            for (j, w) in adj_list[i]:
                if w > max_w:
                    max_w = w
            vertex_costs[i] = max_w
            total += max_w
        return total, vertex_costs
    
    def edges_to_adj(edges):
        adj = [[] for _ in range(n)]
        for (u, v, w) in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        return adj
    
    # Compute MST using Prim's algorithm
    def compute_mst():
        in_mst = [False] * n
        min_edge = [float('inf')] * n
        min_from = [-1] * n
        min_edge[0] = 0
        heap = [(0, 0)]  # (weight, node)
        mst_edges = []
        
        while heap:
            w, u = heapq.heappop(heap)
            if in_mst[u]:
                continue
            in_mst[u] = True
            if min_from[u] != -1:
                mst_edges.append((min_from[u], u, w))
            
            for v in range(n):
                if not in_mst[v]:
                    wt = weight(u, v)
                    if wt < min_edge[v]:
                        min_edge[v] = wt
                        min_from[v] = u
                        heapq.heappush(heap, (wt, v))
        
        return mst_edges
    
    mst_edges = compute_mst()
    
    best_edges = mst_edges[:]
    best_adj = edges_to_adj(best_edges)
    best_obj, best_vcosts = compute_objective(best_adj)
    
    if logger:
        logger.log(best_obj)
    
    # Build tree structure for LCA / path queries
    # For local search: try adding a non-tree edge and removing the edge on the path
    # that gives the best improvement.
    
    # Tree adjacency with edge indices
    tree_adj = [[] for _ in range(n)]
    for ei, (u, v, w) in enumerate(best_edges):
        tree_adj[u].append((v, w, ei))
        tree_adj[v].append((u, w, ei))
    
    # BFS to set up parent structure
    parent = [-1] * n
    parent_edge = [-1] * n
    parent_weight = [0.0] * n
    depth = [0] * n
    order = []
    visited = [False] * n
    queue = [0]
    visited[0] = True
    while queue:
        nq = []
        for u in queue:
            order.append(u)
            for (v, w, ei) in tree_adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    parent_edge[v] = ei
                    parent_weight[v] = w
                    depth[v] = depth[u] + 1
                    nq.append(v)
        queue = nq
    
    def find_path(u, v):
        """Find path from u to v in the tree, return list of (edge_index, weight, node_pair)."""
        path = []
        a, b = u, v
        while a != b:
            if depth[a] > depth[b]:
                path.append((parent_edge[a], parent_weight[a], (a, parent[a])))
                a = parent[a]
            else:
                path.append((parent_edge[b], parent_weight[b], (b, parent[b])))
                b = parent[b]
        return path
    
    # Local search: edge swap
    # For each non-tree edge (u,v,w_new), adding it creates a cycle.
    # We can remove any edge on the path from u to v in the tree.
    # We want to find the removal that maximizes objective decrease.
    
    def try_edge_swaps(current_edges, time_limit_remaining):
        nonlocal best_edges, best_adj, best_obj, best_vcosts
        
        edge_set = set()
        for (u, v, w) in current_edges:
            edge_set.add((min(u,v), max(u,v)))
        
        improved = True
        iteration = 0
        
        while improved and time.time() - start_time < args.time_limit - 1:
            improved = False
            iteration += 1
            
            # Rebuild tree structures
            cur_adj = edges_to_adj(current_edges)
            cur_obj, cur_vcosts = compute_objective(cur_adj)
            
            # Rebuild tree for path queries
            t_adj = [[] for _ in range(n)]
            for ei, (u, v, w) in enumerate(current_edges):
                t_adj[u].append((v, w, ei))
                t_adj[v].append((u, w, ei))
            
            par = [-1] * n
            par_edge = [-1] * n
            par_weight = [0.0] * n
            dep = [0] * n
            vis = [False] * n
            q = [0]
            vis[0] = True
            while q:
                nq = []
                for u in q:
                    for (v, w, ei) in t_adj[u]:
                        if not vis[v]:
                            vis[v] = True
                            par[v] = u
                            par_edge[v] = ei
                            par_weight[v] = w
                            dep[v] = dep[u] + 1
                            nq.append(v)
                q = nq
            
            def get_path(u, v):
                path = []
                a, b = u, v
                while a != b:
                    if dep[a] > dep[b]:
                        path.append((par_edge[a], par_weight[a], a, par[a]))
                        a = par[a]
                    else:
                        path.append((par_edge[b], par_weight[b], b, par[b]))
                        b = par[b]
                return path
            
            # Compute max weight per node
            max_w = [0.0] * n
            second_max_w = [0.0] * n
            max_w_edge = [-1] * n  # edge index giving max weight
            
            for i in range(n):
                weights_with_ei = []
                for (j, w, ei) in t_adj[i]:
                    weights_with_ei.append((w, ei))
                weights_with_ei.sort(reverse=True)
                if len(weights_with_ei) > 0:
                    max_w[i] = weights_with_ei[0][0]
                    max_w_edge[i] = weights_with_ei[0][1]
                if len(weights_with_ei) > 1:
                    second_max_w[i] = weights_with_ei[1][0]
            
            # Try candidate non-tree edges
            # For large n, we can't try all O(n^2) edges, so sample or use heuristics
            candidates = []
            
            if n <= 300:
                # Try all non-tree edges
                for i in range(n):
                    if time.time() - start_time > args.time_limit - 1:
                        break
                    for j in range(i+1, n):
                        if (i, j) not in edge_set:
                            candidates.append((i, j, weight(i, j)))
            else:
                # Sample: for each node, try nearest neighbors not in tree
                # Sort edges by weight and try lightest non-tree edges
                # Use a limited set
                for i in range(n):
                    if time.time() - start_time > args.time_limit - 1:
                        break
                    # compute distances to all other nodes, pick closest non-neighbors
                    dists = []
                    tree_neighbors = set()
                    for (v, w, ei) in t_adj[i]:
                        tree_neighbors.add(v)
                    for j in range(n):
                        if j != i and j not in tree_neighbors:
                            dists.append((weight(i, j), j))
                    dists.sort()
                    for w_ij, j in dists[:min(20, len(dists))]:
                        ii, jj = min(i, j), max(i, j)
                        candidates.append((ii, jj, w_ij))
            
            # Sort candidates by weight (lighter edges more likely to help)
            candidates.sort(key=lambda x: x[2])
            
            best_swap = None
            best_improvement = 0
            
            for (u, v, w_new) in candidates:
                if time.time() - start_time > args.time_limit - 1:
                    break
                
                path = get_path(u, v)
                if not path:
                    continue
                
                # Adding edge (u,v,w_new) means:
                # - u's max weight might increase if w_new > max_w[u]
                # - v's max weight might increase if w_new > max_w[v]
                # Cost increase from adding new edge:
                new_u_cost = max(max_w[u], w_new)
                new_v_cost = max(max_w[v], w_new)
                add_cost = (new_u_cost - max_w[u]) + (new_v_cost - max_w[v])
                
                # For each edge on the path, try removing it
                for (ei, ew, a, b) in path:
                    # Removing edge ei (between a and b with weight ew)
                    # Node a: if ei == max_w_edge[a], new max = second_max_w[a], else same
                    # Node b: similarly
                    
                    save_a = 0
                    if max_w_edge[a] == ei:
                        save_a = max_w[a] - second_max_w[a]
                    
                    save_b = 0
                    if max_w_edge[b] == ei:
                        save_b = max_w[b] - second_max_w[b]
                    
                    # But if a == u or a == v, the new edge also affects a
                    # Need to be careful about overlaps
                    
                    improvement = save_a + save_b - add_cost
                    
                    # Handle overlap: if a == u, the add_cost for u already considered
                    # Actually, let me recompute properly
                    # Affected nodes: u (new edge), v (new edge), a (removed edge), b (removed edge)
                    # Some may overlap
                    
                    affected = set([u, v, a, b])
                    old_total = sum(max_w[x] for x in affected)
                    
                    # New costs
                    new_costs = {}
                    for x in affected:
                        # Current max and edge
                        if max_w_edge[x] == ei:
                            # Removing the max edge
                            base = second_max_w[x]
                        else:
                            base = max_w[x]
                        
                        if x == u or x == v:
                            base = max(base, w_new)
                        
                        new_costs[x] = base
                    
                    new_total = sum(new_costs[x] for x in affected)
                    imp = old_total - new_total
                    
                    if imp > best_improvement + 1e-9:
                        best_improvement = imp
                        best_swap = (u, v, w_new, ei)
            
            if best_swap is not None and best_improvement > 1e-9:
                u, v, w_new, ei_remove = best_swap
                # Perform swap
                new_edges = []
                for ei, e in enumerate(current_edges):
                    if ei != ei_remove:
                        new_edges.append(e)
                new_edges.append((u, v, w_new))
                
                uu, vv = min(u,v), max(u,v)
                re = current_edges[ei_remove]
                ru, rv = min(re[0], re[1]), max(re[0], re[1])
                edge_set.discard((ru, rv))
                edge_set.add((uu, vv))
                
                current_edges = new_edges
                
                new_adj = edges_to_adj(current_edges)
                new_obj, new_vcosts = compute_objective(new_adj)
                
                if new_obj < best_obj - 1e-9:
                    best_edges = current_edges[:]
                    best_adj = new_adj
                    best_obj = new_obj
                    best_vcosts = new_vcosts
                    if logger:
                        logger.log(best_obj)
                
                improved = True
        
        return current_edges
    
    # Run local search
    current_edges = try_edge_swaps(best_edges, args.time_limit - (time.time() - start_time))
    
    # Also try: for each edge in tree, see if we can find a cheaper replacement
    # Additional heuristic: try building tree with different root strategies
    
    # Try a "power-aware" spanning tree construction
    # Greedy: sort all edges by weight, add edge if it connects two components
    # This is just Kruskal's MST - already done via Prim
    
    # Try another approach: minimize bottleneck-aware cost
    # For each node, we want to minimize its max incident edge
    # This is related to the minimum power spanning tree problem
    
    # Additional local search: try adding multiple non-tree edges 
    # (making it a spanning subgraph, not necessarily a tree)
    # Adding an edge can only increase costs for its endpoints but might allow
    # removing edges that save more elsewhere - but in a tree, adding and removing
    # keeps it a tree. For a general subgraph, we need connectivity.
    
    # Let's also try: for each vertex, see if we can reduce its max weight
    # by rerouting through a different path
    
    # Phase 2: Try non-tree spanning subgraphs
    # Actually, the problem says "spanning subgraph" not "spanning tree"
    # So we can have cycles! But adding edges only increases vertex costs.
    # Wait - no, we could have a graph where adding an edge allows removing 
    # another edge that was causing high cost.
    # But our swap search already does this within trees.
    
    # Let's try multiple random spanning trees and pick the best
    if time.time() - start_time < args.time_limit - 2 and n <= 500:
        import random
        random.seed(42)
        
        for trial in range(min(50, max(1, int((args.time_limit - (time.time() - start_time)) / 2)))):
            if time.time() - start_time > args.time_limit - 1:
                break
            
            # Random Prim: add randomness to edge selection
            in_tree = [False] * n
            start_node = random.randint(0, n-1)
            in_tree[start_node] = True
            tree_nodes = [start_node]
            trial_edges = []
            
            for _ in range(n - 1):
                # For each non-tree node, find best edge to tree
                best_e = None
                best_cost = float('inf')
                
                candidates_list = []
                for v in range(n):
                    if in_tree[v]:
                        continue
                    for u in tree_nodes:
                        w = weight(u, v)
                        candidates_list.append((w, u, v))
                
                if not candidates_list:
                    break
                
                # Sort and pick from top few with some randomness
                candidates_list.sort()
                k = min(3, len(candidates_list))
                idx = random.randint(0, k - 1)
                w, u, v = candidates_list[idx]
                
                in_tree[v] = True
                tree_nodes.append(v)
                trial_edges.append((u, v, w))
            
            if len(trial_edges) == n - 1:
                t_adj = edges_to_adj(trial_edges)
                t_obj, t_vcosts = compute_objective(t_adj)
                
                if t_obj < best_obj - 1e-9:
                    best_edges = trial_edges[:]
                    best_adj = t_adj
                    best_obj = t_obj
                    best_vcosts = t_vcosts
                    if logger:
                        logger.log(best_obj)
                    
                    # Run local search on this new tree
                    if time.time() - start_time < args.time_limit - 2:
                        current_edges = try_edge_swaps(trial_edges, args.time_limit - (time.time() - start_time))
    
    # Phase 3: Try a "power-aware" tree construction
    # Idea: Build tree by always adding the edge that minimizes the increase in total power cost
    if time.time() - start_time < args.time_limit - 2:
        in_tree = [False] * n
        in_tree[0] = True
        tree_nodes = {0}
        power_cost = [0.0] * n  # current max weight for each node
        trial_edges = []
        
        for _ in range(n - 1):
            if time.time() - start_time > args.time_limit - 1:
                break
            
            best_increase = float('inf')
            best_e = None
            
            for v in range(n):
                if in_tree[v]:
                    continue
                for u in tree_nodes:
                    w = weight(u, v)
                    # Adding edge (u,v,w):
                    # v goes from cost 0 to w (but v wasn't in tree, so its cost becomes w)
                    # u's cost becomes max(power_cost[u], w)
                    increase = w + max(power_cost[u], w) - power_cost[u]
                    # = w + max(0, w - power_cost[u])
                    
                    if increase < best_increase:
                        best_increase = increase
                        best_e = (u, v, w)
            
            if best_e is None:
                break
            
            u, v, w = best_e
            in_tree[v] = True
            tree_nodes.add(v)
            trial_edges.append((u, v, w))
            power_cost[v] = w
            power_cost[u] = max(power_cost[u], w)
        
        if len(trial_edges) == n - 1:
            t_adj = edges_to_adj(trial_edges)
            t_obj, t_vcosts = compute_objective(t_adj)
            
            if t_obj < best_obj - 1e-9:
                best_edges = trial_edges[:]
                best_adj = t_adj
                best_obj = t_obj
                best_vcosts = t_vcosts
                if logger:
                    logger.log(best_obj)
                
                if time.time() - start_time < args.time_limit - 2:
                    current_edges = try_edge_swaps(best_edges, args.time_limit - (time.time() - start_time))
    
    # Phase 3b: Power-aware from each starting node
    if time.time() - start_time < args.time_limit - 2 and n <= 200:
        for start_node in range(1, n):
            if time.time() - start_time > args.time_limit - 1:
                break
            
            in_tree = [False] * n
            in_tree[start_node] = True
            tree_nodes_set = {start_node}
            pc = [0.0] * n
            trial_edges = []
            
            for _ in range(n - 1):
                best_increase = float('inf')
                best_e = None
                
                for v in range(n):
                    if in_tree[v]:
                        continue
                    for u in tree_nodes_set:
                        w = weight(u, v)
                        increase = w + max(0, w - pc[u])
                        
                        if increase < best_increase:
                            best_increase = increase
                            best_e = (u, v, w)
                
                if best_e is None:
                    break
                
                u, v, w = best_e
                in_tree[v] = True
                tree_nodes_set.add(v)
                trial_edges.append((u, v, w))
                pc[v] = w
                pc[u] = max(pc[u], w)
            
            if len(trial_edges) == n - 1:
                t_adj = edges_to_adj(trial_edges)
                t_obj, t_vcosts = compute_objective(t_adj)
                
                if t_obj < best_obj - 1e-9:
                    best_edges = trial_edges[:]
                    best_adj = t_adj
                    best_obj = t_obj
                    best_vcosts = t_vcosts
                    if logger:
                        logger.log(best_obj)
    
    # Run final local search on best solution
    if time.time() - start_time < args.time_limit - 2:
        current_edges = try_edge_swaps(best_edges, args.time_limit - (time.time() - start_time))
    
    # Format output
    selected_edges = []
    for (u, v, w) in best_edges:
        selected_edges.append([ids[u], ids[v]])
    
    vertex_costs = {}
    final_adj = edges_to_adj(best_edges)
    for i in range(n):
        max_w = 0.0
        for (j, w) in final_adj[i]:
            if w > max_w:
                max_w = w
        vertex_costs[str(ids[i])] = max_w
    
    obj = sum(vertex_costs[str(ids[i])] for i in range(n))
    
    solution = {
        "objective_value": obj,
        "selected_edges": selected_edges,
        "vertex_costs": vertex_costs
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    solve()