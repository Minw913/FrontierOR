import json
import argparse
import time
import math
import random
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description='Generalized TSP Solver')
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()
    time_limit = args.time_limit

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['problem_size']['n']
    m = instance['problem_size']['m']
    nodes = instance['nodes']
    clusters_data = instance['clusters']

    # Build distance matrix
    dist_values = instance['distance_matrix_upper_triangular']['values']
    dist = [[0] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_values[idx]
            dist[i][j] = d
            dist[j][i] = d
            idx += 1

    node_ids = [node['id'] for node in nodes]
    node_id_to_idx = {node['id']: i for i, node in enumerate(nodes)}

    # Cluster structure
    cluster_nodes = []
    node_to_cluster = [0] * n
    for k, cluster in enumerate(clusters_data):
        nindices = [node_id_to_idx[nid] for nid in cluster['node_ids']]
        cluster_nodes.append(nindices)
        for ni in nindices:
            node_to_cluster[ni] = k

    def tour_cost(tour):
        cost = 0
        for i in range(len(tour)):
            cost += dist[tour[i]][tour[(i + 1) % len(tour)]]
        return cost

    def make_solution_dict(tour):
        cost = tour_cost(tour)
        return {
            "objective_value": float(cost),
            "tour": [node_ids[i] for i in tour],
            "visited_nodes": [node_ids[i] for i in tour]
        }

    # === CONSTRUCTION HEURISTICS ===

    def nearest_neighbor(start_node):
        tour = [start_node]
        visited_clusters = {node_to_cluster[start_node]}
        current = start_node
        while len(tour) < m:
            best_next = -1
            best_d = float('inf')
            for ci in range(m):
                if ci in visited_clusters:
                    continue
                for ni in cluster_nodes[ci]:
                    if dist[current][ni] < best_d:
                        best_d = dist[current][ni]
                        best_next = ni
            tour.append(best_next)
            visited_clusters.add(node_to_cluster[best_next])
            current = best_next
        return tour

    def cheapest_insertion():
        # Find closest pair from different clusters
        best_pair = None
        best_d = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                if node_to_cluster[i] != node_to_cluster[j] and dist[i][j] < best_d:
                    best_d = dist[i][j]
                    best_pair = (i, j)
        if best_pair is None:
            return None
        tour = [best_pair[0], best_pair[1]]
        visited_clusters = {node_to_cluster[best_pair[0]], node_to_cluster[best_pair[1]]}

        while len(tour) < m:
            best_insert_cost = float('inf')
            best_insert_pos = -1
            best_node = -1
            for ci in range(m):
                if ci in visited_clusters:
                    continue
                for ni in cluster_nodes[ci]:
                    for pos in range(len(tour)):
                        prev_n = tour[pos]
                        next_n = tour[(pos + 1) % len(tour)]
                        ic = dist[prev_n][ni] + dist[ni][next_n] - dist[prev_n][next_n]
                        if ic < best_insert_cost:
                            best_insert_cost = ic
                            best_insert_pos = pos + 1
                            best_node = ni
            tour.insert(best_insert_pos, best_node)
            visited_clusters.add(node_to_cluster[best_node])
        return tour

    # === LOCAL SEARCH ===

    def node_exchange(tour, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for pos in range(m):
                if time.time() >= deadline:
                    break
                cur = tour[pos]
                ck = node_to_cluster[cur]
                prev_n = tour[(pos - 1) % m]
                next_n = tour[(pos + 1) % m]
                cur_cost = dist[prev_n][cur] + dist[cur][next_n]
                for ni in cluster_nodes[ck]:
                    if ni == cur:
                        continue
                    new_cost = dist[prev_n][ni] + dist[ni][next_n]
                    if new_cost < cur_cost:
                        tour[pos] = ni
                        cur = ni
                        cur_cost = new_cost
                        improved = True
        return tour

    def two_opt(tour, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(m - 1):
                if time.time() >= deadline:
                    break
                for j in range(i + 2, m):
                    if i == 0 and j == m - 1:
                        continue
                    a, b = tour[i], tour[i + 1]
                    c, d = tour[j], tour[(j + 1) % m]
                    if dist[a][c] + dist[b][d] < dist[a][b] + dist[c][d]:
                        tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                        improved = True
        return tour

    def or_opt(tour, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for seg_len in [1, 2, 3]:
                if time.time() >= deadline:
                    break
                for i in range(m):
                    if time.time() >= deadline:
                        break
                    seg_indices = [(i + k) % m for k in range(seg_len)]
                    seg = [tour[idx] for idx in seg_indices]
                    prev_idx = (i - 1) % m
                    next_idx = (i + seg_len) % m
                    prev_n = tour[prev_idx]
                    next_n = tour[next_idx]
                    remove_gain = dist[prev_n][seg[0]] + dist[seg[-1]][next_n] - dist[prev_n][next_n]
                    
                    for j in range(seg_len, m):
                        insert_after = (i + j) % m
                        if insert_after in seg_indices:
                            continue
                        a = tour[insert_after]
                        b = tour[(insert_after + 1) % m]
                        insert_cost = dist[a][seg[0]] + dist[seg[-1]][b] - dist[a][b]
                        if insert_cost < remove_gain - 1e-10:
                            # Remove segment
                            remaining = [tour[k] for k in range(m) if k not in seg_indices]
                            # Find new insert position
                            new_pos = None
                            for p in range(len(remaining)):
                                if remaining[p] == a:
                                    new_pos = p + 1
                                    break
                            if new_pos is None:
                                new_pos = 0
                            new_tour = remaining[:new_pos] + seg + remaining[new_pos:]
                            tour[:] = new_tour
                            improved = True
                            break
                if improved:
                    break
        return tour

    def full_local_search(tour, deadline):
        while time.time() < deadline:
            old_cost = tour_cost(tour)
            tour = node_exchange(tour, deadline)
            if time.time() >= deadline:
                break
            tour = two_opt(tour, deadline)
            if time.time() >= deadline:
                break
            new_cost = tour_cost(tour)
            if new_cost >= old_cost:
                break
        return tour

    # === MAIN SOLVER ===
    
    best_tour = None
    best_cost = float('inf')

    def update_best(tour):
        nonlocal best_tour, best_cost
        c = tour_cost(tour)
        if c < best_cost:
            best_cost = c
            best_tour = tour[:]
            if logger:
                logger.log_solution(best_cost, make_solution_dict(best_tour))

    # Phase 1: Construction
    # Nearest neighbor from multiple starting points
    random.seed(42)
    num_nn_starts = min(m, 5) if time_limit > 10 else 1
    start_candidates = []
    
    # Centroid-based start
    cx = sum(nodes[i]['x'] for i in range(n)) / n
    cy = sum(nodes[i]['y'] for i in range(n)) / n
    centroid_node = min(range(n), key=lambda i: (nodes[i]['x'] - cx)**2 + (nodes[i]['y'] - cy)**2)
    start_candidates.append(centroid_node)
    
    # Random starts from different clusters
    used_clusters = {node_to_cluster[centroid_node]}
    for _ in range(num_nn_starts - 1):
        avail_clusters = [c for c in range(m) if c not in used_clusters]
        if not avail_clusters:
            break
        rc = random.choice(avail_clusters)
        rn = random.choice(cluster_nodes[rc])
        start_candidates.append(rn)
        used_clusters.add(rc)

    for start_node in start_candidates:
        if time.time() - start_time > time_limit * 0.3:
            break
        tour = nearest_neighbor(start_node)
        update_best(tour)

    # Cheapest insertion
    if time.time() - start_time < time_limit * 0.3:
        tour = cheapest_insertion()
        if tour:
            update_best(tour)

    # Phase 2: Local search on best solution
    if best_tour:
        deadline = start_time + time_limit * 0.6
        best_tour = full_local_search(best_tour, deadline)
        update_best(best_tour)

    # Phase 3: Iterated Local Search
    ils_deadline = start_time + time_limit * 0.75
    if best_tour and time.time() < ils_deadline:
        current_tour = best_tour[:]
        current_cost = best_cost
        
        while time.time() < ils_deadline:
            # Perturbation: randomly swap some nodes
            perturbed = current_tour[:]
            num_swaps = max(1, m // 10)
            for _ in range(num_swaps):
                pos = random.randint(0, m - 1)
                ck = node_to_cluster[perturbed[pos]]
                if len(cluster_nodes[ck]) > 1:
                    perturbed[pos] = random.choice(cluster_nodes[ck])
            
            # Local search on perturbed solution
            perturbed = node_exchange(perturbed, time.time() + 2)
            if time.time() >= ils_deadline:
                break
            perturbed = two_opt(perturbed, time.time() + 2)
            
            new_cost = tour_cost(perturbed)
            if new_cost < current_cost:
                current_tour = perturbed
                current_cost = new_cost
                update_best(current_tour)
            elif new_cost < current_cost * 1.05:
                # Accept slightly worse to escape local optima
                current_tour = perturbed
                current_cost = new_cost

    # Phase 4: Gurobi MIP (if time permits and instance is manageable)
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5 and n <= 800:
        try:
            gurobi_result = solve_gurobi(dist, cluster_nodes, node_to_cluster, m, n,
                                          node_ids, remaining_time - 2, logger,
                                          best_tour, best_cost)
            if gurobi_result and gurobi_result[0] < best_cost:
                best_cost = gurobi_result[0]
                best_tour = gurobi_result[1]
                if logger:
                    logger.log_solution(best_cost, make_solution_dict(best_tour))
        except Exception as e:
            pass

    # Final local search polish
    if best_tour and time.time() < start_time + time_limit - 1:
        best_tour = full_local_search(best_tour, start_time + time_limit - 0.5)
        update_best(best_tour)

    # Write solution
    solution = make_solution_dict(best_tour)
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


def solve_gurobi(dist, cluster_nodes, node_to_cluster, m, n, node_ids, time_limit, logger, initial_tour, initial_cost):
    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model("GTSP")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1, time_limit)
    model.Params.OutputFlag = 0

    # Variables
    x = {}
    for i in range(n):
        x[i] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}")

    y = {}
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if node_to_cluster[i] != node_to_cluster[j]:
                y[i, j] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")
                edge_list.append((i, j))

    model.update()

    # Objective
    obj = gp.LinExpr()
    for i, j in edge_list:
        obj += dist[i][j] * y[i, j]
    model.setObjective(obj, GRB.MINIMIZE)

    # Cluster constraints
    for k in range(m):
        model.addConstr(gp.quicksum(x[i] for i in cluster_nodes[k]) == 1, name=f"cluster_{k}")

    # Degree constraints
    for i in range(n):
        edges_i = []
        for j in range(n):
            if j != i and node_to_cluster[j] != node_to_cluster[i]:
                e = (min(i, j), max(i, j))
                if e in y:
                    edges_i.append(e)
        model.addConstr(gp.quicksum(y[e] for e in edges_i) == 2 * x[i], name=f"degree_{i}")

    # Initial solution (MIP start)
    if initial_tour:
        tour_set = set(initial_tour)
        for i in range(n):
            x[i].Start = 1.0 if i in tour_set else 0.0
        
        tour_edges = set()
        for pos in range(len(initial_tour)):
            a = initial_tour[pos]
            b = initial_tour[(pos + 1) % len(initial_tour)]
            tour_edges.add((min(a, b), max(a, b)))
        
        for i, j in edge_list:
            y[i, j].Start = 1.0 if (i, j) in tour_edges else 0.0

    # Lazy constraint callback
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            x_val = model.cbGetSolution(x)
            y_val = model.cbGetSolution(y)

            selected = [i for i in range(n) if x_val[i] > 0.5]
            if len(selected) <= 2:
                return

            # Build adjacency
            adj = defaultdict(set)
            for i, j in edge_list:
                if y_val[i, j] > 0.5:
                    adj[i].add(j)
                    adj[j].add(i)

            # Find connected components
            visited = set()
            for node in selected:
                if node not in visited:
                    component = []
                    stack = [node]
                    while stack:
                        v = stack.pop()
                        if v in visited:
                            continue
                        visited.add(v)
                        component.append(v)
                        for u in adj.get(v, set()):
                            if u not in visited:
                                stack.append(u)

                    if 2 <= len(component) < len(selected):
                        comp_set = set(component)
                        edges_in_comp = [(i, j) for i, j in edge_list if i in comp_set and j in comp_set]
                        if edges_in_comp:
                            model.cbLazy(gp.quicksum(y[e] for e in edges_in_comp) <= len(component) - 1)

    model._x = x
    model._y = y
    model.Params.LazyConstraints = 1
    model.optimize(callback)

    if model.SolCount > 0:
        try:
            x_val = model.getAttr('X', x)
            selected = [i for i in range(n) if x_val[i] > 0.5]

            if len(selected) == m:
                y_val = model.getAttr('X', y)
                adj = defaultdict(list)
                for i, j in edge_list:
                    if y_val[i, j] > 0.5:
                        adj[i].append(j)
                        adj[j].append(i)

                # Reconstruct tour
                tour = [selected[0]]
                visited_set = {selected[0]}
                current = selected[0]
                while len(tour) < m:
                    neighbors = [nb for nb in adj[current] if nb not in visited_set]
                    if not neighbors:
                        break
                    current = neighbors[0]
                    tour.append(current)
                    visited_set.add(current)

                if len(tour) == m:
                    cost = sum(dist[tour[i]][tour[(i + 1) % m]] for i in range(m))
                    return (cost, tour)
        except Exception:
            pass

    return None


if __name__ == '__main__':
    main()