import argparse
import json
import time
import random
from array import array

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()
    deadline = start_time + args.time_limit

    def time_up():
        return time.time() > deadline

    with open(args.instance_path) as f:
        instance = json.load(f)

    n = instance['problem_size']['n']
    m = instance['problem_size']['m']
    nodes_data = instance['nodes']
    clusters_data = instance['clusters']
    flat = array('i', instance['distance_matrix_upper_triangular']['values'])

    # Build node -> cluster mapping
    node_cluster = [0] * n
    for node in nodes_data:
        node_cluster[node['id']] = node['cluster']

    # Map cluster ids to 0..m-1
    cluster_ids = [cd['id'] for cd in clusters_data]
    id_to_idx = {cid: idx for idx, cid in enumerate(cluster_ids)}

    clusters = [[] for _ in range(m)]
    for node in nodes_data:
        c_idx = id_to_idx[node['cluster']]
        clusters[c_idx].append(node['id'])

    node_cluster_idx = [0] * n
    for i in range(n):
        node_cluster_idx[i] = id_to_idx[node_cluster[i]]

    # Precompute row starts for upper-triangular distance matrix
    row_start = [0] * n
    for i in range(n):
        row_start[i] = i * n - i * (i + 1) // 2

    def get_dist(i, j):
        if i == j:
            return 0
        if i > j:
            i, j = j, i
        return flat[row_start[i] + j - i - 1]

    # Precompute minimum distances between clusters
    INF = 10**9
    cluster_dist = [[INF] * m for _ in range(m)]
    for i in range(n):
        ci = node_cluster_idx[i]
        ri = row_start[i]
        for j in range(i + 1, n):
            cj = node_cluster_idx[j]
            if ci != cj:
                d = flat[ri + j - i - 1]
                if d < cluster_dist[ci][cj]:
                    cluster_dist[ci][cj] = d
                    cluster_dist[cj][ci] = d

    def tour_cost(order, nodes):
        cost = 0
        mm = len(order)
        for i in range(mm):
            cost += get_dist(nodes[i], nodes[(i + 1) % mm])
        return cost

    def optimize_nodes(order, nodes):
        """Coordinate descent for node selection given a fixed cluster order."""
        mm = len(order)
        improved = True
        while improved and not time_up():
            improved = False
            for i in range(mm):
                c = order[i]
                left = nodes[(i - 1) % mm]
                right = nodes[(i + 1) % mm]
                current = nodes[i]
                best = current
                best_cost = get_dist(left, current) + get_dist(current, right)
                for v in clusters[c]:
                    if v == current:
                        continue
                    cost = get_dist(left, v) + get_dist(v, right)
                    if cost < best_cost:
                        best = v
                        best_cost = cost
                        improved = True
                        nodes[i] = best
                        break
        return nodes

    def two_opt(order, nodes):
        """Standard 2-opt on the node tour (reverses segments)."""
        mm = len(order)
        improved = True
        while improved and not time_up():
            improved = False
            for i in range(mm):
                for j in range(i + 1, mm):
                    a = nodes[i]
                    b = nodes[(i + 1) % mm]
                    c = nodes[j]
                    d = nodes[(j + 1) % mm]
                    delta = get_dist(a, c) + get_dist(b, d) - get_dist(a, b) - get_dist(c, d)
                    if delta < 0:
                        nodes[i + 1:j + 1] = nodes[i + 1:j + 1][::-1]
                        order[i + 1:j + 1] = order[i + 1:j + 1][::-1]
                        improved = True
                        break
                if improved:
                    break
        return order, nodes

    def local_search(order, nodes):
        nodes = optimize_nodes(order, nodes)
        if time_up():
            return order, nodes
        order, nodes = two_opt(order, nodes)
        if time_up():
            return order, nodes
        nodes = optimize_nodes(order, nodes)
        return order, nodes

    def nearest_neighbor_order(start_cluster):
        visited = [False] * m
        order = [start_cluster]
        visited[start_cluster] = True
        current = start_cluster
        for _ in range(m - 1):
            best_c = -1
            best_d = INF
            for c in range(m):
                if not visited[c]:
                    d = cluster_dist[current][c]
                    if d < best_d:
                        best_d = d
                        best_c = c
            order.append(best_c)
            visited[best_c] = True
            current = best_c
        return order

    best_order = None
    best_nodes = None
    best_cost = INF

    # Initial constructions from multiple starting clusters
    start_clusters = range(m) if m <= 60 else random.sample(range(m), 60)
    for sc in start_clusters:
        if time_up():
            break
        order = nearest_neighbor_order(sc)
        nodes = [clusters[c][0] for c in order]
        order, nodes = local_search(order, nodes)
        cost = tour_cost(order, nodes)
        if cost < best_cost:
            best_cost = cost
            best_order = order[:]
            best_nodes = nodes[:]
            if logger:
                sol = {
                    "objective_value": float(best_cost),
                    "tour": best_nodes,
                    "visited_nodes": best_nodes
                }
                logger.log_solution(best_cost, sol)

    if best_order is None:
        # Fallback if time was too short to finish any construction
        order = list(range(m))
        nodes = [clusters[c][0] for c in order]
        best_order = order[:]
        best_nodes = nodes[:]
        best_cost = tour_cost(order, nodes)
        if logger:
            sol = {
                "objective_value": float(best_cost),
                "tour": best_nodes,
                "visited_nodes": best_nodes
            }
            logger.log_solution(best_cost, sol)

    # Perturbation-based metaheuristic
    random.seed(0)
    no_improve = 0
    while not time_up():
        order = best_order[:]
        nodes = best_nodes[:]
        mm = len(order)

        # Double-bridge perturbation for larger tours, random swaps otherwise
        if mm >= 8:
            cuts = sorted(random.sample(range(1, mm), 4))
            a, b, c, d = cuts
            order = order[:a] + order[c:d] + order[b:c] + order[a:b] + order[d:]
            nodes = nodes[:a] + nodes[c:d] + nodes[b:c] + nodes[a:b] + nodes[d:]
        else:
            for _ in range(3):
                i, j = random.sample(range(mm), 2)
                order[i], order[j] = order[j], order[i]
                nodes[i], nodes[j] = nodes[j], nodes[i]

        # Randomly alter a few node selections
        for _ in range(random.randint(1, max(1, mm // 10))):
            i = random.randrange(mm)
            c = order[i]
            if len(clusters[c]) > 1:
                nodes[i] = random.choice(clusters[c])

        order, nodes = local_search(order, nodes)
        if time_up():
            break
        cost = tour_cost(order, nodes)
        if cost < best_cost - 1e-9:
            best_cost = cost
            best_order = order[:]
            best_nodes = nodes[:]
            no_improve = 0
            if logger:
                sol = {
                    "objective_value": float(best_cost),
                    "tour": best_nodes,
                    "visited_nodes": best_nodes
                }
                logger.log_solution(best_cost, sol)
        else:
            no_improve += 1
            if no_improve > 50:
                # Random restart to diversify search
                order = list(range(m))
                random.shuffle(order)
                nodes = [clusters[c][0] for c in order]
                order, nodes = local_search(order, nodes)
                if not time_up():
                    cost = tour_cost(order, nodes)
                    if cost < best_cost:
                        best_cost = cost
                        best_order = order[:]
                        best_nodes = nodes[:]
                        if logger:
                            sol = {
                                "objective_value": float(best_cost),
                                "tour": best_nodes,
                                "visited_nodes": best_nodes
                            }
                            logger.log_solution(best_cost, sol)
                no_improve = 0

    # Write final solution
    solution = {
        "objective_value": float(best_cost),
        "tour": best_nodes,
        "visited_nodes": best_nodes
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()