import json
import argparse
import time
import math
from solution_logger import SolutionLogger

def total_tour_cost(tour, dist_mat):
    n = len(tour)
    cost = 0
    for i in range(n):
        cost += dist_mat[tour[i]][tour[(i+1) % n]]
    return cost

def run_DP(cluster_order, clusters_list, dist_mat):
    m = len(cluster_order)
    if m == 0:
        return 0.0, []
    if m == 1:
        u = clusters_list[cluster_order[0]][0]
        return 0.0, [u]
    
    dp = [dict() for _ in range(m)]
    cluster0 = cluster_order[0]
    for u in clusters_list[cluster0]:
        dp[0][u] = (0.0, -1)
    
    for i in range(1, m):
        cluster_i = cluster_order[i]
        cluster_prev = cluster_order[i-1]
        for u in clusters_list[cluster_i]:
            best_cost = float('inf')
            best_prev = -1
            for v in clusters_list[cluster_prev]:
                if v not in dp[i-1]:
                    continue
                cost_candidate = dp[i-1][v][0] + dist_mat[v][u]
                if cost_candidate < best_cost:
                    best_cost = cost_candidate
                    best_prev = v
            if best_cost < float('inf'):
                dp[i][u] = (best_cost, best_prev)
    
    best_total = float('inf')
    best_path = None
    cluster_last = cluster_order[m-1]
    for u in clusters_list[cluster_last]:
        if u not in dp[m-1]:
            continue
        path = []
        cur = u
        for j in range(m-1, -1, -1):
            path.append(cur)
            if j > 0:
                cur = dp[j][cur][1]
        start_node = path[-1]
        total_cost = dp[m-1][u][0] + dist_mat[u][start_node]
        if total_cost < best_total:
            best_total = total_cost
            best_path = path
    
    if best_path is None:
        return float('inf'), []
    forward_path = list(reversed(best_path))
    return best_total, forward_path

def run_2opt(tour, dist_mat):
    n = len(tour)
    if n <= 3:
        return tour, False
    
    current_cost = total_tour_cost(tour, dist_mat)
    new_tour = tour[:]
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i+2, n):
                if i == 0 and j == n-1:
                    continue
                a = new_tour[i]
                b = new_tour[(i+1) % n]
                c = new_tour[j]
                d = new_tour[(j+1) % n]
                old_edge_cost = dist_mat[a][b] + dist_mat[c][d]
                new_edge_cost = dist_mat[a][c] + dist_mat[b][d]
                if new_edge_cost < old_edge_cost - 1e-5:
                    if j < n-1:
                        segment = new_tour[i+1:j+1]
                        new_tour = new_tour[:i+1] + segment[::-1] + new_tour[j+1:]
                    else:
                        segment = new_tour[i+1:]
                        new_tour = new_tour[:i+1] + segment[::-1]
                    improved = True
                    break
            if improved:
                break
    
    new_cost = total_tour_cost(new_tour, dist_mat)
    return new_tour, (new_cost < current_cost - 1e-5)

def coordinate_descent_step(cluster_order, current_nodes, clusters_list, dist_mat):
    m = len(cluster_order)
    new_nodes = [None] * m
    for i in range(m):
        prev_index = (i-1) % m
        next_index = (i+1) % m
        if prev_index < i:
            prev_node = new_nodes[prev_index]
        else:
            prev_node = current_nodes[prev_index]
        if next_index < i:
            next_node = new_nodes[next_index]
        else:
            next_node = current_nodes[next_index]
        cluster_i = cluster_order[i]
        best_u = None
        best_cost = float('inf')
        for u in clusters_list[cluster_i]:
            cost_val = dist_mat[prev_node][u] + dist_mat[u][next_node]
            if cost_val < best_cost:
                best_cost = cost_val
                best_u = u
        new_nodes[i] = best_u
    return new_nodes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, "minimize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    n = data['problem_size']['n']
    m_val = data['problem_size']['m']
    nodes_data = data['nodes']
    clusters_data = data['clusters']
    dist_values = data['distance_matrix_upper_triangular']['values']
    
    node_id_to_index = {}
    nodes_index_to_id = [0] * n
    for idx, node in enumerate(nodes_data):
        node_id = node['id']
        node_id_to_index[node_id] = idx
        nodes_index_to_id[idx] = node_id
    
    distance_matrix = [[0] * n for _ in range(n)]
    index = 0
    for i in range(n):
        for j in range(i+1, n):
            d_val = dist_values[index]
            distance_matrix[i][j] = d_val
            distance_matrix[j][i] = d_val
            index += 1
    
    clusters_list = []
    cluster_ids = []
    for cluster in clusters_data:
        cluster_id = cluster['id']
        node_ids = cluster['node_ids']
        node_indices = [node_id_to_index[id] for id in node_ids]
        clusters_list.append(node_indices)
        cluster_ids.append(cluster_id)
    
    cluster_id_to_index = { cid: idx for idx, cid in enumerate(cluster_ids) }
    
    node_cluster = {}
    for cluster_idx, cluster in enumerate(clusters_data):
        cid = cluster['id']
        for node_id in cluster['node_ids']:
            node_idx = node_id_to_index[node_id]
            node_cluster[node_idx] = cid
    
    if m_val == 0:
        output = {
            "objective_value": 0.0,
            "tour": [],
            "visited_nodes": []
        }
        with open(args.solution_path, 'w') as f:
            json.dump(output, f)
        return
    
    C = [[10**9] * m_val for _ in range(m_val)]
    for i in range(m_val):
        for j in range(i+1, m_val):
            min_d = 10**9
            for u in clusters_list[i]:
                for v in clusters_list[j]:
                    if distance_matrix[u][v] < min_d:
                        min_d = distance_matrix[u][v]
            C[i][j] = min_d
            C[j][i] = min_d
    
    best_tour_clusters = None
    best_cost_clusters = 10**18
    for start in range(m_val):
        tour = [start]
        unvisited = set(range(m_val))
        unvisited.remove(start)
        current = start
        while unvisited:
            best_next = None
            best_d = 10**9
            for j in unvisited:
                if C[current][j] < best_d:
                    best_d = C[current][j]
                    best_next = j
            if best_next is None:
                break
            tour.append(best_next)
            unvisited.remove(best_next)
            current = best_next
        total_cost = 0
        for i in range(m_val):
            total_cost += C[tour[i]][tour[(i+1) % m_val]]
        if total_cost < best_cost_clusters:
            best_cost_clusters = total_cost
            best_tour_clusters = tour
    
    if best_tour_clusters is None:
        best_tour_clusters = list(range(m_val))
    
    improved = True
    while improved:
        improved = False
        for i in range(m_val):
            for j in range(i+2, m_val):
                if i == 0 and j == m_val-1:
                    continue
                a = best_tour_clusters[i]
                b = best_tour_clusters[(i+1) % m_val]
                c = best_tour_clusters[j]
                d = best_tour_clusters[(j+1) % m_val]
                old_cost = C[a][b] + C[c][d]
                new_cost = C[a][c] + C[b][d]
                if new_cost < old_cost - 1e-5:
                    if j < m_val-1:
                        segment = best_tour_clusters[i+1:j+1]
                        best_tour_clusters = best_tour_clusters[:i+1] + segment[::-1] + best_tour_clusters[j+1:]
                    else:
                        segment = best_tour_clusters[i+1:]
                        best_tour_clusters = best_tour_clusters[:i+1] + segment[::-1]
                    improved = True
                    break
            if improved:
                break
    
    initial_cost, initial_selected_nodes = run_DP(best_tour_clusters, clusters_list, distance_matrix)
    best_obj = initial_cost
    best_tour_nodes = initial_selected_nodes
    if logger:
        logger.log(best_obj)
    
    current_tour_order = best_tour_clusters
    current_nodes = initial_selected_nodes
    
    start_time = time.time()
    time_limit = args.time_limit
    
    while time.time() - start_time < time_limit - 0.5:
        improvement = False
        
        new_nodes = coordinate_descent_step(current_tour_order, current_nodes, clusters_list, distance_matrix)
        new_cost = total_tour_cost(new_nodes, distance_matrix)
        if new_cost < best_obj - 1e-5:
            best_obj = new_cost
            best_tour_nodes = new_nodes
            current_nodes = new_nodes
            improvement = True
            if logger:
                logger.log(best_obj)
        
        new_tour_nodes, improved_2opt = run_2opt(best_tour_nodes, distance_matrix)
        if improved_2opt:
            new_cluster_order = [node_cluster[node] for node in new_tour_nodes]
            new_cluster_order_indices = [cluster_id_to_index[cid] for cid in new_cluster_order]
            new_cost, new_selected_nodes = run_DP(new_cluster_order_indices, clusters_list, distance_matrix)
            if new_cost < best_obj - 1e-5:
                best_obj = new_cost
                best_tour_nodes = new_selected_nodes
                current_tour_order = new_cluster_order_indices
                current_nodes = new_selected_nodes
                improvement = True
                if logger:
                    logger.log(best_obj)
        
        if not improvement:
            break
    
    tour_node_ids = [nodes_index_to_id[node_idx] for node_idx in best_tour_nodes]
    visited_node_ids = sorted(tour_node_ids)
    
    output = {
        "objective_value": best_obj,
        "tour": tour_node_ids,
        "visited_nodes": visited_node_ids
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(output, f)

if __name__ == '__main__':
    main()