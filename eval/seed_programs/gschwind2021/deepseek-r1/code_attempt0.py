import json
import argparse
import heapq
import collections
import time
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, "minimize") if args.log_path else None
    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    graph_data = instance['graph']
    num_vertices = graph_data['num_vertices']
    num_edges = graph_data['num_edges']
    vertices = graph_data['vertices']
    edges = graph_data['edges']
    
    problem_settings = instance['problem_settings']
    relaxed_clique_type = problem_settings['relaxed_clique_type']
    parameter_s = problem_settings['parameter_s']
    decomposition_type = problem_settings['decomposition_type']
    connectivity_required = problem_settings['connectivity_required']
    
    adj = {v: set() for v in vertices}
    for e in edges:
        u, v = e
        adj[u].add(v)
        adj[v].add(u)
    
    degrees = {v: len(adj[v]) for v in vertices}
    
    if relaxed_clique_type == 'gamma-quasi-clique':
        gamma = parameter_s / 1000.0
    else:
        gamma = None
    
    groups = []
    uncovered = set(vertices)
    
    while uncovered and (time.time() - start_time < args.time_limit):
        seed = max(uncovered, key=lambda v: degrees[v])
        
        if relaxed_clique_type == 's-clique' and parameter_s >= 2:
            radius = parameter_s // 2
            dist = {}
            queue = collections.deque()
            group_set = set()
            dist[seed] = 0
            queue.append(seed)
            group_set.add(seed)
            while queue and (time.time() - start_time < args.time_limit):
                u = queue.popleft()
                if dist[u] < radius:
                    for v in adj[u]:
                        if v not in dist and v in uncovered:
                            dist[v] = dist[u] + 1
                            if dist[v] <= radius:
                                queue.append(v)
                                group_set.add(v)
            groups.append(list(group_set))
            uncovered -= group_set
            if logger:
                logger.log(len(groups) + len(uncovered))
        else:
            current_group = set([seed])
            deg_in_group = {seed: 0}
            if relaxed_clique_type in ['gamma-quasi-clique', 's-defective']:
                current_edges = 0
            else:
                current_edges = None
                
            count_dict = {}
            for v in uncovered:
                if v == seed:
                    continue
                if v in adj[seed]:
                    count_dict[v] = 1
                else:
                    count_dict[v] = 0
                    
            heap = []
            for v in count_dict:
                heapq.heappush(heap, (-count_dict[v], v))
                
            while heap and (time.time() - start_time < args.time_limit):
                found = False
                candidate = None
                while heap and not found:
                    neg_count, v = heapq.heappop(heap)
                    if v not in uncovered or v in current_group:
                        continue
                    if -neg_count != count_dict.get(v, 0):
                        continue
                    if connectivity_required and count_dict.get(v, 0) == 0:
                        continue
                    candidate = v
                    found = True
                    
                if not found:
                    break
                    
                new_n = len(current_group) + 1
                new_deg_in_group = {}
                new_deg_in_group[candidate] = count_dict.get(candidate, 0)
                for u in current_group:
                    if u in adj[candidate]:
                        new_deg_in_group[u] = deg_in_group[u] + 1
                    else:
                        new_deg_in_group[u] = deg_in_group[u]
                        
                valid = True
                if relaxed_clique_type == 'k-core':
                    k = parameter_s
                    if new_deg_in_group[candidate] < k:
                        valid = False
                    else:
                        for u in current_group:
                            if new_deg_in_group[u] < k:
                                valid = False
                                break
                elif relaxed_clique_type == 's-plex':
                    s_val = parameter_s
                    if new_deg_in_group[candidate] < new_n - s_val:
                        valid = False
                    else:
                        for u in current_group:
                            if new_deg_in_group[u] < new_n - s_val:
                                valid = False
                                break
                elif relaxed_clique_type == 's-clique':
                    s_val = 1
                    if new_deg_in_group[candidate] < new_n - s_val:
                        valid = False
                    else:
                        for u in current_group:
                            if new_deg_in_group[u] < new_n - s_val:
                                valid = False
                                break
                elif relaxed_clique_type == 'gamma-quasi-clique':
                    new_edges_val = current_edges + count_dict.get(candidate, 0)
                    total_possible = new_n * (new_n - 1) // 2
                    if new_edges_val < gamma * total_possible:
                        valid = False
                elif relaxed_clique_type == 's-defective':
                    s_def = parameter_s
                    new_edges_val = current_edges + count_dict.get(candidate, 0)
                    total_possible = new_n * (new_n - 1) // 2
                    if new_edges_val < total_possible - s_def:
                        valid = False
                elif relaxed_clique_type == 'k-block':
                    k = parameter_s
                    min_deg = new_deg_in_group[candidate]
                    for u in current_group:
                        if new_deg_in_group[u] < min_deg:
                            min_deg = new_deg_in_group[u]
                    if min_deg < k:
                        valid = False
                elif relaxed_clique_type == 's-bundle':
                    s_bundle = parameter_s
                    min_deg = new_deg_in_group[candidate]
                    for u in current_group:
                        if new_deg_in_group[u] < min_deg:
                            min_deg = new_deg_in_group[u]
                    if min_deg < new_n - s_bundle:
                        valid = False
                elif relaxed_clique_type == 's-club':
                    new_group = current_group | {candidate}
                    local_adj = {}
                    for u in new_group:
                        local_adj[u] = [neigh for neigh in adj[u] if neigh in new_group]
                    diameter = 0
                    disconnected = False
                    for node in new_group:
                        dist = {}
                        queue_bfs = collections.deque([node])
                        dist[node] = 0
                        while queue_bfs:
                            u_bfs = queue_bfs.popleft()
                            for v_bfs in local_adj[u_bfs]:
                                if v_bfs not in dist:
                                    dist[v_bfs] = dist[u_bfs] + 1
                                    queue_bfs.append(v_bfs)
                        if len(dist) != len(new_group):
                            disconnected = True
                            break
                        max_dist = max(dist.values()) if dist else 0
                        if max_dist > diameter:
                            diameter = max_dist
                        if diameter > parameter_s:
                            break
                    if disconnected or diameter > parameter_s:
                        valid = False
                else:
                    valid = False
                    
                if valid:
                    current_group.add(candidate)
                    uncovered.remove(candidate)
                    for u in current_group:
                        if u == candidate:
                            continue
                        if u in adj[candidate]:
                            deg_in_group[u] = deg_in_group.get(u, 0) + 1
                    deg_in_group[candidate] = count_dict.get(candidate, 0)
                    
                    if relaxed_clique_type in ['gamma-quasi-clique', 's-defective']:
                        current_edges += count_dict.get(candidate, 0)
                        
                    for neighbor in adj[candidate]:
                        if neighbor in uncovered and neighbor not in current_group:
                            if neighbor not in count_dict:
                                count_dict[neighbor] = 0
                            count_dict[neighbor] += 1
                            heapq.heappush(heap, (-count_dict[neighbor], neighbor))
            
            groups.append(list(current_group))
            uncovered -= current_group
            if logger:
                logger.log(len(groups) + len(uncovered))
                
    for v in uncovered:
        groups.append([v])
        
    if logger:
        logger.log(len(groups))
        
    solution_json = {
        "objective_value": len(groups),
        "solution": groups
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution_json, f)

if __name__ == '__main__':
    main()