import json
import heapq
import argparse
import time
from solution_logger import SolutionLogger

def compute_mg(i, instance_type, state, data_matrix, neighbors, element_weights, node_weights):
    if instance_type == 'LOC':
        mg = 0.0
        if i < len(neighbors):
            for j in neighbors[i]:
                if data_matrix[j][i] > state['best_per_client'][j]:
                    mg += data_matrix[j][i] - state['best_per_client'][j]
        return mg
    elif instance_type == 'COV':
        mg = 0.0
        if i < len(neighbors):
            for j in neighbors[i]:
                if not state['covered'][j]:
                    mg += node_weights[j]
        return mg
    elif instance_type == 'INF':
        mg = 0.0
        p = element_weights[i]
        if p > 0 and i < len(neighbors):
            for j in neighbors[i]:
                mg += state['prod'][j] * p
        return mg
    else:
        return 0.0

def update_state(i, instance_type, state, data_matrix, neighbors, element_weights):
    if instance_type == 'LOC':
        if i < len(neighbors):
            for j in neighbors[i]:
                if data_matrix[j][i] > state['best_per_client'][j]:
                    state['best_per_client'][j] = data_matrix[j][i]
    elif instance_type == 'COV':
        if i < len(neighbors):
            for j in neighbors[i]:
                state['covered'][j] = True
    elif instance_type == 'INF':
        p = element_weights[i]
        if p > 0 and i < len(neighbors):
            for j in neighbors[i]:
                state['prod'][j] *= (1.0 - p)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.perf_counter()
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    instance_type = data['instance_type']
    n = data['n']
    m = data['m']
    k = data['k']
    gamma = data['gamma']
    type_specific_param = data['type_specific_param']
    data_matrix = data['data_matrix']
    node_weights = data['node_weights']
    element_weights = data['element_weights']

    neighbors = []
    if instance_type == 'LOC':
        for i in range(n):
            lst = []
            for j in range(m):
                if data_matrix[j][i] > 0.0:
                    lst.append(j)
            neighbors.append(lst)
    elif instance_type == 'COV' or instance_type == 'INF':
        for i in range(n):
            lst = []
            for j in range(m):
                if data_matrix[j][i] == 1:
                    lst.append(j)
            neighbors.append(lst)

    if instance_type == 'LOC':
        state = {'best_per_client': [0.0] * m}
    elif instance_type == 'COV':
        state = {'covered': [False] * m}
    elif instance_type == 'INF':
        state = {'prod': [1.0] * m}
    total = 0.0
    selected = []

    log_to_file = args.log_path is not None
    logger = SolutionLogger(args.log_path, "maximize") if log_to_file else None
    
    if logger:
        logger.log(total)
    
    heap = []
    for i in range(n):
        mg = compute_mg(i, instance_type, state, data_matrix, neighbors, element_weights, node_weights)
        heapq.heappush(heap, (-mg, i))
    
    for step in range(k):
        if time.perf_counter() - start_time >= args.time_limit:
            break
        
        candidate = None
        mg_candidate = None
        inner_timeout = False
        
        while heap:
            if time.perf_counter() - start_time >= args.time_limit:
                inner_timeout = True
                break
            
            neg_mg, elem_idx = heapq.heappop(heap)
            mg_val = compute_mg(elem_idx, instance_type, state, data_matrix, neighbors, element_weights, node_weights)
            
            if mg_val <= 0:
                continue
                
            if heap:
                top_neg = heap[0][0]
                top_mg = -top_neg
                if mg_val >= top_mg:
                    candidate = elem_idx
                    mg_candidate = mg_val
                    break
                else:
                    heapq.heappush(heap, (-mg_val, elem_idx))
            else:
                candidate = elem_idx
                mg_candidate = mg_val
                break
        
        if inner_timeout:
            break
        
        if candidate is None:
            break
        
        selected.append(candidate)
        total += mg_candidate
        update_state(candidate, instance_type, state, data_matrix, neighbors, element_weights)
        
        if logger:
            logger.log(total)
    
    solution = {
        "objective_value": total,
        "selected_elements": selected
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()