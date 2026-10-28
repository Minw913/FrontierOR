import time
import json
import argparse
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    num_vertices = instance['num_vertices']
    adjacency_list = instance['adjacency_list']
    
    adj_set = [set(neighbors) for neighbors in adjacency_list]
    degrees = [len(neighbors) for neighbors in adjacency_list]
    
    best_size = 0
    best_clique = []
    
    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="maximize")
    else:
        logger = None

    candidate_set = set(range(num_vertices))
    greedy_clique = []
    while candidate_set:
        if time.time() - start_time > args.time_limit:
            break
        v = None
        max_deg = -1
        for u in candidate_set:
            deg = len(adj_set[u] & candidate_set)
            if deg > max_deg:
                max_deg = deg
                v = u
        if v is None:
            break
        greedy_clique.append(v)
        candidate_set.discard(v)
        candidate_set = candidate_set & adj_set[v]
    
    if len(greedy_clique) > best_size:
        best_size = len(greedy_clique)
        best_clique = greedy_clique
        if logger:
            logger.log(best_size)

    def backtrack(clique, candidate_list):
        nonlocal best_size, best_clique
        if time.time() - start_time > args.time_limit:
            return
        n = len(candidate_list)
        if n == 0:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = list(clique)
                if logger:
                    logger.log(best_size)
            return
        if len(clique) + n <= best_size:
            return
        sorted_candidate = sorted(candidate_list, key=lambda x: -degrees[x])
        for i in range(len(sorted_candidate)):
            if len(clique) + (len(sorted_candidate) - i) <= best_size:
                break
            v = sorted_candidate[i]
            new_candidate = []
            for j in range(i+1, len(sorted_candidate)):
                w = sorted_candidate[j]
                if w in adj_set[v]:
                    new_candidate.append(w)
            clique.append(v)
            backtrack(clique, new_candidate)
            clique.pop()
    
    initial_candidate = list(range(num_vertices))
    current_clique = []
    backtrack(current_clique, initial_candidate)
    
    solution = {
        "objective_value": best_size,
        "clique_vertices": sorted(best_clique)
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()