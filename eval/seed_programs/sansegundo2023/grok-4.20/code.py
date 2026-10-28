import json
import argparse
import time
import sys
from solution_logger import SolutionLogger

def is_clique(vertices, adj):
    if not vertices:
        return True
    vset = set(vertices)
    for u in vertices:
        neighbors = set(adj[str(u)])
        if not vset.issubset(neighbors | {u}):
            return False
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['num_vertices']
    adj = instance['adjacency_list']
    greedy_size = instance['greedy_clique_size']
    greedy = instance.get('greedy_clique', [])

    # Build adjacency sets
    adj_set = {}
    for v_str, neigh in adj.items():
        adj_set[int(v_str)] = set(neigh)

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    best_obj = greedy_size
    best_clique = sorted(greedy)

    if logger and best_obj > 0:
        logger.log(best_obj)

    # Simple greedy improvement with random restarts
    time_limit = float(args.time_limit)
    elapsed = 0.0
    iter_count = 0
    max_iter = 10000

    while elapsed < time_limit and iter_count < max_iter:
        iter_count += 1
        # Start with random permutation
        order = list(range(n))
        import random
        random.shuffle(order)

        current = []
        for v in order:
            can_add = True
            for u in current:
                if v not in adj_set[u]:
                    can_add = False
                    break
            if can_add:
                current.append(v)

        cur_size = len(current)
        if cur_size > best_obj and is_clique(current, adj):
            best_obj = cur_size
            best_clique = sorted(current)
            if logger:
                logger.log(best_obj)
            # Early stop if we reach a known good bound
            if best_obj >= n // 2 + 1:
                break

        elapsed = time.time() - start_time
        if elapsed >= time_limit:
            break

    # Final local search on best clique
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        candidates = []
        for v in range(n):
            if v not in best_clique:
                if all(v in adj_set[u] for u in best_clique):
                    candidates.append(v)
        if candidates:
            best_clique.append(candidates[0])
            best_obj += 1
            improved = True
            if logger:
                logger.log(best_obj)

    # One final validation
    if not is_clique(best_clique, adj):
        # Fallback to greedy
        best_clique = sorted(greedy)
        best_obj = greedy_size

    solution = {
        "objective_value": float(best_obj),
        "clique_vertices": best_clique
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()