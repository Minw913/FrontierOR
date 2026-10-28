import argparse
import json
import random
import time

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    n = instance['n']
    capacity = instance['capacity']
    weights = instance['weights']
    p = instance['profit_matrix']

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path and SolutionLogger else None

    start_time = time.time()

    def local_search(selected, cur_w, cur_obj, delta_add):
        improved = True
        while improved and time.time() - start_time < args.time_limit:
            improved = False
            best_delta = 1e-9
            best_move = None
            
            # 1. Try to remove
            for i in range(n):
                if selected[i]:
                    delta = -(delta_add[i] - p[i][i])
                    if delta > best_delta:
                        best_delta = delta
                        best_move = ('remove', i)
                        
            # 2. Try to add
            for i in range(n):
                if not selected[i] and cur_w + weights[i] <= capacity:
                    delta = delta_add[i]
                    if delta > best_delta:
                        best_delta = delta
                        best_move = ('add', i)
                        
            # 3. Try to swap
            for i in range(n):
                if selected[i]:
                    rem_val = -delta_add[i] + p[i][i]
                    max_w = capacity - cur_w + weights[i]
                    p_i = p[i]
                    for j in range(n):
                        if not selected[j] and weights[j] <= max_w:
                            delta = rem_val + delta_add[j] - p_i[j]
                            if delta > best_delta:
                                best_delta = delta
                                best_move = ('swap', i, j)
                                
            if best_move:
                if best_move[0] == 'remove':
                    i = best_move[1]
                    selected[i] = 0
                    cur_w -= weights[i]
                    cur_obj += best_delta
                    p_i = p[i]
                    for k in range(n):
                        delta_add[k] -= p_i[k]
                elif best_move[0] == 'add':
                    i = best_move[1]
                    selected[i] = 1
                    cur_w += weights[i]
                    cur_obj += best_delta
                    p_i = p[i]
                    for k in range(n):
                        delta_add[k] += p_i[k]
                elif best_move[0] == 'swap':
                    i, j = best_move[1], best_move[2]
                    selected[i] = 0
                    selected[j] = 1
                    cur_w = cur_w - weights[i] + weights[j]
                    cur_obj += best_delta
                    p_i = p[i]
                    for k in range(n):
                        delta_add[k] -= p_i[k]
                    p_j = p[j]
                    for k in range(n):
                        delta_add[k] += p_j[k]
                        
        return selected, cur_w, cur_obj, delta_add

    # Initial setup
    selected = [0] * n
    cur_w = 0
    cur_obj = 0
    delta_add = [p[i][i] for i in range(n)]

    # Greedy initial solution
    items = list(range(n))
    random.shuffle(items)
    items.sort(key=lambda i: p[i][i] / weights[i] if weights[i] > 0 else float('inf'), reverse=True)

    for i in items:
        if delta_add[i] > 0 and cur_w + weights[i] <= capacity:
            selected[i] = 1
            cur_w += weights[i]
            cur_obj += delta_add[i]
            p_i = p[i]
            for k in range(n):
                delta_add[k] += p_i[k]

    best_selected = list(selected)
    best_obj = cur_obj
    best_w = cur_w
    best_delta_add = list(delta_add)

    if logger:
        logger.log_solution(float(best_obj), {
            "objective_value": float(best_obj),
            "selected_items": best_selected
        })

    # Iterated Local Search
    while time.time() - start_time < args.time_limit:
        selected, cur_w, cur_obj, delta_add = local_search(selected, cur_w, cur_obj, delta_add)
        
        if cur_obj > best_obj:
            best_obj = cur_obj
            best_selected = list(selected)
            best_w = cur_w
            best_delta_add = list(delta_add)
            if logger:
                logger.log_solution(float(best_obj), {
                    "objective_value": float(best_obj),
                    "selected_items": best_selected
                })
        
        # Perturbation
        selected = list(best_selected)
        cur_w = best_w
        cur_obj = best_obj
        delta_add = list(best_delta_add)
        
        K = random.randint(1, max(2, n // 10))
        in_S = [i for i in range(n) if selected[i]]
        not_in_S = [i for i in range(n) if not selected[i]]
        
        if in_S:
            remove_count = min(K, len(in_S))
            to_remove = random.sample(in_S, remove_count)
            for i in to_remove:
                selected[i] = 0
                cur_w -= weights[i]
                cur_obj -= (delta_add[i] - p[i][i])
                p_i = p[i]
                for k in range(n):
                    delta_add[k] -= p_i[k]
                
        if not_in_S:
            random.shuffle(not_in_S)
            for i in not_in_S:
                if cur_w + weights[i] <= capacity:
                    selected[i] = 1
                    cur_w += weights[i]
                    cur_obj += delta_add[i]
                    p_i = p[i]
                    for k in range(n):
                        delta_add[k] += p_i[k]

    output = {
        "objective_value": float(best_obj),
        "selected_items": best_selected
    }
    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()