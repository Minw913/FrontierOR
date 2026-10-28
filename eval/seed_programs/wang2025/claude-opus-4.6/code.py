import argparse
import json
import time
import numpy as np
from solution_logger import SolutionLogger


def evaluate_loc(data_matrix, selected):
    if len(selected) == 0:
        return 0.0
    sub = data_matrix[:, selected]
    return float(np.sum(np.max(sub, axis=1)))


def evaluate_cov(data_matrix, node_weights, selected):
    if len(selected) == 0:
        return 0.0
    sub = data_matrix[:, selected]
    covered = np.max(sub, axis=1)
    return float(np.dot(node_weights, covered))


def evaluate_inf(data_matrix, element_weights, selected):
    if len(selected) == 0:
        return 0.0
    probs = element_weights[selected]
    edges = data_matrix[:, selected]
    not_reached = 1.0 - edges * probs[np.newaxis, :]
    prod_not_reached = np.prod(not_reached, axis=1)
    return float(np.sum(1.0 - prod_not_reached))


def marginal_gain_loc(data_matrix, current_max, elem):
    benefits = data_matrix[:, elem]
    gains = np.maximum(benefits - current_max, 0.0)
    return float(np.sum(gains))


def marginal_gain_cov(data_matrix, node_weights, current_covered, elem):
    new_coverage = data_matrix[:, elem]
    newly_covered = np.logical_and(new_coverage > 0, ~current_covered)
    return float(np.dot(node_weights, newly_covered))


def marginal_gain_inf(data_matrix, element_weights, current_not_reached, elem):
    p = element_weights[elem]
    edges = data_matrix[:, elem]
    influence = edges * p
    gains = current_not_reached * influence
    return float(np.sum(gains))


def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    instance_type = instance['instance_type']
    n = instance['n']
    m = instance['m']
    k = instance['k']
    gamma = instance['gamma']
    data_matrix = np.array(instance['data_matrix'], dtype=np.float64)
    node_weights = np.array(instance['node_weights'], dtype=np.float64)
    element_weights = np.array(instance['element_weights'], dtype=np.float64)

    def evaluate(selected):
        sel = list(selected)
        if len(sel) == 0:
            return 0.0
        sel_arr = np.array(sel, dtype=int)
        if instance_type == 'LOC':
            return evaluate_loc(data_matrix, sel_arr)
        elif instance_type == 'COV':
            return evaluate_cov(data_matrix, node_weights, sel_arr)
        elif instance_type == 'INF':
            return evaluate_inf(data_matrix, element_weights, sel_arr)
        return 0.0

    best_selected = []
    best_obj = 0.0

    if instance_type == 'LOC':
        current_max = np.zeros(m, dtype=np.float64)
        selected_set = set()
        selected_list = []
        
        marginals = np.zeros(n, dtype=np.float64)
        for j in range(n):
            marginals[j] = marginal_gain_loc(data_matrix, current_max, j)
        
        for _ in range(k):
            if time.time() - start_time > args.time_limit - 1:
                break
            
            order = np.argsort(-marginals)
            best_elem = -1
            best_gain = -1.0
            
            for idx in range(n):
                j = int(order[idx])
                if j in selected_set:
                    continue
                gain = marginal_gain_loc(data_matrix, current_max, j)
                marginals[j] = gain
                if gain >= best_gain:
                    best_gain = gain
                    best_elem = j
                # Lazy eval check
                found_better = False
                for idx2 in range(idx + 1, n):
                    next_j = int(order[idx2])
                    if next_j not in selected_set:
                        if marginals[next_j] <= best_gain:
                            found_better = True
                        break
                else:
                    found_better = True
                if found_better:
                    break
            
            if best_elem < 0 or best_gain <= 0:
                break
            
            selected_set.add(best_elem)
            selected_list.append(best_elem)
            current_max = np.maximum(current_max, data_matrix[:, best_elem])
            
            obj = float(np.sum(current_max))
            if obj > best_obj:
                best_obj = obj
                best_selected = list(selected_list)
                if logger:
                    logger.log(best_obj)

    elif instance_type == 'COV':
        current_covered = np.zeros(m, dtype=bool)
        selected_set = set()
        selected_list = []
        
        marginals = np.zeros(n, dtype=np.float64)
        for j in range(n):
            marginals[j] = marginal_gain_cov(data_matrix, node_weights, current_covered, j)
        
        for _ in range(k):
            if time.time() - start_time > args.time_limit - 1:
                break
            
            order = np.argsort(-marginals)
            best_elem = -1
            best_gain = -1.0
            
            for idx in range(n):
                j = int(order[idx])
                if j in selected_set:
                    continue
                gain = marginal_gain_cov(data_matrix, node_weights, current_covered, j)
                marginals[j] = gain
                if gain >= best_gain:
                    best_gain = gain
                    best_elem = j
                found_better = False
                for idx2 in range(idx + 1, n):
                    next_j = int(order[idx2])
                    if next_j not in selected_set:
                        if marginals[next_j] <= best_gain:
                            found_better = True
                        break
                else:
                    found_better = True
                if found_better:
                    break
            
            if best_elem < 0 or best_gain <= 0:
                break
            
            selected_set.add(best_elem)
            selected_list.append(best_elem)
            current_covered = np.logical_or(current_covered, data_matrix[:, best_elem] > 0)
            
            obj = float(np.dot(node_weights, current_covered))
            if obj > best_obj:
                best_obj = obj
                best_selected = list(selected_list)
                if logger:
                    logger.log(best_obj)

    elif instance_type == 'INF':
        current_not_reached = np.ones(m, dtype=np.float64)
        selected_set = set()
        selected_list = []
        
        marginals = np.zeros(n, dtype=np.float64)
        for j in range(n):
            marginals[j] = marginal_gain_inf(data_matrix, element_weights, current_not_reached, j)
        
        for _ in range(k):
            if time.time() - start_time > args.time_limit - 1:
                break
            
            order = np.argsort(-marginals)
            best_elem = -1
            best_gain = -1.0
            
            for idx in range(n):
                j = int(order[idx])
                if j in selected_set:
                    continue
                gain = marginal_gain_inf(data_matrix, element_weights, current_not_reached, j)
                marginals[j] = gain
                if gain >= best_gain:
                    best_gain = gain
                    best_elem = j
                found_better = False
                for idx2 in range(idx + 1, n):
                    next_j = int(order[idx2])
                    if next_j not in selected_set:
                        if marginals[next_j] <= best_gain:
                            found_better = True
                        break
                else:
                    found_better = True
                if found_better:
                    break
            
            if best_elem < 0 or best_gain <= 1e-12:
                break
            
            selected_set.add(best_elem)
            selected_list.append(best_elem)
            p = element_weights[best_elem]
            edges = data_matrix[:, best_elem]
            current_not_reached *= (1.0 - edges * p)
            
            obj = float(np.sum(1.0 - current_not_reached))
            if obj > best_obj:
                best_obj = obj
                best_selected = list(selected_list)
                if logger:
                    logger.log(best_obj)

    # Local search: swap-based improvement
    remaining_time = args.time_limit - (time.time() - start_time)
    
    if remaining_time > 2 and len(best_selected) > 0:
        current_selected = list(best_selected)
        current_obj = best_obj
        improved = True
        
        while improved and time.time() - start_time < args.time_limit - 1:
            improved = False
            selected_set_local = set(current_selected)
            unselected = [j for j in range(n) if j not in selected_set_local]
            
            for i_idx in range(len(current_selected)):
                if time.time() - start_time > args.time_limit - 1:
                    break
                for u in unselected:
                    if time.time() - start_time > args.time_limit - 1:
                        break
                    new_selected = list(current_selected)
                    new_selected[i_idx] = u
                    new_obj = evaluate(new_selected)
                    if new_obj > current_obj + 1e-10:
                        current_selected = new_selected
                        current_obj = new_obj
                        improved = True
                        if current_obj > best_obj:
                            best_obj = current_obj
                            best_selected = list(current_selected)
                            if logger:
                                logger.log(best_obj)
                        break
                if improved:
                    break
        
        # Try adding elements if we haven't used all k slots
        if len(best_selected) < k:
            current_selected = list(best_selected)
            current_obj = best_obj
            selected_set_local = set(current_selected)
            unselected = [j for j in range(n) if j not in selected_set_local]
            for u in unselected:
                if time.time() - start_time > args.time_limit - 1:
                    break
                if len(current_selected) >= k:
                    break
                new_selected = current_selected + [u]
                new_obj = evaluate(new_selected)
                if new_obj > current_obj + 1e-10:
                    current_selected = new_selected
                    current_obj = new_obj
                    if current_obj > best_obj:
                        best_obj = current_obj
                        best_selected = list(current_selected)
                        if logger:
                            logger.log(best_obj)

    # Convert to native Python types for JSON serialization
    solution = {
        "objective_value": float(best_obj),
        "selected_elements": [int(x) for x in best_selected]
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    solve()