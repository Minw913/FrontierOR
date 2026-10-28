import argparse
import json
import time
import numpy as np
from solution_logger import SolutionLogger


def evaluate_loc(data_matrix, selected):
    """Facility Location: sum over clients of max benefit from any selected facility."""
    if len(selected) == 0:
        return 0.0
    sub = data_matrix[:, selected]
    return float(np.sum(np.max(sub, axis=1)))


def evaluate_cov(data_matrix, node_weights, selected):
    """Weighted Coverage: sum of weights of items covered by at least one selected sensor."""
    if len(selected) == 0:
        return 0.0
    sub = data_matrix[:, selected]
    covered = np.max(sub, axis=1)  # 1 if any sensor covers, 0 otherwise
    return float(np.dot(node_weights, covered))


def evaluate_inf(data_matrix, element_weights, selected):
    """Bipartite Influence: sum over targets of prob reached by at least one selected source."""
    if len(selected) == 0:
        return 0.0
    # For each target, prob = 1 - prod(1 - p_j * edge_tj) for j in selected
    # edge_tj is data_matrix[t][j], p_j is element_weights[j]
    # influence prob for (t,j) = edge_tj * p_j
    # prob not reached by j = 1 - edge_tj * p_j
    probs = element_weights[selected]  # activation probs of selected
    edges = data_matrix[:, selected]  # m x len(selected)
    # For each target-source pair: 1 - edge * prob
    not_reached = 1.0 - edges * probs[np.newaxis, :]
    prod_not_reached = np.prod(not_reached, axis=1)
    return float(np.sum(1.0 - prod_not_reached))


def marginal_gain_loc(data_matrix, current_max, elem):
    """Marginal gain of adding elem to current set for LOC."""
    benefits = data_matrix[:, elem]
    gains = np.maximum(benefits - current_max, 0.0)
    return float(np.sum(gains))


def marginal_gain_cov(data_matrix, node_weights, current_covered, elem):
    """Marginal gain of adding elem for COV."""
    new_coverage = data_matrix[:, elem]
    # Items not yet covered but now covered
    newly_covered = np.logical_and(new_coverage > 0, ~current_covered)
    return float(np.dot(node_weights, newly_covered))


def marginal_gain_inf(data_matrix, element_weights, current_not_reached, elem):
    """Marginal gain of adding elem for INF."""
    p = element_weights[elem]
    edges = data_matrix[:, elem]
    # For targets with edge: not_reached *= (1 - p)
    # gain = sum of current_not_reached[t] * (1 - (1 - edges[t]*p))
    # = sum of current_not_reached[t] * edges[t] * p
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
        if instance_type == 'LOC':
            return evaluate_loc(data_matrix, sel)
        elif instance_type == 'COV':
            return evaluate_cov(data_matrix, node_weights, sel)
        elif instance_type == 'INF':
            return evaluate_inf(data_matrix, element_weights, sel)
        return 0.0

    # Greedy algorithm with lazy evaluations (accelerated greedy)
    best_selected = []
    best_obj = 0.0

    if instance_type == 'LOC':
        # Track current max benefit per client
        current_max = np.zeros(m, dtype=np.float64)
        selected_set = set()
        selected_list = []
        
        # Initialize marginal gains
        marginals = np.zeros(n, dtype=np.float64)
        for j in range(n):
            marginals[j] = marginal_gain_loc(data_matrix, current_max, j)
        
        for _ in range(k):
            if time.time() - start_time > args.time_limit - 1:
                break
            
            # Lazy greedy
            # Sort by upper bound (current marginals)
            order = np.argsort(-marginals)
            best_elem = -1
            best_gain = -1.0
            
            for idx in range(n):
                j = order[idx]
                if j in selected_set:
                    continue
                # Recompute marginal
                gain = marginal_gain_loc(data_matrix, current_max, j)
                marginals[j] = gain
                if gain >= best_gain:
                    best_gain = gain
                    best_elem = j
                # Check if this is definitely the best (lazy eval)
                # Next element's upper bound
                if idx + 1 < n:
                    next_j = order[idx + 1]
                    while next_j in selected_set and idx + 1 < n - 1:
                        idx_check = idx + 1
                        # Actually we can't skip easily in a for loop, let's just check
                        break
                    if marginals[order[idx + 1]] <= best_gain:
                        break
                else:
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
                j = order[idx]
                if j in selected_set:
                    continue
                gain = marginal_gain_cov(data_matrix, node_weights, current_covered, j)
                marginals[j] = gain
                if gain >= best_gain:
                    best_gain = gain
                    best_elem = j
                if idx + 1 < n:
                    if marginals[order[idx + 1]] <= best_gain:
                        break
                else:
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
                j = order[idx]
                if j in selected_set:
                    continue
                gain = marginal_gain_inf(data_matrix, element_weights, current_not_reached, j)
                marginals[j] = gain
                if gain >= best_gain:
                    best_gain = gain
                    best_elem = j
                if idx + 1 < n:
                    if marginals[order[idx + 1]] <= best_gain:
                        break
                else:
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

    # Now try to improve with local search (swap-based)
    # For each selected element, try swapping with each unselected element
    remaining_time = args.time_limit - (time.time() - start_time)
    
    if remaining_time > 2 and len(best_selected) > 0:
        current_selected = list(best_selected)
        current_obj = best_obj
        improved = True
        
        while improved and time.time() - start_time < args.time_limit - 1:
            improved = False
            unselected = [j for j in range(n) if j not in set(current_selected)]
            
            for i_idx in range(len(current_selected)):
                if time.time() - start_time > args.time_limit - 1:
                    break
                for u in unselected:
                    if time.time() - start_time > args.time_limit - 1:
                        break
                    # Try swap
                    new_selected = list(current_selected)
                    old_elem = new_selected[i_idx]
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
                        break  # restart inner loop
                if improved:
                    break
        
        # Also try adding elements if we haven't used all k slots
        if len(best_selected) < k:
            current_selected = list(best_selected)
            current_obj = best_obj
            unselected = [j for j in range(n) if j not in set(current_selected)]
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

    solution = {
        "objective_value": best_obj,
        "selected_elements": best_selected
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    solve()