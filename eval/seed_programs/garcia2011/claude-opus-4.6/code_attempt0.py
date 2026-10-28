import argparse
import json
import time
import numpy as np
from solution_logger import SolutionLogger


def solve_pmedian(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    n = instance['n']
    p = instance['p']
    cost_matrix = np.array(instance['cost_matrix'], dtype=np.int64)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    def evaluate_solution(medians):
        """Compute total cost for a given set of medians."""
        median_set = list(medians)
        costs = cost_matrix[:, median_set]
        min_costs = np.min(costs, axis=1)
        return int(np.sum(min_costs))
    
    def get_assignments(medians):
        """Get assignments and costs for each node."""
        median_list = list(medians)
        costs = cost_matrix[:, median_list]
        best_idx = np.argmin(costs, axis=1)
        assignments = []
        for i in range(n):
            facility = median_list[best_idx[i]]
            c = int(cost_matrix[i, facility])
            assignments.append({'customer': i, 'facility': facility, 'cost': c})
        return assignments
    
    # For small instances, try exact solution with Gurobi
    if n <= 300 and (time.time() - start_time) < time_limit * 0.9:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            remaining_time = time_limit - (time.time() - start_time) - 1
            if remaining_time > 2:
                model = gp.Model("pmedian")
                model.setParam('OutputFlag', 0)
                model.setParam('TimeLimit', remaining_time)
                model.setParam('Threads', 1)
                
                # y[j] = 1 if j is a median
                y = model.addVars(n, vtype=GRB.BINARY, name="y")
                # x[i,j] = 1 if node i assigned to median j
                x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
                
                # Objective
                model.setObjective(
                    gp.quicksum(cost_matrix[i][j] * x[i, j] for i in range(n) for j in range(n)),
                    GRB.MINIMIZE
                )
                
                # Each node assigned to exactly one median
                for i in range(n):
                    model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1)
                
                # Can only assign to open medians
                for i in range(n):
                    for j in range(n):
                        model.addConstr(x[i, j] <= y[j])
                
                # Exactly p medians
                model.addConstr(gp.quicksum(y[j] for j in range(n)) == p)
                
                # Provide initial solution from heuristic as warm start
                # First do a quick greedy
                # Greedy: pick the node that minimizes total cost when added
                greedy_medians = set()
                # Start with the node that has minimum total cost to all others
                total_costs = np.sum(cost_matrix, axis=0)
                first = int(np.argmin(total_costs))
                greedy_medians.add(first)
                
                for _ in range(p - 1):
                    best_node = -1
                    best_obj = float('inf')
                    current_list = list(greedy_medians)
                    current_min = np.min(cost_matrix[:, current_list], axis=1)
                    for j in range(n):
                        if j not in greedy_medians:
                            new_min = np.minimum(current_min, cost_matrix[:, j])
                            obj = int(np.sum(new_min))
                            if obj < best_obj:
                                best_obj = obj
                                best_node = j
                    greedy_medians.add(best_node)
                
                # Set warm start
                for j in range(n):
                    y[j].Start = 1.0 if j in greedy_medians else 0.0
                
                greedy_assignments = get_assignments(greedy_medians)
                for i in range(n):
                    for j in range(n):
                        if greedy_assignments[i]['facility'] == j:
                            x[i, j].Start = 1.0
                        else:
                            x[i, j].Start = 0.0
                
                greedy_obj = evaluate_solution(greedy_medians)
                if logger:
                    logger.log(greedy_obj)
                
                best_medians = greedy_medians
                best_obj = greedy_obj
                
                # Callback to log incumbents
                def callback(model, where):
                    nonlocal best_obj, best_medians
                    if where == GRB.Callback.MIPSOL:
                        obj = int(round(model.cbGet(GRB.Callback.MIPSOL_OBJ)))
                        if obj < best_obj:
                            best_obj = obj
                            y_vals = model.cbGetSolution(y)
                            best_medians = set(j for j in range(n) if y_vals[j] > 0.5)
                            if logger:
                                logger.log(obj)
                
                model.optimize(callback)
                
                if model.SolCount > 0:
                    final_medians = set(j for j in range(n) if y[j].X > 0.5)
                    final_obj = evaluate_solution(final_medians)
                    if final_obj < best_obj:
                        best_obj = final_obj
                        best_medians = final_medians
                        if logger:
                            logger.log(best_obj)
                
                assignments = get_assignments(best_medians)
                solution = {
                    'objective_value': best_obj,
                    'opened_facilities': sorted(list(best_medians)),
                    'assignments': assignments
                }
                
                with open(solution_path, 'w') as f:
                    json.dump(solution, f)
                return
                
        except Exception:
            pass
    
    # Heuristic approach for larger instances
    # Phase 1: Greedy construction
    greedy_medians = set()
    total_costs = np.sum(cost_matrix, axis=0)
    first = int(np.argmin(total_costs))
    greedy_medians.add(first)
    
    current_min = cost_matrix[:, first].copy().astype(np.int64)
    
    for _ in range(p - 1):
        if time.time() - start_time > time_limit * 0.3:
            # Just pick greedily without full evaluation
            # Pick node that reduces cost most
            improvements = current_min[:, None] - cost_matrix
            improvements = np.maximum(improvements, 0)
            total_improvements = np.sum(improvements, axis=0)
            # Mask already selected
            for m in greedy_medians:
                total_improvements[m] = -1
            best_node = int(np.argmax(total_improvements))
        else:
            best_node = -1
            best_obj = float('inf')
            for j in range(n):
                if j not in greedy_medians:
                    new_min = np.minimum(current_min, cost_matrix[:, j])
                    obj = int(np.sum(new_min))
                    if obj < best_obj:
                        best_obj = obj
                        best_node = j
        
        greedy_medians.add(best_node)
        current_min = np.minimum(current_min, cost_matrix[:, best_node])
    
    best_medians = set(greedy_medians)
    best_obj = evaluate_solution(best_medians)
    if logger:
        logger.log(best_obj)
    
    # Phase 2: Fast swap-based local search (Teitz & Bart / vertex substitution)
    def fast_swap_search(medians):
        nonlocal best_obj, best_medians
        
        median_list = list(medians)
        non_medians = [j for j in range(n) if j not in medians]
        
        # Precompute: for each customer, find nearest and second nearest median
        costs_to_medians = cost_matrix[:, median_list]  # n x p
        sorted_indices = np.argsort(costs_to_medians, axis=1)
        
        nearest = sorted_indices[:, 0]  # index into median_list
        second_nearest = sorted_indices[:, 1] if len(median_list) > 1 else nearest
        
        nearest_cost = costs_to_medians[np.arange(n), nearest]
        second_nearest_cost = costs_to_medians[np.arange(n), second_nearest]
        
        current_obj = int(np.sum(nearest_cost))
        
        improved = True
        while improved and (time.time() - start_time < time_limit * 0.9):
            improved = False
            best_swap_gain = 0
            best_swap = None
            
            for mi_idx in range(len(median_list)):
                if time.time() - start_time > time_limit * 0.9:
                    break
                    
                m_out = median_list[mi_idx]
                
                # For each potential swap-in
                for j in non_medians:
                    if time.time() - start_time > time_limit * 0.9:
                        break
                    
                    # Compute change in objective
                    # For each customer i:
                    # If nearest[i] != mi_idx: new cost = min(nearest_cost[i], cost_matrix[i,j])
                    # If nearest[i] == mi_idx: new cost = min(second_nearest_cost[i], cost_matrix[i,j])
                    
                    cost_to_j = cost_matrix[:, j]
                    
                    is_nearest = (nearest == mi_idx)
                    
                    # Nodes whose nearest is NOT the one being removed
                    new_cost_not_nearest = np.minimum(nearest_cost, cost_to_j)
                    # Nodes whose nearest IS the one being removed
                    new_cost_is_nearest = np.minimum(second_nearest_cost, cost_to_j)
                    
                    new_costs = np.where(is_nearest, new_cost_is_nearest, new_cost_not_nearest)
                    new_obj = int(np.sum(new_costs))
                    gain = current_obj - new_obj
                    
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap = (mi_idx, j)
            
            if best_swap is not None and best_swap_gain > 0:
                mi_idx, j_in = best_swap
                m_out = median_list[mi_idx]
                
                # Perform swap
                median_list[mi_idx] = j_in
                non_medians.remove(j_in)
                non_medians.append(m_out)
                
                # Recompute nearest/second nearest
                costs_to_medians = cost_matrix[:, median_list]
                sorted_indices = np.argsort(costs_to_medians, axis=1)
                nearest = sorted_indices[:, 0]
                second_nearest = sorted_indices[:, 1] if len(median_list) > 1 else nearest
                nearest_cost = costs_to_medians[np.arange(n), nearest]
                second_nearest_cost = costs_to_medians[np.arange(n), second_nearest]
                
                current_obj = int(np.sum(nearest_cost))
                improved = True
                
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_medians = set(median_list)
                    if logger:
                        logger.log(best_obj)
        
        return set(median_list), current_obj
    
    best_medians_ls, best_obj_ls = fast_swap_search(set(greedy_medians))
    if best_obj_ls < best_obj:
        best_obj = best_obj_ls
        best_medians = best_medians_ls
    
    # Phase 3: Perturbation + local search (if time allows)
    rng = np.random.RandomState(42)
    iteration = 0
    while time.time() - start_time < time_limit * 0.85:
        iteration += 1
        # Perturb: randomly swap some medians
        current_list = list(best_medians)
        non_current = [j for j in range(n) if j not in best_medians]
        
        num_swaps = max(1, min(p // 4, len(non_current)))
        num_swaps = min(num_swaps, 3)
        
        perturbed = set(current_list)
        swap_out = rng.choice(current_list, size=num_swaps, replace=False)
        swap_in = rng.choice(non_current, size=num_swaps, replace=False)
        
        for so, si in zip(swap_out, swap_in):
            perturbed.remove(so)
            perturbed.add(si)
        
        perturbed_medians, perturbed_obj = fast_swap_search(perturbed)
        if perturbed_obj < best_obj:
            best_obj = perturbed_obj
            best_medians = perturbed_medians
    
    # Build final solution
    assignments = get_assignments(best_medians)
    solution = {
        'objective_value': best_obj,
        'opened_facilities': sorted(list(best_medians)),
        'assignments': assignments
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve_pmedian(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()