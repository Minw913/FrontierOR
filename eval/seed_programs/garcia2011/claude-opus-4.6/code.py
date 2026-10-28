import argparse
import json
import time
import math
import random
from solution_logger import SolutionLogger


def solve_pmedian(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    n = instance['n']
    p = instance['p']
    
    # Build cost matrix
    if 'cost_matrix' in instance and instance['cost_matrix'] is not None:
        cost_matrix = instance['cost_matrix']
    elif 'node_coordinates' in instance:
        coords = instance['node_coordinates']
        cost_matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                d = round(math.sqrt(dx * dx + dy * dy))
                cost_matrix[i][j] = d
                cost_matrix[j][i] = d
    else:
        raise ValueError("No cost_matrix or node_coordinates provided")
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    def evaluate_solution(medians):
        """Compute total cost for a given set of medians."""
        median_list = list(medians)
        total = 0
        for i in range(n):
            min_cost = cost_matrix[i][median_list[0]]
            for j in median_list[1:]:
                c = cost_matrix[i][j]
                if c < min_cost:
                    min_cost = c
            total += min_cost
        return total
    
    def get_assignments(medians):
        """Get assignments and costs for each node."""
        median_list = list(medians)
        assignments = []
        for i in range(n):
            best_facility = median_list[0]
            best_cost = cost_matrix[i][median_list[0]]
            for j in median_list[1:]:
                c = cost_matrix[i][j]
                if c < best_cost:
                    best_cost = c
                    best_facility = j
            assignments.append({'customer': i, 'facility': best_facility, 'cost': best_cost})
        return assignments
    
    def compute_nearest_info(median_list):
        """Compute nearest and second nearest median for each node."""
        pm = len(median_list)
        nearest_idx = [0] * n  # index into median_list
        nearest_cost = [0] * n
        second_nearest_idx = [0] * n
        second_nearest_cost = [0] * n
        
        for i in range(n):
            best = 0
            best_c = cost_matrix[i][median_list[0]]
            second = -1
            second_c = float('inf')
            
            for k in range(1, pm):
                c = cost_matrix[i][median_list[k]]
                if c < best_c:
                    second = best
                    second_c = best_c
                    best = k
                    best_c = c
                elif c < second_c:
                    second = k
                    second_c = c
            
            nearest_idx[i] = best
            nearest_cost[i] = best_c
            if second == -1:
                second = best
                second_c = best_c
            second_nearest_idx[i] = second
            second_nearest_cost[i] = second_c
        
        return nearest_idx, nearest_cost, second_nearest_idx, second_nearest_cost
    
    # For small instances, try Gurobi
    use_gurobi = n <= 250
    
    # Phase 1: Greedy construction
    total_costs = [0] * n
    for j in range(n):
        s = 0
        for i in range(n):
            s += cost_matrix[i][j]
        total_costs[j] = s
    
    first = min(range(n), key=lambda j: total_costs[j])
    greedy_medians = {first}
    
    current_min = [cost_matrix[i][first] for i in range(n)]
    
    for k in range(p - 1):
        best_node = -1
        best_improvement = -1
        
        for j in range(n):
            if j in greedy_medians:
                continue
            improvement = 0
            for i in range(n):
                diff = current_min[i] - cost_matrix[i][j]
                if diff > 0:
                    improvement += diff
            if improvement > best_improvement:
                best_improvement = improvement
                best_node = j
        
        if best_node == -1:
            # pick any non-median
            for j in range(n):
                if j not in greedy_medians:
                    best_node = j
                    break
        
        greedy_medians.add(best_node)
        for i in range(n):
            c = cost_matrix[i][best_node]
            if c < current_min[i]:
                current_min[i] = c
    
    best_medians = set(greedy_medians)
    best_obj = evaluate_solution(best_medians)
    if logger:
        logger.log(best_obj)
    
    # Phase 2: Swap-based local search (Teitz-Bart style)
    def fast_swap_search(medians):
        nonlocal best_obj, best_medians
        
        median_list = list(medians)
        median_set = set(median_list)
        non_medians = [j for j in range(n) if j not in median_set]
        
        nearest_idx, nearest_cost, second_nearest_idx, second_nearest_cost = compute_nearest_info(median_list)
        
        current_obj = sum(nearest_cost)
        
        improved = True
        while improved and (time.time() - start_time < time_limit * 0.85):
            improved = False
            best_swap_gain = 0
            best_swap = None
            
            for mi_idx in range(len(median_list)):
                if time.time() - start_time > time_limit * 0.85:
                    break
                
                for j in non_medians:
                    gain = 0
                    for i in range(n):
                        if nearest_idx[i] == mi_idx:
                            # This node's nearest is being removed
                            new_cost = min(second_nearest_cost[i], cost_matrix[i][j])
                            gain += nearest_cost[i] - new_cost
                        else:
                            # This node's nearest stays
                            c_j = cost_matrix[i][j]
                            if c_j < nearest_cost[i]:
                                gain += nearest_cost[i] - c_j
                    
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap = (mi_idx, j)
            
            if best_swap is not None and best_swap_gain > 0:
                mi_idx, j_in = best_swap
                m_out = median_list[mi_idx]
                
                median_list[mi_idx] = j_in
                non_medians.remove(j_in)
                non_medians.append(m_out)
                median_set.discard(m_out)
                median_set.add(j_in)
                
                nearest_idx, nearest_cost, second_nearest_idx, second_nearest_cost = compute_nearest_info(median_list)
                current_obj = sum(nearest_cost)
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
    
    # For small instances, try Gurobi
    if use_gurobi and (time.time() - start_time) < time_limit * 0.5:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            remaining_time = time_limit - (time.time() - start_time) - 1
            if remaining_time > 2:
                model = gp.Model("pmedian")
                model.setParam('OutputFlag', 0)
                model.setParam('TimeLimit', remaining_time)
                model.setParam('Threads', 1)
                
                y = model.addVars(n, vtype=GRB.BINARY, name="y")
                x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
                
                model.setObjective(
                    gp.quicksum(cost_matrix[i][j] * x[i, j] for i in range(n) for j in range(n)),
                    GRB.MINIMIZE
                )
                
                for i in range(n):
                    model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1)
                
                for i in range(n):
                    for j in range(n):
                        model.addConstr(x[i, j] <= y[j])
                
                model.addConstr(gp.quicksum(y[j] for j in range(n)) == p)
                
                # Warm start with best known
                for j in range(n):
                    y[j].Start = 1.0 if j in best_medians else 0.0
                
                best_assignments = get_assignments(best_medians)
                for i in range(n):
                    for j in range(n):
                        x[i, j].Start = 1.0 if best_assignments[i]['facility'] == j else 0.0
                
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
        except Exception:
            pass
    
    # Phase 3: Perturbation + local search for remaining time
    rng = random.Random(42)
    while time.time() - start_time < time_limit * 0.80:
        current_list = list(best_medians)
        non_current = [j for j in range(n) if j not in best_medians]
        
        if not non_current:
            break
        
        num_swaps = max(1, min(p // 4, len(non_current), 3))
        
        perturbed = set(current_list)
        swap_out = rng.sample(current_list, min(num_swaps, len(current_list)))
        swap_in = rng.sample(non_current, min(num_swaps, len(non_current)))
        
        for so, si in zip(swap_out, swap_in):
            perturbed.discard(so)
            perturbed.add(si)
        
        while len(perturbed) < p:
            candidates = [j for j in range(n) if j not in perturbed]
            if not candidates:
                break
            perturbed.add(rng.choice(candidates))
        
        while len(perturbed) > p:
            perturbed.pop()
        
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