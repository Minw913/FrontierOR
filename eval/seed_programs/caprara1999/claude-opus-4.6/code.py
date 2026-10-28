import argparse
import json
import time
import numpy as np
from solution_logger import SolutionLogger


def compute_objective(selected, profit_matrix, n):
    """Compute the objective value for a given selection."""
    total = 0
    selected_indices = [i for i in range(n) if selected[i]]
    for i in selected_indices:
        total += profit_matrix[i][i]
        for j in selected_indices:
            if j > i:
                total += 2 * profit_matrix[i][j]
    return total


def compute_objective_fast(selected_indices, profit_mat):
    """Compute objective using numpy for speed."""
    if len(selected_indices) == 0:
        return 0
    sub = profit_mat[np.ix_(selected_indices, selected_indices)]
    # Diagonal contributes once, off-diagonal contributes fully (sum of all = diag + 2*upper_triangle)
    return int(np.sum(sub))


def solve(instance, time_limit, logger):
    n = instance['n']
    capacity = instance['capacity']
    weights = instance['weights']
    profit_matrix = instance['profit_matrix']
    
    start_time = time.time()
    
    profit_mat = np.array(profit_matrix, dtype=np.int64)
    weights_arr = np.array(weights, dtype=np.int64)
    
    # The objective for selected set S is sum_{i in S, j in S} profit_matrix[i][j]
    # which equals sum of diagonal for selected + 2 * sum of upper triangle pairs
    
    best_selected = [0] * n
    best_obj = 0
    
    def update_best(selected_list, obj_val):
        nonlocal best_selected, best_obj
        if obj_val > best_obj:
            best_obj = obj_val
            best_selected = selected_list[:]
            if logger:
                logger.log(obj_val)
    
    # Greedy construction: iteratively add item with best marginal gain
    def greedy_construct(randomize=False, rcl_size=1):
        selected = set()
        total_weight = 0
        remaining = set(range(n))
        
        while remaining:
            best_gains = []
            for i in remaining:
                if total_weight + weights[i] > capacity:
                    continue
                # Marginal gain of adding item i
                gain = profit_mat[i][i]
                for j in selected:
                    gain += 2 * profit_mat[i][j]
                best_gains.append((gain, i))
            
            if not best_gains:
                break
            
            best_gains.sort(key=lambda x: -x[0])
            
            if randomize and len(best_gains) > 1:
                k = min(rcl_size, len(best_gains))
                idx = np.random.randint(0, k)
                chosen = best_gains[idx][1]
            else:
                chosen = best_gains[0][1]
            
            if best_gains[0][0] <= 0 and not randomize:
                # Check if any positive gain exists
                if best_gains[0][0] < 0:
                    break
                # Zero gain - still add for potential future benefit if randomize
                if not randomize:
                    # Try adding zero-gain items too
                    pass
            
            selected.add(chosen)
            total_weight += weights[chosen]
            remaining.remove(chosen)
        
        return selected
    
    # Initial greedy solution
    sel = greedy_construct(randomize=False)
    sel_list = [1 if i in sel else 0 for i in range(n)]
    sel_indices = sorted(sel)
    obj = compute_objective_fast(sel_indices, profit_mat) if sel_indices else 0
    update_best(sel_list, obj)
    
    # Local search: swap-based improvement
    def local_search(selected_set, max_time=None):
        selected = set(selected_set)
        current_weight = sum(weights[i] for i in selected)
        
        # Precompute current contributions
        sel_list_local = sorted(selected)
        current_obj = compute_objective_fast(sel_list_local, profit_mat) if sel_list_local else 0
        
        improved = True
        while improved:
            if max_time and time.time() - start_time > max_time:
                break
            improved = False
            
            # Try remove
            sel_sorted = sorted(selected)
            for i in sel_sorted:
                if max_time and time.time() - start_time > max_time:
                    break
                # Marginal loss of removing i
                loss = profit_mat[i][i]
                for j in selected:
                    if j != i:
                        loss += 2 * profit_mat[i][j]
                if loss < 0:
                    selected.remove(i)
                    current_weight -= weights[i]
                    current_obj -= loss
                    improved = True
            
            # Try add
            not_selected = sorted(set(range(n)) - selected)
            for i in not_selected:
                if max_time and time.time() - start_time > max_time:
                    break
                if current_weight + weights[i] > capacity:
                    continue
                gain = profit_mat[i][i]
                for j in selected:
                    gain += 2 * profit_mat[i][j]
                if gain > 0:
                    selected.add(i)
                    current_weight += weights[i]
                    current_obj += gain
                    improved = True
            
            # Try 1-1 swap
            sel_sorted = sorted(selected)
            not_sel_sorted = sorted(set(range(n)) - selected)
            swap_done = False
            for i in sel_sorted:
                if swap_done:
                    break
                if max_time and time.time() - start_time > max_time:
                    break
                # Cost of removing i
                remove_gain = profit_mat[i][i]
                for j in selected:
                    if j != i:
                        remove_gain += 2 * profit_mat[i][j]
                
                for j in not_sel_sorted:
                    if max_time and time.time() - start_time > max_time:
                        break
                    new_weight = current_weight - weights[i] + weights[j]
                    if new_weight > capacity:
                        continue
                    # Gain of adding j (after removing i)
                    add_gain = profit_mat[j][j]
                    for k in selected:
                        if k != i:
                            add_gain += 2 * profit_mat[j][k]
                    
                    net = add_gain - remove_gain
                    if net > 0:
                        selected.remove(i)
                        selected.add(j)
                        current_weight = new_weight
                        current_obj += net
                        improved = True
                        swap_done = True
                        break
        
        return selected, current_obj
    
    # Apply local search to greedy solution
    sel, obj = local_search(sel, max_time=start_time + time_limit * 0.9)
    sel_list = [1 if i in sel else 0 for i in range(n)]
    update_best(sel_list, obj)
    
    # GRASP: repeated randomized greedy + local search
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 1.0:
        iteration = 0
        while time.time() - start_time < time_limit * 0.95:
            iteration += 1
            rcl = max(2, min(n // 4, 5))
            sel = greedy_construct(randomize=True, rcl_size=rcl)
            sel, obj = local_search(sel, max_time=start_time + time_limit * 0.95)
            sel_list = [1 if i in sel else 0 for i in range(n)]
            update_best(sel_list, obj)
    
    # Try Gurobi if n is small enough and time remains
    remaining_time = time_limit - (time.time() - start_time)
    if n <= 200 and remaining_time > 2:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            model = gp.Model("qkp")
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', remaining_time - 1)
            model.setParam('Threads', 1)
            model.setParam('Cutoff', best_obj + 0.5)
            
            x = model.addVars(n, vtype=GRB.BINARY, name="x")
            
            # Weight constraint
            model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity)
            
            # Objective
            obj_expr = gp.QuadExpr()
            for i in range(n):
                obj_expr += profit_mat[i][i] * x[i]
                for j in range(i + 1, n):
                    if profit_mat[i][j] != 0:
                        obj_expr += 2 * int(profit_mat[i][j]) * x[i] * x[j]
            
            model.setObjective(obj_expr, GRB.MAXIMIZE)
            
            # Warm start
            for i in range(n):
                x[i].Start = best_selected[i]
            
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if logger:
                        logger.log(obj_val)
            
            model.optimize(callback)
            
            if model.SolCount > 0:
                gurobi_obj = model.ObjVal
                gurobi_sel = [int(round(x[i].X)) for i in range(n)]
                if gurobi_obj > best_obj:
                    best_obj = gurobi_obj
                    best_selected = gurobi_sel
                    if logger:
                        logger.log(best_obj)
        except Exception:
            pass
    
    return best_selected, best_obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    selected, obj_val = solve(instance, args.time_limit, logger)
    
    solution = {
        "objective_value": float(obj_val),
        "selected_items": selected
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()