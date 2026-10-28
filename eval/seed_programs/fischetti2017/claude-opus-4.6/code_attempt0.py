import argparse
import json
import time
import numpy as np
import sys

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    start_time = time.time()
    
    n = data['num_facilities']
    m = data['num_customers']
    opening_costs = np.array(data['opening_costs'], dtype=np.float64)
    alloc_costs = np.array(data['allocation_costs'], dtype=np.float64)  # shape (n, m)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # For a set of open facilities, compute optimal allocation and cost
    # For customer j with open facilities S:
    # optimal x_{ij} = (1/c_{ij}) / sum_{k in S} (1/c_{kj})
    # cost_j = 1 / sum_{k in S} (1/c_{kj})
    
    inv_costs = 1.0 / alloc_costs  # shape (n, m)
    
    def evaluate_solution(open_set):
        """Given a set of open facility indices, compute total cost and allocation."""
        if len(open_set) == 0:
            return float('inf'), None
        
        open_list = list(open_set)
        fixed_cost = opening_costs[open_list].sum()
        
        # For each customer, sum of 1/c_{ij} for open facilities
        inv_sum = inv_costs[open_list].sum(axis=0)  # shape (m,)
        
        # Cost per customer = 1 / inv_sum
        alloc_cost = (1.0 / inv_sum).sum()
        
        return fixed_cost + alloc_cost, open_list
    
    def compute_allocation(open_list):
        """Compute the x matrix for given open facilities."""
        x = np.zeros((n, m), dtype=np.float64)
        if len(open_list) == 0:
            return x
        open_arr = np.array(open_list)
        inv_sum = inv_costs[open_arr].sum(axis=0)  # shape (m,)
        for i in open_arr:
            x[i, :] = inv_costs[i, :] / inv_sum
        return x
    
    def make_solution(open_set, obj_val):
        open_list = sorted(open_set)
        y = [0] * n
        for i in open_list:
            y[i] = 1
        x = compute_allocation(open_list)
        sol = {
            "objective_value": float(obj_val),
            "variant": "quadratic",
            "y": y,
            "x": x.tolist()
        }
        return sol
    
    # Precompute: for each facility i, the contribution to allocation cost reduction
    # when adding facility i to an existing set.
    # We'll use incremental evaluation.
    
    # Greedy construction: start empty, add facility with best marginal improvement
    best_obj = float('inf')
    best_open = set()
    
    # Start with each individual facility
    for i in range(n):
        obj, _ = evaluate_solution({i})
        if obj < best_obj:
            best_obj = obj
            best_open = {i}
    
    if logger:
        sol = make_solution(best_open, best_obj)
        logger.log_solution(best_obj, sol)
    
    # Greedy add
    current_open = set()
    current_inv_sum = np.zeros(m, dtype=np.float64)
    current_alloc_cost = float('inf')
    current_fixed_cost = 0.0
    
    # Actually let's do a proper greedy
    remaining = set(range(n))
    
    while remaining and time.time() - start_time < time_limit * 0.3:
        best_marginal = float('inf')
        best_fac = -1
        
        for i in remaining:
            new_inv_sum = current_inv_sum + inv_costs[i]
            new_alloc_cost = (1.0 / new_inv_sum).sum()
            new_fixed_cost = current_fixed_cost + opening_costs[i]
            total = new_fixed_cost + new_alloc_cost
            if total < best_marginal:
                best_marginal = total
                best_fac = i
        
        if best_marginal >= current_fixed_cost + current_alloc_cost and len(current_open) > 0:
            break
        
        current_open.add(best_fac)
        remaining.remove(best_fac)
        current_inv_sum += inv_costs[best_fac]
        current_alloc_cost = (1.0 / current_inv_sum).sum()
        current_fixed_cost += opening_costs[best_fac]
        total = current_fixed_cost + current_alloc_cost
        
        if total < best_obj:
            best_obj = total
            best_open = set(current_open)
            if logger:
                sol = make_solution(best_open, best_obj)
                logger.log_solution(best_obj, sol)
    
    # Local search phase
    def local_search(open_set, obj_val, deadline):
        """Perform add/drop/swap local search."""
        improved = True
        current = set(open_set)
        current_obj = obj_val
        
        # Precompute current inv_sum
        if len(current) > 0:
            cur_inv_sum = inv_costs[list(current)].sum(axis=0)
            cur_alloc = (1.0 / cur_inv_sum).sum()
            cur_fixed = sum(opening_costs[i] for i in current)
        else:
            return current, obj_val
        
        while improved and time.time() < deadline:
            improved = False
            
            # Try DROP moves
            if len(current) > 1:
                best_drop = None
                best_drop_obj = current_obj
                for i in list(current):
                    if time.time() >= deadline:
                        break
                    new_inv_sum = cur_inv_sum - inv_costs[i]
                    # Check no zeros
                    if np.any(new_inv_sum <= 0):
                        continue
                    new_alloc = (1.0 / new_inv_sum).sum()
                    new_fixed = cur_fixed - opening_costs[i]
                    new_obj = new_fixed + new_alloc
                    if new_obj < best_drop_obj - 1e-10:
                        best_drop_obj = new_obj
                        best_drop = i
                
                if best_drop is not None:
                    current.remove(best_drop)
                    cur_inv_sum = cur_inv_sum - inv_costs[best_drop]
                    cur_alloc = (1.0 / cur_inv_sum).sum()
                    cur_fixed -= opening_costs[best_drop]
                    current_obj = best_drop_obj
                    improved = True
                    continue
            
            # Try ADD moves
            closed = set(range(n)) - current
            best_add = None
            best_add_obj = current_obj
            for i in closed:
                if time.time() >= deadline:
                    break
                new_inv_sum = cur_inv_sum + inv_costs[i]
                new_alloc = (1.0 / new_inv_sum).sum()
                new_fixed = cur_fixed + opening_costs[i]
                new_obj = new_fixed + new_alloc
                if new_obj < best_add_obj - 1e-10:
                    best_add_obj = new_obj
                    best_add = i
            
            if best_add is not None:
                current.add(best_add)
                cur_inv_sum = cur_inv_sum + inv_costs[best_add]
                cur_alloc = (1.0 / cur_inv_sum).sum()
                cur_fixed += opening_costs[best_add]
                current_obj = best_add_obj
                improved = True
                continue
            
            # Try SWAP moves
            best_swap = None
            best_swap_obj = current_obj
            open_list = list(current)
            closed_list = list(set(range(n)) - current)
            
            for i_out in open_list:
                if time.time() >= deadline:
                    break
                if len(current) == 1:
                    base_inv_sum = np.zeros(m, dtype=np.float64)
                else:
                    base_inv_sum = cur_inv_sum - inv_costs[i_out]
                base_fixed = cur_fixed - opening_costs[i_out]
                
                for i_in in closed_list:
                    if time.time() >= deadline:
                        break
                    new_inv_sum = base_inv_sum + inv_costs[i_in]
                    if np.any(new_inv_sum <= 0):
                        continue
                    new_alloc = (1.0 / new_inv_sum).sum()
                    new_fixed = base_fixed + opening_costs[i_in]
                    new_obj = new_fixed + new_alloc
                    if new_obj < best_swap_obj - 1e-10:
                        best_swap_obj = new_obj
                        best_swap = (i_out, i_in)
            
            if best_swap is not None:
                i_out, i_in = best_swap
                current.remove(i_out)
                current.add(i_in)
                cur_inv_sum = cur_inv_sum - inv_costs[i_out] + inv_costs[i_in]
                cur_alloc = (1.0 / cur_inv_sum).sum()
                cur_fixed = cur_fixed - opening_costs[i_out] + opening_costs[i_in]
                current_obj = best_swap_obj
                improved = True
                continue
        
        return current, current_obj
    
    deadline = start_time + time_limit * 0.85
    
    # Local search from greedy solution
    ls_open, ls_obj = local_search(best_open, best_obj, deadline)
    if ls_obj < best_obj:
        best_obj = ls_obj
        best_open = ls_open
        if logger:
            sol = make_solution(best_open, best_obj)
            logger.log_solution(best_obj, sol)
    
    # Try multiple random restarts if time permits
    np.random.seed(42)
    
    # Also try starting from each single facility
    if n <= 200 and time.time() < deadline:
        single_objs = []
        for i in range(n):
            obj_i = opening_costs[i] + (1.0 / inv_costs[i]).sum()
            single_objs.append((obj_i, i))
        single_objs.sort()
        
        # Try local search from top-k single facilities
        for obj_i, i in single_objs[:min(10, n)]:
            if time.time() >= deadline:
                break
            ls_open_i, ls_obj_i = local_search({i}, obj_i, deadline)
            if ls_obj_i < best_obj:
                best_obj = ls_obj_i
                best_open = ls_open_i
                if logger:
                    sol = make_solution(best_open, best_obj)
                    logger.log_solution(best_obj, sol)
    
    # Random restarts
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        # Random subset of facilities
        k = max(1, np.random.randint(1, max(2, n // 2 + 1)))
        rand_open = set(np.random.choice(n, size=k, replace=False).tolist())
        rand_inv_sum = inv_costs[list(rand_open)].sum(axis=0)
        if np.any(rand_inv_sum <= 0):
            continue
        rand_alloc = (1.0 / rand_inv_sum).sum()
        rand_fixed = sum(opening_costs[i] for i in rand_open)
        rand_obj = rand_fixed + rand_alloc
        
        ls_open_r, ls_obj_r = local_search(rand_open, rand_obj, deadline)
        if ls_obj_r < best_obj:
            best_obj = ls_obj_r
            best_open = ls_open_r
            if logger:
                sol = make_solution(best_open, best_obj)
                logger.log_solution(best_obj, sol)
    
    # Try Gurobi for small instances if time remains
    gurobi_deadline = start_time + time_limit * 0.98
    remaining_time = gurobi_deadline - time.time()
    
    if remaining_time > 5 and n * m <= 50000:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            model = gp.Model("qUFL")
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(1, remaining_time)
            model.Params.OutputFlag = 0
            
            y = model.addVars(n, vtype=GRB.BINARY, name="y")
            x = model.addVars(n, m, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")
            
            # Set warm start from best solution
            for i in range(n):
                if i in best_open:
                    y[i].Start = 1
                else:
                    y[i].Start = 0
            
            x_warm = compute_allocation(sorted(best_open))
            for i in range(n):
                for j in range(m):
                    x[i, j].Start = x_warm[i, j]
            
            # Objective
            obj = gp.QuadExpr()
            for i in range(n):
                obj += opening_costs[i] * y[i]
            for i in range(n):
                for j in range(m):
                    obj += alloc_costs[i][j] * x[i, j] * x[i, j]
            
            model.setObjective(obj, GRB.MINIMIZE)
            
            # Constraints: each customer fully served
            for j in range(m):
                model.addConstr(gp.quicksum(x[i, j] for i in range(n)) == 1)
            
            # Linking constraints: x[i,j] <= y[i]
            for i in range(n):
                for j in range(m):
                    model.addConstr(x[i, j] <= y[i])
            
            def gurobi_callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    nonlocal best_obj, best_open
                    obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj_val < best_obj - 1e-10:
                        y_vals = model.cbGetSolution([y[i] for i in range(n)])
                        x_vals = {}
                        for i in range(n):
                            for j in range(m):
                                x_vals[i, j] = model.cbGetSolution([x[i, j]])[0]
                        
                        open_set = set()
                        y_out = [0] * n
                        x_out = [[0.0] * m for _ in range(n)]
                        for i in range(n):
                            if y_vals[i] > 0.5:
                                y_out[i] = 1
                                open_set.add(i)
                            for j in range(m):
                                x_out[i][j] = x_vals[i, j]
                        
                        best_obj = obj_val
                        best_open = open_set
                        sol = {
                            "objective_value": float(obj_val),
                            "variant": "quadratic",
                            "y": y_out,
                            "x": x_out
                        }
                        if logger:
                            logger.log_solution(obj_val, sol)
            
            model.optimize(gurobi_callback)
            
            if model.SolCount > 0:
                gurobi_obj = model.ObjVal
                if gurobi_obj < best_obj:
                    best_obj = gurobi_obj
                    y_out = [0] * n
                    x_out = [[0.0] * m for _ in range(n)]
                    best_open = set()
                    for i in range(n):
                        if y[i].X > 0.5:
                            y_out[i] = 1
                            best_open.add(i)
                        for j in range(m):
                            x_out[i][j] = x[i, j].X
                    
                    final_sol = {
                        "objective_value": float(best_obj),
                        "variant": "quadratic",
                        "y": y_out,
                        "x": x_out
                    }
                    if logger:
                        logger.log_solution(best_obj, final_sol)
                    
                    with open(solution_path, 'w') as f:
                        json.dump(final_sol, f)
                    return
        except Exception as e:
            pass  # Fall through to heuristic solution
    
    # Build final solution
    final_sol = make_solution(best_open, best_obj)
    
    # Verify objective
    y_arr = np.array(final_sol['y'])
    x_arr = np.array(final_sol['x'])
    verify_obj = (opening_costs * y_arr).sum() + (alloc_costs * x_arr * x_arr).sum()
    final_sol['objective_value'] = float(verify_obj)
    
    if logger:
        logger.log_solution(verify_obj, final_sol)
    
    with open(solution_path, 'w') as f:
        json.dump(final_sol, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)