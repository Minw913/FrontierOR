import argparse
import json
import time
import numpy as np

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    capacity = instance['capacity']
    weights = instance['weights']
    profit_matrix = instance['profit_matrix']

    start_time = time.time()

    # Convert to numpy for faster access
    P = np.array(profit_matrix, dtype=np.float64)
    W = np.array(weights, dtype=np.int64)

    def compute_objective(selected_indices):
        if len(selected_indices) == 0:
            return 0.0
        idx = np.array(selected_indices, dtype=np.int64)
        sub = P[np.ix_(idx, idx)]
        # Diagonal = standalone profits, upper triangle = interaction profits
        total = np.sum(np.diag(sub)) + np.sum(np.triu(sub, k=1))
        return total

    def compute_objective_from_binary(sol):
        idx = np.where(sol)[0]
        if len(idx) == 0:
            return 0.0
        sub = P[np.ix_(idx, idx)]
        return np.sum(np.diag(sub)) + np.sum(np.triu(sub, k=1))

    # For small instances, use Gurobi QP/MIP
    # For larger instances, use greedy + local search
    
    best_sol = np.zeros(n, dtype=np.int32)
    best_obj = 0.0

    if logger:
        logger.log(best_obj)

    if n <= 200:
        # Use Gurobi
        try:
            import gurobipy as gp
            from gurobipy import GRB

            remaining_time = args.time_limit - (time.time() - start_time)
            if remaining_time < 1:
                remaining_time = 1

            model = gp.Model("qkp")
            model.setParam('TimeLimit', remaining_time)
            model.setParam('OutputFlag', 0)
            model.setParam('Threads', 1)

            x = model.addVars(n, vtype=GRB.BINARY, name="x")

            # Capacity constraint
            model.addConstr(gp.quicksum(W[i] * x[i] for i in range(n)) <= capacity)

            # Objective: sum of P[i][i]*x[i] + sum_{i<j} P[i][j]*x[i]*x[j]
            obj = gp.QuadExpr()
            for i in range(n):
                obj += P[i][i] * x[i]
            for i in range(n):
                for j in range(i + 1, n):
                    if P[i][j] != 0:
                        obj += P[i][j] * x[i] * x[j]

            model.setObjective(obj, GRB.MAXIMIZE)

            # Callback for logging
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if logger:
                        logger.log(obj_val)

            model.optimize(callback)

            if model.SolCount > 0:
                sol = np.array([int(round(x[i].X)) for i in range(n)], dtype=np.int32)
                obj_val = compute_objective_from_binary(sol)
                if obj_val > best_obj:
                    best_sol = sol
                    best_obj = obj_val
                    if logger:
                        logger.log(best_obj)
        except Exception as e:
            # Fall through to greedy/local search
            pass

    # Also run greedy + local search (either as fallback or for large instances)
    remaining_time = args.time_limit - (time.time() - start_time)
    
    if remaining_time > 0.5:
        # Greedy construction based on profit density
        # For each item, compute standalone profit + sum of interaction profits with all other items (as potential)
        standalone = np.diag(P).copy()
        
        # Greedy: add items by best ratio of marginal profit to weight
        selected = set()
        current_weight = 0
        current_obj = 0.0
        sol = np.zeros(n, dtype=np.int32)
        
        # Sort by standalone profit / weight ratio (handle zero weight)
        ratios = np.zeros(n)
        for i in range(n):
            if W[i] == 0:
                ratios[i] = float('inf') if standalone[i] > 0 else 0
            else:
                ratios[i] = standalone[i] / W[i]
        
        order = np.argsort(-ratios)
        
        for i in order:
            if time.time() - start_time > args.time_limit - 0.5:
                break
            if current_weight + W[i] <= capacity:
                # Compute marginal profit of adding i
                marginal = P[i][i]
                for j in selected:
                    marginal += P[i][j]
                if marginal >= 0:
                    selected.add(i)
                    sol[i] = 1
                    current_weight += W[i]
                    current_obj += marginal

        if current_obj > best_obj:
            best_sol = sol.copy()
            best_obj = current_obj
            if logger:
                logger.log(best_obj)

        # Local search: swap/flip
        # Precompute row sums for selected items
        def marginal_add(item, selected_set):
            """Marginal profit of adding item to selected_set"""
            m = P[item][item]
            for j in selected_set:
                m += P[item][j]
            return m

        def marginal_remove(item, selected_set):
            """Marginal profit loss of removing item from selected_set"""
            m = P[item][item]
            for j in selected_set:
                if j != item:
                    m += P[item][j]
            return m

        improved = True
        iteration = 0
        while improved and (time.time() - start_time < args.time_limit - 0.5):
            improved = False
            iteration += 1
            
            selected_list = [i for i in range(n) if sol[i] == 1]
            not_selected_list = [i for i in range(n) if sol[i] == 0]
            
            # Try adding an item
            for i in not_selected_list:
                if time.time() - start_time > args.time_limit - 0.5:
                    break
                if current_weight + W[i] <= capacity:
                    mg = marginal_add(i, selected_list)
                    if mg > 0:
                        sol[i] = 1
                        selected_list.append(i)
                        current_weight += W[i]
                        current_obj += mg
                        improved = True
            
            if current_obj > best_obj:
                best_sol = sol.copy()
                best_obj = current_obj
                if logger:
                    logger.log(best_obj)

            # Try removing an item
            selected_list = [i for i in range(n) if sol[i] == 1]
            for i in list(selected_list):
                if time.time() - start_time > args.time_limit - 0.5:
                    break
                mg = marginal_remove(i, selected_list)
                if mg < 0:  # removing increases profit
                    sol[i] = 0
                    selected_list.remove(i)
                    current_weight -= W[i]
                    current_obj -= mg
                    improved = True

            if current_obj > best_obj:
                best_sol = sol.copy()
                best_obj = current_obj
                if logger:
                    logger.log(best_obj)

            # Try 1-1 swap
            selected_list = [i for i in range(n) if sol[i] == 1]
            not_selected_list = [i for i in range(n) if sol[i] == 0]
            
            best_swap = None
            best_swap_gain = 0
            
            for i in selected_list:
                if time.time() - start_time > args.time_limit - 0.5:
                    break
                remove_mg = marginal_remove(i, selected_list)
                remaining_after_remove = [j for j in selected_list if j != i]
                for j in not_selected_list:
                    if current_weight - W[i] + W[j] <= capacity:
                        add_mg = P[j][j]
                        for k in remaining_after_remove:
                            add_mg += P[j][k]
                        gain = add_mg - remove_mg
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_swap = (i, j)
            
            if best_swap is not None:
                i, j = best_swap
                sol[i] = 0
                sol[j] = 1
                current_weight = current_weight - W[i] + W[j]
                current_obj += best_swap_gain
                improved = True
                
                if current_obj > best_obj:
                    best_sol = sol.copy()
                    best_obj = current_obj
                    if logger:
                        logger.log(best_obj)

    # Final objective computation
    best_obj = compute_objective_from_binary(best_sol)

    solution = {
        "objective_value": float(best_obj),
        "solution": best_sol.tolist()
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    solve()