import argparse
import json
import sys
import time

def heuristic(n, p, cost_matrix, time_limit, start_time, logger):
    """
    Greedy add followed by fast local search (swap) to find a high-quality p-median solution.
    Runs strictly within a budgeted portion of total time limit.
    """
    medians = []
    min_dist = [float('inf')] * n
    
    # Phase 1: Greedy selection of p medians
    for step in range(p):
        if time.time() - start_time > time_limit - 1.0:
            break
            
        best_gain = -float('inf')
        best_candidate = -1
        
        for j in range(n):
            if j in medians:
                continue
                
            if step == 0:
                # First node: maximize negative sum of distances
                gain = -sum(cost_matrix[i][j] for i in range(n))
            else:
                gain = 0
                for i in range(n):
                    if cost_matrix[i][j] < min_dist[i]:
                        gain += min_dist[i] - cost_matrix[i][j]
                        
            if gain > best_gain:
                best_gain = gain
                best_candidate = j
                
        if best_candidate != -1:
            medians.append(best_candidate)
            if step == 0:
                for i in range(n):
                    min_dist[i] = cost_matrix[i][best_candidate]
            else:
                for i in range(n):
                    if cost_matrix[i][best_candidate] < min_dist[i]:
                        min_dist[i] = cost_matrix[i][best_candidate]
                        
    # Graceful timeout fallback to ensure we always have exactly `p` medians
    if len(medians) < p:
        for j in range(n):
            if len(medians) == p: 
                break
            if j not in medians: 
                medians.append(j)

    exact_obj = sum(min(cost_matrix[i][m] for m in medians) for i in range(n))
    if logger:
        logger.log(exact_obj)

    non_medians = set(range(n)) - set(medians)
    
    # Phase 2: Rapid local search with fully evaluated neighbor swaps
    improved = True
    while improved:
        if time.time() - start_time > time_limit - 1.0:
            break
        improved = False
        
        nearest_dist = [float('inf')] * n
        second_nearest_dist = [float('inf')] * n
        nearest_idx = [-1] * n
        
        for i in range(n):
            d1, d2 = float('inf'), float('inf')
            idx1 = -1
            for m in medians:
                d = cost_matrix[i][m]
                if d < d1:
                    d2 = d1
                    d1 = d
                    idx1 = m
                elif d < d2:
                    d2 = d
            nearest_dist[i] = d1
            second_nearest_dist[i] = d2
            nearest_idx[i] = idx1
            
        add_gain = [0] * n
        correction = {u: [0] * n for u in medians}
        non_med_list = list(non_medians)
        
        for i in range(n):
            u_star = nearest_idx[i]
            d1 = nearest_dist[i]
            d2 = second_nearest_dist[i]
            
            row_cost = cost_matrix[i]
            corr_u_star = correction[u_star]
            
            for v in non_med_list:
                cv = row_cost[v]
                if d1 > cv:
                    gain_v = d1 - cv
                else:
                    gain_v = 0
                    
                add_gain[v] += gain_v
                
                if d2 > cv:
                    min_d2_cv = cv
                else:
                    min_d2_cv = d2
                    
                actual_gain = d1 - min_d2_cv
                corr_u_star[v] += actual_gain - gain_v

        best_swap = None
        best_swap_gain = 0
        
        for u in medians:
            for v in non_med_list:
                gain = add_gain[v] + correction[u][v]
                if gain > best_swap_gain:
                    best_swap_gain = gain
                    best_swap = (u, v)
                    
        # Check strict integer improvement
        if best_swap_gain > 0 and best_swap:
            medians.remove(best_swap[0])
            medians.append(best_swap[1])
            non_medians.remove(best_swap[1])
            non_medians.add(best_swap[0])
            improved = True
            
            exact_obj = sum(min(cost_matrix[i][m] for m in medians) for i in range(n))
            if logger:
                logger.log(exact_obj)

    final_obj = sum(min(cost_matrix[i][m] for m in medians) for i in range(n))
    return medians, final_obj


def solve_p_median(args):
    start_time = time.time()
    
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
        
    n = data['n']
    p = data['p']
    cost_matrix = data['cost_matrix']
    
    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except ImportError:
            pass
            
    # Assign up to ~20% of maximum time, capped at 20 seconds for the heuristic phase.
    heur_time_limit = min(float(args.time_limit) * 0.20, 20.0)
    
    medians, heur_obj = heuristic(n, p, cost_matrix, heur_time_limit, start_time, logger)
    
    remaining_time = args.time_limit - (time.time() - start_time)
    
    # Launch Gurobi solver to optimize/prove optimality provided enough time remains
    # Capped at $n \leq 1500$ to circumvent out-of-memory errors on massive instance sizes
    if remaining_time > 5.0 and n <= 1500:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            env = gp.Env(empty=True)
            env.setParam("LogToConsole", 0)
            env.start()
            
            model = gp.Model("p-median", env=env)
            model.Params.TimeLimit = remaining_time - 2.0 
            model.Params.Threads = 1  
            
            class CallbackData:
                def __init__(self, c_logger, initial_obj):
                    self.logger = c_logger
                    self.best_obj = initial_obj
                    
            model._cb_data = CallbackData(logger, heur_obj)
            
            def cb(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj < model._cb_data.best_obj - 0.5:
                        model._cb_data.best_obj = obj
                        if model._cb_data.logger:
                            model._cb_data.logger.log(int(round(obj)))

            costs = {(i, j): cost_matrix[i][j] for i in range(n) for j in range(n)}
            x = model.addVars(n, n, obj=costs, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="x")
            y = model.addVars(n, vtype=GRB.BINARY, name="y")
            
            model.addConstr(y.sum() == p, "p_medians")
            model.addConstrs((x.sum(i, '*') == 1.0 for i in range(n)), "assign")
            model.addConstrs((x[i, j] <= y[j] for i in range(n) for j in range(n) if i != j), "valid")
            
            # Inject heuristic model layout
            for j in range(n):
                y[j].Start = 1.0 if j in medians else 0.0
                    
            model.optimize(cb)
            
            # Extract definitive solution variables avoiding Gurobi numerical boundary floats
            if model.SolCount > 0:
                final_obj = model.ObjVal
                if final_obj <= heur_obj - 0.5:
                    final_medians = [j for j in range(n) if y[j].X > 0.5]
                    # Absolute validation for `p` exact entries
                    if len(final_medians) != p:
                        sorted_j = sorted(range(n), key=lambda j: y[j].X, reverse=True)
                        final_medians = sorted_j[:p]
                        
                    medians = final_medians
                    
        except ImportError:
            pass  
        except Exception:
            pass  

    # Aggregate assignment mappings mapped to selected medians
    assignments = []
    total_cost = 0
    
    for i in range(n):
        best_fac = -1
        best_dist = float('inf')
        for j in medians:
            if cost_matrix[i][j] < best_dist:
                best_dist = cost_matrix[i][j]
                best_fac = j
                
        best_dist = int(best_dist)
        total_cost += best_dist
        
        assignments.append({
            "customer": i,
            "facility": best_fac,
            "cost": best_dist
        })
        
    output = {
        "objective_value": total_cost,
        "opened_facilities": medians,
        "assignments": assignments
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P-Median Optimizer")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path where final output JSON writes")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum execution time limits computationally")
    parser.add_argument("--log_path", type=str, required=False, default=None, help="Utility logger intermediary trajectory JSONL outputs")
    args = parser.parse_args()
    
    solve_p_median(args)