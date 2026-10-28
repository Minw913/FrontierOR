import argparse
import json
import time
import random
from operator import itemgetter

def compute_exact_obj(n, cost_matrix, medians):
    """
    Computes exact objective distance extremely fast using itemgetter optimizations 
    bypassing traditional Python loop bottlenecks.
    """
    if not medians:
        return 0
    if len(medians) == 1:
        m = medians[0]
        return sum(cost_matrix[i][m] for i in range(n))
        
    getter = itemgetter(*medians)
    total = 0
    for i in range(n):
        total += min(getter(cost_matrix[i]))
    return total

def heuristic(n, p, cost_matrix, deadline, logger):
    """
    Greedy additive procedure followed by targeted local search (Teitz and Bart interchange).
    Dynamically tracks boundaries to guarantee validity under strict time limits.
    """
    medians = []
    is_median = [False] * n
    min_dist = [float('inf')] * n
    out_of_time = False
    
    non_medians = set(range(n))
    
    # Phase 1: Greedy selection of p medians
    for step in range(p):
        if out_of_time or time.time() > deadline:
            out_of_time = True
            break
            
        gain = [0] * n
        
        # Step 0 computes global centroids globally, subsequent limits randomly select.
        if step == 0:
            for i in range(n):
                if i % 250 == 0 and time.time() > deadline:
                    out_of_time = True
                    break
                row_cost = cost_matrix[i]
                for j in range(n):
                    gain[j] -= row_cost[j]
            candidates = list(range(n))
        else:
            cand_size = 400
            if len(non_medians) <= cand_size:
                candidates = list(non_medians)
            else:
                candidates = random.sample(list(non_medians), cand_size)
                
            for i in range(n):
                if i % 250 == 0 and time.time() > deadline:
                    out_of_time = True
                    break
                row_cost = cost_matrix[i]
                md = min_dist[i]
                for j in candidates:
                    c = row_cost[j]
                    if c < md:
                        gain[j] += md - c

        if out_of_time:
            break
            
        best_gain = -float('inf')
        best_candidate = -1
        
        for j in candidates:
            if not is_median[j] and gain[j] > best_gain:
                best_gain = gain[j]
                best_candidate = j
                
        if best_candidate != -1:
            medians.append(best_candidate)
            is_median[best_candidate] = True
            non_medians.remove(best_candidate)
            
            if step == 0:
                for i in range(n):
                    min_dist[i] = cost_matrix[i][best_candidate]
            else:
                for i in range(n):
                    c = cost_matrix[i][best_candidate]
                    if c < min_dist[i]:
                        min_dist[i] = c
                        
    # Graceful timeout fallback guaranteeing purely structurally valid arrays filling medians
    if out_of_time or len(medians) < p:
        for j in range(n):
            if len(medians) == p: 
                break
            if not is_median[j]: 
                medians.append(j)
                is_median[j] = True
                if j in non_medians:
                    non_medians.remove(j)

    exact_obj = compute_exact_obj(n, cost_matrix, medians)
        
    if logger:
        logger.log(exact_obj)

    # Phase 2: Memory-efficient subset optimized native local search 
    improved = True
    while improved:
        if time.time() > deadline:
            break
            
        improved = False
        out_of_time = False
        
        nearest_dist = [float('inf')] * n
        second_nearest_dist = [float('inf')] * n
        nearest_idx = [-1] * n
        assigned_to = {u: [] for u in medians}
        
        for i in range(n):
            if i % 250 == 0 and time.time() > deadline:
                out_of_time = True
                break
                
            d1, d2 = float('inf'), float('inf')
            idx1 = -1
            row_cost = cost_matrix[i]
            for m in medians:
                d = row_cost[m]
                if d < d1:
                    d2 = d1
                    d1 = d
                    idx1 = m
                elif d < d2:
                    d2 = d
            nearest_dist[i] = d1
            second_nearest_dist[i] = d2
            nearest_idx[i] = idx1
            if idx1 != -1:
                assigned_to[idx1].append(i)
                
        if out_of_time:
            break
            
        non_med_list = list(non_medians)
        base_gain = [0] * n
        
        for i in range(n):
            if i % 250 == 0 and time.time() > deadline:
                out_of_time = True
                break
                
            d1 = nearest_dist[i]
            row_cost = cost_matrix[i]
            for v in non_med_list:
                cv = row_cost[v]
                if cv < d1:
                    base_gain[v] += d1 - cv
                    
        if out_of_time:
            break

        best_swap_gain = 0
        best_swap = None

        for u in medians:
            if out_of_time:
                break
                
            specific_gain = base_gain[:]
            count_i = 0
            for i in assigned_to[u]:
                count_i += 1
                if count_i % 250 == 0 and time.time() > deadline:
                    out_of_time = True
                    break
                    
                d1 = nearest_dist[i]
                d2 = second_nearest_dist[i]
                row_cost = cost_matrix[i]
                diff = d1 - d2
                for v in non_med_list:
                    cv = row_cost[v]
                    if cv < d1:
                        pass
                    elif cv < d2:
                        specific_gain[v] += d1 - cv
                    else:
                        specific_gain[v] += diff
                        
            if out_of_time:
                break
                
            for v in non_med_list:
                gain = specific_gain[v]
                if gain > best_swap_gain:
                    best_swap_gain = gain
                    best_swap = (u, v)

        if out_of_time or best_swap_gain <= 0 or not best_swap:
            break
            
        medians.remove(best_swap[0])
        medians.append(best_swap[1])
        non_medians.remove(best_swap[1])
        non_medians.add(best_swap[0])
        is_median[best_swap[0]] = False
        is_median[best_swap[1]] = True
        improved = True
        
        exact_obj -= int(best_swap_gain)
        if logger:
            logger.log(exact_obj)

    final_obj = compute_exact_obj(n, cost_matrix, medians)

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
            
    # Buffer properly resolves boundaries escaping external framework rigid constraints preventing abrupt code=1 deaths entirely
    deadline = start_time + max(1.0, float(args.time_limit) - 5.0)
    
    medians, heur_obj = heuristic(n, p, cost_matrix, deadline, logger)
    remaining_time = deadline + 5.0 - time.time()
    
    # Optional Gurobi execution gracefully handles precise optimal bounds securely for specifically miniature sizes safely
    if remaining_time > 5.0 and n <= 400:
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
            
            for j in range(n):
                y[j].Start = 1.0 if j in medians else 0.0
                    
            model.optimize(cb)
            
            if model.SolCount > 0:
                final_obj = model.ObjVal
                if final_obj <= heur_obj - 0.5:
                    final_medians = [j for j in range(n) if y[j].X > 0.5]
                    if len(final_medians) != p:
                        sorted_j = sorted(range(n), key=lambda j: y[j].X, reverse=True)
                        final_medians = sorted_j[:p]
                    medians = final_medians
                    
        except ImportError:
            pass  
        except Exception:
            pass  

    assignments = []
    total_cost = 0
    
    # Establish consistently sound verified assignments optimizing un-interruptible fast execution using standard tuples iteration internally
    if len(medians) == 1:
        m = medians[0]
        for i in range(n):
            c = cost_matrix[i][m]
            total_cost += c
            assignments.append({
                "customer": i,
                "facility": m,
                "cost": c
            })
    else:
        getter = itemgetter(*medians)
        for i in range(n):
            row_cost = cost_matrix[i]
            costs = getter(row_cost)
            best_dist = min(costs)
            best_fac = medians[costs.index(best_dist)]
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
    parser.add_argument("--instance_path", type=str, required=True, help="Path pointing to instance JSON schemas")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write formatting results to designated location")
    parser.add_argument("--time_limit", type=int, required=True, help="Hardened constraint limiting execution bounds mathematically securely")
    parser.add_argument("--log_path", type=str, required=False, default=None, help="Utility module payload trajectory intermediary parameters")
    args = parser.parse_args()
    
    solve_p_median(args)