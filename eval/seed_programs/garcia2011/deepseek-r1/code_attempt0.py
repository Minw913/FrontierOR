import argparse
import json
import random
import time
import gurobipy as gp
from solution_logger import SolutionLogger

def solve_mip(n, p, cost_matrix, time_remaining, logger=None):
    try:
        model = gp.Model()
        model.setParam('OutputFlag', 0)
        model.setParam('Threads', 1)
        
        y = model.addVars(n, vtype=gp.GRB.BINARY, name="y")
        x = model.addVars(n, n, vtype=gp.GRB.BINARY, name="x")
        
        obj = gp.quicksum(cost_matrix[i][j] * x[i, j] for i in range(n) for j in range(n))
        model.setObjective(obj, gp.GRB.MINIMIZE)
        
        model.addConstr(y.sum() == p, "p_medians")
        
        for i in range(n):
            model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1, f"assign_{i}")
            
        for i in range(n):
            for j in range(n):
                model.addConstr(x[i, j] <= y[j], f"open_{i}_{j}")
                
        model.setParam('TimeLimit', max(0.1, time_remaining))
        model.optimize()
        
        if model.SolCount > 0:
            total_cost = model.ObjVal
            opened_facilities = [j for j in range(n) if y[j].X > 0.5]
            
            assignments = []
            for i in range(n):
                for j in range(n):
                    if x[i, j].X > 0.5:
                        assignments.append({
                            'customer': i,
                            'facility': j,
                            'cost': cost_matrix[i][j]
                        })
                        break
            
            solution = {
                'objective_value': int(round(total_cost)),
                'opened_facilities': opened_facilities,
                'assignments': assignments
            }
            
            if logger:
                logger.log(int(round(total_cost)))
                
            return solution
        else:
            return None
    except:
        return None

def generate_random_solution(n, p, cost_matrix, logger=None):
    S = set(random.sample(range(n), p))
    total_cost = 0
    assignments = []
    for i in range(n):
        min_cost = float('inf')
        best_j = -1
        for j in S:
            if cost_matrix[i][j] < min_cost:
                min_cost = cost_matrix[i][j]
                best_j = j
        total_cost += min_cost
        assignments.append({
            'customer': i,
            'facility': best_j,
            'cost': min_cost
        })
    
    solution = {
        'objective_value': int(total_cost),
        'opened_facilities': list(S),
        'assignments': assignments
    }
    
    if logger:
        logger.log(int(total_cost))
        
    return solution

def solve_heuristic(n, p, cost_matrix, time_remaining, logger=None):
    start_time = time.time()
    
    if time_remaining <= 0:
        return generate_random_solution(n, p, cost_matrix, logger)
    
    if n <= 1000 and p <= 100:
        S = set()
        non_S = set(range(n))
        
        best_j = None
        best_total = float('inf')
        for j in range(n):
            total = 0
            for i in range(n):
                total += cost_matrix[i][j]
            if total < best_total:
                best_total = total
                best_j = j
        
        if best_j is None:
            return generate_random_solution(n, p, cost_matrix, logger)
        
        S.add(best_j)
        non_S.remove(best_j)
        
        current_best = [float('inf')] * n
        for i in range(n):
            current_best[i] = cost_matrix[i][best_j]
        
        total_cost = best_total
        
        for k in range(1, p):
            if time.time() - start_time >= time_remaining:
                break
                
            best_candidate = None
            best_candidate_cost = float('inf')
            
            for j in non_S:
                total = 0
                for i in range(n):
                    total += min(current_best[i], cost_matrix[i][j])
                if total < best_candidate_cost:
                    best_candidate_cost = total
                    best_candidate = j
            
            if best_candidate is None:
                break
                
            S.add(best_candidate)
            non_S.remove(best_candidate)
            
            for i in range(n):
                if cost_matrix[i][best_candidate] < current_best[i]:
                    current_best[i] = cost_matrix[i][best_candidate]
            
            total_cost = best_candidate_cost
        
        best_index = [-1] * n
        best_cost_arr = [0] * n
        second_index = [-1] * n
        second_best_arr = [0] * n
        
        for i in range(n):
            min1 = float('inf')
            min2 = float('inf')
            idx1 = -1
            idx2 = -1
            for j in S:
                c = cost_matrix[i][j]
                if c < min1:
                    min2 = min1
                    idx2 = idx1
                    min1 = c
                    idx1 = j
                elif c < min2:
                    min2 = c
                    idx2 = j
            best_index[i] = idx1
            best_cost_arr[i] = min1
            second_index[i] = idx2
            second_best_arr[i] = min2
        
        total_cost = sum(best_cost_arr)
        if logger:
            logger.log(int(total_cost))
    else:
        solution = generate_random_solution(n, p, cost_matrix, logger)
        S = set(solution['opened_facilities'])
        non_S = set(range(n)) - S
        assignments = solution['assignments']
        total_cost = solution['objective_value']
        
        best_index = [-1] * n
        best_cost_arr = [0] * n
        second_index = [-1] * n
        second_best_arr = [0] * n
        
        for i in range(n):
            min1 = float('inf')
            min2 = float('inf')
            idx1 = -1
            idx2 = -1
            for j in S:
                c = cost_matrix[i][j]
                if c < min1:
                    min2 = min1
                    idx2 = idx1
                    min1 = c
                    idx1 = j
                elif c < min2:
                    min2 = c
                    idx2 = j
            best_index[i] = idx1
            best_cost_arr[i] = min1
            second_index[i] = idx2
            second_best_arr[i] = min2
        
        total_cost = sum(best_cost_arr)
        if logger:
            logger.log(int(total_cost))
    
    improved = True
    while improved and (time.time() - start_time < time_remaining):
        improved = False
        S_list = list(S)
        non_S_list = list(non_S)
        random.shuffle(S_list)
        random.shuffle(non_S_list)
        
        for s in S_list:
            if time.time() - start_time >= time_remaining:
                break
                
            base_cost_i = [0] * n
            for i in range(n):
                if best_index[i] == s:
                    base_cost_i[i] = second_best_arr[i]
                else:
                    base_cost_i[i] = best_cost_arr[i]
            
            for t in non_S_list:
                if time.time() - start_time >= time_remaining:
                    break
                    
                new_cost = 0
                for i in range(n):
                    new_cost += min(base_cost_i[i], cost_matrix[i][t])
                
                if new_cost < total_cost - 1e-5:
                    S.remove(s)
                    S.add(t)
                    non_S.remove(t)
                    non_S.add(s)
                    
                    for i in range(n):
                        min1 = float('inf')
                        min2 = float('inf')
                        idx1 = -1
                        idx2 = -1
                        for j in S:
                            c = cost_matrix[i][j]
                            if c < min1:
                                min2 = min1
                                idx2 = idx1
                                min1 = c
                                idx1 = j
                            elif c < min2:
                                min2 = c
                                idx2 = j
                        best_index[i] = idx1
                        best_cost_arr[i] = min1
                        second_index[i] = idx2
                        second_best_arr[i] = min2
                    
                    total_cost = sum(best_cost_arr)
                    if logger:
                        logger.log(int(total_cost))
                    
                    improved = True
                    break
            if improved:
                break
    
    opened_facilities = list(S)
    assignments = []
    for i in range(n):
        assignments.append({
            'customer': i,
            'facility': best_index[i],
            'cost': best_cost_arr[i]
        })
    
    solution = {
        'objective_value': int(total_cost),
        'opened_facilities': opened_facilities,
        'assignments': assignments
    }
    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    n = instance['n']
    p = instance['p']
    cost_matrix = instance['cost_matrix']
    
    start_time = time.time()
    time_remaining = args.time_limit
    
    solution = None
    if n <= 50 and p <= 10:
        time_remaining_mip = max(0, time_remaining - (time.time() - start_time))
        if time_remaining_mip > 0:
            solution = solve_mip(n, p, cost_matrix, time_remaining_mip, logger)
    
    if solution is None:
        time_remaining_heuristic = max(0, time_remaining - (time.time() - start_time))
        if time_remaining_heuristic <= 0:
            solution = generate_random_solution(n, p, cost_matrix, logger)
        else:
            solution = solve_heuristic(n, p, cost_matrix, time_remaining_heuristic, logger)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()