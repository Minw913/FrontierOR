import argparse
import json
import random
import time
import math
import sys
from solution_logger import SolutionLogger

def euclidean_distance(coord1, coord2):
    return round(math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2))

def compute_cost_matrix(node_coordinates):
    n = len(node_coordinates)
    cost_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            cost = euclidean_distance(node_coordinates[i], node_coordinates[j])
            cost_matrix[i][j] = cost
            cost_matrix[j][i] = cost
    return cost_matrix

def get_cost(i, j, cost_matrix, node_coordinates):
    try:
        if cost_matrix is not None:
            return cost_matrix[i][j]
    except (TypeError, IndexError):
        pass
    
    if node_coordinates is not None:
        return euclidean_distance(node_coordinates[i], node_coordinates[j])
    
    return 0

def solve_mip(n, p, cost_matrix, node_coordinates, time_remaining, logger=None):
    try:
        import gurobipy as gp
    except ImportError:
        return None
        
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

def generate_random_solution(n, p, cost_matrix, node_coordinates, logger=None):
    try:
        S = set(random.sample(range(n), p))
        total_cost = 0
        assignments = []
        for i in range(n):
            min_cost = float('inf')
            best_j = -1
            for j in S:
                cost_ij = get_cost(i, j, cost_matrix, node_coordinates)
                if cost_ij < min_cost:
                    min_cost = cost_ij
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
    except:
        # Fallback: open first p facilities and assign everyone to first facility
        opened_facilities = list(range(p))
        assignments = []
        total_cost = 0
        for i in range(n):
            cost_val = 0
            if node_coordinates is not None and i < len(node_coordinates):
                cost_val = euclidean_distance(node_coordinates[i], node_coordinates[0])
            assignments.append({
                'customer': i,
                'facility': 0,
                'cost': cost_val
            })
            total_cost += cost_val
        solution = {
            'objective_value': int(total_cost),
            'opened_facilities': opened_facilities,
            'assignments': assignments
        }
        if logger:
            logger.log(int(total_cost))
        return solution

def solve_heuristic(n, p, cost_matrix, node_coordinates, time_remaining, logger=None):
    try:
        start_time = time.time()
        if time_remaining <= 0:
            return generate_random_solution(n, p, cost_matrix, node_coordinates, logger)
        
        best_j = None
        best_total = float('inf')
        for j in range(n):
            total = 0
            for i in range(n):
                total += get_cost(i, j, cost_matrix, node_coordinates)
            if total < best_total:
                best_total = total
                best_j = j
        
        if best_j is None:
            return generate_random_solution(n, p, cost_matrix, node_coordinates, logger)
        
        S = {best_j}
        non_S = set(range(n)) - S
        
        if p == 1:
            assignments = []
            total_cost = 0
            for i in range(n):
                cost_i = get_cost(i, best_j, cost_matrix, node_coordinates)
                total_cost += cost_i
                assignments.append({
                    'customer': i,
                    'facility': best_j,
                    'cost': cost_i
                })
            solution = {
                'objective_value': int(total_cost),
                'opened_facilities': [best_j],
                'assignments': assignments
            }
            if logger:
                logger.log(int(total_cost))
            return solution
        
        current_best = [get_cost(i, best_j, cost_matrix, node_coordinates) for i in range(n)]
        total_cost = best_total
        for k in range(1, p):
            if time.time() - start_time >= time_remaining:
                additional = random.sample(non_S, p - len(S))
                S.update(additional)
                non_S = non_S - set(additional)
                break
                
            best_candidate = None
            best_candidate_cost = float('inf')
            non_S_list = list(non_S)
            random.shuffle(non_S_list)
            
            max_candidates = min(100, len(non_S_list))
            for j in non_S_list[:max_candidates]:
                total = 0
                for i in range(n):
                    cost_ij = get_cost(i, j, cost_matrix, node_coordinates)
                    total += min(current_best[i], cost_ij)
                if total < best_candidate_cost:
                    best_candidate_cost = total
                    best_candidate = j
                if time.time() - start_time >= time_remaining:
                    break
            
            if best_candidate is None:
                break
                
            S.add(best_candidate)
            non_S.remove(best_candidate)
            
            for i in range(n):
                cost_ij = get_cost(i, best_candidate, cost_matrix, node_coordinates)
                if cost_ij < current_best[i]:
                    current_best[i] = cost_ij
            
            total_cost = best_candidate_cost
        
        if len(S) < p:
            additional = random.sample(non_S, p - len(S))
            S.update(additional)
            non_S = non_S - set(additional)
        
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
                c = get_cost(i, j, cost_matrix, node_coordinates)
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
        try:
            while improved and (time.time() - start_time < time_remaining):
                improved = False
                S_list = list(S)
                non_S_list = list(non_S)
                random.shuffle(S_list)
                random.shuffle(non_S_list)
                
                MAX_CANDIDATES = 100
                candidate_found = False
                
                for s in S_list:
                    if time.time() - start_time >= time_remaining:
                        break
                    candidates_to_try = non_S_list[:MAX_CANDIDATES]
                    for t in candidates_to_try:
                        if time.time() - start_time >= time_remaining:
                            break
                        if s not in S or t not in non_S:
                            continue
                            
                        delta = 0
                        for i in range(n):
                            if best_index[i] == s:
                                new_cost_i = min(second_best_arr[i], get_cost(i, t, cost_matrix, node_coordinates))
                                delta += new_cost_i - best_cost_arr[i]
                            else:
                                cost_it = get_cost(i, t, cost_matrix, node_coordinates)
                                if best_cost_arr[i] > cost_it:
                                    delta += cost_it - best_cost_arr[i]
                        
                        if delta < 0:
                            S.remove(s)
                            non_S.remove(t)
                            S.add(t)
                            non_S.add(s)
                            
                            for i in range(n):
                                min1 = float('inf')
                                min2 = float('inf')
                                idx1 = -1
                                idx2 = -1
                                for j in S:
                                    c = get_cost(i, j, cost_matrix, node_coordinates)
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
                            
                            total_cost += delta
                            if logger:
                                logger.log(int(total_cost))
                            
                            improved = True
                            candidate_found = True
                            break
                    if candidate_found:
                        break
        except:
            pass
        
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
    except:
        return generate_random_solution(n, p, cost_matrix, node_coordinates, logger)

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
    node_coordinates = instance.get('node_coordinates')
    
    cost_matrix = None
    if 'cost_matrix' in instance:
        cost_matrix = instance['cost_matrix']
    elif node_coordinates is not None:
        cost_matrix = compute_cost_matrix(node_coordinates)
    
    start_time = time.time()
    time_remaining = args.time_limit
    
    solution = None
    if n <= 50 and p <= 10:
        time_remaining_mip = max(0, time_remaining - (time.time() - start_time))
        if time_remaining_mip > 0:
            solution = solve_mip(n, p, cost_matrix, node_coordinates, time_remaining_mip, logger)
    
    if solution is None:
        time_remaining_heuristic = max(0, time_remaining - (time.time() - start_time))
        if time_remaining_heuristic <= 0:
            solution = generate_random_solution(n, p, cost_matrix, node_coordinates, logger)
        else:
            solution = solve_heuristic(n, p, cost_matrix, node_coordinates, time_remaining_heuristic, logger)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()