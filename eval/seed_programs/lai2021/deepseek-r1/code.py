import json
import argparse
import time
import random
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def compute_row_starts(N):
    row_start = [0] * N
    if N <= 1:
        return row_start
    for i in range(1, N):
        row_start[i] = row_start[i-1] + (N - i)
    return row_start

def compute_total_d_from_upper_tri(N, distances_upper_triangular):
    total_d = [0.0] * N
    idx = 0
    for i in range(N-1):
        for j in range(i+1, N):
            d_val = distances_upper_triangular[idx]
            total_d[i] += d_val
            total_d[j] += d_val
            idx += 1
    return total_d

def compute_objective(groups, get_distance):
    total = 0.0
    for group in groups:
        elements = group
        n_in_group = len(elements)
        for i_idx in range(n_in_group):
            for j_idx in range(i_idx+1, n_in_group):
                i = elements[i_idx]
                j = elements[j_idx]
                total += get_distance(i, j)
    return total

def greedy_construction(N, m, group_capacities, get_distance, total_d):
    L_g = [group_capacities[g]['L_g'] for g in range(m)]
    U_g = [group_capacities[g]['U_g'] for g in range(m)]
    
    total_lower = sum(L_g)
    if total_lower > N:
        raise Exception("Infeasible: total lower capacity exceeds N")
    if sum(U_g) < N:
        raise Exception("Infeasible: total upper capacity less than N")
    
    groups = [[] for _ in range(m)]
    group_sizes = [0] * m
    remaining = list(range(N))
    
    sorted_elements = sorted(remaining, key=lambda i: total_d[i], reverse=True)
    remaining = sorted_elements[:]
    
    for g in range(m):
        needed = L_g[g]
        for _ in range(needed):
            if not remaining:
                break
            if group_sizes[g] == 0:
                elem = remaining.pop(0)
                groups[g].append(elem)
                group_sizes[g] += 1
            else:
                best_elem = None
                best_gain = -10**15
                for elem in remaining:
                    gain = 0.0
                    for member in groups[g]:
                        gain += get_distance(elem, member)
                    if gain > best_gain:
                        best_gain = gain
                        best_elem = elem
                groups[g].append(best_elem)
                remaining.remove(best_elem)
                group_sizes[g] += 1
    
    while remaining:
        elem = remaining.pop(0)
        best_group = None
        best_gain = -10**15
        for g in range(m):
            if group_sizes[g] < U_g[g]:
                gain = 0.0
                for member in groups[g]:
                    gain += get_distance(elem, member)
                if gain > best_gain:
                    best_gain = gain
                    best_group = g
        if best_group is None:
            best_group = 0
        groups[best_group].append(elem)
        group_sizes[best_group] += 1
    
    assignment = [-1] * N
    for g in range(m):
        for elem in groups[g]:
            assignment[elem] = g
    
    return assignment, groups

def local_search(assignment, groups, get_distance, group_capacities, time_limit, update_best_func=None):
    m = len(group_capacities)
    L_g = [group_capacities[g]['L_g'] for g in range(m)]
    U_g = [group_capacities[g]['U_g'] for g in range(m)]
    group_sizes = [len(group) for group in groups]
    current_obj = compute_objective(groups, get_distance)
    start_time = time.time()
    N = len(assignment)
    
    if update_best_func:
        update_best_func(current_obj, (assignment, groups))
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        order = list(range(N))
        random.shuffle(order)
        for i in order:
            if time.time() - start_time >= time_limit:
                break
            g0 = assignment[i]
            best_delta = 0.0
            best_group = None
            for h in range(m):
                if h == g0:
                    continue
                if group_sizes[g0] <= L_g[g0]:
                    break
                if group_sizes[h] >= U_g[h]:
                    continue
                loss = 0.0
                for j in groups[g0]:
                    if j == i:
                        continue
                    loss += get_distance(i, j)
                gain = 0.0
                for j in groups[h]:
                    gain += get_distance(i, j)
                delta = gain - loss
                if delta > best_delta:
                    best_delta = delta
                    best_group = h
            if best_delta > 0:
                assignment[i] = best_group
                groups[g0].remove(i)
                groups[best_group].append(i)
                group_sizes[g0] -= 1
                group_sizes[best_group] += 1
                current_obj += best_delta
                improved = True
                if update_best_func:
                    update_best_func(current_obj, (assignment, groups))
                break
    return assignment, groups, current_obj

def build_milp_model(N, m, group_capacities, d_matrix):
    model = gp.Model()
    model.setParam("Threads", 1)
    L_g = [group_capacities[g]['L_g'] for g in range(m)]
    U_g = [group_capacities[g]['U_g'] for g in range(m)]
    
    x = model.addVars(N, m, vtype=GRB.BINARY, name="x")
    y = {}
    for i in range(N):
        for j in range(i + 1, N):
            y[(i, j)] = model.addVar(lb=0.0, ub=1.0, name=f"y_{i}_{j}")
    
    for i in range(N):
        model.addConstr(gp.quicksum(x[i, g] for g in range(m)) == 1, name=f"assign_{i}")
    
    for g in range(m):
        total_in_group = gp.quicksum(x[i, g] for i in range(N))
        model.addConstr(total_in_group >= L_g[g], name=f"cap_lb_{g}")
        model.addConstr(total_in_group <= U_g[g], name=f"cap_ub_{g}")
    
    for g in range(m):
        for i in range(N):
            for j in range(i + 1, N):
                model.addConstr(y[(i, j)] >= x[i, g] + x[j, g] - 1, name=f"lb_{i}_{j}_{g}")
                model.addConstr(y[(i, j)] <= x[i, g], name=f"ub1_{i}_{j}_{g}")
                model.addConstr(y[(i, j)] <= x[j, g], name=f"ub2_{i}_{j}_{g}")
    
    obj = gp.quicksum(d_matrix[i][j] * y[(i, j)] for i in range(N) for j in range(i + 1, N))
    model.setObjective(obj, GRB.MAXIMIZE)
    return model, x, y

def extract_milp_solution(model, x, N, m):
    assignment_list = [-1] * N
    for i in range(N):
        for g in range(m):
            if abs(x[i, g].X - 1.0) < 1e-5:
                assignment_list[i] = g
                break
    groups_list = [[] for _ in range(m)]
    for i in range(N):
        g = assignment_list[i]
        groups_list[g].append(i)
    for g in range(m):
        groups_list[g].sort()
    return assignment_list, groups_list

def solution_to_output_format(assignment_list, groups_list, objective_value, N, m):
    assignment_dict = {}
    groups_dict = {}
    for i in range(N):
        assignment_dict[str(i)] = int(assignment_list[i])
    for g in range(m):
        groups_dict[str(g)] = sorted(groups_list[g])
    return {
        "objective_value": float(objective_value),
        "assignment": assignment_dict,
        "groups": groups_dict
    }

def run_heuristic(get_distance, total_d, group_capacities, time_limit, update_best_func=None):
    N = len(total_d)
    m = len(group_capacities)
    start_time = time.time()
    assignment, groups = greedy_construction(N, m, group_capacities, get_distance, total_d)
    obj = compute_objective(groups, get_distance)
    if update_best_func:
        update_best_func(obj, (assignment, groups))
    time_used = time.time() - start_time
    time_left = max(0, time_limit - time_used)
    if time_left > 0:
        assignment, groups, obj = local_search(assignment, groups, get_distance, group_capacities, time_left, update_best_func)
    return assignment, groups, obj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    logger = None
    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="maximize")
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    N = instance['N']
    m = instance['m']
    group_capacities = instance['group_capacities']
    distances_upper_triangular = instance['distances_upper_triangular']
    
    total_lower = sum(g['L_g'] for g in group_capacities)
    total_upper = sum(g['U_g'] for g in group_capacities)
    if total_lower > N or total_upper < N:
        raise Exception("Instance is infeasible: capacity constraints cannot be satisfied")
    
    total_d = compute_total_d_from_upper_tri(N, distances_upper_triangular)
    row_start = compute_row_starts(N)
    
    def get_distance(i, j):
        if i == j:
            return 0.0
        if i > j:
            i, j = j, i
        idx = row_start[i] + j - i - 1
        return distances_upper_triangular[idx]
    
    global_start_time = time.time()
    best_sol_internal = None
    best_obj_value = -10**15
    
    def update_best_solution(new_obj, new_sol_internal):
        nonlocal best_sol_internal, best_obj_value
        if new_obj > best_obj_value:
            best_obj_value = new_obj
            best_sol_internal = new_sol_internal
            if logger:
                logger.log(new_obj)
    
    if N <= 200 and m <= 20:
        try:
            d_matrix = [[0.0]*N for _ in range(N)]
            for i in range(N):
                for j in range(N):
                    if i != j:
                        d_matrix[i][j] = get_distance(i, j)
            model, x, y = build_milp_model(N, m, group_capacities, d_matrix)
            time_left = args.time_limit - (time.time() - global_start_time)
            if time_left > 1:
                def milp_callback(model, where):
                    if where == GRB.Callback.MIPSOL:
                        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                        if logger:
                            logger.log(obj)
                model.setParam('OutputFlag', 0)
                model.setParam('TimeLimit', time_left)
                model.optimize(milp_callback)
                if model.SolCount > 0:
                    assignment_list, groups_list = extract_milp_solution(model, x, N, m)
                    obj_value = model.ObjVal
                    update_best_solution(obj_value, (assignment_list, groups_list))
                else:
                    time_heuristic = time_left
                    assignment_list, groups_list, obj_value = run_heuristic(get_distance, total_d, group_capacities, time_heuristic, update_best_solution)
                    update_best_solution(obj_value, (assignment_list, groups_list))
            else:
                time_heuristic = args.time_limit - (time.time() - global_start_time)
                assignment_list, groups_list, obj_value = run_heuristic(get_distance, total_d, group_capacities, time_heuristic, update_best_solution)
                update_best_solution(obj_value, (assignment_list, groups_list))
        except Exception as e:
            print(f"Using heuristic due to error: {str(e)}")
            time_heuristic = args.time_limit - (time.time() - global_start_time)
            assignment_list, groups_list, obj_value = run_heuristic(get_distance, total_d, group_capacities, time_heuristic, update_best_solution)
            update_best_solution(obj_value, (assignment_list, groups_list))
    else:
        time_heuristic = args.time_limit - (time.time() - global_start_time)
        assignment_list, groups_list, obj_value = run_heuristic(get_distance, total_d, group_capacities, time_heuristic, update_best_solution)
        update_best_solution(obj_value, (assignment_list, groups_list))
    
    if best_sol_internal is None:
        time_heuristic = min(10.0, args.time_limit - (time.time() - global_start_time))
        assignment_list, groups_list, obj_value = run_heuristic(get_distance, total_d, group_capacities, time_heuristic, update_best_solution)
        update_best_solution(obj_value, (assignment_list, groups_list))
    
    assignment_list, groups_list = best_sol_internal
    output_sol = solution_to_output_format(assignment_list, groups_list, best_obj_value, N, m)
    
    with open(args.solution_path, 'w') as f:
        json.dump(output_sol, f)

if __name__ == "__main__":
    main()