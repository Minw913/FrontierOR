import argparse
import json
import random
import time
import sys

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = None
    GRB = None

try:
    from solution_logger import SolutionLogger
except ImportError:
    class SolutionLogger:
        def __init__(self, *args, **kwargs): pass
        def log(self, *args, **kwargs): pass

global_best_obj = -float('inf')
logger = None

def log_incumbent(obj):
    global global_best_obj, logger
    if obj > global_best_obj + 1e-6:
        global_best_obj = obj
        if logger:
            logger.log(obj)

def calc_objective(assignment, D_matrix):
    obj = 0.0
    N = len(assignment)
    for i in range(N):
        g_i = assignment[i]
        for j in range(i + 1, N):
            if assignment[j] == g_i:
                obj += D_matrix[i][j]
    return obj

def local_search(assignment, groups, D_matrix, N, m, L, U, start_time, time_limit):
    cost = [[0.0] * m for _ in range(N)]
    for u in range(N):
        for g in range(m):
            cost[u][g] = sum(D_matrix[u][v] for v in groups[g] if v != u)
            
    obj = sum(cost[u][assignment[u]] for u in range(N)) / 2.0
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit - 1.0:
            break
        improved = False
        
        # moves
        for u in range(N):
            gu = assignment[u]
            if len(groups[gu]) <= L[gu]:
                continue
            
            for gv_cand in range(m):
                if gv_cand == gu or len(groups[gv_cand]) >= U[gv_cand]:
                    continue
                
                delta = cost[u][gv_cand] - cost[u][gu]
                if delta > 1e-6:
                    groups[gu].remove(u)
                    groups[gv_cand].append(u)
                    assignment[u] = gv_cand
                    obj += delta
                    # update cost
                    for x in range(N):
                        cost[x][gu] -= D_matrix[x][u]
                        cost[x][gv_cand] += D_matrix[x][u]
                    improved = True
                    break
            if improved:
                break
                
        if improved:
            continue
            
        # swaps
        for u in range(N):
            gu = assignment[u]
            for v in range(u + 1, N):
                gv = assignment[v]
                if gu == gv:
                    continue
                
                delta = cost[u][gv] - cost[u][gu] + cost[v][gu] - cost[v][gv] - 2.0 * D_matrix[u][v]
                if delta > 1e-6:
                    groups[gu].remove(u)
                    groups[gv].append(u)
                    groups[gv].remove(v)
                    groups[gu].append(v)
                    assignment[u] = gv
                    assignment[v] = gu
                    obj += delta
                    for x in range(N):
                        cost[x][gu] += D_matrix[x][v] - D_matrix[x][u]
                        cost[x][gv] += D_matrix[x][u] - D_matrix[x][v]
                    improved = True
                    break
            if improved:
                break
                
    return assignment, groups, obj

def grasp_and_ls(N, m, L, U, D_matrix, time_limit, start_time):
    best_obj = -1.0
    best_assignment = None
    
    iterations = 0
    target_grasp_time = min(10.0, time_limit * 0.2)
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > target_grasp_time and iterations >= 10:
            break
        if elapsed > time_limit - 2.0:
            break
            
        groups = [[] for _ in range(m)]
        order = list(range(N))
        random.shuffle(order)
        
        R = N
        unmet_L = sum(L)
        
        for u in order:
            if R == unmet_L:
                valid_groups = [g for g in range(m) if len(groups[g]) < L[g]]
            else:
                valid_groups = [g for g in range(m) if len(groups[g]) < U[g]]
                
            scores = []
            for g in valid_groups:
                s = sum(D_matrix[u][v] for v in groups[g])
                scores.append((s, g))
                
            scores.sort(key=lambda x: x[0] + 1e-9 * random.random(), reverse=True)
            if random.random() < 0.2:
                chosen_s, chosen_g = random.choice(scores)
            else:
                rcl_size = min(3, len(scores))
                chosen_s, chosen_g = random.choice(scores[:rcl_size])
                
            groups[chosen_g].append(u)
            R -= 1
            if len(groups[chosen_g]) <= L[chosen_g]:
                unmet_L -= 1
                
        assignment = [-1] * N
        for g in range(m):
            for idx in groups[g]:
                assignment[idx] = g
                
        assignment, groups, obj = local_search(assignment, groups, D_matrix, N, m, L, U, start_time, time_limit)
        
        if obj > best_obj:
            best_obj = obj
            best_assignment = list(assignment)
            log_incumbent(best_obj)
            
        iterations += 1
        
    return best_assignment, best_obj

def format_solution(assignment, m, obj_val):
    assign_dict = {}
    groups_dict = {str(g): [] for g in range(m)}
    for i, g in enumerate(assignment):
        assign_dict[str(i)] = g
        groups_dict[str(g)].append(i)
        
    for g in range(m):
        groups_dict[str(g)].sort()
        
    return {
        "objective_value": float(obj_val),
        "assignment": assign_dict,
        "groups": groups_dict
    }

def solve_mdgp():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    start_time = time.time()
    
    global logger
    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="maximize")
        
    with open(args.instance_path, "r") as f:
        instance_data = json.load(f)
        
    N = instance_data["N"]
    m = instance_data["m"]
    caps = instance_data["group_capacities"]
    L = [c["L_g"] for c in caps]
    U = [c["U_g"] for c in caps]
    dist_upper = instance_data["distances_upper_triangular"]
    
    D = [[0.0] * N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            D[i][j] = D[j][i] = dist_upper[idx]
            idx += 1
            
    best_assignment, best_obj = grasp_and_ls(N, m, L, U, D, args.time_limit, start_time)
    
    rem_time = args.time_limit - (time.time() - start_time)
    
    if gp is not None and rem_time > 1.0:
        model = gp.Model("MDGP")
        model.setParam("TimeLimit", max(1.0, rem_time))
        model.setParam("Threads", 1)
        model.setParam("Symmetry", 2)
        
        x = model.addVars(N, m, vtype=GRB.BINARY, name="x")
        
        if best_assignment:
            for i in range(N):
                for g in range(m):
                    x[i, g].start = 1 if best_assignment[i] == g else 0
                    
        for i in range(N):
            model.addConstr(gp.quicksum(x[i, g] for g in range(m)) == 1, name=f"assign_{i}")
            
        for g in range(m):
            model.addConstr(gp.quicksum(x[i, g] for i in range(N)) >= L[g], name=f"L_{g}")
            model.addConstr(gp.quicksum(x[i, g] for i in range(N)) <= U[g], name=f"U_{g}")
            
        obj_expr = gp.QuadExpr()
        for g in range(m):
            vars1 = []
            vars2 = []
            coeffs = []
            for i in range(N):
                for j in range(i + 1, N):
                    val = D[i][j]
                    if val != 0:
                        vars1.append(x[i, g])
                        vars2.append(x[j, g])
                        coeffs.append(val)
            if coeffs:
                obj_expr.addTerms(coeffs, vars1, vars2)
                
        model.setObjective(obj_expr, GRB.MAXIMIZE)
        
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                log_incumbent(obj_val)
                
        model.optimize(callback)
        
        if model.SolCount > 0:
            final_assignment = [-1] * N
            for i in range(N):
                for g in range(m):
                    if x[i, g].X > 0.5:
                        final_assignment[i] = g
                        
            final_obj = calc_objective(final_assignment, D)
            log_incumbent(final_obj)
            
            if final_obj > best_obj:
                best_assignment = final_assignment
                best_obj = final_obj

    sol_dict = format_solution(best_assignment, m, best_obj)
    
    with open(args.solution_path, "w") as f:
        json.dump(sol_dict, f, indent=2)

if __name__ == "__main__":
    solve_mdgp()