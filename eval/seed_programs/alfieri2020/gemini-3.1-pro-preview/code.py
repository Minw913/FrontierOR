import json
import sys
import time
import random
import argparse
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None


def find_large_clique(n, adj):
    """Greedy heuristic to find a large maximal clique to use as a lower bound."""
    nodes = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    best_clique = []
    for start_node in nodes[:50]:  # Try starting from up to 50 highest degree nodes
        clique = [start_node]
        for v in nodes:
            if v in clique:
                continue
            if all(v in adj[u] for u in clique):
                clique.append(v)
        if len(clique) > len(best_clique):
            best_clique = clique
    return best_clique


def dsatur(n, adj):
    """Degree of Saturation greedy coloring heuristic."""
    color_of = [-1] * n
    uncolored_degree = [len(adj[i]) for i in range(n)]
    neighbor_colors = [set() for _ in range(n)]
    
    uncolored = set(range(n))
    
    while uncolored:
        best_v = -1
        best_sat = -1
        best_deg = -1
        
        for v in uncolored:
            sat = len(neighbor_colors[v])
            deg = uncolored_degree[v]
            if sat > best_sat or (sat == best_sat and deg > best_deg):
                best_sat = sat
                best_deg = deg
                best_v = v
                
        c = 0
        while c in neighbor_colors[best_v]:
            c += 1
            
        color_of[best_v] = c
        uncolored.remove(best_v)
        
        for u in adj[best_v]:
            if color_of[u] == -1:
                neighbor_colors[u].add(c)
                uncolored_degree[u] -= 1
                
    return color_of


def run_tabucol(edges, n, k, initial_coloring, end_time):
    """Tabu Search coloring. Attempts to find a valid coloring using globally strictly k colors."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    first_time = True
    tabu_list = [[0] * k for _ in range(n)]
    
    while time.time() < end_time:
        color_of = list(initial_coloring)
        # Randomize vertices causing overflow to the currently target color `k`
        for v in range(n):
            if color_of[v] == k:
                color_of[v] = random.randint(0, k - 1)
        first_time = False
                
        adj_color_count = [[0] * k for _ in range(n)]
        for v in range(n):
            c = color_of[v]
            for u in adj[v]:
                adj_color_count[u][c] += 1
                
        conflicts = [0] * n
        f = 0
        for v in range(n):
            c = color_of[v]
            confs = adj_color_count[v][c]
            conflicts[v] = confs
            if confs > 0:
                f += 1
        f //= 2
        
        if f == 0:
            return color_of
            
        iter_count = 0
        best_f_ever = f
        max_iters_per_restart = 100000
        
        while iter_count < max_iters_per_restart and time.time() < end_time:
            V_conf = [v for v in range(n) if conflicts[v] > 0]
            if not V_conf:
                return color_of
                
            best_delta = float('inf')
            best_moves = []
            
            sample_size = min(len(V_conf), 30)
            sampled_V = random.sample(V_conf, sample_size) if sample_size < len(V_conf) else V_conf
            
            for v in sampled_V:
                old_c = color_of[v]
                confs_v = conflicts[v]
                for c in range(k):
                    if c == old_c: continue
                    delta = adj_color_count[v][c] - confs_v
                    
                    if tabu_list[v][c] > iter_count and f + delta >= best_f_ever:
                        continue
                        
                    if delta < best_delta:
                        best_delta = delta
                        best_moves = [(v, c)]
                    elif delta == best_delta:
                        best_moves.append((v, c))
                        
            if not best_moves:
                v = random.choice(V_conf)
                c = random.randint(0, k - 1)
                while c == color_of[v]:
                    c = random.randint(0, k - 1)
                best_moves = [(v, c)]
                best_delta = adj_color_count[v][c] - conflicts[v]
                
            move = random.choice(best_moves)
            v, new_c = move
            old_c = color_of[v]
            
            color_of[v] = new_c
            
            tenure = random.randint(0, 9) + int(0.6 * f)
            tabu_list[v][old_c] = iter_count + tenure
            
            f += best_delta
            if f < best_f_ever:
                best_f_ever = f
                
            conflicts[v] = adj_color_count[v][new_c]
            
            for u in adj[v]:
                adj_color_count[u][old_c] -= 1
                if color_of[u] == old_c:
                    conflicts[u] -= 1
                adj_color_count[u][new_c] += 1
                if color_of[u] == new_c:
                    conflicts[u] += 1
                    
            iter_count += 1
            
            if iter_count % 2000 == 0 and time.time() >= end_time:
                break
                
        if f == 0:
            return color_of
            
    return None


def run_gurobi(n, adj, edges, LB, U, best_coloring, clique, time_left, solution_path, logger):
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model("GraphColoring", env=env)
    
    model.setParam('TimeLimit', time_left)
    model.setParam('Threads', 1)
    model.setParam('MIPFocus', 1) 
    
    x = model.addVars(n, U, vtype=GRB.BINARY, name="x")
    y = model.addVars(U, vtype=GRB.BINARY, name="y")
    
    model.setObjective(gp.quicksum(y[c] for c in range(U)), GRB.MINIMIZE)
    
    # Each node gets precisely 1 color
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, c] for c in range(U)) == 1)
        
    is_dense = len(edges) * U > 100000
    
    for u, v in edges:
        for c in range(U):
            if is_dense:
                model.addConstr(x[u, c] + x[v, c] <= 1)
            else:
                model.addConstr(x[u, c] + x[v, c] <= y[c])
                
    for i in range(n):
        for c in range(U):
            model.addConstr(x[i, c] <= y[c])
            
    # Symmetry breaking standard ordering
    for c in range(U - 1):
        model.addConstr(y[c] >= y[c+1])
        
    # Maximal clique symmetry breaking bounds 
    for i in range(min(LB, U)):
        v = clique[i]
        x[v, i].LB = 1.0
        y[i].LB = 1.0
        
    # Translate and deploy best bounds structurally 
    color_map = {}
    for i in range(min(LB, U)):
        c = best_coloring[clique[i]]
        color_map[c] = i
        
    next_color = min(LB, U)
    for i in range(n):
        c = best_coloring[i]
        if c not in color_map:
            color_map[c] = next_color
            next_color += 1
            
        x[i, color_map[c]].Start = 1.0
        
    for c in range(U):
        y[c].Start = 1.0 if c in color_map.values() else 0.0

    global_best_obj = [U]
    
    mapped_best = []
    for i in range(n):
        mapped_best.append(color_map[best_coloring[i]])
    global_best_sol = [mapped_best]
    
    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < global_best_obj[0] - 0.5:
                global_best_obj[0] = int(round(obj))
                if logger:
                    try:
                        logger.log(global_best_obj[0])
                    except Exception:
                        pass
                sol_x = model.cbGetSolution(x)
                new_coloring = [-1]*n
                for i in range(n):
                    for c in range(U):
                        if sol_x[i, c] > 0.5:
                            new_coloring[i] = c
                            break
                global_best_sol[0] = new_coloring
                
    model.optimize(cb)
    write_solution(solution_path, global_best_obj[0], global_best_sol[0])


def write_solution(path, obj, coloring):
    sol = {
        "objective_value": float(obj),
        "coloring": {str(i): int(coloring[i]) for i in range(len(coloring))}
    }
    with open(path, 'w') as f:
        json.dump(sol, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False)
    args = parser.parse_args()
    
    start_time = time.time()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path and SolutionLogger else None
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
        
    n = instance["num_nodes"]
    edges_input = instance["edges"]
    
    adj = [[] for _ in range(n)]
    edges = []
    for u, v in edges_input:
        adj[u].append(v)
        adj[v].append(u)
        edges.append((u, v))
        
    clique = find_large_clique(n, adj)
    LB = len(clique)
    
    coloring = dsatur(n, adj)
    best_obj = len(set(coloring))
    
    # Dense indexing to 0..best_obj-1
    color_set = list(set(coloring))
    c_map = {old: new for new, old in enumerate(color_set)}
    best_coloring = [c_map[c] for c in coloring]
    
    if logger:
        logger.log(best_obj)
        
    if best_obj == LB:
        write_solution(args.solution_path, best_obj, best_coloring)
        return
        
    k = best_obj - 1
    current_for_tabu = best_coloring[:]
    
    limit_ratio = 0.5 if n >= 80 else 0.1
    tabucol_time_limit = start_time + max(5.0, args.time_limit * limit_ratio)
    
    while k >= LB and time.time() < tabucol_time_limit:
        time_left = tabucol_time_limit - time.time()
        limit_for_k = time.time() + min(time_left, max(15.0, time_left * 0.5))
        
        res = run_tabucol(edges, n, k, current_for_tabu, min(limit_for_k, tabucol_time_limit))
        if res is not None:
            best_obj = k
            best_coloring = res[:]
            if logger:
                logger.log(best_obj)
            current_for_tabu = res[:]
            k -= 1
        else:
            break
            
    if best_obj == LB:
        write_solution(args.solution_path, best_obj, best_coloring)
        return
        
    time_left = args.time_limit - (time.time() - start_time)
    if time_left < 2:
        write_solution(args.solution_path, best_obj, best_coloring)
        return
        
    run_gurobi(n, adj, edges, LB, best_obj, best_coloring, clique, time_left, args.solution_path, logger)


if __name__ == "__main__":
    main()