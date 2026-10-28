import argparse
import json
import time
import math
import random
import itertools
import sys
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    return parser.parse_args()

def greedy_cliques(vertices, adj, W_target, num_trials=5):
    cliques = []
    v_list = list(vertices)
    for _ in range(num_trials):
        random.shuffle(v_list)
        C = []
        for v in v_list:
            if all(u in adj[v] for u in C):
                C.append(v)
        if len(C) > W_target:
            cliques.append(C)
    return cliques

def get_large_cliques(vertices, adj, W_target):
    best_clique = []
    large_cliques = []
    
    def search(R, P):
        nonlocal best_clique, large_cliques
        
        if not P:
            if len(R) > len(best_clique):
                best_clique = list(R)
            if len(R) > W_target:
                large_cliques.append(list(R))
            return
            
        P_list = list(P)
        # Sort by degree in the subgraph induced by P
        P_list.sort(key=lambda n: len(P.intersection(adj[n])), reverse=True)
        
        colors = {}
        max_color = 0
        for v in P_list:
            used = {colors[u] for u in adj[v] if u in colors}
            c = 1
            while c in used:
                c += 1
            colors[v] = c
            if c > max_color:
                max_color = c
                
        # Color bound pruning
        prune_limit = len(best_clique)
        if len(R) + max_color <= prune_limit:
            return
            
        P_list.sort(key=lambda n: colors[n], reverse=True)
        
        for v in P_list:
            if len(R) + colors[v] <= prune_limit:
                break
            
            P_new = P.intersection(adj[v])
            R.add(v)
            search(R, P_new)
            R.remove(v)
            P.remove(v)

    search(set(), set(vertices))
    return best_clique, large_cliques

def main():
    args = parse_args()
    
    # Check limit & set to avoid maximum recursion depth in extremely dense/large instances
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
        
    num_vertices = instance["num_vertices"]
    k = instance["interdiction_budget_k"]
    vertices = instance["vertices"]
    edges_list = instance["edges"]
    edges = [tuple(e) for e in edges_list]
    
    sys.setrecursionlimit(max(sys.getrecursionlimit(), num_vertices + 1000))
    
    logger = SolutionLogger(args.log_path, sense="minimize") if SolutionLogger and args.log_path else None
    
    nodes = set(vertices)
    orig_adj = {v: set() for v in vertices}
    for (u, v) in edges:
        orig_adj[u].add(v)
        orig_adj[v].add(u)
        
    # Get initial absolute max clique prior to interdiction
    init_max_clique, _ = get_large_cliques(nodes, orig_adj, num_vertices + 1)
    w_init = len(init_max_clique)
    
    global_best_omega = w_init
    best_y_val = []
    
    if logger:
        logger.log(global_best_omega)
        
    # Sequence of feasibility testing using a pure hitting-set lazily constrained SAT/MIP
    W_target = global_best_omega - 1
    lower_bound = max(1, w_init - k)
    
    start_time = time.time()
    
    while W_target >= lower_bound:
        rem_time = args.time_limit - (time.time() - start_time)
        if rem_time <= 0:
            break
            
        m = gp.Model("interdict_clique")
        m.Params.OutputFlag = 0
        m.Params.LazyConstraints = 1
        m.Params.TimeLimit = rem_time
        m.Params.SolutionLimit = 1 # Find ANY validated integer mapping reaching the constraint threshold
        m.Params.Threads = 1 # Single CPU core specification mapping
        
        y = {}
        for e in edges:
            y[e] = m.addVar(vtype=GRB.BINARY, name=f"y_{e[0]}_{e[1]}")
            
        m.addConstr(gp.quicksum(y.values()) <= k)
        m.setObjective(gp.quicksum(y.values()), GRB.MINIMIZE)
        
        class CBState:
            pass
        cb_state = CBState()
        cb_state.global_best_omega = global_best_omega
        cb_state.best_y_val = best_y_val
        cb_state.logger = logger
        cb_state.W_target = W_target
        
        def cb(model, where):
            if where == GRB.Callback.MIPSOL:
                y_sol = model.cbGetSolution(y)
                
                curr_adj = {v: set() for v in nodes}
                for e in edges:
                    if y_sol[e] <= 0.5:
                        curr_adj[e[0]].add(e[1])
                        curr_adj[e[1]].add(e[0])
                        
                c_greedy = greedy_cliques(nodes, curr_adj, cb_state.W_target)
                best_c, c_exact = get_large_cliques(nodes, curr_adj, cb_state.W_target)
                
                c_size = len(best_c)
                
                if c_size < cb_state.global_best_omega:
                    cb_state.global_best_omega = c_size
                    cb_state.best_y_val = [list(e) for e in edges if y_sol[e] > 0.5]
                    if cb_state.logger:
                        cb_state.logger.log(cb_state.global_best_omega)
                        
                if c_size > cb_state.W_target:
                    all_large = c_greedy + c_exact
                    best_c_sorted = tuple(sorted(best_c))
                    
                    seen = set()
                    unique_large = []
                    for c in all_large:
                        s = tuple(sorted(c))
                        if s not in seen and len(s) > cb_state.W_target:
                            seen.add(s)
                            unique_large.append(c)
                            
                    if best_c_sorted not in seen and c_size > cb_state.W_target:
                        unique_large.append(best_c)
                        seen.add(best_c_sorted)
                        
                    w_plus_1 = cb_state.W_target + 1
                    
                    # Generate subset sampling cuts for efficiency
                    for C in unique_large:
                        sz = len(C)
                        total_combinations = math.comb(sz, w_plus_1)
                        if total_combinations <= 200:
                            subsets = list(itertools.combinations(C, w_plus_1))
                        else:
                            subsets_set = set()
                            C_list = list(C)
                            attempts = 0
                            limit = 200
                            while len(subsets_set) < limit and attempts < limit * 5:
                                s = tuple(sorted(random.sample(C_list, w_plus_1)))
                                subsets_set.add(s)
                                attempts += 1
                            subsets = list(subsets_set)
                            
                        # Apply hitting set lazy conditions (removing >= 1 edge from W_target + 1 component)
                        for s in subsets:
                            s_edges = []
                            for i in range(len(s)):
                                for j in range(i+1, len(s)):
                                    u, v = s[i], s[j]
                                    e_t = (u, v) if (u, v) in y else (v, u)
                                    if e_t in y:
                                        s_edges.append(e_t)
                            if s_edges:
                                model.cbLazy(gp.quicksum(y[e] for e in s_edges) >= 1)
                                
        m.optimize(cb)
        
        global_best_omega = cb_state.global_best_omega
        best_y_val = cb_state.best_y_val
        
        if m.status == GRB.INFEASIBLE:
            break
        elif m.status in [GRB.OPTIMAL, GRB.SOLUTION_LIMIT]:
            W_target = global_best_omega - 1
        else:
            break

    # Save output solution aligned with objective schema
    solution = {
        "objective_value": global_best_omega,
        "interdicted_edges": best_y_val
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == "__main__":
    main()