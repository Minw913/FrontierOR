import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def get_ub(V, E):
    adj = {v: set() for v in V}
    for u, v in E:
        adj[u].add(v)
        adj[v].add(u)
    order = sorted(V, key=lambda x: len(adj[x]), reverse=True)
    color = {}
    for v in order:
        used = {color[u] for u in adj[v] if u in color}
        c = 0
        while c in used:
            c += 1
        color[v] = c
    return color

def get_max_clique(V, E):
    adj = {v: set() for v in V}
    for u, v in E:
        adj[u].add(v)
        adj[v].add(u)
    best_clique = []
    for v in V:
        clique = {v}
        candidates = set(adj[v])
        for u in candidates:
            if all(u in adj[w] for w in clique):
                clique.add(u)
        if len(clique) > len(best_clique):
            best_clique = list(clique)
    return best_clique

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path) as f:
        data = json.load(f)
        
    V = list(range(data["graph"]["num_vertices"]))
    E = [tuple(e) for e in data["graph"]["edges"]]
    
    heuristic_color = get_ub(V, E)
    UB = max(heuristic_color.values()) + 1 if heuristic_color else 1
    
    color_classes = {}
    for v, c in heuristic_color.items():
        if c not in color_classes:
            color_classes[c] = []
        color_classes[c].append(v)
    sol_dict = {
        "objective_value": UB,
        "coloring": {str(v): c for v, c in heuristic_color.items()},
        "color_classes": {str(c): cls for c, cls in color_classes.items()}
    }
    if logger:
        logger.log_solution(UB, sol_dict)
        
    if UB == 1 or len(E) == 0:
        final_obj = UB
        final_color = heuristic_color
    else:
        clique = get_max_clique(V, E)
        
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        
        m = gp.Model("coloring", env=env)
        m.setParam("Seed", 0)
        m.setParam("MIPGap", 1e-4)
        m.setParam("NumericFocus", 0)
        m.setParam("Threads", 1)
        m.setParam("TimeLimit", args.time_limit)
        
        K = UB
        x = m.addVars(V, range(K), vtype=GRB.BINARY, name="x")
        y = m.addVars(range(K), vtype=GRB.BINARY, name="y")
        
        m.setObjective(gp.quicksum(y), GRB.MINIMIZE)
        
        for v in V:
            m.addConstr(gp.quicksum(x[v, c] for c in range(K)) == 1)
            
        for u, v in E:
            for c in range(K):
                m.addConstr(x[u, c] + x[v, c] <= y[c])
                
        for v in V:
            for c in range(K):
                m.addConstr(x[v, c] <= y[c])
                
        for c in range(K - 1):
            m.addConstr(y[c] >= y[c+1])
            
        for i, v in enumerate(clique):
            if i < K:
                m.addConstr(x[v, i] == 1)
                m.addConstr(y[i] == 1)
                
        for v in V:
            for c in range(K):
                x[v, c].Start = 1 if heuristic_color[v] == c else 0
        used_colors = set(heuristic_color.values())
        for c in range(K):
            y[c].Start = 1 if c in used_colors else 0
            
        current_best = UB
        def callback(model, where):
            nonlocal current_best
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                obj_val = int(round(obj))
                if obj_val < current_best:
                    current_best = obj_val
                    sol_x = model.cbGetSolution(x)
                    color = {}
                    for v in V:
                        for c in range(K):
                            if sol_x[v, c] > 0.5:
                                color[v] = c
                                break
                    if logger:
                        cc = {}
                        for v, c in color.items():
                            if c not in cc:
                                cc[c] = []
                            cc[c].append(v)
                        s_dict = {
                            "objective_value": obj_val,
                            "coloring": {str(v): c for v, c in color.items()},
                            "color_classes": {str(c): cls for c, cls in cc.items()}
                        }
                        logger.log_solution(obj_val, s_dict)
        
        m.optimize(callback)
        
        if m.SolCount > 0:
            obj_val = int(round(m.ObjVal))
            if obj_val < UB:
                color = {}
                for v in V:
                    for c in range(K):
                        if x[v, c].X > 0.5:
                            color[v] = c
                            break
                final_color = color
                final_obj = obj_val
            else:
                final_color = heuristic_color
                final_obj = UB
        else:
            final_color = heuristic_color
            final_obj = UB
            
    final_cc = {}
    for v, c in final_color.items():
        if c not in final_cc:
            final_cc[c] = []
        final_cc[c].append(v)
        
    final_sol = {
        "objective_value": final_obj,
        "coloring": {str(v): c for v, c in final_color.items()},
        "color_classes": {str(c): cls for c, cls in final_cc.items()}
    }
    
    with open(args.solution_path, "w") as f:
        json.dump(final_sol, f, indent=4)

if __name__ == "__main__":
    main()