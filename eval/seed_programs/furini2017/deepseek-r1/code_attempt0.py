import argparse
import json
import time
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
        
    graph = instance['graph']
    n = graph['n']
    m = graph['m']
    edges = graph['edges']
    adj_list = graph['adjacency_list']
    
    if n == 0:
        solution = {
            "objective_value": 0,
            "coloring": {},
            "color_classes": {}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        return
    
    # Run DSATUR algorithm
    color = [-1] * n
    sat = [0] * n
    deg = [len(neighbors) for neighbors in adj_list]
    neighbor_colors = [set() for _ in range(n)]
    uncolored = set(range(n))
    
    while uncolored:
        candidate = -1
        best_sat = -1
        best_deg = -1
        for v in uncolored:
            if sat[v] > best_sat or (sat[v] == best_sat and deg[v] > best_deg):
                candidate = v
                best_sat = sat[v]
                best_deg = deg[v]
        
        c = 0
        while c in neighbor_colors[candidate]:
            c += 1
        color[candidate] = c
        uncolored.remove(candidate)
        
        for neighbor in adj_list[candidate]:
            if color[neighbor] == -1 and c not in neighbor_colors[neighbor]:
                neighbor_colors[neighbor].add(c)
                sat[neighbor] += 1
    
    k0 = max(color) + 1
    dsatur_coloring = {v: color[v] for v in range(n)}
    
    if logger:
        logger.log(k0)
    
    final_k = k0
    final_coloring = dsatur_coloring
    time_elapsed = time.time() - start_time
    time_remaining = args.time_limit - time_elapsed
    
    if time_remaining > 1.0 and k0 > 1:
        if n * k0 <= 100000 and (n + m) * k0 <= 1000000:
            try:
                model = gp.Model()
                model.setParam('OutputFlag', 0)
                model.setParam('TimeLimit', time_remaining)
                model.setParam('MIPFocus', 1)
                
                x = model.addVars(n, k0, vtype=GRB.BINARY, name="x")
                y = model.addVars(k0, vtype=GRB.BINARY, name="y")
                
                model.setObjective(gp.quicksum(y[c] for c in range(k0)), GRB.MINIMIZE)
                
                for v in range(n):
                    model.addConstr(gp.quicksum(x[v, c] for c in range(k0)) == 1)
                
                for u, v in edges:
                    for c in range(k0):
                        model.addConstr(x[u, c] + x[v, c] <= 1)
                
                for v in range(n):
                    for c in range(k0):
                        model.addConstr(x[v, c] <= y[c])
                
                for v in range(n):
                    c0 = dsatur_coloring[v]
                    x[v, c0].Start = 1
                for c in range(k0):
                    y[c].Start = 1
                
                def callback(model, where):
                    if where == GRB.Callback.MIPSOL:
                        objval = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                        if logger:
                            logger.log(int(objval))
                
                model.optimize(callback)
                
                if model.SolCount > 0:
                    k1 = int(model.ObjVal)
                    coloring_mip = {}
                    for v in range(n):
                        for c in range(k0):
                            if x[v, c].X > 0.5:
                                coloring_mip[v] = c
                                break
                    final_k = k1
                    final_coloring = coloring_mip
            except Exception:
                pass
    
    coloring_output = {str(v): int(final_coloring[v]) for v in range(n)}
    color_classes = defaultdict(list)
    for v, c in final_coloring.items():
        color_classes[str(c)].append(v)
    
    solution = {
        "objective_value": int(final_k),
        "coloring": coloring_output,
        "color_classes": dict(color_classes)
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()