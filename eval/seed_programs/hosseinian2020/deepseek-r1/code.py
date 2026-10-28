import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser(description='Maximum Edge Weight Clique Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the input JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to the output JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, help='Optional path to log intermediate solutions (JSONL)')
    args = parser.parse_args()

    start_time = time.time()
    
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    vertices = data['vertices']
    edges = data['edges']
    num_vertices = len(vertices)
    num_edges = len(edges)
    
    adj = {v: set() for v in vertices}
    for edge in edges:
        i = edge['i']
        j = edge['j']
        adj[i].add(j)
        adj[j].add(i)
    
    model = gp.Model()
    model.setParam("Threads", 1)
    
    x = {}
    for v in vertices:
        x[v] = model.addVar(vtype=GRB.BINARY, name=f'x_{v}')
    
    y = {}
    for edge in edges:
        i = edge['i']
        j = edge['j']
        key = (min(i, j), max(i, j))
        y[key] = model.addVar(vtype=GRB.BINARY, name=f'y_{key[0]}_{key[1]}')
    
    obj = gp.quicksum(edge['weight'] * y[(min(edge['i'], edge['j']), max(edge['i'], edge['j']))] for edge in edges)
    model.setObjective(obj, GRB.MAXIMIZE)
    
    for edge in edges:
        i = edge['i']
        j = edge['j']
        key = (min(i, j), max(i, j))
        model.addConstr(y[key] <= x[i], name=f'edge_{i}_{j}_constr1')
        model.addConstr(y[key] <= x[j], name=f'edge_{i}_{j}_constr2')
        model.addConstr(y[key] >= x[i] + x[j] - 1, name=f'edge_{i}_{j}_constr3')
    
    sorted_vertices = sorted(vertices)
    n = len(sorted_vertices)
    for idx_i in range(n):
        i = sorted_vertices[idx_i]
        for idx_j in range(idx_i + 1, n):
            j = sorted_vertices[idx_j]
            if j not in adj[i]:
                model.addConstr(x[i] + x[j] <= 1, name=f'non_edge_{i}_{j}')
    
    build_time = time.time() - start_time
    remaining_time = args.time_limit - build_time
    if remaining_time < 0:
        remaining_time = 0
    model.setParam('TimeLimit', remaining_time)
    model.setParam('OutputFlag', 0)
    
    logger = None
    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="maximize")
        model._logger = logger
    
    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            if hasattr(model, '_logger') and model._logger:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                model._logger.log(obj_val)
    
    model.optimize(mycallback)
    
    if model.SolCount > 0:
        objective_value = model.ObjVal
        clique = [v for v in vertices if x[v].X >= 0.5]
    else:
        objective_value = 0.0
        clique = []
    
    with open(args.solution_path, 'w') as f:
        json.dump({
            "objective_value": objective_value,
            "clique": clique
        }, f)

if __name__ == '__main__':
    main()