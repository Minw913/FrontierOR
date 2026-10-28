import json
import argparse
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB

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
    
    num_nodes = instance['num_nodes']
    num_edges = instance['num_edges']
    edges = instance['edges']
    
    if num_nodes == 0:
        solution = {"objective_value": 0, "coloring": {}}
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        return

    adj = [[] for _ in range(num_nodes)]
    for edge in edges:
        u, v = edge
        adj[u].append(v)
        adj[v].append(u)
    
    degrees = [len(neighbors) for neighbors in adj]
    order = sorted(range(num_nodes), key=lambda i: -degrees[i])
    
    greedy_colors = [-1] * num_nodes
    used_colors_set = set()
    for node in order:
        neighbor_colors = set()
        for neighbor in adj[node]:
            if greedy_colors[neighbor] != -1:
                neighbor_colors.add(greedy_colors[neighbor])
        color = 0
        while color in neighbor_colors:
            color += 1
        greedy_colors[node] = color
        used_colors_set.add(color)
    
    K = len(used_colors_set)
    
    distinct_colors = sorted(used_colors_set)
    color_map = {orig: idx for idx, orig in enumerate(distinct_colors)}
    compressed_colors = [color_map[c] for c in greedy_colors]
    
    if logger:
        logger.log(K)
    
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', args.time_limit)
    
    x = model.addVars(num_nodes, K, vtype=GRB.BINARY, name='x')
    y = model.addVars(K, vtype=GRB.BINARY, name='y')
    
    model.addConstrs((gp.quicksum(x[i, k] for k in range(K)) == 1 for i in range(num_nodes)), name='node_assign')
    
    for u, v in edges:
        for k in range(K):
            model.addConstr(x[u, k] + x[v, k] <= 1, name=f'edge_{u}_{v}_{k}')
    
    model.addConstrs((x[i, k] <= y[k] for i in range(num_nodes) for k in range(K)), name='color_used')
    
    for k in range(1, K):
        model.addConstr(y[k] <= y[k-1], name=f'sym_{k}')
    
    model.setObjective(gp.quicksum(y[k] for k in range(K)), GRB.MINIMIZE)
    
    for i in range(num_nodes):
        x[i, compressed_colors[i]].Start = 1
    for k in range(K):
        y[k].Start = 1 if k < K else 0
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)
    
    try:
        model.optimize(callback)
    except gp.GurobiError as e:
        print(f"Gurobi error: {e}")
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
            objective = round(model.ObjVal)
            coloring = {}
            for i in range(num_nodes):
                for k in range(K):
                    if x[i, k].X > 0.5:
                        coloring[str(i)] = k
                        break
        else:
            objective = K
            coloring = {str(i): compressed_colors[i] for i in range(num_nodes)}
    else:
        objective = K
        coloring = {str(i): compressed_colors[i] for i in range(num_nodes)}
    
    solution = {
        "objective_value": objective,
        "coloring": coloring
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()