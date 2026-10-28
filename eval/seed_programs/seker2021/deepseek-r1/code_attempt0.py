import argparse
import json
import time
import gurobipy as gp
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    start_time = time.time()
    
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    graph = data['graph']
    partition = data['partition']
    n = graph['num_vertices']
    edges_list = graph['edges']
    clusters = partition['clusters']
    num_clusters = partition['num_clusters']
    
    vertex_to_cluster = {}
    for idx, cluster in enumerate(clusters):
        for v in cluster:
            vertex_to_cluster[v] = idx
            
    selected_vertices_initial = [min(cluster) for cluster in clusters]
    S = set(selected_vertices_initial)
    
    adj = {}
    for v in S:
        adj[v] = []
    for edge in edges_list:
        u, v = edge
        if u in S and v in S:
            adj[u].append(v)
            adj[v].append(u)
    
    if len(S) == 0:
        coloring_initial = {}
        initial_k = 0
    elif len(adj) == 0:
        coloring_initial = {v: 0 for v in S}
        initial_k = 1
    else:
        vertices_sorted = sorted(S, key=lambda v: -len(adj[v]))
        coloring_initial = {}
        for v in vertices_sorted:
            used_colors = set()
            for neighbor in adj.get(v, []):
                if neighbor in coloring_initial:
                    used_colors.add(coloring_initial[neighbor])
            c = 0
            while c in used_colors:
                c += 1
            coloring_initial[v] = c
        initial_k = len(set(coloring_initial.values()))
    
    selected_vertices_dict_initial = {}
    for idx, cluster in enumerate(clusters):
        selected_vertices_dict_initial[str(idx)] = selected_vertices_initial[idx]
    coloring_dict_initial = {str(v): coloring_initial[v] for v in S}
    
    best_solution = {
        "selected_vertices": selected_vertices_dict_initial,
        "coloring": coloring_dict_initial
    }
    best_k = initial_k
    
    if logger:
        logger.log(best_k)
    
    if best_k > 1 and (time.time() - start_time) < args.time_limit:
        k_val = best_k - 1
        while k_val >= 1 and (time.time() - start_time) < args.time_limit:
            if n * k_val > 100000:
                break
                
            model = gp.Model()
            x_vars = {}
            for i in range(n):
                x_vars[i] = model.addVar(vtype=gp.GRB.BINARY, name=f"x_{i}")
            
            y_vars = {}
            for i in range(n):
                for t in range(k_val):
                    y_vars[(i, t)] = model.addVar(vtype=gp.GRB.BINARY, name=f"y_{i}_{t}")
            
            for idx, cluster in enumerate(clusters):
                model.addConstr(gp.quicksum(x_vars[i] for i in cluster) == 1, name=f"cluster_{idx}")
            
            for i in range(n):
                model.addConstr(gp.quicksum(y_vars[(i, t)] for t in range(k_val)) == x_vars[i], name=f"vtx_color_{i}")
            
            for edge in edges_list:
                u, v = edge
                for t in range(k_val):
                    model.addConstr(y_vars[(u, t)] + y_vars[(v, t)] <= 1, name=f"edge_{u}_{v}_{t}")
            
            model.setObjective(0, gp.GRB.MINIMIZE)
            
            remaining_time = args.time_limit - (time.time() - start_time)
            if remaining_time < 1:
                break
            model.setParam('TimeLimit', remaining_time)
            model.setParam('OutputFlag', 0)
            model.setParam('Threads', 1)
            
            model.optimize()
            
            if model.SolCount > 0:
                selected_vertices_list = []
                coloring_dict = {}
                for i in range(n):
                    if x_vars[i].X > 0.5:
                        selected_vertices_list.append(i)
                        for t in range(k_val):
                            if y_vars[(i, t)].X > 0.5:
                                coloring_dict[str(i)] = t
                                break
                
                distinct_colors = set(coloring_dict.values())
                actual_k = len(distinct_colors)
                
                if actual_k < best_k:
                    selected_vertices_dict = {}
                    for i in selected_vertices_list:
                        cluster_idx = vertex_to_cluster[i]
                        selected_vertices_dict[str(cluster_idx)] = i
                    best_solution = {
                        "selected_vertices": selected_vertices_dict,
                        "coloring": coloring_dict
                    }
                    best_k = actual_k
                    if logger:
                        logger.log(best_k)
                
                k_val = best_k - 1
            else:
                break
    
    solution_output = {
        "objective_value": best_k,
        "selected_vertices": best_solution["selected_vertices"],
        "coloring": best_solution["coloring"]
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution_output, f)

if __name__ == "__main__":
    main()