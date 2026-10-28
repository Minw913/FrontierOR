import argparse
import json
import time
import random
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default='')
    args = parser.parse_args()

    if SolutionLogger and args.log_path:
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['problem_size']['n']
    m = instance['problem_size']['m']
    
    flat_values = instance["distance_matrix_upper_triangular"]["values"]
    dist_matrix = [[0] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        row = dist_matrix[i]
        for j in range(i + 1, n):
            val = flat_values[idx]
            row[j] = val
            dist_matrix[j][i] = val
            idx += 1

    cluster_dict = {}
    for cl in instance['clusters']:
        cluster_dict[cl['id']] = cl['node_ids']

    # Edge cases for small graphs
    if m == 1:
        best_node = instance['clusters'][0]['node_ids'][0]
        obj = 0.0
        if logger:
            logger.log(obj)
        with open(args.solution_path, 'w') as f:
            json.dump({
                "objective_value": obj,
                "tour": [best_node],
                "visited_nodes": [best_node]
            }, f, indent=2)
        return

    if m == 2:
        c1_ids = instance['clusters'][0]['node_ids']
        c2_ids = instance['clusters'][1]['node_ids']
        best_c = float('inf')
        pair = None
        for i in c1_ids:
            for j in c2_ids:
                if dist_matrix[i][j] < best_c:
                    best_c = dist_matrix[i][j]
                    pair = (i, j)
        obj = best_c * 2.0
        if logger:
            logger.log(obj)
        with open(args.solution_path, 'w') as f:
            json.dump({
                "objective_value": obj,
                "tour": [pair[0], pair[1]],
                "visited_nodes": [pair[0], pair[1]]
            }, f, indent=2)
        return

    # Heuristics
    h_start = time.time()
    max_time = min(max(2.0, args.time_limit * 0.1), args.time_limit * 0.5)

    cluster_ids = list(cluster_dict.keys())
    best_obj = float('inf')
    best_tour = None
    
    iter_count = 0
    while time.time() - h_start < max_time:
        tour_clusters = list(cluster_ids)
        random.shuffle(tour_clusters)
        tour_nodes = [random.choice(cluster_dict[c]) for c in tour_clusters]
        
        improved = True
        while improved:
            improved = False
            # 2-opt on clusters
            for i in range(m):
                for j in range(i + 2, m):
                    if i == 0 and j == m - 1:
                        continue
                    
                    u1 = tour_nodes[i]
                    u2 = tour_nodes[i + 1]
                    v1 = tour_nodes[j]
                    v2 = tour_nodes[(j + 1) % m]
                    
                    delta = dist_matrix[u1][v1] + dist_matrix[u2][v2] - dist_matrix[u1][u2] - dist_matrix[v1][v2]
                    if delta < 0:
                        tour_nodes[i + 1:j + 1] = tour_nodes[i + 1:j + 1][::-1]
                        tour_clusters[i + 1:j + 1] = tour_clusters[i + 1:j + 1][::-1]
                        improved = True
            
            # Node improvements
            for i in range(m):
                prev = tour_nodes[i - 1]
                nxt = tour_nodes[(i + 1) % m]
                c = tour_clusters[i]
                
                best_n = tour_nodes[i]
                val_best = dist_matrix[prev][best_n] + dist_matrix[best_n][nxt]
                
                for candidate in cluster_dict[c]:
                    if candidate == best_n:
                        continue
                    val_cand = dist_matrix[prev][candidate] + dist_matrix[candidate][nxt]
                    if val_cand < val_best:
                        val_best = val_cand
                        best_n = candidate
                        improved = True
                tour_nodes[i] = best_n
                
        obj = sum(dist_matrix[tour_nodes[i]][tour_nodes[(i + 1) % m]] for i in range(m))
        if obj < best_obj:
            best_obj = obj
            best_tour = list(tour_nodes)
        
        iter_count += 1
        if iter_count >= 1000:
            break
    
    if logger:
        logger.log(best_obj)

    # Prepare exact modeling
    remaining_time = args.time_limit - (time.time() - h_start) - 0.5
    if remaining_time <= 0:
        remaining_time = 1.0
        
    model = gp.Model("GTSP")
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', remaining_time)
    model.setParam('LazyConstraints', 1)
    # To reduce output verbosity in containers
    model.setParam('OutputFlag', 0)
    
    y = [model.addVar(vtype=GRB.BINARY, name=f"y_{i}") for i in range(n)]
    
    node_cluster = [0] * n
    for c_id, n_ids in cluster_dict.items():
        for node_id in n_ids:
            node_cluster[node_id] = c_id
            
    x = {}
    for i in range(n):
        for j in range(i + 1, n):
            if node_cluster[i] != node_cluster[j]:
                x[i, j] = model.addVar(vtype=GRB.BINARY, obj=dist_matrix[i][j], name=f"x_{i}_{j}")
                
    for c_id, n_ids in cluster_dict.items():
        model.addConstr(gp.quicksum(y[i] for i in n_ids) == 1)
        
    for i in range(n):
        incident = []
        for j in range(n):
            if i < j:
                if (i, j) in x: incident.append(x[i, j])
            elif i > j:
                if (j, i) in x: incident.append(x[j, i])
        model.addConstr(gp.quicksum(incident) == 2 * y[i])
        
    model.NumStart = 1
    model.update()
    for i in range(n):
        y[i].Start = 1.0 if i in best_tour else 0.0
        
    tour_edges = set()
    for k in range(m):
        u = best_tour[k]
        v = best_tour[(k + 1) % m]
        if u > v:
            u, v = v, u
        tour_edges.add((u, v))
        
    for (i, j), var in x.items():
        var.Start = 1.0 if (i, j) in tour_edges else 0.0
        
    model._x = x
    model._clusters = cluster_dict
    model._node_cluster = node_cluster
    model._m = m
    model._logger = logger
    model._best_obj = best_obj
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            x_vals = model.cbGetSolution(model._x)
            adj = {}
            for (i, j), val in model._x.items():
                if x_vals[i, j] > 0.5:
                    adj.setdefault(i, []).append(j)
                    adj.setdefault(j, []).append(i)
                    
            visited_nodes = set()
            comps = []
            for start_node in adj.keys():
                if start_node not in visited_nodes:
                    comp = []
                    q = [start_node]
                    while q:
                        curr_node = q.pop()
                        if curr_node not in visited_nodes:
                            visited_nodes.add(curr_node)
                            comp.append(curr_node)
                            for nxt in adj.get(curr_node, []):
                                if nxt not in visited_nodes:
                                    q.append(nxt)
                    comps.append(comp)
                    
            if len(comps) > 1:
                # Discovered disconnected subtours
                for comp in comps:
                    comp_clusters = set(model._node_cluster[ni] for ni in comp)
                    if 1 < len(comp_clusters) < model._m:
                        nodes_in_cyc = []
                        for c_id in comp_clusters:
                            nodes_in_cyc.extend(model._clusters[c_id])
                        
                        expr = gp.LinExpr()
                        n_cyc = len(nodes_in_cyc)
                        for r in range(n_cyc):
                            u = nodes_in_cyc[r]
                            cu = model._node_cluster[u]
                            for idx in range(r + 1, n_cyc):
                                v = nodes_in_cyc[idx]
                                if cu != model._node_cluster[v]:
                                    if u < v:
                                        expr.addTerms(1.0, model._x[u, v])
                                    else:
                                        expr.addTerms(1.0, model._x[v, u])
                                        
                        model.cbLazy(expr <= len(comp_clusters) - 1)
            elif len(comps) == 1:
                comp_clusters = set(model._node_cluster[ni] for ni in comps[0])
                if len(comp_clusters) == model._m:
                    obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if model._logger:
                        if model._best_obj > obj_val + 1e-4:
                            model._best_obj = obj_val
                            model._logger.log(obj_val)
                            
    model.optimize(callback)
    
    if model.SolCount > 0:
        final_obj = model.ObjVal
        x_vals = model.getAttr('X', x)
        y_vals = model.getAttr('X', y)
        
        selected_nodes = [i for i in range(n) if y_vals[i] > 0.5]
        
        adj = {i: [] for i in selected_nodes}
        for (i, j), var in x.items():
            if x_vals[i, j] > 0.5:
                adj[i].append(j)
                adj[j].append(i)
                
        curr = selected_nodes[0]
        tour_nodes = []
        visited = set()
        while curr not in visited:
            visited.add(curr)
            tour_nodes.append(curr)
            nxts = adj[curr]
            nxt = nxts[0] if nxts[0] not in visited else nxts[1] if nxts[1] not in visited else None
            if nxt is None:
                break
            curr = nxt
    else:
        final_obj = best_obj
        tour_nodes = best_tour
        
    with open(args.solution_path, 'w') as f:
        json.dump({
            "objective_value": final_obj,
            "tour": [int(x) for x in tour_nodes],
            "visited_nodes": [int(x) for x in sorted(tour_nodes)]
        }, f, indent=2)

if __name__ == "__main__":
    solve()