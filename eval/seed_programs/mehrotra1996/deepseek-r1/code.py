import argparse
import json
import heapq
import time
import gurobipy
from solution_logger import SolutionLogger

def dsatur(nodes, adj):
    n = len(nodes)
    color = {}
    neighbor_colors = {}
    saturation = {}
    current_degree = {}
    uncolored_set = set(nodes)
    
    for node in nodes:
        color[node] = -1
        neighbor_colors[node] = set()
        saturation[node] = 0
        current_degree[node] = len(adj[node])
    
    heap = []
    for node in nodes:
        heapq.heappush(heap, (-saturation[node], -current_degree[node], node))
    
    while uncolored_set:
        valid_node = None
        while heap:
            neg_sat, neg_deg, node = heapq.heappop(heap)
            sat = -neg_sat
            deg = -neg_deg
            if node in uncolored_set and saturation[node] == sat and current_degree[node] == deg:
                valid_node = node
                break
        
        if valid_node is None:
            break
        
        uncolored_set.remove(valid_node)
        
        c = 0
        while c in neighbor_colors[valid_node]:
            c += 1
        color[valid_node] = c
        
        for neighbor in adj[valid_node]:
            if neighbor in uncolored_set:
                if c not in neighbor_colors[neighbor]:
                    neighbor_colors[neighbor].add(c)
                    saturation[neighbor] += 1
                    heapq.heappush(heap, (-saturation[neighbor], -current_degree[neighbor], neighbor))
                current_degree[neighbor] -= 1
                heapq.heappush(heap, (-saturation[neighbor], -current_degree[neighbor], neighbor))
    
    colors_used = max(color.values()) + 1 if color else 0
    return color, colors_used

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
    nodes = graph['nodes']
    edges = graph['edges']
    num_nodes = graph['num_nodes']
    num_edges = graph['num_edges']
    
    adj = {node: set() for node in nodes}
    for e in edges:
        u, v = e
        adj[u].add(v)
        adj[v].add(u)
    
    coloring, k0 = dsatur(nodes, adj)
    if logger:
        logger.log(k0)
    
    best_coloring = coloring
    best_k = k0
    
    elapsed = time.time() - start_time
    remaining_time = args.time_limit - elapsed
    
    if num_nodes <= 100 and k0 >= 2 and k0 <= 20 and remaining_time > 1:
        try:
            model = gurobipy.Model()
            model.setParam('OutputFlag', 0)
            model.setParam('Threads', 1)
            model.setParam('TimeLimit', max(0, remaining_time))
            
            k_val = k0 - 1
            x = model.addVars(nodes, range(k_val), vtype=gurobipy.GRB.BINARY, name='x')
            
            for v in nodes:
                model.addConstr(gurobipy.quicksum(x[v, c] for c in range(k_val)) == 1)
            
            for e in edges:
                u, v = e
                for c in range(k_val):
                    model.addConstr(x[u, c] + x[v, c] <= 1)
            
            model.setObjective(0)
            model.optimize()
            
            if model.status == gurobipy.GRB.OPTIMAL or (model.status == gurobipy.GRB.TIME_LIMIT and model.SolCount > 0):
                new_coloring = {}
                for v in nodes:
                    for c in range(k_val):
                        if x[v, c].X > 0.5:
                            new_coloring[v] = c
                best_coloring = new_coloring
                best_k = k_val
                if logger:
                    logger.log(best_k)
        except Exception as e:
            pass
    
    color_classes = [[] for _ in range(best_k)]
    for node, c in best_coloring.items():
        if c < best_k:
            color_classes[c].append(node)
    
    vertex_colors = {str(node): int(c) for node, c in best_coloring.items()}
    
    solution = {
        "objective_value": best_k,
        "vertex_colors": vertex_colors,
        "color_classes": color_classes
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()