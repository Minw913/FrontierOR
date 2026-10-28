import argparse
import json
import re
import heapq
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def get_components(edges, visited_nodes):
    adj = {i: [] for i in visited_nodes}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    components = []
    unvisited = set(visited_nodes)
    while unvisited:
        start = unvisited.pop()
        comp = [start]
        queue = [start]
        while queue:
            curr = queue.pop(0)
            for neighbor in adj[curr]:
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    comp.append(neighbor)
                    queue.append(neighbor)
        components.append(comp)
    return components

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="maximize")

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n = data['n']
    depot = data['depot']
    t0 = data['t0']
    prizes = {int(k): v for k, v in data['prizes'].items()}

    travel_times = {}
    for k, v in data['travel_times'].items():
        match = re.match(r'\((\d+),\s*(\d+)\)', k)
        if match:
            i, j = int(match.group(1)), int(match.group(2))
            travel_times[(min(i, j), max(i, j))] = v

    # Initial graph shortest paths to prune impossible variables
    dist = {i: float('inf') for i in range(1, n + 1)}
    dist[depot] = 0
    pq = [(0, depot)]
    sp_adj = {i: {} for i in range(1, n + 1)}
    for (u, v), t in travel_times.items():
        sp_adj[u][v] = t
        sp_adj[v][u] = t

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: 
            continue
        for v, weight in sp_adj.get(u, {}).items():
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                heapq.heappush(pq, (dist[v], v))

    valid_edges = []
    for (u, v), t in travel_times.items():
        if dist[u] + t + dist[v] <= t0:
            valid_edges.append((u, v))

    # Iterative degree pruning: any valid tour node requires degree >= 2
    changed = True
    while changed:
        changed = False
        degrees = {i: 0 for i in range(1, n + 1)}
        for u, v in valid_edges:
            degrees[u] += 1
            degrees[v] += 1
        new_valid_edges = []
        for u, v in valid_edges:
            if degrees[u] >= 2 and degrees[v] >= 2:
                new_valid_edges.append((u, v))
            else:
                changed = True
        valid_edges = new_valid_edges

    m = gp.Model()
    m.setParam('TimeLimit', args.time_limit)
    m.setParam('OutputFlag', 0)
    m.setParam('Threads', 1)
    m.setParam('LazyConstraints', 1)

    y = {}
    for i in range(1, n + 1):
        y[i] = m.addVar(vtype=GRB.BINARY, name=f'y_{i}')

    x = {}
    for u, v in valid_edges:
        x[(u, v)] = m.addVar(vtype=GRB.BINARY, name=f'x_{u}_{v}')

    m._y = y
    m._x = x
    m._depot = depot

    obj_expr = gp.LinExpr()
    for i in range(1, n + 1):
        if i != depot:
            p = prizes.get(i, 0)
            if p > 0:
                obj_expr.add(y[i], p)
    m.setObjective(obj_expr, GRB.MAXIMIZE)

    # Topological Constraints
    m.addConstr(y[depot] == 1, name='depot_visited')
    m.addConstr(gp.quicksum(travel_times[(u, v)] * x[(u, v)] for u, v in valid_edges) <= t0, name='time_limit')

    for i in range(1, n + 1):
        incident = [(u, v) for (u, v) in valid_edges if u == i or v == i]
        if not incident:
            m.addConstr(y[i] == 0)
        else:
            m.addConstr(gp.quicksum(x[e] for e in incident) == 2 * y[i], name=f'deg_{i}')

    class CallbackData:
        def __init__(self):
            self.best_obj = -float('inf')

    cb_data = CallbackData()

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            x_vals = model.cbGetSolution(model._x)
            y_vals = model.cbGetSolution(model._y)
            
            edges = [e for e, val in x_vals.items() if val > 0.5]
            visited_nodes = [i for i, val in y_vals.items() if val > 0.5]
            
            components = get_components(edges, visited_nodes)
            subtours_found = False
            
            for comp in components:
                if model._depot not in comp and len(comp) >= 2:
                    subtours_found = True
                    expr = gp.LinExpr()
                    for i in range(len(comp)):
                        for j in range(i+1, len(comp)):
                            u, v = comp[i], comp[j]
                            e = (min(u, v), max(u, v))
                            if e in model._x:
                                expr.add(model._x[e])
                    
                    k = comp[0]
                    rhs = gp.LinExpr()
                    for i in comp:
                        if i != k:
                            rhs.add(model._y[i])
                    model.cbLazy(expr <= rhs)
                    
            if not subtours_found:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj > cb_data.best_obj + 1e-5:
                    cb_data.best_obj = obj
                    if logger:
                        logger.log(obj)
                        
        elif where == GRB.Callback.MIPNODE:
            status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
            if status == GRB.OPTIMAL:
                x_vals = model.cbGetNodeRel(model._x)
                y_vals = model.cbGetNodeRel(model._y)
                
                edges = [e for e, val in x_vals.items() if val > 0.5]
                visited_nodes = [i for i, val in y_vals.items() if val > 0.5]
                
                components = get_components(edges, visited_nodes)
                
                for comp in components:
                    if model._depot not in comp and len(comp) >= 2:
                        comp_edges = []
                        for i in range(len(comp)):
                            for j in range(i+1, len(comp)):
                                u, v = comp[i], comp[j]
                                e = (min(u, v), max(u, v))
                                if e in model._x:
                                    comp_edges.append(e)
                        
                        if not comp_edges:
                            continue
                            
                        val_expr = sum(x_vals[e] for e in comp_edges)
                        k = max(comp, key=lambda i: y_vals[i])
                        val_rhs = sum(y_vals[i] for i in comp if i != k)
                        
                        if val_expr > val_rhs + 1e-4:
                            expr = gp.LinExpr()
                            for e in comp_edges:
                                expr.add(model._x[e])
                            rhs = gp.LinExpr()
                            for i in comp:
                                if i != k:
                                    rhs.add(model._y[i])
                            model.cbLazy(expr <= rhs)

    m.optimize(callback)

    if m.SolCount > 0:
        x_vals = {e: v.X for e, v in x.items() if v.X > 0.5}
        y_vals = {i: v.X for i, v in y.items() if v.X > 0.5}
        obj_val = sum(prizes.get(i, 0) for i in y_vals if i != depot)
        
        visited_nodes = list(y_vals.keys())
        edges_list = [[u, v] for u, v in x_vals.keys()]
        
        adj = {}
        for (u, v) in x_vals.keys():
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
            
        tour = [depot]
        current = depot
        prev = None
        
        while True:
            neighbors = adj.get(current, [])
            if len(neighbors) < 2:
                break
            next_node = neighbors[0] if neighbors[0] != prev else neighbors[1]
            tour.append(next_node)
            if next_node == depot:
                break
            prev = current
            current = next_node
            
        solution = {
            "objective_value": float(obj_val),
            "visited_nodes": visited_nodes,
            "edges": edges_list,
            "tour": tour
        }
    else:
        solution = {
            "objective_value": 0.0,
            "visited_nodes": [],
            "edges": [],
            "tour": []
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    solve()