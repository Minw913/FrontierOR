import json
import argparse
import time
import sys
import heapq
from gurobipy import *

def dijkstra(adj, source, num_nodes):
    """Compute shortest paths from source to all nodes."""
    dist = [float('inf')] * num_nodes
    dist[source] = 0
    pq = [(0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist

def build_heuristic_solution(data, arc_index, num_nodes, depot, arcs, profitable_arcs, 
                             VI_nodes, strong_inc, weak_inc, arc_cost):
    """Build a heuristic solution using shortest paths and greedy selection."""
    num_prof = len(profitable_arcs)
    VI_set = set(VI_nodes)
    
    # Build adjacency lists
    adj_forward = [[] for _ in range(num_nodes)]
    adj_backward = [[] for _ in range(num_nodes)]
    for i, j, c in arcs:
        adj_forward[i].append((j, c))
        adj_backward[j].append((i, c))
    
    # Shortest paths
    dist_from_depot = dijkstra(adj_forward, depot, num_nodes)
    dist_to_depot = dijkstra(adj_backward, depot, num_nodes)
    
    # Compute attractiveness of each profitable arc
    candidates = []
    for idx, (i, j, p) in enumerate(profitable_arcs):
        if (i, j) not in arc_index:
            continue
        travel_cost = dist_from_depot[i] + arc_cost[(i, j)] + dist_to_depot[j]
        if travel_cost < float('inf'):
            net = p - travel_cost
            candidates.append((net, idx))
    
    candidates.sort(reverse=True)
    
    # Build incompatibility structures
    strong_adj = {v: set() for v in VI_nodes}
    for u, v in strong_inc:
        if u in strong_adj:
            strong_adj[u].add(v)
        if v in strong_adj:
            strong_adj[v].add(u)
    
    # Greedy selection
    active_nodes = set()
    selected = []
    
    for net, idx in candidates:
        if net <= 0:
            break
        i, j, p = profitable_arcs[idx]
        if i in VI_set:
            conflict = False
            for other in active_nodes:
                if other in strong_adj.get(i, set()):
                    conflict = True
                    break
            if conflict:
                continue
        selected.append(idx)
        if i in VI_set:
            active_nodes.add(i)
    
    if not selected:
        return None
    
    # Build tour by tracing shortest paths
    # For each selected arc, we need path depot->i, arc (i,j), path j->depot
    # We'll reconstruct paths using parent pointers
    
    def get_path_forward(source, target, adj, num_nodes):
        """Get shortest path from source to target, return list of arc indices."""
        dist = [float('inf')] * num_nodes
        parent_arc = [-1] * num_nodes
        parent_node = [-1] * num_nodes
        dist[source] = 0
        pq = [(0, source)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == target:
                break
            for v, w in adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    parent_node[v] = u
                    if (u, v) in arc_index:
                        parent_arc[v] = arc_index[(u, v)]
                    heapq.heappush(pq, (nd, v))
        
        if dist[target] == float('inf'):
            return []
        
        path = []
        node = target
        while node != source:
            if parent_arc[node] >= 0:
                path.append(parent_arc[node])
            node = parent_node[node]
            if node < 0:
                return []
        path.reverse()
        return path
    
    # Count traversals
    x_count = [0] * len(arcs)
    
    for pidx in selected:
        i, j, p = profitable_arcs[pidx]
        # Path from depot to i
        path1 = get_path_forward(depot, i, adj_forward, num_nodes)
        for aidx in path1:
            x_count[aidx] += 1
        # The profitable arc itself
        aidx = arc_index[(i, j)]
        x_count[aidx] += 1
        # Path from j to depot
        path2 = get_path_forward(j, depot, adj_forward, num_nodes)
        for aidx in path2:
            x_count[aidx] += 1
    
    return selected, x_count

def solve(data, time_limit, logger, start_time):
    num_nodes = data['num_nodes']
    depot = data['depot']
    arcs = data['arcs']
    profitable_arcs = data['profitable_arcs']
    VI_nodes = data['VI_nodes']
    strong_inc = data['strong_incompatibilities']
    weak_inc = data['weak_incompatibilities']
    
    num_arcs = len(arcs)
    num_prof = len(profitable_arcs)
    num_weak = len(weak_inc)
    
    # Build lookup structures
    arc_cost = {}
    arc_index = {}
    for idx, (i, j, c) in enumerate(arcs):
        arc_cost[(i, j)] = c
        arc_index[(i, j)] = idx
    
    incoming = [[] for _ in range(num_nodes)]
    outgoing = [[] for _ in range(num_nodes)]
    for idx, (i, j, c) in enumerate(arcs):
        outgoing[i].append(idx)
        incoming[j].append(idx)
    
    # Pre-store arc endpoints for fast callback access
    arc_from = [arcs[idx][0] for idx in range(num_arcs)]
    arc_to = [arcs[idx][1] for idx in range(num_arcs)]
    arc_c = [arcs[idx][2] for idx in range(num_arcs)]
    
    prof_from = [profitable_arcs[idx][0] for idx in range(num_prof)]
    prof_to = [profitable_arcs[idx][1] for idx in range(num_prof)]
    prof_p = [profitable_arcs[idx][2] for idx in range(num_prof)]
    
    # Map profitable arcs to network arc indices
    prof_to_arc = []
    for idx in range(num_prof):
        key = (prof_from[idx], prof_to[idx])
        prof_to_arc.append(arc_index.get(key, None))
    
    VI_set = set(VI_nodes)
    
    # Profitable arcs leaving each VI node
    prof_from_node = {v: [] for v in VI_nodes}
    for idx in range(num_prof):
        if prof_from[idx] in VI_set:
            prof_from_node[prof_from[idx]].append(idx)
    
    # Try heuristic for warm start
    heuristic_result = None
    try:
        heuristic_result = build_heuristic_solution(
            data, arc_index, num_nodes, depot, arcs, profitable_arcs,
            VI_nodes, strong_inc, weak_inc, arc_cost
        )
    except:
        pass
    
    # Create model
    model = Model("hauler")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit
    model.Params.OutputFlag = 0
    
    # Variables
    ub_x = num_nodes + num_prof + 2
    x = model.addVars(num_arcs, vtype=GRB.INTEGER, lb=0, ub=ub_x, name="x")
    y = model.addVars(num_prof, vtype=GRB.BINARY, name="y")
    z = model.addVars(num_weak, vtype=GRB.BINARY, name="z")
    a = model.addVars(VI_nodes, vtype=GRB.BINARY, name="a")
    
    model.update()
    
    # Objective: maximize profit - travel cost - penalties
    obj = LinExpr()
    for idx in range(num_prof):
        obj += prof_p[idx] * y[idx]
    for idx in range(num_arcs):
        obj -= arc_c[idx] * x[idx]
    for idx in range(num_weak):
        obj -= weak_inc[idx][2] * z[idx]
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # Flow balance constraints
    for v in range(num_nodes):
        if incoming[v] or outgoing[v]:
            model.addConstr(
                quicksum(x[idx] for idx in incoming[v]) == quicksum(x[idx] for idx in outgoing[v]),
                name=f"flow_{v}"
            )
    
    # Link y to x: if profitable arc served, must traverse it at least once
    for idx in range(num_prof):
        if prof_to_arc[idx] is not None:
            model.addConstr(x[prof_to_arc[idx]] >= y[idx], name=f"link_{idx}")
    
    # Node activation: if any profitable arc from VI node is served, node is active
    for v in VI_nodes:
        for pidx in prof_from_node[v]:
            model.addConstr(a[v] >= y[pidx], name=f"act_{v}_{pidx}")
    
    # Strong incompatibility constraints
    for idx, (u, v) in enumerate(strong_inc):
        if u in VI_set and v in VI_set:
            model.addConstr(a[u] + a[v] <= 1, name=f"strong_{idx}")
    
    # Weak incompatibility constraints
    for idx, (u, v, pen) in enumerate(weak_inc):
        if u in VI_set and v in VI_set:
            model.addConstr(a[u] + a[v] <= 1 + z[idx], name=f"weak_{idx}")
    
    # Set heuristic as MIP start if available
    if heuristic_result is not None:
        try:
            selected, x_count = heuristic_result
            for idx in range(num_prof):
                y[idx].Start = 1.0 if idx in selected else 0.0
            for idx in range(num_arcs):
                x[idx].Start = x_count[idx]
            for idx in range(num_weak):
                z[idx].Start = 0.0
            for v in VI_nodes:
                a[v].Start = 1.0 if v in set(prof_from[i] for i in selected) else 0.0
        except:
            pass
    
    # Callback for connectivity constraints and solution logging
    best_obj = -float('inf')
    
    def callback(model, where):
        nonlocal best_obj
        
        if where == GRB.Callback.MIPSOL:
            x_val = model.cbGetSolution(x)
            y_val = model.cbGetSolution(y)
            
            # Build support graph and find connected components
            adj = [[] for _ in range(num_nodes)]
            for idx in range(num_arcs):
                if x_val[idx] > 0.5:
                    adj[arc_from[idx]].append(arc_to[idx])
                    adj[arc_to[idx]].append(arc_from[idx])
            
            visited = [False] * num_nodes
            cuts_added = 0
            
            for v in range(num_nodes):
                if visited[v]:
                    continue
                # BFS to find component
                comp = set()
                stack = [v]
                visited[v] = True
                while stack:
                    node = stack.pop()
                    comp.add(node)
                    for nb in adj[node]:
                        if not visited[nb]:
                            visited[nb] = True
                            stack.append(nb)
                
                if depot in comp:
                    continue
                
                # Check if any served profitable arc has both endpoints in comp
                violated = False
                for pidx in range(num_prof):
                    if y_val[pidx] > 0.5:
                        if prof_from[pidx] in comp and prof_to[pidx] in comp:
                            violated = True
                            break
                
                if violated:
                    # Add connectivity cut
                    cut = LinExpr()
                    for aidx in range(num_arcs):
                        if arc_from[aidx] in comp and arc_to[aidx] not in comp:
                            cut += x[aidx]
                    model.cbLazy(cut >= 1)
                    cuts_added += 1
            
            # Log improved solutions
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val > best_obj + 1e-6:
                best_obj = obj_val
                if logger:
                    served = []
                    for idx in range(num_prof):
                        if y_val[idx] > 0.5:
                            served.append({
                                'from': prof_from[idx],
                                'to': prof_to[idx],
                                'profit': prof_p[idx],
                                'cost': arc_cost.get((prof_from[idx], prof_to[idx]), 0)
                            })
                    tour = []
                    for idx in range(num_arcs):
                        if x_val[idx] > 0.5:
                            tour.append({
                                'from': arc_from[idx],
                                'to': arc_to[idx],
                                'count': int(round(x_val[idx])),
                                'cost': arc_c[idx]
                            })
                    sol = {
                        'objective_value': obj_val,
                        'served_arcs': served,
                        'tour_arcs': tour
                    }
                    logger.log_solution(obj_val, sol)
    
    model.Params.LazyConstraints = 1
    model.optimize(callback)
    
    # Extract solution
    if model.SolCount > 0:
        x_val = [x[idx].X for idx in range(num_arcs)]
        y_val = [y[idx].X for idx in range(num_prof)]
        
        served = []
        for idx in range(num_prof):
            if y_val[idx] > 0.5:
                served.append({
                    'from': prof_from[idx],
                    'to': prof_to[idx],
                    'profit': prof_p[idx],
                    'cost': arc_cost.get((prof_from[idx], prof_to[idx]), 0)
                })
        
        tour = []
        for idx in range(num_arcs):
            if x_val[idx] > 0.5:
                tour.append({
                    'from': arc_from[idx],
                    'to': arc_to[idx],
                    'count': int(round(x_val[idx])),
                    'cost': arc_c[idx]
                })
        
        obj_val = model.ObjVal
        
        solution = {
            'objective_value': obj_val,
            'served_arcs': served,
            'tour_arcs': tour
        }
        
        if logger and obj_val > best_obj + 1e-6:
            logger.log_solution(obj_val, solution)
        
        return solution
    
    # Fallback: empty solution
    return {
        'objective_value': 0.0,
        'served_arcs': [],
        'tour_arcs': []
    }

def main():
    parser = argparse.ArgumentParser(description="Hauler Arc Routing Problem Solver")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    start_time = time.time()
    
    solution = solve(data, args.time_limit, logger, start_time)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()