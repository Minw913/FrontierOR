import argparse
import json
import time
import sys
from collections import defaultdict
import heapq

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        inst = json.load(f)
    
    gp = inst['global_parameters']
    lam = gp['lambda']
    net = inst['network']
    modes = inst['modes']
    od_matrix = inst['od_matrix']
    
    freq_ub = gp['frequency_upper_bound_F']
    
    nodes = {n['id']: n for n in net['nodes']}
    edges = {e['id']: e for e in net['edges']}
    
    # Build adjacency for each mode
    # Build directed arc mapping: each undirected edge e -> two directed arcs
    # Arc id: edge_id * 2 for (u->v where u=endpoints[0], v=endpoints[1])
    #         edge_id * 2 + 1 for (v->u)
    
    num_edges = net['num_edges']
    edge_list = net['edges']
    
    # Adjacency for full network (directed)
    adj_dir = defaultdict(list)  # node -> list of (neighbor, arc_id, edge_id, travel_time)
    for e in edge_list:
        eid = e['id']
        u, v = e['endpoints']
        tt = e['traveling_time_seconds']
        arc_uv = eid * 2
        arc_vu = eid * 2 + 1
        adj_dir[u].append((v, arc_uv, eid, tt))
        adj_dir[v].append((u, arc_vu, eid, tt))
    
    # Enumerate candidate lines for each mode
    # A line is a simple path in the mode's subnetwork connecting two distinct terminals
    # with length <= max_line_length_edges
    
    max_line_len = gp.get('max_line_length_edges', None)
    
    candidate_lines = []
    
    def enumerate_lines_for_mode(mode_idx, mode):
        mode_edges_set = set(mode['edge_indices'])
        terminals_set = set(mode['terminals'])
        
        # Build mode adjacency
        mode_adj = defaultdict(list)
        for eidx in mode['edge_indices']:
            e = edges[eidx]
            u, v = e['endpoints']
            mode_adj[u].append((v, eidx))
            mode_adj[v].append((u, eidx))
        
        lines = []
        max_len = max_line_len if max_line_len else 100
        
        # DFS from each terminal
        # To avoid duplicate lines (since undirected), only generate lines where
        # start_terminal <= end_terminal (by id)
        
        for start in mode['terminals']:
            # DFS: stack of (current_node, path_nodes, path_edges, visited_set)
            stack = [(start, [start], [], {start})]
            while stack:
                if time.time() - start_time > time_limit * 0.3:
                    return lines
                cur, path_n, path_e, visited = stack.pop()
                
                # If current is a terminal different from start, record line
                if cur in terminals_set and cur != start and start <= cur:
                    lines.append({
                        'mode_idx': mode_idx,
                        'mode_name': mode['name'],
                        'nodes': list(path_n),
                        'edges': list(path_e),
                        'fixed_cost': mode['fixed_cost_per_line'],
                        'edge_op_cost': sum(edges[eid]['operating_cost'] for eid in path_e),
                        'vehicle_cap': mode['vehicle_capacity']
                    })
                
                if len(path_e) >= max_len:
                    continue
                
                for nb, eidx in mode_adj[cur]:
                    if nb not in visited:
                        stack.append((nb, path_n + [nb], path_e + [eidx], visited | {nb}))
        
        return lines
    
    # Limit enumeration time
    for mi, mode in enumerate(modes):
        lines = enumerate_lines_for_mode(mi, mode)
        candidate_lines.extend(lines)
        if time.time() - start_time > time_limit * 0.35:
            break
    
    # If too many lines, sample/prune
    MAX_LINES = 5000
    if len(candidate_lines) > MAX_LINES:
        # Prioritize shorter lines (lower cost)
        candidate_lines.sort(key=lambda l: len(l['edges']))
        candidate_lines = candidate_lines[:MAX_LINES]
    
    # For each OD pair, find shortest paths (by travel time) in the full directed graph
    # We need to route passengers. Use shortest path routing as a starting heuristic.
    
    # Compute shortest paths from each origin
    od_nodes = set()
    for od in od_matrix:
        od_nodes.add(od['origin'])
        od_nodes.add(od['destination'])
    
    # Dijkstra from each OD node
    def dijkstra(source):
        dist = {}
        prev = {}
        dist[source] = 0
        pq = [(0, source)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float('inf')):
                continue
            for nb, arc_id, eid, tt in adj_dir[u]:
                nd = d + tt
                if nd < dist.get(nb, float('inf')):
                    dist[nb] = nd
                    prev[nb] = (u, arc_id, eid, tt)
                    heapq.heappush(pq, (nd, nb))
        return dist, prev
    
    sp_cache = {}
    for n in od_nodes:
        sp_cache[n] = dijkstra(n)
    
    def get_shortest_path(o, d):
        dist, prev = sp_cache[o]
        if d not in dist:
            return None, None, float('inf')
        path_arcs = []
        path_edges = []
        cur = d
        while cur != o:
            p, arc_id, eid, tt = prev[cur]
            path_arcs.append(arc_id)
            path_edges.append(eid)
            cur = p
        path_arcs.reverse()
        path_edges.reverse()
        return path_arcs, path_edges, dist[d]
    
    # Now formulate with Gurobi
    # Decision variables:
    # y_l: binary, whether line l is activated
    # f_l: continuous, frequency of line l
    # For passenger routing, we use arc-flow formulation per OD pair
    
    # But full arc-flow per OD pair can be huge. Let's try a path-based approach
    # for passengers: pre-compute K shortest paths per OD pair.
    
    # For each OD pair, find up to K shortest directed paths
    K_PATHS = 3
    
    od_paths = {}  # (o,d) -> list of (arcs, edges_set, travel_time)
    
    def k_shortest_paths(o, d, K):
        """Yen's K-shortest paths algorithm"""
        dist_o, prev_o = sp_cache[o]
        if d not in dist_o:
            return []
        
        # Get first shortest path
        arcs, edge_list_path, tt = get_shortest_path(o, d)
        if arcs is None:
            return []
        
        # Reconstruct node sequence
        path_nodes = [o]
        cur = o
        for arc in arcs:
            eid = arc // 2
            e = edges[eid]
            u, v = e['endpoints']
            if arc % 2 == 0:  # u->v
                nxt = v
            else:
                nxt = u
            path_nodes.append(nxt)
            cur = nxt
        
        A = [(tt, path_nodes, arcs)]
        B = []
        
        if K == 1:
            return [(arcs, set(arc // 2 for arc in arcs), tt)]
        
        for k in range(1, K):
            if not A:
                break
            prev_path_nodes = A[-1][1]
            prev_path_arcs = A[-1][2]
            
            for i in range(len(prev_path_nodes) - 1):
                spur_node = prev_path_nodes[i]
                root_path_nodes = prev_path_nodes[:i+1]
                
                # Edges to remove
                removed_arcs = set()
                for (_, pn, _) in A:
                    if pn[:i+1] == root_path_nodes and i < len(pn) - 1:
                        # Find arc from spur_node to pn[i+1]
                        for nb, arc_id, eid, tt2 in adj_dir[spur_node]:
                            if nb == pn[i+1]:
                                removed_arcs.add(arc_id)
                                break
                
                # Removed nodes (nodes in root path except spur node)
                removed_nodes = set(root_path_nodes[:-1])
                
                # Dijkstra from spur_node avoiding removed_nodes and removed_arcs
                dist2 = {spur_node: 0}
                prev2 = {}
                pq = [(0, spur_node)]
                while pq:
                    dd, u = heapq.heappop(pq)
                    if dd > dist2.get(u, float('inf')):
                        continue
                    for nb, arc_id, eid, tt2 in adj_dir[u]:
                        if nb in removed_nodes or arc_id in removed_arcs:
                            continue
                        nd = dd + tt2
                        if nd < dist2.get(nb, float('inf')):
                            dist2[nb] = nd
                            prev2[nb] = (u, arc_id)
                            heapq.heappush(pq, (nd, nb))
                
                if d in dist2:
                    spur_arcs = []
                    spur_nodes = []
                    cur2 = d
                    while cur2 != spur_node:
                        p, aid = prev2[cur2]
                        spur_arcs.append(aid)
                        spur_nodes.append(cur2)
                        cur2 = p
                    spur_arcs.reverse()
                    spur_nodes.reverse()
                    
                    total_nodes = list(root_path_nodes) + spur_nodes
                    # Compute root arcs
                    root_arcs = prev_path_arcs[:i] if i > 0 else []
                    total_arcs = list(root_arcs) + spur_arcs
                    total_tt = sum(edges[a//2]['traveling_time_seconds'] for a in total_arcs)
                    
                    # Check no repeated nodes
                    if len(set(total_nodes)) == len(total_nodes):
                        B.append((total_tt, total_nodes, total_arcs))
            
            if not B:
                break
            
            B.sort(key=lambda x: x[0])
            # Remove duplicates
            added = False
            while B:
                best = B.pop(0)
                if best[1] not in [a[1] for a in A]:
                    A.append(best)
                    added = True
                    break
            if not added:
                break
        
        result = []
        for tt, pn, arcs in A:
            edge_set = set(a // 2 for a in arcs)
            result.append((arcs, edge_set, tt))
        return result
    
    for od in od_matrix:
        o, d = od['origin'], od['destination']
        if time.time() - start_time > time_limit * 0.5:
            # Just use shortest path
            arcs, eset, tt = get_shortest_path(o, d)
            if arcs is not None:
                od_paths[(o,d)] = [(arcs, set(a//2 for a in arcs), tt)]
            else:
                od_paths[(o,d)] = []
        else:
            od_paths[(o,d)] = k_shortest_paths(o, d, K_PATHS)
    
    # Now build Gurobi model
    import gurobipy as gp_solver
    from gurobipy import GRB
    
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time < 5:
        remaining_time = 5
    
    m = gp_solver.Model("transit")
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = max(remaining_time - 10, 5)
    
    num_lines = len(candidate_lines)
    
    # Line variables
    y = {}  # activation binary
    f = {}  # frequency continuous
    
    for l in range(num_lines):
        y[l] = m.addVar(vtype=GRB.BINARY, name=f"y_{l}")
        f[l] = m.addVar(lb=0, ub=freq_ub, vtype=GRB.CONTINUOUS, name=f"f_{l}")
    
    # Passenger path flow variables
    # h[(o,d,k)] = flow on k-th path for OD pair (o,d)
    h = {}
    od_demand = {}
    for od in od_matrix:
        o, d, dem = od['origin'], od['destination'], od['demand']
        od_demand[(o,d)] = dem
        paths = od_paths.get((o,d), [])
        for k in range(len(paths)):
            h[(o,d,k)] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"h_{o}_{d}_{k}")
    
    m.update()
    
    # Constraints:
    
    # 1. Demand satisfaction
    for od in od_matrix:
        o, d, dem = od['origin'], od['destination'], od['demand']
        paths = od_paths.get((o,d), [])
        if paths:
            m.addConstr(
                gp_solver.quicksum(h[(o,d,k)] for k in range(len(paths))) == dem,
                name=f"demand_{o}_{d}"
            )
        # If no paths exist, this is infeasible - shouldn't happen in valid instances
    
    # 2. Link frequency to activation: f_l <= F * y_l
    for l in range(num_lines):
        m.addConstr(f[l] <= freq_ub * y[l], name=f"link_{l}")
    
    # 3. Edge capacity: sum of frequencies of lines using edge e <= edge_capacity
    # Build line->edge mapping
    edge_to_lines = defaultdict(list)  # edge_id -> list of line indices
    for l in range(num_lines):
        for eid in candidate_lines[l]['edges']:
            edge_to_lines[eid].append(l)
    
    for eid, line_list in edge_to_lines.items():
        cap = edges[eid]['edge_capacity']
        m.addConstr(
            gp_solver.quicksum(f[l] for l in line_list) <= cap,
            name=f"edge_cap_{eid}"
        )
    
    # 4. Arc capacity: passenger flow on directed arc <= capacity provided by lines
    # For each directed arc (= edge_id * 2 + direction):
    # Total passenger flow on arc <= sum over lines covering that edge of (vehicle_cap * frequency)
    
    # Build arc -> list of (od, k, demand_flow) for passengers using that arc
    arc_to_paths = defaultdict(list)  # arc_id -> list of (o, d, k)
    for od in od_matrix:
        o, d = od['origin'], od['destination']
        paths = od_paths.get((o,d), [])
        for k, (arcs, eset, tt) in enumerate(paths):
            for arc_id in arcs:
                arc_to_paths[arc_id].append((o, d, k))
    
    # For each arc that has passenger flow:
    for arc_id, path_list in arc_to_paths.items():
        eid = arc_id // 2
        # Lines covering this edge
        lines_on_edge = edge_to_lines.get(eid, [])
        
        if not lines_on_edge and path_list:
            # No lines can provide capacity on this edge - passengers can't use it
            # Force flow to 0 on paths using this arc
            for (o, d, k) in path_list:
                m.addConstr(h[(o,d,k)] == 0, name=f"no_cap_{arc_id}_{o}_{d}_{k}")
        elif lines_on_edge:
            m.addConstr(
                gp_solver.quicksum(h[(o,d,k)] for (o,d,k) in path_list) <=
                gp_solver.quicksum(candidate_lines[l]['vehicle_cap'] * f[l] for l in lines_on_edge),
                name=f"arc_cap_{arc_id}"
            )
    
    # Objective
    # Total line cost = sum_l (fixed_cost * y_l + edge_op_cost_l * f_l)
    total_line_cost = gp_solver.quicksum(
        candidate_lines[l]['fixed_cost'] * y[l] + candidate_lines[l]['edge_op_cost'] * f[l]
        for l in range(num_lines)
    )
    
    # Total passenger travel time = sum over all OD pairs, paths: h[(o,d,k)] * travel_time_k
    total_ptt = gp_solver.LinExpr()
    for od in od_matrix:
        o, d = od['origin'], od['destination']
        paths = od_paths.get((o,d), [])
        for k, (arcs, eset, tt) in enumerate(paths):
            total_ptt += h[(o,d,k)] * tt
    
    obj = lam * total_line_cost + (1 - lam) * total_ptt
    m.setObjective(obj, GRB.MINIMIZE)
    
    # Callback for logging
    best_obj = [float('inf')]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0]:
                best_obj[0] = obj_val
                if logger:
                    logger.log(obj_val)
    
    m.optimize(callback)
    
    # Extract solution
    if m.SolCount > 0:
        active_lines = []
        for l in range(num_lines):
            if y[l].X > 0.5 and f[l].X > 1e-6:
                cl = candidate_lines[l]
                active_lines.append({
                    'line_index': l,
                    'mode': cl['mode_name'],
                    'nodes': cl['nodes'],
                    'edges': cl['edges'],
                    'frequency': f[l].X
                })
        
        active_passenger_paths = []
        for od in od_matrix:
            o, d = od['origin'], od['destination']
            paths = od_paths.get((o,d), [])
            for k, (arcs, eset, tt) in enumerate(paths):
                flow_val = h[(o,d,k)].X
                if flow_val > 1e-6:
                    active_passenger_paths.append({
                        'origin': o,
                        'destination': d,
                        'arcs': arcs,
                        'flow': flow_val
                    })
        
        obj_val = m.ObjVal
        
        solution = {
            'objective_value': obj_val,
            'active_lines': active_lines,
            'active_passenger_paths': active_passenger_paths
        }
        
        if logger:
            logger.log_solution(obj_val, solution)
    else:
        # No feasible solution found - return empty
        solution = {
            'objective_value': float('inf'),
            'active_lines': [],
            'active_passenger_paths': []
        }
        
        # Try a simple heuristic: route all passengers on shortest paths
        # and create lines to cover those paths
        # Actually, we need lines to provide capacity. Let's try a greedy approach.
    
    with open(solution_path, 'w') as fout:
        json.dump(solution, fout, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)