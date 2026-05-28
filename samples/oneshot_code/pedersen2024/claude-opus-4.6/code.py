import argparse
import json
import time
import sys
from collections import defaultdict
import heapq

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    params = instance['parameters']
    nodes_data = instance['nodes']
    edges_data = instance['edges']
    
    quota = params['quota']
    alpha = params['alpha']
    num_nodes = params['total_nodes']
    
    # Build node info
    node_info = {}
    substations = []
    turbines = []
    steiner_nodes = []
    
    for nd in nodes_data:
        nid = nd['id']
        node_info[nid] = nd
        if nd['type'] == 'substation':
            substations.append(nid)
        elif nd['type'] == 'potential_turbine':
            turbines.append(nid)
        else:
            steiner_nodes.append(nid)
    
    all_node_ids = list(node_info.keys())
    
    # Build adjacency
    adj = defaultdict(list)
    edge_map = {}  # (u,v) -> edge data, with u < v
    
    for ed in edges_data:
        u, v = ed['from'], ed['to']
        cost = ed['cost']
        scenic = ed['scenic_impact']
        w = alpha * cost + (1 - alpha) * scenic
        adj[u].append((v, w, ed))
        adj[v].append((u, w, ed))
        key = (min(u, v), max(u, v))
        if key not in edge_map or w < alpha * edge_map[key]['cost'] + (1 - alpha) * edge_map[key]['scenic_impact']:
            edge_map[key] = ed
    
    # Node weighted costs
    node_weight = {}
    for nid, nd in node_info.items():
        node_weight[nid] = alpha * nd['cost'] + (1 - alpha) * nd['scenic_impact']
    
    # Try MIP formulation for small-medium instances, heuristic for large
    # Given complexity, let's use Gurobi MIP
    
    try:
        import gurobipy as gp
        from gurobipy import GRB
        use_gurobi = True
    except:
        use_gurobi = False
    
    if use_gurobi:
        remaining_time = args.time_limit - (time.time() - start_time)
        if remaining_time < 1:
            remaining_time = 1
        
        # MIP Formulation:
        # Variables:
        #   x_e: binary, 1 if edge e is selected
        #   y_v: binary, 1 if node v is in the tree
        #   f_e_uv, f_e_vu: continuous flow on edge e in both directions
        #
        # The tree must contain all substations and be connected.
        # We pick one substation as root and use flow to ensure connectivity.
        # Each non-root node in the tree must receive exactly 1 unit of flow.
        
        model = gp.Model("wind_farm")
        model.setParam('TimeLimit', remaining_time - 2)
        model.setParam('Threads', 1)
        model.setParam('MIPGap', 1e-6)
        
        # Edge variables
        x = {}
        for ed in edges_data:
            eid = ed['id']
            x[eid] = model.addVar(vtype=GRB.BINARY, name=f"x_{eid}")
        
        # Node variables (for turbines and steiner nodes)
        y = {}
        for nid in all_node_ids:
            nd = node_info[nid]
            if nd['type'] == 'substation':
                y[nid] = model.addVar(vtype=GRB.BINARY, name=f"y_{nid}")
                # Substations must be in tree
                model.addConstr(y[nid] == 1)
            else:
                y[nid] = model.addVar(vtype=GRB.BINARY, name=f"y_{nid}")
        
        # Flow variables - directed flow on each edge
        # For each edge, we have flow in both directions
        f_pos = {}  # flow from 'from' to 'to'
        f_neg = {}  # flow from 'to' to 'from'
        N = num_nodes
        
        for ed in edges_data:
            eid = ed['id']
            f_pos[eid] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=N, name=f"fp_{eid}")
            f_neg[eid] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=N, name=f"fn_{eid}")
        
        # Build incidence structure
        node_edges = defaultdict(list)  # node -> list of (edge_id, other_node, is_from)
        for ed in edges_data:
            eid = ed['id']
            u, v = ed['from'], ed['to']
            node_edges[u].append((eid, v, True))   # u is 'from'
            node_edges[v].append((eid, u, False))   # v is 'to'
        
        # Root: pick first substation
        root = substations[0]
        
        # Tree constraint: number of edges = number of nodes in tree - 1
        model.addConstr(gp.quicksum(x[ed['id']] for ed in edges_data) == gp.quicksum(y[nid] for nid in all_node_ids) - 1)
        
        # Node in tree iff it has at least one edge (except isolated root case, but root is always in)
        # Actually, if a node is in tree, it must have at least one incident edge (unless it's the only node)
        # And if an edge is selected, both endpoints must be in tree
        for ed in edges_data:
            eid = ed['id']
            u, v = ed['from'], ed['to']
            model.addConstr(x[eid] <= y[u])
            model.addConstr(x[eid] <= y[v])
        
        # For each non-root node in the tree, it must have degree >= 1
        for nid in all_node_ids:
            if nid == root:
                continue
            incident_edges = [eid for (eid, other, is_from) in node_edges[nid]]
            if incident_edges:
                model.addConstr(y[nid] <= gp.quicksum(x[eid] for eid in incident_edges))
        
        # Flow conservation:
        # Root supplies flow to all other tree nodes
        # For root: outflow - inflow = sum of y[v] for all v != root that are in tree
        # Actually easier: each non-root tree node consumes 1 unit
        # Root: outflow - inflow = sum(y[v] for v != root) = total_tree_nodes - 1
        
        # For non-root node v in tree: inflow - outflow = 1 if y[v]=1, else 0
        # Flow on edge only if edge is selected
        
        for ed in edges_data:
            eid = ed['id']
            model.addConstr(f_pos[eid] <= N * x[eid])
            model.addConstr(f_neg[eid] <= N * x[eid])
        
        # Flow conservation for each node
        for nid in all_node_ids:
            inflow = gp.LinExpr()
            outflow = gp.LinExpr()
            for (eid, other, is_from) in node_edges[nid]:
                if is_from:
                    # nid is 'from', so f_pos goes from nid to other (outflow)
                    # f_neg goes from other to nid (inflow)
                    outflow += f_pos[eid]
                    inflow += f_neg[eid]
                else:
                    # nid is 'to', so f_pos goes from other to nid (inflow)
                    # f_neg goes from nid to other (outflow)
                    inflow += f_pos[eid]
                    outflow += f_neg[eid]
            
            if nid == root:
                # outflow - inflow = total selected nodes - 1
                model.addConstr(outflow - inflow == gp.quicksum(y[v] for v in all_node_ids) - 1)
            else:
                # inflow - outflow = y[nid]
                model.addConstr(inflow - outflow == y[nid])
        
        # Energy quota
        model.addConstr(gp.quicksum(node_info[t]['profit'] * y[t] for t in turbines) >= quota)
        
        # Objective
        obj = gp.LinExpr()
        for ed in edges_data:
            eid = ed['id']
            w = alpha * ed['cost'] + (1 - alpha) * ed['scenic_impact']
            obj += w * x[eid]
        for t in turbines:
            obj += node_weight[t] * y[t]
        # Steiner nodes have 0 cost and 0 scenic impact, substations too
        
        model.setObjective(obj, GRB.MINIMIZE)
        
        # Callback for logging
        best_obj = [float('inf')]
        best_sol = [None]
        
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj_val < best_obj[0] - 1e-9:
                    best_obj[0] = obj_val
                    if logger:
                        logger.log(obj_val)
                    # Store solution
                    xvals = model.cbGetSolution([x[ed['id']] for ed in edges_data])
                    yvals = model.cbGetSolution([y[t] for t in turbines])
                    sel_edges = []
                    for i, ed in enumerate(edges_data):
                        if xvals[i] > 0.5:
                            sel_edges.append({"from": ed['from'], "to": ed['to']})
                    sel_turbines = []
                    for i, t in enumerate(turbines):
                        if yvals[i] > 0.5:
                            sel_turbines.append(t)
                    best_sol[0] = (obj_val, sel_turbines, sel_edges)
        
        model.optimize(callback)
        
        if model.SolCount > 0:
            obj_val = model.ObjVal
            sel_turbines = [t for t in turbines if y[t].X > 0.5]
            sel_edges = []
            for ed in edges_data:
                if x[ed['id']].X > 0.5:
                    sel_edges.append({"from": ed['from'], "to": ed['to']})
            
            solution = {
                "objective_value": obj_val,
                "selected_turbines": sel_turbines,
                "selected_edges": sel_edges
            }
        elif best_sol[0] is not None:
            obj_val, sel_turbines, sel_edges = best_sol[0]
            solution = {
                "objective_value": obj_val,
                "selected_turbines": sel_turbines,
                "selected_edges": sel_edges
            }
        else:
            # Fallback: try heuristic
            solution = heuristic_solve(instance, alpha, quota, node_info, substations, turbines, steiner_nodes, adj, node_weight, edge_map, edges_data, logger, args.time_limit - (time.time() - start_time))
    else:
        solution = heuristic_solve(instance, alpha, quota, node_info, substations, turbines, steiner_nodes, adj, node_weight, edge_map, edges_data, logger, args.time_limit - (time.time() - start_time))
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def heuristic_solve(instance, alpha, quota, node_info, substations, turbines, steiner_nodes, adj, node_weight, edge_map, edges_data, logger, time_limit):
    """Greedy heuristic fallback"""
    start = time.time()
    
    # Sort turbines by efficiency: profit / weighted_cost
    turbine_efficiency = []
    for t in turbines:
        nd = node_info[t]
        w = alpha * nd['cost'] + (1 - alpha) * nd['scenic_impact']
        if w > 0:
            eff = nd['profit'] / w
        else:
            eff = float('inf') if nd['profit'] > 0 else 0
        turbine_efficiency.append((eff, t))
    
    turbine_efficiency.sort(reverse=True)
    
    # Greedily add turbines until quota met
    selected_turbines = []
    total_energy = 0
    for eff, t in turbine_efficiency:
        if total_energy >= quota:
            break
        selected_turbines.append(t)
        total_energy += node_info[t]['profit']
    
    if total_energy < quota:
        # Can't meet quota
        selected_turbines = [t for _, t in turbine_efficiency]
    
    # Build minimum spanning tree connecting all substations and selected turbines
    # Using Dijkstra/Prim on the graph
    required_nodes = set(substations) | set(selected_turbines)
    
    # Use Steiner tree heuristic: shortest path between required nodes
    # Simple approach: build MST on complete graph of required nodes with shortest path weights
    
    # First compute shortest paths between all required nodes
    all_nodes_set = set(node_info.keys())
    
    def dijkstra(source):
        dist = {source: 0}
        prev = {source: None}
        pq = [(0, source)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float('inf')):
                continue
            for v, w, ed in adj[u]:
                nd = d + w
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    prev[v] = (u, ed)
                    heapq.heappush(pq, (nd, v))
        return dist, prev
    
    req_list = list(required_nodes)
    # Compute shortest paths from each required node
    sp_dist = {}
    sp_prev = {}
    for r in req_list:
        if time.time() - start > time_limit - 1:
            break
        d, p = dijkstra(r)
        sp_dist[r] = d
        sp_prev[r] = p
    
    # Build complete graph on required nodes
    complete_edges = []
    for i in range(len(req_list)):
        for j in range(i+1, len(req_list)):
            u, v = req_list[i], req_list[j]
            if u in sp_dist and v in sp_dist[u]:
                complete_edges.append((sp_dist[u][v], u, v))
    
    complete_edges.sort()
    
    # Kruskal's MST
    parent = {n: n for n in req_list}
    rank = {n: 0 for n in req_list}
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True
    
    mst_paths = []
    for w, u, v in complete_edges:
        if find(u) != find(v):
            union(u, v)
            mst_paths.append((u, v))
    
    # Reconstruct actual edges from paths
    selected_edge_set = set()
    tree_nodes = set()
    
    for u, v in mst_paths:
        # Trace path from u to v using sp_prev[u]
        prev = sp_prev.get(u, {})
        path_node = v
        while path_node != u and path_node in prev and prev[path_node] is not None:
            parent_node, ed = prev[path_node]
            key = (min(ed['from'], ed['to']), max(ed['from'], ed['to']))
            selected_edge_set.add(key)
            tree_nodes.add(path_node)
            tree_nodes.add(parent_node)
            path_node = parent_node
    
    tree_nodes |= required_nodes
    
    # Compute objective
    total_cost = 0
    sel_edges_out = []
    for key in selected_edge_set:
        if key in edge_map:
            ed = edge_map[key]
            w = alpha * ed['cost'] + (1 - alpha) * ed['scenic_impact']
            total_cost += w
            sel_edges_out.append({"from": ed['from'], "to": ed['to']})
    
    actual_turbines = [t for t in turbines if t in tree_nodes]
    for t in actual_turbines:
        total_cost += node_weight[t]
    
    if logger:
        logger.log(total_cost)
    
    return {
        "objective_value": total_cost,
        "selected_turbines": actual_turbines,
        "selected_edges": sel_edges_out
    }


if __name__ == '__main__':
    solve()