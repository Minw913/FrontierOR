import argparse
import json
import time
import heapq
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    num_nodes = data['num_nodes']
    depot = data['depot']
    num_arcs = data['num_arcs']
    arcs = data['arcs']
    profitable_arcs = data['profitable_arcs']
    VI_nodes = data['VI_nodes']
    strong_incompatibilities = data['strong_incompatibilities']
    weak_incompatibilities = data['weak_incompatibilities']
    
    # Build arc data structures
    # arc_map: (i,j) -> list of costs (there might be parallel arcs)
    # For each (i,j), we want the minimum cost arc for traversal
    arc_dict = {}  # (i,j) -> minimum cost
    arc_list = []  # all arcs as (i, j, cost)
    
    for a in arcs:
        i, j, cost = a[0], a[1], a[2]
        arc_list.append((i, j, cost))
        key = (i, j)
        if key not in arc_dict or cost < arc_dict[key]:
            arc_dict[key] = cost
    
    # Profitable arcs
    prof_arcs = []
    for pa in profitable_arcs:
        i, j, profit = pa[0], pa[1], pa[2]
        cost = arc_dict.get((i, j), 0)
        prof_arcs.append((i, j, profit, cost))
    
    # Compute shortest paths between all pairs using Dijkstra from each node
    # Build adjacency list
    adj = defaultdict(list)
    for i, j, cost in arc_list:
        adj[i].append((j, cost))
    
    all_nodes = set(range(num_nodes))
    
    def dijkstra(source):
        dist = {}
        dist[source] = 0
        pq = [(0, source)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float('inf')):
                continue
            for v, w in adj[u]:
                nd = d + w
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist
    
    # We need shortest paths from depot and to depot, and between profitable arc endpoints
    # Collect relevant nodes
    relevant_nodes = set()
    relevant_nodes.add(depot)
    for i, j, profit, cost in prof_arcs:
        relevant_nodes.add(i)
        relevant_nodes.add(j)
    
    # Compute shortest paths from all relevant nodes
    sp_from = {}
    for node in relevant_nodes:
        sp_from[node] = dijkstra(node)
    
    # Build reverse graph for shortest paths TO relevant nodes
    # Actually, shortest path to a node = shortest path from that node in reverse graph
    # But we already computed from relevant nodes, so sp_from[depot] gives distances from depot
    # sp to depot: we need reverse dijkstra or compute from all nodes to depot
    # For "to depot", compute reverse graph dijkstra from depot
    rev_adj = defaultdict(list)
    for i, j, cost in arc_list:
        rev_adj[j].append((i, cost))
    
    def dijkstra_rev(source):
        dist = {}
        dist[source] = 0
        pq = [(0, source)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float('inf')):
                continue
            for v, w in rev_adj[u]:
                nd = d + w
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist
    
    # sp_to[node] = shortest path cost from any source to node
    # We need sp_to[relevant_node] = shortest from all nodes to that relevant_node
    # = reverse dijkstra from relevant_node
    sp_to = {}
    for node in relevant_nodes:
        sp_to[node] = dijkstra_rev(node)
    
    # Now formulate as MIP
    # Decision variables:
    # x[k] binary: whether profitable arc k is served
    # For flow: we use arc traversal counts
    # This is a min-cost flow / Euler tour problem
    
    # Let's use the full MIP formulation with flow variables
    
    model = gp.Model("hauler")
    model.setParam('TimeLimit', max(1, time_limit - (time.time() - start_time) - 5))
    model.setParam('Threads', 1)
    model.setParam('LazyConstraints', 1)
    
    # Variables for profitable arcs
    x = {}
    for k, (i, j, profit, cost) in enumerate(prof_arcs):
        x[k] = model.addVar(vtype=GRB.BINARY, name=f"x_{k}")
    
    # Variables for arc traversals
    # f[(i,j,idx)] = number of times arc (i,j) with index idx is traversed
    # To keep it manageable, we aggregate by (i,j) pairs and use minimum cost
    # Actually, for each unique arc in arc_list, we need a flow variable
    # But there might be parallel arcs - we should keep the cheapest for deadheading
    # and the actual cost for profitable arcs
    
    # Let's index arcs uniquely
    # For each (i,j) pair, keep track of the minimum cost (for deadheading)
    # Profitable arcs must be traversed on their actual arc
    
    # Flow variables: for each arc in the network, how many times traversed
    # We'll use one variable per unique arc (i,j,cost) tuple
    # But to simplify, use one variable per (i,j) pair with min cost
    # UNLESS a profitable arc has a different cost
    
    # Actually, let's think about this differently.
    # The key insight: we need an Eulerian subgraph (balanced in/out degree at each node)
    # that forms a connected tour from the depot.
    
    # For each arc a = (i,j,cost) in the network, let f[a] be the number of traversals.
    # For profitable arc k = (i,j,profit), x[k]=1 implies at least one traversal of arc (i,j).
    # Multiple arcs might exist between same (i,j) with different costs.
    
    # For efficiency, let's work with unique (i,j) pairs and cheapest cost,
    # plus separate handling for profitable arcs that need specific arcs.
    
    # Actually, since profitable arcs are a subset of arcs, each profitable arc
    # corresponds to traversing some arc (i,j). If there are multiple arcs (i,j)
    # with different costs, the profitable arc traversal uses one of them.
    # The problem says profitable arcs have their own (i,j,profit) and the arc (i,j)
    # has a traveling cost. The cost field in the profitable_arcs is the profit, not cost.
    
    # Let me re-read: arcs have [i,j,cost], profitable_arcs have [i,j,profit].
    # A profitable arc (i,j) must correspond to an arc (i,j) in the network.
    # The traveling cost of traversing that arc is the arc's cost.
    
    # For the flow formulation, let's define:
    # For each arc in the network (indexed by position in arc_list), f[a] >= 0 integer
    # For profitable arc k, it must use at least one traversal of the corresponding arc
    
    # Map profitable arcs to network arcs
    # A profitable arc (i,j) can use any network arc from i to j
    # We should use the cheapest one
    
    # Build: for each (i,j), list of arc indices with their costs
    ij_to_arcs = defaultdict(list)
    for idx, (i, j, cost) in enumerate(arc_list):
        ij_to_arcs[(i, j)].append((idx, cost))
    
    # For each profitable arc, find the cheapest network arc (i,j)
    prof_arc_to_net_arc = {}
    for k, (i, j, profit, _) in enumerate(prof_arcs):
        candidates = ij_to_arcs[(i, j)]
        best_idx, best_cost = min(candidates, key=lambda x: x[1])
        prof_arc_to_net_arc[k] = (best_idx, best_cost)
        prof_arcs[k] = (i, j, profit, best_cost)
    
    # For the MIP, we'll use flow variables per unique arc
    # Group arcs by (i,j) and for each pair, only use the cheapest arc for deadheading
    # But profitable arcs might need specific ones... actually they use the cheapest too
    
    # Simplify: for each (i,j), use ONE flow variable with the minimum cost
    min_cost_arcs = {}  # (i,j) -> min cost
    for i, j, cost in arc_list:
        key = (i, j)
        if key not in min_cost_arcs or cost < min_cost_arcs[key]:
            min_cost_arcs[key] = cost
    
    # Flow variable for each (i,j) pair
    f = {}
    for (i, j), cost in min_cost_arcs.items():
        f[(i, j)] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"f_{i}_{j}")
    
    # VI node activation
    vi_set = set(VI_nodes)
    # For each VI node, which profitable arcs start from it?
    vi_to_prof = defaultdict(list)
    for k, (i, j, profit, cost) in enumerate(prof_arcs):
        if i in vi_set:
            vi_to_prof[i].append(k)
    
    # y[v] binary: whether VI node v is active
    y = {}
    for v in VI_nodes:
        if vi_to_prof[v]:
            y[v] = model.addVar(vtype=GRB.BINARY, name=f"y_{v}")
    
    # Weak incompatibility override variables
    z = {}
    for idx, wi in enumerate(weak_incompatibilities):
        i_node, j_node, penalty = wi[0], wi[1], wi[2]
        z[idx] = model.addVar(vtype=GRB.BINARY, name=f"z_{idx}")
    
    model.update()
    
    # Objective: maximize total profit - total travel cost - total penalty
    obj = gp.LinExpr()
    for k, (i, j, profit, cost) in enumerate(prof_arcs):
        obj += profit * x[k]
    for (i, j), cost in min_cost_arcs.items():
        obj -= cost * f[(i, j)]
    for idx, wi in enumerate(weak_incompatibilities):
        penalty = wi[2]
        obj -= penalty * z[idx]
    
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # Constraint: if profitable arc k is served, flow on (i,j) >= 1
    for k, (i, j, profit, cost) in enumerate(prof_arcs):
        if (i, j) in f:
            model.addConstr(f[(i, j)] >= x[k], name=f"serve_{k}")
    
    # Flow conservation: in-degree = out-degree at each node
    # For each node, sum of f[(i,node)] = sum of f[(node,j)]
    node_in = defaultdict(list)
    node_out = defaultdict(list)
    for (i, j) in min_cost_arcs:
        node_out[i].append((i, j))
        node_in[j].append((i, j))
    
    for node in all_nodes:
        in_flow = gp.quicksum(f[arc] for arc in node_in.get(node, []))
        out_flow = gp.quicksum(f[arc] for arc in node_out.get(node, []))
        model.addConstr(in_flow == out_flow, name=f"flow_{node}")
    
    # VI node activation constraints
    for v in VI_nodes:
        if v in y and vi_to_prof[v]:
            for k in vi_to_prof[v]:
                model.addConstr(x[k] <= y[v], name=f"activate_{v}_{k}")
    
    # Strong incompatibilities: y[i] + y[j] <= 1
    for si in strong_incompatibilities:
        i_node, j_node = si[0], si[1]
        if i_node in y and j_node in y:
            model.addConstr(y[i_node] + y[j_node] <= 1, name=f"strong_{i_node}_{j_node}")
    
    # Weak incompatibilities: y[i] + y[j] <= 1 + z[idx]
    for idx, wi in enumerate(weak_incompatibilities):
        i_node, j_node, penalty = wi[0], wi[1], wi[2]
        if i_node in y and j_node in y:
            model.addConstr(y[i_node] + y[j_node] <= 1 + z[idx], name=f"weak_{idx}")
    
    # Connectivity constraints via lazy cuts
    # For any subset S not containing depot, if a served profitable arc has both endpoints in S,
    # at least one arc must leave S.
    # We'll add these as lazy constraints using a callback.
    
    # For the callback, we need to detect violated subtour elimination constraints
    # This is the standard approach for arc routing problems
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            elapsed = time.time() - start_time
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)
            
            # Get current solution
            x_val = {}
            for k in x:
                x_val[k] = model.cbGetSolution(x[k])
            f_val = {}
            for arc in f:
                f_val[arc] = model.cbGetSolution(f[arc])
            
            # Build flow graph with only traversed arcs
            # Find connected components excluding depot
            served = [k for k in x_val if x_val[k] > 0.5]
            if not served:
                return
            
            # Build adjacency from flow
            flow_graph = defaultdict(list)
            flow_nodes = set()
            for (i, j), val in f_val.items():
                if val > 0.5:
                    flow_graph[i].append(j)
                    flow_nodes.add(i)
                    flow_nodes.add(j)
            
            # Check if all served profitable arc endpoints are reachable from depot
            # BFS from depot in flow graph
            visited = set()
            queue = [depot]
            visited.add(depot)
            while queue:
                node = queue.pop()
                for neighbor in flow_graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            # For each served profitable arc, check if both endpoints are visited
            # If not, we need to add a cut
            # Actually, the constraint says: for any S not containing depot,
            # if a served profitable arc has both endpoints in S, some arc must leave S.
            
            # Find nodes not reachable from depot that have served profitable arcs
            for k in served:
                i, j, profit, cost = prof_arcs[k]
                if i not in visited or j not in visited:
                    # Find the component containing this arc
                    # BFS from i in flow graph restricted to non-depot nodes
                    comp = set()
                    bfs_queue = [i]
                    comp.add(i)
                    while bfs_queue:
                        node = bfs_queue.pop()
                        for neighbor in flow_graph[node]:
                            if neighbor not in comp and neighbor != depot:
                                comp.add(neighbor)
                                bfs_queue.append(neighbor)
                        # Also add reverse edges
                    
                    # Also do BFS on undirected version
                    rev_flow = defaultdict(list)
                    for (a, b), val in f_val.items():
                        if val > 0.5:
                            rev_flow[b].append(a)
                    
                    comp2 = set()
                    bfs_queue = [i]
                    comp2.add(i)
                    while bfs_queue:
                        node = bfs_queue.pop()
                        for neighbor in flow_graph[node]:
                            if neighbor not in comp2 and neighbor != depot:
                                comp2.add(neighbor)
                                bfs_queue.append(neighbor)
                        for neighbor in rev_flow[node]:
                            if neighbor not in comp2 and neighbor != depot:
                                comp2.add(neighbor)
                                bfs_queue.append(neighbor)
                    
                    if depot not in comp2:
                        # Add cut: for set S = comp2, sum of f[(i,j)] for i in S, j not in S >= x[k] for any k with both endpoints in S
                        S = comp2
                        leaving = gp.LinExpr()
                        for (a, b) in f:
                            if a in S and b not in S:
                                leaving += f[(a, b)]
                        
                        # Find all served profitable arcs with both endpoints in S
                        for k2 in range(len(prof_arcs)):
                            i2, j2, _, _ = prof_arcs[k2]
                            if i2 in S and j2 in S:
                                model.cbLazy(leaving >= x[k2])
    
    model.optimize(callback)
    
    # Extract solution
    if model.SolCount > 0:
        obj_val = model.ObjVal
        
        served_arcs = []
        for k, (i, j, profit, cost) in enumerate(prof_arcs):
            if x[k].X > 0.5:
                served_arcs.append({
                    'from': i,
                    'to': j,
                    'profit': profit,
                    'cost': cost
                })
        
        tour_arcs = []
        for (i, j), var in f.items():
            count = int(round(var.X))
            if count > 0:
                cost = min_cost_arcs[(i, j)]
                tour_arcs.append({
                    'from': i,
                    'to': j,
                    'count': count,
                    'cost': cost
                })
        
        solution = {
            'objective_value': obj_val,
            'served_arcs': served_arcs,
            'tour_arcs': tour_arcs
        }
    else:
        # No solution found, return empty tour with 0 objective
        solution = {
            'objective_value': 0.0,
            'served_arcs': [],
            'tour_arcs': []
        }
    
    with open(solution_path, 'w') as f_out:
        json.dump(solution, f_out, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)

if __name__ == '__main__':
    main()