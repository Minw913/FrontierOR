import argparse
import json
import time
import sys

try:
    import solution_logger
except ImportError:
    solution_logger = None

import gurobipy as gp
from gurobipy import GRB

def solve_tsp(nodes, dist, depot):
    """Simple heuristic to find a TSP tour (Nearest Neighbor + 2-opt) for MIP start"""
    unvisited = set(nodes)
    curr = depot
    tour = []
    
    # Nearest Neighbor
    while unvisited:
        nxt = min(unvisited, key=lambda x: dist[curr][x])
        tour.append(nxt)
        unvisited.remove(nxt)
        curr = nxt
        
    # 2-Opt Local Search
    improved = True
    while improved:
        improved = False
        for i in range(len(tour)):
            for j in range(i + 1, len(tour)):
                if i == 0 and j == len(tour) - 1:
                    continue
                n_i = depot if i == 0 else tour[i-1]
                n_i1 = tour[i]
                n_j = tour[j]
                n_j1 = depot if j == len(tour) - 1 else tour[j+1]
                
                old = dist[n_i][n_i1] + dist[n_j][n_j1]
                new = dist[n_i][n_j] + dist[n_i1][n_j1]
                if new < old:
                    tour[i:j+1] = reversed(tour[i:j+1])
                    improved = True
    return tour

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    time_limit = max(1.0, args.time_limit - 2.0)
    
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
        
    logger = solution_logger.SolutionLogger(args.log_path, sense="minimize") if args.log_path and solution_logger else None
    
    S_max = data["parameters"]["num_stacks_s"]
    L_max = data["parameters"]["stack_height_l"]
    n = data["parameters"]["num_requests_n"]
    
    reqs = data["requests"]
    p_depot = data["pickup_region"]["depot_index"]
    d_depot = data["delivery_region"]["depot_index"]
    
    dist_p = data["pickup_region"]["distance_matrix"]
    dist_d = data["delivery_region"]["distance_matrix"]
    
    # Map array index (0..n-1) to the designated logical request vertex
    req_p_map = {i: reqs[i]["pickup_vertex"] for i in range(n)}
    req_d_map = {i: reqs[i]["delivery_vertex"] for i in range(n)}
    
    V_p = [p_depot] + [req_p_map[i] for i in range(n)]
    V_d = [d_depot] + [req_d_map[i] for i in range(n)]
    
    model = gp.Model("PDTSPMS")
    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', 1)
    model.setParam('LazyConstraints', 1)
    
    model._logger = logger
    model._V_p = V_p
    model._V_d = V_d
    
    # Routing variables
    x_p = model.addVars([(u, v) for u in V_p for v in V_p if u != v], vtype=GRB.BINARY, name="x_p")
    x_d = model.addVars([(u, v) for u in V_d for v in V_d if u != v], vtype=GRB.BINARY, name="x_d")
    
    # MTZ position variables mapped natively on logical requests 
    pos_p = model.addVars(n, vtype=GRB.CONTINUOUS, lb=1, ub=n, name="pos_p")
    pos_d = model.addVars(n, vtype=GRB.CONTINUOUS, lb=1, ub=n, name="pos_d")
    
    # Precedence variables for LIFO policy tracking
    y = model.addVars([(i, j) for i in range(n) for j in range(n) if i < j], vtype=GRB.BINARY, name="y")
    w = model.addVars([(i, j) for i in range(n) for j in range(n) if i < j], vtype=GRB.BINARY, name="w")
    c = model.addVars([(i, j) for i in range(n) for j in range(n) if i < j], vtype=GRB.CONTINUOUS, lb=0, ub=1, name="c")
    
    # Stack capacity and grouping variables
    z = model.addVars(n, S_max, vtype=GRB.BINARY, name="z")
    
    # Structural degree constraints
    for u in V_p:
        model.addConstr(gp.quicksum(x_p[u, v] for v in V_p if u != v) == 1)
        model.addConstr(gp.quicksum(x_p[v, u] for v in V_p if u != v) == 1)
        
    for u in V_d:
        model.addConstr(gp.quicksum(x_d[u, v] for v in V_d if u != v) == 1)
        model.addConstr(gp.quicksum(x_d[v, u] for v in V_d if u != v) == 1)
        
    # Subtour elimination and exact position binding constraints
    for i in range(n):
        for j in range(n):
            if i == j: continue
            u = req_p_map[i]
            v = req_p_map[j]
            model.addConstr(pos_p[j] >= pos_p[i] + 1 - (n + 1) * (1 - x_p[u, v]))
            
            ud = req_d_map[i]
            vd = req_d_map[j]
            model.addConstr(pos_d[j] >= pos_d[i] + 1 - (n + 1) * (1 - x_d[ud, vd]))
            
    # Relational ordering limits
    for i in range(n):
        for j in range(n):
            if i >= j: continue
            model.addConstr(pos_p[j] - pos_p[i] >= 1 - (n + 1) * (1 - y[i, j]))
            model.addConstr(pos_p[i] - pos_p[j] >= 1 - (n + 1) * y[i, j])
            
            model.addConstr(pos_d[j] - pos_d[i] >= 1 - (n + 1) * (1 - w[i, j]))
            model.addConstr(pos_d[i] - pos_d[j] >= 1 - (n + 1) * w[i, j])
            
            # Conflict arises exactly if their order hasn't reversed (LIFO verification)
            model.addConstr(c[i, j] >= y[i, j] + w[i, j] - 1)
            model.addConstr(c[i, j] >= 1 - y[i, j] - w[i, j])
            
    # Loading stacks configuration mapping capacities
    for i in range(n):
        model.addConstr(gp.quicksum(z[i, s] for s in range(S_max)) == 1)
        
    for s in range(S_max):
        model.addConstr(gp.quicksum(z[i, s] for i in range(n)) <= L_max)
        
    for s in range(S_max - 1):
        model.addConstr(gp.quicksum(z[i, s] for i in range(n)) >= gp.quicksum(z[i, s+1] for i in range(n)))
        
    for i in range(n):
        for j in range(n):
            if i >= j: continue
            for s in range(S_max):
                model.addConstr(z[i, s] + z[j, s] + c[i, j] <= 2)
                
    obj = gp.quicksum(dist_p[u][v] * x_p[u, v] for u, v in x_p.keys()) + \
          gp.quicksum(dist_d[u][v] * x_d[u, v] for u, v in x_d.keys())
    model.setObjective(obj, GRB.MINIMIZE)

    # Initial heuristic to compute viable MIP Starts
    p_nodes = [req_p_map[i] for i in range(n)]
    p_tour = solve_tsp(p_nodes, dist_p, p_depot)
    p_req_map_rev = {v: k for k, v in req_p_map.items()}
    base_p_seq = [p_req_map_rev[v] for v in p_tour]
    
    d_nodes = [req_d_map[i] for i in range(n)]
    d_tour = solve_tsp(d_nodes, dist_d, d_depot)
    d_req_map_rev = {v: k for k, v in req_d_map.items()}
    base_d_seq = [d_req_map_rev[v] for v in d_tour]
    
    starts_def = []
    # Build symmetrically opposed traversals guaranteeing strict LIFO satisfaction for feasibility chunks
    for base_seq in [base_p_seq, base_d_seq]:
        stacks_arr = [base_seq[i:i+L_max] for i in range(0, n, L_max)]
        starts_def.append((base_seq, list(reversed(base_seq)), stacks_arr))
        
        d_seq_m = []
        for chunk in stacks_arr: d_seq_m.extend(reversed(chunk))
        starts_def.append((base_seq, d_seq_m, stacks_arr))
        
        d_seq_rb = []
        for chunk in reversed(stacks_arr): d_seq_rb.extend(reversed(chunk))
        starts_def.append((base_seq, d_seq_rb, stacks_arr))
        
    # Same construction prioritizing optimized symmetric delivery blocks as primary
    for base_seq in [base_d_seq]:
        stacks_arr = [base_seq[i:i+L_max] for i in range(0, n, L_max)]
        starts_def.append((list(reversed(base_seq)), base_seq, stacks_arr))
        
        p_seq_m = []
        for chunk in stacks_arr: p_seq_m.extend(reversed(chunk))
        starts_def.append((p_seq_m, base_seq, stacks_arr))
        
        p_seq_rb = []
        for chunk in reversed(stacks_arr): p_seq_rb.extend(reversed(chunk))
        starts_def.append((p_seq_rb, base_seq, stacks_arr))
        
    def eval_start(p_seq, d_seq):
        cost = 0
        curr = p_depot
        for req in p_seq:
            cost += dist_p[curr][req_p_map[req]]
            curr = req_p_map[req]
        cost += dist_p[curr][p_depot]
        
        curr = d_depot
        for req in d_seq:
            cost += dist_d[curr][req_d_map[req]]
            curr = req_d_map[req]
        cost += dist_d[curr][d_depot]
        return cost

    starts_def.sort(key=lambda x: eval_start(x[0], x[1]))
    
    model.NumStart = min(len(starts_def), 10)
    model.update()
    
    # Configure exact MIP start values to help solver rapidly deduce an upper bound
    for start_idx, (p_seq, d_seq, stacks_in) in enumerate(starts_def[:10]):
        model.params.StartNumber = start_idx
        
        stacks = [list(chunk) for chunk in stacks_in if chunk]
        stacks.sort(key=len, reverse=True)
        while len(stacks) < S_max:
            stacks.append([])
            
        for u, v in x_p.keys(): x_p[u, v].Start = 0
        curr = p_depot
        for req in p_seq:
            nxt = req_p_map[req]
            x_p[curr, nxt].Start = 1
            curr = nxt
        x_p[curr, p_depot].Start = 1
        
        for u, v in x_d.keys(): x_d[u, v].Start = 0
        curr = d_depot
        for req in d_seq:
            nxt = req_d_map[req]
            x_d[curr, nxt].Start = 1
            curr = nxt
        x_d[curr, d_depot].Start = 1
        
        for idx, req in enumerate(p_seq): pos_p[req].Start = idx + 1
        for idx, req in enumerate(d_seq): pos_d[req].Start = idx + 1
            
        for i in range(n):
            for s in range(S_max):
                z[i, s].Start = 0
                
        for s_idx, chunk in enumerate(stacks):
            if s_idx < S_max:
                for req in chunk:
                    z[req, s_idx].Start = 1
                
        for i in range(n):
            for j in range(n):
                if i >= j: continue
                pi = p_seq.index(i)
                pj = p_seq.index(j)
                y_val = 1 if pi < pj else 0
                y[i, j].Start = y_val
                
                di = d_seq.index(i)
                dj = d_seq.index(j)
                w_val = 1 if di < dj else 0
                w[i, j].Start = w_val
                
                c_val = 1 if y_val == w_val else 0
                c[i, j].Start = c_val

    model._x_p = x_p
    model._x_d = x_d
    
    def find_subtours(edges, V):
        adj = {u: [] for u in V}
        for u, v in edges:
            if u in adj: adj[u].append(v)
            
        visited = set()
        subtours = []
        for start_node in V:
            if start_node not in visited:
                curr = start_node
                st = []
                while True:
                    visited.add(curr)
                    st.append(curr)
                    if not adj[curr]: break
                    curr = adj[curr][0]
                    if curr in st:
                        idx = st.index(curr)
                        subtours.append(st[idx:])
                        break
                    if curr in visited: break
        return subtours

    def callback_fn(m, where):
        if where == GRB.Callback.MIPSOL:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            vals_p = m.cbGetSolution(m._x_p)
            edges_p = [(u, v) for (u, v), val in vals_p.items() if val > 0.5]
            st_p = find_subtours(edges_p, m._V_p)
            
            vals_d = m.cbGetSolution(m._x_d)
            edges_d = [(u, v) for (u, v), val in vals_d.items() if val > 0.5]
            st_d = find_subtours(edges_d, m._V_d)
            
            # Verifies exact route validity before passing the logged feasible solution
            is_feasible = True
            for st in st_p:
                if len(st) < len(m._V_p):
                    m.cbLazy(gp.quicksum(m._x_p[i, j] for i in st for j in st if i != j) <= len(st) - 1)
                    is_feasible = False
            for st in st_d:
                if len(st) < len(m._V_d):
                    m.cbLazy(gp.quicksum(m._x_d[i, j] for i in st for j in st if i != j) <= len(st) - 1)
                    is_feasible = False
                    
            if is_feasible:
                if not hasattr(m, '_best_obj') or obj < m._best_obj - 1e-5:
                    m._best_obj = obj
                    if m._logger: m._logger.log(obj)

    model.optimize(callback_fn)
    
    if model.SolCount > 0:
        obj_val = model.ObjVal
        
        p_tour_out = []
        curr = p_depot
        for _ in range(len(V_p)):
            p_tour_out.append(curr)
            curr = next(v for v in V_p if curr != v and x_p[curr, v].X > 0.5)
            
        d_tour_out = []
        curr = d_depot
        for _ in range(len(V_d)):
            d_tour_out.append(curr)
            curr = next(v for v in V_d if curr != v and x_d[curr, v].X > 0.5)
            
        output = {
            "objective_value": float(obj_val),
            "pickup_tour": p_tour_out,
            "delivery_tour": d_tour_out
        }
        
        with open(args.solution_path, 'w') as f:
            json.dump(output, f, indent=4)

if __name__ == "__main__":
    main()