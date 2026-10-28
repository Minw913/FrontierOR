import argparse
import json
import math
import time
import sys
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def solve_heuristic(customers, depot, dist, Q):
    """
    Greedy heuristic to find an initial feasible solution.
    """
    routes = []
    rem_demand = {c['id']: c['demand'] for c in customers}
    
    while sum(rem_demand.values()) > 1e-5:
        T = depot['time_window'][0]
        u = depot['id']
        q_rem = Q
        route = [u]
        deliveries = {}
        
        while q_rem > 1e-5:
            best_v = None
            best_score = float('inf')
            
            for c in customers:
                v = c['id']
                if rem_demand[v] <= 1e-5:
                    continue
                arr = T + dist[u][v]
                if arr <= c['time_window'][1]:
                    start_time = max(arr, c['time_window'][0])
                    # Check if return to depot is valid
                    if start_time + c['service_time'] + dist[v][depot['id']] <= depot['time_window'][1]:
                        score = dist[u][v]
                        if score < best_score:
                            best_score = score
                            best_v = v
            if best_v is None:
                break
                
            v = best_v
            c = next(c for c in customers if c['id'] == v)
            arr = T + dist[u][v]
            start_time = max(arr, c['time_window'][0])
            T = start_time + c['service_time']
            u = v
            
            delivery = min(q_rem, rem_demand[v])
            q_rem -= delivery
            rem_demand[v] -= delivery
            deliveries[v] = deliveries.get(v, 0) + delivery
            route.append(v)
            
        if len(route) == 1:
            # Failed to fulfill all demands with this simple heuristic
            return []
            
        route.append(depot['id'] + len(customers) + 1) # Note: virtual depot will be properly mapped later
        routes.append({'route': route, 'deliveries': deliveries})
        
    return routes

def main():
    start_time_exec = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path and SolutionLogger else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    N = data['num_customers']
    Q = data['vehicle_capacity']
    sched_horizon = data['scheduling_horizon']
    a_cost = data['cost_function']['a']
    b_cost = data['cost_function']['b']
    depot = data['depot']
    customers = data['customers']
    dist = data['distance_matrix']

    depot_id = 0
    C = list(range(1, N + 1))
    virtual_id = N + 1

    # Extend distance matrix to include virtual depot
    full_dist = [[0.0] * (N + 2) for _ in range(N + 2)]
    for i in range(N + 1):
        for j in range(N + 1):
            full_dist[i][j] = dist[i][j]

    for i in range(N + 1):
        full_dist[i][virtual_id] = dist[i][0]
        full_dist[virtual_id][i] = dist[0][i]
    full_dist[virtual_id][virtual_id] = 0.0

    time_windows = {}
    time_windows[0] = depot['time_window']
    time_windows[virtual_id] = [depot['time_window'][0], min(depot['time_window'][1], sched_horizon)]
    for c in customers:
        time_windows[c['id']] = c['time_window']

    service_times = {}
    service_times[0] = 0
    service_times[virtual_id] = 0
    for c in customers:
        service_times[c['id']] = c['service_time']

    customer_demands = {c['id']: c['demand'] for c in customers}

    # Edges definition
    E = []
    for c in C:
        E.append((0, c))
    for c1 in C:
        for c2 in C:
            if c1 != c2:
                E.append((c1, c2))
    for c in C:
        E.append((c, virtual_id))
    E.append((0, virtual_id))

    out_edges = {node: [] for node in range(N + 2)}
    in_edges = {node: [] for node in range(N + 2)}
    for (u, v) in E:
        out_edges[u].append(v)
        in_edges[v].append(u)

    # Initial heuristic
    heuristic_routes = solve_heuristic(customers, depot, dist, Q)
    
    # Estimate required vehicles
    min_k = sum(math.ceil(c['demand'] / Q) for c in customers)
    if heuristic_routes:
        K = max(len(heuristic_routes) + 2, min_k + 2)
    else:
        K = min_k + 3

    # Build MIP Model
    model = gp.Model("SDVRPTW")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    
    time_elapsed = time.time() - start_time_exec
    model.Params.TimeLimit = max(1, args.time_limit - time_elapsed - 2)

    # Variables
    x = {}
    y = {}
    t = {}
    d = {}

    for k in range(K):
        for (u, v) in E:
            x[k, u, v] = model.addVar(vtype=GRB.BINARY, name=f"x_{k}_{u}_{v}")
            y[k, u, v] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=Q, name=f"y_{k}_{u}_{v}")
        
        for node in range(N + 2):
            t[k, node] = model.addVar(vtype=GRB.CONTINUOUS, lb=time_windows[node][0], ub=time_windows[node][1], name=f"t_{k}_{node}")
            
        for node in C:
            d[k, node] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=Q, name=f"d_{k}_{node}")

    # Objective
    obj = gp.quicksum(full_dist[u][v] * (a_cost * y[k, u, v] + b_cost * x[k, u, v]) 
                      for k in range(K) for (u, v) in E)
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints
    # 1. Demand fulfillment
    for u in C:
        model.addConstr(gp.quicksum(d[k, u] for k in range(K)) == customer_demands[u], name=f"demand_{u}")

    for k in range(K):
        # 2. Flow conservation for routing
        model.addConstr(gp.quicksum(x[k, 0, v] for v in out_edges[0]) == 1, name=f"leave_depot_{k}")
        model.addConstr(gp.quicksum(x[k, u, virtual_id] for u in in_edges[virtual_id]) == 1, name=f"enter_end_{k}")
        
        for u in C:
            model.addConstr(gp.quicksum(x[k, v, u] for v in in_edges[u]) == 
                            gp.quicksum(x[k, u, v] for v in out_edges[u]), name=f"flow_{k}_{u}")
            model.addConstr(gp.quicksum(x[k, u, v] for v in out_edges[u]) <= 1, name=f"visit_once_{k}_{u}")

        # 3. Load flow and capacity
        for u in C:
            in_flow = gp.quicksum(y[k, v, u] for v in in_edges[u])
            out_flow = gp.quicksum(y[k, u, v] for v in out_edges[u])
            model.addConstr(in_flow - out_flow == d[k, u], name=f"load_flow_{k}_{u}")
            
            # deliveries only if visited
            visited = gp.quicksum(x[k, u, v] for v in out_edges[u])
            model.addConstr(d[k, u] <= Q * visited, name=f"deliv_visit_{k}_{u}")
            
        for (u, v) in E:
            model.addConstr(y[k, u, v] <= Q * x[k, u, v], name=f"capacity_{k}_{u}_{v}")
            
        out_depot = gp.quicksum(y[k, 0, v] for v in out_edges[0])
        total_d = gp.quicksum(d[k, u] for u in C)
        model.addConstr(out_depot == total_d, name=f"depot_load_{k}")
        
        in_end = gp.quicksum(y[k, u, virtual_id] for u in in_edges[virtual_id])
        model.addConstr(in_end == 0, name=f"end_load_{k}")

        # 4. Time windows (Indicator constraints)
        for (u, v) in E:
            model.addGenConstrIndicator(x[k, u, v], True, 
                                        t[k, u] + service_times[u] + full_dist[u][v] <= t[k, v], 
                                        name=f"tw_{k}_{u}_{v}")

    # Symmetry breaking
    for k in range(K - 1):
        model.addConstr(gp.quicksum(d[k, u] for u in C) >= gp.quicksum(d[k+1, u] for u in C), name=f"sym_{k}")

    # MIP Start from heuristic
    if heuristic_routes:
        # Sort to match symmetry breaking constraints
        routes = sorted(heuristic_routes, key=lambda r: sum(r['deliveries'].values()), reverse=True)
        for k in range(K):
            if k < len(routes):
                r = routes[k]
                seq = r['route']
                deliv = r['deliveries']
                seq[-1] = virtual_id
                
                curr_t = time_windows[0][0]
                times = {}
                for i in range(len(seq)):
                    node = seq[i]
                    if node == 0 or node == virtual_id:
                        if i == 0:
                            times[node] = curr_t
                        else:
                            times[node] = max(curr_t, time_windows[node][0])
                    else:
                        arr = curr_t + full_dist[seq[i-1]][node]
                        times[node] = max(arr, time_windows[node][0])
                        curr_t = times[node] + service_times[node]
                
                loads = [0] * (len(seq) - 1)
                curr_l = 0
                for i in range(len(seq)-2, -1, -1):
                    v = seq[i+1]
                    if v in deliv:
                        curr_l += deliv[v]
                    loads[i] = curr_l
                    
                for (u, v) in E:
                    x[k, u, v].Start = 0
                    y[k, u, v].Start = 0
                    
                for i in range(len(seq)-1):
                    u = seq[i]
                    v = seq[i+1]
                    x[k, u, v].Start = 1
                    y[k, u, v].Start = loads[i]
                    
                for u in C:
                    d[k, u].Start = deliv.get(u, 0)
                    
                for u in times:
                    t[k, u].Start = times[u]
            else:
                for (u, v) in E:
                    if u == 0 and v == virtual_id:
                        x[k, u, v].Start = 1
                    else:
                        x[k, u, v].Start = 0
                    y[k, u, v].Start = 0
                    
                for u in C:
                    d[k, u].Start = 0
                    
                t[k, 0].Start = time_windows[0][0]
                t[k, virtual_id].Start = time_windows[0][0] + full_dist[0][virtual_id]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            
            sol_routes = []
            for k in range(K):
                edges = []
                for (u, v) in E:
                    val = model.cbGetSolution(x[k, u, v])
                    if val > 0.5:
                        edges.append((u, v))
                
                if not edges:
                    continue
                if len(edges) == 1 and edges[0] == (0, virtual_id):
                    continue
                    
                succ = {u: v for (u, v) in edges}
                curr = 0
                route = [0]
                visited = set([0])
                
                while curr in succ:
                    nxt = succ[curr]
                    route.append(nxt)
                    curr = nxt
                    if curr == virtual_id:
                        break
                    if curr in visited:
                        break # cycle safely avoided
                    visited.add(curr)
                        
                if route[-1] != virtual_id:
                    continue
                    
                deliv = {}
                for node in route:
                    if 1 <= node <= N:
                        val = model.cbGetSolution(d[k, node])
                        if val > 1e-4:
                            deliv[str(node)] = float(val)
                            
                out_edges_list = []
                for i in range(len(route)-1):
                    u = route[i]
                    v = route[i+1]
                    load = model.cbGetSolution(y[k, u, v])
                    out_edges_list.append({
                        "from": u,
                        "to": v,
                        "load_weight": float(load)
                    })
                    
                sol_routes.append({
                    "vehicle": k,
                    "route": route,
                    "deliveries": deliv,
                    "edges": out_edges_list
                })
                
            sol_dict = {
                "objective_value": obj_val,
                "routes": sol_routes
            }
            if logger:
                logger.log_solution(obj_val, sol_dict)

    model.optimize(callback)

    if model.SolCount > 0:
        obj_val = model.ObjVal
        final_routes = []
        for k in range(K):
            edges = []
            for (u, v) in E:
                if x[k, u, v].X > 0.5:
                    edges.append((u, v))
                    
            if not edges: continue
            if len(edges) == 1 and edges[0] == (0, virtual_id): continue
            
            succ = {u: v for (u, v) in edges}
            curr = 0
            route = [0]
            visited = set([0])
            while curr in succ:
                nxt = succ[curr]
                route.append(nxt)
                curr = nxt
                if curr == virtual_id:
                    break
                if curr in visited:
                    break
                visited.add(curr)
                    
            deliv = {}
            for node in route:
                if 1 <= node <= N:
                    val = d[k, node].X
                    if val > 1e-4:
                        deliv[str(node)] = float(val)
                        
            out_edges_list = []
            for i in range(len(route)-1):
                u = route[i]
                v = route[i+1]
                load = y[k, u, v].X
                out_edges_list.append({
                    "from": u,
                    "to": v,
                    "load_weight": float(load)
                })
                
            final_routes.append({
                "vehicle": k,
                "route": route,
                "deliveries": deliv,
                "edges": out_edges_list
            })
            
        final_solution = {
            "objective_value": obj_val,
            "routes": final_routes
        }
        
        with open(args.solution_path, 'w') as f:
            json.dump(final_solution, f, indent=2)

if __name__ == '__main__':
    main()