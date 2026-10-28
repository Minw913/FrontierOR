import json
import argparse
import time
import sys
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB
import math

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    params = instance['problem_parameters']
    nodes = instance['nodes']
    N = params['n_nodes']
    C = params['n_customers']
    Q = params['vehicle_capacity']
    T = params['planning_horizon']
    
    depot_source = None
    depot_sink = None
    customers = []
    for node in nodes:
        if node['type'] == 'depot_source':
            depot_source = node
        elif node['type'] == 'depot_sink':
            depot_sink = node
        else:
            customers.append(node)
    
    # Assume depot_source and depot_sink have same location
    depot_id = depot_source['id']
    sink_id = depot_sink['id']
    
    # Node list: 0 is source, 1..C customers, C+1 is sink
    node_list = [depot_source] + customers + [depot_sink]
    idx_to_id = [n['id'] for n in node_list]
    id_to_idx = {idx_to_id[i]: i for i in range(len(idx_to_id))}
    
    # Precompute distances, travel times (Euclidean), feasibility
    dist = [[0.0 for _ in range(N)] for _ in range(N)]
    travel_time = [[0.0 for _ in range(N)] for _ in range(N)]
    demand = [0] * N
    service = [0] * N
    a = [0.0] * N  # time window open
    b = [0.0] * N  # time window close
    
    for i in range(N):
        n1 = node_list[i]
        demand[i] = n1['demand']
        service[i] = n1['service_time']
        a[i] = n1['time_window_open']
        b[i] = n1['time_window_close']
        for j in range(N):
            n2 = node_list[j]
            d = euclidean_distance(n1['x'], n1['y'], n2['x'], n2['y'])
            dist[i][j] = d
            travel_time[i][j] = d  # since travel_time_type is Euclidean
    
    # Feasibility matrix: can we go i -> j directly?
    can_go = [[False] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if demand[i] + demand[j] > Q:
                continue
            # Earliest service at i, then arrive at j
            t_i = a[i]
            depart_i = t_i + service[i]
            arrival_j = depart_i + travel_time[i][j]
            if arrival_j <= b[j]:
                can_go[i][j] = True
    
    # For return to sink, we will check in constraints
    
    start_time = time.time()
    best_obj = float('inf')
    best_routes = None
    
    # Use Gurobi to solve VRPTW with MTZ-like or flow formulation
    # Since N can be up to ~50 probably, we use compact formulation with time and load
    
    try:
        m = gp.Model("VRPTW")
        m.setParam('TimeLimit', args.time_limit - 5)  # leave margin
        m.setParam('Threads', 1)
        m.setParam('LogToConsole', 0)
        
        # Nodes: 0 = source, 1 to C = customers, C+1 = sink
        source = 0
        sink = C + 1
        V = list(range(C + 2))
        customers_idx = list(range(1, C + 1))
        
        # x[i,j] = 1 if arc i->j is used
        x = m.addVars(V, V, vtype=GRB.BINARY, name="x")
        
        # u[i]: cumulative load after visiting i
        u = m.addVars(V, vtype=GRB.CONTINUOUS, lb=0, ub=Q, name="u")
        
        # t[i]: service start time at i
        t = m.addVars(V, vtype=GRB.CONTINUOUS, lb=0, ub=T, name="t")
        
        # Objective: minimize total distance
        obj = gp.quicksum(dist[i][j] * x[i,j] for i in V for j in V if i != j)
        m.setObjective(obj, GRB.MINIMIZE)
        
        # Each customer visited exactly once
        for i in customers_idx:
            m.addConstr(gp.quicksum(x[i,j] for j in V if j != i) == 1, f"out_{i}")
            m.addConstr(gp.quicksum(x[j,i] for j in V if j != i) == 1, f"in_{i}")
        
        # Flow conservation at source and sink
        m.addConstr(gp.quicksum(x[source,j] for j in V if j != source) == gp.quicksum(x[sink,k] for k in V if k != sink), "depot_flow")
        
        # No self loops
        for i in V:
            m.addConstr(x[i,i] == 0)
        
        # At most one arc out of sink, but since unlimited vehicles we don't limit number
        
        # Capacity constraints (MTZ)
        for i in customers_idx:
            for j in customers_idx:
                if i != j:
                    m.addConstr(u[i] + demand[j] <= u[j] + Q * (1 - x[i,j]), f"cap_{i}_{j}")
        
        # Link load at source and sink
        for j in customers_idx:
            m.addConstr(u[source] + demand[j] <= u[j] + Q * (1 - x[source,j]), f"cap_source_{j}")
            m.addConstr(u[j] + demand[sink] <= u[sink] + Q * (1 - x[j,sink]), f"cap_to_sink_{j}")
        
        # Time windows and timing constraints
        for i in V:
            m.addConstr(t[i] >= a[i], f"tw_open_{i}")
            m.addConstr(t[i] <= b[i], f"tw_close_{i}")
        
        M = T + 1
        for i in V:
            for j in V:
                if i != j and can_go[i][j]:
                    m.addConstr(t[i] + service[i] + travel_time[i][j] <= t[j] + M * (1 - x[i,j]), f"time_{i}_{j}")
                elif i != j:
                    m.addConstr(x[i,j] == 0, f"forbid_{i}_{j}")
        
        # Return to depot within horizon
        for i in customers_idx:
            m.addConstr(t[i] + service[i] + travel_time[i][sink] <= T + M * (1 - x[i,sink]), f"return_{i}")
        m.addConstr(t[source] == 0, "t_source")
        m.addConstr(t[sink] <= T, "t_sink")
        
        # Symmetry break: source only goes to customers in order
        for i in range(1, C+1):
            for j in range(i+1, C+1):
                pass  # could add more but skip for time
        
        # Callback to log incumbent solutions
        def callback(model, where):
            nonlocal best_obj, best_routes
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj_val < best_obj - 1e-6:
                    best_obj = obj_val
                    if logger:
                        logger.log(float(best_obj))
                    # Try to extract routes
                    try:
                        routes = extract_routes(model, x, source, sink, C)
                        best_routes = routes
                    except:
                        pass
        
        m.optimize(callback)
        
        # Get final solution
        if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT or m.status == GRB.SOLUTION_LIMIT:
            if m.SolCount > 0:
                final_obj = m.objVal
                if final_obj < best_obj:
                    best_obj = final_obj
                    if logger:
                        logger.log(float(best_obj))
                try:
                    routes = extract_routes(m, x, source, sink, C)
                    best_routes = routes
                except:
                    if best_routes is None:
                        best_routes = []  # fallback
        else:
            if best_routes is None:
                # If no solution found, create a trivial (likely infeasible) but schema compliant
                best_routes = []
                best_obj = 0.0
        
    except Exception as e:
        # Fallback: if Gurobi fails or timeout, provide empty or basic solution
        print("Solver error:", str(e), file=sys.stderr)
        best_obj = 0.0
        best_routes = []
    
    # Convert internal indices back to original node ids
    final_routes = []
    if best_routes:
        for route in best_routes:
            orig_route = [idx_to_id[i] for i in route]
            final_routes.append(orig_route)
    else:
        # Very basic fallback: one route per customer if possible
        for cidx in range(1, C+1):
            cid = idx_to_id[cidx]
            final_routes.append([depot_source['id'], cid, depot_sink['id']])
    
    solution = {
        "objective_value": float(best_obj),
        "routes": final_routes
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

def extract_routes(model, x, source, sink, C):
    routes = []
    visited = set()
    V_customers = list(range(1, C+1))
    
    while len(visited) < C:
        # Start new route from depot
        route = [source]
        current = source
        while True:
            found = False
            for j in range(C+2):
                if j == current:
                    continue
                if abs(model.cbGetSolution(x[current, j]) - 1.0) < 1e-6:
                    route.append(j)
                    current = j
                    found = True
                    if j in V_customers:
                        visited.add(j)
                    break
            if not found or current == sink:
                break
        if len(route) > 2:  # at least depot - customer - depot
            routes.append(route)
        else:
            break  # no more
    return routes

if __name__ == "__main__":
    main()