import sys
import json
import argparse
import gurobipy as gp

try:
    from solution_logger import SolutionLogger
except ImportError:
    class SolutionLogger:
        def __init__(self, *args, **kwargs): pass
        def log(self, *args, **kwargs): pass

def compute_earliest_arrival_times(truck_r, drone_r, TT, DT, N):
    """
    Computes precisely the earliest arrival times for all nodes matching
    the combined visit synchronization logic.
    """
    arr = {i: 0 for i in range(N + 2)}
    edges = []
    
    for u, v in zip(truck_r[:-1], truck_r[1:]):
        edges.append((u, v, TT[u][v]))
    for u, v in zip(drone_r[:-1], drone_r[1:]):
        edges.append((u, v, DT[u][v]))
        
    # Bellman-Ford algorithm for longest paths on the DAG
    for _ in range(N + 2):
        changed = False
        for u, v, cost in edges:
            if arr[v] < arr[u] + cost:
                arr[v] = arr[u] + cost
                changed = True
        if not changed:
            break
            
    return arr

def extract_route(edge_vars, start_node, end_node):
    route = [start_node]
    current = start_node
    count = 0
    while current != end_node and count < len(edge_vars):
        next_node = None
        for (i, j), var in edge_vars.items():
            if i == current and var.X > 0.5:
                next_node = j
                break
        if next_node is None:
            break
        route.append(next_node)
        current = next_node
        count += 1
    return route

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default="")
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # 1. Provide instance variables
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    N = data['num_customers']
    TT = data['truck_travel_times']
    DT = data['drone_travel_times']

    # Expand matrix to explicitly represent destination depot (node N+1)
    def expand_matrix(mat):
        new_mat = [row[:] + [row[0]] for row in mat]
        new_mat.append(mat[0] + [mat[0][0]])
        return new_mat
    
    TT_exp = expand_matrix(TT)
    DT_exp = expand_matrix(DT)

    # 2. Setup optimization model
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)  # Silence console output
    env.start()
    
    model = gp.Model("TruckDrone", env=env)
    model.setParam('Threads', 1)  # Enforce single CPU core requirement
    if args.time_limit:
        model.setParam('TimeLimit', args.time_limit)

    C = list(range(1, N + 1))
    
    # Track variables 
    x = {}
    y = {}
    
    # Create valid edges only, ignoring invalid depot shortcuts
    for i in range(N + 2):
        for j in range(N + 2):
            if i == j: continue
            if i == N + 1: continue  # No arcs can leave distinct destination depot Node N+1
            if j == 0: continue      # No arcs can enter distinct origin depot Node 0
            if i == 0 and j == N + 1: continue  # No zero-visit tracks directly connecting origin to destination
            
            x[i, j] = model.addVar(vtype=gp.GRB.BINARY, name=f"x_{i}_{j}")
            y[i, j] = model.addVar(vtype=gp.GRB.BINARY, name=f"y_{i}_{j}")

    # Node attributes mappings structure
    vT = {k: model.addVar(vtype=gp.GRB.BINARY, name=f"vT_{k}") for k in C}
    vD = {k: model.addVar(vtype=gp.GRB.BINARY, name=f"vD_{k}") for k in C}
    c  = {k: model.addVar(vtype=gp.GRB.BINARY, name=f"c_{k}") for k in C}
    
    max_time_val = sum(max(row) for row in TT_exp) + sum(max(row) for row in DT_exp)
    t  = {i: model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=max_time_val, name=f"t_{i}") for i in range(N + 2)}
    pT = {i: model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=N + 2, name=f"pT_{i}") for i in range(N + 2)}
    pD = {i: model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=N + 2, name=f"pD_{i}") for i in range(N + 2)}

    # Flow constraints matching requirements exactly
    for k in C:
        # Degree flows tracking
        model.addConstr(gp.quicksum(x[i, k] for i in range(N + 2) if (i, k) in x) == vT[k])
        model.addConstr(gp.quicksum(x[k, j] for j in range(N + 2) if (k, j) in x) == vT[k])
        
        model.addConstr(gp.quicksum(y[i, k] for i in range(N + 2) if (i, k) in y) == vD[k])
        model.addConstr(gp.quicksum(y[k, j] for j in range(N + 2) if (k, j) in y) == vD[k])
        
        # Every customer must be serviced logically "once" via specific configurations
        model.addConstr(vT[k] + vD[k] >= 1)
        model.addConstr(c[k] == vT[k] + vD[k] - 1)

    # Depot arcs rules
    model.addConstr(gp.quicksum(x[0, j] for j in range(N + 2) if (0, j) in x) == 1)
    model.addConstr(gp.quicksum(x[i, N + 1] for i in range(N + 2) if (i, N + 1) in x) == 1)
    model.addConstr(gp.quicksum(y[0, j] for j in range(N + 2) if (0, j) in y) == 1)
    model.addConstr(gp.quicksum(y[i, N + 1] for i in range(N + 2) if (i, N + 1) in y) == 1)

    # Specific strict drone limitations
    for i in C:
        for j in C:
            if i < j:
                # Limitation governing Drone jumps relative to its combined status assignment dynamically
                model.addConstr(y.get((i, j), 0) + y.get((j, i), 0) <= c[i] + c[j])
                
    for k in C:
        # Exclusion of direct return origin sorting loops
        model.addConstr(y.get((0, k), 0) + y.get((k, N + 1), 0) <= 1)

    # Initial anchor requirements
    model.addConstr(t[0] == 0)
    model.addConstr(pT[0] == 0)
    model.addConstr(pD[0] == 0)

    # Progress tracking linking spatial routing loops implicitly
    for (i, j), var in x.items():
        model.addConstr((var == 1) >> (t[j] >= t[i] + TT_exp[i][j]))
        model.addConstr((var == 1) >> (pT[j] >= pT[i] + 1))
        
    for (i, j), var in y.items():
        model.addConstr((var == 1) >> (t[j] >= t[i] + DT_exp[i][j]))
        model.addConstr((var == 1) >> (pD[j] >= pD[i] + 1))

    # Single tour synchronization destination limit objective minimize
    model.setObjective(t[N + 1], gp.GRB.MINIMIZE)
    
    # 3. Model convergence tracking hooks initialization
    def callback(cb_model, where):
        if where == gp.GRB.Callback.MIPSOL:
            obj = cb_model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)
                
    model.optimize(callback)
    
    # 4. Result Formatting Context Extraction 
    if model.SolCount == 0:
        print("No feasible solution obtained within the requested time envelope.", file=sys.stderr)
        sys.exit(1)

    truck_route = extract_route(x, 0, N + 1)
    drone_route = extract_route(y, 0, N + 1)

    cust_types = {}
    for i in C:
        vt_active = vT[i].X > 0.5
        vd_active = vD[i].X > 0.5
        
        if vt_active and vd_active:
            cust_types[str(i)] = "combined"
        elif vt_active:
            cust_types[str(i)] = "truck"
        elif vd_active:
            cust_types[str(i)] = "drone"

    # Derive precisely aligned exact completion milestones removing potential margin
    exact_arr_times_int = compute_earliest_arrival_times(truck_route, drone_route, TT_exp, DT_exp, N)
    
    # The distinct ending depot maps syntactically to its required ID per original schema guidelines.
    # Outbound sequences project explicit return origin keys safely.
    out_truck_route = [0 if v == (N + 1) else v for v in truck_route]
    out_drone_route = [0 if v == (N + 1) else v for v in drone_route]
    
    solution_arrival_dict = {str(k): float(v) for k, v in exact_arr_times_int.items()}

    solution_payload = {
        "objective_value": float(exact_arr_times_int[N + 1]),
        "truck_route": out_truck_route,
        "drone_route": out_drone_route,
        "customer_types": cust_types,
        "arrival_times": solution_arrival_dict
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution_payload, f, indent=4)

if __name__ == "__main__":
    main()