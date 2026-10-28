import argparse
import json
import os
import sys
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB

# Attempt to load logger
try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True, help="Path to the instance JSON file.")
    parser.add_argument('--solution_path', type=str, required=True, help="Path to write the solution JSON file.")
    parser.add_argument('--time_limit', type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument('--log_path', type=str, required=False, help="Path to the JSONL logging file.")
    args = parser.parse_args()

    # Setup logger
    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    # Read instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    N = data['graph']['num_nodes']
    depot = data['depot']
    K = data['fleet']['num_vehicles']
    C = data['fleet']['vehicle_capacity']
    edges_data = data['edges']
    M = len(edges_data)

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1)
    env.start()
    
    model = gp.Model("CARP", env=env)
    
    # Apply solver parameters based on requirements
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)  # Strictly enforce single-core environment
    model.setParam('Heuristics', 0.5)  # Increase heuristic effort to find feasible bounds faster
    model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions rapidly

    # Extract edge properties
    edge_idx_to_id = []
    e_req = []
    e_dem = []
    e_dc = []
    e_sc = []
    e_u = []
    e_v = []
    
    node_edges = defaultdict(list)
    for idx, e in enumerate(edges_data):
        edge_idx_to_id.append(e['edge_id'])
        e_req.append(e['is_required'])
        e_dem.append(e.get('demand', 0))
        e_dc.append(e.get('cost', 0))
        e_sc.append(e.get('service_cost', 0))
        
        u, v = e['endpoints']
        e_u.append(u)
        e_v.append(v)
        
        node_edges[u].append(idx)
        if u != v:
            node_edges[v].append(idx)  # Avoid duplicate indices for self-loops

    # Variables definition
    X = {}  # Binary: 1 if vehicle services required edge
    Y = {}  # Integer: times vehicle deadheads on edge
    w = {}  # Binary: 1 if vehicle uses edge for network flow (subtour elimination)
    f_uv = {}  # Flow from u -> v
    f_vu = {}  # Flow from v -> u
    z = {}  # Binary: 1 if node visited by vehicle
    P = {}  # Integer helper for node degree parity checking

    for k in range(K):
        for i in range(N):
            z[i,k] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{k}")
            P[i,k] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"P_{i}_{k}")
        
        for idx in range(M):
            if e_req[idx]:
                X[idx,k] = model.addVar(vtype=GRB.BINARY, name=f"X_{idx}_{k}")
            
            # An arbitrary, safe internal upper bound for maximum times an edge is deadheaded
            Y[idx,k] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=M, name=f"Y_{idx}_{k}")
            w[idx,k] = model.addVar(vtype=GRB.BINARY, name=f"w_{idx}_{k}")
            f_uv[idx,k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"fuv_{idx}_{k}")
            f_vu[idx,k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"fvu_{idx}_{k}")

    # Build the 'usage' expression representing all traversals of edge strictly
    U = {}
    for k in range(K):
        for idx in range(M):
            if e_req[idx]:
                U[idx,k] = X[idx,k] + Y[idx,k]
            else:
                U[idx,k] = Y[idx,k]

    # Degree parity and node activity connection
    # Node parity must strictly be even logic: Sum of traversed edges incidence connected to node i = 2 * P[i,k]
    for k in range(K):
        for i in range(N):
            D_expr = gp.quicksum(
                U[idx,k] * (2 if e_u[idx] == e_v[idx] else 1) 
                for idx in node_edges[i]
            )
            model.addConstr(D_expr == 2 * P[i,k], name=f"parity_{i}_{k}")
            
            # Subtour logic condition: If z[i,k] = 0, D_expr must be 0
            model.addGenConstrIndicator(z[i,k], 0, D_expr == 0, name=f"z_ind_{i}_{k}")

    # Connect utilization with flow variables logic
    for k in range(K):
        for idx in range(M):
            # Only edges that are utilized can hold active connectivity marker
            model.addConstr(U[idx,k] >= w[idx,k], name=f"link_U_w_{idx}_{k}")
            # Max flow needed is N - 1 nodes connecting up from depot
            model.addConstr(f_uv[idx,k] <= (N - 1) * w[idx,k], name=f"cap_uv_{idx}_{k}")
            model.addConstr(f_vu[idx,k] <= (N - 1) * w[idx,k], name=f"cap_vu_{idx}_{k}")

    # Hard service constraints natively required by optimization task
    for idx in range(M):
        if e_req[idx]:
            model.addConstr(gp.quicksum(X[idx,k] for k in range(K)) == 1, name=f"req_{idx}")

    # Subtour elimination mechanism: Nodes demand 1 flow packet from network layout to connect
    for k in range(K):
        for i in range(N):
            if i == depot:
                continue
            inflow = gp.quicksum(f_vu[idx,k] for idx in node_edges[i] if e_u[idx] == i) + \
                     gp.quicksum(f_uv[idx,k] for idx in node_edges[i] if e_v[idx] == i)
            outflow = gp.quicksum(f_uv[idx,k] for idx in node_edges[i] if e_u[idx] == i) + \
                      gp.quicksum(f_vu[idx,k] for idx in node_edges[i] if e_v[idx] == i)
            model.addConstr(inflow - outflow == z[i,k], name=f"flow_{i}_{k}")

    # Fleet limit breaking & Capacity limit
    load_vars = model.addVars(K, vtype=GRB.CONTINUOUS, name="load")
    for k in range(K):
        load_expr = gp.quicksum(e_dem[idx] * X[idx,k] for idx in range(M) if e_req[idx])
        model.addConstr(load_vars[k] == load_expr)
        model.addConstr(load_vars[k] <= C, name=f"cap_{k}")
        # Standard strict lexicographical symmetry breaking mapping down vehicles utilization order
        if k > 0:
            model.addConstr(load_vars[k-1] >= load_vars[k], name=f"sym_load_{k}")

    # Construct the final minimization objective dynamically
    total_service_cost = sum(e_sc[idx] for idx in range(M) if e_req[idx])
    deadhead_cost = gp.quicksum(
        e_dc[idx] * Y[idx,k] 
        for k in range(K) 
        for idx in range(M)
    )
    
    model.setObjective(deadhead_cost + total_service_cost, GRB.MINIMIZE)

    # Dynamic solver logging handler hooking
    class CallbackData:
        def __init__(self, logger_handler):
            self.logger_handler = logger_handler
    cb_data = CallbackData(logger)
    
    def mip_callback(mod, where):
        if where == GRB.Callback.MIPSOL:
            obj = mod.cbGet(GRB.Callback.MIPSOL_OBJ)
            if cb_data.logger_handler:
                cb_data.logger_handler.log(obj)

    # Deploy extraction
    model.optimize(mip_callback)

    # Provide safe fallback when no feasible route sets found via hard cut-offs
    if model.SolCount == 0:
        sys.exit(1)

    # Reconstruct formatted routes payload aligned directly with `solution_schema.json` format rules 
    routes = []
    for k in range(K):
        serviced_edges = []
        deadheaded_edges = []
        
        for idx in range(M):
            if e_req[idx]:
                if X[idx,k].X > 0.5:
                    serviced_edges.append(edge_idx_to_id[idx])
                    
            y_val = int(round(Y[idx,k].X))
            if y_val > 0:
                deadheaded_edges.append({
                    "edge_id": edge_idx_to_id[idx],
                    "times": y_val
                })
                
        routes.append({
            "vehicle": k,
            "serviced_edges": serviced_edges,
            "deadheaded_edges": deadheaded_edges
        })

    ans = {
        "objective_value": model.ObjVal,
        "routes": routes
    }

    # Persist derived output sequence structure
    with open(args.solution_path, 'w') as f:
        json.dump(ans, f, indent=2)

if __name__ == "__main__":
    main()