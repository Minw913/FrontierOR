import sys
import json
import argparse
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def solve_pdptw_lifo(args):
    with open(args.instance_path, "r") as f:
        data = json.load(f)
        
    n_requests = data["n_requests"]
    capacity = data["vehicle_capacity"]
    
    nodes = {}
    for nd in data["nodes"]:
        nodes[nd["node_id"]] = nd
        
    travel_cost = {}
    travel_time = {}
    
    for arc in data["arcs"]:
        u, v = arc["from"], arc["to"]
        travel_cost[u, v] = arc["cost"]
        travel_time[u, v] = arc["travel_time"]
        
    P_nodes = list(range(1, n_requests + 1))
    D_nodes = list(range(n_requests + 1, 2 * n_requests + 1))
    dest_depot = 2 * n_requests + 1
    
    # Pre-filter valid edges to reduce model size
    valid_edges = []
    for (u, v) in travel_cost.keys():
        if u == v:
            continue
        # Origin depot cannot go directly to a delivery
        if u == 0 and v in D_nodes:
            continue
        # Pickup cannot go directly to the destination depot
        if u in P_nodes and v == dest_depot:
            continue
        # LIFO: A pickup node cannot be followed immediately by a delivery node of a different request
        if u in P_nodes and v in D_nodes and v != u + n_requests:
            continue
        # Capacity validation for consecutive pickups
        if u in P_nodes and v in P_nodes:
            if nodes[u]['load'] + nodes[v]['load'] > capacity:
                continue
        # Time window validation for any path
        if nodes[u]['tw_early'] + nodes[u]['service_time'] + travel_time[u, v] > nodes[v]['tw_late']:
            continue
            
        valid_edges.append((u, v))
        
    # Build Gurobi Model
    model = gp.Model("PDPTW_LIFO")
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1  # Single CPU environment constraints
    model.Params.OutputFlag = 1
    
    # Flow variables
    x = {}
    for u, v in valid_edges:
        x[u, v] = model.addVar(vtype=GRB.BINARY, obj=travel_cost[u, v], name=f"x_{u}_{v}")
        x[u, v].start = 0  # Default initial solution is empty
        
    T = {}
    L = {}
    S = {}
    Rt = {}
    
    # State variables at each pickup/delivery node
    for i in P_nodes + D_nodes:
        T[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=nodes[i]['tw_early'], ub=nodes[i]['tw_late'], name=f"T_{i}")
        L[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=capacity, name=f"L_{i}")
        S[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=n_requests, name=f"S_{i}")
        Rt[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=1, ub=n_requests, name=f"Rt_{i}")

        # Tighten specific state bounds based on node properties
        if i in P_nodes:
            L[i].lb = float(nodes[i]['load'])
            S[i].lb = 1
        else:
            L[i].ub = float(capacity + nodes[i]['load'])  # load is negative for delivery
            S[i].ub = n_requests - 1
            
    # Trivial distinct vehicle feasible start (dedicated route per request)
    for r in range(1, n_requests + 1):
        P, D = r, r + n_requests
        if (0, P) in x and (P, D) in x and (D, dest_depot) in x:
            x[0, P].start = 1
            x[P, D].start = 1
            x[D, dest_depot].start = 1
            
    # Flow conservation constraints
    for i in P_nodes + D_nodes:
        out_edges = [x[u, v] for u, v in valid_edges if u == i]
        in_edges = [x[u, v] for u, v in valid_edges if v == i]
        model.addConstr(gp.quicksum(out_edges) == 1, name=f"out_{i}")
        model.addConstr(gp.quicksum(in_edges) == 1, name=f"in_{i}")

    # Evolution constraints
    for i, j in valid_edges:
        if j == dest_depot:
            continue
            
        is_P = 1 if j in P_nodes else 0
        delta_j = 1 if is_P else -1
        
        # Service Start Time propagation
        if i != 0:
            M_T = max(0, nodes[i]['tw_late'] + nodes[i]['service_time'] + travel_time[i, j] - nodes[j]['tw_early'])
            model.addConstr(T[j] >= T[i] + nodes[i]['service_time'] + travel_time[i, j] - M_T * (1 - x[i, j]))
            
        # Resource dependencies (Load, Stack Depth, Route Identifier)
        if i == 0:
            model.addConstr(L[j] >= nodes[j]['load'] - capacity * (1 - x[i, j]))
            model.addConstr(L[j] <= nodes[j]['load'] + capacity * (1 - x[i, j]))
            
            model.addConstr(S[j] >= 1 - n_requests * (1 - x[i, j]))
            model.addConstr(S[j] <= 1 + n_requests * (1 - x[i, j]))
            
            model.addConstr(Rt[j] >= j - n_requests * (1 - x[i, j]))
            model.addConstr(Rt[j] <= j + n_requests * (1 - x[i, j]))
        else:
            model.addConstr(L[j] >= L[i] + nodes[j]['load'] - capacity * (1 - x[i, j]))
            model.addConstr(L[j] <= L[i] + nodes[j]['load'] + capacity * (1 - x[i, j]))
            
            model.addConstr(S[j] >= S[i] + delta_j - n_requests * (1 - x[i, j]))
            model.addConstr(S[j] <= S[i] + delta_j + n_requests * (1 - x[i, j]))
            
            model.addConstr(Rt[j] >= Rt[i] - n_requests * (1 - x[i, j]))
            model.addConstr(Rt[j] <= Rt[i] + n_requests * (1 - x[i, j]))
            
    # Relational constraints between respective pickup and delivery matched pairs
    for r in range(1, n_requests + 1):
        P, D = r, r + n_requests
        # Strict continuous vehicle bounding
        model.addConstr(Rt[P] == Rt[D])
        # Direct LIFO sequence enforcement
        model.addConstr(S[D] == S[P] - 1)
        # Proper Precedence constraint
        tt = travel_time.get((P, D), 0)
        model.addConstr(T[D] >= T[P] + nodes[P]['service_time'] + tt)

    # Initialize Logger Config
    logger = SolutionLogger(args.log_path, sense="minimize") if SolutionLogger and args.log_path else None
    model._best_obj = float('inf')
    model._logger = logger

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < model._best_obj - 1e-4:
                model._best_obj = obj
                if model._logger is not None:
                    model._logger.log(obj)

    model.optimize(cb)

    # Post-process routing solution JSON structures
    routes = []
    obj_val = -1.0
    
    if model.SolCount > 0:
        obj_val = model.ObjVal
        
        next_node = {}
        for u, v in valid_edges:
            if x[u, v].X > 0.5:
                if u not in next_node:
                    next_node[u] = []
                next_node[u].append(v)
                
        start_nodes = next_node.get(0, [])
        for vehicle_id, start_node in enumerate(start_nodes, start=1):
            seq = [0]
            details = []
            curr = start_node
            
            arr_t_start = T[curr].X - nodes[0]['service_time'] - travel_time[0, curr]
            arr_t_start = max(float(nodes[0]['tw_early']), arr_t_start)
            
            details.append({
                "node_id": 0,
                "arrival_time": float(arr_t_start),
                "load_after": 0.0
            })
            
            while curr != dest_depot:
                seq.append(curr)
                details.append({
                    "node_id": curr,
                    "arrival_time": float(T[curr].X),
                    "load_after": float(L[curr].X)
                })
                curr = next_node[curr][0]
                
            seq.append(curr)
            last_node = seq[-2]
            arr_t_end = T[last_node].X + nodes[last_node]['service_time'] + travel_time[last_node, curr]
            
            details.append({
                "node_id": curr,
                "arrival_time": float(arr_t_end),
                "load_after": 0.0
            })
            
            routes.append({
                "vehicle_id": vehicle_id,
                "route_sequence": seq,
                "route_details": details
            })
            
    # Serialize exact adherence to schema specifications
    sys.setrecursionlimit(2000)
    with open(args.solution_path, "w") as f:
        json.dump({
            "objective_value": float(obj_val),
            "routes": routes
        }, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False)
    
    args = parser.parse_args()
    solve_pdptw_lifo(args)