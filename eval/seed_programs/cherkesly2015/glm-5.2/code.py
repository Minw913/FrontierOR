import argparse
import json
import math
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

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

    N = instance['n_requests']
    n_nodes = instance['n_nodes']
    C = instance['vehicle_capacity']
    H = instance['planning_horizon'][1]
    
    nodes = instance['nodes']
    arcs_data = instance['arcs']
    
    node_map = {node['node_id']: node for node in nodes}
    
    arcs_out = {}
    arcs_in = {}
    cost = {}
    travel_time = {}
    
    for arc in arcs_data:
        u, v = arc['from'], arc['to']
        if u not in arcs_out:
            arcs_out[u] = []
        if v not in arcs_in:
            arcs_in[v] = []
        arcs_out[u].append(v)
        arcs_in[v].append(u)
        cost[u, v] = arc['cost']
        travel_time[u, v] = arc['travel_time']
        
    d = {}
    s = {}
    e = {}
    f = {}
    for node in nodes:
        i = node['node_id']
        d[i] = node['load']
        s[i] = node['service_time']
        e[i] = node['tw_early']
        f[i] = node['tw_late']
        
    def get_travel_time(u, v):
        if (u, v) in travel_time:
            return travel_time[u, v]
        nu = node_map[u]
        nv = node_map[v]
        return math.sqrt((nu['x'] - nv['x'])**2 + (nu['y'] - nv['y'])**2)

    K = N  # Maximum number of vehicles (each request can be served by a separate vehicle)
    m = gp.Model("PDPTW_LIFO")
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = args.time_limit

    x = {}
    for k in range(K):
        for u in arcs_out:
            for v in arcs_out[u]:
                x[k, u, v] = m.addVar(vtype=GRB.BINARY, obj=cost[u, v], name=f"x_{k}_{u}_{v}")

    t = {}
    l = {}
    y = {}
    for k in range(K):
        for i in node_map.keys():
            t[k, i] = m.addVar(lb=e[i], ub=f[i], name=f"t_{k}_{i}")
            lb = max(0, d[i])
            ub = min(C, C + d[i])
            l[k, i] = m.addVar(lb=lb, ub=ub, name=f"l_{k}_{i}")
        for i in range(1, N + 1):
            y[k, i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"y_{k}_{i}")
            m.addConstr(y[k, i] == sum(x[k, j, i] for j in arcs_in.get(i, [])))

    z = {}
    for k in range(K):
        for i in range(1, N + 1):
            for j in range(i + 1, N + 1):
                z[k, i, j] = m.addVar(vtype=GRB.BINARY, name=f"z_{k}_{i}_{j}")
                m.addConstr(z[k, i, j] <= y[k, i])
                m.addConstr(z[k, i, j] <= y[k, j])

    m.update()

    # Each pickup visited exactly once
    for i in range(1, N + 1):
        m.addConstr(sum(x[k, j, i] for k in range(K) for j in arcs_in.get(i, [])) == 1)

    for k in range(K):
        # Flow conservation
        for i in range(1, 2 * N + 1):
            m.addConstr(sum(x[k, j, i] for j in arcs_in.get(i, [])) == sum(x[k, i, j] for j in arcs_out.get(i, [])))
        
        # Start and end at depot
        m.addConstr(sum(x[k, 0, j] for j in arcs_out.get(0, [])) <= 1)
        m.addConstr(sum(x[k, i, 2*N+1] for i in arcs_in.get(2*N+1, [])) == sum(x[k, 0, j] for j in arcs_out.get(0, [])))
        
        # Pairing constraints
        for i in range(1, N + 1):
            m.addConstr(sum(x[k, j, i] for j in arcs_in.get(i, [])) == sum(x[k, j, i+N] for j in arcs_in.get(i+N, [])))
            
        # Capacity constraints
        for u in arcs_out:
            for v in arcs_out[u]:
                if v != 2*N+1:
                    m.addConstr(l[k, v] >= l[k, u] + d[v] - C * (1 - x[k, u, v]))
                    m.addConstr(l[k, v] <= l[k, u] + d[v] + C * (1 - x[k, u, v]))
                    
        # Time window constraints
        for u in arcs_out:
            for v in arcs_out[u]:
                if v != 2*N+1:
                    m.addConstr(t[k, v] >= t[k, u] + s[u] + get_travel_time(u, v) - H * (1 - x[k, u, v]))
                    
        # Precedence constraints
        for i in range(1, N + 1):
            m.addConstr(t[k, i+N] >= t[k, i] + s[i] + get_travel_time(i, i+N) - H * (1 - y[k, i]))
            
        # LIFO constraints
        for i in range(1, N + 1):
            for j in range(i + 1, N + 1):
                m.addConstr(t[k, j] - t[k, i] >= -H * (1 - z[k, i, j]) - H * (2 - y[k, i] - y[k, j]))
                m.addConstr(t[k, j] - t[k, i] <= H * z[k, i, j] + H * (2 - y[k, i] - y[k, j]))
                m.addConstr(t[k, i+N] - t[k, j+N] >= -H * (1 - z[k, i, j]) - H * (2 - y[k, i] - y[k, j]))
                m.addConstr(t[k, i+N] - t[k, j+N] <= H * z[k, i, j] + H * (2 - y[k, i] - y[k, j]))

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                x_vals = model.cbGetSolution(model._x)
                t_vals = model.cbGetSolution(model._t)
                l_vals = model.cbGetSolution(model._l)
                routes = []
                for k in range(K):
                    seq = [0]
                    curr = 0
                    while curr != 2*N+1:
                        next_node = None
                        for v in arcs_out.get(curr, []):
                            if x_vals[k, curr, v] > 0.5:
                                next_node = v
                                break
                        if next_node is None:
                            break
                        seq.append(next_node)
                        curr = next_node
                    if len(seq) > 2:
                        route_details = []
                        for node in seq:
                            route_details.append({
                                "node_id": node,
                                "arrival_time": t_vals[k, node],
                                "load_after": l_vals[k, node]
                            })
                        routes.append({
                            "vehicle_id": k,
                            "route_sequence": seq,
                            "route_details": route_details
                        })
                solution_dict = {
                    "objective_value": obj,
                    "routes": routes
                }
                logger.log_solution(obj, solution_dict)

    m._x = x
    m._t = t
    m._l = l
    m.optimize(callback)

    if m.SolCount > 0:
        obj = m.ObjVal
        x_vals = m.getAttr('X', x)
        t_vals = m.getAttr('X', t)
        l_vals = m.getAttr('X', l)
        routes = []
        for k in range(K):
            seq = [0]
            curr = 0
            while curr != 2*N+1:
                next_node = None
                for v in arcs_out.get(curr, []):
                    if x_vals[k, curr, v] > 0.5:
                        next_node = v
                        break
                if next_node is None:
                    break
                seq.append(next_node)
                curr = next_node
            if len(seq) > 2:
                route_details = []
                for node in seq:
                    route_details.append({
                        "node_id": node,
                        "arrival_time": t_vals[k, node],
                        "load_after": l_vals[k, node]
                    })
                routes.append({
                    "vehicle_id": k,
                    "route_sequence": seq,
                    "route_details": route_details
                })
        solution_dict = {
            "objective_value": obj,
            "routes": routes
        }
    else:
        solution_dict = {
            "objective_value": 0.0,
            "routes": []
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f, indent=2)

if __name__ == '__main__':
    main()