import json
import argparse
import time
import collections
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    global_params = data["global_parameters"]
    weight_lambda = global_params["lambda"]
    freq_ub = global_params["frequency_upper_bound_F"]
    max_line_length = global_params["max_line_length_edges"]
    node_ids = [n["id"] for n in data["network"]["nodes"]]
    num_nodes = len(node_ids)
    
    # 1. Generate candidate lines
    lines = []
    line_idx = 0
    
    max_gen_time = min(20, max(2, args.time_limit * 0.2))
    num_modes = len(data.get("modes", []))
    max_lines_per_mode = 50000 // num_modes if num_modes > 0 else 50000
    time_per_mode = max_gen_time / num_modes if num_modes > 0 else max_gen_time

    for mode in data.get("modes", []):
        m_name = mode["name"]
        v_cap = mode["vehicle_capacity"]
        terminals = set(mode["terminals"])
        
        adj = collections.defaultdict(list)
        for e_idx in mode["edge_indices"]:
            e = data["network"]["edges"][e_idx]
            u, v = e["endpoints"]
            adj[u].append((v, e_idx))
            adj[v].append((u, e_idx))
            
        queue = collections.deque()
        for t in terminals:
            queue.append((t, t, [t], [])) 
            
        mode_lines = 0
        mode_start_time = time.time()
        while queue:
            if time.time() - mode_start_time > time_per_mode: 
                break
            if mode_lines >= max_lines_per_mode:
                break
                
            start_node, curr, path_nodes, path_edges = queue.popleft()
            
            if len(path_edges) > 0 and curr in terminals and start_node < curr:
                op_cost = sum(data["network"]["edges"][e_idx]["operating_cost"] for e_idx in path_edges)
                lines.append({
                    "line_index": line_idx,
                    "mode": m_name,
                    "nodes": path_nodes,
                    "edges": path_edges,
                    "vehicle_capacity": v_cap,
                    "fixed_cost": mode.get("fixed_cost_per_line", global_params.get("fixed_cost_per_line", 0.0)),
                    "operating_cost": op_cost
                })
                line_idx += 1
                mode_lines += 1
                
            if len(path_edges) < max_line_length:
                for nxt, e_idx in adj[curr]:
                    if nxt not in path_nodes:
                        queue.append((start_node, nxt, path_nodes + [nxt], path_edges + [e_idx]))

    # 2. Build Gurobi Model
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    model = gp.Model("TransitNetwork", env=env)
    
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    
    rem_time = args.time_limit - (time.time() - start_time) - 5
    if rem_time < 5:
        rem_time = 5
    model.setParam('TimeLimit', rem_time)

    y = {}
    f = {}
    for i, l in enumerate(lines):
        y[i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}")
        f[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=freq_ub, name=f"f_{i}")

    OD_demands = collections.defaultdict(dict)
    for od in data.get("od_matrix", []):
        o = od["origin"]
        d = od["destination"]
        dem = od["demand"]
        if dem > 0:
            OD_demands[o][d] = dem

    arc_time = {}
    arc_tail = {}
    arc_head = {}
    for e in data["network"]["edges"]:
        e_id = e["id"]
        u, v = e["endpoints"]
        t = e["traveling_time_seconds"]
        
        a_fwd = 2 * e_id
        arc_time[a_fwd] = t
        arc_tail[a_fwd] = u
        arc_head[a_fwd] = v
        
        a_bwd = 2 * e_id + 1
        arc_time[a_bwd] = t
        arc_tail[a_bwd] = v
        arc_head[a_bwd] = u

    arcs = list(arc_time.keys())
    out_arcs = {v: [] for v in node_ids}
    in_arcs = {v: [] for v in node_ids}
    for a in arcs:
        out_arcs[arc_tail[a]].append(a)
        in_arcs[arc_head[a]].append(a)

    x = collections.defaultdict(dict)
    for o in OD_demands.keys():
        for a in arcs:
            x[o][a] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_{o}_{a}")

    for i in range(len(lines)):
        model.addConstr(f[i] <= freq_ub * y[i])

    lines_on_edge = collections.defaultdict(list)
    for i, l in enumerate(lines):
        for e_idx in l["edges"]:
            lines_on_edge[e_idx].append(i)

    for e in data["network"]["edges"]:
        e_id = e["id"]
        e_cap = e["edge_capacity"]
        model.addConstr(gp.quicksum(f[i] for i in lines_on_edge[e_id]) <= e_cap)
        C_e = gp.quicksum(lines[i]["vehicle_capacity"] * f[i] for i in lines_on_edge[e_id])
        model.addConstr(gp.quicksum(x[o][2 * e_id] for o in OD_demands.keys()) <= C_e)
        model.addConstr(gp.quicksum(x[o][2 * e_id + 1] for o in OD_demands.keys()) <= C_e)

    for o, demands in OD_demands.items():
        S_o = sum(demands.values())
        for v in node_ids:
            req = S_o if v == o else (-demands[v] if v in demands else 0)
            if out_arcs[v] or in_arcs[v] or req != 0:
                outflow = gp.quicksum(x[o][a] for a in out_arcs[v])
                inflow = gp.quicksum(x[o][a] for a in in_arcs[v])
                model.addConstr(outflow - inflow == req)

    total_line_cost = gp.quicksum(lines[i]["fixed_cost"] * y[i] + lines[i]["operating_cost"] * f[i] for i in range(len(lines)))
    total_travel_time = gp.quicksum(arc_time[a] * x[o][a] for o in OD_demands.keys() for a in arcs)

    model.setObjective(weight_lambda * total_line_cost + (1 - weight_lambda) * total_travel_time, GRB.MINIMIZE)

    # 3. Callback and Solution Extraction
    def extract_solution(get_val):
        active_lines = []
        for i, l in enumerate(lines):
            y_val = get_val(y[i])
            if y_val > 0.5:
                f_val = get_val(f[i])
                active_lines.append({
                    "line_index": l["line_index"],
                    "mode": l["mode"],
                    "nodes": l["nodes"],
                    "edges": l["edges"],
                    "frequency": float(max(0.0, f_val))
                })
                
        active_paths = []
        for o, demands in OD_demands.items():
            flows = {a: get_val(x[o][a]) for a in arcs if get_val(x[o][a]) > 1e-5}
            nodes_with_flow = set()
            for a in flows:
                nodes_with_flow.add(arc_tail[a])
                nodes_with_flow.add(arc_head[a])
                
            # Eliminate flow cycles
            while True:
                adj_cyc = collections.defaultdict(list)
                for a, fl in flows.items():
                    if fl > 1e-5:
                        adj_cyc[arc_tail[a]].append(a)
                        
                visited = set()
                in_stack = set()
                parent_arc = {}
                found_cycle = None
                
                for start_nd in nodes_with_flow:
                    if start_nd in visited: continue
                    stack = [(start_nd, 0)]
                    while stack:
                        u, state = stack.pop()
                        if state == 0:
                            visited.add(u)
                            in_stack.add(u)
                            stack.append((u, 1))
                            for a in adj_cyc[u]:
                                v = arc_head[a]
                                if v not in visited:
                                    parent_arc[v] = a
                                    stack.append((v, 0))
                                elif v in in_stack:
                                    cycle_arcs = [a]
                                    curr = u
                                    while curr != v:
                                        p_a = parent_arc[curr]
                                        cycle_arcs.append(p_a)
                                        curr = arc_tail[p_a]
                                    found_cycle = cycle_arcs[::-1]
                                    break
                            if found_cycle: break
                        else:
                            in_stack.remove(u)
                    if found_cycle: break
                
                if not found_cycle:
                    break
                    
                min_f = min(flows[a] for a in found_cycle)
                for a in found_cycle:
                    flows[a] -= min_f

            # Flow decomposition
            for d, demand in demands.items():
                rem_demand = demand
                while rem_demand > 1e-5:
                    adj_path = collections.defaultdict(list)
                    for a, fl in flows.items():
                        if fl > 1e-5:
                            adj_path[arc_tail[a]].append(a)
                    
                    queue = collections.deque([(o, [])])
                    visited_bfs = {o}
                    path = None
                    while queue:
                        curr, cur_path = queue.popleft()
                        if curr == d:
                            path = cur_path
                            break
                        for a in adj_path[curr]:
                            nxt = arc_head[a]
                            if nxt not in visited_bfs:
                                visited_bfs.add(nxt)
                                queue.append((nxt, cur_path + [a]))
                                
                    if path:
                        bottleneck = min([flows[a] for a in path] + [rem_demand])
                        if bottleneck < 1e-5:
                            path = None # trigger fallback
                            
                    if path:
                        active_paths.append({
                            "origin": o,
                            "destination": d,
                            "arcs": path,
                            "flow": float(bottleneck)
                        })
                        for a in path:
                            flows[a] -= bottleneck
                        rem_demand -= bottleneck
                    else:
                        # Fallback path routing to ensure absolute demand match
                        queue = collections.deque([(o, [])])
                        visited_fallback = {o}
                        fallback_path = None
                        while queue:
                            curr, cur_path = queue.popleft()
                            if curr == d:
                                fallback_path = cur_path
                                break
                            for a in out_arcs[curr]:
                                nxt = arc_head[a]
                                if nxt not in visited_fallback:
                                    visited_fallback.add(nxt)
                                    queue.append((nxt, cur_path + [a]))
                        if fallback_path:
                            active_paths.append({
                                "origin": o,
                                "destination": d,
                                "arcs": fallback_path,
                                "flow": float(rem_demand)
                            })
                        break
                        
        line_cost = sum(lines[al["line_index"]]["fixed_cost"] + lines[al["line_index"]]["operating_cost"] * al["frequency"] for al in active_lines)
        travel_time = sum(arc_time[a] * p["flow"] for p in active_paths for a in p["arcs"])
        obj_val = weight_lambda * line_cost + (1 - weight_lambda) * travel_time
        
        return {
            "objective_value": float(obj_val),
            "active_lines": active_lines,
            "active_passenger_paths": active_paths
        }

    if logger:
        def cb(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(float(obj))
                try:
                    sol_dict = extract_solution(model.cbGetSolution)
                    logger.log_solution(sol_dict["objective_value"], sol_dict)
                except Exception:
                    pass
        model.optimize(cb)
    else:
        model.optimize()

    # 4. Final Output Construction
    if model.SolCount > 0:
        sol_dict = extract_solution(lambda var: var.X)
    else:
        sol_dict = {
            "objective_value": 0.0,
            "active_lines": [],
            "active_passenger_paths": []
        }

    with open(args.solution_path, 'w') as f:
        json.dump(sol_dict, f, indent=2)

if __name__ == "__main__":
    main()