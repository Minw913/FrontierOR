import json
import time
import argparse
from collections import deque
import gurobipy as gp
from gurobipy import GRB

def generate_candidate_lines(instance, edge_id_to_operating_cost, max_line_length_edges, max_candidate_paths_per_pair=5):
    candidate_lines = []
    modes = instance["modes"]
    for mode in modes:
        mode_name = mode["name"]
        terminals = mode["terminals"]
        adj = {}
        for idx in mode["edge_indices"]:
            edge = instance["network"]["edges"][idx]
            u, v = edge["endpoints"]
            edge_id = edge["id"]
            travel_time = edge["traveling_time_seconds"]
            if u not in adj:
                adj[u] = []
            if v not in adj:
                adj[v] = []
            adj[u].append((v, edge_id, travel_time))
            adj[v].append((u, edge_id, travel_time))
        
        for i in range(len(terminals)):
            s = terminals[i]
            for j in range(i+1, len(terminals)):
                t = terminals[j]
                best_paths = []
                queue = deque()
                queue.append((s, [s], [], 0))
                while queue:
                    u, path_nodes, path_edges, total_time = queue.popleft()
                    if u == t:
                        best_paths.append((total_time, path_nodes, path_edges))
                        best_paths.sort(key=lambda x: x[0])
                        if len(best_paths) > max_candidate_paths_per_pair:
                            best_paths.pop()
                        continue
                    if len(path_edges) < max_line_length_edges:
                        if u in adj:
                            for neighbor, edge_id, tt in adj[u]:
                                if neighbor not in path_nodes:
                                    new_path_nodes = path_nodes + [neighbor]
                                    new_path_edges = path_edges + [edge_id]
                                    new_time = total_time + tt
                                    queue.append((neighbor, new_path_nodes, new_path_edges, new_time))
                for travel_time, node_list, edge_list in best_paths:
                    op_cost = sum(edge_id_to_operating_cost[eid] for eid in edge_list)
                    fixed_cost = mode["fixed_cost_per_line"]
                    vehicle_capacity = mode["vehicle_capacity"]
                    candidate_lines.append({
                        "mode": mode_name,
                        "nodes": node_list,
                        "edges": edge_list,
                        "fixed_cost": fixed_cost,
                        "operating_cost": op_cost,
                        "vehicle_capacity": vehicle_capacity
                    })
    return candidate_lines

def build_directed_graph(instance):
    arc_list = []
    arc_travel_time = {}
    arc_to_undir_edge_id = {}
    undir_edge_id_to_arcs = {}
    for edge in instance["network"]["edges"]:
        u, v = edge["endpoints"]
        edge_id = edge["id"]
        travel_time = edge["traveling_time_seconds"]
        arc_uv = (u, v)
        arc_vu = (v, u)
        arc_list.append(arc_uv)
        arc_list.append(arc_vu)
        arc_travel_time[arc_uv] = travel_time
        arc_travel_time[arc_vu] = travel_time
        arc_to_undir_edge_id[arc_uv] = edge_id
        arc_to_undir_edge_id[arc_vu] = edge_id
        if edge_id not in undir_edge_id_to_arcs:
            undir_edge_id_to_arcs[edge_id] = []
        undir_edge_id_to_arcs[edge_id].extend([arc_uv, arc_vu])
    return {
        "arc_list": arc_list,
        "arc_travel_time": arc_travel_time,
        "arc_to_undir_edge_id": arc_to_undir_edge_id,
        "undir_edge_id_to_arcs": undir_edge_id_to_arcs
    }

def extract_solution(instance, candidate_lines, z, f, x, directed_graph_info, od_list):
    solution_dict = {
        "active_lines": [],
        "active_passenger_paths": []
    }
    for l_idx, line in enumerate(candidate_lines):
        if z[l_idx].X > 0.5 and f[l_idx].X > 1e-5:
            solution_dict["active_lines"].append({
                "mode": line["mode"],
                "nodes": line["nodes"],
                "edges": line["edges"],
                "frequency": f[l_idx].X
            })
    
    directed_arc_id = {}
    for edge in instance["network"]["edges"]:
        u, v = edge["endpoints"]
        eid = edge["id"]
        directed_arc_id[(u, v)] = 2 * eid
        directed_arc_id[(v, u)] = 2 * eid + 1
    
    arc_list = directed_graph_info["arc_list"]
    for od in od_list:
        o, d = od
        flow_on_arc = {}
        for arc in arc_list:
            if x[od, arc].X > 1e-5:
                flow_on_arc[arc] = x[od, arc].X
        while flow_on_arc:
            parent = {}
            queue = deque([o])
            parent[o] = None
            found = False
            while queue:
                u = queue.popleft()
                if u == d:
                    found = True
                    break
                for arc in arc_list:
                    if arc[0] == u and arc in flow_on_arc and flow_on_arc[arc] > 0:
                        v = arc[1]
                        if v not in parent:
                            parent[v] = (u, arc)
                            queue.append(v)
            if not found:
                break
            path_arcs = []
            u = d
            min_flow = float('inf')
            while u != o:
                prev, arc = parent[u]
                path_arcs.append(arc)
                min_flow = min(min_flow, flow_on_arc[arc])
                u = prev
            path_arcs.reverse()
            solution_dict["active_passenger_paths"].append({
                "origin": o,
                "destination": d,
                "arcs": [directed_arc_id[arc] for arc in path_arcs],
                "flow": min_flow
            })
            for arc in path_arcs:
                flow_on_arc[arc] -= min_flow
                if flow_on_arc[arc] < 1e-5:
                    del flow_on_arc[arc]
    return solution_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    global_params = instance["global_parameters"]
    lambda_val = global_params["lambda"]
    F = global_params["frequency_upper_bound_F"]
    max_line_length_edges = global_params["max_line_length_edges"]

    edge_id_to_operating_cost = {}
    edge_id_to_capacity = {}
    for edge in instance["network"]["edges"]:
        edge_id = edge["id"]
        edge_id_to_operating_cost[edge_id] = edge["operating_cost"]
        edge_id_to_capacity[edge_id] = edge["edge_capacity"]

    candidate_lines = generate_candidate_lines(instance, edge_id_to_operating_cost, max_line_length_edges)
    directed_graph_info = build_directed_graph(instance)
    arc_list = directed_graph_info["arc_list"]
    arc_travel_time = directed_graph_info["arc_travel_time"]
    arc_to_undir_edge_id = directed_graph_info["arc_to_undir_edge_id"]

    od_list = []
    od_demand = {}
    for od in instance["od_matrix"]:
        o = od["origin"]
        d = od["destination"]
        od_pair = (o, d)
        od_list.append(od_pair)
        od_demand[od_pair] = od["demand"]

    incoming = {}
    outgoing = {}
    nodes = [node["id"] for node in instance["network"]["nodes"]]
    for node in nodes:
        incoming[node] = []
        outgoing[node] = []
    for arc in arc_list:
        u, v = arc
        outgoing[u].append(arc)
        incoming[v].append(arc)

    edge_to_lines = {}
    for l_idx, line in enumerate(candidate_lines):
        for eid in line["edges"]:
            if eid not in edge_to_lines:
                edge_to_lines[eid] = []
            edge_to_lines[eid].append(l_idx)

    time_spent = time.time() - start_time
    remaining_time = max(1, args.time_limit - time_spent)

    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.setParam('Seed', 0)
    m.setParam('MIPGap', 1e-4)
    m.setParam('NumericFocus', 0)
    m.setParam('Threads', 1)
    m.setParam('TimeLimit', remaining_time)

    z = m.addVars(len(candidate_lines), vtype=GRB.BINARY, name="z")
    f = m.addVars(len(candidate_lines), lb=0, ub=F, name="f")
    x = m.addVars([(od, arc) for od in od_list for arc in arc_list], lb=0, name="x")

    for l_idx in range(len(candidate_lines)):
        m.addConstr(f[l_idx] <= F * z[l_idx], name=f"link_{l_idx}")

    for eid, lines in edge_to_lines.items():
        expr = gp.quicksum(f[l_idx] for l_idx in lines)
        m.addConstr(expr <= edge_id_to_capacity[eid], name=f"edge_cap_{eid}")

    for od in od_list:
        o, d = od
        for node in nodes:
            in_arcs = incoming[node]
            out_arcs = outgoing[node]
            expr_in = gp.quicksum(x[od, a] for a in in_arcs)
            expr_out = gp.quicksum(x[od, a] for a in out_arcs)
            if node == o:
                m.addConstr(expr_out - expr_in == od_demand[od], name=f"flow_cons_{od}_{node}")
            elif node == d:
                m.addConstr(expr_in - expr_out == od_demand[od], name=f"flow_cons_{od}_{node}")
            else:
                m.addConstr(expr_in - expr_out == 0, name=f"flow_cons_{od}_{node}")

    for arc in arc_list:
        eid = arc_to_undir_edge_id[arc]
        if eid in edge_to_lines:
            cap_expr = gp.quicksum(candidate_lines[l_idx]["vehicle_capacity"] * f[l_idx] for l_idx in edge_to_lines[eid])
        else:
            cap_expr = 0
        flow_expr = gp.quicksum(x[od, arc] for od in od_list)
        m.addConstr(flow_expr <= cap_expr, name=f"arc_cap_{arc}")

    total_line_cost = 0
    for l_idx, line in enumerate(candidate_lines):
        total_line_cost += line["fixed_cost"] * z[l_idx] + line["operating_cost"] * f[l_idx]

    total_passenger_time = 0
    for arc in arc_list:
        flow_expr = gp.quicksum(x[od, arc] for od in od_list)
        total_passenger_time += arc_travel_time[arc] * flow_expr

    obj = lambda_val * total_line_cost + (1 - lambda_val) * total_passenger_time
    m.setObjective(obj, GRB.MINIMIZE)

    def intermediate_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)

    if logger:
        m.optimize(intermediate_callback)
    else:
        m.optimize()

    if m.SolCount > 0:
        solution_dict = extract_solution(instance, candidate_lines, z, f, x, directed_graph_info, od_list)
        objective_value = m.ObjVal
        if logger:
            logger.log_solution(objective_value, solution_dict)
    else:
        solution_dict = {
            "active_lines": [],
            "active_passenger_paths": []
        }
        objective_value = 0
        if logger:
            logger.log_solution(objective_value, solution_dict)

    solution_dict["objective_value"] = objective_value
    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f)

if __name__ == "__main__":
    main()