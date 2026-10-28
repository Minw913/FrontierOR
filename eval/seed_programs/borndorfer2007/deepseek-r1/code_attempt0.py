import json
import math
import time
from collections import deque
import gurobipy as gp
from gurobipy import GRB

def main(instance_path, output_path):
    # Load instance
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    # Extract global parameters
    global_params = instance["global_parameters"]
    max_line_length_edges = global_params["max_line_length_edges"]
    F = global_params["max_frequency"]
    lambda_val = global_params["lambda"]
    
    # Precompute candidate lines
    candidate_lines = []
    mode_dict = {mode["name"]: mode for mode in instance["modes"]}
    edge_id_to_operating_cost = {}
    edge_id_to_capacity = {}
    edge_id_to_travel_time = {}
    for edge in instance["network"]["edges"]:
        edge_id = edge["id"]
        edge_id_to_operating_cost[edge_id] = edge["operating_cost"]
        edge_id_to_capacity[edge_id] = edge["edge_capacity"]
        edge_id_to_travel_time[edge_id] = edge["traveling_time_seconds"]
    
    # For each mode, build subgraph and find candidate lines
    for mode in instance["modes"]:
        mode_name = mode["name"]
        terminals = mode["terminals"]
        terminals_sorted = sorted(terminals)
        
        # Build adjacency list for the mode subgraph: {node: [(neighbor, edge_id, travel_time)]}
        adj = {}
        for edge_idx in mode["edge_indices"]:
            edge = instance["network"]["edges"][edge_idx]
            u, v = edge["endpoints"]
            edge_id = edge["id"]
            travel_time = edge["traveling_time_seconds"]
            if u not in adj:
                adj[u] = []
            if v not in adj:
                adj[v] = []
            adj[u].append((v, edge_id, travel_time))
            adj[v].append((u, edge_id, travel_time))
        
        # For each terminal pair (s,t) with s < t
        k = 5  # top k shortest paths
        for i in range(len(terminals_sorted)):
            s = terminals_sorted[i]
            for j in range(i+1, len(terminals_sorted)):
                t = terminals_sorted[j]
                # BFS by depth to collect all simple paths from s to t
                paths = []  # (travel_time, node_list, edge_list)
                # Queue: (current_node, path_nodes, path_edges, current_time, depth)
                Q = [deque() for _ in range(max_line_length_edges + 1)]
                Q[0].append((s, [s], [], 0))
                for depth in range(0, max_line_length_edges + 1):
                    while Q[depth]:
                        u, path_nodes, path_edges, time_so_far = Q[depth].popleft()
                        if u == t:
                            paths.append((time_so_far, path_nodes, path_edges))
                        if depth < max_line_length_edges:
                            if u in adj:
                                for (v, edge_id, tt) in adj[u]:
                                    if v not in path_nodes:
                                        new_path_nodes = path_nodes + [v]
                                        new_path_edges = path_edges + [edge_id]
                                        new_time = time_so_far + tt
                                        Q[depth+1].append((v, new_path_nodes, new_path_edges, new_time))
                # Sort paths by travel time and take top k
                paths.sort(key=lambda x: x[0])
                for travel_time, node_list, edge_list in paths[:k]:
                    # Compute operating cost for this line: sum of edge operating costs
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
    
    # Build directed graph for passenger routing
    arc_list = []  # list of (from, to)
    arc_travel_time = {}  # (from, to) -> travel_time
    arc_to_undir_edge_id = {}  # (from, to) -> undirected edge id
    undir_edge_id_to_arcs = {}  # undir_edge_id -> list of directed arcs (from, to) in the edge
    for edge in instance["network"]["edges"]:
        u, v = edge["endpoints"]
        edge_id = edge["id"]
        travel_time = edge["traveling_time_seconds"]
        # Arc (u, v)
        arc_uv = (u, v)
        arc_list.append(arc_uv)
        arc_travel_time[arc_uv] = travel_time
        arc_to_undir_edge_id[arc_uv] = edge_id
        # Arc (v, u)
        arc_vu = (v, u)
        arc_list.append(arc_vu)
        arc_travel_time[arc_vu] = travel_time
        arc_to_undir_edge_id[arc_vu] = edge_id
        # Store both arcs for the undirected edge
        if edge_id not in undir_edge_id_to_arcs:
            undir_edge_id_to_arcs[edge_id] = []
        undir_edge_id_to_arcs[edge_id].extend([arc_uv, arc_vu])
    
    # Build OD pairs
    od_list = []  # list of (o, d)
    od_demand = {}  # (o, d) -> demand
    for od in instance["od_matrix"]:
        o = od["origin"]
        d = od["destination"]
        od_list.append((o, d))
        od_demand[(o, d)] = od["demand"]
    
    # Create MIP model
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.setParam('Seed', 0)
    m.setParam('MIPGap', 1e-4)
    m.setParam('NumericFocus', 0)
    m.setParam('Threads', 1)
    m.setParam('TimeLimit', 300)
    
    # Line variables
    z = m.addVars(len(candidate_lines), vtype=GRB.BINARY, name="z")
    f = m.addVars(len(candidate_lines), lb=0, ub=F, name="f")
    
    # Passenger flow variables: x[od, arc] for each od pair and each arc
    x = {}
    for od in od_list:
        for arc in arc_list:
            x[od, arc] = m.addVar(lb=0, name=f"x_{od}_{arc}")
    
    # Constraints
    # (1) f_l <= F * z_l
    for l_idx in range(len(candidate_lines)):
        m.addConstr(f[l_idx] <= F * z[l_idx], name=f"link_{l_idx}")
    
    # (2) For each undirected edge e: sum_{l: e in l} f_l <= edge_capacity[e]
    # First, for each undirected edge, collect candidate lines that use it
    edge_to_lines = {}
    for l_idx, line in enumerate(candidate_lines):
        for eid in line["edges"]:
            if eid not in edge_to_lines:
                edge_to_lines[eid] = []
            edge_to_lines[eid].append(l_idx)
    # Now add constraint for each undirected edge
    for eid, lines in edge_to_lines.items():
        expr = gp.quicksum(f[l_idx] for l_idx in lines)
        m.addConstr(expr <= edge_id_to_capacity[eid], name=f"edge_cap_{eid}")
    
    # (3) Flow conservation for each od and each node
    # Precompute incoming and outgoing arcs per node
    incoming = {node: [] for node in instance["network"]["nodes"]}
    outgoing = {node: [] for node in instance["network"]["nodes"]}
    for arc in arc_list:
        u, v = arc
        outgoing[u].append(arc)
        incoming[v].append(arc)
    # For each od pair and each node
    for od in od_list:
        o, d = od
        for node in instance["network"]["nodes"]:
            in_arcs = incoming[node]
            out_arcs = outgoing[node]
            expr_in = gp.quicksum(x[od, a] for a in in_arcs)
            expr_out = gp.quicksum(x[od, a] for a in out_arcs)
            if node == o:
                m.addConstr(expr_in - expr_out == -od_demand[od], name=f"flow_cons_{od}_{node}")
            elif node == d:
                m.addConstr(expr_in - expr_out == od_demand[od], name=f"flow_cons_{od}_{node}")
            else:
                m.addConstr(expr_in - expr_out == 0, name=f"flow_cons_{od}_{node}")
    
    # (4) For each directed arc a: sum_{od} x[od,a] <= cap_{e_a}
    # cap_{e_a} = sum_{l: e_a in l} (vehicle_capacity[m_l] * f_l)
    for arc in arc_list:
        eid = arc_to_undir_edge_id[arc]
        # Find candidate lines that use this undirected edge
        if eid in edge_to_lines:
            lines = edge_to_lines[eid]
            cap_expr = gp.quicksum(candidate_lines[l_idx]["vehicle_capacity"] * f[l_idx] for l_idx in lines)
        else:
            cap_expr = 0
        flow_expr = gp.quicksum(x[od, arc] for od in od_list)
        m.addConstr(flow_expr <= cap_expr, name=f"arc_cap_{arc}")
    
    # Objective
    total_line_cost = 0
    for l_idx, line in enumerate(candidate_lines):
        total_line_cost += line["fixed_cost"] * z[l_idx] + line["operating_cost"] * f[l_idx]
    
    total_passenger_time = 0
    for arc in arc_list:
        flow_expr = gp.quicksum(x[od, arc] for od in od_list)
        total_passenger_time += arc_travel_time[arc] * flow_expr
    
    obj = lambda_val * total_line_cost + (1 - lambda_val) * total_passenger_time
    m.setObjective(obj, GRB.MINIMIZE)
    
    # Solve
    m.optimize()
    
    # Prepare solution output
    solution = {
        "active_lines": [],
        "active_passenger_paths": []
    }
    
    # Active lines
    for l_idx, line in enumerate(candidate_lines):
        if z[l_idx].X > 0.5 and f[l_idx].X > 0:
            solution["active_lines"].append({
                "mode": line["mode"],
                "nodes": line["nodes"],
                "edges": line["edges"],
                "frequency": f[l_idx].X
            })
    
    # Decompose passenger flows into paths
    # First, assign directed arc ids: for each undirected edge e, assign:
    #   (u,v) -> 2 * eid, (v,u) -> 2 * eid + 1
    directed_arc_id = {}
    for edge in instance["network"]["edges"]:
        u, v = edge["endpoints"]
        eid = edge["id"]
        directed_arc_id[(u, v)] = 2 * eid
        directed_arc_id[(v, u)] = 2 * eid + 1
    
    # For each OD pair, decompose flow
    for od in od_list:
        o, d = od
        # Extract flow values for this OD
        flow_on_arc = {arc: x[od, arc].X for arc in arc_list if x[od, arc].X > 1e-5}
        # While there is flow from o to d
        while True:
            # Find a path from o to d using DFS/BFS
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
                    if arc[0] == u and flow_on_arc.get(arc, 0) > 0:
                        v = arc[1]
                        if v not in parent:
                            parent[v] = (u, arc)
                            queue.append(v)
            if not found:
                break
            # Reconstruct path backwards
            path_arcs = []
            u = d
            min_flow = float('inf')
            while u != o:
                prev, arc = parent[u]
                path_arcs.append(arc)
                min_flow = min(min_flow, flow_on_arc[arc])
                u = prev
            path_arcs.reverse()
            # Record this path
            solution["active_passenger_paths"].append({
                "origin": o,
                "destination": d,
                "arcs": [directed_arc_id[arc] for arc in path_arcs],
                "flow": min_flow
            })
            # Subtract flow from the path
            for arc in path_arcs:
                flow_on_arc[arc] -= min_flow
                if flow_on_arc[arc] < 1e-5:
                    del flow_on_arc[arc]
    
    # Save solution
    with open(output_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    import sys
    instance_path = sys.argv[1]
    output_path = sys.argv[2]
    main(instance_path, output_path)