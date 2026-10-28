import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    # Extract data
    global_params = instance['global_parameters']
    network = instance['network']
    modes = instance['modes']
    od_matrix = instance['od_matrix']
    lambda_weight = global_params['lambda']
    F_global = global_params['frequency_upper_bound_F']
    max_line_length = global_params['max_line_length_edges']
    fixed_cost_global = global_params.get('fixed_cost_per_line', 0.0)
    op_cost_global = global_params.get('operating_cost_per_edge', 0.0)

    nodes = network['nodes']
    edges = network['edges']
    num_nodes = network['num_nodes']
    num_edges = network['num_edges']

    node_id_to_idx = {node['id']: i for i, node in enumerate(nodes)}
    edge_list = edges  # list of edge dicts

    # Build undirected graph for each mode
    mode_dict = {}
    for m_idx, mode in enumerate(modes):
        mode_name = mode['name']
        mode_dict[mode_name] = mode
        mode['edge_set'] = set(mode['edge_indices'])
        mode['terminal_set'] = set(mode['terminals'])

    # Precompute all possible lines (this is the bottleneck, but for small instances feasible)
    # Since max_line_length is small and networks are modest, we enumerate all simple paths between terminals
    all_candidate_lines = []
    line_to_idx = {}
    line_counter = 0

    for mode in modes:
        mode_name = mode['name']
        terminals = mode['terminals']
        edge_indices = mode['edge_indices']
        G_mode = {i: [] for i in range(num_nodes)}
        edge_to_mode_idx = {}
        for e_idx in edge_indices:
            edge = edge_list[e_idx]
            u, v = edge['endpoints']
            u_idx = node_id_to_idx[u]
            v_idx = node_id_to_idx[v]
            G_mode[u_idx].append((v_idx, e_idx))
            G_mode[v_idx].append((u_idx, e_idx))
            edge_to_mode_idx[e_idx] = (u_idx, v_idx)

        def dfs(current, target, path_nodes, path_edges, visited):
            nonlocal line_counter
            if len(path_edges) > max_line_length:
                return
            if current == target and len(path_edges) > 0:
                # valid line
                line = {
                    'mode': mode_name,
                    'nodes': [nodes[p]['id'] for p in path_nodes],
                    'edges': path_edges[:],
                    'length': len(path_edges),
                    'operating_cost': sum(edge_list[e]['operating_cost'] for e in path_edges),
                    'edge_set': set(path_edges)
                }
                all_candidate_lines.append(line)
                line_to_idx[(mode_name, tuple(path_nodes))] = line_counter
                line_counter += 1
                return
            for neighbor, edge_id in G_mode[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                path_nodes.append(neighbor)
                path_edges.append(edge_id)
                dfs(neighbor, target, path_nodes, path_edges, visited)
                path_nodes.pop()
                path_edges.pop()
                visited.remove(neighbor)

        for t_idx in range(len(terminals)):
            for s_idx in range(t_idx + 1, len(terminals)):
                s = node_id_to_idx[terminals[t_idx]]
                t = node_id_to_idx[terminals[s_idx]]
                visited = set([s])
                dfs(s, t, [s], [], visited)

    num_lines = len(all_candidate_lines)
    print(f"Enumerated {num_lines} candidate lines", file=sys.stderr)

    # OD pairs
    od_pairs = []
    demand_dict = {}
    for od in od_matrix:
        o = od['origin']
        d = od['destination']
        dem = od['demand']
        od_pairs.append((o, d))
        demand_dict[(o, d)] = dem

    # We will use path-based formulation for passengers but to keep it tractable, we use arc-based multi-commodity flow
    # with line frequencies as variables.

    try:
        model = gp.Model("Transit_Line_Planning")
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', max(1, args.time_limit - 5))  # leave margin

        # Variables
        f = model.addVars(num_lines, lb=0, ub=F_global, name="f")  # frequency of each line
        x = model.addVars(num_lines, vtype=GRB.BINARY, name="x")   # activation

        # For passengers: arc-based flow. First build directed arcs.
        # Each undirected edge spawns two directed arcs.
        arc_to_edge = {}
        directed_arcs = []  # list of (u, v, edge_idx)
        arc_idx_map = {}
        arc_counter = 0
        for e_idx, edge in enumerate(edge_list):
            u, v = edge['endpoints']
            u_idx = node_id_to_idx[u]
            v_idx = node_id_to_idx[v]
            directed_arcs.append((u_idx, v_idx, e_idx))
            directed_arcs.append((v_idx, u_idx, e_idx))
            arc_to_edge[(u_idx, v_idx)] = e_idx
            arc_to_edge[(v_idx, u_idx)] = e_idx
            arc_idx_map[(u_idx, v_idx)] = arc_counter
            arc_counter += 1
            arc_idx_map[(v_idx, u_idx)] = arc_counter
            arc_counter += 1

        num_arcs = len(directed_arcs)

        # Multi-commodity flow: one commodity per OD pair
        num_od = len(od_pairs)
        od_idx_map = {od_pairs[i]: i for i in range(num_od)}

        flow = model.addVars(num_od, num_arcs, lb=0, name="flow")  # flow[k][a]

        # Objective components
        total_travel_time = gp.LinExpr()
        total_line_cost = gp.LinExpr()

        # Passenger travel time
        for k in range(num_od):
            o_id, d_id = od_pairs[k]
            o = node_id_to_idx[o_id]
            d = node_id_to_idx[d_id]
            for a_idx, (u, v, e_idx) in enumerate(directed_arcs):
                tt = edge_list[e_idx]['traveling_time_seconds']
                total_travel_time += flow[k, a_idx] * tt

        # Line costs
        for i, line in enumerate(all_candidate_lines):
            mode_name = line['mode']
            mode_info = mode_dict[mode_name]
            fixed_c = mode_info.get('fixed_cost_per_line', fixed_cost_global)
            op_c_base = mode_info.get('operating_cost_per_edge', op_cost_global)
            op_cost_line = line['operating_cost']  # already summed from edge data if provided, else use global
            total_line_cost += fixed_c * x[i]
            total_line_cost += op_cost_line * f[i]

        objective = lambda_weight * total_line_cost + (1.0 - lambda_weight) * total_travel_time
        model.setObjective(objective, GRB.MINIMIZE)

        # Constraints

        # Activation and frequency
        for i in range(num_lines):
            model.addConstr(f[i] <= F_global * x[i])
            model.addConstr(f[i] >= 1e-6 * x[i])  # avoid zero frequency on activated if wanted, but not strict

        # Edge frequency capacity
        for e_idx in range(num_edges):
            edge_cap = edge_list[e_idx]['edge_capacity']
            freq_on_edge = gp.LinExpr()
            for i, line in enumerate(all_candidate_lines):
                if e_idx in line['edge_set']:
                    freq_on_edge += f[i]
            model.addConstr(freq_on_edge <= edge_cap)

        # Flow conservation and demand satisfaction
        for k in range(num_od):
            o_id, d_id = od_pairs[k]
            o = node_id_to_idx[o_id]
            d = node_id_to_idx[d_id]
            dem = demand_dict[(o_id, d_id)]

            for n in range(num_nodes):
                inflow = gp.LinExpr()
                outflow = gp.LinExpr()
                for a_idx, (u, v, e_idx) in enumerate(directed_arcs):
                    if v == n:
                        inflow += flow[k, a_idx]
                    if u == n:
                        outflow += flow[k, a_idx]
                if n == o:
                    model.addConstr(outflow - inflow == dem)
                elif n == d:
                    model.addConstr(inflow - outflow == dem)
                else:
                    model.addConstr(inflow == outflow)

        # Capacity on arcs: total passenger capacity from all lines covering the edge
        for a_idx, (u, v, e_idx) in enumerate(directed_arcs):
            edge = edge_list[e_idx]
            veh_cap_sum = gp.LinExpr()
            for i, line in enumerate(all_candidate_lines):
                if e_idx in line['edge_set']:
                    mode_name = line['mode']
                    veh_cap = mode_dict[mode_name]['vehicle_capacity']
                    veh_cap_sum += veh_cap * f[i]
            total_flow_on_arc = gp.LinExpr()
            for k in range(num_od):
                total_flow_on_arc += flow[k, a_idx]
            model.addConstr(total_flow_on_arc <= veh_cap_sum)

        # Callback for incumbent logging
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
                if logger:
                    # We cannot easily extract full solution in callback efficiently, so log objective only
                    logger.log(obj)
            if where == GRB.Callback.MIP:
                runtime = time.time() - start_time
                if runtime > args.time_limit * 0.95:
                    model.terminate()

        # Optimize
        model.optimize(callback)

        # Extract solution
        if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED):
            obj_val = model.objVal if model.objVal is not None else 1e30

            # Build active lines
            active_lines = []
            for i in range(num_lines):
                freq_val = f[i].X
                if freq_val > 1e-6 and x[i].X > 0.5:
                    line = all_candidate_lines[i]
                    active_lines.append({
                        "line_index": i,
                        "mode": line['mode'],
                        "nodes": line['nodes'],
                        "edges": line['edges'],
                        "frequency": float(freq_val)
                    })

            # Build passenger paths - we need to recover path flows from arc flows.
            # This is non-trivial. For the output requirement, we must provide a path decomposition.
            # We will run a simple path extraction per OD by repeatedly finding shortest paths in residual.

            active_passenger_paths = []

            # Build flow values
            flow_values = {}
            for k in range(num_od):
                for a_idx in range(num_arcs):
                    val = flow[k, a_idx].X
                    if val > 1e-8:
                        flow_values[(k, a_idx)] = val

            for k_idx in range(num_od):
                o_id, d_id = od_pairs[k_idx]
                o = node_id_to_idx[o_id]
                d = node_id_to_idx[d_id]
                dem = demand_dict[(o_id, d_id)]
                remaining = dem

                # Build current residual graph for this OD
                while remaining > 1e-6:
                    # Use DFS or BFS to find any path with positive flow
                    visited = [False] * num_nodes
                    pred = [-1] * num_nodes
                    pred_arc = [-1] * num_nodes
                    stack = [o]
                    visited[o] = True

                    found = False
                    while stack and not found:
                        u = stack.pop()
                        for a_idx, (uu, v, e_idx) in enumerate(directed_arcs):
                            if uu == u:
                                if (k_idx, a_idx) in flow_values and flow_values[(k_idx, a_idx)] > 1e-8:
                                    if not visited[v]:
                                        visited[v] = True
                                        pred[v] = u
                                        pred_arc[v] = a_idx
                                        stack.append(v)
                                        if v == d:
                                            found = True
                                            break

                    if not found:
                        break  # should not happen if conservation holds

                    # Find min flow on path
                    path_flow = remaining
                    current = d
                    arcs_in_path = []
                    while current != o:
                        a_idx = pred_arc[current]
                        path_flow = min(path_flow, flow_values.get((k_idx, a_idx), 0.0))
                        arcs_in_path.append(a_idx)
                        current = pred[current]

                    if path_flow < 1e-6:
                        break

                    # Record path
                    arcs_in_path.reverse()  # from origin to dest
                    active_passenger_paths.append({
                        "origin": o_id,
                        "destination": d_id,
                        "arcs": arcs_in_path,
                        "flow": float(path_flow)
                    })

                    # Subtract flow
                    remaining -= path_flow
                    current = d
                    while current != o:
                        a_idx = pred_arc[current]
                        flow_values[(k_idx, a_idx)] -= path_flow
                        if flow_values[(k_idx, a_idx)] < 1e-8:
                            del flow_values[(k_idx, a_idx)]
                        current = pred[current]

            solution_dict = {
                "objective_value": float(obj_val),
                "active_lines": active_lines,
                "active_passenger_paths": active_passenger_paths
            }

            if logger:
                logger.log_solution(obj_val, solution_dict)

            # Write output
            with open(args.solution_path, 'w') as f:
                json.dump(solution_dict, f, indent=2)

        else:
            # No solution found, write empty
            solution_dict = {
                "objective_value": 1e30,
                "active_lines": [],
                "active_passenger_paths": []
            }
            with open(args.solution_path, 'w') as f:
                json.dump(solution_dict, f, indent=2)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        solution_dict = {
            "objective_value": 1e30,
            "active_lines": [],
            "active_passenger_paths": []
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution_dict, f, indent=2)


if __name__ == "__main__":
    main()