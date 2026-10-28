#!/usr/bin/env python3
"""
Transit Network Design Problem Solver
Uses MIP formulation with Gurobi, with heuristic fallback for large instances.
"""

import json
import argparse
import time
import sys
from collections import defaultdict, deque
import heapq


def main():
    parser = argparse.ArgumentParser(description='Transit Network Design Problem Solver')
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    # Check for trivial case
    od_matrix = data.get('od_matrix', [])
    if not od_matrix or data.get('od_summary', {}).get('total_demand', 0) == 0:
        solution = {"objective_value": 0.0, "active_lines": [], "active_passenger_paths": []}
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        if logger:
            logger.log_solution(0.0, solution)
        return

    # Generate candidate lines
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed
    lines = generate_candidate_lines(data, remaining, start_time)

    if not lines:
        # Try heuristic with no lines - likely infeasible but output something
        solution = {"objective_value": 0.0, "active_lines": [], "active_passenger_paths": []}
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        return

    # Decide approach based on problem size
    edges = data['network']['edges']
    num_arcs = 2 * len(edges)
    num_od = len(od_matrix)

    solution = None
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed

    if remaining > 10 and num_arcs * num_od <= 300000 and len(lines) <= 3000:
        solution = solve_with_gurobi(data, lines, remaining, start_time, logger)

    if solution is None:
        solution = solve_heuristic(data, lines, start_time, logger)

    if solution is None:
        solution = {"objective_value": 0.0, "active_lines": [], "active_passenger_paths": []}

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


def generate_candidate_lines(data, time_remaining, start_time, max_total_lines=2000, max_per_pair=30):
    """Generate candidate lines for all modes."""
    lines = []
    max_len = data['global_parameters']['max_line_length_edges']
    edges = data['network']['edges']
    edge_info = {e['id']: e for e in edges}

    for mode_idx, mode_info in enumerate(data['modes']):
        mode_name = mode_info['name']
        terminals = sorted(set(mode_info['terminals']))
        edge_indices = set(mode_info['edge_indices'])

        if len(terminals) < 2:
            continue

        # Build adjacency for this mode
        adj = defaultdict(list)
        for eid in edge_indices:
            if eid < len(edges):
                edge = edges[eid]
                u, v = edge['endpoints']
                adj[u].append((v, eid))
                adj[v].append((u, eid))

        # Find paths between terminal pairs
        for i in range(len(terminals)):
            for j in range(i + 1, len(terminals)):
                if time.time() - start_time > max(1, time_remaining * 0.25):
                    break
                s, t = terminals[i], terminals[j]
                paths = find_paths_bfs(adj, s, t, max_len, max_per_pair)
                for nodes, edge_list in paths:
                    lines.append({
                        'mode': mode_name,
                        'mode_idx': mode_idx,
                        'nodes': nodes,
                        'edges': edge_list
                    })
            if time.time() - start_time > max(1, time_remaining * 0.25):
                break

        if len(lines) >= max_total_lines:
            break

    # Sort by total travel time and limit
    lines.sort(key=lambda l: sum(edge_info.get(e, {}).get('traveling_time_seconds', 0) for e in l['edges']))

    return lines[:max_total_lines]


def find_paths_bfs(adj, s, t, max_len, max_paths):
    """Find simple paths from s to t with at most max_len edges using DFS."""
    paths = []
    stack = [(s, frozenset([s]), [s], [])]

    while stack and len(paths) < max_paths:
        node, visited, path_nodes, path_edges = stack.pop()
        if len(path_edges) >= max_len:
            continue
        for neighbor, eid in adj[node]:
            if neighbor in visited:
                continue
            new_visited = visited | {neighbor}
            new_nodes = path_nodes + [neighbor]
            new_edges = path_edges + [eid]
            if neighbor == t:
                paths.append((new_nodes, new_edges))
                if len(paths) >= max_paths:
                    break
            else:
                stack.append((neighbor, new_visited, new_nodes, new_edges))

    return paths


def solve_with_gurobi(data, lines, time_remaining, start_time, logger):
    """Solve using Gurobi MIP."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None

    edges = data['network']['edges']
    nodes_data = data['network']['nodes']
    od_matrix = data['od_matrix']
    modes = data['modes']
    gp_params = data['global_parameters']
    lam = gp_params['lambda']
    F = gp_params['frequency_upper_bound_F']

    edge_info = {e['id']: e for e in edges}
    mode_info = {m['name']: m for m in modes}

    # Arc convention: for edge eid with endpoints [u, v]:
    # arc 2*eid: u -> v, arc 2*eid+1: v -> u
    num_arcs = 2 * len(edges)

    # Build arc info
    arc_from = [0] * num_arcs
    arc_to = [0] * num_arcs
    arc_edge = [0] * num_arcs
    arc_tt = [0.0] * num_arcs

    for e in edges:
        eid = e['id']
        u, v = e['endpoints']
        tt = e['traveling_time_seconds']
        arc_from[2 * eid] = u
        arc_to[2 * eid] = v
        arc_edge[2 * eid] = eid
        arc_tt[2 * eid] = tt
        arc_from[2 * eid + 1] = v
        arc_to[2 * eid + 1] = u
        arc_edge[2 * eid + 1] = eid
        arc_tt[2 * eid + 1] = tt

    # OD pairs
    od_pairs = [(od['origin'], od['destination'], od['demand']) for od in od_matrix]
    num_od = len(od_pairs)

    # Precompute which lines use each edge
    lines_per_edge = defaultdict(list)
    for l_idx, line in enumerate(lines):
        for eid in line['edges']:
            lines_per_edge[eid].append(l_idx)

    # Precompute line costs
    line_op_cost = []
    line_fixed_cost = []
    line_veh_cap = []
    for line in lines:
        op_cost = sum(edge_info[eid]['operating_cost'] for eid in line['edges'] if eid in edge_info)
        line_op_cost.append(op_cost)
        mode = mode_info.get(line['mode'], {})
        fc = mode.get('fixed_cost_per_line', gp_params.get('fixed_cost_per_line', 0))
        line_fixed_cost.append(fc)
        vc = mode.get('vehicle_capacity', 1)
        line_veh_cap.append(vc)

    # Build node adjacency for flow conservation
    all_node_ids = [n['id'] for n in nodes_data]
    node_out_arcs = defaultdict(list)
    node_in_arcs = defaultdict(list)
    for e in edges:
        eid = e['id']
        u, v = e['endpoints']
        node_out_arcs[u].append(2 * eid)
        node_in_arcs[v].append(2 * eid)
        node_out_arcs[v].append(2 * eid + 1)
        node_in_arcs[u].append(2 * eid + 1)

    num_lines = len(lines)

    # Check if problem is too large
    if num_arcs * num_od > 300000:
        return None

    # Create Gurobi model
    model = gp.Model("transit_design")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', max(1, time_remaining - 3))
    model.setParam('OutputFlag', 0)

    # Decision variables
    y = model.addVars(num_lines, vtype=GRB.BINARY, name="y")
    f = model.addVars(num_lines, lb=0, ub=F, name="f")

    # Only create flow variables for arcs that have potential capacity
    arcs_with_lines = set()
    for eid in lines_per_edge:
        arcs_with_lines.add(2 * eid)
        arcs_with_lines.add(2 * eid + 1)

    x = model.addVars(num_arcs, num_od, lb=0, name="x")

    # Objective
    obj = gp.LinExpr()
    for l in range(num_lines):
        obj += lam * line_fixed_cost[l] * y[l]
        obj += lam * line_op_cost[l] * f[l]

    for arc_id in range(num_arcs):
        tt = arc_tt[arc_id]
        if tt > 0:
            for od_idx in range(num_od):
                obj += (1 - lam) * tt * x[arc_id, od_idx]

    model.setObjective(obj, GRB.MINIMIZE)

    # Constraint 1: f[l] <= F * y[l]
    for l in range(num_lines):
        model.addConstr(f[l] <= F * y[l], name=f"fb_{l}")

    # Constraint 2: Edge capacity
    for eid in edge_info:
        if eid in lines_per_edge and lines_per_edge[eid]:
            model.addConstr(
                gp.quicksum(f[l] for l in lines_per_edge[eid]) <= edge_info[eid]['edge_capacity'],
                name=f"ec_{eid}"
            )

    # Constraint 3: Flow conservation
    for od_idx, (origin, dest, demand) in enumerate(od_pairs):
        for node in all_node_ids:
            out_arcs = node_out_arcs.get(node, [])
            in_arcs = node_in_arcs.get(node, [])
            if not out_arcs and not in_arcs:
                continue
            out_flow = gp.quicksum(x[a, od_idx] for a in out_arcs) if out_arcs else 0
            in_flow = gp.quicksum(x[a, od_idx] for a in in_arcs) if in_arcs else 0
            if node == origin:
                model.addConstr(out_flow - in_flow == demand, name=f"fc_{od_idx}_{node}")
            elif node == dest:
                model.addConstr(out_flow - in_flow == -demand, name=f"fc_{od_idx}_{node}")
            else:
                if out_arcs or in_arcs:
                    model.addConstr(out_flow - in_flow == 0, name=f"fc_{od_idx}_{node}")

    # Constraint 4: Arc capacity
    for arc_id in range(num_arcs):
        eid = arc_edge[arc_id]
        if eid in lines_per_edge and lines_per_edge[eid]:
            cap_expr = gp.quicksum(line_veh_cap[l] * f[l] for l in lines_per_edge[eid])
            model.addConstr(
                gp.quicksum(x[arc_id, od_idx] for od_idx in range(num_od)) <= cap_expr,
                name=f"ac_{arc_id}"
            )
        else:
            # No lines can use this edge - force zero flow
            for od_idx in range(num_od):
                x[arc_id, od_idx].ub = 0

    # Solve
    model.optimize()

    if model.status == GRB.INFEASIBLE:
        return None

    if model.SolCount == 0:
        return None

    # Extract solution
    line_freqs = {}
    for l in range(num_lines):
        freq_val = f[l].X
        if freq_val > 1e-6:
            line_freqs[l] = freq_val

    arc_flows = {}
    for arc_id in range(num_arcs):
        for od_idx in range(num_od):
            flow_val = x[arc_id, od_idx].X
            if flow_val > 1e-6:
                arc_flows[(arc_id, od_idx)] = flow_val

    # Build solution
    solution = build_solution_from_mip(data, lines, line_freqs, arc_flows,
                                       arc_from, arc_to, arc_tt, arc_edge,
                                       od_pairs, edge_info, mode_info, gp_params, lam)

    if logger and solution:
        logger.log_solution(solution['objective_value'], solution)

    return solution


def build_solution_from_mip(data, lines, line_freqs, arc_flows,
                            arc_from, arc_to, arc_tt, arc_edge,
                            od_pairs, edge_info, mode_info, gp_params, lam):
    """Build solution dictionary from MIP results."""

    # Active lines
    active_lines = []
    for l_idx, freq in sorted(line_freqs.items()):
        line = lines[l_idx]
        active_lines.append({
            "line_index": l_idx,
            "mode": line['mode'],
            "nodes": line['nodes'],
            "edges": line['edges'],
            "frequency": freq
        })

    # Decompose flows into paths
    active_paths = decompose_flows(arc_flows, arc_from, arc_to, od_pairs)

    # Compute objective
    total_line_cost = 0.0
    for l_idx, freq in line_freqs.items():
        line = lines[l_idx]
        mode = mode_info.get(line['mode'], {})
        fc = mode.get('fixed_cost_per_line', gp_params.get('fixed_cost_per_line', 0))
        op_cost = sum(edge_info[eid]['operating_cost'] for eid in line['edges'] if eid in edge_info)
        total_line_cost += fc + op_cost * freq

    total_travel_time = 0.0
    for (arc_id, od_idx), flow in arc_flows.items():
        total_travel_time += flow * arc_tt[arc_id]

    objective = lam * total_line_cost + (1 - lam) * total_travel_time

    return {
        "objective_value": objective,
        "active_lines": active_lines,
        "active_passenger_paths": active_paths
    }


def decompose_flows(arc_flows, arc_from, arc_to, od_pairs):
    """Decompose arc flows into paths for each OD pair."""
    paths = []

    for od_idx, (origin, dest, demand) in enumerate(od_pairs):
        # Get flows for this OD pair
        od_arc_flows = {}
        for (arc_id, oi), flow in arc_flows.items():
            if oi == od_idx and flow > 1e-8:
                od_arc_flows[arc_id] = flow

        if not od_arc_flows:
            continue

        # Build adjacency
        adj = defaultdict(list)
        for arc_id in od_arc_flows:
            u = arc_from[arc_id]
            v = arc_to[arc_id]
            adj[u].append(arc_id)

        # Decompose using DFS
        max_iterations = 5000
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            # Find path from origin to dest
            path = find_flow_path_dfs(origin, dest, adj, od_arc_flows, arc_to)
            if path is None:
                break

            path_flow = min(od_arc_flows[a] for a in path)
            if path_flow < 1e-8:
                break

            for a in path:
                od_arc_flows[a] -= path_flow
                if od_arc_flows[a] < 1e-8:
                    u = arc_from[a]
                    if a in adj[u]:
                        adj[u].remove(a)
                    del od_arc_flows[a]

            paths.append({
                "origin": origin,
                "destination": dest,
                "arcs": path,
                "flow": path_flow
            })

            if not od_arc_flows:
                break

    return paths


def find_flow_path_dfs(origin, dest, adj, arc_flows, arc_to):
    """Find a path from origin to dest using arcs with positive flow (DFS)."""
    stack = [(origin, [])]
    visited = set()

    while stack:
        node, path = stack.pop()
        if node == dest:
            return path
        if node in visited:
            continue
        visited.add(node)

        for arc_id in adj.get(node, []):
            if arc_id in arc_flows and arc_flows[arc_id] > 1e-8:
                next_node = arc_to[arc_id]
                if next_node not in visited:
                    stack.append((next_node, path + [arc_id]))

    return None


def solve_heuristic(data, lines, start_time, logger):
    """Heuristic fallback solution."""
    edges = data['network']['edges']
    od_matrix = data['od_matrix']
    modes = data['modes']
    gp_params = data['global_parameters']
    lam = gp_params['lambda']
    F = gp_params['frequency_upper_bound_F']

    if not lines or not od_matrix:
        return {"objective_value": 0.0, "active_lines": [], "active_passenger_paths": []}

    edge_info = {e['id']: e for e in edges}
    mode_info = {m['name']: m for m in modes}

    # Build full network adjacency for shortest paths
    adj = defaultdict(list)
    for e in edges:
        u, v = e['endpoints']
        tt = e['traveling_time_seconds']
        adj[u].append((v, e['id'], tt))
        adj[v].append((u, e['id'], tt))

    od_pairs = [(od['origin'], od['destination'], od['demand']) for od in od_matrix]

    # Find shortest paths for all OD pairs
    edge_flow = defaultdict(float)
    od_shortest_paths = {}

    for origin, dest, demand in od_pairs:
        path_edges = dijkstra_path(adj, origin, dest)
        if path_edges is not None:
            od_shortest_paths[(origin, dest)] = path_edges
            for eid in path_edges:
                edge_flow[eid] += demand

    # Determine which lines to activate and frequencies
    lines_per_edge = defaultdict(list)
    for l_idx, line in enumerate(lines):
        for eid in line['edges']:
            lines_per_edge[eid].append(l_idx)

    # Greedy: for each edge with flow, ensure capacity
    line_freqs = defaultdict(float)
    edge_freq_used = defaultdict(float)

    # Sort edges by flow (descending) to prioritize high-demand edges
    sorted_edges = sorted(edge_flow.items(), key=lambda x: -x[1])

    for eid, flow in sorted_edges:
        if eid not in edge_info:
            continue

        # Find mode for this edge
        edge_mode = None
        for m in modes:
            if eid in set(m['edge_indices']):
                edge_mode = m
                break
        if edge_mode is None:
            continue

        veh_cap = edge_mode['vehicle_capacity']
        if veh_cap <= 0:
            continue

        required_freq = flow / veh_cap
        current_freq_on_edge = edge_freq_used[eid]
        remaining_freq = required_freq - current_freq_on_edge

        if remaining_freq <= 1e-8:
            continue

        # Try to assign frequency to lines using this edge
        candidate_lines = lines_per_edge.get(eid, [])
        for l_idx in candidate_lines:
            if remaining_freq <= 1e-8:
                break

            current_lf = line_freqs[l_idx]
            add_freq = min(remaining_freq, F - current_lf)

            # Check edge capacity constraints for all edges of this line
            for e2 in lines[l_idx]['edges']:
                if e2 in edge_info:
                    cap = edge_info[e2]['edge_capacity']
                    used = edge_freq_used[e2]
                    add_freq = min(add_freq, cap - used)

            if add_freq > 1e-8:
                line_freqs[l_idx] += add_freq
                for e2 in lines[l_idx]['edges']:
                    edge_freq_used[e2] += add_freq
                remaining_freq -= add_freq

    # Build active lines
    active_lines = []
    for l_idx, freq in sorted(line_freqs.items()):
        if freq > 1e-6:
            line = lines[l_idx]
            active_lines.append({
                "line_index": l_idx,
                "mode": line['mode'],
                "nodes": line['nodes'],
                "edges": line['edges'],
                "frequency": freq
            })

    # Build passenger paths from shortest paths
    active_paths = []
    for (origin, dest), path_edges in od_shortest_paths.items():
        demand = next((d for o, dd, d in od_pairs if o == origin and dd == dest), 0)
        if demand <= 0:
            continue
        arc_path = edges_to_arcs(path_edges, origin, dest, edge_info)
        if arc_path is not None:
            active_paths.append({
                "origin": origin,
                "destination": dest,
                "arcs": arc_path,
                "flow": float(demand)
            })

    # Compute objective
    total_line_cost = 0.0
    for l_idx, freq in line_freqs.items():
        if freq > 1e-6:
            line = lines[l_idx]
            mode = mode_info.get(line['mode'], {})
            fc = mode.get('fixed_cost_per_line', gp_params.get('fixed_cost_per_line', 0))
            op_cost = sum(edge_info[eid]['operating_cost'] for eid in line['edges'] if eid in edge_info)
            total_line_cost += fc + op_cost * freq

    total_travel_time = sum(
        edge_info[eid]['traveling_time_seconds'] * flow
        for eid, flow in edge_flow.items() if eid in edge_info
    )

    objective = lam * total_line_cost + (1 - lam) * total_travel_time

    solution = {
        "objective_value": objective,
        "active_lines": active_lines,
        "active_passenger_paths": active_paths
    }

    if logger:
        logger.log_solution(objective, solution)

    return solution


def dijkstra_path(adj, source, target):
    """Find shortest path by travel time. Returns list of edge IDs or None."""
    if source == target:
        return []

    dist = {source: 0.0}
    prev = {}
    heap = [(0.0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if u == target:
            path = []
            node = target
            while node != source:
                edge_id, prev_node = prev[node]
                path.append(edge_id)
                node = prev_node
            path.reverse()
            return path
        if d > dist.get(u, float('inf')):
            continue
        for v, eid, tt in adj.get(u, []):
            new_dist = d + tt
            if new_dist < dist.get(v, float('inf')):
                dist[v] = new_dist
                prev[v] = (eid, u)
                heapq.heappush(heap, (new_dist, v))

    return None


def edges_to_arcs(path_edges, origin, dest, edge_info):
    """Convert edge path to arc path using convention: arc 2*eid is u->v, 2*eid+1 is v->u."""
    arcs = []
    current = origin
    for eid in path_edges:
        if eid not in edge_info:
            return None
        u, v = edge_info[eid]['endpoints']
        if u == current:
            arcs.append(2 * eid)
            current = v
        elif v == current:
            arcs.append(2 * eid + 1)
            current = u
        else:
            return None

    if current != dest:
        return None
    return arcs


if __name__ == '__main__':
    main()