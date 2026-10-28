import argparse
import json
import time
import heapq
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, default=60)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start_time = time.time()

    with open(args.instance_path) as f:
        instance = json.load(f)

    global_params = instance['global_parameters']
    lambda_val = global_params['lambda']
    global_fixed_cost = global_params.get('fixed_cost_per_line', 0.0)
    global_operating_cost = global_params.get('operating_cost_per_edge', 0.0)
    max_len = global_params['max_line_length_edges']
    F = global_params['frequency_upper_bound_F']

    network = instance['network']
    nodes = network['nodes']
    node_ids = [n['id'] for n in nodes]
    edges = network['edges']

    edge_travel_time = {e['id']: e['traveling_time_seconds'] for e in edges}
    edge_operating_cost = {}
    for e in edges:
        op = e.get('operating_cost', 0.0)
        if op is None:
            op = 0.0
        edge_operating_cost[e['id']] = op
    edge_capacity = {e['id']: e['edge_capacity'] for e in edges}

    # Build directed arcs: arc_id = 2*edge_id (u->v), 2*edge_id+1 (v->u)
    arcs = []
    arc_to_edge = {}
    edge_to_arcs = defaultdict(list)
    for e in edges:
        eid = e['id']
        u, v = e['endpoints']
        t = e['traveling_time_seconds']
        a1 = 2 * eid
        arcs.append((a1, eid, u, v, t))
        arc_to_edge[a1] = eid
        edge_to_arcs[eid].append(a1)
        a2 = 2 * eid + 1
        arcs.append((a2, eid, v, u, t))
        arc_to_edge[a2] = eid
        edge_to_arcs[eid].append(a2)

    out_arcs = defaultdict(list)
    in_arcs = defaultdict(list)
    flow_adj = defaultdict(list)  # node -> list of (arc_id, neighbor)
    for a in arcs:
        aid, eid, u, v, t = a
        out_arcs[u].append(aid)
        in_arcs[v].append(aid)
        flow_adj[u].append((aid, v))

    modes = instance['modes']
    od_matrix = instance['od_matrix']
    od_pairs = [(d['origin'], d['destination'], float(d['demand'])) for d in od_matrix]
    W = len(od_pairs)

    # ------------------------------------------------------------------
    # Generate candidate lines (K shortest simple paths between terminals)
    # ------------------------------------------------------------------
    candidate_lines = []
    line_info_dict = {}
    line_index = 0
    seen_paths = set()

    total_terminal_pairs = 0
    for mode in modes:
        terminals = mode['terminals']
        total_terminal_pairs += len(terminals) * (len(terminals) - 1) // 2

    K_lines = 5
    if total_terminal_pairs > 0:
        max_lines = 20000
        K_lines = max(1, min(5, max_lines // total_terminal_pairs))

    def dijkstra(adj, s, t, blocked_edges=None, blocked_nodes=None):
        if blocked_edges is None:
            blocked_edges = set()
        if blocked_nodes is None:
            blocked_nodes = set()
        dist = {}
        prev = {}
        dist[s] = 0.0
        pq = [(0.0, s)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float('inf')):
                continue
            if u == t:
                break
            for v, eid, t_time, op_cost in adj.get(u, []):
                if eid in blocked_edges or v in blocked_nodes:
                    continue
                nd = d + t_time
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    prev[v] = (u, eid)
                    heapq.heappush(pq, (nd, v))
        if t not in dist:
            return None
        path_edges = []
        path_nodes = [t]
        curr = t
        while curr != s:
            p, eid = prev[curr]
            path_edges.append(eid)
            curr = p
            path_nodes.append(curr)
        path_edges.reverse()
        path_nodes.reverse()
        return (dist[t], path_edges, path_nodes)

    def yen_k_shortest(adj, s, t, max_len, K):
        first = dijkstra(adj, s, t)
        if first is None:
            return []
        if len(first[1]) > max_len:
            return []
        A = [first]
        B = []
        for k in range(1, K):
            prev_path = A[-1]
            prev_edges = prev_path[1]
            prev_nodes = prev_path[2]
            for i in range(len(prev_nodes) - 1):
                spur_node = prev_nodes[i]
                root_edges = prev_edges[:i]
                root_nodes = prev_nodes[:i + 1]
                root_time = sum(edge_travel_time[eid] for eid in root_edges)
                blocked_edges = set()
                blocked_nodes = set(root_nodes[:-1])
                for p in A:
                    p_edges = p[1]
                    p_nodes = p[2]
                    if len(p_nodes) > i and p_nodes[:i + 1] == root_nodes:
                        if i < len(p_edges):
                            blocked_edges.add(p_edges[i])
                spur_path = dijkstra(adj, spur_node, t, blocked_edges, blocked_nodes)
                if spur_path is not None:
                    total_dist = root_time + spur_path[0]
                    total_edges = root_edges + spur_path[1]
                    total_nodes = root_nodes + spur_path[2][1:]
                    if len(total_edges) <= max_len:
                        heapq.heappush(B, (total_dist, total_edges, total_nodes))
            if not B:
                break
            while B:
                cand = heapq.heappop(B)
                cand_edges = tuple(cand[1])
                found = False
                for p in A:
                    if tuple(p[1]) == cand_edges:
                        found = True
                        break
                if not found:
                    A.append(cand)
                    break
            else:
                break
        return A

    for mode in modes:
        mode_name = mode['name']
        vehicle_cap = mode['vehicle_capacity']
        fixed_cost = mode.get('fixed_cost_per_line', global_fixed_cost)
        adj = defaultdict(list)
        for ei in mode['edge_indices']:
            e = edges[ei]
            u, v = e['endpoints']
            t = e['traveling_time_seconds']
            op_cost = e.get('operating_cost', 0.0)
            adj[u].append((v, e['id'], t, op_cost))
            adj[v].append((u, e['id'], t, op_cost))
        terminals = mode['terminals']
        for i in range(len(terminals)):
            for j in range(i + 1, len(terminals)):
                s = terminals[i]
                t = terminals[j]
                if time.time() - start_time > args.time_limit * 0.5:
                    break
                paths = yen_k_shortest(adj, s, t, max_len, K_lines)
                for p in paths:
                    dist, path_edges, path_nodes = p
                    edge_tuple = tuple(path_edges)
                    if edge_tuple in seen_paths:
                        continue
                    seen_paths.add(edge_tuple)
                    op_cost_line = sum(edge_operating_cost[eid] for eid in path_edges)
                    line_info = {
                        'line_index': line_index,
                        'mode': mode_name,
                        'nodes': path_nodes,
                        'edges': path_edges,
                        'operating_cost': op_cost_line,
                        'fixed_cost': fixed_cost,
                        'vehicle_capacity': vehicle_cap,
                        'length_edges': len(path_edges)
                    }
                    candidate_lines.append(line_info)
                    line_info_dict[line_index] = line_info
                    line_index += 1
            if time.time() - start_time > args.time_limit * 0.5:
                break

    # Cap number of lines to keep model manageable
    if len(candidate_lines) > 15000:
        candidate_lines = candidate_lines[:15000]
        for new_idx, line in enumerate(candidate_lines):
            line['line_index'] = new_idx
        line_info_dict = {l['line_index']: l for l in candidate_lines}

    num_lines = len(candidate_lines)
    if num_lines == 0:
        solution = {
            'objective_value': 0.0,
            'active_lines': [],
            'active_passenger_paths': []
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        if logger:
            logger.log_solution(0.0, solution)
        return

    # Map edges to lines
    edge_to_lines = defaultdict(list)
    for l_idx, line in enumerate(candidate_lines):
        for eid in line['edges']:
            edge_to_lines[eid].append(l_idx)

    # ------------------------------------------------------------------
    # MILP model
    # ------------------------------------------------------------------
    env = gp.Env(empty=True)
    env.setParam('Seed', 0)
    env.setParam('Threads', 1)
    env.setParam('NumericFocus', 0)
    env.setParam('OutputFlag', 0)
    env.start()
    model = gp.Model(env=env)
    model.setParam('MIPGap', 1e-4)

    time_elapsed = time.time() - start_time
    time_limit_remaining = max(1, args.time_limit - time_elapsed)
    model.setParam('TimeLimit', time_limit_remaining)

    y = model.addVars(num_lines, vtype=GRB.BINARY, name="y")
    f = model.addVars(num_lines, vtype=GRB.CONTINUOUS, lb=0, ub=F, name="f")

    A = len(arcs)
    flow = model.addVars(W, A, vtype=GRB.CONTINUOUS, lb=0, name="flow")
    total_flow = model.addVars(A, vtype=GRB.CONTINUOUS, lb=0, name="total_flow")

    # Aggregate flow per arc
    for a in range(A):
        model.addConstr(total_flow[a] == sum(flow[w, a] for w in range(W)), name=f"agg_{a}")

    # Flow conservation per OD pair
    for w, (o, d, demand) in enumerate(od_pairs):
        for v in node_ids:
            out_flow = sum(flow[w, a] for a in out_arcs.get(v, []))
            in_flow = sum(flow[w, a] for a in in_arcs.get(v, []))
            rhs = 0.0
            if v == o:
                rhs = -demand
            elif v == d:
                rhs = demand
            model.addConstr(out_flow - in_flow == rhs, name=f"fc_{w}_{v}")

    # Arc capacity (passenger flow ≤ vehicle capacity * line frequency)
    for e in edges:
        eid = e['id']
        lines_on_edge = edge_to_lines.get(eid, [])
        if not lines_on_edge:
            for a in edge_to_arcs[eid]:
                model.addConstr(total_flow[a] <= 0, name=f"cap_{a}")
        else:
            cap_expr = sum(candidate_lines[l]['vehicle_capacity'] * f[l] for l in lines_on_edge)
            for a in edge_to_arcs[eid]:
                model.addConstr(total_flow[a] <= cap_expr, name=f"cap_{a}")

    # Edge capacity (sum of line frequencies ≤ edge capacity)
    for e in edges:
        eid = e['id']
        lines_on_edge = edge_to_lines.get(eid, [])
        if lines_on_edge:
            model.addConstr(sum(f[l] for l in lines_on_edge) <= e['edge_capacity'], name=f"ecap_{eid}")

    # Frequency bound
    for l in range(num_lines):
        model.addConstr(f[l] <= F * y[l], name=f"fb_{l}")

    # Objective
    line_cost = sum(candidate_lines[l]['fixed_cost'] * y[l] + candidate_lines[l]['operating_cost'] * f[l] for l in range(num_lines))
    travel_time = sum(arcs[a][4] * total_flow[a] for a in range(A))
    model.setObjective(lambda_val * line_cost + (1 - lambda_val) * travel_time, GRB.MINIMIZE)

    # Callback to log incumbents
    def incumbent_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    model.optimize(incumbent_callback)

    # ------------------------------------------------------------------
    # Extract solution
    # ------------------------------------------------------------------
    if model.SolCount > 0:
        y_vals = model.getAttr('X', y)
        f_vals = model.getAttr('X', f)
        flow_vals = model.getAttr('X', flow)

        active_lines = []
        for l in range(num_lines):
            if y_vals[l] > 0.5:
                line = candidate_lines[l]
                active_lines.append({
                    'line_index': line['line_index'],
                    'mode': line['mode'],
                    'nodes': line['nodes'],
                    'edges': line['edges'],
                    'frequency': f_vals[l]
                })

        # Flow decomposition into paths
        arc_travel_time = {a[0]: a[4] for a in arcs}
        paths = []
        for w, (o, d, demand) in enumerate(od_pairs):
            local_flow = {}
            for a in range(A):
                val = flow_vals[w, a]
                if val > 1e-6:
                    local_flow[a] = val
            remaining = demand
            while remaining > 1e-6:
                stack = [(o, [])]
                visited = {o}
                path_arcs = None
                while stack:
                    u, path = stack.pop()
                    if u == d:
                        path_arcs = path
                        break
                    for a, v in flow_adj.get(u, []):
                        if a in local_flow and local_flow[a] > 1e-6 and v not in visited:
                            visited.add(v)
                            stack.append((v, path + [a]))
                if path_arcs is None:
                    break
                min_flow = min(local_flow[a] for a in path_arcs)
                min_flow = min(min_flow, remaining)
                for a in path_arcs:
                    local_flow[a] -= min_flow
                    if local_flow[a] <= 1e-6:
                        del local_flow[a]
                remaining -= min_flow
                if min_flow > 1e-6:
                    paths.append({
                        'origin': o,
                        'destination': d,
                        'arcs': path_arcs,
                        'flow': min_flow
                    })

        # Recompute objective from extracted solution
        line_cost_val = 0.0
        for line in active_lines:
            info = line_info_dict[line['line_index']]
            line_cost_val += info['fixed_cost']
            line_cost_val += info['operating_cost'] * line['frequency']
        travel_time_val = 0.0
        for p in paths:
            path_time = sum(arc_travel_time[a] for a in p['arcs'])
            travel_time_val += p['flow'] * path_time
        obj_val = lambda_val * line_cost_val + (1 - lambda_val) * travel_time_val

        solution = {
            'objective_value': obj_val,
            'active_lines': active_lines,
            'active_passenger_paths': paths
        }
    else:
        solution = {
            'objective_value': 0.0,
            'active_lines': [],
            'active_passenger_paths': []
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    if logger:
        logger.log_solution(solution['objective_value'], solution)

if __name__ == '__main__':
    main()