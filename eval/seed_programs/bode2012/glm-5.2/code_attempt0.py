import argparse
import json
import time
import heapq
import random
import math
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB

INF = 10**18


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, default=60)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    with open(args.instance_path) as f:
        instance = json.load(f)

    num_nodes = instance['graph']['num_nodes']
    depot = instance['depot']
    num_vehicles = instance['fleet']['num_vehicles']
    capacity = instance['fleet']['vehicle_capacity']
    edges = instance['edges']

    # Map node IDs to contiguous indices if necessary
    node_ids = set()
    for e in edges:
        node_ids.update(e['endpoints'])
    node_ids.add(depot)
    node_id_list = sorted(node_ids)
    if node_id_list == list(range(len(node_id_list))):
        node_to_idx = {n: n for n in node_id_list}
        N = num_nodes
        depot_idx = depot
    else:
        node_to_idx = {n: i for i, n in enumerate(node_id_list)}
        N = len(node_id_list)
        depot_idx = node_to_idx[depot]

    # Build adjacency list
    adj = [[] for _ in range(N)]
    for e in edges:
        u, v = e['endpoints']
        ui = node_to_idx[u]
        vi = node_to_idx[v]
        cost = e['cost']
        eid = e['edge_id']
        adj[ui].append((vi, cost, eid))
        adj[vi].append((ui, cost, eid))

    # All-pairs shortest paths (Dijkstra)
    dist = [[INF] * N for _ in range(N)]
    prev_node = [[-1] * N for _ in range(N)]
    prev_edge = [[-1] * N for _ in range(N)]

    for src in range(N):
        dist[src][src] = 0
        pq = [(0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[src][u]:
                continue
            for v, w, eid in adj[u]:
                nd = d + w
                if nd < dist[src][v]:
                    dist[src][v] = nd
                    prev_node[src][v] = u
                    prev_edge[src][v] = eid
                    heapq.heappush(pq, (nd, v))

    def get_path_edges(src, dst):
        if src == dst:
            return []
        edges_path = []
        cur = dst
        while cur != src:
            p = prev_node[src][cur]
            if p == -1:
                return None
            eid = prev_edge[src][cur]
            edges_path.append(eid)
            cur = p
        return edges_path

    # Required edges
    required_edges = []
    for e in edges:
        if e['is_required']:
            ui = node_to_idx[e['endpoints'][0]]
            vi = node_to_idx[e['endpoints'][1]]
            required_edges.append({
                'eid': e['edge_id'],
                'u': ui,
                'v': vi,
                'demand': e['demand'],
                'service_cost': e['service_cost'],
            })
    num_required = len(required_edges)
    total_service_cost = sum(re['service_cost'] for re in required_edges)

    # Evaluate a sequence of required edge indices: returns (cost, directions, deadhead_segments)
    def evaluate_sequence(seq):
        if not seq:
            return 0, [], []
        k = len(seq)
        # dp[i][0]: cost ending at v of edge i (serviced u->v)
        # dp[i][1]: cost ending at u of edge i (serviced v->u)
        dp = [[None, None] for _ in range(k)]
        e0 = required_edges[seq[0]]
        a0, b0 = e0['u'], e0['v']
        dp[0][0] = (dist[depot_idx][a0] + e0['service_cost'], a0, b0)
        dp[0][1] = (dist[depot_idx][b0] + e0['service_cost'], b0, a0)

        for i in range(1, k):
            ei = required_edges[seq[i]]
            ai, bi = ei['u'], ei['v']
            # Option 0: ai -> bi
            c0 = dp[i-1][0][0] + dist[dp[i-1][0][2]][ai] + ei['service_cost']
            c1 = dp[i-1][1][0] + dist[dp[i-1][1][2]][ai] + ei['service_cost']
            if c0 <= c1:
                dp[i][0] = (c0, ai, bi)
            else:
                dp[i][0] = (c1, ai, bi)
            # Option 1: bi -> ai
            c0 = dp[i-1][0][0] + dist[dp[i-1][0][2]][bi] + ei['service_cost']
            c1 = dp[i-1][1][0] + dist[dp[i-1][1][2]][bi] + ei['service_cost']
            if c0 <= c1:
                dp[i][1] = (c0, bi, ai)
            else:
                dp[i][1] = (c1, bi, ai)

        final0 = dp[k-1][0][0] + dist[dp[k-1][0][2]][depot_idx]
        final1 = dp[k-1][1][0] + dist[dp[k-1][1][2]][depot_idx]
        if final0 <= final1:
            final_cost = final0
            states = [0] * k
            states[-1] = 0
        else:
            final_cost = final1
            states = [0] * k
            states[-1] = 1

        for i in range(k-1, 0, -1):
            cur_state = states[i]
            ei = required_edges[seq[i]]
            target_start = ei['u'] if cur_state == 0 else ei['v']
            c0 = dp[i-1][0][0] + dist[dp[i-1][0][2]][target_start] + ei['service_cost']
            states[i-1] = 0 if c0 == dp[i][cur_state][0] else 1

        directions = []
        deadhead_segments = []
        e0 = required_edges[seq[0]]
        if states[0] == 0:
            s0, t0 = e0['u'], e0['v']
        else:
            s0, t0 = e0['v'], e0['u']
        directions.append((s0, t0))
        deadhead_segments.append((depot_idx, s0))
        prev_end = t0
        for i in range(1, k):
            ei = required_edges[seq[i]]
            if states[i] == 0:
                si, ti = ei['u'], ei['v']
            else:
                si, ti = ei['v'], ei['u']
            directions.append((si, ti))
            deadhead_segments.append((prev_end, si))
            prev_end = ti
        deadhead_segments.append((prev_end, depot_idx))
        return final_cost, directions, deadhead_segments

    def get_deadhead_edges(deadhead_segments):
        counts = defaultdict(int)
        for s, t in deadhead_segments:
            if s == t:
                continue
            path = get_path_edges(s, t)
            if path is None:
                continue
            for eid in path:
                counts[eid] += 1
        return counts

    # 2-opt improvement on sequence (reversal of segments)
    def improve_sequence(seq):
        best_cost, _, _ = evaluate_sequence(seq)
        best_seq = seq[:]
        improved = True
        while improved:
            improved = False
            for i in range(len(best_seq)):
                for j in range(i+1, len(best_seq)):
                    new_seq = best_seq[:i] + best_seq[i:j+1][::-1] + best_seq[j+1:]
                    cost, _, _ = evaluate_sequence(new_seq)
                    if cost < best_cost:
                        best_cost = cost
                        best_seq = new_seq
                        improved = True
        return best_seq, best_cost

    # Insertion-based ordering for a set of edges
    def order_by_insertion(edge_set):
        if not edge_set:
            return []
        seq = [edge_set[0]]
        for i in range(1, len(edge_set)):
            best_seq = None
            best_cost = INF
            for pos in range(len(seq)+1):
                new_seq = seq[:pos] + [edge_set[i]] + seq[pos:]
                cost, _, _ = evaluate_sequence(new_seq)
                if cost < best_cost:
                    best_cost = cost
                    best_seq = new_seq
            seq = best_seq
        return seq

    # Generate routes pool
    routes = []
    route_dict = {}

    def add_route(seq):
        if not seq:
            return
        key = frozenset(seq)
        cost, dirs, segs = evaluate_sequence(seq)
        if key not in route_dict or cost < route_dict[key]['cost']:
            route_dict[key] = {
                'cost': cost,
                'seq': seq,
                'directions': dirs,
                'segments': segs,
                'edges': [required_edges[i]['eid'] for i in seq]
            }

    # Single-edge routes
    for i in range(num_required):
        add_route([i])

    # Feasible partition (first-fit decreasing)
    edge_indices = list(range(num_required))
    edge_indices.sort(key=lambda i: required_edges[i]['demand'], reverse=True)
    vehicle_loads = [0] * num_vehicles
    vehicle_sets = [[] for _ in range(num_vehicles)]
    for i in edge_indices:
        d = required_edges[i]['demand']
        for v in range(num_vehicles):
            if vehicle_loads[v] + d <= capacity:
                vehicle_sets[v].append(i)
                vehicle_loads[v] += d
                break
    for v in range(num_vehicles):
        if vehicle_sets[v]:
            seq = order_by_insertion(vehicle_sets[v])
            seq, _ = improve_sequence(seq)
            add_route(seq)

    # Random route generation
    random.seed(0)
    route_gen_limit = start_time + args.time_limit * 0.5
    max_routes = 8000
    while time.time() < route_gen_limit and len(route_dict) < max_routes:
        if num_required == 0:
            break
        # Random subset by nearest neighbor with random start
        start_i = random.randrange(num_required)
        current_set = [start_i]
        load = required_edges[start_i]['demand']
        remaining = set(range(num_required)) - {start_i}
        # Random walk: add nearest or random until capacity
        while remaining and load < capacity:
            candidates = [i for i in remaining if required_edges[i]['demand'] <= capacity - load]
            if not candidates:
                break
            if random.random() < 0.7:
                # nearest to the last edge in current_set? Use set's endpoints
                last_idx = current_set[-1]
                lu = required_edges[last_idx]['u']
                lv = required_edges[last_idx]['v']
                best_i = min(candidates, key=lambda i: min(
                    dist[lu][required_edges[i]['u']], dist[lu][required_edges[i]['v']],
                    dist[lv][required_edges[i]['u']], dist[lv][required_edges[i]['v']]
                ))
            else:
                best_i = random.choice(candidates)
            current_set.append(best_i)
            load += required_edges[best_i]['demand']
            remaining.remove(best_i)
        seq = order_by_insertion(current_set)
        seq, _ = improve_sequence(seq)
        add_route(seq)

    # Also generate some routes from pairs and triples if small
    if num_required <= 40:
        for i in range(num_required):
            for j in range(i+1, num_required):
                if required_edges[i]['demand'] + required_edges[j]['demand'] <= capacity:
                    add_route([i, j])
        for i in range(num_required):
            for j in range(i+1, num_required):
                for k in range(j+1, num_required):
                    if required_edges[i]['demand'] + required_edges[j]['demand'] + required_edges[k]['demand'] <= capacity:
                        seq = [i, j, k]
                        seq, _ = improve_sequence(seq)
                        add_route(seq)

    routes = list(route_dict.values())

    # If no required edges, output trivial
    if num_required == 0:
        solution = {'objective_value': 0.0, 'routes': []}
        if logger:
            logger.log_solution(0.0, solution)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        return

    # Build and solve set-partitioning MIP
    env = gp.Env(empty=True)
    env.setParam('Seed', 0)
    env.setParam('MIPGap', 1e-4)
    env.setParam('NumericFocus', 0)
    env.setParam('Threads', 1)
    env.start()

    m = gp.Model(env=env)
    remaining_time = max(1, args.time_limit - (time.time() - start_time) - 1)
    m.Params.TimeLimit = remaining_time
    m.Params.OutputFlag = 0

    num_routes = len(routes)
    x = m.addVars(num_routes, vtype=GRB.BINARY, name='x')
    m.setObjective(gp.quicksum(routes[r]['cost'] * x[r] for r in range(num_routes)), GRB.MINIMIZE)

    edge_to_routes = [[] for _ in range(num_required)]
    for r, route in enumerate(routes):
        for idx in route['seq']:
            edge_to_routes[idx].append(r)

    for e in range(num_required):
        m.addConstr(gp.quicksum(x[r] for r in edge_to_routes[e]) == 1, name=f'cover_{e}')
    m.addConstr(gp.quicksum(x[r] for r in range(num_routes)) <= num_vehicles, name='fleet')

    m.optimize()

    if m.SolCount == 0:
        # Fallback: use the feasible partition routes (which are in the pool)
        selected = []
        for v in range(num_vehicles):
            if vehicle_sets[v]:
                key = frozenset(order_by_insertion(vehicle_sets[v]))
                # find in route_dict
                for r in routes:
                    if frozenset(r['seq']) == key:
                        selected.append(r)
                        break
        if not selected:
            # just take any covering set: single edges
            selected = [route_dict[frozenset([i])] for i in range(num_required)]
        total_cost = sum(r['cost'] for r in selected)
    else:
        selected = []
        for r in range(num_routes):
            if x[r].X > 0.5:
                selected.append(routes[r])
        total_cost = m.ObjVal

    # Build output
    out_routes = []
    for v_idx, r in enumerate(selected):
        deadhead_counts = get_deadhead_edges(r['segments'])
        deadhead_list = [{'edge_id': eid, 'times': t} for eid, t in sorted(deadhead_counts.items()) if t > 0]
        out_routes.append({
            'vehicle': v_idx,
            'serviced_edges': r['edges'],
            'deadheaded_edges': deadhead_list
        })

    solution = {
        'objective_value': float(total_cost),
        'routes': out_routes
    }

    if logger:
        logger.log_solution(total_cost, solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()