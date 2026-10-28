import json
import argparse
import heapq
import random
import math
import time

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path and SolutionLogger else None

    with open(args.instance_path) as f:
        inst = json.load(f)

    num_nodes = inst['num_nodes']
    num_arcs = inst['num_arcs']
    source = inst['source_node']
    target = inst['target_node']
    arcs = inst['arcs']
    linear_costs = inst['linear_costs']
    Q = inst['quadratic_costs']

    arcs_by_from = {}
    for arc in arcs:
        arcs_by_from.setdefault(arc['from_node'], []).append((arc['to_node'], arc['id']))

    def calc_obj(path_arcs):
        cost = 0
        for a in path_arcs:
            cost += linear_costs[a]
        for idx1 in range(len(path_arcs)):
            for idx2 in range(idx1 + 1, len(path_arcs)):
                cost += 2 * Q[path_arcs[idx1]][path_arcs[idx2]]
        return cost

    def build_solution_dict(path_arcs, obj):
        selected = []
        for arc_id in path_arcs:
            selected.append({
                "id": arcs[arc_id]['id'],
                "from_node": arcs[arc_id]['from_node'],
                "to_node": arcs[arc_id]['to_node']
            })
        return {"objective_value": float(obj), "solution_arcs": selected}

    def dijkstra_arcs(u, v, weights, excluded_nodes):
        dist = {u: 0}
        prev = {u: None}
        prev_arc = {u: None}
        pq = [(0, u)]
        while pq:
            d, node = heapq.heappop(pq)
            if d > dist.get(node, float('inf')):
                continue
            if node == v:
                break
            for nxt, arc_id in arcs_by_from.get(node, []):
                if nxt != v and nxt in excluded_nodes:
                    continue
                w = weights[arc_id]
                if dist.get(nxt, float('inf')) > d + w:
                    dist[nxt] = d + w
                    prev[nxt] = node
                    prev_arc[nxt] = arc_id
                    heapq.heappush(pq, (dist[nxt], nxt))
        if v not in prev and v != u:
            return None
        if v == u:
            return []
        path_arcs = []
        curr = v
        while curr != u:
            path_arcs.append(prev_arc[curr])
            curr = prev[curr]
        return path_arcs[::-1]

    # Initial shortest path ignoring quadratic costs
    init_path = dijkstra_arcs(source, target, linear_costs[:], set())
    if init_path is None:
        init_path = []
        # Fallback to BFS if Dijkstra fails due to negative weights (though linear costs usually aren't negative)
        # but just in case
        q = [(source, [])]
        visited = {source}
        while q:
            u, p = q.pop(0)
            if u == target:
                init_path = p
                break
            for v, arc_id in arcs_by_from.get(u, []):
                if v not in visited:
                    visited.add(v)
                    q.append((v, p + [arc_id]))

    best_path = init_path
    best_obj = calc_obj(best_path)
    curr_path = init_path
    curr_obj = best_obj

    if logger:
        logger.log_solution(float(best_obj), build_solution_dict(best_path, best_obj))

    start_time = time.time()
    
    # Simulated Annealing parameters
    avg_cost = sum(abs(c) for c in linear_costs) / max(1, num_arcs)
    scale = max(1.0, avg_cost)
    T = max(1.0, abs(best_obj)) * 0.1
    alpha = 0.99

    while time.time() - start_time < args.time_limit:
        if T < 1e-3:
            T = max(1.0, abs(best_obj)) * 0.1

        if not curr_path:
            break

        i = random.randint(0, len(curr_path))
        j = random.randint(i, len(curr_path))

        if i == 0:
            u = source
        else:
            u = arcs[curr_path[i-1]]['to_node']

        if j == len(curr_path):
            v = target
        else:
            v = arcs[curr_path[j]]['from_node']

        fixed_path = curr_path[:i] + curr_path[j:]
        fixed_nodes = set()
        for a in fixed_path:
            fixed_nodes.add(arcs[a]['from_node'])
            fixed_nodes.add(arcs[a]['to_node'])

        # Calculate the linearized weight of each arc given the fixed path
        fixed_sum_Q = [0] * num_arcs
        for b in fixed_path:
            for a in range(num_arcs):
                fixed_sum_Q[a] += 2 * Q[a][b]

        weights = [0] * num_arcs
        for a in range(num_arcs):
            w = linear_costs[a] + fixed_sum_Q[a]
            # Add noise to escape local optima
            w += random.uniform(-0.5, 0.5) * scale
            weights[a] = w

        excluded = fixed_nodes - {u, v}
        sub_path = dijkstra_arcs(u, v, weights, excluded)

        if sub_path is None:
            continue

        new_path = curr_path[:i] + sub_path + curr_path[j:]

        # Ensure the new path is simple (though excluded set should handle this)
        new_nodes = [arcs[a]['from_node'] for a in new_path]
        if new_path:
            new_nodes.append(arcs[new_path[-1]]['to_node'])
        if len(set(new_nodes)) != len(new_nodes):
            continue

        new_obj = calc_obj(new_path)
        delta = new_obj - curr_obj

        if delta < 0 or random.random() < math.exp(-delta / T):
            curr_path = new_path
            curr_obj = new_obj
            if curr_obj < best_obj:
                best_obj = curr_obj
                best_path = curr_path
                if logger:
                    logger.log_solution(float(best_obj), build_solution_dict(best_path, best_obj))

        T *= alpha

    solution = build_solution_dict(best_path, best_obj)
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()