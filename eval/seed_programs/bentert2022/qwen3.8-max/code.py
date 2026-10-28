import json
import argparse
import time
import sys
from collections import defaultdict, deque


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    with open(args.instance_path) as f:
        instance = json.load(f)

    n = instance['n']
    vertices = instance['vertices']
    coords = [(v['x'], v['y']) for v in vertices]

    # Special case: n <= 1
    if n <= 1:
        solution = {
            "objective_value": 0.0,
            "selected_edges": [],
            "vertex_costs": {str(i): 0.0 for i in range(n)}
        }
        if logger:
            logger.log_solution(0.0, solution)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        return

    # Compute all edges (complete graph, squared Euclidean distance)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            w = dx * dx + dy * dy
            edges.append((i, j, w))

    # Compute MST using Kruskal's algorithm
    mst_edges = kruskal(n, edges)
    obj, v_costs = compute_objective(n, mst_edges)

    best_obj = obj
    best_tree = list(mst_edges)

    solution = make_solution(n, best_tree, best_obj, v_costs)
    if logger:
        logger.log_solution(best_obj, solution)

    # Compute lower bound
    lower_bound = sum(v.get('lower_bound', 0.0) for v in vertices)

    # Try MIP for small instances
    remaining = args.time_limit - (time.time() - start_time)
    if n <= 55 and remaining > 5 and best_obj > lower_bound + 1e-9:
        mip_result = solve_mip(n, edges, mst_edges, remaining * 0.6, logger)
        if mip_result is not None:
            mip_obj, mip_tree = mip_result
            if mip_obj < best_obj - 1e-9:
                best_obj = mip_obj
                best_tree = list(mip_tree)
                _, v_costs = compute_objective(n, best_tree)
                solution = make_solution(n, best_tree, best_obj, v_costs)
                if logger:
                    logger.log_solution(best_obj, solution)

    # Local search
    remaining = args.time_limit - (time.time() - start_time)
    if remaining > 1 and best_obj > lower_bound + 1e-9:
        ls_result = local_search(n, edges, best_tree, best_obj, remaining, logger, lower_bound)
        if ls_result is not None:
            ls_obj, ls_tree = ls_result
            if ls_obj < best_obj - 1e-9:
                best_obj = ls_obj
                best_tree = list(ls_tree)

    _, v_costs = compute_objective(n, best_tree)
    solution = make_solution(n, best_tree, best_obj, v_costs)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


def kruskal(n, edges):
    """Compute MST using Kruskal's algorithm."""
    sorted_edges = sorted(edges, key=lambda x: x[2])
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True

    mst = []
    for u, v, w in sorted_edges:
        if union(u, v):
            mst.append((u, v, w))
            if len(mst) == n - 1:
                break
    return mst


def compute_objective(n, tree_edges):
    """Compute objective value and per-vertex costs for a spanning tree."""
    vertex_max = [0.0] * n
    for u, v, w in tree_edges:
        if w > vertex_max[u]:
            vertex_max[u] = w
        if w > vertex_max[v]:
            vertex_max[v] = w
    return sum(vertex_max), vertex_max


def make_solution(n, tree_edges, obj, v_costs):
    """Create solution dictionary in the required output format."""
    return {
        "objective_value": obj,
        "selected_edges": [[u, v] for u, v, w in tree_edges],
        "vertex_costs": {str(i): v_costs[i] for i in range(n)}
    }


def solve_mip(n, edges, mst_edges, time_limit, logger):
    """Solve using MIP formulation with Gurobi for small instances."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None

    m = len(edges)

    # Build adjacency list of edge indices
    adj_edge_indices = defaultdict(list)
    for idx, (u, v, w) in enumerate(edges):
        adj_edge_indices[u].append(idx)
        adj_edge_indices[v].append(idx)

    try:
        model = gp.Model("min_power_spanning_tree")
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', max(1, int(time_limit)))
        model.setParam('OutputFlag', 0)

        # Binary variables for edge selection
        x = [model.addVar(vtype=GRB.BINARY, name=f"x_{i}") for i in range(m)]
        # Continuous variables for vertex costs
        z = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"z_{v}") for v in range(n)]

        # Objective: minimize sum of vertex costs
        model.setObjective(gp.quicksum(z), GRB.MINIMIZE)

        # Spanning tree: exactly n-1 edges
        model.addConstr(gp.quicksum(x) == n - 1)

        # Cost constraints: z_v >= w_e * x_e for all edges incident to v
        for idx, (u, v, w) in enumerate(edges):
            model.addConstr(z[u] >= w * x[idx])
            model.addConstr(z[v] >= w * x[idx])

        # Flow-based connectivity constraints
        root = 0
        f = {}
        for idx in range(m):
            f[idx, 0] = model.addVar(lb=0, ub=n - 1, name=f"f{idx}_0")
            f[idx, 1] = model.addVar(lb=0, ub=n - 1, name=f"f{idx}_1")

        for v in range(n):
            in_terms = []
            out_terms = []
            for idx in adj_edge_indices[v]:
                u_e, v_e, _ = edges[idx]
                if v_e == v:
                    # Edge u_e -> v_e=v: f[idx,0] flows into v, f[idx,1] flows out
                    in_terms.append(f[idx, 0])
                    out_terms.append(f[idx, 1])
                else:
                    # u_e == v: f[idx,0] flows out of v, f[idx,1] flows in
                    out_terms.append(f[idx, 0])
                    in_terms.append(f[idx, 1])

            if v == root:
                model.addConstr(gp.quicksum(out_terms) - gp.quicksum(in_terms) == n - 1)
            else:
                model.addConstr(gp.quicksum(in_terms) - gp.quicksum(out_terms) == 1)

        # Capacity constraints
        for idx in range(m):
            model.addConstr(f[idx, 0] <= (n - 1) * x[idx])
            model.addConstr(f[idx, 1] <= (n - 1) * x[idx])

        # Warm start with MST solution
        mst_set = set()
        for u, v, w in mst_edges:
            mst_set.add((min(u, v), max(u, v)))

        for idx, (u, v, w) in enumerate(edges):
            x[idx].Start = 1.0 if (min(u, v), max(u, v)) in mst_set else 0.0

        model.optimize()

        if model.SolCount > 0:
            selected = []
            for idx, (u, v, w) in enumerate(edges):
                if x[idx].X > 0.5:
                    selected.append((u, v, w))

            if len(selected) == n - 1:
                obj, _ = compute_objective(n, selected)
                return obj, selected

    except Exception:
        pass

    return None


def local_search(n, edges, initial_tree, initial_obj, time_limit, logger, lower_bound):
    """Improve solution using best-improvement local search with edge swaps."""
    start_time = time.time()

    current_tree = list(initial_tree)
    best_obj = initial_obj
    best_tree = list(initial_tree)

    # Sort edges by weight for efficient replacement search
    sorted_edges = sorted(edges, key=lambda x: x[2])

    # Limit the number of crossing edges to check per swap evaluation
    max_crossing_check = min(150, len(edges))

    while time.time() - start_time < time_limit - 0.5:
        if best_obj <= lower_bound + 1e-9:
            break

        # Build adjacency list for current tree
        adj = defaultdict(list)
        for u, v, w in current_tree:
            adj[u].append((v, w))
            adj[v].append((u, w))

        # Compute vertex costs and top-2 heaviest edges at each vertex
        v_cost = [0.0] * n
        top2 = [(0.0, 0.0)] * n

        for v in range(n):
            weights = [w for _, w in adj[v]]
            if len(weights) == 0:
                top2[v] = (0.0, 0.0)
            elif len(weights) == 1:
                top2[v] = (weights[0], 0.0)
            else:
                # Find top 2 without full sort
                max1 = max2 = 0.0
                for w in weights:
                    if w >= max1:
                        max2 = max1
                        max1 = w
                    elif w > max2:
                        max2 = w
                top2[v] = (max1, max2)
            v_cost[v] = top2[v][0]

        best_improvement = 1e-10
        best_swap = None

        # Try removing each tree edge that is the maximum at some endpoint
        for edge_idx in range(len(current_tree)):
            u, v, w = current_tree[edge_idx]

            # Only consider edges that are the max at at least one endpoint
            is_max_u = (abs(w - top2[u][0]) < 1e-12)
            is_max_v = (abs(w - top2[v][0]) < 1e-12)
            if not is_max_u and not is_max_v:
                continue

            # Compute saving from removing this edge
            new_cost_u = top2[u][1] if is_max_u else top2[u][0]
            new_cost_v = top2[v][1] if is_max_v else top2[v][0]
            saving = (v_cost[u] - new_cost_u) + (v_cost[v] - new_cost_v)

            if saving <= 1e-12:
                continue

            # Find component containing u after removing edge (u, v)
            removed_key = (min(u, v), max(u, v))
            visited = set()
            queue = deque([u])
            visited.add(u)
            while queue:
                node = queue.popleft()
                for neighbor, ew in adj[node]:
                    if neighbor not in visited:
                        if (min(node, neighbor), max(node, neighbor)) == removed_key:
                            continue
                        visited.add(neighbor)
                        queue.append(neighbor)

            comp_u = visited

            # Find best replacement edge crossing the cut
            best_increase = float('inf')
            best_add = None
            crossing_found = 0

            for eu, ev, ew in sorted_edges:
                if crossing_found >= max_crossing_check:
                    break

                key = (min(eu, ev), max(eu, ev))
                if key == removed_key:
                    continue

                eu_in = eu in comp_u
                ev_in = ev in comp_u

                if eu_in != ev_in:
                    crossing_found += 1

                    # Compute cost increase from adding this edge
                    cost_eu = new_cost_u if eu == u else (new_cost_v if eu == v else v_cost[eu])
                    cost_ev = new_cost_u if ev == u else (new_cost_v if ev == v else v_cost[ev])

                    increase = max(0.0, ew - cost_eu) + max(0.0, ew - cost_ev)

                    if increase < best_increase:
                        best_increase = increase
                        best_add = (eu, ev, ew)

            if best_add is not None:
                net = saving - best_increase
                if net > best_improvement:
                    best_improvement = net
                    best_swap = (edge_idx, best_add)

        if best_swap is None:
            break

        # Apply the best swap
        remove_idx, add_edge = best_swap
        current_tree[remove_idx] = add_edge

        new_obj, new_costs = compute_objective(n, current_tree)

        if new_obj < best_obj - 1e-9:
            best_obj = new_obj
            best_tree = list(current_tree)
            if logger:
                logger.log_solution(best_obj, make_solution(n, best_tree, best_obj, new_costs))

    return best_obj, best_tree


if __name__ == '__main__':
    main()