import json
import argparse
import time
import heapq
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def parse_instance(path):
    with open(path) as f:
        data = json.load(f)
    n = data['n']
    vertices = data['vertices']
    coords = {}
    for v in vertices:
        coords[v['id']] = (v['x'], v['y'])
    ids = sorted(coords.keys())

    edges = []
    weights = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[ids[i]][0] - coords[ids[j]][0]
            dy = coords[ids[i]][1] - coords[ids[j]][1]
            w = dx * dx + dy * dy
            edges.append((i, j))
            weights.append(w)

    return n, ids, edges, weights


def prim_mst(n, edges, weights):
    if n <= 1:
        return []
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        w = weights[idx]
        adj[u].append((v, w, idx))
        adj[v].append((u, w, idx))

    visited = [False] * n
    visited[0] = True
    heap = []
    for v, w, idx in adj[0]:
        heapq.heappush(heap, (w, idx, v))

    tree_edges = []
    while heap:
        w, idx, v = heapq.heappop(heap)
        if visited[v]:
            continue
        visited[v] = True
        tree_edges.append(idx)
        for nv, nw, nidx in adj[v]:
            if not visited[nv]:
                heapq.heappush(heap, (nw, nidx, nv))

    return tree_edges


def compute_costs(n, edges, weights, edge_indices):
    max_w = [0.0] * n
    for idx in edge_indices:
        u, v = edges[idx]
        w = weights[idx]
        if w > max_w[u]:
            max_w[u] = w
        if w > max_w[v]:
            max_w[v] = w
    return sum(max_w), max_w


def local_search(n, edges, weights, tree_edges, time_budget):
    start = time.time()
    tree_set = set(tree_edges)

    tree_adj = [{} for _ in range(n)]
    for idx in tree_edges:
        u, v = edges[idx]
        tree_adj[u][v] = idx
        tree_adj[v][u] = idx

    node_max = [0.0] * n
    for idx in tree_edges:
        u, v = edges[idx]
        w = weights[idx]
        if w > node_max[u]:
            node_max[u] = w
        if w > node_max[v]:
            node_max[v] = w
    current_obj = sum(node_max)

    improved = True
    while improved:
        improved = False
        if time.time() - start > time_budget:
            break

        for e_out in list(tree_set):
            if time.time() - start > time_budget:
                break

            u, v = edges[e_out]
            w_out = weights[e_out]

            # Find component of u without e_out
            comp_u = set()
            stack = [u]
            comp_u.add(u)
            while stack:
                node = stack.pop()
                for nb in tree_adj[node]:
                    eid = tree_adj[node][nb]
                    if eid != e_out and nb not in comp_u:
                        comp_u.add(nb)
                        stack.append(nb)

            # Second max for u and v after removing e_out
            new_max_u = 0.0
            for nb in tree_adj[u]:
                eid = tree_adj[u][nb]
                if eid != e_out:
                    w = weights[eid]
                    if w > new_max_u:
                        new_max_u = w
            new_max_v = 0.0
            for nb in tree_adj[v]:
                eid = tree_adj[v][nb]
                if eid != e_out:
                    w = weights[eid]
                    if w > new_max_v:
                        new_max_v = w

            gain = (node_max[u] - new_max_u) + (node_max[v] - new_max_v)
            if gain <= 1e-12:
                continue

            best_e_in = -1
            best_delta = 0.0

            for e_in in range(len(edges)):
                if e_in in tree_set:
                    continue
                eu, ev = edges[e_in]
                in_u = eu in comp_u
                in_v = ev in comp_u
                if in_u == in_v:
                    continue

                w_in = weights[e_in]
                a, b = eu, ev

                delta = -gain

                if a == u:
                    delta += max(new_max_u, w_in) - new_max_u
                elif a == v:
                    delta += max(new_max_v, w_in) - new_max_v
                else:
                    if w_in > node_max[a]:
                        delta += w_in - node_max[a]

                if b == u:
                    delta += max(new_max_u, w_in) - new_max_u
                elif b == v:
                    delta += max(new_max_v, w_in) - new_max_v
                else:
                    if w_in > node_max[b]:
                        delta += w_in - node_max[b]

                if delta < best_delta - 1e-9:
                    best_delta = delta
                    best_e_in = e_in

            if best_e_in >= 0:
                tree_set.discard(e_out)
                tree_set.add(best_e_in)

                del tree_adj[u][v]
                del tree_adj[v][u]
                a, b = edges[best_e_in]
                tree_adj[a][b] = best_e_in
                tree_adj[b][a] = best_e_in

                affected = {u, v, a, b}
                for node in affected:
                    node_max[node] = 0.0
                for idx in tree_set:
                    uu, vv = edges[idx]
                    w = weights[idx]
                    if uu in affected and w > node_max[uu]:
                        node_max[uu] = w
                    if vv in affected and w > node_max[vv]:
                        node_max[vv] = w

                current_obj = sum(node_max)
                improved = True
                break

    return list(tree_set), current_obj


def subtour_callback(model, where):
    if where == GRB.Callback.MIPSOL:
        x = model._x
        edges = model._edges
        n = model._n
        num_edges = len(edges)
        x_vals = model.cbGetSolution([x[i] for i in range(num_edges)])

        adj = [[] for _ in range(n)]
        for i in range(num_edges):
            if x_vals[i] > 0.5:
                u, v = edges[i]
                adj[u].append(v)
                adj[v].append(u)

        visited = [False] * n
        for s in range(n):
            if not visited[s]:
                comp = []
                stack = [s]
                visited[s] = True
                while stack:
                    node = stack.pop()
                    comp.append(node)
                    for nb in adj[node]:
                        if not visited[nb]:
                            visited[nb] = True
                            stack.append(nb)
                if len(comp) < n:
                    comp_set = set(comp)
                    within = [i for i in range(num_edges)
                              if edges[i][0] in comp_set and edges[i][1] in comp_set]
                    if len(within) > len(comp) - 1:
                        model.cbLazy(
                            gp.quicksum(x[i] for i in within) <= len(comp) - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, default=60)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start_time = time.time()

    n, ids, edges, weights = parse_instance(args.instance_path)

    def build_solution(edge_indices, obj_val, costs):
        selected = [[ids[edges[i][0]], ids[edges[i][1]]] for i in edge_indices]
        vertex_costs = {str(ids[v]): float(costs[v]) for v in range(n)}
        return {
            "objective_value": float(obj_val),
            "selected_edges": selected,
            "vertex_costs": vertex_costs
        }

    if n <= 1:
        sol_dict = build_solution([], 0.0, [0.0] * n)
        if logger:
            logger.log_solution(0.0, sol_dict)
        with open(args.solution_path, 'w') as f:
            json.dump(sol_dict, f)
        return

    # --- MST initial solution ---
    mst_edges = prim_mst(n, edges, weights)
    mst_obj, mst_costs = compute_costs(n, edges, weights, mst_edges)

    best_edges = mst_edges
    best_obj = mst_obj
    best_costs = mst_costs

    sol_dict = build_solution(best_edges, best_obj, best_costs)
    if logger:
        logger.log_solution(best_obj, sol_dict)

    # --- Local search improvement ---
    ls_time_budget = min(args.time_limit * 0.25, 30.0)
    if n >= 3 and ls_time_budget > 1:
        ls_edges, ls_obj = local_search(n, edges, weights, mst_edges, ls_time_budget)
        if ls_obj < best_obj - 1e-9:
            _, ls_costs = compute_costs(n, edges, weights, ls_edges)
            best_edges = ls_edges
            best_obj = ls_obj
            best_costs = ls_costs
            sol_dict = build_solution(best_edges, best_obj, best_costs)
            if logger:
                logger.log_solution(best_obj, sol_dict)

    # --- Gurobi MIP for exact/near-exact solution ---
    remaining_time = args.time_limit - (time.time() - start_time)
    mip_n_limit = 200

    if remaining_time > 3 and n >= 2 and n <= mip_n_limit:
        try:
            model = gp.Model("min_power")
            model.Params.TimeLimit = max(remaining_time - 2.0, 1.0)
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1

            num_edges = len(edges)
            x = model.addVars(num_edges, vtype=GRB.BINARY, name="x")
            p = model.addVars(n, lb=0.0, vtype=GRB.CONTINUOUS, name="p")

            model.setObjective(gp.quicksum(p[v] for v in range(n)), GRB.MINIMIZE)
            model.addConstr(gp.quicksum(x[i] for i in range(num_edges)) == n - 1)

            for i in range(num_edges):
                u, v = edges[i]
                w = weights[i]
                model.addConstr(p[u] >= w * x[i])
                model.addConstr(p[v] >= w * x[i])

            # Warm start
            best_edge_set = set(best_edges)
            for i in range(num_edges):
                x[i].Start = 1.0 if i in best_edge_set else 0.0
            for v in range(n):
                p[v].Start = best_costs[v]

            model._x = x
            model._edges = edges
            model._n = n
            model.Params.LazyConstraints = 1
            model.optimize(subtour_callback)

            if model.SolCount > 0:
                mip_obj = model.ObjVal
                if mip_obj < best_obj - 1e-6:
                    mip_edges = [i for i in range(num_edges) if x[i].X > 0.5]
                    _, mip_costs = compute_costs(n, edges, weights, mip_edges)
                    best_edges = mip_edges
                    best_obj = sum(mip_costs)
                    best_costs = mip_costs
                    sol_dict = build_solution(best_edges, best_obj, best_costs)
                    if logger:
                        logger.log_solution(best_obj, sol_dict)
        except Exception:
            import traceback
            traceback.print_exc()

    # --- Write final solution ---
    sol_dict = build_solution(best_edges, best_obj, best_costs)
    with open(args.solution_path, 'w') as f:
        json.dump(sol_dict, f)


if __name__ == "__main__":
    main()