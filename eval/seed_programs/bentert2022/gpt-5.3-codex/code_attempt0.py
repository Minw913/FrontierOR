import argparse
import json
import math
import random
import time
from collections import deque, defaultdict

from solution_logger import SolutionLogger

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except Exception:
    GUROBI_AVAILABLE = False


# -------------------------- Utility / Parsing -------------------------- #

class DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1
        return True


def norm_pair(a, b):
    return (a, b) if a < b else (b, a)


def parse_instance(path):
    with open(path, "r") as f:
        inst = json.load(f)

    vertices = inst.get("vertices", [])
    n = inst.get("n", len(vertices))
    if not vertices:
        # fallback if vertices maybe encoded differently (unlikely)
        vertices = [{"id": i, "x": 0.0, "y": 0.0} for i in range(n)]

    ids = [v["id"] for v in vertices]
    id_to_idx = {vid: i for i, vid in enumerate(ids)}
    coords = [(float(v.get("x", 0.0)), float(v.get("y", 0.0))) for v in vertices]

    # Read edges if present under common names
    raw_edges = None
    for k in ["edges", "links", "arcs"]:
        if k in inst and isinstance(inst[k], list):
            raw_edges = inst[k]
            break

    edges = []
    if raw_edges is not None and len(raw_edges) > 0:
        for e in raw_edges:
            if isinstance(e, dict):
                u = e.get("u", e.get("source", e.get("from", e.get("i", e.get("node1")))))
                v = e.get("v", e.get("target", e.get("to", e.get("j", e.get("node2")))))
                w = e.get("weight", e.get("w", e.get("cost", e.get("power", e.get("distance")))))
                if u is None or v is None:
                    continue
                if w is None:
                    iu, iv = id_to_idx[u], id_to_idx[v]
                    dx = coords[iu][0] - coords[iv][0]
                    dy = coords[iu][1] - coords[iv][1]
                    w = dx * dx + dy * dy
                edges.append((id_to_idx[u], id_to_idx[v], float(w)))
            elif isinstance(e, (list, tuple)) and len(e) >= 3:
                u, v, w = e[0], e[1], e[2]
                edges.append((id_to_idx[u], id_to_idx[v], float(w)))
    else:
        # Generate complete graph from coordinates
        for i in range(n):
            xi, yi = coords[i]
            for j in range(i + 1, n):
                xj, yj = coords[j]
                dx, dy = xi - xj, yi - yj
                w = dx * dx + dy * dy
                edges.append((i, j, float(w)))

    # Remove self-loops / duplicate pairs (keep minimum weight)
    best_w = {}
    for u, v, w in edges:
        if u == v:
            continue
        a, b = norm_pair(u, v)
        prev = best_w.get((a, b))
        if prev is None or w < prev:
            best_w[(a, b)] = w

    edges = [(a, b, w) for (a, b), w in best_w.items()]
    m = len(edges)

    return {
        "n": n,
        "m": m,
        "ids": ids,
        "coords": coords,
        "edges": edges,
        "id_to_idx": id_to_idx
    }


# -------------------------- Core graph helpers -------------------------- #

def build_graph_adjacency(n, edges):
    g = [[] for _ in range(n)]
    for idx, (u, v, w) in enumerate(edges):
        g[u].append((v, w, idx))
        g[v].append((u, w, idx))
    return g


def compute_tree_obj_and_costs(n, tree_pairs, weight_lookup):
    adj = [dict() for _ in range(n)]
    for a, b in tree_pairs:
        w = weight_lookup[(a, b)]
        adj[a][b] = w
        adj[b][a] = w

    costs = [0.0] * n
    for i in range(n):
        if adj[i]:
            costs[i] = max(adj[i].values())
        else:
            costs[i] = 0.0
    return sum(costs), costs, adj


def kruskal_tree(n, edges, alpha=0.0, rng=None):
    if rng is None:
        rng = random.Random(0)
    dsu = DSU(n)
    idxs = list(range(len(edges)))
    if alpha <= 0:
        idxs.sort(key=lambda i: edges[i][2])
    else:
        idxs.sort(key=lambda i: edges[i][2] * (1.0 + alpha * rng.random()))

    tree = []
    for i in idxs:
        u, v, _ = edges[i]
        if dsu.union(u, v):
            tree.append(norm_pair(u, v))
            if len(tree) == n - 1:
                break
    return tree if len(tree) == n - 1 else None


def bfs_spanning_tree(n, graph_adj):
    parent = [-1] * n
    parent[0] = 0
    q = deque([0])
    tree = []
    while q:
        u = q.popleft()
        for v, _, _ in graph_adj[u]:
            if parent[v] == -1:
                parent[v] = u
                tree.append(norm_pair(u, v))
                q.append(v)
    return tree if len(tree) == n - 1 else None


def best_star_tree_if_possible(n, weight_lookup):
    best_obj = float("inf")
    best_tree = None
    nodes = list(range(n))
    for c in nodes:
        tree = []
        feasible = True
        for v in nodes:
            if v == c:
                continue
            e = norm_pair(c, v)
            if e not in weight_lookup:
                feasible = False
                break
            tree.append(e)
        if not feasible:
            continue
        # fast objective for star
        center_max = 0.0
        leaves_sum = 0.0
        for v in nodes:
            if v == c:
                continue
            w = weight_lookup[norm_pair(c, v)]
            leaves_sum += w
            if w > center_max:
                center_max = w
        obj = leaves_sum + center_max
        if obj < best_obj:
            best_obj = obj
            best_tree = tree
    return best_tree


# -------------------------- Local search -------------------------- #

def find_path_edges_in_tree(u, v, tree_adj):
    n = len(tree_adj)
    parent = [-1] * n
    pedge = [None] * n
    q = deque([u])
    parent[u] = u
    found = False
    while q and not found:
        x = q.popleft()
        for nb, w in tree_adj[x].items():
            if parent[nb] == -1:
                parent[nb] = x
                pedge[nb] = (x, nb, w)
                if nb == v:
                    found = True
                    break
                q.append(nb)
    if parent[v] == -1:
        return []
    path = []
    cur = v
    while cur != u:
        path.append(pedge[cur])
        cur = parent[cur]
    return path


def eval_swap_delta(add_e, rem_e, tree_adj, current_costs):
    u, v, w_add = add_e
    a, b, w_rem = rem_e
    touched = {u, v, a, b}
    delta = 0.0
    for t in touched:
        vals = list(tree_adj[t].values())
        if t == a or t == b:
            # remove one instance of w_rem
            removed = False
            for i, ww in enumerate(vals):
                if not removed and abs(ww - w_rem) <= 1e-12:
                    vals.pop(i)
                    removed = True
                    break
        if t == u or t == v:
            vals.append(w_add)
        new_c = max(vals) if vals else 0.0
        delta += new_c - current_costs[t]
    return delta


def apply_swap(add_e, rem_e, tree_adj, tree_set, costs):
    u, v, w_add = add_e
    a, b, _ = rem_e
    er = norm_pair(a, b)
    ea = norm_pair(u, v)

    # remove
    if b in tree_adj[a]:
        del tree_adj[a][b]
    if a in tree_adj[b]:
        del tree_adj[b][a]
    tree_set.remove(er)

    # add
    tree_adj[u][v] = w_add
    tree_adj[v][u] = w_add
    tree_set.add(ea)

    # recompute touched costs exactly
    touched = {u, v, a, b}
    for t in touched:
        costs[t] = max(tree_adj[t].values()) if tree_adj[t] else 0.0


def local_search_random(
    n, edges, weight_lookup, initial_tree_pairs, time_limit_end, rng, logger, best_global_obj_ref
):
    obj, costs, tree_adj = compute_tree_obj_and_costs(n, initial_tree_pairs, weight_lookup)
    tree_set = set(initial_tree_pairs)

    # Try random improving edge exchanges
    no_improve = 0
    max_no_improve = max(500, min(5000, 20 * n))

    while time.time() < time_limit_end and no_improve < max_no_improve:
        # sample non-tree edge
        sampled = None
        for _ in range(20):
            u, v, w = edges[rng.randrange(len(edges))]
            e = norm_pair(u, v)
            if e not in tree_set:
                sampled = (u, v, w)
                break
        if sampled is None:
            no_improve += 1
            continue

        u, v, w_add = sampled
        path = find_path_edges_in_tree(u, v, tree_adj)
        if not path:
            no_improve += 1
            continue

        best_delta = 0.0
        best_rem = None
        for rem in path:
            d = eval_swap_delta(sampled, rem, tree_adj, costs)
            if d < best_delta - 1e-12:
                best_delta = d
                best_rem = rem

        if best_rem is not None:
            apply_swap(sampled, best_rem, tree_adj, tree_set, costs)
            obj += best_delta
            no_improve = 0
            if obj < best_global_obj_ref[0] - 1e-9:
                best_global_obj_ref[0] = obj
                if logger:
                    logger.log(obj)
        else:
            no_improve += 1

    return obj, list(tree_set), costs


# -------------------------- MILP (optional exact improvement) -------------------------- #

def try_mip_improve(n, edges, warm_tree_set, warm_costs, remaining_time, logger, current_best):
    if not GUROBI_AVAILABLE:
        return None

    m = len(edges)
    # avoid huge model
    if n > 70 or m > 3000 or remaining_time < 8:
        return None

    try:
        model = gp.Model("min_power_spanning_tree")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = max(1, int(remaining_time))
        model.Params.Threads = 1

        x = {}
        for k, (u, v, w) in enumerate(edges):
            x[k] = model.addVar(vtype=GRB.BINARY, name=f"x_{u}_{v}")

        p = {i: model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"p_{i}") for i in range(n)}

        # Flow vars
        root = 0
        f = {}
        cap = n - 1
        for k, (u, v, w) in enumerate(edges):
            f[(k, 0)] = model.addVar(lb=0.0, ub=cap, vtype=GRB.CONTINUOUS, name=f"f_{u}_{v}")
            f[(k, 1)] = model.addVar(lb=0.0, ub=cap, vtype=GRB.CONTINUOUS, name=f"f_{v}_{u}")

        model.update()

        # objective
        model.setObjective(gp.quicksum(p[i] for i in range(n)), GRB.MINIMIZE)

        # p_i >= w_e x_e for incident edges
        incident = [[] for _ in range(n)]
        for k, (u, v, w) in enumerate(edges):
            incident[u].append((k, w))
            incident[v].append((k, w))
        for i in range(n):
            for k, w in incident[i]:
                model.addConstr(p[i] >= w * x[k])

        # tree edge count
        model.addConstr(gp.quicksum(x[k] for k in range(m)) == n - 1)

        # flow capacity linking
        for k, (u, v, w) in enumerate(edges):
            model.addConstr(f[(k, 0)] <= cap * x[k])
            model.addConstr(f[(k, 1)] <= cap * x[k])

        # flow conservation
        # build directed incidence from undirected edge vars
        out_expr = [gp.LinExpr() for _ in range(n)]
        in_expr = [gp.LinExpr() for _ in range(n)]
        for k, (u, v, _) in enumerate(edges):
            # direction 0: u->v
            out_expr[u] += f[(k, 0)]
            in_expr[v] += f[(k, 0)]
            # direction 1: v->u
            out_expr[v] += f[(k, 1)]
            in_expr[u] += f[(k, 1)]

        for i in range(n):
            if i == root:
                model.addConstr(out_expr[i] - in_expr[i] == n - 1)
            else:
                model.addConstr(in_expr[i] - out_expr[i] == 1)

        # warm start
        warm_set = set(warm_tree_set)
        for k, (u, v, _) in enumerate(edges):
            x[k].Start = 1.0 if norm_pair(u, v) in warm_set else 0.0
        if warm_costs is not None and len(warm_costs) == n:
            for i in range(n):
                p[i].Start = warm_costs[i]

        # incumbent logging callback
        cb_best = [current_best]

        def cb(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj < cb_best[0] - 1e-9:
                    cb_best[0] = obj
                    if logger:
                        logger.log(float(obj))

        model.optimize(cb)

        if model.SolCount == 0:
            return None

        chosen = []
        for k, (u, v, w) in enumerate(edges):
            if x[k].X > 0.5:
                chosen.append(norm_pair(u, v))

        if len(chosen) != n - 1:
            return None

        costs = [p[i].X for i in range(n)]
        obj = sum(costs)
        return obj, chosen, costs

    except Exception:
        return None


# -------------------------- Main solve -------------------------- #

def solve(instance, time_limit, logger=None):
    start = time.time()
    deadline = start + time_limit - 0.02

    n = instance["n"]
    edges = instance["edges"]

    if n <= 1:
        return 0.0, [], [0.0] * n

    if not edges:
        # should not happen for connected instance
        return 0.0, [], [0.0] * n

    # lookup structures
    weight_lookup = {}
    for u, v, w in edges:
        weight_lookup[norm_pair(u, v)] = w

    graph_adj = build_graph_adjacency(n, edges)
    rng = random.Random(42)

    best_obj = float("inf")
    best_tree = None
    best_costs = None

    def register_solution(tree_pairs, obj=None, costs=None):
        nonlocal best_obj, best_tree, best_costs
        if tree_pairs is None or len(tree_pairs) != n - 1:
            return
        if obj is None or costs is None:
            obj2, costs2, _ = compute_tree_obj_and_costs(n, tree_pairs, weight_lookup)
            obj, costs = obj2, costs2
        if obj < best_obj - 1e-9:
            best_obj = obj
            best_tree = list(tree_pairs)
            best_costs = list(costs)
            if logger:
                logger.log(float(best_obj))

    # 1) MST baseline
    mst = kruskal_tree(n, edges, alpha=0.0, rng=rng)
    if mst is None:
        mst = bfs_spanning_tree(n, graph_adj)
    register_solution(mst)

    # 2) Star heuristic (if possible)
    if time.time() < deadline:
        st = best_star_tree_if_possible(n, weight_lookup)
        register_solution(st)

    # 3) A few randomized Kruskal
    trials = 2 if n > 120 else 4
    for _ in range(trials):
        if time.time() >= deadline:
            break
        t = kruskal_tree(n, edges, alpha=0.3, rng=rng)
        register_solution(t)

    if best_tree is None:
        # last resort
        t = bfs_spanning_tree(n, graph_adj)
        register_solution(t)

    # 4) Local search from current best
    if best_tree is not None and time.time() < deadline:
        best_ref = [best_obj]
        ls_obj, ls_tree, ls_costs = local_search_random(
            n=n,
            edges=edges,
            weight_lookup=weight_lookup,
            initial_tree_pairs=best_tree,
            time_limit_end=deadline - 0.5 if (deadline - time.time()) > 1.0 else deadline,
            rng=rng,
            logger=logger,
            best_global_obj_ref=best_ref
        )
        register_solution(ls_tree, ls_obj, ls_costs)

    # 5) MILP polish for small/medium instances
    remaining = max(0.0, deadline - time.time())
    if best_tree is not None and remaining > 6:
        mip_res = try_mip_improve(
            n=n,
            edges=edges,
            warm_tree_set=best_tree,
            warm_costs=best_costs,
            remaining_time=remaining,
            logger=logger,
            current_best=best_obj
        )
        if mip_res is not None:
            obj, tree, costs = mip_res
            register_solution(tree, obj, costs)

    # safety recompute final costs/objective exactly from selected tree
    final_obj, final_costs, _ = compute_tree_obj_and_costs(n, best_tree, weight_lookup)
    return final_obj, best_tree, final_costs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = parse_instance(args.instance_path)
    obj, tree_pairs, costs = solve(instance, args.time_limit, logger=logger)

    ids = instance["ids"]
    selected_edges = [[int(ids[a]), int(ids[b])] for a, b in tree_pairs]
    vertex_costs = {str(ids[i]): float(costs[i]) for i in range(len(ids))}

    sol = {
        "objective_value": float(obj),
        "selected_edges": selected_edges,
        "vertex_costs": vertex_costs
    }

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()