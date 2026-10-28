import argparse
import heapq
import json
import math
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def mst_edges(S, w):
    """Prim MST over the induced complete graph of node set S."""
    S = list(S)
    L = len(S)
    if L <= 1:
        return []
    visited = {S[0]}
    heap = []
    root = S[0]
    for b in S[1:]:
        heapq.heappush(heap, (w[root][b], root, b))
    edges = []
    while heap and len(edges) < L - 1:
        cost, a, b = heapq.heappop(heap)
        if b in visited:
            continue
        visited.add(b)
        edges.append((a, b))
        for cnode in S:
            if cnode not in visited:
                heapq.heappush(heap, (w[b][cnode], b, cnode))
    return edges


def evaluate(S, p, c, w, m):
    """Objective value of activating set S (assignment + MST + node weights)."""
    S = list(S)
    edges = mst_edges(S, w)
    obj = sum(p[j] for j in S)
    obj += sum(w[a][b] for (a, b) in edges)
    for i in range(m):
        ci = c[i]
        obj += min(ci[j] for j in S)
    return obj, edges


def make_solution(S, edges, p, c, w, m):
    """Build a schema-compliant solution dict and return (objective, dict)."""
    S = sorted(S)
    obj = float(sum(p[j] for j in S))
    tree = []
    for (a, b) in edges:
        obj += w[a][b]
        tree.append([a, b])
    assignments = {}
    for i in range(m):
        ci = c[i]
        best_j = S[0]
        best_v = ci[best_j]
        for j in S[1:]:
            v = ci[j]
            if v < best_v:
                best_v = v
                best_j = j
        assignments[str(i)] = best_j
        obj += best_v
    sol = {
        "objective_value": float(obj),
        "active_steiner_nodes": list(S),
        "assignments": assignments,
        "steiner_tree_edges": tree,
    }
    return obj, sol


def is_spanning_tree(S, edges):
    """Check that edges form a spanning tree over S."""
    if len(S) == 0:
        return len(edges) == 0
    if len(edges) != len(S) - 1:
        return False
    Sset = set(S)
    parent = {j: j for j in S}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b) in edges:
        if a not in Sset or b not in Sset:
            return False
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[ra] = rb
    return len({find(j) for j in S}) == 1


def flow_values(S, edges):
    """Feasible single-commodity flow on a tree rooted at the first node of S."""
    if not S:
        return {}, {}
    root = sorted(S)[0]
    adj = {j: [] for j in S}
    for (a, b) in edges:
        adj[a].append(b)
        adj[b].append(a)
    parent = {root: None}
    order = [root]
    for u in order:
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                order.append(v)
    size = {u: 1 for u in S}
    for u in reversed(order):
        if parent[u] is not None:
            size[parent[u]] += size[u]
    fvals = {}
    for u in order[1:]:
        fvals[(parent[u], u)] = float(size[u])
    avals = {root: float(len(S))}
    return avals, fvals


# ---------------------------------------------------------------------------
# Heuristic: greedy additions / removals with local improvement
# ---------------------------------------------------------------------------

def run_heuristic(n, m, c, w, p, time_budget, logger):
    deadline = time.time() + max(0.0, time_budget)
    if n == 0 or m == 0:
        return None

    # Start with the best single Steiner node.
    first = min(range(n), key=lambda j: p[j] + sum(c[i][j] for i in range(m)))
    S = {first}
    best_obj, best_edges = evaluate(S, p, c, w, m)
    best_sol = make_solution(S, best_edges, p, c, w, m)[1]
    if logger:
        logger.log_solution(best_obj, best_sol)

    while time.time() < deadline:
        move = None
        move_obj = best_obj
        move_edges = None
        outside = [j for j in range(n) if j not in S]

        # Addition moves
        for j in outside:
            if time.time() >= deadline:
                break
            cand_edges = None
            o, e = evaluate(S | {j}, p, c, w, m)
            if o < move_obj - 1e-9:
                move_obj = o
                move = ("add", j)
                move_edges = e

        # Removal moves
        if len(S) > 1 and time.time() < deadline:
            for j in list(S):
                if time.time() >= deadline:
                    break
                o, e = evaluate(S - {j}, p, c, w, m)
                if o < move_obj - 1e-9:
                    move_obj = o
                    move = ("remove", j)
                    move_edges = e

        if move is None:
            break

        kind, j = move
        if kind == "add":
            S.add(j)
        else:
            S.discard(j)
        best_obj = move_obj
        best_edges = move_edges
        best_sol = make_solution(S, best_edges, p, c, w, m)[1]
        if logger:
            logger.log_solution(best_obj, best_sol)

    return best_obj, best_sol


# ---------------------------------------------------------------------------
# Exact MIP solved with Gurobi
# ---------------------------------------------------------------------------

def solve_with_gurobi(n, m, c, w, p, remaining_time, logger, start_sol):
    model = gp.Model("steiner_network")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = max(0.0, remaining_time)

    pairs = [(j, k) for j in range(n) for k in range(j + 1, n)]

    y = [model.addVar(vtype=GRB.BINARY, name="y_%d" % j) for j in range(n)]
    x = None
    if m > 0:
        x = [[model.addVar(lb=0.0, ub=1.0, name="x_%d_%d" % (i, j))
              for j in range(n)] for i in range(m)]
    z = [model.addVar(vtype=GRB.BINARY, name="z_%d_%d" % pr) for pr in pairs]
    a = [model.addVar(lb=0.0, name="a_%d" % j) for j in range(n)]
    f = {}
    for (j, k) in pairs:
        f[(j, k)] = model.addVar(lb=0.0, name="f_%d_%d" % (j, k))
        f[(k, j)] = model.addVar(lb=0.0, name="f_%d_%d" % (k, j))

    # Objective
    obj = gp.LinExpr()
    obj.addTerms(p, y)
    for idx, (j, k) in enumerate(pairs):
        obj.add(w[j][k] * z[idx])
    if m > 0:
        for i in range(m):
            for j in range(n):
                obj.add(c[i][j] * x[i][j])
    model.setObjective(obj, GRB.MINIMIZE)

    # Assignment constraints
    if m > 0:
        for i in range(m):
            model.addConstr(gp.quicksum(x[i][j] for j in range(n)) == 1.0)
            for j in range(n):
                model.addConstr(x[i][j] <= y[j])

    # Edge endpoints must be active
    for idx, (j, k) in enumerate(pairs):
        model.addConstr(z[idx] <= y[j])
        model.addConstr(z[idx] <= y[k])

    # Spanning tree edge count
    model.addConstr(gp.quicksum(z) == gp.quicksum(y) - 1.0)

    # Root-arc capacity
    for j in range(n):
        model.addConstr(a[j] <= n * y[j])

    # Flow capacity on undirected edges
    for idx, (j, k) in enumerate(pairs):
        model.addConstr(f[(j, k)] + f[(k, j)] <= n * z[idx])

    # Flow balance: every active node consumes one unit
    for j in range(n):
        inflow = gp.LinExpr()
        outflow = gp.LinExpr()
        for (u, v) in pairs:
            if u == j:
                inflow.add(f[(v, u)])
                outflow.add(f[(u, v)])
            elif v == j:
                inflow.add(f[(u, v)])
                outflow.add(f[(v, u)])
        model.addConstr(a[j] + inflow - outflow == y[j])

    # MIP start from heuristic solution
    if start_sol is not None and m > 0:
        S = set(start_sol["active_steiner_nodes"])
        edges = [tuple(e) for e in start_sol["steiner_tree_edges"]]
        norm_edges = set((min(a2, b2), max(a2, b2)) for (a2, b2) in edges)
        for j in range(n):
            y[j].Start = 1.0 if j in S else 0.0
        if S:
            for i in range(m):
                best_j = min(S, key=lambda jj: c[i][jj])
                for j in range(n):
                    x[i][j].Start = 1.0 if j == best_j else 0.0
            for idx, (u, v) in enumerate(pairs):
                z[idx].Start = 1.0 if (u, v) in norm_edges else 0.0
            if len(edges) == len(S) - 1:
                avals, fvals = flow_values(S, edges)
                for j in range(n):
                    a[j].Start = avals.get(j, 0.0)
                for (j, k) in pairs:
                    f[(j, k)].Start = fvals.get((j, k), 0.0)
                    f[(k, j)].Start = fvals.get((k, j), 0.0)

    # Attach helper data for callback / extraction
    model._n = n
    model._m = m
    model._c = c
    model._w = w
    model._p = p
    model._pprs = pairs
    model._yvars = y
    model._zvars = z
    model._logger = logger

    def callback(mdl, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            yv = mdl.cbGetSolution(mdl._yvars)
            S = [j for j in range(mdl._n) if yv[j] > 0.5]
            if not S:
                return
            zv = mdl.cbGetSolution(mdl._zvars)
            edges = [mdl._pprs[t] for t in range(len(mdl._pprs)) if zv[t] > 0.5]
            if not is_spanning_tree(S, edges):
                edges = mst_edges(S, mdl._w)
            val, sdict = make_solution(S, edges, mdl._p, mdl._c, mdl._w, mdl._m)
            if mdl._logger:
                mdl._logger.log_solution(val, sdict)
        except Exception:
            pass

    model.optimize(callback)

    if model.SolCount == 0:
        return None
    yv = [v.X for v in y]
    S = [j for j in range(n) if yv[j] > 0.5]
    if not S:
        return None
    zv = [v.X for v in z]
    edges = [pairs[t] for t in range(len(pairs)) if zv[t] > 0.5]
    if not is_spanning_tree(S, edges):
        edges = mst_edges(S, w)
    val, sdict = make_solution(S, edges, p, c, w, m)
    if logger:
        logger.log_solution(val, sdict)
    return val, sdict


# ---------------------------------------------------------------------------
# Instance loading / main driver
# ---------------------------------------------------------------------------

def read_instance(path):
    with open(path, "r") as fh:
        data = json.load(fh)
    params = data.get("parameters", {})
    tnodes = data.get("target_nodes", {})
    snodes = data.get("steiner_nodes", {})
    costs = data.get("cost_matrices", {})

    tc = tnodes.get("coordinates", [])
    sc = snodes.get("coordinates", [])
    p = snodes.get("node_weights", [])
    m = params.get("m", tnodes.get("count", len(tc)))
    n = params.get("n", snodes.get("count", len(sc)))

    c = costs.get("assignment_costs", {}).get("matrix")
    w = costs.get("interconnection_costs", {}).get("matrix")

    if c is None and tc and sc:
        c = [[math.hypot(tc[i][0] - sc[j][0], tc[i][1] - sc[j][1])
              for j in range(n)] for i in range(m)]
    if w is None and sc:
        w = [[math.hypot(sc[j][0] - sc[k][0], sc[j][1] - sc[k][1])
              for k in range(n)] for j in range(n)]

    return m, n, (c or []), (w or []), p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start = time.time()
    m, n, c, w, p = read_instance(args.instance_path)

    # Trivial case: no targets -> empty network has zero cost
    if m == 0:
        sol = {
            "objective_value": 0.0,
            "active_steiner_nodes": [],
            "assignments": {},
            "steiner_tree_edges": [],
        }
        if logger:
            logger.log_solution(0.0, sol)
        with open(args.solution_path, "w") as fh:
            json.dump(sol, fh, indent=2)
        return

    best_val, best_sol = None, None

    # --- Heuristic incumbent (bounded time share) ---
    h_budget = min(2.0, max(0.0, 0.1 * float(args.time_limit)))
    h = run_heuristic(n, m, c, w, p, h_budget, logger)
    if h is not None:
        best_val, best_sol = h

    # --- Gurobi exact solve with remaining time ---
    remaining = args.time_limit - (time.time() - start)
    if n > 0 and remaining > 0.0:
        try:
            res = solve_with_gurobi(n, m, c, w, p, remaining, logger, best_sol)
            if res is not None:
                val, sdict = res
                if best_val is None or val < best_val - 1e-9:
                    best_val, best_sol = val, sdict
        except Exception:
            pass

    # --- Fallback (should not be needed) ---
    if best_sol is None:
        S = list(range(n))
        edges = mst_edges(S, w) if n > 0 else []
        best_val, best_sol = make_solution(S, edges, p, c, w, m)

    # Final log of the returned incumbent
    if logger:
        logger.log_solution(best_val, best_sol)

    with open(args.solution_path, "w") as fh:
        json.dump(best_sol, fh, indent=2)


if __name__ == "__main__":
    main()