import argparse
import heapq
import json
import random
import time

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_OK = True
except Exception:
    gp = None
    GRB = None
    GUROBI_OK = False

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)


def num_colors(color):
    return len(set(color)) if color else 0


def normalize(color):
    mapping = {}
    nxt = 0
    out = [0] * len(color)
    for i, c in enumerate(color):
        if c not in mapping:
            mapping[c] = nxt
            nxt += 1
        out[i] = mapping[c]
    return out


def build_solution(color):
    mapping = {}
    nxt = 0
    coloring = {}
    classes = {}
    for v in range(len(color)):
        c = color[v]
        if c not in mapping:
            mapping[c] = nxt
            classes[str(nxt)] = []
            nxt += 1
        nc = mapping[c]
        coloring[str(v)] = nc
        classes[str(nc)].append(v)
    return {
        "objective_value": nxt,
        "coloring": coloring,
        "color_classes": classes,
    }


def verify(color, edges):
    for u, v in edges:
        if color[u] == color[v]:
            return False
    return True


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

def greedy_color(n, adj, order):
    color = [-1] * n
    for v in order:
        used = set()
        for u in adj[v]:
            cu = color[u]
            if cu != -1:
                used.add(cu)
        c = 0
        while c in used:
            c += 1
        color[v] = c
    return color


def dsatur(n, adj):
    degree = [len(a) for a in adj]
    color = [-1] * n
    sat = [0] * n
    nb = [set() for _ in range(n)]
    done = [False] * n
    heap = [(0, -degree[v], v) for v in range(n)]
    heapq.heapify(heap)
    remaining = n
    while remaining > 0 and heap:
        s, d, v = heapq.heappop(heap)
        if done[v] or -s != sat[v] or -d != degree[v]:
            continue
        used = nb[v]
        c = 0
        while c in used:
            c += 1
        color[v] = c
        done[v] = True
        remaining -= 1
        for u in adj[v]:
            if not done[u] and c not in nb[u]:
                nb[u].add(c)
                sat[u] += 1
                heapq.heappush(heap, (-sat[u], -degree[u], u))
    return color


def degeneracy_order(n, adj):
    deg = [len(a) for a in adj]
    heap = [(deg[v], v) for v in range(n)]
    heapq.heapify(heap)
    removed = [False] * n
    order = []
    while heap:
        d, v = heapq.heappop(heap)
        if removed[v] or d != deg[v]:
            continue
        removed[v] = True
        order.append(v)
        for u in adj[v]:
            if not removed[u]:
                deg[u] -= 1
                heapq.heappush(heap, (deg[u], u))
    return order


def greedy_clique(n, adj, rng, deadline, extra_tries=16):
    best = []
    orders = [sorted(range(n), key=lambda v: -len(adj[v]))]
    for _ in range(extra_tries):
        o = list(range(n))
        rng.shuffle(o)
        orders.append(o)
    for order in orders:
        if time.monotonic() > deadline:
            break
        clique = []
        for v in order:
            ok = True
            for u in clique:
                if v not in adj[u]:
                    ok = False
                    break
            if ok:
                clique.append(v)
        if len(clique) > len(best):
            best = clique
    return best


def try_eliminate(n, adj, color, target, rng, deadline, max_attempts=30):
    palette = [c for c in range(max(color) + 1) if c != target]
    removed_vs = [v for v in range(n) if color[v] == target]
    if not removed_vs:
        return None
    for attempt in range(max_attempts):
        if time.monotonic() > deadline:
            return None
        order = removed_vs[:]
        if attempt == 0:
            order.sort(key=lambda v: -len(adj[v]))
        else:
            rng.shuffle(order)
        tmp = color[:]
        ok = True
        for v in order:
            forb = set()
            for u in adj[v]:
                forb.add(tmp[u])
            chosen = -1
            for c in palette:
                if c not in forb:
                    chosen = c
                    break
            if chosen == -1:
                ok = False
                break
            tmp[v] = chosen
        if ok:
            return tmp
    return None


def local_improve(n, adj, color, deadline, rng):
    color = normalize(color)
    while time.monotonic() < deadline:
        sizes = {}
        for c in color:
            sizes[c] = sizes.get(c, 0) + 1
        improved = False
        for target in sorted(sizes, key=lambda c: sizes[c]):
            cand = try_eliminate(n, adj, color, target, rng, deadline)
            if cand is not None:
                color = normalize(cand)
                improved = True
                break
        if not improved:
            break
    return color


# ---------------------------------------------------------------------------
# Gurobi feasibility model: is the graph K-colorable?
# ---------------------------------------------------------------------------

def mip_check_k(env, n, edges, K, clique, time_left):
    m = gp.Model("graph_coloring", env=env)
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = max(0.01, time_left)
    x = m.addVars(n, K, vtype=GRB.BINARY, name="x")
    m.addConstrs((gp.quicksum(x[v, c] for c in range(K)) == 1) for v in range(n))
    m.addConstrs((x[u, c] + x[v, c] <= 1) for (u, v) in edges for c in range(K))
    for i, v in enumerate(clique[:K]):
        m.addConstr(x[v, i] == 1)
    m.setObjective(0.0)
    m.optimize()
    if m.SolCount > 0:
        col = [-1] * n
        for v in range(n):
            for c in range(K):
                if x[v, c].X > 0.5:
                    col[v] = c
                    break
        if all(c >= 0 for c in col):
            return col, "feasible"
        return None, "unknown"
    if m.Status == GRB.INFEASIBLE:
        return None, "infeasible"
    return None, "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start = time.monotonic()
    deadline = start + max(0, args.time_limit)

    def remaining():
        return deadline - time.monotonic()

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path, "r") as f:
        instance = json.load(f)
    graph = instance["graph"]
    n = int(graph["num_vertices"])
    edges_raw = graph.get("edges", [])

    # Deduplicate edges, drop self-loops, normalize orientation
    edge_set = set()
    for e in edges_raw:
        u, v = int(e[0]), int(e[1])
        if u == v:
            continue
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    edges = sorted(edge_set)
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    rng = random.Random(0)

    # Trivial cases ---------------------------------------------------------
    if n == 0:
        sol = {"objective_value": 0, "coloring": {}, "color_classes": {}}
        if logger:
            logger.log_solution(0, sol)
        write_json(args.solution_path, sol)
        return
    if not edges:
        sol = build_solution([0] * n)
        if logger:
            logger.log_solution(1, sol)
        write_json(args.solution_path, sol)
        return

    # Initial heuristic colorings -------------------------------------------
    candidates = []
    candidates.append(dsatur(n, adj))
    candidates.append(greedy_color(n, adj, sorted(range(n), key=lambda v: -len(adj[v]))))
    if remaining() > 0.3:
        sl = degeneracy_order(n, adj)
        candidates.append(greedy_color(n, adj, sl[::-1]))
    best_color = min(candidates, key=lambda c: num_colors(c))
    best_color = normalize(best_color)
    best_k = num_colors(best_color)

    # Clique lower bound -----------------------------------------------------
    clique = greedy_clique(n, adj, rng, deadline)
    LB = len(clique)

    best_sol = build_solution(best_color)

    def update(new_color):
        nonlocal best_color, best_k, best_sol
        new_color = normalize(new_color)
        k = num_colors(new_color)
        if k < best_k:
            best_color = new_color
            best_k = k
            best_sol = build_solution(best_color)
            if logger:
                logger.log_solution(best_k, best_sol)
            write_json(args.solution_path, best_sol)

    if logger:
        logger.log_solution(best_k, best_sol)
    write_json(args.solution_path, best_sol)

    # Decide whether the MIP feasibility route is tractable ------------------
    K0 = best_k - 1
    mip_applicable = (
        GUROBI_OK
        and best_k > LB
        and n > 0
        and n * K0 <= 150000
        and (len(edges) * K0 + n) <= 900000
    )

    # Phase 1: quick local color-class elimination ---------------------------
    if best_k > LB and remaining() > 0.3:
        now = time.monotonic()
        if mip_applicable:
            local_end = now + min(remaining() * 0.08, 5.0)
        else:
            local_end = now + remaining() * 0.5
        improved = local_improve(n, adj, best_color, local_end, rng)
        update(improved)

    # Phase 2: Gurobi feasibility descent ------------------------------------
    if mip_applicable and best_k > LB and remaining() > 0.5:
        env = None
        try:
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()
        except Exception:
            env = None
        if env is not None:
            K = best_k - 1
            while K >= LB and remaining() > 0.5:
                col, status = mip_check_k(env, n, edges, K, clique, remaining())
                if col is not None and verify(col, edges):
                    update(col)
                    K = best_k - 1
                elif status == "infeasible":
                    break
                else:
                    break

    # Phase 3: use any leftover time for further heuristic reduction ---------
    if best_k > LB and remaining() > 0.2:
        improved = local_improve(n, adj, best_color, deadline, rng)
        update(improved)

    write_json(args.solution_path, best_sol)


if __name__ == "__main__":
    main()