import argparse
import heapq
import json
import random
import time

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except Exception:
    gp = None
    GRB = None
    HAS_GUROBI = False

from solution_logger import SolutionLogger


def relabel(color, n):
    """Relabel colors to 0..k-1 by order of first appearance (vertex 0 -> color 0)."""
    mapping = {}
    out = [0] * n
    nxt = 0
    for v in range(n):
        c = color[v]
        nc = mapping.get(c)
        if nc is None:
            nc = nxt
            mapping[c] = nxt
            nxt += 1
        out[v] = nc
    return out, nxt


def dsatur(n, adj, rng=None, deadline=None):
    """DSATUR greedy coloring with optional random tie-breaking."""
    color = [-1] * n
    nbr_colors = [set() for _ in range(n)]
    sat = [0] * n
    deg = [len(a) for a in adj]
    heap = []
    if rng is not None:
        rnd = rng.random
        for v in range(n):
            heapq.heappush(heap, (0, -deg[v], rnd(), v))
    else:
        for v in range(n):
            heapq.heappush(heap, (0, -deg[v], 0.0, v))
    remaining = n
    it = 0
    aborted = False
    while remaining > 0:
        it += 1
        if deadline is not None and (it & 31) == 0 and time.time() > deadline:
            aborted = True
            break
        while True:
            negs, _, _, v = heapq.heappop(heap)
            if color[v] == -1 and -negs == sat[v]:
                break
        cs = nbr_colors[v]
        c = 0
        while c in cs:
            c += 1
        color[v] = c
        remaining -= 1
        for w in adj[v]:
            if color[w] == -1 and c not in nbr_colors[w]:
                nbr_colors[w].add(c)
                sat[w] += 1
                r = rng.random() if rng is not None else 0.0
                heapq.heappush(heap, (-sat[w], -deg[w], r, w))
    if aborted:
        # plain greedy finish so we always return a proper coloring
        for v in range(n):
            if color[v] != -1:
                continue
            cs = nbr_colors[v]
            c = 0
            while c in cs:
                c += 1
            color[v] = c
            for w in adj[v]:
                if color[w] == -1:
                    nbr_colors[w].add(c)
    k = (max(color) + 1) if n > 0 else 0
    return color, k


def tabucol(n, adj, edges, q, init_color, deadline, rng):
    """Try to find a proper q-coloring with TABUCOL. Return coloring or None."""
    if q <= 1:
        return None
    color = [0] * n
    for v in range(n):
        c = init_color[v]
        color[v] = c if 0 <= c < q else rng.randrange(q)
    cnt = [[0] * q for _ in range(n)]
    for (u, w) in edges:
        cnt[u][color[w]] += 1
        cnt[w][color[u]] += 1
    lst = []
    pos = [-1] * n
    f = 0
    for v in range(n):
        d = cnt[v][color[v]]
        if d > 0:
            pos[v] = len(lst)
            lst.append(v)
            f += d
    f //= 2
    if f == 0:
        return color
    tabu = [[0] * q for _ in range(n)]
    best_f = f
    last_improve = 0
    it = 0
    randrange = rng.randrange
    while f > 0:
        if (it & 127) == 0:
            if time.time() >= deadline or it - last_improve > 300000:
                return None
        it += 1
        llen = len(lst)
        if llen == 0:
            return None
        v = lst[randrange(llen)]
        old = color[v]
        cv = cnt[v]
        base = cv[old]
        tv = tabu[v]
        best_delta = None
        choices = []
        for c in range(q):
            if c == old:
                continue
            delta = cv[c] - base
            nf = f + delta
            if tv[c] > it and nf >= best_f:
                continue
            if best_delta is None or delta < best_delta:
                best_delta = delta
                choices = [c]
            elif delta == best_delta:
                choices.append(c)
        if choices:
            c = choices[randrange(len(choices))] if len(choices) > 1 else choices[0]
        else:
            c = randrange(q - 1)
            if c >= old:
                c += 1
        newc = cv[c]
        f += newc - base
        for w in adj[v]:
            cntw = cnt[w]
            cntw[old] -= 1
            cntw[c] += 1
            cw = color[w]
            if cw == old:
                if cntw[old] == 0:
                    i = pos[w]
                    last = lst[-1]
                    lst[i] = last
                    pos[last] = i
                    lst.pop()
                    pos[w] = -1
            elif cw == c:
                if cntw[c] == 1:
                    pos[w] = len(lst)
                    lst.append(w)
        color[v] = c
        if newc == 0:
            i = pos[v]
            last = lst[-1]
            lst[i] = last
            pos[last] = i
            lst.pop()
            pos[v] = -1
        tabu[v][old] = it + 5 + randrange(10)
        if f < best_f:
            best_f = f
            last_improve = it
    return color


def clique_lower_bound(n, adj, deadline):
    if n == 0:
        return 0
    deg = [len(a) for a in adj]
    order = sorted(range(n), key=lambda v: -deg[v])
    best = 1
    starts = 0
    chk = 0
    for s in order:
        if deg[s] + 1 <= best:
            break
        cand = set(adj[s])
        size = 1
        while cand:
            chk += 1
            if (chk & 1023) == 0 and time.time() > deadline:
                return best
            w = max(cand, key=lambda z: deg[z])
            size += 1
            cand.intersection_update(adj[w])
        if size > best:
            best = size
        starts += 1
        if starts >= 20 or time.time() > deadline:
            break
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    t0 = time.time()
    t_end = t0 + max(0, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)
    n = int(inst["num_nodes"])
    edges = []
    for e in inst["edges"]:
        u, w = int(e[0]), int(e[1])
        if u != w:
            edges.append((u, w))
    m = len(edges)

    adj = [set() for _ in range(n)]
    for (u, w) in edges:
        adj[u].add(w)
        adj[w].add(u)

    rng = random.Random(0)

    best = {"k": None, "color": None}

    def solution_dict(color, k):
        return {
            "objective_value": float(k),
            "coloring": {str(v): int(color[v]) for v in range(n)},
        }

    def log_best():
        if logger and best["color"] is not None:
            logger.log_solution(float(best["k"]),
                                solution_dict(best["color"], best["k"]))

    def write_final():
        color = best["color"] if best["color"] is not None else []
        k = best["k"] if best["k"] is not None else 0
        with open(args.solution_path, "w") as f:
            json.dump(solution_dict(color, k), f)

    def remaining():
        return t_end - time.time() - 0.25

    # ---- trivial cases -------------------------------------------------
    if n == 0:
        best["k"] = 0
        best["color"] = []
        write_final()
        return
    if m == 0:
        best["k"] = 1
        best["color"] = [0] * n
        log_best()
        write_final()
        return

    # ---- initial DSATUR coloring ---------------------------------------
    color, k = dsatur(n, adj, rng=None, deadline=t_end - 0.2)
    color, k = relabel(color, n)
    best["k"] = k
    best["color"] = color
    log_best()

    # ---- clique lower bound --------------------------------------------
    lb = 2
    lb_deadline = min(time.time() + 2.0, t_end - 0.2)
    try:
        lb = max(lb, clique_lower_bound(n, adj, lb_deadline))
    except Exception:
        pass

    # ---- decide strategy ------------------------------------------------
    use_tabucol = (n * max(1, best["k"]) <= 4_000_000)
    use_mip = (HAS_GUROBI
               and m * best["k"] <= 500_000
               and n * best["k"] <= 300_000)

    rem = remaining()
    if use_mip:
        heur_time = min(max(rem * 0.25, 1.0), 15.0)
        heur_time = min(heur_time, max(rem - 1.0, 0.0))
    else:
        heur_time = max(rem - 0.05, 0.0)
    heur_deadline = time.time() + heur_time

    # ---- heuristic improvement phase ------------------------------------
    if best["k"] > lb and heur_time > 0.2:
        if use_tabucol:
            attempt = 0
            while best["k"] > lb and time.time() < heur_deadline:
                q = best["k"] - 1
                if q < 2:
                    break
                now = time.time()
                slice_len = min(max((heur_deadline - now) * 0.5, 1.0), 30.0)
                slice_end = min(heur_deadline, now + slice_len)
                if attempt % 3 == 2:
                    init = [rng.randrange(q) for _ in range(n)]
                else:
                    init = best["color"]
                sol = tabucol(n, adj, edges, q, init, slice_end, rng)
                if sol is not None:
                    sol, kk = relabel(sol, n)
                    if kk < best["k"]:
                        best["k"] = kk
                        best["color"] = sol
                        log_best()
                attempt += 1
        else:
            # very large graphs: repeated randomized DSATUR
            while best["k"] > lb and time.time() < heur_deadline:
                c2, k2 = dsatur(n, adj, rng=rng, deadline=heur_deadline)
                c2, k2 = relabel(c2, n)
                if k2 < best["k"]:
                    best["k"] = k2
                    best["color"] = c2
                    log_best()

    # ---- MIP phase (assignment + color-usage formulation) ---------------
    if use_mip and best["k"] > lb and remaining() > 1.0:
        try:
            K = best["k"]
            model = gp.Model("graph_coloring")
            model.setParam("Seed", 0)
            model.setParam("MIPGap", 1e-4)
            model.setParam("NumericFocus", 0)
            model.setParam("Threads", 1)

            x = [[model.addVar(vtype=GRB.BINARY) for _ in range(K)]
                 for _ in range(n)]
            y = [model.addVar(vtype=GRB.BINARY) for _ in range(K)]

            for v in range(n):
                model.addConstr(gp.quicksum(x[v]) == 1)

            # symmetry breaking: vertex v cannot use a color index > v
            lim = min(n, K)
            for v in range(lim):
                row = x[v]
                for c in range(v + 1, K):
                    row[c].UB = 0.0

            # combined edge/usage constraints:
            # for each edge (u,w) and color c: x_u^c + x_w^c <= y_c
            for (u, w) in edges:
                xu = x[u]
                xw = x[w]
                for c in range(K):
                    model.addConstr(xu[c] + xw[c] <= y[c])

            # isolated vertices still need usage linking
            for v in range(n):
                if not adj[v]:
                    xv = x[v]
                    for c in range(K):
                        model.addConstr(xv[c] <= y[c])

            # use colors in order
            for c in range(K - 1):
                model.addConstr(y[c + 1] <= y[c])

            # clique lower bound
            model.addConstr(gp.quicksum(y) >= lb)

            model.setObjective(gp.quicksum(y), GRB.MINIMIZE)

            # warm start from best heuristic coloring
            warm = best["color"]
            for v in range(n):
                x[v][warm[v]].Start = 1.0
            for c in range(K):
                y[c].Start = 1.0

            x_flat = [var for row in x for var in row]

            def mip_callback(model_, where):
                if where == GRB.Callback.MIPSOL:
                    try:
                        obj = model_.cbGet(GRB.Callback.MIPSOL_OBJ)
                    except Exception:
                        return
                    kk = int(round(obj))
                    if kk >= best["k"]:
                        if best["k"] <= lb:
                            model_.terminate()
                        return
                    try:
                        vals = model_.cbGetSolution(x_flat)
                    except Exception:
                        return
                    col = [0] * n
                    for v in range(n):
                        b = v * K
                        seg = vals[b:b + K]
                        mx = max(seg)
                        col[v] = seg.index(mx)
                    col, used = relabel(col, n)
                    if used < best["k"]:
                        best["k"] = used
                        best["color"] = col
                        log_best()
                    if best["k"] <= lb:
                        model_.terminate()

            model.setParam("TimeLimit", max(1.0, remaining()))
            model.optimize(mip_callback)

            if model.SolCount > 0:
                try:
                    vals = model.getAttr("X", x_flat)
                    col = [0] * n
                    for v in range(n):
                        b = v * K
                        seg = vals[b:b + K]
                        mx = max(seg)
                        col[v] = seg.index(mx)
                    col, used = relabel(col, n)
                    if used < best["k"]:
                        best["k"] = used
                        best["color"] = col
                        log_best()
                except Exception:
                    pass
            try:
                model.dispose()
            except Exception:
                pass
        except Exception:
            pass

    # ---- final safety check and output ----------------------------------
    final_color = best["color"]
    proper = True
    for (u, w) in edges:
        if final_color[u] == final_color[w]:
            proper = False
            break
    if not proper:
        final_color, k = dsatur(n, adj, rng=None, deadline=t_end - 0.1)
        final_color, k = relabel(final_color, n)
        best["color"] = final_color
        best["k"] = k

    write_final()


if __name__ == "__main__":
    main()