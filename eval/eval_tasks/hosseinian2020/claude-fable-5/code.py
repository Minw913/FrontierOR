import argparse
import json
import time
import random
import sys

from solution_logger import SolutionLogger


def read_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    deadline = start_time + max(1, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    data = read_instance(args.instance_path)
    vertices = list(data.get("vertices", []))
    n = data.get("num_vertices", len(vertices))
    if not vertices:
        vertices = list(range(n))
    edges = data.get("edges", [])

    # Map vertex labels to indices
    idx = {v: k for k, v in enumerate(vertices)}
    n = len(vertices)

    adj = [set() for _ in range(n)]
    wmap = {}
    for e in edges:
        i = idx[e["i"]]
        j = idx[e["j"]]
        w = float(e["weight"])
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in wmap:
            wmap[(a, b)] = max(wmap[(a, b)], w)
        else:
            wmap[(a, b)] = w
        adj[i].add(j)
        adj[j].add(i)

    def ew(u, v):
        return wmap.get((u, v) if u < v else (v, u), 0.0)

    def clique_weight(C):
        total = 0.0
        lst = list(C)
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                total += ew(lst[a], lst[b])
        return total

    best_clique = [0] if n > 0 else []
    best_obj = 0.0

    def record(obj, clique_idx):
        nonlocal best_obj, best_clique
        if obj > best_obj or (best_obj == 0.0 and obj >= 0.0 and not best_clique):
            if obj > best_obj:
                best_obj = obj
                best_clique = list(clique_idx)
                if logger:
                    logger.log_solution(
                        best_obj,
                        {
                            "objective_value": best_obj,
                            "clique": [vertices[i] for i in best_clique],
                        },
                    )

    # initial trivial log
    if logger and n > 0:
        logger.log_solution(0.0, {"objective_value": 0.0, "clique": [vertices[0]]})

    rng = random.Random(0)

    # -------- Greedy construction from a seed vertex --------
    def greedy_from(seed):
        C = [seed]
        weight = 0.0
        cand = set(adj[seed])
        gains = {u: ew(u, seed) for u in cand}
        while cand:
            # pick candidate with max gain (random tie-break)
            best_g = -1.0
            best_u = None
            for u in cand:
                g = gains[u]
                if g > best_g or (g == best_g and rng.random() < 0.5):
                    best_g = g
                    best_u = u
            u = best_u
            C.append(u)
            weight += best_g
            cand &= adj[u]
            for x in cand:
                gains[x] = gains.get(x, 0.0) + ew(x, u)
        return C, weight

    # -------- Local improvement: drop a vertex, re-extend greedily --------
    def improve(C, weight, budget_end):
        improved = True
        C = list(C)
        while improved and time.time() < budget_end:
            improved = False
            for drop_pos in range(len(C)):
                if time.time() >= budget_end:
                    break
                v = C[drop_pos]
                rest = [u for u in C if u != v]
                loss = sum(ew(v, u) for u in rest)
                # candidates adjacent to all of rest
                if rest:
                    cand = set(adj[rest[0]])
                    for u in rest[1:]:
                        cand &= adj[u]
                    cand.discard(v)
                    for u in rest:
                        cand.discard(u)
                else:
                    cand = set(range(n))
                    cand.discard(v)
                cur = list(rest)
                cur_w = weight - loss
                gains = {u: sum(ew(u, x) for x in cur) for u in cand}
                while cand:
                    bu = max(cand, key=lambda u: gains[u])
                    bg = gains[bu]
                    cur.append(bu)
                    cur_w += bg
                    cand &= adj[bu]
                    for x in cand:
                        gains[x] = gains.get(x, 0.0) + ew(x, bu)
                if cur_w > weight + 1e-9:
                    C = cur
                    weight = cur_w
                    improved = True
                    break
        return C, weight

    # -------- Heuristic phase --------
    total_time = max(1.0, deadline - time.time())
    heur_budget = min(0.30 * total_time, 20.0)
    heur_end = time.time() + heur_budget

    # order seeds by weighted degree
    wdeg = [0.0] * n
    for (a, b), w in wmap.items():
        wdeg[a] += w
        wdeg[b] += w
    seed_order = sorted(range(n), key=lambda v: -wdeg[v])

    si = 0
    while time.time() < heur_end and si < 5 * max(1, n):
        if si < n:
            seed = seed_order[si]
        else:
            seed = rng.randrange(n)
        si += 1
        C, w = greedy_from(seed)
        if w > best_obj:
            record(w, C)
        # local improvement on promising cliques
        if w >= 0.8 * best_obj:
            C2, w2 = improve(C, w, heur_end)
            if w2 > best_obj:
                record(w2, C2)
        if si >= n and time.time() >= heur_end:
            break

    # -------- Exact phase with Gurobi (if feasible in size) --------
    remaining = deadline - time.time() - 2.0
    total_pairs = n * (n - 1) // 2
    num_non_edges = total_pairs - len(wmap)

    use_mip = remaining > 3.0 and num_non_edges <= 2_000_000 and n > 0 and len(wmap) > 0

    if use_mip:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            m = gp.Model("mewc")
            m.Params.OutputFlag = 0
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.TimeLimit = max(1.0, remaining)

            x = m.addVars(n, vtype=GRB.BINARY, name="x")
            edge_list = list(wmap.items())
            y = m.addVars(len(edge_list), lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y")

            obj = gp.quicksum(w * y[k] for k, ((a, b), w) in enumerate(edge_list))
            m.setObjective(obj, GRB.MAXIMIZE)

            for k, ((a, b), w) in enumerate(edge_list):
                m.addConstr(y[k] <= x[a])
                m.addConstr(y[k] <= x[b])

            # non-edge constraints
            for u in range(n):
                au = adj[u]
                for v in range(u + 1, n):
                    if v not in au:
                        m.addConstr(x[u] + x[v] <= 1)

            # warm start
            in_best = set(best_clique)
            for i in range(n):
                x[i].Start = 1.0 if i in in_best else 0.0

            xvars = [x[i] for i in range(n)]

            def cb(model, where):
                if where == GRB.Callback.MIPSOL:
                    objv = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if objv > best_obj + 1e-9:
                        xv = model.cbGetSolution(xvars)
                        clq = [i for i in range(n) if xv[i] > 0.5]
                        # recompute exact weight
                        w = clique_weight(clq)
                        record(w, clq)

            m.optimize(cb)

            if m.SolCount > 0:
                clq = [i for i in range(n) if x[i].X > 0.5]
                w = clique_weight(clq)
                if w > best_obj:
                    record(w, clq)
        except Exception:
            pass

    # -------- Fallback: continue heuristic if time remains --------
    while time.time() < deadline - 0.5 and not use_mip:
        seed = rng.randrange(n) if n > 0 else 0
        if n == 0:
            break
        C, w = greedy_from(seed)
        if w > best_obj:
            record(w, C)
            C2, w2 = improve(C, w, deadline - 0.5)
            if w2 > best_obj:
                record(w2, C2)

    # -------- Write solution --------
    if not best_clique and n > 0:
        best_clique = [0]
        best_obj = 0.0

    solution = {
        "objective_value": float(best_obj),
        "clique": [vertices[i] for i in best_clique],
    }
    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()