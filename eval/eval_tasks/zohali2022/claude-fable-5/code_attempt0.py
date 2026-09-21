import argparse
import json
import math
import random
import time

import numpy as np

INF = float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=60)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    start = time.time()
    deadline = start + max(2, args.time_limit) - 1.0

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    with open(args.instance_path) as f:
        data = json.load(f)

    n = int(data["num_tasks"])
    m = int(data["num_stations"])
    p = np.asarray(data["processing_times"], dtype=float)
    fwd = np.asarray(data["forward_setup_times"], dtype=float)
    bwd = np.asarray(data["backward_setup_times"], dtype=float)
    raw_prec = [(int(e[0]), int(e[1])) for e in data.get("precedence_relations", [])]

    # ---- build precedence graph, auto-detecting 0-based vs 1-based indexing ----
    def build(base):
        pr = [set() for _ in range(n)]
        sc = [set() for _ in range(n)]
        for a, b in raw_prec:
            i, j = a - base, b - base
            if i < 0 or j < 0 or i >= n or j >= n or i == j:
                return None
            pr[j].add(i)
            sc[i].add(j)
        pr = [sorted(s) for s in pr]
        sc = [sorted(s) for s in sc]
        indeg = [len(pr[t]) for t in range(n)]
        stack = [t for t in range(n) if indeg[t] == 0]
        cnt = 0
        while stack:
            t = stack.pop()
            cnt += 1
            for s in sc[t]:
                indeg[s] -= 1
                if indeg[s] == 0:
                    stack.append(s)
        if cnt != n:
            return None
        return pr, sc

    idxs = [x for e in raw_prec for x in e]
    if idxs:
        if max(idxs) >= n:
            tries = [1, 0]
        elif min(idxs) == 0:
            tries = [0, 1]
        else:
            tries = [1, 0]
    else:
        tries = [0]
    built = None
    for base in tries:
        built = build(base)
        if built is not None:
            break
    if built is None:
        built = ([[] for _ in range(n)], [[] for _ in range(n)])
    preds, succs = built

    rng = random.Random(0)

    # ---------------- topological order generators ----------------
    def topo_random():
        indeg = [len(preds[t]) for t in range(n)]
        avail = [t for t in range(n) if indeg[t] == 0]
        order = []
        while avail:
            i = rng.randrange(len(avail))
            avail[i], avail[-1] = avail[-1], avail[i]
            t = avail.pop()
            order.append(t)
            for s in succs[t]:
                indeg[s] -= 1
                if indeg[s] == 0:
                    avail.append(s)
        return order

    def topo_greedy():
        indeg = [len(preds[t]) for t in range(n)]
        avail = [t for t in range(n) if indeg[t] == 0]
        order = []
        prev = -1
        while avail:
            if prev < 0:
                t = min(avail, key=lambda x: p[x])
            else:
                t = min(avail, key=lambda x: fwd[prev][x])
            avail.remove(t)
            order.append(t)
            prev = t
            for s in succs[t]:
                indeg[s] -= 1
                if indeg[s] == 0:
                    avail.append(s)
        return order

    # ---------------- solution formatting ----------------
    def make_solution(perm, bounds, obj):
        assignment = {}
        seqs = {str(s): [] for s in range(1, m + 1)}
        for s, (a, b) in enumerate(bounds, 1):
            seq = [int(perm[i]) + 1 for i in range(a, b + 1)]
            seqs[str(s)] = seq
            for t in seq:
                assignment[str(t)] = s
        return {
            "objective_value": float(obj),
            "assignment": assignment,
            "station_sequences": seqs,
            "cycle_time": float(obj),
        }

    def write_out(sol):
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)

    # ---------------- infeasible station count fallback ----------------
    if n < m or n == 0:
        order = topo_greedy() if n > 0 else []
        assignment = {}
        seqs = {str(s): [] for s in range(1, m + 1)}
        obj = 0.0
        for s, t in enumerate(order, 1):
            seqs[str(s)] = [t + 1]
            assignment[str(t + 1)] = s
            obj = max(obj, float(p[t] + bwd[t][t]))
        sol = {
            "objective_value": obj,
            "assignment": assignment,
            "station_sequences": seqs,
            "cycle_time": obj,
        }
        if logger:
            logger.log_solution(obj, sol)
        write_out(sol)
        return

    # ---------------- evaluation: optimal split of a topological order ----
    rows = np.arange(n)
    if n > 1:
        tri = np.triu_indices(n, 1)
    else:
        tri = None

    def evaluate(perm):
        """perm: list of task ids (topological). Returns (obj, bounds, costs)."""
        idx = np.asarray(perm, dtype=np.int64)
        pa = p[idx]
        P = np.empty(n + 1)
        P[0] = 0.0
        np.cumsum(pa, out=P[1:])
        Fc = np.zeros(n)
        if n > 1:
            Fc[1:] = np.cumsum(fwd[idx[:-1], idx[1:]])
        # S[b, a] = cost of segment covering positions a..b (0-based, inclusive)
        S = (P[1:, None] - P[None, :-1]) + (Fc[:, None] - Fc[None, :]) + bwd[np.ix_(idx, idx)]
        if tri is not None:
            S[tri] = INF
        g = S[:, 0].copy()
        arglist = []
        for _k in range(1, m):
            cand = np.maximum(g[:-1][None, :], S[:, 1:])
            arg = cand.argmin(axis=1)
            g = cand[rows, arg]
            arglist.append(arg)
        obj = float(g[n - 1])
        # backtrack
        b = n - 1
        bounds = []
        for k in range(m - 1, 0, -1):
            a = int(arglist[k - 1][b]) + 1
            bounds.append((a, b))
            b = a - 1
        bounds.append((0, b))
        bounds.reverse()
        costs = [float(S[bb, aa]) for (aa, bb) in bounds]
        return obj, bounds, costs

    def energy_of(obj, costs):
        ss = sum(c * c for c in costs) / max(1, len(costs))
        return obj + 1e-3 * math.sqrt(ss)

    # ---------------- initial solutions ----------------
    best_perm = None
    best_obj = INF
    best_bounds = None
    best_costs = None

    init_cands = [topo_greedy()]
    n_rand = 20 if n <= 300 else 6
    for _ in range(n_rand):
        init_cands.append(topo_random())

    for perm in init_cands:
        if time.time() > deadline:
            break
        obj, bounds, costs = evaluate(perm)
        if obj < best_obj:
            best_perm = perm[:]
            best_obj = obj
            best_bounds = bounds
            best_costs = costs
            if logger:
                logger.log_solution(best_obj, make_solution(best_perm, best_bounds, best_obj))

    if best_perm is None:
        best_perm = topo_greedy()
        best_obj, best_bounds, best_costs = evaluate(best_perm)
        if logger:
            logger.log_solution(best_obj, make_solution(best_perm, best_bounds, best_obj))

    # ---------------- simulated annealing over topological orders ----------
    def neighbor(perm, pos, bounds, costs):
        for _ in range(25):
            if rng.random() < 0.75 or n < 2:
                # insertion move
                if bounds is not None and rng.random() < 0.5:
                    ci = max(range(len(costs)), key=lambda i: costs[i])
                    a, b = bounds[ci]
                    pi = rng.randint(a, b)
                else:
                    pi = rng.randrange(n)
                t = perm[pi]
                lo = max((pos[x] for x in preds[t]), default=-1)
                hi = min((pos[x] for x in succs[t]), default=n)
                jlo = lo + 1
                jhi = min(hi - 1, n - 1)
                if jhi < jlo:
                    continue
                j = rng.randint(jlo, jhi)
                if j == pi:
                    continue
                perm2 = perm[:pi] + perm[pi + 1:]
                return perm2[:j] + [t] + perm2[j:]
            else:
                # feasible swap of two nearby tasks
                pi = rng.randrange(n - 1)
                qi = min(n - 1, pi + rng.randint(1, 4))
                if qi <= pi:
                    continue
                u, v = perm[pi], perm[qi]
                ok = True
                for x in preds[v]:
                    if pos[x] >= pi:
                        ok = False
                        break
                if ok:
                    for x in succs[u]:
                        if pos[x] <= qi:
                            ok = False
                            break
                if not ok:
                    continue
                newp = perm[:]
                newp[pi], newp[qi] = v, u
                return newp
        return None

    cur_perm = best_perm[:]
    cur_obj, cur_bounds, cur_costs = best_obj, best_bounds, best_costs
    cur_energy = energy_of(cur_obj, cur_costs)

    T0 = max(0.05 * best_obj, 1e-6)
    Tend = max(5e-4 * best_obj, 1e-9)
    span = max(deadline - time.time(), 1e-6)
    sa_start = time.time()
    since_best = 0

    while True:
        now = time.time()
        if now > deadline:
            break
        frac = min(1.0, (now - sa_start) / span)
        T = T0 * ((Tend / T0) ** frac)

        pos = [0] * n
        for i, t in enumerate(cur_perm):
            pos[t] = i

        newp = neighbor(cur_perm, pos, cur_bounds, cur_costs)
        if newp is None:
            since_best += 1
            if since_best > 4000:
                cur_perm = best_perm[:]
                cur_obj, cur_bounds, cur_costs = best_obj, best_bounds, best_costs
                cur_energy = energy_of(cur_obj, cur_costs)
                since_best = 0
            continue

        obj, bounds, costs = evaluate(newp)
        en = energy_of(obj, costs)
        delta = en - cur_energy
        if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
            cur_perm = newp
            cur_obj, cur_bounds, cur_costs = obj, bounds, costs
            cur_energy = en
            if obj < best_obj - 1e-9:
                best_perm = newp[:]
                best_obj = obj
                best_bounds = bounds
                best_costs = costs
                since_best = 0
                if logger:
                    logger.log_solution(best_obj, make_solution(best_perm, best_bounds, best_obj))
            else:
                since_best += 1
        else:
            since_best += 1

        if since_best > 4000:
            cur_perm = best_perm[:]
            cur_obj, cur_bounds, cur_costs = best_obj, best_bounds, best_costs
            cur_energy = energy_of(cur_obj, cur_costs)
            since_best = 0

    sol = make_solution(best_perm, best_bounds, best_obj)
    write_out(sol)


if __name__ == "__main__":
    main()