import argparse
import json
import time
import random
import sys

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    time_limit = max(1, args.time_limit)
    deadline = start_time + time_limit - 1.0  # reserve time for output writing

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    graph = data["graph"]
    n = graph["num_vertices"]
    vertices = graph["vertices"]
    edges = graph["edges"]
    weights_in = data["vertex_weights"]

    if n == 0:
        sol = {"objective_value": 0.0, "clique_vertices": []}
        if logger:
            logger.log_solution(0.0, sol)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    # Map vertex id -> original index
    id_to_orig = {vid: i for i, vid in enumerate(vertices)}

    # Sort vertices by weight descending; assign bit positions so that
    # lowest set bit = heaviest vertex.
    order = sorted(range(n), key=lambda i: -weights_in[i])
    pos_of_orig = [0] * n
    for pos, oi in enumerate(order):
        pos_of_orig[oi] = pos
    w = [weights_in[oi] for oi in order]          # weight by position
    ids = [vertices[oi] for oi in order]          # original id by position

    # Adjacency bitsets
    adj = [0] * n
    m = 0
    for e in edges:
        u = pos_of_orig[id_to_orig[e[0]]]
        v = pos_of_orig[id_to_orig[e[1]]]
        if u == v:
            continue
        if not ((adj[u] >> v) & 1):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            m += 1

    best_w = 0
    best_clique = []  # list of positions

    def make_solution(clique_positions, obj):
        return {
            "objective_value": float(obj),
            "clique_vertices": [ids[p] for p in clique_positions],
        }

    def record(clique_positions, obj):
        nonlocal best_w, best_clique
        if obj > best_w:
            best_w = obj
            best_clique = list(clique_positions)
            if logger:
                logger.log_solution(float(obj), make_solution(best_clique, obj))

    rng = random.Random(0)

    def lowest_bits(mask, k):
        """Return up to k lowest set bit positions of mask."""
        out = []
        while mask and len(out) < k:
            b = mask & -mask
            out.append(b.bit_length() - 1)
            mask ^= b
        return out

    def greedy_from(start):
        clique = [start]
        cw = w[start]
        cand = adj[start]
        while cand:
            if rng.random() < 0.7:
                v = (cand & -cand).bit_length() - 1
            else:
                opts = lowest_bits(cand, 8)
                v = opts[rng.randrange(len(opts))]
            clique.append(v)
            cw += w[v]
            cand &= adj[v]
        return clique, cw

    def greedy_fill(mask, clique, cw):
        while mask:
            v = (mask & -mask).bit_length() - 1
            clique.append(v)
            cw += w[v]
            mask &= adj[v]
        return clique, cw

    def local_improve(clique, cw):
        improved = True
        passes = 0
        while improved and passes < 4:
            improved = False
            passes += 1
            k = len(clique)
            if k == 0:
                break
            # prefix/suffix intersection of adjacency
            full = (1 << n) - 1
            pref = [full] * (k + 1)
            for i in range(k):
                pref[i + 1] = pref[i] & adj[clique[i]]
            suf = [full] * (k + 1)
            for i in range(k - 1, -1, -1):
                suf[i] = suf[i + 1] & adj[clique[i]]
            for i in range(k):
                u = clique[i]
                mask = pref[i] & suf[i + 1]
                mask &= ~(1 << u)
                for c in clique:
                    mask &= ~(1 << c)
                if not mask:
                    continue
                new_clique = clique[:i] + clique[i + 1:]
                new_cw = cw - w[u]
                new_clique, new_cw = greedy_fill(mask, new_clique, new_cw)
                if new_cw > cw:
                    clique = new_clique
                    cw = new_cw
                    improved = True
                    break
        return clique, cw

    # ---------- Initial heuristic phase ----------
    complement_pairs = n * (n - 1) // 2 - m
    use_gurobi = complement_pairs <= 3_000_000
    use_bnb = (not use_gurobi) and n <= 20000

    if use_gurobi:
        heur_end = min(deadline, start_time + max(2.0, 0.10 * time_limit))
    elif use_bnb:
        heur_end = min(deadline, start_time + max(2.0, 0.25 * time_limit))
    else:
        heur_end = deadline

    # deterministic greedy from heaviest vertices first
    tried = 0
    for start in range(min(n, 200)):
        if time.time() > heur_end:
            break
        cand = adj[start]
        clique = [start]
        cw = w[start]
        clique, cw = greedy_fill(cand, clique, cw)
        clique, cw = local_improve(clique, cw)
        record(clique, cw)
        tried += 1

    # randomized multi-start
    while time.time() < heur_end:
        start = rng.randrange(n)
        clique, cw = greedy_from(start)
        clique, cw = local_improve(clique, cw)
        record(clique, cw)

    # ---------- Exact / near-exact phase ----------
    if use_gurobi and time.time() < deadline - 1.0:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            model = gp.Model("mwc")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            remaining = deadline - time.time()
            model.Params.TimeLimit = max(1.0, remaining)

            x = model.addVars(n, vtype=GRB.BINARY)
            model.setObjective(
                gp.quicksum(w[i] * x[i] for i in range(n)), GRB.MAXIMIZE
            )

            full = (1 << n) - 1
            for i in range(n):
                # non-neighbors with position > i
                nonadj = (~adj[i]) & full
                nonadj &= ~((1 << (i + 1)) - 1)
                mask = nonadj
                while mask:
                    b = mask & -mask
                    j = b.bit_length() - 1
                    mask ^= b
                    model.addConstr(x[i] + x[j] <= 1)

            # warm start
            in_best = set(best_clique)
            for i in range(n):
                x[i].Start = 1.0 if i in in_best else 0.0

            xlist = [x[i] for i in range(n)]

            def cb(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj > best_w:
                        vals = model.cbGetSolution(xlist)
                        clq = [i for i in range(n) if vals[i] > 0.5]
                        record(clq, int(round(obj)))

            model.optimize(cb)

            if model.SolCount > 0:
                vals = model.getAttr("X", xlist)
                clq = [i for i in range(n) if vals[i] > 0.5]
                obj = sum(w[i] for i in clq)
                record(clq, obj)
        except Exception:
            # fall back to heuristic result
            pass

    elif use_bnb:
        # Branch and bound with bitsets
        sys.setrecursionlimit(10000)
        check_counter = [0]

        class TimeUp(Exception):
            pass

        def wsum(mask):
            s = 0
            while mask:
                b = mask & -mask
                s += w[b.bit_length() - 1]
                mask ^= b
            return s

        cur_clique = []

        def expand(cw, cand):
            check_counter[0] += 1
            if (check_counter[0] & 255) == 0 and time.time() > deadline:
                raise TimeUp()
            while cand:
                if cw + wsum(cand) <= best_w:
                    return
                b = cand & -cand
                v = b.bit_length() - 1
                cand ^= b
                nc = cw + w[v]
                cur_clique.append(v)
                if nc > best_w:
                    record(cur_clique, nc)
                newcand = cand & adj[v]
                if newcand:
                    expand(nc, newcand)
                cur_clique.pop()

        try:
            full = (1 << n) - 1
            expand(0, full)
        except TimeUp:
            pass
        except RecursionError:
            pass

    else:
        # very large sparse graph: keep improving with heuristics until deadline
        while time.time() < deadline:
            start = rng.randrange(n)
            clique, cw = greedy_from(start)
            clique, cw = local_improve(clique, cw)
            record(clique, cw)

    # Validate the best clique (safety)
    ok = True
    for i in range(len(best_clique)):
        for j in range(i + 1, len(best_clique)):
            u, v = best_clique[i], best_clique[j]
            if not ((adj[u] >> v) & 1):
                ok = False
                break
        if not ok:
            break
    if not ok:
        # fall back to a single heaviest vertex
        best_clique = [0]
        best_w = w[0]

    sol = make_solution(best_clique, best_w)
    if logger:
        logger.log_solution(float(best_w), sol)
    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()