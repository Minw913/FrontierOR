import argparse
import heapq
import json
import os
import random
import sys
import time

from solution_logger import SolutionLogger


class _Timeout(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description="Maximum weighted clique solver")
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    t0 = time.time()
    time_limit = max(1, int(args.time_limit))
    # reserve a small buffer for writing the output file
    deadline = t0 + time_limit - min(1.0, max(0.05, 0.02 * time_limit))

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    # ---------------- load instance ----------------
    with open(args.instance_path, "r") as f:
        inst = json.load(f)
    g = inst.get("graph", {})
    ids = list(g.get("vertices", []))
    n = int(g.get("num_vertices", len(ids)))
    index = {v: i for i, v in enumerate(ids)}
    vw = inst.get("vertex_weights")
    if vw is None or len(vw) != n:
        w = [(i % 200) + 1 for i in range(n)]
    else:
        w = [int(vw[i]) for i in range(n)]

    adj_set = [set() for _ in range(n)]
    m = 0
    for e in g.get("edges", []):
        a = index[e[0]]
        b = index[e[1]]
        if a == b:
            continue
        if b not in adj_set[a]:
            adj_set[a].add(b)
            adj_set[b].add(a)
            m += 1

    best = {"w": 0, "mask": 0}

    def mask_bits(mask):
        while mask:
            lsb = mask & -mask
            yield lsb.bit_length() - 1
            mask ^= lsb

    def record_mask(total, mask):
        if total > best["w"]:
            best["w"] = total
            best["mask"] = mask
            if logger:
                verts = [ids[i] for i in mask_bits(mask)]
                logger.log_solution(float(total),
                                    {"objective_value": float(total),
                                     "clique_vertices": verts})

    def write_solution():
        verts = [ids[i] for i in mask_bits(best["mask"])]
        out = {"objective_value": float(best["w"]), "clique_vertices": verts}
        d = os.path.dirname(os.path.abspath(args.solution_path))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.solution_path, "w") as f:
            json.dump(out, f)

    if n == 0:
        write_solution()
        return

    adj_list = [sorted(s) for s in adj_set]
    order_w = sorted(range(n), key=lambda v: (-w[v], v))

    try:
        # ---------------- bitmask adjacency ----------------
        BITMASK_LIMIT = 30000
        adj_mask = None
        if n <= BITMASK_LIMIT:
            adj_mask = [0] * n
            for v in range(n):
                mask = 0
                for u in adj_set[v]:
                    mask |= (1 << u)
                adj_mask[v] = mask

        # ---------------- heuristic phase ----------------
        if adj_mask is not None:
            full_mask = (1 << n) - 1

            def greedy(order):
                cand = full_mask
                tot = 0
                clique = 0
                am = adj_mask
                for v in order:
                    if (cand >> v) & 1:
                        clique |= (1 << v)
                        tot += w[v]
                        cand &= am[v]
                return tot, clique

            def local_improve(clique, tot, max_passes=50):
                am = adj_mask
                for _ in range(max_passes):
                    changed = False
                    for v in order_w:
                        vb = 1 << v
                        if clique & vb:
                            continue
                        diff = clique & ~am[v]
                        if diff == 0:
                            clique |= vb
                            tot += w[v]
                            changed = True
                        elif diff & (diff - 1) == 0:
                            u = diff.bit_length() - 1
                            if w[v] > w[u]:
                                clique = (clique | vb) & ~diff
                                tot += w[v] - w[u]
                                changed = True
                    if not changed or time.time() > deadline:
                        break
                return clique, tot

            tot, clq = greedy(order_w)
            clq, tot = local_improve(clq, tot)
            record_mask(tot, clq)

            rng = random.Random(0)
            heur_end = min(deadline, t0 + min(3.0, max(0.2, 0.05 * time_limit)))
            tries = 0
            while time.time() < heur_end and tries < 500:
                tries += 1
                order = sorted(range(n), key=lambda v: -w[v] * (0.5 + rng.random()))
                t2, c2 = greedy(order)
                c2, t2 = local_improve(c2, t2, max_passes=4)
                record_mask(t2, c2)
        else:
            # very large instance fallback: set-based greedy
            def greedy_set(order):
                cand = None
                tot = 0
                clique = []
                for v in order:
                    if cand is None or v in cand:
                        clique.append(v)
                        tot += w[v]
                        if cand is None:
                            cand = set(adj_set[v])
                        else:
                            cand &= adj_set[v]
                return tot, clique

            def mask_of(clique):
                mask = 0
                for v in clique:
                    mask |= (1 << v)
                return mask

            tot, clq = greedy_set(order_w)
            record_mask(tot, mask_of(clq))

        # ---------------- strong optimization ----------------
        non_edges = n * (n - 1) // 2 - m
        ran_strong = False

        # Dense-ish graphs: few non-edges -> Gurobi on conflict constraints.
        if (adj_mask is not None and n <= 4000 and non_edges <= 250000
                and time.time() < deadline):
            try:
                import gurobipy as gp
                from gurobipy import GRB
                ran_strong = True

                model = gp.Model("max_weight_clique")
                model.Params.Seed = 0
                model.Params.MIPGap = 1e-4
                model.Params.NumericFocus = 0
                model.Params.Threads = 1
                model.Params.LogToConsole = 0

                x = model.addVars(n, vtype=GRB.BINARY, obj=w, name="x")
                model.ModelSense = GRB.MAXIMIZE

                fm = (1 << n) - 1
                pairs = []
                for u in range(n):
                    higher = fm ^ ((1 << (u + 1)) - 1)
                    comp = higher & ~adj_mask[u]
                    while comp:
                        lsb = comp & -comp
                        pairs.append((u, lsb.bit_length() - 1))
                        comp ^= lsb
                model.addConstrs((x[u] + x[v] <= 1 for (u, v) in pairs),
                                 name="conflict")

                bm = best["mask"]
                for v in range(n):
                    x[v].Start = 1.0 if (bm >> v) & 1 else 0.0

                def callback(mdl, where):
                    if where == GRB.Callback.MIPSOL:
                        try:
                            vals = mdl.cbGetSolution(x)
                            mask = 0
                            tot = 0
                            for i in range(n):
                                if vals[i] > 0.5:
                                    mask |= (1 << i)
                                    tot += w[i]
                            record_mask(tot, mask)
                        except Exception:
                            pass

                model.Params.TimeLimit = max(0.05, deadline - time.time())
                model.optimize(callback)

                if model.SolCount > 0:
                    vals = model.getAttr("X", x)
                    mask = 0
                    tot = 0
                    for i in range(n):
                        if vals[i] > 0.5:
                            mask |= (1 << i)
                            tot += w[i]
                    record_mask(tot, mask)
                model.dispose()
            except Exception:
                ran_strong = False

        # Sparse graphs: combinatorial branch & bound with weighted coloring.
        if not ran_strong and adj_mask is not None and time.time() < deadline:
            sys.setrecursionlimit(max(100000, n + 100))
            nodes = [0]

            def color_order(cand_mask):
                am = adj_mask
                wl = w
                verts = []
                mm = cand_mask
                while mm:
                    lsb = mm & -mm
                    verts.append(lsb.bit_length() - 1)
                    mm ^= lsb
                verts.sort(key=lambda v: -wl[v])
                order = []
                bounds = []
                colored = 0
                cum = 0
                L = len(verts)
                for i in range(L):
                    v = verts[i]
                    vb = 1 << v
                    if colored & vb:
                        continue
                    class_mask = vb
                    colored |= vb
                    cum += wl[v]  # vertices processed in desc weight order
                    members = [v]
                    for j in range(i + 1, L):
                        u = verts[j]
                        ub = 1 << u
                        if (colored & ub) or (am[u] & class_mask):
                            continue
                        colored |= ub
                        class_mask |= ub
                        members.append(u)
                    order.extend(members)
                    bounds.extend([cum] * len(members))
                return order, bounds

            def expand(cand_mask, cur_w, cur_mask):
                nodes[0] += 1
                if (nodes[0] & 0x3FF) == 0 and time.time() > deadline:
                    raise _Timeout()
                if not cand_mask:
                    if cur_w > best["w"]:
                        record_mask(cur_w, cur_mask)
                    return
                order, bounds = color_order(cand_mask)
                cm = cand_mask
                am = adj_mask
                wl = w
                for i in range(len(order) - 1, -1, -1):
                    if cur_w + bounds[i] <= best["w"]:
                        return
                    v = order[i]
                    vb = 1 << v
                    if cm & vb:
                        expand(cm & am[v], cur_w + wl[v], cur_mask | vb)
                        cm &= ~vb

            def degeneracy_order():
                deg = [len(a) for a in adj_list]
                removed = [False] * n
                heap = [(deg[v], v) for v in range(n)]
                heapq.heapify(heap)
                order = []
                while heap:
                    d, v = heapq.heappop(heap)
                    if removed[v] or d != deg[v]:
                        continue
                    removed[v] = True
                    order.append(v)
                    for u in adj_list[v]:
                        if not removed[u]:
                            deg[u] -= 1
                            heapq.heappush(heap, (deg[u], u))
                return order

            try:
                later = (1 << n) - 1
                am = adj_mask
                wl = w
                for v in degeneracy_order():
                    vb = 1 << v
                    later &= ~vb
                    if time.time() > deadline:
                        break
                    cand = am[v] & later
                    if not cand:
                        if wl[v] > best["w"]:
                            record_mask(wl[v], vb)
                        continue
                    tw = 0
                    mm = cand
                    while mm:
                        lsb = mm & -mm
                        tw += wl[lsb.bit_length() - 1]
                        mm ^= lsb
                    if wl[v] + tw <= best["w"]:
                        continue
                    expand(cand, wl[v], vb)
            except _Timeout:
                pass

        elif not ran_strong and adj_mask is None and time.time() < deadline:
            # randomized local search for very large instances
            rng = random.Random(1)

            def greedy_set2(order):
                cand = None
                tot = 0
                clique = []
                for v in order:
                    if cand is None or v in cand:
                        clique.append(v)
                        tot += w[v]
                        if cand is None:
                            cand = set(adj_set[v])
                        else:
                            cand &= adj_set[v]
                return tot, clique

            def mask_of2(clique):
                mask = 0
                for v in clique:
                    mask |= (1 << v)
                return mask

            while time.time() < deadline:
                order = sorted(range(n), key=lambda v: -w[v] * (0.3 + rng.random()))
                t2, c2 = greedy_set2(order)
                cl = set(c2)
                for _ in range(2):
                    changed = False
                    for v in range(n):
                        if v in cl:
                            continue
                        cnt = 0
                        who = -1
                        for u in cl:
                            if v not in adj_set[u]:
                                cnt += 1
                                who = u
                                if cnt > 1:
                                    break
                        if cnt == 0:
                            cl.add(v)
                            t2 += w[v]
                            changed = True
                        elif cnt == 1 and w[v] > w[who]:
                            cl.discard(who)
                            cl.add(v)
                            t2 += w[v] - w[who]
                            changed = True
                    if not changed or time.time() > deadline:
                        break
                record_mask(t2, mask_of2(list(cl)))
    except Exception:
        pass
    finally:
        write_solution()


if __name__ == "__main__":
    main()