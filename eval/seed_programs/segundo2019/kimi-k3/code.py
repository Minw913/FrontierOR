#!/usr/bin/env python3
"""
Maximum Edge-Weight Clique Problem (MEWCP) solver.

Select a clique maximizing the sum of weights of edges with both endpoints
selected.  Strategy:
  1. A fast randomized greedy construction + add/swap local-search heuristic
     produces a strong incumbent (used as a MIP start and as a fallback).
  2. Gurobi branch-and-bound on the standard linearized edge formulation:
        maximize  sum_e w_e * y_e
        s.t.      y_e <= x_u,  y_e <= x_v          for every edge e = (u,v)
                  x_u + x_v <= 1                   for every non-edge (u,v)
     Non-edge (conflict) constraints are added up-front when the complement
     graph is small, otherwise they are separated lazily through a callback.

Every improved incumbent is reported to the provided SolutionLogger.
"""

import argparse
import json
import random
import time


def parse_args():
    p = argparse.ArgumentParser(description="Maximum edge-weight clique solver")
    p.add_argument("--instance_path", required=True,
                   help="Path to the JSON problem instance.")
    p.add_argument("--solution_path", required=True,
                   help="Path where the final solution JSON is written.")
    p.add_argument("--time_limit", type=int, required=True,
                   help="Maximum runtime in seconds.")
    p.add_argument("--log_path", default=None,
                   help="Optional path to a JSONL convergence log.")
    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.time()
    time_limit = max(int(args.time_limit), 0)

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="maximize")
        except Exception:
            logger = None

    # ------------------------------------------------------------------ #
    # Read instance
    # ------------------------------------------------------------------ #
    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    g = instance["graph"]
    vertices = list(g.get("vertices", []))
    raw_edges = g.get("edges", [])
    n = len(vertices)
    index_of = {v: i for i, v in enumerate(vertices)}

    eu, ev, ew = [], [], []           # edge endpoint indices / weights
    nbr = [dict() for _ in range(n)]  # nbr[i][j] = weight (i,j)
    for e in raw_edges:
        u, v = e[0], e[1]
        w = float(e[2])
        i, j = index_of[u], index_of[v]
        if i == j:
            continue
        eu.append(i)
        ev.append(j)
        ew.append(w)
        nbr[i][j] = nbr[i].get(j, 0.0) + w
        nbr[j][i] = nbr[j].get(i, 0.0) + w
    m = len(eu)

    best = {"obj": -1.0, "members": []}   # members = list of vertex indices

    # ------------------------------------------------------------------ #
    # Solution projection / bookkeeping
    # ------------------------------------------------------------------ #
    def build_output(members_idx):
        idset = {vertices[i] for i in members_idx}
        clique_edges = []
        obj = 0.0
        for e in raw_edges:
            u, v = e[0], e[1]
            if u != v and u in idset and v in idset:
                a, b = (u, v) if u < v else (v, u)
                clique_edges.append([a, b])
                obj += float(e[2])
        clique_edges.sort(key=lambda p: (p[0], p[1]))
        sol = {
            "objective_value": float(obj),
            "clique_vertices": sorted(idset),
            "clique_edges": clique_edges,
        }
        return sol, obj

    def record(members_idx):
        """Update incumbent if improved and log it."""
        sol, obj = build_output(members_idx)
        if obj > best["obj"] + 1e-9:
            best["obj"] = obj
            best["members"] = list(members_idx)
            if logger is not None:
                try:
                    logger.log_solution(obj, sol)
                except Exception:
                    pass
            return True
        return False

    # ------------------------------------------------------------------ #
    # Heuristic: randomized greedy + add/swap local search
    # ------------------------------------------------------------------ #
    rng = random.Random(20240)

    def greedy(start_v, randomized, deadline):
        clique = [start_v]
        cand = dict(nbr[start_v])  # candidate -> total weight to clique
        while cand:
            if randomized:
                total = 0.0
                for val in cand.values():
                    total += val
                r = rng.random() * total
                acc = 0.0
                c = None
                for u, val in cand.items():
                    acc += val
                    if acc >= r:
                        c = u
                        break
                if c is None:
                    c = next(iter(cand))
            else:
                c = max(cand, key=cand.get)
            del cand[c]
            nc = nbr[c]
            for u in list(cand):
                w = nc.get(u)
                if w is None:
                    del cand[u]
                else:
                    cand[u] += w
            clique.append(c)
            if (len(clique) & 15) == 0 and time.time() > deadline:
                break
        return clique

    def local_improve(clique, deadline):
        """First-improvement-style best add / 1-swap local search."""
        in_c = bytearray(n)
        members = list(clique)
        for v in members:
            in_c[v] = 1
        k = len(members)
        gain = [0.0] * n   # non-member: weight to current clique
        msum = [0.0] * n   # member: weight to other members
        adjc = [0] * n     # non-member: # of adjacent members
        cnt = [0] * n      # non-member: # of non-adjacent members
        last = [-1] * n    # non-member: a non-adjacent member (if cnt==1)
        for v in members:
            s = 0.0
            for u, w in nbr[v].items():
                if in_c[u]:
                    s += w
                else:
                    gain[u] += w
                    adjc[u] += 1
            msum[v] = s
        for u in range(n):
            if not in_c[u]:
                cnt[u] = k - adjc[u]

        def add(v):
            s = 0.0
            nv = nbr[v]
            for u, w in nv.items():
                if in_c[u]:
                    msum[u] += w
                    s += w
                else:
                    gain[u] += w
            msum[v] = s
            in_c[v] = 1
            members.append(v)
            for u in range(n):
                if not in_c[u] and u not in nv:
                    cnt[u] += 1
                    last[u] = v

        def remove(v):
            in_c[v] = 0
            members.remove(v)
            nv = nbr[v]
            for u, w in nv.items():
                if in_c[u]:
                    msum[u] -= w
                else:
                    gain[u] -= w
            msum[v] = 0.0
            for u in range(n):
                if not in_c[u] and u not in nv:
                    cnt[u] -= 1
                    if last[u] == v:
                        last[u] = -1
            cv, gv, lv = 0, 0.0, -1
            for mem in members:
                w = nv.get(mem)
                if w is None:
                    cv += 1
                    lv = mem
                else:
                    gv += w
            cnt[v], gain[v], last[v] = cv, gv, lv

        eps = 1e-9
        it = 0
        while True:
            it += 1
            if (it & 7) == 0 and time.time() > deadline:
                break
            best_gain = eps
            add_u, swap_u, swap_v = -1, -1, -1
            for u in range(n):
                if in_c[u]:
                    continue
                c = cnt[u]
                if c == 0:
                    gu = gain[u]
                    if gu > best_gain:
                        best_gain = gu
                        add_u, swap_u = u, -1
                elif c == 1:
                    lv = last[u]
                    if lv < 0:
                        nm = nbr[u]
                        for mem in members:
                            if mem not in nm:
                                lv = mem
                                break
                        last[u] = lv
                    sg = gain[u] - msum[lv]
                    if sg > best_gain:
                        best_gain = sg
                        swap_u, swap_v, add_u = u, lv, -1
            if add_u >= 0:
                add(add_u)
            elif swap_u >= 0:
                remove(swap_v)
                add(swap_u)
            else:
                break
        return members

    # ------------------------------------------------------------------ #
    # Budget split between heuristic and exact solve
    # ------------------------------------------------------------------ #
    use_gurobi = (m > 0) and (m <= 300000) and (n <= 20000)
    if use_gurobi:
        heur_budget = min(max(0.2 * time_limit, 0.1), 10.0)
    else:
        heur_budget = max(time_limit - 0.5, 0.05)
    heur_deadline = t_start + heur_budget

    if n > 0:
        incw = [0.0] * n
        for e in range(m):
            incw[eu[e]] += ew[e]
            incw[ev[e]] += ew[e]
        order = sorted(range(n), key=lambda i: -incw[i])
        tries = 0
        while True:
            if tries < min(len(order), 8):
                start_v = order[tries]
                randomized = False
            else:
                start_v = order[rng.randrange(min(len(order), 50))]
                randomized = True
            clique = greedy(start_v, randomized, heur_deadline)
            if n <= 10000:
                clique = local_improve(clique, heur_deadline)
            record(clique)
            tries += 1
            if time.time() > heur_deadline:
                break
    else:
        record([])

    # ------------------------------------------------------------------ #
    # Gurobi branch and bound
    # ------------------------------------------------------------------ #
    if use_gurobi:
        remaining = time_limit - (time.time() - t_start) - 0.5
        if remaining > 0.2:
            try:
                import gurobipy as gp
                from gurobipy import GRB

                model = gp.Model("mewcp")
                model.Params.Seed = 0
                model.Params.MIPGap = 1e-4
                model.Params.NumericFocus = 0
                model.Params.Threads = 1

                x = model.addVars(n, vtype=GRB.BINARY, name="x")
                y = model.addVars(m, vtype=GRB.BINARY, name="y")
                model.setObjective(
                    gp.quicksum(ew[e] * y[e] for e in range(m)),
                    GRB.MAXIMIZE,
                )
                model.addConstrs((y[e] <= x[eu[e]]) for e in range(m))
                model.addConstrs((y[e] <= x[ev[e]]) for e in range(m))

                nonedges = n * (n - 1) // 2 - m
                lazy = nonedges > 250000
                if lazy:
                    model.Params.LazyConstraints = 1
                else:
                    for i in range(n):
                        ni = nbr[i]
                        for j in range(i + 1, n):
                            if j not in ni:
                                model.addConstr(x[i] + x[j] <= 1)

                model.update()

                # MIP start from the heuristic incumbent.
                inset = set(best["members"])
                for i in range(n):
                    x[i].Start = 1.0 if i in inset else 0.0
                for e in range(m):
                    y[e].Start = 1.0 if (eu[e] in inset and ev[e] in inset) else 0.0

                xlist = [x[i] for i in range(n)]

                def callback(model, where):
                    if where == GRB.Callback.MIPSOL:
                        try:
                            xsol = model.cbGetSolution(xlist)
                        except Exception:
                            return
                        sel = [i for i in range(n) if xsol[i] > 0.5]
                        if lazy:
                            ncuts = 0
                            for a in range(len(sel)):
                                ia = sel[a]
                                na = nbr[ia]
                                for b in range(a + 1, len(sel)):
                                    ib = sel[b]
                                    if ib not in na:
                                        model.cbLazy(x[ia] + x[ib] <= 1)
                                        ncuts += 1
                                        if ncuts >= 50:
                                            break
                                if ncuts >= 50:
                                    break
                            if ncuts:
                                return
                        try:
                            record(sel)
                        except Exception:
                            pass

                remaining2 = time_limit - (time.time() - t_start) - 0.4
                if remaining2 > 0.05:
                    model.Params.TimeLimit = max(remaining2, 0.05)
                    model.optimize(callback)
                    if model.SolCount > 0:
                        xv = model.getAttr("X", xlist)
                        sel = [i for i in range(n) if xv[i] > 0.5]
                        try:
                            record(sel)
                        except Exception:
                            pass
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Write final solution
    # ------------------------------------------------------------------ #
    sol, _ = build_output(best["members"])
    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()