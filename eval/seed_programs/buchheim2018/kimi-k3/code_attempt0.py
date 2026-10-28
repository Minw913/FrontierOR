#!/usr/bin/env python3
"""
Quadratic Shortest Path Problem (QSPP) solver.

Model:
    min  c^T x + x^T Q x          (x binary, one entry per directed arc)
    s.t. flow conservation (unit flow from source to target),
         in/out degree <= 1 at every intermediate node,
         no disconnected directed cycles (added as cuts, iteratively).

The quadratic term x^T Q x sums over ALL ordered pairs (i, j), matching the
problem statement (symmetric Q, zero diagonal, so each unordered pair of two
distinct arcs contributes twice its matrix entry).

Strategy:
  1. Fast feasible solution via Dijkstra on linear costs.
  2. A few rounds of a linearized marginal-cost shortest-path improvement.
  3. Exact MIQP solve with Gurobi (binary products are linearized by Gurobi),
     incumbent logging through a callback, and iterative cycle-elimination
     cuts so that the returned solution is a single directed path.
"""

import argparse
import heapq
import json
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser(description="Quadratic shortest path solver")
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    t_start = time.time()
    time_limit = float(args.time_limit)

    # ------------------------------------------------------------------ #
    # Load instance
    # ------------------------------------------------------------------ #
    with open(args.instance_path, "r") as f:
        data = json.load(f)

    nodes = data["nodes"]
    arcs = data["arcs"]
    n = len(nodes)
    m = len(arcs)

    node_ids = [nd["id"] for nd in nodes]
    nid2idx = {nid: k for k, nid in enumerate(node_ids)}

    arc_ids = [a["id"] for a in arcs]
    a_from_raw = [a["from_node"] for a in arcs]
    a_to_raw = [a["to_node"] for a in arcs]
    a_from = [nid2idx[v] for v in a_from_raw]
    a_to = [nid2idx[v] for v in a_to_raw]

    source = nid2idx[data["source_node"]]
    target = nid2idx[data["target_node"]]

    cvec = np.asarray(data["linear_costs"], dtype=float)
    Qmat = np.asarray(data["quadratic_costs"], dtype=float)

    out_arcs = [[] for _ in range(n)]
    in_arcs = [[] for _ in range(n)]
    for i in range(m):
        out_arcs[a_from[i]].append(i)
        in_arcs[a_to[i]].append(i)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def path_objective(path):
        """Exact objective: linear costs + all ordered quadratic pairs."""
        if not path:
            return 0.0
        idx = np.asarray(path, dtype=int)
        return float(cvec[idx].sum() + Qmat[np.ix_(idx, idx)].sum())

    def build_solution(path, obj):
        return {
            "objective_value": float(obj),
            "solution_arcs": [
                {
                    "id": int(arc_ids[i]),
                    "from_node": int(a_from_raw[i]),
                    "to_node": int(a_to_raw[i]),
                }
                for i in path
            ],
        }

    best = {"obj": float("inf"), "path": None}

    def maybe_update(path):
        """Record path if it improves the incumbent; log it."""
        if path is None:
            return False
        obj = path_objective(path)
        if obj < best["obj"] - 1e-9:
            best["obj"] = obj
            best["path"] = list(path)
            if logger:
                logger.log_solution(obj, build_solution(path, obj))
            return True
        return False

    def shortest_path(costs):
        """Dijkstra on (clipped) arc costs; returns list of arc indices."""
        costs = np.maximum(np.asarray(costs, dtype=float), 0.0)
        INF = float("inf")
        dist = [INF] * n
        prev_arc = [-1] * n
        dist[source] = 0.0
        pq = [(0.0, source)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u] + 1e-12:
                continue
            if u == target:
                break
            for a in out_arcs[u]:
                v = a_to[a]
                nd = d + costs[a]
                if nd < dist[v] - 1e-12:
                    dist[v] = nd
                    prev_arc[v] = a
                    heapq.heappush(pq, (nd, v))
        if dist[target] == INF:
            return None
        path = []
        u = target
        while u != source:
            a = prev_arc[u]
            if a < 0:
                return None
            path.append(a)
            u = a_from[a]
        path.reverse()
        return path

    def extract_path(xvals):
        """Extract the simple source->target path from selected arcs.

        With the degree/flow constraints the selected arcs form one
        source-target path plus node-disjoint cycles; walking from the
        source yields the path. Returns (path_arc_ids, selected_arc_ids).
        """
        sel = [i for i in range(m) if xvals[i] > 0.5]
        outsel = {}
        for i in sel:
            outsel.setdefault(a_from[i], []).append(i)
        path = []
        node = source
        steps = 0
        while node != target and steps <= m + 1:
            nxts = outsel.get(node)
            if not nxts:
                return None, sel
            a = nxts[0]
            path.append(a)
            node = a_to[a]
            steps += 1
        if node != target:
            return None, sel
        return path, sel

    def find_cycles(leftover):
        """Decompose leftover balanced arcs into directed cycles."""
        out_map = {}
        for i in leftover:
            out_map.setdefault(a_from[i], []).append(i)
        pool = set(leftover)
        cycles = []
        while pool:
            a0 = pool.pop()
            cyc = [a0]
            start_node = a_from[a0]
            node = a_to[a0]
            guard = 0
            limit = len(leftover) + 2
            while node != start_node and guard <= limit:
                nxt = None
                for a in out_map.get(node, ()):
                    if a in pool:
                        nxt = a
                        break
                if nxt is None:
                    break
                cyc.append(nxt)
                pool.discard(nxt)
                node = a_to[nxt]
                guard += 1
            if cyc:
                cycles.append(cyc)
        return cycles

    # ------------------------------------------------------------------ #
    # 1) Quick feasible solution (Dijkstra on linear costs)
    # 2) Linearized marginal-cost improvement rounds
    # ------------------------------------------------------------------ #
    if source == target:
        best["obj"] = 0.0
        best["path"] = []
        if logger:
            logger.log_solution(0.0, build_solution([], 0.0))
    else:
        p0 = shortest_path(cvec)
        if p0 is not None:
            maybe_update(p0)
            cur = list(best["path"])
            heur_budget = min(5.0, max(0.5, 0.15 * time_limit))
            for _ in range(25):
                if time.time() - t_start > heur_budget:
                    break
                idx = np.asarray(cur, dtype=int)
                grad = cvec + 2.0 * Qmat[:, idx].sum(axis=1)
                p1 = shortest_path(grad)
                if p1 is None or p1 == cur:
                    break
                if maybe_update(p1):
                    cur = list(p1)
                else:
                    break

    # ------------------------------------------------------------------ #
    # 3) Exact MIQP solve with Gurobi + iterative cycle elimination
    # ------------------------------------------------------------------ #
    try:
        remaining = time_limit - (time.time() - t_start)
        if remaining > 0.2 and m > 0:
            model = gp.Model("QSPP")
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.OutputFlag = 0
            model.Params.NonConvex = 2

            X = model.addMVar(shape=m, vtype=GRB.BINARY, name="x")

            # Flow conservation: out - in = b.
            b = np.zeros(n)
            b[source] = 1.0
            b[target] = -1.0
            for v in range(n):
                lhs = gp.LinExpr()
                out = out_arcs[v]
                inn = in_arcs[v]
                if out:
                    lhs.addTerms([1.0] * len(out), [X[i] for i in out])
                if inn:
                    lhs.addTerms([-1.0] * len(inn), [X[i] for i in inn])
                model.addConstr(lhs == float(b[v]), name="flow_%d" % v)

            # Degree constraints enforce a single simple path.
            if in_arcs[source]:
                model.addConstr(gp.quicksum(X[i] for i in in_arcs[source]) <= 0.0)
            if out_arcs[target]:
                model.addConstr(gp.quicksum(X[i] for i in out_arcs[target]) <= 0.0)
            for v in range(n):
                if v == source or v == target:
                    continue
                if in_arcs[v]:
                    model.addConstr(gp.quicksum(X[i] for i in in_arcs[v]) <= 1.0)
                if out_arcs[v]:
                    model.addConstr(gp.quicksum(X[i] for i in out_arcs[v]) <= 1.0)

            # Objective: c^T x + x^T Q x (all ordered pairs).
            model.setObjective(cvec @ X + X @ Qmat @ X, GRB.MINIMIZE)

            # MIP start from the heuristic solution.
            if best["path"] is not None:
                start_vec = np.zeros(m)
                if best["path"]:
                    start_vec[np.asarray(best["path"], dtype=int)] = 1.0
                try:
                    X.setAttr("Start", start_vec)
                except Exception:
                    for i in range(m):
                        X[i].Start = float(start_vec[i])

            def incumbent_cb(mdl, where):
                if where == GRB.Callback.MIPSOL:
                    try:
                        xv = mdl.cbGetSolution(mdl._X)
                        path, _sel = extract_path(np.asarray(xv, dtype=float))
                        if path is None:
                            return
                        if source != target and not path:
                            return
                        maybe_update(path)
                    except Exception:
                        pass

            model._X = X

            for _ in range(500):
                rem = time_limit - (time.time() - t_start)
                if rem <= 0.05:
                    break
                model.Params.TimeLimit = rem
                model.optimize(incumbent_cb)

                if model.SolCount == 0:
                    break

                xv = np.asarray(X.getAttr("X"), dtype=float)
                path, sel = extract_path(xv)
                if path is None:
                    break
                maybe_update(path)

                leftover = sorted(set(sel) - set(path))
                if not leftover:
                    break  # incumbent is already a single path

                cycles = find_cycles(leftover)
                if not cycles:
                    break
                for cyc in cycles:
                    cidx = np.asarray(cyc, dtype=int)
                    model.addConstr(X[cidx].sum() <= len(cyc) - 1)
                model.update()
    except Exception:
        pass

    # ------------------------------------------------------------------ #
    # Write final solution
    # ------------------------------------------------------------------ #
    if best["path"] is None:
        final = {"objective_value": 0.0, "solution_arcs": []}
    else:
        final = build_solution(best["path"], best["obj"])

    with open(args.solution_path, "w") as f:
        json.dump(final, f, indent=2)


if __name__ == "__main__":
    main()