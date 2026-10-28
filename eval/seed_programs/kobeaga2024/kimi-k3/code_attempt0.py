import argparse
import json
import os
import random
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def tour_cost(tour, dist, depot):
    cost = 0.0
    prev = depot
    for v in tour:
        cost += dist[prev][v]
        prev = v
    cost += dist[prev][depot]
    return cost


def tour_edges(tour, depot):
    edges = []
    prev = depot
    for v in tour:
        edges.append((prev, v))
        prev = v
    if tour:
        edges.append((prev, depot))
    return edges


def make_solution(obj, visited, edges):
    return {
        "objective_value": float(obj),
        "visited_vertices": sorted(int(v) for v in visited),
        "tour_edges": [[int(a), int(b)] for a, b in edges],
    }


def log_incumbent(state, logger, obj, tour, depot):
    if obj <= state["last"] + 1e-9:
        return
    state["last"] = obj
    if logger:
        sol = make_solution(obj, [depot] + list(tour), tour_edges(tour, depot))
        logger.log_solution(float(obj), sol)


# ---------------------------------------------------------------------------
# Heuristic (greedy insertion + 2-opt + iterated local search)
# ---------------------------------------------------------------------------

def fill_tour(tour, in_tour, length, dist, scores, depot, budget, rng,
              randomize, deadline):
    n = len(scores)
    while True:
        if time.time() > deadline:
            break
        m = len(tour)
        best_key = 0.0
        best_v = -1
        best_p = 0
        best_extra = 0.0
        for v in range(n):
            if (v & 15) == 0 and time.time() > deadline:
                return length
            if v == depot or in_tour[v] or scores[v] <= 0.0:
                continue
            dv = dist[v]
            for p in range(m + 1):
                a = depot if p == 0 else tour[p - 1]
                b = depot if p == m else tour[p]
                extra = dist[a][v] + dv[b] - dist[a][b]
                if length + extra <= budget + 1e-9:
                    key = scores[v] / (extra + 1.0)
                    if randomize:
                        key *= rng.uniform(0.6, 1.4)
                    if key > best_key:
                        best_key = key
                        best_v = v
                        best_p = p
                        best_extra = extra
        if best_v < 0:
            break
        tour.insert(best_p, best_v)
        in_tour[best_v] = True
        length += best_extra
    return length


def two_opt(tour, dist, depot, deadline):
    if len(tour) < 3:
        return tour_cost(tour, dist, depot)
    length = tour_cost(tour, dist, depot)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        m = len(tour)
        found = False
        for i in range(m - 1):
            a = depot if i == 0 else tour[i - 1]
            b = tour[i]
            da = dist[a]
            db = dist[b]
            for j in range(i + 1, m):
                c = tour[j]
                d = depot if j == m - 1 else tour[j + 1]
                delta = da[c] + db[d] - da[b] - dist[c][d]
                if delta < -1e-9:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    length += delta
                    improved = True
                    found = True
                    break
            if found:
                break
    return length


def heuristic_phase(dist, scores, depot, budget, deadline, logger, state):
    n = len(scores)
    rng = random.Random(20240)

    in_tour = [False] * n
    in_tour[depot] = True
    tour = []
    length = fill_tour(tour, in_tour, 0.0, dist, scores, depot, budget,
                       rng, False, deadline)
    length = two_opt(tour, dist, depot, deadline)
    fill_tour(tour, in_tour, length, dist, scores, depot, budget,
              rng, False, deadline)

    best = list(tour)
    best_score = sum(scores[v] for v in best)
    log_incumbent(state, logger, best_score, best, depot)

    while time.time() < deadline:
        cur = list(best)
        if cur:
            k = min(max(1, len(cur) // 3), len(cur))
            s = rng.randrange(len(cur))
            for _ in range(k):
                cur.pop(s % len(cur))
        in_t = [False] * n
        in_t[depot] = True
        for v in cur:
            in_t[v] = True
        length = two_opt(cur, dist, depot, deadline)
        length = fill_tour(cur, in_t, length, dist, scores, depot, budget,
                           rng, True, deadline)
        length = two_opt(cur, dist, depot, deadline)
        fill_tour(cur, in_t, length, dist, scores, depot, budget,
                  rng, True, deadline)
        sc = sum(scores[v] for v in cur)
        if sc > best_score + 1e-9:
            best_score = sc
            best = list(cur)
            log_incumbent(state, logger, best_score, best, depot)

    return best, best_score


# ---------------------------------------------------------------------------
# MIP solve (single-commodity flow formulation of the orienteering problem)
# ---------------------------------------------------------------------------

def solve_with_gurobi(dist, scores, depot, budget, t_start, time_limit,
                      incumbent_tour, logger, state):
    n = len(scores)

    # Candidate edges: an edge longer than the budget can never be used.
    edges = []
    for i in range(n):
        di = dist[i]
        for j in range(i + 1, n):
            if di[j] <= budget:
                edges.append((i, j))

    # Node pruning: distances are (rounded) Euclidean, hence nearly metric,
    # so any tour through i has length >= 2 * dist[depot][i] - rounding slack.
    allowed = [True] * n
    for i in range(n):
        if i != depot and 2.0 * dist[depot][i] - 2.0 > budget:
            allowed[i] = False

    # Sparsify extremely large graphs to keep the model tractable.
    if len(edges) > 150000:
        K = 25
        keep = set()
        for i in range(n):
            nbrs = sorted(range(n), key=lambda j: dist[i][j])
            cnt = 0
            for j in nbrs:
                if j == i:
                    continue
                if dist[i][j] > budget:
                    break
                keep.add((min(i, j), max(i, j)))
                cnt += 1
                if cnt >= K:
                    break
        for j in range(n):
            if j != depot and dist[depot][j] <= budget:
                keep.add((min(depot, j), max(depot, j)))
        edges = sorted(keep)

    model = gp.Model("orienteering")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0
    if len(edges) > 20000:
        model.Params.MIPFocus = 1

    x = model.addVars(edges, vtype=GRB.BINARY, name="x")
    y = model.addVars(n, vtype=GRB.BINARY, name="y")
    arcs = edges + [(j, i) for (i, j) in edges]
    f = model.addVars(arcs, lb=0.0, name="f")

    adj = [[] for _ in range(n)]
    for e in edges:
        adj[e[0]].append(e)
        adj[e[1]].append(e)

    # Degree constraints: every visited node has degree 2 (single cycle).
    model.addConstrs(
        (gp.quicksum(x[e] for e in adj[i]) == 2.0 * y[i] for i in range(n)),
        name="degree")

    # Travel-distance budget.
    model.addConstr(
        gp.quicksum(dist[i][j] * x[i, j] for (i, j) in edges) <= budget,
        name="budget")

    out_arcs = [[] for _ in range(n)]
    in_arcs = [[] for _ in range(n)]
    for a in arcs:
        out_arcs[a[0]].append(a)
        in_arcs[a[1]].append(a)

    # Flow conservation: every visited non-depot node consumes one unit,
    # forcing it to be connected to the depot (no disconnected subtours).
    model.addConstrs(
        (gp.quicksum(f[a] for a in in_arcs[v])
         - gp.quicksum(f[a] for a in out_arcs[v]) == y[v]
         for v in range(n) if v != depot),
        name="flow")

    cap = float(max(1, n))
    model.addConstrs((f[i, j] <= cap * x[i, j] for (i, j) in edges),
                     name="cap_ij")
    model.addConstrs((f[j, i] <= cap * x[i, j] for (i, j) in edges),
                     name="cap_ji")

    y[depot].LB = 1.0
    for i in range(n):
        if not allowed[i]:
            y[i].UB = 0.0

    model.setObjective(gp.quicksum(scores[i] * y[i] for i in range(n)),
                       GRB.MAXIMIZE)
    model.update()

    xvars = [x[e] for e in edges]
    yvars = [y[i] for i in range(n)]
    fvars = [f[a] for a in arcs]

    # Provide the heuristic incumbent as a complete MIP start (a tour with
    # fewer than two customers cannot be represented by the formulation).
    if incumbent_tour and len(incumbent_tour) >= 2:
        eset = set()
        prev = depot
        for v in incumbent_tour:
            eset.add((min(prev, v), max(prev, v)))
            prev = v
        eset.add((min(prev, depot), max(prev, depot)))
        visited = set(incumbent_tour)
        visited.add(depot)
        model.setAttr("Start", xvars,
                      [1.0 if e in eset else 0.0 for e in edges])
        model.setAttr("Start", yvars,
                      [1.0 if i in visited else 0.0 for i in range(n)])
        arc_idx = {a: k for k, a in enumerate(arcs)}
        fvals = [0.0] * len(arcs)
        m = len(incumbent_tour)
        prev = depot
        for t, v in enumerate(incumbent_tour):
            k = arc_idx.get((prev, v))
            if k is not None:
                fvals[k] = float(m - t)
            prev = v
        model.setAttr("Start", fvars, fvals)

    def callback(model, where):
        if where != GRB.Callback.MIPSOL or not logger:
            return
        try:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        except Exception:
            return
        if obj <= state["last"] + 1e-9:
            return
        state["last"] = obj
        try:
            xs = model.cbGetSolution(xvars)
            ys = model.cbGetSolution(yvars)
        except Exception:
            try:
                xs = [model.cbGetSolution(v) for v in xvars]
                ys = [model.cbGetSolution(v) for v in yvars]
            except Exception:
                logger.log(float(obj))
                return
        sel_edges = [edges[k] for k, val in enumerate(xs) if val > 0.5]
        sel_nodes = [i for i, val in enumerate(ys) if val > 0.5]
        logger.log_solution(float(obj),
                            make_solution(obj, sel_nodes, sel_edges))

    remaining = time_limit - (time.time() - t_start) - 0.3
    model.Params.TimeLimit = max(0.1, remaining)
    model.optimize(callback)

    if model.SolCount > 0:
        xs = model.getAttr("X", xvars)
        ys = model.getAttr("X", yvars)
        sel_edges = [edges[k] for k, val in enumerate(xs) if val > 0.5]
        sel_nodes = [i for i, val in enumerate(ys) if val > 0.5]
        total_dist = sum(dist[a][b] for (a, b) in sel_edges)
        if total_dist <= budget + 1e-6 and depot in sel_nodes:
            obj = sum(scores[v] for v in sel_nodes)
            return sel_nodes, sel_edges, obj
    return None


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

    t_start = time.time()
    time_limit = float(args.time_limit)

    with open(args.instance_path, "r") as fh:
        inst = json.load(fh)

    n = int(inst["num_nodes"])
    budget = float(inst["distance_limitation_d0"])
    depot = int(inst["depot_vertex"])
    scores = [float(s) for s in inst["scores"]]
    dist = inst["distance_matrix"]

    logger = (SolutionLogger(args.log_path, sense="maximize")
              if args.log_path else None)
    state = {"last": -1e18}

    # Phase 1: fast heuristic to obtain a strong initial incumbent.
    h_budget = min(2.0, max(0.05, 0.10 * time_limit))
    h_deadline = time.time() + h_budget
    best_tour, best_score = heuristic_phase(dist, scores, depot, budget,
                                            h_deadline, logger, state)

    chosen_visited = [depot] + list(best_tour)
    chosen_edges = tour_edges(best_tour, depot)
    chosen_obj = float(best_score)

    # Phase 2: MIP with the remaining time, warm-started by the heuristic.
    remaining = time_limit - (time.time() - t_start) - 0.5
    if remaining > 0.2 and n >= 2:
        try:
            res = solve_with_gurobi(dist, scores, depot, budget, t_start,
                                    time_limit, best_tour, logger, state)
        except Exception:
            res = None
        if res is not None:
            nodes, mEdges, obj = res
            if obj > chosen_obj + 1e-9:
                chosen_visited = nodes
                chosen_edges = mEdges
                chosen_obj = obj

    solution = make_solution(chosen_obj, chosen_visited, chosen_edges)
    out_dir = os.path.dirname(args.solution_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.solution_path, "w") as fh:
        json.dump(solution, fh)


if __name__ == "__main__":
    main()