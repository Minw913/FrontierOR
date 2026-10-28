import argparse
import itertools
import json
import math
import random
import time

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def _now():
    return time.perf_counter()


def first_not_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


class _Emitter:
    """Thin wrapper around SolutionLogger that remembers the last (best)
    incumbent it emitted, so a crash-safe fallback can reuse it."""

    def __init__(self, logger):
        self.logger = logger
        self.best_obj = None
        self.best_dict = None

    def __call__(self, objective_value, solution_dict):
        self.best_obj = float(objective_value)
        self.best_dict = solution_dict
        try:
            self.logger.log_solution(float(objective_value), solution_dict)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Gurobi helpers
# ---------------------------------------------------------------------------

_GUROBI_OK = None


def _gurobi_available():
    global _GUROBI_OK
    if _GUROBI_OK is None:
        try:
            import gurobipy  # noqa: F401
            _GUROBI_OK = True
        except Exception:
            _GUROBI_OK = False
    return _GUROBI_OK


def _set_gurobi_params(model, time_budget=None):
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    try:
        model.Params.LogToConsole = 0
    except Exception:
        pass
    if time_budget is not None:
        model.Params.TimeLimit = max(0.05, float(time_budget))


# ---------------------------------------------------------------------------
# Instance detection / parsing
# ---------------------------------------------------------------------------

def detect_problem(inst):
    params = inst.get("parameters") or {}
    for holder in (inst, params):
        if not isinstance(holder, dict):
            continue
        for key in ("problem_type", "problem", "type", "problem_class"):
            v = holder.get(key)
            if isinstance(v, str):
                lv = v.lower()
                if "tsp" in lv or "travel" in lv or "travell" in lv or "salesman" in lv:
                    return "TSP"
                if "cut" in lv:
                    return "MAXCUT"
    data = inst.get("data") or {}
    if (params.get("n_cities") is not None or inst.get("n_cities") is not None
            or data.get("distance_matrix") is not None or inst.get("distance_matrix") is not None
            or data.get("coordinates") is not None or inst.get("coordinates") is not None):
        return "TSP"
    return "MAXCUT"


def parse_tsp(inst):
    params = inst.get("parameters") or {}
    data = inst.get("data") or {}
    D = first_not_none(data.get("distance_matrix"), inst.get("distance_matrix"))
    coords = first_not_none(data.get("coordinates"), inst.get("coordinates"))
    n = first_not_none(params.get("n_cities"), inst.get("n_cities"))
    if D is None and coords is not None:
        m = len(coords)
        D = [[0.0] * m for _ in range(m)]
        for i in range(m):
            xi, yi = coords[i][0], coords[i][1]
            for j in range(i + 1, m):
                xj, yj = coords[j][0], coords[j][1]
                d = math.hypot(xi - xj, yi - yj)
                D[i][j] = d
                D[j][i] = d
    if n is None:
        n = len(D) if D is not None else (len(coords) if coords is not None else 0)
    D = [[float(v) for v in row] for row in D]
    return int(n), D


def parse_maxcut(inst):
    params = inst.get("parameters") or {}
    data = inst.get("data") or {}

    edges_raw = first_not_none(
        data.get("edges"), data.get("edge_list"), data.get("links"),
        inst.get("edges"), inst.get("edge_list"), inst.get("links"))
    weights_raw = first_not_none(data.get("weights"), inst.get("weights"))
    adj = first_not_none(
        data.get("adjacency_matrix"), data.get("adjacency"),
        inst.get("adjacency_matrix"), inst.get("adjacency"))
    nodes_param = params.get("nodes")
    n = first_not_none(
        params.get("n_nodes"), params.get("num_nodes"), params.get("n"),
        nodes_param if isinstance(nodes_param, int) else None,
        data.get("n_nodes"), data.get("num_nodes"),
        inst.get("n_nodes"), inst.get("num_nodes"), inst.get("n"))

    edges = []
    if edges_raw is not None:
        for idx, e in enumerate(edges_raw):
            if isinstance(e, dict):
                i = first_not_none(e.get("source"), e.get("u"), e.get("i"),
                                   e.get("from"), e.get("tail"))
                j = first_not_none(e.get("target"), e.get("v"), e.get("j"),
                                   e.get("to"), e.get("head"))
                w = first_not_none(e.get("weight"), e.get("w"), e.get("value"))
                if w is None:
                    w = 1
            else:
                i, j = e[0], e[1]
                if len(e) > 2:
                    w = e[2]
                elif weights_raw is not None and idx < len(weights_raw):
                    w = weights_raw[idx]
                else:
                    w = 1
            edges.append((int(i), int(j), float(w)))
    elif adj is not None:
        rows = list(adj)
        is_matrix = all(isinstance(r, (list, tuple)) and len(r) == len(rows) for r in rows)
        if is_matrix:
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    w = rows[i][j]
                    if w:
                        edges.append((i, j, float(w)))
        else:
            for i, nbrs in enumerate(rows):
                for item in nbrs:
                    if isinstance(item, (list, tuple)):
                        j = int(item[0])
                        w = float(item[1]) if len(item) > 1 else 1.0
                    else:
                        j = int(item)
                        w = 1.0
                    if i < j:
                        edges.append((i, j, w))

    # normalize: undirected, deduplicated (weights summed), no self-loops
    agg = {}
    for (i, j, w) in edges:
        if i == j:
            continue
        if i > j:
            i, j = j, i
        key = (i, j)
        agg[key] = agg.get(key, 0.0) + w
    edges = [(i, j, w) for (i, j), w in agg.items()]

    if edges:
        mn = min(min(i, j) for (i, j, _) in edges)
        mx = max(max(i, j) for (i, j, _) in edges)
        if n is None:
            n = mx + 1
        if mn >= 1 and mx == n:  # labels are 1..n -> shift to 0-based
            edges = [(i - 1, j - 1, w) for (i, j, w) in edges]
    if n is None:
        n = 0
    return int(n), edges


# ---------------------------------------------------------------------------
# TSP
# ---------------------------------------------------------------------------

def tour_cost(D, tour):
    n = len(tour)
    c = 0.0
    for k in range(n):
        c += D[tour[k]][tour[(k + 1) % n]]
    return c


def tsp_solution_dict(n, tour):
    pos = [0] * n
    for t, c in enumerate(tour):
        pos[c] = t
    d = {}
    for c in range(n):
        pc = pos[c]
        for t in range(n):
            d["x_%d_%d" % (c, t)] = 1 if t == pc else 0
    return d


def _nearest_neighbor(n, D, start):
    unvisited = set(range(n))
    unvisited.discard(start)
    tour = [start]
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: D[cur][j])
        tour.append(nxt)
        unvisited.discard(nxt)
        cur = nxt
    return tour


def _two_opt(D, tour, deadline):
    n = len(tour)
    if n < 4:
        return
    while _now() < deadline:
        found = False
        for i in range(n - 1):
            if found:
                break
            a = tour[i]
            b = tour[i + 1]
            dab = D[a][b]
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                c = tour[j]
                d = tour[(j + 1) % n]
                if D[a][c] + D[b][d] < dab + D[c][d] - 1e-12:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    found = True
                    break
        if not found:
            return


def _tsp_bruteforce(n, D, deadline, emit):
    best_tour = list(range(n))
    best_cost = tour_cost(D, best_tour)
    if emit:
        emit(best_cost, tsp_solution_dict(n, best_tour))
    cnt = 0
    for p in itertools.permutations(range(1, n)):
        c = 0.0
        prev = 0
        for node in p:
            c += D[prev][node]
            prev = node
        c += D[prev][0]
        if c < best_cost - 1e-12:
            best_cost = c
            best_tour = [0] + list(p)
            if emit:
                emit(best_cost, tsp_solution_dict(n, best_tour))
        cnt += 1
        if (cnt & 1023) == 0 and _now() > deadline:
            break
    return best_cost, best_tour


def _tsp_mip(n, D, time_budget, emit):
    import gurobipy as gp
    from gurobipy import GRB

    m = gp.Model("tsp")
    _set_gurobi_params(m, time_budget)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    obj = {(i, j): float(D[i][j]) for (i, j) in pairs}
    x = m.addVars(pairs, obj=obj, vtype=GRB.BINARY, name="x")
    u = m.addVars(n, lb=0.0, ub=float(n - 1), name="u")
    m.ModelSense = GRB.MINIMIZE
    for i in range(n):
        m.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) == 1.0)
        m.addConstr(gp.quicksum(x[j, i] for j in range(n) if j != i) == 1.0)
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                m.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1)

    xlist = [x[p] for p in pairs]
    state = {"cost": float("inf"), "tour": None}

    def extract(vals):
        succ = {}
        for k, (i, j) in enumerate(pairs):
            if vals[k] > 0.5:
                succ[i] = j
        tour = [0]
        cur = 0
        for _ in range(n - 1):
            cur = succ.get(cur, -1)
            if cur < 0:
                return None
            tour.append(cur)
        return tour

    def consider(tour):
        c = tour_cost(D, tour)
        if c < state["cost"] - 1e-9:
            state["cost"] = c
            state["tour"] = tour
            if emit:
                emit(c, tsp_solution_dict(n, tour))

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                vals = model.cbGetSolution(xlist)
                tour = extract(vals)
                if tour is not None:
                    consider(tour)
            except Exception:
                pass

    try:
        t = _nearest_neighbor(n, D, 0)
        _two_opt(D, t, _now() + min(2.0, max(0.1, time_budget * 0.1)))
        consider(t)
        for k in range(n):
            x[t[k], t[(k + 1) % n]].Start = 1.0
    except Exception:
        pass

    m.optimize(cb)

    if state["tour"] is None:
        try:
            if m.SolCount > 0:
                vals = [x[p].X for p in pairs]
                tour = extract(vals)
                if tour is not None:
                    consider(tour)
        except Exception:
            pass
    if state["tour"] is None:
        raise RuntimeError("TSP MIP produced no feasible tour")
    return state["cost"], state["tour"]


def _tsp_heuristic(n, D, deadline, emit):
    rng = random.Random(0)
    best_cost = float("inf")
    best_tour = None

    def consider(tour):
        nonlocal best_cost, best_tour
        _two_opt(D, tour, deadline)
        c = tour_cost(D, tour)
        if c < best_cost - 1e-12:
            best_cost = c
            best_tour = tour[:]
            if emit:
                emit(best_cost, tsp_solution_dict(n, best_tour))

    for s in range(min(n, 8)):
        if _now() > deadline:
            break
        consider(_nearest_neighbor(n, D, s))
    while _now() < deadline:
        t = list(range(n))
        rng.shuffle(t)
        consider(t)
    if best_tour is None:
        best_tour = list(range(n))
        best_cost = tour_cost(D, best_tour)
        if emit:
            emit(best_cost, tsp_solution_dict(n, best_tour))
    return best_cost, best_tour


def solve_tsp(n, D, deadline, emit):
    if n <= 1:
        return 0.0, list(range(n))
    if n <= 10:
        return _tsp_bruteforce(n, D, deadline, emit)
    rem = deadline - _now() - 0.25
    if rem > 1.0 and _gurobi_available():
        try:
            return _tsp_mip(n, D, rem, emit)
        except Exception:
            pass
    return _tsp_heuristic(n, D, deadline - 0.1, emit)


# ---------------------------------------------------------------------------
# MaxCut
# ---------------------------------------------------------------------------

def solve_maxcut(n, edges, deadline, emit):
    adj = [[] for _ in range(n)]
    W = 0.0
    for (i, j, w) in edges:
        if 0 <= i < n and 0 <= j < n and i != j:
            adj[i].append((j, w))
            adj[j].append((i, w))
            W += w
    rng = random.Random(0)
    kmax = max(2, n // 15) if n > 0 else 1

    def sigma_of(s):
        sg = [0.0] * n
        for i in range(n):
            acc = 0.0
            for (j, w) in adj[i]:
                acc += w * s[j]
            sg[i] = acc
        return sg

    def value_of(s, sg):
        dot = 0.0
        for i in range(n):
            dot += s[i] * sg[i]
        return (W - 0.5 * dot) / 2.0

    def descend(s, sg):
        # Best-improvement 1-flip local search.
        # gain of flipping i = s_i * sigma_i, sigma_i = sum_j w_ij * s_j
        tot = 0.0
        while True:
            bi = -1
            bd = 1e-9
            for i in range(n):
                d = s[i] * sg[i]
                if d > bd:
                    bd = d
                    bi = i
            if bi < 0:
                return tot
            so = s[bi]
            s[bi] = -so
            for (j, w) in adj[bi]:
                sg[j] -= 2.0 * w * so
            tot += bd

    def to_bits(s):
        return [1 if v > 0 else 0 for v in s]

    def to_dict(bits):
        return {"x_%d" % i: int(bits[i]) for i in range(n)}

    # Initial random solution + local optimum (always runs, very fast).
    s = [1 if rng.random() < 0.5 else -1 for _ in range(n)]
    sg = sigma_of(s)
    v = value_of(s, sg)
    v += descend(s, sg)
    best_s, best_sg, best_v = s, sg, v
    if emit:
        emit(best_v, to_dict(to_bits(best_s)))

    total = deadline - _now()
    use_gurobi = _gurobi_available() and total > 4.0
    if use_gurobi:
        heur_until = _now() + min(30.0, max(1.0, 0.30 * total))
    else:
        heur_until = deadline - 0.15

    # Iterated local search: perturb best, re-optimize, keep improvements.
    stall = 0
    while _now() < heur_until:
        s2 = best_s[:]
        sg2 = best_sg[:]
        k = rng.randint(1, kmax)
        for idx in rng.sample(range(n), k if k <= n else n):
            so = s2[idx]
            s2[idx] = -so
            for (j, w) in adj[idx]:
                sg2[j] -= 2.0 * w * so
        v2 = value_of(s2, sg2)
        v2 += descend(s2, sg2)
        if v2 > best_v + 1e-9:
            best_s, best_sg, best_v = s2, sg2, v2
            stall = 0
            if emit:
                emit(best_v, to_dict(to_bits(best_s)))
        else:
            stall += 1
            if stall >= 1500:
                stall = 0
                s2 = [1 if rng.random() < 0.5 else -1 for _ in range(n)]
                sg2 = sigma_of(s2)
                v2 = value_of(s2, sg2)
                v2 += descend(s2, sg2)
                if v2 > best_v + 1e-9:
                    best_s, best_sg, best_v = s2, sg2, v2
                    if emit:
                        emit(best_v, to_dict(to_bits(best_s)))

    best_bits = to_bits(best_s)

    # Gurobi polishing with the heuristic incumbent as a MIP start.
    if use_gurobi:
        rem = deadline - _now() - 0.25
        if rem > 0.5:
            try:
                gv, gbits = _maxcut_gurobi(n, edges, rem, best_v, best_bits, emit)
                if gv > best_v + 1e-9:
                    best_v, best_bits = gv, gbits
            except Exception:
                pass

    return best_v, best_bits


def _maxcut_gurobi(n, edges, time_budget, start_v, start_bits, emit):
    import gurobipy as gp
    from gurobipy import GRB

    m = gp.Model("maxcut")
    _set_gurobi_params(m, time_budget)
    x = m.addVars(n, vtype=GRB.BINARY, name="x")
    mE = len(edges)
    y = m.addVars(mE, lb=0.0, ub=1.0, name="y")
    m.setObjective(gp.quicksum(edges[e][2] * y[e] for e in range(mE)), GRB.MAXIMIZE)
    for e in range(mE):
        i, j, _w = edges[e]
        m.addConstr(y[e] <= x[i] + x[j])
        m.addConstr(y[e] <= 2.0 - x[i] - x[j])
    for i in range(n):
        x[i].Start = float(start_bits[i])
    for e in range(mE):
        i, j, _w = edges[e]
        y[e].Start = 1.0 if start_bits[i] != start_bits[j] else 0.0

    xlist = [x[i] for i in range(n)]
    state = {"v": start_v, "bits": start_bits}

    def cut_of(bits):
        vv = 0.0
        for (i, j, w) in edges:
            if bits[i] != bits[j]:
                vv += w
        return vv

    def register(bits):
        vv = cut_of(bits)
        if vv > state["v"] + 1e-9:
            state["v"] = vv
            state["bits"] = bits
            if emit:
                emit(vv, {"x_%d" % i: int(bits[i]) for i in range(n)})

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                vals = model.cbGetSolution(xlist)
                register([1 if vals[i] > 0.5 else 0 for i in range(n)])
            except Exception:
                pass

    m.optimize(cb)
    try:
        if m.SolCount > 0:
            register([1 if x[i].X > 0.5 else 0 for i in range(n)])
    except Exception:
        pass
    return state["v"], state["bits"]


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

    start = _now()
    deadline = start + max(0.0, float(args.time_limit))

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    ptype = detect_problem(inst)

    logger = None
    if args.log_path and SolutionLogger is not None:
        try:
            logger = SolutionLogger(
                args.log_path,
                sense=("minimize" if ptype == "TSP" else "maximize"))
        except Exception:
            logger = None
    emit = _Emitter(logger) if logger is not None else None

    try:
        if ptype == "TSP":
            n, D = parse_tsp(inst)
            cost, tour = solve_tsp(n, D, deadline, emit)
            out = {
                "objective_value": float(cost),
                "solution": tsp_solution_dict(n, tour),
                "tour": [int(c) for c in tour],
            }
        else:
            n, edges = parse_maxcut(inst)
            val, bits = solve_maxcut(n, edges, deadline, emit)
            out = {
                "objective_value": float(val),
                "solution": {"x_%d" % i: int(bits[i]) for i in range(n)},
            }
    except Exception:
        # Last-resort: always write a feasible solution. Prefer the best
        # incumbent already logged, if any.
        if emit is not None and emit.best_dict is not None and emit.best_obj is not None:
            if ptype == "TSP":
                n, _D = parse_tsp(inst)
                d = emit.best_dict
                tour = [0] * n
                for c in range(n):
                    for t in range(n):
                        if d.get("x_%d_%d" % (c, t), 0) == 1:
                            tour[t] = c
                out = {
                    "objective_value": float(emit.best_obj),
                    "solution": d,
                    "tour": [int(c) for c in tour],
                }
            else:
                out = {
                    "objective_value": float(emit.best_obj),
                    "solution": emit.best_dict,
                }
        else:
            if ptype == "TSP":
                n, D = parse_tsp(inst)
                tour = list(range(n))
                cost = tour_cost(D, tour) if n > 0 else 0.0
                out = {
                    "objective_value": float(cost),
                    "solution": tsp_solution_dict(n, tour),
                    "tour": tour,
                }
            else:
                n, _edges = parse_maxcut(inst)
                out = {
                    "objective_value": 0.0,
                    "solution": {"x_%d" % i: 0 for i in range(n)},
                }
            if emit is not None:
                emit(out["objective_value"], out["solution"])

    with open(args.solution_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()