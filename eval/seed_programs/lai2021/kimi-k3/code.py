#!/usr/bin/env python3
"""
Maximum-diversity partitioning of N elements into exactly m groups with
per-group lower/upper capacity bounds.

Strategy:
  - For small/moderate instances, solve an exact MIP with Gurobi:
        x[i,g] binary assignment variables
        y[i,j,g] in [0,1] pair-product linearization (valid because we
        maximize with nonnegative distance coefficients, so at integer x
        y equals min(x_i, x_j) = x_i * x_j at the optimum).
  - For very large instances (or as a fallback), run a capacity-feasible
    local-search heuristic with O(1) move deltas and random swap moves.
  - Every improved incumbent is recorded with SolutionLogger.
"""

import argparse
import json
import random
import sys
import time

from solution_logger import SolutionLogger

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAVE_GUROBI = True
except Exception:
    HAVE_GUROBI = False


# --------------------------------------------------------------------------
# Instance parsing / solution building
# --------------------------------------------------------------------------

def parse_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    N = int(data["N"])
    m = int(data["m"])
    caps = data["group_capacities"]
    L = [int(c["L_g"]) for c in caps]
    U = [int(c["U_g"]) for c in caps]
    flat = data.get("distances_upper_triangular", [])
    d = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            v = float(flat[idx])
            if v != 0.0:
                d[(i, j)] = v
            idx += 1
    return N, m, L, U, d


def build_solution(assign, d, m, N):
    groups = {g: [] for g in range(m)}
    for i in range(N):
        groups[int(assign[i])].append(i)
    groups = {g: sorted(members) for g, members in groups.items()}
    obj = 0.0
    for (i, j), v in d.items():
        if assign[i] == assign[j]:
            obj += v
    sol = {
        "objective_value": float(obj),
        "assignment": {str(i): int(assign[i]) for i in range(N)},
        "groups": {str(g): groups[g] for g in range(m)},
    }
    return sol, obj


# --------------------------------------------------------------------------
# Gurobi MIP
# --------------------------------------------------------------------------

def solve_gurobi(N, m, L, U, d, time_limit, logger):
    pairs = [(i, j, v) for (i, j), v in d.items() if v > 0.0]

    model = gp.Model("max_diversity_grouping")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    if time_limit >= 1:
        model.Params.TimeLimit = max(1, time_limit)

    x = model.addVars(N, m, vtype=GRB.BINARY, name="x")

    if pairs:
        keys = [(i, j, g) for (i, j, _) in pairs for g in range(m)]
        y = model.addVars(keys, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="y")

        for i in range(N):
            model.addConstr(gp.quicksum(x[i, g] for g in range(m)) == 1)
        for g in range(m):
            tot = gp.quicksum(x[i, g] for i in range(N))
            model.addConstr(tot >= L[g])
            model.addConstr(tot <= U[g])
        for (i, j, _) in pairs:
            for g in range(m):
                yy = y[(i, j, g)]
                model.addConstr(yy <= x[i, g])
                model.addConstr(yy <= x[j, g])

        model.setObjective(
            gp.quicksum(coef * y[(i, j, g)]
                        for (i, j, coef) in pairs
                        for g in range(m)),
            GRB.MAXIMIZE,
        )
    else:
        for i in range(N):
            model.addConstr(gp.quicksum(x[i, g] for g in range(m)) == 1)
        for g in range(m):
            tot = gp.quicksum(x[i, g] for i in range(N))
            model.addConstr(tot >= L[g])
            model.addConstr(tot <= U[g])
        model.setObjective(0.0, GRB.MAXIMIZE)

    xvars = [x[i, g] for i in range(N) for g in range(m)]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            try:
                vals = model.cbGetSolution(xvars)
                assign = [0] * N
                for i in range(N):
                    bestg, bestv = 0, -1.0
                    for g in range(m):
                        v = vals[i * m + g]
                        if v > bestv:
                            bestv = v
                            bestg = g
                    assign[i] = bestg
                sol, obj = build_solution(assign, d, m, N)
                logger.log_solution(obj, sol)
            except Exception:
                pass

    model.optimize(callback)

    if model.SolCount < 1:
        return None

    vals = model.getAttr("X", xvars)
    assign = [0] * N
    for i in range(N):
        chosen = 0
        bestv = -1.0
        for g in range(m):
            v = vals[i * m + g]
            if v > bestv:
                bestv = v
                chosen = g
        assign[i] = chosen
    return build_solution(assign, d, m, N)


# --------------------------------------------------------------------------
# Heuristic (fallback / large instances)
# --------------------------------------------------------------------------

def objective(assign, d):
    tot = 0.0
    for (i, j), v in d.items():
        if assign[i] == assign[j]:
            tot += v
    return tot


def construct_assignment(N, m, L, U, rng):
    """Random capacity-feasible assignment (assuming a feasible instance)."""
    elems = list(range(N))
    rng.shuffle(elems)
    assign = [0] * N
    sizes = [0] * m
    pos = 0
    for g in range(m):
        while sizes[g] < L[g] and pos < N:
            e = elems[pos]
            pos += 1
            assign[e] = g
            sizes[g] += 1
    for e in elems[pos:]:
        cand = [g for g in range(m) if sizes[g] < U[g]]
        g = rng.choice(cand) if cand else 0
        assign[e] = g
        sizes[g] += 1
    return assign, sizes


def solve_heuristic(N, m, L, U, d, deadline, logger):
    D = [[0.0] * N for _ in range(N)]
    for (i, j), v in d.items():
        D[i][j] = v
        D[j][i] = v
    rng = random.Random(0)

    best_assign = None
    best_obj = -float("inf")

    # Run a few randomized constructions; the longest one uses the deadline,
    # quick restarts are only used while ample time remains.
    while time.time() < deadline and (
        best_assign is None or (deadline - time.time()) > 5.0
    ):
        assign, sizes = construct_assignment(N, m, L, U, rng)
        # S[i][g] = sum of distances from element i to members of group g
        # (self-term counts as zero).
        S = [[0.0] * m for _ in range(N)]
        for (i, j), v in d.items():
            gi, gj = assign[i], assign[j]
            S[i][gj] += v
            S[j][gi] += v
        cur = objective(assign, d)

        while time.time() < deadline:
            improved = False
            # ---- relocation moves ----
            order = list(range(N))
            rng.shuffle(order)
            for i in order:
                if time.time() >= deadline:
                    break
                g = assign[i]
                Si = S[i]
                for h in range(m):
                    if h == g:
                        continue
                    if sizes[g] - 1 >= L[g] and sizes[h] + 1 <= U[h]:
                        delta = Si[h] - Si[g]
                        if delta > 1e-9:
                            Di = D[i]
                            for j in range(N):
                                v = Di[j]
                                if v != 0.0:
                                    S[j][g] -= v
                                    S[j][h] += v
                            sizes[g] -= 1
                            sizes[h] += 1
                            assign[i] = h
                            cur += delta
                            improved = True
                            if cur > best_obj + 1e-9:
                                best_obj = cur
                                best_assign = list(assign)
                                if logger:
                                    try:
                                        sol, obj = build_solution(
                                            best_assign, d, m, N)
                                        logger.log_solution(obj, sol)
                                    except Exception:
                                        pass
                            break  # at most one move per element per pass
            # ---- random swap moves (sizes unchanged -> capacities safe) ----
            swaps = 0
            max_swaps = 3 * N
            while time.time() < deadline and swaps < max_swaps:
                i = rng.randrange(N)
                j = rng.randrange(N)
                if i == j:
                    continue
                g = assign[i]
                h = assign[j]
                if g == h:
                    continue
                swaps += 1
                dij = D[i][j]
                delta = ((S[i][h] - dij) - S[i][g]) + \
                        ((S[j][g] - dij) - S[j][h])
                if delta > 1e-9:
                    Di = D[i]
                    Dj = D[j]
                    for k in range(N):
                        vik = Di[k]
                        if vik != 0.0:
                            S[k][g] -= vik
                            S[k][h] += vik
                        vjk = Dj[k]
                        if vjk != 0.0:
                            S[k][h] -= vjk
                            S[k][g] += vjk
                    S[i][g] += dij
                    S[i][h] -= dij
                    S[j][g] -= dij
                    S[j][h] += dij
                    assign[i] = h
                    assign[j] = g
                    cur += delta
                    improved = True
                    if cur > best_obj + 1e-9:
                        best_obj = cur
                        best_assign = list(assign)
                        if logger:
                            try:
                                sol, obj = build_solution(
                                    best_assign, d, m, N)
                                logger.log_solution(obj, sol)
                            except Exception:
                                pass
            if not improved:
                break  # local optimum: outer loop can restart construction

    if best_assign is None:
        best_assign, _ = construct_assignment(N, m, L, U, rng)
    return build_solution(best_assign, d, m, N)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, required=True)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    logger = (
        SolutionLogger(args.log_path, sense="maximize")
        if args.log_path
        else None
    )

    start = time.time()
    N, m, L, U, d = parse_instance(args.instance_path)

    pair_count = N * (N - 1) // 2
    use_gurobi = HAVE_GUROBI and pair_count * m <= 1_000_000

    result = None
    if use_gurobi:
        elapsed = int(time.time() - start)
        remaining = max(1, args.time_limit - elapsed)
        try:
            result = solve_gurobi(N, m, L, U, d, remaining, logger)
        except Exception:
            result = None

    if result is None:
        elapsed = int(time.time() - start)
        remaining = max(0, args.time_limit - elapsed)
        deadline = time.time() + remaining
        try:
            result = solve_heuristic(N, m, L, U, d, deadline, logger)
        except Exception:
            rng = random.Random(0)
            assign, _ = construct_assignment(N, m, L, U, rng)
            result = build_solution(assign, d, m, N)

    sol, obj = result

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)

    if logger:
        try:
            logger.log_solution(obj, sol)
        except Exception:
            pass


if __name__ == "__main__":
    main()