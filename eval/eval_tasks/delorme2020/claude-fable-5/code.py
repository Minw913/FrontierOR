import argparse
import json
import math
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


# ---------------------------------------------------------------------------
# First Fit Decreasing heuristic (segment tree for O(n log n))
# ---------------------------------------------------------------------------
def first_fit_decreasing(weights, demands, c):
    items = []
    for w, d in zip(weights, demands):
        items.extend([w] * d)
    items.sort(reverse=True)
    n = len(items)
    if n == 0:
        return []
    size = 1
    while size < n:
        size *= 2
    tree = [c] * (2 * size)
    bins = [[] for _ in range(n)]
    maxidx = -1
    for w in items:
        node = 1
        if tree[1] < w:
            # should not happen since w < c
            continue
        while node < size:
            node *= 2
            if tree[node] < w:
                node += 1
        idx = node - size
        tree[node] -= w
        p = node >> 1
        while p:
            tree[p] = tree[2 * p] if tree[2 * p] >= tree[2 * p + 1] else tree[2 * p + 1]
            p >>= 1
        bins[idx].append(w)
        if idx > maxidx:
            maxidx = idx
    return [b for b in bins[:maxidx + 1] if b]


def bins_to_pattern_counts(bins, weight_to_idx, m):
    """Convert bins (lists of widths) to pattern count tuples."""
    pats = []
    for b in bins:
        cnt = [0] * m
        for w in b:
            cnt[weight_to_idx[w]] += 1
        pats.append(tuple(cnt))
    return pats


# ---------------------------------------------------------------------------
# Pricing problems
# ---------------------------------------------------------------------------
def price_exact(duals, w, bounds, c):
    """Exact bounded-knapsack pricing via binary splitting + numpy DP."""
    m = len(w)
    dp = np.zeros(c + 1)
    pieces = []
    for i in range(m):
        if duals[i] <= 1e-12:
            continue
        b = min(bounds[i], c // w[i])
        k = 1
        while b > 0:
            t = min(k, b)
            pieces.append((i, t, w[i] * t, duals[i] * t))
            b -= t
            k *= 2
    takes = []
    for (i, t, wt, vt) in pieces:
        cand = dp[:c + 1 - wt] + vt
        seg = dp[wt:]
        mask = cand > seg + 1e-12
        full = np.zeros(c + 1, dtype=bool)
        full[wt:] = mask
        seg[mask] = cand[mask]
        takes.append(full)
    j = int(np.argmax(dp))
    best = float(dp[j])
    counts = [0] * m
    for p in range(len(pieces) - 1, -1, -1):
        if takes[p][j]:
            i, t, wt, vt = pieces[p]
            counts[i] += t
            j -= wt
    return best, counts


def price_greedy(duals, w, bounds, c):
    """Fast heuristic pricing (used only when DP would be too large)."""
    m = len(w)
    order = sorted(range(m), key=lambda i: -(duals[i] / w[i]))
    rem = c
    counts = [0] * m
    val = 0.0
    for i in order:
        if duals[i] <= 1e-12 or w[i] > rem:
            continue
        t = min(bounds[i], rem // w[i])
        if t > 0:
            counts[i] = t
            rem -= t * w[i]
            val += duals[i] * t
    return val, counts


# ---------------------------------------------------------------------------
# Decode pattern usage into bin assignments (with surplus trimming)
# ---------------------------------------------------------------------------
def decode_solution(vals, patterns, weights, demands):
    m = len(weights)
    raw_bins = []
    for idx, v in enumerate(vals):
        k = int(round(v))
        if k <= 0:
            continue
        for _ in range(k):
            raw_bins.append(list(patterns[idx]))
    total = [0] * m
    for b in raw_bins:
        for i in range(m):
            total[i] += b[i]
    surplus = [total[i] - demands[i] for i in range(m)]
    out = []
    for b in raw_bins:
        row = []
        for i in range(m):
            take = b[i]
            if surplus[i] > 0 and take > 0:
                r = min(take, surplus[i])
                take -= r
                surplus[i] -= r
            row.extend([weights[i]] * take)
        if row:
            out.append(row)
    return out


def make_solution_dict(bins):
    return {
        "objective_value": float(len(bins)),
        "num_bins": len(bins),
        "bin_assignments": bins,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    deadline = start_time + max(5, args.time_limit) - 1.0

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        inst = json.load(f)

    c = int(inst["c"])
    raw_w = [int(x) for x in inst["weights"]]
    raw_d = [int(x) for x in inst["demands"]]

    # Consolidate duplicate weights
    agg = {}
    for w, d in zip(raw_w, raw_d):
        if d > 0:
            agg[w] = agg.get(w, 0) + d
    weights = sorted(agg.keys(), reverse=True)
    demands = [agg[w] for w in weights]
    m = len(weights)
    weight_to_idx = {w: i for i, w in enumerate(weights)}

    if m == 0:
        sol = make_solution_dict([])
        if logger:
            logger.log_solution(0.0, sol)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    total_weight = sum(w * d for w, d in zip(weights, demands))
    lb = int(math.ceil(total_weight / c))

    # ---- FFD heuristic -----------------------------------------------------
    ffd_bins = first_fit_decreasing(weights, demands, c)
    best_bins = ffd_bins
    best_obj = len(ffd_bins)
    if logger:
        logger.log_solution(float(best_obj), make_solution_dict(best_bins))

    if best_obj <= lb:
        with open(args.solution_path, "w") as f:
            json.dump(make_solution_dict(best_bins), f)
        return

    # ---- Column generation master ------------------------------------------
    bounds = [min(demands[i], c // weights[i]) for i in range(m)]

    # decide pricing mode
    est_pieces = sum(int(math.ceil(math.log2(min(bounds[i], max(1, c // weights[i])) + 1))) + 1
                     for i in range(m))
    exact_pricing = (c <= 2_000_000) and (est_pieces * (c + 1) <= 2e8)

    model = gp.Model("bpp")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    cons = []
    for i in range(m):
        cons.append(model.addLConstr(gp.LinExpr(), GRB.GREATER_EQUAL, demands[i]))
    model.update()

    patterns = []
    pat_index = {}
    var_list = []

    def add_pattern(counts):
        key = tuple(counts)
        if key in pat_index:
            return pat_index[key]
        idx = len(patterns)
        patterns.append(key)
        pat_index[key] = idx
        col = gp.Column()
        for i in range(m):
            if key[i] > 0:
                col.addTerms(key[i], cons[i])
        v = model.addVar(obj=1.0, lb=0.0, vtype=GRB.CONTINUOUS, column=col)
        var_list.append(v)
        return idx

    # initial columns: FFD bins + singleton max-fill patterns
    ffd_pats = bins_to_pattern_counts(ffd_bins, weight_to_idx, m)
    ffd_usage = {}
    for p in ffd_pats:
        idx = add_pattern(p)
        ffd_usage[idx] = ffd_usage.get(idx, 0) + 1
    for i in range(m):
        cnt = [0] * m
        cnt[i] = bounds[i]
        add_pattern(cnt)
    model.update()

    cg_deadline = min(deadline, start_time + 0.55 * max(5, args.time_limit))
    lp_bound_valid = exact_pricing
    lp_obj = None
    max_iters = 5000
    it = 0
    while it < max_iters and time.time() < cg_deadline:
        it += 1
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            lp_bound_valid = False
            break
        lp_obj = model.ObjVal
        duals = [max(0.0, cons[i].Pi) for i in range(m)]
        if time.time() >= cg_deadline:
            lp_bound_valid = False
            break
        if exact_pricing:
            best_val, counts = price_exact(duals, weights, bounds, c)
        else:
            best_val, counts = price_greedy(duals, weights, bounds, c)
        if best_val <= 1.0 + 1e-6:
            break  # no negative reduced cost column
        before = len(patterns)
        add_pattern(counts)
        if len(patterns) == before:
            break  # duplicate column -> stop
        model.update()
    else:
        lp_bound_valid = False
    if it >= max_iters:
        lp_bound_valid = False

    if lp_bound_valid and lp_obj is not None:
        lb = max(lb, int(math.ceil(lp_obj - 1e-6)))

    if best_obj <= lb:
        with open(args.solution_path, "w") as f:
            json.dump(make_solution_dict(best_bins), f)
        return

    # ---- Integer master ------------------------------------------------------
    for v in var_list:
        v.VType = GRB.INTEGER
        v.Start = 0.0
    for idx, k in ffd_usage.items():
        var_list[idx].Start = float(k)
    model.update()

    remaining = max(2.0, deadline - time.time())
    model.Params.TimeLimit = remaining
    model.Params.BestObjStop = lb + 1e-3
    model.Params.MIPFocus = 1

    model._vars = var_list
    model._patterns = patterns
    model._best = best_obj

    state = {"best_obj": best_obj, "best_bins": best_bins}

    def callback(mdl, where):
        if where == GRB.Callback.MIPSOL:
            obj = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < state["best_obj"] - 0.5:
                try:
                    vals = mdl.cbGetSolution(mdl._vars)
                    bins = decode_solution(vals, mdl._patterns, weights, demands)
                    if len(bins) < state["best_obj"]:
                        state["best_obj"] = len(bins)
                        state["best_bins"] = bins
                        if logger:
                            logger.log_solution(float(len(bins)),
                                                make_solution_dict(bins))
                except Exception:
                    pass

    try:
        model.optimize(callback)
    except Exception:
        pass

    if model.SolCount > 0:
        try:
            vals = [v.X for v in var_list]
            bins = decode_solution(vals, patterns, weights, demands)
            if len(bins) < state["best_obj"]:
                state["best_obj"] = len(bins)
                state["best_bins"] = bins
                if logger:
                    logger.log_solution(float(len(bins)), make_solution_dict(bins))
        except Exception:
            pass

    best_bins = state["best_bins"]
    sol = make_solution_dict(best_bins)
    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()