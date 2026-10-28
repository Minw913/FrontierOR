import argparse
import json
import time
from bisect import bisect_left, insort
from collections import Counter, defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def aggregate_by_weight(weights, demands):
    agg = defaultdict(int)
    for w, d in zip(weights, demands):
        agg[int(w)] += int(d)
    ws = sorted(agg.keys())
    ds = [agg[w] for w in ws]
    return ws, ds


def best_fit_decreasing(items, capacity):
    items = sorted(items, reverse=True)
    bins = []
    # sorted list of (remaining_capacity, bin_id)
    rem_list = []

    for w in items:
        idx = bisect_left(rem_list, (w, -1))
        if idx == len(rem_list):
            b_id = len(bins)
            bins.append([w])
            rem = capacity - w
            insort(rem_list, (rem, b_id))
        else:
            rem, b_id = rem_list.pop(idx)
            bins[b_id].append(w)
            new_rem = rem - w
            insort(rem_list, (new_rem, b_id))
    return bins


def bin_to_pattern(bin_items, weight_to_idx, m):
    pat = [0] * m
    for w in bin_items:
        pat[weight_to_idx[w]] += 1
    return tuple(pat)


def pricing_unbounded_knapsack(duals, weights, capacity):
    m = len(weights)
    dp = [0.0] * (capacity + 1)
    parent_item = [-1] * (capacity + 1)
    parent_cap = [-1] * (capacity + 1)

    for i in range(m):
        w = weights[i]
        v = duals[i]
        if v <= 1e-12:
            continue
        for cap in range(w, capacity + 1):
            cand = dp[cap - w] + v
            if cand > dp[cap] + 1e-12:
                dp[cap] = cand
                parent_item[cap] = i
                parent_cap[cap] = cap - w

    best_cap = max(range(capacity + 1), key=lambda c: dp[c])
    best_val = dp[best_cap]

    if best_val <= 1.0 + 1e-6:
        return None, best_val

    counts = [0] * m
    c = best_cap
    while c > 0 and parent_item[c] != -1:
        i = parent_item[c]
        counts[i] += 1
        c = parent_cap[c]

    if sum(counts) == 0:
        return None, best_val
    return tuple(counts), best_val


def build_initial_patterns(weights, demands, capacity, heuristic_bins):
    m = len(weights)
    patterns = set()

    # singleton patterns (guaranteed feasibility)
    for i in range(m):
        p = [0] * m
        p[i] = 1
        patterns.add(tuple(p))

    # homogeneous max-fill patterns
    for i, w in enumerate(weights):
        k = capacity // w
        if k > 0:
            p = [0] * m
            p[i] = k
            patterns.add(tuple(p))

    # patterns from heuristic bins
    w2i = {w: i for i, w in enumerate(weights)}
    for b in heuristic_bins:
        patterns.add(bin_to_pattern(b, w2i, m))

    return list(patterns)


def solve_master_with_column_generation(weights, demands, capacity, patterns, time_left_func):
    m = len(weights)
    pattern_set = set(patterns)

    model = gp.Model("RMP")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1

    constrs = [model.addConstr(0.0 == demands[i], name=f"d_{i}") for i in range(m)]
    vars_list = []

    for p in patterns:
        col = gp.Column()
        for i, a in enumerate(p):
            if a:
                col.addTerms(a, constrs[i])
        v = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, obj=1.0, column=col)
        vars_list.append(v)

    model.update()

    while True:
        rem = time_left_func()
        if rem <= 1.0:
            break
        model.Params.TimeLimit = rem
        model.optimize()
        if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            break
        if model.Status == GRB.TIME_LIMIT:
            break

        duals = [c.Pi for c in constrs]
        new_pat, _ = pricing_unbounded_knapsack(duals, weights, capacity)
        if new_pat is None:
            break
        if new_pat in pattern_set:
            break

        pattern_set.add(new_pat)
        patterns.append(new_pat)
        col = gp.Column()
        for i, a in enumerate(new_pat):
            if a:
                col.addTerms(a, constrs[i])
        v = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, obj=1.0, column=col)
        vars_list.append(v)
        model.update()

    return patterns


def build_and_solve_integer_master(
    weights, demands, patterns, time_left_func, logger=None, warm_start_pattern_counts=None, incumbent_ub=None
):
    m = len(weights)
    model = gp.Model("IP")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.MIPFocus = 1

    x = [model.addVar(vtype=GRB.INTEGER, lb=0, obj=1.0, name=f"x_{j}") for j in range(len(patterns))]
    for i in range(m):
        model.addConstr(gp.quicksum(patterns[j][i] * x[j] for j in range(len(patterns))) == demands[i], name=f"d_{i}")

    model.ModelSense = GRB.MINIMIZE

    if incumbent_ub is not None:
        model.Params.Cutoff = incumbent_ub + 1e-6

    if warm_start_pattern_counts:
        for j, p in enumerate(patterns):
            if p in warm_start_pattern_counts:
                x[j].Start = warm_start_pattern_counts[p]

    model._logger = logger
    model._best_logged = float("inf")

    def cb(mdl, where):
        if where == GRB.Callback.MIPSOL and mdl._logger is not None:
            obj = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-6 < mdl._best_logged:
                mdl._best_logged = obj
                mdl._logger.log(float(obj))

    rem = time_left_func()
    if rem <= 0:
        return None, None
    model.Params.TimeLimit = rem
    model.optimize(cb)

    if model.SolCount == 0:
        return None, None

    vals = model.getAttr("X", x)
    selected = []
    for j, v in enumerate(vals):
        k = int(round(v))
        if k > 0:
            selected.append((j, k))
    obj = float(sum(k for _, k in selected))
    return selected, obj


def expand_solution_to_bins(selected_patterns, patterns, weights):
    bins = []
    for j, k in selected_patterns:
        p = patterns[j]
        bin_items = []
        for i, a in enumerate(p):
            if a > 0:
                bin_items.extend([weights[i]] * a)
        for _ in range(k):
            bins.append(list(bin_items))
    return bins


def validate_bins(bins, weights, demands, capacity):
    need = Counter()
    for w, d in zip(weights, demands):
        need[w] += d

    got = Counter()
    for b in bins:
        if sum(b) > capacity:
            return False
        for w in b:
            got[w] += 1

    return got == need


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    capacity = int(inst["c"])
    raw_weights = list(map(int, inst["weights"]))
    raw_demands = list(map(int, inst["demands"]))

    weights, demands = aggregate_by_weight(raw_weights, raw_demands)
    m = len(weights)

    def time_left():
        return max(0.0, args.time_limit - (time.time() - start))

    # Heuristic feasible packing
    items = []
    for w, d in zip(weights, demands):
        items.extend([w] * d)
    heuristic_bins = best_fit_decreasing(items, capacity)
    best_bins = heuristic_bins
    best_obj = float(len(best_bins))
    if logger:
        logger.log(best_obj)

    # Build initial patterns and warm start counts
    patterns = build_initial_patterns(weights, demands, capacity, heuristic_bins)
    w2i = {w: i for i, w in enumerate(weights)}
    warm_counts = Counter()
    for b in heuristic_bins:
        warm_counts[bin_to_pattern(b, w2i, m)] += 1

    # Column generation
    if time_left() > 1.0:
        patterns = solve_master_with_column_generation(weights, demands, capacity, patterns, time_left)

    # Integer master
    if time_left() > 0.0:
        selected, obj = build_and_solve_integer_master(
            weights=weights,
            demands=demands,
            patterns=patterns,
            time_left_func=time_left,
            logger=logger,
            warm_start_pattern_counts=warm_counts,
            incumbent_ub=best_obj,
        )
        if selected is not None:
            bins_ip = expand_solution_to_bins(selected, patterns, weights)
            if validate_bins(bins_ip, weights, demands, capacity):
                if obj + 1e-6 < best_obj:
                    best_obj = obj
                    best_bins = bins_ip
                    if logger:
                        logger.log(best_obj)
                else:
                    # even if same objective, prefer IP solution if valid
                    best_bins = bins_ip

    result = {
        "objective_value": float(len(best_bins)),
        "num_bins": int(round(len(best_bins))),
        "bin_assignments": best_bins,
    }

    with open(args.solution_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()