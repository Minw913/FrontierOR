import argparse
import json
import math
import time
from collections import Counter

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def read_instance(path):
    with open(path, "r") as f:
        data = json.load(f)

    params = data.get("parameters", {})
    W = int(params["W"])
    items_raw = data.get("items", [])

    item_types = []
    weights = []
    demands = []

    for it in items_raw:
        item_types.append(int(it["item_type"]))
        weights.append(int(it["weight"]))
        demands.append(int(it["demand"]))

    return W, item_types, weights, demands


def build_expanded_items(item_types, weights, demands):
    # Returns list of (weight, item_type), one per copy
    expanded = []
    for t, w, d in zip(item_types, weights, demands):
        if d > 0:
            expanded.extend([(w, t)] * d)
    return expanded


def heuristic_pack_ffd(expanded_items, W):
    # First-Fit Decreasing (fallback to Next-Fit style behavior automatically if needed by item order)
    if not expanded_items:
        return []

    # Sort descending by weight
    expanded_items.sort(key=lambda x: x[0], reverse=True)

    bins = []
    rem_caps = []

    n = len(expanded_items)
    # For very large instances, do next-fit decreasing for speed
    if n > 25000:
        current_bin = []
        rem = W
        for w, t in expanded_items:
            if w <= rem:
                current_bin.append(t)
                rem -= w
            else:
                bins.append(current_bin)
                current_bin = [t]
                rem = W - w
        if current_bin:
            bins.append(current_bin)
        return bins

    # Standard FFD
    for w, t in expanded_items:
        placed = False
        for b in range(len(bins)):
            if rem_caps[b] >= w:
                bins[b].append(t)
                rem_caps[b] -= w
                placed = True
                break
        if not placed:
            bins.append([t])
            rem_caps.append(W - w)

    return bins


def bins_to_type_counts(bins):
    # Per-bin Counter(item_type -> count)
    return [Counter(bin_items) for bin_items in bins]


def validate_solution_bins(bins, item_types, demands):
    req = {t: d for t, d in zip(item_types, demands)}
    got = Counter()
    for b in bins:
        got.update(b)
    for t, d in req.items():
        if got[t] < d:
            return False
    return True


def solve_mip(
    W,
    item_types,
    weights,
    demands,
    heuristic_bins,
    time_limit,
    logger=None,
    start_time=None,
):
    m = len(item_types)
    total_items = sum(demands)

    if total_items == 0:
        return []

    UB_bins = max(1, len(heuristic_bins))
    max_per_bin = [W // weights[i] for i in range(m)]

    model = gp.Model("bin_packing_aggregated")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1

    if start_time is None:
        start_time = time.time()
    remaining = max(0.0, float(time_limit) - (time.time() - start_time))
    if remaining <= 0.0:
        return heuristic_bins
    model.Params.TimeLimit = remaining

    # Optional tuning for faster incumbents
    model.Params.MIPFocus = 1

    y = model.addVars(UB_bins, vtype=GRB.BINARY, name="y")
    x = model.addVars(m, UB_bins, vtype=GRB.INTEGER, lb=0, name="x")

    # Upper bounds on x
    for i in range(m):
        ub_i = max_per_bin[i]
        for b in range(UB_bins):
            x[i, b].UB = ub_i

    # Demand satisfaction (use equality: never useful to pack extra positive-weight items)
    for i in range(m):
        model.addConstr(gp.quicksum(x[i, b] for b in range(UB_bins)) == demands[i], name=f"demand_{i}")

    # Bin capacity
    for b in range(UB_bins):
        model.addConstr(
            gp.quicksum(weights[i] * x[i, b] for i in range(m)) <= W * y[b],
            name=f"cap_{b}",
        )

    # Symmetry breaking
    for b in range(UB_bins - 1):
        model.addConstr(y[b] >= y[b + 1], name=f"order_{b}")

    model.setObjective(gp.quicksum(y[b] for b in range(UB_bins)), GRB.MINIMIZE)

    # Warm start from heuristic
    if heuristic_bins:
        counts_per_bin = bins_to_type_counts(heuristic_bins)
        for b in range(UB_bins):
            y[b].Start = 1.0 if b < len(heuristic_bins) else 0.0
        type_to_idx = {t: i for i, t in enumerate(item_types)}
        for b in range(min(len(counts_per_bin), UB_bins)):
            cdict = counts_per_bin[b]
            for t, c in cdict.items():
                i = type_to_idx[t]
                x[i, b].Start = float(c)

    best_logged = [float(len(heuristic_bins)) if heuristic_bins else float("inf")]

    def cb(model_cb, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            obj = model_cb.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < best_logged[0]:
                best_logged[0] = obj
                logger.log(float(obj))

    model.optimize(cb)

    if model.SolCount == 0:
        return heuristic_bins

    # Extract incumbent
    sol_bins = []
    for b in range(UB_bins):
        if y[b].X > 0.5:
            items_in_bin = []
            for i in range(m):
                cnt = int(round(x[i, b].X))
                if cnt > 0:
                    items_in_bin.extend([item_types[i]] * cnt)
            if items_in_bin:
                sol_bins.append(items_in_bin)

    # Safety check; fallback if numerical oddities
    if not validate_solution_bins(sol_bins, item_types, demands):
        return heuristic_bins

    return sol_bins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True, type=str)
    parser.add_argument("--solution_path", required=True, type=str)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", required=False, type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    W, item_types, weights, demands = read_instance(args.instance_path)

    total_items = sum(demands)
    if total_items == 0:
        result = {
            "objective_value": 0.0,
            "num_bins": 0,
            "bins": [],
        }
        with open(args.solution_path, "w") as f:
            json.dump(result, f)
        return

    expanded = build_expanded_items(item_types, weights, demands)

    # Heuristic feasible solution
    heuristic_bins = heuristic_pack_ffd(expanded, W)
    if logger:
        logger.log(float(len(heuristic_bins)))

    # MIP improvement (within remaining time)
    final_bins = solve_mip(
        W=W,
        item_types=item_types,
        weights=weights,
        demands=demands,
        heuristic_bins=heuristic_bins,
        time_limit=args.time_limit,
        logger=logger,
        start_time=start_time,
    )

    # Ensure feasibility; if not, fallback to trivial one-item-per-bin
    if not validate_solution_bins(final_bins, item_types, demands):
        fallback = []
        for t, d in zip(item_types, demands):
            for _ in range(d):
                fallback.append([t])
        final_bins = fallback

    result = {
        "objective_value": float(len(final_bins)),
        "num_bins": int(len(final_bins)),
        "bins": final_bins,
    }

    with open(args.solution_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()