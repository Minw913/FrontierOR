import argparse
import json
import math
import random
import time
from typing import List, Tuple, Dict, Any

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def read_instance(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    n = int(data["n"])
    W = int(data["W"])
    V = int(data["V"])
    items = [(int(it["w"]), int(it["v"])) for it in data["items"]]
    if len(items) != n:
        raise ValueError(f"Instance inconsistency: n={n}, but len(items)={len(items)}")
    return n, W, V, items


def lower_bound_bins(items: List[Tuple[int, int]], W: int, V: int) -> int:
    total_w = sum(w for w, _ in items)
    total_v = sum(v for _, v in items)
    lb_w = math.ceil(total_w / W)
    lb_v = math.ceil(total_v / V)
    return max(1, lb_w, lb_v)


def greedy_pack(order: List[int], items: List[Tuple[int, int]], W: int, V: int):
    bins = []  # each: {"items": [...], "tw": int, "tv": int}
    for i in order:
        w, v = items[i]
        best_b = -1
        best_score = float("inf")

        for b_idx, b in enumerate(bins):
            nw = b["tw"] + w
            nv = b["tv"] + v
            if nw <= W and nv <= V:
                # Best-fit style: minimize normalized leftover slack after insertion
                score = (W - nw) / W + (V - nv) / V
                if score < best_score:
                    best_score = score
                    best_b = b_idx

        if best_b == -1:
            bins.append({"items": [i], "tw": w, "tv": v})
        else:
            b = bins[best_b]
            b["items"].append(i)
            b["tw"] += w
            b["tv"] += v

    return bins


def heuristic_multistart(
    items: List[Tuple[int, int]],
    W: int,
    V: int,
    time_budget: float,
    logger: SolutionLogger = None,
):
    n = len(items)
    start = time.time()

    # Deterministic orderings
    idx = list(range(n))
    orders = []

    orders.append(sorted(idx, key=lambda i: max(items[i][0] / W, items[i][1] / V), reverse=True))
    orders.append(sorted(idx, key=lambda i: (items[i][0] / W + items[i][1] / V), reverse=True))
    orders.append(sorted(idx, key=lambda i: (items[i][0], items[i][1]), reverse=True))
    orders.append(sorted(idx, key=lambda i: (items[i][1], items[i][0]), reverse=True))
    orders.append(sorted(idx, key=lambda i: (items[i][0] * items[i][1], items[i][0] + items[i][1]), reverse=True))

    best_bins = None
    best_obj = float("inf")

    for ord_ in orders:
        b = greedy_pack(ord_, items, W, V)
        if len(b) < best_obj:
            best_obj = len(b)
            best_bins = b
            if logger:
                logger.log(int(best_obj))
        if time.time() - start >= time_budget:
            return best_bins

    # Randomized tie-breaking / perturbation
    rng = random.Random(0)
    base = sorted(idx, key=lambda i: (items[i][0] / W + items[i][1] / V), reverse=True)

    while time.time() - start < time_budget:
        # light perturbation by shuffling chunks
        cand = base[:]
        for _ in range(max(1, n // 20)):
            a = rng.randrange(n)
            b = rng.randrange(n)
            cand[a], cand[b] = cand[b], cand[a]
        b = greedy_pack(cand, items, W, V)
        if len(b) < best_obj:
            best_obj = len(b)
            best_bins = b
            if logger:
                logger.log(int(best_obj))

    return best_bins


def bins_to_output(bins: List[Dict[str, Any]]) -> Dict[str, Any]:
    out_bins = []
    for b in bins:
        out_bins.append(
            {
                "items": list(map(int, b["items"])),
                "total_weight": int(b["tw"]),
                "total_volume": int(b["tv"]),
            }
        )
    return {"objective_value": int(len(out_bins)), "bins": out_bins}


def solve_mip(
    items: List[Tuple[int, int]],
    W: int,
    V: int,
    initial_bins: List[Dict[str, Any]],
    time_limit: float,
    logger: SolutionLogger = None,
):
    n = len(items)
    ub = len(initial_bins)

    # If trivial, return initial
    if ub <= 1 or time_limit <= 0:
        return initial_bins

    model = gp.Model("2d_bin_packing")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.0, time_limit)
    model.Params.MIPFocus = 1

    B = range(ub)
    I = range(n)

    x = model.addVars(I, B, vtype=GRB.BINARY, name="x")
    y = model.addVars(B, vtype=GRB.BINARY, name="y")

    # Each item in exactly one bin
    for i in I:
        model.addConstr(gp.quicksum(x[i, b] for b in B) == 1)

    # Capacities
    for b in B:
        model.addConstr(gp.quicksum(items[i][0] * x[i, b] for i in I) <= W * y[b])
        model.addConstr(gp.quicksum(items[i][1] * x[i, b] for i in I) <= V * y[b])

    # Strengthening
    for i in I:
        for b in B:
            model.addConstr(x[i, b] <= y[b])

    # Symmetry breaking: open bins in order
    for b in range(ub - 1):
        model.addConstr(y[b] >= y[b + 1])

    model.setObjective(gp.quicksum(y[b] for b in B), GRB.MINIMIZE)

    # MIP start from heuristic bins
    for b in B:
        y[b].Start = 1.0 if b < len(initial_bins) else 0.0
    for b, bd in enumerate(initial_bins):
        for i in bd["items"]:
            x[i, b].Start = 1.0

    best_logged = [len(initial_bins)]

    def cb(m, where):
        if where == GRB.Callback.MIPSOL:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-6 < best_logged[0]:
                best_logged[0] = obj
                if logger:
                    logger.log(int(round(obj)))

    model.optimize(cb)

    if model.SolCount == 0:
        return initial_bins

    # Extract incumbent solution
    assign = [-1] * n
    for i in I:
        for b in B:
            if x[i, b].X > 0.5:
                assign[i] = b
                break

    # Build compact bins
    bins_map = {}
    for i, b in enumerate(assign):
        if b == -1:
            # fallback (shouldn't happen), keep initial
            return initial_bins
        if b not in bins_map:
            bins_map[b] = {"items": [], "tw": 0, "tv": 0}
        bins_map[b]["items"].append(i)
        bins_map[b]["tw"] += items[i][0]
        bins_map[b]["tv"] += items[i][1]

    compact = [bins_map[b] for b in sorted(bins_map.keys()) if bins_map[b]["items"]]
    return compact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    t0 = time.time()
    n, W, V, items = read_instance(args.instance_path)

    # Safety checks from statement assumptions
    for i, (w, v) in enumerate(items):
        if w <= 0 or v <= 0 or w > W or v > V:
            raise ValueError(f"Invalid item {i}: (w={w}, v={v}) not compatible with bin (W={W}, V={V}).")

    # Heuristic phase
    total_time = max(0, int(args.time_limit))
    heur_budget = min(2.0, 0.2 * total_time) if total_time > 0 else 0.0
    best_bins = heuristic_multistart(items, W, V, heur_budget, logger=logger)

    # If somehow no heuristic result (shouldn't happen), fallback each item alone
    if best_bins is None:
        best_bins = [{"items": [i], "tw": items[i][0], "tv": items[i][1]} for i in range(n)]
        if logger:
            logger.log(len(best_bins))

    # Early stop if LB met
    lb = lower_bound_bins(items, W, V)
    elapsed = time.time() - t0
    remaining = max(0.0, total_time - elapsed)

    if len(best_bins) > lb and remaining > 0:
        best_bins = solve_mip(items, W, V, best_bins, remaining, logger=logger)

    solution = bins_to_output(best_bins)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()