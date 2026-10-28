import argparse
import json
import time
from typing import List, Tuple, Set

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    return parser.parse_args()


def load_instance(path: str):
    with open(path, "r") as f:
        return json.load(f)


def build_data(instance):
    n = int(instance["num_nodes"])
    commodities = instance["commodities"]
    cnum = len(commodities)  # safer than trusting num_commodities

    dist = instance["distances"]
    fixed = [float(node["fixed_cost"]) for node in instance["nodes"]]

    params = instance["parameters"]
    tau = float(params["tau"])
    chi = float(params["chi"])
    delta = float(params["delta"])

    o = [0] * cnum
    d = [0] * cnum
    flow = [0.0] * cnum
    fchi = [0.0] * cnum
    fdelta = [0.0] * cnum
    ftau = [0.0] * cnum
    same = [False] * cnum

    for c in range(cnum):
        oc = int(commodities[c]["origin"])
        dc = int(commodities[c]["destination"])
        fl = float(commodities[c]["flow"])
        o[c] = oc
        d[c] = dc
        flow[c] = fl
        fchi[c] = fl * chi
        fdelta[c] = fl * delta
        ftau[c] = fl * tau
        same[c] = (oc == dc)

    return {
        "n": n,
        "cnum": cnum,
        "dist": dist,
        "fixed": fixed,
        "o": o,
        "d": d,
        "fchi": fchi,
        "fdelta": fdelta,
        "ftau": ftau,
        "same": same,
        "commodities": commodities,
    }


def single_hub_cost_for_commodity(c, h, data):
    dist = data["dist"]
    o = data["o"]
    d = data["d"]
    fchi = data["fchi"]
    fdelta = data["fdelta"]
    return fchi[c] * dist[o[c]][h] + fdelta[c] * dist[h][d[c]]


def pair_cost_for_commodity(c, i, j, data):
    dist = data["dist"]
    o = data["o"]
    d = data["d"]
    fchi = data["fchi"]
    fdelta = data["fdelta"]
    ftau = data["ftau"]
    return (
        fchi[c] * dist[o[c]][i]
        + ftau[c] * dist[i][j]
        + fdelta[c] * dist[j][d[c]]
    )


def initial_single_hub_solution(data, time_limit, start_time):
    n = data["n"]
    cnum = data["cnum"]
    fixed = data["fixed"]

    best_h = 0
    best_obj = float("inf")

    for h in range(n):
        if time.time() - start_time > max(0.0, time_limit - 0.05):
            break
        s = fixed[h]
        for c in range(cnum):
            s += single_hub_cost_for_commodity(c, h, data)
        if s < best_obj:
            best_obj = s
            best_h = h

    best_costs = [single_hub_cost_for_commodity(c, best_h, data) for c in range(cnum)]
    best_pairs = [(best_h, best_h) for _ in range(cnum)]
    return best_obj, best_h, best_costs, best_pairs


def evaluate_add(
    h: int,
    open_set: Set[int],
    current_obj: float,
    current_costs: List[float],
    current_pairs: List[Tuple[int, int]],
    data
):
    cnum = data["cnum"]
    same = data["same"]
    fixed = data["fixed"]
    dist = data["dist"]
    o = data["o"]
    d = data["d"]
    fchi = data["fchi"]
    fdelta = data["fdelta"]
    ftau = data["ftau"]

    S = list(open_set)
    new_costs = [0.0] * cnum
    new_pairs = [None] * cnum
    delta_trans = 0.0

    for c in range(cnum):
        oldc = current_costs[c]
        best = oldc
        bestp = current_pairs[c]

        # Single-hub through h always feasible
        cand = fchi[c] * dist[o[c]][h] + fdelta[c] * dist[h][d[c]]
        if cand < best:
            best = cand
            bestp = (h, h)

        if not same[c]:
            # Two-hub options involving h and existing hubs
            ah = fchi[c] * dist[o[c]][h]
            bh = fdelta[c] * dist[h][d[c]]
            th = ftau[c]
            row_h = dist[h]

            for j in S:
                v1 = ah + th * row_h[j] + fdelta[c] * dist[j][d[c]]  # (h, j)
                if v1 < best:
                    best = v1
                    bestp = (h, j)

                v2 = fchi[c] * dist[o[c]][j] + th * dist[j][h] + bh  # (j, h)
                if v2 < best:
                    best = v2
                    bestp = (j, h)

        new_costs[c] = best
        new_pairs[c] = bestp
        delta_trans += (best - oldc)

    new_obj = current_obj + fixed[h] + delta_trans
    return new_obj, new_costs, new_pairs


def evaluate_open_set(open_list: List[int], data):
    cnum = data["cnum"]
    fixed = data["fixed"]
    same = data["same"]

    obj = sum(fixed[h] for h in open_list)
    costs = [0.0] * cnum
    pairs = [None] * cnum

    for c in range(cnum):
        best = float("inf")
        bestp = (open_list[0], open_list[0])

        if same[c]:
            for h in open_list:
                v = single_hub_cost_for_commodity(c, h, data)
                if v < best:
                    best = v
                    bestp = (h, h)
        else:
            for i in open_list:
                for j in open_list:
                    v = pair_cost_for_commodity(c, i, j, data)
                    if v < best:
                        best = v
                        bestp = (i, j)

        costs[c] = best
        pairs[c] = bestp
        obj += best

    return obj, costs, pairs


def local_search(data, time_limit, start_time, logger=None):
    n = data["n"]
    cnum = data["cnum"]

    current_obj, h0, current_costs, current_pairs = initial_single_hub_solution(
        data, time_limit, start_time
    )
    open_set = {h0}

    if logger:
        logger.log(float(current_obj))

    # Heuristic controls
    # keep neighborhood small on very large instances
    scale = n * max(1, cnum)
    if scale >= 2_000_000:
        max_add_candidates = 20
        max_drop_candidates = 6
        allow_drop = False
    elif scale >= 500_000:
        max_add_candidates = 35
        max_drop_candidates = 10
        allow_drop = True
    else:
        max_add_candidates = 60
        max_drop_candidates = 20
        allow_drop = True

    fixed = data["fixed"]
    eps = 1e-9

    improved = True
    while improved and (time.time() - start_time) < max(0.0, time_limit - 0.05):
        improved = False
        best_move_obj = current_obj
        best_move = None
        best_costs = None
        best_pairs = None

        # ADD moves
        closed = [i for i in range(n) if i not in open_set]
        if len(closed) > max_add_candidates:
            closed.sort(key=lambda x: fixed[x])
            closed = closed[:max_add_candidates]

        for h in closed:
            if (time.time() - start_time) >= max(0.0, time_limit - 0.05):
                break
            cand_obj, cand_costs, cand_pairs = evaluate_add(
                h, open_set, current_obj, current_costs, current_pairs, data
            )
            if cand_obj + eps < best_move_obj:
                best_move_obj = cand_obj
                best_move = ("add", h)
                best_costs = cand_costs
                best_pairs = cand_pairs

        # DROP moves (optional / limited)
        if allow_drop and len(open_set) > 1 and (time.time() - start_time) < max(0.0, time_limit - 0.05):
            open_list = list(open_set)
            if len(open_list) > max_drop_candidates:
                open_list.sort(key=lambda x: fixed[x], reverse=True)  # try dropping expensive hubs first
                open_list = open_list[:max_drop_candidates]

            for h in open_list:
                if (time.time() - start_time) >= max(0.0, time_limit - 0.05):
                    break
                cand_open = [x for x in open_set if x != h]
                cand_obj, cand_costs, cand_pairs = evaluate_open_set(cand_open, data)
                if cand_obj + eps < best_move_obj:
                    best_move_obj = cand_obj
                    best_move = ("drop", h)
                    best_costs = cand_costs
                    best_pairs = cand_pairs

        if best_move is not None and best_move_obj + eps < current_obj:
            t, h = best_move
            if t == "add":
                open_set.add(h)
            else:
                open_set.remove(h)

            current_obj = best_move_obj
            current_costs = best_costs
            current_pairs = best_pairs
            improved = True

            if logger:
                logger.log(float(current_obj))

    # Recompute objective exactly for safety
    final_obj = sum(data["fixed"][h] for h in open_set) + sum(current_costs)
    return {
        "objective": float(final_obj),
        "open_set": sorted(open_set),
        "best_costs": current_costs,
        "best_pairs": current_pairs,
    }


def fallback_solution(data, logger=None):
    # Always feasible: open cheapest fixed-cost hub and route all commodities through it
    fixed = data["fixed"]
    cnum = data["cnum"]
    h = min(range(len(fixed)), key=lambda i: fixed[i])
    costs = [single_hub_cost_for_commodity(c, h, data) for c in range(cnum)]
    obj = fixed[h] + sum(costs)
    if logger:
        logger.log(float(obj))
    return {
        "objective": float(obj),
        "open_set": [h],
        "best_costs": costs,
        "best_pairs": [(h, h) for _ in range(cnum)],
    }


def build_output(instance, data, sol):
    commodities = data["commodities"]
    assignments = []
    for c, (i, j) in enumerate(sol["best_pairs"]):
        if i == j:
            hub_edge = [int(i)]
        else:
            a, b = int(i), int(j)
            # output unordered pair in symmetric setting
            if a <= b:
                hub_edge = [a, b]
            else:
                hub_edge = [b, a]

        assignments.append(
            {
                "commodity": c,
                "origin": int(commodities[c]["origin"]),
                "destination": int(commodities[c]["destination"]),
                "hub_edge": hub_edge,
                "cost": float(sol["best_costs"][c]),
            }
        )

    output = {
        "objective_value": float(sol["objective"]),
        "open_hubs": [int(h) for h in sorted(sol["open_set"])],
        "assignments": assignments,
    }
    return output


def main():
    args = parse_args()
    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = load_instance(args.instance_path)
    data = build_data(instance)

    try:
        sol = local_search(data, args.time_limit, start_time, logger=logger)
    except Exception:
        # Robust fallback to guarantee a feasible output
        sol = fallback_solution(data, logger=logger)

    output = build_output(instance, data, sol)

    with open(args.solution_path, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()