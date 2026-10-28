import argparse
import json
import time
from collections import defaultdict

from solution_logger import SolutionLogger


def build_size_groups(items, L, W):
    """
    Group items by exact (length, width). Rotation is NOT assumed unless provided as a separate item.
    """
    groups = defaultdict(list)
    item_by_id = {}
    for it in items:
        iid = int(it["id"])
        l = int(it["length"])
        w = int(it["width"])
        p = int(it["profit"])
        c = int(it["copies"])
        item_by_id[iid] = {"length": l, "width": w, "profit": p, "copies": c}
        if l <= L and w <= W and c > 0:
            groups[(l, w)].append((iid, p, c))
    return groups, item_by_id


def assign_feasible_counts(size_counts, groups):
    """
    Given produced counts per size (possibly violating per-item copies),
    assign copies feasibly to item ids per size, maximizing true profit.
    """
    selected = defaultdict(int)
    total_profit = 0.0

    for size, q in size_counts.items():
        if q <= 0:
            continue
        candidates = groups.get(size, [])
        if not candidates:
            continue
        # Highest profit first
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        remaining = q
        for iid, p, cap in candidates:
            if remaining <= 0:
                break
            take = min(remaining, cap)
            if take > 0:
                selected[iid] += take
                total_profit += take * p
                remaining -= take

    return total_profit, dict(selected)


def run_dp(L, W, adj_profit, deadline):
    """
    Unbounded guillotine DP with exact-size terminal values adj_profit[(l,w)].
    Returns (dp, choice) or (None, None) if time exceeded.
    """
    dp = [[0.0] * (W + 1) for _ in range(L + 1)]
    choice = [[("N", None)] * (W + 1) for _ in range(L + 1)]

    check_every_l = 4 if L > 50 else 1

    for l in range(1, L + 1):
        if l % check_every_l == 0 and time.time() >= deadline:
            return None, None
        for w in range(1, W + 1):
            best = adj_profit.get((l, w), 0.0)
            if best > 1e-12:
                best_choice = ("I", None)
            else:
                best = 0.0
                best_choice = ("N", None)

            # Vertical cuts
            half_l = l // 2
            for k in range(1, half_l + 1):
                v = dp[k][w] + dp[l - k][w]
                if v > best + 1e-12:
                    best = v
                    best_choice = ("V", k)

            # Horizontal cuts
            half_w = w // 2
            for k in range(1, half_w + 1):
                v = dp[l][k] + dp[l][w - k]
                if v > best + 1e-12:
                    best = v
                    best_choice = ("H", k)

            dp[l][w] = best
            choice[l][w] = best_choice

    return dp, choice


def recover_size_counts(L, W, choice, deadline):
    """
    Recover produced terminal item-size counts from DP choice tree.
    """
    counts = defaultdict(int)
    stack = [(L, W)]

    while stack:
        if time.time() >= deadline:
            break
        l, w = stack.pop()
        t, k = choice[l][w]
        if t == "I":
            counts[(l, w)] += 1
        elif t == "V":
            stack.append((k, w))
            stack.append((l - k, w))
        elif t == "H":
            stack.append((l, k))
            stack.append((l, w - k))
        # "N" => discard

    return counts


def fallback_single_pattern(L, W, groups):
    """
    Fast fallback heuristic: choose one size group only, in full grid packing.
    """
    best_profit = 0.0
    best_selected = {}

    for (l, w), cand in groups.items():
        if l <= 0 or w <= 0:
            continue
        fit = (L // l) * (W // w)
        if fit <= 0:
            continue

        cand_sorted = sorted(cand, key=lambda x: x[1], reverse=True)  # by profit
        remaining = fit
        total = 0.0
        sel = defaultdict(int)

        for iid, p, cap in cand_sorted:
            if remaining <= 0:
                break
            take = min(remaining, cap)
            if take > 0:
                sel[iid] += take
                total += take * p
                remaining -= take

        if total > best_profit + 1e-12:
            best_profit = total
            best_selected = dict(sel)

    return best_profit, best_selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start = time.time()
    deadline = start + max(1, int(args.time_limit))

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    L = int(inst["panel"]["length"])
    W = int(inst["panel"]["width"])
    items = inst["items"]

    groups, item_by_id = build_size_groups(items, L, W)

    # If nothing fits, return zero
    if not groups:
        out = {"objective_value": 0.0, "items_selected": []}
        with open(args.solution_path, "w") as f:
            json.dump(out, f, indent=2)
        return

    # Quick fallback baseline
    best_obj, best_selected = fallback_single_pattern(L, W, groups)
    if logger and best_obj > 0:
        logger.log(best_obj)

    # If instance likely too large for full DP, rely mostly on fallback
    # (still try at least one DP pass if time allows and dimensions are not extreme).
    area = L * W
    try_dp = area <= 250000 and L <= 700 and W <= 700 and (time.time() + 0.2 < deadline)

    if try_dp:
        # Lagrangian-style iterations on size copy capacities
        size_total_copies = {s: sum(c for _, _, c in lst) for s, lst in groups.items()}
        size_best_profit = {s: max(p for _, p, _ in lst) for s, lst in groups.items()}
        lam = {s: 0.0 for s in groups.keys()}

        # Step based on profits
        max_p = max(size_best_profit.values()) if size_best_profit else 1.0
        step = max(1.0, 0.5 * max_p)

        it = 0
        while time.time() < deadline:
            it += 1
            # Build adjusted terminal values
            adj_profit = {}
            for s, p in size_best_profit.items():
                val = p - lam[s]
                if val > 1e-9:
                    adj_profit[s] = val

            dp, choice = run_dp(L, W, adj_profit, deadline)
            if dp is None:
                break

            # Recover produced size counts
            size_counts = recover_size_counts(L, W, choice, deadline)

            # Feasible projection with true profits and exact per-item caps
            obj, selected = assign_feasible_counts(size_counts, groups)

            if obj > best_obj + 1e-9:
                best_obj = obj
                best_selected = selected
                if logger:
                    logger.log(best_obj)

            # Subgradient update (size-level copy constraints)
            any_violation = False
            for s, cap in size_total_copies.items():
                q = size_counts.get(s, 0)
                g = q - cap
                if g > 0:
                    any_violation = True
                lam[s] = max(0.0, lam[s] + step * g)

            # Step decay
            step *= 0.92
            if step < 0.05:
                step = 0.05

            # If no violations and little time benefit likely, stop early
            if not any_violation and it >= 3:
                # Already feasible wrt size caps; extra iterations rarely help much
                if time.time() + 0.2 >= deadline:
                    break

    # Build output
    items_selected = []
    for iid in sorted(best_selected.keys()):
        c = int(best_selected[iid])
        if c <= 0:
            continue
        it = item_by_id[iid]
        items_selected.append(
            {
                "item_id": int(iid),
                "plate_dims": [int(it["length"]), int(it["width"])],
                "copies": c,
            }
        )

    out = {
        "objective_value": float(best_obj),
        "items_selected": items_selected,
    }

    with open(args.solution_path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()