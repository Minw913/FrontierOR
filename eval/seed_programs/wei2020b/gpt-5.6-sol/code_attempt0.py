import argparse
import json
import math
import random
import time

from solution_logger import SolutionLogger


def make_solution(bins, weights, volumes):
    result_bins = []
    for item_list in bins:
        if not item_list:
            continue
        items = [int(i) for i in item_list]
        result_bins.append({
            "items": items,
            "total_weight": int(sum(weights[i] for i in items)),
            "total_volume": int(sum(volumes[i] for i in items)),
        })
    return {
        "objective_value": len(result_bins),
        "bins": result_bins,
    }


def greedy_pack(order, weights, volumes, W, V, method, deadline=None):
    bins = []
    bin_w = []
    bin_v = []
    n_order = len(order)

    for position, item in enumerate(order):
        if deadline is not None and (position & 127) == 0 and time.monotonic() >= deadline:
            # Preserve feasibility if the heuristic runs out of its budget.
            for remaining in order[position:]:
                bins.append([remaining])
                bin_w.append(weights[remaining])
                bin_v.append(volumes[remaining])
            break

        wi = weights[item]
        vi = volumes[item]
        count = len(bins)

        if count <= 3000:
            candidates = range(count)
        else:
            # Bound the cost on very large instances while retaining both
            # old, tightly packed bins and recently opened bins.
            candidates = list(range(500)) + list(range(count - 2500, count))

        best_bin = -1
        best_score = None

        for b in candidates:
            nw = bin_w[b] + wi
            nv = bin_v[b] + vi
            if nw > W or nv > V:
                continue

            rw = (W - nw) / W
            rv = (V - nv) / V

            if method == 0:
                score = (max(rw, rv), rw + rv)
            elif method == 1:
                score = (rw + rv, max(rw, rv))
            elif method == 2:
                score = (abs(rw - rv), rw + rv)
            else:
                # Prefer filling the dimension in which this item is larger.
                item_balance = wi / W - vi / V
                score = (
                    rw if item_balance >= 0 else rv,
                    rw + rv,
                    max(rw, rv),
                )

            if best_score is None or score < best_score:
                best_score = score
                best_bin = b

        if best_bin < 0:
            bins.append([item])
            bin_w.append(wi)
            bin_v.append(vi)
        else:
            bins[best_bin].append(item)
            bin_w[best_bin] += wi
            bin_v[best_bin] += vi

    return bins


def try_remove_one_bin(bins, weights, volumes, W, V, deadline, node_limit=25000):
    if len(bins) <= 1:
        return None

    totals_w = [sum(weights[i] for i in b) for b in bins]
    totals_v = [sum(volumes[i] for i in b) for b in bins]

    targets = sorted(
        range(len(bins)),
        key=lambda b: (
            len(bins[b]),
            max(totals_w[b] / W, totals_v[b] / V),
            totals_w[b] / W + totals_v[b] / V,
        ),
    )

    for target in targets:
        if time.monotonic() >= deadline:
            return None

        moving = list(bins[target])
        if not moving or len(moving) > 40:
            continue

        destinations = [b for b in range(len(bins)) if b != target]
        residual_w = [W - totals_w[b] for b in destinations]
        residual_v = [V - totals_v[b] for b in destinations]

        if sum(weights[i] for i in moving) > sum(residual_w):
            continue
        if sum(volumes[i] for i in moving) > sum(residual_v):
            continue

        moving.sort(
            key=lambda i: (
                max(weights[i] / W, volumes[i] / V),
                weights[i] / W + volumes[i] / V,
            ),
            reverse=True,
        )

        placement = [-1] * len(moving)
        nodes = [0]

        def dfs(k):
            nodes[0] += 1
            if nodes[0] > node_limit:
                return False
            if (nodes[0] & 255) == 0 and time.monotonic() >= deadline:
                return False
            if k == len(moving):
                return True

            item = moving[k]
            wi = weights[item]
            vi = volumes[item]
            options = []
            seen = set()

            for d in range(len(destinations)):
                if residual_w[d] < wi or residual_v[d] < vi:
                    continue

                signature = (residual_w[d], residual_v[d])
                if signature in seen:
                    continue
                seen.add(signature)

                new_rw = (residual_w[d] - wi) / W
                new_rv = (residual_v[d] - vi) / V
                options.append((max(new_rw, new_rv), new_rw + new_rv, d))

            options.sort()

            for _, __, d in options:
                residual_w[d] -= wi
                residual_v[d] -= vi
                placement[k] = d

                if dfs(k + 1):
                    return True

                residual_w[d] += wi
                residual_v[d] += vi
                placement[k] = -1

            return False

        if dfs(0):
            new_bins = [list(b) for b in bins]
            for k, item in enumerate(moving):
                original_destination = destinations[placement[k]]
                new_bins[original_destination].append(item)
            del new_bins[target]
            return new_bins

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    hard_deadline = start_time + max(0, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    n = int(instance["n"])
    W = int(instance["W"])
    V = int(instance["V"])
    items_data = instance["items"]

    if len(items_data) != n:
        raise ValueError("The number of items does not match field 'n'.")

    weights = [int(item["w"]) for item in items_data]
    volumes = [int(item["v"]) for item in items_data]

    if n == 0:
        best_solution = {"objective_value": 0, "bins": []}
        if logger:
            logger.log_solution(0, best_solution)
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(best_solution, f, separators=(",", ":"))
        return

    # Immediate guaranteed feasible incumbent.
    best_bins = [[i] for i in range(n)]
    best_solution = make_solution(best_bins, weights, volumes)
    best_count = n

    if logger:
        logger.log_solution(best_count, best_solution)

    total_weight = sum(weights)
    total_volume = sum(volumes)
    lower_bound = max(
        math.ceil(total_weight / W),
        math.ceil(total_volume / V),
    )

    def accept(candidate):
        nonlocal best_bins, best_solution, best_count
        if candidate is None:
            return False
        candidate_count = len(candidate)
        if candidate_count >= best_count:
            return False

        solution = make_solution(candidate, weights, volumes)
        best_bins = [list(b) for b in candidate]
        best_solution = solution
        best_count = candidate_count

        if logger:
            logger.log_solution(best_count, best_solution)
        return True

    if time.monotonic() < hard_deadline and best_count > lower_bound:
        initial_budget = min(4.0, max(0.15, args.time_limit * 0.22))
        heuristic_deadline = min(hard_deadline, start_time + initial_budget)

        normalized_sum = [
            weights[i] / W + volumes[i] / V for i in range(n)
        ]
        normalized_max = [
            max(weights[i] / W, volumes[i] / V) for i in range(n)
        ]

        orders = [
            sorted(
                range(n),
                key=lambda i: (normalized_max[i], normalized_sum[i]),
                reverse=True,
            ),
            sorted(
                range(n),
                key=lambda i: (normalized_sum[i], normalized_max[i]),
                reverse=True,
            ),
            sorted(
                range(n),
                key=lambda i: (weights[i] / W, volumes[i] / V),
                reverse=True,
            ),
            sorted(
                range(n),
                key=lambda i: (volumes[i] / V, weights[i] / W),
                reverse=True,
            ),
            sorted(
                range(n),
                key=lambda i: (
                    weights[i] * volumes[i],
                    normalized_sum[i],
                ),
                reverse=True,
            ),
        ]

        for order_index, order in enumerate(orders):
            if time.monotonic() >= heuristic_deadline or best_count == lower_bound:
                break
            for method in range(4):
                if time.monotonic() >= heuristic_deadline:
                    break
                candidate = greedy_pack(
                    order, weights, volumes, W, V, method, heuristic_deadline
                )
                accept(candidate)

        # Deterministic randomized variants diversify ordering while preserving
        # reproducibility.
        rng = random.Random(0)
        random_iterations = 12 if n <= 2000 else 3
        for _ in range(random_iterations):
            if time.monotonic() >= heuristic_deadline or best_count == lower_bound:
                break
            noisy_order = sorted(
                range(n),
                key=lambda i: (
                    normalized_max[i] + rng.random() * 0.12,
                    normalized_sum[i],
                ),
                reverse=True,
            )
            candidate = greedy_pack(
                noisy_order,
                weights,
                volumes,
                W,
                V,
                rng.randrange(4),
                heuristic_deadline,
            )
            accept(candidate)

        # Attempt to eliminate weakly loaded bins through bounded backtracking.
        while (
            time.monotonic() < heuristic_deadline
            and best_count > lower_bound
            and best_count > 1
        ):
            candidate = try_remove_one_bin(
                best_bins,
                weights,
                volumes,
                W,
                V,
                heuristic_deadline,
            )
            if candidate is None or not accept(candidate):
                break

    # Exact MIP phase for instances whose assignment formulation is reasonably
    # sized. The heuristic incumbent is supplied as a MIP start.
    remaining = hard_deadline - time.monotonic()
    if remaining > 0.25 and best_count > lower_bound:
        B = best_count
        pair_count = sum(min(B, i + 1) for i in range(n))

        if pair_count <= 180000 and n <= 1200:
            try:
                import gurobipy as gp
                from gurobipy import GRB

                # Hardest items receive the earliest canonical positions.
                sorted_items = sorted(
                    range(n),
                    key=lambda i: (
                        max(weights[i] / W, volumes[i] / V),
                        weights[i] / W + volumes[i] / V,
                    ),
                    reverse=True,
                )
                position = {item: p for p, item in enumerate(sorted_items)}

                # Canonicalize incumbent bins by their earliest item position.
                start_bins = sorted(
                    [list(b) for b in best_bins],
                    key=lambda b: min(position[i] for i in b),
                )

                model = gp.Model("two_dimensional_bin_packing")
                model.Params.OutputFlag = 0
                model.Params.Seed = 0
                model.Params.MIPGap = 1e-4
                model.Params.NumericFocus = 0
                model.Params.Threads = 1

                remaining = hard_deadline - time.monotonic()
                model.Params.TimeLimit = max(0.01, remaining - 0.03)

                pairs = [
                    (i, b)
                    for i in range(n)
                    for b in range(min(B, i + 1))
                ]
                x = model.addVars(pairs, vtype=GRB.BINARY, name="x")
                y = model.addVars(B, vtype=GRB.BINARY, name="y")

                for i in range(n):
                    model.addConstr(
                        gp.quicksum(x[i, b] for b in range(min(B, i + 1))) == 1
                    )

                for b in range(B):
                    model.addConstr(
                        gp.quicksum(
                            weights[sorted_items[i]] * x[i, b]
                            for i in range(b, n)
                            if (i, b) in x
                        )
                        <= W * y[b]
                    )
                    model.addConstr(
                        gp.quicksum(
                            volumes[sorted_items[i]] * x[i, b]
                            for i in range(b, n)
                            if (i, b) in x
                        )
                        <= V * y[b]
                    )

                for b in range(B - 1):
                    model.addConstr(y[b] >= y[b + 1])

                model.addConstr(gp.quicksum(y[b] for b in range(B)) >= lower_bound)
                model.setObjective(gp.quicksum(y[b] for b in range(B)), GRB.MINIMIZE)

                # MIP start.
                start_assignment = {}
                for b, item_list in enumerate(start_bins):
                    y[b].Start = 1.0
                    for item in item_list:
                        p = position[item]
                        start_assignment[(p, b)] = 1.0

                for b in range(len(start_bins), B):
                    y[b].Start = 0.0
                for key, var in x.items():
                    var.Start = 1.0 if key in start_assignment else 0.0

                x_keys = list(x.keys())
                x_vars = [x[key] for key in x_keys]
                callback_state = {"best": best_count}

                def incumbent_callback(cb_model, where):
                    nonlocal best_bins, best_solution, best_count
                    if where != GRB.Callback.MIPSOL:
                        return

                    objective = int(round(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)))
                    if objective >= callback_state["best"]:
                        return

                    values = cb_model.cbGetSolution(x_vars)
                    callback_bins = [[] for _ in range(B)]

                    for key, value in zip(x_keys, values):
                        if value > 0.5:
                            p, b = key
                            callback_bins[b].append(sorted_items[p])

                    callback_bins = [b for b in callback_bins if b]
                    if len(callback_bins) != objective:
                        return
                    if sum(len(b) for b in callback_bins) != n:
                        return

                    solution = make_solution(callback_bins, weights, volumes)
                    callback_state["best"] = objective
                    best_count = objective
                    best_bins = [list(b) for b in callback_bins]
                    best_solution = solution

                    if logger:
                        logger.log_solution(objective, solution)

                model.optimize(incumbent_callback)

                if model.SolCount > 0:
                    mip_bins = [[] for _ in range(B)]
                    for (p, b), var in x.items():
                        if var.X > 0.5:
                            mip_bins[b].append(sorted_items[p])
                    mip_bins = [b for b in mip_bins if b]
                    accept(mip_bins)

            except Exception:
                # The heuristic solution remains valid if Gurobi is unavailable,
                # a license limit is reached, or model construction is interrupted.
                pass

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution, f, separators=(",", ":"))


if __name__ == "__main__":
    main()