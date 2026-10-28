import argparse
import json
import math
import random
import time

from solution_logger import SolutionLogger


def make_solution(bins, weights, volumes):
    output_bins = []
    for item_list in bins:
        if not item_list:
            continue
        item_list = [int(i) for i in item_list]
        output_bins.append({
            "items": item_list,
            "total_weight": int(sum(weights[i] for i in item_list)),
            "total_volume": int(sum(volumes[i] for i in item_list)),
        })

    return {
        "objective_value": len(output_bins),
        "bins": output_bins,
    }


def greedy_pack(order, weights, volumes, W, V, method, deadline):
    bins = []
    bin_weights = []
    bin_volumes = []

    for position, item in enumerate(order):
        if (position & 127) == 0 and time.monotonic() >= deadline:
            # Finish immediately with singleton bins to preserve feasibility.
            for remaining_item in order[position:]:
                bins.append([remaining_item])
                bin_weights.append(weights[remaining_item])
                bin_volumes.append(volumes[remaining_item])
            break

        item_weight = weights[item]
        item_volume = volumes[item]
        bin_count = len(bins)

        if bin_count <= 3000:
            candidates = range(bin_count)
        else:
            # Bound heuristic work on very large instances.
            candidates = list(range(500)) + list(range(bin_count - 2500, bin_count))

        best_bin = -1
        best_score = None

        for b in candidates:
            new_weight = bin_weights[b] + item_weight
            new_volume = bin_volumes[b] + item_volume

            if new_weight > W or new_volume > V:
                continue

            remaining_weight = (W - new_weight) / W
            remaining_volume = (V - new_volume) / V

            if method == 0:
                score = (
                    max(remaining_weight, remaining_volume),
                    remaining_weight + remaining_volume,
                )
            elif method == 1:
                score = (
                    remaining_weight + remaining_volume,
                    max(remaining_weight, remaining_volume),
                )
            elif method == 2:
                score = (
                    abs(remaining_weight - remaining_volume),
                    remaining_weight + remaining_volume,
                )
            else:
                item_balance = item_weight / W - item_volume / V
                score = (
                    remaining_weight if item_balance >= 0 else remaining_volume,
                    remaining_weight + remaining_volume,
                    max(remaining_weight, remaining_volume),
                )

            if best_score is None or score < best_score:
                best_score = score
                best_bin = b

        if best_bin < 0:
            bins.append([item])
            bin_weights.append(item_weight)
            bin_volumes.append(item_volume)
        else:
            bins[best_bin].append(item)
            bin_weights[best_bin] += item_weight
            bin_volumes[best_bin] += item_volume

    return bins


def try_remove_one_bin(
    bins,
    weights,
    volumes,
    W,
    V,
    deadline,
    node_limit=25000,
):
    if len(bins) <= 1:
        return None

    total_weights = [sum(weights[i] for i in b) for b in bins]
    total_volumes = [sum(volumes[i] for i in b) for b in bins]

    target_order = sorted(
        range(len(bins)),
        key=lambda b: (
            len(bins[b]),
            max(total_weights[b] / W, total_volumes[b] / V),
            total_weights[b] / W + total_volumes[b] / V,
        ),
    )

    for target in target_order:
        if time.monotonic() >= deadline:
            return None

        moving_items = list(bins[target])
        if not moving_items or len(moving_items) > 40:
            continue

        destination_bins = [b for b in range(len(bins)) if b != target]
        residual_weights = [W - total_weights[b] for b in destination_bins]
        residual_volumes = [V - total_volumes[b] for b in destination_bins]

        moving_weight = sum(weights[i] for i in moving_items)
        moving_volume = sum(volumes[i] for i in moving_items)

        if moving_weight > sum(residual_weights):
            continue
        if moving_volume > sum(residual_volumes):
            continue

        moving_items.sort(
            key=lambda i: (
                max(weights[i] / W, volumes[i] / V),
                weights[i] / W + volumes[i] / V,
            ),
            reverse=True,
        )

        placement = [-1] * len(moving_items)
        nodes = [0]

        def search(k):
            nodes[0] += 1

            if nodes[0] > node_limit:
                return False
            if (nodes[0] & 255) == 0 and time.monotonic() >= deadline:
                return False
            if k == len(moving_items):
                return True

            item = moving_items[k]
            item_weight = weights[item]
            item_volume = volumes[item]

            choices = []
            seen_residuals = set()

            for d in range(len(destination_bins)):
                if residual_weights[d] < item_weight:
                    continue
                if residual_volumes[d] < item_volume:
                    continue

                signature = (residual_weights[d], residual_volumes[d])
                if signature in seen_residuals:
                    continue
                seen_residuals.add(signature)

                new_rw = (residual_weights[d] - item_weight) / W
                new_rv = (residual_volumes[d] - item_volume) / V
                choices.append((
                    max(new_rw, new_rv),
                    new_rw + new_rv,
                    d,
                ))

            choices.sort()

            for _, __, d in choices:
                residual_weights[d] -= item_weight
                residual_volumes[d] -= item_volume
                placement[k] = d

                if search(k + 1):
                    return True

                residual_weights[d] += item_weight
                residual_volumes[d] += item_volume
                placement[k] = -1

            return False

        if search(0):
            new_bins = [list(b) for b in bins]

            for k, item in enumerate(moving_items):
                original_destination = destination_bins[placement[k]]
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

    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path
        else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as file:
        instance = json.load(file)

    W = int(instance["W"])
    V = int(instance["V"])
    items_data = instance.get("items", [])

    # Some provided instances have an inconsistent `n` field. The item array is
    # authoritative because its positions define the required zero-based item
    # indices in the solution.
    n = len(items_data)

    weights = [int(item["w"]) for item in items_data]
    volumes = [int(item["v"]) for item in items_data]

    if W <= 0 or V <= 0:
        raise ValueError("Bin capacities W and V must be positive.")

    for i in range(n):
        if weights[i] <= 0 or volumes[i] <= 0:
            raise ValueError(f"Item {i} has a non-positive attribute.")
        if weights[i] > W or volumes[i] > V:
            raise ValueError(f"Item {i} cannot fit in any bin.")

    if n == 0:
        best_solution = {
            "objective_value": 0,
            "bins": [],
        }

        if logger:
            logger.log_solution(0, best_solution)

        with open(args.solution_path, "w", encoding="utf-8") as file:
            json.dump(best_solution, file, separators=(",", ":"))
        return

    # Guaranteed feasible initial solution.
    best_bins = [[i] for i in range(n)]
    best_count = n
    best_solution = make_solution(best_bins, weights, volumes)

    if logger:
        logger.log_solution(best_count, best_solution)

    total_weight = sum(weights)
    total_volume = sum(volumes)
    lower_bound = max(
        math.ceil(total_weight / W),
        math.ceil(total_volume / V),
    )

    def accept(candidate):
        nonlocal best_bins, best_count, best_solution

        if candidate is None or len(candidate) >= best_count:
            return False

        candidate_solution = make_solution(candidate, weights, volumes)
        best_bins = [list(b) for b in candidate]
        best_count = len(candidate)
        best_solution = candidate_solution

        if logger:
            logger.log_solution(best_count, best_solution)

        return True

    if time.monotonic() < hard_deadline and best_count > lower_bound:
        heuristic_budget = min(
            4.0,
            max(0.15, args.time_limit * 0.22),
        )
        heuristic_deadline = min(
            hard_deadline,
            start_time + heuristic_budget,
        )

        normalized_sum = [
            weights[i] / W + volumes[i] / V
            for i in range(n)
        ]
        normalized_max = [
            max(weights[i] / W, volumes[i] / V)
            for i in range(n)
        ]

        orders = [
            sorted(
                range(n),
                key=lambda i: (
                    normalized_max[i],
                    normalized_sum[i],
                ),
                reverse=True,
            ),
            sorted(
                range(n),
                key=lambda i: (
                    normalized_sum[i],
                    normalized_max[i],
                ),
                reverse=True,
            ),
            sorted(
                range(n),
                key=lambda i: (
                    weights[i] / W,
                    volumes[i] / V,
                ),
                reverse=True,
            ),
            sorted(
                range(n),
                key=lambda i: (
                    volumes[i] / V,
                    weights[i] / W,
                ),
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

        for order in orders:
            if (
                time.monotonic() >= heuristic_deadline
                or best_count == lower_bound
            ):
                break

            for method in range(4):
                if time.monotonic() >= heuristic_deadline:
                    break

                candidate = greedy_pack(
                    order,
                    weights,
                    volumes,
                    W,
                    V,
                    method,
                    heuristic_deadline,
                )
                accept(candidate)

        rng = random.Random(0)
        random_iterations = 12 if n <= 2000 else 3

        for _ in range(random_iterations):
            if (
                time.monotonic() >= heuristic_deadline
                or best_count == lower_bound
            ):
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

    # Use an exact assignment model only when its size is manageable.
    remaining_time = hard_deadline - time.monotonic()

    if remaining_time > 0.25 and best_count > lower_bound:
        bin_limit = best_count
        variable_count = sum(
            min(bin_limit, i + 1)
            for i in range(n)
        )

        if variable_count <= 180000 and n <= 1200:
            try:
                import gurobipy as gp
                from gurobipy import GRB

                sorted_items = sorted(
                    range(n),
                    key=lambda i: (
                        max(weights[i] / W, volumes[i] / V),
                        weights[i] / W + volumes[i] / V,
                    ),
                    reverse=True,
                )
                item_position = {
                    item: position
                    for position, item in enumerate(sorted_items)
                }

                # Canonicalize incumbent bins for the triangular formulation.
                start_bins = sorted(
                    [list(b) for b in best_bins],
                    key=lambda b: min(item_position[i] for i in b),
                )

                model = gp.Model("two_dimensional_bin_packing")
                model.Params.OutputFlag = 0
                model.Params.Seed = 0
                model.Params.MIPGap = 1e-4
                model.Params.NumericFocus = 0
                model.Params.Threads = 1
                model.Params.TimeLimit = max(
                    0.01,
                    hard_deadline - time.monotonic() - 0.03,
                )

                pairs = [
                    (i, b)
                    for i in range(n)
                    for b in range(min(bin_limit, i + 1))
                ]

                x = model.addVars(
                    pairs,
                    vtype=GRB.BINARY,
                    name="x",
                )
                y = model.addVars(
                    bin_limit,
                    vtype=GRB.BINARY,
                    name="y",
                )

                for i in range(n):
                    model.addConstr(
                        gp.quicksum(
                            x[i, b]
                            for b in range(min(bin_limit, i + 1))
                        ) == 1
                    )

                for b in range(bin_limit):
                    model.addConstr(
                        gp.quicksum(
                            weights[sorted_items[i]] * x[i, b]
                            for i in range(b, n)
                        ) <= W * y[b]
                    )
                    model.addConstr(
                        gp.quicksum(
                            volumes[sorted_items[i]] * x[i, b]
                            for i in range(b, n)
                        ) <= V * y[b]
                    )

                for b in range(bin_limit - 1):
                    model.addConstr(y[b] >= y[b + 1])

                model.addConstr(
                    gp.quicksum(y[b] for b in range(bin_limit))
                    >= lower_bound
                )
                model.setObjective(
                    gp.quicksum(y[b] for b in range(bin_limit)),
                    GRB.MINIMIZE,
                )

                start_assignment = set()

                for b, item_list in enumerate(start_bins):
                    y[b].Start = 1.0
                    for item in item_list:
                        start_assignment.add((item_position[item], b))

                for b in range(len(start_bins), bin_limit):
                    y[b].Start = 0.0

                for key, variable in x.items():
                    variable.Start = (
                        1.0 if key in start_assignment else 0.0
                    )

                x_keys = list(x.keys())
                x_variables = [x[key] for key in x_keys]
                callback_best = [best_count]

                def incumbent_callback(callback_model, where):
                    nonlocal best_bins, best_count, best_solution

                    if where != GRB.Callback.MIPSOL:
                        return

                    objective = int(round(
                        callback_model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    ))

                    if objective >= callback_best[0]:
                        return

                    values = callback_model.cbGetSolution(x_variables)
                    incumbent_bins = [[] for _ in range(bin_limit)]

                    for key, value in zip(x_keys, values):
                        if value > 0.5:
                            position, b = key
                            incumbent_bins[b].append(sorted_items[position])

                    incumbent_bins = [
                        b for b in incumbent_bins if b
                    ]

                    if len(incumbent_bins) != objective:
                        return
                    if sum(len(b) for b in incumbent_bins) != n:
                        return

                    solution = make_solution(
                        incumbent_bins,
                        weights,
                        volumes,
                    )

                    callback_best[0] = objective
                    best_count = objective
                    best_bins = [list(b) for b in incumbent_bins]
                    best_solution = solution

                    if logger:
                        logger.log_solution(objective, solution)

                model.optimize(incumbent_callback)

                if model.SolCount > 0:
                    mip_bins = [[] for _ in range(bin_limit)]

                    for (position, b), variable in x.items():
                        if variable.X > 0.5:
                            mip_bins[b].append(sorted_items[position])

                    mip_bins = [b for b in mip_bins if b]
                    accept(mip_bins)

            except Exception:
                # The current heuristic incumbent remains feasible if Gurobi
                # is unavailable, rejects the model, or reaches a license limit.
                pass

    with open(args.solution_path, "w", encoding="utf-8") as file:
        json.dump(best_solution, file, separators=(",", ":"))


if __name__ == "__main__":
    main()