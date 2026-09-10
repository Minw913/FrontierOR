import argparse
import json
import math
import random
import time

from solution_logger import SolutionLogger


def make_solution(bins):
    clean_bins = [list(map(int, b)) for b in bins if b]
    return {
        "objective_value": float(len(clean_bins)),
        "bins": clean_bins,
    }


def read_conflicts(data, n):
    pairs = []
    for key in (
        "incompatible_pairs",
        "incompatibilities",
        "conflicts",
        "conflict_pairs",
    ):
        value = data.get(key)
        if value is not None:
            pairs = value
            break

    result = set()
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        a, b = int(pair[0]), int(pair[1])
        if 0 <= a < n and 0 <= b < n and a != b:
            if a > b:
                a, b = b, a
            result.add((a, b))
    return sorted(result)


def greedy_pack(order, weights, capacity, conflict_masks, mode="best", rng=None):
    bins = []
    loads = []
    masks = []

    for item in order:
        feasible = []
        wi = weights[item]
        item_conflicts = conflict_masks[item]

        for b in range(len(bins)):
            if loads[b] + wi <= capacity and (masks[b] & item_conflicts) == 0:
                residual = capacity - loads[b] - wi
                feasible.append((residual, b))
                if mode == "first":
                    break

        if not feasible:
            bins.append([item])
            loads.append(wi)
            masks.append(1 << item)
            continue

        if mode == "first":
            chosen = feasible[0][1]
        else:
            feasible.sort()
            if mode == "random_top" and rng is not None:
                top = min(3, len(feasible))
                chosen = feasible[rng.randrange(top)][1]
            else:
                chosen = feasible[0][1]

        bins[chosen].append(item)
        loads[chosen] += wi
        masks[chosen] |= 1 << item

    return bins


def eliminate_bins(bins, weights, capacity, conflict_masks, deadline):
    bins = [list(b) for b in bins if b]

    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        loads = [sum(weights[i] for i in b) for b in bins]
        source_order = sorted(range(len(bins)), key=lambda b: (loads[b], len(bins[b])))

        for source in source_order:
            if time.monotonic() >= deadline:
                return bins

            destination_bins = [list(b) for i, b in enumerate(bins) if i != source]
            destination_loads = [
                sum(weights[i] for i in b) for b in destination_bins
            ]
            destination_masks = []
            for b in destination_bins:
                mask = 0
                for item in b:
                    mask |= 1 << item
                destination_masks.append(mask)

            items = sorted(bins[source], key=lambda i: (-weights[i], i))
            success = True

            for item in items:
                best_destination = -1
                best_residual = capacity + 1
                wi = weights[item]
                conflicts = conflict_masks[item]

                for d in range(len(destination_bins)):
                    if (
                        destination_loads[d] + wi <= capacity
                        and (destination_masks[d] & conflicts) == 0
                    ):
                        residual = capacity - destination_loads[d] - wi
                        if residual < best_residual:
                            best_residual = residual
                            best_destination = d

                if best_destination < 0:
                    success = False
                    break

                destination_bins[best_destination].append(item)
                destination_loads[best_destination] += wi
                destination_masks[best_destination] |= 1 << item

            if success:
                bins = destination_bins
                improved = True
                break

    return bins


def greedy_clique_lower_bound(conflict_sets):
    n = len(conflict_sets)
    if n == 0:
        return 0

    degree_order = sorted(range(n), key=lambda i: (-len(conflict_sets[i]), i))
    best = 1

    # Each constructed set is a valid clique and therefore a valid lower bound.
    starts = degree_order if n <= 300 else degree_order[:100]
    for start in starts:
        clique = [start]
        candidates = [
            v for v in degree_order if v != start and v in conflict_sets[start]
        ]
        for v in candidates:
            if all(v in conflict_sets[u] for u in clique):
                clique.append(v)
        best = max(best, len(clique))

    return best


def solve_with_gurobi(
    weights,
    capacity,
    conflicts,
    best_bins,
    lower_bound,
    logger,
    deadline,
):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return best_bins

    n = len(weights)
    remaining = deadline - time.monotonic()
    if remaining <= 0.3:
        return best_bins

    # Avoid spending the whole budget merely constructing a quadratic model.
    pair_estimate = n * (n + 1) // 2
    if remaining < 3:
        max_pairs = 50000
    elif remaining < 10:
        max_pairs = 150000
    else:
        max_pairs = 350000

    if pair_estimate > max_pairs or len(conflicts) * max(1, n) > 1_000_000:
        return best_bins

    order = sorted(range(n), key=lambda i: (-weights[i], i))
    sorted_weights = [weights[i] for i in order]
    position = [0] * n
    for p, item in enumerate(order):
        position[item] = p

    conflict_positions = set()
    original_conflicts = set(conflicts)
    for a, b in conflicts:
        pa, pb = position[a], position[b]
        if pa > pb:
            pa, pb = pb, pa
        conflict_positions.add((pa, pb))

    try:
        model = gp.Model("bin_packing")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1

        keys = []
        columns = [[] for _ in range(n)]

        for i in range(n):
            keys.append((i, i))
            columns[i].append(i)
            for j in range(i):
                original_i = order[i]
                original_j = order[j]
                pair = (
                    (original_j, original_i)
                    if original_j < original_i
                    else (original_i, original_j)
                )
                if (
                    sorted_weights[i] + sorted_weights[j] <= capacity
                    and pair not in original_conflicts
                ):
                    keys.append((i, j))
                    columns[j].append(i)

        x = model.addVars(keys, vtype=GRB.BINARY)

        for i in range(n):
            model.addConstr(
                gp.quicksum(x[i, j] for j in range(i + 1) if (i, j) in x) == 1
            )

        for j in range(n):
            model.addConstr(
                gp.quicksum(sorted_weights[i] * x[i, j] for i in columns[j])
                <= capacity * x[j, j]
            )

        for a, b in conflict_positions:
            for j in range(a):
                if (a, j) in x and (b, j) in x:
                    model.addConstr(x[a, j] + x[b, j] <= x[j, j])

        diagonal = gp.quicksum(x[j, j] for j in range(n))
        model.setObjective(diagonal, GRB.MINIMIZE)
        model.addConstr(diagonal >= lower_bound)
        model.addConstr(diagonal <= len(best_bins))

        # Supply the heuristic solution as a complete MIP start.
        for variable in x.values():
            variable.Start = 0.0

        for bin_items in best_bins:
            sorted_positions = [position[item] for item in bin_items]
            representative = min(sorted_positions)
            for i in sorted_positions:
                if (i, representative) in x:
                    x[i, representative].Start = 1.0

        state = {
            "bins": [list(b) for b in best_bins],
            "objective": len(best_bins),
        }
        all_keys = list(x.keys())
        all_vars = [x[key] for key in all_keys]

        def decode(values):
            by_representative = {}
            assigned = set()
            for key, value in zip(all_keys, values):
                if value > 0.5:
                    i, j = key
                    original_item = order[i]
                    by_representative.setdefault(j, []).append(original_item)
                    assigned.add(original_item)

            if len(assigned) != n:
                return None
            return list(by_representative.values())

        def callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return
            try:
                objective = int(round(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)))
                if objective >= state["objective"]:
                    return
                values = cb_model.cbGetSolution(all_vars)
                incumbent_bins = decode(values)
                if incumbent_bins is None or len(incumbent_bins) != objective:
                    return
                state["objective"] = objective
                state["bins"] = incumbent_bins
                if logger:
                    solution = make_solution(incumbent_bins)
                    logger.log_solution(float(objective), solution)
            except Exception:
                pass

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return state["bins"]

        model.Params.TimeLimit = max(0.01, remaining)
        model.optimize(callback)

        if model.SolCount > 0:
            values = [variable.X for variable in all_vars]
            final_bins = decode(values)
            if final_bins is not None and len(final_bins) < state["objective"]:
                state["bins"] = final_bins
                state["objective"] = len(final_bins)
                if logger:
                    solution = make_solution(final_bins)
                    logger.log_solution(float(len(final_bins)), solution)

        return state["bins"]

    except Exception:
        return best_bins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = int(data["n"])
    capacity = int(data["c"])
    weights = [int(w) for w in data["weights"]]

    if len(weights) != n:
        raise ValueError("The weights list length does not equal n.")
    if capacity <= 0 or any(w <= 0 for w in weights):
        raise ValueError("Capacity and item weights must be positive.")
    if any(w > capacity for w in weights):
        raise ValueError("Instance is infeasible: an item exceeds bin capacity.")

    conflicts = read_conflicts(data, n)
    conflict_sets = [set() for _ in range(n)]
    conflict_masks = [0] * n
    for a, b in conflicts:
        conflict_sets[a].add(b)
        conflict_sets[b].add(a)
        conflict_masks[a] |= 1 << b
        conflict_masks[b] |= 1 << a

    descending = sorted(range(n), key=lambda i: (-weights[i], i))

    best_bins = greedy_pack(
        descending, weights, capacity, conflict_masks, mode="best"
    )
    best_solution = make_solution(best_bins)
    if logger:
        logger.log_solution(best_solution["objective_value"], best_solution)

    def consider(candidate):
        nonlocal best_bins, best_solution
        if len(candidate) < len(best_bins):
            best_bins = [list(b) for b in candidate]
            best_solution = make_solution(best_bins)
            if logger:
                logger.log_solution(best_solution["objective_value"], best_solution)

    if time.monotonic() < deadline:
        first_fit = greedy_pack(
            descending, weights, capacity, conflict_masks, mode="first"
        )
        consider(first_fit)

    heuristic_deadline = min(deadline, start_time + min(2.0, 0.12 * args.time_limit))
    if time.monotonic() < heuristic_deadline:
        improved = eliminate_bins(
            best_bins, weights, capacity, conflict_masks, heuristic_deadline
        )
        consider(improved)

    rng = random.Random(0)
    iteration = 0
    while iteration < 100 and time.monotonic() < heuristic_deadline:
        jitter = 0.05 + 0.15 * rng.random()
        randomized_order = sorted(
            range(n),
            key=lambda i: -(weights[i] + rng.random() * capacity * jitter),
        )
        candidate = greedy_pack(
            randomized_order,
            weights,
            capacity,
            conflict_masks,
            mode="random_top",
            rng=rng,
        )
        consider(candidate)

        if len(candidate) <= len(best_bins) + 1:
            candidate = eliminate_bins(
                candidate, weights, capacity, conflict_masks, heuristic_deadline
            )
            consider(candidate)
        iteration += 1

    weight_lower_bound = math.ceil(sum(weights) / capacity) if n else 0
    clique_lower_bound = greedy_clique_lower_bound(conflict_sets)
    lower_bound = max(weight_lower_bound, clique_lower_bound)

    if len(best_bins) > lower_bound and time.monotonic() < deadline:
        best_bins = solve_with_gurobi(
            weights,
            capacity,
            conflicts,
            best_bins,
            lower_bound,
            logger,
            deadline,
        )

    final_solution = make_solution(best_bins)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, separators=(",", ":"))


if __name__ == "__main__":
    main()