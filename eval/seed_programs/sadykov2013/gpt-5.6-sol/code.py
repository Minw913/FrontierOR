import argparse
import json
import math
import random
import time

from solution_logger import SolutionLogger


def canonicalize_bins(bins):
    """Return bins sorted into a canonical order and without empty bins."""
    nonempty = [b for b in bins if b["items"]]
    for b in nonempty:
        b["items"].sort()
    nonempty.sort(key=lambda b: (b["items"][0], tuple(b["items"])))
    return nonempty


def make_solution(bins, item_ids, weights):
    bins = canonicalize_bins(bins)
    output_bins = {}
    for b_idx, b in enumerate(bins):
        items = [item_ids[i] for i in b["items"]]
        output_bins[str(b_idx)] = {
            "items": items,
            "total_weight": int(sum(weights[i] for i in b["items"]))
        }
    return {
        "objective_value": len(output_bins),
        "bins": output_bins
    }


def copy_bins(bins):
    return [
        {
            "items": list(b["items"]),
            "load": int(b["load"]),
            "mask": int(b["mask"])
        }
        for b in bins
    ]


def greedy_pack(order, weights, adjacency, capacity):
    bins = []

    for i in order:
        wi = weights[i]
        bit = 1 << i
        best_bin = -1
        best_score = None

        for b_idx, b in enumerate(bins):
            if b["load"] + wi > capacity:
                continue
            if b["mask"] & adjacency[i]:
                continue

            residual = capacity - b["load"] - wi
            # Best fit first, then prefer bins with more items.
            score = (residual, -len(b["items"]), b_idx)
            if best_score is None or score < best_score:
                best_score = score
                best_bin = b_idx

        if best_bin < 0:
            bins.append({
                "items": [i],
                "load": wi,
                "mask": bit
            })
        else:
            b = bins[best_bin]
            b["items"].append(i)
            b["load"] += wi
            b["mask"] |= bit

    return canonicalize_bins(bins)


def dsatur_pack(weights, adjacency, degrees, capacity):
    """Conflict-aware DSATUR ordering combined with best-fit placement."""
    import heapq

    n = len(weights)
    saturation_masks = [0] * n
    assigned_bin = [-1] * n
    assigned = [False] * n
    heap = []

    for i in range(n):
        heapq.heappush(
            heap,
            (0, -degrees[i], -weights[i], i, 0)
        )

    bins = []
    remaining = n

    while remaining:
        while True:
            neg_sat, neg_deg, neg_weight, i, recorded_mask = heapq.heappop(heap)
            if assigned[i]:
                continue
            if recorded_mask != saturation_masks[i]:
                continue
            break

        wi = weights[i]
        best_bin = -1
        best_score = None

        for b_idx, b in enumerate(bins):
            if b["load"] + wi > capacity:
                continue
            if b["mask"] & adjacency[i]:
                continue
            residual = capacity - b["load"] - wi
            score = (residual, -len(b["items"]), b_idx)
            if best_score is None or score < best_score:
                best_score = score
                best_bin = b_idx

        if best_bin < 0:
            best_bin = len(bins)
            bins.append({
                "items": [i],
                "load": wi,
                "mask": 1 << i
            })
        else:
            b = bins[best_bin]
            b["items"].append(i)
            b["load"] += wi
            b["mask"] |= 1 << i

        assigned[i] = True
        assigned_bin[i] = best_bin
        remaining -= 1

        neighbor_bits = adjacency[i]
        while neighbor_bits:
            lsb = neighbor_bits & -neighbor_bits
            j = lsb.bit_length() - 1
            neighbor_bits ^= lsb

            if assigned[j]:
                continue

            old_mask = saturation_masks[j]
            new_mask = old_mask | (1 << best_bin)
            if new_mask != old_mask:
                saturation_masks[j] = new_mask
                heapq.heappush(
                    heap,
                    (
                        -new_mask.bit_count(),
                        -degrees[j],
                        -weights[j],
                        j,
                        new_mask
                    )
                )

    return canonicalize_bins(bins)


def try_eliminate_bin(bins, source_idx, weights, adjacency, capacity,
                      deadline, node_limit=20000):
    """Try to redistribute every item of one bin using bounded DFS."""
    source_items = list(bins[source_idx]["items"])
    targets = [b for idx, b in enumerate(bins) if idx != source_idx]

    if not source_items:
        return targets

    def initial_candidate_count(i):
        wi = weights[i]
        count = 0
        for b in targets:
            if b["load"] + wi <= capacity and not (b["mask"] & adjacency[i]):
                count += 1
        return count

    source_items.sort(
        key=lambda i: (
            initial_candidate_count(i),
            -degrees_global[i],
            -weights[i],
            i
        )
    )

    nodes = 0

    def dfs(pos):
        nonlocal nodes
        nodes += 1
        if nodes > node_limit or time.monotonic() >= deadline:
            return False
        if pos == len(source_items):
            return True

        i = source_items[pos]
        wi = weights[i]
        candidates = []

        for b_idx, b in enumerate(targets):
            if b["load"] + wi > capacity:
                continue
            if b["mask"] & adjacency[i]:
                continue
            candidates.append((
                capacity - b["load"] - wi,
                -len(b["items"]),
                b_idx
            ))

        candidates.sort()
        if not candidates:
            return False

        for _, _, b_idx in candidates:
            b = targets[b_idx]
            b["items"].append(i)
            b["load"] += wi
            b["mask"] |= 1 << i

            if dfs(pos + 1):
                return True

            b["items"].pop()
            b["load"] -= wi
            b["mask"] ^= 1 << i

        return False

    if dfs(0):
        return canonicalize_bins(targets)
    return None


def relocation_improvement(bins, weights, adjacency, capacity, deadline,
                           on_improvement):
    bins = copy_bins(bins)

    while time.monotonic() < deadline and len(bins) > 1:
        candidates = sorted(
            range(len(bins)),
            key=lambda idx: (
                bins[idx]["load"],
                len(bins[idx]["items"]),
                idx
            )
        )

        improved = False
        for source_idx in candidates:
            if time.monotonic() >= deadline:
                break

            result = try_eliminate_bin(
                bins, source_idx, weights, adjacency, capacity, deadline
            )
            if result is not None:
                bins = result
                on_improvement(bins)
                improved = True
                break

        if not improved:
            break

    return canonicalize_bins(bins)


def clique_lower_bound(adjacency, degrees):
    """Find several valid cliques; their maximum size is a valid lower bound."""
    n = len(adjacency)
    if n == 0:
        return 0

    starts = sorted(range(n), key=lambda i: (-degrees[i], i))
    starts = starts[:min(n, 64)]
    best = 1

    for start in starts:
        size = 1
        candidates = adjacency[start]

        while candidates:
            candidate_list = []
            bits = candidates
            while bits:
                lsb = bits & -bits
                v = lsb.bit_length() - 1
                bits ^= lsb
                candidate_list.append(v)

            v = max(
                candidate_list,
                key=lambda x: ((adjacency[x] & candidates).bit_count(),
                               degrees[x], -x)
            )
            size += 1
            candidates &= adjacency[v]

        best = max(best, size)

    return best


def solve_with_gurobi(best_bins, weights, adjacency, edges, capacity,
                       lower_bound, deadline, logger, item_ids):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return best_bins

    n = len(weights)
    upper_bound = len(best_bins)
    if n == 0 or upper_bound <= lower_bound:
        return best_bins

    remaining = deadline - time.monotonic()
    if remaining <= 0.05:
        return best_bins

    # Canonical-bin symmetry permits item i only in bins 0..i.
    variable_count = sum(min(i + 1, upper_bound) for i in range(n))
    conflict_constraint_count = 0
    for i, j in edges:
        conflict_constraint_count += min(i, j, upper_bound - 1) + 1

    # Avoid spending the entire time constructing an excessively large model.
    if variable_count > 700000 or conflict_constraint_count > 2500000:
        return best_bins

    try:
        model = gp.Model("bin_packing_with_conflicts")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1

        y = model.addVars(upper_bound, vtype=GRB.BINARY, name="y")
        x = {}
        vars_by_bin = [[] for _ in range(upper_bound)]

        for i in range(n):
            for b in range(min(i + 1, upper_bound)):
                var = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{b}")
                x[i, b] = var
                vars_by_bin[b].append((i, var))

        model.update()

        for i in range(n):
            model.addConstr(
                gp.quicksum(x[i, b] for b in range(min(i + 1, upper_bound)))
                == 1
            )

        for b in range(upper_bound):
            model.addConstr(
                gp.quicksum(weights[i] * var for i, var in vars_by_bin[b])
                <= capacity * y[b]
            )
            model.addConstr(
                gp.quicksum(var for _, var in vars_by_bin[b]) <= n * y[b]
            )

        for i, j in edges:
            common = min(i, j, upper_bound - 1)
            for b in range(common + 1):
                model.addConstr(x[i, b] + x[j, b] <= y[b])

        for b in range(upper_bound - 1):
            model.addConstr(y[b] >= y[b + 1])

        if n:
            model.addConstr(y[0] == 1)

        model.addConstr(gp.quicksum(y[b] for b in range(upper_bound))
                        >= lower_bound)
        model.setObjective(
            gp.quicksum(y[b] for b in range(upper_bound)),
            GRB.MINIMIZE
        )

        # Canonicalize the incumbent by the smallest item index in each bin.
        warm_bins = canonicalize_bins(copy_bins(best_bins))
        warm_assignment = {}
        for b, packed_bin in enumerate(warm_bins):
            for i in packed_bin["items"]:
                warm_assignment[i] = b

        for b in range(upper_bound):
            y[b].Start = 1.0 if b < len(warm_bins) else 0.0
        for (i, b), var in x.items():
            var.Start = 1.0 if warm_assignment.get(i) == b else 0.0

        x_keys = list(x.keys())
        x_vars = [x[key] for key in x_keys]
        incumbent_count = [len(best_bins)]
        incumbent_bins = [copy_bins(best_bins)]

        def values_to_bins(values):
            result = [
                {"items": [], "load": 0, "mask": 0}
                for _ in range(upper_bound)
            ]
            chosen = [-1] * n

            for key, value in zip(x_keys, values):
                if value > 0.5:
                    i, b = key
                    chosen[i] = b

            if any(b < 0 for b in chosen):
                return None

            for i, b in enumerate(chosen):
                result[b]["items"].append(i)
                result[b]["load"] += weights[i]
                result[b]["mask"] |= 1 << i

            return canonicalize_bins(result)

        def callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return
            try:
                obj = int(round(cb_model.cbGet(
                    GRB.Callback.MIPSOL_OBJ
                )))
                if obj >= incumbent_count[0]:
                    return

                values = cb_model.cbGetSolution(x_vars)
                candidate = values_to_bins(values)
                if candidate is None or len(candidate) >= incumbent_count[0]:
                    return

                incumbent_count[0] = len(candidate)
                incumbent_bins[0] = copy_bins(candidate)

                if logger:
                    solution = make_solution(candidate, item_ids, weights)
                    logger.log_solution(len(candidate), solution)
            except Exception:
                pass

        remaining = max(0.01, deadline - time.monotonic())
        model.Params.TimeLimit = remaining
        model.optimize(callback)

        if model.SolCount > 0:
            values = [var.X for var in x_vars]
            candidate = values_to_bins(values)
            if candidate is not None and len(candidate) < incumbent_count[0]:
                incumbent_count[0] = len(candidate)
                incumbent_bins[0] = copy_bins(candidate)
                if logger:
                    solution = make_solution(candidate, item_ids, weights)
                    logger.log_solution(len(candidate), solution)

        return canonicalize_bins(incumbent_bins[0])

    except Exception:
        return best_bins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") \
        if args.log_path else None

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    items_data = instance["items"]
    n = int(instance["num_items"])
    capacity = int(instance["bin_capacity"])

    if n != len(items_data):
        n = len(items_data)

    item_ids = [item["id"] for item in items_data]
    weights = [int(item["weight"]) for item in items_data]
    id_to_index = {item_id: i for i, item_id in enumerate(item_ids)}

    adjacency = [0] * n
    edge_set = set()

    for edge in instance.get("conflict_edges", []):
        if len(edge) != 2:
            continue
        a_id, b_id = edge
        if a_id not in id_to_index or b_id not in id_to_index:
            continue

        a = id_to_index[a_id]
        b = id_to_index[b_id]
        if a == b:
            continue
        if a > b:
            a, b = b, a

        if (a, b) not in edge_set:
            edge_set.add((a, b))
            adjacency[a] |= 1 << b
            adjacency[b] |= 1 << a

    edges = sorted(edge_set)
    degrees = [mask.bit_count() for mask in adjacency]

    global degrees_global
    degrees_global = degrees

    # Always begin with a guaranteed feasible incumbent.
    best_bins = [
        {
            "items": [i],
            "load": weights[i],
            "mask": 1 << i
        }
        for i in range(n)
    ]
    best_bins = canonicalize_bins(best_bins)

    best_solution = make_solution(best_bins, item_ids, weights)
    if logger:
        logger.log_solution(len(best_bins), best_solution)

    def register(candidate):
        nonlocal best_bins, best_solution
        candidate = canonicalize_bins(candidate)
        if len(candidate) < len(best_bins):
            best_bins = copy_bins(candidate)
            best_solution = make_solution(best_bins, item_ids, weights)
            if logger:
                logger.log_solution(len(best_bins), best_solution)

    if n == 0:
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(best_solution, f, indent=2)
        return

    # Deterministic greedy orderings.
    deterministic_orders = [
        sorted(range(n), key=lambda i: (-weights[i], -degrees[i], i)),
        sorted(range(n), key=lambda i: (-degrees[i], -weights[i], i)),
        sorted(
            range(n),
            key=lambda i: (
                -(weights[i] * (degrees[i] + 1)),
                -weights[i],
                -degrees[i],
                i
            )
        ),
        sorted(
            range(n),
            key=lambda i: (
                -(weights[i] / max(1, capacity)),
                -degrees[i],
                i
            )
        )
    ]

    for order in deterministic_orders:
        if time.monotonic() >= deadline:
            break
        register(greedy_pack(order, weights, adjacency, capacity))

    # DSATUR is especially useful when conflicts dominate capacity.
    if n <= 700 and time.monotonic() < deadline:
        register(dsatur_pack(weights, adjacency, degrees, capacity))

    # Randomized tie-breaking around weight/degree priorities.
    rng = random.Random(0)
    random_trials = 12 if n <= 1000 else 3
    for _ in range(random_trials):
        if time.monotonic() >= deadline:
            break
        noise = [rng.random() for _ in range(n)]
        mode = rng.randrange(3)
        if mode == 0:
            order = sorted(
                range(n),
                key=lambda i: (-weights[i], -degrees[i], noise[i])
            )
        elif mode == 1:
            order = sorted(
                range(n),
                key=lambda i: (-degrees[i], -weights[i], noise[i])
            )
        else:
            order = sorted(
                range(n),
                key=lambda i: (
                    -(weights[i] * (degrees[i] + 1)),
                    noise[i]
                )
            )
        register(greedy_pack(order, weights, adjacency, capacity))

    # Use a bounded amount of time to remove bins by exact redistribution.
    remaining = max(0.0, deadline - time.monotonic())
    if n <= 1200 and len(best_bins) <= 400 and remaining > 0.1:
        relocation_budget = min(3.0, remaining * 0.20)
        relocation_deadline = min(deadline, time.monotonic() + relocation_budget)
        improved = relocation_improvement(
            best_bins,
            weights,
            adjacency,
            capacity,
            relocation_deadline,
            register
        )
        register(improved)

    total_weight = sum(weights)
    if capacity > 0:
        capacity_lb = math.ceil(total_weight / capacity)
    else:
        capacity_lb = 1 if n else 0

    clique_lb = clique_lower_bound(adjacency, degrees)
    lower_bound = max(capacity_lb, clique_lb, 1 if n else 0)

    # Spend the remaining time proving optimality or improving the incumbent.
    if time.monotonic() < deadline and len(best_bins) > lower_bound:
        candidate = solve_with_gurobi(
            best_bins=best_bins,
            weights=weights,
            adjacency=adjacency,
            edges=edges,
            capacity=capacity,
            lower_bound=lower_bound,
            deadline=deadline,
            logger=logger,
            item_ids=item_ids
        )
        register(candidate)

    best_solution = make_solution(best_bins, item_ids, weights)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution, f, indent=2)


if __name__ == "__main__":
    main()