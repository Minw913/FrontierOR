import argparse
import json
import math
import os
import random
import time
from collections import Counter

from solution_logger import SolutionLogger


def make_solution(bins):
    packed_bins = []
    for items in bins:
        if not items:
            continue
        total = int(sum(items))
        packed_bins.append({
            "items": [int(x) for x in items],
            "total_size": total,
        })
    count = len(packed_bins)
    return {
        "objective_value": count,
        "bin_assignments": packed_bins,
        "num_bins": count,
    }


def first_fit_segment_tree(order, capacity):
    """First-fit packing in O(n log n) using a segment tree of residual space."""
    n = len(order)
    if n == 0:
        return []

    leaf_count = 1
    while leaf_count < n:
        leaf_count <<= 1

    tree = [0] * (2 * leaf_count)
    residuals = [0] * n
    bins = []
    used = 0

    def update(index, value):
        pos = leaf_count + index
        tree[pos] = value
        pos >>= 1
        while pos:
            new_value = max(tree[pos << 1], tree[(pos << 1) | 1])
            if tree[pos] == new_value:
                tree[pos] = new_value
                pos >>= 1
                continue
            tree[pos] = new_value
            pos >>= 1

    def leftmost_with_space(required):
        pos = 1
        while pos < leaf_count:
            left = pos << 1
            if tree[left] >= required:
                pos = left
            else:
                pos = left | 1
        return pos - leaf_count

    for size in order:
        if used == 0 or tree[1] < size:
            idx = used
            used += 1
            bins.append([size])
            residuals[idx] = capacity - size
            update(idx, residuals[idx])
        else:
            idx = leftmost_with_space(size)
            bins[idx].append(size)
            residuals[idx] -= size
            update(idx, residuals[idx])

    return bins


def validate_solution(solution, original_items, capacity):
    if solution["objective_value"] != solution["num_bins"]:
        return False
    if solution["num_bins"] != len(solution["bin_assignments"]):
        return False

    produced = Counter()
    for entry in solution["bin_assignments"]:
        items = entry["items"]
        total = sum(items)
        if total != entry["total_size"] or total > capacity or total <= 0:
            return False
        produced.update(items)

    return produced == Counter(original_items)


def configure_gurobi(model, output=False):
    model.Params.OutputFlag = 1 if output else 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1


def solve_with_patterns(
    capacity,
    items,
    best_state,
    logger,
    start_time,
    deadline,
    pattern_seed_bins,
):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return

    if time.monotonic() >= deadline:
        return

    counts = Counter(items)
    sizes = sorted(counts.keys(), reverse=True)
    demands = [counts[s] for s in sizes]
    m = len(sizes)
    size_to_type = {s: i for i, s in enumerate(sizes)}

    patterns = []
    pattern_set = set()

    def add_pattern(pattern):
        pattern = tuple(int(x) for x in pattern)
        if not any(pattern) or pattern in pattern_set:
            return False
        if sum(pattern[i] * sizes[i] for i in range(m)) > capacity:
            return False
        pattern_set.add(pattern)
        patterns.append(pattern)
        return True

    # Homogeneous patterns guarantee that the restricted master is feasible.
    for i, size in enumerate(sizes):
        p = [0] * m
        p[i] = min(demands[i], capacity // size)
        add_pattern(p)

    # Seed the master with all patterns appearing in heuristic packings.
    for bins in pattern_seed_bins:
        for packed_bin in bins:
            p = [0] * m
            for size in packed_bin:
                p[size_to_type[size]] += 1
            add_pattern(p)

    if not patterns:
        return

    total_limit = max(0.0, deadline - start_time)
    now = time.monotonic()
    remaining = max(0.0, deadline - now)
    if remaining <= 0.02:
        return

    # Reserve most of the time for the integer restricted master.
    cg_budget = min(remaining * 0.38, max(1.0, total_limit * 0.38))
    cg_deadline = min(deadline, now + cg_budget)

    try:
        master = gp.Model("bin_packing_column_generation")
        configure_gurobi(master)

        cover = []
        for i in range(m):
            cover.append(master.addConstr(gp.LinExpr() >= demands[i], name=f"demand_{i}"))

        master_vars = []
        for j, p in enumerate(patterns):
            col = gp.Column()
            for i, coeff in enumerate(p):
                if coeff:
                    col.addTerms(coeff, cover[i])
            master_vars.append(
                master.addVar(
                    obj=1.0,
                    lb=0.0,
                    vtype=GRB.CONTINUOUS,
                    column=col,
                    name=f"p_{j}",
                )
            )
        master.ModelSense = GRB.MINIMIZE
        master.update()

        pricing = gp.Model("pricing")
        configure_gurobi(pricing)
        pricing.Params.PoolSearchMode = 1
        pricing.Params.PoolSolutions = 16

        qvars = []
        for i in range(m):
            upper = min(demands[i], capacity // sizes[i])
            qvars.append(
                pricing.addVar(
                    lb=0,
                    ub=upper,
                    vtype=GRB.INTEGER,
                    name=f"q_{i}",
                )
            )
        pricing.addConstr(
            gp.quicksum(sizes[i] * qvars[i] for i in range(m)) <= capacity,
            name="capacity",
        )
        pricing.ModelSense = GRB.MAXIMIZE
        pricing.update()

        iterations = 0
        while (
            time.monotonic() < cg_deadline
            and len(patterns) < 5000
            and iterations < 500
        ):
            iterations += 1
            remaining_cg = cg_deadline - time.monotonic()
            if remaining_cg <= 0.01:
                break

            master.Params.TimeLimit = max(0.01, remaining_cg)
            master.optimize()
            if master.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
                break
            if master.SolCount == 0:
                break

            try:
                duals = [max(0.0, float(c.Pi)) for c in cover]
            except Exception:
                break

            pricing.setObjective(
                gp.quicksum(duals[i] * qvars[i] for i in range(m)),
                GRB.MAXIMIZE,
            )
            remaining_cg = cg_deadline - time.monotonic()
            if remaining_cg <= 0.01:
                break
            pricing.Params.TimeLimit = max(0.01, min(2.0, remaining_cg))
            pricing.optimize()

            if pricing.SolCount == 0:
                break

            candidates = []
            solution_count = min(pricing.SolCount, 16)
            for sol_no in range(solution_count):
                pricing.Params.SolutionNumber = sol_no
                try:
                    value = float(pricing.PoolObjVal)
                    p = tuple(int(round(v.Xn)) for v in qvars)
                except Exception:
                    continue
                if value > 1.0 + 1e-7 and any(p):
                    candidates.append((value, p))

            candidates.sort(reverse=True)
            new_patterns = []
            for _, p in candidates:
                if add_pattern(p):
                    new_patterns.append(p)

            if not new_patterns:
                # If pricing was proven optimal, no improving unseen column exists.
                break

            for p in new_patterns:
                col = gp.Column()
                for i, coeff in enumerate(p):
                    if coeff:
                        col.addTerms(coeff, cover[i])
                master_vars.append(
                    master.addVar(
                        obj=1.0,
                        lb=0.0,
                        vtype=GRB.CONTINUOUS,
                        column=col,
                        name=f"p_{len(master_vars)}",
                    )
                )
            master.update()

    except Exception:
        # The heuristic solution remains available if column generation fails.
        return

    if time.monotonic() >= deadline:
        return

    def pattern_values_to_solution(values):
        remaining_demand = demands[:]
        packed_bins = []

        for j, raw_value in enumerate(values):
            multiplicity = int(round(raw_value))
            if multiplicity <= 0:
                continue
            p = patterns[j]

            for _ in range(multiplicity):
                packed = []
                for i, amount in enumerate(p):
                    take = min(amount, remaining_demand[i])
                    if take:
                        packed.extend([sizes[i]] * take)
                        remaining_demand[i] -= take
                if packed:
                    packed_bins.append(packed)

        if any(x > 0 for x in remaining_demand):
            return None
        return make_solution(packed_bins)

    try:
        integer_model = gp.Model("bin_packing_integer_master")
        configure_gurobi(integer_model)

        zvars = [
            integer_model.addVar(
                lb=0,
                ub=len(items),
                vtype=GRB.INTEGER,
                obj=1.0,
                name=f"z_{j}",
            )
            for j in range(len(patterns))
        ]
        integer_model.ModelSense = GRB.MINIMIZE

        for i in range(m):
            integer_model.addConstr(
                gp.quicksum(patterns[j][i] * zvars[j] for j in range(len(patterns)))
                >= demands[i],
                name=f"cover_{i}",
            )

        # Warm-start from the current best packing when its patterns are present.
        start_counts = Counter()
        for entry in best_state["solution"]["bin_assignments"]:
            p = [0] * m
            for size in entry["items"]:
                p[size_to_type[size]] += 1
            start_counts[tuple(p)] += 1

        pattern_index = {p: j for j, p in enumerate(patterns)}
        for p, amount in start_counts.items():
            j = pattern_index.get(p)
            if j is not None:
                zvars[j].Start = amount

        integer_model.update()

        lower_bound = math.ceil(sum(items) / capacity)

        def accept_solution(candidate):
            if candidate is None:
                return
            obj = candidate["objective_value"]
            if obj < best_state["objective"]:
                best_state["objective"] = obj
                best_state["solution"] = candidate
                if logger:
                    logger.log_solution(obj, candidate)

        def callback(model, where):
            if where != GRB.Callback.MIPSOL:
                return
            try:
                values = model.cbGetSolution(zvars)
                candidate = pattern_values_to_solution(values)
                accept_solution(candidate)
                if best_state["objective"] <= lower_bound:
                    model.terminate()
                elif time.monotonic() >= deadline:
                    model.terminate()
            except Exception:
                return

        remaining = deadline - time.monotonic()
        if remaining <= 0.01:
            return

        integer_model.Params.TimeLimit = max(0.01, remaining)
        integer_model.Params.BestObjStop = float(lower_bound)
        integer_model.optimize(callback)

        if integer_model.SolCount > 0:
            candidate = pattern_values_to_solution([v.X for v in zvars])
            accept_solution(candidate)

    except Exception:
        return


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
        instance = json.load(f)

    capacity = int(instance["bin_capacity"])
    items = [int(x) for x in instance["items"]]
    declared_n = int(instance["num_items"])

    if declared_n != len(items):
        raise ValueError("num_items does not match the length of items")
    if capacity <= 0:
        raise ValueError("bin_capacity must be positive")
    if any(x <= 0 or x > capacity for x in items):
        raise ValueError("Every item must have size in [1, bin_capacity]")

    if not items:
        solution = make_solution([])
        if logger:
            logger.log_solution(0, solution)
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(solution, f, separators=(",", ":"))
        return

    # Deterministic first-fit decreasing gives an immediate feasible incumbent.
    descending = sorted(items, reverse=True)
    initial_bins = first_fit_segment_tree(descending, capacity)
    initial_solution = make_solution(initial_bins)

    best_state = {
        "objective": initial_solution["objective_value"],
        "solution": initial_solution,
    }
    pattern_seed_bins = [initial_bins]

    if logger:
        logger.log_solution(best_state["objective"], best_state["solution"])

    lower_bound = math.ceil(sum(items) / capacity)

    # Generate several inexpensive diversified packings and use their patterns
    # to strengthen the later restricted master problem.
    rng = random.Random(0)
    if best_state["objective"] > lower_bound:
        trial_count = 0
        max_trials = 8 if len(items) <= 100000 else 3

        while trial_count < max_trials and time.monotonic() < deadline:
            trial_count += 1

            # Perturb the descending order while retaining a strong size bias.
            decorated = []
            for idx, size in enumerate(items):
                noise = rng.random()
                key = -(size * (1.0 + 0.10 * noise))
                decorated.append((key, idx, size))
            decorated.sort()
            order = [entry[2] for entry in decorated]

            bins = first_fit_segment_tree(order, capacity)
            pattern_seed_bins.append(bins)
            candidate = make_solution(bins)

            if candidate["objective_value"] < best_state["objective"]:
                best_state["objective"] = candidate["objective_value"]
                best_state["solution"] = candidate
                if logger:
                    logger.log_solution(candidate["objective_value"], candidate)

            if best_state["objective"] <= lower_bound:
                break

    if best_state["objective"] > lower_bound and time.monotonic() < deadline:
        solve_with_patterns(
            capacity=capacity,
            items=items,
            best_state=best_state,
            logger=logger,
            start_time=start_time,
            deadline=deadline,
            pattern_seed_bins=pattern_seed_bins,
        )

    final_solution = best_state["solution"]
    if not validate_solution(final_solution, items, capacity):
        # This should never be needed, but preserves a valid output if a solver
        # callback was interrupted at an inconvenient point.
        final_solution = initial_solution

    solution_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if solution_dir:
        os.makedirs(solution_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, separators=(",", ":"))


if __name__ == "__main__":
    main()