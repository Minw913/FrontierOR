import argparse
import bisect
import json
import math
import os
import time
from collections import Counter, defaultdict

from solution_logger import SolutionLogger

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None


def lower_bound_items(width_demand, capacity):
    """Simple valid lower bounds on the number of bins."""
    total_width = sum(w * d for w, d in width_demand.items())
    lb = int(math.ceil(total_width / capacity)) if capacity else 0

    # If (m+1)*w > capacity, at most m such items fit in one bin.
    for m in range(1, 6):
        count = sum(d for w, d in width_demand.items()
                    if (m + 1) * w > capacity)
        if count:
            lb = max(lb, int(math.ceil(count / float(m))))
    return lb


def best_fit_decreasing(items, capacity, width_by_type):
    """
    items: list of (width, type_id).
    Returns a list of bins; every bin is a dict type_id -> number of copies.
    """
    active = []     # Sorted list of (residual_capacity, original_bin_id)
    bins = []

    for width, tid in sorted(items, key=lambda x: (-x[0], x[1])):
        pos = bisect.bisect_left(active, (width, -1))
        if pos == len(active):
            bins.append({tid: 1})
            active.append((capacity - width, len(bins) - 1))
        else:
            residual, original_id = active.pop(pos)
            bins[original_id][tid] = bins[original_id].get(tid, 0) + 1
            new_residual = residual - width
            bisect.insort(active, (new_residual, original_id))
    return bins


def try_empty_bin(bins, capacity, width_by_type, target_pos):
    """Greedily redistribute all items from bins[target_pos] into other bins."""
    if target_pos < 0 or target_pos >= len(bins):
        return False

    loads = []
    for rec in bins:
        loads.append(sum(width_by_type[tid] * cnt for tid, cnt in rec.items()))

    target_items = []
    for tid, cnt in bins[target_pos].items():
        width = width_by_type[tid]
        target_items.extend((tid, width) for _ in range(cnt))
    target_items.sort(key=lambda x: (-x[1], x[0]))

    residual_by_pos = {}
    sorted_residuals = []
    for pos in range(len(bins)):
        if pos == target_pos:
            continue
        residual = capacity - loads[pos]
        residual_by_pos[pos] = residual
        sorted_residuals.append((residual, pos))
    sorted_residuals.sort()

    placements = []
    for tid, width in target_items:
        idx = bisect.bisect_left(sorted_residuals, (width, -1))
        if idx == len(sorted_residuals):
            return False
        residual, pos = sorted_residuals.pop(idx)
        new_residual = residual - width
        bisect.insort(sorted_residuals, (new_residual, pos))
        placements.append((pos, tid))

    if not placements and all(cnt == 0 for cnt in bins[target_pos].values()):
        pass
    else:
        for pos, tid in placements:
            bins[pos][tid] = bins[pos].get(tid, 0) + 1

    del bins[target_pos]
    return True


def improve_locally(bins, capacity, width_by_type, lower_bound, deadline):
    """Try a bounded number of easy bin-removal improvements."""
    if len(bins) <= max(lower_bound, 1):
        return bins

    while time.monotonic() < deadline and len(bins) > max(lower_bound, 1):
        loads = [
            (sum(width_by_type[tid] * cnt for tid, cnt in rec.items()), pos)
            for pos, rec in enumerate(bins)
        ]
        # Lightly filled bins are the most promising candidates to eliminate.
        candidates = sorted(loads)[:30]
        changed = False
        for _, pos in candidates:
            if time.monotonic() >= deadline:
                break
            if try_empty_bin(bins, capacity, width_by_type, pos):
                changed = True
                break
        if not changed:
            break
    return bins


def pattern_to_arcs(bins, width_by_type, capacity):
    """
    Convert packed bins to an aggregated arc-flow witness.
    x is indexed by (type_id, width, start_length); z by ending length.
    Returns (x_counts, z_counts).
    """
    x_counts = Counter()
    z_counts = Counter()

    for rec in bins:
        node = 0
        ordered_items = []
        for tid, count in rec.items():
            width = width_by_type[tid]
            ordered_items.extend((tid, width) for _ in range(count))
        ordered_items.sort(key=lambda x: (-x[1], x[0]))

        for tid, width in ordered_items:
            x_counts[(tid, width, node)] += 1
            node += width

        # Only partial/empty bins need an explicit loss arc to capacity.
        if node < capacity:
            z_counts[node] += 1

    return x_counts, z_counts


def make_solution_dict(objective, bins, width_by_type, capacity):
    x_counts, z_counts = pattern_to_arcs(bins, width_by_type, capacity)
    model_variables = {}

    if objective > 0:
        model_variables["n_bins"] = int(objective)

    for (tid, width, start), count in sorted(x_counts.items()):
        model_variables[f"x_{start}_{start + width}_{tid}"] = int(count)

    for start, count in sorted(z_counts.items()):
        model_variables[f"z_{start}_{capacity}"] = int(count)

    return {
        "objective_value": int(objective),
        "model_variables": model_variables
    }


def reachable_positions(widths, capacity):
    """Subset sums attainable within one bin, using a large integer bitset."""
    mask = (1 << (capacity + 1)) - 1
    bits = 1
    for width in widths:
        if width <= 0:
            continue
        remaining_copies = capacity // width
        power = 1
        while remaining_copies:
            take = min(remaining_copies, power)
            bits = ((bits << (take * width)) | bits) & mask
            remaining_copies -= take
            power <<= 1

    positions = []
    while bits:
        low_bit = bits & -bits
        node = low_bit.bit_length() - 1
        if node <= capacity:
            positions.append(node)
        bits -= low_bit

    # Loss arcs need the capacity node even when it is not an item-sum.
    if capacity not in positions:
        positions.append(capacity)
    positions.sort()
    return positions


def solve_with_arc_flow(types, capacity, upper_bound, logger):
    """
    Build and solve an exact integer arc-flow formulation.

    Each item type has a parallel arc class.  Variable n_bins is a boundary
    flow from node 0 to node capacity; its value is minimized.
    """
    if gp is None or upper_bound <= 0:
        return None

    # A hierarchy of model-size safeguards keeps infeasibly large graphs on
    # the deterministic heuristic fallback.
    now = time.monotonic()
    remaining_estimate_global = SOLVE_CONTEXT["deadline"] - now
    if remaining_estimate_global <= 1.5:
        return None

    if remaining_estimate_global >= 90:
        variable_safeguard = 1_500_000
    elif remaining_estimate_global >= 30:
        variable_safeguard = 800_000
    else:
        variable_safeguard = 300_000

    type_count = len(types)
    pessimistic_estimate = (capacity + 1) * (type_count + 1)
    if pessimistic_estimate > variable_safeguard:
        return None

    widths = sorted({t["width"] for t in types})
    try:
        nodes = reachable_positions(widths, capacity)
    except (MemoryError, OverflowError):
        return None

    distinct_width_positions = len(nodes)
    estimated_variables = distinct_width_positions * (type_count + 1)
    if estimated_variables > variable_safeguard:
        return None

    x_vars = {}
    z_vars = {}
    demand_lists = defaultdict(list)
    node_terms = {node: [] for node in nodes}

    try:
        model = gp.Model("cutting_stock_arc_flow")
        model.setParam("Seed", 0)
        model.setParam("MIPGap", 1e-4)
        model.setParam("NumericFocus", 0)
        model.setParam("Threads", 1)
        model.setParam("OutputFlag", 0)

        n_bins = model.addVar(
            lb=SOLVE_CONTEXT["lower_bound"],
            ub=upper_bound,
            vtype=GRB.INTEGER,
            name="n_bins"
        )

        # Item arcs.
        for item in types:
            tid = item["type_id"]
            width = item["width"]
            demand = item["demand"]
            for start in nodes:
                end = start + width
                if end > capacity:
                    continue
                var = model.addVar(
                    lb=0,
                    ub=demand,
                    vtype=GRB.INTEGER,
                    name=f"x_{start}_{end}_{tid}"
                )
                key = (tid, width, start)
                x_vars[key] = var
                demand_lists[tid].append(var)
                node_terms[start].append((var, -1.0))
                node_terms[end].append((var, 1.0))

        # Loss arcs complete each bin path at capacity.
        for start in nodes:
            if start == capacity:
                continue
            var = model.addVar(
                lb=0,
                ub=upper_bound,
                vtype=GRB.INTEGER,
                name=f"z_{start}_{capacity}"
            )
            z_vars[start] = var
            node_terms[start].append((var, -1.0))
            node_terms[capacity].append((var, 1.0))

        # Inject flow at node 0 and absorb it at capacity.
        node_terms[0].append((n_bins, 1.0))
        node_terms[capacity].append((n_bins, -1.0))

        # Flow balance ensures that every unit of flow is one complete bin.
        for node, terms in node_terms.items():
            if not terms:
                continue
            expr = gp.LinExpr()
            expr.addTerms([coef for _, coef in terms],
                          [var for var, _ in terms])
            model.addConstr(expr == 0, name=f"flow_balance_{node}")

        # Demand for each type is the total flow over its parallel arc class.
        for item in types:
            tid = item["type_id"]
            model.addConstr(
                gp.quicksum(demand_lists[tid]) == item["demand"],
                name=f"demand_{tid}"
            )

        model.setObjective(n_bins, GRB.MINIMIZE)

        # Install the heuristic packing as a MIP start.
        x_start, z_start = SOLVE_CONTEXT["arc_start"]
        model.update()
        try:
            n_bins.Start = upper_bound
            for key, value in x_start.items():
                if key in x_vars:
                    x_vars[key].Start = value
            for key, value in z_start.items():
                if key in z_vars:
                    z_vars[key].Start = value
        except Exception:
            pass

        solver_time = SOLVE_CONTEXT["deadline"] - time.monotonic() - 0.5
        if solver_time <= 0:
            model.dispose()
            return None
        model.setParam("TimeLimit", max(0.01, solver_time))

        best_seen = upper_bound

        def incumbent_callback(cb_model, where):
            nonlocal best_seen
            if logger is None or where != GRB.Callback.MIPSOL:
                return
            try:
                val = int(round(cb_model.cbGetSolution(n_bins)))
                if val < best_seen:
                    best_seen = val
                    logger.log(val)
            except Exception:
                pass

        model.optimize(incumbent_callback if logger is not None else None)

        if model.SolCount <= 0:
            model.dispose()
            return None

        objective = int(round(n_bins.X))
        model_variables = {}
        if objective > 0:
            model_variables["n_bins"] = objective

        for (tid, width, start), var in x_vars.items():
            value = int(round(var.X))
            if value > 0:
                model_variables[f"x_{start}_{start + width}_{tid}"] = value

        for start, var in z_vars.items():
            value = int(round(var.X))
            if value > 0:
                model_variables[f"z_{start}_{capacity}"] = value

        result = {
            "objective_value": objective,
            "model_variables": model_variables
        }
        model.dispose()
        return result

    except Exception:
        try:
            model.dispose()
        except Exception:
            pass
        return None


# Context avoids repeatedly passing a large collection of immutable data.
SOLVE_CONTEXT = {
    "deadline": 0.0,
    "lower_bound": 0,
    "arc_start": (Counter(), Counter())
}


def write_solution(path, solution):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(solution, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    total_time = max(0.0, float(args.time_limit))
    deadline = start_time + total_time

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        instance = json.load(f)

    capacity = int(instance["bin_capacity"])
    types = []
    width_demand = defaultdict(int)
    width_by_type = {}

    for item in instance.get("items", []):
        tid = int(item["type_id"])
        width = int(item["width"])
        demand = int(item["demand"])
        if demand <= 0:
            continue
        if width <= 0 or width > capacity:
            raise ValueError(f"Item type {tid} has infeasible width {width}")
        normalized = {
            "type_id": tid,
            "width": width,
            "demand": demand
        }
        types.append(normalized)
        width_demand[width] += demand
        width_by_type[tid] = width

    if not types:
        empty_solution = {
            "objective_value": 0,
            "model_variables": {}
        }
        write_solution(args.solution_path, empty_solution)
        return

    all_items = []
    for item in types:
        all_items.extend((item["width"], item["type_id"])
                         for _ in range(item["demand"]))

    bins = best_fit_decreasing(all_items, capacity, width_by_type)
    lower = lower_bound_items(width_demand, capacity)

    local_soft_deadline = min(
        deadline,
        start_time + min(1.0, max(0.05, total_time * 0.10))
    )
    bins = improve_locally(
        bins,
        capacity,
        width_by_type,
        lower,
        local_soft_deadline
    )

    objective = len(bins)
    solution = make_solution_dict(objective, bins, width_by_type, capacity)

    if logger and solution["objective_value"] > 0:
        logger.log_solution(objective, solution)

    # If the deterministic bound is already attained, it is provably optimal.
    if objective > lower:
        SOLVE_CONTEXT["deadline"] = deadline
        SOLVE_CONTEXT["lower_bound"] = lower
        SOLVE_CONTEXT["arc_start"] = pattern_to_arcs(
            bins, width_by_type, capacity
        )
        improved = solve_with_arc_flow(types, capacity, objective, logger)
        if (improved is not None and
                int(improved["objective_value"]) <= objective):
            objective = int(improved["objective_value"])
            solution = improved
            if logger and objective < len(bins):
                logger.log_solution(objective, solution)

    write_solution(args.solution_path, solution)


if __name__ == "__main__":
    main()