"""One-dimensional bin packing (cutting stock) solver.

Strategy:
  1. Fast constructive heuristic (best-fit decreasing) + a bin-elimination
     local search to obtain a strong initial feasible solution.
  2. An exact aggregated assignment MIP (items of equal size are grouped by
     type) solved with Gurobi, warm-started from the heuristic solution and
     strengthened with a lower-bound cut plus symmetry-breaking constraints.
  3. The best solution found within --time_limit is written to disk, and every
     improving incumbent is recorded through solution_logger.
"""

import argparse
import bisect
import json
import time
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(description="One-dimensional bin packing solver.")
    parser.add_argument("--instance_path", required=True,
                        help="Path to the JSON problem instance.")
    parser.add_argument("--solution_path", required=True,
                        help="Path where the solution JSON is written.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", default=None,
                        help="Optional path to a JSONL convergence log.")
    return parser.parse_args()


def build_solution(bins):
    """Project the internal list-of-bins representation onto the output schema."""
    assignments = []
    for b in bins:
        assignments.append({
            "items": [int(s) for s in b],
            "total_size": int(sum(b)),
        })
    return {
        "objective_value": len(assignments),
        "bin_assignments": assignments,
        "num_bins": len(assignments),
    }


def best_fit_decreasing(items, capacity):
    """Classic best-fit decreasing heuristic; returns a list of bins (item lists)."""
    bins = []
    open_rem = []  # sorted list of (remaining_capacity, bin_index)
    for s in sorted(items, reverse=True):
        pos = bisect.bisect_left(open_rem, (s, -1))
        if pos < len(open_rem):
            rem, idx = open_rem.pop(pos)
            bins[idx].append(s)
            rem -= s
            if rem > 0:
                bisect.insort(open_rem, (rem, idx))
        else:
            bins.append([s])
            rem = capacity - s
            if rem > 0:
                bisect.insort(open_rem, (rem, len(bins) - 1))
    return bins


def improve_solution(bins, capacity, deadline):
    """Try to empty the least-loaded bins by moving their items into other bins."""
    bins = [list(b) for b in bins]
    while time.time() < deadline:
        sums = [sum(b) for b in bins]
        order = sorted(range(len(bins)), key=lambda i: sums[i])
        removed = False
        for idx in order[:10]:
            if time.time() >= deadline:
                break
            rems = [capacity - sums[k] for k in range(len(bins))]
            rems[idx] = -1  # do not place items back into the bin being emptied
            moving = sorted(bins[idx], reverse=True)
            targets = {}
            feasible = True
            for s in moving:
                if time.time() >= deadline:
                    feasible = False
                    break
                best = -1
                best_rem = capacity + 1
                for k in range(len(bins)):
                    r = rems[k]
                    if s <= r < best_rem:
                        best = k
                        best_rem = r
                if best < 0:
                    feasible = False
                    break
                rems[best] -= s
                targets.setdefault(best, []).append(s)
            if feasible:
                for k, added in targets.items():
                    bins[k].extend(added)
                del bins[idx]
                removed = True
                break
        if not removed:
            break
    return bins


def solve_with_gurobi(counts, capacity, init_bins, time_budget, logger, best_logged):
    """Aggregated assignment MIP over item types, warm-started from init_bins."""
    import gurobipy as gp
    from gurobipy import GRB

    started = time.time()
    sizes = sorted(counts.keys())
    demands = [counts[s] for s in sizes]
    num_types = len(sizes)
    num_bins = len(init_bins)
    total_size = sum(s * d for s, d in zip(sizes, demands))
    lower_bound = max(
        (total_size + capacity - 1) // capacity,
        sum(d for s, d in zip(sizes, demands) if 2 * s > capacity),
    )

    model = gp.Model("bin_packing")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0

    # y[j] = 1 if bin j is used; x[t, j] = copies of type t packed into bin j.
    y = model.addVars(num_bins, vtype=GRB.BINARY, name="y")
    x = {}
    for t in range(num_types):
        ub_t = min(demands[t], capacity // sizes[t])
        for j in range(num_bins):
            x[t, j] = model.addVar(lb=0, ub=ub_t, vtype=GRB.INTEGER,
                                   name="x_%d_%d" % (t, j))

    model.setObjective(gp.quicksum(y[j] for j in range(num_bins)), GRB.MINIMIZE)
    model.addConstr(gp.quicksum(y[j] for j in range(num_bins)) >= lower_bound,
                    name="bin_lower_bound")

    for t in range(num_types):
        model.addConstr(
            gp.quicksum(x[t, j] for j in range(num_bins)) == demands[t],
            name="demand_%d" % t)

    load_exprs = []
    for j in range(num_bins):
        load = gp.quicksum(sizes[t] * x[t, j] for t in range(num_types))
        load_exprs.append(load)
        model.addConstr(load <= capacity * y[j], name="capacity_%d" % j)
    model.addConstr(gp.quicksum(load_exprs) == total_size, name="total_load")

    # Symmetry breaking: bins are opened in order and sorted by load.
    for j in range(num_bins - 1):
        model.addConstr(y[j] >= y[j + 1], name="open_order_%d" % j)
        model.addConstr(load_exprs[j] >= load_exprs[j + 1], name="load_order_%d" % j)

    # MIP start from the heuristic solution.
    for j, b in enumerate(init_bins):
        y[j].Start = 1.0
        bc = Counter(b)
        for t, s in enumerate(sizes):
            v = bc.get(s)
            if v:
                x[t, j].Start = float(v)

    var_list = [x[t, j] for j in range(num_bins) for t in range(num_types)]

    def bins_from_values(values):
        result = []
        k = 0
        for j in range(num_bins):
            bin_items = []
            for t in range(num_types):
                cnt = int(round(values[k]))
                k += 1
                if cnt > 0:
                    bin_items.extend([sizes[t]] * cnt)
            if bin_items:
                result.append(bin_items)
        return result

    state = {"best": int(best_logged)}

    def _callback(m, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            try:
                obj = int(round(m.cbGet(GRB.Callback.MIPSOL_OBJ)))
                if obj < state["best"]:
                    values = m.cbGetSolution(var_list)
                    logger.log_solution(obj, build_solution(bins_from_values(values)))
                    state["best"] = obj
            except Exception:
                pass

    # Spend only the time that is actually left after model construction.
    model.Params.TimeLimit = max(1.0, float(time_budget - (time.time() - started)))
    model.optimize(_callback)

    if model.SolCount > 0:
        values = model.getAttr("X", var_list)
        return bins_from_values(values), state["best"]
    return None, state["best"]


def main():
    args = parse_args()
    start_time = time.time()
    time_limit = max(1, int(args.time_limit))

    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path, "r") as fh:
        instance = json.load(fh)

    capacity = int(instance["bin_capacity"])
    items = [int(v) for v in instance.get("items", [])]
    counts = Counter(items)

    # --- Heuristic incumbent -------------------------------------------------
    bins = best_fit_decreasing(items, capacity) if items else []
    if bins and len(bins) <= 5000:
        budget = max(0.5, min(3.0, 0.05 * time_limit))
        bins = improve_solution(bins, capacity, time.time() + budget)
    bins.sort(key=lambda b: sum(b), reverse=True)  # matches load-order constraints

    best_bins = bins
    best_obj = len(best_bins)
    best_solution = build_solution(best_bins)
    best_logged = best_obj
    if logger is not None:
        logger.log_solution(best_obj, best_solution)

    # --- Exact refinement with Gurobi ---------------------------------------
    total_size = sum(items)
    lower_bound = max(
        (total_size + capacity - 1) // capacity if items else 0,
        sum(d for s, d in counts.items() if 2 * s > capacity),
    )
    remaining = time_limit - (time.time() - start_time)
    model_size = len(counts) * max(1, len(best_bins))
    if (items and best_obj > lower_bound and remaining > 2.0
            and model_size <= 400000):
        try:
            mip_bins, best_logged = solve_with_gurobi(
                counts, capacity, best_bins, remaining, logger, best_logged)
            if mip_bins is not None and len(mip_bins) < best_obj:
                best_bins = mip_bins
                best_obj = len(best_bins)
                best_solution = build_solution(best_bins)
                if logger is not None and best_obj < best_logged:
                    logger.log_solution(best_obj, best_solution)
                    best_logged = best_obj
        except Exception:
            pass  # keep the heuristic solution

    with open(args.solution_path, "w") as fh:
        json.dump(best_solution, fh)


if __name__ == "__main__":
    main()