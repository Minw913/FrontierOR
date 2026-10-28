import argparse
import json
import time
import random
from collections import Counter

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except Exception:
    GUROBI_AVAILABLE = False

from solution_logger import SolutionLogger


def best_fit_packing(items, capacity):
    """Best-Fit (on given order). Returns list of bins, each bin is list of item sizes."""
    bins = []
    loads = []
    for it in items:
        best_idx = -1
        best_rem_after = None
        for b in range(len(bins)):
            if loads[b] + it <= capacity:
                rem_after = capacity - (loads[b] + it)
                if best_rem_after is None or rem_after < best_rem_after:
                    best_rem_after = rem_after
                    best_idx = b
        if best_idx == -1:
            bins.append([it])
            loads.append(it)
        else:
            bins[best_idx].append(it)
            loads[best_idx] += it
    return bins


def first_fit_packing(items, capacity):
    bins = []
    loads = []
    for it in items:
        placed = False
        for b in range(len(bins)):
            if loads[b] + it <= capacity:
                bins[b].append(it)
                loads[b] += it
                placed = True
                break
        if not placed:
            bins.append([it])
            loads.append(it)
    return bins


def validate_solution(bins, items, capacity):
    for b in bins:
        if sum(b) > capacity:
            return False
    return Counter(x for b in bins for x in b) == Counter(items)


def try_empty_bin_once(bins, capacity, idx):
    """Try to move all items from bin idx into other bins greedily (best-fit)."""
    candidate_items = sorted(bins[idx], reverse=True)
    other_bins = [list(b) for j, b in enumerate(bins) if j != idx]
    rem = [capacity - sum(b) for b in other_bins]

    for it in candidate_items:
        best_j = -1
        best_rem_after = None
        for j in range(len(other_bins)):
            if rem[j] >= it:
                rem_after = rem[j] - it
                if best_rem_after is None or rem_after < best_rem_after:
                    best_rem_after = rem_after
                    best_j = j
        if best_j == -1:
            return None
        other_bins[best_j].append(it)
        rem[best_j] -= it

    return other_bins


def local_improvement_by_bin_elimination(bins, capacity, time_deadline, update_incumbent):
    """Repeatedly try to eliminate one bin."""
    current = [list(b) for b in bins]
    improved = True
    while improved and time.time() < time_deadline:
        improved = False
        order = sorted(range(len(current)), key=lambda i: sum(current[i]))  # smallest load first
        for idx in order:
            if time.time() >= time_deadline:
                break
            if len(current) <= 1:
                break
            new_bins = try_empty_bin_once(current, capacity, idx)
            if new_bins is not None and len(new_bins) < len(current):
                current = new_bins
                update_incumbent(current)
                improved = True
                break
    return current


def mip_polish(items, capacity, incumbent_bins, remaining_time, logger, current_best_obj):
    """MIP polishing with aggregated item types. Returns improved bins or incumbent."""
    if not GUROBI_AVAILABLE or remaining_time <= 0.2:
        return incumbent_bins

    n_bins_ub = len(incumbent_bins)
    if n_bins_ub <= 1:
        return incumbent_bins

    counts = Counter(items)
    sizes = sorted(counts.keys(), reverse=True)
    demand = [counts[s] for s in sizes]
    m = len(sizes)
    U = n_bins_ub

    # Guard against too-large models
    if m * U > 300000:
        return incumbent_bins

    try:
        model = gp.Model("bin_packing")
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(0.1, remaining_time)
        model.Params.MIPFocus = 1

        x = {}
        y = {}
        for b in range(U):
            y[b] = model.addVar(vtype=GRB.BINARY, name=f"y_{b}")

        for t in range(m):
            ub_t = min(demand[t], capacity // sizes[t])
            for b in range(U):
                x[t, b] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=ub_t, name=f"x_{t}_{b}")

        # Demand satisfaction
        for t in range(m):
            model.addConstr(gp.quicksum(x[t, b] for b in range(U)) == demand[t], name=f"demand_{t}")

        # Capacity and linking
        loads = {}
        for b in range(U):
            loads[b] = gp.quicksum(sizes[t] * x[t, b] for t in range(m))
            model.addConstr(loads[b] <= capacity * y[b], name=f"cap_{b}")
            for t in range(m):
                ub_t = min(demand[t], capacity // sizes[t])
                model.addConstr(x[t, b] <= ub_t * y[b], name=f"link_{t}_{b}")

        # Symmetry breaking
        for b in range(U - 1):
            model.addConstr(y[b] >= y[b + 1], name=f"sym_y_{b}")
            model.addConstr(loads[b] >= loads[b + 1], name=f"sym_load_{b}")

        model.setObjective(gp.quicksum(y[b] for b in range(U)), GRB.MINIMIZE)

        # Warm start from incumbent
        for b in range(U):
            if b < len(incumbent_bins):
                y[b].Start = 1.0
                cbin = Counter(incumbent_bins[b])
                for t, s in enumerate(sizes):
                    x[t, b].Start = float(cbin.get(s, 0))
            else:
                y[b].Start = 0.0
                for t in range(m):
                    x[t, b].Start = 0.0

        cb_state = {"best": current_best_obj}

        def cb(model_cb, where):
            if where == GRB.Callback.MIPSOL:
                obj = model_cb.cbGet(GRB.Callback.MIPSOL_OBJ)
                val = int(round(obj))
                if val < cb_state["best"]:
                    cb_state["best"] = val
                    if logger:
                        logger.log(val)

        model.optimize(cb)

        if model.SolCount == 0:
            return incumbent_bins

        # Extract solution
        new_bins = []
        for b in range(U):
            if y[b].X > 0.5:
                b_items = []
                for t, s in enumerate(sizes):
                    q = int(round(x[t, b].X))
                    if q > 0:
                        b_items.extend([s] * q)
                if b_items:
                    new_bins.append(b_items)

        if len(new_bins) < len(incumbent_bins) and validate_solution(new_bins, items, capacity):
            return new_bins
        return incumbent_bins

    except Exception:
        return incumbent_bins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    args = parser.parse_args()

    start = time.time()
    rnd = random.Random(0)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    W = int(inst["bin_capacity"])
    items = list(inst["items"])
    # num_items field exists by schema; trust items list for actual data
    if len(items) == 0:
        sol = {"objective_value": 0, "bin_assignments": [], "num_bins": 0}
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    best_bins = None
    best_obj = 10**18

    def update_incumbent(candidate_bins):
        nonlocal best_bins, best_obj
        if candidate_bins is None:
            return
        k = len(candidate_bins)
        if k < best_obj and validate_solution(candidate_bins, items, W):
            best_obj = k
            best_bins = [list(b) for b in candidate_bins]
            if logger:
                logger.log(best_obj)

    # Deterministic constructive heuristics
    sorted_desc = sorted(items, reverse=True)
    update_incumbent(best_fit_packing(sorted_desc, W))
    update_incumbent(first_fit_packing(sorted_desc, W))

    # Randomized multi-start constructive phase
    def remaining_time():
        return args.time_limit - (time.time() - start)

    construct_deadline = start + min(max(0.5, 0.25 * args.time_limit), 3.0)
    while time.time() < construct_deadline and remaining_time() > 0.2:
        # randomized tie-breaking among similar sizes
        order = sorted(items, key=lambda x: (-x, rnd.random()))
        bins = best_fit_packing(order, W)
        update_incumbent(bins)

    # Local improvement phase (bin elimination)
    if best_bins is not None and remaining_time() > 0.5:
        improve_deadline = start + min(max(1.0, 0.65 * args.time_limit), args.time_limit - 0.2)
        improve_deadline = min(improve_deadline, start + args.time_limit - 0.2)
        best_bins = local_improvement_by_bin_elimination(
            best_bins, W, improve_deadline, update_incumbent
        )

    # MIP polishing phase with remaining time
    if best_bins is None:
        # fallback (should not happen)
        best_bins = best_fit_packing(sorted_desc, W)
        update_incumbent(best_bins)

    rem = remaining_time()
    if rem > 0.5:
        polished = mip_polish(items, W, best_bins, rem - 0.1, logger, best_obj)
        update_incumbent(polished)

    # Finalize
    if best_bins is None:
        best_bins = best_fit_packing(sorted_desc, W)
        best_obj = len(best_bins)

    # Prepare output schema
    bin_assignments = []
    for b in best_bins:
        total = int(sum(b))
        bin_assignments.append({
            "items": [int(x) for x in b],
            "total_size": total
        })

    solution = {
        "objective_value": int(len(best_bins)),
        "bin_assignments": bin_assignments,
        "num_bins": int(len(best_bins))
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()