import json
import argparse
import time

import numpy as np

INF = np.iinfo(np.int64).max // 4


def evaluate_selected(C, selected):
    """Evaluate a set of open medians: assignment cost, second-best cost, assigned median."""
    n = C.shape[0]
    best_cost = np.full(n, INF, dtype=np.int64)
    second_cost = np.full(n, INF, dtype=np.int64)
    best_med = np.full(n, -1, dtype=np.int32)

    for j in selected:
        j = int(j)
        col = C[:, j]

        mask = col < best_cost
        if np.any(mask):
            second_cost[mask] = best_cost[mask]
            best_cost[mask] = col[mask]
            best_med[mask] = j

        mask2 = (~mask) & (col < second_cost)
        if np.any(mask2):
            second_cost[mask2] = col[mask2]

    objective = int(best_cost.sum(dtype=np.int64))
    return objective, best_cost, second_cost, best_med


def build_solution(C, selected, best_med):
    """Build the required output dictionary from a median set and assignments."""
    n = C.shape[0]
    assignments = []
    total = 0

    for i in range(n):
        facility = int(best_med[i])
        cost = int(C[i, facility])
        total += cost
        assignments.append({
            "customer": i,
            "facility": facility,
            "cost": cost
        })

    return {
        "objective_value": int(total),
        "opened_facilities": sorted(int(j) for j in selected),
        "assignments": assignments
    }


def sanitize_selected(selected, n, p):
    """Make sure the selected facility list has exactly p distinct valid indices."""
    seen = set()
    result = []

    for j in selected:
        try:
            j = int(j)
        except Exception:
            continue
        if 0 <= j < n and j not in seen:
            seen.add(j)
            result.append(j)
            if len(result) == p:
                return result

    if len(result) < p:
        for j in range(n):
            if j not in seen:
                seen.add(j)
                result.append(j)
                if len(result) == p:
                    break

    return result[:p]


def random_selected(n, p, rng):
    if p <= 0:
        return []
    if p >= n:
        return list(range(n))
    return [int(x) for x in rng.choice(n, size=p, replace=False)]


def weighted_construct(C, p, rng, end_time):
    """Fast construction: start from best 1-median, then sample poorly served nodes."""
    n = C.shape[0]
    if p <= 0:
        return []
    if p >= n:
        return list(range(n))

    col_sums = C.sum(axis=0, dtype=np.int64)
    first = int(np.argmin(col_sums))

    selected = [first]
    selected_bool = np.zeros(n, dtype=bool)
    selected_bool[first] = True

    current_cost = C[:, first].astype(np.int64)

    while len(selected) < p and time.time() < end_time:
        weights = current_cost.astype(np.float64)
        weights[selected_bool] = 0.0
        total = float(weights.sum())

        if not np.isfinite(total) or total <= 0.0:
            closed = np.where(~selected_bool)[0]
            if len(closed) == 0:
                break
            j = int(rng.choice(closed))
        else:
            cum = np.cumsum(weights)
            r = rng.random() * total
            j = int(np.searchsorted(cum, r, side="right"))
            if j >= n:
                j = n - 1
            if selected_bool[j]:
                closed = np.where(~selected_bool)[0]
                if len(closed) == 0:
                    break
                j = int(rng.choice(closed))

        selected.append(j)
        selected_bool[j] = True
        np.minimum(current_cost, C[:, j].astype(np.int64), out=current_cost)

    if len(selected) < p:
        closed = np.where(~selected_bool)[0]
        need = p - len(selected)
        if len(closed) > need:
            extra = rng.choice(closed, size=need, replace=False)
        else:
            extra = closed
        selected.extend(int(x) for x in extra)

    return selected


def top_candidates_by_gain(C, current_cost, closed_mask, k, end_time, rng):
    """Return candidates with large estimated gain if added as a new median."""
    n = C.shape[0]
    valid_all = np.where(closed_mask)[0]

    if k <= 0 or len(valid_all) == 0:
        return np.array([], dtype=np.int64)

    k = min(k, len(valid_all))

    # For larger instances, estimate gains on a sample of customer rows.
    max_rows = n if n <= 1500 else min(1500, max(300, n // 4))
    full = max_rows >= n

    if not full:
        rows = rng.choice(n, size=max_rows, replace=False)
        rows = np.sort(rows)
        row_count = max_rows
    else:
        rows = None
        row_count = n

    processed = np.zeros(n, dtype=bool)
    gains = np.zeros(n, dtype=np.int64)

    chunk = max(1, min(n, int(16_000_000 // max(1, 8 * row_count))))

    for st in range(0, n, chunk):
        if time.time() > end_time:
            break

        en = min(n, st + chunk)
        processed[st:en] = True

        if not np.any(closed_mask[st:en]):
            continue

        if full:
            diff = current_cost[:, None] - C[:, st:en]
        else:
            diff = current_cost[rows, None] - C[rows, st:en]

        np.maximum(diff, 0, out=diff)
        gains[st:en] = diff.sum(axis=0, dtype=np.int64)

    valid_proc = valid_all[processed[valid_all]]

    if len(valid_proc) == 0:
        return rng.choice(valid_all, size=k, replace=False)

    if len(valid_proc) < k:
        return rng.choice(valid_all, size=k, replace=False)

    g = gains[valid_proc]
    part = np.argpartition(g, -k)[-k:]
    top = valid_proc[part]
    top = top[np.argsort(gains[top])[::-1]]

    return top.astype(np.int64)


def greedy_construct(C, p, rng, end_time):
    """Greedy add construction using (possibly approximate) gain evaluations."""
    n = C.shape[0]
    if p <= 0:
        return []
    if p >= n:
        return list(range(n))

    col_sums = C.sum(axis=0, dtype=np.int64)
    first = int(np.argmin(col_sums))

    selected = [first]
    selected_bool = np.zeros(n, dtype=bool)
    selected_bool[first] = True

    current_cost = C[:, first].astype(np.int64)

    while len(selected) < p and time.time() < end_time:
        closed_mask = ~selected_bool
        candidates = top_candidates_by_gain(C, current_cost, closed_mask, 1, end_time, rng)
        if len(candidates) == 0:
            break

        j = int(candidates[0])
        selected.append(j)
        selected_bool[j] = True
        np.minimum(current_cost, C[:, j].astype(np.int64), out=current_cost)

    if len(selected) < p:
        closed = np.where(~selected_bool)[0]
        need = p - len(selected)
        if len(closed) > need:
            extra = rng.choice(closed, size=need, replace=False)
        else:
            extra = closed
        selected.extend(int(x) for x in extra)

    return selected


def kmeanspp_construct(coords, p, rng, end_time):
    """k-means++ style initialization using coordinates, when available."""
    n = len(coords)
    if p <= 0:
        return []
    if p >= n:
        return list(range(n))

    arr = np.asarray(coords, dtype=np.float64)

    first = int(rng.integers(n))
    selected = [first]
    selected_bool = np.zeros(n, dtype=bool)
    selected_bool[first] = True

    min_d2 = ((arr - arr[first]) ** 2).sum(axis=1)

    while len(selected) < p and time.time() < end_time:
        weights = min_d2.copy()
        weights[selected_bool] = 0.0
        total = float(weights.sum())

        if not np.isfinite(total) or total <= 0.0:
            closed = np.where(~selected_bool)[0]
            if len(closed) == 0:
                break
            j = int(rng.choice(closed))
        else:
            cum = np.cumsum(weights)
            r = rng.random() * total
            j = int(np.searchsorted(cum, r, side="right"))
            if j >= n:
                j = n - 1
            if selected_bool[j]:
                closed = np.where(~selected_bool)[0]
                if len(closed) == 0:
                    break
                j = int(rng.choice(closed))

        selected.append(j)
        selected_bool[j] = True
        d2 = ((arr - arr[j]) ** 2).sum(axis=1)
        np.minimum(min_d2, d2, out=min_d2)

    if len(selected) < p:
        closed = np.where(~selected_bool)[0]
        need = p - len(selected)
        if len(closed) > need:
            extra = rng.choice(closed, size=need, replace=False)
        else:
            extra = closed
        selected.extend(int(x) for x in extra)

    return selected


def find_best_swap(C, curr_cost, second_cost, assign_pos, p, candidates, end_time):
    """Evaluate candidate swaps for the current solution and return the best one."""
    n = C.shape[0]

    if len(candidates) == 0:
        return None, 0.0

    best_swap = None
    best_delta = -1.0e-9

    col64 = np.empty(n, dtype=np.int64)
    base = np.empty(n, dtype=np.int64)
    adj = np.empty(n, dtype=np.int64)
    fweights = np.empty(n, dtype=np.float64)

    for b in candidates:
        if time.time() > end_time:
            break

        bi = int(b)

        col64[:] = C[:, bi]

        np.subtract(col64, curr_cost, out=base)
        np.minimum(base, 0, out=base)
        s_val = int(base.sum(dtype=np.int64))

        np.minimum(col64, second_cost, out=adj)
        adj -= curr_cost
        adj -= base

        fweights[:] = adj
        adj_sum = np.bincount(assign_pos, weights=fweights, minlength=p)
        delta = adj_sum + float(s_val)

        a_pos = int(np.argmin(delta))
        d_val = float(delta[a_pos])

        if d_val < best_delta:
            best_delta = d_val
            best_swap = (a_pos, bi)

    return best_swap, best_delta


def local_search(C, selected, end_time, record, rng, full_threshold=500):
    """Teitz-Bart style swap local search, exact for small n and sampled for large n."""
    n = C.shape[0]
    p = len(selected)

    selected = list(selected)
    if p == 0:
        return selected, 0

    obj, curr_cost, second_cost, best_med = evaluate_selected(C, selected)
    failed_rounds = 0

    while time.time() < end_time:
        med_pos = np.full(n, -1, dtype=np.int32)
        sel_arr = np.asarray(selected, dtype=np.int32)
        med_pos[sel_arr] = np.arange(p, dtype=np.int32)

        assign_pos = med_pos[best_med]
        closed_mask = med_pos < 0
        closed_count = int(np.count_nonzero(closed_mask))

        if closed_count == 0:
            break

        exact = n <= full_threshold

        if exact:
            candidates = np.where(closed_mask)[0]
        else:
            sample_size = min(600, max(150, n // 10))
            k = min(closed_count, sample_size)

            if failed_rounds % 2 == 0:
                candidates = top_candidates_by_gain(
                    C, curr_cost, closed_mask, k, end_time, rng
                )
            else:
                closed = np.where(closed_mask)[0]
                candidates = rng.choice(closed, size=k, replace=False)

            if len(candidates) == 0:
                break

        best_swap, best_delta = find_best_swap(
            C, curr_cost, second_cost, assign_pos, p, candidates, end_time
        )

        if best_swap is not None and best_delta < -0.5:
            a_pos, b = best_swap

            if 0 <= a_pos < p:
                new_selected = selected.copy()
                new_selected[a_pos] = b

                new_obj, new_curr, new_second, new_med = evaluate_selected(C, new_selected)

                if new_obj < obj:
                    selected = new_selected
                    obj = new_obj
                    curr_cost = new_curr
                    second_cost = new_second
                    best_med = new_med

                    record(selected, obj, curr_cost, best_med)
                    failed_rounds = 0
                    continue

            if exact:
                break
            failed_rounds += 1
        else:
            if exact:
                break
            failed_rounds += 1

        if failed_rounds >= 4:
            break

    return selected, obj


def perturb(selected, n, rng, k=None):
    """Randomly swap k open facilities with closed facilities."""
    selected = list(selected)
    p = len(selected)

    if p == 0 or p == n:
        return selected

    selected_set = set(selected)
    closed = [j for j in range(n) if j not in selected_set]

    if not closed:
        return selected

    if k is None:
        k = max(1, min(p, len(closed), p // 5))

    k = max(1, min(k, p, len(closed)))

    for _ in range(k):
        idx = int(rng.integers(len(selected)))
        bidx = int(rng.integers(len(closed)))

        removed = selected[idx]
        added = closed[bidx]

        selected[idx] = added
        closed[bidx] = removed

    return selected


def solve_gurobi(C, n, p, end_time, initial_selected):
    """Exact MIP for small instances only."""
    remaining = end_time - time.time()
    if remaining < 5.0 or n > 150:
        return None

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return None

    try:
        model = gp.Model("p_median")
        model.setParam("OutputFlag", 0)
        model.setParam("Seed", 0)
        model.setParam("MIPGap", 1e-4)
        model.setParam("NumericFocus", 0)
        model.setParam("Threads", 1)
        model.setParam("TimeLimit", max(0.5, remaining))

        y = model.addVars(n, vtype=GRB.BINARY, name="y")
        x = model.addVars(n, n, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="x")

        model.setObjective(
            gp.quicksum(float(C[i, j]) * x[i, j] for i in range(n) for j in range(n)),
            GRB.MINIMIZE
        )

        for i in range(n):
            model.addConstr(
                gp.quicksum(x[i, j] for j in range(n)) == 1,
                name=f"assign_{i}"
            )

        model.addConstr(
            gp.quicksum(y[j] for j in range(n)) == p,
            name="cardinality"
        )

        for i in range(n):
            for j in range(n):
                model.addConstr(x[i, j] <= y[j])

        if initial_selected is not None and len(initial_selected) == p:
            init_set = set(int(j) for j in initial_selected)
            for j in range(n):
                y[j].Start = 1.0 if j in init_set else 0.0

            try:
                _, _, _, best_med = evaluate_selected(C, initial_selected)
                for i in range(n):
                    x[i, int(best_med[i])].Start = 1.0
            except Exception:
                pass

        model.optimize()

        if model.SolCount > 0:
            vals = [y[j].X for j in range(n)]
            selected = [j for j in range(n) if vals[j] > 0.5]

            if len(selected) != p:
                order = sorted(range(n), key=lambda j: vals[j], reverse=True)
                selected = order[:p]

            return selected

    except Exception:
        return None

    return None


def sample_raw_max(raw, n):
    """Quick sample scan to decide whether int32 is safe."""
    max_val = 0
    if n <= 0:
        return max_val

    row_step = max(1, n // 20)

    for i in range(0, n, row_step):
        if i >= len(raw):
            break

        row = raw[i]
        if not isinstance(row, list):
            continue

        m = len(row)
        if m == 0:
            continue

        col_step = max(1, m // 50)
        positions = set(range(0, m, col_step))
        positions.add(0)
        positions.add(m - 1)
        positions.add(i % m)

        for j in positions:
            if 0 <= j < m:
                try:
                    v = raw[i][j]
                    if v < 0:
                        v = -v
                    if v > max_val:
                        max_val = v
                except Exception:
                    pass

    return max_val


def cost_from_coordinates(coords, n, dtype):
    """Fallback: compute rounded Euclidean distances if cost matrix is unavailable."""
    arr = np.asarray(coords, dtype=np.float64)
    C = np.empty((n, n), dtype=dtype)

    for i in range(n):
        diff = arr - arr[i]
        d = np.sqrt(np.sum(diff * diff, axis=1))
        np.rint(d, out=d)
        C[i, :] = d.astype(dtype)

    return C


def load_cost_matrix(instance, n):
    """Memory-conscious loader for the dense cost matrix."""
    if n <= 0:
        return np.empty((0, 0), dtype=np.int64)

    dtype = np.int64 if n <= 1500 else np.int32
    raw = instance.get("cost_matrix")

    if raw is not None:
        try:
            if isinstance(raw, list) and n > 0 and isinstance(raw[0], list):
                if dtype == np.int32:
                    if sample_raw_max(raw, n) > np.iinfo(np.int32).max:
                        dtype = np.int64

                C = np.empty((n, n), dtype=dtype)

                for i in range(n):
                    row = raw[i]
                    C[i, :] = row
                    raw[i] = None

                if "cost_matrix" in instance:
                    del instance["cost_matrix"]

                return C

            arr = np.asarray(raw, dtype=dtype)
            if arr.size == n * n:
                arr = arr.reshape((n, n))
                if "cost_matrix" in instance:
                    del instance["cost_matrix"]
                return arr

        except Exception:
            # Fall back to coordinates if the dense matrix cannot be used safely.
            pass

    coords = instance.get("node_coordinates")
    if coords is not None and len(coords) == n:
        try:
            return cost_from_coordinates(coords, n, dtype)
        except Exception:
            pass

    # Last-resort feasible dummy matrix; valid instances should never reach this.
    return np.zeros((n, n), dtype=dtype)


def main():
    parser = argparse.ArgumentParser(description="p-Median solver")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    # Infer / validate n.
    n_raw = instance.get("n")
    if n_raw is None:
        if "cost_matrix" in instance and isinstance(instance["cost_matrix"], list):
            n_raw = len(instance["cost_matrix"])
        elif "node_coordinates" in instance:
            n_raw = len(instance["node_coordinates"])
        else:
            n_raw = 0

    n = int(n_raw)

    # If row-list cost matrix shape disagrees with n, trust the matrix shape.
    cm = instance.get("cost_matrix")
    if isinstance(cm, list) and len(cm) > 0 and isinstance(cm[0], list) and len(cm) != n:
        n = len(cm)

    p = int(instance.get("p", 0))
    coords = instance.get("node_coordinates")

    C = load_cost_matrix(instance, n)
    del instance

    if n == 0:
        solution = {
            "objective_value": 0,
            "opened_facilities": [],
            "assignments": []
        }
        if logger:
            logger.log_solution(0, solution)
        with open(args.solution_path, "w") as f:
            json.dump(solution, f)
        return

    if p > n:
        p = n
    if p <= 0:
        p = 1

    time_limit = max(1, int(args.time_limit))
    end_time = start_time + time_limit - 1.0
    if end_time <= time.time():
        end_time = time.time() + 0.5

    rng = np.random.default_rng(12345)

    best_obj = 10 ** 100
    best_selected = None
    best_solution = None

    def record(selected, obj, curr_cost=None, best_med=None):
        nonlocal best_obj, best_selected, best_solution

        if obj >= best_obj:
            return

        if best_med is None:
            obj2, _, _, best_med = evaluate_selected(C, selected)
            obj = obj2
            if obj >= best_obj:
                return

        solution = build_solution(C, selected, best_med)
        true_obj = solution["objective_value"]

        if true_obj < best_obj:
            best_obj = true_obj
            best_selected = [int(j) for j in selected]
            best_solution = solution

            if logger:
                logger.log_solution(best_obj, best_solution)

    # Trivial all-open case.
    if p >= n:
        assignments = []
        for i in range(n):
            assignments.append({
                "customer": i,
                "facility": i,
                "cost": 0
            })

        solution = {
            "objective_value": 0,
            "opened_facilities": list(range(n)),
            "assignments": assignments
        }

        if logger:
            logger.log_solution(0, solution)

        with open(args.solution_path, "w") as f:
            json.dump(solution, f)
        return

    # -------------------
    # Initial solutions.
    # -------------------
    init_budget = max(0.5, min(10.0, (end_time - time.time()) * 0.25))
    init_end = min(end_time, time.time() + init_budget)

    selected = sanitize_selected(weighted_construct(C, p, rng, init_end), n, p)
    obj, curr, sec, med = evaluate_selected(C, selected)
    record(selected, obj, curr, med)

    if time.time() < init_end and (n <= 700 or (p <= 30 and n <= 2000)):
        selected_g = sanitize_selected(greedy_construct(C, p, rng, init_end), n, p)
        obj_g, curr_g, sec_g, med_g = evaluate_selected(C, selected_g)
        record(selected_g, obj_g, curr_g, med_g)

    if coords is not None and len(coords) == n and time.time() < init_end:
        try:
            selected_k = sanitize_selected(kmeanspp_construct(coords, p, rng, init_end), n, p)
            obj_k, curr_k, sec_k, med_k = evaluate_selected(C, selected_k)
            record(selected_k, obj_k, curr_k, med_k)
        except Exception:
            pass

    if best_selected is None:
        selected = sanitize_selected(random_selected(n, p, rng), n, p)
        obj, curr, sec, med = evaluate_selected(C, selected)
        record(selected, obj, curr, med)

    # For p=1, the best column sum is optimal.
    if p == 1:
        if best_solution is None:
            selected = sanitize_selected(best_selected if best_selected else [0], n, p)
            obj, _, _, med = evaluate_selected(C, selected)
            best_solution = build_solution(C, selected, med)
            best_obj = best_solution["objective_value"]
            best_selected = selected
            if logger:
                logger.log_solution(best_obj, best_solution)

        with open(args.solution_path, "w") as f:
            json.dump(best_solution, f)
        return

    # -------------------
    # Iterated local search.
    # -------------------
    current = list(best_selected)
    iteration = 0

    while time.time() < end_time - 0.5:
        remaining = end_time - time.time()
        call_end = min(end_time, time.time() + max(2.0, remaining * 0.5))

        improved, improved_obj = local_search(
            C, current, call_end, record, rng, full_threshold=500
        )

        if improved_obj < best_obj:
            current = list(improved)
            best_selected = list(improved)
            iteration = 0
        else:
            iteration += 1
            if time.time() > end_time - 1.0:
                break
            if p <= 1 or p >= n:
                break

            k = max(1, min(p, n - p, max(1, p // 8 + (iteration % 3))))
            current = perturb(best_selected, n, rng, k=k)

    # -------------------
    # Small-instance exact refinement.
    # -------------------
    if n <= 150 and p < n and time.time() < end_time - 5.0:
        g_selected = solve_gurobi(C, n, p, end_time, best_selected)
        if g_selected is not None:
            g_selected = sanitize_selected(g_selected, n, p)
            obj_g, curr_g, sec_g, med_g = evaluate_selected(C, g_selected)
            record(g_selected, obj_g, curr_g, med_g)

    # Final fallback / output.
    if best_solution is None:
        selected = sanitize_selected(best_selected if best_selected else random_selected(n, p, rng), n, p)
        obj, _, _, med = evaluate_selected(C, selected)
        best_solution = build_solution(C, selected, med)
        best_obj = best_solution["objective_value"]
        best_selected = selected

        if logger:
            logger.log_solution(best_obj, best_solution)

    with open(args.solution_path, "w") as f:
        json.dump(best_solution, f)


if __name__ == "__main__":
    main()