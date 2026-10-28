import argparse
import json
import math
import time
import gc

try:
    import numpy as np
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False

from solution_logger import SolutionLogger


# ---------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------

def write_zero_solution(path, n, m):
    with open(path, "w") as f:
        f.write('{"objective_value": 0.0, "flows": {')
        batch = []
        first = True
        for i in range(n):
            for j in range(m):
                batch.append(f'"x_{i}_{j}":0.0')
                if len(batch) >= 10000:
                    if first:
                        f.write(",".join(batch))
                        first = False
                    else:
                        f.write("," + ",".join(batch))
                    batch = []
        if batch:
            if first:
                f.write(",".join(batch))
            else:
                f.write("," + ",".join(batch))
        f.write("}}")


def write_solution_matrix(path, obj, n, m, X):
    with open(path, "w") as f:
        f.write('{"objective_value": ' + json.dumps(float(obj)) + ', "flows": {')
        batch = []
        first = True
        for i in range(n):
            row = X[i]
            for j in range(m):
                v = float(row[j])
                if not math.isfinite(v):
                    v = 0.0
                elif v < 0.0:
                    v = 0.0
                batch.append(f'"x_{i}_{j}":{v:.12g}')
                if len(batch) >= 10000:
                    if first:
                        f.write(",".join(batch))
                        first = False
                    else:
                        f.write("," + ",".join(batch))
                    batch = []
        if batch:
            if first:
                f.write(",".join(batch))
            else:
                f.write("," + ",".join(batch))
        f.write("}}")


# ---------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------

def make_log_func(logger, n, m, max_dict_arcs=50000):
    if logger is None:
        return lambda obj, X=None: None

    def log_func(obj, X=None):
        obj = float(obj)
        if X is not None and n * m <= max_dict_arcs:
            flows = {}
            for i in range(n):
                row = X[i]
                for j in range(m):
                    v = float(row[j])
                    if not math.isfinite(v) or v < 0.0:
                        v = 0.0
                    flows[f"x_{i}_{j}"] = v
            logger.log_solution(obj, {"objective_value": obj, "flows": flows})
        else:
            logger.log(obj)

    return log_func


# ---------------------------------------------------------------------
# NumPy-based feasibility repair
# ---------------------------------------------------------------------

if HAS_NUMPY:

    def make_feasible_dense(X, C, Q, supplies, demands, U):
        """
        Repairs a dense flow matrix so that:
          - column sums match demands as closely as possible,
          - row sums do not exceed supplies,
          - 0 <= X <= U.
        This is intended as a final projection/repair for near-optimal
        dual-derived flows, and also as a greedy constructor from zeros.
        """
        n, m = X.shape
        if n == 0 or m == 0:
            return

        tol = 1e-6
        np.clip(X, 0.0, U, out=X)

        rowsum = X.sum(axis=1)

        # Scale down overfull suppliers.
        over = rowsum > supplies + tol
        if np.any(over):
            for i in np.where(over)[0]:
                if rowsum[i] > 0.0:
                    factor = max(0.0, float(supplies[i]) / float(rowsum[i]))
                    if factor < 1.0:
                        X[i, :] *= factor
                        rowsum[i] *= factor

        rowsum = X.sum(axis=1)
        spare = np.maximum(supplies - rowsum, 0.0)
        colsum = X.sum(axis=0)

        # Remove column surpluses by deleting expensive flow first.
        surp_idx = np.where(colsum > demands + tol)[0]
        if surp_idx.size:
            surp_idx = surp_idx[np.argsort(-(colsum[surp_idx] - demands[surp_idx]))]
            for j in surp_idx:
                excess = float(colsum[j] - demands[j])
                if excess <= tol:
                    continue

                if Q is None:
                    marg = C[:, j].copy()

                while excess > tol:
                    valid = X[:, j] > tol
                    if not np.any(valid):
                        break

                    if Q is not None:
                        marg = C[:, j] + 2.0 * Q[:, j] * X[:, j]

                    masked = np.where(valid, marg, -np.inf)
                    i = int(np.argmax(masked))
                    delta = min(excess, float(X[i, j]))

                    if delta <= tol:
                        X[i, j] = 0.0
                        continue

                    X[i, j] -= delta
                    excess -= delta
                    colsum[j] -= delta
                    rowsum[i] -= delta
                    spare[i] += delta

        # Fill column deficits from suppliers with remaining capacity.
        def_idx = np.where(colsum < demands - tol)[0]
        if def_idx.size:
            def_idx = def_idx[np.argsort(-(demands[def_idx] - colsum[def_idx]))]
            for j in def_idx:
                need = float(demands[j] - colsum[j])
                if need <= tol:
                    continue

                if Q is None:
                    marg = C[:, j].copy()

                while need > tol:
                    valid = (spare > tol) & (X[:, j] < U - tol)
                    if not np.any(valid):
                        valid = (spare > 0.0) & (X[:, j] < U)
                        if not np.any(valid):
                            break

                    if Q is not None:
                        marg = C[:, j] + 2.0 * Q[:, j] * X[:, j]

                    masked = np.where(valid, marg, np.inf)
                    i = int(np.argmin(masked))
                    delta = min(need, float(spare[i]), float(U - X[i, j]))

                    if delta <= 1e-9:
                        if spare[i] <= 1e-8:
                            spare[i] = 0.0
                        else:
                            X[i, j] = U
                        continue

                    X[i, j] += delta
                    need -= delta
                    colsum[j] += delta
                    rowsum[i] += delta
                    spare[i] -= delta

        # Final tiny correction for residual column errors.
        colsum = X.sum(axis=0)
        for j in range(m):
            diff = float(demands[j] - colsum[j])
            if diff > tol:
                avail = np.minimum(np.maximum(spare, 0.0), np.maximum(U - X[:, j], 0.0))
                tot = float(avail.sum())
                if tot > 0.0:
                    scale = min(1.0, diff / tot)
                    add = avail * scale
                    X[:, j] += add
                    spare -= add
                    rowsum += add
                    colsum[j] += add.sum()
            elif diff < -tol:
                pos = np.maximum(X[:, j], 0.0)
                tot = float(pos.sum())
                if tot > 0.0:
                    scale = min(1.0, -diff / tot)
                    red = pos * scale
                    X[:, j] -= red
                    spare += red
                    rowsum -= red
                    colsum[j] -= red.sum()

        np.clip(X, 0.0, U, out=X)


    def objective_chunked(X, C, Q, slices):
        obj = 0.0
        for s, e in slices:
            Xc = X[s:e]
            obj += float(np.sum(C[s:e] * Xc))
            if Q is not None:
                obj += float(np.sum(Q[s:e] * Xc * Xc))
        return obj


    # -----------------------------------------------------------------
    # Custom convex transportation solver based on dual coordinate ascent
    # -----------------------------------------------------------------

    def solve_custom(C, Q, n, m, supplies, demands, U, cost_type, deadline, log_func):
        supplies_arr = np.asarray(supplies, dtype=np.float64)
        demands_arr = np.asarray(demands, dtype=np.float64)
        total_demand = float(demands_arr.sum())

        X = np.zeros((n, m), dtype=np.float64)

        if n == 0 or m == 0 or total_demand <= 1e-12:
            return X, 0.0

        C = np.ascontiguousarray(C, dtype=np.float64)
        if Q is not None:
            Q = np.ascontiguousarray(Q, dtype=np.float64)

        # A small quadratic regularization is used for linear/zero-q arcs
        # to obtain a smooth strictly convex subproblem. The reported
        # objective always uses the original cost coefficients.
        if C.size:
            cmin = float(C.min())
            cmax = float(C.max())
            max_c = max(1.0, abs(cmin), abs(cmax))
        else:
            max_c = 1.0

        max_d = float(np.max(demands_arr)) if demands_arr.size else 0.0
        flow_scale = max(
            1.0,
            min(
                float(U),
                max_d if max_d > 0.0 else 1.0,
                total_demand / max(1, n),
            ),
        )

        eps = 1e-5 * max_c / flow_scale
        eps = min(max(eps, 1e-10), 1e-2)
        denom_scalar = 2.0 * eps if Q is None else None

        # Row chunks used to keep temporary arrays bounded.
        max_elems = 2_000_000
        step = max(1, max_elems // max(1, m))
        slices = [(s, min(n, s + step)) for s in range(0, n, step)]

        mu = np.zeros(n, dtype=np.float64)
        best_obj = float("inf")
        best_X = None
        did_nu = False

        def nu_bounds(mu_vals):
            low = np.full(m, np.inf, dtype=np.float64)
            high = np.full(m, -np.inf, dtype=np.float64)

            for s, e in slices:
                base = mu_vals[s:e, None] + C[s:e]
                low = np.minimum(low, base.min(axis=0))

                if Q is None:
                    high_chunk = base.max(axis=0) + denom_scalar * U
                else:
                    denom = np.maximum(Q[s:e], eps)
                    denom *= 2.0
                    high_chunk = (base + denom * U).max(axis=0)

                high = np.maximum(high, high_chunk)

            high = np.maximum(high, low)
            return low, high

        def col_sums_for_nu(nu_vals, mu_vals):
            acc = np.zeros(m, dtype=np.float64)
            for s, e in slices:
                temp = nu_vals[None, :] - mu_vals[s:e, None] - C[s:e]
                if Q is None:
                    temp /= denom_scalar
                else:
                    denom = np.maximum(Q[s:e], eps)
                    denom *= 2.0
                    np.divide(temp, denom, out=temp)
                np.clip(temp, 0.0, U, out=temp)
                acc += temp.sum(axis=0)
            return acc

        def update_nu(mu_vals, iters):
            low, high = nu_bounds(mu_vals)

            for _ in range(iters):
                mid = (low + high) * 0.5
                colsum = col_sums_for_nu(mid, mu_vals)
                mask = colsum < demands_arr
                low[mask] = mid[mask]
                high[~mask] = mid[~mask]

            nu_vals = (low + high) * 0.5
            rowsum = np.zeros(n, dtype=np.float64)
            obj = 0.0

            for s, e in slices:
                temp = nu_vals[None, :] - mu_vals[s:e, None] - C[s:e]
                if Q is None:
                    temp /= denom_scalar
                else:
                    denom = np.maximum(Q[s:e], eps)
                    denom *= 2.0
                    np.divide(temp, denom, out=temp)

                np.clip(temp, 0.0, U, out=temp)
                X[s:e] = temp
                rowsum[s:e] = temp.sum(axis=1)

                obj += float(np.sum(C[s:e] * temp))
                if Q is not None:
                    obj += float(np.sum(Q[s:e] * temp * temp))

            return nu_vals, rowsum, obj

        def row_sums_for_mu(mu_vals, nu_vals):
            rs = np.empty(n, dtype=np.float64)
            for s, e in slices:
                temp = nu_vals[None, :] - mu_vals[s:e, None] - C[s:e]
                if Q is None:
                    temp /= denom_scalar
                else:
                    denom = np.maximum(Q[s:e], eps)
                    denom *= 2.0
                    np.divide(temp, denom, out=temp)
                np.clip(temp, 0.0, U, out=temp)
                rs[s:e] = temp.sum(axis=1)
            return rs

        def high_mu(nu_vals):
            high = np.zeros(n, dtype=np.float64)
            for s, e in slices:
                vals = nu_vals[None, :] - C[s:e]
                high[s:e] = np.maximum(0.0, vals.max(axis=1))
            return high

        def update_mu(nu_vals, iters):
            zero_mu = np.zeros(n, dtype=np.float64)
            rs0 = row_sums_for_mu(zero_mu, nu_vals)
            active = rs0 > supplies_arr + 1e-5

            mu_new = np.zeros(n, dtype=np.float64)
            if np.any(active):
                low = np.zeros(n, dtype=np.float64)
                high = high_mu(nu_vals)

                bad = active & (high <= 1e-12)
                high[bad] = 1.0

                for _ in range(iters):
                    mid = (low + high) * 0.5
                    rs = row_sums_for_mu(mid, nu_vals)
                    mask = rs > supplies_arr + 1e-5
                    low[mask] = mid[mask]
                    high[~mask] = mid[~mask]

                mu_new[active] = high[active]

            return mu_new

        max_sweeps = 100
        nu = None

        for sweep in range(max_sweeps):
            if time.time() >= deadline:
                break

            nu_iters = 32 if sweep == 0 else 22
            mu_iters = 32 if sweep == 0 else 22

            nu, rowsum, obj = update_nu(mu, nu_iters)
            did_nu = True

            if np.all(rowsum <= supplies_arr + 1e-3):
                if obj < best_obj - 1e-6:
                    best_obj = obj
                    log_func(obj, X)
                    if n * m <= 3_000_000:
                        try:
                            best_X = X.copy()
                        except Exception:
                            best_X = None

            if time.time() >= deadline:
                break

            mu = update_mu(nu, mu_iters)

        # If possible, do one final demand-satisfying update with the last mu.
        if did_nu and time.time() < deadline:
            nu, rowsum, obj = update_nu(mu, 26)
            if np.all(rowsum <= supplies_arr + 1e-3):
                if obj < best_obj - 1e-6:
                    best_obj = obj
                    log_func(obj, X)
                    if n * m <= 3_000_000:
                        try:
                            best_X = X.copy()
                        except Exception:
                            best_X = None

        if best_X is not None:
            X = best_X

        make_feasible_dense(X, C, Q, supplies_arr, demands_arr, float(U))
        obj = objective_chunked(X, C, Q, slices)

        if obj < best_obj - 1e-6 or not did_nu:
            log_func(obj, X)

        return X, obj


    # -----------------------------------------------------------------
    # Streaming proportional fallback for extremely large dense matrices
    # -----------------------------------------------------------------

    def solve_fallback_stream(
        C,
        Q,
        n,
        m,
        supplies,
        demands,
        U,
        cost_type,
        solution_path,
        logger,
    ):
        supplies_arr = np.asarray(supplies, dtype=np.float64)
        demands_arr = np.asarray(demands, dtype=np.float64)
        total_supply = float(supplies_arr.sum())
        total_demand = float(demands_arr.sum())

        if n == 0 or m == 0 or total_demand <= 1e-12 or total_supply <= 1e-12:
            if logger:
                logger.log(0.0)
            write_zero_solution(solution_path, n, m)
            return 0.0

        U = float(U)
        rowsum = np.zeros(n, dtype=np.float64)
        deficits = demands_arr.copy()
        obj = 0.0

        # Base proportional allocation, capped by arc capacity.
        for i in range(n):
            if supplies_arr[i] <= 0.0:
                continue
            ratio = supplies_arr[i] / total_supply
            base = np.minimum(U, demands_arr * ratio)
            rowsum[i] = float(base.sum())
            deficits -= base

            obj += float(np.sum(C[i] * base))
            if Q is not None:
                obj += float(np.sum(Q[i] * base * base))

        remaining = np.maximum(supplies_arr - rowsum, 0.0)
        extra = {}

        deficit_idx = np.where(deficits > 1e-8)[0]
        if deficit_idx.size:
            deficit_idx = deficit_idx[np.argsort(-deficits[deficit_idx])]

            for j in deficit_idx:
                need = float(deficits[j])
                if need <= 1e-8:
                    continue

                # For moderate n, use cost order; otherwise use natural order.
                if n <= 2000:
                    order = np.argsort(C[:, j])
                else:
                    order = range(n)

                for i in order:
                    if need <= 1e-8:
                        break
                    if remaining[i] <= 1e-8:
                        continue

                    base_ij = min(U, float(demands_arr[j] * supplies_arr[i] / total_supply))
                    cur = base_ij + extra.get((i, j), 0.0)
                    cap = U - cur
                    if cap <= 1e-8:
                        continue

                    delta = min(need, float(remaining[i]), float(cap))
                    if delta <= 1e-9:
                        continue

                    extra[(i, j)] = extra.get((i, j), 0.0) + delta
                    need -= delta
                    remaining[i] -= delta

                    if Q is not None:
                        q = float(Q[i, j])
                        obj += float(C[i, j]) * delta + q * ((cur + delta) * (cur + delta) - cur * cur)
                    else:
                        obj += float(C[i, j]) * delta

                # If still short, try any supplier.
                if need > 1e-6:
                    for i in range(n):
                        if need <= 1e-8:
                            break
                        if remaining[i] <= 1e-8:
                            continue

                        base_ij = min(U, float(demands_arr[j] * supplies_arr[i] / total_supply))
                        cur = base_ij + extra.get((i, j), 0.0)
                        cap = U - cur
                        if cap <= 1e-8:
                            continue

                        delta = min(need, float(remaining[i]), float(cap))
                        if delta <= 1e-9:
                            continue

                        extra[(i, j)] = extra.get((i, j), 0.0) + delta
                        need -= delta
                        remaining[i] -= delta

                        if Q is not None:
                            q = float(Q[i, j])
                            obj += float(C[i, j]) * delta + q * ((cur + delta) * (cur + delta) - cur * cur)
                        else:
                            obj += float(C[i, j]) * delta

        if logger:
            logger.log(float(obj))

        # Stream output without materializing all flows.
        with open(solution_path, "w") as f:
            f.write('{"objective_value": ' + json.dumps(float(obj)) + ', "flows": {')
            batch = []
            first = True

            for i in range(n):
                si = float(supplies_arr[i])
                for j in range(m):
                    if total_supply > 0.0 and si > 0.0:
                        base = min(U, float(demands_arr[j] * si / total_supply))
                    else:
                        base = 0.0
                    val = base + extra.get((i, j), 0.0)
                    if not math.isfinite(val) or val < 0.0:
                        val = 0.0

                    batch.append(f'"x_{i}_{j}":{val:.12g}')
                    if len(batch) >= 10000:
                        if first:
                            f.write(",".join(batch))
                            first = False
                        else:
                            f.write("," + ",".join(batch))
                        batch = []

            if batch:
                if first:
                    f.write(",".join(batch))
                else:
                    f.write("," + ",".join(batch))

            f.write("}}")

        return obj


# ---------------------------------------------------------------------
# Gurobi path for small/medium instances
# ---------------------------------------------------------------------

if HAS_NUMPY:

    def try_gurobi(
        C,
        Q,
        n,
        m,
        supplies,
        demands,
        U,
        cost_type,
        time_left,
        log_func,
    ):
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except Exception:
            return None, None, False

        try:
            model = gp.Model("transportation")
            model.setParam("Seed", 0)
            model.setParam("MIPGap", 1e-4)
            model.setParam("NumericFocus", 0)
            model.setParam("Threads", 1)
            model.setParam("TimeLimit", max(1, int(time_left)))
            model.setParam("OutputFlag", 0)

            x = model.addMVar((n, m), lb=0.0, ub=float(U), vtype=GRB.CONTINUOUS)

            demands_arr = np.asarray(demands, dtype=np.float64)
            supplies_arr = np.asarray(supplies, dtype=np.float64)

            model.addConstr(x.sum(axis=0) == demands_arr)
            model.addConstr(x.sum(axis=1) <= supplies_arr)

            if cost_type == "quadratic" and Q is not None:
                # Prefer matrix expression if available; fall back to explicit loop.
                try:
                    model.setObjective((C * x).sum() + (Q * x * x).sum(), GRB.MINIMIZE)
                except Exception:
                    obj = gp.QuadExpr()
                    for i in range(n):
                        for j in range(m):
                            obj += float(C[i, j]) * x[i, j]
                            obj += float(Q[i, j]) * x[i, j] * x[i, j]
                    model.setObjective(obj, GRB.MINIMIZE)
            else:
                # Linear objective via matrix API.
                try:
                    model.setObjective(x.reshape(-1) @ C.reshape(-1), GRB.MINIMIZE)
                except Exception:
                    model.setObjective((C * x).sum(), GRB.MINIMIZE)

            model.optimize()

            if model.SolCount > 0:
                X = np.asarray(x.X, dtype=np.float64).reshape(n, m)
                np.clip(X, 0.0, float(U), out=X)

                obj = float(np.sum(C * X))
                if cost_type == "quadratic" and Q is not None:
                    obj += float(np.sum(Q * X * X))

                log_func(obj, X)
                return X, obj, True

            return None, None, False

        except Exception:
            return None, None, False


# ---------------------------------------------------------------------
# Pure-Python greedy fallback (used only if NumPy is unavailable)
# ---------------------------------------------------------------------

def solve_simple_python(instance, solution_path, logger):
    n = int(instance["n"])
    m = int(instance["m"])
    supplies = list(instance["supplies"])
    demands = list(instance["demands"])
    U = float(instance["arc_capacity"])
    linear_costs = instance["linear_costs"]
    cost_type = instance.get("cost_type", "linear_integer")
    quadratic_costs = instance.get("quadratic_costs", None)

    if n == 0 or m == 0 or sum(demands) <= 0:
        if logger:
            logger.log(0.0)
        write_zero_solution(solution_path, n, m)
        return

    rem = [float(s) for s in supplies]
    X = [[0.0 for _ in range(m)] for _ in range(n)]

    # Serve larger demands first; for each customer, use cheaper suppliers first.
    for j in sorted(range(m), key=lambda jj: -float(demands[jj])):
        need = float(demands[j])
        if need <= 0.0:
            continue

        order = sorted(range(n), key=lambda i: float(linear_costs[i][j]))
        for i in order:
            if need <= 1e-9:
                break
            amt = min(need, rem[i], U)
            if amt > 0.0:
                X[i][j] = amt
                rem[i] -= amt
                need -= amt

        # If still needed, take any remaining capacity.
        if need > 1e-9:
            for i in range(n):
                if need <= 1e-9:
                    break
                avail_cap = U - X[i][j]
                amt = min(need, rem[i], avail_cap)
                if amt > 0.0:
                    X[i][j] += amt
                    rem[i] -= amt
                    need -= amt

    obj = 0.0
    for i in range(n):
        for j in range(m):
            f = float(X[i][j])
            obj += float(linear_costs[i][j]) * f
            if cost_type == "quadratic" and quadratic_costs is not None:
                obj += float(quadratic_costs[i][j]) * f * f

    if logger:
        logger.log(float(obj))

    write_solution_matrix(solution_path, obj, n, m, X)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    time_limit = max(1, int(args.time_limit))

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    n = int(instance["n"])
    m = int(instance["m"])
    supplies = instance["supplies"]
    demands = instance["demands"]
    U = float(instance.get("arc_capacity", 0))
    cost_type = str(instance.get("cost_type", "linear_integer"))

    total_demand = int(instance.get("total_demand", sum(demands)))

    # Trivial zero-demand case.
    if n == 0 or m == 0 or total_demand <= 0 or U <= 0.0:
        if logger:
            logger.log(0.0)
        write_zero_solution(args.solution_path, n, m)
        return

    log_func = make_log_func(logger, n, m, max_dict_arcs=50000)

    # If NumPy is unavailable, use a simple pure-Python greedy.
    if not HAS_NUMPY:
        solve_simple_python(instance, args.solution_path, logger)
        return

    # Convert cost matrices to compact arrays and release the JSON payload.
    try:
        C = np.asarray(instance["linear_costs"], dtype=np.float64)
        if C.ndim == 1:
            C = C.reshape(n, m)
        elif C.shape != (n, m):
            C = C.reshape(n, m)
    except Exception:
        C = np.zeros((n, m), dtype=np.float64)

    Q = None
    if cost_type == "quadratic" and instance.get("quadratic_costs", None) is not None:
        try:
            Q = np.asarray(instance["quadratic_costs"], dtype=np.float64)
            if Q.ndim == 1:
                Q = Q.reshape(n, m)
            elif Q.shape != (n, m):
                Q = Q.reshape(n, m)
        except Exception:
            Q = None

    del instance
    gc.collect()

    num_arcs = n * m

    # Reserve time for final repair/output.
    reserve = min(300, max(5, int(0.10 * time_limit)))
    solve_deadline = start_time + max(1, time_limit - reserve)

    # Use Gurobi for small/medium instances where it is likely to be memory-safe.
    use_gurobi = False
    if cost_type == "quadratic" and num_arcs <= 80_000:
        use_gurobi = True
    elif cost_type != "quadratic" and num_arcs <= 500_000:
        use_gurobi = True

    if use_gurobi:
        time_left = max(1, int(solve_deadline - time.time()))
        X, obj, ok = try_gurobi(
            C,
            Q,
            n,
            m,
            supplies,
            demands,
            U,
            cost_type,
            time_left,
            log_func,
        )
        if ok:
            write_solution_matrix(args.solution_path, obj, n, m, X)
            return

    # Custom solver for larger instances.
    try:
        X, obj = solve_custom(
            C,
            Q,
            n,
            m,
            supplies,
            demands,
            U,
            cost_type,
            solve_deadline,
            log_func,
        )
        write_solution_matrix(args.solution_path, obj, n, m, X)
    except MemoryError:
        # Very large instance: stream a feasible proportional solution.
        solve_fallback_stream(
            C,
            Q,
            n,
            m,
            supplies,
            demands,
            U,
            cost_type,
            args.solution_path,
            logger,
        )
    except Exception:
        # Last-resort fallback: still attempt a streaming proportional solution.
        try:
            solve_fallback_stream(
                C,
                Q,
                n,
                m,
                supplies,
                demands,
                U,
                cost_type,
                args.solution_path,
                logger,
            )
        except Exception:
            # If everything else fails, output zeros to preserve schema.
            if logger:
                logger.log(0.0)
            write_zero_solution(args.solution_path, n, m)


if __name__ == "__main__":
    main()