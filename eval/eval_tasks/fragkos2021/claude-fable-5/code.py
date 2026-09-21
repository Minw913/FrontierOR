import argparse
import json
import time
import random
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=60)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t_start = time.time()
    deadline = t_start + max(1, args.time_limit) - 0.5

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    with open(args.instance_path, "r") as fh:
        data = json.load(fh)

    m = int(data["num_facilities"])
    n = int(data["num_customers"])
    nodes = sorted(data["nodes"], key=lambda x: x["id"])

    d = np.array([float(nodes[j]["demand"]) for j in range(n)], dtype=float)
    f = np.array([float(nodes[i]["fixed_setup_cost"]) for i in range(m)], dtype=float)
    p = np.array([float(nodes[i]["survival_probability"]) for i in range(m)], dtype=float)
    # Clip survival probabilities away from exactly 1 to keep drop-delta formula finite
    p = np.clip(p, 0.0, 1.0 - 1e-9)
    C = np.asarray(data["transportation_cost_matrix"], dtype=float).reshape(n, m)
    pen = np.asarray(data["penalty_cost"], dtype=float).reshape(n)

    # Trivial case: no facilities
    if m == 0:
        obj = float(np.dot(d, pen))
        sol = {"objective_value": obj, "facility_locations": []}
        if logger:
            logger.log_solution(obj, sol)
        with open(args.solution_path, "w") as fh:
            json.dump(sol, fh)
        return

    # Precompute per-customer facility ordering by unit transportation cost
    order = np.argsort(C, axis=1)                      # n x m
    c_sorted = np.take_along_axis(C, order, axis=1)    # n x m
    p_sorted = p[order]                                # n x m
    dcol = d[:, None]
    ratio = p / (1.0 - p)

    def evaluate_state(y):
        """Given boolean open-vector y, return (total_cost, G) where
        G[k] = sum_j d_j * (C[j,k]*prefix_o[j,k] - suffix_o[j,k]).
        Add delta (k closed): f[k] + p[k]*G[k]
        Drop delta (k open):  -f[k] - p[k]/(1-p[k])*G[k]
        Assumes independent facility disruptions with survival probs p.
        """
        mask = y[order]
        s = p_sorted * mask                      # survival prob at sorted slots
        om = 1.0 - s
        cp = np.cumprod(om, axis=1)
        prefix = np.empty_like(cp)
        prefix[:, 0] = 1.0
        if m > 1:
            prefix[:, 1:] = cp[:, :-1]
        end = cp[:, -1]
        contrib = c_sorted * s * prefix
        # suffix[j, r] = sum_{l>=r} contrib[j, l] + pen_j * end_j
        rev = np.cumsum(contrib[:, ::-1], axis=1)[:, ::-1]
        suffix = rev + (pen * end)[:, None]
        serve = suffix[:, 0]
        total = float(np.dot(f, y)) + float(np.dot(d, serve))
        # Scatter prefix/suffix back to original facility indices
        prefix_o = np.empty_like(prefix)
        suffix_o = np.empty_like(suffix)
        np.put_along_axis(prefix_o, order, prefix, axis=1)
        np.put_along_axis(suffix_o, order, suffix, axis=1)
        G = ((C * prefix_o - suffix_o) * dcol).sum(axis=0)
        return total, G

    def flip_deltas(y, G):
        da = f + p * G            # delta if we open k (valid where y[k] is False)
        dd = -f - ratio * G       # delta if we close k (valid where y[k] is True)
        return np.where(y, dd, da)

    best_total = None
    best_y = None

    def record_best(total, y):
        nonlocal best_total, best_y
        if best_total is None or total < best_total - 1e-9:
            best_total = total
            best_y = y.copy()
            if logger:
                sol = {
                    "objective_value": float(best_total),
                    "facility_locations": [int(v) for v in best_y],
                }
                try:
                    logger.log_solution(float(best_total), sol)
                except Exception:
                    pass
            return True
        return False

    def local_search(y, total, G):
        """Best-improvement local search: single flips (add/drop) + swaps."""
        while True:
            if time.time() > deadline:
                return total, G
            tol = 1e-9 * (1.0 + abs(total))
            dlt = flip_deltas(y, G)
            k = int(np.argmin(dlt))
            if dlt[k] < -tol:
                y[k] = not y[k]
                total, G = evaluate_state(y)
                continue
            # Swap phase: try closing each open facility and opening best closed one
            open_idx = np.where(y)[0]
            best_delta = -tol
            best_pair = None
            for ko in open_idx:
                if time.time() > deadline:
                    break
                y[ko] = False
                t2, G2 = evaluate_state(y)
                da2 = f + p * G2
                da2 = np.where(y, np.inf, da2)
                da2[ko] = np.inf
                ki = int(np.argmin(da2))
                y[ko] = True
                if not np.isfinite(da2[ki]):
                    continue
                delta = (t2 + da2[ki]) - total
                if delta < best_delta:
                    best_delta = delta
                    best_pair = (ko, ki)
            if best_pair is None:
                return total, G
            ko, ki = best_pair
            y[ko] = False
            y[ki] = True
            total, G = evaluate_state(y)

    try:
        # Initial solution: nothing open (all demand penalized), then greedy/LS
        y = np.zeros(m, dtype=bool)
        total, G = evaluate_state(y)
        record_best(total, y)

        total, G = local_search(y, total, G)
        record_best(total, y)

        # Iterated local search: perturb best solution and re-optimize
        rng = random.Random(0)
        while time.time() < deadline:
            y = best_y.copy()
            k = rng.randint(1, min(3, m))
            for _ in range(k):
                i = rng.randrange(m)
                y[i] = not y[i]
            total, G = evaluate_state(y)
            total, G = local_search(y, total, G)
            record_best(total, y)
    except Exception:
        # Fall through to write the best solution found so far
        pass

    if best_y is None:
        best_y = np.zeros(m, dtype=bool)
        best_total, _ = evaluate_state(best_y)

    sol = {
        "objective_value": float(best_total),
        "facility_locations": [int(v) for v in best_y],
    }
    with open(args.solution_path, "w") as fh:
        json.dump(sol, fh)


if __name__ == "__main__":
    main()