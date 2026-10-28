import argparse
import json
import time
from typing import Optional

import numpy as np


class SafeLogger:
    """
    Robust no-op logger wrapper.

    Note:
    Some execution environments can fail hard when importing/initializing
    external logging utilities (e.g., DBus-related failures). To guarantee
    solver stability, this wrapper intentionally avoids external dependencies.
    """
    def __init__(self, log_path: Optional[str]):
        self.log_path = log_path  # kept for interface compatibility

    def log_solution(self, objective_value, solution_dict):
        # Intentionally no-op for robustness
        return

    def log(self, objective_value):
        # Intentionally no-op for robustness
        return


def build_solution_dict(objective_value, y, x):
    return {
        "objective_value": float(objective_value),
        "variant": "quadratic",
        "y": [int(v) for v in y],
        "x": x,
    }


def make_x_from_open(inv_costs: np.ndarray, open_mask: np.ndarray, denom: np.ndarray):
    n, m = inv_costs.shape
    x = np.zeros((n, m), dtype=float)
    open_idx = np.where(open_mask)[0]
    if open_idx.size > 0:
        x[open_idx, :] = inv_costs[open_idx, :] / denom[None, :]
    return x.tolist()


def evaluate_objective(open_mask: np.ndarray, opening_costs: np.ndarray, denom: np.ndarray):
    return float(opening_costs[open_mask].sum() + np.sum(1.0 / denom))


def best_add_move(
    inv_costs: np.ndarray,
    opening_costs: np.ndarray,
    open_mask: np.ndarray,
    denom: np.ndarray,
    start_time: float,
    time_limit: int,
    rng: np.random.Generator,
):
    n, _ = inv_costs.shape
    closed = np.where(~open_mask)[0]
    if closed.size == 0:
        return 0.0, -1

    if closed.size > 300:
        k1 = min(180, closed.size)
        k2 = min(120, closed.size)
        low_open = closed[np.argpartition(opening_costs[closed], k1 - 1)[:k1]]
        rand_idx = rng.choice(closed, size=k2, replace=False)
        cand = np.unique(np.concatenate([low_open, rand_idx]))
    else:
        cand = closed

    base = 1.0 / denom
    best_imp = 0.0
    best_k = -1

    for t, k in enumerate(cand):
        if (t & 31) == 0 and (time.time() - start_time) >= time_limit:
            break
        d2 = denom + inv_costs[k]
        imp = float(np.sum(base - 1.0 / d2) - opening_costs[k])  # objective decrease
        if imp > best_imp + 1e-12:
            best_imp = imp
            best_k = int(k)

    return best_imp, best_k


def best_drop_move(
    inv_costs: np.ndarray,
    opening_costs: np.ndarray,
    open_mask: np.ndarray,
    denom: np.ndarray,
    start_time: float,
    time_limit: int,
):
    open_idx = np.where(open_mask)[0]
    if open_idx.size <= 1:
        return 0.0, -1

    base = 1.0 / denom
    best_imp = 0.0
    best_i = -1

    for t, i in enumerate(open_idx):
        if (t & 31) == 0 and (time.time() - start_time) >= time_limit:
            break
        d2 = denom - inv_costs[i]
        if np.any(d2 <= 0.0):
            continue
        imp = float(opening_costs[i] - np.sum(1.0 / d2 - base))  # objective decrease
        if imp > best_imp + 1e-12:
            best_imp = imp
            best_i = int(i)

    return best_imp, best_i


def local_search(
    inv_costs: np.ndarray,
    opening_costs: np.ndarray,
    open_mask: np.ndarray,
    denom: np.ndarray,
    start_time: float,
    time_limit: int,
    rng: np.random.Generator,
):
    while (time.time() - start_time) < time_limit:
        add_imp, add_k = best_add_move(
            inv_costs, opening_costs, open_mask, denom, start_time, time_limit, rng
        )
        drop_imp, drop_i = best_drop_move(
            inv_costs, opening_costs, open_mask, denom, start_time, time_limit
        )

        if add_imp <= 1e-12 and drop_imp <= 1e-12:
            break

        if add_imp >= drop_imp:
            open_mask[add_k] = True
            denom += inv_costs[add_k]
        else:
            open_mask[drop_i] = False
            denom -= inv_costs[drop_i]

    return open_mask, denom


def greedy_construct_from_seed(
    seed_i: int,
    inv_costs: np.ndarray,
    opening_costs: np.ndarray,
    start_time: float,
    time_limit: int,
    rng: np.random.Generator,
):
    n, _ = inv_costs.shape
    open_mask = np.zeros(n, dtype=bool)
    open_mask[seed_i] = True
    denom = inv_costs[seed_i].copy()

    while (time.time() - start_time) < time_limit:
        imp, k = best_add_move(
            inv_costs, opening_costs, open_mask, denom, start_time, time_limit, rng
        )
        if imp <= 1e-12 or k < 0:
            break
        open_mask[k] = True
        denom += inv_costs[k]

    open_mask, denom = local_search(
        inv_costs, opening_costs, open_mask, denom, start_time, time_limit, rng
    )
    return open_mask, denom


def solve_instance(instance: dict, time_limit: int, logger: SafeLogger):
    n = int(instance["num_facilities"])
    m = int(instance["num_customers"])

    opening_costs = np.asarray(instance["opening_costs"], dtype=float)
    alloc_costs = np.asarray(instance["allocation_costs"], dtype=float)

    if n == 0:
        obj = 0.0 if m == 0 else 1e100
        return build_solution_dict(obj, [], [])

    if m == 0:
        sol = build_solution_dict(0.0, [0] * n, [[] for _ in range(n)])
        logger.log_solution(0.0, sol)
        return sol

    alloc_costs = np.maximum(alloc_costs, 1e-15)  # numerical safety
    inv_costs = 1.0 / alloc_costs

    start_time = time.time()
    rng = np.random.default_rng(0)

    single_obj = opening_costs + alloc_costs.sum(axis=1)
    order = np.argsort(single_obj)

    if n <= 50:
        max_starts = min(n, 12)
    elif n <= 200:
        max_starts = min(n, 8)
    else:
        max_starts = min(n, 5)

    best_obj = float("inf")
    best_mask = None
    best_denom = None

    def register(mask, denom):
        nonlocal best_obj, best_mask, best_denom
        obj = evaluate_objective(mask, opening_costs, denom)
        if obj + 1e-12 < best_obj:
            best_obj = obj
            best_mask = mask.copy()
            best_denom = denom.copy()
            sol = build_solution_dict(
                best_obj,
                best_mask.astype(int).tolist(),
                make_x_from_open(inv_costs, best_mask, best_denom),
            )
            logger.log_solution(best_obj, sol)

    for s in range(max_starts):
        if (time.time() - start_time) >= time_limit:
            break
        seed = int(order[s])
        mask, denom = greedy_construct_from_seed(
            seed, inv_costs, opening_costs, start_time, time_limit, rng
        )
        register(mask, denom)

    if best_mask is None:
        seed = int(order[0])
        best_mask = np.zeros(n, dtype=bool)
        best_mask[seed] = True
        best_denom = inv_costs[seed].copy()
        best_obj = evaluate_objective(best_mask, opening_costs, best_denom)

    return build_solution_dict(
        best_obj,
        best_mask.astype(int).tolist(),
        make_x_from_open(inv_costs, best_mask, best_denom),
    )


def main():
    parser = argparse.ArgumentParser(description="qUFL quadratic solver")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SafeLogger(args.log_path)

    try:
        with open(args.instance_path, "r") as f:
            instance = json.load(f)
        sol = solve_instance(instance, int(args.time_limit), logger)
    except BaseException:
        n = 0
        m = 0
        try:
            with open(args.instance_path, "r") as f:
                inst2 = json.load(f)
            n = int(inst2.get("num_facilities", 0))
            m = int(inst2.get("num_customers", 0))
        except BaseException:
            pass

        y = [0] * max(n, 0)
        x = [[0.0 for _ in range(max(m, 0))] for _ in range(max(n, 0))]
        sol = build_solution_dict(1e100 if m > 0 else 0.0, y, x)

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()