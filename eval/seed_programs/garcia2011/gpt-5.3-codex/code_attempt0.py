import argparse
import json
import time
import random
from typing import List, Tuple, Optional

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def compute_assignment(cost: np.ndarray, open_facilities: List[int]) -> Tuple[int, List[int], List[int]]:
    """Assign each customer to nearest open facility."""
    n = cost.shape[0]
    opens = np.array(open_facilities, dtype=int)
    sub = cost[:, opens]  # n x p
    idx = np.argmin(sub, axis=1)
    assigned_fac = opens[idx]
    assigned_cost = sub[np.arange(n), idx].astype(int)
    obj = int(assigned_cost.sum())
    return obj, assigned_fac.tolist(), assigned_cost.tolist()


def greedy_initial_solution(
    cost: np.ndarray,
    p: int,
    rng: random.Random,
    time_end: float
) -> List[int]:
    """Greedy add facilities by maximum marginal gain."""
    n = cost.shape[0]
    selected = np.zeros(n, dtype=bool)
    best_cost = np.full(n, 10**15, dtype=np.int64)
    opens = []

    for _ in range(p):
        if time.time() >= time_end:
            break
        best_j = -1
        best_gain = -1

        # evaluate every unopened node
        for j in range(n):
            if selected[j]:
                continue
            # gain = sum(max(0, best_cost - c[:,j]))
            col = cost[:, j]
            gain = int(np.maximum(0, best_cost - col).sum())
            if gain > best_gain:
                best_gain = gain
                best_j = j

        if best_j == -1:
            # fallback: random unopened
            cand = [j for j in range(n) if not selected[j]]
            best_j = rng.choice(cand)

        selected[best_j] = True
        opens.append(best_j)
        best_cost = np.minimum(best_cost, cost[:, best_j])

    # If interrupted before selecting p, fill randomly
    if len(opens) < p:
        remaining = [j for j in range(n) if not selected[j]]
        rng.shuffle(remaining)
        need = p - len(opens)
        opens.extend(remaining[:need])

    return opens


def local_search_swaps(
    cost: np.ndarray,
    opens: List[int],
    p: int,
    rng: random.Random,
    time_end: float,
    max_iter: int = 30,
    sample_closed_per_open: int = 40
) -> List[int]:
    """Simple first-improvement 1-swap local search."""
    n = cost.shape[0]
    open_set = set(opens)

    for _ in range(max_iter):
        if time.time() >= time_end:
            break

        open_list = list(open_set)
        closed_list = [j for j in range(n) if j not in open_set]
        if not closed_list:
            break

        # Compute best and second-best open facility for each customer
        sub = cost[:, open_list]  # n x p
        best_pos = np.argmin(sub, axis=1)
        best_cost = sub[np.arange(n), best_pos]
        assigned_fac = np.array([open_list[k] for k in best_pos], dtype=int)

        if len(open_list) >= 2:
            second_pos = np.argpartition(sub, 1, axis=1)[:, 1]
            second_cost = sub[np.arange(n), second_pos]
        else:
            second_cost = np.full(n, 10**15, dtype=np.int64)

        improved = False
        rng.shuffle(open_list)

        for a in open_list:
            if time.time() >= time_end:
                break

            # sample candidate closed facilities
            if len(closed_list) > sample_closed_per_open:
                b_candidates = rng.sample(closed_list, sample_closed_per_open)
            else:
                b_candidates = closed_list[:]
                rng.shuffle(b_candidates)

            mask_a = (assigned_fac == a)
            keep_best_or_second = np.where(mask_a, second_cost, best_cost)

            for b in b_candidates:
                if time.time() >= time_end:
                    break

                new_cost = np.minimum(keep_best_or_second, cost[:, b])
                delta = int(new_cost.sum() - best_cost.sum())
                if delta < 0:
                    open_set.remove(a)
                    open_set.add(b)
                    improved = True
                    break

            if improved:
                break

        if not improved:
            break

    result = list(open_set)
    # Ensure exactly p facilities (safety)
    if len(result) > p:
        result = result[:p]
    elif len(result) < p:
        missing = [j for j in range(n) if j not in open_set]
        result.extend(missing[: (p - len(result))])
    return result


def choose_candidates(cost: np.ndarray, p: int, heuristic_opens: List[int], max_full_vars: int = 250000) -> List[int]:
    """Choose facility candidate set for MIP."""
    n = cost.shape[0]
    if n * n <= max_full_vars:
        return list(range(n))

    # restricted candidates
    cand_limit = min(n, max(120, p + 20))
    col_sums = cost.sum(axis=0)
    best_by_score = np.argsort(col_sums)[:cand_limit].tolist()

    s = set(best_by_score)
    for j in heuristic_opens:
        s.add(j)

    # cap if too large
    if len(s) > cand_limit:
        # keep all heuristic opens + best remaining by score
        keep = set(heuristic_opens)
        for j in best_by_score:
            if len(keep) >= cand_limit:
                break
            keep.add(j)
        s = keep

    return sorted(list(s))


def solve_mip(
    cost: np.ndarray,
    p: int,
    candidates: List[int],
    warm_opens: List[int],
    logger: Optional[SolutionLogger],
    best_logged_ref: List[float],
    time_limit: float
) -> Optional[Tuple[int, List[int], List[int], List[int]]]:
    """Solve (possibly restricted) p-median MIP with Gurobi."""
    n = cost.shape[0]
    m = len(candidates)
    if m < p or time_limit <= 0.05:
        return None

    cand_pos = {c: j for j, c in enumerate(candidates)}
    warm_open_set = set(warm_opens)

    model = gp.Model("p_median")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.05, float(time_limit))
    model.Params.MIPGap = 0.0

    y = model.addVars(m, vtype=GRB.BINARY, name="y")
    x = model.addVars(n, m, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")

    model.addConstr(gp.quicksum(y[j] for j in range(m)) == p, name="open_p")
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(m)) == 1, name=f"assign_{i}")
        for j in range(m):
            model.addConstr(x[i, j] <= y[j], name=f"link_{i}_{j}")

    model.setObjective(
        gp.quicksum(float(cost[i, candidates[j]]) * x[i, j] for i in range(n) for j in range(m)),
        GRB.MINIMIZE
    )

    # Warm start
    for j, c in enumerate(candidates):
        y[j].Start = 1.0 if c in warm_open_set else 0.0

    # Assign each customer to nearest warm open (all warm opens are included in candidates by construction)
    warm_list = [f for f in warm_opens if f in cand_pos]
    if len(warm_list) >= 1:
        warm_arr = np.array(warm_list, dtype=int)
        sub = cost[:, warm_arr]
        idx = np.argmin(sub, axis=1)
        assigned = warm_arr[idx]
        for i in range(n):
            jj = cand_pos[int(assigned[i])]
            x[i, jj].Start = 1.0

    def cb(model_cb, where):
        if where == GRB.Callback.MIPSOL:
            obj = model_cb.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-6 < best_logged_ref[0]:
                best_logged_ref[0] = obj
                if logger:
                    logger.log(int(round(obj)))

    model.optimize(cb)

    if model.SolCount == 0:
        return None

    yvals = np.array([y[j].X for j in range(m)])
    open_idx = np.where(yvals > 0.5)[0].tolist()

    if len(open_idx) != p:
        # numeric fallback
        order = np.argsort(-yvals)
        open_idx = order[:p].tolist()

    opened = [candidates[j] for j in open_idx]
    obj, assigned_fac, assigned_cost = compute_assignment(cost, opened)
    return obj, opened, assigned_fac, assigned_cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    args = parser.parse_args()

    t0 = time.time()
    deadline = t0 + max(1, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    rng = random.Random(0)

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    n = int(instance["n"])
    p = int(instance["p"])
    cost = np.array(instance["cost_matrix"], dtype=np.int64)

    # --- Heuristic phase ---
    heur_time = min(20.0, max(1.0, 0.2 * args.time_limit))
    heur_deadline = min(deadline - 0.2, time.time() + heur_time)

    opens = greedy_initial_solution(cost, p, rng, heur_deadline)
    opens = local_search_swaps(cost, opens, p, rng, heur_deadline)

    best_obj, best_assign_fac, best_assign_cost = compute_assignment(cost, opens)
    best_opens = opens[:]

    if logger:
        logger.log(int(best_obj))
    best_logged_ref = [float(best_obj)]

    # --- MIP improvement phase ---
    remaining = deadline - time.time()
    if remaining > 0.5:
        candidates = choose_candidates(cost, p, best_opens)
        mip_result = solve_mip(
            cost=cost,
            p=p,
            candidates=candidates,
            warm_opens=best_opens,
            logger=logger,
            best_logged_ref=best_logged_ref,
            time_limit=max(0.05, deadline - time.time() - 0.05),
        )
        if mip_result is not None:
            obj2, opens2, assign_fac2, assign_cost2 = mip_result
            if obj2 < best_obj:
                best_obj = obj2
                best_opens = opens2
                best_assign_fac = assign_fac2
                best_assign_cost = assign_cost2

    # Build output
    assignments = [
        {"customer": int(i), "facility": int(best_assign_fac[i]), "cost": int(best_assign_cost[i])}
        for i in range(n)
    ]

    solution = {
        "objective_value": int(best_obj),
        "opened_facilities": [int(x) for x in best_opens],
        "assignments": assignments,
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()