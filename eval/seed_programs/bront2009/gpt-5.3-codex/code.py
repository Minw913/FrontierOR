import argparse
import json
import time
from typing import List, Dict, Tuple, Set

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Airline offer-set time-allocation optimizer")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    return parser.parse_args()


def build_data(instance: dict):
    lam = float(instance["lambda"])
    alpha = float(instance.get("alpha", 1.0))
    T = float(instance["booking_horizon"]["T"])

    legs = instance["network"]["legs"]
    m = len(legs)
    leg_ids = [leg["leg_id"] for leg in legs]
    leg_id_to_idx = {lid: i for i, lid in enumerate(leg_ids)}
    capacities = [alpha * float(leg["capacity"]) for leg in legs]

    products = instance["products"]
    n = len(products)
    product_ids = [p["product_id"] for p in products]
    pid_to_idx = {pid: j for j, pid in enumerate(product_ids)}
    fares = [float(p["fare"]) for p in products]

    # incidence a_ij: leg i used by product j
    A = [[0.0] * n for _ in range(m)]
    for j, p in enumerate(products):
        for lid in p["legs_used"]:
            i = leg_id_to_idx[lid]
            A[i][j] = 1.0

    segments = instance["segments"]
    L = len(segments)
    lam_l = [float(seg["lambda_l"]) for seg in segments]
    total_seg_rate = sum(lam_l)
    if total_seg_rate > 0:
        seg_probs = [x / total_seg_rate for x in lam_l]
    else:
        seg_probs = [1.0 / L] * L if L > 0 else []

    seg_cons_idx: List[List[int]] = []
    seg_pref: List[List[float]] = []
    seg_v0: List[float] = []
    # product -> list of (segment l, preference v_lj)
    prod_in_segments: List[List[Tuple[int, float]]] = [[] for _ in range(n)]

    for l, seg in enumerate(segments):
        cset_ids = seg["consideration_set"]
        pref_vec = seg["preference_vector"]
        v0 = float(seg["no_purchase_preference"])

        cidx = [pid_to_idx[pid] for pid in cset_ids]
        vlist = [float(v) for v in pref_vec]

        seg_cons_idx.append(cidx)
        seg_pref.append(vlist)
        seg_v0.append(v0)

        for j, v in zip(cidx, vlist):
            prod_in_segments[j].append((l, v))

    return {
        "lambda": lam,
        "T": T,
        "m": m,
        "n": n,
        "L": L,
        "capacities": capacities,
        "A": A,
        "fares": fares,
        "product_ids": product_ids,
        "seg_probs": seg_probs,
        "seg_cons_idx": seg_cons_idx,
        "seg_pref": seg_pref,
        "seg_v0": seg_v0,
        "prod_in_segments": prod_in_segments,
    }


def compute_metrics_for_set(
    S: Set[int],
    fares: List[float],
    A: List[List[float]],
    seg_probs: List[float],
    seg_cons_idx: List[List[int]],
    seg_pref: List[List[float]],
    seg_v0: List[float],
):
    """Return (expected_revenue_per_arrival, expected_leg_consumption_per_arrival)."""
    n = len(fares)
    m = len(A)
    probs = [0.0] * n

    for l in range(len(seg_probs)):
        cidx = seg_cons_idx[l]
        vp = seg_pref[l]
        denom = seg_v0[l]
        chosen_positions = []

        for pos, j in enumerate(cidx):
            if j in S:
                denom += vp[pos]
                chosen_positions.append(pos)

        if not chosen_positions:
            continue

        pl = seg_probs[l]
        for pos in chosen_positions:
            j = cidx[pos]
            probs[j] += pl * vp[pos] / denom

    rev = sum(fares[j] * probs[j] for j in range(n))
    q = [0.0] * m
    for i in range(m):
        ai = A[i]
        q[i] = sum(ai[j] * probs[j] for j in range(n))
    return rev, q


def local_search_pricing(
    n: int,
    seg_probs: List[float],
    seg_v0: List[float],
    prod_in_segments: List[List[Tuple[int, float]]],
    w: List[float],
    initial_set: Set[int],
    time_deadline: float,
):
    """
    Heuristic maximize:
      G(S) = sum_l p_l * [sum_{j in S∩C_l} w_j v_lj] / [v0_l + sum_{j in S∩C_l} v_lj]
    """
    L = len(seg_probs)
    S = set(initial_set)

    x = [0.0] * L  # denominator additions
    y = [0.0] * L  # numerator sums
    for j in S:
        wj = w[j]
        for l, v in prod_in_segments[j]:
            x[l] += v
            y[l] += wj * v

    eps = 1e-10

    while time.time() < time_deadline:
        best_delta = 0.0
        best_j = None
        best_add = True

        for j in range(n):
            in_set = j in S
            sign = -1.0 if in_set else 1.0
            delta = 0.0
            wj = w[j]

            for l, v in prod_in_segments[j]:
                old_den = seg_v0[l] + x[l]
                old_term = y[l] / old_den

                new_x = x[l] + sign * v
                new_y = y[l] + sign * (wj * v)
                new_den = seg_v0[l] + new_x
                new_term = new_y / new_den

                delta += seg_probs[l] * (new_term - old_term)

            if delta > best_delta + eps:
                best_delta = delta
                best_j = j
                best_add = not in_set

        if best_j is None:
            break

        # apply move
        j = best_j
        sgn = 1.0 if best_add else -1.0
        wj = w[j]
        for l, v in prod_in_segments[j]:
            x[l] += sgn * v
            y[l] += sgn * (wj * v)

        if best_add:
            S.add(j)
        else:
            S.remove(j)

    # compute objective value
    g = 0.0
    for l in range(L):
        g += seg_probs[l] * (y[l] / (seg_v0[l] + x[l]))
    return S, g


def pricing_heuristic(data: dict, pi: List[float], sigma: float, end_time: float):
    lam = data["lambda"]
    n = data["n"]
    fares = data["fares"]
    A = data["A"]

    # dual-adjusted product values
    w = []
    for j in range(n):
        penalty = 0.0
        for i in range(data["m"]):
            penalty += pi[i] * A[i][j]
        w.append(fares[j] - penalty)

    seeds = [set()]

    pos = [j for j in range(n) if w[j] > 0.0]
    if pos:
        seeds.append(set(pos))
        pos_sorted = sorted(pos, key=lambda j: w[j], reverse=True)
        for k in [1, 2, 3, 5, 8, 12]:
            if len(pos_sorted) >= k:
                seeds.append(set(pos_sorted[:k]))

    for cidx in data["seg_cons_idx"][: min(10, len(data["seg_cons_idx"]))]:
        seeds.append(set(cidx))
        seeds.append({j for j in cidx if w[j] > 0.0})

    # deduplicate
    uniq = []
    seen = set()
    for s in seeds:
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            uniq.append(s)

    if time.time() >= end_time:
        return set(), -1e100

    best_set = set()
    best_g = 0.0

    remaining = max(0.0, end_time - time.time())
    if remaining <= 0:
        return set(), -1e100
    per_seed = max(0.005, remaining / max(1, len(uniq)))

    for s in uniq:
        if time.time() >= end_time:
            break
        deadline = min(end_time, time.time() + per_seed)
        cand_set, cand_g = local_search_pricing(
            n=n,
            seg_probs=data["seg_probs"],
            seg_v0=data["seg_v0"],
            prod_in_segments=data["prod_in_segments"],
            w=w,
            initial_set=s,
            time_deadline=deadline,
        )
        if cand_g > best_g:
            best_g = cand_g
            best_set = cand_set

    rc = lam * best_g - sigma
    return best_set, rc


def solve(instance: dict, time_limit: int, logger: SolutionLogger = None):
    data = build_data(instance)
    start = time.time()
    end_time = start + max(1, int(time_limit))

    lam = data["lambda"]
    T = data["T"]
    m = data["m"]
    n = data["n"]

    model = gp.Model("offer_set_lp")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.ModelSense = GRB.MAXIMIZE

    # constraints: sum_s lambda*q_is*t_s <= cap_i ; sum_s t_s <= T
    leg_constrs = []
    for i in range(m):
        c = model.addConstr(gp.LinExpr() <= data["capacities"][i], name=f"cap_{i}")
        leg_constrs.append(c)
    time_constr = model.addConstr(gp.LinExpr() <= T, name="time")

    columns = []  # each: {"key","set_idx","var"}
    key_to_col = {}

    def add_column(set_idx: Set[int]):
        key = tuple(sorted(set_idx))
        if len(key) == 0:
            return False
        if key in key_to_col:
            return False

        rev, q = compute_metrics_for_set(
            set(set_idx),
            data["fares"],
            data["A"],
            data["seg_probs"],
            data["seg_cons_idx"],
            data["seg_pref"],
            data["seg_v0"],
        )

        col = gp.Column()
        for i in range(m):
            col.addTerms(lam * q[i], leg_constrs[i])
        col.addTerms(1.0, time_constr)

        v = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, obj=lam * rev, column=col, name=f"t_{len(columns)}")
        model.update()

        key_to_col[key] = len(columns)
        columns.append({"key": key, "set_idx": set(set_idx), "var": v})
        return True

    # Initial columns
    for j in range(n):
        add_column({j})
    if n > 0:
        add_column(set(range(n)))

    best_obj = float("-inf")

    while time.time() < end_time - 0.05:
        rem = end_time - time.time()
        if rem <= 0:
            break
        model.Params.TimeLimit = rem
        model.optimize()

        if model.SolCount > 0:
            obj = float(model.ObjVal)
            if obj > best_obj + 1e-9:
                best_obj = obj
                if logger:
                    logger.log(best_obj)
        else:
            break

        # Need optimal LP solution to safely use dual prices
        if model.Status != GRB.OPTIMAL:
            break

        if time.time() >= end_time - 0.05:
            break

        pi = [leg_constrs[i].Pi for i in range(m)]
        sigma = time_constr.Pi

        pricing_end = min(end_time - 0.02, time.time() + max(0.03, 0.25 * (end_time - time.time())))
        cand_set, rc = pricing_heuristic(data, pi, sigma, pricing_end)

        if (not cand_set) or rc <= 1e-8:
            break

        if not add_column(cand_set):
            break

    # Final optimize with any leftover time
    if time.time() < end_time - 0.01:
        model.Params.TimeLimit = end_time - time.time()
        model.optimize()
        if model.SolCount > 0:
            obj = float(model.ObjVal)
            if obj > best_obj + 1e-9 and logger:
                logger.log(obj)

    if model.SolCount > 0:
        objective_value = float(model.ObjVal)
    else:
        objective_value = 0.0

    active_columns = []
    if model.SolCount > 0:
        for c in columns:
            x = c["var"].X
            if x > 1e-8:
                offer_ids = [data["product_ids"][j] for j in sorted(c["set_idx"])]
                active_columns.append(
                    {
                        "offer_set": offer_ids,
                        "time_allocated": float(x),
                    }
                )

    return {
        "objective_value": objective_value,
        "active_columns": active_columns,
    }


def main():
    args = parse_args()
    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    sol = solve(instance, args.time_limit, logger)

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()