import argparse
import json
import math
import random
import time
from typing import Dict, List, Tuple, Optional

from solution_logger import SolutionLogger


def load_instance(path: str):
    with open(path, "r") as f:
        data = json.load(f)

    num_facilities = int(data["num_facilities"])
    num_customers = int(data["num_customers"])
    nodes = data["nodes"]

    demands = [float(nodes[j]["demand"]) for j in range(num_customers)]
    fixed_cost = [float(nodes[i]["fixed_setup_cost"]) for i in range(num_facilities)]

    survival_prob = []
    for i in range(num_facilities):
        nd = nodes[i]
        if "survival_probability" in nd:
            s = float(nd["survival_probability"])
        elif "disruption_probability" in nd:
            s = 1.0 - float(nd["disruption_probability"])
        else:
            s = 1.0
        survival_prob.append(min(1.0, max(0.0, s)))

    trans = data["transportation_cost_matrix"]
    penalty = [float(x) for x in data["penalty_cost"]]

    # Sanity conversion
    trans_cost = [[float(trans[j][i]) for i in range(num_facilities)] for j in range(num_customers)]

    return num_facilities, num_customers, demands, fixed_cost, survival_prob, trans_cost, penalty


def solve(
    num_facilities: int,
    num_customers: int,
    demands: List[float],
    fixed_cost: List[float],
    survival_prob: List[float],
    trans_cost: List[List[float]],
    penalty: List[float],
    time_limit: int,
    logger: Optional[SolutionLogger] = None,
):
    start = time.time()
    deadline = start + max(1, time_limit) - 1e-3

    # Pre-sort facilities per customer by transportation cost (ascending)
    ordered_fac = [
        sorted(range(num_facilities), key=lambda i, j=j: trans_cost[j][i])
        for j in range(num_customers)
    ]

    # Caches
    obj_cache: Dict[int, float] = {}

    def objective(mask: int) -> float:
        if mask in obj_cache:
            return obj_cache[mask]

        # Fixed setup cost
        fixed = 0.0
        mm = mask
        idx = 0
        while mm:
            if mm & 1:
                fixed += fixed_cost[idx]
            mm >>= 1
            idx += 1

        expected = 0.0
        for j in range(num_customers):
            p_none = 1.0
            ec = 0.0
            for i in ordered_fac[j]:
                if (mask >> i) & 1:
                    s = survival_prob[i]
                    if s <= 0.0:
                        continue
                    take = p_none * s
                    ec += trans_cost[j][i] * take
                    p_none *= (1.0 - s)
                    if p_none < 1e-14:
                        p_none = 0.0
                        break
            ec += penalty[j] * p_none
            expected += demands[j] * ec

        val = fixed + expected
        obj_cache[mask] = val
        return val

    best_mask = 0
    best_obj = objective(0)

    if logger:
        logger.log(best_obj)

    full_mask = (1 << num_facilities) - 1
    if time.time() < deadline:
        full_obj = objective(full_mask)
        if full_obj < best_obj - 1e-12:
            best_obj, best_mask = full_obj, full_mask
            if logger:
                logger.log(best_obj)

    def try_update(mask: int, val: float):
        nonlocal best_mask, best_obj
        if val < best_obj - 1e-12:
            best_obj = val
            best_mask = mask
            if logger:
                logger.log(best_obj)

    def local_search_1flip(mask: int) -> Tuple[int, float]:
        cur = objective(mask)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_n_mask = mask
            best_n_obj = cur

            for i in range(num_facilities):
                if time.time() >= deadline:
                    break
                nmask = mask ^ (1 << i)  # flip i
                nobj = objective(nmask)
                if nobj < best_n_obj - 1e-12:
                    best_n_obj = nobj
                    best_n_mask = nmask

            if best_n_mask != mask:
                mask = best_n_mask
                cur = best_n_obj
                improved = True
                try_update(mask, cur)

        return mask, cur

    # Greedy add from empty (deterministic)
    def greedy_add(randomized: bool = False, alpha: float = 0.2) -> Tuple[int, float]:
        mask = 0
        cur = objective(mask)

        while time.time() < deadline:
            candidates = []
            best_val = cur
            worst_val = cur
            for i in range(num_facilities):
                if (mask >> i) & 1:
                    continue
                nmask = mask | (1 << i)
                nobj = objective(nmask)
                if nobj < best_val:
                    best_val = nobj
                if nobj > worst_val:
                    worst_val = nobj
                candidates.append((nobj, i))

            if not candidates:
                break

            # Only additions that improve
            improv = [(v, i) for (v, i) in candidates if v < cur - 1e-12]
            if not improv:
                break

            if randomized:
                cutoff = best_val + alpha * (max(best_val, worst_val) - best_val)
                rcl = [(v, i) for (v, i) in improv if v <= cutoff + 1e-12]
                if not rcl:
                    rcl = improv
                v, i = random.choice(rcl)
            else:
                v, i = min(improv)

            mask |= (1 << i)
            cur = v
            try_update(mask, cur)

        return mask, cur

    # Deterministic constructive + LS
    if time.time() < deadline:
        m0, _ = greedy_add(randomized=False)
        m0, v0 = local_search_1flip(m0)
        try_update(m0, v0)

    # Randomized multi-start / iterated local improvement
    random.seed(0)
    # Simple attractiveness for randomized starts
    attractiveness = []
    for i in range(num_facilities):
        gain = 0.0
        s = survival_prob[i]
        for j in range(num_customers):
            gain += demands[j] * max(0.0, penalty[j] - trans_cost[j][i]) * s
        attractiveness.append(gain - fixed_cost[i])

    sorted_fac = sorted(range(num_facilities), key=lambda i: attractiveness[i], reverse=True)

    while time.time() < deadline:
        # Alternate between randomized greedy and perturb-from-best
        if random.random() < 0.5:
            alpha = random.uniform(0.05, 0.5)
            m, _ = greedy_add(randomized=True, alpha=alpha)
        else:
            # Perturb best solution
            m = best_mask
            k = 1 if num_facilities <= 1 else random.randint(1, min(4, num_facilities))
            for _ in range(k):
                i = random.randrange(num_facilities)
                m ^= (1 << i)
            # Bias repair toward attractive facilities
            for i in sorted_fac[: max(1, num_facilities // 10)]:
                if random.random() < 0.2:
                    m |= (1 << i)

        m, v = local_search_1flip(m)
        try_update(m, v)

    return best_mask, best_obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    (
        num_facilities,
        num_customers,
        demands,
        fixed_cost,
        survival_prob,
        trans_cost,
        penalty,
    ) = load_instance(args.instance_path)

    best_mask, best_obj = solve(
        num_facilities=num_facilities,
        num_customers=num_customers,
        demands=demands,
        fixed_cost=fixed_cost,
        survival_prob=survival_prob,
        trans_cost=trans_cost,
        penalty=penalty,
        time_limit=args.time_limit,
        logger=logger,
    )

    facility_locations = [1 if ((best_mask >> i) & 1) else 0 for i in range(num_facilities)]
    solution = {
        "objective_value": float(best_obj),
        "facility_locations": facility_locations,
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()