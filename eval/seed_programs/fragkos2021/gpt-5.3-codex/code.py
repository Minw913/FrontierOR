import argparse
import ast
import gzip
import json
import random
import re
import time
from typing import Dict, List, Optional, Tuple

from solution_logger import SolutionLogger


def _read_text_any(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()

    # Handle gzipped input transparently
    if len(raw) >= 2 and raw[0] == 0x1F and raw[1] == 0x8B:
        raw = gzip.decompress(raw)

    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    # Last resort
    return raw.decode("utf-8", errors="replace")


def _parse_json_robust(path: str):
    text = _read_text_any(path)

    # 1) Standard JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) Remove comments and trailing commas (JSONC/relaxed JSON)
    try:
        cleaned = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        cleaned = re.sub(r"//.*?$", "", cleaned, flags=re.M)
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        return json.loads(cleaned)
    except Exception:
        pass

    # 3) Python literal fallback
    try:
        return ast.literal_eval(text)
    except Exception as e:
        raise ValueError(f"Could not parse instance file as JSON/relaxed JSON/Python literal: {e}")


def _get_first(data: dict, keys: List[str], default=None):
    for k in keys:
        if k in data:
            return data[k]
    return default


def load_instance(path: str):
    data = _parse_json_robust(path)
    if not isinstance(data, dict):
        raise ValueError("Parsed instance is not a JSON object.")

    num_facilities = _get_first(data, ["num_facilities", "n_facilities", "m", "n"])
    num_customers = _get_first(data, ["num_customers", "n_customers", "k", "c"])

    nodes = _get_first(data, ["nodes", "node_data"], default=None)

    # Transportation matrix
    trans = _get_first(
        data,
        [
            "transportation_cost_matrix",
            "transport_cost_matrix",
            "shipping_cost_matrix",
            "cost_matrix",
            "c_ij",
            "transportation_cost",
        ],
    )

    # Penalty vector
    penalty = _get_first(
        data,
        ["penalty_cost", "penalty_costs", "penalties", "unserved_penalty", "p_j", "penalty"],
    )

    # If dimensions missing, infer from matrix
    if trans is not None and isinstance(trans, list) and len(trans) > 0 and isinstance(trans[0], list):
        if num_customers is None:
            num_customers = len(trans)
        if num_facilities is None:
            num_facilities = len(trans[0])

    if num_facilities is None or num_customers is None:
        raise ValueError("Could not infer num_facilities/num_customers from instance.")

    num_facilities = int(num_facilities)
    num_customers = int(num_customers)

    # Demands
    demands = None
    if nodes is not None and isinstance(nodes, list) and len(nodes) >= num_customers:
        demands = [float(nodes[j].get("demand", 0.0)) for j in range(num_customers)]
    else:
        d_arr = _get_first(data, ["demand", "demands", "customer_demands", "d_j"], default=None)
        if d_arr is not None:
            demands = [float(d_arr[j]) for j in range(num_customers)]
    if demands is None:
        demands = [1.0] * num_customers

    # Fixed setup costs
    fixed_cost = None
    if nodes is not None and isinstance(nodes, list) and len(nodes) >= num_facilities:
        fixed_cost = [float(nodes[i].get("fixed_setup_cost", 0.0)) for i in range(num_facilities)]
    else:
        f_arr = _get_first(
            data,
            ["fixed_setup_cost", "fixed_setup_costs", "fixed_cost", "fixed_costs", "open_cost", "f_i"],
            default=None,
        )
        if f_arr is not None:
            fixed_cost = [float(f_arr[i]) for i in range(num_facilities)]
    if fixed_cost is None:
        fixed_cost = [0.0] * num_facilities

    # Survival probabilities
    survival_prob = None
    if nodes is not None and isinstance(nodes, list) and len(nodes) >= num_facilities:
        surv = []
        for i in range(num_facilities):
            nd = nodes[i]
            if "survival_probability" in nd:
                s = float(nd["survival_probability"])
            elif "disruption_probability" in nd:
                s = 1.0 - float(nd["disruption_probability"])
            else:
                s = None
            surv.append(s)
        if all(v is not None for v in surv):
            survival_prob = [max(0.0, min(1.0, float(v))) for v in surv]

    if survival_prob is None:
        s_arr = _get_first(data, ["survival_probability", "survival_probabilities", "q_i"], default=None)
        if s_arr is not None:
            survival_prob = [max(0.0, min(1.0, float(s_arr[i]))) for i in range(num_facilities)]

    if survival_prob is None:
        d_arr = _get_first(
            data, ["disruption_probability", "disruption_probabilities", "failure_probability"], default=None
        )
        if d_arr is not None:
            survival_prob = [max(0.0, min(1.0, 1.0 - float(d_arr[i]))) for i in range(num_facilities)]

    if survival_prob is None:
        survival_prob = [1.0] * num_facilities

    # Transportation matrix required
    if trans is None:
        raise ValueError("Transportation cost matrix not found in instance.")

    # Normalize matrix shape [num_customers][num_facilities]
    if len(trans) != num_customers:
        # Try transpose interpretation
        if len(trans) == num_facilities and len(trans[0]) == num_customers:
            trans = [list(row) for row in zip(*trans)]
        else:
            raise ValueError("Transportation matrix dimensions inconsistent with num_customers/num_facilities.")

    trans_cost = [[float(trans[j][i]) for i in range(num_facilities)] for j in range(num_customers)]

    # Penalty required (or infer conservative fallback)
    if penalty is None:
        penalty = []
        for j in range(num_customers):
            mx = max(trans_cost[j]) if num_facilities > 0 else 0.0
            penalty.append(mx)
    penalty = [float(penalty[j]) for j in range(num_customers)]

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
    deadline = start + max(1, int(time_limit)) - 1e-3

    ordered_fac = [sorted(range(num_facilities), key=lambda i, j=j: trans_cost[j][i]) for j in range(num_customers)]
    obj_cache: Dict[int, float] = {}

    def objective(mask: int) -> float:
        if mask in obj_cache:
            return obj_cache[mask]

        fixed = 0.0
        mm = mask
        i = 0
        while mm:
            if mm & 1:
                fixed += fixed_cost[i]
            mm >>= 1
            i += 1

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

    full_mask = (1 << num_facilities) - 1 if num_facilities < 62 else None
    if full_mask is not None and time.time() < deadline:
        v = objective(full_mask)
        if v < best_obj - 1e-12:
            best_obj, best_mask = v, full_mask
            if logger:
                logger.log(best_obj)

    def try_update(mask: int, val: float):
        nonlocal best_mask, best_obj
        if val < best_obj - 1e-12:
            best_obj, best_mask = val, mask
            if logger:
                logger.log(best_obj)

    def local_search(mask: int) -> Tuple[int, float]:
        cur = objective(mask)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_n_mask = mask
            best_n_obj = cur
            for i in range(num_facilities):
                if time.time() >= deadline:
                    break
                nmask = mask ^ (1 << i)
                nobj = objective(nmask)
                if nobj < best_n_obj - 1e-12:
                    best_n_obj = nobj
                    best_n_mask = nmask
            if best_n_mask != mask:
                mask, cur = best_n_mask, best_n_obj
                improved = True
                try_update(mask, cur)
        return mask, cur

    def greedy_add(randomized=False, alpha=0.2) -> Tuple[int, float]:
        mask = 0
        cur = objective(mask)
        while time.time() < deadline:
            cand = []
            best_val = cur
            worst_val = cur
            for i in range(num_facilities):
                if (mask >> i) & 1:
                    continue
                nmask = mask | (1 << i)
                nobj = objective(nmask)
                cand.append((nobj, i))
                if nobj < best_val:
                    best_val = nobj
                if nobj > worst_val:
                    worst_val = nobj
            if not cand:
                break
            improv = [(v, i) for (v, i) in cand if v < cur - 1e-12]
            if not improv:
                break

            if randomized:
                cutoff = best_val + alpha * (worst_val - best_val)
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

    if time.time() < deadline:
        m0, _ = greedy_add(randomized=False)
        m0, v0 = local_search(m0)
        try_update(m0, v0)

    random.seed(0)
    attractiveness = []
    for i in range(num_facilities):
        gain = 0.0
        s = survival_prob[i]
        for j in range(num_customers):
            gain += demands[j] * max(0.0, penalty[j] - trans_cost[j][i]) * s
        attractiveness.append(gain - fixed_cost[i])
    sorted_fac = sorted(range(num_facilities), key=lambda i: attractiveness[i], reverse=True)

    while time.time() < deadline:
        if random.random() < 0.5:
            alpha = random.uniform(0.05, 0.5)
            m, _ = greedy_add(randomized=True, alpha=alpha)
        else:
            m = best_mask
            if num_facilities > 0:
                k = random.randint(1, min(4, num_facilities))
                for _ in range(k):
                    i = random.randrange(num_facilities)
                    m ^= (1 << i)
                for i in sorted_fac[: max(1, num_facilities // 10)]:
                    if random.random() < 0.2:
                        m |= (1 << i)

        m, v = local_search(m)
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

    try:
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
        sol = {"objective_value": float(best_obj), "facility_locations": facility_locations}

    except Exception:
        # Guaranteed valid fallback output
        # Try minimal parse for num_facilities; else 0-length vector
        nfac = 0
        try:
            data = _parse_json_robust(args.instance_path)
            if isinstance(data, dict):
                nfac = int(_get_first(data, ["num_facilities", "n_facilities", "m", "n"], default=0) or 0)
        except Exception:
            nfac = 0
        sol = {"objective_value": 1e100, "facility_locations": [0] * max(0, nfac)}

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()