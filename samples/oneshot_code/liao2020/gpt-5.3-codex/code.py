import argparse
import json
import time
from typing import Dict, List, Tuple, Optional

from solution_logger import SolutionLogger


def get_costs(instance: dict) -> Tuple[float, float, float]:
    cs = instance.get("cost_structures", {})
    if not cs:
        return 1.0, 1.0, 1.0
    k = next(iter(cs.keys()))
    d = cs[k]
    return float(d["c_w"]), float(d["c_g"]), float(d["c_o"])


def build_data(instance: dict):
    patients = instance["patients"]
    P = len(patients)

    patient_ids = []
    proc_type = []
    mean_d = []
    d_low = []
    d_up = []
    mean_u = []
    u_low = []
    u_up = []

    for p in patients:
        patient_ids.append(int(p["patient_index"]))
        proc_type.append(p["procedure_type"])

        mq = float(p["mean_prep_adequacy"])
        muA = float(p["mean_duration_adequate_prep"])
        muI = float(p["mean_duration_inadequate_prep"])
        mean_d.append(mq * muA + (1.0 - mq) * muI)

        dAL = float(p["lower_bound_duration_adequate_prep"])
        dAU = float(p["upper_bound_duration_adequate_prep"])
        dIL = float(p["lower_bound_duration_inadequate_prep"])
        dIU = float(p["upper_bound_duration_inadequate_prep"])
        d_low.append(min(dAL, dIL))
        d_up.append(max(dAU, dIU))

        mean_u.append(float(p["mean_arrival_time_deviation"]))
        u_low.append(float(p["lower_bound_arrival_time_deviation"]))
        u_up.append(float(p["upper_bound_arrival_time_deviation"]))

    return P, patient_ids, proc_type, mean_d, d_low, d_up, mean_u, u_low, u_up


def make_scenarios(mean_d, d_low, d_up, mean_u, u_low, u_up):
    # Small adversarial set to approximate distributional robustness.
    # Each scenario gives deterministic durations and arrival deviations per patient.
    return [
        ("mean", mean_d, mean_u),
        ("high_high", d_up, u_up),
        ("high_low", d_up, u_low),
        ("low_high", d_low, u_up),
        ("low_low", d_low, u_low),
    ]


def eval_schedule_order(
    order: List[int],
    schedule: List[float],
    scenarios,
    c_w: float,
    c_g: float,
    c_o: float,
    L: float,
) -> float:
    P = len(order)
    worst = -1e30

    for _, dvals, uvals in scenarios:
        prev_comp = 0.0
        waiting = 0.0
        idle = 0.0

        for i in range(P):
            p = order[i]
            s_i = schedule[i]

            start_i = max(s_i, s_i + uvals[p], prev_comp)
            waiting += (start_i - s_i)

            if i > 0:
                idle += max(0.0, s_i - prev_comp)

            comp_i = start_i + dvals[p]
            prev_comp = comp_i

        overtime = max(0.0, prev_comp - L)
        cost = c_w * waiting + c_g * idle + c_o * overtime
        if cost > worst:
            worst = cost

    return worst


def build_schedule_from_params(
    order: List[int],
    L: float,
    base_d: List[float],
    mean_u: List[float],
    alpha: float,
    beta: float,
) -> List[float]:
    P = len(order)
    s = [0.0] * P
    for i in range(1, P):
        p_prev = order[i - 1]
        gap = alpha * base_d[p_prev] + beta * max(0.0, mean_u[p_prev])
        if gap < 0.0:
            gap = 0.0
        s[i] = s[i - 1] + gap
        if s[i] > L:
            s[i] = L
    return s


def isotonic_clip_nondec(s: List[float], L: float) -> List[float]:
    # Ensure 0 = s0 <= ... <= s_{P-1} <= L
    if not s:
        return s
    s2 = s[:]
    s2[0] = 0.0
    for i in range(1, len(s2)):
        if s2[i] < s2[i - 1]:
            s2[i] = s2[i - 1]
        if s2[i] > L:
            s2[i] = L
    return s2


def optimize_schedule_for_order(
    order: List[int],
    L: float,
    mean_d: List[float],
    d_up: List[float],
    mean_u: List[float],
    scenarios,
    c_w: float,
    c_g: float,
    c_o: float,
    t_start: float,
    time_limit: float,
    quick: bool = False,
):
    best_s = None
    best_obj = float("inf")

    # Candidate schedule parameters
    alpha_grid = [0.0, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.6]
    beta_grid = [0.0, 0.3, 0.6, 1.0]
    if quick:
        alpha_grid = [0.5, 0.9, 1.3]
        beta_grid = [0.0, 0.6]

    base_options = [mean_d, d_up]

    # Add equal-spacing schedule
    P = len(order)
    if P <= 1:
        cand = [0.0]
        obj = eval_schedule_order(order, cand, scenarios, c_w, c_g, c_o, L)
        best_s, best_obj = cand, obj
    else:
        eq = [i * (L / (P - 1)) for i in range(P)]
        eq = isotonic_clip_nondec(eq, L)
        obj = eval_schedule_order(order, eq, scenarios, c_w, c_g, c_o, L)
        best_s, best_obj = eq, obj

    # Parameter sweep
    for base_d in base_options:
        for a in alpha_grid:
            for b in beta_grid:
                if time.time() - t_start >= time_limit - 0.05:
                    return best_s, best_obj
                s = build_schedule_from_params(order, L, base_d, mean_u, a, b)
                s = isotonic_clip_nondec(s, L)
                obj = eval_schedule_order(order, s, scenarios, c_w, c_g, c_o, L)
                if obj < best_obj:
                    best_obj = obj
                    best_s = s

    # Local coordinate refinement
    if best_s is None:
        best_s = [0.0] * P

    delta = max(1.0, L / (10.0 if quick else 6.0))
    min_delta = 1.0 if quick else 0.5

    while delta >= min_delta and (time.time() - t_start) < time_limit - 0.05:
        improved = False
        for i in range(1, P):  # keep s[0]=0
            if time.time() - t_start >= time_limit - 0.05:
                break

            cur = best_s[i]
            low_bound = best_s[i - 1]
            up_bound = L if i == P - 1 else best_s[i + 1]

            candidates = []
            lo = max(low_bound, cur - delta)
            hi = min(up_bound, cur + delta)
            if lo != cur:
                candidates.append(lo)
            if hi != cur and hi != lo:
                candidates.append(hi)

            local_best_val = best_obj
            local_best = cur

            for val in candidates:
                s_try = best_s[:]
                s_try[i] = val
                obj = eval_schedule_order(order, s_try, scenarios, c_w, c_g, c_o, L)
                if obj < local_best_val - 1e-9:
                    local_best_val = obj
                    local_best = val

            if local_best != cur:
                best_s[i] = local_best
                best_obj = local_best_val
                improved = True

        if not improved:
            delta *= 0.5

    return best_s, best_obj


def make_initial_orders(
    P: int,
    proc_type: List[str],
    mean_d: List[float],
    d_low: List[float],
    d_up: List[float],
    mean_u: List[float],
) -> List[List[int]]:
    idx = list(range(P))
    rng = [d_up[i] - d_low[i] for i in range(P)]

    orders = []

    # Simple deterministic sorts
    orders.append(sorted(idx, key=lambda i: mean_d[i]))  # SPT-like
    orders.append(sorted(idx, key=lambda i: mean_d[i], reverse=True))  # LPT-like
    orders.append(sorted(idx, key=lambda i: max(0.0, mean_u[i])))  # early arrivers first
    orders.append(sorted(idx, key=lambda i: (proc_type[i], mean_d[i])))  # C then UC (lexicographic)
    orders.append(sorted(idx, key=lambda i: (0 if proc_type[i] == "C" else 1, mean_d[i])))
    orders.append(sorted(idx, key=lambda i: (0 if proc_type[i] == "UC" else 1, mean_d[i])))

    # Risk-adjusted keys
    for rho in [0.2, 0.5, 1.0]:
        for eta in [0.2, 0.5]:
            orders.append(sorted(idx, key=lambda i: mean_d[i] + rho * rng[i] + eta * max(0.0, mean_u[i])))

    # Remove duplicates
    uniq = []
    seen = set()
    for o in orders:
        t = tuple(o)
        if t not in seen:
            seen.add(t)
            uniq.append(o)
    return uniq


def adjacent_swap_local_search(
    current_order: List[int],
    current_schedule: List[float],
    current_obj: float,
    L: float,
    mean_d: List[float],
    d_up: List[float],
    mean_u: List[float],
    scenarios,
    c_w: float,
    c_g: float,
    c_o: float,
    t_start: float,
    time_limit: float,
    logger: Optional[SolutionLogger],
):
    P = len(current_order)
    improved_global = True

    while improved_global and (time.time() - t_start) < time_limit - 0.05:
        improved_global = False
        for i in range(P - 1):
            if time.time() - t_start >= time_limit - 0.05:
                break

            cand_order = current_order[:]
            cand_order[i], cand_order[i + 1] = cand_order[i + 1], cand_order[i]

            # Quick optimization for neighbor
            cand_schedule, cand_obj = optimize_schedule_for_order(
                cand_order, L, mean_d, d_up, mean_u, scenarios, c_w, c_g, c_o,
                t_start, time_limit, quick=True
            )

            if cand_obj + 1e-9 < current_obj:
                current_order, current_schedule, current_obj = cand_order, cand_schedule, cand_obj
                improved_global = True
                if logger:
                    logger.log(float(current_obj))
                break  # first-improvement restart

    return current_order, current_schedule, current_obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    t_start = time.time()
    time_limit = max(1, int(args.time_limit))

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    P, patient_ids, proc_type, mean_d, d_low, d_up, mean_u, u_low, u_up = build_data(instance)
    L = float(instance["provider_service_hours_L_minutes"])
    c_w, c_g, c_o = get_costs(instance)
    scenarios = make_scenarios(mean_d, d_low, d_up, mean_u, u_low, u_up)

    orders = make_initial_orders(P, proc_type, mean_d, d_low, d_up, mean_u)

    best_obj = float("inf")
    best_order = list(range(P))
    best_schedule = [0.0] * P

    # Evaluate initial candidate orders
    for order in orders:
        if time.time() - t_start >= time_limit - 0.05:
            break

        quick = (P > 60)  # speed guard for very large instances
        sched, obj = optimize_schedule_for_order(
            order, L, mean_d, d_up, mean_u, scenarios, c_w, c_g, c_o,
            t_start, time_limit, quick=quick
        )

        if obj < best_obj:
            best_obj = obj
            best_order = order[:]
            best_schedule = sched[:]
            if logger:
                logger.log(float(best_obj))

    # Local search around incumbent
    if time.time() - t_start < time_limit - 0.05:
        best_order, best_schedule, best_obj = adjacent_swap_local_search(
            best_order, best_schedule, best_obj,
            L, mean_d, d_up, mean_u, scenarios, c_w, c_g, c_o,
            t_start, time_limit, logger
        )

    # Safety fallback
    if best_schedule is None or len(best_schedule) != P:
        best_order = sorted(range(P), key=lambda i: mean_d[i])
        best_schedule = [0.0] * P
        for i in range(1, P):
            best_schedule[i] = min(L, best_schedule[i - 1] + mean_d[best_order[i - 1]])
        best_obj = eval_schedule_order(best_order, best_schedule, scenarios, c_w, c_g, c_o, L)
        if logger:
            logger.log(float(best_obj))

    # Build output schema
    # assignment: patient_index -> position (1..P)
    # schedule: position_index -> start time
    # patient_start_times: patient_index -> assigned position start
    local_pos = {p: i for i, p in enumerate(best_order)}

    assignment_out: Dict[str, int] = {}
    patient_start_out: Dict[str, float] = {}
    for local_idx, pid in enumerate(patient_ids):
        pos = local_pos[local_idx] + 1
        assignment_out[str(pid)] = int(pos)
        patient_start_out[str(pid)] = float(best_schedule[pos - 1])

    schedule_out: Dict[str, float] = {}
    for i in range(P):
        schedule_out[str(i + 1)] = float(best_schedule[i])

    sol = {
        "objective_value": float(best_obj),
        "assignment": assignment_out,
        "schedule": schedule_out,
        "patient_start_times": patient_start_out,
    }

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()