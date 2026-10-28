import argparse
import json
import math
import time
from typing import List, Tuple, Optional

from solution_logger import SolutionLogger


def build_cumulative_arrivals(
    T: int, NT: int, theta: int, lambdas: List[float]
) -> List[float]:
    """A[t] = cumulative arrivals from 0 up to time t (inclusive in continuous sense), for integer t."""
    pref = [0.0] * (NT + 1)
    for p in range(NT):
        pref[p + 1] = pref[p] + theta * float(lambdas[p])

    total = pref[NT]
    A = [0.0] * (T + 1)
    for t in range(T + 1):
        if t >= T:
            A[t] = total
        else:
            p = t // theta
            if p >= NT:
                A[t] = total
            else:
                A[t] = pref[p] + (t - p * theta) * float(lambdas[p])
    return A


def evaluate_schedule(times: List[int], A: List[float], T: int, C: float) -> float:
    """Compute total waiting-time objective for a full departure schedule."""
    D = len(times)
    if D == 0:
        # No trains: only accumulation over whole horizon
        n = A[T]
        return T * (0.5 * n)

    obj = 0.0

    # Initial interval [0, t1]
    t1 = times[0]
    n0 = A[t1]
    obj += t1 * (0.5 * n0)

    # Interior intervals [ti, t{i+1}]
    for i in range(1, D):
        s = times[i - 1]
        t = times[i]
        h = t - s
        w_start = max(0.0, A[s] - i * C)  # i trains have departed at time s? here s = t_i, i starts at1 => correct
        n = A[t] - A[s]
        obj += h * (w_start + 0.5 * n)

    # Final interval [tD, T]
    tD = times[-1]
    hF = T - tD
    wF = max(0.0, A[tD] - D * C)
    nF = A[T] - A[tD]
    obj += hF * (wF + 0.5 * nF)

    return obj


def build_simple_feasible_schedule(
    T: int, D: int, h_min: int, h_max: int
) -> Optional[List[int]]:
    """Construct a quick feasible schedule if possible (integer minutes)."""
    if D == 0:
        return []
    if T <= 0:
        return None

    if D == 1:
        return [max(0, min(T - 1, T // 2))]

    if h_min > h_max:
        return None
    if (D - 1) * h_min > (T - 1):
        return None

    # Use minimum interior headway and center the pattern.
    span = (D - 1) * h_min
    slack = (T - 1) - span
    t1 = slack // 2
    times = [t1 + i * h_min for i in range(D)]

    # Validate upper spacing against h_max (using h_min construction, always <= h_max if h_min<=h_max)
    for i in range(D - 1):
        dh = times[i + 1] - times[i]
        if dh < h_min or dh > h_max:
            return None
    if times[0] < 0 or times[-1] > T - 1:
        return None
    return times


def solve_instance(instance: dict, time_limit: int, logger: Optional[SolutionLogger]):
    params = instance["parameters"]

    T = int(params["T_planning_horizon_minutes"])
    D = int(params["D_num_train_services"])
    NT = int(params["NT_num_demand_periods"])
    theta = int(params["theta_period_length_minutes"])
    h_min = int(params["h_min_minutes"])
    h_max = int(params["h_max_minutes"])
    C = float(params["C_train_capacity_passengers"])

    demand_periods = instance["demand_periods"]
    lambdas = [float(dp["lambda_pax_per_min"]) for dp in demand_periods]

    start_time = time.time()
    deadline = start_time + max(1, int(time_limit))

    A = build_cumulative_arrivals(T, NT, theta, lambdas)

    best_obj = math.inf
    best_times = None

    # Quick feasibility checks
    if D < 0:
        return best_obj, []
    if D == 0:
        obj0 = evaluate_schedule([], A, T, C)
        if logger:
            logger.log(obj0)
        return obj0, []
    if T <= 0:
        return math.inf, None
    if D >= 2:
        if h_min > h_max or (D - 1) * h_min > (T - 1):
            return math.inf, None

    # Heuristic incumbent
    heuristic = build_simple_feasible_schedule(T, D, h_min, h_max)
    if heuristic is not None:
        h_obj = evaluate_schedule(heuristic, A, T, C)
        best_obj = h_obj
        best_times = heuristic
        if logger:
            logger.log(best_obj)

    # Exact DP over integer minute departures
    # parents[i][t] = predecessor departure time for train i at time t (1-indexed train index)
    # parents[1] unused
    parents = [[-1] * T for _ in range(D + 1)]

    INF = 1e100
    prev_cost = [INF] * T

    # Stage i=1
    tmax1 = T - 1 - (D - 1) * h_min if D >= 2 else T - 1
    if tmax1 < 0:
        return best_obj, best_times

    for t in range(0, tmax1 + 1):
        if time.time() >= deadline:
            return best_obj, best_times
        # initial interval cost
        prev_cost[t] = t * (0.5 * A[t])

        if D == 1:
            total = prev_cost[t] + (T - t) * (max(0.0, A[t] - C) + 0.5 * (A[T] - A[t]))
            if total + 1e-9 < best_obj:
                best_obj = total
                best_times = [t]
                if logger:
                    logger.log(best_obj)

    if D == 1:
        return best_obj, best_times

    # Stages i=1..D-1 transitions from train i to i+1
    # i denotes current train index
    for i in range(1, D):
        if time.time() >= deadline:
            return best_obj, best_times

        next_cost = [INF] * T
        tmax_curr = T - 1 - (D - i) * h_min
        tmax_next = T - 1 - (D - (i + 1)) * h_min

        for s in range(0, tmax_curr + 1):
            base = prev_cost[s]
            if base >= INF / 2:
                continue

            if time.time() >= deadline:
                return best_obj, best_times

            lo = s + h_min
            hi = min(s + h_max, tmax_next)

            if lo > hi:
                continue

            w_start = max(0.0, A[s] - i * C)

            for t in range(lo, hi + 1):
                h = t - s
                n = A[t] - A[s]
                cand = base + h * (w_start + 0.5 * n)
                if cand + 1e-12 < next_cost[t]:
                    next_cost[t] = cand
                    parents[i + 1][t] = s

                    # If this is the last departure stage, we have a complete schedule candidate.
                    if i + 1 == D:
                        final_cost = (T - t) * (max(0.0, A[t] - D * C) + 0.5 * (A[T] - A[t]))
                        total = cand + final_cost
                        if total + 1e-9 < best_obj:
                            # Reconstruct
                            sol = [0] * D
                            sol[D - 1] = t
                            cur_t = t
                            for k in range(D, 1, -1):
                                prev_t = parents[k][cur_t]
                                if prev_t < 0:
                                    sol = None
                                    break
                                sol[k - 2] = prev_t
                                cur_t = prev_t
                            if sol is not None:
                                best_obj = total
                                best_times = sol
                                if logger:
                                    logger.log(best_obj)

        prev_cost = next_cost

    return best_obj, best_times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    best_obj, best_times = solve_instance(instance, args.time_limit, logger)

    if best_times is None:
        # Fallback output in case of infeasibility
        departure_schedule = {}
        objective_value = float("inf")
    else:
        departure_schedule = {str(i + 1): int(best_times[i]) for i in range(len(best_times))}
        objective_value = float(best_obj)

    solution = {
        "objective_value": objective_value,
        "departure_schedule_minutes": departure_schedule,
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()