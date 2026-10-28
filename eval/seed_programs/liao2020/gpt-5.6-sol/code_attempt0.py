import argparse
import json
import math
import os
import random
import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from solution_logger import SolutionLogger


def read_instance(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def authoritative_cost(instance):
    structures = instance["cost_structures"]
    name = next(iter(structures))
    data = structures[name]
    return name, float(data["c_w"]), float(data["c_g"]), float(data["c_o"])


def patient_arrays(instance):
    patients = sorted(instance["patients"], key=lambda x: int(x["patient_index"]))
    p = len(patients)

    q_mean = np.empty(p)
    da_mean = np.empty(p)
    di_mean = np.empty(p)
    da_low = np.empty(p)
    da_high = np.empty(p)
    di_low = np.empty(p)
    di_high = np.empty(p)
    u_mean = np.empty(p)
    u_low = np.empty(p)
    u_high = np.empty(p)

    for j, patient in enumerate(patients):
        q_mean[j] = float(patient["mean_prep_adequacy"])
        da_mean[j] = float(patient["mean_duration_adequate_prep"])
        di_mean[j] = float(patient["mean_duration_inadequate_prep"])
        da_low[j] = float(patient["lower_bound_duration_adequate_prep"])
        da_high[j] = float(patient["upper_bound_duration_adequate_prep"])
        di_low[j] = float(patient["lower_bound_duration_inadequate_prep"])
        di_high[j] = float(patient["upper_bound_duration_inadequate_prep"])
        u_mean[j] = float(patient["mean_arrival_time_deviation"])
        u_low[j] = float(patient["lower_bound_arrival_time_deviation"])
        u_high[j] = float(patient["upper_bound_arrival_time_deviation"])

    return {
        "patients": patients,
        "P": p,
        "q_mean": q_mean,
        "da_mean": da_mean,
        "di_mean": di_mean,
        "da_low": da_low,
        "da_high": da_high,
        "di_low": di_low,
        "di_high": di_high,
        "u_mean": u_mean,
        "u_low": u_low,
        "u_high": u_high,
    }


def interpolation_probability(mean, low, high):
    result = np.zeros_like(mean, dtype=float)
    width = high - low
    mask = np.abs(width) > 1e-12
    result[mask] = (mean[mask] - low[mask]) / width[mask]
    return np.clip(result, 0.0, 1.0)


def feature_targets(data):
    q = data["q_mean"]
    return np.concatenate([
        q,
        q * data["da_mean"],
        (1.0 - q) * data["di_mean"],
        data["u_mean"],
    ])


def states_to_features(proc_state, arrival_high, data):
    """
    proc_state:
      0 = inadequate, low duration
      1 = inadequate, high duration
      2 = adequate, low duration
      3 = adequate, high duration
    """
    proc_state = np.asarray(proc_state, dtype=np.int8)
    arrival_high = np.asarray(arrival_high, dtype=np.int8)

    q = (proc_state >= 2).astype(float)
    duration_high = (proc_state & 1).astype(float)

    da = data["da_low"][None, :] + duration_high * (
        data["da_high"] - data["da_low"]
    )[None, :]
    di = data["di_low"][None, :] + duration_high * (
        data["di_high"] - data["di_low"]
    )[None, :]
    u = data["u_low"][None, :] + arrival_high * (
        data["u_high"] - data["u_low"]
    )[None, :]

    return np.concatenate([q, q * da, (1.0 - q) * di, u], axis=1)


def states_to_duration_arrival(proc_state, arrival_high, data):
    proc_state = np.asarray(proc_state, dtype=np.int8)
    arrival_high = np.asarray(arrival_high, dtype=np.int8)

    adequate = proc_state >= 2
    duration_high = (proc_state & 1).astype(float)

    da = data["da_low"][None, :] + duration_high * (
        data["da_high"] - data["da_low"]
    )[None, :]
    di = data["di_low"][None, :] + duration_high * (
        data["di_high"] - data["di_low"]
    )[None, :]
    durations = np.where(adequate, da, di)

    arrivals = data["u_low"][None, :] + arrival_high * (
        data["u_high"] - data["u_low"]
    )[None, :]
    return durations, arrivals


def scenario_costs(proc_state, arrival_high, order, schedule, data, length,
                   c_w, c_g, c_o):
    durations, arrivals = states_to_duration_arrival(proc_state, arrival_high, data)
    n = durations.shape[0]
    completion = np.zeros(n, dtype=float)
    total = np.zeros(n, dtype=float)

    for position, patient in enumerate(order):
        t = float(schedule[position])
        ready_from_arrival = t + arrivals[:, patient]

        if position == 0:
            start = np.maximum(t, ready_from_arrival)
        else:
            idle = np.maximum(0.0, t - completion)
            total += c_g * idle
            start = np.maximum(np.maximum(t, ready_from_arrival), completion)

        waiting = np.maximum(0.0, start - t)
        total += c_w * waiting
        completion = start + durations[:, patient]

    total += c_o * np.maximum(0.0, completion - length)
    return total


def generate_state_pool(data, count, rng):
    p = data["P"]
    count = max(count, 32)

    q_prob = np.clip(data["q_mean"], 0.0, 1.0)
    da_hi_prob = interpolation_probability(
        data["da_mean"], data["da_low"], data["da_high"]
    )
    di_hi_prob = interpolation_probability(
        data["di_mean"], data["di_low"], data["di_high"]
    )
    u_hi_prob = interpolation_probability(
        data["u_mean"], data["u_low"], data["u_high"]
    )

    proc = np.empty((count, p), dtype=np.int8)
    arr = np.empty((count, p), dtype=np.int8)

    random_count = max(1, count // 2)
    q_draw = rng.random((random_count, p)) < q_prob[None, :]
    hi_prob = np.where(q_draw, da_hi_prob[None, :], di_hi_prob[None, :])
    hi_draw = rng.random((random_count, p)) < hi_prob
    proc[:random_count] = 2 * q_draw.astype(np.int8) + hi_draw.astype(np.int8)
    arr[:random_count] = (
        rng.random((random_count, p)) < u_hi_prob[None, :]
    ).astype(np.int8)

    # Correlated threshold scenarios are important under unrestricted dependence.
    for row in range(random_count, count):
        u0 = ((row - random_count) + 0.5) / max(1, count - random_count)
        reverse_q = ((row // 7) & 1) != 0
        reverse_d = ((row // 11) & 1) != 0
        reverse_u = ((row // 13) & 1) != 0

        uq = 1.0 - u0 if reverse_q else u0
        ud = 1.0 - u0 if reverse_d else u0
        ua = 1.0 - u0 if reverse_u else u0

        q_draw = uq < q_prob
        hi_prob = np.where(q_draw, da_hi_prob, di_hi_prob)
        hi_draw = ud < hi_prob
        proc[row] = 2 * q_draw.astype(np.int8) + hi_draw.astype(np.int8)
        arr[row] = (ua < u_hi_prob).astype(np.int8)

    # Explicit support corners.
    corner_proc = [
        np.zeros(p, dtype=np.int8),
        np.ones(p, dtype=np.int8),
        np.full(p, 2, dtype=np.int8),
        np.full(p, 3, dtype=np.int8),
    ]
    corner_arr = [
        np.zeros(p, dtype=np.int8),
        np.ones(p, dtype=np.int8),
    ]
    k = 0
    for ps in corner_proc:
        for ar in corner_arr:
            if k >= count:
                break
            proc[k] = ps
            arr[k] = ar
            k += 1

    return proc, arr


def expected_duration(data):
    q = data["q_mean"]
    return q * data["da_mean"] + (1.0 - q) * data["di_mean"]


def make_schedule(order, expected, length, scale=1.0, first_shift=0.0):
    p = len(order)
    schedule = np.zeros(p, dtype=float)
    current = max(0.0, float(first_shift))
    schedule[0] = 0.0

    for i in range(1, p):
        current += scale * max(0.0, expected[order[i - 1]])
        schedule[i] = current

    if p > 1 and schedule[-1] > length:
        schedule *= length / schedule[-1]

    schedule[0] = 0.0
    schedule = np.maximum.accumulate(np.clip(schedule, 0.0, float(length)))
    return schedule


def candidate_orders(data, rng, max_random):
    p = data["P"]
    exp_d = expected_duration(data)
    q = data["q_mean"]
    u = data["u_mean"]

    orders = []
    seen = set()

    def add(order):
        key = tuple(int(x) for x in order)
        if key not in seen:
            seen.add(key)
            orders.append(np.asarray(key, dtype=int))

    add(np.arange(p))
    add(np.argsort(exp_d))
    add(np.argsort(-exp_d))
    add(np.argsort(q))
    add(np.argsort(-q))
    add(np.argsort(u))
    add(np.argsort(-u))
    add(np.argsort(exp_d + np.maximum(0.0, u)))
    add(np.argsort(-(exp_d + np.maximum(0.0, u))))

    base = np.argsort(exp_d)
    for _ in range(max_random):
        order = base.copy()
        swaps = 1 + int(rng.integers(1, max(2, min(5, p + 1))))
        for _ in range(swaps):
            if p >= 2:
                a, b = rng.integers(0, p, size=2)
                order[a], order[b] = order[b], order[a]
        add(order)

    return orders


def choose_schedule_and_order(data, length, c_w, c_g, c_o,
                              proc_pool, arr_pool, deadline, rng):
    exp_d = expected_duration(data)
    p = data["P"]

    max_random = min(250, max(20, 15 * p))
    orders = candidate_orders(data, rng, max_random)
    scales = [0.65, 0.8, 0.9, 1.0, 1.1, 1.25]

    best_score = math.inf
    best_order = orders[0]
    best_schedule = make_schedule(best_order, exp_d, length, 1.0)

    # A smaller pool is enough for ranking heuristic schedules.
    rank_n = min(len(proc_pool), 450)
    ps = proc_pool[:rank_n]
    ar = arr_pool[:rank_n]

    for order in orders:
        if time.monotonic() >= deadline:
            break
        for scale in scales:
            schedule = make_schedule(order, exp_d, length, scale)
            costs = scenario_costs(
                ps, ar, order, schedule, data, length, c_w, c_g, c_o
            )
            # Blend average and upper-tail behavior to reflect adversarial dependence.
            tail_index = max(0, int(0.85 * len(costs)))
            tail = np.partition(costs, tail_index)[tail_index:]
            score = 0.45 * float(np.mean(costs)) + 0.55 * float(np.mean(tail))

            if score + 1e-9 < best_score:
                best_score = score
                best_order = order.copy()
                best_schedule = schedule.copy()

    return best_order, best_schedule


def solve_moment_dual(features, costs, target, seconds):
    n, m = features.shape
    model = gp.Model("moment_dual")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.05, float(seconds))
    model.Params.Method = 1

    intercept = model.addVar(lb=-GRB.INFINITY, name="intercept")
    coefficients = model.addVars(m, lb=-GRB.INFINITY, name="coef")

    for s in range(n):
        expression = intercept + gp.quicksum(
            float(features[s, j]) * coefficients[j] for j in range(m)
            if abs(features[s, j]) > 1e-14
        )
        model.addConstr(expression >= float(costs[s]))

    objective = intercept + gp.quicksum(
        float(target[j]) * coefficients[j] for j in range(m)
        if abs(target[j]) > 1e-14
    )
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()

    if model.SolCount <= 0:
        b = float(np.max(costs)) if len(costs) else 0.0
        return b, np.zeros(m), b

    b = float(intercept.X)
    a = np.asarray([float(coefficients[j].X) for j in range(m)])
    value = float(b + np.dot(target, a))
    return b, a, value


def single_state_cost(proc, arr, order, schedule, data, length, c_w, c_g, c_o):
    return float(scenario_costs(
        proc[None, :], arr[None, :], order, schedule,
        data, length, c_w, c_g, c_o
    )[0])


def hill_climb_violation(proc, arr, coefficients, intercept, order, schedule,
                         data, length, c_w, c_g, c_o, max_passes=3):
    p = data["P"]
    proc = proc.copy()
    arr = arr.copy()

    def violation(ps, ar):
        f = states_to_features(ps[None, :], ar[None, :], data)[0]
        cost = single_state_cost(
            ps, ar, order, schedule, data, length, c_w, c_g, c_o
        )
        return cost - intercept - float(np.dot(coefficients, f))

    current = violation(proc, arr)

    for _ in range(max_passes):
        improved = False
        for patient in range(p):
            old_state = int(proc[patient])
            best_state = old_state
            best_value = current

            for state in range(4):
                if state == old_state:
                    continue
                proc[patient] = state
                value = violation(proc, arr)
                if value > best_value + 1e-8:
                    best_value = value
                    best_state = state

            proc[patient] = best_state
            if best_value > current + 1e-8:
                current = best_value
                improved = True

            old_arrival = int(arr[patient])
            arr[patient] = 1 - old_arrival
            value = violation(proc, arr)
            if value > current + 1e-8:
                current = value
                improved = True
            else:
                arr[patient] = old_arrival

        if not improved:
            break

    return proc, arr, current


def refine_dual(proc_pool, arr_pool, order, schedule, data, length,
                c_w, c_g, c_o, deadline, rng):
    target = feature_targets(data)
    proc_list = [row.copy() for row in proc_pool]
    arr_list = [row.copy() for row in arr_pool]

    intercept = 0.0
    coefficients = np.zeros(4 * data["P"])
    objective = math.inf

    for iteration in range(8):
        remaining = deadline - time.monotonic()
        if remaining <= 0.15:
            break

        proc_array = np.asarray(proc_list, dtype=np.int8)
        arr_array = np.asarray(arr_list, dtype=np.int8)
        features = states_to_features(proc_array, arr_array, data)
        costs = scenario_costs(
            proc_array, arr_array, order, schedule, data,
            length, c_w, c_g, c_o
        )

        intercept, coefficients, objective = solve_moment_dual(
            features, costs, target, min(remaining * 0.45, 10.0)
        )

        if time.monotonic() >= deadline:
            break

        starts = min(24, 5 + 2 * data["P"])
        additions = []
        largest_violation = -math.inf

        for k in range(starts):
            if time.monotonic() >= deadline:
                break
            if k < min(6, len(proc_list)):
                index = int(rng.integers(0, len(proc_list)))
                proc = proc_list[index].copy()
                arr = arr_list[index].copy()
            else:
                proc = rng.integers(0, 4, size=data["P"], dtype=np.int8)
                arr = rng.integers(0, 2, size=data["P"], dtype=np.int8)

            ps, ar, violation = hill_climb_violation(
                proc, arr, coefficients, intercept, order, schedule, data,
                length, c_w, c_g, c_o, max_passes=2
            )
            largest_violation = max(largest_violation, violation)
            if violation > 1e-5:
                additions.append((ps, ar, violation))

        if not additions:
            break

        additions.sort(key=lambda item: item[2], reverse=True)
        existing = {
            (bytes(np.asarray(ps, dtype=np.int8)),
             bytes(np.asarray(ar, dtype=np.int8)))
            for ps, ar in zip(proc_list, arr_list)
        }

        added = 0
        for ps, ar, _ in additions[:12]:
            key = (bytes(ps), bytes(ar))
            if key not in existing:
                existing.add(key)
                proc_list.append(ps.copy())
                arr_list.append(ar.copy())
                added += 1

        if added == 0:
            break

    return intercept, coefficients, objective


def clean_number(value):
    value = float(value)
    if abs(value) < 1e-9:
        return 0.0
    return value


def build_solution(instance, data, cost_name, c_w, c_g, c_o,
                   order, schedule, intercept, coefficients, objective):
    p = data["P"]
    model_variables = {}

    assignment = {}
    patient_start_times = {}
    position_of = np.empty(p, dtype=int)

    for i, patient_zero in enumerate(order):
        patient = int(patient_zero) + 1
        position = i + 1
        position_of[patient_zero] = position
        assignment[str(patient)] = position
        patient_start_times[str(patient)] = clean_number(schedule[i])
        model_variables[f"x[{patient},{position}]"] = 1.0

    schedule_dict = {}
    for i in range(p):
        value = clean_number(schedule[i])
        schedule_dict[str(i + 1)] = value
        if value != 0.0:
            model_variables[f"t[{i + 1}]"] = value

    # The finite-support moment dual has one intercept and four moment
    # coefficients per patient. Project them to appointment-position indices.
    rho = coefficients[0:p]
    alpha = coefficients[p:2 * p]
    lamb = coefficients[2 * p:3 * p]
    gamma = coefficients[3 * p:4 * p]

    if abs(intercept) > 1e-9:
        model_variables["beta[1]"] = clean_number(intercept)

    for i, patient_zero in enumerate(order):
        pos = i + 1
        patient = int(patient_zero) + 1

        values = [
            ("rho", rho[patient_zero]),
            ("alpha", alpha[patient_zero]),
            ("lambda", lamb[patient_zero]),
            ("gamma", gamma[patient_zero]),
        ]
        products = [
            ("eta", rho[patient_zero]),
            ("tau", alpha[patient_zero]),
            ("nu", lamb[patient_zero]),
            ("Gam", gamma[patient_zero]),
        ]

        for family, value in values:
            value = clean_number(value)
            if value != 0.0:
                model_variables[f"{family}[{pos}]"] = value

        for family, value in products:
            value = clean_number(value)
            if value != 0.0:
                model_variables[f"{family}[{patient},{pos}]"] = value

    return {
        "objective_value": clean_number(max(0.0, objective)),
        "model_variables": model_variables,
        "cost_structure": cost_name,
        "c_w": float(c_w),
        "c_g": float(c_g),
        "c_o": float(c_o),
        "assignment": assignment,
        "schedule": schedule_dict,
        "patient_start_times": patient_start_times,
    }


def write_solution(path, solution):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(solution, f, indent=2, sort_keys=False)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.monotonic()
    deadline = start_time + max(1, int(args.time_limit))
    rng = np.random.default_rng(0)
    random.seed(0)

    instance = read_instance(args.instance_path)
    data = patient_arrays(instance)
    length = float(instance["provider_service_hours_L_minutes"])
    cost_name, c_w, c_g, c_o = authoritative_cost(instance)

    p = data["P"]
    pool_size = min(3500, max(350, 100 + 55 * p))
    proc_pool, arr_pool = generate_state_pool(data, pool_size, rng)

    order, schedule = choose_schedule_and_order(
        data, length, c_w, c_g, c_o,
        proc_pool, arr_pool, deadline, rng
    )

    # Preserve time for the moment problem and output.
    dual_deadline = min(deadline - 0.05, time.monotonic() + max(
        0.1, 0.70 * (deadline - time.monotonic())
    ))

    intercept, coefficients, objective = refine_dual(
        proc_pool, arr_pool, order, schedule, data, length,
        c_w, c_g, c_o, dual_deadline, rng
    )

    if not np.isfinite(objective):
        costs = scenario_costs(
            proc_pool, arr_pool, order, schedule, data,
            length, c_w, c_g, c_o
        )
        intercept = float(np.max(costs)) if len(costs) else 0.0
        coefficients = np.zeros(4 * p)
        objective = intercept

    solution = build_solution(
        instance, data, cost_name, c_w, c_g, c_o,
        order, schedule, intercept, coefficients, objective
    )

    if logger:
        logger.log_solution(solution["objective_value"], solution)

    write_solution(args.solution_path, solution)


if __name__ == "__main__":
    main()