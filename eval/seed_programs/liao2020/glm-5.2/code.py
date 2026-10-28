import argparse
import json
import time
from itertools import permutations
import gurobipy as gp
from gurobipy import GRB


def solve_schedule_lp(assign, P, L, c_w, c_g, c_o, scenarios, remaining_time):
    """Solve LP for optimal schedule given assignment.
    assign[i] = patient index (0-based) at position i.
    Returns (cost, schedule) where schedule is list of start times.
    """
    model = gp.Model("schedule")
    model.setParam('Seed', 0)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', max(1, remaining_time))

    t = model.addVars(P, lb=0, ub=L, name="t")
    model.addConstr(t[0] == 0)
    for i in range(1, P):
        model.addConstr(t[i] >= t[i - 1])

    total_cost = gp.LinExpr(0)

    for sc_idx, (prob, sc) in enumerate(scenarios):
        if prob < 1e-12:
            continue

        C_prev = None

        for i in range(P):
            p_idx = assign[i]
            d_val, u_val = sc[p_idx]

            R = model.addVar(lb=-GRB.INFINITY, name=f"R_{sc_idx}_{i}")
            W = model.addVar(lb=0, name=f"W_{sc_idx}_{i}")
            G = model.addVar(lb=0, name=f"G_{sc_idx}_{i}")
            C = model.addVar(lb=-GRB.INFINITY, name=f"C_{sc_idx}_{i}")

            model.addConstr(R >= t[i] + u_val)
            if i == 0:
                model.addConstr(R >= 0)
            else:
                model.addConstr(R >= C_prev)

            model.addConstr(W >= R - t[i])
            model.addConstr(G >= t[i] - R)
            model.addConstr(C == t[i] + W + d_val)

            total_cost += prob * (c_w * W + c_g * G)

            if i == P - 1:
                OT = model.addVar(lb=0, name=f"OT_{sc_idx}")
                model.addConstr(OT >= C - L)
                total_cost += prob * c_o * OT

            C_prev = C

    model.setObjective(total_cost, GRB.MINIMIZE)
    model.optimize()

    if model.SolCount > 0:
        schedule = [t[i].X for i in range(P)]
        cost = model.ObjVal
        model.dispose()
        return cost, schedule

    model.dispose()
    return float('inf'), [0.0] * P


def build_solution(cost, assign, schedule, P, cost_name, c_w, c_g, c_o):
    assignment = {}
    for i in range(P):
        assignment[str(assign[i] + 1)] = i + 1

    sched = {}
    for i in range(P):
        sched[str(i + 1)] = float(schedule[i])

    patient_start = {}
    for i in range(P):
        patient_start[str(assign[i] + 1)] = float(schedule[i])

    model_vars = {}
    for i in range(P):
        model_vars[f"x[{assign[i]+1},{i+1}]"] = 1.0
        model_vars[f"t[{i+1}]"] = float(schedule[i])

    return {
        "objective_value": float(cost),
        "model_variables": model_vars,
        "cost_structure": cost_name,
        "c_w": float(c_w),
        "c_g": float(c_g),
        "c_o": float(c_o),
        "assignment": assignment,
        "schedule": sched,
        "patient_start_times": patient_start,
        "model_variable_name_patterns": {
            "x[p,i]": "patient-position assignment, p and i are 1..P",
            "t[i]": "position start time",
            "rho[i], alpha[i], lambda[i], gamma[i]": "dual variables",
            "beta[i]": "i is 1..P+2",
            "z[i], v[i], r[i], e[i]": "one-dimensional auxiliary variables",
            "eta[p,i], tau[p,i], nu[p,i], Gam[p,i]": "McCormick variables",
            "zeta[p,i], psi[p,i], phi[p,i]": "nonnegative McCormick variables"
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    P = instance['num_patients_P']
    L = instance['provider_service_hours_L_minutes']

    cost_structures = instance['cost_structures']
    cost_name = list(cost_structures.keys())[0]
    cs = cost_structures[cost_name]
    c_w = float(cs['c_w'])
    c_g = float(cs['c_g'])
    c_o = float(cs['c_o'])

    patients = instance['patients']

    # Per-patient data
    mu_d = []
    patient_dur_data = []  # list of sorted (value, cumulative_prob)
    patient_arr_data = []  # list of (u_low, u_high, threshold)

    for p_idx in range(P):
        pat = patients[p_idx]
        mu_q = pat['mean_prep_adequacy']
        mu_A = pat['mean_duration_adequate_prep']
        mu_I = pat['mean_duration_inadequate_prep']
        d_AL = pat['lower_bound_duration_adequate_prep']
        d_AU = pat['upper_bound_duration_adequate_prep']
        d_IL = pat['lower_bound_duration_inadequate_prep']
        d_IU = pat['upper_bound_duration_inadequate_prep']

        # Duration outcomes: 4 extreme points with probabilities
        dur_outcomes = {}

        if d_AU - d_AL > 1e-9:
            p_ade_lo = mu_q * (d_AU - mu_A) / (d_AU - d_AL)
            p_ade_hi = mu_q * (mu_A - d_AL) / (d_AU - d_AL)
        else:
            p_ade_lo = 0.0
            p_ade_hi = mu_q
        dur_outcomes[d_AL] = dur_outcomes.get(d_AL, 0.0) + max(0.0, p_ade_lo)
        dur_outcomes[d_AU] = dur_outcomes.get(d_AU, 0.0) + max(0.0, p_ade_hi)

        if d_IU - d_IL > 1e-9:
            p_inade_lo = (1 - mu_q) * (d_IU - mu_I) / (d_IU - d_IL)
            p_inade_hi = (1 - mu_q) * (mu_I - d_IL) / (d_IU - d_IL)
        else:
            p_inade_lo = 0.0
            p_inade_hi = (1 - mu_q)
        dur_outcomes[d_IL] = dur_outcomes.get(d_IL, 0.0) + max(0.0, p_inade_lo)
        dur_outcomes[d_IU] = dur_outcomes.get(d_IU, 0.0) + max(0.0, p_inade_hi)

        sorted_durs = sorted(dur_outcomes.items())
        cum = 0.0
        dur_cum = []
        for val, prob in sorted_durs:
            cum += prob
            dur_cum.append((val, min(1.0, cum)))
        if dur_cum:
            dur_cum[-1] = (dur_cum[-1][0], 1.0)

        patient_dur_data.append(dur_cum)
        mu_d.append(mu_q * mu_A + (1 - mu_q) * mu_I)

        # Arrival data
        u_lo = pat['lower_bound_arrival_time_deviation']
        u_hi = pat['upper_bound_arrival_time_deviation']
        mu_u_val = pat['mean_arrival_time_deviation']

        if u_hi - u_lo > 1e-9:
            p_u_hi = (mu_u_val - u_lo) / (u_hi - u_lo)
        else:
            p_u_hi = 0.5
        p_u_hi = max(0.0, min(1.0, p_u_hi))
        arr_thresh = 1.0 - p_u_hi

        patient_arr_data.append((u_lo, u_hi, arr_thresh))

    # Generate comonotone coupling scenarios
    all_thresholds = {0.0, 1.0}
    for p_idx in range(P):
        for val, cum in patient_dur_data[p_idx]:
            all_thresholds.add(max(0.0, min(1.0, cum)))
        all_thresholds.add(max(0.0, min(1.0, patient_arr_data[p_idx][2])))

    t_vals = sorted(all_thresholds)

    # Merge close thresholds for large P to limit scenario count
    if P > 10:
        merged = [t_vals[0]]
        for v in t_vals[1:]:
            if v - merged[-1] > 0.005:
                merged.append(v)
        if 1.0 not in merged:
            merged.append(1.0)
        t_vals = sorted(set(merged))

    scenarios = []
    for s in range(len(t_vals) - 1):
        prob = t_vals[s + 1] - t_vals[s]
        if prob < 1e-12:
            continue
        mid = (t_vals[s] + t_vals[s + 1]) / 2

        sc = {}
        for p_idx in range(P):
            d_val = patient_dur_data[p_idx][-1][0]
            for val, cum in patient_dur_data[p_idx]:
                if mid < cum:
                    d_val = val
                    break
            u_lo, u_hi, arr_thresh = patient_arr_data[p_idx]
            u_val = u_hi if mid >= arr_thresh else u_lo
            sc[p_idx] = (d_val, u_val)
        scenarios.append((prob, sc))

    # Generate candidate assignments
    assignments = []

    var_d = []
    for p_idx in range(P):
        var = 0.0
        mean = mu_d[p_idx]
        cum_prev = 0.0
        for val, cum in patient_dur_data[p_idx]:
            prob = cum - cum_prev
            var += prob * (val - mean) ** 2
            cum_prev = cum
        var_d.append(var)

    assignments.append(sorted(range(P), key=lambda p: mu_d[p]))
    assignments.append(sorted(range(P), key=lambda p: -mu_d[p]))
    assignments.append(sorted(range(P), key=lambda p: var_d[p]))
    assignments.append(sorted(range(P), key=lambda p: -var_d[p]))
    cv = [var_d[p] ** 0.5 / max(mu_d[p], 1e-6) for p in range(P)]
    assignments.append(sorted(range(P), key=lambda p: cv[p]))
    assignments.append(sorted(range(P), key=lambda p: -cv[p]))

    if P <= 7:
        for perm in permutations(range(P)):
            assignments.append(list(perm))

    seen = set()
    unique_assignments = []
    for a in assignments:
        key = tuple(a)
        if key not in seen:
            seen.add(key)
            unique_assignments.append(a)

    best_cost = float('inf')
    best_assignment = None
    best_schedule = None

    for assign in unique_assignments:
        elapsed = time.time() - start_time
        if elapsed > args.time_limit * 0.6:
            break

        remaining = max(1, args.time_limit - elapsed)
        cost, schedule = solve_schedule_lp(assign, P, L, c_w, c_g, c_o, scenarios, remaining)

        if cost < best_cost:
            best_cost = cost
            best_assignment = list(assign)
            best_schedule = list(schedule)

            solution = build_solution(best_cost, best_assignment, best_schedule,
                                       P, cost_name, c_w, c_g, c_o)
            if logger:
                logger.log_solution(best_cost, solution)

    if best_assignment is None:
        best_assignment = list(range(P))
        cum = 0.0
        best_schedule = []
        for i in range(P):
            best_schedule.append(cum)
            cum += mu_d[i]
        best_cost = 0.0

    # Local search: adjacent swaps
    improved = True
    while improved and time.time() - start_time < args.time_limit * 0.85:
        improved = False
        for i in range(P - 1):
            if time.time() - start_time > args.time_limit * 0.85:
                break
            new_assign = list(best_assignment)
            new_assign[i], new_assign[i + 1] = new_assign[i + 1], new_assign[i]
            key = tuple(new_assign)
            if key in seen:
                continue
            seen.add(key)
            remaining = max(1, args.time_limit - (time.time() - start_time))
            cost, schedule = solve_schedule_lp(new_assign, P, L, c_w, c_g, c_o, scenarios, remaining)
            if cost < best_cost - 1e-6:
                best_cost = cost
                best_assignment = new_assign
                best_schedule = schedule
                improved = True
                solution = build_solution(best_cost, best_assignment, best_schedule,
                                           P, cost_name, c_w, c_g, c_o)
                if logger:
                    logger.log_solution(best_cost, solution)

    # 2-opt local search (reverse segments)
    for i in range(P):
        for j in range(i + 1, min(i + 4, P)):
            if time.time() - start_time > args.time_limit * 0.93:
                break
            new_assign = list(best_assignment)
            new_assign[i:j + 1] = reversed(new_assign[i:j + 1])
            key = tuple(new_assign)
            if key in seen:
                continue
            seen.add(key)
            remaining = max(1, args.time_limit - (time.time() - start_time))
            cost, schedule = solve_schedule_lp(new_assign, P, L, c_w, c_g, c_o, scenarios, remaining)
            if cost < best_cost - 1e-6:
                best_cost = cost
                best_assignment = new_assign
                best_schedule = schedule
                solution = build_solution(best_cost, best_assignment, best_schedule,
                                           P, cost_name, c_w, c_g, c_o)
                if logger:
                    logger.log_solution(best_cost, solution)

    # Insertion local search
    for i in range(P):
        for j in range(P):
            if i == j or time.time() - start_time > args.time_limit * 0.95:
                break
            new_assign = list(best_assignment)
            val = new_assign.pop(i)
            new_assign.insert(j, val)
            key = tuple(new_assign)
            if key in seen:
                continue
            seen.add(key)
            remaining = max(1, args.time_limit - (time.time() - start_time))
            cost, schedule = solve_schedule_lp(new_assign, P, L, c_w, c_g, c_o, scenarios, remaining)
            if cost < best_cost - 1e-6:
                best_cost = cost
                best_assignment = new_assign
                best_schedule = schedule
                solution = build_solution(best_cost, best_assignment, best_schedule,
                                           P, cost_name, c_w, c_g, c_o)
                if logger:
                    logger.log_solution(best_cost, solution)

    solution = build_solution(best_cost, best_assignment, best_schedule,
                               P, cost_name, c_w, c_g, c_o)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()