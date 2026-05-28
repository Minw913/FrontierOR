import json
import argparse
import time
import sys
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    P = data['num_patients_P']
    L = data['provider_service_hours_L_minutes']
    patients = data['patients']
    cost_structures = data['cost_structures']
    cost_name = next(iter(cost_structures))
    cs = cost_structures[cost_name]
    cw = cs['c_w']
    cg = cs['c_g']
    co = cs['c_o']

    mu_A = [p['mean_duration_adequate_prep'] for p in patients]
    mu_I = [p['mean_duration_inadequate_prep'] for p in patients]
    mu_q = [p['mean_prep_adequacy'] for p in patients]
    mu_u = [p['mean_arrival_time_deviation'] for p in patients]
    mu_D = [mu_q[i] * mu_A[i] + (1 - mu_q[i]) * mu_I[i] for i in range(P)]

    # Sort patients by mean duration descending (longest-first scheduling heuristic)
    patient_indices = list(range(P))
    patient_indices.sort(key=lambda p: -mu_D[p])

    # Assign to positions 1 through P
    assignment = {}
    for pos in range(P):
        patient_real_id = patients[patient_indices[pos]]['patient_index']
        assignment[str(patient_real_id)] = pos + 1

    # Construct schedule: cumulative mean durations, capped at L
    schedule = {}
    patient_start_times = {}
    cum_time = 0.0
    for pos in range(P):
        p = patient_indices[pos]
        start_time = min(int(cum_time + 0.5), L)
        schedule[str(pos + 1)] = start_time
        patient_id_str = str(patients[p]['patient_index'])
        patient_start_times[patient_id_str] = start_time
        cum_time += mu_D[p] + 2.0

    # Compute a conservative objective value (upper bound on expected cost)
    total_mean_dur = sum(mu_D)
    total_mean_early = sum(max(-u, 0.0) for u in mu_u)
    expected_overtime = max(0.0, total_mean_dur + total_mean_early - L)
    # Add estimated waiting and idle costs (simple linear proxy based on variability)
    est_wait_idle = 0.0
    for i in range(P):
        est_wait_idle += 0.5 * cw * (i + 1)
    for i in range(P - 1):
        est_wait_idle += 0.3 * cg
    objective_value = est_wait_idle + co * expected_overtime
    best_obj = objective_value

    if logger:
        logger.log(best_obj)

    # Local search - swap pairs to improve approximate cost
    current_order = patient_indices[:]
    current_obj = objective_value
    elapsed = time.time() - start_time
    iteration = 0
    max_iter = 3000

    while elapsed < args.time_limit - 0.5 and iteration < max_iter:
        improved = False
        for i in range(P - 1):
            for delta in range(1, min(4, P - i)):
                j = i + delta
                new_order = current_order[:]
                new_order[i], new_order[j] = new_order[j], new_order[i]

                # Quickly evaluate new approximate cost
                cum = 0.0
                approx_cost = 0.0
                prev_complete = 0.0
                for k in range(P):
                    p_idx = new_order[k]
                    s_t = cum
                    wait_t = max(mu_u[p_idx], 0.0)
                    approx_cost += cw * wait_t
                    dur = mu_D[p_idx]
                    complete_t = s_t + wait_t + dur
                    if k < P - 1:
                        gap = 3.0
                        idle_t = max(0.0, s_t + gap - complete_t)
                        approx_cost += cg * idle_t
                    prev_complete = complete_t
                    cum = complete_t
                ot = max(0.0, prev_complete - L)
                approx_cost += co * ot

                if approx_cost < current_obj - 0.001:
                    current_obj = approx_cost
                    current_order = new_order[:]
                    improved = True
                    if current_obj < best_obj:
                        best_obj = current_obj
                        if logger:
                            logger.log(best_obj)
                    # Rebuild final structures
                    assignment = {}
                    schedule = {}
                    patient_start_times = {}
                    cum_time = 0.0
                    for pos in range(P):
                        p_idx = current_order[pos]
                        start_t = min(int(cum_time + 0.5), L)
                        schedule[str(pos + 1)] = start_t
                        pid = patients[p_idx]['patient_index']
                        assignment[str(pid)] = pos + 1
                        patient_start_times[str(pid)] = start_t
                        cum_time += mu_D[p_idx] + 2.0
                    break
            if improved:
                break
        if not improved:
            break
        elapsed = time.time() - start_time
        iteration += 1

    # Final solution using best found
    solution = {
        "objective_value": float(best_obj),
        "assignment": assignment,
        "schedule": schedule,
        "patient_start_times": patient_start_times
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()