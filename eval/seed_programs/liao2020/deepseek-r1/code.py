import json
import math
import random
import argparse
import time
from solution_logger import SolutionLogger

def norm_ppf(p):
    if p <= 0 or p >= 1:
        return None
    p_low = 0.02425
    p_high = 0.97575
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        numerator = (((((-7.784894002430293e-03 * q - 3.223964580411365e-01) * q - 2.400758277161838e+00) * q - 2.549732539343734e+00) * q + 4.374664141464968e+00) * q + 2.938163982698783e+00)
        denominator = ((((7.784695709041462e-03 * q + 3.224671290700398e-01) * q + 2.445134137142996e+00) * q + 3.754408661907416e+00) * q + 1.0)
        x = numerator / denominator
        return x
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        poly = (((((((-4.14045315348448e-10 * r + 3.220865476912e-08) * r - 1.131782319e-06) * r + 1.9768423e-05) * r - 0.000196639) * r + 0.008516) * r - 0.020749) * r + 0.025) * q
        return poly
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        numerator = (((((-7.784894002430293e-03 * q - 3.223964580411365e-01) * q - 2.400758277161838e+00) * q - 2.549732539343734e+00) * q + 4.374664141464968e+00) * q + 2.938163982698783e+00)
        denominator = ((((7.784695709041462e-03 * q + 3.224671290700398e-01) * q + 2.445134137142996e+00) * q + 3.754408661907416e+00) * q + 1.0)
        x = -numerator / denominator
        return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=float, required=True)
    parser.add_argument('--log_path', type=str, default=None, required=False)
    args = parser.parse_args()

    start_time = time.time()
    if args.log_path:
        logger = SolutionLogger(args.log_path, sense='minimize')
    else:
        logger = None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    L = instance['provider_service_hours_L_minutes']
    patients_list = instance['patients']
    patients = {}
    for p in patients_list:
        patients[str(p['patient_index'])] = p
    
    cost_structures = instance['cost_structures']
    cost_structure_name = next(iter(cost_structures.keys()))
    cost_structure = cost_structures[cost_structure_name]
    c_w = float(cost_structure['c_w'])
    c_g = float(cost_structure['c_g'])
    c_o = float(cost_structure['c_o'])

    patient_mean_dur_list = []
    for pid, p in patients.items():
        mu_q = p['mean_prep_adequacy']
        mu_A = p['mean_duration_adequate_prep']
        mu_I = p['mean_duration_inadequate_prep']
        mean_dur = mu_q * mu_A + (1 - mu_q) * mu_I
        patient_mean_dur_list.append((pid, mean_dur))
    
    patient_mean_dur_list.sort(key=lambda x: x[1], reverse=True)
    patient_ids_sorted = [pid for (pid, mean_dur) in patient_mean_dur_list]
    P = len(patient_ids_sorted)
    
    s = [0.0]
    buffers = []
    for i in range(P - 1):
        pid = patient_ids_sorted[i]
        p = patients[pid]
        mu_q_p = p['mean_prep_adequacy']
        mu_A_p = p['mean_duration_adequate_prep']
        mu_I_p = p['mean_duration_inadequate_prep']
        d_AL_p = p['lower_bound_duration_adequate_prep']
        d_AU_p = p['upper_bound_duration_adequate_prep']
        d_IL_p = p['lower_bound_duration_inadequate_prep']
        d_IU_p = p['upper_bound_duration_inadequate_prep']
        var_A = (mu_A_p - d_AL_p) * (d_AU_p - mu_A_p)
        var_I = (mu_I_p - d_IL_p) * (d_IU_p - mu_I_p)
        mean_dur_p = mu_q_p * mu_A_p + (1 - mu_q_p) * mu_I_p
        var_dur = mu_q_p * (var_A + (mu_A_p - mean_dur_p)**2) + (1 - mu_q_p) * (var_I + (mu_I_p - mean_dur_p)**2)

        next_pid = patient_ids_sorted[i+1]
        next_p = patients[next_pid]
        mu_u = next_p['mean_arrival_time_deviation']
        u_L = next_p['lower_bound_arrival_time_deviation']
        u_U = next_p['upper_bound_arrival_time_deviation']
        var_arr = (mu_u - u_L) * (u_U - mu_u)
        total_variance = var_dur + var_arr

        critical_ratio = c_g / (c_g + c_w)
        z_val = norm_ppf(critical_ratio)
        if z_val is None or total_variance <= 0:
            buffer_i = 0.0
        else:
            buffer_i = z_val * math.sqrt(total_variance)
        buffers.append(buffer_i)

        s_i = s[-1] + mean_dur_p + buffer_i
        s.append(s_i)
    
    total_mean_duration = sum(mean_dur for (_, mean_dur) in patient_mean_dur_list)
    total_buffer = sum(buffers) if buffers else 0
    if total_mean_duration + total_buffer > L:
        available_padding = max(0, L - total_mean_duration)
        if total_buffer > 0:
            factor = available_padding / total_buffer
        else:
            factor = 0
        adjusted_buffers = [factor * buf for buf in buffers]
        s_adjusted = [0.0]
        for i in range(P-1):
            pid = patient_ids_sorted[i]
            p = patients[pid]
            mu_q_p = p['mean_prep_adequacy']
            mu_A_p = p['mean_duration_adequate_prep']
            mu_I_p = p['mean_duration_inadequate_prep']
            mean_dur_p = mu_q_p * mu_A_p + (1 - mu_q_p) * mu_I_p
            s_next = s_adjusted[-1] + mean_dur_p + adjusted_buffers[i]
            s_adjusted.append(s_next)
        s = s_adjusted
    else:
        pass

    assignment = {}
    patient_start_times = {}
    schedule = {}
    for idx, pid in enumerate(patient_ids_sorted):
        pos = idx + 1
        assignment[pid] = pos
        patient_start_times[pid] = s[idx]
        schedule[str(pos)] = s[idx]
    
    total_sim_cost = 0.0
    num_sim = 0
    random.seed(0)
    remaining_time = args.time_limit - (time.time() - start_time)
    if remaining_time <= 0:
        if logger:
            logger.log_solution(0.0, None)
    else:
        if P > 0:
            for n in range(1000):
                if time.time() - start_time > args.time_limit:
                    break
                completion_prev = 0.0
                waiting_times = []
                idle_times = []
                for i in range(P):
                    pid = patient_ids_sorted[i]
                    p = patients[pid]
                    s_time = s[i]
                    mu_q = p['mean_prep_adequacy']
                    q_adequate = 1 if random.random() < mu_q else 0
                    if q_adequate == 1:
                        mu_dur = p['mean_duration_adequate_prep']
                        low_bound = p['lower_bound_duration_adequate_prep']
                        up_bound = p['upper_bound_duration_adequate_prep']
                        p_low_prob = (up_bound - mu_dur) / (up_bound - low_bound)
                        dur = low_bound if random.random() < p_low_prob else up_bound
                    else:
                        mu_dur = p['mean_duration_inadequate_prep']
                        low_bound = p['lower_bound_duration_inadequate_prep']
                        up_bound = p['upper_bound_duration_inadequate_prep']
                        p_low_prob = (up_bound - mu_dur) / (up_bound - low_bound)
                        dur = low_bound if random.random() < p_low_prob else up_bound
                    
                    mu_arr_dev = p['mean_arrival_time_deviation']
                    arr_dev_lb = p['lower_bound_arrival_time_deviation']
                    arr_dev_ub = p['upper_bound_arrival_time_deviation']
                    p_low_arr = (arr_dev_ub - mu_arr_dev) / (arr_dev_ub - arr_dev_lb)
                    arr_dev = arr_dev_lb if random.random() < p_low_arr else arr_dev_ub
                    arrival_time = s_time + arr_dev
                    
                    ready_time = max(arrival_time, completion_prev)
                    waiting_time = max(0, ready_time - s_time)
                    current_completion = ready_time + dur
                    if i > 0:
                        idle_time = max(0, ready_time - completion_prev)
                        idle_times.append(idle_time)
                    
                    completion_prev = current_completion
                    waiting_times.append(waiting_time)
                
                overtime = max(0, current_completion - L)
                cost_value = c_w * sum(waiting_times) + c_g * sum(idle_times) + c_o * overtime
                total_sim_cost += cost_value
                num_sim += 1
        if num_sim > 0:
            objective_value = total_sim_cost / num_sim
        else:
            objective_value = 0.0

    model_variable_name_patterns = {
        "x[p,i]": "patient-position assignment, p and i are 1..P",
        "t[i]": "position start time",
        "rho[i], alpha[i], lambda[i], gamma[i]": "dual variables",
        "beta[i]": "i is 1..P+2",
        "z[i], v[i], r[i], e[i]": "one-dimensional auxiliary variables",
        "eta[p,i], tau[p,i], nu[p,i], Gam[p,i]": "McCormick variables",
        "zeta[p,i], psi[p,i], phi[p,i]": "nonnegative McCormick variables"
    }

    solution_dict = {
        "objective_value": objective_value,
        "model_variables": {},
        "cost_structure": cost_structure_name,
        "c_w": c_w,
        "c_g": c_g,
        "c_o": c_o,
        "assignment": assignment,
        "schedule": schedule,
        "patient_start_times": patient_start_times,
        "model_variable_name_patterns": model_variable_name_patterns
    }

    if logger:
        logger.log_solution(objective_value, solution_dict)

    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f)

if __name__ == "__main__":
    main()