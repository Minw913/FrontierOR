import json
import math
import random
import argparse
import time

def norm_ppf(p):
    if p <= 0 or p >= 1:
        return None
    p_low = 0.02425
    p_high = 0.97575
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((-7.784894002430293e-03 * q + -3.223964580411365e-01) * q + -2.400758277161838e+00) * q + -2.549732539343734e+00) * q + 4.374664141464968e+00) * q + 2.938163982698783e+00) / ((((7.784695709041462e-03 * q + 3.224671290700398e-01) * q + 2.445134137142996e+00) * q + 3.754408661907416e+00) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((((-4.14045315348448e-10 * r + 3.220865476912e-08) * r + -1.131782319e-06) * r + 1.9768423e-05) * r + -0.000196639) * r + 0.008516) * r + -0.020749) * r + 0.025) * q
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        return -((((((-7.784894002430293e-03 * q + -3.223964580411365e-01) * q + -2.400758277161838e+00) * q + -2.549732539343734e+00) * q + 4.374664141464968e+00) * q + 2.938163982698783e+00) / ((((7.784695709041462e-03 * q + 3.224671290700398e-01) * q + 2.445134137142996e+00) * q + 3.754408661907416e+00) * q + 1.0))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=str, required=True)
    parser.add_argument('--time_limit', type=float, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    start_time = time.time()
    with open(args.instance, 'r') as f:
        instance = json.load(f)

    cost_structure = next(iter(instance['cost_structures'].values()))
    c_w = cost_structure['c_w']
    c_g = cost_structure['c_g']
    c_o = cost_structure['c_o']
    L = instance['session_length']
    patients = instance['patients']

    mean_durations = []
    for p in patients.values():
        mu_q = p['mean_prep_adequacy']
        mu_A = p['mean_duration_adequate_prep']
        mu_I = p['mean_duration_inadequate_prep']
        mean_dur = mu_q * mu_A + (1 - mu_q) * mu_I
        mean_durations.append(mean_dur)

    sorted_patients = sorted(patients.items(), key=lambda x: mean_durations[int(x[0])], reverse=True)
    patient_ids_sorted = [pid for pid, _ in sorted_patients]

    s = [0.0]
    buffers = []
    for i in range(len(patient_ids_sorted) - 1):
        pid = patient_ids_sorted[i]
        p = patients[pid]
        mu_q = p['mean_prep_adequacy']
        mu_A = p['mean_duration_adequate_prep']
        mu_I = p['mean_duration_inadequate_prep']
        d_AL = p['lower_bound_duration_adequate_prep']
        d_AU = p['upper_bound_duration_adequate_prep']
        d_IL = p['lower_bound_duration_inadequate_prep']
        d_IU = p['upper_bound_duration_inadequate_prep']
        
        var_A = (mu_A - d_AL) * (d_AU - mu_A)
        var_I = (mu_I - d_IL) * (d_IU - mu_I)
        mean_dur = mu_q * mu_A + (1 - mu_q) * mu_I
        var_dur = mu_q * (var_A + (mu_A - mean_dur)**2) + (1 - mu_q) * (var_I + (mu_I - mean_dur)**2)
        
        next_pid = patient_ids_sorted[i+1]
        next_p = patients[next_pid]
        u_L = next_p['lower_bound_arrival_time_deviation']
        u_U = next_p['upper_bound_arrival_time_deviation']
        mu_u = next_p['mean_arrival_time_deviation']
        var_arr = (mu_u - u_L) * (u_U - mu_u)
        
        total_var = var_dur + var_arr
        critical_ratio = c_g / (c_g + c_w)
        z = norm_ppf(critical_ratio)
        buffer_i = z * math.sqrt(total_var) if total_var > 0 else 0
        buffers.append(buffer_i)
        s_next = s[-1] + mean_dur + buffer_i
        s.append(s_next)
    
    total_buffer = sum(buffers)
    total_mean_duration = sum(mean_durations)
    if total_mean_duration + total_buffer > L:
        available = L - total_mean_duration
        factor = available / total_buffer if total_buffer > 0 else 0
        buffers = [b * factor for b in buffers]
        s = [0.0]
        for i in range(len(patient_ids_sorted) - 1):
            pid = patient_ids_sorted[i]
            p = patients[pid]
            mu_q = p['mean_prep_adequacy']
            mu_A = p['mean_duration_adequate_prep']
            mu_I = p['mean_duration_inadequate_prep']
            mean_dur = mu_q * mu_A + (1 - mu_q) * mu_I
            s_next = s[-1] + mean_dur + buffers[i]
            s.append(s_next)
    
    assignment = {}
    patient_start_times = {}
    schedule = {}
    for pos, pid in enumerate(patient_ids_sorted, 1):
        assignment[pid] = str(pos)
        patient_start_times[pid] = s[pos-1]
        schedule[str(pos)] = s[pos-1]
    
    N = 1000
    cost_sum = 0.0
    count = 0
    random.seed(0)
    for _ in range(N):
        completion_prev = 0.0
        waiting_times = []
        idle_times = []
        for i, pid in enumerate(patient_ids_sorted):
            p = patients[pid]
            s_i = s[i]
            mu_q = p['mean_prep_adequacy']
            q = 1 if random.random() < mu_q else 0
            if q == 1:
                d_AL = p['lower_bound_duration_adequate_prep']
                d_AU = p['upper_bound_duration_adequate_prep']
                mu_A = p['mean_duration_adequate_prep']
                p_low = (d_AU - mu_A) / (d_AU - d_AL)
                duration = d_AL if random.random() < p_low else d_AU
            else:
                d_IL = p['lower_bound_duration_inadequate_prep']
                d_IU = p['upper_bound_duration_inadequate_prep']
                mu_I = p['mean_duration_inadequate_prep']
                p_low = (d_IU - mu_I) / (d_IU - d_IL)
                duration = d_IL if random.random() < p_low else d_IU
            
            u_L = p['lower_bound_arrival_time_deviation']
            u_U = p['upper_bound_arrival_time_deviation']
            mu_u = p['mean_arrival_time_deviation']
            p_low_arr = (u_U - mu_u) / (u_U - u_L)
            arrival_dev = u_L if random.random() < p_low_arr else u_U
            arrival_time = s_i + arrival_dev
            
            ready_time = max(arrival_time, completion_prev)
            waiting_time = max(0, ready_time - s_i)
            completion_time = ready_time + duration
            
            if i > 0:
                idle_time = max(0, ready_time - completion_prev)
                idle_times.append(idle_time)
            
            completion_prev = completion_time
            waiting_times.append(waiting_time)
        
        overtime = max(0, completion_prev - L)
        total_cost = c_w * sum(waiting_times) + c_g * sum(idle_times) + c_o * overtime
        cost_sum += total_cost
        count += 1
    
    objective_value = cost_sum / count if count > 0 else 0.0
    
    solution = {
        "assignment": assignment,
        "schedule": schedule,
        "patient_start_times": patient_start_times,
        "cost_structure_name": "Cost1",
        "objective_value": objective_value,
        "model_variables": {}
    }
    
    with open(args.output, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()