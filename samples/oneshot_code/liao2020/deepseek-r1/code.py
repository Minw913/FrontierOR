import argparse
import json
import time
from solution_logger import SolutionLogger

def main(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    with open(instance_path, 'r') as f:
        instance_data = json.load(f)
    
    cost_structure = next(iter(instance_data['cost_structures'].values()))
    c_w = cost_structure['c_w']
    c_g = cost_structure['c_g']
    c_o = cost_structure['c_o']
    
    patients = instance_data['patients']
    n = instance_data['num_patients_P']
    L = instance_data['provider_service_hours_L_minutes']
    
    if n == 0:
        result = {
            "objective_value": 0.0,
            "assignment": {},
            "schedule": {},
            "patient_start_times": {}
        }
        with open(solution_path, 'w') as f_out:
            json.dump(result, f_out, indent=2)
        return
    
    ED = [0.0] * n
    SD = [0.0] * n
    worst_duration = [0.0] * n
    worst_arrival = [0.0] * n
    lower_bound_min = [0.0] * n
    upper_bound_max = [0.0] * n
    
    for i, patient in enumerate(patients):
        q = patient['mean_prep_adequacy']
        mu_A = patient['mean_duration_adequate_prep']
        mu_I = patient['mean_duration_inadequate_prep']
        d_AL = patient['lower_bound_duration_adequate_prep']
        d_AU = patient['upper_bound_duration_adequate_prep']
        d_IL = patient['lower_bound_duration_inadequate_prep']
        d_IU = patient['upper_bound_duration_inadequate_prep']
        u_L = patient['lower_bound_arrival_time_deviation']
        u_U = patient['upper_bound_arrival_time_deviation']
        
        ED[i] = q * mu_A + (1 - q) * mu_I
        low_bound = min(d_AL, d_IL)
        high_bound = max(d_AU, d_IU)
        SD[i] = (high_bound - low_bound) / 2.0
        worst_duration[i] = q * d_AU + (1 - q) * d_IU
        worst_arrival[i] = u_U
        lower_bound_min[i] = low_bound
        upper_bound_max[i] = high_bound
    
    total_expected_duration = sum(ED)
    total_SD = sum(SD)
    available_buffer = L - total_expected_duration
    
    rules = [
        ('increasing_ED', lambda i: ED[i]),
        ('decreasing_ED', lambda i: -ED[i]),
        ('increasing_worst_duration', lambda i: worst_duration[i]),
        ('decreasing_worst_duration', lambda i: -worst_duration[i]),
        ('increasing_lower_bound_duration', lambda i: lower_bound_min[i]),
        ('decreasing_lower_bound_duration', lambda i: -lower_bound_min[i]),
        ('increasing_upper_bound_duration', lambda i: upper_bound_max[i]),
        ('decreasing_upper_bound_duration', lambda i: -upper_bound_max[i])
    ]
    
    best_sequence_indices = None
    best_start_times = None
    best_cost = float('inf')
    best_rule_name = None
    
    for rule_name, key_func in rules:
        if time.time() - start_time > time_limit:
            break
        
        sorted_indices = sorted(range(n), key=key_func)
        start_times = [0.0] * n
        
        if available_buffer < 0 or total_SD == 0 or n == 1:
            for j in range(1, n):
                start_times[j] = start_times[j-1] + ED[sorted_indices[j-1]]
        else:
            for j in range(1, n):
                buffer_j = available_buffer * (SD[sorted_indices[j-1]] / total_SD)
                start_times[j] = start_times[j-1] + ED[sorted_indices[j-1]] + buffer_j
        
        a = [0.0] * n
        w = [0.0] * n
        c = [0.0] * n
        idle = [0.0] * (n-1) if n > 1 else []
        
        idx0 = sorted_indices[0]
        a0 = max(0.0, start_times[0] + worst_arrival[idx0])
        w0 = max(0.0, a0 - start_times[0])
        c0 = a0 + worst_duration[idx0]
        a[0] = a0
        w[0] = w0
        c[0] = c0
        
        for j in range(1, n):
            idxj = sorted_indices[j]
            a_j = max(c[j-1], start_times[j] + worst_arrival[idxj])
            w_j = max(0.0, a_j - start_times[j])
            a[j] = a_j
            w[j] = w_j
            idle_j_minus1 = max(0.0, start_times[j] - (c[j-1] + w_j))
            if j-1 < len(idle):
                idle[j-1] = idle_j_minus1
            c[j] = a_j + worst_duration[idxj]
        
        overtime = max(0.0, c[n-1] - L)
        total_waiting = sum(w)
        total_idle = sum(idle) if n > 1 else 0.0
        total_cost = c_w * total_waiting + c_g * total_idle + c_o * overtime
        
        if total_cost < best_cost:
            best_sequence_indices = sorted_indices
            best_start_times = start_times
            best_cost = total_cost
            best_rule_name = rule_name
            if logger:
                logger.log(best_cost)
    
    if best_sequence_indices is None:
        best_sequence_indices = list(range(n))
        best_start_times = [0.0] * n
        for j in range(1, n):
            best_start_times[j] = best_start_times[j-1] + ED[j-1]
        best_rule_name = 'default_sequence'
        a = [0.0] * n
        w = [0.0] * n
        c = [0.0] * n
        idle = [0.0] * (n-1) if n > 1 else []
        idx0 = best_sequence_indices[0]
        a0 = max(0.0, best_start_times[0] + worst_arrival[idx0])
        w0 = max(0.0, a0 - best_start_times[0])
        c0 = a0 + worst_duration[idx0]
        a[0] = a0
        w[0] = w0
        c[0] = c0
        for j in range(1, n):
            idxj = best_sequence_indices[j]
            a_j = max(c[j-1], best_start_times[j] + worst_arrival[idxj])
            w_j = max(0.0, a_j - best_start_times[j])
            a[j] = a_j
            w[j] = w_j
            idle_j_minus1 = max(0.0, best_start_times[j] - (c[j-1] + w_j))
            if j-1 < len(idle):
                idle[j-1] = idle_j_minus1
            c[j] = a_j + worst_duration[idxj]
        overtime = max(0.0, c[n-1] - L)
        total_waiting = sum(w)
        total_idle = sum(idle) if n > 1 else 0.0
        best_cost = c_w * total_waiting + c_g * total_idle + c_o * overtime
        if logger:
            logger.log(best_cost)
    
    assignment_dict = {}
    schedule_dict = {}
    patient_start_times_dict = {}
    
    patient_index_to_position = {}
    for pos, idx in enumerate(best_sequence_indices):
        patient_index = patients[idx]['patient_index']
        assignment_dict[str(patient_index)] = pos + 1
        schedule_dict[str(pos + 1)] = best_start_times[pos]
        patient_start_times_dict[str(patient_index)] = best_start_times[pos]
    
    result = {
        "objective_value": best_cost,
        "assignment": assignment_dict,
        "schedule": schedule_dict,
        "patient_start_times": patient_start_times_dict
    }
    
    with open(solution_path, 'w') as f_out:
        json.dump(result, f_out, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    main(args.instance_path, args.solution_path, args.time_limit, args.log_path)