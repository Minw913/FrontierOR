import argparse
import json
import time
import sys
from itertools import combinations
from collections import defaultdict

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    num_patients = data['num_patients']
    num_surgeons = data['num_surgeons']
    num_hospitals = data['num_hospitals']
    ORs_per_hospital = data['ORs_per_hospital']
    num_days = data['num_days']
    alpha = data['alpha']
    
    patients = {p['patient_id']: p for p in data['patients']}
    surgeons = {s['surgeon_id']: s for s in data['surgeons']}
    hospitals = {h['hospital_id']: h for h in data['hospitals']}
    
    # Precompute OR info
    or_info = {}  # (hospital_id, or_id) -> OR data
    for h in data['hospitals']:
        for or_data in h['ORs']:
            or_info[(h['hospital_id'], or_data['or_id'])] = or_data
    
    # Precompute surgeon fixed costs
    surgeon_fixed_costs = {}  # (surgeon_id, hospital_id) -> cost
    for h in data['hospitals']:
        for s_id_str, cost in h['surgeon_fixed_costs_per_day'].items():
            surgeon_fixed_costs[(int(s_id_str), h['hospital_id'])] = cost
    
    # Get OR regular time and max overtime for a specific day
    def get_or_times(hospital_id, or_id, day):
        od = or_info[(hospital_id, or_id)]
        day_str = str(day)
        if 'daily' in od and day_str in od['daily']:
            return od['daily'][day_str]['regular_time'], od['daily'][day_str]['max_overtime']
        return od['regular_time'], od['max_overtime']
    
    def get_or_fixed_cost(hospital_id, or_id):
        od = or_info[(hospital_id, or_id)]
        return od['fixed_cost_per_hour']
    
    def get_or_overtime_cost(hospital_id, or_id):
        od = or_info[(hospital_id, or_id)]
        return od['overtime_cost_per_hour']
    
    # Compute T_ps: total OR occupation time for patient p with surgeon s
    def compute_T_ps(patient, surgeon_id):
        prep = patient['preparation_time']
        surg = patient['surgeon_specific_times'][str(surgeon_id)]
        clean = patient['cleaning_time']
        return prep + surg + clean
    
    # Compute workload contribution: alpha * surg_time + (1 - alpha) * T_ps
    def compute_workload(patient, surgeon_id):
        surg = patient['surgeon_specific_times'][str(surgeon_id)]
        t_ps = compute_T_ps(patient, surgeon_id)
        return alpha * surg + (1 - alpha) * t_ps
    
    # Build possible assignments for each patient
    # An assignment is (patient_id, surgeon_id, hospital_id, day, or_id)
    mandatory_patients = [p for p in data['patients'] if p['is_mandatory']]
    optional_patients = [p for p in data['patients'] if not p['is_mandatory']]
    
    # Use a greedy/heuristic approach given the complexity
    # Strategy: For each mandatory patient, find the best assignment
    # Then try to fit optional patients
    
    # Build surgeon availability
    surgeon_avail = {}  # surgeon_id -> {day: available_minutes}
    for s in data['surgeons']:
        surgeon_avail[s['surgeon_id']] = {}
        for d_str, mins in s['availability_by_day'].items():
            surgeon_avail[s['surgeon_id']][int(d_str)] = mins
    
    # Track state
    # or_usage[hospital_id][day][or_id] = list of (patient_id, surgeon_id, T_ps, surg_time, prep_time, clean_time)
    or_usage = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # or_opened[hospital_id][day][or_id] = True/False
    or_opened = defaultdict(lambda: defaultdict(lambda: defaultdict(bool)))
    # surgeon_day_hospital[surgeon_id][day] = hospital_id or None
    surgeon_day_hospital = defaultdict(lambda: defaultdict(lambda: None))
    # surgeon_day_workload[surgeon_id][day] = total workload
    surgeon_day_workload = defaultdict(lambda: defaultdict(float))
    
    def or_remaining_time(hospital_id, or_id, day):
        reg, maxot = get_or_times(hospital_id, or_id, day)
        total_capacity = reg + maxot
        used = sum(item[2] for item in or_usage[hospital_id][day][or_id])
        return total_capacity - used
    
    def or_used_time(hospital_id, or_id, day):
        return sum(item[2] for item in or_usage[hospital_id][day][or_id])
    
    def surgeon_remaining_workload(surgeon_id, day):
        if day not in surgeon_avail.get(surgeon_id, {}):
            return 0
        return surgeon_avail[surgeon_id][day] - surgeon_day_workload[surgeon_id][day]
    
    def compute_assignment_cost(patient, surgeon_id, hospital_id, day, or_id):
        """Compute incremental cost of assigning this patient."""
        t_ps = compute_T_ps(patient, surgeon_id)
        reg, maxot = get_or_times(hospital_id, or_id, day)
        
        cost = 0.0
        
        # OR opening cost (if not already open)
        if not or_opened[hospital_id][day][or_id]:
            # Fixed cost is per hour of regular time
            fixed_cost_ph = get_or_fixed_cost(hospital_id, or_id)
            cost += fixed_cost_ph * (reg / 60.0)
        
        # Surgeon assignment cost (if surgeon not already at this hospital on this day)
        if surgeon_day_hospital[surgeon_id][day] is None:
            cost += surgeon_fixed_costs.get((surgeon_id, hospital_id), 0)
        
        # Overtime cost change
        current_used = or_used_time(hospital_id, or_id, day)
        new_used = current_used + t_ps
        
        ot_cost_ph = get_or_overtime_cost(hospital_id, or_id)
        old_overtime = max(0, current_used - reg)
        new_overtime = max(0, new_used - reg)
        cost += (new_overtime - old_overtime) * (ot_cost_ph / 60.0)
        
        # Reward for optional patients
        if not patient['is_mandatory']:
            cost -= patient['reward']
        
        return cost
    
    def can_assign(patient, surgeon_id, hospital_id, day, or_id):
        """Check if assignment is feasible."""
        pid = patient['patient_id']
        
        # Check surgeon eligibility
        if surgeon_id not in patient['eligible_surgeons']:
            return False
        
        # Check OR eligibility
        h_str = str(hospital_id)
        if h_str not in patient['eligible_ORs_by_hospital']:
            return False
        if or_id not in patient['eligible_ORs_by_hospital'][h_str]:
            return False
        
        # Check surgeon available on this day
        if day not in surgeon_avail.get(surgeon_id, {}):
            return False
        
        # Check due date for mandatory
        if patient['is_mandatory'] and day > patient['due_date']:
            return False
        
        # Check day is within planning horizon
        if day < 1 or day > num_days:
            return False
        
        # Check surgeon not assigned to different hospital on this day
        if surgeon_day_hospital[surgeon_id][day] is not None and surgeon_day_hospital[surgeon_id][day] != hospital_id:
            return False
        
        # Check OR capacity
        t_ps = compute_T_ps(patient, surgeon_id)
        if or_remaining_time(hospital_id, or_id, day) < t_ps:
            return False
        
        # Check surgeon workload capacity
        wl = compute_workload(patient, surgeon_id)
        if surgeon_remaining_workload(surgeon_id, day) < wl - 1e-6:
            return False
        
        return True
    
    def do_assign(patient, surgeon_id, hospital_id, day, or_id):
        pid = patient['patient_id']
        t_ps = compute_T_ps(patient, surgeon_id)
        surg_time = patient['surgeon_specific_times'][str(surgeon_id)]
        prep_time = patient['preparation_time']
        clean_time = patient['cleaning_time']
        
        or_usage[hospital_id][day][or_id].append((pid, surgeon_id, t_ps, surg_time, prep_time, clean_time))
        or_opened[hospital_id][day][or_id] = True
        
        if surgeon_day_hospital[surgeon_id][day] is None:
            surgeon_day_hospital[surgeon_id][day] = hospital_id
        
        wl = compute_workload(patient, surgeon_id)
        surgeon_day_workload[surgeon_id][day] += wl
    
    # Sort mandatory patients by due date (earliest first), then by number of options (fewest first)
    def count_options(patient):
        count = 0
        for s_id in patient['eligible_surgeons']:
            for d in surgeons[s_id]['operating_days'] if s_id in surgeons else []:
                if patient['is_mandatory'] and d > patient['due_date']:
                    continue
                if d < 1 or d > num_days:
                    continue
                h_str_keys = patient['eligible_ORs_by_hospital'].keys()
                for h_str in h_str_keys:
                    h_id = int(h_str)
                    for or_id in patient['eligible_ORs_by_hospital'][h_str]:
                        count += 1
        return count
    
    mandatory_sorted = sorted(mandatory_patients, key=lambda p: (p['due_date'], count_options(p)))
    
    assignments = {}  # patient_id -> (surgeon_id, hospital_id, day, or_id)
    
    # Greedy assignment for mandatory patients
    for patient in mandatory_sorted:
        if time.time() - start_time > time_limit * 0.8:
            break
        
        best_cost = float('inf')
        best_assign = None
        
        pid = patient['patient_id']
        
        for s_id in patient['eligible_surgeons']:
            s_data = surgeons.get(s_id)
            if s_data is None:
                continue
            
            for d in s_data['operating_days']:
                if patient['is_mandatory'] and d > patient['due_date']:
                    continue
                if d < 1 or d > num_days:
                    continue
                
                for h_str in patient['eligible_ORs_by_hospital'].keys():
                    h_id = int(h_str)
                    
                    for or_id in patient['eligible_ORs_by_hospital'][h_str]:
                        if can_assign(patient, s_id, h_id, d, or_id):
                            cost = compute_assignment_cost(patient, s_id, h_id, d, or_id)
                            if cost < best_cost:
                                best_cost = cost
                                best_assign = (s_id, h_id, d, or_id)
        
        if best_assign is not None:
            s_id, h_id, d, or_id = best_assign
            do_assign(patient, s_id, h_id, d, or_id)
            assignments[pid] = best_assign
    
    # Check all mandatory patients assigned
    unassigned_mandatory = [p for p in mandatory_patients if p['patient_id'] not in assignments]
    if unassigned_mandatory:
        # Try harder for unassigned mandatory patients
        for patient in unassigned_mandatory:
            if time.time() - start_time > time_limit * 0.9:
                break
            pid = patient['patient_id']
            best_cost = float('inf')
            best_assign = None
            
            for s_id in patient['eligible_surgeons']:
                s_data = surgeons.get(s_id)
                if s_data is None:
                    continue
                for d in s_data['operating_days']:
                    if patient['is_mandatory'] and d > patient['due_date']:
                        continue
                    if d < 1 or d > num_days:
                        continue
                    for h_str in patient['eligible_ORs_by_hospital'].keys():
                        h_id = int(h_str)
                        for or_id in patient['eligible_ORs_by_hospital'][h_str]:
                            if can_assign(patient, s_id, h_id, d, or_id):
                                cost = compute_assignment_cost(patient, s_id, h_id, d, or_id)
                                if cost < best_cost:
                                    best_cost = cost
                                    best_assign = (s_id, h_id, d, or_id)
            
            if best_assign is not None:
                s_id, h_id, d, or_id = best_assign
                do_assign(patient, s_id, h_id, d, or_id)
                assignments[pid] = best_assign
    
    # Greedy assignment for optional patients (sorted by reward descending)
    optional_sorted = sorted(optional_patients, key=lambda p: -p['reward'])
    
    for patient in optional_sorted:
        if time.time() - start_time > time_limit * 0.85:
            break
        
        pid = patient['patient_id']
        if patient['reward'] <= 0:
            continue
        
        best_cost = float('inf')
        best_assign = None
        
        for s_id in patient['eligible_surgeons']:
            s_data = surgeons.get(s_id)
            if s_data is None:
                continue
            for d in s_data['operating_days']:
                if d < 1 or d > num_days:
                    continue
                for h_str in patient['eligible_ORs_by_hospital'].keys():
                    h_id = int(h_str)
                    for or_id in patient['eligible_ORs_by_hospital'][h_str]:
                        if can_assign(patient, s_id, h_id, d, or_id):
                            cost = compute_assignment_cost(patient, s_id, h_id, d, or_id)
                            if cost < best_cost:
                                best_cost = cost
                                best_assign = (s_id, h_id, d, or_id)
        
        if best_assign is not None and best_cost < 0:
            # Only assign if it reduces total cost (negative incremental cost)
            s_id, h_id, d, or_id = best_assign
            do_assign(patient, s_id, h_id, d, or_id)
            assignments[pid] = best_assign
    
    # Now build the detailed schedule with timing
    # For each (hospital, day, or_id), schedule patients sequentially
    
    result_assignments = []
    result_opened_ors = []
    result_surgeon_assignments = []
    result_or_sequences = []
    result_surgeon_sequences = []
    
    # Track surgeon start/end times
    surgeon_times = defaultdict(lambda: defaultdict(lambda: {'start': float('inf'), 'end': 0.0}))
    
    # Process each OR
    total_or_fixed_cost = 0.0
    total_overtime_cost = 0.0
    total_surgeon_cost = 0.0
    total_reward = 0.0
    
    opened_ors_set = set()
    surgeon_hospital_day_set = set()
    
    for h_id in or_usage:
        for day in or_usage[h_id]:
            for or_id in or_usage[h_id][day]:
                cases = or_usage[h_id][day][or_id]
                if not cases:
                    continue
                
                opened_ors_set.add((h_id, day, or_id))
                
                # Schedule cases sequentially in the OR
                current_time = 0.0
                ordered_cases = list(cases)  # Keep the order they were added
                
                for i, (pid, s_id, t_ps, surg_time, prep_time, clean_time) in enumerate(ordered_cases):
                    patient = patients[pid]
                    
                    room_entry_time = current_time
                    surgery_start_time = room_entry_time + prep_time
                    finish_time = surgery_start_time + surg_time
                    room_exit_time = finish_time + clean_time
                    
                    result_assignments.append({
                        "patient_id": pid,
                        "surgeon_id": s_id,
                        "hospital_id": h_id,
                        "day": day,
                        "or_id": or_id,
                        "is_mandatory": patient['is_mandatory'],
                        "T_ps": t_ps,
                        "finish_time": finish_time,
                        "surgery_start_time": surgery_start_time,
                        "room_entry_time": room_entry_time,
                        "room_exit_time": room_exit_time
                    })
                    
                    # Update surgeon times
                    surgeon_times[(s_id, h_id, day)]['start'] = min(
                        surgeon_times[(s_id, h_id, day)]['start'], surgery_start_time)
                    surgeon_times[(s_id, h_id, day)]['end'] = max(
                        surgeon_times[(s_id, h_id, day)]['end'], finish_time)
                    
                    surgeon_hospital_day_set.add((s_id, h_id, day))
                    
                    if not patient['is_mandatory']:
                        total_reward += patient['reward']
                    
                    current_time = room_exit_time
                
                # OR completion time and overtime
                completion_time = current_time
                reg, maxot = get_or_times(h_id, or_id, day)
                overtime = max(0.0, completion_time - reg)
                overtime = min(overtime, maxot)
                
                result_opened_ors.append({
                    "hospital_id": h_id,
                    "day": day,
                    "or_id": or_id,
                    "regular_time": reg,
                    "completion_time": completion_time,
                    "overtime": overtime
                })
                
                # OR fixed cost
                fixed_cost_ph = get_or_fixed_cost(h_id, or_id)
                total_or_fixed_cost += fixed_cost_ph * (reg / 60.0)
                
                # Overtime cost
                ot_cost_ph = get_or_overtime_cost(h_id, or_id)
                total_overtime_cost += overtime * (ot_cost_ph / 60.0)
                
                # OR sequences
                for i in range(len(ordered_cases)):
                    for j in range(i + 1, len(ordered_cases)):
                        p_id = ordered_cases[i][0]
                        k_id = ordered_cases[j][0]
                        # i comes before j, so k is after p => p_after_k = 0 for (p,k)
                        result_or_sequences.append({
                            "hospital_id": h_id,
                            "day": day,
                            "or_id": or_id,
                            "patient_p": p_id,
                            "patient_k": k_id,
                            "p_after_k": 0
                        })
    
    # Now we need to handle surgeon sequences properly
    # We need to check timing constraints for surgeon sharing across ORs
    # First, collect all assignments by (surgeon, hospital, day)
    surgeon_day_cases = defaultdict(list)
    for a in result_assignments:
        key = (a['surgeon_id'], a['hospital_id'], a['day'])
        surgeon_day_cases[key].append(a)
    
    # Re-schedule considering surgeon conflicts
    # For each (hospital, day), group all cases by OR and by surgeon
    # We need to ensure no surgeon overlap
    
    # Let's do a proper sequential scheduling
    # Reset and redo timing
    
    # Group assignments by (hospital, day)
    hd_cases = defaultdict(list)
    for a in result_assignments:
        hd_cases[(a['hospital_id'], a['day'])].append(a)
    
    # For each (hospital, day), schedule all ORs considering surgeon constraints
    new_assignments = []
    new_opened_ors = []
    new_or_sequences = []
    new_surgeon_sequences = []
    surgeon_times_new = defaultdict(lambda: {'start': float('inf'), 'end': 0.0})
    
    total_or_fixed_cost = 0.0
    total_overtime_cost = 0.0
    total_reward = 0.0
    
    for (h_id, day), cases in hd_cases.items():
        # Group by OR
        or_cases = defaultdict(list)
        for c in cases:
            or_cases[c['or_id']].append(c)
        
        # Group by surgeon
        surg_cases = defaultdict(list)
        for c in cases:
            surg_cases[c['surgeon_id']].append(c)
        
        # Simple scheduling: for each OR, schedule cases sequentially
        # But respect surgeon availability (surgeon can only do one case at a time)
        
        # Track surgeon availability (next free time for each surgeon)
        surgeon_free_time = defaultdict(float)
        
        # Schedule OR by OR
        all_scheduled = []
        
        for or_id in or_cases:
            current_or_time = 0.0
            or_case_list = or_cases[or_id]
            
            for c in or_case_list:
                pid = c['patient_id']
                s_id = c['surgeon_id']
                patient = patients[pid]
                prep_time = patient['preparation_time']
                surg_time = patient['surgeon_specific_times'][str(s_id)]
                clean_time = patient['cleaning_time']
                t_ps = prep_time + surg_time + clean_time
                
                # Patient can enter OR at current_or_time
                # Surgeon must be free by the time surgery starts
                room_entry_time = max(current_or_time, surgeon_free_time[s_id] - prep_time)
                # Ensure room_entry_time >= 0
                room_entry_time = max(0, room_entry_time)
                # Ensure surgeon is free when surgery starts
                surgery_start_time = room_entry_time + prep_time
                if surgery_start_time < surgeon_free_time[s_id]:
                    surgery_start_time = surgeon_free_time[s_id]
                    room_entry_time = surgery_start_time - prep_time
                    if room_entry_time < current_or_time:
                        room_entry_time = current_or_time
                        surgery_start_time = room_entry_time + prep_time
                        # If surgeon still not free, wait
                        if surgery_start_time < surgeon_free_time[s_id]:
                            # Need to delay room entry
                            surgery_start_time = surgeon_free_time[s_id]
                            room_entry_time = surgery_start_time - prep_time
                
                finish_time = surgery_start_time + surg_time
                room_exit_time = finish_time + clean_time
                
                scheduled_case = {
                    "patient_id": pid,
                    "surgeon_id": s_id,
                    "hospital_id": h_id,
                    "day": day,
                    "or_id": or_id,
                    "is_mandatory": patient['is_mandatory'],
                    "T_ps": t_ps,
                    "finish_time": finish_time,
                    "surgery_start_time": surgery_start_time,
                    "room_entry_time": room_entry_time,
                    "room_exit_time": room_exit_time
                }
                
                all_scheduled.append(scheduled_case)
                
                surgeon_free_time[s_id] = finish_time
                current_or_time = room_exit_time
                
                # Update surgeon times
                key = (s_id, h_id, day)
                surgeon_times_new[key]['start'] = min(surgeon_times_new[key]['start'], surgery_start_time)
                surgeon_times_new[key]['end'] = max(surgeon_times_new[key]['end'], finish_time)
                
                if not patient['is_mandatory']:
                    total_reward += patient['reward']
        
        new_assignments.extend(all_scheduled)
        
        # OR completion and costs
        for or_id in or_cases:
            or_scheduled = [c for c in all_scheduled if c['or_id'] == or_id]
            if not or_scheduled:
                continue
            
            completion_time = max(c['room_exit_time'] for c in or_scheduled)
            reg, maxot = get_or_times(h_id, or_id, day)
            overtime = max(0.0, completion_time - reg)
            overtime = min(overtime, maxot)
            
            new_opened_ors.append({
                "hospital_id": h_id,
                "day": day,
                "or_id": or_id,
                "regular_time": reg,
                "completion_time": completion_time,
                "overtime": overtime
            })
            
            fixed_cost_ph = get_or_fixed_cost(h_id, or_id)
            total_or_fixed_cost += fixed_cost_ph * (reg / 60.0)
            
            ot_cost_ph = get_or_overtime_cost(h_id, or_id)
            total_overtime_cost += overtime * (ot_cost_ph / 60.0)
            
            # OR sequences
            or_patients = [c['patient_id'] for c in or_scheduled]
            for i in range(len(or_patients)):
                for j in range(i + 1, len(or_patients)):
                    new_or_sequences.append({
                        "hospital_id": h_id,
                        "day": day,
                        "or_id": or_id,
                        "patient_p": or_patients[i],
                        "patient_k": or_patients[j],
                        "p_after_k": 0  # i before j
                    })
        
        # Surgeon sequences
        for s_id in surg_cases:
            s_scheduled = [c for c in all_scheduled if c['surgeon_id'] == s_id]
            # Sort by surgery start time
            s_scheduled.sort(key=lambda c: c['surgery_start_time'])
            s_patients = [c['patient_id'] for c in s_scheduled]
            for i in range(len(s_patients)):
                for j in range(i + 1, len(s_patients)):
                    # Check if they are in different ORs (same OR already covered)
                    or_i = next(c['or_id'] for c in s_scheduled if c['patient_id'] == s_patients[i])
                    or_j = next(c['or_id'] for c in s_scheduled if c['patient_id'] == s_patients[j])
                    new_surgeon_sequences.append({
                        "hospital_id": h_id,
                        "day": day,
                        "surgeon_id": s_id,
                        "patient_p": s_patients[i],
                        "patient_k": s_patients[j],
                        "p_after_k": 0  # i before j
                    })
    
    # Surgeon assignment costs
    for (s_id, h_id, day) in surgeon_hospital_day_set:
        total_surgeon_cost += surgeon_fixed_costs.get((s_id, h_id), 0)
    
    # Build surgeon assignments
    result_surgeon_assignments = []
    for key, times in surgeon_times_new.items():
        s_id, h_id, day = key
        if times['start'] == float('inf'):
            continue
        result_surgeon_assignments.append({
            "surgeon_id": s_id,
            "hospital_id": h_id,
            "day": day,
            "start_time": times['start'],
            "end_time": times['end']
        })
    
    objective_value = total_or_fixed_cost + total_surgeon_cost + total_overtime_cost - total_reward
    
    if logger:
        logger.log(objective_value)
    
    solution = {
        "objective_value": objective_value,
        "assignments": new_assignments,
        "opened_ors": new_opened_ors,
        "surgeon_assignments": result_surgeon_assignments,
        "or_sequences": new_or_sequences,
        "surgeon_sequences": new_surgeon_sequences
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()