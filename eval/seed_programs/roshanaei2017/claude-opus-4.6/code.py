import argparse
import json
import time
import sys
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

    patients = {}
    for p in data['patients']:
        patients[p['patient_id']] = p

    surgeons = {}
    for s in data['surgeons']:
        surgeons[s['surgeon_id']] = s

    hospitals = {}
    for h in data['hospitals']:
        hospitals[h['hospital_id']] = h

    # Precompute OR info: (hospital_id, or_id) -> OR data
    or_info = {}
    for h in data['hospitals']:
        for or_data in h['ORs']:
            or_info[(h['hospital_id'], or_data['or_id'])] = or_data

    # Precompute surgeon fixed costs: (surgeon_id, hospital_id) -> cost
    surgeon_fixed_costs = {}
    for h in data['hospitals']:
        for s_id_str, cost in h.get('surgeon_fixed_costs_per_day', {}).items():
            surgeon_fixed_costs[(int(s_id_str), h['hospital_id'])] = cost

    def get_or_times(hospital_id, or_id, day):
        od = or_info[(hospital_id, or_id)]
        day_str = str(day)
        if 'daily' in od and day_str in od['daily']:
            return od['daily'][day_str]['regular_time'], od['daily'][day_str]['max_overtime']
        return od['regular_time'], od['max_overtime']

    def get_or_fixed_cost(hospital_id, or_id):
        return or_info[(hospital_id, or_id)]['fixed_cost_per_hour']

    def get_or_overtime_cost(hospital_id, or_id):
        return or_info[(hospital_id, or_id)]['overtime_cost_per_hour']

    def compute_T_ps(patient, surgeon_id):
        prep = patient['preparation_time']
        surg = patient['surgeon_specific_times'][str(surgeon_id)]
        clean = patient['cleaning_time']
        return prep + surg + clean

    def compute_workload(patient, surgeon_id):
        surg = patient['surgeon_specific_times'][str(surgeon_id)]
        t_ps = compute_T_ps(patient, surgeon_id)
        return alpha * surg + (1 - alpha) * t_ps

    # Build surgeon availability: surgeon_id -> {day(int): available_minutes}
    surgeon_avail = {}
    for s in data['surgeons']:
        sid = s['surgeon_id']
        surgeon_avail[sid] = {}
        for d_str, mins in s.get('availability_by_day', {}).items():
            surgeon_avail[sid][int(d_str)] = mins

    # surgeon operating days
    surgeon_op_days = {}
    for s in data['surgeons']:
        sid = s['surgeon_id']
        surgeon_op_days[sid] = list(s.get('operating_days', []))

    mandatory_patients = [p for p in data['patients'] if p['is_mandatory']]
    optional_patients = [p for p in data['patients'] if not p['is_mandatory']]

    # State tracking
    or_usage = {}  # (h_id, day, or_id) -> list of (pid, s_id, t_ps, surg_time, prep_time, clean_time)
    or_opened_set = set()
    surgeon_day_hospital = {}  # (surgeon_id, day) -> hospital_id or None
    surgeon_day_workload = {}  # (surgeon_id, day) -> float

    def get_or_usage(h_id, day, or_id):
        key = (h_id, day, or_id)
        if key not in or_usage:
            or_usage[key] = []
        return or_usage[key]

    def or_used_time(h_id, or_id, day):
        return sum(item[2] for item in get_or_usage(h_id, day, or_id))

    def or_remaining_time(h_id, or_id, day):
        reg, maxot = get_or_times(h_id, or_id, day)
        total_capacity = reg + maxot
        used = or_used_time(h_id, or_id, day)
        return total_capacity - used

    def surgeon_remaining_workload(surgeon_id, day):
        if day not in surgeon_avail.get(surgeon_id, {}):
            return 0
        avail = surgeon_avail[surgeon_id][day]
        used = surgeon_day_workload.get((surgeon_id, day), 0.0)
        return avail - used

    def can_assign(patient, surgeon_id, hospital_id, day, or_id):
        pid = patient['patient_id']

        if surgeon_id not in patient['eligible_surgeons']:
            return False

        h_str = str(hospital_id)
        if h_str not in patient.get('eligible_ORs_by_hospital', {}):
            return False
        if or_id not in patient['eligible_ORs_by_hospital'][h_str]:
            return False

        if day not in surgeon_avail.get(surgeon_id, {}):
            return False

        if patient['is_mandatory'] and day > patient['due_date']:
            return False

        if day < 1 or day > num_days:
            return False

        sdh = surgeon_day_hospital.get((surgeon_id, day), None)
        if sdh is not None and sdh != hospital_id:
            return False

        t_ps = compute_T_ps(patient, surgeon_id)
        if or_remaining_time(hospital_id, or_id, day) < t_ps - 1e-9:
            return False

        wl = compute_workload(patient, surgeon_id)
        if surgeon_remaining_workload(surgeon_id, day) < wl - 1e-9:
            return False

        return True

    def compute_assignment_cost(patient, surgeon_id, hospital_id, day, or_id):
        t_ps = compute_T_ps(patient, surgeon_id)
        reg, maxot = get_or_times(hospital_id, or_id, day)

        cost = 0.0

        # OR opening cost
        if (hospital_id, day, or_id) not in or_opened_set:
            fixed_cost_ph = get_or_fixed_cost(hospital_id, or_id)
            cost += fixed_cost_ph * (reg / 60.0)

        # Surgeon assignment cost
        if surgeon_day_hospital.get((surgeon_id, day), None) is None:
            cost += surgeon_fixed_costs.get((surgeon_id, hospital_id), 0)

        # Overtime cost change
        current_used = or_used_time(hospital_id, or_id, day)
        new_used = current_used + t_ps

        ot_cost_ph = get_or_overtime_cost(hospital_id, or_id)
        old_overtime = max(0.0, current_used - reg)
        new_overtime = max(0.0, new_used - reg)
        cost += (new_overtime - old_overtime) * (ot_cost_ph / 60.0)

        # Reward for optional patients
        if not patient['is_mandatory']:
            cost -= patient.get('reward', 0)

        return cost

    def do_assign(patient, surgeon_id, hospital_id, day, or_id):
        pid = patient['patient_id']
        t_ps = compute_T_ps(patient, surgeon_id)
        surg_time = patient['surgeon_specific_times'][str(surgeon_id)]
        prep_time = patient['preparation_time']
        clean_time = patient['cleaning_time']

        key = (hospital_id, day, or_id)
        if key not in or_usage:
            or_usage[key] = []
        or_usage[key].append((pid, surgeon_id, t_ps, surg_time, prep_time, clean_time))
        or_opened_set.add(key)

        if surgeon_day_hospital.get((surgeon_id, day), None) is None:
            surgeon_day_hospital[(surgeon_id, day)] = hospital_id

        wl = compute_workload(patient, surgeon_id)
        surgeon_day_workload[(surgeon_id, day)] = surgeon_day_workload.get((surgeon_id, day), 0.0) + wl

    # Count options for sorting
    def count_options(patient):
        count = 0
        for s_id in patient['eligible_surgeons']:
            if s_id not in surgeons:
                continue
            for d in surgeon_op_days.get(s_id, []):
                if patient['is_mandatory'] and d > patient['due_date']:
                    continue
                if d < 1 or d > num_days:
                    continue
                for h_str in patient.get('eligible_ORs_by_hospital', {}).keys():
                    count += len(patient['eligible_ORs_by_hospital'][h_str])
        return count

    mandatory_sorted = sorted(mandatory_patients, key=lambda p: (p['due_date'], count_options(p)))

    assignments = {}  # patient_id -> (surgeon_id, hospital_id, day, or_id)

    # Greedy assignment for mandatory patients
    for patient in mandatory_sorted:
        if time.time() - start_time > time_limit * 0.7:
            break

        pid = patient['patient_id']
        best_cost = float('inf')
        best_assign = None

        for s_id in patient['eligible_surgeons']:
            if s_id not in surgeons:
                continue
            for d in surgeon_op_days.get(s_id, []):
                if patient['is_mandatory'] and d > patient['due_date']:
                    continue
                if d < 1 or d > num_days:
                    continue
                for h_str in patient.get('eligible_ORs_by_hospital', {}).keys():
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

    # Try again for unassigned mandatory
    unassigned_mandatory = [p for p in mandatory_patients if p['patient_id'] not in assignments]
    for patient in unassigned_mandatory:
        if time.time() - start_time > time_limit * 0.85:
            break
        pid = patient['patient_id']
        best_cost = float('inf')
        best_assign = None

        for s_id in patient['eligible_surgeons']:
            if s_id not in surgeons:
                continue
            for d in surgeon_op_days.get(s_id, []):
                if patient['is_mandatory'] and d > patient['due_date']:
                    continue
                if d < 1 or d > num_days:
                    continue
                for h_str in patient.get('eligible_ORs_by_hospital', {}).keys():
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

    # Greedy for optional patients
    optional_sorted = sorted(optional_patients, key=lambda p: -p.get('reward', 0))
    for patient in optional_sorted:
        if time.time() - start_time > time_limit * 0.85:
            break
        pid = patient['patient_id']
        reward = patient.get('reward', 0)
        if reward <= 0:
            continue

        best_cost = float('inf')
        best_assign = None

        for s_id in patient['eligible_surgeons']:
            if s_id not in surgeons:
                continue
            for d in surgeon_op_days.get(s_id, []):
                if d < 1 or d > num_days:
                    continue
                for h_str in patient.get('eligible_ORs_by_hospital', {}).keys():
                    h_id = int(h_str)
                    for or_id in patient['eligible_ORs_by_hospital'][h_str]:
                        if can_assign(patient, s_id, h_id, d, or_id):
                            cost = compute_assignment_cost(patient, s_id, h_id, d, or_id)
                            if cost < best_cost:
                                best_cost = cost
                                best_assign = (s_id, h_id, d, or_id)

        if best_assign is not None and best_cost < 0:
            s_id, h_id, d, or_id = best_assign
            do_assign(patient, s_id, h_id, d, or_id)
            assignments[pid] = best_assign

    # Build detailed schedule with timing
    # Group by (hospital, day)
    hd_or_cases = defaultdict(lambda: defaultdict(list))
    for (h_id, day, or_id), cases in or_usage.items():
        for case in cases:
            hd_or_cases[(h_id, day)][or_id].append(case)

    result_assignments = []
    result_opened_ors = []
    result_surgeon_assignments_list = []
    result_or_sequences = []
    result_surgeon_sequences = []

    surgeon_times = {}  # (s_id, h_id, day) -> {'start': ..., 'end': ...}
    surgeon_hospital_day_set = set()

    total_or_fixed_cost = 0.0
    total_overtime_cost = 0.0
    total_surgeon_cost = 0.0
    total_reward = 0.0

    for (h_id, day), or_dict in hd_or_cases.items():
        # Track surgeon free time across ORs on this hospital-day
        surgeon_free = defaultdict(float)

        all_scheduled_this_hd = []

        for or_id, cases in or_dict.items():
            current_or_time = 0.0
            or_scheduled = []

            for (pid, s_id, t_ps, surg_time, prep_time, clean_time) in cases:
                patient = patients[pid]

                # Room entry: max of current OR time and (surgeon free - prep)
                # But room entry can't be before current_or_time
                room_entry_time = current_or_time
                surgery_start_time = room_entry_time + prep_time

                # Surgeon must be free by surgery start
                if surgery_start_time < surgeon_free[s_id]:
                    surgery_start_time = surgeon_free[s_id]
                    room_entry_time = surgery_start_time - prep_time
                    if room_entry_time < current_or_time:
                        room_entry_time = current_or_time
                        surgery_start_time = room_entry_time + prep_time
                        if surgery_start_time < surgeon_free[s_id]:
                            surgery_start_time = surgeon_free[s_id]
                            room_entry_time = surgery_start_time - prep_time

                finish_time = surgery_start_time + surg_time
                room_exit_time = finish_time + clean_time

                sc = {
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

                or_scheduled.append(sc)
                all_scheduled_this_hd.append(sc)

                surgeon_free[s_id] = finish_time
                current_or_time = room_exit_time

                key = (s_id, h_id, day)
                if key not in surgeon_times:
                    surgeon_times[key] = {'start': float('inf'), 'end': 0.0}
                surgeon_times[key]['start'] = min(surgeon_times[key]['start'], surgery_start_time)
                surgeon_times[key]['end'] = max(surgeon_times[key]['end'], finish_time)
                surgeon_hospital_day_set.add(key)

                if not patient['is_mandatory']:
                    total_reward += patient.get('reward', 0)

            result_assignments.extend(or_scheduled)

            # OR info
            if or_scheduled:
                completion_time = max(c['room_exit_time'] for c in or_scheduled)
            else:
                completion_time = 0.0

            reg, maxot = get_or_times(h_id, or_id, day)
            overtime = max(0.0, completion_time - reg)
            overtime = min(overtime, float(maxot))

            result_opened_ors.append({
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
            or_pids = [c['patient_id'] for c in or_scheduled]
            for i in range(len(or_pids)):
                for j in range(i + 1, len(or_pids)):
                    result_or_sequences.append({
                        "hospital_id": h_id,
                        "day": day,
                        "or_id": or_id,
                        "patient_p": or_pids[i],
                        "patient_k": or_pids[j],
                        "p_after_k": 0
                    })

        # Surgeon sequences for this hospital-day
        surg_cases_hd = defaultdict(list)
        for sc in all_scheduled_this_hd:
            surg_cases_hd[sc['surgeon_id']].append(sc)

        for s_id, s_cases in surg_cases_hd.items():
            s_cases_sorted = sorted(s_cases, key=lambda c: c['surgery_start_time'])
            for i in range(len(s_cases_sorted)):
                for j in range(i + 1, len(s_cases_sorted)):
                    result_surgeon_sequences.append({
                        "hospital_id": h_id,
                        "day": day,
                        "surgeon_id": s_id,
                        "patient_p": s_cases_sorted[i]['patient_id'],
                        "patient_k": s_cases_sorted[j]['patient_id'],
                        "p_after_k": 0
                    })

    # Surgeon assignment costs
    for (s_id, h_id, day) in surgeon_hospital_day_set:
        total_surgeon_cost += surgeon_fixed_costs.get((s_id, h_id), 0)

    # Build surgeon assignments output
    for key, times in surgeon_times.items():
        s_id, h_id, day = key
        if times['start'] == float('inf'):
            continue
        result_surgeon_assignments_list.append({
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
        "assignments": result_assignments,
        "opened_ors": result_opened_ors,
        "surgeon_assignments": result_surgeon_assignments_list,
        "or_sequences": result_or_sequences,
        "surgeon_sequences": result_surgeon_sequences
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