import json
import argparse
import sys
import time as time_module
from gurobipy import *
from solution_logger import SolutionLogger


def get_surg_time(patient, surgeon_id):
    """Get surgical procedure time for patient with given surgeon."""
    st = patient.get('surgeon_specific_times', {})
    return float(st.get(str(surgeon_id), patient.get('base_surgical_time', 0)))


def get_total_or_time(patient, surgeon_id):
    """Total OR occupation time: prep + surgery + cleaning."""
    return patient['preparation_time'] + get_surg_time(patient, surgeon_id) + patient['cleaning_time']


def build_solution_from_model(model, x, y, z, ot, f, st, w, v, patients, surgeons, hospitals,
                              hospital_ids, days, or_data, feasible, patient_ids):
    """Extract solution dictionary from solved Gurobi model."""
    solution = {
        "objective_value": model.ObjVal,
        "assignments": [],
        "opened_ors": [],
        "surgeon_assignments": [],
        "or_sequences": [],
        "surgeon_sequences": []
    }

    # Extract assignments
    assigned_patients = {}  # p_id -> (s_id, h_id, d, o_id)
    for p_id in patient_ids:
        for (s_id, h_id, d, o_id) in feasible[p_id]:
            if (p_id, s_id, h_id, d, o_id) in x and x[p_id, s_id, h_id, d, o_id].X > 0.5:
                assigned_patients[p_id] = (s_id, h_id, d, o_id)
                break

    for p_id, (s_id, h_id, d, o_id) in assigned_patients.items():
        p = patients[p_id]
        surg_time = get_surg_time(p, s_id)
        finish_time = f[p_id].X
        surgery_start = finish_time - surg_time
        room_entry = surgery_start - p['preparation_time']
        room_exit = finish_time + p['cleaning_time']
        total_or_time = get_total_or_time(p, s_id)

        solution["assignments"].append({
            "patient_id": p_id,
            "surgeon_id": s_id,
            "hospital_id": h_id,
            "day": d,
            "or_id": o_id,
            "is_mandatory": p['is_mandatory'],
            "T_ps": total_or_time,
            "finish_time": round(finish_time, 6),
            "surgery_start_time": round(surgery_start, 6),
            "room_entry_time": round(room_entry, 6),
            "room_exit_time": round(room_exit, 6)
        })

    # Extract opened ORs
    for (h_id, d, o_id), var in y.items():
        if var.X > 0.5:
            data = or_data[(h_id, d, o_id)]
            # Compute completion time
            completion = 0.0
            for p_id, (s_id, hh, dd, oo) in assigned_patients.items():
                if hh == h_id and dd == d and oo == o_id:
                    room_exit = f[p_id].X + patients[p_id]['cleaning_time']
                    completion = max(completion, room_exit)

            overtime_val = ot[h_id, d, o_id].X if (h_id, d, o_id) in ot else 0.0
            solution["opened_ors"].append({
                "hospital_id": h_id,
                "day": d,
                "or_id": o_id,
                "regular_time": data['regular_time'],
                "completion_time": round(completion, 6),
                "overtime": round(max(0, overtime_val), 6)
            })

    # Extract surgeon assignments
    for (s_id, h_id, d), var in z.items():
        if var.X > 0.5:
            # Compute start and end times
            start_time = float('inf')
            end_time = 0.0
            for p_id, (ss, hh, dd, oo) in assigned_patients.items():
                if ss == s_id and hh == h_id and dd == d:
                    surg_time = get_surg_time(patients[p_id], s_id)
                    s_start = f[p_id].X - surg_time
                    s_end = f[p_id].X
                    start_time = min(start_time, s_start)
                    end_time = max(end_time, s_end)
            if start_time == float('inf'):
                start_time = 0.0
                end_time = 0.0

            solution["surgeon_assignments"].append({
                "surgeon_id": s_id,
                "hospital_id": h_id,
                "day": d,
                "start_time": round(start_time, 6),
                "end_time": round(end_time, 6)
            })

    # Extract OR sequences
    for (p_id, k_id, h_id, d, o_id), var in w.items():
        if p_id in assigned_patients and k_id in assigned_patients:
            (s1, h1, d1, o1) = assigned_patients[p_id]
            (s2, h2, d2, o2) = assigned_patients[k_id]
            if h1 == h_id and d1 == d and o1 == o_id and h2 == h_id and d2 == d and o2 == o_id:
                solution["or_sequences"].append({
                    "hospital_id": h_id,
                    "day": d,
                    "or_id": o_id,
                    "patient_p": p_id,
                    "patient_k": k_id,
                    "p_after_k": int(round(var.X))
                })

    # Extract surgeon sequences
    for (p_id, k_id, s_id, h_id, d), var in v.items():
        if p_id in assigned_patients and k_id in assigned_patients:
            (s1, h1, d1, o1) = assigned_patients[p_id]
            (s2, h2, d2, o2) = assigned_patients[k_id]
            if s1 == s_id and h1 == h_id and d1 == d and s2 == s_id and h2 == h_id and d2 == d:
                solution["surgeon_sequences"].append({
                    "hospital_id": h_id,
                    "day": d,
                    "surgeon_id": s_id,
                    "patient_p": p_id,
                    "patient_k": k_id,
                    "p_after_k": int(round(var.X))
                })

    return solution


def greedy_heuristic(inst):
    """Simple greedy heuristic to construct a feasible solution."""
    patients = {p['patient_id']: p for p in inst['patients']}
    surgeons = {s['surgeon_id']: s for s in inst['surgeons']}
    hospitals = {h['hospital_id']: h for h in inst['hospitals']}
    num_days = inst['num_days']
    alpha = inst['alpha']
    hospital_ids = sorted(hospitals.keys())
    patient_ids = sorted(patients.keys())

    # Sort patients: mandatory first by due date, then optional by reward descending
    mandatory = sorted([p for p in patient_ids if patients[p]['is_mandatory']],
                       key=lambda p: patients[p]['due_date'])
    optional = sorted([p for p in patient_ids if not patients[p]['is_mandatory']],
                      key=lambda p: -patients[p].get('reward', 0))

    # Track OR usage and surgeon usage
    or_usage = {}  # (h, d, o) -> list of (entry, exit, patient_id, surgeon_id)
    surgeon_usage = {}  # (s, h, d) -> list of (surg_start, surg_end, patient_id)

    assignments = []
    opened_ors_set = set()
    surgeon_assign_set = set()

    def get_or_times(h_id, d, o_id):
        h = hospitals[h_id]
        for or_info in h['ORs']:
            if or_info['or_id'] == o_id:
                rt = or_info['regular_time']
                mot = or_info['max_overtime']
                if str(d) in or_info.get('daily', {}):
                    rt = or_info['daily'][str(d)]['regular_time']
                    mot = or_info['daily'][str(d)]['max_overtime']
                return rt, mot
        return 0, 0

    def try_assign(p_id):
        p = patients[p_id]
        max_day = min(p['due_date'], num_days - 1) if p['is_mandatory'] else num_days - 1
        best = None
        best_cost = float('inf')

        for s_id in p['eligible_surgeons']:
            s = surgeons[s_id]
            for d in s['operating_days']:
                if d > max_day or d >= num_days:
                    continue
                for h_id in hospital_ids:
                    if str(h_id) not in p.get('eligible_ORs_by_hospital', {}):
                        continue
                    for o_id in p['eligible_ORs_by_hospital'][str(h_id)]:
                        rt, mot = get_or_times(h_id, d, o_id)
                        total_time = get_total_or_time(p, s_id)
                        surg_time = get_surg_time(p, s_id)

                        # Check OR capacity
                        current_usage = sum(e[1] - e[0] for e in or_usage.get((h_id, d, o_id), []))
                        if current_usage + total_time > rt + mot:
                            continue

                        # Check surgeon capacity
                        workload = surg_time + (1 - alpha) * (p['preparation_time'] + p['cleaning_time'])
                        current_surg_load = sum(e[1] - e[0] for e in surgeon_usage.get((s_id, h_id, d), []))
                        avail = int(s['availability_by_day'].get(str(d), 0))
                        if current_surg_load + workload > avail:
                            continue

                        # Check surgeon not at another hospital
                        conflict = False
                        for (ss, hh, dd) in surgeon_assign_set:
                            if ss == s_id and dd == d and hh != h_id:
                                conflict = True
                                break
                        if conflict:
                            continue

                        # Find a feasible time slot
                        # Simple: append at end
                        or_list = or_usage.get((h_id, d, o_id), [])
                        if or_list:
                            entry_time = max(e[1] for e in or_list)
                        else:
                            entry_time = 0

                        exit_time = entry_time + total_time
                        if exit_time > rt + mot:
                            continue

                        surg_start = entry_time + p['preparation_time']
                        surg_end = surg_start + surg_time

                        # Check surgeon sequencing
                        surg_list = surgeon_usage.get((s_id, h_id, d), [])
                        feasible_slot = True
                        for (ss, se, _) in surg_list:
                            if not (surg_end <= ss or surg_start >= se):
                                feasible_slot = False
                                break
                        if not feasible_slot:
                            continue

                        # Compute cost
                        cost = 0
                        if (h_id, d, o_id) not in opened_ors_set:
                            h_data = hospitals[h_id]
                            for or_info in h_data['ORs']:
                                if or_info['or_id'] == o_id:
                                    cost += or_info['fixed_cost_per_hour'] * rt / 60.0
                                    break
                        if (s_id, h_id, d) not in surgeon_assign_set:
                            cost += hospitals[h_id]['surgeon_fixed_costs_per_day'].get(str(s_id), 0)

                        overtime_needed = max(0, exit_time - rt)
                        current_ot = max(0, (max(e[1] for e in or_list) if or_list else 0) - rt)
                        additional_ot = max(0, overtime_needed - current_ot)
                        h_data = hospitals[h_id]
                        for or_info in h_data['ORs']:
                            if or_info['or_id'] == o_id:
                                cost += additional_ot * or_info['overtime_cost_per_hour'] / 60.0
                                break

                        if not p['is_mandatory']:
                            cost -= p.get('reward', 0)

                        if cost < best_cost:
                            best_cost = cost
                            best = (s_id, h_id, d, o_id, entry_time, exit_time, surg_start, surg_end)

        if best is not None:
            s_id, h_id, d, o_id, entry_time, exit_time, surg_start, surg_end = best
            or_usage.setdefault((h_id, d, o_id), []).append((entry_time, exit_time, p_id, s_id))
            surgeon_usage.setdefault((s_id, h_id, d), []).append((surg_start, surg_end, p_id))
            opened_ors_set.add((h_id, d, o_id))
            surgeon_assign_set.add((s_id, h_id, d))
            assignments.append({
                'patient_id': p_id,
                'surgeon_id': s_id,
                'hospital_id': h_id,
                'day': d,
                'or_id': o_id,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'surg_start': surg_start,
                'surg_end': surg_end
            })
            return True
        return False

    for p_id in mandatory:
        try_assign(p_id)
    for p_id in optional:
        try_assign(p_id)

    return assignments, opened_ors_set, surgeon_assign_set


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    global_start = time_module.time()

    with open(args.instance_path, 'r') as f:
        inst = json.load(f)

    num_patients = inst['num_patients']
    num_surgeons = inst['num_surgeons']
    num_hospitals = inst['num_hospitals']
    ORs_per_hospital = inst['ORs_per_hospital']
    num_days = inst['num_days']
    alpha = inst['alpha']

    patients = {p['patient_id']: p for p in inst['patients']}
    surgeons = {s['surgeon_id']: s for s in inst['surgeons']}
    hospitals = {h['hospital_id']: h for h in inst['hospitals']}

    patient_ids = sorted(patients.keys())
    surgeon_ids = sorted(surgeons.keys())
    hospital_ids = sorted(hospitals.keys())
    days = list(range(num_days))

    # Precompute OR data
    or_data = {}
    for h_id in hospital_ids:
        h = hospitals[h_id]
        for or_info in h['ORs']:
            o_id = or_info['or_id']
            for d in days:
                rt = or_info['regular_time']
                mot = or_info['max_overtime']
                if str(d) in or_info.get('daily', {}):
                    rt = or_info['daily'][str(d)]['regular_time']
                    mot = or_info['daily'][str(d)]['max_overtime']
                fixed_cost = or_info['fixed_cost_per_hour'] * rt / 60.0
                ot_cost_rate = or_info['overtime_cost_per_hour'] / 60.0
                or_data[(h_id, d, o_id)] = {
                    'regular_time': rt,
                    'max_overtime': mot,
                    'fixed_cost': fixed_cost,
                    'ot_cost_rate': ot_cost_rate
                }

    # Precompute feasible assignments
    feasible = {}
    for p_id in patient_ids:
        p = patients[p_id]
        feasible[p_id] = []
        max_day = min(p['due_date'], num_days - 1) if p['is_mandatory'] else num_days - 1
        for s_id in p['eligible_surgeons']:
            if s_id not in surgeons:
                continue
            s = surgeons[s_id]
            for d in s['operating_days']:
                if d > max_day or d >= num_days:
                    continue
                for h_id in hospital_ids:
                    if str(h_id) in p.get('eligible_ORs_by_hospital', {}):
                        for o_id in p['eligible_ORs_by_hospital'][str(h_id)]:
                            if (h_id, d, o_id) in or_data:
                                feasible[p_id].append((s_id, h_id, d, o_id))

    # Compute big-M values
    max_prep = max(patients[p]['preparation_time'] for p in patient_ids) if patient_ids else 0
    max_clean = max(patients[p]['cleaning_time'] for p in patient_ids) if patient_ids else 0
    max_surg = max(get_surg_time(patients[p], s) for p in patient_ids for s in patients[p]['eligible_surgeons'] if s in surgeons) if patient_ids else 0

    # Try MIP approach
    try:
        elapsed = time_module.time() - global_start
        remaining = args.time_limit - elapsed - 5
        if remaining < 10:
            raise Exception("Not enough time for MIP")

        model = Model("surgical_scheduling")
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', max(remaining, 10))
        model.setParam('OutputFlag', 0)

        # Variables
        x = {}
        for p_id in patient_ids:
            for (s_id, h_id, d, o_id) in feasible[p_id]:
                x[p_id, s_id, h_id, d, o_id] = model.addVar(
                    vtype=GRB.BINARY, name=f"x_{p_id}_{s_id}_{h_id}_{d}_{o_id}")

        y = {}
        for key in or_data:
            y[key] = model.addVar(vtype=GRB.BINARY, name=f"y_{key}")

        z = {}
        for s_id in surgeon_ids:
            s = surgeons[s_id]
            for d in s['operating_days']:
                if d < num_days:
                    for h_id in hospital_ids:
                        z[s_id, h_id, d] = model.addVar(vtype=GRB.BINARY, name=f"z_{s_id}_{h_id}_{d}")

        ot = {}
        for key, data in or_data.items():
            ot[key] = model.addVar(lb=0, ub=data['max_overtime'], vtype=GRB.CONTINUOUS, name=f"ot_{key}")

        f = {}
        for p_id in patient_ids:
            f[p_id] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"f_{p_id}")

        st_var = {}
        for p_id in patient_ids:
            st_var[p_id] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"st_{p_id}")

        model.update()

        # Precompute patients per OR and per surgeon-hospital-day
        patients_per_or = {}
        patients_per_surg_hd = {}
        for p_id in patient_ids:
            for (s_id, h_id, d, o_id) in feasible[p_id]:
                patients_per_or.setdefault((h_id, d, o_id), set()).add(p_id)
                patients_per_surg_hd.setdefault((s_id, h_id, d), set()).add(p_id)

        # Sequencing variables
        w = {}
        for key, pats in patients_per_or.items():
            if len(pats) < 2:
                continue
            h_id, d, o_id = key
            pat_list = sorted(pats)
            for i in range(len(pat_list)):
                for j in range(i + 1, len(pat_list)):
                    w[pat_list[i], pat_list[j], h_id, d, o_id] = model.addVar(
                        vtype=GRB.BINARY, name=f"w_{pat_list[i]}_{pat_list[j]}_{h_id}_{d}_{o_id}")

        v = {}
        for key, pats in patients_per_surg_hd.items():
            if len(pats) < 2:
                continue
            s_id, h_id, d = key
            pat_list = sorted(pats)
            for i in range(len(pat_list)):
                for j in range(i + 1, len(pat_list)):
                    v[pat_list[i], pat_list[j], s_id, h_id, d] = model.addVar(
                        vtype=GRB.BINARY, name=f"v_{pat_list[i]}_{pat_list[j]}_{s_id}_{h_id}_{d}")

        model.update()

        # Constraints
        # 1. Assignment constraints
        for p_id in patient_ids:
            p = patients[p_id]
            if feasible[p_id]:
                if p['is_mandatory']:
                    model.addConstr(
                        quicksum(x[p_id, s, h, d, o] for (s, h, d, o) in feasible[p_id]) == 1,
                        name=f"assign_{p_id}")
                else:
                    model.addConstr(
                        quicksum(x[p_id, s, h, d, o] for (s, h, d, o) in feasible[p_id]) <= 1,
                        name=f"assign_{p_id}")

        # 2. Linking x to y and z
        for p_id in patient_ids:
            for (s_id, h_id, d, o_id) in feasible[p_id]:
                model.addConstr(x[p_id, s_id, h_id, d, o_id] <= y[h_id, d, o_id])
                model.addConstr(x[p_id, s_id, h_id, d, o_id] <= z[s_id, h_id, d])

        # 3. Surgeon at most one hospital per day
        for s_id in surgeon_ids:
            s = surgeons[s_id]
            for d in s['operating_days']:
                if d < num_days:
                    model.addConstr(quicksum(z[s_id, h_id, d] for h_id in hospital_ids) <= 1)

        # 4. Surgical time definition
        for p_id in patient_ids:
            if feasible[p_id]:
                model.addConstr(
                    st_var[p_id] == quicksum(
                        get_surg_time(patients[p_id], s) * x[p_id, s, h, d, o]
                        for (s, h, d, o) in feasible[p_id]),
                    name=f"st_def_{p_id}")

        # 5. Finish time lower bound
        for p_id in patient_ids:
            model.addConstr(f[p_id] >= patients[p_id]['preparation_time'] + st_var[p_id],
                            name=f"finish_lb_{p_id}")

        # 6. OR capacity
        for key, data in or_data.items():
            h_id, d, o_id = key
            if key in patients_per_or and patients_per_or[key]:
                terms = []
                for p_id in patients_per_or[key]:
                    for (s, h, dd, o) in feasible[p_id]:
                        if h == h_id and dd == d and o == o_id:
                            total_time = get_total_or_time(patients[p_id], s)
                            terms.append(total_time * x[p_id, s, h_id, d, o_id])
                if terms:
                    model.addConstr(quicksum(terms) <= data['regular_time'] + ot[key])

        # 7. Surgeon workload
        for (s_id, h_id, d), pats in patients_per_surg_hd.items():
            if s_id not in surgeons:
                continue
            avail = int(surgeons[s_id]['availability_by_day'].get(str(d), 0))
            terms = []
            for p_id in pats:
                for (s, h, dd, o) in feasible[p_id]:
                    if s == s_id and h == h_id and dd == d:
                        surg_t = get_surg_time(patients[p_id], s_id)
                        workload = alpha * surg_t + (1 - alpha) * get_total_or_time(patients[p_id], s_id)
                        terms.append(workload * x[p_id, s_id, h_id, d, o])
            if terms:
                model.addConstr(quicksum(terms) <= avail * z[s_id, h_id, d])

        # 8. Overtime lower bound from completion
        for key, data in or_data.items():
            h_id, d, o_id = key
            if key in patients_per_or:
                for p_id in patients_per_or[key]:
                    in_or_expr = quicksum(x[p_id, s, h_id, d, o_id]
                                          for (s, h, dd, o) in feasible[p_id]
                                          if h == h_id and dd == d and o == o_id)
                    M_val = data['regular_time'] + data['max_overtime'] + max_prep + max_clean + max_surg
                    model.addConstr(
                        ot[key] >= f[p_id] + patients[p_id]['cleaning_time'] - data['regular_time']
                        - M_val * (1 - in_or_expr))

        # 9. OR sequencing constraints
        for (p_id, k_id, h_id, d, o_id), var in w.items():
            data = or_data.get((h_id, d, o_id))
            if data is None:
                continue
            M_val = data['regular_time'] + data['max_overtime'] + max_prep + max_clean + max_surg

            in_or_p = quicksum(x[p_id, s, h_id, d, o_id]
                               for (s, h, dd, o) in feasible[p_id]
                               if h == h_id and dd == d and o == o_id)
            in_or_k = quicksum(x[k_id, s, h_id, d, o_id]
                               for (s, h, dd, o) in feasible[k_id]
                               if h == h_id and dd == d and o == o_id)

            # If w=1: k is after p in this OR
            model.addConstr(
                f[k_id] >= f[p_id] + patients[p_id]['cleaning_time'] + patients[k_id]['preparation_time']
                + st_var[k_id] - M_val * (1 - var) - M_val * (1 - in_or_p) - M_val * (1 - in_or_k))

            # If w=0: p is after k in this OR
            model.addConstr(
                f[p_id] >= f[k_id] + patients[k_id]['cleaning_time'] + patients[p_id]['preparation_time']
                + st_var[p_id] - M_val * var - M_val * (1 - in_or_p) - M_val * (1 - in_or_k))

        # 10. Surgeon sequencing constraints
        for (p_id, k_id, s_id, h_id, d), var in v.items():
            if s_id not in surgeons:
                continue
            avail = int(surgeons[s_id]['availability_by_day'].get(str(d), 0))
            M_val = avail + max_surg + max_prep + max_clean

            with_surg_p = quicksum(x[p_id, s_id, h_id, d, o]
                                   for (s, h, dd, o) in feasible[p_id]
                                   if s == s_id and h == h_id and dd == d)
            with_surg_k = quicksum(x[k_id, s_id, h_id, d, o]
                                   for (s, h, dd, o) in feasible[k_id]
                                   if s == s_id and h == h_id and dd == d)

            # If v=1: k is after p with this surgeon
            model.addConstr(
                f[k_id] >= f[p_id] + st_var[k_id]
                - M_val * (1 - var) - M_val * (1 - with_surg_p) - M_val * (1 - with_surg_k))

            # If v=0: p is after k with this surgeon
            model.addConstr(
                f[p_id] >= f[k_id] + st_var[p_id]
                - M_val * var - M_val * (1 - with_surg_p) - M_val * (1 - with_surg_k))

        # Objective
        obj = LinExpr()
        for key, data in or_data.items():
            obj += data['fixed_cost'] * y[key]
            obj += data['ot_cost_rate'] * ot[key]

        for (s_id, h_id, d), var in z.items():
            cost = hospitals[h_id].get('surgeon_fixed_costs_per_day', {}).get(str(s_id), 0)
            obj += cost * var

        for p_id in patient_ids:
            p = patients[p_id]
            if not p['is_mandatory'] and p.get('reward', 0) > 0:
                obj -= p['reward'] * quicksum(x[p_id, s, h, d, o] for (s, h, d, o) in feasible[p_id])

        model.setObjective(obj, GRB.MINIMIZE)
        model.update()

        # Solve
        model.optimize()

        if model.SolCount > 0:
            solution = build_solution_from_model(
                model, x, y, z, ot, f, st_var, w, v, patients, surgeons, hospitals,
                hospital_ids, days, or_data, feasible, patient_ids)

            if logger:
                logger.log_solution(solution['objective_value'], solution)

            with open(args.solution_path, 'w') as fout:
                json.dump(solution, fout, indent=2)
            return

    except Exception as e:
        pass

    # Fallback: greedy heuristic
    try:
        assignments_raw, opened_ors_set, surgeon_assign_set = greedy_heuristic(inst)

        solution = {
            "objective_value": 0.0,
            "assignments": [],
            "opened_ors": [],
            "surgeon_assignments": [],
            "or_sequences": [],
            "surgeon_sequences": []
        }

        # Compute objective
        total_cost = 0.0
        for (h_id, d, o_id) in opened_ors_set:
            data = or_data.get((h_id, d, o_id))
            if data:
                total_cost += data['fixed_cost']

        for (s_id, h_id, d) in surgeon_assign_set:
            cost = hospitals[h_id].get('surgeon_fixed_costs_per_day', {}).get(str(s_id), 0)
            total_cost += cost

        # Build assignments and compute overtime
        or_patients = {}  # (h,d,o) -> list of assignments
        for a in assignments_raw:
            key = (a['hospital_id'], a['day'], a['or_id'])
            or_patients.setdefault(key, []).append(a)

        for a in assignments_raw:
            p = patients[a['patient_id']]
            solution["assignments"].append({
                "patient_id": a['patient_id'],
                "surgeon_id": a['surgeon_id'],
                "hospital_id": a['hospital_id'],
                "day": a['day'],
                "or_id": a['or_id'],
                "is_mandatory": p['is_mandatory'],
                "T_ps": get_total_or_time(p, a['surgeon_id']),
                "finish_time": round(a['surg_end'], 6),
                "surgery_start_time": round(a['surg_start'], 6),
                "room_entry_time": round(a['entry_time'], 6),
                "room_exit_time": round(a['exit_time'], 6)
            })
            if not p['is_mandatory']:
                total_cost -= p.get('reward', 0)

        # Opened ORs with overtime
        for (h_id, d, o_id) in opened_ors_set:
            data = or_data.get((h_id, d, o_id))
            if data is None:
                continue
            pats = or_patients.get((h_id, d, o_id), [])
            completion = max((a['exit_time'] for a in pats), default=0)
            overtime = max(0, completion - data['regular_time'])
            total_cost += data['ot_cost_rate'] * overtime
            solution["opened_ors"].append({
                "hospital_id": h_id,
                "day": d,
                "or_id": o_id,
                "regular_time": data['regular_time'],
                "completion_time": round(completion, 6),
                "overtime": round(overtime, 6)
            })

        # Surgeon assignments
        surg_times = {}
        for a in assignments_raw:
            key = (a['surgeon_id'], a['hospital_id'], a['day'])
            surg_times.setdefault(key, []).append(a)

        for (s_id, h_id, d) in surgeon_assign_set:
            pats = surg_times.get((s_id, h_id, d), [])
            if pats:
                start_t = min(a['surg_start'] for a in pats)
                end_t = max(a['surg_end'] for a in pats)
            else:
                start_t = 0
                end_t = 0
            solution["surgeon_assignments"].append({
                "surgeon_id": s_id,
                "hospital_id": h_id,
                "day": d,
                "start_time": round(start_t, 6),
                "end_time": round(end_t, 6)
            })

        # OR sequences
        for (h_id, d, o_id), pats in or_patients.items():
            sorted_pats = sorted(pats, key=lambda a: a['entry_time'])
            for i in range(len(sorted_pats)):
                for j in range(i + 1, len(sorted_pats)):
                    p_first = sorted_pats[i]
                    p_second = sorted_pats[j]
                    solution["or_sequences"].append({
                        "hospital_id": h_id,
                        "day": d,
                        "or_id": o_id,
                        "patient_p": p_second['patient_id'],
                        "patient_k": p_first['patient_id'],
                        "p_after_k": 1
                    })

        # Surgeon sequences
        for (s_id, h_id, d), pats in surg_times.items():
            sorted_pats = sorted(pats, key=lambda a: a['surg_start'])
            for i in range(len(sorted_pats)):
                for j in range(i + 1, len(sorted_pats)):
                    p_first = sorted_pats[i]
                    p_second = sorted_pats[j]
                    solution["surgeon_sequences"].append({
                        "hospital_id": h_id,
                        "day": d,
                        "surgeon_id": s_id,
                        "patient_p": p_second['patient_id'],
                        "patient_k": p_first['patient_id'],
                        "p_after_k": 1
                    })

        solution["objective_value"] = round(total_cost, 6)

        if logger:
            logger.log_solution(solution['objective_value'], solution)

        with open(args.solution_path, 'w') as fout:
            json.dump(solution, fout, indent=2)

    except Exception as e2:
        # Last resort: empty solution
        solution = {
            "objective_value": 0.0,
            "assignments": [],
            "opened_ors": [],
            "surgeon_assignments": [],
            "or_sequences": [],
            "surgeon_sequences": []
        }
        with open(args.solution_path, 'w') as fout:
            json.dump(solution, fout, indent=2)


if __name__ == '__main__':
    main()