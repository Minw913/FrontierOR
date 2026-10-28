import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    num_days = data['num_days']
    num_patients = data['num_patients']
    num_surgeons = data['num_surgeons']
    num_hospitals = data['num_hospitals']
    ORs_per_hospital = data['ORs_per_hospital']
    alpha = data['alpha']

    patients_data = data['patients']
    surgeons_data = data['surgeons']
    hospitals_data = data['hospitals']

    hospitals = [h['hospital_id'] for h in hospitals_data]
    ORs_in_hospital = {h['hospital_id']: [or_info['or_id'] for or_info in h['ORs']] for h in hospitals_data}
    surgeons_list = [s['surgeon_id'] for s in surgeons_data]

    preparation_time = {}
    cleaning_time = {}
    due_date = {}
    is_mandatory = {}
    reward = {}
    eligible_surgeons = {}
    eligible_ORs_by_hospital = {}
    surgeon_specific_times = {}
    base_surgical_time = {}

    for p in patients_data:
        pid = p['patient_id']
        preparation_time[pid] = p['preparation_time']
        cleaning_time[pid] = p['cleaning_time']
        due_date[pid] = p['due_date']
        is_mandatory[pid] = p['is_mandatory']
        reward[pid] = p['reward']
        eligible_surgeons[pid] = p['eligible_surgeons']
        eligible_ORs_by_hospital[pid] = {int(hid): or_list for hid, or_list in p['eligible_ORs_by_hospital'].items()}
        surgeon_specific_times[pid] = {int(sid): time for sid, time in p['surgeon_specific_times'].items()}
        base_surgical_time[pid] = p['base_surgical_time']

    operating_days = {}
    availability = {}
    for s in surgeons_data:
        sid = s['surgeon_id']
        operating_days[sid] = s['operating_days']
        availability[sid] = {int(day): avail for day, avail in s['availability_by_day'].items()}

    regular_time = {}
    max_overtime = {}
    fixed_cost_per_hour = {}
    overtime_cost_per_hour = {}
    surgeon_fixed_costs_per_day = {}

    for h in hospitals_data:
        hid = h['hospital_id']
        surgeon_fixed_costs_per_day[hid] = {int(sid): cost for sid, cost in h['surgeon_fixed_costs_per_day'].items()}
        for or_info in h['ORs']:
            oid = or_info['or_id']
            fixed_cost_per_hour[(hid, oid)] = or_info['fixed_cost_per_hour']
            overtime_cost_per_hour[(hid, oid)] = or_info['overtime_cost_per_hour']
            default_regular = or_info['regular_time']
            default_max_ot = or_info['max_overtime']
            for day in range(1, num_days+1):
                day_str = str(day)
                if 'daily' in or_info and day_str in or_info['daily']:
                    regular_time[(hid, day, oid)] = or_info['daily'][day_str].get('regular_time', default_regular)
                    max_overtime[(hid, day, oid)] = or_info['daily'][day_str].get('max_overtime', default_max_ot)
                else:
                    regular_time[(hid, day, oid)] = default_regular
                    max_overtime[(hid, day, oid)] = default_max_ot

    model = gp.Model('SurgicalScheduling')
    model.setParam('OutputFlag', 1)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)

    feasible_assignments = []
    for p in patients_data:
        pid = p['patient_id']
        for sid in eligible_surgeons[pid]:
            if sid not in operating_days:
                continue
            for hid, or_list in eligible_ORs_by_hospital[pid].items():
                if hid not in hospitals:
                    continue
                hospital_ORs = ORs_in_hospital[hid]
                for oid in or_list:
                    if oid not in hospital_ORs:
                        continue
                    for day in operating_days[sid]:
                        if day < 1 or day > num_days:
                            continue
                        if is_mandatory[pid]:
                            if day <= due_date[pid]:
                                feasible_assignments.append((pid, sid, hid, day, oid))
                        else:
                            feasible_assignments.append((pid, sid, hid, day, oid))

    x = {}
    for assign in feasible_assignments:
        p, s, h, d, o = assign
        x[p, s, h, d, o] = model.addVar(vtype=GRB.BINARY, name=f'x_{p}_{s}_{h}_{d}_{o}')

    assigned = {}
    patient_ids = [p['patient_id'] for p in patients_data]
    for pid in patient_ids:
        assigned[pid] = model.addVar(vtype=GRB.BINARY, name=f'assigned_{pid}')

    for pid in patient_ids:
        model.addConstr(assigned[pid] == gp.quicksum(
            x[assign] for assign in feasible_assignments if assign[0] == pid
        ), name=f'assigned_constr_{pid}')

    y = {}
    for h in hospitals:
        for day in range(1, num_days+1):
            for oid in ORs_in_hospital[h]:
                y[h, day, oid] = model.addVar(vtype=GRB.BINARY, name=f'y_{h}_{day}_{oid}')

    z = {}
    for s in surgeons_list:
        if s in operating_days:
            for day in operating_days[s]:
                if day < 1 or day > num_days:
                    continue
                for h in hospitals:
                    z[s, h, day] = model.addVar(vtype=GRB.BINARY, name=f'z_{s}_{h}_{day}')

    room_entry = {}
    surgery_start = {}
    finish_time = {}
    room_exit = {}
    for pid in patient_ids:
        room_entry[pid] = model.addVar(lb=0, name=f'room_entry_{pid}')
        surgery_start[pid] = model.addVar(lb=0, name=f'surgery_start_{pid}')
        finish_time[pid] = model.addVar(lb=0, name=f'finish_time_{pid}')
        room_exit[pid] = model.addVar(lb=0, name=f'room_exit_{pid}')

    assigned_s = {}
    for pid in patient_ids:
        for sid in eligible_surgeons[pid]:
            assigned_s[pid, sid] = model.addVar(vtype=GRB.BINARY, name=f'assigned_s_{pid}_{sid}')

    for (pid, sid) in assigned_s:
        model.addConstr(assigned_s[pid, sid] == gp.quicksum(
            x[assign] for assign in feasible_assignments if assign[0] == pid and assign[1] == sid
        ), name=f'assigned_s_constr_{pid}_{sid}')

    M_big = 100000
    for pid in patient_ids:
        model.addConstr(surgery_start[pid] == room_entry[pid] + preparation_time[pid] * assigned[pid], 
                        name=f'timing_prep_{pid}')
        surg_time_expr = gp.LinExpr()
        for sid in eligible_surgeons[pid]:
            if (pid, sid) in assigned_s:
                surg_time = surgeon_specific_times[pid].get(sid, base_surgical_time[pid])
                surg_time_expr += surg_time * assigned_s[pid, sid]
        model.addConstr(finish_time[pid] == surgery_start[pid] + surg_time_expr, 
                        name=f'timing_surg_{pid}')
        model.addConstr(room_exit[pid] == finish_time[pid] + cleaning_time[pid] * assigned[pid], 
                        name=f'timing_clean_{pid}')
        model.addConstr(room_entry[pid] <= M_big * assigned[pid])
        model.addConstr(surgery_start[pid] <= M_big * assigned[pid])
        model.addConstr(finish_time[pid] <= M_big * assigned[pid])
        model.addConstr(room_exit[pid] <= M_big * assigned[pid])

    completion_time = {}
    overtime = {}
    for h in hospitals:
        for day in range(1, num_days+1):
            for oid in ORs_in_hospital[h]:
                completion_time[h, day, oid] = model.addVar(lb=0, name=f'completion_{h}_{day}_{oid}')
                overtime[h, day, oid] = model.addVar(lb=0, ub=max_overtime[h, day, oid], 
                                                    name=f'overtime_{h}_{day}_{oid}')

    patients_by_or = {}
    for assign in feasible_assignments:
        p, s, h, d, o = assign
        key = (h, d, o)
        if key in patients_by_or:
            patients_by_or[key].append(p)
        else:
            patients_by_or[key] = [p]

    for key in patients_by_or:
        h, day, oid = key
        patient_list = patients_by_or[key]
        expr_list = []
        for pid in patient_list:
            surgeons_in_key = [s for s in eligible_surgeons[pid] if (pid, s, h, day, oid) in x]
            if surgeons_in_key:
                expr = gp.quicksum(x[pid, s, h, day, oid] for s in surgeons_in_key)
                expr_list.append(expr)
                model.addConstr(completion_time[h, day, oid] >= room_exit[pid] - M_big * (1 - expr))
        model.addConstr(completion_time[h, day, oid] <= (regular_time[h, day, oid] + max_overtime[h, day, oid]) * y[h, day, oid], 
                        name=f'max_completion_{h}_{day}_{oid}')

    for (h, day, oid) in completion_time:
        model.addConstr(overtime[h, day, oid] >= completion_time[h, day, oid] - regular_time[h, day, oid])
        model.addConstr(overtime[h, day, oid] <= max_overtime[h, day, oid] * y[h, day, oid])

    for s in surgeons_list:
        if s in operating_days:
            for day in operating_days[s]:
                if day < 1 or day > num_days:
                    continue
                workload_expr = gp.LinExpr()
                for p in patient_ids:
                    if (p, s, day) in [(a[0], a[1], a[3]) for a in feasible_assignments]:
                        for assign in [a for a in feasible_assignments if a[0]==p and a[1]==s and a[3]==day]:
                            h, oid = assign[2], assign[4]
                            proc_time = surgeon_specific_times[p].get(s, base_surgical_time[p])
                            w = (1 - alpha) * (preparation_time[p] + cleaning_time[p]) + proc_time
                            workload_expr += w * x[assign]
                model.addConstr(workload_expr <= availability[s][day], name=f'workload_{s}_{day}')

    start_time = {}
    end_time = {}
    for s in surgeons_list:
        if s in operating_days:
            for day in operating_days[s]:
                if day < 1 or day > num_days:
                    continue
                for h in hospitals:
                    start_time[s, h, day] = model.addVar(lb=0, name=f'start_{s}_{h}_{day}')
                    end_time[s, h, day] = model.addVar(lb=0, name=f'end_{s}_{h}_{day}')

    for (s, h, day) in z:
        patient_assignments_for_s = [assign for assign in feasible_assignments if assign[1] == s and assign[2] == h and assign[3] == day]
        for assign in patient_assignments_for_s:
            p = assign[0]
            model.addConstr(start_time[s, h, day] <= surgery_start[p] + M_big * (1 - x[assign]), 
                            name=f'start_time_{p}_{s}_{h}_{day}')
            model.addConstr(end_time[s, h, day] >= finish_time[p] - M_big * (1 - x[assign]), 
                            name=f'end_time_{p}_{s}_{h}_{day}')
        model.addConstr(end_time[s, h, day] - start_time[s, h, day] <= availability[s][day] * z[s, h, day], 
                        name=f'surgeon_workday_{s}_{h}_{day}')

    for assign in feasible_assignments:
        p, s, h, d, o = assign
        model.addConstr(x[p, s, h, d, o] <= y[h, d, o], name=f'open_or_{p}_{s}_{h}_{d}_{o}')
        model.addConstr(x[p, s, h, d, o] <= z[s, h, d], name=f'surgeon_assigned_{p}_{s}_{h}_{d}_{o}')

    for s in surgeons_list:
        if s in operating_days:
            for day in operating_days[s]:
                if day < 1 or day > num_days:
                    continue
                model.addConstr(gp.quicksum(z[s, h, day] for h in hospitals) <= 1, name=f'surgeon_location_{s}_{day}')

    a = {}
    patient_assignments_per_or = {}
    for assign in feasible_assignments:
        p, s, h, d, o = assign
        key = (h, d, o)
        if key not in patient_assignments_per_or:
            patient_assignments_per_or[key] = set()
        patient_assignments_per_or[key].add(p)

    for key, patients_set in patient_assignments_per_or.items():
        h, d, o = key
        patients_list = list(patients_set)
        n = len(patients_list)
        for i in range(n):
            p_i = patients_list[i]
            for j in range(i+1, n):
                p_j = patients_list[j]
                p_i_expr = assigned[p_i]
                p_j_expr = assigned[p_j]
                a_key = (p_i, p_j, h, d, o) if p_i < p_j else (p_j, p_i, h, d, o)
                if a_key not in a:
                    a_var = model.addVar(vtype=GRB.BINARY, name=f'a_{a_key[0]}_{a_key[1]}_{h}_{d}_{o}')
                    a[a_key] = a_var
                    if p_i < p_j:
                        first = p_i
                        second = p_j
                    else:
                        first = p_j
                        second = p_i
                    model.addConstr(room_exit[first] <= room_entry[second] + M_big * (1 - a_var) + M_big * (2 - assigned[first] - assigned[second]))
                    model.addConstr(room_exit[second] <= room_entry[first] + M_big * a_var + M_big * (2 - assigned[first] - assigned[second]))

    b = {}
    surgeon_assignments_per_day = {}
    for assign in feasible_assignments:
        p, s, h, d, o = assign
        key = (s, h, d)
        if key not in surgeon_assignments_per_day:
            surgeon_assignments_per_day[key] = set()
        surgeon_assignments_per_day[key].add(p)

    for key, patients_set in surgeon_assignments_per_day.items():
        s, h, d = key
        patients_list = list(patients_set)
        n = len(patients_list)
        for i in range(n):
            p_i = patients_list[i]
            for j in range(i+1, n):
                p_j = patients_list[j]
                b_key = (p_i, p_j, s, h, d) if p_i < p_j else (p_j, p_i, s, h, d)
                if b_key not in b:
                    b_var = model.addVar(vtype=GRB.BINARY, name=f'b_{b_key[0]}_{b_key[1]}_{s}_{h}_{d}')
                    b[b_key] = b_var
                    first, second = (p_i, p_j) if p_i < p_j else (p_j, p_i)
                    model.addConstr(finish_time[first] <= surgery_start[second] + M_big * (1 - b_var) + M_big * (2 - assigned[first] - assigned[second]))
                    model.addConstr(finish_time[second] <= surgery_start[first] + M_big * b_var + M_big * (2 - assigned[first] - assigned[second]))

    obj = gp.LinExpr()
    for (h, oid) in set((h, oid) for h in hospitals for oid in ORs_in_hospital[h]):
        default_fixed_cost = fixed_cost_per_hour.get((h, oid), 0)
        for day in range(1, num_days+1):
            fixed_cost_dollars = default_fixed_cost * (regular_time.get((h, day, oid), 0) / 60.0)
            obj += fixed_cost_dollars * y[h, day, oid]
    for (s, h, day) in z:
        if (s, h, day) in z:
            fixed_cost = surgeon_fixed_costs_per_day.get(h, {}).get(s, 0)
            obj += fixed_cost * z[s, h, day]
    for (h, day, oid) in overtime:
        ot_cost_per_min = overtime_cost_per_hour.get((h, oid), 0) / 60.0
        obj += ot_cost_per_min * overtime[h, day, oid]
    for pid in patient_ids:
        if not is_mandatory[pid]:
            obj -= reward[pid] * assigned[pid]
    model.setObjective(obj, GRB.MINIMIZE)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    best_obj = float('inf')

    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            current_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if current_obj < best_obj:
                best_obj = current_obj
                if logger:
                    logger.log(current_obj)

    model.optimize(callback)

    solution = {
        "objective_value": model.ObjVal,
        "assignments": [],
        "opened_ors": [],
        "surgeon_assignments": [],
        "or_sequences": [],
        "surgeon_sequences": []
    }

    for assign in feasible_assignments:
        p, s, h, d, o = assign
        if x[assign].X > 0.5:
            T_ps = preparation_time[p] + surgeon_specific_times[p].get(s, base_surgical_time[p]) + cleaning_time[p]
            surgery_start_val = surgery_start[p].X
            finish_val = finish_time[p].X
            room_entry_val = room_entry[p].X
            room_exit_val = room_exit[p].X
            solution['assignments'].append({
                "patient_id": p,
                "surgeon_id": s,
                "hospital_id": h,
                "day": d,
                "or_id": o,
                "is_mandatory": is_mandatory[p],
                "T_ps": T_ps,
                "finish_time": finish_val,
                "surgery_start_time": surgery_start_val,
                "room_entry_time": room_entry_val,
                "room_exit_time": room_exit_val
            })

    for (h, d, o) in y:
        if y[h, d, o].X > 0.5:
            comp_time = completion_time[h, d, o].X
            ot = overtime[h, d, o].X
            reg_time = regular_time.get((h,d,o), 0)
            solution['opened_ors'].append({
                "hospital_id": h,
                "day": d,
                "or_id": o,
                "regular_time": reg_time,
                "completion_time": comp_time,
                "overtime": ot
            })

    for (s, h, d) in z:
        if (s, h, d) in z and z[s, h, d].X > 0.5:
            start_val = start_time[s, h, d].X
            end_val = end_time[s, h, d].X
            solution['surgeon_assignments'].append({
                "surgeon_id": s,
                "hospital_id": h,
                "day": d,
                "start_time": start_val,
                "end_time": end_val
            })

    for key, var in a.items():
        if var.X > 0.5:
            p1, p2, h, day, oid = key
            solution['or_sequences'].append({
                "hospital_id": h,
                "day": day,
                "or_id": oid,
                "patient_p": p1,
                "patient_k": p2,
                "p_after_k": 1
            })

    for key, var in b.items():
        if var.X > 0.5:
            p1, p2, s, h, day = key
            solution['surgeon_sequences'].append({
                "hospital_id": h,
                "day": day,
                "surgeon_id": s,
                "patient_p": p1,
                "patient_k": p2,
                "p_after_k": 1
            })

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()