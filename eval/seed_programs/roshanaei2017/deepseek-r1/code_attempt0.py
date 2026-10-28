import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import time
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
            for hid, or_list in eligible_ORs_by_hospital[pid].items():
                for oid in or_list:
                    for day in range(1, num_days+1):
                        if day not in operating_days.get(sid, []):
                            continue
                        if is_mandatory[pid] and day > due_date[pid]:
                            continue
                        if oid not in ORs_in_hospital[hid]:
                            continue
                        feasible_assignments.append((pid, sid, hid, day, oid))

    x = {}
    for assign in feasible_assignments:
        p, s, h, d, o = assign
        x[p,s,h,d,o] = model.addVar(vtype=GRB.BINARY, name=f'x_{p}_{s}_{h}_{d}_{o}')

    assigned = {}
    for p in [p['patient_id'] for p in patients_data]:
        assigned[p] = model.addVar(vtype=GRB.BINARY, name=f'assigned_{p}')

    for p in assigned:
        model.addConstr(assigned[p] == gp.quicksum(
            x[assign] for assign in feasible_assignments if assign[0] == p
        ), name=f'assigned_constr_{p}')

    y = {}
    for h in hospitals:
        for day in range(1, num_days+1):
            for oid in ORs_in_hospital[h]:
                y[h, day, oid] = model.addVar(vtype=GRB.BINARY, name=f'y_{h}_{day}_{oid}')

    z = {}
    for s in surgeons_list:
        for day in operating_days[s]:
            for h in hospitals:
                z[s, h, day] = model.addVar(vtype=GRB.BINARY, name=f'z_{s}_{h}_{day}')

    room_entry = {}
    surgery_start = {}
    finish_time = {}
    room_exit = {}
    for p in [p['patient_id'] for p in patients_data]:
        room_entry[p] = model.addVar(lb=0, name=f'room_entry_{p}')
        surgery_start[p] = model.addVar(lb=0, name=f'surgery_start_{p}')
        finish_time[p] = model.addVar(lb=0, name=f'finish_time_{p}')
        room_exit[p] = model.addVar(lb=0, name=f'room_exit_{p}')

    assigned_s = {}
    for p in [p['patient_id'] for p in patients_data]:
        for s in eligible_surgeons[p]:
            assigned_s[p, s] = model.addVar(vtype=GRB.BINARY, name=f'assigned_s_{p}_{s}')

    for p, s in assigned_s:
        model.addConstr(assigned_s[p, s] == gp.quicksum(
            x[assign] for assign in feasible_assignments if assign[0] == p and assign[1] == s
        ), name=f'assigned_s_constr_{p}_{s}')

    for p in [p['patient_id'] for p in patients_data]:
        model.addConstr(surgery_start[p] == room_entry[p] + preparation_time[p] * assigned[p], 
                        name=f'timing_prep_{p}')
        surg_time_expr = gp.LinExpr()
        for s in eligible_surgeons[p]:
            surg_time_expr += surgeon_specific_times[p].get(s, base_surgical_time[p]) * assigned_s[p, s]
        model.addConstr(finish_time[p] == surgery_start[p] + surg_time_expr, 
                        name=f'timing_surg_{p}')
        model.addConstr(room_exit[p] == finish_time[p] + cleaning_time[p] * assigned[p], 
                        name=f'timing_clean_{p}')

    completion_time = {}
    overtime = {}
    for h in hospitals:
        for day in range(1, num_days+1):
            for oid in ORs_in_hospital[h]:
                completion_time[h, day, oid] = model.addVar(lb=0, name=f'completion_{h}_{day}_{oid}')
                overtime[h, day, oid] = model.addVar(lb=0, ub=max_overtime[h, day, oid], 
                                                    name=f'overtime_{h}_{day}_{oid}')

    for h, day, oid in completion_time:
        model.addConstr(completion_time[h, day, oid] >= room_exit[p] - 10000*(1 - x.get((p, s, h, day, oid), 0))
                        for p in [p['patient_id'] for p in patients_data] for s in eligible_surgeons[p] 
                        if (p, s, h, day, oid) in x)
        model.addConstr(overtime[h, day, oid] >= completion_time[h, day, oid] - regular_time[h, day, oid])
        model.addConstr(completion_time[h, day, oid] <= (regular_time[h, day, oid] + max_overtime[h, day, oid]) * 
                        y[h, day, oid])
        model.addConstr(overtime[h, day, oid] <= max_overtime[h, day, oid] * y[h, day, oid])

    for s in surgeons_list:
        for day in operating_days[s]:
            workload_expr = gp.LinExpr()
            for assign in feasible_assignments:
                p, s_assign, h, d, o = assign
                if s_assign == s and d == day:
                    w = (1 - alpha) * (preparation_time[p] + cleaning_time[p]) + \
                        surgeon_specific_times[p].get(s, base_surgical_time[p])
                    workload_expr += w * x[assign]
            model.addConstr(workload_expr <= availability[s][day], name=f'workload_{s}_{day}')

    start_time = {}
    end_time = {}
    for s in surgeons_list:
        for day in operating_days[s]:
            for h in hospitals:
                start_time[s, h, day] = model.addVar(lb=0, name=f'start_{s}_{h}_{day}')
                end_time[s, h, day] = model.addVar(lb=0, name=f'end_{s}_{h}_{day}')

    for s, h, day in start_time:
        for assign in feasible_assignments:
            p, s_assign, h_assign, d, o = assign
            if s_assign == s and h_assign == h and d == day:
                model.addGenConstrIndicator(x[assign], True, start_time[s, h, day] <= surgery_start[p])
                model.addGenConstrIndicator(x[assign], True, end_time[s, h, day] >= finish_time[p])
        model.addGenConstrIndicator(z[s, h, day], True, 
                                    end_time[s, h, day] - start_time[s, h, day] <= availability[s][day])

    for assign in feasible_assignments:
        p, s, h, d, o = assign
        model.addConstr(x[assign] <= y[h, d, o])
        model.addConstr(x[assign] <= z[s, h, d])

    for s in surgeons_list:
        for day in operating_days[s]:
            model.addConstr(gp.quicksum(z[s, h, day] for h in hospitals) <= 1)

    M = 10000
    a = {}
    for h in hospitals:
        for day in range(1, num_days+1):
            for oid in ORs_in_hospital[h]:
                patients_in_or = [p for p in [p['patient_id'] for p in patients_data] 
                                 if any((p, s, h, day, oid) in x for s in eligible_surgeons[p])]
                for i in range(len(patients_in_or)):
                    for j in range(i+1, len(patients_in_or)):
                        p = patients_in_or[i]
                        k = patients_in_or[j]
                        a_key = (min(p, k), max(p, k), h, day, oid)
                        a[a_key] = model.addVar(vtype=GRB.BINARY, name=f'a_{p}_{k}_{h}_{day}_{oid}')
                        model.addConstr(room_exit[p] <= room_entry[k] + M * (1 - a[a_key]) + M * (2 - assigned[p] - assigned[k]))
                        model.addConstr(room_exit[k] <= room_entry[p] + M * a[a_key] + M * (2 - assigned[p] - assigned[k]))

    b = {}
    for s in surgeons_list:
        for day in operating_days[s]:
            for h in hospitals:
                patients_for_surgeon = [p for p in [p['patient_id'] for p in patients_data] 
                                       if any((p, s, h, day, o) in x for o in ORs_in_hospital[h])]
                for i in range(len(patients_for_surgeon)):
                    for j in range(i+1, len(patients_for_surgeon)):
                        p = patients_for_surgeon[i]
                        k = patients_for_surgeon[j]
                        b_key = (min(p, k), max(p, k), s, h, day)
                        b[b_key] = model.addVar(vtype=GRB.BINARY, name=f'b_{p}_{k}_{s}_{h}_{day}')
                        model.addConstr(finish_time[p] <= surgery_start[k] + M * (1 - b[b_key]) + M * (2 - assigned[p] - assigned[k]))
                        model.addConstr(finish_time[k] <= surgery_start[p] + M * b[b_key] + M * (2 - assigned[p] - assigned[k]))

    obj = gp.LinExpr()
    for h, day, oid in y:
        fixed_cost = (fixed_cost_per_hour[h, oid] * regular_time[h, day, oid]) / 60.0
        obj += fixed_cost * y[h, day, oid]
    for s, h, day in z:
        fixed_cost = surgeon_fixed_costs_per_day[h].get(s, 0)
        obj += fixed_cost * z[s, h, day]
    for h, day, oid in overtime:
        overtime_cost = (overtime_cost_per_hour[h, oid] / 60.0) * overtime[h, day, oid]
        obj += overtime_cost
    for p in [p['patient_id'] for p in patients_data]:
        if not is_mandatory[p]:
            obj -= reward[p] * assigned[p]
    model.setObjective(obj, GRB.MINIMIZE)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    best_obj = None

    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            current_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if best_obj is None or current_obj < best_obj:
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

    for h, d, o in y:
        if y[h, d, o].X > 0.5:
            comp_time = completion_time[h, d, o].X
            ot = overtime[h, d, o].X
            solution['opened_ors'].append({
                "hospital_id": h,
                "day": d,
                "or_id": o,
                "regular_time": regular_time[h, d, o],
                "completion_time": comp_time,
                "overtime": ot
            })

    for s, h, d in z:
        if z[s, h, d].X > 0.5:
            start_val = start_time[s, h, d].X
            end_val = end_time[s, h, d].X
            solution['surgeon_assignments'].append({
                "surgeon_id": s,
                "hospital_id": h,
                "day": d,
                "start_time": start_val,
                "end_time": end_val
            })

    for key in a:
        p, k, h, d, o = key[0], key[1], key[2], key[3], key[4]
        if a[key].X > 0.5:
            p_after_k = 1
        else:
            p_after_k = 0
        solution['or_sequences'].append({
            "hospital_id": h,
            "day": d,
            "or_id": o,
            "patient_p": p,
            "patient_k": k,
            "p_after_k": p_after_k
        })

    for key in b:
        p, k, s, h, d = key[0], key[1], key[2], key[3], key[4]
        if b[key].X > 0.5:
            p_after_k = 1
        else:
            p_after_k = 0
        solution['surgeon_sequences'].append({
            "hospital_id": h,
            "day": d,
            "surgeon_id": s,
            "patient_p": p,
            "patient_k": k,
            "p_after_k": p_after_k
        })

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()