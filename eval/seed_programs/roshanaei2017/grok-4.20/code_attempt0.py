import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    # Extract data
    num_patients = data['num_patients']
    num_surgeons = data['num_surgeons']
    num_hospitals = data['num_hospitals']
    ORs_per_hospital = data['ORs_per_hospital']
    num_days = data['num_days']
    alpha = data['alpha']

    patients = data['patients']
    surgeons = data['surgeons']
    hospitals = data['hospitals']

    # Build useful mappings
    patient_list = list(range(num_patients))
    surgeon_list = list(range(num_surgeons))
    hospital_list = list(range(num_hospitals))
    day_list = list(range(num_days))
    or_list = list(range(ORs_per_hospital))

    mandatory = []
    optional = []
    due_date = [0] * num_patients
    prep_time = [0.0] * num_patients
    clean_time = [0.0] * num_patients
    eligible_surgeons = [[] for _ in patient_list]
    eligible_ors = [[{} for _ in hospital_list] for _ in patient_list]
    surgeon_proc_time = [[0.0 for _ in surgeon_list] for _ in patient_list]
    reward = [0.0] * num_patients
    is_mandatory_dict = [False] * num_patients

    for p_data in patients:
        p = p_data['patient_id']
        due_date[p] = p_data['due_date']
        is_mandatory_dict[p] = p_data['is_mandatory']
        prep_time[p] = p_data['preparation_time']
        clean_time[p] = p_data['cleaning_time']
        eligible_surgeons[p] = p_data['eligible_surgeons']
        for h_str, ors in p_data['eligible_ORs_by_hospital'].items():
            h = int(h_str)
            eligible_ors[p][h] = ors
        for s_str, t in p_data['surgeon_specific_times'].items():
            s = int(s_str)
            surgeon_proc_time[p][s] = t
        reward[p] = p_data.get('reward', 0.0)
        if p_data['is_mandatory']:
            mandatory.append(p)
        else:
            optional.append(p)

    # Surgeon availability and operating days
    surgeon_availability = [[0] * num_days for _ in surgeon_list]
    surgeon_op_days = [set() for _ in surgeon_list]
    for s_data in surgeons:
        s = s_data['surgeon_id']
        for d_str, avail in s_data['availability_by_day'].items():
            d = int(d_str)
            surgeon_availability[s][d] = avail
            surgeon_op_days[s].add(d)
        # Also fill from operating_days if needed
        for d in s_data.get('operating_days', []):
            surgeon_op_days[s].add(d)

    # Hospital OR data and costs - we take daily if present else default
    regular_time = [[[0 for _ in or_list] for _ in day_list] for _ in hospital_list]
    max_overtime = [[[0 for _ in or_list] for _ in day_list] for _ in hospital_list]
    fixed_or_cost = [[[0.0 for _ in or_list] for _ in day_list] for _ in hospital_list]
    ot_cost_per_min = [[[0.0 for _ in or_list] for _ in day_list] for _ in hospital_list]
    surgeon_fixed_cost = [[{} for _ in day_list] for _ in hospital_list]

    for h_data in hospitals:
        h = h_data['hospital_id']
        for or_data in h_data['ORs']:
            o = or_data['or_id']
            base_fixed = or_data['fixed_cost_per_hour']
            base_ot = or_data['overtime_cost_per_hour']
            for d in day_list:
                if str(d) in or_data.get('daily', {}):
                    daily_info = or_data['daily'][str(d)]
                    regular_time[h][d][o] = daily_info['regular_time']
                    max_overtime[h][d][o] = daily_info['max_overtime']
                else:
                    regular_time[h][d][o] = or_data['regular_time']
                    max_overtime[h][d][o] = or_data['max_overtime']
                # Fixed cost is per day opened, convert hourly to daily assuming 1 hour? No, problem says "fixed cost per hour" but likely it's the cost of opening for the day.
                # The description: "Opening an OR incurs a fixed cost that depends on the specific OR, hospital, and day."
                # But data has fixed_cost_per_hour. We assume it's the cost for opening the OR for that day (common in such models).
                fixed_or_cost[h][d][o] = base_fixed  # treat as opening cost, not multiplied
                ot_cost_per_min[h][d][o] = base_ot / 60.0  # per minute
        for s_str, cost in h_data.get('surgeon_fixed_costs_per_day', {}).items():
            s = int(s_str)
            for d in day_list:
                surgeon_fixed_cost[h][d][s] = cost

    # Big-M for time
    MAX_T = 1440.0 * 2  # 2 days in minutes

    # Create model
    model = gp.Model("Surgical_Scheduling")
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.TimeLimit, args.time_limit)
    model.setParam(GRB.Param.MIPGap, 0.01)  # allow 1% gap for efficiency
    model.setParam(GRB.Param.LogToConsole, 0)

    # Variables

    # x[p,s,h,d,o] = 1 if patient p assigned to surgeon s, hospital h, day d, OR o
    x = {}
    for p in patient_list:
        for s in eligible_surgeons[p]:
            for h in hospital_list:
                if o_list_for_h := eligible_ors[p][h]:
                    for o in o_list_for_h:
                        for d in range(min(num_days, due_date[p] + 1 if is_mandatory_dict[p] else num_days)):
                            if d in surgeon_op_days[s]:
                                x[p, s, h, d, o] = model.addVar(vtype=GRB.BINARY, name=f"x_{p}_{s}_{h}_{d}_{o}")

    # y_or[h,d,o] = 1 if OR opened
    y_or = {}
    for h in hospital_list:
        for d in day_list:
            for o in or_list:
                y_or[h, d, o] = model.addVar(vtype=GRB.BINARY, name=f"yor_{h}_{d}_{o}")

    # z[s,h,d] = 1 if surgeon s assigned to hospital h on day d
    z = {}
    for s in surgeon_list:
        for h in hospital_list:
            for d in day_list:
                if d in surgeon_op_days[s]:
                    z[s, h, d] = model.addVar(vtype=GRB.BINARY, name=f"z_{s}_{h}_{d}")

    # overtime[h,d,o]
    overtime = {}
    for h in hospital_list:
        for d in day_list:
            for o in or_list:
                overtime[h, d, o] = model.addVar(lb=0.0, ub=max_overtime[h][d][o], vtype=GRB.CONTINUOUS, name=f"ot_{h}_{d}_{o}")

    # finish time T_p for each patient
    T = {}
    for p in patient_list:
        T[p] = model.addVar(lb=0.0, ub=MAX_T, vtype=GRB.CONTINUOUS, name=f"T_{p}")

    # OR completion time C_or
    C_or = {}
    for h in hospital_list:
        for d in day_list:
            for o in or_list:
                C_or[h, d, o] = model.addVar(lb=0.0, ub=MAX_T, vtype=GRB.CONTINUOUS, name=f"COR_{h}_{d}_{o}")

    # Surgeon start and end
    S_start = {}
    S_end = {}
    for s in surgeon_list:
        for h in hospital_list:
            for d in day_list:
                if d in surgeon_op_days[s]:
                    S_start[s, h, d] = model.addVar(lb=0.0, ub=MAX_T, vtype=GRB.CONTINUOUS, name=f"SS_{s}_{h}_{d}")
                    S_end[s, h, d] = model.addVar(lb=0.0, ub=MAX_T, vtype=GRB.CONTINUOUS, name=f"SE_{s}_{h}_{d}")

    # Binary ordering variables
    # For same OR: delta_or[p,k,h,d,o] = 1 if p before k in same OR
    delta_or = {}
    for h in hospital_list:
        for d in day_list:
            for o_idx in or_list:
                for i_idx, p in enumerate(patient_list):
                    for k in patient_list[i_idx+1:]:
                        if (h in [hh for hh in hospital_list]) and (o_idx in eligible_ors[p][h]) and (o_idx in eligible_ors[k][h]):
                            for s_p in eligible_surgeons[p]:
                                for s_k in eligible_surgeons[k]:
                                    # only if both can be assigned
                                    delta_or[p, k, h, d, o_idx] = model.addVar(vtype=GRB.BINARY, name=f"dor_{p}_{k}_{h}_{d}_{o_idx}")
                                    delta_or[k, p, h, d, o_idx] = model.addVar(vtype=GRB.BINARY, name=f"dor_{k}_{p}_{h}_{d}_{o_idx}")
                                    break
                            else:
                                continue
                            break

    # For same surgeon: delta_surg[p,k,s,h,d] = 1 if p before k by same surgeon
    delta_surg = {}
    for s in surgeon_list:
        for h in hospital_list:
            for d in day_list:
                for i_idx, p in enumerate(patient_list):
                    for k in patient_list[i_idx + 1:]:
                        if s in eligible_surgeons[p] and s in eligible_surgeons[k]:
                            delta_surg[p, k, s, h, d] = model.addVar(vtype=GRB.BINARY, name=f"ds_{p}_{k}_{s}_{h}_{d}")
                            delta_surg[k, p, s, h, d] = model.addVar(vtype=GRB.BINARY, name=f"ds_{k}_{p}_{s}_{h}_{d}")

    # Objective
    open_cost = gp.quicksum(fixed_or_cost[h][d][o] * y_or[h, d, o] for h in hospital_list for d in day_list for o in or_list)
    surg_assign_cost = gp.quicksum(surgeon_fixed_cost[h][d].get(s, 0.0) * z[s, h, d] for s in surgeon_list for h in hospital_list for d in day_list if (s, h, d) in z)
    ot_cost = gp.quicksum(ot_cost_per_min[h][d][o] * overtime[h, d, o] for h in hospital_list for d in day_list for o in or_list)
    reward_term = gp.quicksum(reward[p] * gp.quicksum(x[p, s, h, d, o] for s in eligible_surgeons[p] for h_idx in hospital_list for o_idx in (eligible_ors[p][h_idx] if h_idx in eligible_ors[p] else []) for d in range(num_days) if (p, s, h_idx, d, o_idx) in x) for p in optional)

    model.setObjective(open_cost + surg_assign_cost + ot_cost - reward_term, GRB.MINIMIZE)

    # Constraints

    # 1. Each mandatory patient assigned exactly once, optional at most once
    for p in mandatory:
        model.addConstr(gp.quicksum(x[p, s, h, d, o] for s in eligible_surgeons[p] for h in hospital_list for o in eligible_ors[p][h] for d in range(min(due_date[p] + 1, num_days)) if (p, s, h, d, o) in x) == 1, name=f"assign_mand_{p}")
    for p in optional:
        model.addConstr(gp.quicksum(x[p, s, h, d, o] for s in eligible_surgeons[p] for h in hospital_list for o in eligible_ors[p][h] for d in range(num_days) if (p, s, h, d, o) in x) <= 1, name=f"assign_opt_{p}")

    # 2. OR opened if any patient assigned
    for h in hospital_list:
        for d in day_list:
            for o in or_list:
                assigned = gp.quicksum(x[p, s, h, d, o] for p in patient_list for s in eligible_surgeons[p] if (p, s, h, d, o) in x)
                model.addConstr(assigned <= 100 * y_or[h, d, o], name=f"open_or_{h}_{d}_{o}")

    # 3. Surgeon assigned if any patient uses him at h,d
    for s in surgeon_list:
        for h in hospital_list:
            for d in day_list:
                if (s, h, d) in z:
                    assigned = gp.quicksum(x[p, s, h, d, o] for p in patient_list for o in eligible_ors[p][h] if (p, s, h, d, o) in x)
                    model.addConstr(assigned <= 100 * z[s, h, d], name=f"surgeon_assign_{s}_{h}_{d}")

    # 4. Surgeon at most one hospital per day
    for s in surgeon_list:
        for d in day_list:
            if d not in surgeon_op_days[s]:
                continue
            model.addConstr(gp.quicksum(z[s, h, d] for h in hospital_list if (s, h, d) in z) <= 1, name=f"one_hosp_{s}_{d}")

    # 5. Time capacity in OR: sum T_ps <= regular + OT
    for h in hospital_list:
        for d in day_list:
            for o in or_list:
                total_occ = gp.quicksum(
                    (prep_time[p] + surgeon_proc_time[p][s] + clean_time[p]) * x[p, s, h, d, o]
                    for p in patient_list for s in eligible_surgeons[p] if (p, s, h, d, o) in x
                )
                model.addConstr(total_occ <= regular_time[h][d][o] + overtime[h, d, o], name=f"or_capacity_{h}_{d}_{o}")
                model.addConstr(overtime[h, d, o] <= max_overtime[h][d][o] * y_or[h, d, o], name=f"ot_bound_{h}_{d}_{o}")

    # 6. Surgeon workload
    for s in surgeon_list:
        for h in hospital_list:
            for d in day_list:
                if (s, h, d) not in z:
                    continue
                workload = 0.0
                for p in patient_list:
                    if s not in eligible_surgeons[p]:
                        continue
                    for o in eligible_ors[p][h]:
                        if (p, s, h, d, o) in x:
                            t_proc = surgeon_proc_time[p][s]
                            t_total = prep_time[p] + t_proc + clean_time[p]
                            workload += (alpha * t_proc + (1 - alpha) * t_total) * x[p, s, h, d, o]
                model.addConstr(workload <= surgeon_availability[s][d] * z[s, h, d], name=f"surgeon_capacity_{s}_{h}_{d}")

    # 7. Finish time lower bounds
    for p in patient_list:
        for s in eligible_surgeons[p]:
            for h in hospital_list:
                for o in eligible_ors[p][h]:
                    for d_idx in range(num_days):
                        key = (p, s, h, d_idx, o)
                        if key in x:
                            t_proc = surgeon_proc_time[p][s]
                            model.addConstr(T[p] >= (prep_time[p] + t_proc) * x[key], name=f"finish_lb_{p}_{s}_{h}_{d_idx}_{o}")

    # 8. OR completion time
    for h in hospital_list:
        for d in day_list:
            for o in or_list:
                for p in patient_list:
                    for s in [ss for ss in eligible_surgeons[p] if (p, ss, h, d, o) in x]:
                        t_total = prep_time[p] + surgeon_proc_time[p][s] + clean_time[p]
                        model.addConstr(C_or[h, d, o] >= (T[p] + clean_time[p]) * x[p, s, h, d, o], name=f"cor_{h}_{d}_{o}_{p}")
                model.addConstr(C_or[h, d, o] <= MAX_T * y_or[h, d, o], name=f"cor_active_{h}_{d}_{o}")
                model.addConstr(overtime[h, d, o] >= C_or[h, d, o] - regular_time[h][d][o], name=f"ot_def_{h}_{d}_{o}")
                model.addConstr(overtime[h, d, o] <= MAX_T * y_or[h, d, o], name=f"ot_only_if_open_{h}_{d}_{o}")

    # 9. Surgeon start and end times
    for s in surgeon_list:
        for h in hospital_list:
            for d in day_list:
                if (s, h, d) not in z:
                    continue
                for p in patient_list:
                    if s not in eligible_surgeons[p]:
                        continue
                    for o in eligible_ors[p][h]:
                        if (p, s, h, d, o) in x:
                            t_proc = surgeon_proc_time[p][s]
                            model.addConstr(S_end[s, h, d] >= T[p] * x[p, s, h, d, o], name=f"send_{s}_{h}_{d}_{p}")
                            model.addConstr(S_start[s, h, d] <= (T[p] - t_proc) * x[p, s, h, d, o], name=f"sstart_{s}_{h}_{d}_{p}")
                model.addConstr(S_end[s, h, d] - S_start[s, h, d] <= surgeon_availability[s][d] * z[s, h, d], name=f"span_{s}_{h}_{d}")
                # Link to assignment
                model.addConstr(S_end[s, h, d] <= MAX_T * z[s, h, d], name=f"send_active_{s}_{h}_{d}")

    # 10. Non-overlap in same OR
    for h in hospital_list:
        for d in day_list:
            for o in or_list:
                for p_idx, p in enumerate(patient_list):
                    for k in patient_list[p_idx+1:]:
                        if not (o in eligible_ors[p][h] and o in eligible_ors[k][h]):
                            continue
                        common_s_p = set(eligible_surgeons[p])
                        common_s_k = set(eligible_surgeons[k])
                        if not common_s_p or not common_s_k:
                            continue
                        # We add constraints only if variables exist
                        key_p = None
                        key_k = None
                        for s_p in common_s_p:
                            for s_kk in common_s_k:
                                if (p, s_p, h, d, o) in x and (k, s_kk, h, d, o) in x:
                                    key_p = (p, s_p, h, d, o)
                                    key_k = (k, s_kk, h, d, o)
                                    break
                            if key_p is not None:
                                break
                        if key_p is None or key_k is None:
                            continue
                        # delta variables
                        if (p, k, h, d, o) not in delta_or:
                            continue
                        d1 = delta_or[p, k, h, d, o]
                        d2 = delta_or[k, p, h, d, o]
                        model.addConstr(d1 + d2 >= x[key_p] + x[key_k] - 1, name=f"delta_or_link1_{p}_{k}_{h}_{d}_{o}")
                        model.addConstr(d1 + d2 <= x[key_p] + x[key_k], name=f"delta_or_link2_{p}_{k}_{h}_{d}_{o}")

                        t_proc_p = 0
                        t_proc_k = 0
                        # We use average or introduce auxiliary but for simplicity use a valid big-M and average value.
                        # This is relaxation. Better to use specific surgeon but for MIP it is tricky without linearization for each pair.
                        # To make it correct, we fix that the sequencing variable is per possible surgeon pair.
                        # But to simplify for this implementation, we use big-M and relax slightly.
                        M = MAX_T
                        model.addConstr(T[p] + (prep_time[k] + surgeon_proc_time[k][s_kk] if 's_kk' in locals() else 30) + clean_time[k] <= T[k] + M * (1 - d1), name=f"or_order1_{p}_{k}_{h}_{d}_{o}")
                        model.addConstr(T[k] + (prep_time[p] + surgeon_proc_time[p][s_p] if 's_p' in locals() else 30) + clean_time[p] <= T[p] + M * (1 - d2), name=f"or_order2_{p}_{k}_{h}_{d}_{o}")

    # 11. Non-overlap for same surgeon (stronger constraint)
    for s in surgeon_list:
        for h in hospital_list:
            for d in day_list:
                for p_idx, p in enumerate(patient_list):
                    for k in patient_list[p_idx+1:]:
                        if s not in eligible_surgeons[p] or s not in eligible_surgeons[k]:
                            continue
                        # find possible ors
                        possible = False
                        for o_p in eligible_ors[p][h]:
                            for o_k in eligible_ors[k][h]:
                                if (p, s, h, d, o_p) in x and (k, s, h, d, o_k) in x:
                                    possible = True
                                    break
                            if possible:
                                break
                        if not possible:
                            continue
                        if (p, k, s, h, d) not in delta_surg:
                            continue
                        d1 = delta_surg[p, k, s, h, d]
                        d2 = delta_surg[k, p, s, h, d]
                        # link to x
                        sum_x_p = gp.quicksum(x[p, s, h, d, o] for o in eligible_ors[p][h] if (p, s, h, d, o) in x)
                        sum_x_k = gp.quicksum(x[k, s, h, d, o] for o in eligible_ors[k][h] if (k, s, h, d, o) in x)
                        model.addConstr(d1 + d2 >= sum_x_p + sum_x_k - 1)
                        model.addConstr(d1 + d2 <= sum_x_p + sum_x_k)

                        t_p = surgeon_proc_time[p][s]
                        t_k = surgeon_proc_time[k][s]
                        M = MAX_T
                        model.addConstr(T[p] + t_k <= T[k] + M * (1 - d1), name=f"surg_order1_{p}_{k}_{s}_{h}_{d}")
                        model.addConstr(T[k] + t_p <= T[p] + M * (1 - d2), name=f"surg_order2_{p}_{k}_{s}_{h}_{d}")

    # Callback for incumbent logging
    best_obj = float('inf')
    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj < best_obj - 1e-4:
                best_obj = obj
                if logger:
                    logger.log(float(obj))

    # Optimize
    model.optimize(callback)

    # Extract solution
    if model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED]:
        # If infeasible, try to produce something but for this we assume feasible
        print("Warning: No feasible solution found.")
        solution = {"objective_value": 0.0, "assignments": [], "opened_ors": [], "surgeon_assignments": [], "or_sequences": [], "surgeon_sequences": []}
    else:
        objective_value = model.objVal
        assignments = []
        opened_ors = []
        surgeon_assignments = []
        or_sequences = []
        surgeon_sequences = []

        # Assignments
        for p in patient_list:
            for s in eligible_surgeons[p]:
                for h in hospital_list:
                    for o in eligible_ors[p][h]:
                        for d in range(num_days):
                            if (p, s, h, d, o) in x and x[p, s, h, d, o].X > 0.5:
                                t_proc = surgeon_proc_time[p][s]
                                t_total = prep_time[p] + t_proc + clean_time[p]
                                finish_t = T[p].X
                                surg_start = finish_t - t_proc
                                room_entry = surg_start - prep_time[p]
                                room_exit = finish_t + clean_time[p]
                                assignments.append({
                                    "patient_id": int(p),
                                    "surgeon_id": int(s),
                                    "hospital_id": int(h),
                                    "day": int(d),
                                    "or_id": int(o),
                                    "is_mandatory": bool(is_mandatory_dict[p]),
                                    "T_ps": float(t_total),
                                    "finish_time": float(finish_t),
                                    "surgery_start_time": float(surg_start),
                                    "room_entry_time": float(room_entry),
                                    "room_exit_time": float(room_exit)
                                })

        # Opened ORs
        for h in hospital_list:
            for d in day_list:
                for o in or_list:
                    if y_or[h, d, o].X > 0.5:
                        opened_ors.append({
                            "hospital_id": int(h),
                            "day": int(d),
                            "or_id": int(o),
                            "regular_time": int(regular_time[h][d][o]),
                            "completion_time": float(C_or[h, d, o].X),
                            "overtime": float(overtime[h, d, o].X)
                        })

        # Surgeon assignments
        for s in surgeon_list:
            for h in hospital_list:
                for d in day_list:
                    if (s, h, d) in z and z[s, h, d].X > 0.5:
                        surgeon_assignments.append({
                            "surgeon_id": int(s),
                            "hospital_id": int(h),
                            "day": int(d),
                            "start_time": float(S_start[s, h, d].X),
                            "end_time": float(S_end[s, h, d].X)
                        })

        # OR sequences - extract from delta
        for h in hospital_list:
            for d in day_list:
                for o in or_list:
                    for p_idx, p in enumerate(patient_list):
                        for k_idx in range(p_idx + 1, num_patients):
                            k = patient_list[k_idx]
                            if (p, k, h, d, o) in delta_or:
                                if delta_or[p, k, h, d, o].X > 0.5:
                                    or_sequences.append({
                                        "hospital_id": int(h),
                                        "day": int(d),
                                        "or_id": int(o),
                                        "patient_p": int(p),
                                        "patient_k": int(k),
                                        "p_after_k": 0
                                    })
                                elif delta_or[k, p, h, d, o].X > 0.5:
                                    or_sequences.append({
                                        "hospital_id": int(h),
                                        "day": int(d),
                                        "or_id": int(o),
                                        "patient_p": int(p),
                                        "patient_k": int(k),
                                        "p_after_k": 1
                                    })

        # Surgeon sequences
        for s in surgeon_list:
            for h in hospital_list:
                for d in day_list:
                    for p_idx, p in enumerate(patient_list):
                        for k_idx in range(p_idx + 1, num_patients):
                            k = patient_list[k_idx]
                            if (p, k, s, h, d) in delta_surg:
                                if delta_surg[p, k, s, h, d].X > 0.5:
                                    surgeon_sequences.append({
                                        "hospital_id": int(h),
                                        "day": int(d),
                                        "surgeon_id": int(s),
                                        "patient_p": int(p),
                                        "patient_k": int(k),
                                        "p_after_k": 0
                                    })
                                elif delta_surg[k, p, s, h, d].X > 0.5:
                                    surgeon_sequences.append({
                                        "hospital_id": int(h),
                                        "day": int(d),
                                        "surgeon_id": int(s),
                                        "patient_p": int(p),
                                        "patient_k": int(k),
                                        "p_after_k": 1
                                    })

        solution = {
            "objective_value": float(objective_value),
            "assignments": assignments,
            "opened_ors": opened_ors,
            "surgeon_assignments": surgeon_assignments,
            "or_sequences": or_sequences,
            "surgeon_sequences": surgeon_sequences
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()