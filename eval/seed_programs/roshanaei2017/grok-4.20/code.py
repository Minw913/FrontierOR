import json
import argparse
import time
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

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    # ====================== Data Extraction ======================
    num_p = data['num_patients']
    num_s = data['num_surgeons']
    num_h = data['num_hospitals']
    num_o = data['ORs_per_hospital']
    num_d = data['num_days']
    alpha = data['alpha']

    P = list(range(num_p))
    S = list(range(num_s))
    H = list(range(num_h))
    O = list(range(num_o))
    D = list(range(num_d))

    patients = data['patients']
    due = [p['due_date'] for p in patients]
    is_mandatory = [p['is_mandatory'] for p in patients]
    prep = [p['preparation_time'] for p in patients]
    clean = [p['cleaning_time'] for p in patients]
    reward = [p.get('reward', 0.0) for p in patients]
    elig_surgeons = [p['eligible_surgeons'] for p in patients]
    surgeon_specific_times = [p['surgeon_specific_times'] for p in patients]

    elig_ors = [[{} for _ in H] for _ in P]
    for p_data in patients:
        pid = p_data['patient_id']
        for h_str, ors in p_data['eligible_ORs_by_hospital'].items():
            elig_ors[pid][int(h_str)] = ors

    proc = [[0.0 for _ in S] for _ in P]
    for p_idx, p_data in enumerate(patients):
        for s_str, t in p_data['surgeon_specific_times'].items():
            proc[p_idx][int(s_str)] = float(t)

    avail = [[0] * num_d for _ in S]
    op_days = [set() for _ in S]
    for s_data in data['surgeons']:
        sid = s_data['surgeon_id']
        for d_str, mins in s_data['availability_by_day'].items():
            d_idx = int(d_str)
            avail[sid][d_idx] = int(mins)
            op_days[sid].add(d_idx)
        for d_idx in s_data.get('operating_days', []):
            op_days[sid].add(int(d_idx))

    reg_time = [[[0]*num_o for _ in D] for _ in H]
    max_ot = [[[0]*num_o for _ in D] for _ in H]
    fix_or_cost = [[[0.0]*num_o for _ in D] for _ in H]
    ot_cost_per_min = [[[0.0]*num_o for _ in D] for _ in H]
    surg_fix_cost = [[{} for _ in D] for _ in H]

    for h_data in data['hospitals']:
        hid = h_data['hospital_id']
        for or_data in h_data['ORs']:
            oid = or_data['or_id']
            base_reg = or_data['regular_time']
            base_maxot = or_data['max_overtime']
            fcost = or_data['fixed_cost_per_hour']
            ocost = or_data['overtime_cost_per_hour'] / 60.0

            for d_idx in D:
                day_key = str(d_idx)
                if 'daily' in or_data and day_key in or_data['daily']:
                    info = or_data['daily'][day_key]
                    reg_time[hid][d_idx][oid] = info['regular_time']
                    max_ot[hid][d_idx][oid] = info['max_overtime']
                else:
                    reg_time[hid][d_idx][oid] = base_reg
                    max_ot[hid][d_idx][oid] = base_maxot
                fix_or_cost[hid][d_idx][oid] = fcost
                ot_cost_per_min[hid][d_idx][oid] = ocost

        for s_str, cost in h_data.get('surgeon_fixed_costs_per_day', {}).items():
            sid = int(s_str)
            for d_idx in D:
                surg_fix_cost[hid][d_idx][sid] = float(cost)

    MAX_T = 1440.0 * 2
    M = MAX_T * 2

    # ====================== Model ======================
    model = gp.Model("Surgical_Scheduling")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.Params.TimeLimit = max(10, args.time_limit - 10)
    model.Params.LogToConsole = 0
    model.Params.MIPFocus = 1

    # Variables
    x = {}
    for p in P:
        max_d = min(due[p] + 1, num_d) if is_mandatory[p] else num_d
        for s in elig_surgeons[p]:
            if s not in op_days[s]:
                continue
            for h in H:
                for o in elig_ors[p][h]:
                    for d in range(max_d):
                        if d in op_days[s]:
                            x[p, s, h, d, o] = model.addVar(vtype=GRB.BINARY, name=f"x_{p}_{s}_{h}_{d}_{o}")

    y = {}
    for h in H:
        for d in D:
            for o_idx in O:
                y[h, d, o_idx] = model.addVar(vtype=GRB.BINARY, name=f"y_{h}_{d}_{o_idx}")

    z = {}
    for s in S:
        for h in H:
            for d in D:
                if d in op_days[s]:
                    z[s, h, d] = model.addVar(vtype=GRB.BINARY, name=f"z_{s}_{h}_{d}")

    overtime = {}
    for h in H:
        for d in D:
            for o_idx in O:
                overtime[h, d, o_idx] = model.addVar(0.0, max_ot[h][d][o_idx], vtype=GRB.CONTINUOUS,
                                                     name=f"ot_{h}_{d}_{o_idx}")

    finish = {p: model.addVar(0.0, MAX_T, vtype=GRB.CONTINUOUS, name=f"T_{p}") for p in P}
    C_or = {}
    for h in H:
        for d in D:
            for o_idx in O:
                C_or[h, d, o_idx] = model.addVar(0.0, MAX_T, vtype=GRB.CONTINUOUS, name=f"C_{h}_{d}_{o_idx}")

    S_start = {}
    S_end = {}
    for s, h, d in z:
        S_start[s, h, d] = model.addVar(0.0, MAX_T, vtype=GRB.CONTINUOUS, name=f"SS_{s}_{h}_{d}")
        S_end[s, h, d] = model.addVar(0.0, MAX_T, vtype=GRB.CONTINUOUS, name=f"SE_{s}_{h}_{d}")

    # Sequencing variables (only when both patients can be together)
    delta_or = {}
    delta_surg = {}

    for p1 in P:
        for p2 in P:
            if p1 >= p2: 
                continue
            common_s = set(elig_surgeons[p1]) & set(elig_surgeons[p2])
            for s in common_s:
                for h in H:
                    common_o = set(elig_ors[p1][h]) & set(elig_ors[p2][h])
                    for d in D:
                        if d not in op_days[s]: 
                            continue
                        for o_idx in common_o:
                            key = (p1, p2, h, d, o_idx)
                            delta_or[key] = model.addVar(vtype=GRB.BINARY, name=f"orseq_{p1}_{p2}_{h}_{d}_{o_idx}")
                            delta_or[(p2, p1, h, d, o_idx)] = model.addVar(vtype=GRB.BINARY,
                                                                            name=f"orseq_{p2}_{p1}_{h}_{d}_{o_idx}")
                            break
                    key_s = (p1, p2, s, h, d)
                    if key_s not in delta_surg:
                        delta_surg[key_s] = model.addVar(vtype=GRB.BINARY, name=f"surgseq_{p1}_{p2}_{s}_{h}_{d}")
                        delta_surg[(p2, p1, s, h, d)] = model.addVar(vtype=GRB.BINARY,
                                                                      name=f"surgseq_{p2}_{p1}_{s}_{h}_{d}")

    # ====================== Objective ======================
    cost_open = gp.quicksum(fix_or_cost[h][d][o] * y[h, d, o] for h in H for d in D for o in O)
    cost_surg = gp.quicksum(surg_fix_cost[h][d].get(s, 0.0) * z[s, h, d] for s, h, d in z)
    cost_ot = gp.quicksum(ot_cost_per_min[h][d][o] * overtime[h, d, o] for h in H for d in D for o in O)
    total_reward = gp.quicksum(reward[p] * gp.quicksum(x[k] for k in x if k[0] == p) for p in P if not is_mandatory[p])

    model.setObjective(cost_open + cost_surg + cost_ot - total_reward, GRB.MINIMIZE)

    # ====================== Constraints ======================
    # Each mandatory patient assigned exactly once
    for p in P:
        if is_mandatory[p]:
            model.addConstr(gp.quicksum(x[k] for k in x if k[0] == p) == 1, f"assign_m_{p}")
        else:
            model.addConstr(gp.quicksum(x[k] for k in x if k[0] == p) <= 1, f"assign_o_{p}")

    # Open OR if used
    for h in H:
        for d in D:
            for o_idx in O:
                used = gp.quicksum(x.get((p, s, h, d, o_idx), 0) 
                                   for p in P for s in elig_surgeons[p])
                model.addConstr(used <= 100 * y[h, d, o_idx], f"open_or_{h}_{d}_{o_idx}")

    # Surgeon-hospital-day linking
    for s, h, d in z:
        used = gp.quicksum(x.get((p, s, h, d, o_idx), 0) for p in P for o_idx in elig_ors[p][h])
        model.addConstr(used <= 50 * z[s, h, d], f"zlink_{s}_{h}_{d}")

    # One hospital per surgeon per day
    for s in S:
        for d in D:
            if d not in op_days[s]:
                continue
            model.addConstr(gp.quicksum(z.get((s, h, d), 0) for h in H) <= 1, f"one_hosp_{s}_{d}")

    # OR capacity
    for h in H:
        for d in D:
            for o_idx in O:
                total_time = gp.quicksum((prep[p] + proc[p][s] + clean[p]) * x.get((p, s, h, d, o_idx), 0)
                                         for p in P for s in elig_surgeons[p])
                model.addConstr(total_time <= reg_time[h][d][o_idx] + overtime[h, d, o_idx],
                                f"or_cap_{h}_{d}_{o_idx}")
                model.addConstr(overtime[h, d, o_idx] <= max_ot[h][d][o_idx] * y[h, d, o_idx])

    # Surgeon workload
    for s, h, d in z:
        workload = gp.quicksum((alpha * proc[p][s] + (1 - alpha) * (prep[p] + proc[p][s] + clean[p])) *
                               x.get((p, s, h, d, o_idx), 0)
                               for p in P for o_idx in elig_ors[p][h])
        model.addConstr(workload <= avail[s][d] * z[s, h, d], f"surgeon_work_{s}_{h}_{d}")

    # Finish time lower bound
    for p in P:
        for s in elig_surgeons[p]:
            for h in H:
                for o_idx in elig_ors[p][h]:
                    for d_idx in D:
                        if (p, s, h, d_idx, o_idx) in x:
                            model.addConstr(finish[p] >= (prep[p] + proc[p][s]) * x[p, s, h, d_idx, o_idx])

    # OR completion time
    for h in H:
        for d in D:
            for o_idx in O:
                for p in P:
                    for s in elig_surgeons[p]:
                        if (p, s, h, d, o_idx) in x:
                            model.addConstr(C_or[h, d, o_idx] >= (finish[p] + clean[p]) * x[p, s, h, d, o_idx])
                model.addConstr(C_or[h, d, o_idx] <= MAX_T * y[h, d, o_idx])
                model.addConstr(overtime[h, d, o_idx] >= C_or[h, d, o_idx] - reg_time[h][d][o_idx])
                model.addConstr(overtime[h, d, o_idx] <= MAX_T * y[h, d, o_idx])

    # Surgeon start/end times
    for s, h, d in z:
        for p in P:
            if s not in elig_surgeons[p]:
                continue
            for o_idx in elig_ors[p][h]:
                if (p, s, h, d, o_idx) in x:
                    tp = proc[p][s]
                    model.addConstr(S_end[s, h, d] >= finish[p] * x[p, s, h, d, o_idx])
                    model.addConstr(S_start[s, h, d] <= (finish[p] - tp) * x[p, s, h, d, o_idx])
        model.addConstr(S_end[s, h, d] - S_start[s, h, d] <= avail[s][d] * z[s, h, d] + 1e-4)
        model.addConstr(S_end[s, h, d] <= MAX_T * z[s, h, d])

    # Sequencing in same OR
    for (p1, p2, h, d, o_idx), var in list(delta_or.items()):
        if p1 >= p2: 
            continue
        d1 = delta_or[(p1, p2, h, d, o_idx)]
        d2 = delta_or[(p2, p1, h, d, o_idx)]
        xp1 = x.get((p1, s, h, d, o_idx), 0)  # s will be fixed in loop below
        xp2 = x.get((p2, s, h, d, o_idx), 0)
        model.addConstr(d1 + d2 >= xp1 + xp2 - 1)
        model.addConstr(d1 + d2 <= xp1 + xp2)

        for s_cand in set(elig_surgeons[p1]) & set(elig_surgeons[p2]):
            if (p1, s_cand, h, d, o_idx) in x and (p2, s_cand, h, d, o_idx) in x:
                tp1 = proc[p1][s_cand]
                tp2 = proc[p2][s_cand]
                model.addConstr(finish[p1] + (prep[p2] + tp2 + clean[p2]) <= finish[p2] + M * (1 - d1))
                model.addConstr(finish[p2] + (prep[p1] + tp1 + clean[p1]) <= finish[p1] + M * (1 - d2))
                break

    # Sequencing for same surgeon
    for (p1, p2, s, h, d), var in list(delta_surg.items()):
        if p1 >= p2: 
            continue
        d1 = delta_surg[(p1, p2, s, h, d)]
        d2 = delta_surg[(p2, p1, s, h, d)]
        sumx1 = gp.quicksum(x.get((p1, s, h, d, o_idx), 0) for o_idx in elig_ors[p1][h])
        sumx2 = gp.quicksum(x.get((p2, s, h, d, o_idx), 0) for o_idx in elig_ors[p2][h])
        model.addConstr(d1 + d2 >= sumx1 + sumx2 - 1)
        model.addConstr(d1 + d2 <= sumx1 + sumx2)

        tp1 = proc[p1][s]
        tp2 = proc[p2][s]
        model.addConstr(finish[p1] + tp2 <= finish[p2] + M * (1 - d1))
        model.addConstr(finish[p2] + tp1 <= finish[p1] + M * (1 - d2))

    # ====================== Callback ======================
    best_obj = float('inf')
    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj_val < best_obj - 1e-6:
                best_obj = obj_val
                if logger:
                    logger.log(float(obj_val))

    model.optimize(callback)

    # ====================== Extract Solution ======================
    if model.SolCount == 0:
        solution = {
            "objective_value": 0.0,
            "assignments": [],
            "opened_ors": [],
            "surgeon_assignments": [],
            "or_sequences": [],
            "surgeon_sequences": []
        }
    else:
        obj_val = model.objVal
        assignments = []
        opened_ors = []
        surgeon_assignments = []
        or_sequences = []
        surgeon_sequences = []

        # Assignments
        for (p, s, h, d, o_idx), var in x.items():
            if var.X > 0.5:
                tp = proc[p][s]
                t_total = prep[p] + tp + clean[p]
                f_time = finish[p].X
                assignments.append({
                    "patient_id": int(p),
                    "surgeon_id": int(s),
                    "hospital_id": int(h),
                    "day": int(d),
                    "or_id": int(o_idx),
                    "is_mandatory": bool(is_mandatory[p]),
                    "T_ps": float(t_total),
                    "finish_time": float(f_time),
                    "surgery_start_time": float(f_time - tp),
                    "room_entry_time": float(f_time - tp - prep[p]),
                    "room_exit_time": float(f_time + clean[p])
                })

        # Opened ORs
        for h in H:
            for d in D:
                for o_idx in O:
                    if y[h, d, o_idx].X > 0.5:
                        opened_ors.append({
                            "hospital_id": int(h),
                            "day": int(d),
                            "or_id": int(o_idx),
                            "regular_time": int(reg_time[h][d][o_idx]),
                            "completion_time": float(C_or[h, d, o_idx].X),
                            "overtime": float(overtime[h, d, o_idx].X)
                        })

        # Surgeon assignments
        for (s, h, d), var in z.items():
            if var.X > 0.5:
                surgeon_assignments.append({
                    "surgeon_id": int(s),
                    "hospital_id": int(h),
                    "day": int(d),
                    "start_time": float(S_start[s, h, d].X),
                    "end_time": float(S_end[s, h, d].X)
                })

        # OR sequences
        for (p1, p2, h, d, o_idx), var in delta_or.items():
            if p1 < p2 and var.X > 0.5:
                or_sequences.append({
                    "hospital_id": int(h),
                    "day": int(d),
                    "or_id": int(o_idx),
                    "patient_p": int(p1),
                    "patient_k": int(p2),
                    "p_after_k": 0
                })

        # Surgeon sequences
        for (p1, p2, s, h, d), var in delta_surg.items():
            if p1 < p2 and var.X > 0.5:
                surgeon_sequences.append({
                    "hospital_id": int(h),
                    "day": int(d),
                    "surgeon_id": int(s),
                    "patient_p": int(p1),
                    "patient_k": int(p2),
                    "p_after_k": 0
                })

        solution = {
            "objective_value": float(obj_val),
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