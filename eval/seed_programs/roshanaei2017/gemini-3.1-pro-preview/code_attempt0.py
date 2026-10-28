import json
import argparse
import sys
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def main():
    parser = argparse.ArgumentParser(description="Solve Hospital Surgical Scheduling Multi-Day Horizon")
    parser.add_argument('--instance_path', type=str, required=True, help="Path to the user instance JSON file.")
    parser.add_argument('--solution_path', type=str, required=True, help="Path where final output JSON must be saved.")
    parser.add_argument('--time_limit', type=int, required=True, help="Max runtime in seconds.")
    parser.add_argument('--log_path', type=str, default=None, help="Path for JSONL logging.")
    
    args = parser.parse_args()
    
    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")
        
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
        
    num_days = data['num_days']
    alpha = data['alpha']
    patients = data['patients']
    surgeons = data['surgeons']
    hospitals = data['hospitals']
    
    hosp_dict = {}
    for h in hospitals:
        hid = h['hospital_id']
        ors = {}
        for r in h['ORs']:
            ors[r['or_id']] = r
        hosp_dict[hid] = {
            'ORs': ors,
            'surgeon_fixed_costs_per_day': h.get('surgeon_fixed_costs_per_day', {})
        }
        
    model = gp.Model("Scheduling")
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    
    # Optional logic for reducing messages unless debugging
    model.setParam('LogToConsole', 1)
    
    X = {}
    Y = {}
    Z = {}
    F = {}
    C = {}
    OT = {}
    
    start_S = {}
    end_S = {}
    bOR = {}
    bSurg = {}
    
    # Identify variables and valid assignments
    for p in patients:
        pid = p['patient_id']
        is_man = p['is_mandatory']
        due = p['due_date']
        F[pid] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"f_{pid}")
        
        for s_id in p['eligible_surgeons']:
            s_data = next((s for s in surgeons if s['surgeon_id'] == s_id), None)
            if not s_data: continue
            
            for d in s_data['operating_days']:
                if d > num_days: continue
                if is_man and d > due: continue
                
                # Surgeon available at which hospitals?
                for h_str, r_list in p['eligible_ORs_by_hospital'].items():
                    hid = int(h_str)
                    h_data = hosp_dict[hid]
                    if str(s_id) not in h_data['surgeon_fixed_costs_per_day']: continue
                    
                    for rid in r_list:
                        # Check OR constraints
                        r_data = h_data['ORs'][rid]
                        d_str = str(d)
                        reg = r_data.get('daily', {}).get(d_str, {}).get('regular_time', r_data['regular_time'])
                        if reg >= 0:
                            X[pid, s_id, hid, d, rid] = model.addVar(vtype=GRB.BINARY, name=f"x_{pid}_{s_id}_{hid}_{d}_{rid}")
                            
    for hid, h_data in hosp_dict.items():
        for r_data in h_data['ORs'].values():
            rid = r_data['or_id']
            for d in range(1, num_days + 1):
                Y[hid, rid, d] = model.addVar(vtype=GRB.BINARY, name=f"y_{hid}_{rid}_{d}")
                C[hid, rid, d] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"c_{hid}_{rid}_{d}")
                OT[hid, rid, d] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"ot_{hid}_{rid}_{d}")
                
    for s in surgeons:
        sid = s['surgeon_id']
        for d in s['operating_days']:
            if d > num_days: continue
            for hid in hosp_dict.keys():
                if str(sid) in hosp_dict[hid]['surgeon_fixed_costs_per_day']:
                    Z[sid, hid, d] = model.addVar(vtype=GRB.BINARY, name=f"z_{sid}_{hid}_{d}")
                    start_S[sid, hid, d] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"ss_{sid}_{hid}_{d}")
                    end_S[sid, hid, d] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"es_{sid}_{hid}_{d}")

    # Patient Assignments constraints
    X_p = { p['patient_id']: [] for p in patients }
    for k in X.keys():
        X_p[k[0]].append(X[k])
        
    for p in patients:
        pid = p['patient_id']
        if p['is_mandatory']:
            model.addConstr(gp.quicksum(X_p[pid]) == 1)
        else:
            model.addConstr(gp.quicksum(X_p[pid]) <= 1)
            
    # Relational linking
    for t, var in X.items():
        pid, sid, hid, d, rid = t
        model.addConstr(var <= Y[hid, rid, d])
        model.addConstr(var <= Z[sid, hid, d])

    # Surgeon distinct location constraint
    for s in surgeons:
        sid = s['surgeon_id']
        for d in range(1, num_days + 1):
            h_vars = [Z[sid, hid, d] for hid in hosp_dict.keys() if (sid, hid, d) in Z]
            if h_vars:
                model.addConstr(gp.quicksum(h_vars) <= 1)

    # Patient procedure finishing times
    for p in patients:
        pid = p['patient_id']
        prep = p['preparation_time']
        dur_expr = gp.LinExpr()
        sum_x = gp.LinExpr()
        for t in X:
            if t[0] == pid:
                sid = t[1]
                proc = p['surgeon_specific_times'][str(sid)]
                dur_expr += (prep + proc) * X[t]
                sum_x += X[t]
        model.addConstr(F[pid] >= dur_expr)
        model.addConstr(F[pid] <= 3000 * sum_x)

    # OR Capacity, Schedule and Overtime Bounds
    for hid, h_data in hosp_dict.items():
        for r_data in h_data['ORs'].values():
            rid = r_data['or_id']
            for d in range(1, num_days + 1):
                d_str = str(d)
                reg = r_data.get('daily', {}).get(d_str, {}).get('regular_time', r_data['regular_time'])
                max_ot = r_data.get('daily', {}).get(d_str, {}).get('max_overtime', r_data['max_overtime'])
                
                dur_expr = gp.LinExpr()
                for t in X:
                    if t[2] == hid and t[3] == d and t[4] == rid:
                        pid, sid = t[0], t[1]
                        p_data = next(pt for pt in patients if pt['patient_id'] == pid)
                        T_ps = p_data['preparation_time'] + p_data['surgeon_specific_times'][str(sid)] + p_data['cleaning_time']
                        dur_expr += T_ps * X[t]
                
                model.addConstr(dur_expr <= reg * Y[hid, rid, d] + OT[hid, rid, d])
                model.addConstr(OT[hid, rid, d] <= max_ot * Y[hid, rid, d])
                
                for t in X:
                    if t[2] == hid and t[3] == d and t[4] == rid:
                        pid = t[0]
                        clean = next(pt for pt in patients if pt['patient_id'] == pid)['cleaning_time']
                        model.addConstr(C[hid, rid, d] >= F[pid] + clean - 3000 * (1 - X[t]))
                        
                model.addConstr(OT[hid, rid, d] >= C[hid, rid, d] - reg)
                model.addConstr(C[hid, rid, d] <= 3000 * Y[hid, rid, d])
                
    # Surgeon Span & Workloads Constraints
    for s in surgeons:
        sid = s['surgeon_id']
        for d in s['operating_days']:
            if d > num_days: continue
            avail = s['availability_by_day'].get(str(d), 0)
            
            wl_expr = gp.LinExpr()
            for t in X:
                if t[1] == sid and t[3] == d:
                    pid = t[0]
                    p_data = next(pt for pt in patients if pt['patient_id'] == pid)
                    T_ps = p_data['preparation_time'] + p_data['surgeon_specific_times'][str(sid)] + p_data['cleaning_time']
                    proc = p_data['surgeon_specific_times'][str(sid)]
                    wl = alpha * proc + (1 - alpha) * T_ps
                    wl_expr += wl * X[t]
            model.addConstr(wl_expr <= avail)
            
            for hid in hosp_dict.keys():
                if (sid, hid, d) in Z:
                    for t in X:
                        if t[1] == sid and t[2] == hid and t[3] == d:
                            pid = t[0]
                            p_data = next(pt for pt in patients if pt['patient_id'] == pid)
                            proc = p_data['surgeon_specific_times'][str(sid)]
                            
                            model.addConstr(start_S[sid, hid, d] <= F[pid] - proc + 3000 * (1 - X[t]))
                            model.addConstr(end_S[sid, hid, d] >= F[pid] - 3000 * (1 - X[t]))
                            
                    model.addConstr(end_S[sid, hid, d] - start_S[sid, hid, d] <= avail * Z[sid, hid, d])
                    model.addConstr(start_S[sid, hid, d] <= 3000 * Z[sid, hid, d])
                    model.addConstr(end_S[sid, hid, d] <= 3000 * Z[sid, hid, d])

    # Temporal logic for sequencing avoiding overlapping
    OR_groups = {}
    Surg_groups = {}
    for t in X:
        pid, sid, hid, d, rid = t
        OR_groups.setdefault((hid, d, rid), []).append(pid)
        Surg_groups.setdefault((sid, d), []).append(pid)

    OR_pairs = set()
    for g, p_list in OR_groups.items():
        uniq_p = list(set(p_list))
        for i in range(len(uniq_p)):
            for j in range(i+1, len(uniq_p)):
                p1, p2 = uniq_p[i], uniq_p[j]
                if p1 > p2: p1, p2 = p2, p1
                OR_pairs.add((p1, p2, g[0], g[1], g[2]))
                
    Surg_pairs = set()
    for g, p_list in Surg_groups.items():
        uniq_p = list(set(p_list))
        for i in range(len(uniq_p)):
            for j in range(i+1, len(uniq_p)):
                p1, p2 = uniq_p[i], uniq_p[j]
                if p1 > p2: p1, p2 = p2, p1
                Surg_pairs.add((p1, p2, g[0], g[1]))
    
    unique_OR_pairs = list(set((t[0], t[1]) for t in OR_pairs))
    for p1, p2 in unique_OR_pairs:
        bOR[p1, p2] = model.addVar(vtype=GRB.BINARY, name=f"bor_{p1}_{p2}")
        
    for (p1, p2, hid, d, rid) in OR_pairs:
        X1 = [k for k in X if k[0] == p1 and k[2] == hid and k[3] == d and k[4] == rid]
        X2 = [k for k in X if k[0] == p2 and k[2] == hid and k[3] == d and k[4] == rid]
        if not X1 or not X2: continue
        
        sum_x1 = gp.quicksum(X[k] for k in X1)
        sum_x2 = gp.quicksum(X[k] for k in X2)
        p1_data = next(pt for pt in patients if pt['patient_id'] == p1)
        p2_data = next(pt for pt in patients if pt['patient_id'] == p2)
        
        dur1 = gp.LinExpr()
        for k in X1:
            dur1 += (p1_data['preparation_time'] + p1_data['surgeon_specific_times'][str(k[1])]) * X[k]
        dur2 = gp.LinExpr()
        for k in X2:
            dur2 += (p2_data['preparation_time'] + p2_data['surgeon_specific_times'][str(k[1])]) * X[k]
            
        model.addConstr(F[p1] - dur1 >= F[p2] + p2_data['cleaning_time'] - 3000 * (1 - bOR[p1, p2]) - 3000 * (2 - sum_x1 - sum_x2))
        model.addConstr(F[p2] - dur2 >= F[p1] + p1_data['cleaning_time'] - 3000 * bOR[p1, p2] - 3000 * (2 - sum_x1 - sum_x2))
        
    unique_Surg_pairs = list(set((t[0], t[1]) for t in Surg_pairs))
    for p1, p2 in unique_Surg_pairs:
        bSurg[p1, p2] = model.addVar(vtype=GRB.BINARY, name=f"bsurg_{p1}_{p2}")
        
    for (p1, p2, sid, d) in Surg_pairs:
        X1 = [k for k in X if k[0] == p1 and k[1] == sid and k[3] == d]
        X2 = [k for k in X if k[0] == p2 and k[1] == sid and k[3] == d]
        if not X1 or not X2: continue
        
        sum_x1 = gp.quicksum(X[k] for k in X1)
        sum_x2 = gp.quicksum(X[k] for k in X2)
        p1_data = next(pt for pt in patients if pt['patient_id'] == p1)
        p2_data = next(pt for pt in patients if pt['patient_id'] == p2)
        
        proc1 = gp.LinExpr()
        for k in X1: proc1 += p1_data['surgeon_specific_times'][str(sid)] * X[k]
            
        proc2 = gp.LinExpr()
        for k in X2: proc2 += p2_data['surgeon_specific_times'][str(sid)] * X[k]
            
        model.addConstr(F[p1] - proc1 >= F[p2] - 3000 * (1 - bSurg[p1, p2]) - 3000 * (2 - sum_x1 - sum_x2))
        model.addConstr(F[p2] - proc2 >= F[p1] - 3000 * bSurg[p1, p2] - 3000 * (2 - sum_x1 - sum_x2))

    # Objective logic Construction
    obj = gp.LinExpr()
    
    # Cost for Open ORs
    for hid, h_data in hosp_dict.items():
        for r_data in h_data['ORs'].values():
            rid = r_data['or_id']
            fc_rate = r_data['fixed_cost_per_hour']
            ot_rate = r_data['overtime_cost_per_hour']
            for d in range(1, num_days + 1):
                d_str = str(d)
                reg = r_data.get('daily', {}).get(d_str, {}).get('regular_time', r_data['regular_time'])
                obj += (fc_rate * (reg / 60.0)) * Y[hid, rid, d]
                obj += (ot_rate / 60.0) * OT[hid, rid, d]

    # Sub-costs corresponding to deployed surgeons
    for sid, hid, d in Z:
        cost = hosp_dict[hid]['surgeon_fixed_costs_per_day'][str(sid)]
        obj += cost * Z[sid, hid, d]

    # Deduction for Optional patients
    for p in patients:
        pid = p['patient_id']
        if not p['is_mandatory']:
            obj -= p['reward'] * gp.quicksum(X_p[pid])

    # Tie breaker (to penalize unnecessary gaps in F schedule)
    for pid in F:
        obj += 1e-5 * F[pid]
        
    model.setObjective(obj, GRB.MINIMIZE)
    
    def event_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                raw_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                total_f = sum(model.cbGetSolution(fvar) for fvar in F.values())
                true_obj = raw_obj - 1e-5 * total_f
                if logger:
                    logger.log(true_obj)
            except Exception:
                pass
            
    model.optimize(event_callback)
    
    if model.SolCount > 0:
        assignments = []
        for k, var in X.items():
            if var.X > 0.5:
                pid, sid, hid, d, rid = k
                p_data = next(pt for pt in patients if pt['patient_id'] == pid)
                prep = float(p_data['preparation_time'])
                clean = float(p_data['cleaning_time'])
                proc = float(p_data['surgeon_specific_times'][str(sid)])
                
                f_p = float(F[pid].X)
                assignments.append({
                    "patient_id": int(pid),
                    "surgeon_id": int(sid),
                    "hospital_id": int(hid),
                    "day": int(d),
                    "or_id": int(rid),
                    "is_mandatory": bool(p_data['is_mandatory']),
                    "T_ps": prep + proc + clean,
                    "finish_time": f_p,
                    "surgery_start_time": f_p - proc,
                    "room_entry_time": f_p - proc - prep,
                    "room_exit_time": f_p + clean
                })

        opened_ors = []
        for k, var in Y.items():
            if var.X > 0.5:
                hid, rid, d = k
                r_data = hosp_dict[hid]['ORs'][rid]
                d_str = str(d)
                reg = r_data.get('daily', {}).get(d_str, {}).get('regular_time', r_data['regular_time'])
                
                or_pats = [a for a in assignments if a['hospital_id'] == hid and a['day'] == d and a['or_id'] == rid]
                comp_time = max([a['room_exit_time'] for a in or_pats], default=0.0)
                overtime = float(max(0.0, comp_time - reg))

                opened_ors.append({
                    "hospital_id": int(hid),
                    "day": int(d),
                    "or_id": int(rid),
                    "regular_time": int(reg),
                    "completion_time": float(comp_time),
                    "overtime": overtime
                })

        surgeon_assignments = []
        for k, var in Z.items():
            if var.X > 0.5:
                sid, hid, d = k
                s_pats = [a for a in assignments if a['surgeon_id'] == sid and a['day'] == d and a['hospital_id'] == hid]
                st = min([a['surgery_start_time'] for a in s_pats], default=0.0)
                et = max([a['finish_time'] for a in s_pats], default=0.0)
                
                surgeon_assignments.append({
                    "surgeon_id": int(sid),
                    "hospital_id": int(hid),
                    "day": int(d),
                    "start_time": float(st),
                    "end_time": float(et)
                })
        
        or_sequences = []
        or_assigns = {}
        for a in assignments:
            or_assigns.setdefault((a['hospital_id'], a['day'], a['or_id']), []).append(a)
        
        for key, lst in or_assigns.items():
            hid, d, rid = key
            for i in range(len(lst)):
                for j in range(i+1, len(lst)):
                    p1 = lst[i]
                    p2 = lst[j]
                    p_after_k = 1 if p1['finish_time'] > p2['finish_time'] else 0
                    
                    or_sequences.append({
                        "hospital_id": int(hid),
                        "day": int(d),
                        "or_id": int(rid),
                        "patient_p": int(p1['patient_id']),
                        "patient_k": int(p2['patient_id']),
                        "p_after_k": int(p_after_k)
                    })
                    
        surgeon_sequences = []
        surg_assigns = {}
        for a in assignments:
            surg_assigns.setdefault((a['hospital_id'], a['day'], a['surgeon_id']), []).append(a)
            
        for key, lst in surg_assigns.items():
            hid, d, sid = key
            for i in range(len(lst)):
                for j in range(i+1, len(lst)):
                    p1 = lst[i]
                    p2 = lst[j]
                    p_after_k = 1 if p1['finish_time'] > p2['finish_time'] else 0
                    
                    surgeon_sequences.append({
                        "hospital_id": int(hid),
                        "day": int(d),
                        "surgeon_id": int(sid),
                        "patient_p": int(p1['patient_id']),
                        "patient_k": int(p2['patient_id']),
                        "p_after_k": int(p_after_k)
                    })
                    
        objective_value = 0.0
        for or_inst in opened_ors:
            hid, rid, d = or_inst['hospital_id'], or_inst['or_id'], or_inst['day']
            r_data = hosp_dict[hid]['ORs'][rid]
            fc_rate = float(r_data['fixed_cost_per_hour'])
            ot_rate = float(r_data['overtime_cost_per_hour'])
            reg = float(or_inst['regular_time'])
            objective_value += (fc_rate * (reg / 60.0))
            objective_value += ((ot_rate / 60.0) * or_inst['overtime'])

        for z in surgeon_assignments:
            sid, hid = z['surgeon_id'], z['hospital_id']
            objective_value += float(hosp_dict[hid]['surgeon_fixed_costs_per_day'][str(sid)])

        for a in assignments:
            if not a['is_mandatory']:
                pid = a['patient_id']
                p_data = next(pt for pt in patients if pt['patient_id'] == pid)
                objective_value -= float(p_data['reward'])

        with open(args.solution_path, 'w') as f:
            json.dump({
                "objective_value": float(objective_value),
                "assignments": assignments,
                "opened_ors": opened_ors,
                "surgeon_assignments": surgeon_assignments,
                "or_sequences": or_sequences,
                "surgeon_sequences": surgeon_sequences
            }, f, indent=2)

if __name__ == "__main__":
    main()