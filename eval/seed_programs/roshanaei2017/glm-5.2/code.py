import json
import argparse
from gurobipy import Model, GRB, quicksum

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, default=300)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except Exception:
        logger = None

    with open(args.instance_path) as fh:
        inst = json.load(fh)

    P = inst['num_patients']
    S = inst['num_surgeons']
    H = inst['num_hospitals']
    O_per = inst['ORs_per_hospital']
    D = inst['num_days']
    alpha = inst['alpha']

    patients = inst['patients']
    surgeons = inst['surgeons']
    hospitals = inst['hospitals']

    patients_by_id = {pat['patient_id']: pat for pat in patients}
    surgeons_by_id = {surg['surgeon_id']: surg for surg in surgeons}
    hospitals_by_id = {hosp['hospital_id']: hosp for hosp in hospitals}

    patient_ids = [pat['patient_id'] for pat in patients]
    surgeon_ids = [surg['surgeon_id'] for surg in surgeons]
    hospital_ids = [hosp['hospital_id'] for hosp in hospitals]

    h_or_ids = {hid: [od['or_id'] for od in hospitals_by_id[hid]['ORs']] for hid in hospital_ids}
    h_or_info = {hid: {od['or_id']: od for od in hospitals_by_id[hid]['ORs']} for hid in hospital_ids}

    def get_or_reg_max(hid, oid, d):
        od = h_or_info[hid][oid]
        rt = od['regular_time']
        mo = od['max_overtime']
        daily = od.get('daily', {})
        if str(d) in daily:
            dd = daily[str(d)]
            rt = dd['regular_time']
            mo = dd['max_overtime']
        return rt, mo

    def get_or_costs(hid, oid):
        od = h_or_info[hid][oid]
        return od['fixed_cost_per_hour'], od['overtime_cost_per_hour']

    all_days = sorted(set(d for surg in surgeons for d in surg['operating_days']))
    max_day = max(all_days) if all_days else -1

    surg_time = {}
    for pid in patient_ids:
        pat = patients_by_id[pid]
        for sid in pat['eligible_surgeons']:
            surg_time[(pid, sid)] = pat['surgeon_specific_times'].get(str(sid), pat.get('base_surgical_time', 0))

    def T_ps(pid, sid):
        return patients_by_id[pid]['preparation_time'] + surg_time[(pid, sid)] + patients_by_id[pid]['cleaning_time']

    feas_x = []
    feas_x_set = set()
    feas_by_p = {pid: [] for pid in patient_ids}
    feas_by_sghd = {}
    or_capacity_terms = {}
    for pid in patient_ids:
        pat = patients_by_id[pid]
        dd = pat['due_date']
        max_d = min(dd, max_day) if max_day >= 0 else -1
        if max_d < 0:
            continue
        for sid in pat['eligible_surgeons']:
            surg = surgeons_by_id[sid]
            op_days = set(surg['operating_days'])
            for d in all_days:
                if d > max_d:
                    continue
                if d not in op_days:
                    continue
                for hid in hospital_ids:
                    ors_p = pat['eligible_ORs_by_hospital'].get(str(hid), [])
                    for oid in ors_p:
                        tpl = (pid, sid, hid, d, oid)
                        feas_x.append(tpl)
                        feas_x_set.add(tpl)
                        feas_by_p[pid].append((sid, hid, d, oid))
                        feas_by_sghd.setdefault((sid, hid, d), []).append((pid, oid))
                        or_capacity_terms.setdefault((hid, d, oid), []).append((pid, sid))

    M_orig = 0
    for hid in hospital_ids:
        for oid in h_or_ids[hid]:
            for d in all_days:
                rt, mo = get_or_reg_max(hid, oid, d)
                M_orig = max(M_orig, rt + mo)
    max_small = 0
    for (pid, sid) in surg_time:
        pat = patients_by_id[pid]
        s = surg_time[(pid, sid)]
        max_small = max(max_small, pat['preparation_time'] + pat['cleaning_time'] + s)
    M = M_orig + max_small + 100
    if M < 2000:
        M = 2000

    model = Model('surgery')
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', max(args.time_limit - 3, 1))
    model.setParam('OutputFlag', 0)

    x = {}
    for tpl in feas_x:
        x[tpl] = model.addVar(vtype=GRB.BINARY, name='x_%s' % (tpl,))

    z = {}
    for sid in surgeon_ids:
        for d in surgeons_by_id[sid]['operating_days']:
            for hid in hospital_ids:
                z[(sid, hid, d)] = model.addVar(vtype=GRB.BINARY, name='z_%s_%s_%s' % (sid, hid, d))

    y = {}
    overtime = {}
    C = {}
    for hid in hospital_ids:
        for oid in h_or_ids[hid]:
            for d in all_days:
                y[(hid, d, oid)] = model.addVar(vtype=GRB.BINARY, name='y_%s_%s_%s' % (hid, d, oid))
                overtime[(hid, d, oid)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='ot_%s_%s_%s' % (hid, d, oid))
                C[(hid, d, oid)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='C_%s_%s_%s' % (hid, d, oid))

    f = {pid: model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='f_%s' % pid) for pid in patient_ids}

    st = {}
    et = {}
    for sid in surgeon_ids:
        for d in surgeons_by_id[sid]['operating_days']:
            for hid in hospital_ids:
                st[(sid, hid, d)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='st_%s_%s_%s' % (sid, hid, d))
                et[(sid, hid, d)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='et_%s_%s_%s' % (sid, hid, d))

    slack = {}
    for pid in patient_ids:
        if patients_by_id[pid]['is_mandatory']:
            slack[pid] = model.addVar(vtype=GRB.BINARY, name='slack_%s' % pid)

    or_seq = {}
    sg_seq = {}

    # Assignment constraints
    for pid in patient_ids:
        pat = patients_by_id[pid]
        terms = [x[(pid, sid, hid, d, oid)] for (sid, hid, d, oid) in feas_by_p[pid]]
        if pat['is_mandatory']:
            model.addConstr(quicksum(terms) + slack[pid] == 1, name='assign_%s' % pid)
        else:
            model.addConstr(quicksum(terms) <= 1, name='assign_%s' % pid)

    # Surgeon one hospital per day
    for sid in surgeon_ids:
        for d in surgeons_by_id[sid]['operating_days']:
            terms = [z[(sid, hid, d)] for hid in hospital_ids]
            model.addConstr(quicksum(terms) <= 1, name='surg_one_%s_%s' % (sid, d))

    # Link x-z, x-y
    for (pid, sid, hid, d, oid) in feas_x:
        model.addConstr(x[(pid, sid, hid, d, oid)] <= z[(sid, hid, d)])
        model.addConstr(x[(pid, sid, hid, d, oid)] <= y[(hid, d, oid)])

    # OR capacity and overtime bounds
    for (hid, d, oid), ps_list in or_capacity_terms.items():
        rt, mo = get_or_reg_max(hid, oid, d)
        terms = [x[(pid, sid, hid, d, oid)] * T_ps(pid, sid) for (pid, sid) in ps_list]
        model.addConstr(quicksum(terms) <= rt + overtime[(hid, d, oid)], name='cap_%s_%s_%s' % (hid, d, oid))
        model.addConstr(overtime[(hid, d, oid)] <= mo * y[(hid, d, oid)], name='otb_%s_%s_%s' % (hid, d, oid))

    # OR completion and overtime definition
    for hid in hospital_ids:
        for oid in h_or_ids[hid]:
            for d in all_days:
                rt, mo = get_or_reg_max(hid, oid, d)
                for (pid, sid) in or_capacity_terms.get((hid, d, oid), []):
                    in_or = quicksum(x[(pid, ss, hid, d, oid)] for ss in patients_by_id[pid]['eligible_surgeons'] if (pid, ss, hid, d, oid) in feas_x_set)
                    model.addConstr(C[(hid, d, oid)] >= f[pid] + patients_by_id[pid]['cleaning_time'] - M * (1 - in_or))
                model.addConstr(C[(hid, d, oid)] <= (rt + mo) * y[(hid, d, oid)])
                model.addConstr(overtime[(hid, d, oid)] >= C[(hid, d, oid)] - rt - M * (1 - y[(hid, d, oid)]))

    # Patient finish time
    for pid in patient_ids:
        terms = [x[(pid, sid, hid, d, oid)] * (patients_by_id[pid]['preparation_time'] + surg_time[(pid, sid)]) for (sid, hid, d, oid) in feas_by_p[pid]]
        if terms:
            model.addConstr(f[pid] >= quicksum(terms))
        total_assign = quicksum(x[(pid, sid, hid, d, oid)] for (sid, hid, d, oid) in feas_by_p[pid])
        extra = slack[pid] if pid in slack else 0
        model.addConstr(f[pid] <= M * (total_assign + extra))

    # Surgeon workload and start/end
    for (sid, hid, d), pids_oids in feas_by_sghd.items():
        avail = surgeons_by_id[sid]['availability_by_day'].get(str(d), 0)
        wl_terms = []
        for (pid, oid) in pids_oids:
            occ = T_ps(pid, sid)
            sg = surg_time[(pid, sid)]
            w = alpha * sg + (1 - alpha) * occ
            wl_terms.append(x[(pid, sid, hid, d, oid)] * w)
        if wl_terms:
            model.addConstr(quicksum(wl_terms) <= avail * z[(sid, hid, d)])
        pids = set(p for (p, o) in pids_oids)
        for pid in pids:
            in_sg = quicksum(x[(pid, sid, hid, d, oid)] for oid in patients_by_id[pid]['eligible_ORs_by_hospital'].get(str(hid), []) if (pid, sid, hid, d, oid) in feas_x_set)
            model.addConstr(et[(sid, hid, d)] >= f[pid] - M * (1 - in_sg))
            model.addConstr(st[(sid, hid, d)] <= f[pid] - surg_time[(pid, sid)] + M * (1 - in_sg))
        model.addConstr(et[(sid, hid, d)] - st[(sid, hid, d)] <= avail * z[(sid, hid, d)])
        model.addConstr(et[(sid, hid, d)] <= M * z[(sid, hid, d)])
        model.addConstr(st[(sid, hid, d)] <= M * z[(sid, hid, d)])
        model.addConstr(st[(sid, hid, d)] <= et[(sid, hid, d)])

    # OR sequencing disjunctions
    for (hid, d, oid), ps_list in or_capacity_terms.items():
        ps = sorted(set(p for (p, s) in ps_list))
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                p = ps[i]
                k = ps[j]
                b = model.addVar(vtype=GRB.BINARY, name='ors_%s_%s_%s_%s_%s' % (hid, d, oid, p, k))
                or_seq[(hid, d, oid, p, k)] = b
                in_p = quicksum(x[(p, s, hid, d, oid)] for s in patients_by_id[p]['eligible_surgeons'] if (p, s, hid, d, oid) in feas_x_set)
                in_k = quicksum(x[(k, s, hid, d, oid)] for s in patients_by_id[k]['eligible_surgeons'] if (k, s, hid, d, oid) in feas_x_set)
                surg_k = quicksum(x[(k, s, hid, d, oid)] * surg_time[(k, s)] for s in patients_by_id[k]['eligible_surgeons'] if (k, s, hid, d, oid) in feas_x_set)
                surg_p = quicksum(x[(p, s, hid, d, oid)] * surg_time[(p, s)] for s in patients_by_id[p]['eligible_surgeons'] if (p, s, hid, d, oid) in feas_x_set)
                model.addConstr(f[p] + patients_by_id[p]['cleaning_time'] + patients_by_id[k]['preparation_time'] + surg_k <= f[k] + M * (2 - in_p - in_k) + M * (1 - b))
                model.addConstr(f[k] + patients_by_id[k]['cleaning_time'] + patients_by_id[p]['preparation_time'] + surg_p <= f[p] + M * (2 - in_p - in_k) + M * b)

    # Surgeon sequencing disjunctions
    for (sid, hid, d), pids_oids in feas_by_sghd.items():
        pids = sorted(set(p for (p, o) in pids_oids))
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                p = pids[i]
                k = pids[j]
                b = model.addVar(vtype=GRB.BINARY, name='sgs_%s_%s_%s_%s_%s' % (sid, hid, d, p, k))
                sg_seq[(sid, hid, d, p, k)] = b
                in_p = quicksum(x[(p, sid, hid, d, oid)] for oid in patients_by_id[p]['eligible_ORs_by_hospital'].get(str(hid), []) if (p, sid, hid, d, oid) in feas_x_set)
                in_k = quicksum(x[(k, sid, hid, d, oid)] for oid in patients_by_id[k]['eligible_ORs_by_hospital'].get(str(hid), []) if (k, sid, hid, d, oid) in feas_x_set)
                model.addConstr(f[p] + surg_time[(k, sid)] <= f[k] + M * (2 - in_p - in_k) + M * (1 - b))
                model.addConstr(f[k] + surg_time[(p, sid)] <= f[p] + M * (2 - in_p - in_k) + M * b)

    # Objective
    obj_terms = []
    BIG_PENALTY = 1e7
    for hid in hospital_ids:
        for oid in h_or_ids[hid]:
            fc, oc = get_or_costs(hid, oid)
            for d in all_days:
                rt, mo = get_or_reg_max(hid, oid, d)
                obj_terms.append(y[(hid, d, oid)] * fc * (rt / 60.0))
                obj_terms.append(overtime[(hid, d, oid)] * oc / 60.0)
    for sid in surgeon_ids:
        for d in surgeons_by_id[sid]['operating_days']:
            for hid in hospital_ids:
                sc = hospitals_by_id[hid]['surgeon_fixed_costs_per_day'].get(str(sid), 0)
                obj_terms.append(z[(sid, hid, d)] * sc)
    for pid in patient_ids:
        if not patients_by_id[pid]['is_mandatory']:
            r = patients_by_id[pid]['reward']
            obj_terms.append(-r * quicksum(x[(pid, sid, hid, d, oid)] for (sid, hid, d, oid) in feas_by_p[pid]))
    for pid in slack:
        obj_terms.append(slack[pid] * BIG_PENALTY)

    model.setObjective(quicksum(obj_terms), GRB.MINIMIZE)

    last_obj = [float('inf')]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < last_obj[0] - 1e-6:
                last_obj[0] = obj_val
                if logger:
                    logger.log(obj_val)

    model.optimize(callback)

    if model.SolCount > 0:
        assignments = []
        for pid in patient_ids:
            for (sid, hid, d, oid) in feas_by_p[pid]:
                if x[(pid, sid, hid, d, oid)].X > 0.5:
                    T = T_ps(pid, sid)
                    finish = f[pid].X
                    sg = surg_time[(pid, sid)]
                    prep = patients_by_id[pid]['preparation_time']
                    clean = patients_by_id[pid]['cleaning_time']
                    assignments.append({
                        'patient_id': pid,
                        'surgeon_id': sid,
                        'hospital_id': hid,
                        'day': d,
                        'or_id': oid,
                        'is_mandatory': patients_by_id[pid]['is_mandatory'],
                        'T_ps': T,
                        'finish_time': finish,
                        'surgery_start_time': finish - sg,
                        'room_entry_time': finish - sg - prep,
                        'room_exit_time': finish + clean
                    })
                    break

        opened_ors = []
        for hid in hospital_ids:
            for oid in h_or_ids[hid]:
                for d in all_days:
                    if y[(hid, d, oid)].X > 0.5:
                        rt, _ = get_or_reg_max(hid, oid, d)
                        opened_ors.append({
                            'hospital_id': hid,
                            'day': d,
                            'or_id': oid,
                            'regular_time': rt,
                            'overtime': overtime[(hid, d, oid)].X,
                            'completion_time': C[(hid, d, oid)].X
                        })

        surgeon_assignments = []
        for sid in surgeon_ids:
            for d in surgeons_by_id[sid]['operating_days']:
                for hid in hospital_ids:
                    if z[(sid, hid, d)].X > 0.5:
                        surgeon_assignments.append({
                            'surgeon_id': sid,
                            'hospital_id': hid,
                            'day': d,
                            'start_time': st[(sid, hid, d)].X,
                            'end_time': et[(sid, hid, d)].X
                        })

        # Extract OR sequences
        or_sequences = []
        for (hid, d, oid, p, k), b in or_seq.items():
            in_p = False
            in_k = False
            for a in assignments:
                if a['day'] == d and a['hospital_id'] == hid and a['or_id'] == oid and a['patient_id'] == p:
                    in_p = True
                if a['day'] == d and a['hospital_id'] == hid and a['or_id'] == oid and a['patient_id'] == k:
                    in_k = True
            if in_p and in_k:
                or_sequences.append({
                    'hospital_id': hid,
                    'day': d,
                    'or_id': oid,
                    'patient_p': p,
                    'patient_k': k,
                    'p_after_k': 0 if b.X > 0.5 else 1
                })

        # Extract Surgeon sequences
        surgeon_sequences = []
        for (sid, hid, d, p, k), b in sg_seq.items():
            in_p = False
            in_k = False
            for a in assignments:
                if a['day'] == d and a['hospital_id'] == hid and a['surgeon_id'] == sid and a['patient_id'] == p:
                    in_p = True
                if a['day'] == d and a['hospital_id'] == hid and a['surgeon_id'] == sid and a['patient_id'] == k:
                    in_k = True
            if in_p and in_k:
                surgeon_sequences.append({
                    'surgeon_id': sid,
                    'hospital_id': hid,
                    'day': d,
                    'patient_p': p,
                    'patient_k': k,
                    'p_after_k': 0 if b.X > 0.5 else 1
                })

        objective_value = model.ObjVal
        solution = {
            'objective_value': objective_value,
            'assignments': assignments,
            'opened_ors': opened_ors,
            'surgeon_assignments': surgeon_assignments,
            'or_sequences': or_sequences,
            'surgeon_sequences': surgeon_sequences
        }

        if logger:
            logger.log_solution(objective_value, solution)

        with open(args.solution_path, 'w') as fh:
            json.dump(solution, fh, indent=2)
    else:
        solution = {
            'objective_value': None,
            'assignments': [],
            'opened_ors': [],
            'surgeon_assignments': [],
            'or_sequences': [],
            'surgeon_sequences': []
        }
        with open(args.solution_path, 'w') as fh:
            json.dump(solution, fh, indent=2)

if __name__ == '__main__':
    main()