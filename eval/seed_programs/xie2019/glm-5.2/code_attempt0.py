import argparse
import json
import random
import itertools
import time
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    start_time = time.time()
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except Exception:
        logger = None

    with open(args.instance_path) as f:
        inst = json.load(f)

    num_vessels = inst['num_vessels']
    num_berths = inst['num_berths']
    total_qc = inst['total_qc']
    T = inst['time_steps_per_shift']
    num_shifts = inst['num_shifts_per_step']
    horizon = inst['total_time_steps']
    hours = inst['time_horizon_hours']
    c1 = inst['c1']
    c2 = inst['c2']
    vessels = inst['vessels']
    berths = inst['berths']
    big = inst['qc_profiles']['big']
    small = inst['qc_profiles']['small']

    seed = inst.get('seed', None)
    if seed is None:
        seed = inst.get('random_seed', None)
    if seed is None:
        seed = inst.get('qc_profiles', {}).get('seed', 0)
    if not isinstance(seed, int):
        seed = 0
    rng = random.Random(seed)

    def gen_sequences(num, mnq, mxq, mnh, mxh):
        if num <= 0 or mnq > mxq or mnh > mxh:
            return []
        qc_range = list(range(mnq, mxq + 1))
        total = sum(len(qc_range) ** H for H in range(mnh, mxh + 1))
        seqs = []
        if 0 < total <= 200000:
            all_seqs = []
            for H in range(mnh, mxh + 1):
                for combo in itertools.product(qc_range, repeat=H):
                    all_seqs.append(list(combo))
            if len(all_seqs) <= num:
                seqs = all_seqs
            else:
                idx = rng.sample(range(len(all_seqs)), num)
                seqs = [all_seqs[i] for i in sorted(idx)]
        else:
            for _ in range(num):
                H = rng.randint(mnh, mxh)
                seq = [rng.randint(mnq, mxq) for _ in range(H)]
                seqs.append(seq)
        return seqs

    big_seqs = gen_sequences(inst['num_big_profiles'], big['min_qc'], big['max_qc'],
                             big['min_handling_shifts'], big['max_handling_shifts'])
    small_seqs = gen_sequences(inst['num_small_profiles'], small['min_qc'], small['max_qc'],
                               small['min_handling_shifts'], small['max_handling_shifts'])

    shift_hours = hours / num_shifts if num_shifts > 0 else 1.0

    def expand_profile(seq, r):
        H = len(seq)
        qc_list = [seq[0]] * (T - r)
        for i in range(1, H):
            qc_list.extend([seq[i]] * T)
        return len(qc_list), qc_list

    vessel_info = {}
    for v in vessels:
        vid = v['id']
        expected_H = int(round(v['duration_hours'] / shift_hours)) if shift_hours > 0 else 1
        if expected_H <= small['max_handling_shifts']:
            seqs = small_seqs
            is_big = False
            mnh, mxh = small['min_handling_shifts'], small['max_handling_shifts']
        else:
            seqs = big_seqs
            is_big = True
            mnh, mxh = big['min_handling_shifts'], big['max_handling_shifts']
        kept = [(i, s) for i, s in enumerate(seqs) if abs(len(s) - expected_H) <= 1]
        if not kept:
            kept = [(i, s) for i, s in enumerate(seqs) if mnh <= len(s) <= mxh]
        if not kept:
            kept = list(enumerate(seqs))
        if len(kept) > 15:
            kept = kept[:15]
        vessel_info[vid] = {'kept': kept, 'is_big': is_big}

    def build_candidates(max_profs_per_bt, relax_end, per_vessel_cap):
        candidates = {}
        for v in vessels:
            vid = v['id']
            a_v, b_v = v['a'], v['b']
            k_bar = v['k_bar']
            t_bar = v['t_bar']
            kept = vessel_info[vid]['kept']
            cand_list = []
            for b in range(num_berths):
                berth = berths[b]
                a_b = berth['a_k']
                b_b = berth['b_k']
                lo = max(a_v, a_b)
                hi = min(b_v, b_b)
                if lo > hi:
                    continue
                for t in range(lo, hi + 1):
                    r = t % T
                    profs = []
                    for pidx, seq in kept:
                        D, qc_list = expand_profile(seq, r)
                        end = t + D - 1
                        if end > horizon - 1:
                            continue
                        if not relax_end and end > b_b:
                            continue
                        profs.append((pidx, D, qc_list, end))
                    if len(profs) > max_profs_per_bt:
                        profs_by_dur = sorted(profs, key=lambda x: x[1])
                        profs_by_qc = sorted(profs, key=lambda x: sum(x[2]))
                        chosen = [profs_by_dur[0]]
                        if max_profs_per_bt >= 2 and profs_by_qc[0][0] != profs_by_dur[0][0]:
                            chosen.append(profs_by_qc[0])
                        profs = chosen[:max_profs_per_bt]
                    cost = c1 * abs(b - k_bar) + c2 * abs(t - t_bar)
                    for pidx, D, qc_list, end in profs:
                        cand_list.append((b, t, pidx, D, end, qc_list, cost))
            if len(cand_list) > per_vessel_cap:
                cand_list.sort(key=lambda c: c[6])
                cand_list = cand_list[:per_vessel_cap]
            candidates[vid] = cand_list
        return candidates

    candidates = build_candidates(2, False, 3000)
    total_cands = sum(len(candidates[v['id']]) for v in vessels)
    if total_cands > 80000:
        candidates = build_candidates(1, False, 3000)
        total_cands = sum(len(candidates[v['id']]) for v in vessels)

    need_relax = any(len(candidates[v['id']]) == 0 for v in vessels)
    if need_relax:
        candidates = build_candidates(2, True, 3000)
        if any(len(candidates[v['id']]) == 0 for v in vessels):
            candidates = build_candidates(1, True, 3000)

    def greedy(order):
        berth_occ = {}
        qc_used = {}
        assignment = {}
        for vi in order:
            vid = vessels[vi]['id']
            cands = candidates[vid]
            if not cands:
                return None
            sorted_cands = sorted(range(len(cands)), key=lambda ci: cands[ci][6])
            assigned = False
            for ci in sorted_cands:
                c = cands[ci]
                b, t, pidx, D, end, qc_list, cost = c
                ok = True
                for tau in range(t, end + 1):
                    if (b, tau) in berth_occ:
                        ok = False
                        break
                if not ok:
                    continue
                for tau in range(t, end + 1):
                    qc = qc_list[tau - t]
                    if qc_used.get(tau, 0) + qc > total_qc:
                        ok = False
                        break
                if not ok:
                    continue
                assignment[vid] = ci
                for tau in range(t, end + 1):
                    berth_occ[(b, tau)] = vid
                    qc_used[tau] = qc_used.get(tau, 0) + qc_list[tau - t]
                assigned = True
                break
            if not assigned:
                return None
        return assignment

    def make_solution_dict(assignment):
        sol_vessels = []
        total = 0.0
        for v in vessels:
            vid = v['id']
            ci = assignment[vid]
            c = candidates[vid][ci]
            b, t, pidx, D, end, qc_list, cost = c
            sol_vessels.append({
                'id': vid,
                'berth': b,
                'start_time': t,
                'end_time': end,
                'profile': pidx
            })
            total += cost
        return total, {'objective_value': total, 'vessels': sol_vessels}

    best_obj = [None]
    best_sol = [None]

    def try_greedy(assign, obj_holder, sol_holder, log=True):
        if assign is None:
            return False
        obj, sol = make_solution_dict(assign)
        if obj_holder[0] is None or obj < obj_holder[0] - 1e-9:
            obj_holder[0] = obj
            sol_holder[0] = sol
            if log and logger:
                logger.log_solution(obj, sol)
        return True

    # try a few greedy orders
    orders = [
        sorted(range(num_vessels), key=lambda vi: vessels[vi]['a']),
        sorted(range(num_vessels), key=lambda vi: vessels[vi]['t_bar']),
        sorted(range(num_vessels), key=lambda vi: -vessels[vi]['duration_hours']),
        list(range(num_vessels)),
    ]
    for od in orders:
        ga = greedy(od)
        try_greedy(ga, best_obj, best_sol)
        if best_sol[0] is not None:
            break

    def write_output(sol, path):
        with open(path, 'w') as f:
            if sol is None:
                json.dump({'objective_value': None, 'vessels': []}, f, indent=2)
            else:
                json.dump(sol, f, indent=2)

    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed - 2
    infeasible = any(len(candidates[v['id']]) == 0 for v in vessels)
    if remaining < 2 or infeasible:
        write_output(best_sol[0], args.solution_path)
        return

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        write_output(best_sol[0], args.solution_path)
        return

    model = gp.Model()
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', max(1, remaining))
    model.ModelSense = GRB.MINIMIZE

    x = {}
    xkeys = []
    for v in vessels:
        vid = v['id']
        for ci, c in enumerate(candidates[vid]):
            var = model.addVar(vtype=GRB.BINARY, obj=c[6], name=f'x_{vid}_{ci}')
            x[(vid, ci)] = var
            xkeys.append((vid, ci))

    for v in vessels:
        vid = v['id']
        model.addConstr(gp.quicksum(x[(vid, ci)] for ci in range(len(candidates[vid]))) == 1)

    occ = {}
    qcu = {}
    for v in vessels:
        vid = v['id']
        for ci, c in enumerate(candidates[vid]):
            b, t, pidx, D, end, qc_list, cost = c
            var = x[(vid, ci)]
            for tau in range(t, end + 1):
                occ.setdefault((b, tau), []).append(var)
                qcu.setdefault(tau, []).append((var, qc_list[tau - t]))

    for key, varlist in occ.items():
        if len(varlist) > 1:
            model.addConstr(gp.quicksum(varlist) <= 1)
    for tau, pairs in qcu.items():
        model.addConstr(gp.quicksum(qc * var for var, qc in pairs) <= total_qc)

    if best_sol[0] is not None:
        greedy_assign = {}
        for v in vessels:
            vid = v['id']
            for ci in range(len(candidates[vid])):
                c = candidates[vid][ci]
                if c[0] == best_sol[0]['vessels'][vid]['berth'] and c[1] == best_sol[0]['vessels'][vid]['start_time'] and c[2] == best_sol[0]['vessels'][vid]['profile']:
                    greedy_assign[vid] = ci
                    break
        for (vid, ci) in xkeys:
            x[(vid, ci)].Start = 1.0 if greedy_assign.get(vid) == ci else 0.0

    model._xvars = [x[k] for k in xkeys]
    model._xkeys = xkeys

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if best_obj[0] is None or obj < best_obj[0] - 1e-9:
                vals = model.cbGetSolution(model._xvars)
                assignment = {}
                for k, val in zip(model._xkeys, vals):
                    if val > 0.5:
                        assignment[k[0]] = k[1]
                if len(assignment) == num_vessels:
                    obj2, sol = make_solution_dict(assignment)
                    best_obj[0] = obj2
                    best_sol[0] = sol
                    if logger:
                        logger.log_solution(obj2, sol)

    model.optimize(callback)

    if model.SolCount > 0:
        obj_val = model.ObjVal
        if best_obj[0] is None or obj_val < best_obj[0] - 1e-6:
            assignment = {}
            for v in vessels:
                vid = v['id']
                found = False
                for ci in range(len(candidates[vid])):
                    if x[(vid, ci)].Xn > 0.5:
                        assignment[vid] = ci
                        found = True
                        break
                if not found:
                    assignment = None
                    break
            if assignment is not None:
                obj2, sol = make_solution_dict(assignment)
                best_obj[0] = obj2
                best_sol[0] = sol
                if logger:
                    logger.log_solution(obj2, sol)

    write_output(best_sol[0], args.solution_path)

if __name__ == '__main__':
    main()