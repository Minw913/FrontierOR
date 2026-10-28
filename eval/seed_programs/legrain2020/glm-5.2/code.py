import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        instance = json.load(f)

    N = instance['num_nurses']
    D = instance['num_days']
    num_weeks = instance['num_weeks']
    S = instance['num_shifts']
    K = instance['num_skills']

    skill_names = instance['skills']
    shift_info = instance['shifts']
    pw = instance['penalty_weights']
    nurses = instance['nurses']
    demand = instance['demand']

    skill_idx = {name: i for i, name in enumerate(skill_names)}
    shift_idx = {s['name']: i for i, s in enumerate(shift_info)}

    forbidden_set = set()
    for pair in instance['forbidden_shift_successions']:
        if pair[0] in shift_idx and pair[1] in shift_idx:
            forbidden_set.add((shift_idx[pair[0]], shift_idx[pair[1]]))

    nurse_skills = [set(skill_idx[s] for s in nurses[n]['skills']) for n in range(N)]
    nurses_with_skill = {k: [n for n in range(N) if k in nurse_skills[n]] for k in range(K)}

    if N == 0 or D == 0:
        solution = {"objective_value": 0.0, "schedule": []}
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        if logger:
            logger.log_solution(0.0, solution)
        return

    t_start = time.time()

    model = gp.Model('nurse_scheduling')
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0

    # Decision variables
    x_keys = [(n, d, s, k) for n in range(N) for d in range(D)
              for s in range(S) for k in nurse_skills[n]]
    x = model.addVars(x_keys, vtype=GRB.BINARY, name='x')
    y = model.addVars(N, D, S, lb=0, ub=1, name='y')
    w = model.addVars(N, D, lb=0, ub=1, name='w')
    r = model.addVars(N, D, lb=0, ub=1, name='r')
    u = model.addVars(D, S, K, lb=0, name='u')

    # Linking + at most one shift per day
    for n in range(N):
        for d in range(D):
            for s in range(S):
                model.addConstr(y[n, d, s] == gp.quicksum(x[n, d, s, k] for k in nurse_skills[n]))
            model.addConstr(w[n, d] == gp.quicksum(y[n, d, s] for s in range(S)))
            model.addConstr(r[n, d] == 1 - w[n, d])
            model.addConstr(gp.quicksum(x[n, d, s, k] for s in range(S) for k in nurse_skills[n]) <= 1)

    # Demand (hard min, soft opt)
    for d in range(D):
        for s in range(S):
            sname = shift_info[s]['name']
            if sname not in demand[d]:
                continue
            for k in range(K):
                kname = skill_names[k]
                if kname not in demand[d][sname]:
                    continue
                min_d = demand[d][sname][kname]['minimum']
                opt_d = demand[d][sname][kname]['optimal']
                expr = gp.quicksum(x[n, d, s, k] for n in nurses_with_skill[k])
                model.addConstr(expr >= min_d)
                model.addConstr(expr + u[d, s, k] >= opt_d)

    # Forbidden shift successions (hard)
    for n in range(N):
        for d in range(D - 1):
            for (s1, s2) in forbidden_set:
                model.addConstr(y[n, d, s1] + y[n, d + 1, s2] <= 1)

    # Bounds from contracts
    max_cd = {n: nurses[n]['contract']['CD_plus'] for n in range(N)}
    min_cd = {n: nurses[n]['contract']['CD_minus'] for n in range(N)}
    max_cr = {n: nurses[n]['contract']['CR_plus'] for n in range(N)}
    min_cr = {n: nurses[n]['contract']['CR_minus'] for n in range(N)}
    l_minus = {n: nurses[n]['contract']['L_minus'] for n in range(N)}
    l_plus = {n: nurses[n]['contract']['L_plus'] for n in range(N)}
    we_plus = {n: nurses[n]['contract']['WE_plus'] for n in range(N)}
    
    max_cs = {s: shift_info[s]['CS_plus'] for s in range(S)}
    min_cs = {s: shift_info[s]['CS_minus'] for s in range(S)}

    # Consecutive working days: max consecutive (hard via sliding window)
    for n in range(N):
        L = max_cd[n]
        for d in range(D - L):
            model.addConstr(gp.quicksum(w[n, d + i] for i in range(L + 1)) <= L)

    # Consecutive days off: max consecutive (hard via sliding window)
    for n in range(N):
        L = max_cr[n]
        for d in range(D - L):
            model.addConstr(gp.quicksum(r[n, d + i] for i in range(L + 1)) <= L)

    # Consecutive days: min working days (soft) - penalize isolated working days
    pen_iso = model.addVars(N, D, lb=0, ub=1, name='pen_iso')
    for n in range(N):
        for d in range(D):
            if d == 0:
                left = 1
            else:
                left = r[n, d - 1]
            if d == D - 1:
                right = 1
            else:
                right = r[n, d + 1]
            model.addConstr(pen_iso[n, d] >= w[n, d] + left + right - 2)

    # Consecutive days off: min days off (soft) - penalize isolated days off
    pen_iso_r = model.addVars(N, D, lb=0, ub=1, name='pen_iso_r')
    for n in range(N):
        for d in range(D):
            if d == 0:
                left = 1
            else:
                left = w[n, d - 1]
            if d == D - 1:
                right = 1
            else:
                right = w[n, d + 1]
            model.addConstr(pen_iso_r[n, d] >= r[n, d] + left + right - 2)

    # Consecutive same shift: max consecutive (soft)
    exc_shift = model.addVars(N, S, D, lb=0, name='exc_shift')
    for n in range(N):
        for s in range(S):
            L = max_cs[s]
            for d in range(D - L):
                model.addConstr(exc_shift[n, s, d] >= gp.quicksum(y[n, d + i, s] for i in range(L + 1)) - L)

    # Consecutive same shift: min consecutive (soft) - penalize isolated shifts
    pen_shift_iso = model.addVars(N, S, D, lb=0, ub=1, name='pen_shift_iso')
    for n in range(N):
        for s in range(S):
            for d in range(D):
                if d == 0:
                    left = 1
                else:
                    left = 1 - y[n, d - 1, s]
                if d == D - 1:
                    right = 1
                else:
                    right = 1 - y[n, d + 1, s]
                model.addConstr(pen_shift_iso[n, s, d] >= y[n, d, s] + left + right - 2)

    # Preferences (soft)
    pref_pen = model.addVars(N, D, S, lb=0, ub=1, name='pref_pen')
    for n in range(N):
        prefs = nurses[n].get('preferences', [])
        pref_set = {(p['day'], p['shift']) for p in prefs}
        for d, sname in pref_set:
            if sname in shift_idx:
                s = shift_idx[sname]
                model.addConstr(pref_pen[n, d, s] >= y[n, d, s])

    # Weekend: complete weekends (soft)
    weekend_pairs = []
    for w_idx in range(num_weeks):
        sat = w_idx * 7 + 5
        sun = w_idx * 7 + 6
        if sat < D and sun < D:
            weekend_pairs.append((sat, sun))
    weekend_pen = model.addVars(N, len(weekend_pairs), lb=0, ub=1, name='weekend_pen')
    for n in range(N):
        for wi, (sat, sun) in enumerate(weekend_pairs):
            model.addConstr(weekend_pen[n, wi] >= w[n, sat] - w[n, sun])
            model.addConstr(weekend_pen[n, wi] >= w[n, sun] - w[n, sat])

    # Total assignments: min and max (soft)
    under_total = model.addVars(N, lb=0, name='under_total')
    over_total = model.addVars(N, lb=0, name='over_total')
    for n in range(N):
        total_assign = gp.quicksum(w[n, d] for d in range(D))
        model.addConstr(under_total[n] >= l_minus[n] - total_assign)
        model.addConstr(over_total[n] >= total_assign - l_plus[n])

    # Total worked weekends: max (soft)
    weekend_worked = model.addVars(N, len(weekend_pairs), vtype=GRB.BINARY, name='weekend_worked')
    excess_weekends = model.addVars(N, lb=0, name='excess_weekends')
    for n in range(N):
        for wi, (sat, sun) in enumerate(weekend_pairs):
            model.addConstr(weekend_worked[n, wi] >= w[n, sat])
            model.addConstr(weekend_worked[n, wi] >= w[n, sun])
        model.addConstr(excess_weekends[n] >= gp.quicksum(weekend_worked[n, wi] for wi in range(len(weekend_pairs))) - we_plus[n])

    # Objective
    obj = gp.LinExpr()
    
    # c_S1: coverage
    obj += pw['c_S1'] * gp.quicksum(u[d, s, k] for d in range(D) for s in range(S) for k in range(K))

    # c_S2a: consecutive worked days
    obj += pw['c_S2a'] * gp.quicksum(pen_iso[n, d] for n in range(N) for d in range(D))

    # c_S2b: consecutive same shift
    obj += pw['c_S2b'] * gp.quicksum(exc_shift[n, s, d] for n in range(N) for s in range(S) for d in range(D))
    obj += pw['c_S2b'] * gp.quicksum(pen_shift_iso[n, s, d] for n in range(N) for s in range(S) for d in range(D))

    # c_S3: consecutive days off
    obj += pw['c_S3'] * gp.quicksum(pen_iso_r[n, d] for n in range(N) for d in range(D))

    # c_S4: preferences
    obj += pw['c_S4'] * gp.quicksum(pref_pen[n, d, s] for n in range(N) for d in range(D) for s in range(S))

    # c_S5: incomplete weekends
    obj += pw['c_S5'] * gp.quicksum(weekend_pen[n, wi] for n in range(N) for wi in range(len(weekend_pairs)))

    # c_S6: total assignments
    obj += pw['c_S6'] * gp.quicksum(under_total[n] + over_total[n] for n in range(N))

    # c_S7: excess weekends
    obj += pw['c_S7'] * gp.quicksum(excess_weekends[n] for n in range(N))

    model.setObjective(obj, GRB.MINIMIZE)

    # Greedy MIPStart
    try:
        start_vals = {}
        for n in range(N):
            for d in range(D):
                for s in range(S):
                    for k in nurse_skills[n]:
                        start_vals[x[n, d, s, k]] = 0
        for d in range(D):
            for s in range(S):
                sname = shift_info[s]['name']
                if sname not in demand[d]:
                    continue
                for k in range(K):
                    kname = skill_names[k]
                    if kname not in demand[d][sname]:
                        continue
                    min_d = demand[d][sname][kname]['minimum']
                    assigned = 0
                    for n in nurses_with_skill[k]:
                        if assigned >= min_d:
                            break
                        ok = True
                        if d > 0:
                            for (s1, s2) in forbidden_set:
                                if s2 == s and start_vals.get(x[n, d - 1, s1, k], 0) > 0.5:
                                    ok = False
                                    break
                        if ok:
                            start_vals[x[n, d, s, k]] = 1
                            assigned += 1
        for n in range(N):
            for d in range(D):
                for s in range(S):
                    val = sum(start_vals.get(x[n, d, s, k], 0) for k in nurse_skills[n])
                    start_vals[y[n, d, s]] = val
                wv = sum(start_vals.get(y[n, d, s], 0) for s in range(S))
                start_vals[w[n, d]] = wv
                start_vals[r[n, d]] = 1 - wv
        for n in range(N):
            for d in range(D):
                for s in range(S):
                    for k in nurse_skills[n]:
                        model.addConstr(x[n, d, s, k] == start_vals.get(x[n, d, s, k], 0), name=f'start_{n}_{d}_{s}_{k}')
        model.Params.TimeLimit = 5
        model.optimize()
        model.remove([c for c in model.getConstrs() if c.VarName.startswith('start_')])
        model.Params.TimeLimit = max(1, args.time_limit - int(time.time() - t_start))
    except Exception:
        pass

    model.Params.TimeLimit = max(1, args.time_limit - int(time.time() - t_start))
    model.optimize()

    # Extract solution
    schedule = []
    if model.SolCount > 0:
        for n in range(N):
            nurse_assignments = []
            for d in range(D):
                for s in range(S):
                    for k in nurse_skills[n]:
                        if x[n, d, s, k].X > 0.5:
                            nurse_assignments.append({
                                "day": d,
                                "shift": shift_info[s]['name'],
                                "skill": skill_names[k]
                            })
            if nurse_assignments:
                schedule.append({
                    "nurse_id": nurses[n]['id'],
                    "nurse_name": nurses[n]['name'],
                    "assignments": nurse_assignments
                })
        obj_val = model.ObjVal
    else:
        obj_val = 0.0

    solution = {
        "objective_value": obj_val,
        "schedule": schedule
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    if logger:
        logger.log_solution(obj_val, solution)


if __name__ == '__main__':
    main()