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

    # Consecutive days: max consecutive working days (hard via sliding window)
    max_consec = pw.get('max_consecutive_days', 6)
    for n in range(N):
        for d in range(D - max_consec):
            model.addConstr(gp.quicksum(w[n, d + i] for i in range(max_consec + 1)) <= max_consec)

    # Consecutive days: min consecutive working days (soft)
    min_consec = pw.get('min_consecutive_days', 2)
    min_consec_weight = pw.get('insufficient_consecutive_days', 20)
    # Detect stretches of exactly 1 working day (isolated)
    for n in range(N):
        for d in range(D):
            # w[n,d]=1 and (d==0 or r[n,d-1]=1) and (d==D-1 or r[n,d+1]=1)
            # => isolated working day => penalty
            if d == 0:
                left = 1
            else:
                left = r[n, d - 1]
            if d == D - 1:
                right = 1
            else:
                right = r[n, d + 1]
            # isolated = w[n,d] * left * right (nonlinear) => use auxiliary
            # We can add: if w=1 and left=1 and right=1 then penalty
            # Use indicator-like via linearization with binary bounds
            # Since w, r are binary (0/1 continuous), we can use:
            # pen_iso[n,d] >= w[n,d] + left + right - 2
            pen_iso = model.addVar(lb=0, ub=1, name=f'pen_iso_{n}_{d}')
            model.addConstr(pen_iso >= w[n, d] + left + right - 2)
            # This will be added to objective

    # Consecutive same shift: max consecutive (soft)
    max_consec_shift = pw.get('max_consecutive_same_shift', 5)
    max_consec_shift_weight = pw.get('excess_consecutive_same_shift', 15)
    for n in range(N):
        for s in range(S):
            for d in range(D - max_consec_shift):
                # sum of y[n,d+i,s] for i in 0..max_consec_shift > max_consec_shift => penalty
                excess = model.addVar(lb=0, name=f'exc_shift_{n}_{s}_{d}')
                model.addConstr(excess >= gp.quicksum(y[n, d + i, s] for i in range(max_consec_shift + 1)) - max_consec_shift)

    # Consecutive same shift: min consecutive (soft)
    min_consec_shift = pw.get('min_consecutive_same_shift', 2)
    min_consec_shift_weight = pw.get('insufficient_consecutive_same_shift', 15)
    # Detect isolated same shift (single day of a shift surrounded by different/rest)
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
                pen_shift_iso = model.addVar(lb=0, ub=1, name=f'pen_shift_iso_{n}_{s}_{d}')
                model.addConstr(pen_shift_iso >= y[n, d, s] + left + right - 2)

    # Preferences (soft)
    pref_weight = pw.get('preferences', 10)
    pref_pen = model.addVars(N, D, S, lb=0, ub=1, name='pref_pen')
    for n in range(N):
        prefs = nurses[n].get('preferences', {})
        for d in range(D):
            dkey = str(d)
            if dkey in prefs:
                for sname, val in prefs[dkey].items():
                    if sname in shift_idx:
                        s = shift_idx[sname]
                        if val == -1:  # nurse does not want this shift
                            model.addConstr(pref_pen[n, d, s] >= y[n, d, s])
                        elif val == 1:  # nurse wants this shift
                            model.addConstr(pref_pen[n, d, s] >= 1 - y[n, d, s])

    # Weekend: complete weekends (soft)
    complete_weekend_weight = pw.get('complete_weekend', 5)
    # Identify weekend days (Saturday=5, Sunday=6 in 0-indexed week)
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
    total_assign_weight = pw.get('total_assignments', 30)
    min_total = pw.get('min_total_assignments', 0)
    max_total = pw.get('max_total_assignments', D)
    total_assign = [gp.quicksum(w[n, d] for d in range(D)) for n in range(N)]
    under_total = model.addVars(N, lb=0, name='under_total')
    over_total = model.addVars(N, lb=0, name='over_total')
    for n in range(N):
        model.addConstr(under_total[n] >= min_total - total_assign[n])
        model.addConstr(over_total[n] >= total_assign[n] - max_total)

    # Objective
    obj = gp.LinExpr()
    # Coverage shortfall
    coverage_weight = pw.get('coverage', 30)
    for d in range(D):
        for s in range(S):
            for k in range(K):
                obj += coverage_weight * u[d, s, k]

    # Insufficient consecutive days
    for n in range(N):
        for d in range(D):
            pen_var = model.getVarByName(f'pen_iso_{n}_{d}')
            if pen_var is not None:
                obj += min_consec_weight * pen_var

    # Excess consecutive same shift
    for n in range(N):
        for s in range(S):
            for d in range(D - max_consec_shift):
                exc_var = model.getVarByName(f'exc_shift_{n}_{s}_{d}')
                if exc_var is not None:
                    obj += max_consec_shift_weight * exc_var

    # Insufficient consecutive same shift
    for n in range(N):
        for s in range(S):
            for d in range(D):
                pen_var = model.getVarByName(f'pen_shift_iso_{n}_{s}_{d}')
                if pen_var is not None:
                    obj += min_consec_shift_weight * pen_var

    # Preferences
    for n in range(N):
        for d in range(D):
            for s in range(S):
                obj += pref_weight * pref_pen[n, d, s]

    # Weekend
    for n in range(N):
        for wi in range(len(weekend_pairs)):
            obj += complete_weekend_weight * weekend_pen[n, wi]

    # Total assignments
    for n in range(N):
        obj += total_assign_weight * (under_total[n] + over_total[n])

    model.setObjective(obj, GRB.MINIMIZE)

    # Greedy MIPStart
    try:
        start_vals = {}
        for n in range(N):
            for d in range(D):
                for s in range(S):
                    for k in nurse_skills[n]:
                        start_vals[x[n, d, s, k]] = 0
        # Greedy: assign to meet minimum demand
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
                        # Check forbidden succession
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
            for d in range(D):
                for s in range(S):
                    for k in nurse_skills[n]:
                        if x[n, d, s, k].X > 0.5:
                            schedule.append({
                                "nurse": n,
                                "day": d,
                                "shift": shift_info[s]['name'],
                                "skill": skill_names[k]
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