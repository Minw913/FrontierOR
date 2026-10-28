import json
import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, default=300)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    start_time = time.time()

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            pass

    with open(args.instance_path) as f:
        instance = json.load(f)

    H = instance['planning_horizon_days']
    crew_members = instance['crew_members']
    pairings_list = instance['pairings']
    wr = instance['work_rules']

    NC = len(crew_members)
    NP = len(pairings_list)

    # Build trivial solution (all uncovered)
    trivial_obj = sum(p['duration_minutes'] * p['coverage_requirement'] for p in pairings_list)
    trivial_rosters = {}
    for c in crew_members:
        total_min = H * 24 * 60
        pre_min = sum(a['duration_days'] * 24 * 60 for a in c.get('pre_assigned_activities', []))
        trivial_rosters[str(c['id'])] = {
            'roster_index': 0,
            'pairings': [],
            'unproductive_time': int(max(0, total_min - pre_min))
        }
    trivial_uncovered = [
        {'pairing_id': p['id'], 'uncovered_count': p['coverage_requirement'],
         'duration_minutes': p['duration_minutes']}
        for p in pairings_list
    ]
    trivial_solution = {
        'objective_value': float(trivial_obj),
        'rosters': trivial_rosters,
        'uncovered_pairings': trivial_uncovered
    }
    if logger:
        logger.log_solution(float(trivial_obj), trivial_solution)

    if NP == 0 or NC == 0:
        with open(args.solution_path, 'w') as f:
            json.dump(trivial_solution, f, indent=2)
        return

    # Pairing time intervals [start, end) in hours
    p_intervals = []
    for p in pairings_list:
        s = p['start_day'] * 24.0 + p['departure_hour']
        e = (p['start_day'] + p['duration_days'] - 1) * 24.0 + p['return_hour']
        p_intervals.append((s, e))

    # Pre-assigned activity intervals
    c_pre = []
    for c in crew_members:
        acts = []
        for a in c.get('pre_assigned_activities', []):
            s = a['start_day'] * 24.0
            e = (a['end_day'] + 1) * 24.0
            acts.append((s, e, a.get('type', '')))
        c_pre.append(acts)

    # Determine eligibility (qualifications + no overlap with pre-assigned)
    eligible = [set() for _ in range(NC)]
    for ci in range(NC):
        quals = set(crew_members[ci].get('qualifications', []))
        for pi in range(NP):
            if pairings_list[pi]['aircraft_type'] not in quals:
                continue
            ps, pe = p_intervals[pi]
            ok = True
            for (as_, ae, _) in c_pre[ci]:
                if ps < ae and as_ < pe:
                    ok = False
                    break
            if ok:
                eligible[ci].add(pi)

    # Limit problem size if too many variables
    total_vars = sum(len(eligible[ci]) for ci in range(NC))
    MAX_VARS = 250000
    if total_vars > MAX_VARS and NC > 0:
        max_per_crew = max(10, MAX_VARS // NC)
        for ci in range(NC):
            if len(eligible[ci]) > max_per_crew:
                sorted_pis = sorted(eligible[ci],
                                   key=lambda pi: -pairings_list[pi]['duration_minutes'])
                eligible[ci] = set(sorted_pis[:max_per_crew])

    # Build MIP
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        with open(args.solution_path, 'w') as f:
            json.dump(trivial_solution, f, indent=2)
        return

    model = gp.Model("crew_rostering")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0

    elapsed = time.time() - start_time
    remaining = max(1, args.time_limit - elapsed)
    model.Params.TimeLimit = remaining * 0.85

    # Decision variables
    x = {}
    for ci in range(NC):
        for pi in eligible[ci]:
            x[ci, pi] = model.addVar(vtype=GRB.BINARY, name=f"x_{ci}_{pi}")

    u = {}
    for pi in range(NP):
        u[pi] = model.addVar(lb=0, ub=pairings_list[pi]['coverage_requirement'],
                             vtype=GRB.INTEGER, name=f"u_{pi}")

    model.update()

    # Coverage constraints
    for pi in range(NP):
        crew_vars = [x[ci, pi] for ci in range(NC) if (ci, pi) in x]
        model.addConstr(gp.quicksum(crew_vars) + u[pi] == pairings_list[pi]['coverage_requirement'])

    # Overlap constraints via interval graph cliques
    for ci in range(NC):
        if len(eligible[ci]) < 2:
            continue
        intervals = [(p_intervals[pi][0], p_intervals[pi][1], pi) for pi in eligible[ci]]
        events = []
        for s, e, pi in intervals:
            events.append((s, 1, pi))  # start
            events.append((e, 0, pi))  # end
        events.sort(key=lambda ev: (ev[0], ev[1]))  # ends before starts at same time
        active = set()
        for t, typ, pi in events:
            if typ == 1:
                active.add(pi)
                if len(active) > 1:
                    model.addConstr(gp.quicksum(x[ci, p] for p in active) <= 1)
            else:
                active.discard(pi)

    # Monthly flight hour limits
    for ci in range(NC):
        terms = [(pairings_list[pi]['flight_hours'], x[ci, pi])
                 for pi in eligible[ci] if (ci, pi) in x]
        if terms:
            model.addConstr(gp.quicksum(fh * var for fh, var in terms) <= wr['monthly_flight_limit_hours'])

    # Rolling flight hour limits (first 15 days of current month + last 15 of previous)
    for ci in range(NC):
        prev_h = crew_members[ci].get('flight_hours_previous_15_days', 0.0)
        terms = [(pairings_list[pi]['flight_hours'], x[ci, pi])
                 for pi in eligible[ci]
                 if (ci, pi) in x and pairings_list[pi]['start_day'] < 15]
        model.addConstr(gp.quicksum(fh * var for fh, var in terms) + prev_h <= wr['rolling_flight_limit_hours'])

    # Cross-pairing limits
    for ci in range(NC):
        cross = [x[ci, pi] for pi in eligible[ci]
                 if (ci, pi) in x and pairings_list[pi].get('is_cross_pairing', False)]
        if cross:
            model.addConstr(gp.quicksum(cross) <= wr['max_cross_pairings_per_month'])

    # Consecutive early departure constraints
    early_by_day = {}
    for pi in range(NP):
        if pairings_list[pi]['departure_hour'] < 8.0:
            early_by_day.setdefault(pairings_list[pi]['start_day'], []).append(pi)

    max_early = wr['max_consecutive_early_departures_before_8am']
    if early_by_day and max_early < H:
        for ci in range(NC):
            d_vars = {}
            for day in sorted(early_by_day.keys()):
                ee = [pi for pi in early_by_day[day] if pi in eligible[ci] and (ci, pi) in x]
                if not ee:
                    continue
                d_vars[day] = model.addVar(vtype=GRB.BINARY, name=f"d_{ci}_{day}")
                for pi in ee:
                    model.addConstr(d_vars[day] >= x[ci, pi])
            for day in range(H - max_early):
                dn = [day + k for k in range(max_early + 1)]
                if all(d in d_vars for d in dn):
                    model.addConstr(gp.quicksum(d_vars[d] for d in dn) <= max_early)

    # Objective: minimize total uncovered duration
    model.setObjective(
        gp.quicksum(pairings_list[pi]['duration_minutes'] * u[pi] for pi in range(NP)),
        GRB.MINIMIZE
    )

    # Callback for logging incumbents
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    model.optimize(callback)

    # Extract solution
    assignments = [[] for _ in range(NC)]
    if model.SolCount > 0:
        for ci in range(NC):
            for pi in eligible[ci]:
                if (ci, pi) in x and x[ci, pi].X > 0.5:
                    assignments[ci].append(pi)
    else:
        with open(args.solution_path, 'w') as f:
            json.dump(trivial_solution, f, indent=2)
        return

    # Post-process: check rest period feasibility and fix violations
    try:
        for ci in range(NC):
            if not assignments[ci]:
                continue
            fix_roster(assignments[ci], crew_members[ci], pairings_list, p_intervals, H, wr)
    except Exception:
        pass

    # Compute final coverage
    cov_count = [0] * NP
    for ci in range(NC):
        for pi in assignments[ci]:
            cov_count[pi] += 1

    unc_list = []
    total_unc = 0
    for pi in range(NP):
        unc = pairings_list[pi]['coverage_requirement'] - cov_count[pi]
        if unc > 0:
            unc_list.append({
                'pairing_id': pairings_list[pi]['id'],
                'uncovered_count': unc,
                'duration_minutes': pairings_list[pi]['duration_minutes']
            })
            total_unc += unc * pairings_list[pi]['duration_minutes']

    # Build rosters output
    rosters = {}
    for ci in range(NC):
        cm = crew_members[ci]
        pids = [pairings_list[pi]['id'] for pi in assignments[ci]]
        tmin = H * 24 * 60
        pmin = sum(pairings_list[pi]['duration_minutes'] for pi in assignments[ci])
        pre = sum(a['duration_days'] * 24 * 60 for a in cm.get('pre_assigned_activities', []))
        rosters[str(cm['id'])] = {
            'roster_index': 0,
            'pairings': pids,
            'unproductive_time': int(max(0, tmin - pmin - pre))
        }

    solution = {
        'objective_value': float(total_unc),
        'rosters': rosters,
        'uncovered_pairings': unc_list
    }

    if logger:
        logger.log_solution(float(total_unc), solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def compute_free_intervals(busy, t_start, t_end):
    """Compute free intervals in [t_start, t_end) given busy intervals."""
    filtered = [(max(s, t_start), min(e, t_end)) for s, e in busy if s < t_end and e > t_start]
    filtered.sort()
    free = []
    cur = t_start
    for s, e in filtered:
        if s > cur:
            free.append((cur, s))
        cur = max(cur, e)
    if cur < t_end:
        free.append((cur, t_end))
    return free


def check_rest_feasibility(assigned, crew_member, pairings, p_intervals, H, wr):
    """Check if weekly and monthly rest periods can be placed."""
    busy = [p_intervals[pi] for pi in assigned]
    for a in crew_member.get('pre_assigned_activities', []):
        if a.get('type', '') == 'annual_leave':
            continue  # annual leave counts as free for rest placement
        busy.append((a['start_day'] * 24.0, (a['end_day'] + 1) * 24.0))

    num_weeks = (H + 6) // 7
    weekly_ok = [False] * num_weeks

    weekly_rest = float(wr['weekly_rest_period_hours'])
    for w in range(num_weeks):
        ws_day = w * 7
        rsm = ws_day * 24.0
        rsx = (ws_day + 6) * 24.0 + 6.0  # Sunday 06:00
        cs = rsm
        ce = min(rsm + 7 * 24.0 + 30.0, H * 24.0)  # overflow up to 30h into next week
        free = compute_free_intervals(busy, cs, ce)
        for fs, fe in free:
            if max(fs, rsm) <= min(fe - weekly_rest, rsx):
                weekly_ok[w] = True
                break

    # Monthly rest: extended weekly rest in at least one week
    total_inactive = sum(a['duration_days'] for a in crew_member.get('pre_assigned_activities', []))
    reduction = 12 * (total_inactive // 7)
    additional = max(0, wr['monthly_rest_additional_hours'] - reduction)
    monthly_dur = float(wr['weekly_rest_period_hours'] + additional)

    monthly_ok = False
    for w in range(num_weeks):
        ws_day = w * 7
        rsm = ws_day * 24.0
        rsx = (ws_day + 6) * 24.0 + 6.0
        cs = rsm
        ce = min(rsm + 7 * 24.0 + 30.0, H * 24.0)
        free = compute_free_intervals(busy, cs, ce)
        for fs, fe in free:
            if max(fs, rsm) <= min(fe - monthly_dur, rsx):
                monthly_ok = True
                break
        if monthly_ok:
            break

    return all(weekly_ok) and monthly_ok


def check_reserve_rules(assigned, crew_member, pairings, p_intervals):
    """Check that reserve blocks are followed by noon-or-later departures or rests."""
    for a in crew_member.get('pre_assigned_activities', []):
        if a.get('type', '') != 'reserve_block':
            continue
        rb_end = (a['end_day'] + 1) * 24.0
        earliest = None
        for pi in assigned:
            if p_intervals[pi][0] >= rb_end - 0.001:
                if earliest is None or p_intervals[pi][0] < p_intervals[earliest][0]:
                    earliest = pi
        if earliest is not None and pairings[earliest]['departure_hour'] < 12.0:
            return False, earliest
    return True, None


def fix_roster(assigned, crew_member, pairings, p_intervals, H, wr):
    """Remove pairings greedily until all rest/reserve constraints are satisfied."""
    # Fix reserve rule violations
    while True:
        ok, bad = check_reserve_rules(assigned, crew_member, pairings, p_intervals)
        if ok or bad is None:
            break
        assigned.remove(bad)

    # Fix rest period violations
    while assigned:
        if check_rest_feasibility(assigned, crew_member, pairings, p_intervals, H, wr):
            break
        shortest = min(assigned, key=lambda pi: pairings[pi]['duration_minutes'])
        assigned.remove(shortest)


if __name__ == '__main__':
    main()