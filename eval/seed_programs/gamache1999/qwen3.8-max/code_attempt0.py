import json
import argparse
import time
import sys

def read_instance(path):
    with open(path) as f:
        return json.load(f)

def pairings_overlap(p1, p2):
    """Check if two pairings overlap in time."""
    start1 = p1['start_day'] * 24.0 + p1['departure_hour']
    end1 = (p1['start_day'] + p1['duration_days'] - 1) * 24.0 + p1['return_hour']
    start2 = p2['start_day'] * 24.0 + p2['departure_hour']
    end2 = (p2['start_day'] + p2['duration_days'] - 1) * 24.0 + p2['return_hour']
    return start1 < end2 and start2 < end1

def pairing_overlaps_activity(pairing, activity):
    """Check if a pairing overlaps with a pre-assigned activity."""
    p_start = pairing['start_day'] * 24.0 + pairing['departure_hour']
    p_end = (pairing['start_day'] + pairing['duration_days'] - 1) * 24.0 + pairing['return_hour']
    a_start = activity['start_day'] * 24.0
    a_end = (activity['end_day'] + 1) * 24.0
    return p_start < a_end and a_start < p_end

def pairing_active_on_day(pairing, day):
    """Check if a pairing is active on a given day."""
    return pairing['start_day'] <= day <= pairing['start_day'] + pairing['duration_days'] - 1

def solve_instance(instance, time_limit, logger=None):
    from gurobipy import Model, GRB, quicksum
    
    start_time = time.time()
    
    planning_days = instance['planning_horizon_days']
    pairings = instance['pairings']
    crew_members = instance['crew_members']
    work_rules = instance.get('work_rules', {})
    
    num_pairings = len(pairings)
    num_crew = len(crew_members)
    
    # Work rule parameters
    monthly_flight_limit = work_rules.get('monthly_flight_limit_hours', 77)
    rolling_flight_limit = work_rules.get('rolling_flight_limit_hours', 85)
    max_cross = work_rules.get('max_cross_pairings_per_month', 1)
    max_early_consec = work_rules.get('max_consecutive_early_departures_before_8am', 2)
    
    # Preprocessing: determine compatibility
    compatible = [[] for _ in range(num_crew)]
    compatible_set = [set() for _ in range(num_crew)]
    
    for c_idx, crew in enumerate(crew_members):
        quals = set(crew.get('qualifications', []))
        pre_activities = crew.get('pre_assigned_activities', [])
        
        for p_idx, pairing in enumerate(pairings):
            if pairing['aircraft_type'] not in quals:
                continue
            conflict = False
            for act in pre_activities:
                if pairing_overlaps_activity(pairing, act):
                    conflict = True
                    break
            if conflict:
                continue
            compatible[c_idx].append(p_idx)
            compatible_set[c_idx].add(p_idx)
    
    # Precompute which crew can cover each pairing
    crew_for_pairing = [[] for _ in range(num_pairings)]
    for c_idx in range(num_crew):
        for p_idx in compatible[c_idx]:
            crew_for_pairing[p_idx].append(c_idx)
    
    # Find overlapping pairs for each crew member
    overlaps = [[] for _ in range(num_crew)]
    
    for c_idx in range(num_crew):
        comp = compatible[c_idx]
        if len(comp) < 2:
            continue
        comp_sorted = sorted(comp, key=lambda p: (pairings[p]['start_day'], pairings[p]['departure_hour']))
        
        for i in range(len(comp_sorted)):
            p1 = pairings[comp_sorted[i]]
            p1_end = (p1['start_day'] + p1['duration_days'] - 1) * 24.0 + p1['return_hour']
            for j in range(i + 1, len(comp_sorted)):
                p2 = pairings[comp_sorted[j]]
                p2_start = p2['start_day'] * 24.0 + p2['departure_hour']
                if p2_start >= p1_end:
                    break
                if pairings_overlap(p1, p2):
                    overlaps[c_idx].append((comp_sorted[i], comp_sorted[j]))
    
    # Find pairings active on each day for each crew member (for weekly rest approximation)
    pairings_on_day = [[[] for _ in range(planning_days)] for _ in range(num_crew)]
    for c_idx in range(num_crew):
        for p_idx in compatible[c_idx]:
            p = pairings[p_idx]
            for d in range(p['start_day'], min(p['start_day'] + p['duration_days'], planning_days)):
                pairings_on_day[c_idx][d].append(p_idx)
    
    # Build MIP
    model = Model("crew_rostering")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', max(1, time_limit - 2))
    model.setParam('OutputFlag', 0)
    
    # Variables
    x = {}
    for c_idx in range(num_crew):
        for p_idx in compatible[c_idx]:
            x[c_idx, p_idx] = model.addVar(vtype=GRB.BINARY, name=f"x_{c_idx}_{p_idx}")
    
    u = {}
    for p_idx in range(num_pairings):
        u[p_idx] = model.addVar(vtype=GRB.INTEGER, lb=0, 
                                ub=pairings[p_idx]['coverage_requirement'],
                                name=f"u_{p_idx}")
    
    model.update()
    
    # Objective: minimize total uncovered duration
    obj = quicksum(pairings[p_idx]['duration_minutes'] * u[p_idx] for p_idx in range(num_pairings))
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Constraint 1: Coverage
    for p_idx in range(num_pairings):
        if crew_for_pairing[p_idx]:
            model.addConstr(
                quicksum(x[c_idx, p_idx] for c_idx in crew_for_pairing[p_idx]) + u[p_idx] >= pairings[p_idx]['coverage_requirement'],
                name=f"coverage_{p_idx}"
            )
        else:
            model.addConstr(u[p_idx] >= pairings[p_idx]['coverage_requirement'], name=f"coverage_{p_idx}")
    
    # Constraint 2: Non-overlap
    for c_idx in range(num_crew):
        for (p1_idx, p2_idx) in overlaps[c_idx]:
            model.addConstr(x[c_idx, p1_idx] + x[c_idx, p2_idx] <= 1)
    
    # Constraint 3: Monthly flight limit
    for c_idx in range(num_crew):
        if compatible[c_idx]:
            model.addConstr(
                quicksum(pairings[p_idx]['flight_hours'] * x[c_idx, p_idx] for p_idx in compatible[c_idx]) <= monthly_flight_limit,
                name=f"monthly_flight_{c_idx}"
            )
    
    # Constraint 4: Rolling flight limit (first 15 days of current month)
    for c_idx in range(num_crew):
        prev_hours = crew_members[c_idx].get('flight_hours_previous_15_days', 0.0)
        rolling_pairings = [p_idx for p_idx in compatible[c_idx] if pairings[p_idx]['start_day'] <= 14]
        if rolling_pairings and prev_hours > 0:
            remaining = rolling_flight_limit - prev_hours
            if remaining < 0:
                # Can't fly at all in first 15 days
                for p_idx in rolling_pairings:
                    model.addConstr(x[c_idx, p_idx] == 0)
            else:
                model.addConstr(
                    quicksum(pairings[p_idx]['flight_hours'] * x[c_idx, p_idx] for p_idx in rolling_pairings) <= remaining,
                    name=f"rolling_flight_{c_idx}"
                )
    
    # Constraint 5: Cross-pairing limit
    for c_idx in range(num_crew):
        cross_pairings = [p_idx for p_idx in compatible[c_idx] if pairings[p_idx].get('is_cross_pairing', False)]
        if cross_pairings and max_cross < len(cross_pairings):
            model.addConstr(
                quicksum(x[c_idx, p_idx] for p_idx in cross_pairings) <= max_cross,
                name=f"cross_{c_idx}"
            )
    
    # Constraint 6: Early departure limit (no more than max_early_consec consecutive days)
    for c_idx in range(num_crew):
        for d in range(planning_days - max_early_consec):
            early_in_window = []
            for dd in range(d, d + max_early_consec + 1):
                for p_idx in compatible[c_idx]:
                    if pairings[p_idx]['start_day'] == dd and pairings[p_idx]['departure_hour'] < 8.0:
                        early_in_window.append(p_idx)
            if len(early_in_window) > max_early_consec:
                model.addConstr(
                    quicksum(x[c_idx, p_idx] for p_idx in early_in_window) <= max_early_consec,
                    name=f"early_{c_idx}_{d}"
                )
    
    # Constraint 7: Reserve blocks followed by rest or late departures
    for c_idx, crew in enumerate(crew_members):
        for act in crew.get('pre_assigned_activities', []):
            if act.get('type', '') == 'reserve_block':
                end_day = act['end_day']
                next_day = end_day + 1
                if next_day < planning_days:
                    for p_idx in compatible[c_idx]:
                        if pairings[p_idx]['start_day'] == next_day and pairings[p_idx]['departure_hour'] < 12.0:
                            model.addConstr(x[c_idx, p_idx] == 0)
    
    # Constraint 8: Weekly rest approximation
    # For each crew member and each full week, ensure at least 2 days are free
    # This is a necessary condition for 48-hour rest
    num_full_weeks = planning_days // 7
    for c_idx in range(num_crew):
        for w in range(num_full_weeks):
            week_start = w * 7
            week_end = week_start + 7
            # For each day in the week, check if any pairing is active
            # We want: at most 5 days with active pairings
            day_busy_vars = []
            for d in range(week_start, week_end):
                active_p = pairings_on_day[c_idx][d]
                if active_p:
                    # Create indicator: is any pairing active on this day?
                    # Since at most one pairing can be active at a time,
                    # sum of x for active pairings on this day is 0 or 1 (or possibly 2 if one ends and another starts)
                    day_busy_vars.append(quicksum(x[c_idx, p_idx] for p_idx in active_p if (c_idx, p_idx) in x))
            
            if day_busy_vars:
                model.addConstr(quicksum(day_busy_vars) <= 5)
    
    # Compute greedy initial solution for MIP start
    greedy_assignment = compute_greedy(instance, compatible, overlaps, monthly_flight_limit, 
                                        rolling_flight_limit, max_cross, max_early_consec)
    
    # Set MIP start
    try:
        for c_idx in range(num_crew):
            for p_idx in compatible[c_idx]:
                if (c_idx, p_idx) in x:
                    x[c_idx, p_idx].Start = 1.0 if p_idx in greedy_assignment.get(c_idx, set()) else 0.0
        for p_idx in range(num_pairings):
            assigned_count = sum(1 for c_idx in range(num_crew) if p_idx in greedy_assignment.get(c_idx, set()))
            u[p_idx].Start = max(0, pairings[p_idx]['coverage_requirement'] - assigned_count)
    except:
        pass
    
    # Solve
    model.optimize()
    
    # Extract solution
    solution = extract_solution(model, instance, x, u, compatible)
    
    if logger and solution:
        logger.log_solution(solution['objective_value'], solution)
    
    return solution

def compute_greedy(instance, compatible, overlaps, monthly_flight_limit, 
                   rolling_flight_limit, max_cross, max_early_consec):
    """Compute a greedy initial solution."""
    pairings = instance['pairings']
    crew_members = instance['crew_members']
    planning_days = instance['planning_horizon_days']
    num_crew = len(crew_members)
    num_pairings = len(pairings)
    
    # Sort pairings by duration * coverage_requirement (descending)
    pairing_order = sorted(range(num_pairings), 
                          key=lambda p: pairings[p]['duration_minutes'] * pairings[p]['coverage_requirement'],
                          reverse=True)
    
    assignment = {c: set() for c in range(num_crew)}
    flight_hours = {c: 0.0 for c in range(num_crew)}
    rolling_hours = {c: crew_members[c].get('flight_hours_previous_15_days', 0.0) for c in range(num_crew)}
    cross_count = {c: 0 for c in range(num_crew)}
    coverage_count = {p: 0 for p in range(num_pairings)}
    
    # Build overlap lookup
    overlap_set = [set() for _ in range(num_crew)]
    for c_idx in range(num_crew):
        for (p1, p2) in overlaps[c_idx]:
            overlap_set[c_idx].add((p1, p2))
            overlap_set[c_idx].add((p2, p1))
    
    for p_idx in pairing_order:
        p = pairings[p_idx]
        needed = p['coverage_requirement'] - coverage_count[p_idx]
        if needed <= 0:
            continue
        
        # Sort compatible crew by remaining capacity
        candidates = []
        for c_idx in crew_for_pairing_global[p_idx]:
            if p_idx not in compatible[c_idx]:
                continue
            # Check flight hour limits
            if flight_hours[c_idx] + p['flight_hours'] > monthly_flight_limit:
                continue
            if p['start_day'] <= 14 and rolling_hours[c_idx] + p['flight_hours'] > rolling_flight_limit:
                continue
            # Check cross-pairing limit
            if p.get('is_cross_pairing', False) and cross_count[c_idx] >= max_cross:
                continue
            # Check overlap with already assigned pairings
            conflict = False
            for assigned_p in assignment[c_idx]:
                if (p_idx, assigned_p) in overlap_set[c_idx]:
                    conflict = True
                    break
            if conflict:
                continue
            # Check early departure constraint
            if p['departure_hour'] < 8.0:
                early_violation = False
                for offset in range(-max_early_consec + 1, 1):
                    count = 0
                    for dd in range(p['start_day'] + offset, p['start_day'] + offset + max_early_consec + 1):
                        if 0 <= dd < planning_days:
                            for ap in assignment[c_idx]:
                                if pairings[ap]['start_day'] == dd and pairings[ap]['departure_hour'] < 8.0:
                                    count += 1
                                    break
                    if count > max_early_consec:
                        early_violation = True
                        break
                if early_violation:
                    continue
            
            # Score: prefer crew with fewer assigned pairings
            candidates.append((len(assignment[c_idx]), c_idx))
        
        candidates.sort()
        assigned_this = 0
        for _, c_idx in candidates:
            if assigned_this >= needed:
                break
            assignment[c_idx].add(p_idx)
            flight_hours[c_idx] += p['flight_hours']
            if p['start_day'] <= 14:
                rolling_hours[c_idx] += p['flight_hours']
            if p.get('is_cross_pairing', False):
                cross_count[c_idx] += 1
            coverage_count[p_idx] += 1
            assigned_this += 1
    
    return assignment

# Global variable for greedy
crew_for_pairing_global = []

def extract_solution(model, instance, x, u, compatible):
    pairings = instance['pairings']
    crew_members = instance['crew_members']
    planning_days = instance['planning_horizon_days']
    num_pairings = len(pairings)
    num_crew = len(crew_members)
    
    has_solution = model.SolCount > 0
    
    total_uncovered = 0.0
    rosters = {}
    uncovered_list = []
    
    for c_idx, crew in enumerate(crew_members):
        assigned = []
        pairing_minutes = 0
        
        if has_solution:
            for p_idx in compatible[c_idx]:
                if (c_idx, p_idx) in x and x[c_idx, p_idx].X > 0.5:
                    assigned.append(pairings[p_idx]['id'])
                    pairing_minutes += pairings[p_idx]['duration_minutes']
        
        total_minutes = planning_days * 1440
        pre_assigned_minutes = sum(act.get('duration_days', (act['end_day'] - act['start_day'] + 1)) * 1440 
                                   for act in crew.get('pre_assigned_activities', []))
        unproductive = max(0, total_minutes - pairing_minutes - pre_assigned_minutes)
        
        rosters[str(crew['id'])] = {
            "roster_index": 0,
            "pairings": assigned,
            "unproductive_time": int(unproductive)
        }
    
    for p_idx in range(num_pairings):
        if has_solution:
            uncovered_count = max(0, int(round(u[p_idx].X)))
        else:
            uncovered_count = pairings[p_idx]['coverage_requirement']
        
        if uncovered_count > 0:
            uncovered_list.append({
                "pairing_id": pairings[p_idx]['id'],
                "uncovered_count": uncovered_count,
                "duration_minutes": pairings[p_idx]['duration_minutes']
            })
            total_uncovered += uncovered_count * pairings[p_idx]['duration_minutes']
    
    return {
        "objective_value": float(total_uncovered),
        "rosters": rosters,
        "uncovered_pairings": uncovered_list
    }

def build_trivial_solution(instance):
    """Build a trivial solution where all pairings are uncovered."""
    pairings = instance['pairings']
    crew_members = instance['crew_members']
    planning_days = instance['planning_horizon_days']
    
    total_uncovered = 0.0
    rosters = {}
    uncovered_list = []
    
    for crew in crew_members:
        total_minutes = planning_days * 1440
        pre_assigned_minutes = sum(act.get('duration_days', (act['end_day'] - act['start_day'] + 1)) * 1440 
                                   for act in crew.get('pre_assigned_activities', []))
        unproductive = max(0, total_minutes - pre_assigned_minutes)
        
        rosters[str(crew['id'])] = {
            "roster_index": 0,
            "pairings": [],
            "unproductive_time": int(unproductive)
        }
    
    for pairing in pairings:
        if pairing['coverage_requirement'] > 0:
            uncovered_list.append({
                "pairing_id": pairing['id'],
                "uncovered_count": pairing['coverage_requirement'],
                "duration_minutes": pairing['duration_minutes']
            })
            total_uncovered += pairing['coverage_requirement'] * pairing['duration_minutes']
    
    return {
        "objective_value": float(total_uncovered),
        "rosters": rosters,
        "uncovered_pairings": uncovered_list
    }

def main():
    parser = argparse.ArgumentParser(description="Airline Crew Rostering Optimizer")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    instance = read_instance(args.instance_path)
    
    # Log trivial solution first
    trivial = build_trivial_solution(instance)
    if logger:
        logger.log_solution(trivial['objective_value'], trivial)
    
    try:
        # Set global for greedy
        global crew_for_pairing_global
        pairings = instance['pairings']
        crew_members = instance['crew_members']
        num_pairings = len(pairings)
        num_crew = len(crew_members)
        
        crew_for_pairing_global = [[] for _ in range(num_pairings)]
        for c_idx, crew in enumerate(crew_members):
            quals = set(crew.get('qualifications', []))
            pre_activities = crew.get('pre_assigned_activities', [])
            for p_idx, pairing in enumerate(pairings):
                if pairing['aircraft_type'] not in quals:
                    continue
                conflict = False
                for act in pre_activities:
                    if pairing_overlaps_activity(pairing, act):
                        conflict = True
                        break
                if not conflict:
                    crew_for_pairing_global[p_idx].append(c_idx)
        
        solution = solve_instance(instance, args.time_limit, logger)
        
        if solution is None:
            solution = trivial
            
    except Exception as e:
        print(f"Error during solving: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        solution = trivial
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)
    
    print(f"Objective value: {solution['objective_value']}")

if __name__ == '__main__':
    main()