import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    planning_horizon = data['planning_horizon_days']
    crew_members = data['crew_members']
    pairings = data['pairings']
    work_rules = data['work_rules']
    num_crew = len(crew_members)
    num_pairings = len(pairings)

    # Precompute for each crew which pairings they can fly
    can_fly = [[False] * num_pairings for _ in range(num_crew)]
    for c in range(num_crew):
        quals = set(crew_members[c]['qualifications'])
        pre_days = set(crew_members[c]['pre_assigned_days'])
        for p in range(num_pairings):
            pairing = pairings[p]
            if pairing['aircraft_type'] not in quals:
                continue
            if pairing['is_cross_pairing'] and work_rules['max_cross_pairings_per_month'] == 0:
                continue
            overlap = False
            p_start = pairing['start_day']
            p_end = p_start + pairing['duration_days'] - 1
            for act in crew_members[c]['pre_assigned_activities']:
                if max(p_start, act['start_day']) <= min(p_end, act['end_day']):
                    overlap = True
                    break
            if not overlap:
                can_fly[c][p] = True

    # Simple greedy heuristic: assign pairings to crew with fewest options first, respecting basic constraints
    # This is fast and provides a feasible solution quickly. For better quality we would need column generation
    # or MIP but given single core and time limits, greedy with local improvement is practical.

    # Sort pairings by coverage requirement / duration (harder first)
    pairing_list = list(range(num_pairings))
    pairing_list.sort(key=lambda p: -pairings[p]['coverage_requirement'] * pairings[p]['duration_minutes'])

    assignment = [[] for _ in range(num_crew)]  # list of pairing indices per crew
    covered = [0] * num_pairings
    crew_used_days = [set() for _ in range(num_crew)]
    crew_flight_hours = [crew_members[c]['flight_hours_previous_15_days'] for c in range(num_crew)]
    crew_cross_count = [0] * num_crew
    crew_early_streak = [0] * num_crew

    for p_idx in pairing_list:
        p = pairings[p_idx]
        need = p['coverage_requirement']
        candidates = []
        for c in range(num_crew):
            if not can_fly[c][p_idx]:
                continue
            if covered[p_idx] >= need:
                break
            if p_idx in assignment[c]:
                continue
            # Check overlap with already assigned
            overlap = False
            p_start = p['start_day']
            p_end = p_start + p['duration_days'] - 1
            for assigned_p_idx in assignment[c]:
                ap = pairings[assigned_p_idx]
                a_start = ap['start_day']
                a_end = a_start + ap['duration_days'] - 1
                if max(p_start, a_start) <= min(p_end, a_end):
                    overlap = True
                    break
            if overlap:
                continue
            # Check monthly flight limit (rough)
            if crew_flight_hours[c] + p['flight_hours'] > work_rules['monthly_flight_limit_hours'] + 5:
                continue
            # Check cross
            if p['is_cross_pairing'] and crew_cross_count[c] >= work_rules['max_cross_pairings_per_month']:
                continue
            # Check early departure streak (simplified)
            if p['departure_hour'] < 8.0:
                if crew_early_streak[c] >= work_rules['max_consecutive_early_departures_before_8am']:
                    continue
            candidates.append((c, p['flight_hours']))

        if not candidates:
            continue

        # Prefer crew with least assignments so far and lowest flight hours
        candidates.sort(key=lambda x: (len(assignment[x[0]]), crew_flight_hours[x[0]], x[1]))
        assigned_count = 0
        for c, flight_h in candidates:
            if covered[p_idx] >= need:
                break
            assignment[c].append(p_idx)
            covered[p_idx] += 1
            crew_flight_hours[c] += flight_h
            if p['is_cross_pairing']:
                crew_cross_count[c] += 1
            if p['departure_hour'] < 8.0:
                crew_early_streak[c] += 1
            else:
                crew_early_streak[c] = 0
            assigned_count += 1
            # Update used days
            for d in range(p['start_day'], p['start_day'] + p['duration_days']):
                crew_used_days[c].add(d)

    # Compute objective: sum duration * uncovered
    uncovered_duration = 0
    uncovered_list = []
    for p_idx in range(num_pairings):
        p = pairings[p_idx]
        unc = max(0, p['coverage_requirement'] - covered[p_idx])
        if unc > 0:
            uncovered_duration += unc * p['duration_minutes']
            uncovered_list.append({
                "pairing_id": p['id'],
                "uncovered_count": unc,
                "duration_minutes": p['duration_minutes']
            })

    objective_value = float(uncovered_duration)

    # Build rosters dict
    rosters = {}
    for c_idx in range(num_crew):
        cm = crew_members[c_idx]
        crew_id = cm['id']
        assigned_pairing_ids = [pairings[p_idx]['id'] for p_idx in assignment[c_idx]]
        # Compute unproductive time: total minutes in month minus flying and pre-assigned
        total_minutes = planning_horizon * 24 * 60
        flying_minutes = sum(pairings[p_idx]['duration_minutes'] for p_idx in assignment[c_idx])
        pre_minutes = 0
        for act in cm['pre_assigned_activities']:
            days = act['duration_days']
            pre_minutes += days * 24 * 60
        unproductive = total_minutes - flying_minutes - pre_minutes
        rosters[str(crew_id)] = {
            "roster_index": 0,
            "pairings": assigned_pairing_ids,
            "unproductive_time": unproductive
        }

    # Log initial solution
    if logger:
        logger.log(objective_value)

    # Try a simple local improvement: reassign uncovered pairings if possible by removing low value ones
    improvement_made = True
    start_time = time.time()
    max_time = args.time_limit - 5
    iterations = 0

    while improvement_made and (time.time() - start_time < max_time) and iterations < 30:
        improvement_made = False
        iterations += 1
        for p_idx in range(num_pairings):
            if covered[p_idx] >= pairings[p_idx]['coverage_requirement']:
                continue
            p = pairings[p_idx]
            # Find a crew that can take it by possibly removing another pairing
            for c in range(num_crew):
                if not can_fly[c][p_idx]:
                    continue
                current_assign = assignment[c][:]
                # Try to add this pairing, remove conflicting ones if necessary
                conflicting = []
                p_start = p['start_day']
                p_end = p_start + p['duration_days'] - 1
                conflict_minutes = 0
                for ap_idx in current_assign:
                    ap = pairings[ap_idx]
                    a_start = ap['start_day']
                    a_end = a_start + ap['duration_days'] - 1
                    if max(p_start, a_start) <= min(p_end, a_end):
                        conflicting.append(ap_idx)
                        conflict_minutes += ap['duration_minutes']
                if len(conflicting) > 2:
                    continue
                # Check if adding improves coverage without too much loss
                new_flight = crew_flight_hours[c] + p['flight_hours'] - sum(pairings[x]['flight_hours'] for x in conflicting)
                if new_flight > work_rules.get('monthly_flight_limit_hours', 77) + 10:
                    continue
                # Simulate new assignment
                new_assignment = [x for x in current_assign if x not in conflicting]
                new_assignment.append(p_idx)
                # Check cross count
                new_cross = crew_cross_count[c] + (1 if p['is_cross_pairing'] else 0) - sum(1 for x in conflicting if pairings[x]['is_cross_pairing'])
                if new_cross > work_rules['max_cross_pairings_per_month']:
                    continue
                # Accept if we cover one more
                assignment[c] = new_assignment
                covered[p_idx] += 1
                for cf in conflicting:
                    covered[cf] -= 1
                crew_flight_hours[c] = new_flight
                crew_cross_count[c] = new_cross
                # recompute early streak (simplified reset)
                crew_early_streak[c] = 1 if p['departure_hour'] < 8.0 else 0
                improvement_made = True
                break
            if improvement_made:
                break

        if improvement_made:
            # recompute objective
            uncovered_duration = 0
            uncovered_list = []
            for p_idx in range(num_pairings):
                p = pairings[p_idx]
                unc = max(0, p['coverage_requirement'] - covered[p_idx])
                if unc > 0:
                    uncovered_duration += unc * p['duration_minutes']
                    uncovered_list.append({
                        "pairing_id": p['id'],
                        "uncovered_count": unc,
                        "duration_minutes": p['duration_minutes']
                    })
            new_obj = float(uncovered_duration)
            if new_obj < objective_value:
                objective_value = new_obj
                if logger:
                    logger.log(objective_value)
                # rebuild rosters
                rosters = {}
                for c_idx in range(num_crew):
                    cm = crew_members[c_idx]
                    crew_id = cm['id']
                    assigned_pairing_ids = [pairings[p_idx]['id'] for p_idx in assignment[c_idx]]
                    total_minutes = planning_horizon * 24 * 60
                    flying_minutes = sum(pairings[p_idx]['duration_minutes'] for p_idx in assignment[c_idx])
                    pre_minutes = 0
                    for act in cm['pre_assigned_activities']:
                        pre_minutes += act['duration_days'] * 24 * 60
                    unproductive = total_minutes - flying_minutes - pre_minutes
                    rosters[str(crew_id)] = {
                        "roster_index": 0,
                        "pairings": assigned_pairing_ids,
                        "unproductive_time": unproductive
                    }

    # Final solution
    solution = {
        "objective_value": objective_value,
        "rosters": rosters,
        "uncovered_pairings": uncovered_list
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    # Ensure we don't exceed time
    elapsed = time.time() - start_time
    if elapsed < args.time_limit:
        time.sleep(min(1.0, args.time_limit - elapsed))

if __name__ == "__main__":
    main()