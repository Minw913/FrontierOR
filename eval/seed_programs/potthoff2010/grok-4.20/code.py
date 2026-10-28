import json
import argparse
import time
import sys
from collections import defaultdict
from solution_logger import SolutionLogger

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("Gurobi not available. Please ensure gurobipy is installed.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    # Extract key data
    tasks = instance['tasks']
    active_duties = instance['active_duties']
    reserve_duties = instance['reserve_duties']
    all_duties = active_duties + reserve_duties
    parameters = instance['parameters']
    network = instance['network']
    disruption = instance['disruption']

    task_dict = {t['task_id']: t for t in tasks}
    duty_dict = {d['duty_id']: d for d in all_duties}

    # Only unfinished tasks after rescheduling time
    resched_time = instance['disruption']['rescheduling_time_minutes']
    unfinished_tasks = []
    for t in tasks:
        if t['departure_time_minutes'] >= resched_time and t['task_id'] not in disruption.get('canceled_tasks', []):
            unfinished_tasks.append(t['task_id'])

    # Precompute possible task assignments per duty (simple feasibility filter)
    possible_assignments = defaultdict(list)
    for duty in all_duties:
        duty_id = duty['duty_id']
        licenses = set(duty['rolling_stock_licenses'])
        route_know = set(duty['route_knowledge'])
        current_station = duty.get('current_station', duty.get('standby_station', duty['crew_base']))
        for tid in unfinished_tasks:
            task = task_dict[tid]
            if task['rolling_stock_type'] in licenses and task['departure_station'] in route_know:
                possible_assignments[duty_id].append(tid)

    # Simple greedy heuristic: assign tasks to first compatible duty, prefer original
    assignment = {}
    covered = set()
    canceled = set(unfinished_tasks)
    total_cost = 0
    duty_assignments = {}

    # First pass: keep original tasks where possible
    for duty in all_duties:
        duty_id = duty['duty_id']
        original_tasks = duty.get('tasks', [])
        feasible_original = [t for t in original_tasks if t in unfinished_tasks and t not in covered]
        if feasible_original:
            assignment[duty_id] = feasible_original
            for t in feasible_original:
                covered.add(t)
                canceled.discard(t)
            duty_assignments[duty_id] = {
                "completion_index": 0,
                "cost": 0.0,
                "covered_tasks": feasible_original[:],
                "uses_taxi": False,
                "release_at_relief_point": False,
                "return_station": duty['crew_base'],
                "return_time_minutes": duty.get('end_time_minutes', resched_time + 60)
            }

    # Second pass: assign remaining tasks greedily
    for tid in sorted(unfinished_tasks):
        if tid in covered:
            continue
        task = task_dict[tid]
        assigned = False
        for duty in all_duties:
            duty_id = duty['duty_id']
            if tid in possible_assignments[duty_id]:
                if duty_id not in assignment:
                    assignment[duty_id] = []
                assignment[duty_id].append(tid)
                covered.add(tid)
                canceled.discard(tid)
                assigned = True
                
                # Compute simple cost
                change_cost = parameters['cost_duty_changed']
                reassigned = len(assignment[duty_id])
                transfer_cost = max(0, len(assignment[duty_id]) - 1) * parameters['cost_new_transfer']
                comp_cost = change_cost + reassigned * parameters['cost_task_reassigned'] + transfer_cost
                
                duty_assignments[duty_id] = {
                    "completion_index": 1,
                    "cost": float(comp_cost),
                    "covered_tasks": assignment[duty_id][:],
                    "uses_taxi": False,
                    "release_at_relief_point": False,
                    "return_station": duty['crew_base'],
                    "return_time_minutes": task['arrival_time_minutes'] + 30
                }
                total_cost += comp_cost
                break
        if not assigned:
            # Must cancel
            cancel_cost = (parameters['cost_cancel_A_to_B'] 
                          if task['task_type'] == 'A_to_B' 
                          else parameters['cost_cancel_A_to_A'])
            total_cost += cancel_cost

    # Add remaining unchanged duties with zero cost
    for duty in all_duties:
        duty_id = duty['duty_id']
        if duty_id not in duty_assignments:
            duty_assignments[duty_id] = {
                "completion_index": 0,
                "cost": 0.0,
                "covered_tasks": [],
                "uses_taxi": False,
                "release_at_relief_point": False,
                "return_station": duty['crew_base'],
                "return_time_minutes": duty.get('end_time_minutes', resched_time + 30)
            }

    objective_value = float(total_cost)

    solution = {
        "released_problem_variant": "relief_point_completion_v1",
        "objective_value": objective_value,
        "canceled_tasks": sorted(list(canceled)),
        "duty_assignments": duty_assignments
    }

    # Log the solution
    if logger:
        logger.log_solution(objective_value, solution)

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    elapsed = time.time() - start_time
    if elapsed > args.time_limit:
        print(f"Warning: Exceeded time limit. Took {elapsed:.2f}s", file=sys.stderr)

    print(f"Solution written with objective {objective_value}")


if __name__ == "__main__":
    main()