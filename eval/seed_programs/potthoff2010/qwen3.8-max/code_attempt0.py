#!/usr/bin/env python3
"""
Railway Crew Rescheduling Solver
Variant: relief_point_completion_v1
"""

import json
import argparse
import time
import heapq
import sys
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Railway Crew Rescheduling Solver')
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    return parser.parse_args()


def compute_travel_times(instance):
    """Compute shortest travel times between all pairs of stations using route segments."""
    network = instance['network']
    stations = [s['id'] for s in network['stations']]
    
    adj = defaultdict(list)
    for seg in network['route_segments']:
        adj[seg['from_station']].append((seg['to_station'], seg['travel_time_minutes']))
        if seg.get('bidirectional', True):
            adj[seg['to_station']].append((seg['from_station'], seg['travel_time_minutes']))
    
    travel_time = {}
    for source in stations:
        dist = {s: float('inf') for s in stations}
        dist[source] = 0
        pq = [(0, source)]
        visited = set()
        
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            for v, w in adj.get(u, []):
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    heapq.heappush(pq, (dist[v], v))
        
        travel_time[source] = dist
    
    return travel_time


def compute_taxi_times(instance):
    """Get taxi travel times between station pairs."""
    taxi = {}
    for conn in instance['network'].get('taxi_connections', []):
        taxi[(conn['from_station'], conn['to_station'])] = conn['travel_time_minutes']
        if conn.get('bidirectional', True):
            taxi[(conn['to_station'], conn['from_station'])] = conn['travel_time_minutes']
    return taxi


def get_best_travel_time(from_st, to_st, travel_time, taxi_times):
    """Get the best travel time between two stations (train or taxi)."""
    if from_st == to_st:
        return 0, False
    train_tt = travel_time.get(from_st, {}).get(to_st, float('inf'))
    taxi_tt = taxi_times.get((from_st, to_st), float('inf'))
    if taxi_tt < train_tt:
        return taxi_tt, True
    return train_tt, False


def get_driver_current_position(duty, tasks, rescheduling_time):
    """Determine the driver's current station and effective start time."""
    if duty['duty_type'] == 'reserve':
        return duty.get('standby_station', duty['crew_base']), rescheduling_time
    
    pre_tasks = duty.get('pre_disruption_tasks', [])
    if not pre_tasks:
        return duty['crew_base'], max(duty.get('start_time_minutes', 0), rescheduling_time)
    
    last_station = duty['crew_base']
    last_time = duty.get('start_time_minutes', 0)
    
    for tid in pre_tasks:
        if tid in tasks:
            task = tasks[tid]
            if task['arrival_time_minutes'] <= rescheduling_time:
                last_station = task['arrival_station']
                last_time = task['arrival_time_minutes']
            else:
                last_station = task['arrival_station']
                last_time = task['arrival_time_minutes']
                break
    
    return last_station, max(last_time, rescheduling_time)


def check_sequence_feasibility(task_ids, tasks, start_station, start_time,
                                travel_time, taxi_times, params, max_duty_end, crew_base):
    """Check if a sequence of tasks is feasible and determine return info."""
    current_station = start_station
    current_time = start_time
    min_conn = params.get('min_connection_time_minutes', 0)
    uses_taxi = False
    
    for tid in task_ids:
        if tid not in tasks:
            return None
        task = tasks[tid]
        dep_station = task['departure_station']
        dep_time = task['departure_time_minutes']
        
        if dep_time < current_time:
            return None
        
        if current_station != dep_station:
            tt, is_taxi = get_best_travel_time(current_station, dep_station, travel_time, taxi_times)
            if tt == float('inf') or current_time + tt > dep_time:
                return None
            if is_taxi:
                uses_taxi = True
        
        if task['arrival_time_minutes'] > max_duty_end:
            return None
        
        current_station = task['arrival_station']
        current_time = task['arrival_time_minutes'] + min_conn
    
    # Check return to base
    release_at_relief_point = False
    return_station = crew_base
    return_time = current_time
    
    if current_station != crew_base:
        tt, is_taxi = get_best_travel_time(current_station, crew_base, travel_time, taxi_times)
        if tt == float('inf') or current_time + tt > max_duty_end:
            # Can't return to base - use relief point release
            release_at_relief_point = True
            return_station = current_station
            return_time = current_time - min_conn if task_ids else current_time
        else:
            if is_taxi:
                uses_taxi = True
            return_time = current_time + tt
    else:
        return_time = current_time - min_conn if task_ids else current_time
    
    return {
        'feasible': True,
        'uses_taxi': uses_taxi,
        'release_at_relief_point': release_at_relief_point,
        'return_station': return_station,
        'return_time_minutes': int(return_time)
    }


def compute_completion_cost(duty, comp_task_ids, tasks, params):
    """Compute the released cost of a completion."""
    original_tasks = duty.get('tasks', [])
    
    if comp_task_ids == original_tasks:
        return 0.0
    
    if not comp_task_ids and not original_tasks:
        return 0.0
    
    cost = float(params.get('cost_duty_changed', 0))
    
    original_set = set(original_tasks)
    new_tasks = [t for t in comp_task_ids if t not in original_set]
    cost += params.get('cost_task_reassigned', 0) * len(new_tasks)
    
    # Count transfers in completion
    transfers = 0
    for i in range(len(comp_task_ids) - 1):
        if comp_task_ids[i] in tasks and comp_task_ids[i + 1] in tasks:
            t1 = tasks[comp_task_ids[i]]
            t2 = tasks[comp_task_ids[i + 1]]
            if t1['arrival_station'] != t2['departure_station']:
                transfers += 1
    
    # Count transfers in original
    orig_transfers = 0
    for i in range(len(original_tasks) - 1):
        if original_tasks[i] in tasks and original_tasks[i + 1] in tasks:
            t1 = tasks[original_tasks[i]]
            t2 = tasks[original_tasks[i + 1]]
            if t1['arrival_station'] != t2['departure_station']:
                orig_transfers += 1
    
    new_transfers = max(0, transfers - orig_transfers)
    cost += params.get('cost_new_transfer', 0) * new_transfers
    
    return cost


def generate_duty_completions(duty, tasks, travel_time, taxi_times, params,
                               rescheduling_time, global_start, time_limit):
    """Generate feasible completions for a duty."""
    route_knowledge = set(duty.get('route_knowledge', []))
    rolling_stock_licenses = set(duty.get('rolling_stock_licenses', []))
    crew_base = duty['crew_base']
    max_duty_end = duty.get('end_time_minutes', 1440) + params.get('max_overtime_minutes', 60)
    
    current_station, current_time = get_driver_current_position(duty, tasks, rescheduling_time)
    
    # Filter available tasks
    available = []
    for tid, task in tasks.items():
        if task['departure_station'] not in route_knowledge:
            continue
        if task['arrival_station'] not in route_knowledge:
            continue
        if task['rolling_stock_type'] not in rolling_stock_licenses:
            continue
        if task['departure_time_minutes'] < current_time:
            continue
        if task['arrival_time_minutes'] > max_duty_end:
            continue
        available.append(task)
    
    available.sort(key=lambda t: (t['departure_time_minutes'], t['task_id']))
    
    completions = []
    
    # 1. Empty completion
    empty_result = check_sequence_feasibility([], tasks, current_station, current_time,
                                               travel_time, taxi_times, params, max_duty_end, crew_base)
    completions.append({
        'tasks': [],
        'uses_taxi': False,
        'release_at_relief_point': False,
        'return_station': current_station,
        'return_time_minutes': int(current_time)
    })
    
    # 2. Original completion
    original_task_ids = [t for t in duty.get('tasks', []) if t in tasks]
    if original_task_ids:
        result = check_sequence_feasibility(original_task_ids, tasks, current_station, current_time,
                                             travel_time, taxi_times, params, max_duty_end, crew_base)
        if result and result['feasible']:
            completions.append({
                'tasks': original_task_ids[:],
                'uses_taxi': result['uses_taxi'],
                'release_at_relief_point': result['release_at_relief_point'],
                'return_station': result['return_station'],
                'return_time_minutes': result['return_time_minutes']
            })
    
    # 3. DFS generation of additional completions
    max_completions = 150
    min_conn = params.get('min_connection_time_minutes', 0)
    time_budget = max(0.5, (time_limit - (time.time() - global_start)) * 0.4)
    dfs_start = time.time()
    
    def can_reach(from_st, to_st, by_time):
        if from_st == to_st:
            return True, False
        tt, is_taxi = get_best_travel_time(from_st, to_st, travel_time, taxi_times)
        if tt == float('inf'):
            return False, False
        return True, is_taxi
    
    def dfs(cur_station, cur_time, path, start_idx):
        if len(completions) >= max_completions:
            return
        if time.time() - dfs_start > time_budget:
            return
        
        for i in range(start_idx, len(available)):
            task = available[i]
            tid = task['task_id']
            
            if tid in path:
                continue
            
            dep_station = task['departure_station']
            dep_time = task['departure_time_minutes']
            
            if dep_time < cur_time:
                continue
            
            # Check reachability
            if cur_station != dep_station:
                reachable, _ = can_reach(cur_station, dep_station, dep_time)
                if not reachable:
                    continue
                tt, _ = get_best_travel_time(cur_station, dep_station, travel_time, taxi_times)
                if cur_time + tt > dep_time:
                    continue
            
            if task['arrival_time_minutes'] > max_duty_end:
                continue
            
            new_path = path + [tid]
            new_station = task['arrival_station']
            new_time = task['arrival_time_minutes'] + min_conn
            
            # Check if this partial path can lead to a valid completion
            result = check_sequence_feasibility(new_path, tasks, current_station, current_time,
                                                 travel_time, taxi_times, params, max_duty_end, crew_base)
            if result and result['feasible']:
                completions.append({
                    'tasks': new_path[:],
                    'uses_taxi': result['uses_taxi'],
                    'release_at_relief_point': result['release_at_relief_point'],
                    'return_station': result['return_station'],
                    'return_time_minutes': result['return_time_minutes']
                })
            
            # Recurse if we have budget
            if len(completions) < max_completions and time.time() - dfs_start < time_budget:
                dfs(new_station, new_time, new_path, i + 1)
    
    dfs(current_station, current_time, [], 0)
    
    return completions


def build_greedy_solution(duties, all_completions, tasks, tasks_to_schedule, params):
    """Build a greedy solution as initial/fallback."""
    solution = {
        'released_problem_variant': 'relief_point_completion_v1',
        'canceled_tasks': [],
        'duty_assignments': {}
    }
    
    # For each duty, pick the best completion (prefer unchanged, then lowest cost)
    covered_tasks = set()
    
    for duty in duties:
        duty_id = duty['duty_id']
        completions = all_completions[duty_id]
        original_tasks = duty.get('tasks', [])
        
        best_idx = 0
        best_cost = float('inf')
        
        for c_idx, comp in enumerate(completions):
            cost = compute_completion_cost(duty, comp['tasks'], tasks, params)
            # Prefer completions that cover more uncovered tasks
            uncovered_covered = len(set(comp['tasks']) - covered_tasks)
            score = cost - uncovered_covered * 0.01
            
            if comp['tasks'] == original_tasks:
                score = -1  # Strongly prefer unchanged
            
            if score < best_cost:
                best_cost = score
                best_idx = c_idx
        
        comp = completions[best_idx]
        covered_tasks.update(comp['tasks'])
        
        cost = compute_completion_cost(duty, comp['tasks'], tasks, params)
        solution['duty_assignments'][duty_id] = {
            'completion_index': best_idx,
            'cost': cost,
            'covered_tasks': comp['tasks'],
            'uses_taxi': comp.get('uses_taxi', False),
            'release_at_relief_point': comp.get('release_at_relief_point', False),
            'return_station': comp.get('return_station', duty['crew_base']),
            'return_time_minutes': comp.get('return_time_minutes', 0)
        }
    
    # Determine canceled tasks
    for tid in tasks_to_schedule:
        if tid not in covered_tasks:
            solution['canceled_tasks'].append(tid)
    
    # Compute objective
    total_cost = 0.0
    for duty_id, assignment in solution['duty_assignments'].items():
        total_cost += assignment['cost']
    
    for tid in solution['canceled_tasks']:
        task = tasks[tid]
        if task['task_type'] == 'A_to_B':
            total_cost += params.get('cost_cancel_A_to_B', 1000)
        else:
            total_cost += params.get('cost_cancel_A_to_A', 100)
    
    solution['objective_value'] = total_cost
    return solution


def solve_with_gurobi(duties, all_completions, tasks, tasks_to_schedule, params,
                      time_limit, global_start, logger):
    """Solve the set covering problem using Gurobi."""
    import gurobipy as gp
    from gurobipy import GRB
    
    elapsed = time.time() - global_start
    remaining = max(1, time_limit - elapsed - 3)
    
    model = gp.Model("crew_rescheduling")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', remaining)
    model.setParam('OutputFlag', 0)
    
    # Variables
    x = {}
    for duty in duties:
        duty_id = duty['duty_id']
        n_comp = len(all_completions[duty_id])
        for c_idx in range(n_comp):
            x[duty_id, c_idx] = model.addVar(vtype=GRB.BINARY, name=f"x_{duty_id}_{c_idx}")
    
    y = {}
    for tid in tasks_to_schedule:
        y[tid] = model.addVar(vtype=GRB.BINARY, name=f"y_{tid}")
    
    model.update()
    
    # Objective
    obj = gp.LinExpr()
    
    comp_costs = {}
    for duty in duties:
        duty_id = duty['duty_id']
        for c_idx, comp in enumerate(all_completions[duty_id]):
            cost = compute_completion_cost(duty, comp['tasks'], tasks, params)
            # Add taxi cost if applicable
            if comp.get('uses_taxi', False):
                cost += params.get('cost_taxi', 0)
            comp_costs[duty_id, c_idx] = cost
            obj += cost * x[duty_id, c_idx]
    
    for tid in tasks_to_schedule:
        task = tasks[tid]
        if task['task_type'] == 'A_to_B':
            cancel_cost = params.get('cost_cancel_A_to_B', 1000)
        else:
            cancel_cost = params.get('cost_cancel_A_to_A', 100)
        obj += cancel_cost * y[tid]
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Constraint: each duty gets exactly one completion
    for duty in duties:
        duty_id = duty['duty_id']
        n_comp = len(all_completions[duty_id])
        model.addConstr(
            gp.quicksum(x[duty_id, c_idx] for c_idx in range(n_comp)) == 1,
            name=f"one_completion_{duty_id}"
        )
    
    # Constraint: each task is covered or canceled
    # Pre-compute which completions cover which tasks
    task_coverage = defaultdict(list)
    for duty in duties:
        duty_id = duty['duty_id']
        for c_idx, comp in enumerate(all_completions[duty_id]):
            for tid in comp['tasks']:
                if tid in tasks_to_schedule:
                    task_coverage[tid].append((duty_id, c_idx))
    
    for tid in tasks_to_schedule:
        covering_vars = [x[duty_id, c_idx] for duty_id, c_idx in task_coverage[tid]]
        if covering_vars:
            model.addConstr(
                gp.quicksum(covering_vars) + y[tid] >= 1,
                name=f"cover_{tid}"
            )
        else:
            model.addConstr(y[tid] == 1, name=f"must_cancel_{tid}")
    
    # Set initial solution from greedy
    greedy = build_greedy_solution(duties, all_completions, tasks, tasks_to_schedule, params)
    
    try:
        for duty in duties:
            duty_id = duty['duty_id']
            n_comp = len(all_completions[duty_id])
            sel_idx = greedy['duty_assignments'][duty_id]['completion_index']
            for c_idx in range(n_comp):
                x[duty_id, c_idx].Start = 1.0 if c_idx == sel_idx else 0.0
        
        for tid in tasks_to_schedule:
            y[tid].Start = 1.0 if tid in greedy['canceled_tasks'] else 0.0
    except Exception:
        pass
    
    # Solve
    model.optimize()
    
    # Extract solution
    if model.SolCount > 0:
        solution = {
            'released_problem_variant': 'relief_point_completion_v1',
            'objective_value': model.ObjVal,
            'canceled_tasks': [],
            'duty_assignments': {}
        }
        
        for tid in tasks_to_schedule:
            if y[tid].X > 0.5:
                solution['canceled_tasks'].append(tid)
        
        for duty in duties:
            duty_id = duty['duty_id']
            n_comp = len(all_completions[duty_id])
            
            selected_idx = 0
            for c_idx in range(n_comp):
                if x[duty_id, c_idx].X > 0.5:
                    selected_idx = c_idx
                    break
            
            comp = all_completions[duty_id][selected_idx]
            cost = comp_costs[duty_id, selected_idx]
            
            solution['duty_assignments'][duty_id] = {
                'completion_index': selected_idx,
                'cost': cost,
                'covered_tasks': comp['tasks'],
                'uses_taxi': comp.get('uses_taxi', False),
                'release_at_relief_point': comp.get('release_at_relief_point', False),
                'return_station': comp.get('return_station', duty['crew_base']),
                'return_time_minutes': comp.get('return_time_minutes', 0)
            }
        
        return solution
    
    return None


def solve(instance, time_limit, global_start, logger):
    """Main solve function."""
    params = instance['parameters']
    disruption = instance['disruption']
    rescheduling_time = disruption['rescheduling_time_minutes']
    
    tasks = {t['task_id']: t for t in instance['tasks']}
    duties = instance['active_duties'] + instance['reserve_duties']
    
    # Compute travel times
    travel_time = compute_travel_times(instance)
    taxi_times = compute_taxi_times(instance)
    
    # Determine tasks to schedule
    pre_disruption = set()
    for d in duties:
        pre_disruption.update(d.get('pre_disruption_tasks', []))
    
    canceled_by_disruption = set(disruption.get('canceled_tasks', []))
    tasks_to_schedule = set(tasks.keys()) - pre_disruption - canceled_by_disruption
    
    # Generate completions for each duty
    all_completions = {}
    for duty in duties:
        duty_id = duty['duty_id']
        completions = generate_duty_completions(
            duty, tasks, travel_time, taxi_times, params,
            rescheduling_time, global_start, time_limit
        )
        all_completions[duty_id] = completions
    
    # Try Gurobi
    solution = None
    try:
        solution = solve_with_gurobi(
            duties, all_completions, tasks, tasks_to_schedule, params,
            time_limit, global_start, logger
        )
    except Exception as e:
        print(f"Gurobi failed: {e}", file=sys.stderr)
    
    if solution is None:
        # Fallback to greedy
        solution = build_greedy_solution(duties, all_completions, tasks, tasks_to_schedule, params)
    
    if logger:
        logger.log_solution(solution['objective_value'], solution)
    
    return solution


def main():
    args = parse_args()
    global_start = time.time()
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    solution = solve(instance, args.time_limit, global_start, logger)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()