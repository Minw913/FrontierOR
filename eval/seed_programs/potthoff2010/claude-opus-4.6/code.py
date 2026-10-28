import json
import argparse
import time
import sys
from collections import defaultdict
from itertools import combinations

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        inst = json.load(f)
    
    params = inst['parameters']
    cost_cancel_ab = params['cost_cancel_A_to_B']
    cost_cancel_aa = params['cost_cancel_A_to_A']
    cost_duty_changed = params['cost_duty_changed']
    cost_task_reassigned = params['cost_task_reassigned']
    cost_new_transfer = params['cost_new_transfer']
    cost_taxi = params['cost_taxi']
    min_conn_time = params['min_connection_time_minutes']
    max_overtime = params.get('max_overtime_minutes', 0)
    
    # Build station info
    stations = {}
    for s in inst['network']['stations']:
        stations[s['id']] = s
    
    crew_bases = set(inst['network']['crew_bases'])
    
    # Build task info
    tasks = {}
    for t in inst['tasks']:
        tasks[t['task_id']] = t
    
    # Build taxi connections
    taxi_conns = {}
    for tc in inst['network']['taxi_connections']:
        taxi_conns[(tc['from_station'], tc['to_station'])] = tc['travel_time_minutes']
        if tc.get('bidirectional', False):
            taxi_conns[(tc['to_station'], tc['from_station'])] = tc['travel_time_minutes']
    
    # Build route segments for shortest path
    route_segments = defaultdict(list)
    for rs in inst['network']['route_segments']:
        route_segments[rs['from_station']].append((rs['to_station'], rs['travel_time_minutes']))
        if rs.get('bidirectional', True):
            route_segments[rs['to_station']].append((rs['from_station'], rs['travel_time_minutes']))
    
    # Shortest path between stations using route segments (for checking connections)
    # We'll use BFS/Dijkstra for taxi travel times
    
    # Canceled tasks from disruption
    disruption_canceled = set(inst['disruption'].get('canceled_tasks', []))
    
    # All duties
    all_duties = []
    for d in inst.get('active_duties', []):
        all_duties.append(d)
    for d in inst.get('reserve_duties', []):
        all_duties.append(d)
    
    # Tasks that need coverage: all tasks minus disruption-canceled ones
    # Tasks in duties that need rescheduling
    all_task_ids = set(tasks.keys())
    coverable_tasks = all_task_ids - disruption_canceled
    
    # For each duty, determine current station and available time
    rescheduling_time = inst['disruption']['rescheduling_time_minutes']
    
    def get_duty_current_station(duty):
        """Get the station where the driver is at rescheduling time."""
        if duty['duty_type'] == 'reserve':
            return duty.get('standby_station', duty['crew_base'])
        
        # For active duties, look at pre_disruption_tasks and current tasks
        pre_tasks = duty.get('pre_disruption_tasks', [])
        if pre_tasks:
            last_pre_task_id = pre_tasks[-1]
            if last_pre_task_id in tasks:
                t = tasks[last_pre_task_id]
                # If task is still in progress at rescheduling time
                if t['departure_time_minutes'] <= rescheduling_time <= t['arrival_time_minutes']:
                    return t['arrival_station']
                elif t['arrival_time_minutes'] <= rescheduling_time:
                    return t['arrival_station']
                else:
                    return t['departure_station']
            else:
                # Task might be in task_details
                for td in duty.get('task_details', []):
                    if td['task_id'] == last_pre_task_id:
                        arr_parts = td['arrival_time'].split(':')
                        arr_min = int(arr_parts[0]) * 60 + int(arr_parts[1])
                        if arr_min <= rescheduling_time:
                            return td['arrival_station']
                        else:
                            return td['arrival_station']
                return duty['crew_base']
        else:
            return duty['crew_base']
    
    def get_duty_available_time(duty):
        """Get the time from which the driver is available."""
        pre_tasks = duty.get('pre_disruption_tasks', [])
        if pre_tasks:
            last_pre_task_id = pre_tasks[-1]
            if last_pre_task_id in tasks:
                t = tasks[last_pre_task_id]
                return max(rescheduling_time, t['arrival_time_minutes'])
            else:
                for td in duty.get('task_details', []):
                    if td['task_id'] == last_pre_task_id:
                        arr_parts = td['arrival_time'].split(':')
                        arr_min = int(arr_parts[0]) * 60 + int(arr_parts[1])
                        return max(rescheduling_time, arr_min)
        return max(rescheduling_time, duty['start_time_minutes'])
    
    # Compute shortest paths between relief points using Dijkstra
    import heapq
    
    relief_points = set()
    for s in inst['network']['stations']:
        if s['is_relief_point']:
            relief_points.add(s['id'])
    
    # Build adjacency for train travel (using train lines for connections)
    # For connection feasibility, we need to know if a driver can get from
    # one task's arrival to another task's departure
    
    # Precompute: can a driver connect from task A arrival to task B departure?
    # Connection requires: arrival_station of A == departure_station of B (same station)
    # OR taxi/train connection with enough time
    # And both stations must be relief points for transfer
    
    # For simplicity with the enumeration approach, let's focus on:
    # 1. Same-station connections with minimum connection time
    # 2. Taxi connections
    
    def can_connect(arr_station, arr_time, dep_station, dep_time, use_taxi=False):
        """Check if driver can get from arr_station at arr_time to dep_station at dep_time."""
        if arr_station == dep_station:
            return dep_time >= arr_time + min_conn_time, False, 0
        
        # Check taxi
        if (arr_station, dep_station) in taxi_conns:
            taxi_time = taxi_conns[(arr_station, dep_station)]
            if dep_time >= arr_time + taxi_time:
                return True, True, taxi_time
        
        return False, False, 0
    
    def can_return_to_base(station, arr_time, base, end_time):
        """Check if driver can return to crew base by end_time."""
        if station == base:
            return True, False, arr_time
        
        # Check taxi
        if (station, base) in taxi_conns:
            taxi_time = taxi_conns[(station, base)]
            ret_time = arr_time + taxi_time
            if ret_time <= end_time:
                return True, True, ret_time
        
        return False, False, 0
    
    # For each duty, enumerate feasible completions
    # A completion is a sequence of tasks that the driver can perform
    
    # Strategy: For each duty, do a DFS/BFS to find feasible task sequences
    # Limit enumeration to keep it tractable
    
    MAX_COMPLETIONS_PER_DUTY = 500
    MAX_ENUM_TIME_PER_DUTY = 2.0  # seconds
    
    # Sort tasks by departure time for efficient enumeration
    task_list = sorted(
        [t for t in tasks.values() if t['task_id'] not in disruption_canceled],
        key=lambda t: t['departure_time_minutes']
    )
    task_by_dep_station = defaultdict(list)
    for t in task_list:
        task_by_dep_station[t['departure_station']].append(t)
    
    # Check route knowledge
    def has_route_knowledge(duty, task):
        """Check if driver has route knowledge for all stations in the task."""
        rk = set(duty.get('route_knowledge', []))
        if not rk:
            return True
        dep = task['departure_station']
        arr = task['arrival_station']
        # The driver needs knowledge of departure and arrival stations at minimum
        # Actually need knowledge of all stations along the route
        # For simplicity, check departure and arrival
        if dep not in rk or arr not in rk:
            return False
        # Check intermediate stations on the line
        line_id = task.get('line_id', '')
        for line in inst.get('train_lines', []):
            if line['line_id'] == line_id:
                line_stations = line['stations']
                dep_idx = None
                arr_idx = None
                for i, s in enumerate(line_stations):
                    if s == dep:
                        dep_idx = i
                    if s == arr:
                        arr_idx = i
                if dep_idx is not None and arr_idx is not None:
                    start_idx = min(dep_idx, arr_idx)
                    end_idx = max(dep_idx, arr_idx)
                    for i in range(start_idx, end_idx + 1):
                        if line_stations[i] not in rk:
                            return False
                break
        return True
    
    def has_rs_license(duty, task):
        """Check if driver has rolling stock license for the task."""
        licenses = set(duty.get('rolling_stock_licenses', []))
        if not licenses:
            return True
        return task.get('rolling_stock_type', '') in licenses
    
    def task_is_feasible_for_duty(duty, task):
        """Check basic feasibility of a task for a duty."""
        return has_route_knowledge(duty, task) and has_rs_license(duty, task)
    
    # Pre-filter tasks for each duty
    duty_feasible_tasks = {}
    for duty in all_duties:
        feas = []
        for t in task_list:
            if task_is_feasible_for_duty(duty, t):
                feas.append(t)
        duty_feasible_tasks[duty['duty_id']] = feas
    
    # Build line station lookup
    line_info = {}
    for line in inst.get('train_lines', []):
        line_info[line['line_id']] = line
    
    # Enumerate completions for each duty
    all_completions = {}  # duty_id -> list of completions
    
    for duty in all_duties:
        duty_id = duty['duty_id']
        base = duty['crew_base']
        current_station = get_duty_current_station(duty)
        available_time = get_duty_available_time(duty)
        duty_end = duty['end_time_minutes'] + max_overtime
        original_tasks = set(duty.get('tasks', []))
        
        feasible_tasks = duty_feasible_tasks[duty_id]
        
        # Filter tasks that depart after available_time (with connection time)
        # and arrive before duty end
        eligible_tasks = []
        for t in feasible_tasks:
            if t['arrival_time_minutes'] <= duty_end:
                eligible_tasks.append(t)
        
        completions = []
        
        # Empty completion (driver does nothing new)
        # Check if driver can return to base from current station
        can_ret, uses_taxi_ret, ret_time = can_return_to_base(
            current_station, available_time, base, duty_end
        )
        
        # Cost of empty completion
        empty_cost = 0
        empty_covered = []
        # If original duty had tasks, not doing them means the duty changed
        if original_tasks:
            empty_cost = cost_duty_changed
        
        if can_ret:
            completions.append({
                'tasks': [],
                'covered': [],
                'cost': empty_cost if original_tasks else 0,
                'uses_taxi': uses_taxi_ret,
                'release_at_relief_point': False,
                'return_station': base,
                'return_time': ret_time if uses_taxi_ret else available_time
            })
        else:
            # Release at relief point
            if current_station in relief_points:
                completions.append({
                    'tasks': [],
                    'covered': [],
                    'cost': empty_cost if original_tasks else 0,
                    'uses_taxi': False,
                    'release_at_relief_point': True,
                    'return_station': current_station,
                    'return_time': available_time
                })
        
        # DFS enumeration of task sequences
        enum_start = time.time()
        
        # Sort eligible tasks by departure time
        eligible_tasks.sort(key=lambda t: t['departure_time_minutes'])
        
        # Index for quick lookup
        tasks_from_station_time = defaultdict(list)
        for t in eligible_tasks:
            tasks_from_station_time[t['departure_station']].append(t)
        
        # BFS/DFS with pruning
        # State: (current_station, current_time, task_sequence)
        # Use iterative DFS with stack
        
        stack = [(current_station, available_time, [], set())]
        visited_states = set()
        
        while stack and len(completions) < MAX_COMPLETIONS_PER_DUTY:
            if time.time() - enum_start > MAX_ENUM_TIME_PER_DUTY:
                break
            
            station, avail_t, task_seq, task_set = stack.pop()
            
            # Try to form a completion from current state
            # Check if can return to base
            can_ret, uses_taxi_ret, ret_time = can_return_to_base(
                station, avail_t, base, duty_end
            )
            
            if task_seq:  # Non-empty completion
                covered = [t['task_id'] for t in task_seq]
                # Calculate cost
                comp_cost = cost_duty_changed
                
                num_reassigned = sum(1 for t in task_seq if t['task_id'] not in original_tasks)
                comp_cost += num_reassigned * cost_task_reassigned
                
                # Count new transfers: consecutive tasks where the second is not 
                # the natural successor in original duty
                num_new_transfers = 0
                orig_task_list = duty.get('tasks', [])
                for i in range(len(task_seq)):
                    if i == 0:
                        # Connection from current position to first task
                        if task_seq[i]['task_id'] not in original_tasks:
                            num_new_transfers += 1
                        elif original_tasks and orig_task_list:
                            # Check if this was the next expected task
                            if orig_task_list[0] != task_seq[i]['task_id']:
                                num_new_transfers += 1
                    else:
                        # Check if consecutive pair was in original
                        prev_id = task_seq[i-1]['task_id']
                        curr_id = task_seq[i]['task_id']
                        is_original_consecutive = False
                        for j in range(len(orig_task_list) - 1):
                            if orig_task_list[j] == prev_id and orig_task_list[j+1] == curr_id:
                                is_original_consecutive = True
                                break
                        if not is_original_consecutive:
                            num_new_transfers += 1
                
                comp_cost += num_new_transfers * cost_new_transfer
                
                uses_taxi_total = uses_taxi_ret
                # Check if any connection in sequence used taxi
                # We need to track this during enumeration
                
                if can_ret:
                    taxi_cost = cost_taxi if uses_taxi_total else 0
                    completions.append({
                        'tasks': [t['task_id'] for t in task_seq],
                        'covered': covered,
                        'cost': comp_cost + taxi_cost,
                        'uses_taxi': uses_taxi_total,
                        'release_at_relief_point': False,
                        'return_station': base,
                        'return_time': ret_time
                    })
                else:
                    # Release at relief point if station is relief point
                    if station in relief_points:
                        completions.append({
                            'tasks': [t['task_id'] for t in task_seq],
                            'covered': covered,
                            'cost': comp_cost,
                            'uses_taxi': False,
                            'release_at_relief_point': True,
                            'return_station': station,
                            'return_time': avail_t
                        })
            
            # Expand: try adding next tasks
            if len(completions) >= MAX_COMPLETIONS_PER_DUTY:
                break
            
            # Find tasks reachable from current station/time
            next_tasks = []
            
            # Same station tasks
            for t in tasks_from_station_time.get(station, []):
                if t['task_id'] in task_set:
                    continue
                if t['departure_time_minutes'] >= avail_t + min_conn_time:
                    if t['arrival_time_minutes'] <= duty_end:
                        next_tasks.append((t, False))
            
            # Taxi connection tasks
            for (from_s, to_s), taxi_time in taxi_conns.items():
                if from_s == station:
                    for t in tasks_from_station_time.get(to_s, []):
                        if t['task_id'] in task_set:
                            continue
                        if t['departure_time_minutes'] >= avail_t + taxi_time:
                            if t['arrival_time_minutes'] <= duty_end:
                                next_tasks.append((t, True))
            
            # Limit branching - take most promising
            next_tasks.sort(key=lambda x: x[0]['departure_time_minutes'])
            
            # Prioritize original tasks
            orig_next = [nt for nt in next_tasks if nt[0]['task_id'] in original_tasks]
            non_orig_next = [nt for nt in next_tasks if nt[0]['task_id'] not in original_tasks]
            
            # Limit expansion
            max_branch = 15
            selected = orig_next[:max_branch] + non_orig_next[:max(0, max_branch - len(orig_next))]
            
            for t, via_taxi in selected:
                new_task_set = task_set | {t['task_id']}
                new_seq = task_seq + [t]
                stack.append((
                    t['arrival_station'],
                    t['arrival_time_minutes'],
                    new_seq,
                    new_task_set
                ))
        
        # If no completions found, add a trivial one
        if not completions:
            if current_station in relief_points:
                completions.append({
                    'tasks': [],
                    'covered': [],
                    'cost': cost_duty_changed if original_tasks else 0,
                    'uses_taxi': False,
                    'release_at_relief_point': True,
                    'return_station': current_station,
                    'return_time': available_time
                })
            else:
                # Must still have a completion
                completions.append({
                    'tasks': [],
                    'covered': [],
                    'cost': cost_duty_changed if original_tasks else 0,
                    'uses_taxi': False,
                    'release_at_relief_point': True,
                    'return_station': current_station,
                    'return_time': available_time
                })
        
        # Check if original duty assignment is feasible (no change needed)
        # Try to find a completion that exactly matches original tasks
        orig_task_ids = duty.get('tasks', [])
        if orig_task_ids:
            # Check if all original tasks are still available (not disruption-canceled)
            all_available = all(tid not in disruption_canceled for tid in orig_task_ids)
            if all_available:
                # Check if original sequence is feasible
                feasible = True
                cur_sta = current_station
                cur_time = available_time
                
                for tid in orig_task_ids:
                    t = tasks.get(tid)
                    if t is None:
                        feasible = False
                        break
                    if t['departure_station'] != cur_sta:
                        feasible = False
                        break
                    if t['departure_time_minutes'] < cur_time + min_conn_time:
                        # Allow if it's the first task and timing works
                        if t['departure_time_minutes'] < cur_time:
                            feasible = False
                            break
                    cur_sta = t['arrival_station']
                    cur_time = t['arrival_time_minutes']
                
                if feasible:
                    can_ret_orig, taxi_orig, ret_time_orig = can_return_to_base(
                        cur_sta, cur_time, base, duty_end
                    )
                    if can_ret_orig and not taxi_orig:
                        # Original assignment is feasible with zero cost
                        # Check if we already have it
                        has_orig = False
                        for c in completions:
                            if c['tasks'] == orig_task_ids and c['cost'] == 0:
                                has_orig = True
                                break
                        if not has_orig:
                            completions.insert(0, {
                                'tasks': orig_task_ids,
                                'covered': orig_task_ids,
                                'cost': 0,
                                'uses_taxi': False,
                                'release_at_relief_point': False,
                                'return_station': base,
                                'return_time': ret_time_orig
                            })
        else:
            # No original tasks - empty completion is zero cost
            # Make sure we have it
            has_empty_zero = False
            for c in completions:
                if not c['tasks'] and c['cost'] == 0:
                    has_empty_zero = True
                    break
            if not has_empty_zero and completions:
                # The empty completion for a duty with no tasks should be zero cost
                for c in completions:
                    if not c['tasks']:
                        c['cost'] = 0
                        break
        
        all_completions[duty_id] = completions
    
    # Now solve the set cover problem with Gurobi
    # Decision variables:
    # x[d,c] = 1 if duty d uses completion c
    # z[t] = 1 if task t is canceled
    
    # Determine which tasks need coverage
    tasks_needing_coverage = set()
    for t_id in coverable_tasks:
        # Check if task is in any duty's original assignment or needs to be covered
        tasks_needing_coverage.add(t_id)
    
    # Build task coverage mapping
    # For each task, which (duty, completion) pairs cover it
    task_coverage = defaultdict(list)  # task_id -> list of (duty_id, comp_idx)
    
    for duty_id, comps in all_completions.items():
        for c_idx, comp in enumerate(comps):
            for t_id in comp['covered']:
                task_coverage[t_id].append((duty_id, c_idx))
    
    elapsed = time.time() - start_time
    remaining_time = max(time_limit - elapsed - 5, 10)
    
    try:
        import gurobipy as gp
        from gurobipy import GRB
        
        model = gp.Model("crew_reschedule")
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', remaining_time)
        model.setParam('OutputFlag', 0)
        
        # Variables
        x = {}  # x[duty_id, comp_idx] = binary
        for duty_id, comps in all_completions.items():
            for c_idx in range(len(comps)):
                x[duty_id, c_idx] = model.addVar(vtype=GRB.BINARY, name=f"x_{duty_id}_{c_idx}")
        
        z = {}  # z[task_id] = binary (1 if canceled)
        for t_id in tasks_needing_coverage:
            z[t_id] = model.addVar(vtype=GRB.BINARY, name=f"z_{t_id}")
        
        model.update()
        
        # Constraints
        # 1. Each duty uses exactly one completion
        for duty_id, comps in all_completions.items():
            model.addConstr(
                gp.quicksum(x[duty_id, c_idx] for c_idx in range(len(comps))) == 1,
                name=f"duty_{duty_id}"
            )
        
        # 2. Each task is either covered or canceled
        for t_id in tasks_needing_coverage:
            coverage_vars = []
            for duty_id, c_idx in task_coverage.get(t_id, []):
                if (duty_id, c_idx) in x:
                    coverage_vars.append(x[duty_id, c_idx])
            
            if coverage_vars:
                model.addConstr(
                    gp.quicksum(coverage_vars) + z[t_id] >= 1,
                    name=f"cover_{t_id}"
                )
            else:
                # No completion covers this task, must cancel
                model.addConstr(z[t_id] == 1, name=f"force_cancel_{t_id}")
        
        # Objective
        obj = gp.LinExpr()
        
        # Completion costs
        for duty_id, comps in all_completions.items():
            for c_idx, comp in enumerate(comps):
                if comp['cost'] > 0:
                    obj += comp['cost'] * x[duty_id, c_idx]
        
        # Cancellation costs
        for t_id in tasks_needing_coverage:
            t = tasks[t_id]
            if t['task_type'] == 'A_to_B':
                obj += cost_cancel_ab * z[t_id]
            else:
                obj += cost_cancel_aa * z[t_id]
        
        model.setObjective(obj, GRB.MINIMIZE)
        
        # Solve
        model.optimize()
        
        # Extract solution
        if model.SolCount > 0:
            solution = extract_solution(model, x, z, all_completions, all_duties, 
                                       tasks, tasks_needing_coverage, cost_cancel_ab, 
                                       cost_cancel_aa, disruption_canceled)
            
            if logger:
                logger.log_solution(solution['objective_value'], solution)
            
            with open(solution_path, 'w') as f:
                json.dump(solution, f, indent=2)
            return
    
    except Exception as e:
        print(f"Gurobi error: {e}", file=sys.stderr)
    
    # Fallback: greedy solution
    solution = greedy_solution(all_completions, all_duties, tasks, 
                              tasks_needing_coverage, cost_cancel_ab, 
                              cost_cancel_aa, disruption_canceled)
    
    if logger:
        logger.log_solution(solution['objective_value'], solution)
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def extract_solution(model, x, z, all_completions, all_duties, tasks, 
                     tasks_needing_coverage, cost_cancel_ab, cost_cancel_aa, 
                     disruption_canceled):
    """Extract solution from Gurobi model."""
    duty_assignments = {}
    total_cost = 0
    canceled_tasks = []
    
    for duty in all_duties:
        duty_id = duty['duty_id']
        comps = all_completions[duty_id]
        
        selected_idx = 0
        for c_idx in range(len(comps)):
            if x[duty_id, c_idx].X > 0.5:
                selected_idx = c_idx
                break
        
        comp = comps[selected_idx]
        total_cost += comp['cost']
        
        duty_assignments[duty_id] = {
            'completion_index': selected_idx,
            'cost': comp['cost'],
            'covered_tasks': comp['covered'],
            'uses_taxi': comp['uses_taxi'],
            'release_at_relief_point': comp['release_at_relief_point'],
            'return_station': comp['return_station'],
            'return_time_minutes': comp['return_time']
        }
    
    for t_id in tasks_needing_coverage:
        if z[t_id].X > 0.5:
            canceled_tasks.append(t_id)
            t = tasks[t_id]
            if t['task_type'] == 'A_to_B':
                total_cost += cost_cancel_ab
            else:
                total_cost += cost_cancel_aa
    
    # Add disruption-canceled tasks
    canceled_tasks.extend(list(disruption_canceled))
    
    return {
        'released_problem_variant': 'relief_point_completion_v1',
        'objective_value': total_cost,
        'canceled_tasks': canceled_tasks,
        'duty_assignments': duty_assignments
    }


def greedy_solution(all_completions, all_duties, tasks, tasks_needing_coverage,
                    cost_cancel_ab, cost_cancel_aa, disruption_canceled):
    """Simple greedy: each duty picks cheapest completion, cancel uncovered tasks."""
    duty_assignments = {}
    total_cost = 0
    covered = set()
    
    for duty in all_duties:
        duty_id = duty['duty_id']
        comps = all_completions[duty_id]
        
        # Pick completion with best coverage/cost ratio
        # Simple: pick the one that covers most original tasks at lowest cost
        best_idx = 0
        best_score = float('inf')
        
        for c_idx, comp in enumerate(comps):
            score = comp['cost'] - len(comp['covered']) * 100
            if score < best_score:
                best_score = score
                best_idx = c_idx
        
        comp = comps[best_idx]
        total_cost += comp['cost']
        covered.update(comp['covered'])
        
        duty_assignments[duty_id] = {
            'completion_index': best_idx,
            'cost': comp['cost'],
            'covered_tasks': comp['covered'],
            'uses_taxi': comp['uses_taxi'],
            'release_at_relief_point': comp['release_at_relief_point'],
            'return_station': comp['return_station'],
            'return_time_minutes': comp['return_time']
        }
    
    canceled_tasks = list(disruption_canceled)
    for t_id in tasks_needing_coverage:
        if t_id not in covered:
            canceled_tasks.append(t_id)
            t = tasks[t_id]
            if t['task_type'] == 'A_to_B':
                total_cost += cost_cancel_ab
            else:
                total_cost += cost_cancel_aa
    
    return {
        'released_problem_variant': 'relief_point_completion_v1',
        'objective_value': total_cost,
        'canceled_tasks': canceled_tasks,
        'duty_assignments': duty_assignments
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)