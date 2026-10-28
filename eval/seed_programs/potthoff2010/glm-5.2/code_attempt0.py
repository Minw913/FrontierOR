import json
import argparse
import time
import math
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB

def parse_time_str(t):
    h, m = t.split(':')
    return int(h) * 60 + int(m)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    start_time = time.time()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        instance = json.load(f)

    params = instance['parameters']
    disruption = instance['disruption']
    rescheduling_time = disruption['rescheduling_time_minutes']

    stations = {s['id']: s for s in instance['network']['stations']}
    lines = {l['line_id']: l for l in instance['train_lines']}
    all_tasks = {t['task_id']: t for t in instance['tasks']}

    # Taxi connections
    taxi_conns = defaultdict(dict)
    for tc in instance['network']['taxi_connections']:
        taxi_conns[tc['from_station']][tc['to_station']] = tc['travel_time_minutes']
        if tc.get('bidirectional', True):
            taxi_conns[tc['to_station']][tc['from_station']] = tc['travel_time_minutes']

    # Precompute train travel between station pairs
    station_lines = defaultdict(list)  # (s1, s2) -> [(line, travel_time, freq)]
    for line in instance['train_lines']:
        sts = line['stations']
        seg_times = line['segment_travel_times']
        freq = line['frequency_per_hour']
        n = len(sts)
        for i in range(n):
            for j in range(n):
                if i != j:
                    if i < j:
                        tt = sum(seg_times[i:j])
                    else:
                        tt = sum(seg_times[j:i])
                    station_lines[(sts[i], sts[j])].append((line, tt, freq))

    blocked_from = disruption['blocked_segment']['from_station']
    blocked_to = disruption['blocked_segment']['to_station']
    disruption_start = disruption['start_time_minutes']
    disruption_end = disruption['end_time_minutes']

    def is_segment_blocked(s1, s2):
        return (s1 == blocked_from and s2 == blocked_to) or (s1 == blocked_to and s2 == blocked_from)

    def is_route_blocked_on_line(line, s1, s2, depart_time):
        sts = line['stations']
        seg_times = line['segment_travel_times']
        idx1 = sts.index(s1) if s1 in sts else -1
        idx2 = sts.index(s2) if s2 in sts else -1
        if idx1 == -1 or idx2 == -1:
            return False
        if idx1 < idx2:
            cum = 0
            for i in range(idx1, idx2):
                if is_segment_blocked(sts[i], sts[i+1]):
                    t_seg = depart_time + cum
                    if disruption_start <= t_seg <= disruption_end:
                        return True
                cum += seg_times[i]
        else:
            cum = 0
            for i in range(idx1-1, idx2-1, -1):
                if is_segment_blocked(sts[i], sts[i+1]):
                    t_seg = depart_time + cum
                    if disruption_start <= t_seg <= disruption_end:
                        return True
                cum += seg_times[i]
        return False

    def compute_train_arrival(from_st, to_st, depart_time):
        if from_st == to_st:
            return float(depart_time)
        best = None
        for (line, tt, freq) in station_lines.get((from_st, to_st), []):
            headway = 60.0 / freq if freq > 0 else 60.0
            n = int(depart_time / headway)
            if n * headway < depart_time:
                n += 1
            dep = n * headway
            if is_route_blocked_on_line(line, from_st, to_st, dep):
                continue
            arrival = dep + tt
            if best is None or arrival < best:
                best = arrival
        return best

    def get_driver_state(duty):
        if duty['duty_type'] == 'reserve':
            return duty['standby_station'], max(duty['start_time_minutes'], rescheduling_time)
        current_station = duty['crew_base']
        available_time = max(duty['start_time_minutes'], rescheduling_time)
        td_lookup = {td['task_id']: td for td in duty.get('task_details', [])}
        for task_id in duty.get('pre_disruption_tasks', []):
            arr_station = None
            dep_t = arr_t = None
            if task_id in td_lookup:
                td = td_lookup[task_id]
                arr_station = td['arrival_station']
                dep_t = parse_time_str(td['departure_time'])
                arr_t = parse_time_str(td['arrival_time'])
            elif task_id in all_tasks:
                t = all_tasks[task_id]
                arr_station = t['arrival_station']
                dep_t = t['departure_time_minutes']
                arr_t = t['arrival_time_minutes']
            else:
                continue
            if dep_t is not None and arr_t is not None:
                if dep_t <= rescheduling_time <= arr_t:
                    current_station = arr_station
                    available_time = arr_t
                elif arr_t <= rescheduling_time:
                    current_station = arr_station
                    available_time = max(arr_t, rescheduling_time)
        return current_station, available_time

    def is_task_feasible(task_id, duty):
        task = all_tasks[task_id]
        line = lines[task['line_id']]
        if task['rolling_stock_type'] not in duty['rolling_stock_licenses']:
            return False
        sts = line['stations']
        dep_idx = sts.index(task['departure_station'])
        arr_idx = sts.index(task['arrival_station'])
        if dep_idx < arr_idx:
            route = sts[dep_idx:arr_idx+1]
        else:
            route = list(reversed(sts[arr_idx:dep_idx+1]))
        rk = set(duty['route_knowledge'])
        for s in route:
            if s not in rk:
                return False
        return True

    def check_connection(cur_station, cur_time, last_rst, task_id):
        task = all_tasks[task_id]
        dep_time = task['departure_time_minutes']
        rst = task['rolling_stock_type']
        min_conn = params['min_connection_time_minutes'] if last_rst is not None and last_rst != rst else 0
        if cur_station == task['departure_station']:
            if dep_time >= cur_time + min_conn:
                return (True, False)
            return None
        train_arr = compute_train_arrival(cur_station, task['departure_station'], cur_time)
        if train_arr is not None:
            if dep_time >= int(train_arr) + min_conn:
                return (True, False)
        if cur_station in taxi_conns and task['departure_station'] in taxi_conns[cur_station]:
            tt = taxi_conns[cur_station][task['departure_station']]
            ta = cur_time + tt
            if dep_time >= ta + min_conn:
                return (True, True)
        return None

    def check_return(cur_station, cur_time, duty):
        base = duty['crew_base']
        horizon = duty['end_time_minutes'] + params['max_overtime_minutes']
        max_dur = params.get('max_duty_duration_hours', 24) * 60
        max_ret = min(horizon, duty['start_time_minutes'] + max_dur)
        if cur_station == base:
            return (base, min(cur_time, max_ret), False, False)
        train_arr = compute_train_arrival(cur_station, base, cur_time)
        if train_arr is not None and int(train_arr) <= max_ret:
            return (base, int(train_arr), False, False)
        if cur_station in taxi_conns and base in taxi_conns[cur_station]:
            tt = taxi_conns[cur_station][base]
            ta = cur_time + tt
            if ta <= max_ret:
                return (base, ta, True, False)
        if stations.get(cur_station, {}).get('is_relief_point', False):
            return (cur_station, cur_time, False, True)
        return (cur_station, cur_time, False, True)

    # Determine unfinished tasks
    disruption_canceled = set(disruption.get('canceled_tasks', []))
    unfinished_tasks = []
    for tid, task in all_tasks.items():
        if task['departure_time_minutes'] >= rescheduling_time and tid not in disruption_canceled:
            unfinished_tasks.append(tid)

    all_duties = instance['active_duties'] + instance['reserve_duties']
    duty_completions = {}

    enum_time_limit = args.time_limit * 0.4

    for duty in all_duties:
        duty_id = duty['duty_id']
        cur_st, cur_t = get_driver_state(duty)
        feasible_tasks = sorted(
            [tid for tid in unfinished_tasks if is_task_feasible(tid, duty)],
            key=lambda tid: all_tasks[tid]['departure_time_minutes']
        )
        original_tasks = list(duty.get('tasks', []))
        original_task_set = set(original_tasks)
        original_pairs = set()
        for i in range(len(original_tasks) - 1):
            original_pairs.add((original_tasks[i], original_tasks[i+1]))

        completions = []

        def compute_cost(tasks_so_far, taxis, uses_taxi_ret):
            if tasks_so_far == original_tasks:
                return 0
            cost = params['cost_duty_changed']
            new_tasks = sum(1 for t in tasks_so_far if t not in original_task_set)
            cost += new_tasks * params['cost_task_reassigned']
            new_pairs = 0
            for i in range(len(tasks_so_far) - 1):
                if (tasks_so_far[i], tasks_so_far[i+1]) not in original_pairs:
                    new_pairs += 1
            cost += new_pairs * params['cost_new_transfer']
            total_taxis = taxis + (1 if uses_taxi_ret else 0)
            if total_taxis > 0:
                cost += total_taxis * params['cost_taxi']
            return cost

        # Empty completion
        ret = check_return(cur_st, cur_t, duty)
        if ret:
            rs, rt, ut, rel = ret
            c = 0 if not original_tasks else params['cost_duty_changed']
            if ut:
                c += params['cost_taxi']
            completions.append({
                'tasks': [], 'cost': c, 'uses_taxi': ut,
                'release_at_relief_point': rel, 'return_station': rs,
                'return_time_minutes': rt
            })

        # Original completion
        if original_tasks:
            feas = True
            cs, ct, lr = cur_st, cur_t, None
            tx = 0
            for tid in original_tasks:
                if tid not in all_tasks or not is_task_feasible(tid, duty):
                    feas = False
                    break
                conn = check_connection(cs, ct, lr, tid)
                if conn is None:
                    feas = False
                    break
                if conn[1]:
                    tx += 1
                task = all_tasks[tid]
                cs = task['arrival_station']
                ct = task['arrival_time_minutes']
                lr = task['rolling_stock_type']
            if feas:
                ret = check_return(cs, ct, duty)
                if ret:
                    rs, rt, ut, rel = ret
                    completions.append({
                        'tasks': original_tasks.copy(), 'cost': 0,
                        'uses_taxi': tx > 0 or ut,
                        'release_at_relief_point': rel,
                        'return_station': rs, 'return_time_minutes': rt
                    })

        # DFS enumeration
        max_completions = min(200, max(20, 500 // max(1, len(all_duties))))
        max_depth = min(8, len(feasible_tasks))

        def dfs(cs, ct, lr, tasks_so_far, taxis, depth):
            if len(completions) >= max_completions:
                return
            if depth >= max_depth:
                return
            if time.time() - start_time > enum_time_limit:
                return
            if tasks_so_far:
                ret = check_return(cs, ct, duty)
                if ret:
                    rs, rt, ut, rel = ret
                    completions.append({
                        'tasks': tasks_so_far.copy(),
                        'cost': compute_cost(tasks_so_far, taxis, ut),
                        'uses_taxi': taxis > 0 or ut,
                        'release_at_relief_point': rel,
                        'return_station': rs, 'return_time_minutes': rt
                    })
            task_set = set(tasks_so_far)
            for tid in feasible_tasks:
                if tid in task_set:
                    continue
                task = all_tasks[tid]
                if task['departure_time_minutes'] < ct:
                    continue
                conn = check_connection(cs, ct, lr, tid)
                if conn is None:
                    continue
                dfs(task['arrival_station'], task['arrival_time_minutes'],
                    task['rolling_stock_type'], tasks_so_far + [tid],
                    taxis + (1 if conn[1] else 0), depth + 1)

        dfs(cur_st, cur_t, None, [], 0, 0)

        # Deduplicate
        seen = set()
        unique = []
        for comp in completions:
            key = tuple(comp['tasks'])
            if key not in seen:
                seen.add(key)
                unique.append(comp)
        duty_completions[duty_id] = unique if unique else completions

    # Build MIP
    remaining = max(1, args.time_limit - (time.time() - start_time) - 2)

    model = gp.Model('crew_rescheduling')
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', remaining)
    model.setParam('OutputFlag', 0)

    x = {}
    for d_id, comps in duty_completions.items():
        for c_idx in range(len(comps)):
            x[d_id, c_idx] = model.addVar(vtype=GRB.BINARY, name=f'x_{d_id}_{c_idx}')

    y = {}
    for t_id in unfinished_tasks:
        cc = params['cost_cancel_A_to_B'] if all_tasks[t_id]['task_type'] == 'A_to_B' else params['cost_cancel_A_to_A']
        y[t_id] = model.addVar(vtype=GRB.BINARY, name=f'y_{t_id}')

    model.update()

    for d_id in duty_completions:
        model.addConstr(gp.quicksum(x[d_id, c_idx] for c_idx in range(len(duty_completions[d_id]))) == 1)

    for t_id in unfinished_tasks:
        covering = []
        for d_id, comps in duty_completions.items():
            for c_idx, comp in enumerate(comps):
                if t_id in comp['tasks']:
                    covering.append(x[d_id, c_idx])
        model.addConstr(gp.quicksum(covering) + y[t_id] >= 1)

    obj = gp.LinExpr()
    for d_id, comps in duty_completions.items():
        for c_idx, comp in enumerate(comps):
            obj += comp['cost'] * x[d_id, c_idx]
    for t_id in unfinished_tasks:
        cc = params['cost_cancel_A_to_B'] if all_tasks[t_id]['task_type'] == 'A_to_B' else params['cost_cancel_A_to_A']
        obj += cc * y[t_id]
    model.setObjective(obj, GRB.MINIMIZE)

    model.optimize()

    # Construct solution
    def construct_fallback():
        total = 0
        canceled = list(unfinished_tasks)
        da = {}
        for duty in all_duties:
            d_id = duty['duty_id']
            cs, ct = get_driver_state(duty)
            ret = check_return(cs, ct, duty)
            if ret:
                rs, rt, ut, rel = ret
            else:
                rs, rt, ut, rel = cs, ct, False, True
            orig = duty.get('tasks', [])
            c = 0 if not orig else params['cost_duty_changed']
            if ut:
                c += params['cost_taxi']
            total += c
            da[d_id] = {
                'completion_index': 0,
                'cost': float(c),
                'covered_tasks': [],
                'uses_taxi': ut,
                'release_at_relief_point': rel,
                'return_station': rs,
                'return_time_minutes': rt
            }
        for t_id in canceled:
            cc = params['cost_cancel_A_to_B'] if all_tasks[t_id]['task_type'] == 'A_to_B' else params['cost_cancel_A_to_A']
            total += cc
        return {
            'released_problem_variant': 'relief_point_completion_v1',
            'objective_value': float(total),
            'canceled_tasks': canceled,
            'duty_assignments': da
        }

    if model.SolCount > 0:
        selected = {}
        for d_id in duty_completions:
            for c_idx in range(len(duty_completions[d_id])):
                if x[d_id, c_idx].X > 0.5:
                    selected[d_id] = c_idx
                    break
            if d_id not in selected:
                selected[d_id] = 0

        canceled = []
        for t_id in unfinished_tasks:
            if y[t_id].X > 0.5:
                canceled.append(t_id)

        total_cost = 0
        duty_assignments = {}
        for duty in all_duties:
            d_id = duty['duty_id']
            c_idx = selected.get(d_id, 0)
            comp = duty_completions[d_id][c_idx]
            total_cost += comp['cost']
            duty_assignments[d_id] = {
                'completion_index': c_idx,
                'cost': float(comp['cost']),
                'covered_tasks': comp['tasks'],
                'uses_taxi': comp['uses_taxi'],
                'release_at_relief_point': comp['release_at_relief_point'],
                'return_station': comp['return_station'],
                'return_time_minutes': comp['return_time_minutes']
            }

        for t_id in canceled:
            cc = params['cost_cancel_A_to_B'] if all_tasks[t_id]['task_type'] == 'A_to_B' else params['cost_cancel_A_to_A']
            total_cost += cc

        solution = {
            'released_problem_variant': 'relief_point_completion_v1',
            'objective_value': float(total_cost),
            'canceled_tasks': canceled,
            'duty_assignments': duty_assignments
        }

        if logger:
            logger.log_solution(total_cost, solution)
    else:
        solution = construct_fallback()
        if logger:
            logger.log_solution(solution['objective_value'], solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()