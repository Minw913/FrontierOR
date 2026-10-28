import sys
import json
import argparse
import time
from bisect import bisect_left
import gurobipy as gp
from gurobipy import GRB

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    depots = data['depots']
    trips = data['trips']
    travel_times = data['travel_times']
    fc = data['vehicle_parameters']['fictitious_cost_per_minute_empty_outside_depot']
    crew_params = data['crew_parameters']
    duty_types = data['duty_types']

    # 1. Vehicle Scheduling Problem (VSP)
    vsp = gp.Model("VSP")
    vsp.setParam('OutputFlag', 0)
    vsp.setParam('Threads', 1)
    
    # Time limits allocation (allow 20% for VSP if total time is large)
    vlimit = max(5.0, args.time_limit * 0.2)
    vsp.setParam('TimeLimit', vlimit)

    y, z, x_short, x_long = {}, {}, {}, {}

    for d in depots:
        d_name = d['name']
        for i in trips:
            y[d_name, i['trip_id']] = vsp.addVar(vtype=GRB.BINARY, obj=100000.0)
            z[d_name, i['trip_id']] = vsp.addVar(vtype=GRB.BINARY, obj=0.0)

    for d in depots:
        d_name = d['name']
        for i in trips:
            i_id = i['trip_id']
            i_end = i['end_time']
            i_loc = i['end_location']
            for j in trips:
                j_id = j['trip_id']
                if i_id >= j_id:
                    continue
                j_start = j['start_time']
                j_loc = j['start_location']
                if i_end <= j_start:
                    tt_short = travel_times[i_loc][j_loc]
                    if i_end + tt_short <= j_start:
                        cost_short = (j_start - i_end) * fc
                        x_short[d_name, i_id, j_id] = vsp.addVar(vtype=GRB.BINARY, obj=cost_short)
                    
                    tt_to_d = travel_times[i_loc][d_name]
                    tt_from_d = travel_times[d_name][j_loc]
                    if i_end + tt_to_d + tt_from_d <= j_start:
                        x_long[d_name, i_id, j_id] = vsp.addVar(vtype=GRB.BINARY, obj=0.0)

    vsp.modelSense = GRB.MINIMIZE

    for i in trips:
        i_id = i['trip_id']
        vsp.addConstr(
            gp.quicksum(y[d['name'], i_id] for d in depots) +
            gp.quicksum(x_short[d['name'], j['trip_id'], i_id] for d in depots for j in trips if (d['name'], j['trip_id'], i_id) in x_short) +
            gp.quicksum(x_long[d['name'], j['trip_id'], i_id] for d in depots for j in trips if (d['name'], j['trip_id'], i_id) in x_long) == 1
        )
        vsp.addConstr(
            gp.quicksum(z[d['name'], i_id] for d in depots) +
            gp.quicksum(x_short[d['name'], i_id, j['trip_id']] for d in depots for j in trips if (d['name'], i_id, j['trip_id']) in x_short) +
            gp.quicksum(x_long[d['name'], i_id, j['trip_id']] for d in depots for j in trips if (d['name'], i_id, j['trip_id']) in x_long) == 1
        )

    for d in depots:
        d_name = d['name']
        for i in trips:
            i_id = i['trip_id']
            in_flow = y[d_name, i_id] + \
                      gp.quicksum(x_short[d_name, j['trip_id'], i_id] for j in trips if (d_name, j['trip_id'], i_id) in x_short) + \
                      gp.quicksum(x_long[d_name, j['trip_id'], i_id] for j in trips if (d_name, j['trip_id'], i_id) in x_long)
            out_flow = z[d_name, i_id] + \
                       gp.quicksum(x_short[d_name, i_id, j['trip_id']] for j in trips if (d_name, i_id, j['trip_id']) in x_short) + \
                       gp.quicksum(x_long[d_name, i_id, j['trip_id']] for j in trips if (d_name, i_id, j['trip_id']) in x_long)
            vsp.addConstr(in_flow == out_flow)

    vsp.optimize()

    # Extract VSP Solution
    vehicle_schedule = {d['name']: [] for d in depots}
    blocks = []
    vblock_id_counter = 0

    if vsp.SolCount > 0:
        for d in depots:
            d_name = d['name']
            for i in trips:
                if y[d_name, i['trip_id']].X > 0.5:
                    curr = i['trip_id']
                    path = [curr]
                    edges = []
                    while True:
                        found = False
                        for j in trips:
                            j_id = j['trip_id']
                            if (d_name, curr, j_id) in x_short and x_short[d_name, curr, j_id].X > 0.5:
                                path.append(j_id)
                                edges.append('short')
                                curr = j_id
                                found = True
                                break
                            if (d_name, curr, j_id) in x_long and x_long[d_name, curr, j_id].X > 0.5:
                                path.append(j_id)
                                edges.append('long')
                                curr = j_id
                                found = True
                                break
                        if not found:
                            break
                    vehicle_schedule[d_name].append(path)
                    blocks.append((vblock_id_counter, d_name, path, edges))
                    vblock_id_counter += 1

    # 2. Crew Task Generation
    trip_dict = {t['trip_id']: t for t in trips}
    total_tasks = 0
    global_tasks = []
    pieces = []

    valid_ROs_base = set(data.get('relief_locations', []))
    max_piece_len = max(dt.get('piece_length_max', 999999) for dt in duty_types.values())

    for vblock_id, d_name, path, edges in blocks:
        sub_blocks = []
        curr_sub = [path[0]]
        for idx in range(len(edges)):
            if edges[idx] == 'long':
                sub_blocks.append(curr_sub)
                curr_sub = [path[idx+1]]
            else:
                curr_sub.append(path[idx+1])
        sub_blocks.append(curr_sub)
        
        valid_ROs = valid_ROs_base.copy()
        valid_ROs.add(d_name)

        for sub in sub_blocks:
            P = []
            t_first = trip_dict[sub[0]]
            dep_tt = travel_times[d_name][t_first['start_location']]
            P.append((t_first['start_time'] - dep_tt, d_name))
            
            for t_id in sub:
                T = trip_dict[t_id]
                P.append((T['start_time'], T['start_location']))
                P.append((T['end_time'], T['end_location']))
            
            t_last = trip_dict[sub[-1]]
            ret_tt = travel_times[t_last['end_location']][d_name]
            P.append((t_last['end_time'] + ret_tt, d_name))
            
            start_task_idx = total_tasks
            for k in range(len(P)-1):
                global_tasks.append({
                    'id': total_tasks,
                    'start_time': P[k][0],
                    'end_time': P[k+1][0],
                    'start_loc': P[k][1],
                    'end_loc': P[k+1][1],
                    'is_trip': (k % 2 == 1),
                    'trip_id': sub[(k-1)//2] if k % 2 == 1 else None
                })
                total_tasks += 1
                
            for i in range(len(P)):
                if P[i][1] in valid_ROs:
                    for j in range(i+1, len(P)):
                        if P[j][1] in valid_ROs:
                            dur = P[j][0] - P[i][0]
                            if dur <= max_piece_len:
                                pieces.append({
                                    'vblock_id': vblock_id,
                                    'depot': d_name,
                                    'start_time': P[i][0],
                                    'end_time': P[j][0],
                                    'start_loc': P[i][1],
                                    'end_loc': P[j][1],
                                    'duration': dur,
                                    'tasks': list(range(start_task_idx + i, start_task_idx + j))
                                })

    pieces_by_depot = {}
    for p in pieces:
        pieces_by_depot.setdefault(p['depot'], []).append(p)

    def get_sign_on(loc, d_n):
        if loc == d_n: return crew_params['sign_on_time_depot_minutes']
        ex = crew_params['extra_time_non_depot_relief_minutes']
        if crew_params['extra_time_includes_deadhead_to_depot']: return ex
        return ex + travel_times[loc][d_n]

    def get_sign_off(loc, d_n):
        if loc == d_n: return crew_params['sign_off_time_depot_minutes']
        ex = crew_params['extra_time_non_depot_relief_minutes']
        if crew_params['extra_time_includes_deadhead_to_depot']: return ex
        return ex + travel_times[loc][d_n]

    duties = []
    change_allowed = crew_params['changeovers_allowed']

    for dt_name, dt in duty_types.items():
        n_pieces = dt['num_pieces']
        p_min = dt['piece_length_min']
        p_max = dt['piece_length_max']
        b_min = dt['break_length_min'] or 0
        d_max = dt['duty_length_max'] or 999999
        w_max = dt['work_time_max'] or 999999
        s_min = dt['start_time_min'] if dt['start_time_min'] is not None else -999999
        s_max = dt['start_time_max'] if dt['start_time_max'] is not None else 999999
        e_max = dt['end_time_max'] if dt['end_time_max'] is not None else 999999
        
        for d in depots:
            d_name = d['name']
            d_pieces = pieces_by_depot.get(d_name, [])
            valid_p = [p for p in d_pieces if p_min <= p['duration'] <= p_max]
            
            if n_pieces == 1:
                for p in valid_p:
                    so = get_sign_on(p['start_loc'], d_name)
                    sf = get_sign_off(p['end_loc'], d_name)
                    dlen = so + p['duration'] + sf
                    if dlen > d_max or p['duration'] > w_max: continue
                    st = p['start_time'] - so
                    et = p['end_time'] + sf
                    if st < s_min or st > s_max or et > e_max: continue
                    
                    duties.append({
                        'type': dt_name, 'depot': d_name, 'pieces': [p], 'tasks': p['tasks']
                    })
            elif n_pieces == 2:
                valid_p.sort(key=lambda x: x['start_time'])
                st_values = [p['start_time'] for p in valid_p]
                
                for p1 in valid_p:
                    min_st = p1['end_time'] + b_min
                    idx = bisect_left(st_values, min_st)
                    for i2 in range(idx, len(valid_p)):
                        p2 = valid_p[i2]
                        if not change_allowed and p1['vblock_id'] != p2['vblock_id']: continue
                        if set(p1['tasks']).intersection(p2['tasks']): continue
                        
                        so = get_sign_on(p1['start_loc'], d_name)
                        sf = get_sign_off(p2['end_loc'], d_name)
                        break_len = p2['start_time'] - p1['end_time']
                        dlen = so + p1['duration'] + break_len + p2['duration'] + sf
                        if dlen > d_max: continue
                        
                        wt = p1['duration'] + p2['duration']
                        if wt > w_max: continue
                        
                        st = p1['start_time'] - so
                        et = p2['end_time'] + sf
                        if st < s_min or st > s_max or et > e_max: continue
                        
                        duties.append({
                            'type': dt_name, 'depot': d_name, 'pieces': [p1, p2], 'tasks': p1['tasks'] + p2['tasks']
                        })

    # 3. Crew Scheduling Problem (CSP)
    rem_time = max(5.0, args.time_limit - (time.time() - start_time) - 1.0)
    csp = gp.Model("CSP")
    csp.setParam('TimeLimit', rem_time)
    csp.setParam('OutputFlag', 0)
    csp.setParam('Threads', 1)
    csp.setParam('MIPGap', 1e-4)

    num_duties = len(duties)
    u = csp.addVars(num_duties, vtype=GRB.BINARY)
    slacks = csp.addVars(total_tasks, vtype=GRB.INTEGER, lb=0)

    task_to_du = [[] for _ in range(total_tasks)]
    for j, duty in enumerate(duties):
        for t in duty['tasks']:
            task_to_du[t].append(j)

    for t in range(total_tasks):
        csp.addConstr(gp.quicksum(u[j] for j in task_to_du[t]) + slacks[t] == 1)

    csp.setObjective(gp.quicksum(u[j] for j in range(num_duties)) + 10000.0 * gp.quicksum(slacks[t] for t in range(total_tasks)), GRB.MINIMIZE)

    csp.optimize()

    # Form Solution Output
    crew_output = []
    uncovered_trips = set([t['trip_id'] for t in trips])
    
    if csp.SolCount > 0:
        for j in range(num_duties):
            if u[j].X > 0.5:
                duty = duties[j]
                duty_trips = []
                for t in duty['tasks']:
                    gt = global_tasks[t]
                    if gt['is_trip']:
                        duty_trips.append(gt['trip_id'])
                        if gt['trip_id'] in uncovered_trips:
                            uncovered_trips.remove(gt['trip_id'])
                if duty_trips:
                    crew_output.append({
                        "depot": duty['depot'],
                        "type": duty['type'],
                        "trips": duty_trips
                    })

    # Fallback to output dummy duties for any strictly missing trips for schema
    dt_name = list(duty_types.keys())[0]
    for tr in list(uncovered_trips):
        td = depots[0]['name']
        for b in blocks:
            if tr in b[2]:
                td = b[1]
                break
        crew_output.append({
            "depot": td,
            "type": dt_name,
            "trips": [tr]
        })

    obj_val = len(blocks) + len(crew_output)
    if logger:
        logger.log(obj_val)

    solution = {
        "objective_value": obj_val,
        "vehicle_schedule": vehicle_schedule,
        "crew_schedule": crew_output
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    solve()