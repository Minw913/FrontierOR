import argparse
import json
import collections
import bisect
import time
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def time_to_mins(t_str):
    h, m = map(int, t_str.split(':'))
    return h * 60 + m

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if (SolutionLogger and args.log_path) else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    # Extract Network
    stations_data = data['network']['stations']
    stations = [s['id'] for s in stations_data]
    is_relief = {s['id']: s['is_relief_point'] for s in stations_data}
    is_base = {s['id']: s['is_crew_base'] for s in stations_data}
    
    taxi_conns = data['network'].get('taxi_connections', [])
    taxi_time = {s1: {s2: float('inf') for s2 in stations} for s1 in stations}
    for s in stations:
        taxi_time[s][s] = 0
    for tc in taxi_conns:
        s1, s2, t = tc['from_station'], tc['to_station'], tc['travel_time_minutes']
        taxi_time[s1][s2] = min(taxi_time[s1][s2], t)
        if tc['bidirectional']:
            taxi_time[s2][s1] = min(taxi_time[s2][s1], t)

    for k in stations:
        for i in stations:
            for j in stations:
                if taxi_time[i][k] + taxi_time[k][j] < taxi_time[i][j]:
                    taxi_time[i][j] = taxi_time[i][k] + taxi_time[k][j]

    train_lines = {l['line_id']: l for l in data.get('train_lines', [])}
    
    tasks = data['tasks']
    tasks_dict = {t['task_id']: t for t in tasks}
    
    reschedule_time = data['disruption']['rescheduling_time_minutes']
    hard_canceled = set(data['disruption'].get('canceled_tasks', []))
    
    params = data['parameters']
    cost_cancel_A2B = params['cost_cancel_A_to_B']
    cost_cancel_A2A = params['cost_cancel_A_to_A']
    cost_duty_change = params['cost_duty_changed']
    cost_reassign = params['cost_task_reassigned']
    cost_new_trans = params['cost_new_transfer']
    cost_taxi = params['cost_taxi']
    min_conn = params['min_connection_time_minutes']
    max_dur = params['max_duty_duration_hours'] * 60

    all_duties = []
    for d in data.get('active_duties', []):
        if d.get('pre_disruption_tasks'):
            last_id = d['pre_disruption_tasks'][-1]
            last_det = next(t for t in d['task_details'] if t['task_id'] == last_id)
            arr_m = time_to_mins(last_det['arrival_time'])
            src_s = last_det['arrival_station']
            src_t = max(reschedule_time, arr_m)
        else:
            src_s = d['crew_base']
            src_t = max(reschedule_time, d['start_time_minutes'])
        all_duties.append((d, src_s, src_t))
        
    for d in data.get('reserve_duties', []):
        src_s = d['standby_station']
        src_t = max(reschedule_time, d['start_time_minutes'])
        all_duties.append((d, src_s, src_t))

    times = collections.defaultdict(set)
    for t in tasks:
        times[t['departure_station']].add(t['departure_time_minutes'])
        times[t['arrival_station']].add(t['arrival_time_minutes'])
        
    for d, src_s, src_t in all_duties:
        times[src_s].add(src_t)
        horizon = d['start_time_minutes'] + max_dur
        for s in stations:
            times[s].add(horizon)

    sorted_times = {s: sorted(list(times[s])) for s in stations}
    graph = {}
    for s in stations:
        st_list = sorted_times[s]
        for i, t in enumerate(st_list):
            graph[(s, t)] = []

    for s in stations:
        st_list = sorted_times[s]
        for i in range(len(st_list) - 1):
            graph[(s, st_list[i])].append(((s, st_list[i+1]), 0))

    for t in tasks:
        u = (t['departure_station'], t['departure_time_minutes'])
        v = (t['arrival_station'], t['arrival_time_minutes'])
        graph[u].append((v, 0))

    for s1 in stations:
        for t in sorted_times[s1]:
            u = (s1, t)
            for s2 in stations:
                if s1 == s2: continue
                tt = taxi_time[s1][s2]
                if tt < float('inf'):
                    arr_t = t + tt
                    j = bisect.bisect_left(sorted_times[s2], arr_t)
                    if j < len(sorted_times[s2]):
                        v = (s2, sorted_times[s2][j])
                        graph[u].append((v, 1))

    def bfs(start_node):
        dist = {start_node: 0}
        dq = collections.deque([start_node])
        while dq:
            u = dq.popleft()
            d_u = dist[u]
            for v, w in graph[u]:
                new_d = d_u + w
                if new_d < dist.get(v, float('inf')):
                    dist[v] = new_d
                    if w == 0:
                        dq.appendleft(v)
                    else:
                        dq.append(v)
        return dist

    start_nodes = set()
    for t in tasks:
        start_nodes.add((t['arrival_station'], t['arrival_time_minutes']))
    for d, src_s, src_t in all_duties:
        start_nodes.add((src_s, src_t))

    dist_maps = {}
    for sn in start_nodes:
        dist_maps[sn] = bfs(sn)

    def EAT_query(s_start, t_start, s_end, t_end):
        sn = (s_start, t_start)
        if sn not in dist_maps: return float('inf')
        d_map = dist_maps[sn]
        st_list = sorted_times[s_end]
        j = bisect.bisect_right(st_list, t_end) - 1
        if j < 0: return float('inf')
        return d_map.get((s_end, st_list[j]), float('inf'))

    duty_data = {}
    for d, src_s, src_t in all_duties:
        did = d['duty_id']
        rk = set(d['route_knowledge'])
        rsl = set(d['rolling_stock_licenses'])
        horizon = d['start_time_minutes'] + max_dur
        cb = d['crew_base']
        
        V_d = []
        for t in tasks:
            if t['task_id'] in hard_canceled: continue
            if t['rolling_stock_type'] not in rsl: continue
            l_info = train_lines.get(t['line_id'])
            t_sts = []
            if l_info:
                l_sts = l_info['stations']
                d_idx = l_sts.index(t['departure_station'])
                a_idx = l_sts.index(t['arrival_station'])
                if t['direction'] == 'forward':
                    t_sts = l_sts[d_idx : a_idx+1]
                else:
                    t_sts = l_sts[a_idx : d_idx+1]
            if all(s in rk for s in t_sts):
                V_d.append(t)
                
        E_d = []
        for t in V_d:
            dist = EAT_query(src_s, src_t, t['departure_station'], t['departure_time_minutes'])
            if dist < float('inf'):
                E_d.append(('src', t['task_id'], min(dist, 1)))
            
            dist_sink = EAT_query(t['arrival_station'], t['arrival_time_minutes'], cb, horizon)
            if dist_sink < float('inf'):
                E_d.append((t['task_id'], 'sink', min(dist_sink, 1)))
            elif is_relief.get(t['arrival_station']) and t['arrival_time_minutes'] <= horizon:
                E_d.append((t['task_id'], 'sink', 0))
                
        for t1 in V_d:
            for t2 in V_d:
                if t1['task_id'] == t2['task_id']: continue
                if t1['rolling_stock_type'] != t2['rolling_stock_type']:
                    if t2['departure_time_minutes'] < t1['arrival_time_minutes'] + min_conn: continue
                dist = EAT_query(t1['arrival_station'], t1['arrival_time_minutes'], t2['departure_station'], t2['departure_time_minutes'])
                if dist < float('inf'):
                    E_d.append((t1['task_id'], t2['task_id'], min(dist, 1)))
                    
        dist_s = EAT_query(src_s, src_t, cb, horizon)
        if dist_s < float('inf'):
            E_d.append(('src', 'sink', min(dist_s, 1)))
        elif is_relief.get(src_s) and src_t <= horizon:
            E_d.append(('src', 'sink', 0))
            
        orig_tasks = d['tasks']
        z_valid = True
        v_ids = set(t['task_id'] for t in V_d)
        edges_set = set((u, v) for u, v, c in E_d)
        
        for ot in orig_tasks:
            if ot not in v_ids:
                z_valid = False; break
        if z_valid:
            curr = 'src'
            for ot in orig_tasks:
                if (curr, ot) not in edges_set: z_valid = False; break
                curr = ot
            if (curr, 'sink') not in edges_set: z_valid = False
            
        duty_data[did] = {
            'src_s': src_s, 'src_t': src_t, 'cb': cb, 'horizon': horizon,
            'V_d': V_d, 'E_d': E_d, 'orig_tasks': orig_tasks, 'z_valid': z_valid
        }

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Seed", 0)
    env.setParam("MIPGap", 1e-4)
    env.setParam("NumericFocus", 0)
    env.setParam("Threads", 1)
    env.setParam("TimeLimit", args.time_limit)
    env.start()
    
    m = gp.Model("crew_rescheduling", env=env)
    
    Z = {}
    Y = {}
    X = {}
    Taxi = {}
    
    for did, dd in duty_data.items():
        if dd['z_valid']:
            Z[did] = m.addVar(vtype=GRB.BINARY, name=f"Z_{did}")
        else:
            Z[did] = m.addVar(lb=0, ub=0, vtype=GRB.BINARY, name=f"Z_{did}")
            
        Taxi[did] = m.addVar(vtype=GRB.BINARY, name=f"Taxi_{did}")
        for t in dd['V_d']:
            Y[did, t['task_id']] = m.addVar(vtype=GRB.BINARY, name=f"Y_{did}_{t['task_id']}")
        for e in dd['E_d']:
            X[did, e[0], e[1]] = m.addVar(vtype=GRB.BINARY, name=f"X_{did}_{e[0]}_{e[1]}")
            
    Cancel = {}
    for t in tasks:
        tid = t['task_id']
        if tid not in hard_canceled:
            Cancel[tid] = m.addVar(vtype=GRB.BINARY, name=f"Cancel_{tid}")
            
    m.update()
    
    obj = gp.LinExpr()
    for did, dd in duty_data.items():
        obj += cost_duty_change * (1 - Z[did])
        obj += cost_taxi * Taxi[did]
        orig_set = set(dd['orig_tasks'])
        for t in dd['V_d']:
            if t['task_id'] not in orig_set:
                obj += cost_reassign * Y[did, t['task_id']]
                
        orig_adj = set(zip(dd['orig_tasks'][:-1], dd['orig_tasks'][1:]))
        for u, v, c in dd['E_d']:
            if u != 'src' and v != 'sink':
                if (u, v) not in orig_adj:
                    obj += cost_new_trans * X[did, u, v]
                    
    for t in tasks:
        tid = t['task_id']
        if tid not in hard_canceled:
            cc = cost_cancel_A2B if t['task_type'] == 'A_to_B' else cost_cancel_A2A
            obj += cc * Cancel[tid]
            
    m.setObjective(obj, GRB.MINIMIZE)
    
    for did, dd in duty_data.items():
        for t in dd['V_d']:
            tid = t['task_id']
            in_edges = [X[did, u, v] for u, v, c in dd['E_d'] if v == tid]
            out_edges = [X[did, u, v] for u, v, c in dd['E_d'] if u == tid]
            m.addConstr(gp.quicksum(in_edges) == Y[did, tid])
            m.addConstr(gp.quicksum(out_edges) == Y[did, tid])
            
        src_out = [X[did, u, v] for u, v, c in dd['E_d'] if u == 'src']
        sink_in = [X[did, u, v] for u, v, c in dd['E_d'] if v == 'sink']
        m.addConstr(gp.quicksum(src_out) == 1 - Z[did])
        m.addConstr(gp.quicksum(sink_in) == 1 - Z[did])
        
        for u, v, c in dd['E_d']:
            if c > 0:
                m.addConstr(X[did, u, v] <= Taxi[did])
                
    for t in tasks:
        tid = t['task_id']
        if tid not in hard_canceled:
            cov = gp.LinExpr()
            for did, dd in duty_data.items():
                if t in dd['V_d']:
                    cov += Y[did, tid]
                if tid in dd['orig_tasks']:
                    cov += Z[did]
            m.addConstr(cov + Cancel[tid] >= 1)

    def extract_solution(model):
        sol_assignments = {}
        for did, dd in duty_data.items():
            z_val = model.cbGetSolution(Z[did]) if model else Z[did].X
            if z_val > 0.5:
                seq = dd['orig_tasks']
                cost = 0.0
            else:
                seq = []
                curr = 'src'
                while curr != 'sink':
                    next_node = None
                    for u, v, c in dd['E_d']:
                        if u == curr:
                            val = model.cbGetSolution(X[did, u, v]) if model else X[did, u, v].X
                            if val > 0.5:
                                next_node = v
                                break
                    if next_node is None: break
                    if next_node != 'sink': seq.append(next_node)
                    curr = next_node
                    
                cost = cost_duty_change
                t_val = model.cbGetSolution(Taxi[did]) if model else Taxi[did].X
                if t_val > 0.5: cost += cost_taxi
                
                orig_set = set(dd['orig_tasks'])
                orig_adj = set(zip(dd['orig_tasks'][:-1], dd['orig_tasks'][1:]))
                for tid in seq:
                    if tid not in orig_set: cost += cost_reassign
                for i in range(len(seq)-1):
                    if (seq[i], seq[i+1]) not in orig_adj: cost += cost_new_trans
                    
            uses_taxi = False
            curr_s, curr_t = dd['src_s'], dd['src_t']
            for tid in seq:
                t = tasks_dict[tid]
                dist = EAT_query(curr_s, curr_t, t['departure_station'], t['departure_time_minutes'])
                if dist > 0: uses_taxi = True
                curr_s, curr_t = t['arrival_station'], t['arrival_time_minutes']
                
            cb = dd['cb']
            ret_s, ret_t, rel = cb, curr_t, False
            if curr_s == cb:
                pass
            else:
                found_base = False
                if (curr_s, curr_t) in dist_maps:
                    d_map = dist_maps[(curr_s, curr_t)]
                    for st in sorted_times[cb]:
                        if st <= dd['horizon']:
                            c = d_map.get((cb, st), float('inf'))
                            if c < float('inf'):
                                if c > 0: uses_taxi = True
                                ret_s = cb
                                ret_t = st
                                found_base = True
                                break
                if not found_base:
                    rel = True
                    ret_s = curr_s
                    ret_t = curr_t
                    
            if z_val > 0.5: uses_taxi = False
            
            sol_assignments[did] = {
                "completion_index": 0,
                "cost": float(cost),
                "covered_tasks": seq,
                "uses_taxi": bool(uses_taxi),
                "release_at_relief_point": bool(rel),
                "return_station": str(ret_s),
                "return_time_minutes": int(ret_t)
            }
            
        canc = list(hard_canceled)
        for t in tasks:
            tid = t['task_id']
            if tid not in hard_canceled:
                c_val = model.cbGetSolution(Cancel[tid]) if model else Cancel[tid].X
                if c_val > 0.5: canc.append(tid)
                
        obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ) if model else model.ObjVal
        return obj_val, canc, sol_assignments

    def callback(model, where):
        if where == GRB.Callback.MIPSOL and logger:
            try:
                obj_val, canc, sol_assignments = extract_solution(model)
                sol_dict = {
                    "released_problem_variant": "relief_point_completion_v1",
                    "objective_value": float(obj_val),
                    "canceled_tasks": canc,
                    "duty_assignments": sol_assignments
                }
                logger.log_solution(obj_val, sol_dict)
            except Exception:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj_val)

    m.optimize(callback)

    if m.SolCount > 0:
        obj_val, canc, sol_assignments = extract_solution(None)
        final_sol = {
            "released_problem_variant": "relief_point_completion_v1",
            "objective_value": float(obj_val),
            "canceled_tasks": canc,
            "duty_assignments": sol_assignments
        }
        with open(args.solution_path, 'w') as f:
            json.dump(final_sol, f, indent=2)

if __name__ == '__main__':
    solve()