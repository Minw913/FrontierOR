import argparse
import json
import time
import math
from itertools import combinations

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Parse instance data
    cost_params = data['cost_parameters']
    fuel_price = cost_params['fuel_price_usd_per_tonne']
    handling_cost = cost_params['handling_cost_usd_per_hour']
    delay_cost = cost_params['delay_cost_usd_per_hour']
    waiting_cost = cost_params['waiting_cost_usd_per_hour']
    lft_penalty = cost_params['lft_penalty_usd']
    
    beta = data['handling_time_deviation_factor_beta_per_meter']
    
    ports = {p['id']: p for p in data['ports']}
    distances = data['distances_nautical_miles']
    speeds = data['speeds_knots']
    travel_time_per_nm = data['travel_time_hours_per_nm_by_speed']
    
    ships = data['ships']
    external_ships_data = data.get('external_ships', [])
    
    # Build external ships by port
    external_by_port = {}
    for ext_port in external_ships_data:
        pid = ext_port['port_id']
        if pid not in external_by_port:
            external_by_port[pid] = []
        for es in ext_port['ships']:
            external_by_port[pid].append({
                'id': es['id'],
                'position': es['berthing_position'],
                'length': es['length_meters'],
                'start': es['start_time_hours'],
                'end': es['end_time_hours'],
                'handling': es['handling_time_hours']
            })
    
    # For each ship, find optimal speed for each leg (minimize fuel cost alone as baseline)
    # Then try to improve with a greedy/heuristic approach
    
    def get_fuel_cost(ship, speed, dist):
        fc_per_nm = ship['fuel_consumption_per_nm_by_speed'][str(speed)]
        return fuel_price * fc_per_nm * dist
    
    def get_travel_time(speed, dist):
        tt_per_nm = travel_time_per_nm[str(speed)]
        return tt_per_nm * dist
    
    def get_handling_time(pv, pos):
        dev = abs(pos - pv['ideal_berthing_position'])
        return pv['min_handling_time_hours'] * (1 + beta * dev)
    
    # Greedy construction: process ships one by one, assign positions and times
    # For each port visit, try ideal position first, then adjust for conflicts
    
    # Collect all assignments (for conflict checking)
    # assignments[port_id] = list of (position, length, start, end, ship_id)
    assignments = {}
    for pid in ports:
        assignments[pid] = []
        if pid in external_by_port:
            for es in external_by_port[pid]:
                assignments[pid].append({
                    'pos': es['position'],
                    'len': es['length'],
                    'start': es['start'],
                    'end': es['start'] + es['handling'],
                    'ship_id': ('ext', es['id']),
                    'fixed': True
                })
    
    def check_conflicts(port_id, pos, length, start, end, exclude_ship_id=None):
        """Check if placing a ship at (pos, length, start, end) conflicts with existing assignments."""
        conflicts = []
        for a in assignments[port_id]:
            if exclude_ship_id and a['ship_id'] == exclude_ship_id:
                continue
            # Check overlap in both space and time
            space_overlap = not (pos + length <= a['pos'] or a['pos'] + a['len'] <= pos)
            time_overlap = not (end <= a['start'] or a['end'] <= start)
            if space_overlap and time_overlap:
                conflicts.append(a)
        return conflicts
    
    def find_earliest_start(port_id, pos, length, earliest, handling_time, exclude_ship_id=None):
        """Find earliest start time >= earliest such that no conflicts exist."""
        start = earliest
        end = start + handling_time
        max_iter = 200
        for _ in range(max_iter):
            conflicts = check_conflicts(port_id, pos, length, start, end, exclude_ship_id)
            if not conflicts:
                return start
            # Push start past the earliest conflict end
            min_end = min(c['end'] for c in conflicts)
            start = min_end
            end = start + handling_time
        return start
    
    def find_best_position_and_start(port_id, ship_length, pv, arrival_time, exclude_ship_id=None):
        """Find best position and start time for a port visit."""
        quay_len = ports[port_id]['quay_length_meters']
        ideal_pos = pv['ideal_berthing_position']
        earliest = max(pv['earliest_start_time_hours'], arrival_time)
        
        best_cost = float('inf')
        best_pos = None
        best_start = None
        best_ht = None
        
        # Try a set of candidate positions
        candidate_positions = set()
        candidate_positions.add(ideal_pos)
        
        # Add positions at boundaries of external ships
        for a in assignments[port_id]:
            if exclude_ship_id and a['ship_id'] == exclude_ship_id:
                continue
            candidate_positions.add(a['pos'] + a['len'])  # right after this ship
            candidate_positions.add(a['pos'] - ship_length)  # right before this ship
        
        candidate_positions.add(0)
        candidate_positions.add(quay_len - ship_length)
        
        # Add positions near ideal
        for delta in range(-50, 51, 5):
            candidate_positions.add(ideal_pos + delta)
        for delta in range(-200, 201, 25):
            candidate_positions.add(ideal_pos + delta)
        
        # Filter valid positions
        valid_positions = []
        for pos in candidate_positions:
            pos = max(0, min(pos, quay_len - ship_length))
            pos = round(pos, 2)
            valid_positions.append(pos)
        valid_positions = sorted(set(valid_positions))
        
        for pos in valid_positions:
            if pos < 0 or pos + ship_length > quay_len:
                continue
            
            ht = get_handling_time(pv, pos)
            start = find_earliest_start(port_id, pos, ship_length, earliest, ht, exclude_ship_id)
            end = start + ht
            
            dev = abs(pos - ideal_pos)
            
            # Cost components for this visit
            wait = start - arrival_time
            delay = max(0, end - pv['expected_finish_time_hours'])
            lft_viol = max(0, end - pv['latest_finish_time_hours'])
            
            visit_cost = (waiting_cost * wait + handling_cost * ht + 
                         delay_cost * delay + lft_penalty * lft_viol)
            
            if visit_cost < best_cost:
                best_cost = visit_cost
                best_pos = pos
                best_start = start
                best_ht = ht
        
        return best_pos, best_start, best_ht, best_cost
    
    # Build solution
    solution_ships = []
    total_objective = 0.0
    
    # Sort ships by earliest start time of first port visit
    ship_order = sorted(range(len(ships)), key=lambda i: ships[i]['port_visits'][0]['earliest_start_time_hours'])
    
    for si in ship_order:
        ship = ships[si]
        ship_id = ship['id']
        ship_length = ship['length_meters']
        route = ship['route']
        pvs = ship['port_visits']
        
        port_visit_results = []
        speed_selections = []
        
        prev_end_time = None
        prev_port = None
        
        for vi, pv in enumerate(pvs):
            port_id = pv['port_id']
            
            # Determine arrival time
            if vi == 0:
                arrival_time = pv['earliest_start_time_hours']
                selected_speed = None
                fuel_cost_leg = 0
            else:
                # Choose speed for this leg
                prev_pv_result = port_visit_results[-1]
                departure_time = prev_pv_result['start'] + prev_pv_result['handling_time']
                
                dist_key = f"{prev_port}_{port_id}"
                dist = distances[dist_key]
                
                best_speed = None
                best_leg_cost = float('inf')
                best_arrival = None
                
                for speed in speeds:
                    tt = get_travel_time(speed, dist)
                    arr = departure_time + tt
                    fc = get_fuel_cost(ship, speed, dist)
                    
                    # Estimate visit cost at this arrival
                    earliest = max(pv['earliest_start_time_hours'], arr)
                    ideal_pos = pv['ideal_berthing_position']
                    ht_ideal = pv['min_handling_time_hours']
                    end_ideal = earliest + ht_ideal
                    
                    wait_est = earliest - arr
                    delay_est = max(0, end_ideal - pv['expected_finish_time_hours'])
                    lft_est = max(0, end_ideal - pv['latest_finish_time_hours'])
                    
                    leg_cost = fc + waiting_cost * wait_est + delay_cost * delay_est + lft_penalty * lft_est
                    
                    if leg_cost < best_leg_cost:
                        best_leg_cost = leg_cost
                        best_speed = speed
                        best_arrival = arr
                
                arrival_time = best_arrival
                selected_speed = best_speed
                fuel_cost_leg = get_fuel_cost(ship, best_speed, dist)
                
                speed_selections.append({
                    'from_call': vi,
                    'to_call': vi + 1,
                    'speed_knots': best_speed
                })
                
                total_objective += fuel_cost_leg
            
            # Find best position and start
            pos, start, ht, visit_cost = find_best_position_and_start(
                port_id, ship_length, pv, arrival_time)
            
            end = start + ht
            dev = abs(pos - pv['ideal_berthing_position'])
            delay = max(0, end - pv['expected_finish_time_hours'])
            lft_viol = max(0, end - pv['latest_finish_time_hours'])
            wait = start - arrival_time
            
            port_visit_results.append({
                'port_call_index': pv['port_call_index'],
                'port_id': port_id,
                'berthing_position': pos,
                'start': start,
                'handling_time': ht,
                'arrival_time': arrival_time,
                'delay': delay,
                'lft_violation': lft_viol,
                'position_deviation': dev
            })
            
            # Add to assignments
            assignments[port_id].append({
                'pos': pos,
                'len': ship_length,
                'start': start,
                'end': end,
                'ship_id': ('opt', ship_id),
                'fixed': False
            })
            
            total_objective += visit_cost
            
            prev_end_time = end
            prev_port = port_id
        
        sol_pvs = []
        for pvr in port_visit_results:
            sol_pvs.append({
                'port_call_index': pvr['port_call_index'],
                'port_id': pvr['port_id'],
                'berthing_position': pvr['berthing_position'],
                'start_time': pvr['start'],
                'handling_time': pvr['handling_time'],
                'arrival_time': pvr['arrival_time'],
                'delay': pvr['delay'],
                'lft_violation': pvr['lft_violation'],
                'position_deviation': pvr['position_deviation']
            })
        
        solution_ships.append({
            'ship_id': ship_id,
            'port_visits': sol_pvs,
            'speed_selections': speed_selections
        })
    
    if logger:
        logger.log(total_objective)
    
    # Now try to improve with local search
    remaining_time = time_limit - (time.time() - start_time)
    
    def recompute_objective(solution_ships_list):
        """Recompute total objective from solution."""
        obj = 0.0
        for ss in solution_ships_list:
            ship = next(s for s in ships if s['id'] == ss['ship_id'])
            for pv_sol in ss['port_visits']:
                pv_data = next(pv for pv in ship['port_visits'] if pv['port_call_index'] == pv_sol['port_call_index'])
                wait = pv_sol['start_time'] - pv_sol['arrival_time']
                obj += waiting_cost * wait
                obj += handling_cost * pv_sol['handling_time']
                obj += delay_cost * pv_sol['delay']
                obj += lft_penalty * pv_sol['lft_violation']
            
            for sp in ss['speed_selections']:
                from_idx = sp['from_call'] - 1
                to_idx = sp['to_call'] - 1
                from_port = ship['route'][from_idx]
                to_port = ship['route'][to_idx]
                dist_key = f"{from_port}_{to_port}"
                dist = distances[dist_key]
                fc = get_fuel_cost(ship, sp['speed_knots'], dist)
                obj += fc
        return obj
    
    # Rebuild assignments from solution
    def rebuild_assignments(solution_ships_list):
        new_assignments = {}
        for pid in ports:
            new_assignments[pid] = []
            if pid in external_by_port:
                for es in external_by_port[pid]:
                    new_assignments[pid].append({
                        'pos': es['position'],
                        'len': es['length'],
                        'start': es['start'],
                        'end': es['start'] + es['handling'],
                        'ship_id': ('ext', es['id']),
                        'fixed': True
                    })
        
        for ss in solution_ships_list:
            ship = next(s for s in ships if s['id'] == ss['ship_id'])
            for pv_sol in ss['port_visits']:
                new_assignments[pv_sol['port_id']].append({
                    'pos': pv_sol['berthing_position'],
                    'len': ship['length_meters'],
                    'start': pv_sol['start_time'],
                    'end': pv_sol['start_time'] + pv_sol['handling_time'],
                    'ship_id': ('opt', ss['ship_id']),
                    'fixed': False
                })
        return new_assignments
    
    # Try Gurobi MIP for small instances
    def try_mip_solve(time_remaining):
        """Attempt MIP formulation if instance is small enough."""
        import gurobipy as gp
        from gurobipy import GRB
        
        # Count total port visits
        total_visits = sum(len(s['port_visits']) for s in ships)
        total_legs = sum(len(s['port_visits']) - 1 for s in ships)
        
        # Collect visits per port for conflict constraints
        visits_per_port = {}
        for s in ships:
            for pv in s['port_visits']:
                pid = pv['port_id']
                if pid not in visits_per_port:
                    visits_per_port[pid] = []
                visits_per_port[pid].append((s['id'], pv['port_call_index']))
        
        # Count conflict pairs
        total_pairs = 0
        for pid in visits_per_port:
            n = len(visits_per_port[pid])
            ext_n = len(external_by_port.get(pid, []))
            total_pairs += n * (n - 1) // 2 + n * ext_n
        
        # If too large, skip MIP
        if total_visits > 60 or total_pairs > 200:
            return None
        
        M_time = data.get('planning_horizon_hours', 10000)
        
        try:
            model = gp.Model("BAP")
            model.Params.TimeLimit = max(1, time_remaining - 5)
            model.Params.Threads = 1
            model.Params.OutputFlag = 0
            
            # Decision variables
            # b[s,v] = berthing position
            # t[s,v] = start time
            # sp[s,leg,k] = 1 if speed k selected for leg
            
            b = {}  # berthing position
            t = {}  # start time
            arr = {}  # arrival time
            ht = {}  # handling time
            delay_var = {}  # delay
            lft_var = {}  # lft violation
            wait_var = {}  # waiting time
            dev_pos = {}  # positive deviation
            dev_neg = {}  # negative deviation
            
            speed_var = {}  # binary speed selection
            
            ship_map = {s['id']: s for s in ships}
            
            for s in ships:
                sid = s['id']
                for pv in s['port_visits']:
                    vi = pv['port_call_index']
                    pid = pv['port_id']
                    ql = ports[pid]['quay_length_meters']
                    
                    b[sid, vi] = model.addVar(lb=0, ub=ql - s['length_meters'], name=f"b_{sid}_{vi}")
                    t[sid, vi] = model.addVar(lb=pv['earliest_start_time_hours'], ub=M_time, name=f"t_{sid}_{vi}")
                    arr[sid, vi] = model.addVar(lb=0, ub=M_time, name=f"arr_{sid}_{vi}")
                    
                    dev_pos[sid, vi] = model.addVar(lb=0, ub=ql, name=f"devp_{sid}_{vi}")
                    dev_neg[sid, vi] = model.addVar(lb=0, ub=ql, name=f"devn_{sid}_{vi}")
                    
                    ht[sid, vi] = model.addVar(lb=pv['min_handling_time_hours'], ub=pv['max_handling_time_hours'], name=f"ht_{sid}_{vi}")
                    delay_var[sid, vi] = model.addVar(lb=0, name=f"delay_{sid}_{vi}")
                    lft_var[sid, vi] = model.addVar(lb=0, name=f"lft_{sid}_{vi}")
                    wait_var[sid, vi] = model.addVar(lb=0, name=f"wait_{sid}_{vi}")
                
                for vi_idx in range(len(s['port_visits']) - 1):
                    pv = s['port_visits'][vi_idx]
                    vi = pv['port_call_index']
                    for k, speed in enumerate(speeds):
                        speed_var[sid, vi, k] = model.addVar(vtype=GRB.BINARY, name=f"sp_{sid}_{vi}_{k}")
            
            model.update()
            
            # Constraints
            for s in ships:
                sid = s['id']
                for pv in s['port_visits']:
                    vi = pv['port_call_index']
                    ideal = pv['ideal_berthing_position']
                    
                    # Deviation: b - ideal = dev_pos - dev_neg
                    model.addConstr(b[sid, vi] - ideal == dev_pos[sid, vi] - dev_neg[sid, vi])
                    
                    # Handling time = min_ht * (1 + beta * (dev_pos + dev_neg))
                    model.addConstr(ht[sid, vi] == pv['min_handling_time_hours'] * (1 + beta * (dev_pos[sid, vi] + dev_neg[sid, vi])))
                    
                    # Start >= arrival
                    model.addConstr(t[sid, vi] >= arr[sid, vi])
                    
                    # Start >= earliest
                    model.addConstr(t[sid, vi] >= pv['earliest_start_time_hours'])
                    
                    # Waiting time
                    model.addConstr(wait_var[sid, vi] == t[sid, vi] - arr[sid, vi])
                    
                    # Delay
                    model.addConstr(delay_var[sid, vi] >= t[sid, vi] + ht[sid, vi] - pv['expected_finish_time_hours'])
                    
                    # LFT violation
                    model.addConstr(lft_var[sid, vi] >= t[sid, vi] + ht[sid, vi] - pv['latest_finish_time_hours'])
                
                # First visit: arrival = earliest_start_time (or just set it free since we only need arr <= t)
                first_pv = s['port_visits'][0]
                fvi = first_pv['port_call_index']
                model.addConstr(arr[sid, fvi] <= t[sid, fvi])
                # arrival at first port can be as early as earliest_start
                model.addConstr(arr[sid, fvi] == first_pv['earliest_start_time_hours'])
                
                # Speed selection and arrival constraints
                for vi_idx in range(len(s['port_visits']) - 1):
                    pv_from = s['port_visits'][vi_idx]
                    pv_to = s['port_visits'][vi_idx + 1]
                    vi_from = pv_from['port_call_index']
                    vi_to = pv_to['port_call_index']
                    
                    from_port = pv_from['port_id']
                    to_port = pv_to['port_id']
                    dist_key = f"{from_port}_{to_port}"
                    dist = distances[dist_key]
                    
                    # Exactly one speed
                    model.addConstr(gp.quicksum(speed_var[sid, vi_from, k] for k in range(len(speeds))) == 1)
                    
                    # Arrival at next port = departure from current + travel time
                    # departure = t[sid, vi_from] + ht[sid, vi_from]
                    # travel_time = sum(speed_var * tt_per_nm * dist)
                    travel_time_expr = gp.quicksum(
                        speed_var[sid, vi_from, k] * float(travel_time_per_nm[str(speeds[k])]) * dist
                        for k in range(len(speeds))
                    )
                    
                    model.addConstr(arr[sid, vi_to] == t[sid, vi_from] + ht[sid, vi_from] + travel_time_expr)
            
            # Non-overlap constraints between optimized ships at same port
            M_pos = max(p['quay_length_meters'] for p in data['ports'])
            
            for pid in visits_per_port:
                visit_list = visits_per_port[pid]
                ext_list = external_by_port.get(pid, [])
                
                for i in range(len(visit_list)):
                    sid_i, vi_i = visit_list[i]
                    ship_i = ship_map[sid_i]
                    len_i = ship_i['length_meters']
                    
                    # Between optimized ships
                    for j in range(i + 1, len(visit_list)):
                        sid_j, vi_j = visit_list[j]
                        ship_j = ship_map[sid_j]
                        len_j = ship_j['length_meters']
                        
                        # At least one of 4 conditions:
                        # b_i + len_i <= b_j (i left of j)
                        # b_j + len_j <= b_i (j left of i)
                        # t_i + ht_i <= t_j (i before j)
                        # t_j + ht_j <= t_i (j before i)
                        
                        z1 = model.addVar(vtype=GRB.BINARY)
                        z2 = model.addVar(vtype=GRB.BINARY)
                        z3 = model.addVar(vtype=GRB.BINARY)
                        z4 = model.addVar(vtype=GRB.BINARY)
                        
                        model.addConstr(z1 + z2 + z3 + z4 >= 1)
                        
                        model.addConstr(b[sid_i, vi_i] + len_i <= b[sid_j, vi_j] + M_pos * (1 - z1))
                        model.addConstr(b[sid_j, vi_j] + len_j <= b[sid_i, vi_i] + M_pos * (1 - z2))
                        model.addConstr(t[sid_i, vi_i] + ht[sid_i, vi_i] <= t[sid_j, vi_j] + M_time * (1 - z3))
                        model.addConstr(t[sid_j, vi_j] + ht[sid_j, vi_j] <= t[sid_i, vi_i] + M_time * (1 - z4))
                    
                    # Between optimized and external ships
                    for es in ext_list:
                        e_pos = es['position']
                        e_len = es['length']
                        e_start = es['start']
                        e_end = es['start'] + es['handling']
                        
                        z1 = model.addVar(vtype=GRB.BINARY)
                        z2 = model.addVar(vtype=GRB.BINARY)
                        z3 = model.addVar(vtype=GRB.BINARY)
                        z4 = model.addVar(vtype=GRB.BINARY)
                        
                        model.addConstr(z1 + z2 + z3 + z4 >= 1)
                        
                        model.addConstr(b[sid_i, vi_i] + len_i <= e_pos + M_pos * (1 - z1))
                        model.addConstr(e_pos + e_len <= b[sid_i, vi_i] + M_pos * (1 - z2))
                        model.addConstr(t[sid_i, vi_i] + ht[sid_i, vi_i] <= e_start + M_time * (1 - z3))
                        model.addConstr(e_end <= t[sid_i, vi_i] + M_time * (1 - z4))
            
            # Objective
            obj_expr = gp.LinExpr()
            
            for s in ships:
                sid = s['id']
                for pv in s['port_visits']:
                    vi = pv['port_call_index']
                    obj_expr += waiting_cost * wait_var[sid, vi]
                    obj_expr += handling_cost * ht[sid, vi]
                    obj_expr += delay_cost * delay_var[sid, vi]
                    obj_expr += lft_penalty * lft_var[sid, vi]
                
                for vi_idx in range(len(s['port_visits']) - 1):
                    pv = s['port_visits'][vi_idx]
                    vi = pv['port_call_index']
                    from_port = pv['port_id']
                    to_port = s['port_visits'][vi_idx + 1]['port_id']
                    dist_key = f"{from_port}_{to_port}"
                    dist = distances[dist_key]
                    
                    for k, speed in enumerate(speeds):
                        fc_per_nm = s['fuel_consumption_per_nm_by_speed'][str(speed)]
                        obj_expr += fuel_price * fc_per_nm * dist * speed_var[sid, vi, k]
            
            model.setObjective(obj_expr, GRB.MINIMIZE)
            
            # Set initial solution from greedy
            for ss in solution_ships:
                sid = ss['ship_id']
                s = ship_map[sid]
                for pv_sol in ss['port_visits']:
                    vi = pv_sol['port_call_index']
                    b[sid, vi].Start = pv_sol['berthing_position']
                    t[sid, vi].Start = pv_sol['start_time']
                    arr[sid, vi].Start = pv_sol['arrival_time']
                
                for sp_sel in ss['speed_selections']:
                    vi_from = sp_sel['from_call']
                    speed_val = sp_sel['speed_knots']
                    for k, speed in enumerate(speeds):
                        if abs(speed - speed_val) < 0.01:
                            speed_var[sid, vi_from, k].Start = 1
                        else:
                            speed_var[sid, vi_from, k].Start = 0
            
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if logger:
                        logger.log(obj_val)
            
            model.optimize(callback)
            
            if model.SolCount > 0:
                # Extract solution
                new_solution_ships = []
                new_obj = model.ObjVal
                
                for s in ships:
                    sid = s['id']
                    pvs_sol = []
                    for pv in s['port_visits']:
                        vi = pv['port_call_index']
                        bp = b[sid, vi].X
                        st = t[sid, vi].X
                        ar = arr[sid, vi].X
                        h = ht[sid, vi].X
                        end_t = st + h
                        d = max(0, end_t - pv['expected_finish_time_hours'])
                        lv = max(0, end_t - pv['latest_finish_time_hours'])
                        pd_val = abs(bp - pv['ideal_berthing_position'])
                        
                        pvs_sol.append({
                            'port_call_index': vi,
                            'port_id': pv['port_id'],
                            'berthing_position': round(bp, 6),
                            'start_time': round(st, 6),
                            'handling_time': round(h, 6),
                            'arrival_time': round(ar, 6),
                            'delay': round(d, 6),
                            'lft_violation': round(lv, 6),
                            'position_deviation': round(pd_val, 6)
                        })
                    
                    sps = []
                    for vi_idx in range(len(s['port_visits']) - 1):
                        pv = s['port_visits'][vi_idx]
                        vi = pv['port_call_index']
                        for k, speed in enumerate(speeds):
                            if speed_var[sid, vi, k].X > 0.5:
                                sps.append({
                                    'from_call': vi,
                                    'to_call': s['port_visits'][vi_idx + 1]['port_call_index'],
                                    'speed_knots': speed
                                })
                                break
                    
                    new_solution_ships.append({
                        'ship_id': sid,
                        'port_visits': pvs_sol,
                        'speed_selections': sps
                    })
                
                return new_solution_ships, new_obj
            
            return None
            
        except Exception as e:
            return None
    
    # Try MIP
    remaining = time_limit - (time.time() - start_time)
    if remaining > 10:
        mip_result = try_mip_solve(remaining)
        if mip_result is not None:
            mip_sol, mip_obj = mip_result
            if mip_obj < total_objective:
                solution_ships = mip_sol
                total_objective = mip_obj
                if logger:
                    logger.log(total_objective)
    
    # Recompute objective to be sure
    final_obj = 0.0
    for ss in solution_ships:
        ship = next(s for s in ships if s['id'] == ss['ship_id'])
        for pv_sol in ss['port_visits']:
            wait = max(0, pv_sol['start_time'] - pv_sol['arrival_time'])
            final_obj += waiting_cost * wait
            final_obj += handling_cost * pv_sol['handling_time']
            final_obj += delay_cost * pv_sol['delay']
            final_obj += lft_penalty * pv_sol['lft_violation']
        
        for sp in ss['speed_selections']:
            from_call_idx = sp['from_call']
            to_call_idx = sp['to_call']
            # Find port ids
            from_pv = next(pv for pv in ss['port_visits'] if pv['port_call_index'] == from_call_idx)
            to_pv = next(pv for pv in ss['port_visits'] if pv['port_call_index'] == to_call_idx)
            dist_key = f"{from_pv['port_id']}_{to_pv['port_id']}"
            dist = distances[dist_key]
            fc_per_nm = ship['fuel_consumption_per_nm_by_speed'][str(sp['speed_knots'])]
            final_obj += fuel_price * fc_per_nm * dist
    
    solution = {
        'objective_value': round(final_obj, 6),
        'ships': solution_ships
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)