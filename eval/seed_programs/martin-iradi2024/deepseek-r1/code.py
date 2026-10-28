import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    planning_horizon = instance['planning_horizon_hours']
    cost_params = instance['cost_parameters']
    beta = instance['handling_time_deviation_factor_beta_per_meter']
    ports = {p['id']: p for p in instance['ports']}
    distances = instance['distances_nautical_miles']
    speeds = instance['speeds_knots']
    travel_time_per_nm_by_speed = instance['travel_time_hours_per_nm_by_speed']
    ships = instance['ships']
    external_ships_list = instance['external_ships']

    model = gp.Model("BerthAllocation")
    model.setParam('OutputFlag', 1)
    model.setParam('Threads', 1)

    port_events = {p['id']: [] for p in instance['ports']}
    for ext in external_ships_list:
        port_id = ext['port_id']
        for ship_info in ext['ships']:
            event = {
                'type': 'external',
                'id': ship_info['id'],
                'length': ship_info['length_meters'],
                'start': ship_info['start_time_hours'],
                'end': ship_info['end_time_hours'],
                'pos': ship_info['berthing_position']
            }
            port_events[port_id].append(event)

    all_visits = []
    legs = []
    for s_idx, ship in enumerate(ships):
        ship_visits = []
        for v_idx, visit in enumerate(ship['port_visits']):
            port_id = visit['port_id']
            quay_length = ports[port_id]['quay_length_meters']
            ship_length = ship['length_meters']
            pos_var = model.addVar(lb=0.0, ub=quay_length - ship_length, name=f"pos_s{s_idx}_v{v_idx}")
            start_var = model.addVar(lb=visit['earliest_start_time_hours'], ub=planning_horizon, name=f"start_s{s_idx}_v{v_idx}")
            dev_var = model.addVar(lb=0.0, name=f"dev_s{s_idx}_v{v_idx}")
            ideal_pos = visit['ideal_berthing_position']
            model.addConstr(dev_var >= pos_var - ideal_pos)
            model.addConstr(dev_var >= ideal_pos - pos_var)
            min_handling = visit['min_handling_time_hours']
            handling_time_expr = min_handling * (1 + beta * dev_var)
            end_time_expr = start_var + handling_time_expr
            arrival_var = model.addVar(lb=0.0, ub=planning_horizon, name=f"arrival_s{s_idx}_v{v_idx}")
            if v_idx == 0:
                model.addConstr(arrival_var == start_var)
            delay_var = model.addVar(lb=0.0, name=f"delay_s{s_idx}_v{v_idx}")
            lft_violation_var = model.addVar(lb=0.0, name=f"lft_violation_s{s_idx}_v{v_idx}")
            expected_finish = visit['expected_finish_time_hours']
            latest_finish = visit['latest_finish_time_hours']
            model.addConstr(delay_var >= end_time_expr - expected_finish)
            model.addConstr(lft_violation_var >= end_time_expr - latest_finish)
            visit_data = {
                'port_id': port_id,
                'pos_var': pos_var,
                'start_var': start_var,
                'dev_var': dev_var,
                'handling_time_expr': handling_time_expr,
                'end_time_expr': end_time_expr,
                'arrival_var': arrival_var,
                'delay_var': delay_var,
                'lft_violation_var': lft_violation_var,
                'input_visit': visit
            }
            ship_visits.append(visit_data)
            event = {
                'type': 'optimized',
                'ship_idx': s_idx,
                'visit_idx': v_idx,
                'length': ship['length_meters'],
                'start_var': start_var,
                'end_var': end_time_expr,
                'pos_var': pos_var
            }
            port_events[port_id].append(event)
        all_visits.append(ship_visits)
        
        ship_legs = []
        for v_idx in range(len(ship['port_visits']) - 1):
            port1 = ship['port_visits'][v_idx]['port_id']
            port2 = ship['port_visits'][v_idx+1]['port_id']
            dist_key = f"{port1}_{port2}"
            dist = distances.get(dist_key)
            if dist is None:
                raise ValueError(f"Distance not found for {dist_key}")
            speed_vars = []
            travel_time_contrib = []
            fuel_contrib = []
            for k, speed_val in enumerate(speeds):
                speed_str = str(speed_val)
                travel_time_per_nm = travel_time_per_nm_by_speed.get(speed_str)
                if travel_time_per_nm is None:
                    raise ValueError(f"Travel time per nm not found for speed {speed_str}")
                fuel_per_nm = ship['fuel_consumption_per_nm_by_speed'].get(speed_str)
                if fuel_per_nm is None:
                    raise ValueError(f"Fuel consumption per nm not found for ship {s_idx} at speed {speed_str}")
                var = model.addVar(vtype=GRB.BINARY, name=f"speed_s{s_idx}_leg{v_idx}_k{k}")
                speed_vars.append(var)
                travel_time_contrib.append(var * (travel_time_per_nm * dist))
                fuel_contrib.append(var * (fuel_per_nm * dist))
            model.addConstr(sum(speed_vars) == 1)
            travel_time_leg = gp.quicksum(travel_time_contrib)
            fuel_consumed_leg = gp.quicksum(fuel_contrib)
            model.addConstr(
                all_visits[s_idx][v_idx+1]['arrival_var'] == 
                all_visits[s_idx][v_idx]['start_var'] + 
                all_visits[s_idx][v_idx]['handling_time_expr'] + 
                travel_time_leg
            )
            model.addConstr(all_visits[s_idx][v_idx+1]['start_var'] >= all_visits[s_idx][v_idx+1]['arrival_var'])
            leg_data = {
                'from_idx': v_idx,
                'to_idx': v_idx+1,
                'speed_vars': speed_vars,
                'travel_time_leg': travel_time_leg,
                'fuel_consumed_leg': fuel_consumed_leg,
                'speeds': speeds
            }
            ship_legs.append(leg_data)
        legs.append(ship_legs)
    
    for port_id, events in port_events.items():
        quay_length = ports[port_id]['quay_length_meters']
        n_events = len(events)
        M_spatial = quay_length
        M_temporal = planning_horizon
        for i in range(n_events):
            for j in range(i+1, n_events):
                event_i = events[i]
                event_j = events[j]
                z1 = model.addVar(vtype=GRB.BINARY, name=f"z1_{port_id}_{i}_{j}")
                z2 = model.addVar(vtype=GRB.BINARY, name=f"z2_{port_id}_{i}_{j}")
                z3 = model.addVar(vtype=GRB.BINARY, name=f"z3_{port_id}_{i}_{j}")
                z4 = model.addVar(vtype=GRB.BINARY, name=f"z4_{port_id}_{i}_{j}")
                if event_i['type'] == 'optimized':
                    pos_i = event_i['pos_var']
                    start_i = event_i['start_var']
                    end_i = event_i['end_var']
                else:
                    pos_i = event_i['pos']
                    start_i = event_i['start']
                    end_i = event_i['end']
                len_i = event_i['length']
                if event_j['type'] == 'optimized':
                    pos_j = event_j['pos_var']
                    start_j = event_j['start_var']
                    end_j = event_j['end_var']
                else:
                    pos_j = event_j['pos']
                    start_j = event_j['start']
                    end_j = event_j['end']
                len_j = event_j['length']
                model.addConstr(pos_i + len_i <= pos_j + M_spatial * (1 - z1))
                model.addConstr(pos_j + len_j <= pos_i + M_spatial * (1 - z2))
                model.addConstr(end_i <= start_j + M_temporal * (1 - z3))
                model.addConstr(end_j <= start_i + M_temporal * (1 - z4))
                model.addConstr(z1 + z2 + z3 + z4 >= 1)
    
    total_cost = 0
    for s_idx, ship in enumerate(ships):
        for v_idx in range(len(ship['port_visits'])):
            visit_data = all_visits[s_idx][v_idx]
            total_cost += cost_params['waiting_cost_usd_per_hour'] * (visit_data['start_var'] - visit_data['arrival_var'])
            total_cost += cost_params['handling_cost_usd_per_hour'] * visit_data['handling_time_expr']
            total_cost += cost_params['delay_cost_usd_per_hour'] * visit_data['delay_var']
            total_cost += cost_params['lft_penalty_usd'] * visit_data['lft_violation_var']
        if s_idx < len(legs):
            for leg in legs[s_idx]:
                total_cost += cost_params['fuel_price_usd_per_tonne'] * leg['fuel_consumed_leg']
    model.setObjective(total_cost, GRB.MINIMIZE)
    
    model.setParam('TimeLimit', args.time_limit)
    
    if logger:
        model._current_best = None
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if model._current_best is None or obj < model._current_best:
                    model._current_best = obj
                    logger.log(obj)
        model.optimize(callback)
    else:
        model.optimize()
    
    if model.status == GRB.OPTIMAL or (model.status == GRB.TIME_LIMIT and model.SolCount > 0):
        solution = {
            "objective_value": model.ObjVal,
            "ships": []
        }
        for s_idx, ship in enumerate(ships):
            ship_sol = {
                "ship_id": ship['id'],
                "port_visits": [],
                "speed_selections": []
            }
            for v_idx, visit_data in enumerate(all_visits[s_idx]):
                input_visit = visit_data['input_visit']
                port_visit_sol = {
                    "port_call_index": input_visit['port_call_index'],
                    "port_id": visit_data['port_id'],
                    "berthing_position": visit_data['pos_var'].X,
                    "start_time": visit_data['start_var'].X,
                    "handling_time": visit_data['handling_time_expr'].getValue(),
                    "arrival_time": visit_data['arrival_var'].X,
                    "delay": visit_data['delay_var'].X,
                    "lft_violation": visit_data['lft_violation_var'].X,
                    "position_deviation": visit_data['dev_var'].X
                }
                ship_sol['port_visits'].append(port_visit_sol)
            if s_idx < len(legs):
                for leg in legs[s_idx]:
                    from_idx = leg['from_idx']
                    to_idx = leg['to_idx']
                    from_call = all_visits[s_idx][from_idx]['input_visit']['port_call_index']
                    to_call = all_visits[s_idx][to_idx]['input_visit']['port_call_index']
                    chosen_speed = None
                    for k, var in enumerate(leg['speed_vars']):
                        if var.X > 0.5:
                            chosen_speed = leg['speeds'][k]
                            break
                    if chosen_speed is None and leg['speed_vars']:
                        chosen_speed = leg['speeds'][0]
                    ship_sol['speed_selections'].append({
                        "from_call": from_call,
                        "to_call": to_call,
                        "speed_knots": chosen_speed
                    })
            solution['ships'].append(ship_sol)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        solution = {
            "objective_value": 1e20,
            "ships": []
        }
        for s_idx, ship in enumerate(ships):
            ship_sol = {
                "ship_id": ship['id'],
                "port_visits": [],
                "speed_selections": []
            }
            design_speed = ship['design_speed_knots']
            chosen_speed = min(speeds, key=lambda x: abs(x - design_speed)) if speeds else design_speed
            for v_idx, visit in enumerate(ship['port_visits']):
                start_time = visit['earliest_start_time_hours']
                handling_time = visit['min_handling_time_hours']
                end_time = start_time + handling_time
                delay = max(0, end_time - visit['expected_finish_time_hours'])
                lft_violation = max(0, end_time - visit['latest_finish_time_hours'])
                port_visit_sol = {
                    "port_call_index": visit['port_call_index'],
                    "port_id": visit['port_id'],
                    "berthing_position": visit['ideal_berthing_position'],
                    "start_time": start_time,
                    "handling_time": handling_time,
                    "arrival_time": start_time,
                    "delay": delay,
                    "lft_violation": lft_violation,
                    "position_deviation": 0.0
                }
                ship_sol['port_visits'].append(port_visit_sol)
            for v_idx in range(len(ship['port_visits']) - 1):
                from_call = ship['port_visits'][v_idx]['port_call_index']
                to_call = ship['port_visits'][v_idx+1]['port_call_index']
                ship_sol['speed_selections'].append({
                    "from_call": from_call,
                    "to_call": to_call,
                    "speed_knots": chosen_speed
                })
            solution['ships'].append(ship_sol)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()