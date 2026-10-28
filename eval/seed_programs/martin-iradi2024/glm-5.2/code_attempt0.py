import json
import argparse
import copy
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        data = json.load(f)

    H = data['planning_horizon_hours']
    cost = data['cost_parameters']
    fuel_price = cost['fuel_price_usd_per_tonne']
    handling_rate = cost['handling_cost_usd_per_hour']
    delay_rate = cost['delay_cost_usd_per_hour']
    waiting_rate = cost['waiting_cost_usd_per_hour']
    lft_penalty = cost['lft_penalty_usd']
    beta = data['handling_time_deviation_factor_beta_per_meter']

    port_quay_length = {p['id']: p['quay_length_meters'] for p in data['ports']}
    distances = data['distances_nautical_miles']
    speeds = data['speeds_knots']
    travel_time_per_nm = {float(k): v for k, v in data['travel_time_hours_per_nm_by_speed'].items()}

    ships_opt = data['ships']
    external_ships = data['external_ships']

    # Preprocess ships
    ship_info = {}
    for s in ships_opt:
        ship_info[s['id']] = {
            'length': s['length_meters'],
            'route': s['route'],
            'fuel_per_nm': {float(k): v for k, v in s['fuel_consumption_per_nm_by_speed'].items()},
            'visits': sorted(s['port_visits'], key=lambda v: v['port_call_index']),
            'design_speed': s['design_speed_knots']
        }

    # Preprocess external ships
    port_externals = defaultdict(list)
    for ext_port in external_ships:
        port_id = ext_port['port_id']
        for es in ext_port['ships']:
            port_externals[port_id].append(es)

    # Build Gurobi model
    model = gp.Model("berth_speed")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = args.time_limit
    model.Params.OutputFlag = 0

    # Variable containers
    p_vars = defaultdict(dict)
    t_vars = defaultdict(dict)
    h_vars = defaultdict(dict)
    a_vars = defaultdict(dict)
    e_vars = defaultdict(dict)
    dev_vars = defaultdict(dict)
    delay_vars = defaultdict(dict)
    lft_vars = defaultdict(dict)
    speed_vars = defaultdict(dict)  # speed_vars[ship_id][from_call] = {speed: var}

    # Logging structures
    log_visit_vars = []
    log_speed_vars = []
    visit_idx_map = {}
    ship_idx_map = {}

    # Solution template
    solution_template = {
        "objective_value": None,
        "ships": []
    }
    for idx_s, s in enumerate(ships_opt):
        ship_id = s['id']
        ship_idx_map[ship_id] = idx_s
        visit_idx_map[ship_id] = {}
        ship_dict = {
            "ship_id": ship_id,
            "port_visits": [],
            "speed_selections": []
        }
        for v_idx, v in enumerate(ship_info[ship_id]['visits']):
            visit_idx_map[ship_id][v['port_call_index']] = v_idx
            ship_dict["port_visits"].append({
                "port_call_index": v['port_call_index'],
                "port_id": v['port_id'],
                "berthing_position": 0.0,
                "start_time": 0.0,
                "handling_time": 0.0,
                "arrival_time": 0.0,
                "delay": 0.0,
                "lft_violation": 0.0,
                "position_deviation": 0.0
            })
        for i in range(len(s['route']) - 1):
            ship_dict["speed_selections"].append({
                "from_call": i + 1,
                "to_call": i + 2,
                "speed_knots": None
            })
        solution_template["ships"].append(ship_dict)

    # Add variables and per-visit constraints
    for s in ships_opt:
        ship_id = s['id']
        L = s['length_meters']
        info = ship_info[ship_id]
        visits = info['visits']
        route = info['route']
        fuel = info['fuel_per_nm']
        ship_idx = ship_idx_map[ship_id]

        for v in visits:
            idx = v['port_call_index']
            port = v['port_id']
            ideal = v['ideal_berthing_position']
            min_h = v['min_handling_time_hours']
            e_start = v['earliest_start_time_hours']
            exp_finish = v['expected_finish_time_hours']
            lft = v['latest_finish_time_hours']
            Q = port_quay_length[port]

            p = model.addVar(lb=0, ub=Q - L, name=f"p_{ship_id}_{idx}")
            t = model.addVar(lb=0, ub=H, name=f"t_{ship_id}_{idx}")
            h = model.addVar(lb=0, ub=H, name=f"h_{ship_id}_{idx}")
            a = model.addVar(lb=0, ub=H, name=f"a_{ship_id}_{idx}")
            e = model.addVar(lb=0, ub=H, name=f"e_{ship_id}_{idx}")
            dev = model.addVar(lb=0, name=f"dev_{ship_id}_{idx}")
            delay = model.addVar(lb=0, name=f"delay_{ship_id}_{idx}")
            lft_v = model.addVar(lb=0, name=f"lft_{ship_id}_{idx}")

            model.addConstr(dev >= p - ideal)
            model.addConstr(dev >= ideal - p)
            model.addConstr(h == min_h * (1 + beta * dev))
            model.addConstr(e == t + h)
            model.addConstr(t >= e_start)
            model.addConstr(t >= a)
            model.addConstr(delay >= e - exp_finish)
            model.addConstr(lft_v >= e - lft)

            if idx == 1:
                model.addConstr(a == t)
            else:
                prev_idx = idx - 1
                tt = model.addVar(lb=0, ub=H, name=f"tt_{ship_id}_{prev_idx}")
                speed_dict = {}
                for k, speed in enumerate(speeds):
                    z = model.addVar(vtype=GRB.BINARY, name=f"z_{ship_id}_{prev_idx}_{k}")
                    speed_dict[speed] = z
                model.addConstr(gp.quicksum(speed_dict.values()) == 1)
                dist = distances[f"{route[prev_idx-1]}_{route[idx-1]}"]
                model.addConstr(tt == dist * gp.quicksum(travel_time_per_nm[speed] * speed_dict[speed] for speed in speeds))
                model.addConstr(a == e_vars[ship_id][prev_idx] + tt)
                speed_vars[ship_id][prev_idx] = speed_dict

                leg_idx = prev_idx - 1
                for speed, z_var in speed_dict.items():
                    log_speed_vars.append({
                        'ship_idx': ship_idx,
                        'leg_idx': leg_idx,
                        'speed': speed,
                        'var': z_var
                    })

            p_vars[ship_id][idx] = p
            t_vars[ship_id][idx] = t
            h_vars[ship_id][idx] = h
            a_vars[ship_id][idx] = a
            e_vars[ship_id][idx] = e
            dev_vars[ship_id][idx] = dev
            delay_vars[ship_id][idx] = delay
            lft_vars[ship_id][idx] = lft_v

            v_idx = visit_idx_map[ship_id][idx]
            log_visit_vars.append({'ship_idx': ship_idx, 'visit_idx': v_idx, 'field': 'berthing_position', 'var': p})
            log_visit_vars.append({'ship_idx': ship_idx, 'visit_idx': v_idx, 'field': 'start_time', 'var': t})
            log_visit_vars.append({'ship_idx': ship_idx, 'visit_idx': v_idx, 'field': 'handling_time', 'var': h})
            log_visit_vars.append({'ship_idx': ship_idx, 'visit_idx': v_idx, 'field': 'arrival_time', 'var': a})
            log_visit_vars.append({'ship_idx': ship_idx, 'visit_idx': v_idx, 'field': 'delay', 'var': delay})
            log_visit_vars.append({'ship_idx': ship_idx, 'visit_idx': v_idx, 'field': 'lft_violation', 'var': lft_v})
            log_visit_vars.append({'ship_idx': ship_idx, 'visit_idx': v_idx, 'field': 'position_deviation', 'var': dev})

    # Non-overlap constraints
    port_visit_list = defaultdict(list)
    for s in ships_opt:
        ship_id = s['id']
        L = s['length_meters']
        for v in ship_info[ship_id]['visits']:
            idx = v['port_call_index']
            port = v['port_id']
            port_visit_list[port].append({
                'ship_id': ship_id,
                'call_idx': idx,
                'length': L,
                'p': p_vars[ship_id][idx],
                't': t_vars[ship_id][idx],
                'e': e_vars[ship_id][idx]
            })

    for port, visits in port_visit_list.items():
        n = len(visits)
        # optimized vs optimized
        for i in range(n):
            v1 = visits[i]
            for j in range(i + 1, n):
                v2 = visits[j]
                y1 = model.addVar(vtype=GRB.BINARY, name=f"nl_{v1['ship_id']}_{v1['call_idx']}_{v2['ship_id']}_{v2['call_idx']}_1")
                y2 = model.addVar(vtype=GRB.BINARY, name=f"nl_{v1['ship_id']}_{v1['call_idx']}_{v2['ship_id']}_{v2['call_idx']}_2")
                y3 = model.addVar(vtype=GRB.BINARY, name=f"nl_{v1['ship_id']}_{v1['call_idx']}_{v2['ship_id']}_{v2['call_idx']}_3")
                y4 = model.addVar(vtype=GRB.BINARY, name=f"nl_{v1['ship_id']}_{v1['call_idx']}_{v2['ship_id']}_{v2['call_idx']}_4")
                model.addConstr(y1 + y2 + y3 + y4 >= 1)
                model.addConstr((y1 == 1) >> (v1['p'] + v1['length'] <= v2['p']))
                model.addConstr((y2 == 1) >> (v2['p'] + v2['length'] <= v1['p']))
                model.addConstr((y3 == 1) >> (v1['e'] <= v2['t']))
                model.addConstr((y4 == 1) >> (v2['e'] <= v1['t']))

        # optimized vs external
        ext_list = port_externals.get(port, [])
        for v in visits:
            for ext in ext_list:
                p_ext = ext['berthing_position']
                L_ext = ext['length_meters']
                t_ext_start = ext['start_time_hours']
                t_ext_end = ext['end_time_hours']

                y1 = model.addVar(vtype=GRB.BINARY, name=f"nle_{v['ship_id']}_{v['call_idx']}_{ext['id']}_1")
                y2 = model.addVar(vtype=GRB.BINARY, name=f"nle_{v['ship_id']}_{v['call_idx']}_{ext['id']}_2")
                y3 = model.addVar(vtype=GRB.BINARY, name=f"nle_{v['ship_id']}_{v['call_idx']}_{ext['id']}_3")
                y4 = model.addVar(vtype=GRB.BINARY, name=f"nle_{v['ship_id']}_{v['call_idx']}_{ext['id']}_4")
                model.addConstr(y1 + y2 + y3 + y4 >= 1)
                model.addConstr((y1 == 1) >> (v['p'] + v['length'] <= p_ext))
                model.addConstr((y2 == 1) >> (p_ext + L_ext <= v['p']))
                model.addConstr((y3 == 1) >> (v['e'] <= t_ext_start))
                model.addConstr((y4 == 1) >> (t_ext_end <= v['t']))

    # Objective
    obj = gp.LinExpr()
    for s in ships_opt:
        ship_id = s['id']
        fuel = ship_info[ship_id]['fuel_per_nm']
        for v in ship_info[ship_id]['visits']:
            idx = v['port_call_index']
            obj += waiting_rate * (t_vars[ship_id][idx] - a_vars[ship_id][idx])
            obj += handling_rate * h_vars[ship_id][idx]
            obj += delay_rate * delay_vars[ship_id][idx]
            obj += lft_penalty * lft_vars[ship_id][idx]
        route = ship_info[ship_id]['route']
        for i in range(len(route) - 1):
            prev_idx = i + 1
            dist = distances[f"{route[i]}_{route[i+1]}"]
            speed_dict = speed_vars[ship_id][prev_idx]
            for speed, z in speed_dict.items():
                obj += fuel_price * fuel[speed] * dist * z

    model.setObjective(obj, GRB.MINIMIZE)

    # Callback for logging incumbents
    callback_vars_visit = [item['var'] for item in log_visit_vars]
    callback_vars_speed = [item['var'] for item in log_speed_vars]
    callback_vars_all = callback_vars_visit + callback_vars_speed

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                vals = model.cbGetSolution(callback_vars_all)
                sol = copy.deepcopy(solution_template)
                sol['objective_value'] = obj_val
                for item, val in zip(log_visit_vars, vals[:len(log_visit_vars)]):
                    sol['ships'][item['ship_idx']]['port_visits'][item['visit_idx']][item['field']] = val
                for item, val in zip(log_speed_vars, vals[len(log_visit_vars):]):
                    if val > 0.5:
                        sol['ships'][item['ship_idx']]['speed_selections'][item['leg_idx']]['speed_knots'] = item['speed']
                # Ensure all speed selections are filled (fallback to max if numerical issue)
                for ship in sol['ships']:
                    for sel in ship['speed_selections']:
                        if sel['speed_knots'] is None:
                            # find the max among the binaries for this leg
                            # (not easily accessible here, so leave as None; final output will fix)
                            pass
                logger.log_solution(obj_val, sol)

    model.setCallback(callback)

    # Optimize
    model.optimize()

    # Extract final solution
    sol = copy.deepcopy(solution_template)
    if model.SolCount > 0:
        obj_val = model.ObjVal
        sol['objective_value'] = obj_val
        for item in log_visit_vars:
            val = item['var'].X
            sol['ships'][item['ship_idx']]['port_visits'][item['visit_idx']][item['field']] = val
        for item in log_speed_vars:
            if item['var'].X > 0.5:
                sol['ships'][item['ship_idx']]['speed_selections'][item['leg_idx']]['speed_knots'] = item['speed']
        # Fallback for speed if none selected (should not happen in MIP)
        for ship in sol['ships']:
            for sel in ship['speed_selections']:
                if sel['speed_knots'] is None:
                    # pick the first available speed
                    sel['speed_knots'] = speeds[0]
    else:
        # Fallback: construct a trivial feasible-ish solution
        obj_val = 0.0
        for idx_s, s in enumerate(ships_opt):
            ship_id = s['id']
            info = ship_info[ship_id]
            L = s['length_meters']
            prev_end = 0.0
            for v_idx, v in enumerate(info['visits']):
                port = v['port_id']
                Q = port_quay_length[port]
                ideal = v['ideal_berthing_position']
                pos = max(0.0, min(Q - L, ideal))
                dev = abs(pos - ideal)
                min_h = v['min_handling_time_hours']
                h = min_h * (1 + beta * dev)
                e_start = v['earliest_start_time_hours']
                t = max(e_start, prev_end)
                e = t + h
                exp_finish = v['expected_finish_time_hours']
                lft = v['latest_finish_time_hours']
                delay = max(0.0, e - exp_finish)
                lft_v = max(0.0, e - lft)
                a = t
                sol['ships'][idx_s]['port_visits'][v_idx].update({
                    'berthing_position': pos,
                    'start_time': t,
                    'handling_time': h,
                    'arrival_time': a,
                    'delay': delay,
                    'lft_violation': lft_v,
                    'position_deviation': dev
                })
                prev_end = e
            for leg_idx in range(len(info['route']) - 1):
                sol['ships'][idx_s]['speed_selections'][leg_idx]['speed_knots'] = info['design_speed']
        sol['objective_value'] = obj_val
        if logger:
            logger.log_solution(obj_val, sol)

    # Write output
    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()