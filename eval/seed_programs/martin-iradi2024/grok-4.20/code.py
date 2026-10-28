import json
import argparse
import time
import sys
import math
from itertools import combinations
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

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()
    time_limit = args.time_limit

    # Extract data
    ports = {p['id']: p['quay_length_meters'] for p in instance['ports']}
    distances = instance['distances_nautical_miles']
    speeds = instance['speeds_knots']
    travel_times = instance['travel_time_hours_per_nm_by_speed']
    beta = instance['handling_time_deviation_factor_beta_per_meter']
    cost = instance['cost_parameters']
    fuel_price = cost['fuel_price_usd_per_tonne']
    handling_cost = cost['handling_cost_usd_per_hour']
    delay_cost = cost['delay_cost_usd_per_hour']
    waiting_cost = cost['waiting_cost_usd_per_hour']
    lft_penalty = cost['lft_penalty_usd']

    ships = instance['ships']
    external_ships = {}
    for entry in instance['external_ships']:
        pid = entry['port_id']
        external_ships[pid] = entry['ships']

    # Map ship id to ship data
    ship_dict = {s['id']: s for s in ships}

    # Create model
    model = gp.Model("Berth_Speed_Optimization")
    model.Params.TimeLimit = time_limit - 5
    model.Params.Threads = 1
    model.Params.MIPGap = 1e-4
    model.Params.LogToConsole = 0

    # Decision variables
    bigM = 1e6
    eps = 1e-4

    # For each optimized ship
    x = {}      # berth position (left)
    t_start = {}  # berthing start time
    arrival = {}  # arrival time (for first port it's dummy)
    handling = {}
    delay = {}
    lft_viol = {}
    speed_choice = {}  # binary for speed selection per leg
    dev_pos = {}
    dev_neg = {}
    y_left = {}    # binary ordering: ship i left of ship j
    y_before = {}  # binary ordering: ship i finishes before j starts

    # Precompute for external ships
    all_berthings = {}  # (port, vessel_id, is_external, call_idx_if_opt) -> data

    for ship in ships:
        sid = ship['id']
        route = ship['route']
        for pv in ship['port_visits']:
            pidx = pv['port_call_index']
            pid = pv['port_id']
            key = (pid, sid, False, pidx)
            all_berthings[key] = {
                'length': ship['length_meters'],
                'ideal': pv['ideal_berthing_position'],
                'min_h': pv['min_handling_time_hours'],
                'max_h': pv['max_handling_time_hours'],
                'est': pv['earliest_start_time_hours'],
                'eft': pv['expected_finish_time_hours'],
                'lft': pv['latest_finish_time_hours'],
                'is_external': False,
                'ship_id': sid,
                'call_idx': pidx
            }

    for pid, ext_list in external_ships.items():
        for ext in ext_list:
            key = (pid, ext['id'], True, -1)
            all_berthings[key] = {
                'length': ext['length_meters'],
                'berth_pos': ext['berthing_position'],
                'start': ext['start_time_hours'],
                'handle': ext['handling_time_hours'],
                'end': ext['end_time_hours'],
                'is_external': True,
                'ship_id': ext['id'],
                'call_idx': -1
            }

    # Create variables for optimized ships
    for ship in ships:
        sid = ship['id']
        route = ship['route']
        n_calls = len(route)
        for pv in ship['port_visits']:
            pidx = pv['port_call_index']
            pid = pv['port_id']
            key = (pid, sid, False, pidx)
            data = all_berthings[key]
            x[(pid, sid, pidx)] = model.addVar(lb=0.0, ub=ports[pid] - data['length'] + eps, name=f"x_{sid}_{pidx}")
            t_start[(pid, sid, pidx)] = model.addVar(lb=0.0, ub=instance['planning_horizon_hours'] + 100, name=f"tstart_{sid}_{pidx}")
            arrival[(pid, sid, pidx)] = model.addVar(lb=0.0, ub=instance['planning_horizon_hours'] + 100, name=f"arr_{sid}_{pidx}")
            handling[(pid, sid, pidx)] = model.addVar(lb=data['min_h'], ub=data['max_h'], name=f"h_{sid}_{pidx}")
            delay[(pid, sid, pidx)] = model.addVar(lb=0.0, ub=instance['planning_horizon_hours'], name=f"delay_{sid}_{pidx}")
            lft_viol[(pid, sid, pidx)] = model.addVar(lb=0.0, ub=instance['planning_horizon_hours'], name=f"lftv_{sid}_{pidx}")
            dev_pos[(pid, sid, pidx)] = model.addVar(lb=0.0, ub=ports[pid], name=f"devpos_{sid}_{pidx}")
            dev_neg[(pid, sid, pidx)] = model.addVar(lb=0.0, ub=ports[pid], name=f"devneg_{sid}_{pidx}")

        # Speed choices for legs
        for leg in range(1, n_calls):
            from_idx = leg
            to_idx = leg + 1
            for sp_idx, sp in enumerate(speeds):
                speed_choice[(sid, from_idx, to_idx, sp_idx)] = model.addVar(vtype=GRB.BINARY, name=f"speed_{sid}_{from_idx}_{to_idx}_{sp_idx}")

    # Constraints

    # 1. Exactly one speed per leg
    for ship in ships:
        sid = ship['id']
        n_calls = len(ship['route'])
        for leg in range(1, n_calls):
            from_idx = leg
            to_idx = leg + 1
            model.addConstr(gp.quicksum(speed_choice[(sid, from_idx, to_idx, sp_idx)] for sp_idx in range(len(speeds))) == 1)

    # 2. Arrival time linking
    for ship in ships:
        sid = ship['id']
        route = ship['route']
        n_calls = len(route)
        for c in range(1, n_calls):
            pidx_from = c
            pidx_to = c + 1
            pid_from = route[c-1]
            pid_to = route[c]
            dist_key = f"{pid_from}_{pid_to}"
            dist = distances[dist_key]
            tt_sum = gp.LinExpr()
            for sp_idx, sp in enumerate(speeds):
                tt_per_nm = float(travel_times[str(sp)])
                tt_sum += speed_choice[(sid, pidx_from, pidx_to, sp_idx)] * (dist * tt_per_nm)
            model.addConstr(arrival[(pid_to, sid, pidx_to)] == t_start[(pid_from, sid, pidx_from)] + handling[(pid_from, sid, pidx_from)] + tt_sum)

    # 3. First port arrival <= start
    for ship in ships:
        sid = ship['id']
        route = ship['route']
        first_pid = route[0]
        first_pidx = 1
        model.addConstr(arrival[(first_pid, sid, first_pidx)] <= t_start[(first_pid, sid, first_pidx)])

    # 4. Start time >= arrival and >= EST
    for ship in ships:
        sid = ship['id']
        for pv in ship['port_visits']:
            pid = pv['port_id']
            pidx = pv['port_call_index']
            est = pv['earliest_start_time_hours']
            model.addConstr(t_start[(pid, sid, pidx)] >= arrival[(pid, sid, pidx)])
            model.addConstr(t_start[(pid, sid, pidx)] >= est)

    # 5. Handling time and deviation (linearized absolute value)
    for ship in ships:
        sid = ship['id']
        for pv in ship['port_visits']:
            pid = pv['port_id']
            pidx = pv['port_call_index']
            data = all_berthings[(pid, sid, False, pidx)]
            min_h = data['min_h']
            ideal = data['ideal']
            pos_var = x[(pid, sid, pidx)]
            dpos = dev_pos[(pid, sid, pidx)]
            dneg = dev_neg[(pid, sid, pidx)]
            model.addConstr(dpos - dneg == pos_var - ideal)
            model.addConstr(dpos >= 0)
            model.addConstr(dneg >= 0)
            model.addConstr(handling[(pid, sid, pidx)] == min_h * (1.0 + beta * (dpos + dneg)))

    # 6. Delay and LFT violation (linearized)
    for ship in ships:
        sid = ship['id']
        for pv in ship['port_visits']:
            pid = pv['port_id']
            pidx = pv['port_call_index']
            data = all_berthings[(pid, sid, False, pidx)]
            eft = data['eft']
            lft = data['lft']
            end_time = t_start[(pid, sid, pidx)] + handling[(pid, sid, pidx)]
            model.addConstr(delay[(pid, sid, pidx)] >= end_time - eft)
            model.addConstr(delay[(pid, sid, pidx)] >= 0)
            model.addConstr(lft_viol[(pid, sid, pidx)] >= end_time - lft)
            model.addConstr(lft_viol[(pid, sid, pidx)] >= 0)

    # 7. Non-overlap constraints (only between optimized ships and all others)
    port_vessels = {}
    for key, data in all_berthings.items():
        pid = key[0]
        if pid not in port_vessels:
            port_vessels[pid] = []
        port_vessels[pid].append(key)

    for pid, vessel_list in port_vessels.items():
        for v_opt_key in [v for v in vessel_list if not v[2]]:  # optimized only
            sid_opt = v_opt_key[1]
            pidx_opt = v_opt_key[3]
            pos_opt = x[(pid, sid_opt, pidx_opt)]
            len_opt = all_berthings[v_opt_key]['length']
            start_opt = t_start[(pid, sid_opt, pidx_opt)]
            end_opt = start_opt + handling[(pid, sid_opt, pidx_opt)]

            for v_other_key in vessel_list:
                if v_other_key == v_opt_key:
                    continue
                is_ext = v_other_key[2]
                if is_ext:
                    pos_other = all_berthings[v_other_key]['berth_pos']
                    len_other = all_berthings[v_other_key]['length']
                    start_other = all_berthings[v_other_key]['start']
                    end_other = all_berthings[v_other_key]['end']
                else:
                    sid_other = v_other_key[1]
                    pidx_other = v_other_key[3]
                    pos_other = x[(pid, sid_other, pidx_other)]
                    len_other = all_berthings[v_other_key]['length']
                    start_other = t_start[(pid, sid_other, pidx_other)]
                    end_other = start_other + handling[(pid, sid_other, pidx_other)]

                # Binary ordering variables
                left_bin = model.addVar(vtype=GRB.BINARY, name=f"left_{pid}_{sid_opt}_{pidx_opt}_{v_other_key[1]}_{v_other_key[3]}")
                before_bin = model.addVar(vtype=GRB.BINARY, name=f"before_{pid}_{sid_opt}_{pidx_opt}_{v_other_key[1]}_{v_other_key[3]}")

                M_pos = ports[pid] + 1000
                M_time = instance['planning_horizon_hours'] * 2 + 1000

                # Non-overlap: either one is completely left of the other OR one finishes before the other starts
                model.addConstr(pos_opt + len_opt <= pos_other + M_pos * (1 - left_bin))
                model.addConstr(pos_other + len_other <= pos_opt + M_pos * left_bin)
                model.addConstr(end_opt <= start_other + M_time * (1 - before_bin))
                model.addConstr(end_other <= start_opt + M_time * before_bin)

                # At least one of the two conditions must hold
                model.addConstr(left_bin + before_bin >= 1)

    # Objective
    total_cost = gp.LinExpr()

    # Waiting, handling, delay, penalty for optimized ships
    for ship in ships:
        sid = ship['id']
        for pv in ship['port_visits']:
            pid = pv['port_id']
            pidx = pv['port_call_index']
            wait_time = t_start[(pid, sid, pidx)] - arrival[(pid, sid, pidx)]
            total_cost += waiting_cost * wait_time
            total_cost += handling_cost * handling[(pid, sid, pidx)]
            total_cost += delay_cost * delay[(pid, sid, pidx)]
            total_cost += lft_penalty * lft_viol[(pid, sid, pidx)]

    # Fuel cost
    for ship in ships:
        sid = ship['id']
        route = ship['route']
        n_calls = len(route)
        fuel_map = ship.get('fuel_consumption_per_nm_by_speed', {})
        for leg in range(1, n_calls):
            from_p = route[leg-1]
            to_p = route[leg]
            dist_key = f"{from_p}_{to_p}"
            dist = distances[dist_key]
            for sp_idx, sp_knots in enumerate(speeds):
                fuel_per_nm = fuel_map.get(str(sp_knots), fuel_map.get(sp_knots))
                total_cost += fuel_price * fuel_per_nm * dist * speed_choice[(sid, leg, leg+1, sp_idx)]

    model.setObjective(total_cost, GRB.MINIMIZE)

    # Callback for incumbent logging
    best_obj = float('inf')
    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj < best_obj - 1e-3:
                best_obj = obj
                if logger:
                    logger.log(float(obj))
        if time.time() - start_time > time_limit - 2:
            model.terminate()

    # Optimize
    model.optimize(callback)

    # Extract solution
    if model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.SUBOPTIMAL]:
        solution = {
            "objective_value": 1e10,
            "ships": []
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        return

    # Build output
    output_ships = []
    for ship in ships:
        sid = ship['id']
        ship_out = {
            "ship_id": sid,
            "port_visits": [],
            "speed_selections": []
        }
        for pv in ship['port_visits']:
            pid = pv['port_id']
            pidx = pv['port_call_index']
            pos_val = x[(pid, sid, pidx)].X
            start_val = t_start[(pid, sid, pidx)].X
            h_val = handling[(pid, sid, pidx)].X
            arr_val = arrival[(pid, sid, pidx)].X
            end_val = start_val + h_val
            eft = pv['expected_finish_time_hours']
            lft = pv['latest_finish_time_hours']
            delay_val = max(0.0, end_val - eft)
            lft_val = max(0.0, end_val - lft)
            dev_val = abs(pos_val - pv['ideal_berthing_position'])

            pv_out = {
                "port_call_index": pidx,
                "port_id": pid,
                "berthing_position": round(pos_val, 6),
                "start_time": round(start_val, 6),
                "handling_time": round(h_val, 6),
                "arrival_time": round(arr_val, 6),
                "delay": round(delay_val, 6),
                "lft_violation": round(lft_val, 6),
                "position_deviation": round(dev_val, 6)
            }
            ship_out["port_visits"].append(pv_out)

        # speeds
        n_calls = len(ship['route'])
        for leg in range(1, n_calls):
            from_idx = leg
            to_idx = leg + 1
            chosen_sp = None
            for sp_idx, sp in enumerate(speeds):
                if speed_choice[(sid, from_idx, to_idx, sp_idx)].X > 0.5:
                    chosen_sp = float(sp)
                    break
            ship_out["speed_selections"].append({
                "from_call": from_idx,
                "to_call": to_idx,
                "speed_knots": chosen_sp
            })

        output_ships.append(ship_out)

    final_obj = model.objVal if model.objVal < 1e9 else best_obj
    solution = {
        "objective_value": round(float(final_obj), 6),
        "ships": output_ships
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()