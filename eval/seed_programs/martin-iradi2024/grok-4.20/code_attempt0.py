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
    model.Params.TimeLimit = time_limit - 5  # reserve some margin
    model.Params.Threads = 1
    model.Params.MIPGap = 0.01
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
            x[(pid, sid, pidx)] = model.addVar(lb=0.0, ub=ports[pid] - data['length'], name=f"x_{sid}_{pidx}")
            t_start[(pid, sid, pidx)] = model.addVar(lb=0.0, ub=instance['planning_horizon_hours'], name=f"tstart_{sid}_{pidx}")
            arrival[(pid, sid, pidx)] = model.addVar(lb=0.0, ub=instance['planning_horizon_hours'], name=f"arr_{sid}_{pidx}")
            handling[(pid, sid, pidx)] = model.addVar(lb=0.0, name=f"h_{sid}_{pidx}")
            delay[(pid, sid, pidx)] = model.addVar(lb=0.0, name=f"delay_{sid}_{pidx}")
            lft_viol[(pid, sid, pidx)] = model.addVar(lb=0.0, name=f"lftv_{sid}_{pidx}")

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
                tt_sum += speed_choice[(sid, pidx_from, pidx_to, sp_idx)] * dist * tt_per_nm
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

    # 5. Handling time
    for ship in ships:
        sid = ship['id']
        for pv in ship['port_visits']:
            pid = pv['port_id']
            pidx = pv['port_call_index']
            data = all_berthings[(pid, sid, False, pidx)]
            min_h = data['min_h']
            ideal = data['ideal']
            max_possible_h = data['max_h']
            dev = model.addVar(lb=0, name=f"dev_{sid}_{pidx}")
            model.addConstr(dev == gp.abs_(x[(pid, sid, pidx)] - ideal))
            model.addConstr(handling[(pid, sid, pidx)] == min_h * (1 + beta * dev))
            # Bound handling time
            model.addConstr(handling[(pid, sid, pidx)] <= max_possible_h)

    # 6. Delay and LFT violation
    for ship in ships:
        sid = ship['id']
        for pv in ship['port_visits']:
            pid = pv['port_id']
            pidx = pv['port_call_index']
            data = all_berthings[(pid, sid, False, pidx)]
            eft = data['eft']
            lft = data['lft']
            end_time = t_start[(pid, sid, pidx)] + handling[(pid, sid, pidx)]
            model.addConstr(delay[(pid, sid, pidx)] == gp.max_(0, end_time - eft))
            model.addConstr(lft_viol[(pid, sid, pidx)] == gp.max_(0, end_time - lft))

    # 7. Non-overlap between all pairs at same port (including externals)
    port_vessels = {}
    for key, data in all_berthings.items():
        pid = key[0]
        if pid not in port_vessels:
            port_vessels[pid] = []
        port_vessels[pid].append(key)

    for pid, vessel_list in port_vessels.items():
        for v1, v2 in combinations(vessel_list, 2):
            is_ext1 = v1[2]
            is_ext2 = v2[2]
            if is_ext1 and is_ext2:
                continue  # both external -> already fixed, assume input is feasible

            # Define intervals and positions
            if not is_ext1:
                sid1 = v1[1]
                pidx1 = v1[3]
                pos1 = x[(pid, sid1, pidx1)]
                len1 = all_berthings[v1]['length']
                start1 = t_start[(pid, sid1, pidx1)]
                end1 = start1 + handling[(pid, sid1, pidx1)]
            else:
                sid1 = v1[1]
                pos1 = all_berthings[v1]['berth_pos']
                len1 = all_berthings[v1]['length']
                start1 = all_berthings[v1]['start']
                end1 = all_berthings[v1]['end']

            if not is_ext2:
                sid2 = v2[1]
                pidx2 = v2[3]
                pos2 = x[(pid, sid2, pidx2)]
                len2 = all_berthings[v2]['length']
                start2 = t_start[(pid, sid2, pidx2)]
                end2 = start2 + handling[(pid, sid2, pidx2)]
            else:
                sid2 = v2[1]
                pos2 = all_berthings[v2]['berth_pos']
                len2 = all_berthings[v2]['length']
                start2 = all_berthings[v2]['start']
                end2 = all_berthings[v2]['end']

            # Binary variables for ordering
            left_ij = model.addVar(vtype=GRB.BINARY, name=f"left_{pid}_{v1[1]}_{v1[3]}_{v2[1]}_{v2[3]}")
            before_ij = model.addVar(vtype=GRB.BINARY, name=f"before_{pid}_{v1[1]}_{v1[3]}_{v2[1]}_{v2[3]}")

            # No spatial overlap if not before in time
            # pos1 + len1 <= pos2 + M*(left_ij + before_ij)
            model.addConstr(pos1 + len1 <= pos2 + bigM * (left_ij + before_ij))
            # pos2 + len2 <= pos1 + M*(1-left_ij + before_ij)
            model.addConstr(pos2 + len2 <= pos1 + bigM * ((1 - left_ij) + before_ij))

            # No time overlap if not separated spatially
            # end1 <= start2 + M*( (1-left_ij) + (1-before_ij) + before_ji ) wait, better use two binaries
            # We use before_ij = 1 if i finishes before j starts
            before_ji = model.addVar(vtype=GRB.BINARY, name=f"beforeji_{pid}_{v1[1]}_{v1[3]}_{v2[1]}_{v2[3]}")
            model.addConstr(before_ij + before_ji >= 1 - left_ij - (1 - left_ij))  # at least one ordering in time or space
            # Actually simpler: use bigM on time constraints with ordering binaries

            model.addConstr(end1 <= start2 + bigM * (1 - before_ij))
            model.addConstr(end2 <= start1 + bigM * (1 - before_ji))

            # Link before_ji
            model.addConstr(before_ij + before_ji >= 1)  # at least one must finish before the other starts (if they overlap in space they must be ordered in time)

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
        fuel_map = ship['fuel_consumption_per_nm_by_speed']
        for leg in range(1, n_calls):
            from_p = route[leg-1]
            to_p = route[leg]
            dist = distances[f"{from_p}_{to_p}"]
            for sp_idx, sp_knots in enumerate(speeds):
                fuel_per_nm = fuel_map[str(sp_knots)]
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

    # Optimize
    model.optimize(callback)

    # Extract solution
    if model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED]:
        # If infeasible, return a dummy feasible if possible or just the schema with high cost
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