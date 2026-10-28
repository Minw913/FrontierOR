import json
import argparse
import time
import sys
from gurobipy import *


def extract_solution_from_model(model, instance, ships, speeds, x_pos, t_st, h_var,
                                arr_var, delay_var, over_var, dev_var, spd_var, num_speeds):
    """Extract solution using .X attribute (after optimization)."""
    solution = {"ships": []}

    for ship in ships:
        sid = ship['id']
        route = ship['route']
        ship_sol = {"ship_id": sid, "port_visits": [], "speed_selections": []}

        for pv in ship['port_visits']:
            idx = pv['port_call_index']
            key = (sid, idx)

            pos_val = x_pos[key].X
            start_val = t_st[key].X
            h_val = h_var[key].X
            arr_val = arr_var[key].X
            delay_val = max(0.0, delay_var[key].X)
            over_val = max(0.0, over_var[key].X)
            dev_val = max(0.0, dev_var[key].X)

            pv_sol = {
                "port_call_index": idx,
                "port_id": pv['port_id'],
                "berthing_position": round(pos_val, 6),
                "start_time": round(start_val, 6),
                "handling_time": round(h_val, 6),
                "arrival_time": round(arr_val, 6),
                "delay": round(delay_val, 6),
                "lft_violation": round(over_val, 6),
                "position_deviation": round(dev_val, 6)
            }
            ship_sol["port_visits"].append(pv_sol)

        num_legs = len(route) - 1
        for leg in range(num_legs):
            chosen_speed = speeds[0]
            for k in range(num_speeds):
                if spd_var[(sid, leg, k)].X > 0.5:
                    chosen_speed = speeds[k]
                    break

            ship_sol["speed_selections"].append({
                "from_call": leg + 1,
                "to_call": leg + 2,
                "speed_knots": chosen_speed
            })

        solution["ships"].append(ship_sol)

    solution["objective_value"] = round(model.ObjVal, 6)
    return solution


def extract_solution_from_callback(model, instance, ships, speeds, x_pos, t_st, h_var,
                                   arr_var, delay_var, over_var, dev_var, spd_var, num_speeds):
    """Extract solution using cbGetSolution (inside callback)."""
    solution = {"ships": []}

    for ship in ships:
        sid = ship['id']
        route = ship['route']
        ship_sol = {"ship_id": sid, "port_visits": [], "speed_selections": []}

        for pv in ship['port_visits']:
            idx = pv['port_call_index']
            key = (sid, idx)

            pos_val = model.cbGetSolution(x_pos[key])
            start_val = model.cbGetSolution(t_st[key])
            h_val = model.cbGetSolution(h_var[key])
            arr_val = model.cbGetSolution(arr_var[key])
            delay_val = max(0.0, model.cbGetSolution(delay_var[key]))
            over_val = max(0.0, model.cbGetSolution(over_var[key]))
            dev_val = max(0.0, model.cbGetSolution(dev_var[key]))

            pv_sol = {
                "port_call_index": idx,
                "port_id": pv['port_id'],
                "berthing_position": round(pos_val, 6),
                "start_time": round(start_val, 6),
                "handling_time": round(h_val, 6),
                "arrival_time": round(arr_val, 6),
                "delay": round(delay_val, 6),
                "lft_violation": round(over_val, 6),
                "position_deviation": round(dev_val, 6)
            }
            ship_sol["port_visits"].append(pv_sol)

        num_legs = len(route) - 1
        for leg in range(num_legs):
            chosen_speed = speeds[0]
            for k in range(num_speeds):
                if model.cbGetSolution(spd_var[(sid, leg, k)]) > 0.5:
                    chosen_speed = speeds[k]
                    break

            ship_sol["speed_selections"].append({
                "from_call": leg + 1,
                "to_call": leg + 2,
                "speed_knots": chosen_speed
            })

        solution["ships"].append(ship_sol)

    obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
    solution["objective_value"] = round(obj_val, 6)
    return solution


def main():
    parser = argparse.ArgumentParser(description="Berth Allocation and Speed Optimization")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    # Initialize logger
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Parse instance data
    planning_horizon = instance['planning_horizon_hours']
    cost_params = instance['cost_parameters']
    fuel_price = cost_params['fuel_price_usd_per_tonne']
    handling_cost_rate = cost_params['handling_cost_usd_per_hour']
    delay_cost_rate = cost_params['delay_cost_usd_per_hour']
    waiting_cost_rate = cost_params['waiting_cost_usd_per_hour']
    lft_penalty_rate = cost_params['lft_penalty_usd_per_hour']
    beta = instance['handling_time_deviation_factor_beta_per_meter']

    ports = {p['id']: p for p in instance['ports']}
    distances = instance['distances_nautical_miles']
    speeds = instance['speeds_knots']
    num_speeds = len(speeds)

    # Travel time per nm by speed index
    tt_per_nm = []
    for spd in speeds:
        key = str(spd)
        tt_per_nm.append(instance['travel_time_hours_per_nm_by_speed'][key])

    ships = instance['ships']

    # External ships by port
    external_ships_by_port = {}
    for es_group in instance.get('external_ships', []):
        pid = es_group['port_id']
        if pid not in external_ships_by_port:
            external_ships_by_port[pid] = []
        external_ships_by_port[pid].extend(es_group['ships'])

    # Build Gurobi model
    model = Model("BASP")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', 0)

    # Decision variables
    x_pos = {}
    t_st = {}
    dev_var = {}
    h_var = {}
    arr_var = {}
    wait_var = {}
    delay_var = {}
    over_var = {}
    spd_var = {}

    for ship in ships:
        sid = ship['id']
        ship_len = ship['length_meters']

        for pv in ship['port_visits']:
            idx = pv['port_call_index']
            pid = pv['port_id']
            key = (sid, idx)
            quay_len = ports[pid]['quay_length_meters']

            x_pos[key] = model.addVar(lb=0, ub=quay_len - ship_len, name=f"x_{sid}_{idx}")
            t_st[key] = model.addVar(lb=0, ub=planning_horizon, name=f"t_{sid}_{idx}")
            dev_var[key] = model.addVar(lb=0, name=f"dev_{sid}_{idx}")
            h_var[key] = model.addVar(lb=0, name=f"h_{sid}_{idx}")
            arr_var[key] = model.addVar(lb=0, ub=planning_horizon, name=f"arr_{sid}_{idx}")
            wait_var[key] = model.addVar(lb=0, name=f"wait_{sid}_{idx}")
            delay_var[key] = model.addVar(lb=0, name=f"delay_{sid}_{idx}")
            over_var[key] = model.addVar(lb=0, name=f"over_{sid}_{idx}")

        num_legs = len(ship['route']) - 1
        for leg in range(num_legs):
            for k in range(num_speeds):
                spd_var[(sid, leg, k)] = model.addVar(vtype=GRB.BINARY, name=f"s_{sid}_{leg}_{k}")

    model.update()

    # Constraints for each ship
    for ship in ships:
        sid = ship['id']
        route = ship['route']

        for pv in ship['port_visits']:
            idx = pv['port_call_index']
            key = (sid, idx)
            ideal = pv['ideal_berthing_position']
            min_h = pv['min_handling_time_hours']
            max_h = pv['max_handling_time_hours']
            est = pv['earliest_start_time_hours']
            eft = pv['expected_finish_time_hours']
            lft = pv['latest_finish_time_hours']

            # Deviation from ideal position (absolute value linearization)
            model.addConstr(dev_var[key] >= x_pos[key] - ideal, name=f"devp_{sid}_{idx}")
            model.addConstr(dev_var[key] >= ideal - x_pos[key], name=f"devm_{sid}_{idx}")

            # Handling time: h = min_h * (1 + beta * dev)
            model.addConstr(h_var[key] == min_h + min_h * beta * dev_var[key], name=f"ht_{sid}_{idx}")

            # Upper bound on handling time
            model.addConstr(h_var[key] <= max_h, name=f"htmax_{sid}_{idx}")

            # Earliest start time
            model.addConstr(t_st[key] >= est, name=f"est_{sid}_{idx}")

            # Waiting time = start - arrival
            model.addConstr(wait_var[key] == t_st[key] - arr_var[key], name=f"wait_{sid}_{idx}")

            # Delay = max(0, end - EFT)
            model.addConstr(delay_var[key] >= t_st[key] + h_var[key] - eft, name=f"del_{sid}_{idx}")

            # LFT overshoot = max(0, end - LFT)
            model.addConstr(over_var[key] >= t_st[key] + h_var[key] - lft, name=f"lft_{sid}_{idx}")

        # Exactly one speed per leg
        num_legs = len(route) - 1
        for leg in range(num_legs):
            model.addConstr(
                quicksum(spd_var[(sid, leg, k)] for k in range(num_speeds)) == 1,
                name=f"spdsel_{sid}_{leg}"
            )

        # Arrival time constraints
        for pv in ship['port_visits']:
            idx = pv['port_call_index']
            key = (sid, idx)

            if idx == 1:
                # First port: no preceding voyage, set arrival = start (zero waiting)
                model.addConstr(arr_var[key] == t_st[key], name=f"arr1_{sid}_{idx}")
            else:
                # Arrival = prev_start + prev_handling + travel_time
                prev_key = (sid, idx - 1)
                prev_pid = route[idx - 2]
                curr_pid = route[idx - 1]
                dist_key = f"{prev_pid}_{curr_pid}"
                dist = distances[dist_key]

                travel_expr = quicksum(
                    spd_var[(sid, idx - 2, k)] * tt_per_nm[k] * dist
                    for k in range(num_speeds)
                )
                model.addConstr(
                    arr_var[key] == t_st[prev_key] + h_var[prev_key] + travel_expr,
                    name=f"arr_{sid}_{idx}"
                )

    # Non-overlap constraints
    # Collect all ships (optimized + external) at each port
    port_visitors = {}

    for ship in ships:
        sid = ship['id']
        for pv in ship['port_visits']:
            pid = pv['port_id']
            if pid not in port_visitors:
                port_visitors[pid] = []
            port_visitors[pid].append(('opt', sid, pv['port_call_index']))

    for pid, ext_list in external_ships_by_port.items():
        if pid not in port_visitors:
            port_visitors[pid] = []
        for es in ext_list:
            port_visitors[pid].append(('ext', es))

    # Ship lookup for quick access
    ship_by_id = {s['id']: s for s in ships}

    for pid, visitors in port_visitors.items():
        n = len(visitors)
        quay_len = ports[pid]['quay_length_meters']

        for i in range(n):
            for j in range(i + 1, n):
                vi = visitors[i]
                vj = visitors[j]

                # Get attributes for ship i
                if vi[0] == 'opt':
                    sid_i, idx_i = vi[1], vi[2]
                    ship_i = ship_by_id[sid_i]
                    pos_i = x_pos[(sid_i, idx_i)]
                    len_i = ship_i['length_meters']
                    st_i = t_st[(sid_i, idx_i)]
                    ht_i = h_var[(sid_i, idx_i)]
                else:
                    ext_i = vi[1]
                    pos_i = ext_i['berthing_position']
                    len_i = ext_i['length_meters']
                    st_i = ext_i['start_time_hours']
                    ht_i = ext_i['handling_time_hours']

                # Get attributes for ship j
                if vj[0] == 'opt':
                    sid_j, idx_j = vj[1], vj[2]
                    ship_j = ship_by_id[sid_j]
                    pos_j = x_pos[(sid_j, idx_j)]
                    len_j = ship_j['length_meters']
                    st_j = t_st[(sid_j, idx_j)]
                    ht_j = h_var[(sid_j, idx_j)]
                else:
                    ext_j = vj[1]
                    pos_j = ext_j['berthing_position']
                    len_j = ext_j['length_meters']
                    st_j = ext_j['start_time_hours']
                    ht_j = ext_j['handling_time_hours']

                # For two external ships, check if constraint is trivially satisfied
                if vi[0] == 'ext' and vj[0] == 'ext':
                    # Check if any of the 4 conditions holds
                    if (pos_i + len_i <= pos_j + 1e-6 or
                        pos_j + len_j <= pos_i + 1e-6 or
                        st_i + ht_i <= st_j + 1e-6 or
                        st_j + ht_j <= st_i + 1e-6):
                        continue  # Trivially satisfied, skip

                # 4 binary variables for disjunction
                b1 = model.addVar(vtype=GRB.BINARY, name=f"b1_{pid}_{i}_{j}")
                b2 = model.addVar(vtype=GRB.BINARY, name=f"b2_{pid}_{i}_{j}")
                b3 = model.addVar(vtype=GRB.BINARY, name=f"b3_{pid}_{i}_{j}")
                b4 = model.addVar(vtype=GRB.BINARY, name=f"b4_{pid}_{i}_{j}")

                model.addConstr(b1 + b2 + b3 + b4 >= 1, name=f"disj_{pid}_{i}_{j}")

                # i is entirely left of j
                model.addConstr(pos_i + len_i <= pos_j + quay_len * (1 - b1),
                               name=f"left_{pid}_{i}_{j}")
                # j is entirely left of i
                model.addConstr(pos_j + len_j <= pos_i + quay_len * (1 - b2),
                               name=f"right_{pid}_{i}_{j}")
                # i finishes before j starts
                model.addConstr(st_i + ht_i <= st_j + planning_horizon * (1 - b3),
                               name=f"before_{pid}_{i}_{j}")
                # j finishes before i starts
                model.addConstr(st_j + ht_j <= st_i + planning_horizon * (1 - b4),
                               name=f"after_{pid}_{i}_{j}")

    model.update()

    # Objective function
    obj = LinExpr()

    for ship in ships:
        sid = ship['id']
        route = ship['route']

        for pv in ship['port_visits']:
            idx = pv['port_call_index']
            key = (sid, idx)

            obj += waiting_cost_rate * wait_var[key]
            obj += handling_cost_rate * h_var[key]
            obj += delay_cost_rate * delay_var[key]
            obj += lft_penalty_rate * over_var[key]

        # Fuel cost for each leg
        num_legs = len(route) - 1
        for leg in range(num_legs):
            prev_pid = route[leg]
            curr_pid = route[leg + 1]
            dist_key = f"{prev_pid}_{curr_pid}"
            dist = distances[dist_key]

            fuel_by_speed = ship['fuel_consumption_per_nm_by_speed']
            for k in range(num_speeds):
                spd_key = str(speeds[k])
                fuel_per_nm = fuel_by_speed[spd_key]
                obj += fuel_price * fuel_per_nm * dist * spd_var[(sid, leg, k)]

    model.setObjective(obj, GRB.MINIMIZE)

    # Callback for logging incumbents
    best_logged_obj = float('inf')

    def callback(model, where):
        nonlocal best_logged_obj
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_logged_obj - 1e-6:
                best_logged_obj = obj_val
                try:
                    sol = extract_solution_from_callback(
                        model, instance, ships, speeds, x_pos, t_st, h_var,
                        arr_var, delay_var, over_var, dev_var, spd_var, num_speeds
                    )
                    if logger:
                        logger.log_solution(obj_val, sol)
                except Exception:
                    if logger:
                        logger.log(obj_val)

    # Optimize
    model.optimize(callback)

    # Extract final solution
    if model.SolCount > 0:
        solution = extract_solution_from_model(
            model, instance, ships, speeds, x_pos, t_st, h_var,
            arr_var, delay_var, over_var, dev_var, spd_var, num_speeds
        )
        # Log final solution if not already logged
        if logger and solution["objective_value"] < best_logged_obj - 1e-6:
            logger.log_solution(solution["objective_value"], solution)
    else:
        # No feasible solution found; output a trivial solution
        solution = {"objective_value": 1e12, "ships": []}
        for ship in ships:
            sid = ship['id']
            route = ship['route']
            ship_sol = {"ship_id": sid, "port_visits": [], "speed_selections": []}
            for pv in ship['port_visits']:
                idx = pv['port_call_index']
                ship_sol["port_visits"].append({
                    "port_call_index": idx,
                    "port_id": pv['port_id'],
                    "berthing_position": pv['ideal_berthing_position'],
                    "start_time": pv['earliest_start_time_hours'],
                    "handling_time": pv['min_handling_time_hours'],
                    "arrival_time": pv['earliest_start_time_hours'],
                    "delay": 0.0,
                    "lft_violation": 0.0,
                    "position_deviation": 0.0
                })
            num_legs = len(route) - 1
            for leg in range(num_legs):
                ship_sol["speed_selections"].append({
                    "from_call": leg + 1,
                    "to_call": leg + 2,
                    "speed_knots": speeds[0]
                })
            solution["ships"].append(ship_sol)

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    print(f"Objective value: {solution['objective_value']}")
    print(f"Solution written to {args.solution_path}")


if __name__ == "__main__":
    main()