import argparse
import json
import math
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB


def solve_shipping_optimization(instance_path, solution_path, time_limit, log_path):
    # Read instance data
    with open(instance_path, 'r') as f:
        data = json.load(f)

    # Extract parameters
    planning_horizon = data['planning_horizon_hours']
    fuel_price = data['cost_parameters']['fuel_price_usd_per_tonne']
    handling_cost = data['cost_parameters']['handling_cost_usd_per_hour']
    delay_cost = data['cost_parameters']['delay_cost_usd_per_hour']
    waiting_cost = data['cost_parameters']['waiting_cost_usd_per_hour']
    lft_penalty = data['cost_parameters']['lft_penalty_usd']
    beta = data['handling_time_deviation_factor_beta_per_meter']

    ports = {p['id']: p for p in data['ports']}
    distances = data['distances_nautical_miles']
    speeds = data['speeds_knots']
    travel_times = data['travel_time_hours_per_nm_by_speed']  # hours per nautical mile
    ships_data = {s['id']: s for s in data['ships']}
    
    # Process external ships  
    external_ships_by_port = defaultdict(list)
    for ext_port_info in data['external_ships']:
        port_id = ext_port_info['port_id']
        for ext_ship in ext_port_info['ships']:
            external_ships_by_port[port_id].append(ext_ship)

    # Model initialization
    model = gp.Model('ShippingOptimization')
    model.setParam("Threads", 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions

    # Decision Variables
    
    # For each ship and port visit, we have:
    # 1. berthing position (continuous)
    # 2. start time (continuous)
    # 3. speed selection (binary indicator for each leg)
    
    ship_vars = {}
    
    for ship_id, ship in ships_data.items():
        ship_vars[ship_id] = {'port_visits': {}, 'speed_selections': []}
        
        # Port visit variables
        for idx, pv in enumerate(ship['port_visits']):
            var_dict = {}
            
            # Position variable: continuous
            pos_lb = 0
            pos_ub = ports[pv['port_id']]['quay_length_meters'] - ship['length_meters']
            if pos_ub < 0:
                raise ValueError(f"Ship {ship_id} is longer ({ship['length_meters']}m) than available quay at port {pv['port_id']} ({ports[pv['port_id']]['quay_length_meters']}m)")
                
            pos_var = model.addVar(
                lb=pos_lb, 
                ub=pos_ub,
                name=f"ship_{ship_id}_pos_{pv['port_call_index']}_{pv['port_id']}",
                vtype=GRB.CONTINUOUS
            )
            var_dict['position'] = pos_var
            
            time_lb = pv['earliest_start_time_hours']
            time_ub = planning_horizon
            time_var = model.addVar(
                lb=time_lb,
                ub=time_ub,
                name=f"ship_{ship_id}_time_{pv['port_call_index']}_{pv['port_id']}",
                vtype=GRB.CONTINUOUS
            )
            var_dict['start_time'] = time_var
            
            ship_vars[ship_id]['port_visits'][idx] = var_dict
        
        # Speed variables for inter-port legs (between consecutive port visits)
        for leg_idx in range(len(ship['port_visits']) - 1):
            speed_vars = []
            for j, speed in enumerate(speeds):
                speed_var_name = f"ship_{ship_id}_speed_leg_{leg_idx}_at_{speed}"
                speed_var = model.addVar(
                    vtype=GRB.BINARY,
                    name=speed_var_name
                )
                speed_vars.append(speed_var)
            # For this leg, store the available speed binary variables
            ship_vars[ship_id]['speed_selections'].append(speed_vars)
    
    # Precompute variables for handling time, delays, etc.
    computed_vars = {}
    
    for ship_id, ship in ships_data.items():
        computed_vars[ship_id] = {'port_visits': {}}
        
        for idx, pv in enumerate(ship['port_visits']):
            vars_dict = ship_vars[ship_id]['port_visits'][idx]
            
            # Position deviation absolute value (linearization)
            # pos_deviation_abs >= position - ideal
            # pos_deviation_abs >= -(position - ideal)
            pos_dev_abs = model.addVar(
                name=f"ship_{ship_id}_pos_dev_abs_{pv['port_call_index']}", 
                vtype=GRB.CONTINUOUS, 
                lb=0
            )
            # Linearization constraints
            model.addConstr(
                pos_dev_abs >= vars_dict['position'] - pv['ideal_berthing_position'],
                name=f"pos_dev_geq_{ship_id}_{pv['port_call_index']}_a"
            )
            model.addConstr(
                pos_dev_abs >= pv['ideal_berthing_position'] - vars_dict['position'],
                name=f"pos_dev_geq_{ship_id}_{pv['port_call_index']}_b"
            )
            
            # Handling time = min_handling * (1 + beta * |position - ideal|)
            handling_time = model.addVar(
                name=f"ship_{ship_id}_handling_time_{pv['port_call_index']}", 
                vtype=GRB.CONTINUOUS
            )
            model.addConstr(
                handling_time == pv['min_handling_time_hours'] * (1 + beta * pos_dev_abs),
                name=f"handling_time_def_{ship_id}_{pv['port_call_index']}"
            )
            
            # End time = start_time + handling_time
            end_time = model.addVar(
                name=f"ship_{ship_id}_end_time_{pv['port_call_index']}", 
                vtype=GRB.CONTINUOUS, 
                ub=planning_horizon
            )
            model.addConstr(
                end_time == vars_dict['start_time'] + handling_time,
                name=f"end_time_definition_{ship_id}_{pv['port_call_index']}"
            )
            
            # Arrival time and travel time calculation
            arrival_time = model.addVar(
                name=f"ship_{ship_id}_arrival_{pv['port_call_index']}", 
                vtype=GRB.CONTINUOUS, 
                ub=planning_horizon
            )
            
            if idx == 0:  # For first port visit: arrival_time <= start_time
                model.addConstr(
                    arrival_time <= vars_dict['start_time'],  # Ship arrives before it can start
                    name=f"first_arrival_constr_{ship_id}_{pv['port_call_index']}"
                )
            else:  # Has prior arrival (after previous handling and travel)
                prev_visit_idx = idx - 1
                prev_end_time = computed_vars[ship_id]['port_visits'][prev_visit_idx]['end_time']
                leg_idx = idx - 1
                speed_vars_for_leg = ship_vars[ship_id]['speed_selections'][leg_idx]
                
                # Origin and destination port IDs for this leg
                origin_port_id = ship['route'][prev_visit_idx]
                dest_port_id = ship['route'][idx]
                dist_nm = distances.get(f"{origin_port_id}_{dest_port_id}") 
                if dist_nm is None:
                    dist_nm = distances.get(f"{dest_port_id}_{origin_port_id}")
                    if dist_nm is None:
                        raise ValueError(f"No distance defined between ports {origin_port_id} and {dest_port_id}")
                        
                # Travel time depends on selected speed
                travel_time_total = model.addVar(vtype=GRB.CONTINUOUS, name=f"ship_{ship_id}_leg_{leg_idx}_travel_time")
                
                # Calculate travel duration for each possible speed
                travel_durations = []
                for k, speed in enumerate(speeds):
                    speed_str = str(speed)
                    dur = dist_nm * float(travel_times[speed_str])
                    travel_durations.append(dur)
                
                # Link travel time to selected speed
                for k in range(len(speeds)):
                    model.addConstr(
                        travel_time_total >= travel_durations[k] - (1 - speed_vars_for_leg[k]) * planning_horizon,
                        name=f"travel_time_link_min_{ship_id}_leg_{leg_idx}_speed_{k}"
                    )
                    model.addConstr(
                        travel_time_total <= travel_durations[k] + (1 - speed_vars_for_leg[k]) * planning_horizon,
                        name=f"travel_time_link_max_{ship_id}_leg_{leg_idx}_speed_{k}"
                    )
                
                # Total arrival time = previous end_time + travel_time
                model.addConstr(
                    arrival_time == prev_end_time + travel_time_total,
                    name=f"arrival_from_prev_{ship_id}_visit_{idx}"
                )
            
            # Waiting time = max(0, start - arrival)
            waiting_time = model.addVar(
                name=f"ship_{ship_id}_waiting_time_{pv['port_call_index']}", 
                vtype=GRB.CONTINUOUS, 
                lb=0
            )
            model.addConstr(
                waiting_time >= vars_dict['start_time'] - arrival_time,
                name=f"waiting_time_definition_{ship_id}_{pv['port_call_index']}_a"
            )
            # Note: For linear relaxation, we assume we'll get a good result with just one-sided inequality
            
            # Delay = max(0, end_time - expected_finish_time)
            delay_time = model.addVar(
                name=f"ship_{ship_id}_delay_time_{pv['port_call_index']}", 
                vtype=GRB.CONTINUOUS, 
                lb=0
            )
            expected_finish = pv['expected_finish_time_hours']
            model.addConstr(
                delay_time >= end_time - expected_finish,
                name=f"delay_definition_{ship_id}_{pv['port_call_index']}_a"
            )
            
            # LFT violation = max(0, end - latest_finish)
            lft_violation = model.addVar(
                name=f"ship_{ship_id}_lft_violation_{pv['port_call_index']}", 
                vtype=GRB.CONTINUOUS, 
                lb=0
            )
            latest_finish = pv['latest_finish_time_hours']
            model.addConstr(
                lft_violation >= end_time - latest_finish,
                name=f"lft_violation_definition_{ship_id}_{pv['port_call_index']}_a"
            )
            
            # Store computed variables
            computed_vars[ship_id]['port_visits'][idx] = {
                'handling_time': handling_time,
                'end_time': end_time,
                'arrival_time': arrival_time,
                'waiting_time': waiting_time,
                'delay_time': delay_time,
                'lft_violation': lft_violation,
                'pos_dev_abs': pos_dev_abs
            }

    # Constraint: exactly one speed per leg for all ships
    for ship_id in ships_data.keys():
        ship = ships_data[ship_id]
        for leg_idx in range(len(ship['port_visits']) - 1):
            speed_vars = ship_vars[ship_id]['speed_selections'][leg_idx]
            model.addConstr(
                gp.quicksum(speed_vars) == 1,
                name=f"exactly_one_speed_ship_{ship_id}_leg_{leg_idx}"
            )

    # Constraints for non-overlapping ships at same port
    # 1. Between optimized ships
    # 2. Between optimized and external ships
    
    for port_id in ports.keys():
        # Gather all optimized ships visiting this port
        opt_ships_at_port = []
        for ship_id in ships_data.keys():
            for idx, pv in enumerate(ships_data[ship_id]['port_visits']):
                if pv['port_id'] == port_id:
                    opt_ships_at_port.append((ship_id, idx))
                    
        # Gather external ships also
        external_ships_at_this_port = external_ships_by_port.get(port_id, [])
        
        # 1. Between pairs of optimized ships at same port
        for i in range(len(opt_ships_at_port)):
            for j in range(i+1, len(opt_ships_at_port)):
                si, vi = opt_ships_at_port[i]  # si = ship ID, vi = visit index
                sj, vj = opt_ships_at_port[j]
                
                # Variables for si[vi] and sj[vj]
                pos_i = ship_vars[si]['port_visits'][vi]['position']
                start_i = ship_vars[si]['port_visits'][vi]['start_time']
                end_i = computed_vars[si]['port_visits'][vi]['end_time']
                len_i = ships_data[si]['length_meters']
                
                pos_j = ship_vars[sj]['port_visits'][vj]['position']
                start_j = ship_vars[sj]['port_visits'][vj]['start_time']
                end_j = computed_vars[sj]['port_visits'][vj]['end_time']
                len_j = ships_data[sj]['length_meters']
                
                # To model exclusive OR (non-overlap conditions), introduce binary variables
                # Let's make sure that either:
                #   1. si ends before sj starts OR sj ends before si starts, OR
                #   2. si is completely to the left of sj OR sj is completely to the left of si
                #
                # To formulate this we will need binary variables to indicate which condition holds
                bin_ti = model.addVar(vtype=GRB.BINARY, name=f"nooverlap_s{si}_v{vi}_to_s{sj}_v{vj}_time")
                bin_si = model.addVar(vtype=GRB.BINARY, name=f"nooverlap_s{si}_v{vi}_to_s{sj}_v{vj}_space")
                
                # Time separation constraints (either A ends before B starts or vice versa)
                model.addConstr(
                    end_i <= start_j + bin_ti * planning_horizon,
                    name=f"time_separation_1_{si}v{vi}_to_{sj}v{vj}")
                model.addConstr(
                    end_j <= start_i + (1 - bin_ti) * planning_horizon,
                    name=f"time_separation_2_{si}v{vi}_to_{sj}v{vj}")
                    
                # Space separation constraints (either A fully to left of B or vice versa)
                model.addConstr(
                    pos_i + len_i <= pos_j + bin_si * ports[port_id]['quay_length_meters'],
                    name=f"space_separation_1_{si}v{vi}_to_{sj}v{vj}")
                model.addConstr(
                    pos_j + len_j <= pos_i + (1 - bin_si) * ports[port_id]['quay_length_meters'],
                    name=f"space_separation_2_{si}v{vi}_to_{sj}v{vj}")
        
        # 2. Between optimized and external ships at same port
        for s_opt, v_opt in opt_ships_at_port:
            for k_ext, ext_ship in enumerate(external_ships_at_this_port):
                # Variables for optimized ship
                pos_opt = ship_vars[s_opt]['port_visits'][v_opt]['position']
                start_opt = ship_vars[s_opt]['port_visits'][v_opt]['start_time']
                end_opt = computed_vars[s_opt]['port_visits'][v_opt]['end_time']
                len_opt = ships_data[s_opt]['length_meters']
                
                # Parameters for external ship
                pos_ext = ext_ship['berthing_position']
                start_ext = ext_ship['start_time_hours']
                end_ext = ext_ship['end_time_hours']
                len_ext = ext_ship['length_meters']
                
                # Again use binary vars to determine mode of non-overlap
                bin_ti_oe = model.addVar(vtype=GRB.BINARY, name=f"nooverlap_opt_s{s_opt}_v{v_opt}_to_ext_{port_id}_k{k_ext}_time")
                
                # Time separation
                model.addConstr(
                    end_opt <= start_ext + bin_ti_oe * planning_horizon,
                    name=f"opt_ext_time_separation_1_{s_opt}v{v_opt}_k{k_ext}")
                model.addConstr(
                    end_ext <= start_opt + (1 - bin_ti_oe) * planning_horizon,
                    name=f"opt_ext_time_separation_2_{s_opt}v{v_opt}_k{k_ext}")
                    
                # External ship doesn't move, we only adjust the space of our ship
                bin_sp_oe = model.addVar(vtype=GRB.BINARY, name=f"nooverlap_opt_s{s_opt}_v{v_opt}_to_ext_{port_id}_k{k_ext}_space")
                
                # Space: either opt ship is left of ext or right of it
                model.addConstr(
                    pos_opt + len_opt <= pos_ext + bin_sp_oe * ports[port_id]['quay_length_meters'],
                    name=f"opt_ext_space_separation_1_{s_opt}v{v_opt}_k{k_ext}")
                model.addConstr(
                    pos_ext + len_ext <= pos_opt + (1 - bin_sp_oe) * ports[port_id]['quay_length_meters'],
                    name=f"opt_ext_space_separation_2_{s_opt}v{v_opt}_k{k_ext}")

    # Construct objective expression
    total_obj = gp.quicksum([
        waiting_cost * computed_vars[ship_id]['port_visits'][idx]['waiting_time'] +
        handling_cost * computed_vars[ship_id]['port_visits'][idx]['handling_time'] +
        delay_cost * computed_vars[ship_id]['port_visits'][idx]['delay_time'] +
        lft_penalty * computed_vars[ship_id]['port_visits'][idx]['lft_violation']
        for ship_id in ships_data.keys()
        for idx in range(len(ships_data[ship_id]['port_visits']))
    ])

    # Add fuel costs for each leg for each ship
    fuel_cost_terms = []
    for ship_id, ship in ships_data.items():
        for leg_idx in range(len(ship['port_visits']) - 1):
            # Get origin and destination port IDs
            origin_port_id = ship['route'][leg_idx]  
            dest_port_id = ship['route'][leg_idx+1]
            dist_nm = distances.get(f"{origin_port_id}_{dest_port_id}")  
            if dist_nm is None:
                dist_nm = distances.get(f"{dest_port_id}_{origin_port_id}")
                if dist_nm is None:
                    raise ValueError(f"No distance found for leg {origin_port_id}->{dest_port_id}")
            
            speed_vars_for_leg = ship_vars[ship_id]['speed_selections'][leg_idx]
            
            for k, speed in enumerate(speeds):
                speed_str = str(speed)
                fuel_cons_per_nm = float(ship['fuel_consumption_per_nm_by_speed'][speed_str])
                total_fuel_cost = fuel_cons_per_nm * dist_nm * fuel_price
                
                fuel_cost_terms.append(gp.quicksum(
                    [total_fuel_cost * speed_vars_for_leg[k]]
                ))
                
    total_obj = total_obj + gp.quicksum(fuel_cost_terms)

    # Set objective in model
    model.setObjective(total_obj, GRB.MINIMIZE)

    # Initialize logger
    if log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(log_path, sense="minimize")
    else:
        logger = None

    # Set a callback for logging objective values during optimization
    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                try:
                    logger.log(obj)
                except Exception:
                    pass  # Pass silently if logger fails

    # Optimize the model
    model.optimize(mycallback)

    # Prepare solution output
    
    # Check if solution exists
    if model.SolCount == 0:
        final_solution = {
            "objective_value": float('inf'),
            "ships": []
        }
    else:
        # Build solution results
        final_ships_output = []
        
        for ship_id in ships_data.keys():
            ship_output = {
                "ship_id": ship_id, 
                "port_visits": [], 
                "speed_selections": []
            }
            
            ship_info = ships_data[ship_id]
            
            # Port visits data
            for idx, pv in enumerate(ship_info['port_visits']):
                var_dict = ship_vars[ship_id]['port_visits'][idx]
                comp_vars = computed_vars[ship_id]['port_visits'][idx]
                
                # Extract values
                position_val = var_dict['position'].X 
                start_time_val = var_dict['start_time'].X
                handling_time_val = comp_vars['handling_time'].X
                arrival_time_val = comp_vars['arrival_time'].X
                end_time_val = comp_vars['end_time'].X
                delay_time_val = comp_vars['delay_time'].X
                lft_violation_val = comp_vars['lft_violation'].X
                pos_dev_abs_val = comp_vars['pos_dev_abs'].X
                
                waiting_time_val = max(0, start_time_val - arrival_time_val)  # Recompute for consistency
                
                port_visit_out = {
                    "port_call_index": pv['port_call_index'],
                    "port_id": pv['port_id'],
                    "berthing_position": position_val,
                    "start_time": start_time_val,
                    "handling_time": handling_time_val,
                    "arrival_time": arrival_time_val,
                    "delay": delay_time_val,
                    "lft_violation": lft_violation_val,
                    "position_deviation": pos_dev_abs_val
                }
                
                ship_output["port_visits"].append(port_visit_out)
                
            # Speed selections for legs
            for leg_idx in range(len(ship_info['port_visits']) - 1):
                speed_vars_this_leg = ship_vars[ship_id]['speed_selections'][leg_idx]
                
                selected_speed = None
                for k, speed in enumerate(speeds):
                    if speed_vars_this_leg[k].X > 0.99:  # Binary variable set to 1?
                        selected_speed = speed
                        break
                        
                speed_out = {
                    "from_call": ship_info['port_visits'][leg_idx]['port_call_index'],
                    "to_call": ship_info['port_visits'][leg_idx + 1]['port_call_index'],
                    "speed_knots": selected_speed
                }
                
                ship_output["speed_selections"].append(speed_out)
                
            final_ships_output.append(ship_output)

        final_solution = {
            "objective_value": model.ObjVal,
            "ships": final_ships_output
        }
    
    # Write the solution to the specified file
    with open(solution_path, 'w') as f:
        json.dump(final_solution, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solve Shipping Line Container Ship Optimization Problem')
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the input JSON instance file')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path to save the output JSON solution file')
    parser.add_argument('--time_limit', type=int, default=600,
                        help='Time limit for the solver in seconds')
    parser.add_argument('--log_path', type=str, default=None,
                        help='Path to log intermediate solutions')
    args = parser.parse_args()
    
    solve_shipping_optimization(args.instance_path, args.solution_path, args.time_limit, args.log_path)