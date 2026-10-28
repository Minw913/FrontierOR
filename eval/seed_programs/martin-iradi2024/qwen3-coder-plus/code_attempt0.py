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
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions

    # Decision Variables
    
    # For each ship and port visit, we have:
    # 1. berthing position (continuous)
    # 2. start time (continuous)
    # 3. speed selection (binary indicator)
    
    ship_vars = {}
    
    for ship_id, ship in ships_data.items():
        ship_vars[ship_id] = {'port_visits': {}, 'speed_selections': []}
        
        # Port visit variables
        for idx, pv in enumerate(ship['port_visits']):
            var_dict = {}
            
            # Position and time variables
            pos_lb = 0
            pos_ub = ports[pv['port_id']]['quay_length_meters'] - ship['length_meters']
            if pos_ub < 0:
                raise ValueError(f"Ship {ship_id} is longer than available quay at port {pv['port_id']}")
                
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
        
        # Speed variables for inter-port legs
        for i in range(len(ship['port_visits']) - 1):
            speed_vars = []
            for j, speed in enumerate(speeds):
                speed_var = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"ship_{ship_id}_speed_leg_{i}_at_{speed}"
                )
                speed_vars.append(speed_var)
            ship_vars[ship_id]['speed_selections'].append({'vars': speed_vars})
    
    # Precompute variables for handling time, delays, etc.
    computed_vars = {}
    
    for ship_id, ship in ships_data.items():
        computed_vars[ship_id] = {'port_visits': {}}
        
        for idx, pv in enumerate(ship['port_visits']):
            vars_dict = ship_vars[ship_id]['port_visits'][idx]
            
            # Handling time = min_handling * (1 + beta * |position - ideal|)
            # We linearize this using auxiliary variables
            # Use pos_deviation_abs >= position - ideal and pos_deviation_abs >= -(position - ideal)
            pos_dev_abs = model.addVar(name=f"ship_{ship_id}_pos_dev_abs_{pv['port_call_index']}", vtype=GRB.CONTINUOUS, lb=0)
            # Linearization constraints
            model.addConstr(pos_dev_abs >= vars_dict['position'] - pv['ideal_berthing_position'])
            model.addConstr(pos_dev_abs >= pv['ideal_berthing_position'] - vars_dict['position'])
            
            # Handling time
            handling_time = model.addVar(name=f"ship_{ship_id}_handling_time_{pv['port_call_index']}", vtype=GRB.CONTINUOUS)
            model.addConstr(handling_time == pv['min_handling_time_hours'] * (1 + beta * pos_dev_abs))
            
            # End time = start_time + handling_time
            end_time = model.addVar(name=f"ship_{ship_id}_end_time_{pv['port_call_index']}", vtype=GRB.CONTINUOUS, ub=planning_horizon)
            model.addConstr(end_time == vars_dict['start_time'] + handling_time)
            
            # Arrival time and waiting time
            arrival_time = None
            if idx == 0:  # First leg doesn't have arrival defined by travel
                arrival_time = model.addVar(name=f"ship_{ship_id}_arrival_{pv['port_call_index']}", vtype=GRB.CONTINUOUS, lb=0, ub=planning_horizon)
                # Enforce arrival <= start time
                model.addConstr(arrival_time <= vars_dict['start_time'])
            else:  # Has prior arrival
                prev_end_time = computed_vars[ship_id]['port_visits'][idx-1]['end_time']
                leg_id = idx - 1
                speed_selection = ship_vars[ship_id]['speed_selections'][leg_id]
                
                # Compute arrival_time = prev_end + travel_time based on selected speed
                dist_nm = distances[f"{ship['route'][idx-1]}_{ship['route'][idx]}"]
                arrival_time = model.addVar(name=f"ship_{ship_id}_arrival_{pv['port_call_index']}", vtype=GRB.CONTINUOUS, ub=planning_horizon)
                
                # Linearize depending on speed choice
                for k, speed in enumerate(speeds):
                    travel_dur_for_leg = dist_nm * float(travel_times[str(speed)])
                    model.addConstr(
                        arrival_time >= prev_end_time + travel_dur_for_leg - planning_horizon * (1 - speed_selection['vars'][k])
                    )
                    
                # Ensure arrival_time is tight to max of options
                big_M = float(planning_horizon)  # Large enough number
                for k, speed in enumerate(speeds):
                    travel_dur_for_leg = dist_nm * float(travel_times[str(speed)])
                    model.addConstr(
                        arrival_time <= prev_end_time + travel_dur_for_leg + big_M * (1 - speed_selection['vars'][k])
                    )
            
            # Waiting time = max(0, start - arrival)
            waiting_time = model.addVar(name=f"ship_{ship_id}_waiting_time_{pv['port_call_index']}", vtype=GRB.CONTINUOUS, lb=0)
            # Linearize: waiting_time = max(0, start_time - arrival_time)
            model.addConstr(waiting_time >= vars_dict['start_time'] - arrival_time)
            
            # Delay = max(0, end - expected_finish)
            delay_time = model.addVar(name=f"ship_{ship_id}_delay_time_{pv['port_call_index']}", vtype=GRB.CONTINUOUS, lb=0)
            expected_finish = pv['expected_finish_time_hours']
            model.addConstr(delay_time >= end_time - expected_finish)
            
            # LFT violation = max(0, end - latest_finish)
            lft_violation = model.addVar(name=f"ship_{ship_id}_lft_violation_{pv['port_call_index']}", vtype=GRB.CONTINUOUS, lb=0)
            latest_finish = pv['latest_finish_time_hours']
            model.addConstr(lft_violation >= end_time - latest_finish)
            
            # Store values for objective computation
            computed_vars[ship_id]['port_visits'][idx] = {
                'handling_time': handling_time,
                'end_time': end_time,
                'arrival_time': arrival_time,
                'waiting_time': waiting_time,
                'delay_time': delay_time,
                'lft_violation': lft_violation,
                'pos_dev_abs': pos_dev_abs
            }

    # Add constraints for non-overlapping ships within each port
    for port_id in ports.keys():
        all_ships_at_port = []  # List of (ship_id, visit_idx) pairs
        for ship_id in ships_data.keys():
            for idx, pv in enumerate(ships_data[ship_id]['port_visits']):
                if pv['port_id'] == port_id:
                    all_ships_at_port.append((ship_id, idx))
        
        # Add external ships too
        external_ships_at_port = external_ships_by_port[port_id]
        
        # Pairwise conflicts between optimized ships
        for i in range(len(all_ships_at_port)):
            for j in range(i+1, len(all_ships_at_port)):
                s1, idx1 = all_ships_at_port[i]
                s2, idx2 = all_ships_at_port[j]
                
                s1_vars = ship_vars[s1]['port_visits'][idx1]
                s2_vars = ship_vars[s2]['port_visits'][idx2]
                
                s1_computed = computed_vars[s1]['port_visits'][idx1]
                s2_computed = computed_vars[s2]['port_visits'][idx2]
                
                len_s1 = ships_data[s1]['length_meters']
                len_s2 = ships_data[s2]['length_meters']
                
                # Time-based conflict: either s1 ends before s2 starts, or s2 ends before s1 starts
                time_before_cons = model.addConstr(s1_computed['end_time'] <= s2_vars['start_time'] + planning_horizon * model.addVar(vtype=GRB.BINARY, name=f"tb_s{s1}_v{idx1}_to_s{s2}_v{idx2}"))
                model.addConstr(s2_computed['end_time'] <= s1_vars['start_time'] + planning_horizon * (2 - time_before_cons) - model.getVarByName(time_before_cons.getAttr("VarName")))
                
                # Spatial conflict: either s1 positioned to the left of s2, or s2 to left of s1
                s1_right = s1_vars['position'] + len_s1
                s2_right = s2_vars['position'] + len_s2
                
                spatial_left_cons = model.addConstr(s1_right <= s2_vars['position'] + planning_horizon * model.addVar(vtype=GRB.BINARY, name=f"sl_s{s1}_v{idx1}_to_s{s2}_v{idx2}"))
                model.addConstr(s2_right <= s1_vars['position'] + planning_horizon * (2 - spatial_left_cons) - model.getVarByName(spatial_left_cons.getAttr("VarName")))

        # Conflict between optimized and external ships
        for opt_ship, opt_idx in all_ships_at_port:
            for ext_ship in external_ships_at_port:
                opt_pv = ships_data[opt_ship]['port_visits'][opt_idx]
                
                if(opt_pv['port_id'] != port_id): continue
                    
                opt_vars = ship_vars[opt_ship]['port_visits'][opt_idx]
                opt_computed = computed_vars[opt_ship]['port_visits'][opt_idx]
                opt_len = ships_data[opt_ship]['length_meters']
                
                # ext: start from ext_ship['start_time'] to ext_ship['end_time'], ext_pos, with length ext_len
                ext_start_time = ext_ship['start_time_hours']
                ext_end_time = ext_ship['end_time_hours']
                ext_pos = ext_ship['berthing_position']
                ext_len = ext_ship['length_meters']
                
                # Either non-overlapping in time: opt end <= ext start OR ext end <= opt start  
                t_bin = model.addVar(vtype=GRB.BINARY, name=f"time_no_overlap_opt_s{opt_ship}_v{opt_idx}_ext_{ext_ship['id']}")
                model.addConstr(opt_computed['end_time'] <= ext_start_time + planning_horizon * t_bin)
                model.addConstr(ext_end_time <= opt_vars['start_time'] + planning_horizon * (1-t_bin))
                
                # Or non-overlapping in space: opt right <= ext_pos OR ext right <= opt left
                s_bin = model.addVar(vtype=GRB.BINARY, name=f"space_no_overlap_opt_s{opt_ship}_v{opt_idx}_ext")
                opt_right = opt_vars['position'] + opt_len  
                ext_right = ext_pos + ext_len  
                opt_left = opt_vars['position']
                
                model.addConstr(opt_right <= ext_pos + ports[port_id]['quay_length_meters'] * s_bin)
                model.addConstr(ext_right <= opt_left + ports[port_id]['quay_length_meters'] * (1-s_bin))

    # Ensure exactly one speed is selected per leg
    for ship_id, ship in ships_data.items():
        for leg_idx in range(len(ship['port_visits']) - 1):
            speed_vars = ship_vars[ship_id]['speed_selections'][leg_idx]['vars']
            model.addConstr(gp.quicksum(speed_vars) == 1)

    # Objective function
    total_obj = gp.LinExpr()

    # Costs per port visit across all optimized ships
    for ship_id in ships_data.keys():
        ship = ships_data[ship_id]
        for idx, pv in enumerate(ship['port_visits']):
            comp_vars = computed_vars[ship_id]['port_visits'][idx]
            
            # Add costs: waiting, handling, delay, lft_penalty, position_dev_penalty
            total_obj += (
                waiting_cost * comp_vars['waiting_time'] +
                handling_cost * comp_vars['handling_time'] +
                delay_cost * comp_vars['delay_time'] +
                lft_penalty * comp_vars['lft_violation']
            )

    # Add fuel costs
    for ship_id, ship in ships_data.items():
        for leg_idx in range(len(ship['port_visits']) - 1):
            speed_vars = ship_vars[ship_id]['speed_selections'][leg_idx]['vars']
            orig_port = ship['route'][leg_idx]
            dest_port = ship['route'][leg_idx + 1]
            dist_nm = distances[f"{orig_port}_{dest_port}"]
            
            # Sum over speed choices for this leg: (consumption_per_nm_at_speed * distance * fuel_price) * binary_indicator
            fuel_cost_expr = gp.QuadExpr()
            for k, speed in enumerate(speeds):
                speed_str = str(speed)
                fuel_cons_nm = ship['fuel_consumption_per_nm_by_speed'][speed_str]
                fuel_cost_per_leg = fuel_cons_nm * dist_nm * fuel_price
                fuel_cost_expr += fuel_cost_per_leg * speed_vars[k]
                
            total_obj += fuel_cost_expr

    model.setObjective(total_obj, GRB.MINIMIZE)

    # Initialize logger
    from solution_logger import SolutionLogger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    # Optimize
    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)
    
    model.optimize(mycallback)

    # Prepare the solution for export
    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible")
        model.computeIIS()
        model.write("infeasible.ilp")
        return None
        
    if model.SolCount == 0:
        print("No solution found within time limit")
        final_solution = {
            "objective_value": float('inf'),
            "ships": []
        }
    else:
        # Build solution
        final_ships_output = []
        
        for ship_id, ship in ships_data.items():
            ship_output = {"ship_id": ship_id, "port_visits": [], "speed_selections": []}
            
            for idx, pv in enumerate(ship['port_visits']):
                var_dict = ship_vars[ship_id]['port_visits'][idx]
                comp_vars = computed_vars[ship_id]['port_visits'][idx]
                
                start_time_val = var_dict['start_time'].X
                position_val = var_dict['position'].X
                handling_time_val = comp_vars['handling_time'].X
                arrival_time_val = comp_vars['arrival_time'].X
                end_time_val = comp_vars['end_time'].X
                waiting_time_val = comp_vars['waiting_time'].X
                delay_time_val = comp_vars['delay_time'].X
                lft_violation_val = comp_vars['lft_violation'].X
                pos_dev_abs_val = comp_vars['pos_dev_abs'].X
                
                port_visit_output = {
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
                
                ship_output["port_visits"].append(port_visit_output)
                
            # Speeds for each leg
            for leg_idx in range(len(ship['port_visits']) - 1):
                speed_vars = ship_vars[ship_id]['speed_selections'][leg_idx]['vars']
                
                # Find which speed was selected
                selected_speed = None
                for k, speed_var in enumerate(speed_vars):
                    if speed_var.X > 0.99:  # Binary should be 0 or 1, add small tolerance
                        selected_speed = speeds[k]
                        break
                        
                speed_output = {
                    "from_call": ship['port_visits'][leg_idx]['port_call_index'],
                    "to_call": ship['port_visits'][leg_idx + 1]['port_call_index'],
                    "speed_knots": selected_speed
                }
                
                ship_output["speed_selections"].append(speed_output)
                
            final_ships_output.append(ship_output)

        final_solution = {
            "objective_value": model.ObjVal,
            "ships": final_ships_output
        }
    
    # Write solution to output file
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