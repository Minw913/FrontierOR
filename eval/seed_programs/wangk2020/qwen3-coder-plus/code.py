import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_atm_instance(instance_data, time_limit, log_path):
    # Parse instance data
    num_periods = instance_data['num_time_periods']
    max_disp = instance_data['maximum_displacement_delta']
    airports_list = instance_data['airports']
    flights = {f['id']: f for f in instance_data['flights']}
    aircraft_conn_map = {(c['flight_i'], c['flight_j']): c for c in instance_data['aircraft_connections']}
    passenger_conn_map = {(c['flight_i'], c['flight_j']): c for c in instance_data['passenger_connections']}
    capacity_env = instance_data['capacity_envelopes']
    scenarios_data = instance_data['scenarios']
    
    all_flight_ids = [f['id'] for f in instance_data['flights']]

    # Set rho parameter as per problem description
    rho = 0.5

    # Determine airport to index mapping
    airport_to_idx = {airport: i for i, airport in enumerate(airports_list)}

    # Create model
    model = gp.Model("ATM")
    model.setParam("Threads", 1)
    model.setParam(GRB.Param.TimeLimit, time_limit)

    # Initialize logger if provided
    logger = None
    if log_path:
        logger = SolutionLogger(log_path, sense="minimize")

        def my_callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if logger:
                    logger.log(obj)
    else:
        def my_callback(model, where):
            pass  # No logging

    # Decision variables for Scheduling stage
    # Using indicators for each flight and time period to denote scheduled departure/arrival time
    y_dep = {}  # Scheduled departure indicators { (fid, p): variable }
    y_arr = {}  # Scheduled arrival indicators

    # Valid periods for each flight (based on max displacement)
    flight_valid_ranges = {}
    for fid, flight in flights.items():
        req_dep = flight['dep_period']
        start_period = max(0, req_dep - max_disp)
        end_period = min(num_periods - 1, req_dep + max_disp)
        flight_valid_ranges[fid] = {"start": start_period, "end": end_period}

        # Departure windows
        for p in range(num_periods):
            var_name = f"y_dep_{fid}_{p}"
            y_dep[(fid, p)] = model.addVar(vtype=GRB.BINARY, name=var_name)

        # Arrival windows - adjust according to scheduled duration constraint
        for p in range(num_periods):
            var_name = f"y_arr_{fid}_{p}"
            y_arr[(fid, p)] = model.addVar(vtype=GRB.BINARY, name=var_name)

    # Second stage variables for each scenario
    x_dep_scen = {}
    x_arr_scen = {}
    v_dep_scen = {}
    v_arr_scen = {}

    for scen in scenarios_data:
        sid = scen['id']
        x_dep_scen[sid] = {}
        x_arr_scen[sid] = {}

        for fid in flights.keys():
            for p in range(num_periods):
                var_x_dep = f"x_dep_{fid}_{sid}_{p}"
                var_x_arr = f"x_arr_{fid}_{sid}_{p}"
                x_dep_scen[sid][(fid, p)] = model.addVar(vtype=GRB.BINARY, name=var_x_dep)
                x_arr_scen[sid][(fid, p)] = model.addVar(vtype=GRB.BINARY, name=var_x_arr)

            # Delay variables for this scenario
            var_v_dep = f"v_dep_{fid}_{sid}"
            var_v_arr = f"v_arr_{fid}_{sid}"
            flight_info = flights[fid]
            max_del_dep = flight_info['max_dep_delay']
            max_del_arr = flight_info['max_arr_delay']
            v_dep_scen[sid][fid] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_del_dep, name=var_v_dep)
            v_arr_scen[sid][fid] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_del_arr, name=var_v_arr)

    # Non-increasing constraints for scheduled indicators
    for fid in flights.keys():
        for p in range(num_periods - 1):
            model.addConstr(y_dep[(fid, p)] >= y_dep[(fid, p + 1)], name=f"noninc_sch_dep_{fid}_{p}")
            model.addConstr(y_arr[(fid, p)] >= y_arr[(fid, p + 1)], name=f"noninc_sch_arr_{fid}_{p}")

    # Enforce fixed en-route duration for scheduled times
    for fid, flight in flights.items():
        sch_dur = flight['delta_sch']
        dep_sum = gp.quicksum(y_dep[(fid, p)] for p in range(num_periods))
        arr_sum = gp.quicksum(y_arr[(fid, p)] for p in range(num_periods))
        model.addConstr(arr_sum - dep_sum == sch_dur, name=f"duration_scheduled_{fid}")

    # Passenger connection constraints for scheduled times
    for (flight_i, flight_j), con in passenger_conn_map.items():
        conn_time = con['min_connection_time']
        # We need at least conn_time periods between arrival of i and departure of j
        # So: dep_sum[j] - arr_sum[i] >= conn_time
        dep_sum_j = gp.quicksum(y_dep[(flight_j, p)] for p in range(num_periods))
        arr_sum_i = gp.quicksum(y_arr[(flight_i, p)] for p in range(num_periods))
        model.addConstr(dep_sum_j - arr_sum_i >= conn_time, name=f"pass_conn_{flight_i}_{flight_j}")

    # Constraints for each scenario
    scenario_ids = [s['id'] for s in scenarios_data]
    for sid, scenario in enumerate(scenarios_data):  # Note: using enumerate if scenarios don't guarantee integer ids
        sid = scenario['id']  # This correctly uses the actual ID
        
        # Non-increasing constraints for operated times
        for fid in flights.keys():
            for p in range(num_periods - 1):
                model.addConstr(
                    x_dep_scen[sid][(fid, p)] >= x_dep_scen[sid][(fid, p + 1)],
                    name=f"noninc_op_dep_{fid}_{p}_s{sid}"
                )
                model.addConstr(
                    x_arr_scen[sid][(fid, p)] >= x_arr_scen[sid][(fid, p + 1)],
                    name=f"noninc_op_arr_{fid}_{p}_s{sid}"
                )

        # Aircraft connection constraints for operated times
        for (flight_i, flight_j), con in aircraft_conn_map.items():
            conn_time = con['min_connection_time']
            dep_sum_j = gp.quicksum(x_dep_scen[sid][(flight_j, p)] for p in range(num_periods))
            arr_sum_i = gp.quicksum(x_arr_scen[sid][(flight_i, p)] for p in range(num_periods))
            model.addConstr(dep_sum_j - arr_sum_i >= conn_time, name=f"aircraft_conn_{flight_i}_{flight_j}_s{sid}")

        # Relate delay variables to the differences of indicators
        for fid, flight in flights.items():
            # v_dep = x_dep_scen - y_dep  
            act_dep_scen_sum = gp.quicksum(x_dep_scen[sid][(fid, p)] for p in range(num_periods))
            sch_dep_sum = gp.quicksum(y_dep[(fid, p)] for p in range(num_periods))
            model.addConstr(v_dep_scen[sid][fid] == act_dep_scen_sum - sch_dep_sum, name=f"v_dep_def_{fid}_s{sid}")

            # v_arr ≥ x_arr_scen - y_arr (we might have early arrivals that create negative arrival delays)
            act_arr_scen_sum = gp.quicksum(x_arr_scen[sid][(fid, p)] for p in range(num_periods))
            sch_arr_sum = gp.quicksum(y_arr[(fid, p)] for p in range(num_periods))
            diff = act_arr_scen_sum - sch_arr_sum
            model.addConstr(v_arr_scen[sid][fid] >= diff, name=f"v_arr_def_geq_{fid}_s{sid}")

            # Enforce operational en-route duration limits
            enroute_actual = act_arr_scen_sum - act_dep_scen_sum
            if flight['delta_min'] <= enroute_actual.getValue() <= flight['delta_max']:
                continue  # These constraints make sense with values; not needed as expressions here until solved
            model.addConstr(enroute_actual >= flight['delta_min'], name=f"min_op_dur_{fid}_s{sid}")
            model.addConstr(enroute_actual <= flight['delta_max'], name=f"max_op_dur_{fid}_s{sid}")

        # Capacity constraints for each airport and time period
        conditions_list = scenario['operating_conditions']
        for airport_code in airports_list:
            cond_list = conditions_list[airport_code]

            # Get flights originating from this airport and ending at this airport
            depart_flights = [fid for fid in flights.keys() if flights[fid]['dep_airport'] == airport_code]
            arrival_flights = [fid for fid in flights.keys() if flights[fid]['arr_airport'] == airport_code]

            envs = capacity_env[airport_code]
            segs = envs.get('VMC', []) or envs.get('IMC', [])
            
            for t in range(num_periods):
                weather_cond = cond_list[t]
                cap_segments = envs[weather_cond]

                # Calculate departures and arrivals at this airport in time period t
                # Departures in time period t: x[p-1,t] - x[p,t] = 1 means actual operation happened
                depart_expr = gp.LinExpr()
                for fid in depart_flights:
                    if t == 0:
                        de = 1 - x_dep_scen[sid][(fid, 0)]
                    else:
                        de = x_dep_scen[sid][(fid, t-1)] - x_dep_scen[sid][(fid, t)]
                    depart_expr.add(de)

                # Arrivals in time period t  
                arrive_expr = gp.LinExpr()
                for fid in arrival_flights:
                    if t == 0:
                        ae = 1 - x_arr_scen[sid][(fid, 0)]
                    else:
                        ae = x_arr_scen[sid][(fid, t-1)] - x_arr_scen[sid][(fid, t)]
                    arrive_expr.add(ae)

                # Add capacity constraint for each segment
                for seg_idx, seg in enumerate(cap_segments):
                    cap_constr = seg['a'] * depart_expr + seg['b'] * arrive_expr <= seg['Q']
                    model.addConstr(cap_constr, name=f"cap_{airport_code}_t{t}_seg{seg_idx}_s{sid}")

    # Objective computation
    # Part 1: Schedule displacement cost
    schedule_displacement_cost = gp.LinExpr(0)
    # Add binary variables to indicate specific scheduled time periods
    z_dep = {}
    for fid, flight in flights.items():
        req_t = flight['dep_period']
        
        # Introduce binary variables to indicate which time was chosen
        z_fid = {}  # z[t] indicates if scheduled departure is at time t
        for t in range(num_periods):
            z_name = f"z_dep_{fid}_{t}"
            z_fid[t] = model.addVar(vtype=GRB.BINARY, name=z_name)
            z_dep[(fid, t)] = z_fid[t]

        # Link y indicator to actual position selected (binary decomposition constraint)
        for t in range(num_periods - 1):
            # When y[t] decreases from 1 to 0, that indicates actual departure
            # For this we use: if y[t] and not y[t+1] -> z[t]=1
            # Model as: z[t] >= y[t] - y[t+1]
            model.addConstr(z_fid[t] >= y_dep[(fid, t)] - y_dep[(fid, t+1)], name=f"link_y_to_z_{fid}_{t}")
        
        # For the last period, if y[last] = 1, then z[last] should be 1
        model.addConstr(z_fid[num_periods - 1] >= y_dep[(fid, num_periods - 1)], name=f"last_z_link_{fid}")

        # Also ensure at most one z = 1 (the others are handled by non-increasing property and link above)
        model.addConstr(gp.quicksum(z_fid[t] for t in range(num_periods)) == 1, name=f"one_scheduled_departure_{fid}")

        # Displacement cost: sum of |t - req_t| * z[t]
        local_cost = gp.quicksum(abs(t - req_t) * z_fid[t] for t in range(num_periods))
        schedule_displacement_cost.add(local_cost)

    # Part 2: Expected delay cost over all scenarios
    expected_delay_cost = gp.LinExpr(0)
    
    for scenario in scenarios_data:
        sid = scenario['id']
        prob = scenario['probability']
        
        # Calculate total delay cost in this scenario
        scenario_delay_cost = gp.quicksum(
            flights[fid]['cost_dep_delay'] * v_dep_scen[sid][fid] +
            flights[fid]['cost_arr_delay'] * v_arr_scen[sid][fid]
            for fid in flights.keys()
        )
        expected_delay_cost.add(prob * scenario_delay_cost)

    # Final objective: rho * schedule_cost + (1-rho) * delay_cost
    obj_expr = rho * schedule_displacement_cost + (1 - rho) * expected_delay_cost
    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Solve the model
    model.optimize(lambda model, where: my_callback(model, where))

    # Process solution
    objective_value = float('inf')
    if model.SolCount > 0:
        objective_value = model.ObjVal
        
        schedule = {}
        for fid in flights.keys():
            # Decode scheduled time from indicators
            sch_dep = -1
            sch_arr = -1
            
            # Find period where indicator transitions from 1 to 0 for departure
            found = False
            for p in range(num_periods - 1):
                if abs(y_dep[(fid, p)].X - y_dep[(fid, p + 1)].X) > 0.5 and y_dep[(fid, p)].X > 0.5:
                    sch_dep = p
                    found = True
                    break
            if not found and abs(y_dep[(fid, num_periods - 1)].X - 1.0) < 0.5:
                sch_dep = num_periods - 1
                
            # Find period where indicator transitions from 1 to 0 for arrival
            found = False
            for p in range(num_periods - 1):
                if abs(y_arr[(fid, p)].X - y_arr[(fid, p + 1)].X) > 0.5 and y_arr[(fid, p)].X > 0.5:
                    sch_arr = p
                    found = True
                    break
            if not found and abs(y_arr[(fid, num_periods - 1)].X - 1.0) < 0.5:
                sch_arr = num_periods - 1

            # If not decoded directly, use z variables if available
            if sch_dep == -1:
                for p in range(num_periods):
                    if z_dep[(fid, p)].X > 0.5:
                        sch_dep = p
                        break
                        
            # If still not found, use request time as fallback
            if sch_dep == -1:
                sch_dep = flights[fid]['dep_period']
            if sch_arr == -1:
                sch_arr = flights[fid]['arr_period']
            
            schedule[fid] = {
                "scheduled_dep": sch_dep,
                "scheduled_arr": sch_arr
            }

        scenario_solutions = []
        for scenario in scenarios_data:
            sid = scenario['id']
            
            x_dep_dict = {}
            x_arr_dict = {}
            v_dep_dict = {}
            v_arr_dict = {}
            
            for fid in flights.keys():
                for p in range(num_periods):
                    x_dep_key = f"{fid}_{p}"
                    x_dep_dict[x_dep_key] = int(round(x_dep_scen[sid][(fid, p)].X))
                    
                    x_arr_key = f"{fid}_{p}"
                    x_arr_dict[x_arr_key] = int(round(x_arr_scen[sid][(fid, p)].X))
                
                v_dep_dict[fid] = int(round(v_dep_scen[sid][fid].X))
                v_arr_dict[fid] = int(round(v_arr_scen[sid][fid].X))

            temp_scen_sol = {
                "scenario_id": sid,
                "x_dep": x_dep_dict,
                "x_arr": x_arr_dict,
                "v_dep": v_dep_dict,
                "v_arr": v_arr_dict
            }
            scenario_solutions.append(temp_scen_sol)

        solution = {
            "objective_value": objective_value,
            "schedule": schedule,
            "scenario_solutions": scenario_solutions
        }
    else:
        # Return a valid but empty structure if no solution found
        solution = {
            "objective_value": float('inf'),
            "schedule": {},
            "scenario_solutions": []
        }

    return solution


def main():
    parser = argparse.ArgumentParser(description='Solve Air Traffic Management Optimization Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, help='Path to JSONL log file (optional)')

    args = parser.parse_args()

    # Read the instance
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Solve the instance
    solution = solve_atm_instance(instance_data, args.time_limit, args.log_path)

    # Write the solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()