import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np
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
    scenarios = {s['id']: s for s in instance_data['scenarios']}
    
    # Create mapping from airport codes to indices
    airport_to_idx = {airport: i for i, airport in enumerate(airports_list)}
    
    # Extract flight IDs and create lists
    flight_ids = list(flights.keys())
    flight_to_idx = {fid: idx for idx, fid in enumerate(flight_ids)}
    
    # rho is not in the example but based on text, let's assume it was passed separately or has a default
    # This value needs to be provided by the problem or is missing in schema
    # Since it's not in the schema directly, we'll need a default or derive from somewhere
    # Looking at the problem description, it's referenced as a parameter
    # Let me look at the overall objective: weighted combination where first component is 
    # multiplied by rho and second by (1-rho). So we'll set a reasonable default value
    rho = 0.5  # Default value since it's not provided in schema
    
    # Create model
    model = gp.Model("ATM")
    model.setParam(GRB.Param.TimeLimit, time_limit)
    if log_path:
        model.setParam(GRB.Param.LogToConsole, 0)  # Suppress console logs since we're using custom logger
    
    # Decision variables
    # Scheduling variables (first stage)
    y_dep = {}  # Scheduled departure indicators
    y_arr = {}  # Scheduled arrival indicators
    
    # For each flight, create variables for time range [requested - max_displacement, requested + max_displacement]
    for fid, flight in flights.items():
        req_dep = flight['dep_period']
        start_period = max(0, req_dep - max_disp)
        end_period = min(num_periods - 1, req_dep + max_disp)
        
        # Variables for scheduled departure indicators
        for p in range(num_periods):
            var_name = f"y_dep_{fid}_{p}"
            y_dep[(fid, p)] = model.addVar(vtype=GRB.BINARY, name=var_name)
            
        # Variables for scheduled arrival indicators
        # Ensure arrival window aligns with departure window and duration constraints
        arr_start = max(0, req_dep - max_disp + flight['delta_sch'])
        arr_end = min(num_periods - 1, req_dep + max_disp + flight['delta_sch'])
        
        for p in range(num_periods):
            var_name = f"y_arr_{fid}_{p}"
            y_arr[(fid, p)] = model.addVar(vtype=GRB.BINARY, name=var_name)
    
    # Second stage variables for each scenario
    x_dep_scen = {}  # Operated departure indicators by scenario
    x_arr_scen = {}  # Operated arrival indicators by scenario
    v_dep_scen = {}  # Departure delays by scenario
    v_arr_scen = {}  # Arrival delays by scenario
    
    for sid in scenarios.keys():
        x_dep_scen[sid] = {}
        x_arr_scen[sid] = {}
        
        for fid, flight in flights.items():
            # Operated departure and arrival indicators for this scenario
            for p in range(num_periods):
                var_name = f"x_dep_{fid}_{sid}_{p}"
                x_dep_scen[sid][(fid, p)] = model.addVar(vtype=GRB.BINARY, name=var_name)
                
                var_name = f"x_arr_{fid}_{sid}_{p}"
                x_arr_scen[sid][(fid, p)] = model.addVar(vtype=GRB.BINARY, name=var_name)
            
            # Delay variables for this scenario
            var_name = f"v_dep_{fid}_{sid}"
            v_dep_scen[sid][fid] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=flight['max_dep_delay'], name=var_name)
            
            var_name = f"v_arr_{fid}_{sid}"
            v_arr_scen[sid][fid] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=flight['max_arr_delay'], name=var_name)

    # Non-increasing constraints for scheduled indicators
    for fid in flight_ids:
        for p in range(num_periods - 1):
            model.addConstr(y_dep[(fid, p)] >= y_dep[(fid, p+1)])
            model.addConstr(y_arr[(fid, p)] >= y_arr[(fid, p+1)])
    
    # Enforce scheduled en-route duration
    for fid, flight in flights.items():
        scheduled_dur = sum(y_dep[(fid, p)] for p in range(num_periods)) - sum(y_arr[(fid, p)] for p in range(num_periods))
        model.addConstr(scheduled_dur == -flight['delta_sch'])  # Difference is negative because we want arrival index > departure index
        
    # Passenger connection constraints (in scheduling)
    for (flight_i, flight_j), conn_info in passenger_conn_map.items():
        if flight_i not in flights or flight_j not in flights:
            continue  # Skip if flight doesn't exist
            
        min_conn_time = conn_info['min_connection_time']
        diff = sum(x_dep_scen[0][(flight_j, p)] for p in range(num_periods)) - sum(y_arr[(flight_i, p)] for p in range(num_periods))
        model.addConstr(diff >= min_conn_time)
    
    # Apply passenger and aircraft connection constraints for all scenarios
    scenario_ids = list(scenarios.keys())
    for sid in scenario_ids:
        # Enforce aircraft connections in operation (only aircraft)
        for (flight_i, flight_j), conn_info in aircraft_conn_map.items():
            if flight_i not in flights or flight_j not in flights:
                continue
                
            min_conn_time = conn_info['min_connection_time']
            diff = (sum(x_dep_scen[sid][(flight_j, p)] for p in range(num_periods)) - 
                    sum(x_arr_scen[sid][(flight_i, p)] for p in range(num_periods)))
            model.addConstr(diff >= min_conn_time)
        
        # Non-increasing constraints for operated indicators in each scenario
        for fid in flight_ids:
            for p in range(num_periods - 1):
                model.addConstr(x_dep_scen[sid][(fid, p)] >= x_dep_scen[sid][(fid, p + 1)])
                model.addConstr(x_arr_scen[sid][(fid, p)] >= x_arr_scen[sid][(fid, p + 1)])
        
        # Delay constraints: actual delay must match calculated difference
        for fid, flight in flights.items():
            # Calculate departure delay based on difference in sums of indicators
            departure_delay_calculation = (sum(x_dep_scen[sid][(fid, p)] for p in range(num_periods)) - 
                                          sum(y_dep[(fid, p)] for p in range(num_periods)))
            # Ensure calculated delay matches v_dep (the delay variable)
            model.addConstr(v_dep_scen[sid][fid] >= departure_delay_calculation)
            # For arrival delay:
            arrival_delay_calculation = (sum(x_arr_scen[sid][(fid, p)] for p in range(num_periods)) - 
                                        sum(y_arr[(fid, p)] for p in range(num_periods)))
            # Ensure calculated delay is bounded by v_arr
            model.addConstr(v_arr_scen[sid][fid] >= arrival_delay_calculation)
            
            # Max delay constraints already enforced by variable bounds
            
            # Enforce that actual en-route duration is within min/max for each scenario
            en_route_duration = (sum(x_arr_scen[sid][(fid, p)] for p in range(num_periods)) - 
                                 sum(x_dep_scen[sid][(fid, p)] for p in range(num_periods)))
            model.addConstr(en_route_duration <= -flight['delta_min'])  # Negative as arrival index > dep index
            model.addConstr(en_route_duration >= -flight['delta_max'])  # Lower bound of allowed range
            
        # Airport capacity constraints for each scenario
        scenario_data = scenarios[sid]
        for airport_code in airports_list:
            airport_idx = airport_to_idx[airport_code]
            conditions = scenario_data['operating_conditions'].get(airport_code, ['VMC'] * num_periods)
            
            dep_flights = []  # Flights departing from this airport
            arr_flights = []  # Flights arriving to this airport
            for fid, flight in flights.items():
                if flight['dep_airport'] == airport_code:
                    dep_flights.append(fid)
                if flight['arr_airport'] == airport_code:
                    arr_flights.append(fid)
            
            # Process each time period
            for p in range(num_periods):
                weather_cond = conditions[p]
                env_segments = capacity_env[airport_code][weather_cond]
                
                # For each segment of envelope, add constraint
                for seg in env_segments:
                    departures_in_period_p = 0
                    arrivals_in_period_p = 0
                    
                    # Count departures: difference y(p) - y(p+1), or for operated difference x(p) - x(p+1)
                    # We use drop from previous: in period p, the count is x[p-1] - x[p] assuming x[-1] = 1 initially
                    # Since we need the change at boundary of p, departures in period p is 
                    # [value before period p] - [value after period p] = x[p-1] - x[p],
                    # but for non-increasing, at period p, this is equivalent to:
                    # if p==0 then it's (1-x[p]) else x[p-1] - x[p] (if we assume x starts high)
                    # Actually, we interpret x[p] = 1 means flight operates at time >= p.
                    # So flight operates ON period p means x(p) was 1 then becomes 0 at point just after p.
                    # So flight operates at period p if x[p]=1 and x[p+1]=0 (assuming non-increasing x)
                    # So, departures in period p = x[p-1] - x[p] (with x[-1]=1), which would be (1 - x[0]) when p=0
                    
                    # At period 0: departures = 1 - x[0][any_flight]
                    # At period p>0: departures = x[p-1] - x[p]
                    for fid in dep_flights:
                        if p == 0:
                            departures_in_period_p += (1 - x_dep_scen[sid][(fid, p)]) if fid in x_dep_scen[sid] else 0
                        else:
                            # Calculate flight starts at period p (from period p-1 to p)
                            departures_in_period_p += (x_dep_scen[sid][(fid, p-1)] - x_dep_scen[sid][(fid, p)])
                    
                    for fid in arr_flights:
                        if p == 0:
                            arrivals_in_period_p += (1 - x_arr_scen[sid][(fid, p)])
                        else:
                            arrivals_in_period_p += (x_arr_scen[sid][(fid, p-1)] - x_arr_scen[sid][(fid, p)])
                    
                    # Add capacity constraint for this segment
                    model.addConstr(seg['a'] * departures_in_period_p + seg['b'] * arrivals_in_period_p <= seg['Q'])

    # Objective function
    # First component: Schedule displacement cost
    schedule_displacement_cost = 0
    for fid, flight in flights.items():
        # Weights for displacement cost from requested time, which depends on distance from target indicator pattern
        req_period = flight['dep_period']
        displacement_cost = 0
        for p in range(num_periods):
            # We want to minimize sum over all periods of |scheduled_period - requested_period|
            # Using indicator approach: sum of abs diff between y and req pattern
            # The shift corresponds to a single "0" -> "1" transition, say at position t
            # So if true sched is t, cost = |t - req|
            # To express this, let sched[t] = 1 means scheduled at period t
            # But current formulation sums all y values to represent period
            # For simplicity, we can approximate the absolute deviation:
            # Cost due to period p = coeff * abs(sum_indices_for_schedule_dep - req) 
            # This requires special handling, but let's do the simpler version:
            # Minimize sum over p of coeff * |position of transition from p|
            # Better approach - let's define binary vars that represent exact scheduled period
            # For now, we will compute displacement as total deviation from requested pattern
            
            # Let z be 1 if flight is scheduled to dep at/after period p, 0 elsewhere
            # Then the position of transition (the scheduled departure) is sum of all z[p]*w[p]
            # The difference |actual_pos - req_pos| is harder
            # We should re-parametrize as: for each period p, Z[sched_dep_fid = p]
            # This is more standard
            
            # Define binary vars for exact scheduled time
            Z_dep = {}  # Binary var for each possible scheduled time
            for pp in range(max(0, req_period - max_disp), min(num_periods, req_period + max_disp + 1)):
                var_name = f"Z_sched_dep_{fid}_{pp}" 
                Z_dep[pp] = model.addVar(vtype=GRB.BINARY, name=var_name)
                
                # Link with original y vars: y[p] for p>=pp should reflect that schedule happens >= pp
                # If we're scheduled at time pp, then y[q] = 1 for q <= pp, y[q] = 0 for q > pp
                for q in range(num_periods):
                    if q <= pp:
                        # When scheduled at pp, y[q] = 1 for q <= pp
                        model.addConstr(y_dep[(fid, q)] >= Z_dep[pp])
                    else:
                        # When scheduled at pp, y[q] = 0 for q > pp
                        model.addConstr(y_dep[(fid, q)] <= Z_dep[pp])
                        
            model.addConstr(gp.quicksum(Z_dep[pp] for pp in Z_dep.keys()) == 1)
            
            # Now calculate displacement as sum of weights for moving from req
            disp_cost_total = gp.quicksum(
                abs(pp - req_period) * Z_dep[pp] for pp in Z_dep.keys()
            )
            schedule_displacement_cost += disp_cost_total
    
    # Second component: Expected operational delay cost
    expected_delay_cost = 0
    for sid, scenario in scenarios.items():
        prob = scenario['probability']
        scenario_delay_cost = 0
        
        for fid, flight in flights.items():
            dep_delay_cost = flight['cost_dep_delay'] * v_dep_scen[sid][fid]
            arr_delay_cost = flight['cost_arr_delay'] * v_arr_scen[sid][fid]
            scenario_delay_cost += dep_delay_cost + arr_delay_cost
            
        expected_delay_cost += prob * scenario_delay_cost
    
    # Combined objective
    total_obj = rho * schedule_displacement_cost + (1 - rho) * expected_delay_cost
    
    model.setObjective(total_obj, GRB.MINIMIZE)
    
    # Optimize
    if log_path:
        logger = SolutionLogger(log_path, sense="minimize")
        
        def my_callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj)
        
        model.optimize(my_callback)
    else:
        model.optimize()

    # Prepare solution output
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.objVal
        
        schedule = {}
        for fid in flight_ids:
            # Decode y indicators to get exact scheduled departure/arrival periods
            # Sum y values represents the index from end of timeline, so actual time = num_periods - sum
            dep_sum = sum(int(y_dep[(fid, p)].x) for p in range(num_periods))
            # If y changes from 1 to 0 at pos sched_time, then that sum equals num_periods - sched_time
            # So sched_time = num_periods - dep_sum
            # Actually, if y[t] = 1 for t <= sched_time and y[t] = 0 for t > sched_time, 
            # then sum y[p] over all p = sched_time + 1 if scheduled at time sched_time
            # Or we can say: sum of all y[p] = number of time steps where y[p]=1
            # Which = time of transition + 1
            # So scheduled time = sum of y[p] - 1
            scheduled_dep_period = int(dep_sum - 1)
            
            arr_sum = sum(int(y_arr[(fid, p)].x) for p in range(num_periods))
            scheduled_arr_period = int(arr_sum - 1)
            
            schedule[fid] = {
                "scheduled_dep": scheduled_dep_period,
                "scheduled_arr": scheduled_arr_period
            }
            
        scenario_solutions = []
        for sid in scenarios.keys():
            x_dep = {}
            x_arr = {}
            v_dep = {}
            v_arr = {}

            for fid in flight_ids:
                for p in range(num_periods):
                    x_dep_key = f"{fid}_{p}"
                    x_arr_key = f"{fid}_{p}"
                    x_dep[x_dep_key] = int(x_dep_scen[sid][(fid, p)].x)
                    x_arr[x_arr_key] = int(x_arr_scen[sid][(fid, p)].x)
                    
                v_dep[fid] = int(v_dep_scen[sid][fid].x)
                v_arr[fid] = int(v_arr_scen[sid][fid].x)
                
            scenario_solutions.append({
                "scenario_id": sid,
                "x_dep": x_dep,
                "x_arr": x_arr,
                "v_dep": v_dep,
                "v_arr": v_arr
            })

        solution = {
            "objective_value": objective_value,
            "schedule": schedule,
            "scenario_solutions": scenario_solutions
        }
    else:
        # Return empty/dummy solution if not solved
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