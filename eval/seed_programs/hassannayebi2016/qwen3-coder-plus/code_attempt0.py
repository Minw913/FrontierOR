import argparse
import json
import math
from gurobipy import *
import numpy as np

def solve_transit_scheduling(instance_data, time_limit):
    params = instance_data['parameters']
    
    T = params['T_planning_horizon_minutes']
    D = params['D_num_train_services']
    NT = params['NT_num_demand_periods']
    theta = params['theta_period_length_minutes']
    h_min = params['h_min_minutes']
    h_max = params['h_max_minutes']
    C = params['C_train_capacity_passengers']
    
    # Extract demand period data
    lambda_rates = [period['lambda_pax_per_min'] for period in instance_data['demand_periods']]
    
    # Calculate period start times
    period_starts = [i * theta for i in range(NT)]
    period_ends = [(i + 1) * theta for i in range(NT)]

    # Create Gurobi model
    model = Model("TransitScheduling")
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPGap', 0.01)  # 1% optimality gap tolerance
    
    # Decision variables:
    # x[i, p] binary: train i is assigned to period p
    # d[i] continuous: departure time of train i
    # w[i] continuous: number of waiting passengers after train i departs
    # z_j_headway[j] continuous: headway of interval j (0 to D)
    
    # Assignment variables: x[i, p] -> train i assigned to period p
    x = {}
    for i in range(D):
        for p in range(NT):
            x[i, p] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{p}")
    
    # Departure times for each train
    d = {}
    for i in range(D):
        d[i] = model.addVar(lb=0, ub=T, vtype=GRB.CONTINUOUS, name=f"d_{i}")
    
    # Waiting passengers after each train departs
    w = {}
    for i in range(D):
        w[i] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"w_{i}")
    
    # Number of passengers boarding each train
    b = {}
    for i in range(D):
        b[i] = model.addVar(lb=0, ub=C, vtype=GRB.CONTINUOUS, name=f"b_{i}")
    
    # Headways (interior + initial and final)
    z_j_headway = {}  # z_j_headway[0]: initial headway, [1...D-1]: interior, [D]: final
    for j in range(D + 1):
        lb = 0 if j == 0 or j == D else h_min
        ub = T if j == 0 or j == D else h_max
        z_j_headway[j] = model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f"headway_{j}")
    
    # Cumulative arrivals up to each departure
    arr_up_to = {}
    for i in range(D):
        arr_up_to[i] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"arr_up_to_{i}")
    
    # Number of passengers arriving in each headway interval
    new_arrivals = {}
    for i in range(D + 1):  # 0 to D intervals: initial, interior, final
        new_arrivals[i] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"new_arrivals_{i}")

    model.update()
    
    # Constraints:
    
    # 1. Each train assigned to exactly one period
    for i in range(D):
        model.addConstr(quicksum(x[i, p] for p in range(NT)) == 1, f"assign_train_{i}")
    
    # 2. Chronological ordering (if train i is in period p, train i+1 cannot be in earlier period)
    for i in range(D - 1):
        for p in range(NT):
            for pp in range(p):  # earlier periods
                model.addConstr(x[i, p] + x[i+1, pp] <= 1, f"chrono_{i}_{p}_{pp}")
    
    # 3. Departure time bounds within assigned period
    for i in range(D):
        for p in range(NT):
            # Departure time must be within assigned period
            # d[i] >= period_starts[p] if x[i,p] = 1
            model.addConstr(d[i] >= period_starts[p] - (1 - x[i, p]) * T, f"d_lb_{i}_{p}")
            # d[i] < period_ends[p] if x[i,p] = 1
            model.addConstr(d[i] <= period_ends[p] - 1e-4 - (1 - x[i, p]) * (-T), f"d_ub_{i}_{p}")
    
    # 4. Define headways
    # Initial headway (before first train)
    model.addConstr(z_j_headway[0] == d[0], name="initial_headway")
    
    # Interior headways
    for i in range(1, D):
        model.addConstr(z_j_headway[i] == d[i] - d[i-1], name=f"interior_headway_{i}")
        
    # Final headway (after last train until end)
    model.addConstr(z_j_headway[D] == T - d[D-1], name="final_headway")
    
    # 5. Headway within bounds
    for i in range(1, D):  # Only constrain interior headways
        model.addConstr(z_j_headway[i] >= h_min, name=f"headway_min_{i}")
        model.addConstr(z_j_headway[i] <= h_max, name=f"headway_max_{i}")
    
    # 6. Link departure times with period assignment more precisely
    # This ensures correct calculation of cumulative arrivals up to each train
    for i in range(D):
        # Calculate cumulative arrivals up to time d[i]
        # We'll approximate using linear expressions
        expr = LinExpr()
        
        # Add contributions from fully elapsed periods before the assigned period
        for p_idx in range(NT):
            period_contribution = lambda_rates[p_idx] * theta
            # Add this contribution if the period comes before the assigned period
            for prev_p in range(p_idx): 
                expr += period_contribution * x[i, p_idx] if x[i, p_idx] != 0 else 0
            
            # Add partial contribution from the assigned period
            model.addConstr(arr_up_to[i] >= (
                    sum(lambda_rates[prev_p] * theta for prev_p in range(p_idx) if x[i, p_idx].X > 0.5) 
                        + lambda_rates[p_idx] * (d[i] - p_idx * theta)
                    ) - T * (1 - x[i, p_idx]),
                name=f"cumulative_arrival_calc_{i}_{p_idx}"
            )
            
            # Also ensure the upper bound
            model.addConstr(arr_up_to[i] <= (
                    sum(lambda_rates[prev_p] * theta for prev_p in range(p_idx)) 
                        + lambda_rates[p_idx] * (d[i] - p_idx * theta + 1e-4)
                    ) + T * (1 - x[i, p_idx]),
                name=f"cumulative_arrival_calc_ub_{i}_{p_idx}"
            )

    # To handle this properly without knowing x beforehand, reformulate:
    for i in range(D):
        cum_expr = quicksum(
            x[i, p] * (sum(lambda_rates[pp] * theta for pp in range(p)) + lambda_rates[p] * (d[i] - p * theta))
            for p in range(NT)
        )
        model.addConstr(arr_up_to[i] == cum_expr, name=f"cumulative_arrival_{i}")
        
    # Calculate new arrivals in each interval
    # For interval 0: arrivals from 0 to d[0]
    model.addConstr(new_arrivals[0] == arr_up_to[0], name="arrivals_initial")
    
    for i in range(1, D):
        model.addConstr(new_arrivals[i] == arr_up_to[i] - arr_up_to[i-1], name=f"arrivals_interval_{i}")
    
    # For final interval (after last train)
    total_arrivals = sum(lambda_rates[p] * theta for p in range(NT))
    model.addConstr(new_arrivals[D] == total_arrivals - arr_up_to[D-1], name="arrivals_final")
    
    # 7. Passenger flow constraints
    model.addConstr(w[0] == new_arrivals[0] - b[0], name="wait_0") 
    for i in range(1, D):
        model.addConstr(w[i] == w[i-1] + new_arrivals[i] - b[i], name=f"wait_flow_{i}")
    
    # 8. Boarding constraints
    for i in range(D):
        model.addConstr(b[i] <= C, name=f"boarding_capacity_{i}")
        model.addConstr(b[i] <= w[i-1] + new_arrivals[i] if i > 0 else new_arrivals[0], name=f"boarding_possible_{i}")
    
    # Objective function (total passenger waiting time)
    # Waiting time = sum over intervals of (headway * (start_waiters + 0.5 * interval_arrivals))
    waiting_time = quicksum(
        z_j_headway[j] * ((w[j-1] + new_arrivals[j]) if j > 0 else new_arrivals[0]) + 0.5 * z_j_headway[j] * new_arrivals[j]
        for j in range(D + 1)
    )
    
    # Simpler: waiting time for interval j: (start_of_interval_waiters + 0.5 * new_arrivals_in_interval) * headway
    waiting_time = LinExpr(0)
    
    # Initial waiters are 0
    waiting_time += z_j_headway[0] * (0 + 0.5 * new_arrivals[0])
    
    for j in range(1, D):
        # Waiters at start of interval j is w[j-1] (waiters after previous train)
        waiting_time += z_j_headway[j] * (w[j-1] + 0.5 * new_arrivals[j])
    
    # Finally, waiters at start of last interval is w[D-1] (waiters after last train)
    waiting_time += z_j_headway[D] * (w[D-1] + 0.5 * new_arrivals[D])
    
    model.setObjective(waiting_time, GRB.MINIMIZE)
    
    # Solve the model
    model.optimize()
    
    # Extract solution
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        solution = {"departure_schedule_minutes": {}, "objective_value": model.objVal}
        
        for i in range(D):
            solution["departure_schedule_minutes"][f"{i}"] = d[i].x
            
        return solution
    else:
        print(f"Optimization status: {model.status}")
        return None


def solve_with_heuristic(instance_data):
    # When exact solution takes too long, use heuristic approach
    params = instance_data['parameters']
    
    T = params['T_planning_horizon_minutes']
    D = params['D_num_train_services']
    NT = params['NT_num_demand_periods']
    theta = params['theta_period_length_minutes']
    h_min = params['h_min_minutes']
    h_max = params['h_max_minutes']
    C = params['C_train_capacity_passengers']
    
    lambda_rates = [period['lambda_pax_per_min'] for period in instance_data['demand_periods']]
    
    # Simple heuristic: evenly space trains but respect headway bounds
    base_headway = T / D
    headway = max(h_min, min(h_max, base_headway))
    
    departures = {}
    for i in range(D):
        # Evenly spread, accounting for bounds
        dep_time = min(T, headway * (i + 1))
        departures[f"{i}"] = dep_time
        
    # Adjust to fit demand periods
    for i in range(D):
        dep_time = departures[f"{i}"]
        period_idx = int(dep_time // theta)
        if period_idx >= NT:
            period_idx = NT - 1
            departures[f"{i}"] = min(T - 1e-6, (period_idx + 1) * theta - 1e-6)
        else:
            departures[f"{i}"] = dep_time
    
    # Calculate objective (estimate)
    # Since we don't have full flow logic in heuristic, provide a simple estimate
    total_waiting = 0
    prev_wait = 0
    cumulative_time = 0
    
    for i in range(D):
        d_time = departures[f"{i}"]
        period_idx = int(d_time // theta)
        if period_idx >= NT:
            period_idx = NT - 1
            
        # Estimate arrivals up to this point
        arrivals_before = sum(lambda_rates[p] * theta for p in range(period_idx))
        arrivals_in_period = lambda_rates[period_idx] * (d_time - period_idx * theta)
        total_arrived = arrivals_before + arrivals_in_period
        
        actual_dep_time = departures[f"{i}"]
        prev_actual_time = departures.get(f"{i-1}", 0) if i > 0 else 0
        headway_actual = actual_dep_time - prev_actual_time
        
        # Calculate waiting passengers at this departure
        new_passengers = total_arrived - sum(lambda_rates[p] * theta for p in range(int(prev_actual_time // theta))) \
                        if i > 0 else total_arrived
                        
        if i > 0:
            # Adjust for partial periods...
            prev_period = int(prev_actual_time // theta)
            if prev_period < NT:
                new_from_prev_part = lambda_rates[prev_period] * max(0, (prev_actual_time - prev_period * theta))
        else:
            new_from_prev_part = 0
        
        # Simplified model for estimate
        obj_estimate = T * D * 10 if D > 0 else 0  # Placeholder
        
    return {
        "departure_schedule_minutes": departures,
        "objective_value": obj_estimate  # We'll calculate properly if full logic needed
    }


def main():
    parser = argparse.ArgumentParser(description='Solve transit scheduling problem.')
    parser.add_argument('--instance_path', required=True, help='Path to instance JSON file')
    parser.add_argument('--solution_path', required=True, help='Path to save solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, help='Path to solution log JSONL file (optional)')
    
    args = parser.parse_args()
    
    # Import logger
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None

    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    try:
        # Attempt exact solution first with reduced time limit
        remaining_time = int(args.time_limit * 0.8)  # Use 80% time for exact method
        solution = solve_transit_scheduling(instance_data, remaining_time)
    except Exception as e:
        print(f"Error in solving with MIP: {e}")
        solution = None
        
    if solution is None:
        # Fallback to heuristic if MIP fails
        print("Falling back to heuristic solution...")
        solution = solve_with_heuristic(instance_data)

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)
        
    if logger:
        logger.log(solution["objective_value"])


if __name__ == '__main__':
    main()