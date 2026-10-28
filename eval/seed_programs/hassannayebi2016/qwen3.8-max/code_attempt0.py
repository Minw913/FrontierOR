import json
import argparse
import time
from gurobipy import *

def solve(instance_path, solution_path, time_limit, log_path=None):
    # Read instance
    with open(instance_path) as f:
        instance = json.load(f)
    
    params = instance['parameters']
    T = params['T_planning_horizon_minutes']
    D = params['D_num_train_services']
    NT = params['NT_num_demand_periods']
    theta = params['theta_period_length_minutes']
    h_min = params['h_min_minutes']
    h_max = params['h_max_minutes']
    C = params['C_train_capacity_passengers']
    
    # Arrival rates
    lambdas = [dp['lambda_pax_per_min'] for dp in instance['demand_periods']]
    
    # Precompute cumulative arrivals A(t) for t = 0, 1, ..., T
    A = [0.0] * (T + 1)
    for t in range(1, T + 1):
        period = min((t - 1) // theta, NT - 1)  # period at time t-1 to t
        # Actually, A(t) = cumulative arrivals from 0 to t
        # At time t, we're in period floor(t / theta) if t < T, but at t = T we've completed all periods
        # Let me compute more carefully
        pass
    
    # Recompute A(t) properly
    # A(t) = sum of arrivals from time 0 to time t
    # If t is in period k (k*theta <= t < (k+1)*theta), then
    # A(t) = sum_{j=0}^{k-1} lambda_j * theta + lambda_k * (t - k*theta)
    # At t = T = NT * theta, A(T) = sum_{j=0}^{NT-1} lambda_j * theta = total_demand
    
    A = [0.0] * (T + 1)
    for t in range(T + 1):
        k = t // theta  # period index
        if k >= NT:
            k = NT - 1
        # Sum of full periods before k
        cum = sum(lambdas[j] * theta for j in range(k))
        # Partial period k
        cum += lambdas[k] * (t - k * theta)
        A[t] = cum
    
    # Verify
    total_demand = sum(lambdas[j] * theta for j in range(NT))
    
    # Create Gurobi model
    model = Model("train_scheduling")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit
    
    # Binary variables: x[d,t] = 1 if train d departs at time t
    x = {}
    for d in range(D):
        for t in range(T):  # departure at time t, t = 0, ..., T-1
            x[d, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{d}_{t}")
    
    # Each train departs exactly once
    for d in range(D):
        model.addConstr(quicksum(x[d, t] for t in range(T)) == 1)
    
    # Departure time of train d
    # t_d = sum_t t * x[d,t]
    t_dep = {}
    for d in range(D):
        t_dep[d] = quicksum(t * x[d, t] for t in range(T))
    
    # Cumulative arrivals at departure of train d
    A_dep = {}
    for d in range(D):
        A_dep[d] = quicksum(A[t] * x[d, t] for t in range(T))
    
    # Headway constraints
    for d in range(1, D):
        model.addConstr(t_dep[d] - t_dep[d-1] >= h_min)
        model.addConstr(t_dep[d] - t_dep[d-1] <= h_max)
    
    # Waiting passengers
    W = {}
    for d in range(D):
        W[d] = model.addVar(lb=0, name=f"W_{d}")
    
    # W[0] = max(0, A_dep[0] - C)
    model.addConstr(W[0] >= A_dep[0] - C)
    
    # W[d] = max(0, W[d-1] + A_dep[d] - A_dep[d-1] - C)
    for d in range(1, D):
        model.addConstr(W[d] >= W[d-1] + A_dep[d] - A_dep[d-1] - C)
    
    # Objective
    obj = LinExpr()
    
    # Initial interval: t_0 * (0 + 0.5 * A_dep[0]) = 0.5 * t_dep[0] * A_dep[0]
    obj += 0.5 * t_dep[0] * A_dep[0]
    
    # Interior intervals
    for d in range(1, D):
        headway = t_dep[d] - t_dep[d-1]
        new_arrivals = A_dep[d] - A_dep[d-1]
        obj += headway * (W[d-1] + 0.5 * new_arrivals)
    
    # Final interval
    headway_final = T - t_dep[D-1]
    new_arrivals_final = total_demand - A_dep[D-1]
    obj += headway_final * (W[D-1] + 0.5 * new_arrivals_final)
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    model.optimize()
    
    # Extract solution
    if model.SolCount > 0:
        departure_times = {}
        for d in range(D):
            t_d = sum(t * x[d, t].X for t in range(T))
            departure_times[str(d)] = round(t_d, 6)
        
        obj_val = model.ObjVal
        
        solution = {
            "objective_value": obj_val,
            "departure_schedule_minutes": departure_times
        }
    else:
        # Fallback: evenly spaced
        departure_times = {}
        for d in range(D):
            t_d = (d + 1) * T / (D + 1)
            departure_times[str(d)] = round(t_d, 6)
        # Compute objective for this solution
        obj_val = compute_objective(departure_times, ...)
        solution = {
            "objective_value": obj_val,
            "departure_schedule_minutes": departure_times
        }
    
    # Write solution
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)
    
    return solution