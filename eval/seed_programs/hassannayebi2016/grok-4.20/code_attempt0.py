import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    T = instance['parameters']['T_planning_horizon_minutes']
    D = instance['parameters']['D_num_train_services']
    NT = instance['parameters']['NT_num_demand_periods']
    theta = instance['parameters']['theta_period_length_minutes']
    h_min = instance['parameters']['h_min_minutes']
    h_max = instance['parameters']['h_max_minutes']
    C = instance['parameters']['C_train_capacity_passengers']

    demand_list = instance['demand_periods']
    lambda_p = [d['lambda_pax_per_min'] for d in demand_list]

    # Precompute cumulative arrivals up to start of each period
    cum_lambda = [0.0] * (NT + 1)
    for p in range(1, NT + 1):
        cum_lambda[p] = cum_lambda[p - 1] + theta * lambda_p[p - 1]

    total_demand = cum_lambda[NT]
    # We assume total_demand <= D * C, as per feasibility

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model = gp.Model("Train_Scheduling")
    model.Params.TimeLimit = args.time_limit
    model.Params.LogToConsole = 0
    model.Params.MIPGap = 0.01  # Allow some gap for larger instances
    model.Params.Threads = 1

    # Decision variables: departure times (continuous)
    t = model.addVars(D, lb=0, ub=T, vtype=GRB.CONTINUOUS, name="t")

    # Auxiliary: headways (including initial and final)
    h = model.addVars(D + 1, lb=0, ub=T, vtype=GRB.CONTINUOUS, name="h")

    # Assignment: binary x[d,k] = 1 if train d assigned to period k (0-based)
    x = model.addVars(D, NT, vtype=GRB.BINARY, name="x")

    # For cumulative passengers up to each departure
    cum_pax = model.addVars(D, lb=0, ub=total_demand + 1, vtype=GRB.CONTINUOUS, name="cum_pax")

    # Passengers boarding each train
    b = model.addVars(D, lb=0, ub=C, vtype=GRB.CONTINUOUS, name="b")

    # Waiting passengers just before each departure (after previous boardings)
    w = model.addVars(D + 1, lb=0, vtype=GRB.CONTINUOUS, name="w")
    # w[0] = 0 (before first train)

    # Set initial waiting = 0
    model.addConstr(w[0] == 0)

    # Headway definitions
    model.addConstr(h[0] == t[0])  # initial headway
    for d in range(1, D):
        model.addConstr(h[d] == t[d] - t[d - 1])
    model.addConstr(h[D] == T - t[D - 1])  # final headway

    # Headway bounds for interior intervals
    for d in range(1, D):
        model.addConstr(h[d] >= h_min)
        model.addConstr(h[d] <= h_max)

    # All headways sum to T
    model.addConstr(gp.quicksum(h[i] for i in range(D + 1)) == T)

    # Train ordering: departure times non-decreasing
    for d in range(1, D):
        model.addConstr(t[d] >= t[d - 1])

    # Assignment constraints
    for d in range(D):
        model.addConstr(gp.quicksum(x[d, k] for k in range(NT)) == 1)

    # Chronological ordering of assignments
    for d in range(D - 1):
        for k in range(NT):
            model.addConstr(gp.quicksum(x[d, p] for p in range(k + 1)) >= 
                           gp.quicksum(x[d + 1, p] for p in range(k + 1)))

    # Link departure time to assigned period
    period_start = [k * theta for k in range(NT)]
    period_end = [(k + 1) * theta for k in range(NT)]
    for d in range(D):
        for k in range(NT):
            model.addConstr(t[d] >= period_start[k] * x[d, k])
            model.addConstr(t[d] <= (period_end[k] - 1e-5) * x[d, k] + (1 - x[d, k]) * T)

    # Cumulative passengers up to departure of train d
    for d in range(D):
        cum_expr = gp.LinExpr()
        for k in range(NT):
            # full periods before k
            full_before = sum(theta * lambda_p[p] for p in range(k))
            # fractional in k: (t[d] - k*theta) * lambda_k
            frac = t[d] - k * theta
            cum_expr += x[d, k] * (full_before + frac * lambda_p[k])
        model.addConstr(cum_pax[d] == cum_expr)

    # Passengers arriving during each interval
    # arrivals[0] = cum_pax[0]
    # arrivals[d] = cum_pax[d] - cum_pax[d-1] for d=1..D-1
    # arrivals[D] = total_demand - cum_pax[D-1]

    arrivals = model.addVars(D + 1, lb=0, vtype=GRB.CONTINUOUS, name="arrivals")
    model.addConstr(arrivals[0] == cum_pax[0])
    for d in range(1, D):
        model.addConstr(arrivals[d] == cum_pax[d] - cum_pax[d - 1])
    model.addConstr(arrivals[D] == total_demand - cum_pax[D - 1])

    # Boarding and waiting balance
    for d in range(D):
        # waiting before boarding = w[d] + arrivals[d]
        # b[d] <= waiting before boarding
        # b[d] <= C (already in var ub)
        model.addConstr(b[d] <= w[d] + arrivals[d])
        # w[d+1] = w[d] + arrivals[d] - b[d]
        model.addConstr(w[d + 1] == w[d] + arrivals[d] - b[d])

    # All passengers must eventually board or be counted in final waiting, but since we have final interval,
    # final waiting w[D] should be <= arrivals[D] (but actually enforced by equations)
    # We also add constraint that total boarded = total demand (no one left behind)
    model.addConstr(gp.quicksum(b[d] for d in range(D)) == total_demand)

    # Objective: sum over intervals of h_i * (w_at_start_i + 0.5 * arrivals_i)
    # w_at_start_0 = 0
    # w_at_start_i for i>=1 = w[i]  (waiting just before train i departs, before it boards)
    # After train departs, waiting becomes w[i+1] = previous waiting after arrival minus boarding.
    # But during interval i (between departure i-1 and i), the starting waiting (right after previous departure) is w[i]
    # Yes.

    obj_expr = gp.LinExpr()
    # Initial interval (i=0): starts with w=0
    obj_expr += h[0] * (0 + 0.5 * arrivals[0])

    for i in range(1, D + 1):
        # For interval i (after departure i-1, before departure i or end)
        # Platform starts with w[i] (which is after previous departure's boarding)
        obj_expr += h[i] * (w[i] + 0.5 * arrivals[i])

    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Callback to log improved solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(obj)

    start_time = time.time()
    model.optimize(callback)

    # Extract solution
    if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
        dep_times = {}
        for d in range(D):
            val = t[d].X
            # round to nearest reasonable precision (0.01 min)
            dep_times[f"train_{d+1}"] = round(float(val), 4)

        objective = model.ObjVal

        solution = {
            "objective_value": float(objective),
            "departure_schedule_minutes": dep_times
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # If infeasible or no solution, write a dummy (should not happen for valid instances)
        solution = {
            "objective_value": 0.0,
            "departure_schedule_minutes": {f"train_{d+1}": 0.0 for d in range(D)}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()