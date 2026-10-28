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

    H = instance['planning_horizon_H']
    n = instance['num_orders_n']
    sigma = instance['unit_cost_nonregular_capacity_sigma']
    capacities = instance['capacities']
    orders = instance['orders']

    # Precompute feasible intervals for each order
    feasible_intervals = []
    for o_idx in range(n):
        ord_data = orders[o_idx]
        r = ord_data['release_date_r']
        d = ord_data['due_date_d']
        L = ord_data['lower_exec_interval_length_l']
        Ubar = ord_data['upper_exec_interval_length_l_bar']
        w = ord_data['work_content_p']
        LB = ord_data['intensity_lower_bound_LB']
        UB = ord_data['intensity_upper_bound_UB']
        tard_cost = ord_data['tardiness_cost_w']
        order_id = ord_data['order_id']

        intervals = []
        for s in range(r, H + 1):  # s from r to H (1-based? periods 1..H assumed)
            for f in range(s, H + 1):
                length = f - s + 1
                if length < L or length > Ubar:
                    continue
                # Check compatibility with bounds (though schema already provides L and Ubar)
                if length > 0 and (1.0 / length <= UB) and (1.0 / length >= LB or length == 1):
                    intervals.append((s, f))
        feasible_intervals.append(intervals)

    # Create model
    model = gp.Model("OrderScheduling")
    model.Params.TimeLimit = args.time_limit
    model.Params.LogToConsole = 0
    model.Params.Threads = 1
    model.Params.MIPGap = 0.01  # Allow some gap for speed if needed

    # Decision variables: x[o][k] = 1 if order o uses interval k
    x = []
    for o in range(n):
        x.append(model.addVars(len(feasible_intervals[o]), vtype=GRB.BINARY, name=f"x_{o}"))

    # y[p][t] = intensity of order p in period t (only for feasible)
    # But to avoid too many vars, we'll use a dict or compute on fly but for LP we need vars
    # Since intensities can be continuous, but to enforce bounds only in active windows, we use formulation with intervals

    # Actually since interval chosen determines the window, we can have intensity per (order, interval, period)
    # But that can be heavy. Since n is small in expected instances, and H probably <=100, we can do:
    # For each order, for each feasible interval, have a binary, and then have continuous intensity per period per order, linked.

    y = {}  # y[o,t] intensity of order o in period t
    for o in range(n):
        for t in range(1, H+1):
            y[o, t] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"y_{o}_{t}")

    # overtime
    overtime = model.addVars(H+1, lb=0, vtype=GRB.CONTINUOUS, name="overtime")

    # Constraints

    # Each order chooses exactly one interval
    for o in range(n):
        model.addConstr(gp.quicksum(x[o][k] for k in range(len(feasible_intervals[o]))) == 1, name=f"one_interval_{o}")

    # Intensity linking and bounds
    for o in range(n):
        ord_data = orders[o]
        LB = ord_data['intensity_lower_bound_LB']
        UB = ord_data['intensity_upper_bound_UB']
        p_work = ord_data['work_content_p']
        for k_idx, (s, f) in enumerate(feasible_intervals[o]):
            xvar = x[o][k_idx]
            for t in range(1, H+1):
                yvar = y[o, t]
                if s <= t <= f:
                    model.addConstr(yvar >= LB * xvar, name=f"lb_y_{o}_{k_idx}_{t}")
                    model.addConstr(yvar <= UB * xvar, name=f"ub_y_{o}_{k_idx}_{t}")
                else:
                    model.addConstr(yvar <= (1 - xvar), name=f"zero_out_{o}_{k_idx}_{t}")  # since UB<=1

    # Sum of intensities per order = 1
    for o in range(n):
        model.addConstr(gp.quicksum(y[o, t] for t in range(1, H+1)) == 1, name=f"sum_int_{o}")

    # Capacity and overtime
    for t in range(1, H+1):
        total_work = gp.quicksum(y[o, t] * orders[o]['work_content_p'] for o in range(n))
        model.addConstr(total_work <= capacities[t-1] + overtime[t], name=f"cap_{t}")
    
    # Objective
    total_tardiness_cost = 0
    for o in range(n):
        ord_data = orders[o]
        w = ord_data['tardiness_cost_w']
        d = ord_data['due_date_d']
        tard_contrib = 0
        for k_idx, (s, f) in enumerate(feasible_intervals[o]):
            tard = max(f - d, 0)
            tard_contrib += tard * w * x[o][k_idx]
        total_tardiness_cost += tard_contrib

    total_overtime_cost = sigma * gp.quicksum(overtime[t] for t in range(1, H+1))
    model.setObjective(total_tardiness_cost + total_overtime_cost, GRB.MINIMIZE)

    # Callback for logging incumbents
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    best_obj = None

    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
            if best_obj is None or obj_val < best_obj:
                best_obj = obj_val
                if logger:
                    logger.log(float(obj_val))

    # Optimize
    start_time = time.time()
    model.optimize(callback)

    # Extract solution
    if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.USER_INTERRUPT]:
        # Infeasible? Try to return something but assume feasible instances
        print("No feasible solution found.", file=sys.stderr)
        obj_val = 0.0
        orders_sol = []
        nonreg = {}
    else:
        obj_val = model.objVal
        orders_sol = []
        for o_idx in range(n):
            ord_data = orders[o_idx]
            order_id = ord_data['order_id']
            chosen_k = None
            for k in range(len(feasible_intervals[o_idx])):
                if x[o_idx][k].X > 0.5:
                    chosen_k = k
                    break
            if chosen_k is None:
                # Should not happen
                continue
            s, f = feasible_intervals[o_idx][chosen_k]
            intensities = {}
            for t in range(s, f+1):
                val = y[o_idx, t].X
                if val > 1e-6:
                    intensities[str(t)] = round(val, 6)  # clean float
            orders_sol.append({
                "order_id": order_id,
                "chosen_interval": [s, f],
                "intensities": intensities
            })

        nonreg = {}
        for t in range(1, H+1):
            val = overtime[t].X
            if val > 1e-5:
                nonreg[str(t)] = round(val, 6)

    solution = {
        "objective_value": float(obj_val),
        "orders_solution": orders_sol,
        "nonregular_capacity_by_period": nonreg
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    if logger:
        logger.log(float(obj_val))  # final log

if __name__ == "__main__":
    main()