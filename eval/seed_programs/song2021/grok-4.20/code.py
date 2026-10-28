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
        L = ord_data['lower_exec_interval_length_l']
        Ubar = ord_data['upper_exec_interval_length_l_bar']
        order_id = ord_data['order_id']

        intervals = []
        for s in range(r, H + 1):
            for f in range(s, H + 1):
                length = f - s + 1
                if L <= length <= Ubar:
                    intervals.append((s, f))
        feasible_intervals.append(intervals)

    # Create model
    model = gp.Model("OrderScheduling")
    model.Params.TimeLimit = args.time_limit
    model.Params.LogToConsole = 0
    model.Params.Threads = 1

    # Decision variables: x[o][k] = 1 if order o uses interval k
    x = []
    for o in range(n):
        x.append(model.addVars(len(feasible_intervals[o]), vtype=GRB.BINARY, name=f"x_{o}"))

    # y[o,t] = intensity of order o in period t
    y = {}
    for o in range(n):
        for t in range(1, H+1):
            y[o, t] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"y_{o}_{t}")

    # overtime
    overtime = {}
    for t in range(1, H+1):
        overtime[t] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"over_{t}")

    # Each order chooses exactly one interval
    for o in range(n):
        model.addConstr(gp.quicksum(x[o][k] for k in range(len(feasible_intervals[o]))) == 1, name=f"one_interval_{o}")

    # Intensity bounds and zero outside window
    for o in range(n):
        ord_data = orders[o]
        LB = ord_data['intensity_lower_bound_LB']
        UB = ord_data['intensity_upper_bound_UB']
        for k_idx, (s, f) in enumerate(feasible_intervals[o]):
            xvar = x[o][k_idx]
            for t in range(1, H+1):
                yvar = y[o, t]
                if s <= t <= f:
                    model.addConstr(yvar >= LB * xvar, name=f"lb_y_{o}_{k_idx}_{t}")
                    model.addConstr(yvar <= UB * xvar, name=f"ub_y_{o}_{k_idx}_{t}")
                else:
                    model.addConstr(yvar <= 1 - xvar, name=f"zero_out_{o}_{k_idx}_{t}")

    # Sum of intensities per order = 1
    for o in range(n):
        model.addConstr(gp.quicksum(y[o, t] for t in range(1, H+1)) == 1.0, name=f"sum_int_{o}")

    # Capacity constraints
    for t in range(1, H+1):
        total_work = gp.quicksum(y[o, t] * orders[o]['work_content_p'] for o in range(n))
        model.addConstr(total_work <= capacities[t-1] + overtime[t], name=f"cap_{t}")

    # Objective
    total_tard_cost = 0.0
    for o in range(n):
        ord_data = orders[o]
        w = ord_data['tardiness_cost_w']
        d = ord_data['due_date_d']
        for k_idx, (s, f) in enumerate(feasible_intervals[o]):
            tard = max(f - d, 0)
            total_tard_cost += tard * w * x[o][k_idx]

    total_ot_cost = sigma * gp.quicksum(overtime[t] for t in range(1, H+1))
    model.setObjective(total_tard_cost + total_ot_cost, GRB.MINIMIZE)

    # Logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    best_obj = float('inf')

    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj - 1e-6:
                best_obj = obj_val
                if logger:
                    logger.log(float(obj_val))

    # Optimize
    model.optimize(callback)

    # Extract solution
    if model.SolCount == 0:
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
            chosen_val = -1
            for k in range(len(feasible_intervals[o_idx])):
                val = x[o_idx][k].X
                if val > chosen_val:
                    chosen_val = val
                    chosen_k = k
            if chosen_k is None or chosen_val < 0.5:
                # Fallback: pick first
                chosen_k = 0
            s, f = feasible_intervals[o_idx][chosen_k]
            intensities = {}
            for t in range(s, f+1):
                val = y[o_idx, t].X
                if val > 1e-8:
                    intensities[str(t)] = round(float(val), 8)
            orders_sol.append({
                "order_id": int(order_id),
                "chosen_interval": [int(s), int(f)],
                "intensities": intensities
            })

        nonreg = {}
        for t in range(1, H+1):
            val = overtime[t].X
            if val > 1e-6:
                nonreg[str(t)] = round(float(val), 6)

    solution = {
        "objective_value": float(obj_val),
        "orders_solution": orders_sol,
        "nonregular_capacity_by_period": nonreg
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    if logger and model.SolCount > 0:
        logger.log(float(obj_val))

if __name__ == "__main__":
    main()