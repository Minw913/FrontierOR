import argparse
import json
import math
import time
from itertools import product

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    H = data['planning_horizon_H']
    n = data['num_orders_n']
    sigma = data['unit_cost_nonregular_capacity_sigma']
    capacities = data['capacities']
    orders = data['orders']
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    start_time = time.time()
    
    import gurobipy as gp
    from gurobipy import GRB
    
    model = gp.Model("order_scheduling")
    model.Params.TimeLimit = max(1, time_limit - 5)
    model.Params.Threads = 1
    model.Params.MIPFocus = 1  # Focus on finding feasible solutions
    
    # Precompute feasible intervals for each order
    # An interval is (start, finish) where start >= release_date, finish <= H
    # length = finish - start + 1
    # length must be in [lower_exec_interval_length, upper_exec_interval_length]
    
    order_intervals = {}  # order_id -> list of (start, finish) tuples
    
    for order in orders:
        oid = order['order_id']
        r = order['release_date_r']
        l_min = order['lower_exec_interval_length_l']
        l_max = order['upper_exec_interval_length_l_bar']
        
        intervals = []
        for length in range(l_min, l_max + 1):
            for start in range(r, H - length + 2):  # start from r, finish = start + length - 1 <= H
                finish = start + length - 1
                if finish <= H:
                    intervals.append((start, finish))
        order_intervals[oid] = intervals
    
    # Decision variables:
    # y[oid, s, f] = 1 if order oid uses interval [s, f]
    # x[oid, t] = intensity of order oid in period t
    # o[t] = nonregular capacity in period t
    
    y = {}
    for order in orders:
        oid = order['order_id']
        for (s, f) in order_intervals[oid]:
            y[oid, s, f] = model.addVar(vtype=GRB.BINARY, name=f"y_{oid}_{s}_{f}")
    
    x = {}
    for order in orders:
        oid = order['order_id']
        r = order['release_date_r']
        for (s, f) in order_intervals[oid]:
            for t in range(s, f + 1):
                if (oid, t) not in x:
                    x[oid, t] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"x_{oid}_{t}")
    
    # Nonregular capacity variables
    o = {}
    for t in range(1, H + 1):
        o[t] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"o_{t}")
    
    # Tardiness variables
    tardiness = {}
    for order in orders:
        oid = order['order_id']
        tardiness[oid] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"tard_{oid}")
    
    model.update()
    
    # Constraint 1: Each order selects exactly one interval
    for order in orders:
        oid = order['order_id']
        model.addConstr(
            gp.quicksum(y[oid, s, f] for (s, f) in order_intervals[oid]) == 1,
            name=f"one_interval_{oid}"
        )
    
    # Constraint 2: Intensity sum equals 1 for each order
    for order in orders:
        oid = order['order_id']
        # Collect all periods where x[oid, t] exists
        periods_for_order = set()
        for (s, f) in order_intervals[oid]:
            for t in range(s, f + 1):
                periods_for_order.add(t)
        model.addConstr(
            gp.quicksum(x[oid, t] for t in periods_for_order) == 1,
            name=f"sum_intensity_{oid}"
        )
    
    # Constraint 3: Intensity bounds - x[oid,t] must be between LB and UB if active, 0 otherwise
    # x[oid, t] <= UB * sum of y[oid,s,f] that cover period t
    # x[oid, t] >= LB * sum of y[oid,s,f] that cover period t
    
    for order in orders:
        oid = order['order_id']
        UB = order['intensity_upper_bound_UB']
        LB = order['intensity_lower_bound_LB']
        
        # Group intervals by which periods they cover
        period_to_intervals = {}
        for (s, f) in order_intervals[oid]:
            for t in range(s, f + 1):
                if t not in period_to_intervals:
                    period_to_intervals[t] = []
                period_to_intervals[t].append((s, f))
        
        for t, intervals_covering in period_to_intervals.items():
            active = gp.quicksum(y[oid, s, f] for (s, f) in intervals_covering)
            model.addConstr(x[oid, t] <= UB * active, name=f"ub_{oid}_{t}")
            model.addConstr(x[oid, t] >= LB * active, name=f"lb_{oid}_{t}")
    
    # Also ensure x[oid, t] = 0 for periods not covered by any selected interval
    # This is handled by the UB constraint: if no interval covers t, active=0, so x<=0
    
    # Constraint 4: Tardiness definition
    # tardiness[oid] >= finish - due_date for the selected interval
    for order in orders:
        oid = order['order_id']
        d = order['due_date_d']
        for (s, f) in order_intervals[oid]:
            tard_val = max(0, f - d)
            if tard_val > 0:
                model.addConstr(
                    tardiness[oid] >= tard_val * y[oid, s, f],
                    name=f"tard_{oid}_{s}_{f}"
                )
    
    # Constraint 5: Resource capacity and nonregular capacity
    for t in range(1, H + 1):
        # Collect all x[oid, t] that exist
        terms = []
        for order in orders:
            oid = order['order_id']
            p = order['work_content_p']
            if (oid, t) in x:
                terms.append(x[oid, t] * p)
        
        if terms:
            model.addConstr(
                gp.quicksum(terms) - o[t] <= capacities[t - 1],
                name=f"capacity_{t}"
            )
        else:
            # No orders can be active in period t
            pass
    
    # Objective: minimize total weighted tardiness + total nonregular capacity cost
    obj = gp.quicksum(order['tardiness_cost_w'] * tardiness[order['order_id']] for order in orders)
    obj += sigma * gp.quicksum(o[t] for t in range(1, H + 1))
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Callback for logging
    best_obj = [float('inf')]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0] - 1e-6:
                best_obj[0] = obj_val
                if logger:
                    logger.log(obj_val)
    
    model.optimize(callback)
    
    # Extract solution
    if model.SolCount > 0:
        objective_value = model.ObjVal
        
        orders_solution = []
        for order in orders:
            oid = order['order_id']
            
            # Find chosen interval
            chosen_s, chosen_f = None, None
            for (s, f) in order_intervals[oid]:
                if y[oid, s, f].X > 0.5:
                    chosen_s, chosen_f = s, f
                    break
            
            # Get intensities
            intensities = {}
            for t in range(chosen_s, chosen_f + 1):
                val = x[oid, t].X
                if val > 1e-10:
                    intensities[str(t)] = round(val, 10)
            
            orders_solution.append({
                "order_id": oid,
                "chosen_interval": [chosen_s, chosen_f],
                "intensities": intensities
            })
        
        # Nonregular capacity
        nonregular = {}
        for t in range(1, H + 1):
            val = o[t].X
            if val > 1e-10:
                nonregular[str(t)] = round(val, 10)
        
        solution = {
            "objective_value": objective_value,
            "orders_solution": orders_solution,
            "nonregular_capacity_by_period": nonregular
        }
    else:
        # No feasible solution found - return empty
        solution = {
            "objective_value": float('inf'),
            "orders_solution": [],
            "nonregular_capacity_by_period": {}
        }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)