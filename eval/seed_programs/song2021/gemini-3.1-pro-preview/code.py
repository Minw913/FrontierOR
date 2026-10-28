import sys
import json
import argparse
import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser(description="Orders Scheduling Optimization")
    parser.add_argument('--instance_path', type=str, required=True, help="Path to instance JSON file")
    parser.add_argument('--solution_path', type=str, required=True, help="Path to output solution JSON file")
    parser.add_argument('--time_limit', type=int, required=True, help="Time limit in seconds")
    parser.add_argument('--log_path', type=str, default=None, help="Path to log intermediate solutions")
    args = parser.parse_args()

    # Initialize Logger
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    # Load instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    max_H = data["planning_horizon_H"]
    n = data["num_orders_n"]
    sigma = data["unit_cost_nonregular_capacity_sigma"]
    capacities = data["capacities"]
    orders = data["orders"]

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 1)
    env.start()
    
    model = gp.Model("Scheduling", env=env)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)

    # Variables mapping
    s = {}
    e = {}
    y = {}
    z = {}
    T = {}
    O_var = {}

    for order in orders:
        i = order["order_id"]
        r_i = order["release_date_r"]
        l_i = order["lower_exec_interval_length_l"]
        
        # Start variables valid bounded from r_i to max_H - l_i
        for t in range(r_i, max_H - l_i + 1):
            s[i, t] = model.addVar(vtype=GRB.BINARY, name=f"s_{i}_{t}")
            
        # End variables valid bounded from r_i + l_i - 1 to max_H - 1
        for t in range(r_i + l_i - 1, max_H):
            e[i, t] = model.addVar(vtype=GRB.BINARY, name=f"e_{i}_{t}")
            
        # Ongoing status y and intensity z are valid from r_i to max_H - 1
        for t in range(r_i, max_H):
            y[i, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{t}")
            z[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"z_{i}_{t}")
            
        T[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"T_{i}")

    for t in range(max_H):
        O_var[t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"O_{t}")

    # Objective function: Minimize Weighted Tardiness + Nonregular Capacity Cost
    obj_expr = gp.LinExpr()
    for order in orders:
        i = order["order_id"]
        w_i = order["tardiness_cost_w"]
        obj_expr += w_i * T[i]

    for t in range(max_H):
        obj_expr += sigma * O_var[t]

    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Constraints for each order
    for order in orders:
        i = order["order_id"]
        r_i = order["release_date_r"]
        d_i = order["due_date_d"]
        l_i = order["lower_exec_interval_length_l"]
        l_bar_i = order["upper_exec_interval_length_l_bar"]
        LB_i = order["intensity_lower_bound_LB"]
        UB_i = order["intensity_upper_bound_UB"]

        # Exactly one execution interval start and end assigned
        s_sum = gp.quicksum(s.get((i, t), 0) for t in range(r_i, max_H - l_i + 1))
        e_sum = gp.quicksum(e.get((i, t), 0) for t in range(r_i + l_i - 1, max_H))
        model.addConstr(s_sum == 1, name=f"one_start_{i}")
        model.addConstr(e_sum == 1, name=f"one_end_{i}")

        # Active periods flow sequence maintaining contiguity via logic bounds
        for t in range(r_i, max_H):
            s_val = s.get((i, t), 0)
            e_val = e.get((i, t-1), 0)
            y_prev = y.get((i, t-1), 0)
            model.addConstr(y[i, t] - y_prev == s_val - e_val, name=f"flow_{i}_{t}")

        # Intensities distribution bounds must strictly equal 1 across lifetime
        model.addConstr(gp.quicksum(z.get((i, t), 0) for t in range(r_i, max_H)) == 1, name=f"tot_int_{i}")

        # Execution intensity window constraints
        for t in range(r_i, max_H):
            model.addConstr(z[i, t] >= LB_i * y[i, t], name=f"LB_{i}_{t}")
            model.addConstr(z[i, t] <= UB_i * y[i, t], name=f"UB_{i}_{t}")

        # Admissible limits placed on periods engaged bounds
        y_sum = gp.quicksum(y.get((i, t), 0) for t in range(r_i, max_H))
        model.addConstr(y_sum >= l_i, name=f"min_length_{i}")
        model.addConstr(y_sum <= l_bar_i, name=f"max_length_{i}")

        # Computes tardiness evaluating valid penalty timeline
        F_expr = gp.quicksum(t * e.get((i, t), 0) for t in range(r_i + l_i - 1, max_H))
        model.addConstr(T[i] >= F_expr - d_i, name=f"tardiness_{i}")

    # Aggregating period overhead resource cost bindings 
    for t in range(max_H):
        U_t = gp.LinExpr()
        for order in orders:
            i = order["order_id"]
            if t >= order["release_date_r"]:
                U_t += order["work_content_p"] * z[i, t]
        
        model.addConstr(O_var[t] >= U_t - capacities[t], name=f"overtime_bound_{t}")

    # Callback structure evaluating interim best candidates
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger is not None:
                logger.log(obj)

    model.optimize(callback)

    # Reconstruct optimal / near-optimum solution output
    if model.SolCount > 0:
        sol_orders = []
        for order in orders:
            i = order["order_id"]
            r_i = order["release_date_r"]
            l_i = order["lower_exec_interval_length_l"]
            
            start_p = -1
            for t in range(r_i, max_H - l_i + 1):
                if s.get((i, t)) is not None and s[i, t].X > 0.5:
                    start_p = t
                    break
                    
            finish_p = -1
            for t in range(r_i + l_i - 1, max_H):
                if e.get((i, t)) is not None and e[i, t].X > 0.5:
                    finish_p = t
                    break
                    
            intensities = {}
            for t in range(r_i, max_H):
                if y.get((i, t)) is not None and y[i, t].X > 0.5:
                    intensities[str(t)] = z[i, t].X
                    
            sol_orders.append({
                "order_id": i,
                "chosen_interval": [start_p, finish_p],
                "intensities": intensities
            })
            
        nonreg_cap = {}
        for t in range(max_H):
            val = O_var[t].X
            if val > 1e-5:
                nonreg_cap[str(t)] = val
                
        solution = {
            "objective_value": model.ObjVal,
            "orders_solution": sol_orders,
            "nonregular_capacity_by_period": nonreg_cap
        }
        
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        print("No feasible solution was found within the computation limits.")
        sys.exit(1)

if __name__ == '__main__':
    main()