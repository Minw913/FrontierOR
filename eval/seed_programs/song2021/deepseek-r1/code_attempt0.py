import json
import argparse
import gurobipy as gp
from solution_logger import SolutionLogger

def fallback_solution(orders, H, capacities, sigma):
    R = [0.0] * H
    orders_solution = []
    total_tardiness = 0.0
    total_nonregular = 0.0
    nonregular_dict = {}

    for order in orders:
        p_i = order['work_content_p']
        l_i = order['lower_exec_interval_length_l']
        l_bar_i = order['upper_exec_interval_length_l_bar']
        r_i = order['release_date_r']
        d_i = order['due_date_d']
        w_i = order['tardiness_cost_w']
        LB_i = order['intensity_lower_bound_LB']
        UB_i = order['intensity_upper_bound_UB']
        
        if H - r_i < l_i:
            start = max(r_i, H - l_i)
            if start < r_i:
                start = r_i
                L = H - r_i
            else:
                L = l_i
            end = start + L - 1
        else:
            L = min(l_bar_i, H - r_i)
            if L < l_i:
                start = max(r_i, H - l_i)
                if start < r_i:
                    start = r_i
                    L = H - r_i
                else:
                    L = l_i
                end = start + L - 1
            else:
                start = r_i
                end = start + L - 1
        
        intensity_per_period = 1.0 / L if L > 0 else 0.0
        for t in range(start, end + 1):
            if t < H:
                R[t] += intensity_per_period * p_i
        
        tardiness = max(0, end - d_i)
        total_tardiness += w_i * tardiness
        
        intensities_dict = {}
        for t in range(start, end + 1):
            intensities_dict[str(t)] = intensity_per_period
        
        orders_solution.append({
            "order_id": order['order_id'],
            "chosen_interval": [start, end],
            "intensities": intensities_dict
        })
    
    for t in range(H):
        nt_val = max(0, R[t] - capacities[t])
        if nt_val > 1e-6:
            nonregular_dict[str(t)] = nt_val
        total_nonregular += nt_val
    
    objective_value = total_tardiness + sigma * total_nonregular
    return orders_solution, nonregular_dict, objective_value

def main():
    parser = argparse.ArgumentParser(description='Order Scheduling Optimization')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to the output solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to the log file for intermediate solutions')
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    H = instance['planning_horizon_H']
    n = instance['num_orders_n']
    sigma = instance['unit_cost_nonregular_capacity_sigma']
    capacities = instance['capacities']
    orders = instance['orders']
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    model = gp.Model()
    model.setParam('OutputFlag', 1)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    
    s = {}
    e = {}
    z = {}
    a = {}
    T_var = {}
    N_var = [0] * H
    
    for i in range(n):
        for t in range(H):
            s[i, t] = model.addVar(vtype=gp.GRB.BINARY, name=f"s_{i}_{t}")
            e[i, t] = model.addVar(vtype=gp.GRB.BINARY, name=f"e_{i}_{t}")
            z[i, t] = model.addVar(lb=0, ub=1, vtype=gp.GRB.CONTINUOUS, name=f"z_{i}_{t}")
            UB_i = orders[i]['intensity_upper_bound_UB']
            a[i, t] = model.addVar(lb=0, ub=UB_i, vtype=gp.GRB.CONTINUOUS, name=f"a_{i}_{t}")
        T_var[i] = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"T_{i}")
    
    for t in range(H):
        N_var[t] = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"N_{t}")
    
    for i in range(n):
        order = orders[i]
        l_i = order['lower_exec_interval_length_l']
        l_bar_i = order['upper_exec_interval_length_l_bar']
        LB_i = order['intensity_lower_bound_LB']
        UB_i = order['intensity_upper_bound_UB']
        r_i = order['release_date_r']
        d_i = order['due_date_d']
        w_i = order['tardiness_cost_w']
        p_i = order['work_content_p']
        
        model.addConstr(gp.quicksum(s[i, t] for t in range(H)) == 1, name=f"C1_{i}")
        model.addConstr(gp.quicksum(e[i, t] for t in range(H)) == 1, name=f"C2_{i}")
        model.addConstr(z[i, 0] == s[i, 0], name=f"C3_{i}")
        
        for t in range(1, H):
            model.addConstr(z[i, t] == z[i, t-1] + s[i, t] - e[i, t-1], name=f"C4_{i}_{t}")
        
        T_s = gp.quicksum(t * s[i, t] for t in range(H))
        T_e = gp.quicksum(t * e[i, t] for t in range(H))
        model.addConstr(T_e - T_s + 1 >= l_i, name=f"C5_lb_{i}")
        model.addConstr(T_e - T_s + 1 <= l_bar_i, name=f"C5_ub_{i}")
        model.addConstr(T_s >= r_i, name=f"C6_{i}")
        
        for t in range(H):
            model.addConstr(a[i, t] >= LB_i * z[i, t], name=f"C7a_{i}_{t}")
            model.addConstr(a[i, t] <= UB_i * z[i, t], name=f"C7b_{i}_{t}")
        
        model.addConstr(gp.quicksum(a[i, t] for t in range(H)) == 1, name=f"C8_{i}")
        model.addConstr(T_var[i] >= T_e - d_i, name=f"Tardiness_{i}")
    
    for t in range(H):
        model.addConstr(N_var[t] >= gp.quicksum(a[i, t] * orders[i]['work_content_p'] for i in range(n)) - capacities[t], name=f"N_def_{t}")
    
    total_tardiness_cost = gp.quicksum(T_var[i] * orders[i]['tardiness_cost_w'] for i in range(n))
    total_nonregular_cost = sigma * gp.quicksum(N_var[t] for t in range(H))
    model.setObjective(total_tardiness_cost + total_nonregular_cost, gp.GRB.MINIMIZE)
    
    def mycallback(model, where):
        if where == gp.GRB.Callback.MIPSOL:
            objval = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
            if model._logger:
                model._logger.log(objval)
    
    if logger:
        model._logger = logger
        model.optimize(mycallback)
    else:
        model.optimize()
    
    if model.SolCount > 0:
        orders_solution = []
        nonregular_dict = {}
        for t in range(H):
            nt_val = N_var[t].X
            if nt_val > 1e-6:
                nonregular_dict[str(t)] = nt_val
        
        for i in range(n):
            start, end = None, None
            for t in range(H):
                if s[i, t].X > 0.5:
                    start = t
                if e[i, t].X > 0.5:
                    end = t
            if start is None or end is None:
                active_periods = [t for t in range(H) if z[i, t].X > 0.5]
                if active_periods:
                    start = min(active_periods)
                    end = max(active_periods)
                else:
                    start, end, _ = fallback_solution([orders[i]], H, capacities, sigma)[0][0]['chosen_interval']
            intensities_dict = {}
            for t in range(start, end + 1):
                if t < H:
                    intensities_dict[str(t)] = a[i, t].X
            orders_solution.append({
                "order_id": orders[i]['order_id'],
                "chosen_interval": [start, end],
                "intensities": intensities_dict
            })
        objective_value = model.ObjVal
    else:
        orders_solution, nonregular_dict, objective_value = fallback_solution(orders, H, capacities, sigma)
    
    solution = {
        "objective_value": objective_value,
        "orders_solution": orders_solution,
        "nonregular_capacity_by_period": nonregular_dict
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()