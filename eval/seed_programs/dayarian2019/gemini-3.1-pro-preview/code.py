import json
import argparse
import math
import gurobipy as gp
from gurobipy import GRB

def solve_catering_problem(instance_path, solution_path, time_limit, log_path):
    # Parse instance
    with open(instance_path, 'r') as f:
        data = json.load(f)
        
    num_orders = data["num_orders"]
    num_products = data["num_products"]
    global_params = data["global_parameters"]
    orders = data["orders"]
    products = data["products"]
    periods = data["periods"]
    subperiods = data["subperiods"]
    travel = data["travel_time_matrix"]
    vehicle_capacity = data["vehicle_capacity"]
    
    N_nodes = num_orders
    # 1-based indexing for orders to match travel matrix
    O_idx = list(range(1, N_nodes + 1))
    nodes = list(range(N_nodes + 1))
    
    M_periods = global_params["num_periods"]
    SP = global_params["num_subperiods"]
    K = data["num_vehicles"]
    T = global_params["max_trips_per_route"]
    
    L = global_params["setup_length_subperiods"]
    sub_per_period = global_params["subperiods_per_period"]
    
    prod_idx = {p['product_id']: i for i, p in enumerate(products)}
    subp_id_to_idx = {sp["subperiod_id"]: i for i, sp in enumerate(subperiods)}
    
    # Initialize logger
    logger = None
    if log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(log_path, sense="minimize")
        
    m = gp.Model("Catering_Optimization")
    m.setParam('Seed', 0)
    m.setParam('MIPGap', 1e-4)
    m.setParam('NumericFocus', 0)
    m.setParam('Threads', 1)
    if time_limit:
        m.setParam('TimeLimit', time_limit)
        
    # --- Variables ---
    # Routing
    y = m.addVars(K, T, O_idx, vtype=GRB.BINARY, name="y")
    x = m.addVars(K, T, nodes, nodes, vtype=GRB.BINARY, name="x")
    is_used = m.addVars(K, T, vtype=GRB.BINARY, name="is_used")
    U = m.addVars(K, T, M_periods, vtype=GRB.BINARY, name="U")
    
    A = m.addVars(O_idx, vtype=GRB.CONTINUOUS, name="A")
    D = m.addVars(K, T, vtype=GRB.CONTINUOUS, name="D")
    L_time = m.addVars(K, T, vtype=GRB.CONTINUOUS, name="L_time")
    Ret = m.addVars(K, T, vtype=GRB.CONTINUOUS, name="Ret")
    
    RS = m.addVars(K, vtype=GRB.CONTINUOUS, name="RS")
    RE = m.addVars(K, vtype=GRB.CONTINUOUS, name="RE")
    RD = m.addVars(K, vtype=GRB.CONTINUOUS, name="RD")
    DriverCost = m.addVars(K, vtype=GRB.CONTINUOUS, name="DriverCost")
    
    # Production
    E = m.addVars(num_products, SP, vtype=GRB.INTEGER, lb=0, name="E")
    S = m.addVars(num_products, SP, vtype=GRB.INTEGER, lb=0, name="S")
    W = m.addVars(M_periods, vtype=GRB.INTEGER, lb=0, name="W")
    X = m.addVars(M_periods, vtype=GRB.INTEGER, lb=0, name="X")
    V = m.addVars(O_idx, M_periods, vtype=GRB.BINARY, name="V")
    w_var = m.addVars(K, T, O_idx, M_periods, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="w_var")
    
    # --- Constraints ---
    # 1. Routing
    for i in O_idx:
        m.addConstr(gp.quicksum(y[k, t, i] for k in range(K) for t in range(T)) == 1)
        
    BIG_M = global_params["horizon_end_minutes"] + 100000
    for k in range(K):
        for t in range(T):
            m.addConstr(gp.quicksum(U[k, t, m_p] for m_p in range(M_periods)) == is_used[k, t])
            
            if t < T - 1:
                m.addConstr(is_used[k, t+1] <= is_used[k, t])
                m.addConstr(gp.quicksum(m_p * U[k, t+1, m_p] for m_p in range(M_periods)) >= 
                            gp.quicksum(m_p * U[k, t, m_p] for m_p in range(M_periods)) + 1 - M_periods * (1 - is_used[k, t+1]))
                
            m.addConstr(D[k, t] == gp.quicksum(periods[m_p]["start_time"] * U[k, t, m_p] for m_p in range(M_periods)))
            m.addConstr(L_time[k, t] == D[k, t] + global_params["preparation_time_minutes"] * is_used[k, t])
            
            m.addConstr(gp.quicksum(x[k, t, 0, j] for j in O_idx) == is_used[k, t])
            m.addConstr(gp.quicksum(x[k, t, i, 0] for i in O_idx) == is_used[k, t])
            
            for i in O_idx:
                m.addConstr(gp.quicksum(x[k, t, i, j] for j in nodes if i != j) == y[k, t, i])
                m.addConstr(gp.quicksum(x[k, t, j, i] for j in nodes if i != j) == y[k, t, i])
                m.addConstr(x[k, t, i, i] == 0)
            m.addConstr(x[k, t, 0, 0] == 0)
            
            m.addConstr(gp.quicksum(orders[i-1]["space_requirement"] * y[k, t, i] for i in O_idx) <= vehicle_capacity * is_used[k, t])
            
            for j in O_idx:
                m.addConstr(A[j] >= L_time[k, t] + travel[0][j] - BIG_M * (1 - x[k, t, 0, j]))
                m.addConstr(Ret[k, t] >= A[j] + orders[j-1]["service_time"] + travel[j][0] - BIG_M * (1 - x[k, t, j, 0]))
                for i in O_idx:
                    if i != j:
                        m.addConstr(A[j] >= A[i] + orders[i-1]["service_time"] + travel[i][j] - BIG_M * (1 - x[k, t, i, j]))
                            
        # Route logic and driver costs
        m.addConstr(RS[k] == gp.quicksum(periods[m_p]["start_time"] * U[k, 0, m_p] for m_p in range(M_periods)))
        for t in range(T):
            m.addConstr(RE[k] >= Ret[k, t])
            # Span bound
            m.addConstr(gp.quicksum(m_p * U[k, t, m_p] for m_p in range(M_periods)) - 
                        gp.quicksum(m_p * U[k, 0, m_p] for m_p in range(M_periods)) + 1 <= 
                        global_params["max_shift_route_length_periods"] + M_periods * (1 - is_used[k, t]))
            
        m.addConstr(RD[k] >= RE[k] - RS[k])
        m.addConstr(RD[k] <= global_params["max_shift_route_length_minutes"] * is_used[k, 0])
        
        m.addConstr(DriverCost[k] >= global_params["driver_salary_per_minute"] * RD[k])
        m.addConstr(DriverCost[k] >= global_params["driver_salary_per_minute"] * global_params["min_paid_time_minutes"] * is_used[k, 0])
        
        if k > 0:
            m.addConstr(is_used[k, 0] <= is_used[k-1, 0])

    for i in O_idx:
        m.addConstr(A[i] >= orders[i-1]["time_window_start"])
        m.addConstr(A[i] <= orders[i-1]["time_window_end"])
        
    for i in O_idx:
        for m_p in range(M_periods):
            m.addConstr(V[i, m_p] == gp.quicksum(w_var[k, t, i, m_p] for k in range(K) for t in range(T)))
        for k in range(K):
            for t in range(T):
                for m_p in range(M_periods):
                    m.addConstr(w_var[k, t, i, m_p] <= y[k, t, i])
                    m.addConstr(w_var[k, t, i, m_p] <= U[k, t, m_p])
                    m.addConstr(w_var[k, t, i, m_p] >= y[k, t, i] + U[k, t, m_p] - 1)
                    
    # 2. Production
    for p_idx in range(num_products):
        for t in range(SP):
            prev_E = E[p_idx, t-1] if t > 0 else 0
            S_idx = t - L
            if S_idx >= 0:
                m.addConstr(E[p_idx, t] - prev_E <= S[p_idx, S_idx])
            else:
                m.addConstr(E[p_idx, t] - prev_E <= 0)
                
    Q = {}
    z = {}
    for o_idx, order in enumerate(orders):
        o = order["order_id"]
        i = o_idx + 1
        for p_id_str, qty in order["demands"].items():
            if qty <= 0: continue
            p_id = int(p_id_str)
            p_idx = prod_idx[p_id]
            valid_subs = data["production_windows"][str(o)][str(p_id)]["valid_subperiod_ids"]
            
            for t_id in valid_subs:
                t = subp_id_to_idx[t_id]
                sub_end = subperiods[t]["end_time"]
                life = products[p_idx]["life_span_minutes"]
                maxA = sub_end + life
                
                if maxA < order["time_window_start"]:
                    continue
                    
                valid_m = [m_p for m_p in range(M_periods) if periods[m_p]["start_time"] + global_params["preparation_time_minutes"] >= sub_end]
                if not valid_m:
                    continue
                    
                Q[i, p_idx, t] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"Q_{i}_{p_idx}_{t}")
                
                if maxA < order["time_window_end"]:
                    z[i, p_idx, t] = m.addVar(vtype=GRB.BINARY, name=f"z_{i}_{p_idx}_{t}")
                    m.addConstr(Q[i, p_idx, t] <= qty * z[i, p_idx, t])
                    m.addConstr(A[i] <= maxA + BIG_M * (1 - z[i, p_idx, t]))
                
                m.addConstr(Q[i, p_idx, t] <= qty * gp.quicksum(V[i, m_p] for m_p in valid_m))
            
            m.addConstr(gp.quicksum(Q.get((i, p_idx, subp_id_to_idx[t_id]), 0) for t_id in valid_subs) >= qty)
            
    for p_idx in range(num_products):
        rate = products[p_idx]["production_rate_per_subperiod"]
        for t in range(SP):
            m.addConstr(gp.quicksum(Q.get((i, p_idx, t), 0) for i in O_idx) <= rate * E[p_idx, t])
            
    for t in range(SP):
        m_p = t // sub_per_period
        total_emp = gp.quicksum(E[p_idx, t] + (gp.quicksum(S[p_idx, t - i] for i in range(L) if t - i >= 0) if L > 0 else 0) for p_idx in range(num_products))
        m.addConstr(total_emp <= W[m_p])
        
    for m_p in range(M_periods):
        m.addConstr(W[m_p] <= data["num_workstations"])
        m.addConstr(W[m_p] <= data["num_production_employees"])
        
        prev_W = W[m_p-1] if m_p > 0 else 0
        m.addConstr(W[m_p] - prev_W <= X[m_p])
        
        K_min = math.ceil(global_params["min_paid_time_subperiods"] / sub_per_period)
        m.addConstr(W[m_p] >= gp.quicksum(X[j] for j in range(max(0, m_p - K_min + 1), m_p + 1)))
        
        K_max = global_params["max_shift_route_length_periods"]
        m.addConstr(W[m_p] <= gp.quicksum(X[j] for j in range(max(0, m_p - K_max + 1), m_p + 1)))

    # Objective
    veh_op_cost = global_params["vehicle_operating_cost_per_minute"] * gp.quicksum(
        travel[i][j] * x[k, t, i, j] for k in range(K) for t in range(T) for i in nodes for j in nodes)
    
    emp_cost = global_params["employee_salary_per_subperiod"] * sub_per_period * gp.quicksum(W[m_p] for m_p in range(M_periods))
    setup_cost_val = global_params["setup_cost"] * gp.quicksum(S[p_idx, t] for p_idx in range(num_products) for t in range(SP))
    
    m.setObjective(gp.quicksum(DriverCost[k] for k in range(K)) + veh_op_cost + emp_cost + setup_cost_val, GRB.MINIMIZE)
    
    def extract_solution(model, cb=False):
        def get_val(var):
            return model.cbGetSolution(var) if cb else var.X

        sol = {}
        sol["objective_value"] = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ) if cb else model.ObjVal
        
        routes = []
        for k in range(K):
            if get_val(is_used[k, 0]) > 0.5:
                trips = []
                for t in range(T):
                    if get_val(is_used[k, t]) > 0.5:
                        curr = 0
                        seq = []
                        visited = set()
                        while True:
                            next_node = None
                            for j in range(N_nodes + 1):
                                if curr != j and get_val(x[k, t, curr, j]) > 0.5:
                                    next_node = j
                                    break
                            if next_node is None or next_node == 0 or next_node in visited:
                                break
                            visited.add(next_node)
                            seq.append(next_node)
                            curr = next_node
                            
                        start_p = 0
                        for m_p in range(M_periods):
                            if get_val(U[k, t, m_p]) > 0.5:
                                start_p = m_p
                                break
                        trips.append({
                            "trip": t,
                            "customers": [orders[j-1]["order_id"] for j in seq],
                            "start_period": start_p
                        })
                routes.append({
                    "vehicle": k,
                    "trips": trips
                })
        sol["routes"] = routes
        
        prod_sched = []
        for o_idx, order in enumerate(orders):
            o = order["order_id"]
            i = o_idx + 1
            for p_idx in range(num_products):
                p_id = products[p_idx]["product_id"]
                for t in range(SP):
                    if (i, p_idx, t) in Q and get_val(Q[i, p_idx, t]) > 1e-4:
                        prod_sched.append({
                            "order": o,
                            "product": p_id,
                            "subperiod": subperiods[t]["subperiod_id"],
                            "quantity": float(get_val(Q[i, p_idx, t]))
                        })
        sol["production_schedule"] = prod_sched
        
        emp_by_ps = []
        setup_by_ps = []
        for p_idx in range(num_products):
            p_id = products[p_idx]["product_id"]
            for t in range(SP):
                e_val = int(round(get_val(E[p_idx, t])))
                s_val = int(round(get_val(S[p_idx, t])))
                if e_val > 0:
                    emp_by_ps.append({"product": p_id, "subperiod": subperiods[t]["subperiod_id"], "employees": e_val})
                if s_val > 0:
                    setup_by_ps.append({"product": p_id, "subperiod": subperiods[t]["subperiod_id"], "setups": s_val})
                    
        sol["production_staffing"] = {
            "employees_by_product_subperiod": emp_by_ps,
            "setups_by_product_subperiod": setup_by_ps,
            "workstations_by_period": [int(round(get_val(W[m_p]))) for m_p in range(M_periods)],
            "shift_starts_by_period": [int(round(get_val(X[m_p]))) for m_p in range(M_periods)]
        }
        return sol
        
    def cb(model, where):
        if where == gp.GRB.Callback.MIPSOL:
            if logger:
                try:
                    sol = extract_solution(model, cb=True)
                    logger.log_solution(sol["objective_value"], sol)
                except Exception:
                    obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
                    logger.log(obj)
                    
    m.optimize(cb)
    
    if m.SolCount > 0:
        final_sol = extract_solution(m, cb=False)
        with open(solution_path, 'w') as f:
            json.dump(final_sol, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default="")
    args = parser.parse_args()
    
    solve_catering_problem(args.instance_path, args.solution_path, args.time_limit, args.log_path)