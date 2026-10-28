import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB

# Attempt to load the pre-installed solution logger
try:
    from solution_logger import SolutionLogger
except ImportError:
    class SolutionLogger:
        def __init__(self, *args, **kwargs): pass
        def log(self, *args): pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default="")
    args = parser.parse_args()

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Load instance
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    params = data["parameters"]
    T = params["T_planning_horizon_minutes"]
    D = params["D_num_train_services"]
    NT = params["NT_num_demand_periods"]
    theta = params["theta_period_length_minutes"]
    h_min = params["h_min_minutes"]
    h_max = params["h_max_minutes"]
    C_capacity = float(params["C_train_capacity_passengers"])
    
    # Sort demand periods to ensure correct chronological sequence
    periods_data = sorted(data["demand_periods"], key=lambda x: x["period"])
    lambda_pax = [pd["lambda_pax_per_min"] for pd in periods_data]

    # Handle edge case with 0 trains
    if D == 0:
        total_pax = sum(lambda_pax[p] * theta for p in range(NT))
        obj_val = float(T * 0.5 * total_pax)
        res = {"objective_value": obj_val, "departure_schedule_minutes": {}}
        with open(args.solution_path, 'w') as f:
            json.dump(res, f, indent=2)
        if logger:
            logger.log(obj_val)
        return

    # Cumulative arrival points for PWL mapping
    pts_x = [p * theta for p in range(NT + 1)]
    pts_y = [0.0]
    curr_y = 0.0
    for p in range(NT):
        curr_y += lambda_pax[p] * theta
        pts_y.append(curr_y)
    
    TotalDemand = pts_y[-1]

    # Python evaluator for the correct cumulative arrival curve
    def get_cum_A(t_val):
        period = int(t_val // theta)
        if period >= NT:
            period = NT - 1
            return pts_y[-1] + lambda_pax[-1] * (t_val - NT * theta)
        base = pts_y[period]
        rem = t_val - period * theta
        return base + lambda_pax[period] * rem

    # Exact Python evaluator for total wait time
    def compute_objective_py(t_vars):
        N_vals = [get_cum_A(tv) for tv in t_vars] + [TotalDemand]
        w_curr = 0.0
        obj_val = 0.0
        t_prev = 0.0
        N_prev = 0.0
        for d in range(D):
            H_d = t_vars[d] - t_prev
            A_d = N_vals[d] - N_prev
            obj_val += H_d * (w_curr + 0.5 * A_d)
            w_curr = max(0.0, w_curr + A_d - C_capacity)
            t_prev = t_vars[d]
            N_prev = N_vals[d]
        # Final interval
        H_D = T - t_prev
        A_D = N_vals[-1] - N_prev
        obj_val += H_D * (w_curr + 0.5 * A_D)
        return obj_val

    # Construct provably feasible uniform initial solution
    H_avg = max(h_min, min(h_max, T / D))
    total_interior = (D - 1) * H_avg
    if total_interior > T:
        H_avg = T / max(1, (D - 1))
        total_interior = (D - 1) * H_avg
    t0 = (T - total_interior) / 2
    t_start = [t0 + d * H_avg for d in range(D)]

    # Initial evaluation & logging
    init_obj = compute_objective_py(t_start)
    if logger:
        logger.log(init_obj)
    
    best_t = t_start.copy()
    best_obj = init_obj

    # Calculate remaining time for Gurobi
    rem_time = args.time_limit - (time.time() - start_time)
    
    # Run the Gurobi solver if time permits
    if rem_time > 1.0:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        m = gp.Model("TrainSchedule", env=env)
        
        # Configure model parameters
        m.setParam("TimeLimit", rem_time)
        m.setParam("NonConvex", 2)
        m.setParam("MIPFocus", 1)  # Focus on finding feasible solutions rapidly
        m.setParam("Threads", 1)   # Restricted execution environment requirement

        # Decision Variables
        t = m.addVars(D, vtype=GRB.CONTINUOUS, lb=0, ub=T, name="t")
        H = m.addVars(D+1, vtype=GRB.CONTINUOUS, lb=0, ub=T, name="H")
        N = m.addVars(D, vtype=GRB.CONTINUOUS, lb=0, ub=TotalDemand, name="N")
        A = m.addVars(D+1, vtype=GRB.CONTINUOUS, lb=0, ub=TotalDemand, name="A")
        W = m.addVars(D, vtype=GRB.CONTINUOUS, lb=0, ub=max(TotalDemand, C_capacity), name="W")

        # Basic headway connections
        m.addConstr(H[0] == t[0])
        for d in range(1, D):
            m.addConstr(t[d] - t[d-1] == H[d])
            m.addConstr(H[d] >= h_min)
            m.addConstr(H[d] <= h_max)
        m.addConstr(H[D] == T - t[D-1])

        # PWL tracking and logical bounds
        for d in range(D):
            m.addConstr(t[d] >= d * h_min)
            m.addConstr(t[d] <= T - (D - 1 - d) * h_min)
            m.addGenConstrPWL(t[d], N[d], pts_x, pts_y, f"PWL_N_{d}")

        # Interval specific arrivals
        m.addConstr(A[0] == N[0])
        for d in range(1, D):
            m.addConstr(A[d] == N[d] - N[d-1])
        m.addConstr(A[D] == TotalDemand - N[D-1])

        # Track passenger boardings with capacity bounds
        m.addConstr(W[0] >= A[0] - C_capacity)
        for d in range(1, D):
            m.addConstr(W[d] >= W[d-1] + A[d] - C_capacity)

        # Trapzoidal integration objective
        obj = gp.QuadExpr()
        obj += 0.5 * H[0] * A[0]
        for d in range(1, D):
            obj += H[d] * W[d-1] + 0.5 * H[d] * A[d]
        obj += H[D] * W[D-1] + 0.5 * H[D] * A[D]

        m.setObjective(obj, GRB.MINIMIZE)

        # Inject initial starting solution state
        N_start = [get_cum_A(x) for x in t_start]
        H_start = [t_start[0]] + [t_start[d] - t_start[d-1] for d in range(1, D)] + [T - t_start[-1]]
        A_start = [N_start[0]] + [N_start[d] - N_start[d-1] for d in range(1, D)] + [TotalDemand - N_start[-1]]
        W_start = []
        wc = 0.0
        for d in range(D):
            wc = max(0.0, wc + A_start[d] - C_capacity)
            W_start.append(wc)

        for d in range(D):
            t[d].Start = t_start[d]
            N[d].Start = N_start[d]
            W[d].Start = W_start[d]
        for d in range(D+1):
            H[d].Start = H_start[d]
            A[d].Start = A_start[d]

        # MIP Solution Logger definition
        class MipLogger:
            def __init__(self, c_logger, initial_best):
                self.logger = c_logger
                self.best_obj = initial_best

            def callback(self, model, where):
                if where == gp.GRB.Callback.MIPSOL:
                    cb_obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
                    if cb_obj < self.best_obj - 1e-5:
                        self.best_obj = cb_obj
                        if self.logger:
                            self.logger.log(cb_obj)

        mip_logger = MipLogger(logger, init_obj)
        m.optimize(mip_logger.callback)

        # Update best solution if improvement exists
        if m.SolCount > 0:
            sol_obj = m.ObjVal
            if sol_obj < init_obj:
                sol_t = [t[d].X for d in range(D)]
                true_obj = compute_objective_py(sol_t)
                best_t = sol_t
                best_obj = true_obj
    
    # Save Output Formatted Standard
    dep_schedule = {str(d): float(best_t[d]) for d in range(D)}
    
    out = {
        "objective_value": float(best_obj),
        "departure_schedule_minutes": dep_schedule
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()