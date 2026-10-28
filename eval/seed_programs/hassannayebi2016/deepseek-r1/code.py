import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    params = instance["parameters"]
    T = params["T_planning_horizon_minutes"]
    D = params["D_num_train_services"]
    NT = params["NT_num_demand_periods"]
    theta = params["theta_period_length_minutes"]
    h_min = params["h_min_minutes"]
    h_max = params["h_max_minutes"]
    C = params["C_train_capacity_passengers"]
    total_demand = params["total_demand_passengers"]
    demand_periods = instance["demand_periods"]
    
    L = [period['lambda_pax_per_min'] for period in demand_periods]
    L_max = max(L) if L else 0
    
    base = [0.0]
    for i in range(NT):
        base.append(base[-1] + theta * L[i])
    
    model = gp.Model("TrainScheduling")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam('NonConvex', 2)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', 1)
    
    x = model.addVars(D, NT, vtype=GRB.BINARY, name="x")
    t = model.addVars(D, lb=0.0, ub=T, name="t")
    z_time = model.addVars(D, NT, lb=0.0, ub=theta, name="z_time")
    
    h0 = model.addVar(lb=0.0, ub=T, name="h0")
    h = model.addVars(D-1, lb=h_min, ub=h_max, name="h")
    h_final = model.addVar(lb=0.0, ub=T, name="h_final")
    
    w = model.addVars(D, lb=0.0, ub=total_demand, name="w")
    b = model.addVars(D, lb=0.0, ub=C, name="b")
    
    F = model.addVars(D, lb=0.0, ub=total_demand, name="F")
    A0 = model.addVar(lb=0.0, ub=total_demand, name="A0")
    A = model.addVars(D-1, lb=0.0, ub=total_demand, name="A")
    A_final = model.addVar(lb=0.0, ub=total_demand, name="A_final")
    
    y0 = model.addVar(lb=0.0, name="y0")
    y1 = model.addVars(D-1, lb=0.0, name="y1")
    y2 = model.addVars(D-1, lb=0.0, name="y2")
    y3 = model.addVar(lb=0.0, name="y3")
    y4 = model.addVar(lb=0.0, name="y4")
    
    for d in range(D):
        model.addConstr(gp.quicksum(x[d, p] for p in range(NT)) == 1, f"assign_train_{d}")
    
    for d in range(D-1):
        model.addConstr(
            gp.quicksum(p * x[d+1, p] for p in range(NT)) >= 
            gp.quicksum(p * x[d, p] for p in range(NT)), 
            f"monotonic_{d}"
        )
    
    for d in range(D):
        model.addConstr(
            t[d] == gp.quicksum(p * theta * x[d, p] + z_time[d, p] for p in range(NT)),
            f"t_{d}"
        )
        for p in range(NT):
            model.addConstr(z_time[d, p] <= theta * x[d, p], f"z_time_ub_{d}_{p}")
    
    model.addConstr(h0 == t[0], "h0_def")
    for d in range(D-1):
        model.addConstr(h[d] == t[d+1] - t[d], f"h_{d}_def")
    model.addConstr(h_final == T - t[D-1], "h_final_def")
    
    for d in range(D):
        model.addConstr(
            F[d] == gp.quicksum(x[d, p] * (base[p] + z_time[d, p] * L[p]) for p in range(NT)),
            f"F_{d}"
        )
    
    model.addConstr(A0 == F[0], "A0_def")
    for d in range(D-1):
        model.addConstr(A[d] == F[d+1] - F[d], f"A_{d}_def")
    model.addConstr(A_final == total_demand - F[D-1], "A_final_def")
    
    model.addConstr(b[0] <= C, "b0_ub")
    model.addConstr(b[0] <= A0, "b0_avail")
    model.addConstr(w[0] == A0 - b[0], "w0_def")
    for d in range(1, D):
        model.addConstr(b[d] <= C, f"b{d}_ub")
        model.addConstr(b[d] <= w[d-1] + A[d-1], f"b{d}_avail")
        model.addConstr(w[d] == w[d-1] + A[d-1] - b[d], f"w{d}_def")
    
    model.addConstr(y0 >= 0, "mc_y0_1")
    model.addConstr(y0 >= T * A0 + L_max * T * h0 - L_max * T * T, "mc_y0_2")
    model.addConstr(y0 <= T * A0, "mc_y0_3")
    model.addConstr(y0 <= L_max * T * h0, "mc_y0_4")
    
    for d in range(D-1):
        model.addConstr(y1[d] >= h_min * w[d] + 0 * h[d] - h_min * 0, f"mc_y1_{d}_1")
        model.addConstr(y1[d] >= h_max * w[d] + total_demand * h[d] - h_max * total_demand, f"mc_y1_{d}_2")
        model.addConstr(y1[d] <= h_max * w[d] + 0 * h[d] - h_max * 0, f"mc_y1_{d}_3")
        model.addConstr(y1[d] <= total_demand * h[d] + h_min * w[d] - h_min * total_demand, f"mc_y1_{d}_4")
        
        model.addConstr(y2[d] >= h_min * A[d] + 0 * h[d] - h_min * 0, f"mc_y2_{d}_1")
        model.addConstr(y2[d] >= h_max * A[d] + total_demand * h[d] - h_max * total_demand, f"mc_y2_{d}_2")
        model.addConstr(y2[d] <= h_max * A[d] + 0 * h[d] - h_max * 0, f"mc_y2_{d}_3")
        model.addConstr(y2[d] <= total_demand * h[d] + h_min * A[d] - h_min * total_demand, f"mc_y2_{d}_4")
    
    model.addConstr(y3 >= 0, "mc_y3_1")
    model.addConstr(y3 >= T * w[D-1] + total_demand * h_final - T * total_demand, "mc_y3_2")
    model.addConstr(y3 <= T * w[D-1], "mc_y3_3")
    model.addConstr(y3 <= total_demand * h_final, "mc_y3_4")
    
    model.addConstr(y4 >= 0, "mc_y4_1")
    model.addConstr(y4 >= T * A_final + total_demand * h_final - T * total_demand, "mc_y4_2")
    model.addConstr(y4 <= T * A_final, "mc_y4_3")
    model.addConstr(y4 <= total_demand * h_final, "mc_y4_4")
    
    total_waiting_time = 0.5 * y0 + \
                         gp.quicksum(y1[d] + 0.5 * y2[d] for d in range(D-1)) + \
                         y3 + 0.5 * y4
    model.setObjective(total_waiting_time, GRB.MINIMIZE)
    
    logger = SolutionLogger(args.log_path, "minimize") if args.log_path else None
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)
    
    try:
        if logger:
            model.optimize(callback)
        else:
            model.optimize()
    except gp.GurobiError as e:
        print(f"Optimization error: {e}")
        return
    
    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible")
        return
    elif model.status in (GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.SUBOPTIMAL, GRB.OPTIMAL):
        if model.SolCount > 0:
            obj_value = model.ObjVal
            departure_schedule = {}
            for d in range(D):
                departure_schedule[str(d)] = t[d].X
            solution = {
                "objective_value": obj_value,
                "departure_schedule_minutes": departure_schedule
            }
            with open(args.solution_path, 'w') as f:
                json.dump(solution, f, indent=2)
        else:
            print("No solution found")
    else:
        print(f"Optimization status: {model.status}")

if __name__ == "__main__":
    main()