import argparse
import json
import gurobipy as gp
from solution_logger import SolutionLogger

def get_u_past(initial_status, tau):
    if initial_status > 0:
        if tau > -initial_status:
            return 1
        else:
            return 0
    elif initial_status < 0:
        if tau == initial_status:
            return 1
        else:
            return 0
    else:
        return 0

def solve_unit_commitment(args):
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    def cb(model, where):
        if where == gp.GRB.Callback.MIPSOL:
            obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    model = gp.Model("UnitCommitment")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    
    T = data['num_time_periods']
    H = data['time_period_hours']
    num_g = data['num_generators']

    # Variables
    u = {}
    v = {}
    w = {}
    p = {}
    pb = {}
    c_su = {}

    for g in range(num_g):
        for t in range(1, T + 1):
            u[g, t] = model.addVar(vtype=gp.GRB.BINARY, name=f"u_{g}_{t}")
            v[g, t] = model.addVar(vtype=gp.GRB.BINARY, name=f"v_{g}_{t}")
            w[g, t] = model.addVar(vtype=gp.GRB.BINARY, name=f"w_{g}_{t}")
            p[g, t] = model.addVar(lb=0, name=f"p_{g}_{t}")
            pb[g, t] = model.addVar(lb=0, name=f"pb_{g}_{t}")
            c_su[g, t] = model.addVar(lb=0, name=f"c_su_{g}_{t}")

    obj_expr = gp.LinExpr()

    for g, gen in enumerate(data['generators']):
        init_status = gen['initial_status']
        u_prev_0 = 1 if init_status > 0 else 0
        p_prev_0 = gen['initial_power']
        
        p_min = gen['p_min']
        p_max = gen['p_max']
        RU = gen['ramp_up_limit']
        RD = gen['ramp_down_limit']
        SU = gen['startup_ramp_limit']
        SD = gen['shutdown_ramp_limit']
        UT = gen['min_up_time']
        DT = gen['min_down_time']
        
        L_up = max(0, UT - init_status) if init_status > 0 else 0
        L_dn = max(0, DT - (-init_status)) if init_status < 0 else 0
        
        segs = sorted(gen['production_cost_segments'], key=lambda x: x['output_mw_start'])
        is_convex = True
        for i in range(len(segs)-1):
            if segs[i]['marginal_cost_per_mwh'] > segs[i+1]['marginal_cost_per_mwh']:
                is_convex = False

        S = sorted(gen['startup_costs'], key=lambda x: x['max_downtime_hours'] if x['max_downtime_hours'] is not None else float('inf'))
        SC_1 = S[0]['cost'] if len(S) > 0 else 0.0

        for t in range(1, T + 1):
            # 1. State Logic
            u_prev = u[g, t - 1] if t > 1 else u_prev_0
            model.addConstr(v[g, t] - w[g, t] == u[g, t] - u_prev, name=f"logic_{g}_{t}")

            # 2. Ramping Constraints
            p_prev = p[g, t - 1] if t > 1 else p_prev_0
            model.addConstr(p[g, t] - p_prev <= RU * u_prev + SU * v[g, t], name=f"ramp_up_{g}_{t}")
            model.addConstr(p_prev - p[g, t] <= RD * u[g, t] + SD * w[g, t], name=f"ramp_dn_{g}_{t}")

            # 3. Min Up/Down time initial
            if t <= L_up:
                model.addConstr(u[g, t] == 1, name=f"init_up_{g}_{t}")
            if t <= L_dn:
                model.addConstr(u[g, t] == 0, name=f"init_dn_{g}_{t}")

            # Intermediate Min Up/Down times
            end_up = min(t + UT - 1, T)
            duration_up = end_up - t + 1
            model.addConstr(gp.quicksum(u[g, tau] for tau in range(t, end_up + 1)) >= duration_up * v[g, t], name=f"min_up_{g}_{t}")
            
            end_dn = min(t + DT - 1, T)
            duration_dn = end_dn - t + 1
            model.addConstr(gp.quicksum(1 - u[g, tau] for tau in range(t, end_dn + 1)) >= duration_dn * w[g, t], name=f"min_dn_{g}_{t}")

            # 4. Capacity & Reserve Constraints
            model.addConstr(p[g, t] >= p_min * u[g, t], name=f"cap_min_{g}_{t}")
            model.addConstr(p[g, t] <= p_max * u[g, t], name=f"cap_max_{g}_{t}")
            model.addConstr(pb[g, t] >= p[g, t], name=f"res_min_{g}_{t}")
            model.addConstr(pb[g, t] <= p_max * u[g, t], name=f"res_max_{g}_{t}")

            # 5. Production Cost
            first_start = segs[0]['output_mw_start']
            var_cost_expr = gp.LinExpr()
            
            if is_convex:
                p_sum = first_start * u[g, t]
                for s_idx, seg in enumerate(segs):
                    delta = seg['output_mw_end'] - seg['output_mw_start']
                    p_seg = model.addVar(lb=0, ub=delta, name=f"pseg_{g}_{t}_{s_idx}")
                    model.addConstr(p_seg <= delta * u[g, t], name=f"pseg_lim_{g}_{t}_{s_idx}")
                    p_sum += p_seg
                    var_cost_expr += seg['marginal_cost_per_mwh'] * H * p_seg
                model.addConstr(p[g, t] == p_sum, name=f"p_sum_{g}_{t}")
            else:
                var_cost_val = model.addVar(lb=0, name=f"pc_{g}_{t}")
                var_cost_expr += var_cost_val
                
                xpts = [0.0]
                ypts = [0.0]
                if first_start > 0.0:
                    xpts.append(first_start)
                    ypts.append(0.0)
                    
                cum_cost = 0.0
                for seg in segs:
                    cum_cost += (seg['output_mw_end'] - seg['output_mw_start']) * seg['marginal_cost_per_mwh'] * H
                    xpts.append(seg['output_mw_end'])
                    ypts.append(cum_cost)
                model.addGenConstrPWL(p[g, t], var_cost_val, xpts, ypts, name=f"pwl_{g}_{t}")

            # 6. Startup Costs
            if len(S) > 0:
                model.addConstr(c_su[g, t] >= SC_1 * v[g, t], name=f"su_base_{g}_{t}")
                for m in range(1, len(S)):
                    SC_m = S[m]['cost']
                    limit_hours = S[m-1]['max_downtime_hours']
                    K_prev = int(limit_hours // H)
                    
                    start_idx = t - K_prev - 1
                    end_idx = t - 1
                    
                    sum_u = gp.LinExpr()
                    for tau in range(start_idx, end_idx + 1):
                        if tau <= 0:
                            sum_u += get_u_past(init_status, tau)
                        else:
                            sum_u += u[g, tau]
                            
                    model.addConstr(c_su[g, t] >= SC_m * v[g, t] - (SC_m - SC_1) * sum_u, name=f"su_tier_{g}_{t}_{m}")

            # Objective terms for this (g, t)
            obj_expr += gen['no_load_cost'] * u[g, t] + var_cost_expr + c_su[g, t]

    # Global Demand and Reserve Requirements
    demand = data['demand']
    reserve = data['reserve_requirement']
    for t in range(1, T + 1):
        model.addConstr(gp.quicksum(p[g, t] for g in range(num_g)) == demand[t-1], name=f"sys_demand_{t}")
        model.addConstr(gp.quicksum(pb[g, t] for g in range(num_g)) >= demand[t-1] + reserve[t-1], name=f"sys_reserve_{t}")

    model.setObjective(obj_expr, gp.GRB.MINIMIZE)
    model.optimize(cb)

    if model.SolCount > 0:
        schedule = {}
        for g, gen in enumerate(data['generators']):
            g_id = gen['id']
            schedule[g_id] = {
                "u": [int(round(u[g, t].X)) for t in range(1, T + 1)],
                "p": [p[g, t].X for t in range(1, T + 1)],
                "pb": [pb[g, t].X for t in range(1, T + 1)]
            }
            
        out = {
            "objective_value": model.ObjVal,
            "schedule": schedule
        }
        
        with open(args.solution_path, 'w') as f:
            json.dump(out, f, indent=4)
    else:
        print("No feasible solution found within the time limit.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False)
    args = parser.parse_args()
    solve_unit_commitment(args)