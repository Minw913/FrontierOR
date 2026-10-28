import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    env = gp.Env(empty=True)
    env.setParam("Seed", 0)
    env.setParam("MIPGap", 1e-4)
    env.setParam("NumericFocus", 0)
    env.setParam("Threads", 1)
    env.start()
    model = gp.Model("ucgna", env=env)
    
    T = instance["time_periods"]
    beta = instance["objective_weight_beta"]
    eta_p = instance["stress_parameters"]["eta_p"]
    eta_g = instance["stress_parameters"]["eta_g"]

    elec = instance["electricity_network"]
    gens = instance["generators"]["generators"]
    gas = instance["gas_network"]

    o = {}
    vp = {}
    vm = {}
    w = {}
    r = {}
    phimax = {}
    p = {}
    se = {}
    f = {}
    theta = {}
    sg = {}
    sgs = {}
    pisq = {}
    phig = {}
    lg = {}
    qg = {}
    gamma = {}
    psi = {}

    for g in gens:
        gid = g["id"]
        for t in range(0, T + 1):
            if t == 0:
                o[gid, t] = model.addVar(vtype=GRB.BINARY, name=f"o_{gid}_{t}")
                p[gid, t] = model.addVar(lb=0, ub=g["max_power"], name=f"p_{gid}_{t}")
                if g["initial_status"] == 1:
                    o[gid, t].lb = 1
                    o[gid, t].ub = 1
                else:
                    o[gid, t].lb = 0
                    o[gid, t].ub = 0
                p[gid, t].lb = g["initial_gen"]
                p[gid, t].ub = g["initial_gen"]
            else:
                o[gid, t] = model.addVar(vtype=GRB.BINARY, name=f"o_{gid}_{t}")
                vp[gid, t] = model.addVar(vtype=GRB.BINARY, name=f"vp_{gid}_{t}")
                vm[gid, t] = model.addVar(vtype=GRB.BINARY, name=f"vm_{gid}_{t}")
                r[gid, t] = model.addVar(lb=0, name=f"r_{gid}_{t}")
                p[gid, t] = model.addVar(lb=0, ub=g["max_power"], name=f"p_{gid}_{t}")
                if g["is_gfpp"]:
                    phimax[gid, t] = model.addVar(lb=0, ub=gas["max_gas_price_mmBtu"], name=f"phimax_{gid}_{t}")
                
                for b in g["bids"]:
                    bid_id = b["id"]
                    w[gid, bid_id, t] = model.addVar(vtype=GRB.BINARY, name=f"w_{gid}_{bid_id}_{t}")
                    se[gid, bid_id, t] = model.addVar(lb=0, ub=b["max_amount"], name=f"se_{gid}_{bid_id}_{t}")
    
    for line in elec["lines"]:
        line_id = line["id"]
        for t in range(1, T + 1):
            f[line_id, t] = model.addVar(lb=-line["thermal_limit"], ub=line["thermal_limit"], name=f"f_{line_id}_{t}")
            
    for bus in elec["buses"]:
        bus_id = bus["id"]
        for t in range(1, T + 1):
            theta[bus_id, t] = model.addVar(lb=bus["voltage_angle_lb"], ub=bus["voltage_angle_ub"], name=f"theta_{bus_id}_{t}")
            
    for j in gas["junctions"]:
        jid = j["id"]
        for t in range(1, T + 1):
            sg[jid, t] = model.addVar(lb=0, name=f"sg_{jid}_{t}")
            pisq[jid, t] = model.addVar(lb=j["pressure_lb_squared"], ub=j["pressure_ub_squared"], name=f"pisq_{jid}_{t}")
            lg[jid, t] = model.addVar(lb=0, name=f"lg_{jid}_{t}")
            qg[jid, t] = model.addVar(lb=0, ub=max(0, j["gas_demand_profile"][t-1]), name=f"qg_{jid}_{t}")
            gamma[jid, t] = model.addVar(lb=0, name=f"gamma_{jid}_{t}")
            if j["is_source"]:
                for interval in j["supply_intervals"]:
                    int_id = interval["id"]
                    sgs[jid, int_id, t] = model.addVar(lb=interval["interval_lb"], ub=interval["interval_ub"], name=f"sgs_{jid}_{int_id}_{t}")
            else:
                sg[jid, t].ub = 0
                
    for conn in gas["connections"]:
        conn_id = conn["id"]
        for t in range(1, T + 1):
            phig[conn_id, t] = model.addVar(lb=0, name=f"phig_{conn_id}_{t}")
            
    for zone in gas["pricing_zones"]:
        zone_id = zone["id"]
        for t in range(1, T + 1):
            psi[zone_id, t] = model.addVar(lb=gas["min_gas_price_mmBtu"], ub=gas["max_gas_price_mmBtu"], name=f"psi_{zone_id}_{t}")

    model.update()
    
    for g in gens:
        gid = g["id"]
        for t in range(1, T + 1):
            model.addConstr(vp[gid, t] - vm[gid, t] == o[gid, t] - o[gid, t-1])
            
            if t + g["min_up_time"] - 1 <= T:
                model.addConstr(gp.quicksum(vp[gid, t_] for t_ in range(t, t + g["min_up_time"])) <= o[gid, t + g["min_up_time"] - 1])
            if t + g["min_down_time"] - 1 <= T:
                model.addConstr(gp.quicksum(vp[gid, t_] for t_ in range(t, t + g["min_down_time"])) <= 1 - o[gid, t - 1])
                
            if g["initial_status"] == 1:
                must_on = g["min_up_time"] - g["initial_active_periods"]
                if t <= must_on:
                    o[gid, t].lb = 1
            else:
                must_off = g["min_down_time"] - g["initial_inactive_periods"]
                if t <= must_off:
                    o[gid, t].ub = 0
                    
            model.addConstr(p[gid, t] >= g["min_power"] * o[gid, t])
            model.addConstr(p[gid, t] <= g["max_power"] * o[gid, t])
            
            model.addConstr(p[gid, t] - p[gid, t-1] <= g["ramp_up"] * o[gid, t-1] + g["max_power"] * vp[gid, t])
            model.addConstr(p[gid, t-1] - p[gid, t] <= g["ramp_down"] * o[gid, t] + g["min_power"] * vm[gid, t])
            
            for tier in g["startup_cost_params"]:
                lookback = int(tier[0])
                cost = tier[1]
                if t - lookback >= 0:
                    model.addConstr(r[gid, t] >= cost * (vp[gid, t] - gp.quicksum(o[gid, t-t_] for t_ in range(1, lookback))))
                    
            total_se = gp.quicksum(se[gid, b["id"], t] for b in g["bids"])
            model.addConstr(p[gid, t] == total_se)
            
            for i, b in enumerate(g["bids"]):
                bid_id = b["id"]
                model.addConstr(se[gid, bid_id, t] <= b["max_amount"] * w[gid, bid_id, t])
                if g["is_gfpp"]:
                    if i < len(g["bids"]) - 1:
                        next_bid_id = g["bids"][i+1]["id"]
                        model.addConstr(w[gid, bid_id, t] >= w[gid, next_bid_id, t])
                        model.addConstr(se[gid, bid_id, t] >= b["max_amount"] * w[gid, next_bid_id, t])
                        
            if g["is_gfpp"]:
                zone_id = None
                for zone in gas["pricing_zones"]:
                    if g["gas_junction"] in zone["junctions"]:
                        zone_id = zone["id"]
                        break
                if zone_id is not None:
                    model.addConstr(phimax[gid, t] >= psi[zone_id, t] * o[gid, t])
                    max_frac = g["max_gas_price_fraction"] if g["max_gas_price_fraction"] is not None else 1.0
                    model.addConstr(phimax[gid, t] <= gas["max_gas_price_mmBtu"] * max_frac * o[gid, t])

    for bus in elec["buses"]:
        bus_id = bus["id"]
        for t in range(1, T + 1):
            gen_sum = gp.quicksum(p[g["id"], t] for g in gens if g["bus"] == bus_id)
            demand = bus["demand_profile"][t-1] * eta_p
            flow_out = gp.quicksum(f[line["id"], t] for line in elec["lines"] if line["from_bus"] == bus_id)
            flow_in = gp.quicksum(f[line["id"], t] for line in elec["lines"] if line["to_bus"] == bus_id)
            model.addConstr(gen_sum - demand == flow_out - flow_in)
            
    for line in elec["lines"]:
        line_id = line["id"]
        for t in range(1, T + 1):
            model.addConstr(f[line_id, t] == line["susceptance"] * (theta[line["from_bus"], t] - theta[line["to_bus"], t]))
            model.addConstr(theta[line["from_bus"], t] - theta[line["to_bus"], t] <= line["angle_diff_limit"])
            model.addConstr(theta[line["to_bus"], t] - theta[line["from_bus"], t] <= line["angle_diff_limit"])
            
    for j in gas["junctions"]:
        jid = j["id"]
        for t in range(1, T + 1):
            if j["is_source"]:
                model.addConstr(sg[jid, t] == gp.quicksum(sgs[jid, int_id["id"], t] for int_id in j["supply_intervals"]))
            else:
                model.addConstr(sg[jid, t] == 0)
                
            demand = j["gas_demand_profile"][t-1] * eta_g
            model.addConstr(lg[jid, t] + qg[jid, t] == demand)
            
            flow_out = gp.quicksum(phig[conn["id"], t] for conn in gas["connections"] if conn["from_junction"] == jid)
            flow_in = gp.quicksum(phig[conn["id"], t] for conn in gas["connections"] if conn["to_junction"] == jid)
            model.addConstr(sg[jid, t] - lg[jid, t] - gamma[jid, t] == flow_out - flow_in)
            
    for conn in gas["connections"]:
        conn_id = conn["id"]
        for t in range(1, T + 1):
            if conn["type"] == "pipeline":
                model.addConstr(pisq[conn["from_junction"], t] - pisq[conn["to_junction"], t] >= conn["weymouth_factor"] * phig[conn_id, t] * phig[conn_id, t])
            elif conn["type"] == "compressor":
                if conn["compression_ratio_lb"] is not None:
                    model.addConstr(pisq[conn["to_junction"], t] >= (conn["compression_ratio_lb"]**2) * pisq[conn["from_junction"], t])
                if conn["compression_ratio_ub"] is not None:
                    model.addConstr(pisq[conn["to_junction"], t] <= (conn["compression_ratio_ub"]**2) * pisq[conn["from_junction"], t])
            elif conn["type"] == "control_valve":
                if conn["control_ratio_lb"] is not None:
                    model.addConstr(pisq[conn["to_junction"], t] >= (conn["control_ratio_lb"]**2) * pisq[conn["from_junction"], t])
                if conn["control_ratio_ub"] is not None:
                    model.addConstr(pisq[conn["to_junction"], t] <= (conn["control_ratio_ub"]**2) * pisq[conn["from_junction"], t])
                    
    for j in gas["junctions"]:
        jid = j["id"]
        for t in range(1, T + 1):
            gfpps_at_j = [g for g in gens if g["is_gfpp"] and g["gas_junction"] == jid]
            if gfpps_at_j:
                heat_sum = gp.QuadExpr()
                for g in gfpps_at_j:
                    gid = g["id"]
                    H_u2 = g["heat_rate_coefficients"]["H_u2"] if g["heat_rate_coefficients"]["H_u2"] is not None else 0.0
                    H_u1 = g["heat_rate_coefficients"]["H_u1"] if g["heat_rate_coefficients"]["H_u1"] is not None else 0.0
                    H_u0 = g["heat_rate_coefficients"]["H_u0"] if g["heat_rate_coefficients"]["H_u0"] is not None else 0.0
                    heat_sum += H_u2 * p[gid, t] * p[gid, t] + H_u1 * p[gid, t] + H_u0 * o[gid, t]
                model.addConstr(gamma[jid, t] >= heat_sum)
            else:
                model.addConstr(gamma[jid, t] == 0)

    elec_cost = gp.LinExpr()
    for g in gens:
        gid = g["id"]
        for t in range(1, T + 1):
            elec_cost += g["no_load_cost"] * o[gid, t] + r[gid, t]
            for b in g["bids"]:
                elec_cost += b["price"] * se[gid, b["id"], t]
                
    gas_cost = gp.LinExpr()
    for j in gas["junctions"]:
        jid = j["id"]
        for t in range(1, T + 1):
            gas_cost += j["demand_shedding_cost"] * qg[jid, t]
            if j["is_source"]:
                for interval in j["supply_intervals"]:
                    gas_cost += interval["slope"] * sgs[jid, interval["id"], t]
                    
    model.setObjective(beta * elec_cost + (1 - beta) * gas_cost, GRB.MINIMIZE)
    model.setParam("TimeLimit", args.time_limit)
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL and logger:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            sol = {
                "objective_value": obj,
                "released_problem_variant": "integrated_primal_ucgna",
                "objective_weight_beta": beta,
                "primary_variables": {}
            }
            pv = sol["primary_variables"]
            for g in gens:
                gid = g["id"]
                for t in range(0, T + 1):
                    pv[f"o_{gid}_{t}"] = model.cbGetSolution(o[gid, t]) if t > 0 else g["initial_status"]
                    pv[f"p_{gid}_{t}"] = model.cbGetSolution(p[gid, t]) if t > 0 else g["initial_gen"]
                    if t > 0:
                        pv[f"vp_{gid}_{t}"] = model.cbGetSolution(vp[gid, t])
                        pv[f"vm_{gid}_{t}"] = model.cbGetSolution(vm[gid, t])
                        pv[f"r_{gid}_{t}"] = model.cbGetSolution(r[gid, t])
                        if g["is_gfpp"]:
                            pv[f"phimax_{gid}_{t}"] = model.cbGetSolution(phimax[gid, t])
                        for b in g["bids"]:
                            pv[f"w_{gid}_{b['id']}_{t}"] = model.cbGetSolution(w[gid, b['id'], t])
                            pv[f"se_{gid}_{b['id']}_{t}"] = model.cbGetSolution(se[gid, b['id'], t])
            
            for line in elec["lines"]:
                for t in range(1, T + 1):
                    pv[f"f_{line['id']}_{t}"] = model.cbGetSolution(f[line['id'], t])
                    
            for bus in elec["buses"]:
                for t in range(1, T + 1):
                    pv[f"theta_{bus['id']}_{t}"] = model.cbGetSolution(theta[bus['id'], t])
                    
            for j in gas["junctions"]:
                for t in range(1, T + 1):
                    pv[f"sg_{j['id']}_{t}"] = model.cbGetSolution(sg[j['id'], t])
                    pv[f"pisq_{j['id']}_{t}"] = model.cbGetSolution(pisq[j['id'], t])
                    pv[f"lg_{j['id']}_{t}"] = model.cbGetSolution(lg[j['id'], t])
                    pv[f"qg_{j['id']}_{t}"] = model.cbGetSolution(qg[j['id'], t])
                    pv[f"gamma_{j['id']}_{t}"] = model.cbGetSolution(gamma[j['id'], t])
                    if j["is_source"]:
                        for interval in j["supply_intervals"]:
                            pv[f"sgs_{j['id']}_{interval['id']}_{t}"] = model.cbGetSolution(sgs[j['id'], interval['id'], t])
                            
            for conn in gas["connections"]:
                for t in range(1, T + 1):
                    pv[f"phig_{conn['id']}_{t}"] = model.cbGetSolution(phig[conn['id'], t])
                    
            for zone in gas["pricing_zones"]:
                for t in range(1, T + 1):
                    pv[f"psi_{zone['id']}_{t}"] = model.cbGetSolution(psi[zone['id'], t])
                    
            logger.log_solution(obj, sol)

    model.optimize(callback)

    obj_val = model.ObjVal if model.SolCount > 0 else 0.0
    pv = {}
    
    if model.SolCount > 0:
        for g in gens:
            gid = g["id"]
            for t in range(0, T + 1):
                pv[f"o_{gid}_{t}"] = o[gid, t].X if t > 0 else g["initial_status"]
                pv[f"p_{gid}_{t}"] = p[gid, t].X if t > 0 else g["initial_gen"]
                if t > 0:
                    pv[f"vp_{gid}_{t}"] = vp[gid, t].X
                    pv[f"vm_{gid}_{t}"] = vm[gid, t].X
                    pv[f"r_{gid}_{t}"] = r[gid, t].X
                    if g["is_gfpp"]:
                        pv[f"phimax_{gid}_{t}"] = phimax[gid, t].X
                    for b in g["bids"]:
                        pv[f"w_{gid}_{b['id']}_{t}"] = w[gid, b['id'], t].X
                        pv[f"se_{gid}_{b['id']}_{t}"] = se[gid, b['id'], t].X
        
        for line in elec["lines"]:
            for t in range(1, T + 1):
                pv[f"f_{line['id']}_{t}"] = f[line['id'], t].X
                
        for bus in elec["buses"]:
            for t in range(1, T + 1):
                pv[f"theta_{bus['id']}_{t}"] = theta[bus['id'], t].X
                
        for j in gas["junctions"]:
            for t in range(1, T + 1):
                pv[f"sg_{j['id']}_{t}"] = sg[j['id'], t].X
                pv[f"pisq_{j['id']}_{t}"] = pisq[j['id'], t].X
                pv[f"lg_{j['id']}_{t}"] = lg[j['id'], t].X
                pv[f"qg_{j['id']}_{t}"] = qg[j['id'], t].X
                pv[f"gamma_{j['id']}_{t}"] = gamma[j['id'], t].X
                if j["is_source"]:
                    for interval in j["supply_intervals"]:
                        pv[f"sgs_{j['id']}_{interval['id']}_{t}"] = sgs[j['id'], interval['id'], t].X
                        
        for conn in gas["connections"]:
            for t in range(1, T + 1):
                pv[f"phig_{conn['id']}_{t}"] = phig[conn['id'], t].X
                
        for zone in gas["pricing_zones"]:
            for t in range(1, T + 1):
                pv[f"psi_{zone['id']}_{t}"] = psi[zone['id'], t].X
    else:
        for g in gens:
            gid = g["id"]
            for t in range(0, T + 1):
                pv[f"o_{gid}_{t}"] = g["initial_status"] if t == 0 else 0
                pv[f"p_{gid}_{t}"] = g["initial_gen"] if t == 0 else 0
                if t > 0:
                    pv[f"vp_{gid}_{t}"] = 0
                    pv[f"vm_{gid}_{t}"] = 0
                    pv[f"r_{gid}_{t}"] = 0
                    if g["is_gfpp"]:
                        pv[f"phimax_{gid}_{t}"] = 0
                    for b in g["bids"]:
                        pv[f"w_{gid}_{b['id']}_{t}"] = 0
                        pv[f"se_{gid}_{b['id']}_{t}"] = 0
        for line in elec["lines"]:
            for t in range(1, T + 1):
                pv[f"f_{line['id']}_{t}"] = 0
        for bus in elec["buses"]:
            for t in range(1, T + 1):
                pv[f"theta_{bus['id']}_{t}"] = 0
        for j in gas["junctions"]:
            for t in range(1, T + 1):
                pv[f"sg_{j['id']}_{t}"] = 0
                pv[f"pisq_{j['id']}_{t}"] = j["pressure_lb_squared"]
                pv[f"lg_{j['id']}_{t}"] = 0
                pv[f"qg_{j['id']}_{t}"] = j["gas_demand_profile"][t-1] * eta_g
                pv[f"gamma_{j['id']}_{t}"] = 0
                if j["is_source"]:
                    for interval in j["supply_intervals"]:
                        pv[f"sgs_{j['id']}_{interval['id']}_{t}"] = interval["interval_lb"]
        for conn in gas["connections"]:
            for t in range(1, T + 1):
                pv[f"phig_{conn['id']}_{t}"] = 0
        for zone in gas["pricing_zones"]:
            for t in range(1, T + 1):
                pv[f"psi_{zone['id']}_{t}"] = gas["min_gas_price_mmBtu"]

    solution = {
        "objective_value": obj_val,
        "released_problem_variant": "integrated_primal_ucgna",
        "objective_weight_beta": beta,
        "primary_variables": pv
    }
    
    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=4)

if __name__ == "__main__":
    main()