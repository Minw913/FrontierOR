import sys
import json
import argparse
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def main():
    parser = argparse.ArgumentParser(description="UCGNA Solver")
    parser.add_argument("--instance_path", required=True, help="Path to instance JSON file")
    parser.add_argument("--solution_path", required=True, help="Path to output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", required=False, help="Path to log intermediate solutions")
    args = parser.parse_args()

    # Load instance data
    with open(args.instance_path, "r") as f:
        data = json.load(f)

    # Extract parameters
    T = data["time_periods"]
    beta = data["objective_weight_beta"]
    eta_p = data["stress_parameters"]["eta_p"]
    eta_g = data["stress_parameters"]["eta_g"]

    electricity = data["electricity_network"]
    buses = electricity["buses"]
    lines = electricity["lines"]
    
    generators_data = data["generators"]
    gen_list = generators_data["generators"]
    
    gas_network = data["gas_network"]
    junctions = gas_network["junctions"]
    connections = gas_network["connections"]
    pricing_zones = gas_network["pricing_zones"]
    max_gas_price_mmBtu = gas_network["max_gas_price_mmBtu"]

    # Pre-sort bids by price to enforce sequential activation logically
    for g in gen_list:
        if "bids" in g and g["bids"]:
            g["bids"] = sorted(g["bids"], key=lambda x: x["price"])

    # Build mappings
    bus_to_gens = {b["id"]: [] for b in buses}
    for g in gen_list:
        bus_to_gens[g["bus"]].append(g)

    lines_to = {b["id"]: [] for b in buses}
    lines_from = {b["id"]: [] for b in buses}
    for l in lines:
        lines_to[l["to_bus"]].append(l)
        lines_from[l["from_bus"]].append(l)

    conn_to = {j["id"]: [] for j in junctions}
    conn_from = {j["id"]: [] for j in junctions}
    for c in connections:
        conn_to[c["to_junction"]].append(c)
        conn_from[c["from_junction"]].append(c)

    junc_to_gfpps = {j["id"]: [] for j in junctions}
    for g in gen_list:
        if g["is_gfpp"] and g["gas_junction"] is not None:
            junc_to_gfpps[g["gas_junction"]].append(g)

    junc_to_zone = {}
    for z in pricing_zones:
        for j in z["junctions"]:
            junc_to_zone[j] = z["id"]

    # Historical state resolution
    hist_o = {}
    hist_vp = {}
    for g in gen_list:
        g_id = g["id"]
        hist_o[g_id] = {}
        hist_vp[g_id] = {}
        init_status = g["initial_status"]
        init_act = g["initial_active_periods"]
        init_inact = g["initial_inactive_periods"]
        
        max_lookback = 0
        if "startup_cost_params" in g:
            for params in g["startup_cost_params"]:
                max_lookback = max(max_lookback, int(params[0]))
                
        max_hist = max(g["min_up_time"], g["min_down_time"], max_lookback) + 2
        
        for tau in range(-max_hist, 1):
            if init_status == 1:
                hist_o[g_id][tau] = 1 if tau > -init_act else 0
            else:
                hist_o[g_id][tau] = 0 if tau > -init_inact else 1
                
        for tau in range(-max_hist, 1):
            hist_vp[g_id][tau] = 1 if (hist_o[g_id][tau] == 1 and hist_o[g_id].get(tau-1, 0) == 0) else 0

    # Model Setup
    model = gp.Model("UCGNA")
    model.setParam("TimeLimit", args.time_limit)
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)

    logger = SolutionLogger(args.log_path, sense="minimize") if SolutionLogger and args.log_path else None

    # Primary Variables
    o = {}
    vp = {}
    vm = {}
    r = {}
    phimax = {}
    p = {}
    se = {}
    w = {}
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

    for t in range(1, T+1):
        for g in gen_list:
            g_id = g["id"]
            o[g_id, t] = model.addVar(vtype=GRB.BINARY, name=f"o_{g_id}_{t}")
            vp[g_id, t] = model.addVar(lb=0, ub=1, name=f"vp_{g_id}_{t}")
            vm[g_id, t] = model.addVar(lb=0, ub=1, name=f"vm_{g_id}_{t}")
            r[g_id, t] = model.addVar(lb=0, name=f"r_{g_id}_{t}")
            p[g_id, t] = model.addVar(lb=0, name=f"p_{g_id}_{t}")
            if g["is_gfpp"]:
                phimax[g_id, t] = model.addVar(lb=0, name=f"phimax_{g_id}_{t}")
            for bid in g["bids"]:
                b_id = bid["id"]
                w[g_id, b_id, t] = model.addVar(lb=0, ub=1, name=f"w_{g_id}_{b_id}_{t}")
                se[g_id, b_id, t] = model.addVar(lb=0, name=f"se_{g_id}_{b_id}_{t}")
                
        for l in lines:
            f[l["id"], t] = model.addVar(lb=-l["thermal_limit"], ub=l["thermal_limit"], name=f"f_{l['id']}_{t}")
        for b in buses:
            theta[b["id"], t] = model.addVar(lb=b["voltage_angle_lb"], ub=b["voltage_angle_ub"], name=f"theta_{b['id']}_{t}")
        for j in junctions:
            j_id = j["id"]
            sg[j_id, t] = model.addVar(lb=0, name=f"sg_{j_id}_{t}")
            pisq[j_id, t] = model.addVar(lb=j["pressure_lb_squared"], ub=j["pressure_ub_squared"], name=f"pisq_{j_id}_{t}")
            lg[j_id, t] = model.addVar(lb=0, name=f"lg_{j_id}_{t}")
            qg[j_id, t] = model.addVar(lb=0, name=f"qg_{j_id}_{t}")
            gamma[j_id, t] = model.addVar(lb=0, name=f"gamma_{j_id}_{t}")
            if j["is_source"]:
                for intv in j["supply_intervals"]:
                    sgs[j_id, intv["id"], t] = model.addVar(lb=intv["interval_lb"], ub=intv["interval_ub"], name=f"sgs_{j_id}_{intv['id']}_{t}")
        for c in connections:
            phig[c["id"], t] = model.addVar(lb=0, name=f"phig_{c['id']}_{t}")
        for z in pricing_zones:
            psi[z["id"], t] = model.addVar(lb=gas_network["min_gas_price_mmBtu"], ub=gas_network["max_gas_price_mmBtu"], name=f"psi_{z['id']}_{t}")

    # Constraints
    for t in range(1, T+1):
        # 1. Electricity constraints
        for g in gen_list:
            g_id = g["id"]
            o_prev = o[g_id, t-1] if t > 1 else hist_o[g_id][0]
            p_prev = p[g_id, t-1] if t > 1 else g["initial_gen"]
            
            model.addConstr(o[g_id, t] - o_prev == vp[g_id, t] - vm[g_id, t])
            model.addConstr(vp[g_id, t] + vm[g_id, t] <= 1)
            
            # Min up time
            UT = g["min_up_time"]
            if UT > 0:
                model.addConstr(sum(vp[g_id, k] if k > 0 else hist_vp[g_id][k] for k in range(t - UT + 1, t + 1)) <= o[g_id, t])
            
            # Min down time
            DT = g["min_down_time"]
            if DT > 0:
                o_prev_dt = o[g_id, t - DT] if t - DT > 0 else hist_o[g_id][t - DT]
                model.addConstr(sum(vp[g_id, k] if k > 0 else hist_vp[g_id][k] for k in range(t - DT + 1, t + 1)) <= 1 - o_prev_dt)
            
            # Startup costs
            for params in g.get("startup_cost_params", []):
                W = int(params[0])
                C = params[1]
                sum_o = sum(o[g_id, t-i] if t-i > 0 else hist_o[g_id][t-i] for i in range(1, W + 1))
                model.addConstr(r[g_id, t] >= C * (vp[g_id, t] - sum_o))
                
            # Power Bounds & Ramping
            model.addConstr(p[g_id, t] >= g["min_power"] * o[g_id, t])
            model.addConstr(p[g_id, t] <= g["max_power"] * o[g_id, t])
            model.addConstr(p[g_id, t] - p_prev <= g["ramp_up"] * o_prev + g["max_power"] * vp[g_id, t])
            model.addConstr(p_prev - p[g_id, t] <= g["ramp_down"] * o[g_id, t] + g["min_power"] * vm[g_id, t])

            # Bids logic
            bids = g.get("bids", [])
            for i, bid in enumerate(bids):
                b_id = bid["id"]
                model.addConstr(se[g_id, b_id, t] <= bid["max_amount"] * w[g_id, b_id, t])
                if i < len(bids) - 1:
                    next_bid_id = bids[i+1]["id"]
                    model.addConstr(se[g_id, b_id, t] >= bid["max_amount"] * w[g_id, next_bid_id, t])
                    model.addConstr(w[g_id, b_id, t] >= w[g_id, next_bid_id, t])
                if i == 0:
                    model.addConstr(w[g_id, b_id, t] <= o[g_id, t])
            
            model.addConstr(p[g_id, t] == sum(se[g_id, bid["id"], t] for bid in bids))

            # GFPP Phimax
            if g["is_gfpp"]:
                max_gp_bound = g["max_gas_price_fraction"] * max_gas_price_mmBtu
                H_u1 = g["heat_rate_coefficients"]["H_u1"]
                thresh = []
                for bid in bids:
                    val = max_gp_bound
                    if H_u1 is not None and H_u1 > 0:
                        val = min(bid["price"] / H_u1, max_gp_bound)
                    thresh.append(val)
                
                expr = 0
                for i in range(len(bids) - 1):
                    expr += thresh[i] * (w[g_id, bids[i]["id"], t] - w[g_id, bids[i+1]["id"], t])
                if len(bids) > 0:
                    expr += thresh[-1] * w[g_id, bids[-1]["id"], t]
                
                model.addConstr(phimax[g_id, t] == expr)
                
                j_idx = g["gas_junction"]
                z_idx = junc_to_zone.get(j_idx)
                if z_idx is not None:
                    model.addConstr(phimax[g_id, t] >= psi[z_idx, t] - max_gas_price_mmBtu * (1 - o[g_id, t]))

        # 2. Electricity Network
        for b in buses:
            b_id = b["id"]
            D_elec = b["demand_profile"][t-1] * eta_p
            gen_sum = sum(p[g["id"], t] for g in bus_to_gens[b_id])
            in_sum = sum(f[l["id"], t] for l in lines_to[b_id])
            out_sum = sum(f[l["id"], t] for l in lines_from[b_id])
            model.addConstr(gen_sum - D_elec == out_sum - in_sum)

        for l in lines:
            l_id = l["id"]
            model.addConstr(f[l_id, t] == l["susceptance"] * (theta[l["from_bus"], t] - theta[l["to_bus"], t]))
            model.addConstr(theta[l["from_bus"], t] - theta[l["to_bus"], t] <= l["angle_diff_limit"])
            model.addConstr(theta[l["from_bus"], t] - theta[l["to_bus"], t] >= -l["angle_diff_limit"])

        # 3. Gas Network
        for j in junctions:
            j_id = j["id"]
            if j["is_source"]:
                model.addConstr(sg[j_id, t] == sum(sgs[j_id, intv["id"], t] for intv in j["supply_intervals"]))
            else:
                model.addConstr(sg[j_id, t] == 0)
                
            D_gas = j["gas_demand_profile"][t-1] * eta_g
            model.addConstr(lg[j_id, t] + qg[j_id, t] == D_gas)
            model.addConstr(qg[j_id, t] <= D_gas)
            
            in_sum = sum(phig[c["id"], t] for c in conn_to[j_id])
            out_sum = sum(phig[c["id"], t] for c in conn_from[j_id])
            model.addConstr(sg[j_id, t] - lg[j_id, t] - gamma[j_id, t] == out_sum - in_sum)
            
            gfpps_at_j = junc_to_gfpps[j_id]
            if len(gfpps_at_j) > 0:
                expr = 0
                for g in gfpps_at_j:
                    H_u2 = g["heat_rate_coefficients"]["H_u2"] or 0.0
                    H_u1 = g["heat_rate_coefficients"]["H_u1"] or 0.0
                    H_u0 = g["heat_rate_coefficients"]["H_u0"] or 0.0
                    expr += H_u2 * p[g["id"], t] * p[g["id"], t] + H_u1 * p[g["id"], t] + H_u0 * o[g["id"], t]
                model.addQConstr(gamma[j_id, t] >= expr)
            else:
                model.addConstr(gamma[j_id, t] == 0)

        for c in connections:
            c_id = c["id"]
            from_j = c["from_junction"]
            to_j = c["to_junction"]
            if c["type"] == "pipeline":
                W_f = c.get("weymouth_factor")
                if W_f is not None:
                    model.addQConstr(pisq[from_j, t] - pisq[to_j, t] >= W_f * phig[c_id, t] * phig[c_id, t])
            else:
                lb_comp = c.get("compression_ratio_lb")
                ub_comp = c.get("compression_ratio_ub")
                if lb_comp is not None:
                    model.addConstr(pisq[to_j, t] >= (lb_comp**2) * pisq[from_j, t])
                if ub_comp is not None:
                    model.addConstr(pisq[to_j, t] <= (ub_comp**2) * pisq[from_j, t])
                
                lb_ctrl = c.get("control_ratio_lb")
                ub_ctrl = c.get("control_ratio_ub")
                if lb_ctrl is not None:
                    model.addConstr(pisq[to_j, t] >= (lb_ctrl**2) * pisq[from_j, t])
                if ub_ctrl is not None:
                    model.addConstr(pisq[to_j, t] <= (ub_ctrl**2) * pisq[from_j, t])

    # Objective
    cost_elec = 0
    cost_gas = 0
    for t in range(1, T+1):
        for g in gen_list:
            g_id = g["id"]
            cost_elec += g["no_load_cost"] * o[g_id, t]
            cost_elec += r[g_id, t]
            for bid in g.get("bids", []):
                cost_elec += bid["price"] * se[g_id, bid["id"], t]

        for j in junctions:
            j_id = j["id"]
            cost_gas += j["demand_shedding_cost"] * qg[j_id, t]
            if j["is_source"]:
                for intv in j["supply_intervals"]:
                    cost_gas += intv["slope"] * sgs[j_id, intv["id"], t]

    model.setObjective(beta * cost_elec + (1 - beta) * cost_gas, GRB.MINIMIZE)

    # Callback for logging solutions
    def cb(model, where):
        if where == GRB.Callback.MIPSOL and logger:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            try:
                prim = {}
                for t_val in range(1, T+1):
                    for g in gen_list:
                        g_id = g["id"]
                        prim[f"o_{g_id}_{t_val}"] = float(model.cbGetSolution(o[g_id, t_val]))
                        prim[f"vp_{g_id}_{t_val}"] = float(model.cbGetSolution(vp[g_id, t_val]))
                        prim[f"vm_{g_id}_{t_val}"] = float(model.cbGetSolution(vm[g_id, t_val]))
                        prim[f"r_{g_id}_{t_val}"] = float(model.cbGetSolution(r[g_id, t_val]))
                        prim[f"p_{g_id}_{t_val}"] = float(model.cbGetSolution(p[g_id, t_val]))
                        if g["is_gfpp"]:
                            prim[f"phimax_{g_id}_{t_val}"] = float(model.cbGetSolution(phimax[g_id, t_val]))
                        for bid in g.get("bids", []):
                            b_id = bid["id"]
                            prim[f"w_{g_id}_{b_id}_{t_val}"] = float(model.cbGetSolution(w[g_id, b_id, t_val]))
                            prim[f"se_{g_id}_{b_id}_{t_val}"] = float(model.cbGetSolution(se[g_id, b_id, t_val]))
                            
                    for l in lines:
                        prim[f"f_{l['id']}_{t_val}"] = float(model.cbGetSolution(f[l["id"], t_val]))
                    for b in buses:
                        prim[f"theta_{b['id']}_{t_val}"] = float(model.cbGetSolution(theta[b["id"], t_val]))
                    for j in junctions:
                        j_id = j["id"]
                        prim[f"sg_{j_id}_{t_val}"] = float(model.cbGetSolution(sg[j_id, t_val]))
                        prim[f"pisq_{j_id}_{t_val}"] = float(model.cbGetSolution(pisq[j_id, t_val]))
                        prim[f"lg_{j_id}_{t_val}"] = float(model.cbGetSolution(lg[j_id, t_val]))
                        prim[f"qg_{j_id}_{t_val}"] = float(model.cbGetSolution(qg[j_id, t_val]))
                        prim[f"gamma_{j_id}_{t_val}"] = float(model.cbGetSolution(gamma[j_id, t_val]))
                        if j["is_source"]:
                            for intv in j["supply_intervals"]:
                                prim[f"sgs_{j_id}_{intv['id']}_{t_val}"] = float(model.cbGetSolution(sgs[j_id, intv["id"], t_val]))
                    for c in connections:
                        prim[f"phig_{c['id']}_{t_val}"] = float(model.cbGetSolution(phig[c["id"], t_val]))
                    for z in pricing_zones:
                        prim[f"psi_{z['id']}_{t_val}"] = float(model.cbGetSolution(psi[z["id"], t_val]))

                for g in gen_list:
                    g_id = g["id"]
                    prim[f"o_{g_id}_0"] = float(hist_o[g_id][0])
                    prim[f"p_{g_id}_0"] = float(g["initial_gen"])

                sol = {
                    "objective_value": obj,
                    "released_problem_variant": "integrated_primal_ucgna",
                    "objective_weight_beta": beta,
                    "primary_variables": prim
                }
                logger.log_solution(obj, sol)
            except Exception:
                logger.log(obj)

    model.optimize(cb)

    # Save final solution
    if model.SolCount > 0:
        prim = {}
        for t in range(1, T+1):
            for g in gen_list:
                g_id = g["id"]
                prim[f"o_{g_id}_{t}"] = float(o[g_id, t].X)
                prim[f"vp_{g_id}_{t}"] = float(vp[g_id, t].X)
                prim[f"vm_{g_id}_{t}"] = float(vm[g_id, t].X)
                prim[f"r_{g_id}_{t}"] = float(r[g_id, t].X)
                prim[f"p_{g_id}_{t}"] = float(p[g_id, t].X)
                if g["is_gfpp"]:
                    prim[f"phimax_{g_id}_{t}"] = float(phimax[g_id, t].X)
                for bid in g.get("bids", []):
                    b_id = bid["id"]
                    prim[f"w_{g_id}_{b_id}_{t}"] = float(w[g_id, b_id, t].X)
                    prim[f"se_{g_id}_{b_id}_{t}"] = float(se[g_id, b_id, t].X)

            for l in lines:
                prim[f"f_{l['id']}_{t}"] = float(f[l["id"], t].X)
            for b in buses:
                prim[f"theta_{b['id']}_{t}"] = float(theta[b["id"], t].X)
            for j in junctions:
                j_id = j["id"]
                prim[f"sg_{j_id}_{t}"] = float(sg[j_id, t].X)
                prim[f"pisq_{j_id}_{t}"] = float(pisq[j_id, t].X)
                prim[f"lg_{j_id}_{t}"] = float(lg[j_id, t].X)
                prim[f"qg_{j_id}_{t}"] = float(qg[j_id, t].X)
                prim[f"gamma_{j_id}_{t}"] = float(gamma[j_id, t].X)
                if j["is_source"]:
                    for intv in j["supply_intervals"]:
                        prim[f"sgs_{j_id}_{intv['id']}_{t}"] = float(sgs[j_id, intv["id"], t].X)
            for c in connections:
                prim[f"phig_{c['id']}_{t}"] = float(phig[c["id"], t].X)
            for z in pricing_zones:
                prim[f"psi_{z['id']}_{t}"] = float(psi[z["id"], t].X)

        for g in gen_list:
            g_id = g["id"]
            prim[f"o_{g_id}_0"] = float(hist_o[g_id][0])
            prim[f"p_{g_id}_0"] = float(g["initial_gen"])

        final_sol = {
            "objective_value": model.ObjVal,
            "released_problem_variant": "integrated_primal_ucgna",
            "objective_weight_beta": beta,
            "primary_variables": prim
        }
        
        with open(args.solution_path, "w") as f:
            json.dump(final_sol, f, indent=2)

if __name__ == "__main__":
    main()