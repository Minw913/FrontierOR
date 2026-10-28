import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    # Extract data
    sys = data["system"]
    scen_params = data["scenario_parameters"]
    tree = data["scenario_tree"]
    rep_days = data["representative_days"]
    lower_scens = data["lower_level_scenarios"]
    glob_params = data["global_parameters"]

    # Precompute node info
    node_dict = {node["id"]: node for node in tree}
    root_nodes = [node for node in tree if node["parent_id"] is None]
    
    # Compute demand multipliers and cost multipliers for each node
    node_demand_mult = {}
    node_cost_mult = {}
    
    def compute_mults(node, d_mult, c_mult):
        node_demand_mult[node["id"]] = d_mult
        node_cost_mult[node["id"]] = c_mult
        for i, child_id in enumerate(node["children"]):
            child = node_dict[child_id]
            if i == 0:
                new_d = d_mult * (1 + scen_params["demand_growth_high_percent"])
                new_c = c_mult * (1 + glob_params.get("cost_variation_percent", 0))
            elif i == 1:
                new_d = d_mult * (1 + scen_params["demand_growth_low_percent"])
                new_c = c_mult * (1 - glob_params.get("cost_variation_percent", 0))
            else:
                new_d = d_mult * (1 + (scen_params["demand_growth_high_percent"] + scen_params["demand_growth_low_percent"]) / 2)
                new_c = c_mult
            compute_mults(child, new_d, new_c)

    for root in root_nodes:
        compute_mults(root, 1.0, 1.0)

    # Total peak demand at node
    total_peak = {}
    for node in tree:
        total_peak[node["id"]] = sum(l["annual_peak_load_MW"] for l in sys["loads"]) * node_demand_mult[node["id"]]

    # Build Gurobi model
    env = gp.Env(empty=True)
    env.setParam("Seed", 0)
    env.setParam("MIPGap", 1e-4)
    env.setParam("NumericFocus", 0)
    env.setParam("Threads", 1)
    env.start()
    model = gp.Model("power_expansion", env=env)
    model.Params.TimeLimit = args.time_limit
    model.Params.OutputFlag = 0

    # Investment variables
    built_gen = {}
    built_line = {}
    built_stor = {}
    avail_gen = {}
    avail_line = {}
    avail_stor = {}

    for node in tree:
        nid = node["id"]
        for g in sys["generators"]:
            if not g["existing"]:
                built_gen[nid, g["id"]] = model.addVar(vtype=GRB.BINARY, name=f"bg_{nid}_{g['id']}")
                avail_gen[nid, g["id"]] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=g["Pmax_MW"], name=f"ag_{nid}_{g['id']}")
                if node["parent_id"] is None:
                    model.addConstr(avail_gen[nid, g["id"]] == g["Pmax_MW"] * built_gen[nid, g["id"]])
                else:
                    model.addConstr(avail_gen[nid, g["id"]] == avail_gen[node["parent_id"], g["id"]] + g["Pmax_MW"] * built_gen[nid, g["id"]])
        for l in sys["transmission_lines"]:
            if not l["existing"]:
                built_line[nid, l["id"]] = model.addVar(vtype=GRB.BINARY, name=f"bl_{nid}_{l['id']}")
                avail_line[nid, l["id"]] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=l["flow_limit_MW"], name=f"al_{nid}_{l['id']}")
                if node["parent_id"] is None:
                    model.addConstr(avail_line[nid, l["id"]] == l["flow_limit_MW"] * built_line[nid, l["id"]])
                else:
                    model.addConstr(avail_line[nid, l["id"]] == avail_line[node["parent_id"], l["id"]] + l["flow_limit_MW"] * built_line[nid, l["id"]])
        for s in sys["storage"]:
            if not s["existing"]:
                built_stor[nid, s["id"]] = model.addVar(vtype=GRB.BINARY, name=f"bs_{nid}_{s['id']}")
                avail_stor[nid, s["id"]] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=s["capacity_MW"], name=f"as_{nid}_{s['id']}")
                if node["parent_id"] is None:
                    model.addConstr(avail_stor[nid, s["id"]] == s["capacity_MW"] * built_stor[nid, s["id"]])
                else:
                    model.addConstr(avail_stor[nid, s["id"]] == avail_stor[node["parent_id"], s["id"]] + s["capacity_MW"] * built_stor[nid, s["id"]])

    # Reserve margin constraint
    for node in tree:
        nid = node["id"]
        derated_cap = gp.LinExpr()
        for g in sys["generators"]:
            df = glob_params["derating_factors"].get(g["type"], 0)
            if g["existing"]:
                derated_cap += g["Pmax_MW"] * df
            else:
                derated_cap += avail_gen[nid, g["id"]] * df
        model.addConstr(derated_cap >= (1 + glob_params["reserve_margin_requirement"]) * total_peak[nid])

    # Operational variables
    alpha = {}
    gamma = {}
    p_gen = {}
    s_res = {}
    f_fwd = {}
    f_bwd = {}
    u_stor = {}
    v_stor = {}
    r_stor = {}

    obj = gp.LinExpr()

    rep_day_weight = 1.0 / len(rep_days)

    for node in tree:
        nid = node["id"]
        prob = node["probability"]
        cost_mult = node_cost_mult[nid]
        
        # Investment cost
        for g in sys["generators"]:
            if not g["existing"]:
                base_cost = g["capital_cost_per_kW"]
                if base_cost == 0:
                    base_cost = glob_params["capital_costs_per_kW"].get(g["type"], 0)
                obj += prob * base_cost * 1000 * g["Pmax_MW"] * built_gen[nid, g["id"]] * cost_mult
        for l in sys["transmission_lines"]:
            if not l["existing"]:
                base_cost = l["capital_cost_per_kW"]
                if base_cost == 0:
                    base_cost = glob_params["capital_costs_per_kW"]["transmission_line"]
                obj += prob * base_cost * 1000 * l["flow_limit_MW"] * built_line[nid, l["id"]] * cost_mult
        for s in sys["storage"]:
            if not s["existing"]:
                base_cost = s["capital_cost_per_kW"]
                if base_cost == 0:
                    base_cost = glob_params["capital_costs_per_kW"]["energy_storage"]
                obj += prob * base_cost * 1000 * s["capacity_MW"] * built_stor[nid, s["id"]] * cost_mult

        for day in rep_days:
            day_id = day["id"]
            scens = lower_scens[str(day_id)]
            num_scens = len(scens)
            scen_weight = 1.0 / num_scens
            for si, scen in enumerate(scens):
                for h in range(scen_params["hours_per_day"]):
                    demand = {b: 0.0 for b in sys["buses"]}
                    for load in sys["loads"]:
                        b = load["bus"]
                        d = load["annual_peak_load_MW"] * node_demand_mult[nid] * day["load_profile"][h] * scen["demand_multipliers"][h]
                        demand[b] = d
                    
                    for g in sys["generators"]:
                        key = (nid, day_id, si, h, g["id"])
                        alpha[key] = model.addVar(vtype=GRB.BINARY, name=f"a_{'_'.join(map(str, key))}")
                        gamma[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"g_{'_'.join(map(str, key))}")
                        p_gen[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"p_{'_'.join(map(str, key))}")
                        s_res[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"s_{'_'.join(map(str, key))}")
                        
                        if g["type"] == "solar":
                            profile = day["solar_profile"][h] * scen["solar_multipliers"][h]
                        elif g["type"] == "wind":
                            profile = scen["wind_power_factors"][h]
                        else:
                            profile = 1.0
                            
                        if g["existing"]:
                            cap_bound = g["Pmax_MW"]
                            model.addConstr(p_gen[key] + s_res[key] <= cap_bound * profile * alpha[key])
                        else:
                            cap_bound = avail_gen[nid, g["id"]]
                            model.addConstr(p_gen[key] + s_res[key] <= cap_bound * profile)
                            if g["Pmax_MW"] > 0:
                                model.addConstr(alpha[key] <= cap_bound / g["Pmax_MW"])
                        
                        model.addConstr(p_gen[key] >= g["Pmin_MW"] * alpha[key])
                        
                        if h > 0:
                            prev_key = (nid, day_id, si, h-1, g["id"])
                            model.addConstr(gamma[key] >= alpha[key] - alpha[prev_key])
                            model.addConstr(p_gen[key] - p_gen[prev_key] <= g["ramp_MW_per_h"])
                            model.addConstr(p_gen[prev_key] - p_gen[key] <= g["ramp_MW_per_h"])
                        else:
                            model.addConstr(gamma[key] >= alpha[key])
                        
                        if g["min_on_h"] > 1:
                            H = scen_params["hours_per_day"]
                            end_h = min(h + g["min_on_h"], H)
                            model.addConstr(gp.quicksum(alpha[(nid, day_id, si, t, g["id"])] for t in range(h, end_h)) >= g["min_on_h"] * gamma[key])
                        if g["min_off_h"] > 1:
                            H = scen_params["hours_per_day"]
                            end_h = min(h + g["min_off_h"], H)
                            if h > 0:
                                prev_alpha = alpha[(nid, day_id, si, h-1, g["id"])]
                                model.addConstr(gp.quicksum(1 - alpha[(nid, day_id, si, t, g["id"])] for t in range(h, end_h)) >= g["min_off_h"] * (prev_alpha - alpha[key]))
                        
                        op_cost = g["a_cost"] * alpha[key] + (g["b_cost"] + g["c_cost"] * g["Pmax_MW"]) * p_gen[key] + g["startup_cost"] * gamma[key]
                        obj += prob * rep_day_weight * scen_weight * op_cost

                    for b in sys["buses"]:
                        res_at_b = gp.quicksum(s_res[(nid, day_id, si, h, g["id"])] for g in sys["generators"] if g["bus"] == b)
                        model.addConstr(res_at_b >= glob_params["spinning_reserve_percent"] * demand[b])
                        
                    for l in sys["transmission_lines"]:
                        key = (nid, day_id, si, h, l["id"])
                        cap = l["flow_limit_MW"] if l["existing"] else avail_line[nid, l["id"]]
                        f_fwd[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=cap, name=f"ff_{'_'.join(map(str, key))}")
                        f_bwd[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=cap, name=f"fb_{'_'.join(map(str, key))}")
                        
                    for s in sys["storage"]:
                        key = (nid, day_id, si, h, s["id"])
                        cap = s["capacity_MW"] if s["existing"] else avail_stor[nid, s["id"]]
                        u_stor[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=cap, name=f"u_{'_'.join(map(str, key))}")
                        v_stor[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=cap, name=f"v_{'_'.join(map(str, key))}")
                        r_stor[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=cap, name=f"r_{'_'.join(map(str, key))}")
                        
                        eff = s["efficiency"]
                        if h == 0:
                            model.addConstr(r_stor[key] == 0)
                        else:
                            prev_key = (nid, day_id, si, h-1, s["id"])
                            model.addConstr(r_stor[key] == r_stor[prev_key] + eff * u_stor[prev_key] - v_stor[prev_key])
                        model.addConstr(v_stor[key] <= r_stor[key])
                        
                    for b in sys["buses"]:
                        gen_b = gp.quicksum(p_gen[(nid, day_id, si, h, g["id"])] for g in sys["generators"] if g["bus"] == b)
                        stor_b = gp.quicksum(v_stor[(nid, day_id, si, h, s["id"])] - u_stor[(nid, day_id, si, h, s["id"])] for s in sys["storage"] if s["bus"] == b)
                        
                        flow_out = gp.LinExpr()
                        flow_in = gp.LinExpr()
                        for l in sys["transmission_lines"]:
                            if l["from_bus"] == b:
                                flow_out += f_fwd[(nid, day_id, si, h, l["id"])]
                                flow_in += f_bwd[(nid, day_id, si, h, l["id"])] * (1 - l["loss_factor"])
                            if l["to_bus"] == b:
                                flow_out += f_bwd[(nid, day_id, si, h, l["id"])]
                                flow_in += f_fwd[(nid, day_id, si, h, l["id"])] * (1 - l["loss_factor"])
                                
                        model.addConstr(demand[b] + flow_out - flow_in == gen_b + stor_b)

    model.setObjective(obj, GRB.MINIMIZE)
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)
    
    model.optimize(callback)
    
    solution = {
        "objective_value": model.ObjVal if model.SolCount > 0 else 0.0,
        "alpha": {},
        "gamma": {},
        "p_gen": {},
        "s_res": {},
        "f_fwd": {},
        "f_bwd": {},
        "u_stor": {},
        "v_stor": {},
        "r_stor": {},
        "investments": {}
    }
    
    if model.SolCount > 0:
        for node in tree:
            nid = node["id"]
            solution["investments"][str(nid)] = {
                "generators": {},
                "lines": {},
                "storage": {}
            }
            for g in sys["generators"]:
                if not g["existing"]:
                    val = round(built_gen[nid, g["id"]].X)
                    solution["investments"][str(nid)]["generators"][g["id"]] = val
            for l in sys["transmission_lines"]:
                if not l["existing"]:
                    val = round(built_line[nid, l["id"]].X)
                    solution["investments"][str(nid)]["lines"][l["id"]] = val
            for s in sys["storage"]:
                if not s["existing"]:
                    val = round(built_stor[nid, s["id"]].X)
                    solution["investments"][str(nid)]["storage"][s["id"]] = val

        for key, var in alpha.items():
            if abs(var.X) > 1e-6:
                k = f"{key[0]}_{key[1]}_{key[2]}_{key[3]}_{key[4]}"
                solution["alpha"][k] = var.X
        for key, var in gamma.items():
            if abs(var.X) > 1e-6:
                k = f"{key[0]}_{key[1]}_{key[2]}_{key[3]}_{key[4]}"
                solution["gamma"][k] = var.X
        for key, var in p_gen.items():
            if abs(var.X) > 1e-6:
                k = f"{key[0]}_{key[1]}_{key[2]}_{key[3]}_{key[4]}"
                solution["p_gen"][k] = var.X
        for key, var in s_res.items():
            if abs(var.X) > 1e-6:
                k = f"{key[0]}_{key[1]}_{key[2]}_{key[3]}_{key[4]}"
                solution["s_res"][k] = var.X
        for key, var in f_fwd.items():
            if abs(var.X) > 1e-6:
                k = f"{key[0]}_{key[1]}_{key[2]}_{key[3]}_{key[4]}"
                solution["f_fwd"][k] = var.X
        for key, var in f_bwd.items():
            if abs(var.X) > 1e-6:
                k = f"{key[0]}_{key[1]}_{key[2]}_{key[3]}_{key[4]}"
                solution["f_bwd"][k] = var.X
        for key, var in u_stor.items():
            if abs(var.X) > 1e-6:
                k = f"{key[0]}_{key[1]}_{key[2]}_{key[3]}_{key[4]}"
                solution["u_stor"][k] = var.X
        for key, var in v_stor.items():
            if abs(var.X) > 1e-6:
                k = f"{key[0]}_{key[1]}_{key[2]}_{key[3]}_{key[4]}"
                solution["v_stor"][k] = var.X
        for key, var in r_stor.items():
            if abs(var.X) > 1e-6:
                k = f"{key[0]}_{key[1]}_{key[2]}_{key[3]}_{key[4]}"
                solution["r_stor"][k] = var.X

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    solve()