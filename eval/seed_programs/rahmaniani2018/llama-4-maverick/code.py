import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def create_model(instance_data):
    model = Model("ElectricitySystemPlanning")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    
    num_generators = instance_data["problem_dimensions"]["num_generators"]
    num_storage_resources = instance_data["problem_dimensions"]["num_storage_resources"]
    num_transmission_lines = instance_data["problem_dimensions"]["num_transmission_lines"]
    num_zones = instance_data["problem_dimensions"]["num_zones"]
    num_subperiods = instance_data["subperiods"]["num_subperiods"]
    hours_per_subperiod = instance_data["subperiods"]["hours_per_subperiod"]
    total_timesteps = instance_data["problem_dimensions"]["total_timesteps"]
    subperiod_weights = instance_data["subperiods"]["subperiod_weights"]
    generators = instance_data["generators"]
    transmission_lines = instance_data["transmission_lines"]
    demand = instance_data["demand"]
    nse_segments = instance_data["nse_segments"]
    availability_profiles = instance_data["availability_profiles"]
    RPS_enabled = instance_data["policy"]["RPS_enabled"]
    RPS_share = instance_data["policy"]["RPS_share"]
    CO2_cap_enabled = instance_data["policy"]["CO2_cap_enabled"]
    CO2_cap_tons_per_MWh = instance_data["policy"]["CO2_cap_tons_per_MWh"]
    RPS_noncompliance_cost_per_MWh = instance_data["policy"]["RPS_noncompliance_cost_per_MWh"]
    CO2_noncompliance_cost_per_ton = instance_data["policy"]["CO2_noncompliance_cost_per_ton"]

    yPnew = {i: model.addVar(vtype=GRB.INTEGER, name=f"yPnew_{i}") for i in range(num_generators)}
    yPret = {i: model.addVar(vtype=GRB.INTEGER, name=f"yPret_{i}") for i in range(num_generators)}
    yP = {i: model.addVar(name=f"yP_{i}") for i in range(num_generators)}
    yEnew = {i: model.addVar(vtype=GRB.INTEGER, name=f"yEnew_{i}") for i in range(num_generators) if generators[i]["is_storage"]}
    yEret = {i: model.addVar(vtype=GRB.INTEGER, name=f"yEret_{i}") for i in range(num_generators) if generators[i]["is_storage"]}
    yE = {i: model.addVar(name=f"yE_{i}") for i in range(num_generators) if generators[i]["is_storage"]}
    yFnew = {l: model.addVar(vtype=GRB.INTEGER, name=f"yFnew_{l}") for l in range(num_transmission_lines)}
    xinj = {(i, t): model.addVar(name=f"xinj_{i}_{t}") for i in range(num_generators) for t in range(total_timesteps)}
    xwdw = {(i, t): model.addVar(name=f"xwdw_{i}_{t}") for i in range(num_generators) if generators[i]["is_storage"] for t in range(total_timesteps)}
    xsoc = {(i, t): model.addVar(name=f"xsoc_{i}_{t}") for i in range(num_generators) if generators[i]["is_storage"] for t in range(total_timesteps)}
    xflow = {(l, t): model.addVar(name=f"xflow_{l}_{t}") for l in range(num_transmission_lines) for t in range(total_timesteps)}
    xnse = {(s, z, t): model.addVar(name=f"xnse_{s}_{z}_{t}") for s in range(len(nse_segments)) for z in range(num_zones) for t in range(total_timesteps)}
    xcommit = {(i, t): model.addVar(vtype=GRB.INTEGER, name=f"xcommit_{i}_{t}") for i in range(num_generators) if generators[i]["is_UC"] for t in range(total_timesteps)}
    xstart = {(i, t): model.addVar(vtype=GRB.INTEGER, name=f"xstart_{i}_{t}") for i in range(num_generators) if generators[i]["is_UC"] for t in range(total_timesteps)}
    xshut = {(i, t): model.addVar(vtype=GRB.INTEGER, name=f"xshut_{i}_{t}") for i in range(num_generators) if generators[i]["is_UC"] for t in range(total_timesteps)}
    xRPS = {w: model.addVar(name=f"xRPS_{w}") for w in range(num_subperiods)} if RPS_enabled else {}
    xCO2 = {w: model.addVar(name=f"xCO2_{w}") for w in range(num_subperiods)} if CO2_cap_enabled else {}

    for i, gen in enumerate(generators):
        model.addConstr(yP[i] == gen["existing_capacity_MW"] + gen["capacity_size_MW"] * (yPnew[i] - yPret[i]), name=f"yP_{i}")
        model.addConstr(yPnew[i] * gen["capacity_size_MW"] <= gen["max_capacity_MW"] - gen["existing_capacity_MW"], name=f"max_new_cap_{i}")
        model.addConstr(yPret[i] * gen["capacity_size_MW"] <= gen["existing_capacity_MW"], name=f"max_ret_cap_{i}")
        if not gen["can_retire"]:
            model.addConstr(yPret[i] == 0, name=f"nonretirable_{i}")
        if gen["is_storage"]:
            model.addConstr(yE[i] == gen["existing_storage_capacity_MWh"] + gen["storage_capacity_size_MWh"] * (yEnew[i] - yEret[i]), name=f"yE_{i}")
            model.addConstr(yEnew[i] * gen["storage_capacity_size_MWh"] <= gen["max_storage_capacity_MWh"] - gen["existing_storage_capacity_MWh"], name=f"max_new_energy_cap_{i}")
            model.addConstr(yEret[i] * gen["storage_capacity_size_MWh"] <= gen["existing_storage_capacity_MWh"], name=f"max_ret_energy_cap_{i}")
            model.addConstr(yE[i] >= gen["min_duration_MWh_per_MW"] * yP[i], name=f"min_duration_{i}")
            model.addConstr(yE[i] <= gen["max_duration_MWh_per_MW"] * yP[i], name=f"max_duration_{i}")

    for l, line in enumerate(transmission_lines):
        model.addConstr(yFnew[l] <= line["max_new_capacity_MW"], name=f"max_new_transmission_{l}")
        for t in range(total_timesteps):
            model.addConstr(xflow[l, t] <= line["existing_capacity_MW"] + yFnew[l], name=f"transmission_upper_bound_{l}_{t}")
            model.addConstr(xflow[l, t] >= -(line["existing_capacity_MW"] + yFnew[l]), name=f"transmission_lower_bound_{l}_{t}")

    for z in range(num_zones):
        for t in range(total_timesteps):
            gen_in_zone = [i for i, gen in enumerate(generators) if gen["zone"] == z]
            inflow_lines = [l for l, line in enumerate(transmission_lines) if line["to_zone"] == z]
            outflow_lines = [l for l, line in enumerate(transmission_lines) if line["from_zone"] == z]
            lhs = quicksum(xinj.get((i, t), 0) for i in gen_in_zone)
            lhs -= quicksum(xwdw.get((i, t), 0) for i in gen_in_zone)
            lhs += quicksum(xflow[l, t] for l in inflow_lines)
            lhs -= quicksum(xflow[l, t] for l in outflow_lines)
            lhs += quicksum(xnse[s, z, t] for s in range(len(nse_segments)))
            rhs = demand[str(z)][t]
            model.addConstr(lhs == rhs, name=f"power_balance_{z}_{t}")

    objective = quicksum(gen["inv_cost_per_MW_yr"] * gen["capacity_size_MW"] * yPnew[i] for i, gen in enumerate(generators))
    objective += quicksum(gen["fom_cost_per_MW_yr"] * yP[i] for i, gen in enumerate(generators))
    objective += quicksum(gen.get("inv_cost_energy_per_MWh_yr", 0) * gen.get("storage_capacity_size_MWh", 0) * yEnew.get(i, 0) for i, gen in enumerate(generators) if gen["is_storage"])
    objective += quicksum(gen.get("fom_cost_energy_per_MWh_yr", 0) * yE.get(i, 0) for i, gen in enumerate(generators) if gen["is_storage"])
    objective += quicksum(line["inv_cost_per_MW_yr"] * yFnew[l] for l, line in enumerate(transmission_lines))
    model.setObjective(objective, GRB.MINIMIZE)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path")
    args = parser.parse_args()
    
    with open(args.instance_path, "r") as f:
        instance_data = json.load(f)
    
    model = create_model(instance_data)
    model.Params.TimeLimit = args.time_limit
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    model.optimize()
    
    if model.SolCount > 0:
        solution = {
            "objective_value": model.ObjVal,
            "model_variables": {},
            "investment_decisions": [],
            "transmission_decisions": []
        }
        
        for i, gen in enumerate(instance_data["generators"]):
            solution["investment_decisions"].append({
                "generator_id": i,
                "resource_type": gen["resource_type"],
                "zone": gen["zone"],
                "y_P_new": model.getVarByName(f"yPnew_{i}").X,
                "y_P_ret": model.getVarByName(f"yPret_{i}").X,
                "y_P_total_MW": model.getVarByName(f"yP_{i}").X,
                "y_E_new": model.getVarByName(f"yEnew_{i}").X if gen["is_storage"] else None,
                "y_E_ret": model.getVarByName(f"yEret_{i}").X if gen["is_storage"] else None,
                "y_E_total_MWh": model.getVarByName(f"yE_{i}").X if gen["is_storage"] else None
            })
        
        for l, line in enumerate(instance_data["transmission_lines"]):
            solution["transmission_decisions"].append({
                "line_id": l,
                "y_F_new_MW": model.getVarByName(f"yFnew_{l}").X,
                "total_capacity_MW": line["existing_capacity_MW"] + model.getVarByName(f"yFnew_{l}").X
            })
        
        with open(args.solution_path, "w") as f:
            json.dump(solution, f, indent=2)
    else:
        print("No feasible solution found.")

if __name__ == "__main__":
    main()