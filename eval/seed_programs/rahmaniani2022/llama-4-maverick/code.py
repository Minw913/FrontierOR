import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def create_model(instance_data):
    model = Model("power_grid_planning")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    
    num_zones = instance_data["problem_parameters"]["num_zones"]
    num_weeks = instance_data["problem_parameters"]["num_weeks"]
    hours_per_subperiod = instance_data["problem_parameters"]["hours_per_subperiod"]
    total_hours = instance_data["problem_parameters"]["total_hours"]
    subperiod_weights = instance_data["subperiod_weights"]
    generators = instance_data["generators"]
    generator_parameters = instance_data["generator_parameters"]
    existing_capacity = instance_data["existing_capacity"]
    demand = instance_data["demand"]
    consumer_segments = instance_data["consumer_segments"]
    transmission_lines = instance_data["transmission_lines"]
    availability = instance_data["availability"]
    hydro_inflow = instance_data.get("hydro_inflow", {})
    policy = instance_data["policy"]
    released_investment_convention = instance_data["released_investment_convention"]

    y_P_new = {}  
    y_P_ret = {}  
    y_E_new = {}  
    y_F_new = {}  
    
    x_inj = {}  
    x_wdw = {}  
    x_soc = {}  
    x_nse = {}  
    x_lvl = {}  
    x_spill = {}  
    x_flow = {}  
    x_commit = {}  
    x_start = {}  
    x_shut = {}  
    x_RPS = {}  
    x_CO2 = {}  
    
    for gen in generators:
        cluster_id = gen["cluster_id"]
        is_UC = gen.get("is_UC", False)
        if is_UC:
            y_P_new[cluster_id] = model.addVar(vtype=GRB.INTEGER, name=f"y_P_new[{cluster_id}]")
            y_P_ret[cluster_id] = model.addVar(vtype=GRB.INTEGER, name=f"y_P_ret[{cluster_id}]")
        else:
            y_P_new[cluster_id] = model.addVar(vtype=GRB.CONTINUOUS, name=f"y_P_new[{cluster_id}]")
            y_P_ret[cluster_id] = model.addVar(vtype=GRB.CONTINUOUS, name=f"y_P_ret[{cluster_id}]")
        if gen.get("is_storage", False):
            if is_UC:
                y_E_new[cluster_id] = model.addVar(vtype=GRB.INTEGER, name=f"y_E_new[{cluster_id}]")
            else:
                y_E_new[cluster_id] = model.addVar(vtype=GRB.CONTINUOUS, name=f"y_E_new[{cluster_id}]")
    
    for line in transmission_lines:
        line_id = line["line_id"]
        y_F_new[line_id] = model.addVar(vtype=GRB.INTEGER, name=f"y_F_new[{line_id}]")
    
    for w in range(num_weeks):
        for t in range(hours_per_subperiod):
            hour = w * hours_per_subperiod + t
            for gen in generators:
                cluster_id = gen["cluster_id"]
                if gen.get("is_storage", False):
                    x_inj[cluster_id, hour] = model.addVar(name=f"x_inj[{cluster_id},{hour}]")
                    x_wdw[cluster_id, hour] = model.addVar(name=f"x_wdw[{cluster_id},{hour}]")
                    x_soc[cluster_id, hour] = model.addVar(name=f"x_soc[{cluster_id},{hour}]")
                else:
                    x_inj[cluster_id, hour] = model.addVar(name=f"x_inj[{cluster_id},{hour}]")
                if gen.get("is_hydro", False):
                    x_lvl[cluster_id, hour] = model.addVar(name=f"x_lvl[{cluster_id},{hour}]")
                    x_spill[cluster_id, hour] = model.addVar(name=f"x_spill[{cluster_id},{hour}]")
                if gen.get("is_UC", False):
                    x_commit[cluster_id, hour] = model.addVar(name=f"x_commit[{cluster_id},{hour}]")
                    x_start[cluster_id, hour] = model.addVar(name=f"x_start[{cluster_id},{hour}]")
                    x_shut[cluster_id, hour] = model.addVar(name=f"x_shut[{cluster_id},{hour}]")
            for line in transmission_lines:
                line_id = line["line_id"]
                x_flow[line_id, hour] = model.addVar(name=f"x_flow[{line_id},{hour}]")
            for z in range(num_zones):
                for segment in consumer_segments:
                    if segment["zone"] == z:
                        x_nse[segment["segment_id"], z, hour] = model.addVar(name=f"x_nse[{segment['segment_id']},{z},{hour}]")
        if policy["type"] == "RPS":
            x_RPS[w] = model.addVar(name=f"x_RPS[{w}]")
        if policy["type"] == "CO2":
            x_CO2[w] = model.addVar(name=f"x_CO2[{w}]")
    
    objective = quicksum(
        generator_parameters.get(gen["resource_type"], {}).get("investment_cost_per_MW_yr", 0) * y_P_new[gen["cluster_id"]]
        for gen in generators 
    )
    objective += quicksum(
        generator_parameters.get("battery", {}).get("energy_investment_cost_per_MWh_yr", 0) * y_E_new[gen["cluster_id"]]
        for gen in generators if gen.get("is_storage", False)
    )
    objective += quicksum(
        line.get("investment_cost_per_MW_yr", 0) * y_F_new[line["line_id"]]
        for line in transmission_lines
    )
    objective += quicksum(
        generator_parameters.get(gen["resource_type"], {}).get("fixed_OM_cost_per_MW_yr", 0) * (existing_capacity.get(str(gen["cluster_id"]), 0) + y_P_new[gen["cluster_id"]] - y_P_ret[gen["cluster_id"]])
        for gen in generators
    )
    objective += quicksum(
        generator_parameters.get("battery", {}).get("energy_fixed_OM_cost_per_MWh_yr", 0) * (gen.get("capacity_size_MW", 0) * gen.get("energy_capacity_size_MWh", 0) + y_E_new.get(gen["cluster_id"], 0))
        for gen in generators if gen.get("is_storage", False)
    )
    for w in range(num_weeks):
        weight = subperiod_weights[w]
        for t in range(hours_per_subperiod):
            hour = w * hours_per_subperiod + t
            objective += weight * quicksum(
                generator_parameters.get(gen["resource_type"], {}).get("variable_cost_per_MWh", 0) * x_inj.get((gen["cluster_id"], hour), 0)
                for gen in generators 
            )
            objective += weight * quicksum(
                generator_parameters.get("battery", {}).get("variable_cost_per_MWh", 0) * x_wdw.get((gen["cluster_id"], hour), 0)
                for gen in generators if gen.get("is_storage", False)
            )
            objective += weight * quicksum(
                segment["cost_per_MWh"] * x_nse.get((segment["segment_id"], z, hour), 0)
                for z in range(num_zones) for segment in consumer_segments if segment["zone"] == z
            )
            objective += weight * quicksum(
                generator_parameters.get(gen["resource_type"], {}).get("start_up_cost", 0) * x_start.get((gen["cluster_id"], hour), 0)
                for gen in generators if gen.get("is_UC", False)
            )
        if policy["type"] == "RPS":
            objective += weight * policy.get("RPS_noncompliance_cost_per_MWh", 0) * x_RPS.get(w, 0)
        if policy["type"] == "CO2":
            objective += weight * policy.get("CO2_noncompliance_cost_per_ton", 0) * x_CO2.get(w, 0)
    model.setObjective(objective, GRB.MINIMIZE)
    
    for gen in generators:
        cluster_id = gen["cluster_id"]
        model.addConstr(y_P_ret[cluster_id] <= existing_capacity.get(str(cluster_id), 0))
        model.addConstr(y_P_new[cluster_id] >= 0)
        if gen.get("is_storage", False):
            model.addConstr(y_E_new[cluster_id] >= 0)
    
    for line in transmission_lines:
        line_id = line["line_id"]
        model.addConstr(y_F_new[line_id] >= 0)
    
    for w in range(num_weeks):
        for t in range(hours_per_subperiod):
            hour = w * hours_per_subperiod + t
            for z in range(num_zones):
                model.addConstr(
                    quicksum(x_inj.get((gen["cluster_id"], hour), 0) for gen in generators if gen["zone"] == z)
                    - quicksum(x_wdw.get((gen["cluster_id"], hour), 0) for gen in generators if gen.get("is_storage", False) and gen["zone"] == z)
                    + quicksum(x_flow.get((line["line_id"], hour), 0) for line in transmission_lines if line["to_zone"] == z)
                    - quicksum(x_flow.get((line["line_id"], hour), 0) for line in transmission_lines if line["from_zone"] == z)
                    + quicksum(x_nse.get((segment["segment_id"], z, hour), 0) for segment in consumer_segments if segment["zone"] == z)
                    == demand.get(str(z), [0]*total_hours)[hour]
                )
                for segment in consumer_segments:
                    if segment["zone"] == z:
                        model.addConstr(x_nse.get((segment["segment_id"], z, hour), 0) <= segment["max_nse_fraction"] * demand.get(str(z), [0]*total_hours)[hour])
    
    if policy["type"] == "RPS":
        for w in range(num_weeks):
            weight = subperiod_weights[w]
            rps_generation = quicksum(
                weight * x_inj.get((gen["cluster_id"], hour), 0)
                for t in range(hours_per_subperiod)
                for gen in generators if gen.get("is_RPS_qualifying", False)
                for hour in [w * hours_per_subperiod + t]
            )
            total_demand = weight * sum(demand.get(str(z), [0]*total_hours)[w * hours_per_subperiod + t] for z in range(num_zones) for t in range(hours_per_subperiod))
            model.addConstr(rps_generation + x_RPS.get(w, 0) >= policy.get("RPS_share", 0) * total_demand)
    
    if policy["type"] == "CO2":
        for w in range(num_weeks):
            weight = subperiod_weights[w]
            co2_emissions = quicksum(
                weight * (generator_parameters.get(gen["resource_type"], {}).get("CO2_emission_tons_per_MWh", 0) * x_inj.get((gen["cluster_id"], hour), 0) if gen.get("resource_type") in generator_parameters else 0)
                + weight * (generator_parameters.get("battery", {}).get("CO2_emission_tons_per_MWh", 0) * x_wdw.get((gen["cluster_id"], hour), 0) if gen.get("is_storage", False) else 0)
                for t in range(hours_per_subperiod)
                for gen in generators
                for hour in [w * hours_per_subperiod + t]
            )
            total_demand = weight * sum(demand.get(str(z), [0]*total_hours)[w * hours_per_subperiod + t] for z in range(num_zones) for t in range(hours_per_subperiod))
            model.addConstr(co2_emissions - x_CO2.get(w, 0) <= policy.get("CO2_cap_tons_per_MWh", 0) * total_demand)
    
    model.update()
    return model, y_P_new, y_P_ret, y_E_new, y_F_new, x_inj, x_wdw, x_soc, x_nse, x_lvl, x_spill, x_flow, x_commit, x_start, x_shut, x_RPS, x_CO2

def extract_solution(model, y_P_new, y_P_ret, y_E_new, y_F_new, x_inj, x_wdw, x_soc, x_nse, x_lvl, x_spill, x_flow, x_commit, x_start, x_shut, x_RPS, x_CO2, generators, transmission_lines, consumer_segments, existing_capacity, generator_parameters):
    solution = {
        "objective_value": model.ObjVal,
        "investments": {},
        "storage_energy_investments": {},
        "transmission_investments": {},
    }
    for gen in generators:
        cluster_id = gen["cluster_id"]
        solution["investments"][str(cluster_id)] = {
            "resource_type": gen["resource_type"],
            "zone": gen["zone"],
            "new_units": y_P_new[cluster_id].X if y_P_new[cluster_id].VType != GRB.CONTINUOUS else y_P_new[cluster_id].X,
            "new_MW": y_P_new[cluster_id].X,
            "retired_units": y_P_ret[cluster_id].X if y_P_ret[cluster_id].VType != GRB.CONTINUOUS else y_P_ret[cluster_id].X,
            "retired_MW": y_P_ret[cluster_id].X,
            "total_capacity_MW": existing_capacity.get(str(cluster_id), 0) + y_P_new[cluster_id].X - y_P_ret[cluster_id].X,
        }
        if gen.get("is_storage", False):
            solution["storage_energy_investments"][str(cluster_id)] = {
                "new_units": y_E_new[cluster_id].X if y_E_new[cluster_id].VType != GRB.CONTINUOUS else y_E_new[cluster_id].X,
                "new_MWh": y_E_new[cluster_id].X,
                "total_energy_MWh": gen.get("energy_capacity_size_MWh", 0) * (existing_capacity.get(str(cluster_id), 0) + y_P_new[cluster_id].X - y_P_ret[cluster_id].X) + y_E_new[cluster_id].X,
            }
    for line in transmission_lines:
        line_id = line["line_id"]
        solution["transmission_investments"][str(line_id)] = {
            "new_MW": y_F_new[line_id].X,
            "total_capacity_MW": line["existing_capacity_MW"] + y_F_new[line_id].X,
        }
    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path")
    args = parser.parse_args()
    
    with open(args.instance_path, "r") as f:
        instance_data = json.load(f)
    
    model, y_P_new, y_P_ret, y_E_new, y_F_new, x_inj, x_wdw, x_soc, x_nse, x_lvl, x_spill, x_flow, x_commit, x_start, x_shut, x_RPS, x_CO2 = create_model(instance_data)
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    model.Params.TimeLimit = args.time_limit
    model.optimize()
    
    if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
        solution = extract_solution(model, y_P_new, y_P_ret, y_E_new, y_F_new, x_inj, x_wdw, x_soc, x_nse, x_lvl, x_spill, x_flow, x_commit, x_start, x_shut, x_RPS, x_CO2, instance_data["generators"], instance_data["transmission_lines"], instance_data["consumer_segments"], instance_data["existing_capacity"], instance_data["generator_parameters"])
        with open(args.solution_path, "w") as f:
            json.dump(solution, f, indent=2)
        if logger:
            logger.log_solution(model.ObjVal, solution)
    else:
        print("No solution found")

if __name__ == "__main__":
    main()