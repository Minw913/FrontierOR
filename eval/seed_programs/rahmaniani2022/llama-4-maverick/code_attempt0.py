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
    
    # Extract relevant data from instance_data
    num_zones = instance_data["problem_parameters"]["num_zones"]
    num_weeks = instance_data["problem_parameters"]["num_weeks"]
    hours_per_subperiod = instance_data["problem_parameters"]["hours_per_subperiod"]
    total_hours = instance_data["problem_parameters"]["total_hours"]
    num_generator_clusters = instance_data["problem_parameters"]["num_generator_clusters"]
    subperiod_weights = instance_data["subperiod_weights"]
    generators = instance_data["generators"]
    generator_parameters = instance_data["generator_parameters"]
    existing_capacity = instance_data["existing_capacity"]
    demand = instance_data["demand"]
    consumer_segments = instance_data["consumer_segments"]
    transmission_lines = instance_data["transmission_lines"]
    availability = instance_data["availability"]
    hydro_inflow = instance_data["hydro_inflow"]
    policy = instance_data["policy"]
    
    # Define variables
    y_P_new = {}  # New power capacity for generators
    y_P_ret = {}  # Retired power capacity for generators
    y_E_new = {}  # New energy capacity for storage
    y_F_new = {}  # New transmission capacity
    
    x_inj = {}  # Generation
    x_wdw = {}  # Storage withdrawal
    x_soc = {}  # Storage state of charge
    x_nse = {}  # Non-served energy
    x_lvl = {}  # Hydro reservoir level
    x_spill = {}  # Hydro spillage
    x_flow = {}  # Transmission flow
    x_commit = {}  # Unit commitment
    x_start = {}  # Unit start-up
    x_shut = {}  # Unit shutdown
    x_RPS = {}  # RPS slack
    x_CO2 = {}  # CO2 slack
    
    for i, gen in enumerate(generators):
        cluster_id = gen["cluster_id"]
        if gen["is_UC"]:
            y_P_new[cluster_id] = model.addVar(vtype=GRB.INTEGER, name=f"y_P_new[{cluster_id}]")
            y_P_ret[cluster_id] = model.addVar(vtype=GRB.INTEGER, name=f"y_P_ret[{cluster_id}]")
        else:
            y_P_new[cluster_id] = model.addVar(vtype=GRB.INTEGER, name=f"y_P_new[{cluster_id}]")
            y_P_ret[cluster_id] = model.addVar(vtype=GRB.CONTINUOUS, name=f"y_P_ret[{cluster_id}]")
        if gen["is_storage"]:
            y_E_new[cluster_id] = model.addVar(vtype=GRB.INTEGER, name=f"y_E_new[{cluster_id}]")
    
    for l, line in enumerate(transmission_lines):
        line_id = line["line_id"]
        y_F_new[line_id] = model.addVar(vtype=GRB.INTEGER, name=f"y_F_new[{line_id}]")
    
    for w in range(num_weeks):
        for t in range(hours_per_subperiod):
            hour = w * hours_per_subperiod + t
            for i, gen in enumerate(generators):
                cluster_id = gen["cluster_id"]
                if gen["is_storage"]:
                    x_inj[cluster_id, hour] = model.addVar(name=f"x_inj[{cluster_id},{hour}]")
                    x_wdw[cluster_id, hour] = model.addVar(name=f"x_wdw[{cluster_id},{hour}]")
                    x_soc[cluster_id, hour] = model.addVar(name=f"x_soc[{cluster_id},{hour}]")
                else:
                    x_inj[cluster_id, hour] = model.addVar(name=f"x_inj[{cluster_id},{hour}]")
                if gen["is_hydro"]:
                    x_lvl[cluster_id, hour] = model.addVar(name=f"x_lvl[{cluster_id},{hour}]")
                    x_spill[cluster_id, hour] = model.addVar(name=f"x_spill[{cluster_id},{hour}]")
                if gen["is_UC"]:
                    x_commit[cluster_id, hour] = model.addVar(name=f"x_commit[{cluster_id},{hour}]")
                    x_start[cluster_id, hour] = model.addVar(name=f"x_start[{cluster_id},{hour}]")
                    x_shut[cluster_id, hour] = model.addVar(name=f"x_shut[{cluster_id},{hour}]")
            for l, line in enumerate(transmission_lines):
                line_id = line["line_id"]
                x_flow[line_id, hour] = model.addVar(name=f"x_flow[{line_id},{hour}]")
            for z in range(num_zones):
                for s, segment in enumerate(consumer_segments):
                    if segment["zone"] == z:
                        x_nse[s, z, hour] = model.addVar(name=f"x_nse[{s},{z},{hour}]")
        if policy["type"] == "RPS":
            x_RPS[w] = model.addVar(name=f"x_RPS[{w}]")
        if policy["type"] == "CO2":
            x_CO2[w] = model.addVar(name=f"x_CO2[{w}]")
    
    # Objective function
    objective = quicksum(
        generator_parameters[gen["resource_type"]]["investment_cost_per_MW_yr"] * y_P_new[gen["cluster_id"]]
        for gen in generators if gen["resource_type"] in generator_parameters
    )
    objective += quicksum(
        generator_parameters["battery"]["energy_investment_cost_per_MWh_yr"] * y_E_new[gen["cluster_id"]]
        for gen in generators if gen["is_storage"]
    )
    objective += quicksum(
        line["investment_cost_per_MW_yr"] * y_F_new[line["line_id"]]
        for line in transmission_lines
    )
    objective += quicksum(
        generator_parameters[gen["resource_type"]]["fixed_OM_cost_per_MW_yr"] * (existing_capacity[str(gen["cluster_id"])] + y_P_new[gen["cluster_id"]] - y_P_ret[gen["cluster_id"]])
        for gen in generators if gen["resource_type"] in generator_parameters
    )
    objective += quicksum(
        generator_parameters["battery"]["energy_fixed_OM_cost_per_MWh_yr"] * (gen["capacity_size_MW"] * gen["energy_capacity_size_MWh"] + y_E_new[gen["cluster_id"]])
        for gen in generators if gen["is_storage"]
    )
    for w in range(num_weeks):
        weight = subperiod_weights[w]
        for t in range(hours_per_subperiod):
            hour = w * hours_per_subperiod + t
            objective += weight * quicksum(
                generator_parameters[gen["resource_type"]]["variable_cost_per_MWh"] * x_inj[gen["cluster_id"], hour]
                for gen in generators if gen["resource_type"] in generator_parameters
            )
            objective += weight * quicksum(
                generator_parameters["battery"]["variable_cost_per_MWh"] * x_wdw[gen["cluster_id"], hour]
                for gen in generators if gen["is_storage"]
            )
            objective += weight * quicksum(
                segment["cost_per_MWh"] * x_nse[s, z, hour]
                for z in range(num_zones) for s, segment in enumerate(consumer_segments) if segment["zone"] == z
            )
            objective += weight * quicksum(
                generator_parameters[gen["resource_type"]]["start_up_cost"] * x_start[gen["cluster_id"], hour]
                for gen in generators if gen["is_UC"]
            )
        if policy["type"] == "RPS":
            objective += weight * policy["RPS_noncompliance_cost_per_MWh"] * x_RPS[w]
        if policy["type"] == "CO2":
            objective += weight * policy["CO2_noncompliance_cost_per_ton"] * x_CO2[w]
    model.setObjective(objective, GRB.MINIMIZE)
    
    # Constraints
    for gen in generators:
        cluster_id = gen["cluster_id"]
        model.addConstr(y_P_ret[cluster_id] <= existing_capacity[str(cluster_id)])
        model.addConstr(y_P_new[cluster_id] >= 0)
        if gen["is_storage"]:
            model.addConstr(y_E_new[cluster_id] >= 0)
    
    for line in transmission_lines:
        line_id = line["line_id"]
        model.addConstr(y_F_new[line_id] >= 0)
    
    for w in range(num_weeks):
        for t in range(hours_per_subperiod):
            hour = w * hours_per_subperiod + t
            for z in range(num_zones):
                # Demand balance
                model.addConstr(
                    quicksum(x_inj[gen["cluster_id"], hour] for gen in generators if gen["zone"] == z)
                    - quicksum(x_wdw[gen["cluster_id"], hour] for gen in generators if gen["is_storage"] and gen["zone"] == z)
                    + quicksum(x_flow[line["line_id"], hour] for line in transmission_lines if line["to_zone"] == z)
                    - quicksum(x_flow[line["line_id"], hour] for line in transmission_lines if line["from_zone"] == z)
                    + quicksum(x_nse[s, z, hour] for s, segment in enumerate(consumer_segments) if segment["zone"] == z)
                    == demand[str(z)][hour]
                )
                for s, segment in enumerate(consumer_segments):
                    if segment["zone"] == z:
                        model.addConstr(x_nse[s, z, hour] <= segment["max_nse_fraction"] * demand[str(z)][hour])
    
    # Storage and hydro constraints
    for gen in generators:
        if gen["is_storage"]:
            for w in range(num_weeks):
                for t in range(hours_per_subperiod):
                    hour = w * hours_per_subperiod + t
                    prev_hour = (hour - 1) % total_hours
                    model.addConstr(x_soc[gen["cluster_id"], hour] == x_soc[gen["cluster_id"], prev_hour] + generator_parameters["battery"]["charging_efficiency"] * x_wdw[gen["cluster_id"], hour] - x_inj[gen["cluster_id"], hour] / generator_parameters["battery"]["discharging_efficiency"] - generator_parameters["battery"]["self_discharge_rate_pct"] * x_soc[gen["cluster_id"], prev_hour])
                    model.addConstr(x_soc[gen["cluster_id"], hour] <= gen["energy_capacity_size_MWh"] * (existing_capacity[str(gen["cluster_id"])] + y_P_new[gen["cluster_id"]] - y_P_ret[gen["cluster_id"]]) + y_E_new[gen["cluster_id"]])
    
    # Transmission constraints
    for line in transmission_lines:
        line_id = line["line_id"]
        for w in range(num_weeks):
            for t in range(hours_per_subperiod):
                hour = w * hours_per_subperiod + t
                model.addConstr(x_flow[line_id, hour] <= line["existing_capacity_MW"] + y_F_new[line_id])
                model.addConstr(x_flow[line_id, hour] >= -(line["existing_capacity_MW"] + y_F_new[line_id]))
    
    # Policy constraints
    if policy["type"] == "RPS":
        for w in range(num_weeks):
            weight = subperiod_weights[w]
            rps_generation = quicksum(
                weight * x_inj[gen["cluster_id"], hour]
                for t in range(hours_per_subperiod)
                for gen in generators
                if gen["is_RPS_qualifying"]
                for hour in [w * hours_per_subperiod + t]
            )
            total_demand = weight * sum(demand[str(z)][w * hours_per_subperiod + t] for z in range(num_zones) for t in range(hours_per_subperiod))
            model.addConstr(rps_generation + x_RPS[w] >= policy["RPS_share"] * total_demand)
    
    if policy["type"] == "CO2":
        for w in range(num_weeks):
            weight = subperiod_weights[w]
            co2_emissions = quicksum(
                weight * generator_parameters[gen["resource_type"]]["CO2_emission_tons_per_MWh"] * x_inj[gen["gen["cluster_id"], hour] + generator_parameters["battery"]["CO2_emission_tons_per_MWh"] * x_wdw[gen["cluster_id"], hour]
                for t in range(hours_per_subperiod)
                for gen in generators
                if gen["resource_type"] in generator_parameters or gen["is_storage"]
                for hour in [w * hours_per_subperiod + t]
            )
            total_demand = weight * sum(demand[str(z)][w * hours_per_subperiod + t] for z in range(num_zones) for t in range(hours_per_subperiod))
            model.addConstr(co2_emissions - x_CO2[w] <= policy["CO2_cap_tons_per_MWh"] * total_demand)
    
    model.update()
    return model, y_P_new, y_P_ret, y_E_new, y_F_new, x_inj, x_wdw, x_soc, x_nse, x_lvl, x_spill, x_flow, x_commit, x_start, x_shut, x_RPS, x_CO2

def extract_solution(model, y_P_new, y_P_ret, y_E_new, y_F_new, x_inj, x_wdw, x_soc, x_nse, x_lvl, x_spill, x_flow, x_commit, x_start, x_shut, x_RPS, x_CO2, generators, transmission_lines, consumer_segments):
    solution = {
        "objective_value": model.ObjVal,
        "investments": {},
        "storage_energy_investments": {},
        "transmission_investments": {},
    }
    for gen in generators:
        cluster_id = gen["cluster_id"]
        solution["investments"][cluster_id] = {
            "resource_type": gen["resource_type"],
            "zone": gen["zone"],
            "new_units": y_P_new[cluster_id].X,
            "new_MW": y_P_new[cluster_id].X,
            "retired_units": y_P_ret[cluster_id].X,
            "retired_MW": y_P_ret[cluster_id].X,
            "total_capacity_MW": existing_capacity[str(cluster_id)] + y_P_new[cluster_id].X - y_P_ret[cluster_id].X,
        }
        if gen["is_storage"]:
            solution["storage_energy_investments"][cluster_id] = {
                "new_units": y_E_new[cluster_id].X,
                "new_MWh": y_E_new[cluster_id].X,
                "total_energy_MWh": gen["energy_capacity_size_MWh"] * (existing_capacity[str(cluster_id)] + y_P_new[cluster_id].X - y_P_ret[cluster_id].X) + y_E_new[cluster_id].X,
            }
    for line in transmission_lines:
        line_id = line["line_id"]
        solution["transmission_investments"][line_id] = {
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
        solution = extract_solution(model, y_P_new, y_P_ret, y_E_new, y_F_new, x_inj, x_wdw, x_soc, x_nse, x_lvl, x_spill, x_flow, x_commit, x_start, x_shut, x_RPS, x_CO2, instance_data["generators"], instance_data["transmission_lines"], instance_data["consumer_segments"])
        with open(args.solution_path, "w") as f:
            json.dump(solution, f, indent=2)
        if logger:
            logger.log_solution(model.ObjVal, solution)
    else:
        print("No solution found")

if __name__ == "__main__":
    main()