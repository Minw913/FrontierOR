import argparse
import json
import math
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from solution_logger import SolutionLogger

def solve_instance(instance_path, solution_path, time_limit, log_path):
    # Load instance data
    with open(instance_path, 'r') as f:
        data = json.load(f)

    # Extract dimensions
    num_zones = data['problem_dimensions']['num_zones']
    num_weeks = data['problem_dimensions']['num_weeks']
    hours_per_week = data['problem_dimensions']['hours_per_week']
    total_timesteps = data['problem_dimensions']['total_timesteps']
    num_generators = data['problem_dimensions']['num_generators']
    num_UC_generators = data['problem_dimensions']['num_UC_generators']
    num_storage_resources = data['problem_dimensions']['num_storage_resources']
    num_hydro_resources = data['problem_dimensions']['num_hydro_resources']
    num_transmission_lines = data['problem_dimensions']['num_transmission_lines']
    num_demand_segments = data['problem_dimensions']['num_demand_segments']

    # Policy settings
    RPS_enabled = data['policy']['RPS_enabled']
    RPS_share = data['policy']['RPS_share']
    CO2_cap_enabled = data['policy']['CO2_cap_enabled']
    CO2_cap_tons_per_MWh = data['policy']['CO2_cap_tons_per_MWh']
    RPS_noncompliance_cost_per_MWh = data['policy']['RPS_noncompliance_cost_per_MWh']
    CO2_noncompliance_cost_per_ton = data['policy']['CO2_noncompliance_cost_per_ton']

    # Subperiods
    num_subperiods = data['subperiods']['num_subperiods']
    hours_per_subperiod = data['subperiods']['hours_per_subperiod']
    subperiod_weights = data['subperiods']['subperiod_weights']
    circular_indexing = data['subperiods']['circular_indexing']

    # Extract generator data
    generators = data['generators']
    
    # Create mappings for indices
    gen_to_idx = {gen['id']: i for i, gen in enumerate(generators)}
    zone_to_gens = [[] for _ in range(num_zones)]
    for i, gen in enumerate(generators):
        zone_to_gens[gen['zone']].append(i)

    # Extract transmission data
    transmission_lines = data['transmission_lines']
    line_to_idx = {line['id']: i for i, line in enumerate(transmission_lines)}

    # Extract demand data
    demand = {}
    for z in range(num_zones):
        demand[z] = data['demand'][str(z)]

    # Calculate total weighted demand
    total_weighted_demand = 0.0
    for w in range(num_weeks):
        for t in range(hours_per_week):
            timestep_idx = w * hours_per_week + t
            weight = subperiod_weights[w] / hours_per_week
            for z in range(num_zones):
                total_weighted_demand += weight * demand[z][timestep_idx]

    # Extract NSE segments
    nse_segments = data['nse_segments']

    # Extract availability profiles
    availability = {}
    for gen_id_str, profile in data['availability_profiles'].items():
        gen_id = int(gen_id_str)
        gen_idx = gen_to_idx[gen_id]
        if profile['type'] == 'constant':
            availability[gen_idx] = [profile['value']] * total_timesteps
        else:  # timeseries
            availability[gen_idx] = profile['values']

    # Initialize Gurobi model
    model = gp.Model("power_system_planning")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)

    # Create logger if path provided
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    # Define variables
    # Investment variables
    yPnew = {}  # New power capacity units
    yPret = {}  # Retired power capacity units
    yP = {}     # Total power capacity
    yEnew = {}  # New energy capacity units (for storage)
    yEret = {}  # Retired energy capacity units (for storage)
    yE = {}     # Total energy capacity (for storage)

    for i, gen in enumerate(generators):
        # Power capacity variables
        max_new_units = math.floor(gen['max_capacity_MW'] / gen['capacity_size_MW']) if gen['max_capacity_MW'] < float('inf') else 999999
        existing_units = gen['existing_capacity_MW'] / gen['capacity_size_MW']
        
        yPnew[i] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_new_units, name=f"yPnew_{i}")
        yPret[i] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=existing_units if gen['can_retire'] else 0, name=f"yPret_{i}")
        yP[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"yP_{i}")

        # Link total capacity
        model.addConstr(yP[i] == gen['existing_capacity_MW'] + 
                       yPnew[i] * gen['capacity_size_MW'] - 
                       yPret[i] * gen['capacity_size_MW'], name=f"link_capacity_{i}")

        # Energy capacity variables (for storage resources)
        if gen['is_storage']:
            max_new_energy_units = math.floor(gen['max_storage_capacity_MWh'] / gen['storage_capacity_size_MWh'])
            existing_energy_units = gen['existing_storage_capacity_MWh'] / gen['storage_capacity_size_MWh']
            
            yEnew[i] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_new_energy_units, name=f"yEnew_{i}")
            yEret[i] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=existing_energy_units, name=f"yEret_{i}")
            yE[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"yE_{i}")

            # Link total energy capacity
            model.addConstr(yE[i] == gen['existing_storage_capacity_MWh'] + 
                           yEnew[i] * gen['storage_capacity_size_MWh'] - 
                           yEret[i] * gen['storage_capacity_size_MWh'], name=f"link_energy_capacity_{i}")

            # Duration constraints for storage
            model.addConstr(yE[i] >= yP[i] * gen['min_duration_MWh_per_MW'], name=f"min_duration_{i}")
            model.addConstr(yE[i] <= yP[i] * gen['max_duration_MWh_per_MW'], name=f"max_duration_{i}")
        else:
            yEnew[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=0, name=f"yEnew_dummy_{i}")  # Dummy variable
            yEret[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=0, name=f"yEret_dummy_{i}")  # Dummy variable
            yE[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"yE_dummy_{i}")  # Dummy variable

    # Transmission investment variables
    yFnew = {}  # New transmission capacity
    yF = {}     # Total transmission capacity

    for l, line in enumerate(transmission_lines):
        yFnew[l] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=line['max_new_capacity_MW'], name=f"yFnew_{l}")
        yF[l] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"yF_{l}")

        # Link total capacity
        model.addConstr(yF[l] == line['existing_capacity_MW'] + yFnew[l], name=f"link_line_capacity_{l}")

    # Operational variables
    xinj = {}  # Generation/injection
    xwdw = {}  # Withdrawal (for storage)
    xsoc = {}  # State of charge (for storage)
    xhydro = {}  # Hydro generation (for hydro)
    
    xflow = {}  # Transmission flows
    xnse = {}   # Non-served energy
    
    # Unit commitment variables (for UC resources)
    xcommit = {}
    xstart = {}
    xshut = {}

    # Policy slack variables
    xRPS = {}  # RPS noncompliance (if enabled)
    xCO2 = {}  # CO2 noncompliance (if enabled)

    # Initialize operational variables
    for i in range(num_generators):
        gen = generators[i]
        if gen['is_storage']:
            xinj[i] = {}
            xwdw[i] = {}
            xsoc[i] = {}
            for t in range(total_timesteps):
                xinj[i][t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"xinj_{i}_{t}")
                xwdw[i][t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"xwdw_{i}_{t}")
                xsoc[i][t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"xsoc_{i}_{t}")
        elif gen['is_hydro']:
            xinj[i] = {}
            xhydro[i] = {}
            for t in range(total_timesteps):
                xinj[i][t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"xinj_{i}_{t}")
                xhydro[i][t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"xhydro_{i}_{t}")
        else:
            xinj[i] = {}
            for t in range(total_timesteps):
                xinj[i][t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"xinj_{i}_{t}")

    # Flow variables
    for l in range(num_transmission_lines):
        xflow[l] = {}
        for t in range(total_timesteps):
            xflow[l][t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"xflow_{l}_{t}")

    # NSE variables
    for s in range(num_demand_segments):
        xnse[s] = {}
        for z in range(num_zones):
            xnse[s][z] = {}
            for t in range(total_timesteps):
                max_nse = nse_segments[s]['max_frac'] * demand[z][t]
                xnse[s][z][t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max_nse, name=f"xnse_{s}_{z}_{t}")

    # Unit commitment variables for UC resources
    uc_gen_indices = []
    for i, gen in enumerate(generators):
        if gen['is_UC']:
            uc_gen_indices.append(i)
            xcommit[i] = {}
            xstart[i] = {}
            xshut[i] = {}
            for t in range(total_timesteps):
                # Upper bound for committed units is based on total capacity
                max_units = math.ceil((gen['existing_capacity_MW'] + gen['max_capacity_MW']) / gen['capacity_size_MW'])
                xcommit[i][t] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_units, name=f"xcommit_{i}_{t}")
                xstart[i][t] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_units, name=f"xstart_{i}_{t}")
                xshut[i][t] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_units, name=f"xshut_{i}_{t}")

    # Policy slack variables
    if RPS_enabled:
        for w in range(num_weeks):
            xRPS[w] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"xRPS_{w}")
    
    if CO2_cap_enabled:
        for w in range(num_weeks):
            xCO2[w] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"xCO2_{w}")

    # Constraints
    # Power balance constraints
    for w in range(num_weeks):
        for h in range(hours_per_week):
            t = w * hours_per_week + h
            for z in range(num_zones):
                lhs = gp.LinExpr()
                
                # Add generation from generators in this zone
                for i in zone_to_gens[z]:
                    lhs.add(xinj[i][t])
                    
                # Subtract storage withdrawals in this zone
                for i in zone_to_gens[z]:
                    if generators[i]['is_storage']:
                        lhs.add(-xwdw[i][t])
                
                # Subtract flows leaving this zone
                for l, line in enumerate(transmission_lines):
                    if line['from_zone'] == z:
                        lhs.add(-xflow[l][t])
                    elif line['to_zone'] == z:
                        lhs.add(xflow[l][t])
                        
                # Add NSE
                for s in range(num_demand_segments):
                    lhs.add(xnse[s][z][t])
                
                # RHS is demand
                rhs = demand[z][t]
                
                model.addConstr(lhs == rhs, name=f"power_balance_{z}_{t}")

    # Availability constraints for non-UC, non-storage, non-hydro
    for i, gen in enumerate(generators):
        if not gen['is_UC'] and not gen['is_storage'] and not gen['is_hydro']:
            for t in range(total_timesteps):
                model.addConstr(xinj[i][t] <= availability[i][t] * yP[i], name=f"availability_{i}_{t}")

    # Storage constraints
    for i, gen in enumerate(generators):
        if gen['is_storage']:
            for t in range(total_timesteps):
                # Availability constraint for withdrawals
                model.addConstr(xwdw[i][t] <= availability[i][t] * yP[i], name=f"storage_availability_wdw_{i}_{t}")
                
                # Total power constraint (generation + withdrawal <= capacity)
                model.addConstr(xinj[i][t] + xwdw[i][t] <= yP[i], name=f"storage_power_limit_{i}_{t}")
                
                # Charging constraint
                model.addConstr(gen['charge_efficiency'] * xwdw[i][t] <= yE[i] - xsoc[i][t], name=f"storage_charging_{i}_{t}")
                
                # Discharging constraint
                model.addConstr(xinj[i][t] / gen['discharge_efficiency'] <= xsoc[i][t], name=f"storage_discharging_{i}_{t}")
                
                # SOC upper bound
                model.addConstr(xsoc[i][t] <= yE[i], name=f"storage_soc_bound_{i}_{t}")

    # Hydro constraints
    for i, gen in enumerate(generators):
        if gen['is_hydro']:
            for t in range(total_timesteps):
                # Reservoir level constraint (hydro generation + spillage)
                # Assuming the reservoir level is related to the energy capacity
                # The hydro energy capacity is determined by duration ratio * power capacity
                model.addConstr(xinj[i][t] <= gen['duration_MWh_per_MW'] * yP[i], name=f"hydro_generation_limit_{i}_{t}")

    # Minimum output constraints
    for i, gen in enumerate(generators):
        if not gen['is_UC'] and not gen['is_storage'] and not gen['is_hydro']:
            for t in range(total_timesteps):
                model.addConstr(xinj[i][t] >= gen['min_output_frac'] * yP[i], name=f"min_output_{i}_{t}")

    # Ramp constraints for non-UC resources
    for i, gen in enumerate(generators):
        if not gen['is_UC'] and not gen['is_storage'] and not gen['is_hydro']:
            for w in range(num_weeks):
                for h in range(hours_per_week):
                    t = w * hours_per_week + h
                    # Next hour with circular indexing
                    next_h = (h + 1) % hours_per_week
                    next_t = w * hours_per_week + next_h
                    
                    # Ramp up constraint
                    model.addConstr(xinj[i][next_t] - xinj[i][t] <= gen['ramp_up_frac_per_hr'] * yP[i], 
                                   name=f"ramp_up_{i}_{t}")
                    
                    # Ramp down constraint
                    model.addConstr(xinj[i][t] - xinj[i][next_t] <= gen['ramp_dn_frac_per_hr'] * yP[i], 
                                   name=f"ramp_down_{i}_{t}")

    # Unit commitment constraints
    for i in uc_gen_indices:
        gen = generators[i]
        for t in range(total_timesteps):
            # Committed units cannot exceed total capacity
            model.addConstr(xcommit[i][t] * gen['capacity_size_MW'] <= yP[i], name=f"uc_commit_limit_{i}_{t}")
            
            # Startup/shutdown bounds
            max_units = math.ceil((gen['existing_capacity_MW'] + gen['max_capacity_MW']) / gen['capacity_size_MW'])
            model.addConstr(xstart[i][t] <= max_units, name=f"uc_start_limit_{i}_{t}")
            model.addConstr(xshut[i][t] <= max_units, name=f"uc_shut_limit_{i}_{t}")
            
            # Commitment transition with circular indexing
            if t == 0:
                # Connect to the last hour of the simulation
                prev_t = total_timesteps - 1
            else:
                prev_t = t - 1
            
            model.addConstr(xcommit[i][t] == xcommit[i][prev_t] + xstart[i][t] - xshut[i][t],
                           name=f"uc_transition_{i}_{t}")
            
            # Min output constraint for UC
            model.addConstr(xinj[i][t] >= xcommit[i][t] * gen['min_output_frac'] * gen['capacity_size_MW'],
                           name=f"uc_min_output_{i}_{t}")
            
            # Max output constraint for UC
            model.addConstr(xinj[i][t] <= xcommit[i][t] * availability[i][t] * gen['capacity_size_MW'],
                           name=f"uc_max_output_{i}_{t}")

    # Storage state of charge dynamics
    for i, gen in enumerate(generators):
        if gen['is_storage']:
            for w in range(num_weeks):
                for h in range(hours_per_week):
                    t = w * hours_per_week + h
                    if t == 0:
                        # For first hour, assume initial SOC is half full
                        model.addConstr(xsoc[i][t] == yE[i] * 0.5 + 
                                       gen['charge_efficiency'] * xwdw[i][t] - 
                                       xinj[i][t] / gen['discharge_efficiency'] - 
                                       gen['self_discharge_rate'] * yE[i] * 0.5,
                                       name=f"soc_initial_{i}_{t}")
                    else:
                        prev_t = t - 1
                        model.addConstr(xsoc[i][t] == xsoc[i][prev_t] + 
                                       gen['charge_efficiency'] * xwdw[i][prev_t] - 
                                       xinj[i][prev_t] / gen['discharge_efficiency'] - 
                                       gen['self_discharge_rate'] * xsoc[i][prev_t],
                                       name=f"soc_dynamics_{i}_{t}")

    # RPS constraint
    if RPS_enabled:
        total_rps_generation = gp.LinExpr()
        for w in range(num_weeks):
            weight = subperiod_weights[w] / hours_per_week
            rps_gen_in_week = gp.LinExpr()
            for h in range(hours_per_week):
                t = w * hours_per_week + h
                for i, gen in enumerate(generators):
                    if gen['is_RPS_qualifying']:
                        rps_gen_in_week.add(weight * xinj[i][t])
            
            total_rps_generation.add(rps_gen_in_week)
        
        model.addConstr(total_rps_generation + gp.quicksum([xRPS[w] for w in range(num_weeks)]) >= 
                       RPS_share * total_weighted_demand,
                       name="rps_requirement")

    # CO2 constraint
    if CO2_cap_enabled:
        co2_cap_limit = CO2_cap_tons_per_MWh * total_weighted_demand if CO2_cap_tons_per_MWh else 0.05 * total_weighted_demand
        
        total_co2_emissions = gp.LinExpr()
        for w in range(num_weeks):
            weight = subperiod_weights[w] / hours_per_week
            co2_emissions_in_week = gp.LinExpr()
            for h in range(hours_per_week):
                t = w * hours_per_week + h
                for i, gen in enumerate(generators):
                    if gen['co2_tons_per_MWh'] > 0:
                        co2_emissions_in_week.add(weight * gen['co2_tons_per_MWh'] * xinj[i][t])
            
            total_co2_emissions.add(co2_emissions_in_week)
        
        model.addConstr(total_co2_emissions - gp.quicksum([xCO2[w] for w in range(num_weeks)]) <= co2_cap_limit,
                       name="co2_cap")

    # Objective function
    # Fixed costs
    fixed_costs = gp.LinExpr()
    
    # Investment costs
    for i, gen in enumerate(generators):
        fixed_costs.add(gen['inv_cost_per_MW_yr'] * yPnew[i] * gen['capacity_size_MW'])
        if gen['is_hydro']:
            # For hydro, we might need to account for energy capacity investments differently
            # Based on the problem description, hydro has a fixed duration ratio
            fixed_costs.add(gen['inv_cost_per_MW_yr'] * gen['duration_MWh_per_MW'] * yPnew[i] * gen['capacity_size_MW'])
    
    for i in [idx for idx, gen in enumerate(generators) if gen['is_storage']]:
        gen = generators[i]
        fixed_costs.add(gen['inv_cost_energy_per_MWh_yr'] * yEnew[i] * gen['storage_capacity_size_MWh'])
    
    for l in range(num_transmission_lines):
        fixed_costs.add(transmission_lines[l]['inv_cost_per_MW_yr'] * yFnew[l])
    
    # Fixed O&M costs
    for i, gen in enumerate(generators):
        fixed_costs.add(gen['fom_cost_per_MW_yr'] * yP[i])
        if gen['is_hydro']:
            # For hydro, include energy capacity FOM
            fixed_costs.add(gen['fom_cost_per_MW_yr'] * gen['duration_MWh_per_MW'] * yP[i])
    
    for i in [idx for idx, gen in enumerate(generators) if gen['is_storage']]:
        gen = generators[i]
        fixed_costs.add(gen['fom_cost_energy_per_MWh_yr'] * yE[i])

    # Variable costs
    var_costs = gp.LinExpr()
    for i, gen in enumerate(generators):
        for t in range(total_timesteps):
            weight = subperiod_weights[t // hours_per_week] / hours_per_week
            var_costs.add(weight * gen['var_cost_per_MWh'] * xinj[i][t])
            if gen['is_storage']:
                var_costs.add(weight * gen['var_cost_per_MWh'] * xwdw[i][t])

    # NSE costs
    nse_costs = gp.LinExpr()
    for s in range(num_demand_segments):
        for z in range(num_zones):
            for t in range(total_timesteps):
                weight = subperiod_weights[t // hours_per_week] / hours_per_week
                nse_costs.add(weight * nse_segments[s]['cost_per_MWh'] * xnse[s][z][t])

    # Startup costs
    startup_costs = gp.LinExpr()
    for i in uc_gen_indices:
        gen = generators[i]
        for t in range(total_timesteps):
            weight = subperiod_weights[t // hours_per_week] / hours_per_week
            startup_costs.add(weight * gen['start_cost_per_unit'] * xstart[i][t])

    # Policy violation costs
    policy_costs = gp.LinExpr()
    if RPS_enabled:
        for w in range(num_weeks):
            policy_costs.add(RPS_noncompliance_cost_per_MWh * xRPS[w])
    
    if CO2_cap_enabled:
        for w in range(num_weeks):
            policy_costs.add(CO2_noncompliance_cost_per_ton * xCO2[w])

    total_cost = fixed_costs + var_costs + nse_costs + startup_costs + policy_costs
    model.setObjective(total_cost, GRB.MINIMIZE)

    # Optimize
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    model.optimize(callback)

    # Prepare solution
    solution = {
        "objective_value": model.objVal if model.solCount > 0 else float('inf'),
        "model_variables": {},
        "investment_decisions": [],
        "transmission_decisions": [],
        "model_variable_name_patterns": {
            "yPnew_{i}, yPret_{i}, yP_{i}": "generator investment/retirement/total capacity",
            "yEnew_{i}, yEret_{i}, yE_{i}": "storage-energy investment variables",
            "yFnew_{l}": "transmission investment",
            "xinj_{i}_{t}, xwdw_{i}_{t}, xsoc_{i}_{t}, xhydro_{i}_{t}": "operations",
            "xflow_{l}_{t}": "transmission flow",
            "xnse_{s}_{z}_{t}": "non-served energy",
            "xcommit_{i}_{t}, xstart_{i}_{t}, xshut_{i}_{t}": "unit commitment",
            "xRPS_{w}, xCO2_{w}": "policy slack variables"
        }
    }

    if model.solCount > 0:
        # Populate investment decisions
        for i, gen in enumerate(generators):
            inv_decision = {
                "generator_id": gen['id'],
                "resource_type": gen['resource_type'],
                "zone": gen['zone'],
                "y_P_new": round(yPnew[i].X),
                "y_P_ret": round(yPret[i].X),
                "y_P_total_MW": yP[i].X
            }
            if gen['is_storage']:
                inv_decision["y_E_new"] = round(yEnew[i].X)
                inv_decision["y_E_ret"] = round(yEret[i].X)
                inv_decision["y_E_total_MWh"] = yE[i].X
            else:
                inv_decision["y_E_new"] = 0
                inv_decision["y_E_ret"] = 0
                inv_decision["y_E_total_MWh"] = 0
            
            solution["investment_decisions"].append(inv_decision)

        # Populate transmission decisions
        for l, line in enumerate(transmission_lines):
            trans_decision = {
                "line_id": line['id'],
                "y_F_new_MW": round(yFnew[l].X),
                "total_capacity_MW": yF[l].X
            }
            solution["transmission_decisions"].append(trans_decision)

        # Add model variables that are non-zero to the sparse solution
        for var in model.getVars():
            if abs(var.X) > 1e-6:  # Only include non-zero variables
                solution["model_variables"][var.varName] = var.X

        # Log solution if available
        if logger:
            logger.log_solution(model.objVal, solution)

    # Write solution
    with open(solution_path, 'w') as f:
        json.dump(solution, f)

def main():
    parser = argparse.ArgumentParser(description='Solve power system planning problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to save the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log intermediate solutions')
    
    args = parser.parse_args()
    
    solve_instance(args.instance_path, args.solution_path, args.time_limit, args.log_path)

if __name__ == '__main__':
    main()