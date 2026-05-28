import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def setup_model(instance):
    model = gp.Model()
    model.Params.LogToConsole = 0
    
    num_generators = instance['problem_dimensions']['num_generators']
    num_storage_resources = instance['problem_dimensions']['num_storage_resources']
    num_hydro_resources = instance['problem_dimensions']['num_hydro_resources']
    num_transmission_lines = instance['problem_dimensions']['num_transmission_lines']
    num_zones = instance['problem_dimensions']['num_zones']
    num_weeks = instance['problem_dimensions']['num_weeks']
    hours_per_week = instance['problem_dimensions']['hours_per_week']
    total_timesteps = instance['problem_dimensions']['total_timesteps']
    subperiod_weights = instance['subperiods']['subperiod_weights']
    generators = instance['generators']
    transmission_lines = instance['transmission_lines']
    demand = instance['demand']
    nse_segments = instance['nse_segments']
    availability_profiles = instance['availability_profiles']
    policy = instance['policy']
    
    y_P_new = model.addVars(num_generators, vtype=GRB.INTEGER, name='y_P_new')
    y_P_ret = model.addVars(num_generators, vtype=GRB.INTEGER, name='y_P_ret')
    y_E_new = model.addVars(num_storage_resources, vtype=GRB.INTEGER, name='y_E_new')
    y_E_ret = model.addVars(num_storage_resources, vtype=GRB.INTEGER, name='y_E_ret')
    y_F_new = model.addVars(num_transmission_lines, vtype=GRB.CONTINUOUS, name='y_F_new')
    
    generation = model.addVars(num_generators, total_timesteps, vtype=GRB.CONTINUOUS, name='generation')
    storage_withdrawal = model.addVars(num_storage_resources, total_timesteps, vtype=GRB.CONTINUOUS, name='storage_withdrawal')
    transmission_flow = model.addVars(num_transmission_lines, total_timesteps, vtype=GRB.CONTINUOUS, name='transmission_flow')
    nse = model.addVars(num_zones, len(nse_segments), total_timesteps, vtype=GRB.CONTINUOUS, name='nse')
    state_of_charge = model.addVars(num_storage_resources, total_timesteps, vtype=GRB.CONTINUOUS, name='state_of_charge')
    reservoir_level = model.addVars(num_hydro_resources, total_timesteps, vtype=GRB.CONTINUOUS, name='reservoir_level')
    
    committed_units = model.addVars([g for g in range(num_generators) if generators[g]['is_UC']], total_timesteps, vtype=GRB.INTEGER, name='committed_units')
    startups = model.addVars([g for g in range(num_generators) if generators[g]['is_UC']], total_timesteps, vtype=GRB.INTEGER, name='startups')
    shutdowns = model.addVars([g for g in range(num_generators) if generators[g]['is_UC']], total_timesteps, vtype=GRB.INTEGER, name='shutdowns')
    
    RPS_slack = model.addVar(vtype=GRB.CONTINUOUS, name='RPS_slack') if policy.get('RPS_enabled', False) else None
    CO2_slack = model.addVar(vtype=GRB.CONTINUOUS, name='CO2_slack') if policy.get('CO2_cap_enabled', False) else None
    
    for g in range(num_generators):
        model.addConstr(y_P_new[g] * generators[g]['capacity_size_MW'] <= generators[g].get('max_capacity_MW', float('inf')) - generators[g]['existing_capacity_MW'])
        model.addConstr(y_P_ret[g] * generators[g]['capacity_size_MW'] <= generators[g]['existing_capacity_MW'])
        total_capacity = generators[g]['existing_capacity_MW'] + (y_P_new[g] - y_P_ret[g]) * generators[g]['capacity_size_MW']
        
        for t in range(total_timesteps):
            avail = availability_profiles.get(str(g), {}).get('value', 1.0) if availability_profiles.get(str(g), {}).get('type') == 'constant' else availability_profiles.get(str(g), {}).get('values', [1.0]*total_timesteps)[t]
            
            if generators[g]['is_UC']:
                model.addConstr(generation[g, t] >= committed_units[g, t] * generators[g]['capacity_size_MW'] * generators[g]['min_output_frac'])
                model.addConstr(generation[g, t] <= committed_units[g, t] * generators[g]['capacity_size_MW'] * avail)
            else:
                model.addConstr(generation[g, t] >= total_capacity * generators[g]['min_output_frac'])
                model.addConstr(generation[g, t] <= total_capacity * avail)
    
    for s, g in enumerate(range(num_generators - num_storage_resources, num_generators)):
        model.addConstr(y_E_new[s] * generators[g]['storage_capacity_size_MWh'] <= generators[g].get('max_storage_capacity_MWh', float('inf')) - generators[g]['existing_storage_capacity_MWh'])
        model.addConstr(y_E_ret[s] * generators[g]['storage_capacity_size_MWh'] <= generators[g]['existing_storage_capacity_MWh'])
        total_energy_capacity = generators[g]['existing_storage_capacity_MWh'] + (y_E_new[s] - y_E_ret[s]) * generators[g]['storage_capacity_size_MWh']
        
        for t in range(total_timesteps):
            model.addConstr(state_of_charge[s, t] <= total_energy_capacity)
            if t > 0:
                model.addConstr(state_of_charge[s, t] == state_of_charge[s, t-1] + generators[g]['charge_efficiency'] * storage_withdrawal[s, t-1] - generation[g, t-1] / generators[g]['discharge_efficiency'] - state_of_charge[s, t-1] * generators[g]['self_discharge_rate'])
            else:
                model.addConstr(state_of_charge[s, t] == state_of_charge[s, total_timesteps-1] + generators[g]['charge_efficiency'] * storage_withdrawal[s, total_timesteps-1] - generation[g, total_timesteps-1] / generators[g]['discharge_efficiency'] - state_of_charge[s, total_timesteps-1] * generators[g]['self_discharge_rate'])
    
    for l in range(num_transmission_lines):
        model.addConstr(y_F_new[l] <= transmission_lines[l]['max_new_capacity_MW'])
        total_line_capacity = transmission_lines[l]['existing_capacity_MW'] + y_F_new[l]
        for t in range(total_timesteps):
            model.addConstr(transmission_flow[l, t] <= total_line_capacity)
            model.addConstr(transmission_flow[l, t] >= -total_line_capacity)
    
    for z in range(num_zones):
        for t in range(total_timesteps):
            model.addConstr(sum(generation[g, t] for g in range(num_generators) if generators[g]['zone'] == z) - 
                            sum(storage_withdrawal[s, t] for s in range(num_storage_resources) if generators[s + num_generators - num_storage_resources]['zone'] == z) - 
                            sum(transmission_flow[l, t] for l in range(num_transmission_lines) if transmission_lines[l]['from_zone'] == z) + 
                            sum(transmission_flow[l, t] for l in range(num_transmission_lines) if transmission_lines[l]['to_zone'] == z) + 
                            sum(nse[z, seg, t] for seg in range(len(nse_segments))) == float(demand[str(z)][t]))
            
            for seg in range(len(nse_segments)):
                model.addConstr(nse[z, seg, t] <= float(nse_segments[seg]['max_frac']) * float(demand[str(z)][t]))
    
    if policy.get('RPS_enabled', False):
        RPS_qualifying_generation = sum(subperiod_weights[t // hours_per_week] * sum(generation[g, t] for g in range(num_generators) if generators[g].get('is_RPS_qualifying', False)) for t in range(total_timesteps))
        total_demand = sum(subperiod_weights[t // hours_per_week] * sum(float(demand[str(z)][t]) for z in range(num_zones)) for t in range(total_timesteps))
        model.addConstr(RPS_qualifying_generation + RPS_slack >= policy['RPS_share'] * total_demand)
    
    if policy.get('CO2_cap_enabled', False):
        CO2_emissions = sum(subperiod_weights[t // hours_per_week] * sum(generation[g, t] * generators[g]['co2_tons_per_MWh'] for g in range(num_generators)) for t in range(total_timesteps))
        total_demand = sum(subperiod_weights[t // hours_per_week] * sum(float(demand[str(z)][t]) for z in range(num_zones)) for t in range(total_timesteps))
        model.addConstr(CO2_emissions - CO2_slack <= policy.get('CO2_cap_tons_per_MWh', 0.05) * total_demand)
    
    fixed_costs = sum(generators[g]['inv_cost_per_MW_yr'] * generators[g]['capacity_size_MW'] * y_P_new[g] for g in range(num_generators)) + \
                  sum(generators[g].get('inv_cost_energy_per_MWh_yr', 0) * generators[g].get('storage_capacity_size_MWh', 0) * y_E_new[s] for s, g in enumerate(range(num_generators - num_storage_resources, num_generators))) + \
                  sum(transmission_lines[l]['inv_cost_per_MW_yr'] * y_F_new[l] for l in range(num_transmission_lines)) + \
                  sum(generators[g]['fom_cost_per_MW_yr'] * (generators[g]['existing_capacity_MW'] + (y_P_new[g] - y_P_ret[g]) * generators[g]['capacity_size_MW']) for g in range(num_generators)) + \
                  sum(generators[g].get('fom_cost_energy_per_MWh_yr', 0) * (generators[g].get('existing_storage_capacity_MWh', 0) + (y_E_new[s] - y_E_ret[s]) * generators[g].get('storage_capacity_size_MWh', 0)) for s, g in enumerate(range(num_generators - num_storage_resources, num_generators)))
    variable_costs = sum(subperiod_weights[t // hours_per_week] * sum(generation[g, t] * generators[g]['var_cost_per_MWh'] for g in range(num_generators)) for t in range(total_timesteps)) + \
                     sum(subperiod_weights[t // hours_per_week] * sum(storage_withdrawal[s, t] * generators[s + num_generators - num_storage_resources]['var_cost_per_MWh'] for s in range(num_storage_resources)) for t in range(total_timesteps))
    nse_costs = sum(subperiod_weights[t // hours_per_week] * sum(nse[z, seg, t] * nse_segments[seg]['cost_per_MWh'] for z in range(num_zones) for seg in range(len(nse_segments))) for t in range(total_timesteps))
    startup_costs = sum(subperiod_weights[t // hours_per_week] * sum(startups[g, t] * generators[g].get('start_cost_per_unit', 0) for g in range(num_generators) if generators[g]['is_UC']) for t in range(total_timesteps))
    policy_costs = (RPS_slack * policy['RPS_noncompliance_cost_per_MWh'] if policy.get('RPS_enabled', False) else 0) + (CO2_slack * policy['CO2_noncompliance_cost_per_ton'] if policy.get('CO2_cap_enabled', False) else 0)
    model.setObjective(fixed_costs + variable_costs + nse_costs + startup_costs + policy_costs, GRB.MINIMIZE)
    
    return model

def extract_solution(instance, model):
    generators = instance['generators']
    transmission_lines = instance['transmission_lines']
    num_storage_resources = instance['problem_dimensions']['num_storage_resources']
    num_generators = instance['problem_dimensions']['num_generators']
    
    solution = {
        'objective_value': model.ObjVal,
        'investment_decisions': [],
        'transmission_decisions': []
    }
    
    for g in range(num_generators):
        decision = {
            'generator_id': g,
            'resource_type': generators[g]['resource_type'],
            'zone': generators[g]['zone'],
            'y_P_new': model.getVarByName(f'y_P_new[{g}]').X,
            'y_P_ret': model.getVarByName(f'y_P_ret[{g}]').X,
            'y_P_total_MW': generators[g]['existing_capacity_MW'] + (model.getVarByName(f'y_P_new[{g}]').X - model.getVarByName(f'y_P_ret[{g}]').X) * generators[g]['capacity_size_MW'],
        }
        
        if g >= num_generators - num_storage_resources:
            s = g - num_generators + num_storage_resources
            decision['y_E_new'] = model.getVarByName(f'y_E_new[{s}]').X
            decision['y_E_ret'] = model.getVarByName(f'y_E_ret[{s}]').X
            decision['y_E_total_MWh'] = generators[g].get('existing_storage_capacity_MWh', 0) + (model.getVarByName(f'y_E_new[{s}]').X - model.getVarByName(f'y_E_ret[{s}]').X) * generators[g].get('storage_capacity_size_MWh', 0)
        
        solution['investment_decisions'].append(decision)
    
    for l in range(len(transmission_lines)):
        solution['transmission_decisions'].append({
            'line_id': l,
            'y_F_new_MW': model.getVarByName(f'y_F_new[{l}]').X,
            'total_capacity_MW': transmission_lines[l]['existing_capacity_MW'] + model.getVarByName(f'y_F_new[{l}]').X
        })
    
    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path')
    args = parser.parse_args()
    
    instance = read_instance(args.instance_path)
    model = setup_model(instance)
    model.Params.TimeLimit = args.time_limit
    
    logger = SolutionLogger(args.log_path, sense='minimize') if args.log_path else None
    
    model.optimize()
    
    if logger and model.SolCount > 0:
        logger.log(model.ObjVal)
    
    if model.SolCount > 0:
        solution = extract_solution(instance, model)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        print("No feasible solution found.")

if __name__ == "__main__":
    main()