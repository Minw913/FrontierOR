import argparse
import json
from gurobipy import Model, GRB, quicksum
import numpy as np
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def create_model(instance):
    model = Model('power_system_planning')
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    
    # Extract relevant data from instance
    system = instance['system']
    scenario_parameters = instance['scenario_parameters']
    scenario_tree = instance['scenario_tree']
    representative_days = instance['representative_days']
    lower_level_scenarios = instance['lower_level_scenarios']
    global_parameters = instance['global_parameters']
    
    num_buses = system['num_buses']
    buses = system['buses']
    generators = system['generators']
    transmission_lines = system['transmission_lines']
    storage = system['storage']
    loads = system['loads']
    
    num_stages = scenario_parameters['num_stages']
    num_upper_level_scenarios = scenario_parameters['num_upper_level_scenarios']
    num_lower_level_scenarios_per_day = scenario_parameters['num_lower_level_scenarios_per_day']
    num_representative_days = scenario_parameters['num_representative_days']
    hours_per_day = scenario_parameters['hours_per_day']
    
    existing_generators = [g for g in generators if g['existing']]
    candidate_generators = [g for g in generators if not g['existing']]
    existing_lines = [l for l in transmission_lines if l['existing']]
    candidate_lines = [l for l in transmission_lines if not l['existing']]
    existing_storage = [s for s in storage if s['existing']]
    candidate_storage = [s for s in storage if not s['existing']]
    
    # Create variables
    investments = {}
    alpha = {}
    gamma = {}
    p_gen = {}
    s_res = {}
    f_fwd = {}
    f_bwd = {}
    u_stor = {}
    v_stor = {}
    r_stor = {}
    
    for node in scenario_tree:
        node_id = node['id']
        investments[node_id] = {
            'generators': {g['id']: model.addVar(vtype=GRB.BINARY) for g in candidate_generators},
            'lines': {l['id']: model.addVar(vtype=GRB.BINARY) for l in candidate_lines},
            'storage': {s['id']: model.addVar(vtype=GRB.BINARY) for s in candidate_storage}
        }
        
        for day in representative_days:
            day_id = day['id']
            for lower_level_scenario in lower_level_scenarios[str(day_id)]:
                scenario_id = lower_level_scenario['scenario_id']
                for hour in range(hours_per_day):
                    alpha[(node_id, day_id, scenario_id, hour)] = {}
                    gamma[(node_id, day_id, scenario_id, hour)] = {}
                    p_gen[(node_id, day_id, scenario_id, hour)] = {}
                    s_res[(node_id, day_id, scenario_id, hour)] = {}
                    f_fwd[(node_id, day_id, scenario_id, hour)] = {}
                    f_bwd[(node_id, day_id, scenario_id, hour)] = {}
                    u_stor[(node_id, day_id, scenario_id, hour)] = {}
                    v_stor[(node_id, day_id, scenario_id, hour)] = {}
                    r_stor[(node_id, day_id, scenario_id, hour)] = {}
                    
                    for g in generators:
                        g_id = g['id']
                        alpha[(node_id, day_id, scenario_id, hour)][g_id] = model.addVar(vtype=GRB.BINARY)
                        gamma[(node_id, day_id, scenario_id, hour)][g_id] = model.addVar(vtype=GRB.BINARY)
                        p_gen[(node_id, day_id, scenario_id, hour)][g_id] = model.addVar(lb=0)
                        s_res[(node_id, day_id, scenario_id, hour)][g_id] = model.addVar(lb=0)
                    
                    for l in transmission_lines:
                        l_id = l['id']
                        f_fwd[(node_id, day_id, scenario_id, hour)][l_id] = model.addVar(lb=0)
                        f_bwd[(node_id, day_id, scenario_id, hour)][l_id] = model.addVar(lb=0)
                    
                    for s in storage:
                        s_id = s['id']
                        u_stor[(node_id, day_id, scenario_id, hour)][s_id] = model.addVar(lb=0)
                        v_stor[(node_id, day_id, scenario_id, hour)][s_id] = model.addVar(lb=0)
                        r_stor[(node_id, day_id, scenario_id, hour)][s_id] = model.addVar(lb=0)
    
    # Objective function
    objective = quicksum(node['probability'] * (
        quicksum(g['capital_cost_per_kW'] * 1000 * investments[node['id']]['generators'][g['id']] * g['Pmax_MW'] for g in candidate_generators) +
        quicksum(l['capital_cost_per_kW'] * 1000 * investments[node['id']]['lines'][l['id']] * l['flow_limit_MW'] for l in candidate_lines) +
        quicksum(s['capital_cost_per_kW'] * 1000 * investments[node['id']]['storage'][s['id']] * s['capacity_MW'] for s in candidate_storage) +
        quicksum(
            quicksum(
                quicksum(
                    quicksum(
                        g['startup_cost'] * gamma[(node['id'], day['id'], scenario_id, hour)][g['id']] +
                        (g['a_cost'] * alpha[(node['id'], day['id'], scenario_id, hour)][g['id']] +
                         (g['b_cost'] + g['c_cost'] * g['Pmax_MW']) * p_gen[(node['id'], day['id'], scenario_id, hour)][g['id']])
                        for hour in range(hours_per_day)
                    ) for lower_level_scenario in lower_level_scenarios[str(day['id'])] for scenario_id in [lower_level_scenario['scenario_id']]
                ) for day in representative_days
            ) / num_lower_level_scenarios_per_day
        ) / num_representative_days
    ) for node in scenario_tree)
    model.setObjective(objective, GRB.MINIMIZE)
    
    # Constraints
    for node in scenario_tree:
        node_id = node['id']
        for g in candidate_generators:
            g_id = g['id']
            if node['parent_id'] is not None:
                model.addConstr(investments[node_id]['generators'][g_id] >= investments[node['parent_id']]['generators'][g_id])
        
        for l in candidate_lines:
            l_id = l['id']
            if node['parent_id'] is not None:
                model.addConstr(investments[node_id]['lines'][l_id] >= investments[node['parent_id']]['lines'][l_id])
        
        for s in candidate_storage:
            s_id = s['id']
            if node['parent_id'] is not None:
                model.addConstr(investments[node_id]['storage'][s_id] >= investments[node['parent_id']]['storage'][s_id])
        
        for day in representative_days:
            day_id = day['id']
            for lower_level_scenario in lower_level_scenarios[str(day_id)]:
                scenario_id = lower_level_scenario['scenario_id']
                for hour in range(hours_per_day):
                    for g in generators:
                        g_id = g['id']
                        model.addConstr(p_gen[(node_id, day_id, scenario_id, hour)][g_id] + s_res[(node_id, day_id, scenario_id, hour)][g_id] <=
                                        g['Pmax_MW'] * alpha[(node_id, day_id, scenario_id, hour)][g_id] if g['existing'] else
                                        g['Pmax_MW'] * investments[node_id]['generators'][g_id])
                        model.addConstr(p_gen[(node_id, day_id, scenario_id, hour)][g_id] >= g['Pmin_MW'] * alpha[(node_id, day_id, scenario_id, hour)][g_id])
                        if hour > 0:
                            model.addConstr(gamma[(node_id, day_id, scenario_id, hour)][g_id] >= alpha[(node_id, day_id, scenario_id, hour)][g_id] - alpha[(node_id, day_id, scenario_id, hour-1)][g_id])
                            model.addConstr(p_gen[(node_id, day_id, scenario_id, hour)][g_id] - p_gen[(node_id, day_id, scenario_id, hour-1)][g_id] <= g['ramp_MW_per_h'])
                            model.addConstr(p_gen[(node_id, day_id, scenario_id, hour-1)][g_id] - p_gen[(node_id, day_id, scenario_id, hour)][g_id] <= g['ramp_MW_per_h'])
                    
                    for bus in buses:
                        bus_load = next((load['annual_peak_load_MW'] for load in loads if load['bus'] == bus), 0)
                        load_profile = day['load_profile'][hour]
                        demand_multiplier = lower_level_scenario['demand_multipliers'][hour]
                        demand = bus_load * load_profile * demand_multiplier
                        
                        total_gen = quicksum(p_gen[(node_id, day_id, scenario_id, hour)][g['id']] for g in generators if g['bus'] == bus)
                        total_storage_withdrawal = quicksum(v_stor[(node_id, day_id, scenario_id, hour)][s['id']] for s in storage if s['bus'] == bus)
                        total_storage_injection = quicksum(u_stor[(node_id, day_id, scenario_id, hour)][s['id']] for s in storage if s['bus'] == bus)
                        total_fwd_flow = quicksum(f_fwd[(node_id, day_id, scenario_id, hour)][l['id']] for l in transmission_lines if l['from_bus'] == bus)
                        total_bwd_flow = quicksum(f_bwd[(node_id, day_id, scenario_id, hour)][l['id']] for l in transmission_lines if l['to_bus'] == bus)
                        total_loss = quicksum(l['loss_factor'] * (f_fwd[(node_id, day_id, scenario_id, hour)][l['id']] + f_bwd[(node_id, day_id, scenario_id, hour)][l['id']]) for l in transmission_lines if l['from_bus'] == bus or l['to_bus'] == bus)
                        
                        model.addConstr(total_gen + total_storage_withdrawal - total_storage_injection + total_bwd_flow - total_fwd_flow >= demand)
                        model.addConstr(total_gen + total_storage_withdrawal - total_storage_injection + total_bwd_flow - total_fwd_flow - total_loss <= demand)
                    
                    for l in transmission_lines:
                        l_id = l['id']
                        flow_limit = l['flow_limit_MW'] if l['existing'] else investments[node_id]['lines'][l_id] * l['flow_limit_MW']
                        model.addConstr(f_fwd[(node_id, day_id, scenario_id, hour)][l_id] <= flow_limit)
                        model.addConstr(f_bwd[(node_id, day_id, scenario_id, hour)][l_id] <= flow_limit)
                    
                    for s in storage:
                        s_id = s['id']
                        model.addConstr(r_stor[(node_id, day_id, scenario_id, hour)][s_id] <= s['capacity_MW'] * investments[node_id]['storage'][s_id] if not s['existing'] else s['capacity_MW'])
                        model.addConstr(v_stor[(node_id, day_id, scenario_id, hour)][s_id] <= r_stor[(node_id, day_id, scenario_id, hour)][s_id])
                        if hour > 0:
                            model.addConstr(r_stor[(node_id, day_id, scenario_id, hour)][s_id] == r_stor[(node_id, day_id, scenario_id, hour-1)][s_id] + s['efficiency'] * u_stor[(node_id, day_id, scenario_id, hour-1)][s_id] - v_stor[(node_id, day_id, scenario_id, hour-1)][s_id])
                        else:
                            model.addConstr(r_stor[(node_id, day_id, scenario_id, hour)][s_id] == s['efficiency'] * u_stor[(node_id, day_id, scenario_id, hour)][s_id] - v_stor[(node_id, day_id, scenario_id, hour)][s_id])
    
    return model, investments, alpha, gamma, p_gen, s_res, f_fwd, f_bwd, u_stor, v_stor, r_stor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()
    
    instance = read_instance(args.instance_path)
    model, investments, alpha, gamma, p_gen, s_res, f_fwd, f_bwd, u_stor, v_stor, r_stor = create_model(instance)
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    model.Params.TimeLimit = args.time_limit
    model.optimize()
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.objVal
        
        solution = {
            'objective_value': objective_value,
            'alpha': {},
            'gamma': {},
            'p_gen': {},
            's_res': {},
            'f_fwd': {},
            'f_bwd': {},
            'u_stor': {},
            'v_stor': {},
            'r_stor': {},
            'investments': {}
        }
        
        for node in instance['scenario_tree']:
            node_id = node['id']
            solution['investments'][node_id] = {
                'generators': {g['id']: investments[node_id]['generators'][g['id']].x for g in instance['system']['generators'] if not g['existing']},
                'lines': {l['id']: investments[node_id]['lines'][l['id']].x for l in instance['system']['transmission_lines'] if not l['existing']},
                'storage': {s['id']: investments[node_id]['storage'][s['id']].x for s in instance['system']['storage'] if not s['existing']}
            }
        
        for node in instance['scenario_tree']:
            node_id = node['id']
            for day in instance['representative_days']:
                day_id = day['id']
                for lower_level_scenario in instance['lower_level_scenarios'][str(day_id)]:
                    scenario_id = lower_level_scenario['scenario_id']
                    for hour in range(instance['scenario_parameters']['hours_per_day']):
                        solution['alpha'][(node_id, day_id, scenario_id, hour)] = {g['id']: alpha[(node_id, day_id, scenario_id, hour)][g['id']].x for g in instance['system']['generators']}
                        solution['gamma'][(node_id, day_id, scenario_id, hour)] = {g['id']: gamma[(node_id, day_id, scenario_id, hour)][g['id']].x for g in instance['system']['generators']}
                        solution['p_gen'][(node_id, day_id, scenario_id, hour)] = {g['id']: p_gen[(node_id, day_id, scenario_id, hour)][g['id']].x for g in instance['system']['generators']}
                        solution['s_res'][(node_id, day_id, scenario_id, hour)] = {g['id']: s_res[(node_id, day_id, scenario_id, hour)][g['id']].x for g in instance['system']['generators']}
                        solution['f_fwd'][(node_id, day_id, scenario_id, hour)] = {l['id']: f_fwd[(node_id, day_id, scenario_id, hour)][l['id']].x for l in instance['system']['transmission_lines']}
                        solution['f_bwd'][(node_id, day_id, scenario_id, hour)] = {l['id']: f_bwd[(node_id, day_id, scenario_id, hour)][l['id']].x for l in instance['system']['transmission_lines']}
                        solution['u_stor'][(node_id, day_id, scenario_id, hour)] = {s['id']: u_stor[(node_id, day_id, scenario_id, hour)][s['id']].x for s in instance['system']['storage']}
                        solution['v_stor'][(node_id, day_id, scenario_id, hour)] = {s['id']: v_stor[(node_id, day_id, scenario_id, hour)][s['id']].x for s in instance['system']['storage']}
                        solution['r_stor'][(node_id, day_id, scenario_id, hour)] = {s['id']: r_stor[(node_id, day_id, scenario_id, hour)][s['id']].x for s in instance['system']['storage']}
        
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        
        if logger:
            logger.log_solution(objective_value, solution)

if __name__ == "__main__":
    main()