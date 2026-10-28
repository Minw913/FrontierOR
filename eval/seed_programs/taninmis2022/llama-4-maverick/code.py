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
    
    system = instance['system']
    scenario_parameters = instance['scenario_parameters']
    scenario_tree = instance['scenario_tree']
    representative_days = instance['representative_days']
    lower_level_scenarios = instance.get('lower_level_scenarios', {})
    global_parameters = instance['global_parameters']
    
    num_buses = system['num_buses']
    buses = system['buses']
    generators = system['generators']
    transmission_lines = system['transmission_lines']
    storage = system.get('storage', [])
    loads = system['loads']
    
    num_stages = scenario_parameters['num_stages']
    num_upper_level_scenarios = scenario_parameters['num_upper_level_scenarios']
    num_lower_level_scenarios_per_day = scenario_parameters['num_lower_level_scenarios_per_day']
    num_representative_days = scenario_parameters['num_representative_days']
    hours_per_day = scenario_parameters['hours_per_day']
    
    existing_generators = [g for g in generators if g.get('existing', False)]
    candidate_generators = [g for g in generators if not g.get('existing', False)]
    existing_lines = [l for l in transmission_lines if l.get('existing', False)]
    candidate_lines = [l for l in transmission_lines if not l.get('existing', False)]
    existing_storage = [s for s in storage if s.get('existing', False)]
    candidate_storage = [s for s in storage if not s.get('existing', False)]
    
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
            for lower_level_scenario in lower_level_scenarios.get(str(day_id), []):
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
    
    objective = quicksum(node['probability'] * (
        quicksum(g.get('capital_cost_per_kW', 0) * 1000 * investments[node['id']]['generators'][g['id']] * g.get('Pmax_MW', 0) for g in candidate_generators) +
        quicksum(l.get('capital_cost_per_kW', 0) * 1000 * investments[node['id']]['lines'][l['id']] * l.get('flow_limit_MW', 0) for l in candidate_lines) +
        quicksum(s.get('capital_cost_per_kW', 0) * 1000 * investments[node['id']]['storage'][s['id']] * s.get('capacity_MW', 0) for s in candidate_storage) +
        quicksum(
            quicksum(
                quicksum(
                    quicksum(
                        g.get('startup_cost', 0) * gamma[(node['id'], day['id'], scenario_id, hour)][g['id']] +
                        (g.get('a_cost', 0) * alpha[(node['id'], day['id'], scenario_id, hour)][g['id']] +
                         (g.get('b_cost', 0) + g.get('c_cost', 0) * g.get('Pmax_MW', 0)) * p_gen[(node['id'], day['id'], scenario_id, hour)][g['id']])
                        for hour in range(hours_per_day)
                    ) for lower_level_scenario in lower_level_scenarios.get(str(day['id']), []) for scenario_id in [lower_level_scenario['scenario_id']]
                ) / max(len(lower_level_scenarios.get(str(day['id']), [])), 1)
            ) for day in representative_days
        ) / max(num_representative_days, 1)
    ) for node in scenario_tree)
    model.setObjective(objective, GRB.MINIMIZE)
    
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
            for lower_level_scenario in lower_level_scenarios.get(str(day_id), []):
                scenario_id = lower_level_scenario['scenario_id']
                for hour in range(hours_per_day):
                    for g in generators:
                        g_id = g['id']
                        available_capacity = g.get('Pmax_MW', 0) if g.get('existing', True) else g.get('Pmax_MW', 0) * investments[node_id]['generators'][g_id]
                        model.addConstr(p_gen[(node_id, day_id, scenario_id, hour)][g_id] + s_res[(node_id, day_id, scenario_id, hour)][g_id] <= available_capacity)
                        model.addConstr(p_gen[(node_id, day_id, scenario_id, hour)][g_id] >= g.get('Pmin_MW', 0) * alpha[(node_id, day_id, scenario_id, hour)][g_id])
                        if hour > 0:
                            model.addConstr(gamma[(node_id, day_id, scenario_id, hour)][g_id] >= alpha[(node_id, day_id, scenario_id, hour)][g_id] - alpha[(node_id, day_id, scenario_id, hour-1)][g_id])
                            if 'ramp_MW_per_h' in g:
                                model.addConstr(p_gen[(node_id, day_id, scenario_id, hour)][g_id] - p_gen[(node_id, day_id, scenario_id, hour-1)][g_id] <= g['ramp_MW_per_h'])
                                model.addConstr(p_gen[(node_id, day_id, scenario_id, hour-1)][g_id] - p_gen[(node_id, day_id, scenario_id, hour)][g_id] <= g['ramp_MW_per_h'])
                    
                    for bus in buses:
                        bus_load = next((load['annual_peak_load_MW'] for load in loads if load.get('bus') == bus), 0)
                        load_profile = day.get('load_profile', [0]*hours_per_day)[hour]
                        demand_multiplier = lower_level_scenario.get('demand_multipliers', [1]*hours_per_day)[hour]
                        demand = bus_load * load_profile * demand_multiplier
                        
                        total_gen = quicksum(p_gen[(node_id, day_id, scenario_id, hour)][g['id']] for g in generators if g.get('bus') == bus)
                        total_storage_withdrawal = quicksum(v_stor[(node_id, day_id, scenario_id, hour)][s['id']] for s in storage if s.get('bus') == bus)
                        total_storage_injection = quicksum(u_stor[(node_id, day_id, scenario_id, hour)][s['id']] for s in storage if s.get('bus') == bus)
                        total_fwd_flow = quicksum(f_fwd[(node_id, day_id, scenario_id, hour)][l['id']] for l in transmission_lines if l.get('from_bus') == bus)
                        total_bwd_flow = quicksum(f_bwd[(node_id, day_id, scenario_id, hour)][l['id']] for l in transmission_lines if l.get('to_bus') == bus)
                        total_loss = quicksum(l.get('loss_factor', 0) * (f_fwd[(node_id, day_id, scenario_id, hour)][l['id']] + f_bwd[(node_id, day_id, scenario_id, hour)][l['id']]) for l in transmission_lines if l.get('from_bus') == bus or l.get('to_bus') == bus)
                        
                        model.addConstr(total_gen + total_storage_withdrawal - total_storage_injection + total_bwd_flow - total_fwd_flow >= demand)
                        model.addConstr(total_gen + total_storage_withdrawal - total_storage_injection + total_bwd_flow - total_fwd_flow - total_loss <= demand)
                    
                    for l in transmission_lines:
                        l_id = l['id']
                        flow_limit = l.get('flow_limit_MW', 0) if l.get('existing', True) else investments[node_id]['lines'][l_id] * l.get('flow_limit_MW', 0)
                        model.addConstr(f_fwd[(node_id, day_id, scenario_id, hour)][l_id] <= flow_limit)
                        model.addConstr(f_bwd[(node_id, day_id, scenario_id, hour)][l_id] <= flow_limit)
                    
                    for s in storage:
                        s_id = s['id']
                        available_capacity = s.get('capacity_MW', 0) if s.get('existing', True) else s.get('capacity_MW', 0) * investments[node_id]['storage'][s_id]
                        model.addConstr(r_stor[(node_id, day_id, scenario_id, hour)][s_id] <= available_capacity)
                        model.addConstr(v_stor[(node_id, day_id, scenario_id, hour)][s_id] <= r_stor[(node_id, day_id, scenario_id, hour)][s_id])
                        if hour > 0:
                            model.addConstr(r_stor[(node_id, day_id, scenario_id, hour)][s_id] == r_stor[(node_id, day_id, scenario_id, hour-1)][s_id] + s.get('efficiency', 0) * u_stor[(node_id, day_id, scenario_id, hour-1)][s_id] - v_stor[(node_id, day_id, scenario_id, hour-1)][s_id])
                        else:
                            model.addConstr(r_stor[(node_id, day_id, scenario_id, hour)][s_id] == 0)
    
    model.update()
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
                'generators': {g['id']: investments[node_id]['generators'][g['id']].x for g in instance['system']['generators'] if not g.get('existing', True)},
                'lines': {l['id']: investments[node_id]['lines'][l['id']].x for l in instance['system']['transmission_lines'] if not l.get('existing', True)},
                'storage': {s['id']: investments[node_id]['storage'][s['id']].x for s in instance.get('system', {}).get('storage', []) if not s.get('existing', True)}
            }
        
        for node in instance['scenario_tree']:
            node_id = node['id']
            for day in instance['representative_days']:
                day_id = day['id']
                for lower_level_scenario in instance.get('lower_level_scenarios', {}).get(str(day_id), []):
                    scenario_id = lower_level_scenario['scenario_id']
                    for hour in range(instance['scenario_parameters']['hours_per_day']):
                        if (node_id, day_id, scenario_id, hour) not in solution['alpha']:
                            solution['alpha'][(node_id, day_id, scenario_id, hour)] = {}
                            solution['gamma'][(node_id, day_id, scenario_id, hour)] = {}
                            solution['p_gen'][(node_id, day_id, scenario_id, hour)] = {}
                            solution['s_res'][(node_id, day_id, scenario_id, hour)] = {}
                            solution['f_fwd'][(node_id, day_id, scenario_id, hour)] = {}
                            solution['f_bwd'][(node_id, day_id, scenario_id, hour)] = {}
                            solution['u_stor'][(node_id, day_id, scenario_id, hour)] = {}
                            solution['v_stor'][(node_id, day_id, scenario_id, hour)] = {}
                            solution['r_stor'][(node_id, day_id, scenario_id, hour)] = {}
                        for g in instance['system']['generators']:
                            g_id = g['id']
                            solution['alpha'][(node_id, day_id, scenario_id, hour)][g_id] = alpha[(node_id, day_id, scenario_id, hour)][g_id].x
                            solution['gamma'][(node_id, day_id, scenario_id, hour)][g_id] = gamma[(node_id, day_id, scenario_id, hour)][g_id].x
                            solution['p_gen'][(node_id, day_id, scenario_id, hour)][g_id] = p_gen[(node_id, day_id, scenario_id, hour)][g_id].x
                            solution['s_res'][(node_id, day_id, scenario_id, hour)][g_id] = s_res[(node_id, day_id, scenario_id, hour)][g_id].x
                        for l in instance['system']['transmission_lines']:
                            l_id = l['id']
                            solution['f_fwd'][(node_id, day_id, scenario_id, hour)][l_id] = f_fwd[(node_id, day_id, scenario_id, hour)][l_id].x
                            solution['f_bwd'][(node_id, day_id, scenario_id, hour)][l_id] = f_bwd[(node_id, day_id, scenario_id, hour)][l_id].x
                        for s in instance.get('system', {}).get('storage', []):
                            s_id = s['id']
                            solution['u_stor'][(node_id, day_id, scenario_id, hour)][s_id] = u_stor[(node_id, day_id, scenario_id, hour)][s_id].x
                            solution['v_stor'][(node_id, day_id, scenario_id, hour)][s_id] = v_stor[(node_id, day_id, scenario_id, hour)][s_id].x
                            solution['r_stor'][(node_id, day_id, scenario_id, hour)][s_id] = r_stor[(node_id, day_id, scenario_id, hour)][s_id].x
        
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, default=lambda x: str(x))
        
        if logger:
            logger.log_solution(objective_value, solution)

if __name__ == "__main__":
    main()