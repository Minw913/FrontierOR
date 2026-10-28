import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    system = instance['system']
    scenario_params = instance['scenario_parameters']
    scenario_tree = instance['scenario_tree']
    rep_days = instance['representative_days']
    lower_scenarios = instance['lower_level_scenarios']
    global_params = instance['global_parameters']

    num_buses = system['num_buses']
    buses = system['buses']
    generators = system['generators']
    transmission_lines = system['transmission_lines']
    storage_units = system['storage']
    loads = system['loads']

    hours_per_day = scenario_params['hours_per_day']
    num_rep_days = len(rep_days)
    num_lower_scenarios_per_day = scenario_params['num_lower_level_scenarios_per_day']

    node_map = {node['id']: node for node in scenario_tree}
    for node in scenario_tree:
        node['parent'] = node['parent_id']
        node['children'] = node['children']

    stages = sorted(set(node['stage'] for node in scenario_tree))
    bfs_order = []
    for stage in stages:
        bfs_order.extend([node for node in scenario_tree if node['stage'] == stage])

    existing_gens = [g for g in generators if g['existing']]
    candidate_gens = [g for g in generators if not g['existing']]
    existing_lines = [l for l in transmission_lines if l['existing']]
    candidate_lines = [l for l in transmission_lines if not l['existing']]
    existing_storage = [s for s in storage_units if s['existing']]
    candidate_storage = [s for s in storage_units if not s['existing']]

    avail_gen = {g['id']: g['Pmax_MW'] for g in existing_gens}
    for g in candidate_gens:
        avail_gen[g['id']] = 0.0

    avail_line = {l['id']: l['flow_limit_MW'] for l in existing_lines}
    for l in candidate_lines:
        avail_line[l['id']] = 0.0

    avail_stor_power = {s['id']: s['capacity_MW'] for s in existing_storage}
    avail_stor_energy = {s['id']: s['capacity_MW'] for s in existing_storage}
    for s in candidate_storage:
        avail_stor_power[s['id']] = 0.0
        avail_stor_energy[s['id']] = 0.0

    bus_loads = {bus: 0.0 for bus in buses}
    for load in loads:
        bus = load['bus']
        bus_loads[bus] = load['annual_peak_load_MW']

    gen_map = {g['id']: g for g in generators}
    line_map = {l['id']: l for l in transmission_lines}
    stor_map = {s['id']: s for s in storage_units}

    rep_day_map = {d['id']: d for d in rep_days}
    lower_scenario_map = {}
    for rep_day_id, scenarios in lower_scenarios.items():
        rep_day_id = int(rep_day_id)
        lower_scenario_map[rep_day_id] = {s['scenario_id']: s for s in scenarios}

    spinning_reserve_req = global_params['spinning_reserve_percent']

    total_objective = 0.0
    solution_alpha = {}
    solution_gamma = {}
    solution_p_gen = {}
    solution_s_res = {}
    solution_f_fwd = {}
    solution_f_bwd = {}
    solution_u_stor = {}
    solution_v_stor = {}
    solution_r_stor = {}
    
    gen_investments = {}
    line_investments = {}
    stor_investments = {}

    elapsed_time = time.time() - start_time
    remaining_time = args.time_limit - elapsed_time

    for node in bfs_order:
        if remaining_time <= 1:
            break

        node_id = node['id']
        parent_id = node['parent']
        prob_node = node['probability']

        if parent_id is not None:
            avail_gen = avail_gen.copy()
            avail_line = avail_line.copy()
            avail_stor_power = avail_stor_power.copy()
            avail_stor_energy = avail_stor_energy.copy()

        num_scenarios_node = num_rep_days * num_lower_scenarios_per_day
        time_per_scenario = max(1, remaining_time / num_scenarios_node) if num_scenarios_node > 0 else remaining_time
        time_limit_node = min(remaining_time, time_per_scenario * num_scenarios_node)

        model = gp.Model(f"node_{node_id}")
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', time_limit_node)

        x_g = {}
        for g in candidate_gens:
            gen_id = g['id']
            parent_avail = avail_gen[gen_id]
            if parent_avail >= g['Pmax_MW'] - 1e-5:
                x_g[gen_id] = model.addVar(vtype=GRB.BINARY, name=f"x_g_{gen_id}", lb=0, ub=0)
            else:
                x_g[gen_id] = model.addVar(vtype=GRB.BINARY, name=f"x_g_{gen_id}")

        x_l = {}
        for l in candidate_lines:
            line_id = l['id']
            parent_avail = avail_line[line_id]
            if parent_avail >= l['flow_limit_MW'] - 1e-5:
                x_l[line_id] = model.addVar(vtype=GRB.BINARY, name=f"x_l_{line_id}", lb=0, ub=0)
            else:
                x_l[line_id] = model.addVar(vtype=GRB.BINARY, name=f"x_l_{line_id}")

        x_s = {}
        for s in candidate_storage:
            stor_id = s['id']
            parent_power = avail_stor_power[stor_id]
            parent_energy = avail_stor_energy[stor_id]
            if parent_power >= s['capacity_MW'] - 1e-5 and parent_energy >= s['capacity_MW'] - 1e-5:
                x_s[stor_id] = model.addVar(vtype=GRB.BINARY, name=f"x_s_{stor_id}", lb=0, ub=0)
            else:
                x_s[stor_id] = model.addVar(vtype=GRB.BINARY, name=f"x_s_{stor_id}")

        alpha = {}
        gamma = {}
        p_gen = {}
        s_res = {}
        f_fwd = {}
        f_bwd = {}
        u_stor = {}
        v_stor = {}
        r_stor = {}
        for rep_day in rep_days:
            rep_day_id = rep_day['id']
            day_scenarios = lower_scenario_map[rep_day_id]
            for scenario_id, scenario in day_scenarios.items():
                scenario_key = (node_id, rep_day_id, scenario_id)
                alpha[scenario_key] = {}
                gamma[scenario_key] = {}
                p_gen[scenario_key] = {}
                s_res[scenario_key] = {}
                f_fwd[scenario_key] = {}
                f_bwd[scenario_key] = {}
                u_stor[scenario_key] = {}
                v_stor[scenario_key] = {}
                r_stor[scenario_key] = {}
                
                for gen in generators:
                    gen_id = gen['id']
                    alpha[scenario_key][gen_id] = {}
                    gamma[scenario_key][gen_id] = {}
                    p_gen[scenario_key][gen_id] = {}
                    s_res[scenario_key][gen_id] = {}
                    for hour in range(hours_per_day):
                        alpha[scenario_key][gen_id][hour] = model.addVar(vtype=GRB.BINARY, name=f"alpha_{scenario_key}_{gen_id}_{hour}")
                        gamma[scenario_key][gen_id][hour] = model.addVar(vtype=GRB.BINARY, name=f"gamma_{scenario_key}_{gen_id}_{hour}")
                        p_gen[scenario_key][gen_id][hour] = model.addVar(vtype=GRB.CONTINUOUS, name=f"p_gen_{scenario_key}_{gen_id}_{hour}", lb=0.0)
                        s_res[scenario_key][gen_id][hour] = model.addVar(vtype=GRB.CONTINUOUS, name=f"s_res_{scenario_key}_{gen_id}_{hour}", lb=0.0)
                
                for line in transmission_lines:
                    line_id = line['id']
                    f_fwd[scenario_key][line_id] = {}
                    f_bwd[scenario_key][line_id] = {}
                    for hour in range(hours_per_day):
                        f_fwd[scenario_key][line_id][hour] = model.addVar(vtype=GRB.CONTINUOUS, name=f"f_fwd_{scenario_key}_{line_id}_{hour}", lb=0.0)
                        f_bwd[scenario_key][line_id][hour] = model.addVar(vtype=GRB.CONTINUOUS, name=f"f_bwd_{scenario_key}_{line_id}_{hour}", lb=0.0)
                
                for stor in storage_units:
                    stor_id = stor['id']
                    u_stor[scenario_key][stor_id] = {}
                    v_stor[scenario_key][stor_id] = {}
                    r_stor[scenario_key][stor_id] = {}
                    for hour in range(hours_per_day):
                        u_stor[scenario_key][stor_id][hour] = model.addVar(vtype=GRB.CONTINUOUS, name=f"u_stor_{scenario_key}_{stor_id}_{hour}", lb=0.0)
                        v_stor[scenario_key][stor_id][hour] = model.addVar(vtype=GRB.CONTINUOUS, name=f"v_stor_{scenario_key}_{stor_id}_{hour}", lb=0.0)
                        r_stor[scenario_key][stor_id][hour] = model.addVar(vtype=GRB.CONTINUOUS, name=f"r_stor_{scenario_key}_{stor_id}_{hour}", lb=0.0)

        investment_cost = 0.0
        for g in candidate_gens:
            gen_id = g['id']
            investment_cost += x_g[gen_id] * g['capital_cost_per_kW'] * g['Pmax_MW'] * 1000
        for l in candidate_lines:
            line_id = l['id']
            investment_cost += x_l[line_id] * l['capital_cost_per_kW'] * l['flow_limit_MW'] * 1000
        for s in candidate_storage:
            stor_id = s['id']
            investment_cost += x_s[stor_id] * s['capital_cost_per_kW'] * s['capacity_MW'] * 1000

        operational_cost = 0.0
        for rep_day in rep_days:
            rep_day_id = rep_day['id']
            day_scenarios = lower_scenario_map[rep_day_id]
            for scenario_id, scenario in day_scenarios.items():
                scenario_key = (node_id, rep_day_id, scenario_id)
                for gen in generators:
                    gen_id = gen['id']
                    variable_cost = gen['b_cost'] + gen['c_cost'] * gen['Pmax_MW']
                    for hour in range(hours_per_day):
                        operational_cost += prob_node * (
                            gen['a_cost'] * alpha[scenario_key][gen_id][hour] +
                            variable_cost * p_gen[scenario_key][gen_id][hour] +
                            gen['startup_cost'] * gamma[scenario_key][gen_id][hour]
                        )

        model.setObjective(investment_cost + operational_cost, GRB.MINIMIZE)

        for g in candidate_gens:
            gen_id = g['id']
            model.addConstr(avail_gen[gen_id] + x_g[gen_id] * g['Pmax_MW'] <= g['Pmax_MW'])
        for l in candidate_lines:
            line_id = l['id']
            model.addConstr(avail_line[line_id] + x_l[line_id] * l['flow_limit_MW'] <= l['flow_limit_MW'])
        for s in candidate_storage:
            stor_id = s['id']
            model.addConstr(avail_stor_power[stor_id] + x_s[stor_id] * s['capacity_MW'] <= s['capacity_MW'])
            model.addConstr(avail_stor_energy[stor_id] + x_s[stor_id] * s['capacity_MW'] <= s['capacity_MW'])

        for rep_day in rep_days:
            rep_day_id = rep_day['id']
            day_scenarios = lower_scenario_map[rep_day_id]
            for scenario_id, scenario in day_scenarios.items():
                scenario_key = (node_id, rep_day_id, scenario_id)
                for hour in range(hours_per_day):
                    for bus in buses:
                        demand_multiplier = scenario['demand_multipliers'][hour]
                        base_demand = bus_loads[bus] * rep_day['load_profile'][hour]
                        demand = base_demand * demand_multiplier
                        
                        gen_sum = 0.0
                        for gen in generators:
                            if gen['bus'] == bus:
                                gen_id = gen['id']
                                gen_sum += p_gen[scenario_key][gen_id][hour]
                        
                        stor_net = 0.0
                        for stor in storage_units:
                            if stor['bus'] == bus:
                                stor_id = stor['id']
                                stor_net += v_stor[scenario_key][stor_id][hour] - u_stor[scenario_key][stor_id][hour]
                        
                        flow_in = 0.0
                        flow_out = 0.0
                        for line in transmission_lines:
                            line_id = line['id']
                            if line['from_bus'] == bus:
                                flow_out += f_fwd[scenario_key][line_id][hour] * (1 - line['loss_factor'])
                                flow_in += f_bwd[scenario_key][line_id][hour] * (1 - line['loss_factor'])
                            if line['to_bus'] == bus:
                                flow_in += f_fwd[scenario_key][line_id][hour] * (1 - line['loss_factor'])
                                flow_out += f_bwd[scenario_key][line_id][hour] * (1 - line['loss_factor'])
                        
                        model.addConstr(gen_sum + stor_net + flow_in == demand + flow_out)

                    reserve_req = 0.0
                    for bus in buses:
                        demand_multiplier = scenario['demand_multipliers'][hour]
                        base_demand = bus_loads[bus] * rep_day['load_profile'][hour]
                        demand = base_demand * demand_multiplier
                        reserve_req += demand * spinning_reserve_req
                    
                    reserve_sum = 0.0
                    for gen in generators:
                        gen_id = gen['id']
                        reserve_sum += s_res[scenario_key][gen_id][hour]
                    model.addConstr(reserve_sum >= reserve_req)

                    for gen in generators:
                        gen_id = gen['id']
                        max_avail = avail_gen[gen_id]
                        if gen_id in x_g:
                            max_avail += x_g[gen_id] * gen['Pmax_MW']
                        model.addConstr(p_gen[scenario_key][gen_id][hour] + s_res[scenario_key][gen_id][hour] <= max_avail * alpha[scenario_key][gen_id][hour])
                        model.addConstr(p_gen[scenario_key][gen_id][hour] >= gen['Pmin_MW'] * alpha[scenario_key][gen_id][hour])
                        
                        if hour > 0:
                            model.addConstr(p_gen[scenario_key][gen_id][hour] - p_gen[scenario_key][gen_id][hour-1] <= gen['ramp_MW_per_h'])
                            model.addConstr(p_gen[scenario_key][gen_id][hour-1] - p_gen[scenario_key][gen_id][hour] <= gen['ramp_MW_per_h'])
                        
                        if hour > 0:
                            model.addConstr(gamma[scenario_key][gen_id][hour] >= alpha[scenario_key][gen_id][hour] - alpha[scenario_key][gen_id][hour-1])
                        else:
                            model.addConstr(gamma[scenario_key][gen_id][hour] >= alpha[scenario_key][gen_id][hour])

                    for line in transmission_lines:
                        line_id = line['id']
                        max_avail = avail_line[line_id]
                        if line_id in x_l:
                            max_avail += x_l[line_id] * line['flow_limit_MW']
                        model.addConstr(f_fwd[scenario_key][line_id][hour] <= max_avail)
                        model.addConstr(f_bwd[scenario_key][line_id][hour] <= max_avail)

                    for stor in storage_units:
                        stor_id = stor['id']
                        max_power = avail_stor_power[stor_id]
                        max_energy = avail_stor_energy[stor_id]
                        if stor_id in x_s:
                            max_power += x_s[stor_id] * stor['capacity_MW']
                            max_energy += x_s[stor_id] * stor['capacity_MW']
                        model.addConstr(u_stor[scenario_key][stor_id][hour] <= max_power)
                        model.addConstr(v_stor[scenario_key][stor_id][hour] <= max_power)
                        model.addConstr(r_stor[scenario_key][stor_id][hour] <= max_energy)
                        
                        if hour == 0:
                            prev_soc = 0
                        else:
                            prev_soc = r_stor[scenario_key][stor_id][hour-1]
                        model.addConstr(
                            r_stor[scenario_key][stor_id][hour] == 
                            prev_soc + 
                            stor['efficiency'] * u_stor[scenario_key][stor_id][hour] - 
                            v_stor[scenario_key][stor_id][hour]
                        )

        model.optimize()

        if model.status == GRB.OPTIMAL or (model.status == GRB.TIME_LIMIT and model.SolCount > 0):
            total_objective += model.objVal
            
            for g in candidate_gens:
                gen_id = g['id']
                if x_g[gen_id].X > 0.5:
                    gen_investments[gen_id] = node_id
                    avail_gen[gen_id] = g['Pmax_MW']
            
            for l in candidate_lines:
                line_id = l['id']
                if x_l[line_id].X > 0.5:
                    line_investments[line_id] = node_id
                    avail_line[line_id] = l['flow_limit_MW']
            
            for s in candidate_storage:
                stor_id = s['id']
                if x_s[stor_id].X > 0.5:
                    stor_investments[stor_id] = node_id
                    avail_stor_power[stor_id] = s['capacity_MW']
                    avail_stor_energy[stor_id] = s['capacity_MW']
            
            for rep_day in rep_days:
                rep_day_id = rep_day['id']
                day_scenarios = lower_scenario_map[rep_day_id]
                for scenario_id, scenario in day_scenarios.items():
                    scenario_key = (node_id, rep_day_id, scenario_id)
                    for gen in generators:
                        gen_id = gen['id']
                        for hour in range(hours_per_day):
                            if node_id not in solution_alpha:
                                solution_alpha[node_id] = {}
                            if rep_day_id not in solution_alpha[node_id]:
                                solution_alpha[node_id][rep_day_id] = {}
                            if scenario_id not in solution_alpha[node_id][rep_day_id]:
                                solution_alpha[node_id][rep_day_id][scenario_id] = {}
                            if hour not in solution_alpha[node_id][rep_day_id][scenario_id]:
                                solution_alpha[node_id][rep_day_id][scenario_id][hour] = {}
                            solution_alpha[node_id][rep_day_id][scenario_id][hour][gen_id] = alpha[scenario_key][gen_id][hour].X
                            
                            if node_id not in solution_gamma:
                                solution_gamma[node_id] = {}
                            if rep_day_id not in solution_gamma[node_id]:
                                solution_gamma[node_id][rep_day_id] = {}
                            if scenario_id not in solution_gamma[node_id][rep_day_id]:
                                solution_gamma[node_id][rep_day_id][scenario_id] = {}
                            if hour not in solution_gamma[node_id][rep_day_id][scenario_id]:
                                solution_gamma[node_id][rep_day_id][scenario_id][hour] = {}
                            solution_gamma[node_id][rep_day_id][scenario_id][hour][gen_id] = gamma[scenario_key][gen_id][hour].X
                            
                            if node_id not in solution_p_gen:
                                solution_p_gen[node_id] = {}
                            if rep_day_id not in solution_p_gen[node_id]:
                                solution_p_gen[node_id][rep_day_id] = {}
                            if scenario_id not in solution_p_gen[node_id][rep_day_id]:
                                solution_p_gen[node_id][rep_day_id][scenario_id] = {}
                            if hour not in solution_p_gen[node_id][rep_day_id][scenario_id]:
                                solution_p_gen[node_id][rep_day_id][scenario_id][hour] = {}
                            solution_p_gen[node_id][rep_day_id][scenario_id][hour][gen_id] = p_gen[scenario_key][gen_id][hour].X
                            
                            if node_id not in solution_s_res:
                                solution_s_res[node_id] = {}
                            if rep_day_id not in solution_s_res[node_id]:
                                solution_s_res[node_id][rep_day_id] = {}
                            if scenario_id not in solution_s_res[node_id][rep_day_id]:
                                solution_s_res[node_id][rep_day_id][scenario_id] = {}
                            if hour not in solution_s_res[node_id][rep_day_id][scenario_id]:
                                solution_s_res[node_id][rep_day_id][scenario_id][hour] = {}
                            solution_s_res[node_id][rep_day_id][scenario_id][hour][gen_id] = s_res[scenario_key][gen_id][hour].X
                    
                    for line in transmission_lines:
                        line_id = line['id']
                        for hour in range(hours_per_day):
                            if node_id not in solution_f_fwd:
                                solution_f_fwd[node_id] = {}
                            if rep_day_id not in solution_f_fwd[node_id]:
                                solution_f_fwd[node_id][rep_day_id] = {}
                            if scenario_id not in solution_f_fwd[node_id][rep_day_id]:
                                solution_f_fwd[node_id][rep_day_id][scenario_id] = {}
                            if hour not in solution_f_fwd[node_id][rep_day_id][scenario_id]:
                                solution_f_fwd[node_id][rep_day_id][scenario_id][hour] = {}
                            solution_f_fwd[node_id][rep_day_id][scenario_id][hour][line_id] = f_fwd[scenario_key][line_id][hour].X
                            
                            if node_id not in solution_f_bwd:
                                solution_f_bwd[node_id] = {}
                            if rep_day_id not in solution_f_bwd[node_id]:
                                solution_f_bwd[node_id][rep_day_id] = {}
                            if scenario_id not in solution_f_bwd[node_id][rep_day_id]:
                                solution_f_bwd[node_id][rep_day_id][scenario_id] = {}
                            if hour not in solution_f_bwd[node_id][rep_day_id][scenario_id]:
                                solution_f_bwd[node_id][rep_day_id][scenario_id][hour] = {}
                            solution_f_bwd[node_id][rep_day_id][scenario_id][hour][line_id] = f_bwd[scenario_key][line_id][hour].X
                    
                    for stor in storage_units:
                        stor_id = stor['id']
                        for hour in range(hours_per_day):
                            if node_id not in solution_u_stor:
                                solution_u_stor[node_id] = {}
                            if rep_day_id not in solution_u_stor[node_id]:
                                solution_u_stor[node_id][rep_day_id] = {}
                            if scenario_id not in solution_u_stor[node_id][rep_day_id]:
                                solution_u_stor[node_id][rep_day_id][scenario_id] = {}
                            if hour not in solution_u_stor[node_id][rep_day_id][scenario_id]:
                                solution_u_stor[node_id][rep_day_id][scenario_id][hour] = {}
                            solution_u_stor[node_id][rep_day_id][scenario_id][hour][stor_id] = u_stor[scenario_key][stor_id][hour].X
                            
                            if node_id not in solution_v_stor:
                                solution_v_stor[node_id] = {}
                            if rep_day_id not in solution_v_stor[node_id]:
                                solution_v_stor[node_id][rep_day_id] = {}
                            if scenario_id not in solution_v_stor[node_id][rep_day_id]:
                                solution_v_stor[node_id][rep_day_id][scenario_id] = {}
                            if hour not in solution_v_stor[node_id][rep_day_id][scenario_id]:
                                solution_v_stor[node_id][rep_day_id][scenario_id][hour] = {}
                            solution_v_stor[node_id][rep_day_id][scenario_id][hour][stor_id] = v_stor[scenario_key][stor_id][hour].X
                            
                            if node_id not in solution_r_stor:
                                solution_r_stor[node_id] = {}
                            if rep_day_id not in solution_r_stor[node_id]:
                                solution_r_stor[node_id][rep_day_id] = {}
                            if scenario_id not in solution_r_stor[node_id][rep_day_id]:
                                solution_r_stor[node_id][rep_day_id][scenario_id] = {}
                            if hour not in solution_r_stor[node_id][rep_day_id][scenario_id]:
                                solution_r_stor[node_id][rep_day_id][scenario_id][hour] = {}
                            solution_r_stor[node_id][rep_day_id][scenario_id][hour][stor_id] = r_stor[scenario_key][stor_id][hour].X

        elapsed_time = time.time() - start_time
        remaining_time = args.time_limit - elapsed_time

    candidate_gen_ids = [g['id'] for g in candidate_gens]
    candidate_line_ids = [l['id'] for l in candidate_lines]
    candidate_stor_ids = [s['id'] for s in candidate_storage]

    investments_dict = {}
    for node in bfs_order:
        node_id = node['id']
        gen_dict = {}
        for gen_id in candidate_gen_ids:
            if gen_id in gen_investments and gen_investments[gen_id] == node_id:
                gen_dict[gen_id] = 1
            else:
                gen_dict[gen_id] = 0
        
        line_dict = {}
        for line_id in candidate_line_ids:
            if line_id in line_investments and line_investments[line_id] == node_id:
                line_dict[line_id] = 1
            else:
                line_dict[line_id] = 0
        
        stor_dict = {}
        for stor_id in candidate_stor_ids:
            if stor_id in stor_investments and stor_investments[stor_id] == node_id:
                stor_dict[stor_id] = 1
            else:
                stor_dict[stor_id] = 0
        
        investments_dict[node_id] = {
            "generators": gen_dict,
            "lines": line_dict,
            "storage": stor_dict
        }

    solution = {
        "objective_value": total_objective,
        "investments": investments_dict,
        "alpha": solution_alpha,
        "gamma": solution_gamma,
        "p_gen": solution_p_gen,
        "s_res": solution_s_res,
        "f_fwd": solution_f_fwd,
        "f_bwd": solution_f_bwd,
        "u_stor": solution_u_stor,
        "v_stor": solution_v_stor,
        "r_stor": solution_r_stor
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()