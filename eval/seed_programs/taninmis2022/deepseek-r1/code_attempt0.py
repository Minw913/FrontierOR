import argparse
import json
import time
from collections import deque
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

    bus_loads = {}
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

    total_scenarios = sum(1 for _ in bfs_order) * num_rep_days * num_lower_scenarios_per_day
    total_objective = 0.0
    investments = {}
    alpha_sol = {}
    gamma_sol = {}
    p_gen_sol = {}
    s_res_sol = {}
    f_fwd_sol = {}
    f_bwd_sol = {}
    u_stor_sol = {}
    v_stor_sol = {}
    r_stor_sol = {}

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
        time_per_scenario = max(1, remaining_time / num_scenarios_node) if total_scenarios > 0 else remaining_time
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
            if parent_power >= s['capacity_MW'] - 1e-5 and parent_energy >= s['capacity_MWh'] - 1e-5:
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
                for bus in buses:
                    bus_id = bus['id']
                    alpha[scenario_key][bus_id] = {}
                    gamma[scenario_key][bus_id] = {}
                    for hour in range(hours_per_day):
                        alpha[scenario_key][bus_id][hour] = model.addVar(vtype=GRB.CONTINUOUS, name=f"alpha_{scenario_key}_{bus_id}_{hour}")
                        gamma[scenario_key][bus_id][hour] = model.addVar(vtype=GRB.CONTINUOUS, name=f"gamma_{scenario_key}_{bus_id}_{hour}")
                for gen in generators:
                    gen_id = gen['id']
                    p_gen[scenario_key][gen_id] = {}
                    s_res[scenario_key][gen_id] = {}
                    for hour in range(hours_per_day):
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
            investment_cost += x_g[gen_id] * g['investment_cost']
        for l in candidate_lines:
            line_id = l['id']
            investment_cost += x_l[line_id] * l['investment_cost']
        for s in candidate_storage:
            stor_id = s['id']
            investment_cost += x_s[stor_id] * s['investment_cost']

        operational_cost = 0.0
        for rep_day in rep_days:
            rep_day_id = rep_day['id']
            day_scenarios = lower_scenario_map[rep_day_id]
            day_prob = rep_day['probability']
            for scenario_id, scenario in day_scenarios.items():
                scenario_key = (node_id, rep_day_id, scenario_id)
                scenario_prob = scenario['probability']
                total_prob = prob_node * day_prob * scenario_prob
                for gen in generators:
                    gen_id = gen['id']
                    for hour in range(hours_per_day):
                        operational_cost += total_prob * (p_gen[scenario_key][gen_id][hour] * gen['variable_cost'] + s_res[scenario_key][gen_id][hour] * gen['reserve_cost'])
                for stor in storage_units:
                    stor_id = stor['id']
                    for hour in range(hours_per_day):
                        operational_cost += total_prob * (u_stor[scenario_key][stor_id][hour] * stor['variable_charge_cost'] + v_stor[scenario_key][stor_id][hour] * stor['variable_discharge_cost'])

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
            model.addConstr(avail_stor_energy[stor_id] + x_s[stor_id] * s['capacity_MWh'] <= s['capacity_MWh'])

        for rep_day in rep_days:
            rep_day_id = rep_day['id']
            day_scenarios = lower_scenario_map[rep_day_id]
            for scenario_id, scenario in day_scenarios.items():
                scenario_key = (node_id, rep_day_id, scenario_id)
                for hour in range(hours_per_day):
                    for bus in buses:
                        bus_id = bus['id']
                        load_factor = scenario['load_factors'][str(bus_id)][hour]
                        demand = bus_loads[bus_id] * load_factor
                        gen_sum = 0.0
                        for gen in generators:
                            if gen['bus'] == bus_id:
                                gen_id = gen['id']
                                gen_sum += p_gen[scenario_key][gen_id][hour]
                        stor_charge_sum = 0.0
                        stor_discharge_sum = 0.0
                        for stor in storage_units:
                            if stor['bus'] == bus_id:
                                stor_id = stor['id']
                                stor_charge_sum += u_stor[scenario_key][stor_id][hour]
                                stor_discharge_sum += v_stor[scenario_key][stor_id][hour]
                        flow_in_sum = 0.0
                        flow_out_sum = 0.0
                        for line in transmission_lines:
                            if line['from_bus'] == bus_id:
                                flow_out_sum += f_fwd[scenario_key][line_id][hour]
                            if line['to_bus'] == bus_id:
                                flow_in_sum += f_fwd[scenario_key][line_id][hour]
                            if line['from_bus'] == bus_id:
                                flow_in_sum += f_bwd[scenario_key][line_id][hour]
                            if line['to_bus'] == bus_id:
                                flow_out_sum += f_bwd[scenario_key][line_id][hour]
                        model.addConstr(gen_sum + stor_discharge_sum + flow_in_sum == demand + stor_charge_sum + flow_out_sum + alpha[scenario_key][bus_id][hour] - gamma[scenario_key][bus_id][hour])

                    reserve_req = 0.0
                    for bus in buses:
                        bus_id = bus['id']
                        load_factor = scenario['load_factors'][str(bus_id)][hour]
                        demand = bus_loads[bus_id] * load_factor
                        reserve_req += demand * spinning_reserve_req
                    reserve_sum = 0.0
                    for gen in generators:
                        gen_id = gen['id']
                        reserve_sum += s_res[scenario_key][gen_id][hour]
                    for stor in storage_units:
                        stor_id = stor['id']
                        reserve_sum += r_stor[scenario_key][stor_id][hour]
                    model.addConstr(reserve_sum >= reserve_req)

                    for gen in generators:
                        gen_id = gen['id']
                        max_avail = avail_gen[gen_id]
                        if gen_id in x_g:
                            max_avail += x_g[gen_id] * gen['Pmax_MW']
                        model.addConstr(p_gen[scenario_key][gen_id][hour] + s_res[scenario_key][gen_id][hour] <= max_avail)
                        model.addConstr(p_gen[scenario_key][gen_id][hour] >= gen['Pmin_MW'])
                        model.addConstr(s_res[scenario_key][gen_id][hour] <= gen['max_reserve_MW'])

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
                            max_energy += x_s[stor_id] * stor['capacity_MWh']
                        model.addConstr(u_stor[scenario_key][stor_id][hour] <= max_power)
                        model.addConstr(v_stor[scenario_key][stor_id][hour] <= max_power)
                        model.addConstr(r_stor[scenario_key][stor_id][hour] <= max_power)
                        if hour == 0:
                            soc = stor['initial_SOC_MWh']
                        else:
                            soc = r_stor[scenario_key][stor_id][hour-1]
                        charge_eff = stor['charging_efficiency']
                        discharge_eff = stor['discharging_efficiency']
                        next_soc = soc + u_stor[scenario_key][stor_id][hour] * charge_eff - v_stor[scenario_key][stor_id][hour] / discharge_eff
                        model.addConstr(r_stor[scenario_key][stor_id][hour] == next_soc)
                        model.addConstr(r_stor[scenario_key][stor_id][hour] <= max_energy)
                        model.addConstr(r_stor[scenario_key][stor_id][hour] >= stor['min_SOC_MWh'])
                        model.addConstr(u_stor[scenario_key][stor_id][hour] + v_stor[scenario_key][stor_id][hour] + s_res[scenario_key][stor_id][hour] <= max_power)

        model.optimize()

        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT and model.SolCount > 0:
            total_objective += model.objVal
            for g in candidate_gens:
                gen_id = g['id']
                if x_g[gen_id].X > 0.5:
                    investments[gen_id] = node_id
                    avail_gen[gen_id] = g['Pmax_MW']
            for l in candidate_lines:
                line_id = l['id']
                if x_l[line_id].X > 0.5:
                    investments[line_id] = node_id
                    avail_line[line_id] = l['flow_limit_MW']
            for s in candidate_storage:
                stor_id = s['id']
                if x_s[stor_id].X > 0.5:
                    investments[stor_id] = node_id
                    avail_stor_power[stor_id] = s['capacity_MW']
                    avail_stor_energy[stor_id] = s['capacity_MWh']
            for rep_day in rep_days:
                rep_day_id = rep_day['id']
                day_scenarios = lower_scenario_map[rep_day_id]
                for scenario_id, scenario in day_scenarios.items():
                    scenario_key = (node_id, rep_day_id, scenario_id)
                    for bus in buses:
                        bus_id = bus['id']
                        for hour in range(hours_per_day):
                            alpha_sol[scenario_key, bus_id, hour] = alpha[scenario_key][bus_id][hour].X
                            gamma_sol[scenario_key, bus_id, hour] = gamma[scenario_key][bus_id][hour].X
                    for gen in generators:
                        gen_id = gen['id']
                        for hour in range(hours_per_day):
                            p_gen_sol[scenario_key, gen_id, hour] = p_gen[scenario_key][gen_id][hour].X
                            s_res_sol[scenario_key, gen_id, hour] = s_res[scenario_key][gen_id][hour].X
                    for line in transmission_lines:
                        line_id = line['id']
                        for hour in range(hours_per_day):
                            f_fwd_sol[scenario_key, line_id, hour] = f_fwd[scenario_key][line_id][hour].X
                            f_bwd_sol[scenario_key, line_id, hour] = f_bwd[scenario_key][line_id][hour].X
                    for stor in storage_units:
                        stor_id = stor['id']
                        for hour in range(hours_per_day):
                            u_stor_sol[scenario_key, stor_id, hour] = u_stor[scenario_key][stor_id][hour].X
                            v_stor_sol[scenario_key, stor_id, hour] = v_stor[scenario_key][stor_id][hour].X
                            r_stor_sol[scenario_key, stor_id, hour] = r_stor[scenario_key][stor_id][hour].X

        elapsed_time = time.time() - start_time
        remaining_time = args.time_limit - elapsed_time

    solution = {
        "investments": investments,
        "alpha": alpha_sol,
        "gamma": gamma_sol,
        "p_gen": p_gen_sol,
        "s_res": s_res_sol,
        "f_fwd": f_fwd_sol,
        "f_bwd": f_bwd_sol,
        "u_stor": u_stor_sol,
        "v_stor": v_stor_sol,
        "r_stor": r_stor_sol,
        "objective": total_objective
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()