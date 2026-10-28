import json
import argparse
import time
import sys
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    # Extract data
    system = data['system']
    scenario_params = data['scenario_parameters']
    scenario_tree = data['scenario_tree']
    rep_days = data['representative_days']
    lower_scens = data['lower_level_scenarios']
    global_params = data['global_parameters']

    buses = system['buses']
    generators = system['generators']
    lines = system['transmission_lines']
    storages = system['storage']
    loads = {ld['bus']: ld['annual_peak_load_MW'] for ld in system['loads']}

    num_stages = scenario_params['num_stages']
    num_rep_days = scenario_params['num_representative_days']
    hours_per_day = scenario_params['hours_per_day']
    num_lower_scens = scenario_params['num_lower_level_scenarios_per_day']
    investment_years = scenario_params['investment_period_years']

    # Build scenario tree map
    tree_nodes = {node['id']: node for node in scenario_tree}
    root_id = next(nid for nid, n in tree_nodes.items() if n['parent_id'] is None)
    leaf_nodes = [nid for nid, n in tree_nodes.items() if not n['children']]

    # Precompute probabilities and ancestors
    node_prob = {n['id']: n['probability'] for n in scenario_tree}
    ancestors = {}
    for nid in tree_nodes:
        anc = []
        current = nid
        while current is not None:
            anc.append(current)
            current = tree_nodes[current]['parent_id']
        ancestors[nid] = list(reversed(anc))

    # Generator, line, storage lists
    gens = generators
    gen_ids = [g['id'] for g in gens]
    gen_by_id = {g['id']: g for g in gens}
    line_ids = [l['id'] for l in lines]
    storage_ids = [s['id'] for s in storages]

    # Candidate sets
    candidate_gens = [g for g in gens if not g['existing']]
    candidate_lines = [l for l in lines if not l['existing']]
    candidate_stor = [s for s in storages if not s['existing']]

    # Representative days
    day_ids = [d['id'] for d in rep_days]
    day_by_id = {d['id']: d for d in rep_days}

    # Build model
    model = gp.Model("PowerExpansion")
    model.setParam(GRB.Param.Seed, 0)
    model.setParam(GRB.Param.MIPGap, 1e-4)
    model.setParam(GRB.Param.NumericFocus, 0)
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.TimeLimit, max(args.time_limit - 30, 5))

    # Investment variables: built at this node (binary)
    build_gen = {}
    for nid in tree_nodes:
        for g in candidate_gens:
            build_gen[nid, g['id']] = model.addVar(vtype=GRB.BINARY, name=f"build_g_{nid}_{g['id']}")

    build_line = {}
    for nid in tree_nodes:
        for ln in candidate_lines:
            build_line[nid, ln['id']] = model.addVar(vtype=GRB.BINARY, name=f"build_l_{nid}_{ln['id']}")

    build_stor = {}
    for nid in tree_nodes:
        for st in candidate_stor:
            build_stor[nid, st['id']] = model.addVar(vtype=GRB.BINARY, name=f"build_s_{nid}_{st['id']}")

    # Available capacity at each node (cumulative)
    avail_gen = {}
    for nid in tree_nodes:
        for g in candidate_gens:
            anc = ancestors[nid]
            avail_gen[nid, g['id']] = gp.quicksum(build_gen[a, g['id']] for a in anc)

    avail_line = {}
    for nid in tree_nodes:
        for ln in candidate_lines:
            anc = ancestors[nid]
            avail_line[nid, ln['id']] = gp.quicksum(build_line[a, ln['id']] for a in anc)

    avail_stor = {}
    for nid in tree_nodes:
        for st in candidate_stor:
            anc = ancestors[nid]
            avail_stor[nid, st['id']] = gp.quicksum(build_stor[a, st['id']] for a in anc)

    # Operational variables
    alpha = {}
    gamma = {}
    p_gen = {}
    s_res = {}
    for nid in tree_nodes:
        for d_idx, d in enumerate(day_ids):
            for s in range(num_lower_scens):
                for h in range(hours_per_day):
                    for g_idx, g_id in enumerate(gen_ids):
                        key = (nid, d, s, h, g_id)
                        alpha[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"alpha_{nid}_{d}_{s}_{h}_{g_id}")
                        gamma[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"gamma_{nid}_{d}_{s}_{h}_{g_id}")
                        p_gen[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"p_{nid}_{d}_{s}_{h}_{g_id}")
                        s_res[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"sr_{nid}_{d}_{s}_{h}_{g_id}")

    f_fwd = {}
    f_bwd = {}
    for nid in tree_nodes:
        for d_idx, d in enumerate(day_ids):
            for s in range(num_lower_scens):
                for h in range(hours_per_day):
                    for l_idx, ln_id in enumerate(line_ids):
                        key = (nid, d, s, h, ln_id)
                        f_fwd[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"fwd_{nid}_{d}_{s}_{h}_{ln_id}")
                        f_bwd[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"bwd_{nid}_{d}_{s}_{h}_{ln_id}")

    u_stor = {}
    v_stor = {}
    r_stor = {}
    for nid in tree_nodes:
        for d_idx, d in enumerate(day_ids):
            for s in range(num_lower_scens):
                for h in range(hours_per_day):
                    for st_idx, st_id in enumerate(storage_ids):
                        key = (nid, d, s, h, st_id)
                        u_stor[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"u_{nid}_{d}_{s}_{h}_{st_id}")
                        v_stor[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"v_{nid}_{d}_{s}_{h}_{st_id}")
                        r_stor[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"r_{nid}_{d}_{s}_{h}_{st_id}")

    # Objective
    total_cost = 0.0

    # Investment cost
    for nid in tree_nodes:
        prob = node_prob[nid]
        invest_cost = 0.0
        for g in candidate_gens:
            invest_cost += build_gen[nid, g['id']] * g.get('capital_cost_per_kW', 0.0) * g['Pmax_MW'] * 1000.0
        for ln in candidate_lines:
            invest_cost += build_line[nid, ln['id']] * ln.get('capital_cost_per_kW', 0.0) * ln['flow_limit_MW'] * 1000.0
        for st in candidate_stor:
            invest_cost += build_stor[nid, st['id']] * st.get('capital_cost_per_kW', 0.0) * st['capacity_MW'] * 1000.0
        total_cost += prob * invest_cost

    # Operating cost weight (approximate annualization)
    op_weight = 365.0 * investment_years / (num_rep_days * num_lower_scens)
    for nid in tree_nodes:
        prob = node_prob[nid]
        for d_idx, d in enumerate(day_ids):
            for s in range(num_lower_scens):
                for h in range(hours_per_day):
                    for g_idx, g_id in enumerate(gen_ids):
                        g = gen_by_id[g_id]
                        key = (nid, d, s, h, g_id)
                        lin_cost = g['a_cost'] * alpha[key] + (g['b_cost'] + g['c_cost'] * g['Pmax_MW']) * p_gen[key]
                        total_cost += prob * op_weight * lin_cost
                    for g_idx, g_id in enumerate(gen_ids):
                        total_cost += prob * op_weight * g['startup_cost'] * gamma[(nid, d, s, h, g_id)]

    model.setObjective(total_cost, GRB.MINIMIZE)

    # Constraints

    # Commitment, bounds, startup
    M = 2.0
    for nid in tree_nodes:
        for d_idx, d in enumerate(day_ids):
            for s in range(num_lower_scens):
                for g_idx, g_id in enumerate(gen_ids):
                    g = gen_by_id[g_id]
                    is_cand = not g['existing']
                    pmax = g['Pmax_MW']
                    min_on = g.get('min_on_h', 1)
                    min_off = g.get('min_off_h', 1)
                    ramp = g.get('ramp_MW_per_h', pmax)
                    for h in range(hours_per_day):
                        key = (nid, d, s, h, g_id)
                        prev_key = (nid, d, s, h-1, g_id) if h > 0 else None

                        if is_cand:
                            model.addConstr(alpha[key] <= avail_gen[nid, g_id], name=f"avail_g_{nid}_{d}_{s}_{h}_{g_id}")
                        else:
                            model.addConstr(alpha[key] <= 1.0)

                        model.addConstr(p_gen[key] <= pmax * alpha[key])
                        model.addConstr(p_gen[key] >= g['Pmin_MW'] * alpha[key])
                        model.addConstr(s_res[key] <= pmax * alpha[key] - p_gen[key])
                        model.addConstr(s_res[key] >= 0)

                        if prev_key is not None:
                            model.addConstr(gamma[key] >= alpha[key] - alpha[prev_key])
                        else:
                            model.addConstr(gamma[key] >= alpha[key])

    # Min up / down time (simplified window enforcement)
    for nid in tree_nodes:
        for d_idx, d in enumerate(day_ids):
            for s in range(num_lower_scens):
                for g_idx, g_id in enumerate(gen_ids):
                    g = gen_by_id[g_id]
                    min_on = g.get('min_on_h', 1)
                    min_off = g.get('min_off_h', 1)
                    for h in range(hours_per_day):
                        key = (nid, d, s, h, g_id)
                        # Enforce min up
                        for k in range(1, min(min_on, hours_per_day - h)):
                            next_key = (nid, d, s, h + k, g_id)
                            model.addConstr(alpha[next_key] >= alpha[key] - (1.0 - gamma[key]) * M)
                        # Enforce min down (if turning off)
                        if h > 0:
                            prev_key = (nid, d, s, h-1, g_id)
                            model.addConstr(alpha[key] <= 1.0 - (alpha[prev_key] - gamma[prev_key]) * M)

    # Ramp constraints
    for nid in tree_nodes:
        for d_idx, d in enumerate(day_ids):
            for s in range(num_lower_scens):
                for g_idx, g_id in enumerate(gen_ids):
                    g = gen_by_id[g_id]
                    ramp = g.get('ramp_MW_per_h', 10000.0)
                    for h in range(1, hours_per_day):
                        key = (nid, d, s, h, g_id)
                        prev = (nid, d, s, h-1, g_id)
                        model.addConstr(p_gen[key] - p_gen[prev] <= ramp)
                        model.addConstr(p_gen[prev] - p_gen[key] <= ramp)

    # Spinning reserve - corrected bus indexing
    bus_gens = defaultdict(list)
    for g in gens:
        bus_gens[g['bus']].append(g['id'])

    for nid in tree_nodes:
        for d_idx, d_id in enumerate(day_ids):
            day_data = day_by_id[d_id]
            load_profile = day_data['load_profile']
            for s in range(num_lower_scens):
                lower_data = lower_scens[str(d_id)][s]
                demand_mult = lower_data['demand_multipliers']
                for h in range(hours_per_day):
                    for b in buses:
                        peak_load = loads.get(b, 0.0)
                        hourly_load = peak_load * load_profile[h] * demand_mult[h]
                        req = global_params['spinning_reserve_percent'] * hourly_load
                        model.addConstr(
                            gp.quicksum(s_res[(nid, d_id, s, h, gid)] for gid in bus_gens[b]) >= req,
                            name=f"spin_{nid}_{d_id}_{s}_{h}_{b}"
                        )

    # Storage dynamics
    eff = global_params.get('storage_efficiency', 0.9)
    for nid in tree_nodes:
        for d_idx, d_id in enumerate(day_ids):
            for s in range(num_lower_scens):
                for st_idx, st_id in enumerate(storage_ids):
                    st = next((s for s in storages if s['id'] == st_id), None)
                    is_cand = not st['existing']
                    cap = st['capacity_MW']
                    for h in range(hours_per_day):
                        key = (nid, d_id, s, h, st_id)
                        if h == 0:
                            model.addConstr(r_stor[key] == 0.0)
                        else:
                            prev_key = (nid, d_id, s, h-1, st_id)
                            model.addConstr(r_stor[key] == r_stor[prev_key] + eff * u_stor[prev_key] - v_stor[prev_key])
                            model.addConstr(v_stor[key] <= r_stor[prev_key])

                        avail = avail_stor[nid, st_id] if is_cand else 1.0
                        model.addConstr(u_stor[key] <= cap * avail)
                        model.addConstr(v_stor[key] <= cap * avail)
                        model.addConstr(r_stor[key] <= cap * avail)
                        model.addConstr(r_stor[key] >= 0.0)
                        model.addConstr(u_stor[key] >= 0.0)
                        model.addConstr(v_stor[key] >= 0.0)

    # Energy balance
    gen_at_bus = defaultdict(list)
    for g in gens:
        gen_at_bus[g['bus']].append(g['id'])
    stor_at_bus = defaultdict(list)
    for s in storages:
        stor_at_bus[s['bus']].append(s['id'])
    line_from = defaultdict(list)
    line_to = defaultdict(list)
    for ln in lines:
        line_from[ln['from_bus']].append(ln['id'])
        line_to[ln['to_bus']].append(ln['id'])

    for nid in tree_nodes:
        for d_idx, d_id in enumerate(day_ids):
            day_data = day_by_id[d_id]
            load_profile = day_data['load_profile']
            solar_profile = day_data.get('solar_profile', [0.0]*hours_per_day)
            for s in range(num_lower_scens):
                lower_data = lower_scens[str(d_id)][s]
                demand_mult = lower_data['demand_multipliers']
                solar_mult = lower_data.get('solar_multipliers', [1.0]*hours_per_day)
                wind_pf = lower_data.get('wind_power_factors', [0.0]*hours_per_day)
                for h in range(hours_per_day):
                    for b in buses:
                        # Demand
                        demand = loads.get(b, 0.0) * load_profile[h] * demand_mult[h]

                        # Renewables (using commitment as proxy for availability)
                        ren_out = 0.0
                        for g_id in gen_at_bus[b]:
                            g = gen_by_id[g_id]
                            if g['type'] == 'solar':
                                ren_out += g['Pmax_MW'] * solar_profile[h] * solar_mult[h] * alpha[(nid, d_id, s, h, g_id)]
                            elif g['type'] == 'wind':
                                ren_out += g['Pmax_MW'] * wind_pf[h] * alpha[(nid, d_id, s, h, g_id)]

                        # Net transmission
                        net_trans = 0.0
                        for ln_id in line_from[b]:
                            ln = next(l for l in lines if l['id'] == ln_id)
                            loss = ln.get('loss_factor', 0.0)
                            fwd_key = (nid, d_id, s, h, ln_id)
                            net_trans += f_fwd[fwd_key] * (1.0 + loss)
                        for ln_id in line_to[b]:
                            ln = next(l for l in lines if l['id'] == ln_id)
                            loss = ln.get('loss_factor', 0.0)
                            bwd_key = (nid, d_id, s, h, ln_id)
                            net_trans -= f_bwd[bwd_key] * (1.0 - loss)

                        # Storage
                        stor_net = gp.quicksum(v_stor[(nid, d_id, s, h, st)] - u_stor[(nid, d_id, s, h, st)]
                                               for st in stor_at_bus[b])

                        # Conventional generation
                        conv_gen = gp.quicksum(p_gen[(nid, d_id, s, h, gid)]
                                               for gid in gen_at_bus[b]
                                               if gen_by_id[gid]['type'] not in ('solar', 'wind'))

                        model.addConstr(demand + net_trans == ren_out + conv_gen + stor_net,
                                        name=f"bal_{nid}_{d_id}_{s}_{h}_{b}")

    # Transmission limits
    for nid in tree_nodes:
        for d_idx, d_id in enumerate(day_ids):
            for s in range(num_lower_scens):
                for h in range(hours_per_day):
                    for ln in lines:
                        ln_id = ln['id']
                        key = (nid, d_id, s, h, ln_id)
                        if ln['existing']:
                            limit = ln['flow_limit_MW']
                        else:
                            limit = avail_line[nid, ln_id] * ln['flow_limit_MW']
                        model.addConstr(f_fwd[key] <= limit)
                        model.addConstr(f_bwd[key] <= limit)

    # Reserve margin (per node, using peak at that stage)
    for nid in tree_nodes:
        stage = tree_nodes[nid]['stage']
        growth_factor = 1.0 + stage * max(scenario_params.get('demand_growth_high_percent', 0.1),
                                          scenario_params.get('demand_growth_low_percent', 0.05))
        peak_demand = sum(loads.get(b, 0.0) for b in buses) * growth_factor
        req_cap = (1.0 + global_params.get('reserve_margin_requirement', 0.15)) * peak_demand

        derated = 0.0
        for g in gens:
            derate = global_params['derating_factors'].get(g['type'], 0.9)
            if g['existing']:
                derated += g['Pmax_MW'] * derate
            else:
                derated += avail_gen[nid, g['id']] * g['Pmax_MW'] * derate
        model.addConstr(derated >= req_cap, name=f"rm_{nid}")

    # Callback
    best_obj = float('inf')
    solution_dict_cache = None

    def callback(model, where):
        nonlocal best_obj, solution_dict_cache
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj < best_obj - 1e-6:
                best_obj = obj
                sol = {}
                sol['objective_value'] = float(obj)
                sol['investments'] = {}
                for nid_val in tree_nodes:
                    nid_str = str(nid_val)
                    sol['investments'][nid_str] = {'generators': {}, 'lines': {}, 'storage': {}}
                    for g in candidate_gens:
                        val = model.cbGetSolution(build_gen[nid_val, g['id']])
                        sol['investments'][nid_str]['generators'][g['id']] = 1 if val > 0.5 else 0
                    for ln_ in candidate_lines:
                        val = model.cbGetSolution(build_line[nid_val, ln_['id']])
                        sol['investments'][nid_str]['lines'][ln_['id']] = 1 if val > 0.5 else 0
                    for st_ in candidate_stor:
                        val = model.cbGetSolution(build_stor[nid_val, st_['id']])
                        sol['investments'][nid_str]['storage'][st_['id']] = 1 if val > 0.5 else 0

                # Operational variables (only root for size)
                root_n = root_id
                sol['alpha'] = {}
                sol['gamma'] = {}
                sol['p_gen'] = {}
                sol['s_res'] = {}
                sol['f_fwd'] = {}
                sol['f_bwd'] = {}
                sol['u_stor'] = {}
                sol['v_stor'] = {}
                sol['r_stor'] = {}

                for d_id in day_ids:
                    for sc in range(num_lower_scens):
                        for h in range(hours_per_day):
                            for g_id in gen_ids:
                                k = f"{root_n},{d_id},{sc},{h},{g_id}"
                                sol['alpha'][k] = float(model.cbGetSolution(alpha[(root_n, d_id, sc, h, g_id)]))
                                sol['gamma'][k] = float(model.cbGetSolution(gamma[(root_n, d_id, sc, h, g_id)]))
                                sol['p_gen'][k] = float(model.cbGetSolution(p_gen[(root_n, d_id, sc, h, g_id)]))
                                sol['s_res'][k] = float(model.cbGetSolution(s_res[(root_n, d_id, sc, h, g_id)]))

                for d_id in day_ids:
                    for sc in range(num_lower_scens):
                        for h in range(hours_per_day):
                            for ln_id in line_ids:
                                k = f"{root_n},{d_id},{sc},{h},{ln_id}"
                                sol['f_fwd'][k] = float(model.cbGetSolution(f_fwd[(root_n, d_id, sc, h, ln_id)]))
                                sol['f_bwd'][k] = float(model.cbGetSolution(f_bwd[(root_n, d_id, sc, h, ln_id)]))

                for d_id in day_ids:
                    for sc in range(num_lower_scens):
                        for h in range(hours_per_day):
                            for st_id in storage_ids:
                                k = f"{root_n},{d_id},{sc},{h},{st_id}"
                                sol['u_stor'][k] = float(model.cbGetSolution(u_stor[(root_n, d_id, sc, h, st_id)]))
                                sol['v_stor'][k] = float(model.cbGetSolution(v_stor[(root_n, d_id, sc, h, st_id)]))
                                sol['r_stor'][k] = float(model.cbGetSolution(r_stor[(root_n, d_id, sc, h, st_id)]))

                if logger:
                    logger.log_solution(obj, sol)
                solution_dict_cache = sol

    model.optimize(callback)

    # Final solution
    if model.SolCount > 0:
        final_obj = model.ObjVal
        sol = {}
        sol['objective_value'] = float(final_obj)
        sol['investments'] = {}
        for nid_val in tree_nodes:
            nid_str = str(nid_val)
            sol['investments'][nid_str] = {'generators': {}, 'lines': {}, 'storage': {}}
            for g in candidate_gens:
                val = avail_gen[nid_val, g['id']].getValue()
                built = 1 if val > 0.5 else 0
                sol['investments'][nid_str]['generators'][g['id']] = built
            for ln_ in candidate_lines:
                val = avail_line[nid_val, ln_['id']].getValue()
                built = 1 if val > 0.5 else 0
                sol['investments'][nid_str]['lines'][ln_['id']] = built
            for st_ in candidate_stor:
                val = avail_stor[nid_val, st_['id']].getValue()
                built = 1 if val > 0.5 else 0
                sol['investments'][nid_str]['storage'][st_['id']] = built

        root_n = root_id
        sol['alpha'] = {}
        sol['gamma'] = {}
        sol['p_gen'] = {}
        sol['s_res'] = {}
        sol['f_fwd'] = {}
        sol['f_bwd'] = {}
        sol['u_stor'] = {}
        sol['v_stor'] = {}
        sol['r_stor'] = {}

        for d_id in day_ids:
            for sc in range(num_lower_scens):
                for h in range(hours_per_day):
                    for g_id in gen_ids:
                        k = f"{root_n},{d_id},{sc},{h},{g_id}"
                        sol['alpha'][k] = float(alpha[(root_n, d_id, sc, h, g_id)].X)
                        sol['gamma'][k] = float(gamma[(root_n, d_id, sc, h, g_id)].X)
                        sol['p_gen'][k] = float(p_gen[(root_n, d_id, sc, h, g_id)].X)
                        sol['s_res'][k] = float(s_res[(root_n, d_id, sc, h, g_id)].X)

        for d_id in day_ids:
            for sc in range(num_lower_scens):
                for h in range(hours_per_day):
                    for ln_id in line_ids:
                        k = f"{root_n},{d_id},{sc},{h},{ln_id}"
                        sol['f_fwd'][k] = float(f_fwd[(root_n, d_id, sc, h, ln_id)].X)
                        sol['f_bwd'][k] = float(f_bwd[(root_n, d_id, sc, h, ln_id)].X)

        for d_id in day_ids:
            for sc in range(num_lower_scens):
                for h in range(hours_per_day):
                    for st_id in storage_ids:
                        k = f"{root_n},{d_id},{sc},{h},{st_id}"
                        sol['u_stor'][k] = float(u_stor[(root_n, d_id, sc, h, st_id)].X)
                        sol['v_stor'][k] = float(v_stor[(root_n, d_id, sc, h, st_id)].X)
                        sol['r_stor'][k] = float(r_stor[(root_n, d_id, sc, h, st_id)].X)

        if logger and solution_dict_cache is None:
            logger.log_solution(final_obj, sol)

        with open(args.solution_path, 'w') as f:
            json.dump(sol, f, indent=2)
    else:
        fallback = {
            "objective_value": 1e10,
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
        for nid_val in tree_nodes:
            nid_str = str(nid_val)
            fallback['investments'][nid_str] = {
                'generators': {g['id']: 0 for g in candidate_gens},
                'lines': {ln['id']: 0 for ln in candidate_lines},
                'storage': {st['id']: 0 for st in candidate_stor}
            }
        with open(args.solution_path, 'w') as f:
            json.dump(fallback, f, indent=2)


if __name__ == "__main__":
    main()