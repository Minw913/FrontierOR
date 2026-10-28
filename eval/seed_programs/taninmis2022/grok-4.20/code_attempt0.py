import json
import argparse
import time
import sys
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import numpy as np
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
    model.setParam(GRB.Param.TimeLimit, args.time_limit - 10)  # buffer

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
    # alpha: commitment
    alpha = {}
    gamma = {}  # startup
    p_gen = {}
    s_res = {}
    for nid in tree_nodes:
        for d in day_ids:
            for s in range(num_lower_scens):
                for h in range(hours_per_day):
                    for g in gen_ids:
                        key = (nid, d, s, h, g)
                        alpha[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"alpha_{nid}_{d}_{s}_{h}_{g}")
                        gamma[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"gamma_{nid}_{d}_{s}_{h}_{g}")
                        p_gen[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"p_{nid}_{d}_{s}_{h}_{g}")
                        s_res[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"sr_{nid}_{d}_{s}_{h}_{g}")

    # Transmission
    f_fwd = {}
    f_bwd = {}
    for nid in tree_nodes:
        for d in day_ids:
            for s in range(num_lower_scens):
                for h in range(hours_per_day):
                    for ln in line_ids:
                        key = (nid, d, s, h, ln)
                        f_fwd[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"fwd_{nid}_{d}_{s}_{h}_{ln}")
                        f_bwd[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"bwd_{nid}_{d}_{s}_{h}_{ln}")

    # Storage
    u_stor = {}  # charge
    v_stor = {}  # discharge
    r_stor = {}  # SOC
    for nid in tree_nodes:
        for d in day_ids:
            for s in range(num_lower_scens):
                for h in range(hours_per_day):
                    for st in storage_ids:
                        key = (nid, d, s, h, st)
                        u_stor[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"u_{nid}_{d}_{s}_{h}_{st}")
                        v_stor[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"v_{nid}_{d}_{s}_{h}_{st}")
                        r_stor[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"r_{nid}_{d}_{s}_{h}_{st}")

    # Objective
    total_cost = 0.0

    # Investment cost (annualized over investment period, but here we use direct as per benchmark)
    for nid in tree_nodes:
        prob = node_prob[nid]
        node_stage = tree_nodes[nid]['stage']
        # Capital costs at this node
        invest_cost = 0.0
        for g in candidate_gens:
            invest_cost += build_gen[nid, g['id']] * g['capital_cost_per_kW'] * g['Pmax_MW'] * 1000  # kW to MW
        for ln in candidate_lines:
            invest_cost += build_line[nid, ln['id']] * ln['capital_cost_per_kW'] * ln['flow_limit_MW'] * 1000
        for st in candidate_stor:
            invest_cost += build_stor[nid, st['id']] * st['capital_cost_per_kW'] * st['capacity_MW'] * 1000
        total_cost += prob * invest_cost

    # Operating costs
    op_weight = 1.0 / num_rep_days / num_lower_scens * 365 * investment_years  # approximate annualization factor
    for nid in tree_nodes:
        prob = node_prob[nid]
        for d_idx, d in enumerate(day_ids):
            day_data = day_by_id[d]
            for scen in range(num_lower_scens):
                lower_data = lower_scens[str(d)][scen]
                for h in range(hours_per_day):
                    for g_id in gen_ids:
                        g = gen_by_id[g_id]
                        key = (nid, d, scen, h, g_id)
                        # linearized cost: a * alpha + (b + c * Pmax) * p
                        lin_cost = g['a_cost'] * alpha[key] + (g['b_cost'] + g['c_cost'] * g['Pmax_MW']) * p_gen[key]
                        total_cost += prob * op_weight * lin_cost
                    total_cost += prob * op_weight * gp.quicksum(g['startup_cost'] * gamma[(nid, d, scen, h, g_id)] for g_id in gen_ids)

    model.setObjective(total_cost, GRB.MINIMIZE)

    # Constraints

    # 1. Commitment logic and min up/down (simplified due to time; use big-M style with rolling windows)
    M = 1.0
    for nid in tree_nodes:
        for d in day_ids:
            for s in range(num_lower_scens):
                for g_idx, g_id in enumerate(gen_ids):
                    g = gen_by_id[g_id]
                    is_candidate = not g['existing']
                    pmax = g['Pmax_MW']
                    min_on = g['min_on_h']
                    min_off = g['min_off_h']
                    for h in range(hours_per_day):
                        key = (nid, d, s, h, g_id)
                        prev_key = (nid, d, s, h-1, g_id) if h > 0 else None

                        # Commitment bounded by availability
                        if is_candidate:
                            model.addConstr(alpha[key] <= avail_gen[nid, g_id])
                        else:
                            model.addConstr(alpha[key] <= 1.0)

                        # Power bounds
                        model.addConstr(p_gen[key] <= pmax * alpha[key])
                        model.addConstr(p_gen[key] >= g['Pmin_MW'] * alpha[key])
                        model.addConstr(s_res[key] <= pmax * alpha[key] - p_gen[key])

                        # Startup
                        if prev_key:
                            model.addConstr(gamma[key] >= alpha[key] - alpha[prev_key])
                        else:
                            model.addConstr(gamma[key] >= alpha[key])  # first hour

                        # Min up time (approximate with window)
                        for k in range(1, min(min_on, hours_per_day - h)):
                            model.addConstr(alpha[(nid, d, s, h+k, g_id)] >= alpha[key] - (1 - gamma[key]) * M)

                        # Min down time (approximate)
                        for k in range(1, min(min_off, hours_per_day - h)):
                            model.addConstr(alpha[(nid, d, s, h+k, g_id)] <= 1 - (alpha[key] - gamma[key]))

    # 2. Ramp constraints
    for nid in tree_nodes:
        for d in day_ids:
            for s in range(num_lower_scens):
                for g_id in gen_ids:
                    g = gen_by_id[g_id]
                    ramp = g['ramp_MW_per_h']
                    for h in range(1, hours_per_day):
                        key = (nid, d, s, h, g_id)
                        prev = (nid, d, s, h-1, g_id)
                        model.addConstr(p_gen[key] - p_gen[prev] <= ramp)
                        model.addConstr(p_gen[prev] - p_gen[key] <= ramp)

    # 3. Spinning reserve
    bus_gens = defaultdict(list)
    for g in gens:
        bus_gens[g['bus']].append(g['id'])

    for nid in tree_nodes:
        for d in day_ids:
            for s in range(num_lower_scens):
                for h in range(hours_per_day):
                    for b in buses:
                        req = global_params['spinning_reserve_percent'] * loads[b] * day_by_id[d]['load_profile'][h] * \
                              lower_scens[str(d)][s]['demand_multipliers'][h]  # approximate peak scaling
                        model.addConstr(gp.quicksum(s_res[(nid,d,s,h,gid)] for gid in bus_gens[b]) >= req)

    # 4. Storage dynamics
    eff = global_params['storage_efficiency']
    for nid in tree_nodes:
        for d in day_ids:
            for s in range(num_lower_scens):
                for st_idx, st_id in enumerate(storage_ids):
                    st = next(s for s in storages if s['id'] == st_id)
                    is_cand = not st['existing']
                    cap = st['capacity_MW']
                    for h in range(hours_per_day):
                        key = (nid, d, s, h, st_id)
                        if h == 0:
                            model.addConstr(r_stor[key] == 0)  # start of day SOC=0
                            model.addConstr(u_stor[key] <= cap * (avail_stor[nid, st_id] if is_cand else 1.0))
                            model.addConstr(v_stor[key] <= cap * (avail_stor[nid, st_id] if is_cand else 1.0))
                        else:
                            prev_key = (nid, d, s, h-1, st_id)
                            model.addConstr(r_stor[key] == r_stor[prev_key] + eff * u_stor[prev_key] - v_stor[prev_key])
                            model.addConstr(r_stor[key] <= cap * (avail_stor[nid, st_id] if is_cand else 1.0))
                            model.addConstr(v_stor[key] <= r_stor[prev_key])
                            model.addConstr(u_stor[key] <= cap * (avail_stor[nid, st_id] if is_cand else 1.0))
                            model.addConstr(v_stor[key] <= cap * (avail_stor[nid, st_id] if is_cand else 1.0))

    # 5. Energy balance
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
        for d in day_ids:
            day_data = day_by_id[d]
            load_profile = day_data['load_profile']
            solar_profile = day_data['solar_profile']
            for s in range(num_lower_scens):
                lower_data = lower_scens[str(d)][s]
                demand_mult = lower_data['demand_multipliers']
                solar_mult = lower_data['solar_multipliers']
                wind_pf = lower_data['wind_power_factors']
                for h in range(hours_per_day):
                    for b in buses:
                        # Compute renewable output
                        ren_out = 0.0
                        for g_id in gen_at_bus[b]:
                            g = gen_by_id[g_id]
                            if g['type'] == 'solar':
                                ren_out += g['Pmax_MW'] * solar_profile[h] * solar_mult[h] * alpha[(nid,d,s,h,g_id)]
                            elif g['type'] == 'wind':
                                ren_out += g['Pmax_MW'] * wind_pf[h] * alpha[(nid,d,s,h,g_id)]

                        lhs = loads[b] * load_profile[h] * demand_mult[h]  # demand
                        # outflows
                        for ln_id in line_from[b]:
                            ln = next(l for l in lines if l['id'] == ln_id)
                            loss = ln['loss_factor']
                            fwd_key = (nid, d, s, h, ln_id)
                            lhs += f_fwd[fwd_key] * (1 + loss)  # approx loss on sending
                        # inflows
                        for ln_id in line_to[b]:
                            ln = next(l for l in lines if l['id'] == ln_id)
                            loss = ln['loss_factor']
                            bwd_key = (nid, d, s, h, ln_id)
                            lhs -= f_bwd[bwd_key] * (1 - loss)

                        rhs = ren_out + gp.quicksum(p_gen[(nid,d,s,h,gid)] for gid in gen_at_bus[b] if gen_by_id[gid]['type'] not in ['solar','wind'])
                        rhs += gp.quicksum(v_stor[(nid,d,s,h,st)] - u_stor[(nid,d,s,h,st)] for st in stor_at_bus[b])

                        model.addConstr(lhs == rhs, name=f"balance_{nid}_{d}_{s}_{h}_{b}")

    # 6. Transmission limits
    for nid in tree_nodes:
        for d in day_ids:
            for s in range(num_lower_scens):
                for h in range(hours_per_day):
                    for ln in lines:
                        ln_id = ln['id']
                        key = (nid, d, s, h, ln_id)
                        if ln['existing']:
                            limit = ln['flow_limit_MW']
                        else:
                            limit = avail_line[nid, ln_id] * ln['flow_limit_MW']
                        model.addConstr(f_fwd[key] <= limit)
                        model.addConstr(f_bwd[key] <= limit)

    # 7. Reserve margin (only at leaves or per node)
    for nid in tree_nodes:
        peak_demand = sum(loads[b] for b in buses) * (1 + tree_nodes[nid]['stage'] * 0.1)  # rough growth
        req_cap = (1 + global_params['reserve_margin_requirement']) * peak_demand
        derated_cap = 0.0
        for g in gens:
            if g['existing']:
                derated_cap += g['Pmax_MW'] * global_params['derating_factors'].get(g['type'], 0.9)
            else:
                derated_cap += avail_gen[nid, g['id']] * g['Pmax_MW'] * global_params['derating_factors'].get(g['type'], 0.9)
        model.addConstr(derated_cap >= req_cap, name=f"reserve_margin_{nid}")

    # Callback for incumbent logging
    best_obj = float('inf')
    solution_dict_cache = None

    def callback(model, where):
        nonlocal best_obj, solution_dict_cache
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj < best_obj - 1e-6:
                best_obj = obj
                # Extract solution
                sol = {}
                sol['objective_value'] = float(obj)
                sol['investments'] = {}
                for nid in tree_nodes:
                    sol['investments'][str(nid)] = {
                        'generators': {},
                        'lines': {},
                        'storage': {}
                    }
                    for g in candidate_gens:
                        val = model.cbGetSolution(build_gen[nid, g['id']])
                        sol['investments'][str(nid)]['generators'][g['id']] = 1 if val > 0.5 else 0
                    for ln in candidate_lines:
                        val = model.cbGetSolution(build_line[nid, ln['id']])
                        sol['investments'][str(nid)]['lines'][ln['id']] = 1 if val > 0.5 else 0
                    for st in candidate_stor:
                        val = model.cbGetSolution(build_stor[nid, st['id']])
                        sol['investments'][str(nid)]['storage'][st['id']] = 1 if val > 0.5 else 0

                # Extract operational variables (only for root node to save time/space)
                root_nid = root_id
                sol['alpha'] = {}
                sol['gamma'] = {}
                sol['p_gen'] = {}
                sol['s_res'] = {}
                sol['f_fwd'] = {}
                sol['f_bwd'] = {}
                sol['u_stor'] = {}
                sol['v_stor'] = {}
                sol['r_stor'] = {}

                for d in day_ids:
                    for sc in range(num_lower_scens):
                        for h in range(hours_per_day):
                            for g_id in gen_ids:
                                k = f"{root_nid},{d},{sc},{h},{g_id}"
                                sol['alpha'][k] = float(model.cbGetSolution(alpha[(root_nid,d,sc,h,g_id)]))
                                sol['gamma'][k] = float(model.cbGetSolution(gamma[(root_nid,d,sc,h,g_id)]))
                                sol['p_gen'][k] = float(model.cbGetSolution(p_gen[(root_nid,d,sc,h,g_id)]))
                                sol['s_res'][k] = float(model.cbGetSolution(s_res[(root_nid,d,sc,h,g_id)]))

                for d in day_ids:
                    for sc in range(num_lower_scens):
                        for h in range(hours_per_day):
                            for ln_id in line_ids:
                                k = f"{root_nid},{d},{sc},{h},{ln_id}"
                                sol['f_fwd'][k] = float(model.cbGetSolution(f_fwd[(root_nid,d,sc,h,ln_id)]))
                                sol['f_bwd'][k] = float(model.cbGetSolution(f_bwd[(root_nid,d,sc,h,ln_id)]))

                for d in day_ids:
                    for sc in range(num_lower_scens):
                        for h in range(hours_per_day):
                            for st_id in storage_ids:
                                k = f"{root_nid},{d},{sc},{h},{st_id}"
                                sol['u_stor'][k] = float(model.cbGetSolution(u_stor[(root_nid,d,sc,h,st_id)]))
                                sol['v_stor'][k] = float(model.cbGetSolution(v_stor[(root_nid,d,sc,h,st_id)]))
                                sol['r_stor'][k] = float(model.cbGetSolution(r_stor[(root_nid,d,sc,h,st_id)]))

                if logger:
                    logger.log_solution(obj, sol)
                solution_dict_cache = sol

    model.optimize(callback)

    # Final solution extraction
    if model.SolCount > 0:
        final_obj = model.ObjVal
        sol = {}
        sol['objective_value'] = float(final_obj)
        sol['investments'] = {}
        for nid_val in tree_nodes:
            nid = str(nid_val)
            sol['investments'][nid] = {
                'generators': {},
                'lines': {},
                'storage': {}
            }
            for g in candidate_gens:
                val = avail_gen[nid_val, g['id']].getValue()  # use final available
                built = 1 if val > 0.5 else 0
                sol['investments'][nid]['generators'][g['id']] = built
            for ln in candidate_lines:
                val = avail_line[nid_val, ln['id']].getValue()
                built = 1 if val > 0.5 else 0
                sol['investments'][nid]['lines'][ln['id']] = built
            for st in candidate_stor:
                val = avail_stor[nid_val, st['id']].getValue()
                built = 1 if val > 0.5 else 0
                sol['investments'][nid]['storage'][st['id']] = built

        # Fill operational variables from best solution (root only for practicality)
        root_nid = root_id
        sol['alpha'] = {}
        sol['gamma'] = {}
        sol['p_gen'] = {}
        sol['s_res'] = {}
        sol['f_fwd'] = {}
        sol['f_bwd'] = {}
        sol['u_stor'] = {}
        sol['v_stor'] = {}
        sol['r_stor'] = {}

        for d in day_ids:
            for sc in range(num_lower_scens):
                for h in range(hours_per_day):
                    for g_id in gen_ids:
                        k = f"{root_nid},{d},{sc},{h},{g_id}"
                        sol['alpha'][k] = float(alpha[(root_nid,d,sc,h,g_id)].X)
                        sol['gamma'][k] = float(gamma[(root_nid,d,sc,h,g_id)].X)
                        sol['p_gen'][k] = float(p_gen[(root_nid,d,sc,h,g_id)].X)
                        sol['s_res'][k] = float(s_res[(root_nid,d,sc,h,g_id)].X)

        for d in day_ids:
            for sc in range(num_lower_scens):
                for h in range(hours_per_day):
                    for ln_id in line_ids:
                        k = f"{root_nid},{d},{sc},{h},{ln_id}"
                        sol['f_fwd'][k] = float(f_fwd[(root_nid,d,sc,h,ln_id)].X)
                        sol['f_bwd'][k] = float(f_bwd[(root_nid,d,sc,h,ln_id)].X)

        for d in day_ids:
            for sc in range(num_lower_scens):
                for h in range(hours_per_day):
                    for st_id in storage_ids:
                        k = f"{root_nid},{d},{sc},{h},{st_id}"
                        sol['u_stor'][k] = float(u_stor[(root_nid,d,sc,h,st_id)].X)
                        sol['v_stor'][k] = float(v_stor[(root_nid,d,sc,h,st_id)].X)
                        sol['r_stor'][k] = float(r_stor[(root_nid,d,sc,h,st_id)].X)

        if logger and solution_dict_cache is None:
            logger.log_solution(final_obj, sol)

        with open(args.solution_path, 'w') as f:
            json.dump(sol, f, indent=2)
    else:
        # No solution found
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
            "investments": {str(nid): {"generators": {}, "lines": {}, "storage": {}} for nid in tree_nodes}
        }
        for nid in tree_nodes:
            for g in candidate_gens:
                fallback['investments'][str(nid)]['generators'][g['id']] = 0
            for ln in candidate_lines:
                fallback['investments'][str(nid)]['lines'][ln['id']] = 0
            for st in candidate_stor:
                fallback['investments'][str(nid)]['storage'][st['id']] = 0
        with open(args.solution_path, 'w') as f:
            json.dump(fallback, f, indent=2)

if __name__ == "__main__":
    main()