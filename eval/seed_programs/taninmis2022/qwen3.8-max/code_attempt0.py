import argparse
import json
import time
import sys
import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        inst = json.load(f)

    system = inst['system']
    sp = inst['scenario_parameters']
    tree_nodes = inst['scenario_tree']
    rep_days = inst['representative_days']
    lower_scen = inst['lower_level_scenarios']
    gparams = inst['global_parameters']

    buses = system['buses']
    generators = system['generators']
    lines = system['transmission_lines']
    storage_units = system['storage']
    loads = system['loads']

    H = sp['hours_per_day']
    D = sp['num_representative_days']
    W = sp['num_lower_level_scenarios_per_day']
    inv_years = sp['investment_period_years']

    # Index sets
    gen_ids = [g['id'] for g in generators]
    line_ids = [l['id'] for l in lines]
    stor_ids = [s['id'] for s in storage_units]
    node_ids = [n['id'] for n in tree_nodes]

    # Generator lookup
    gen_by_id = {g['id']: g for g in generators}
    line_by_id = {l['id']: l for l in lines}
    stor_by_id = {s['id']: s for s in storage_units}

    existing_gens = [g for g in generators if g['existing']]
    candidate_gens = [g for g in generators if not g['existing']]
    existing_lines = [l for l in lines if l['existing']]
    candidate_lines = [l for l in lines if not l['existing']]
    existing_stor = [s for s in storage_units if s['existing']]
    candidate_stor = [s for s in storage_units if not s['existing']]

    # Build tree structure
    nodes_dict = {n['id']: n for n in tree_nodes}
    root_id = None
    for n in tree_nodes:
        if n['parent_id'] is None:
            root_id = n['id']
            break

    # Compute demand and cost multipliers via BFS
    demand_mult = {}
    cost_mult = {}

    def compute_mults(nid, dm, cm):
        demand_mult[nid] = dm
        cost_mult[nid] = cm
        node = nodes_dict[nid]
        children = node['children']
        bf = sp['upper_level_branching_factor']
        for i, cid in enumerate(children):
            if bf <= 1:
                new_dm = dm * (1 + (sp['demand_growth_high_percent'] + sp['demand_growth_low_percent']) / 2)
                new_cm = cm
            elif bf == 2:
                if i == 0:
                    new_dm = dm * (1 + sp['demand_growth_high_percent'])
                    new_cm = cm * (1 + sp['cost_variation_percent'])
                else:
                    new_dm = dm * (1 + sp['demand_growth_low_percent'])
                    new_cm = cm * (1 - sp['cost_variation_percent'])
            else:
                frac = i / (bf - 1)
                growth = sp['demand_growth_low_percent'] + frac * (sp['demand_growth_high_percent'] - sp['demand_growth_low_percent'])
                cvar = -sp['cost_variation_percent'] + frac * 2 * sp['cost_variation_percent']
                new_dm = dm * (1 + growth)
                new_cm = cm * (1 + cvar)
            compute_mults(cid, new_dm, new_cm)

    compute_mults(root_id, 1.0, 1.0)

    # Peak demand at each (node, bus)
    peak_demand = {}
    for nid in node_ids:
        for load in loads:
            peak_demand[(nid, load['bus'])] = load['annual_peak_load_MW'] * demand_mult.get(nid, 1.0)

    # Total peak demand per node
    total_peak = {}
    for nid in node_ids:
        total_peak[nid] = sum(peak_demand.get((nid, b), 0.0) for b in buses)

    # Derating factors
    derating = gparams.get('derating_factors', {})

    def get_derating(gen):
        gtype = gen.get('type', 'fossil').lower()
        if 'wind' in gtype:
            return derating.get('wind', 0.1)
        elif 'solar' in gtype:
            return derating.get('solar', 0.2)
        else:
            return derating.get('fossil', 0.9)

    # Precompute demand for each (node, day, scenario, hour, bus)
    # demand_val[nid][d][w][h][bus]
    demand_val = {}
    for nid in node_ids:
        demand_val[nid] = {}
        for d_idx, rd in enumerate(rep_days):
            demand_val[nid][d_idx] = {}
            day_key = str(rd['id'])
            scenarios_for_day = lower_scen.get(day_key, [])
            load_profile = rd['load_profile']
            for w_idx, scen in enumerate(scenarios_for_day):
                demand_val[nid][d_idx][w_idx] = {}
                dm = scen['demand_multipliers']
                for h in range(H):
                    demand_val[nid][d_idx][w_idx][h] = {}
                    for load in loads:
                        bus = load['bus']
                        base = peak_demand.get((nid, bus), 0.0)
                        demand_val[nid][d_idx][w_idx][h][bus] = base * load_profile[h] * dm[h]

    # Precompute renewable capacity factors
    # solar_cf[nid][d][w][h] for solar generators
    # wind_pf[nid][d][w][h] for wind generators
    solar_cf = {}
    wind_pf = {}
    for d_idx, rd in enumerate(rep_days):
        day_key = str(rd['id'])
        scenarios_for_day = lower_scen.get(day_key, [])
        solar_profile = rd['solar_profile']
        solar_cf[d_idx] = {}
        wind_pf[d_idx] = {}
        for w_idx, scen in enumerate(scenarios_for_day):
            solar_cf[d_idx][w_idx] = {}
            wind_pf[d_idx][w_idx] = {}
            sm = scen['solar_multipliers']
            wf = scen['wind_power_factors']
            for h in range(H):
                solar_cf[d_idx][w_idx][h] = min(solar_profile[h] * sm[h], 1.0)
                wind_pf[d_idx][w_idx][h] = min(wf[h], 1.0)

    # Determine which generators are solar/wind
    def is_solar(gen):
        return 'solar' in gen.get('type', '').lower()

    def is_wind(gen):
        return 'wind' in gen.get('type', '').lower()

    # Build Gurobi model
    model = gp.Model("capacity_expansion")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = args.time_limit - 5  # Leave some buffer
    model.Params.OutputFlag = 0

    # Decision variables
    # Investment variables
    x_gen = {}
    for s in node_ids:
        for g in candidate_gens:
            x_gen[(s, g['id'])] = model.addVar(vtype=GRB.BINARY, name=f"xg_{s}_{g['id']}")

    x_line = {}
    for s in node_ids:
        for l in candidate_lines:
            x_line[(s, l['id'])] = model.addVar(vtype=GRB.BINARY, name=f"xl_{s}_{l['id']}")

    x_stor = {}
    for s in node_ids:
        for st in candidate_stor:
            x_stor[(s, st['id'])] = model.addVar(vtype=GRB.BINARY, name=f"xs_{s}_{st['id']}")

    # Available capacity tracking
    # For generators: avail_gen[s][g] = sum of x_gen on path from root to s
    # We'll compute this via constraints
    # built_gen[s,g] = 1 if built by node s (binary derived)
    built_gen = {}
    for s in node_ids:
        for g in candidate_gens:
            built_gen[(s, g['id'])] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"bg_{s}_{g['id']}")

    built_line = {}
    for s in node_ids:
        for l in candidate_lines:
            built_line[(s, l['id'])] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"bl_{s}_{l['id']}")

    built_stor = {}
    for s in node_ids:
        for st in candidate_stor:
            built_stor[(s, st['id'])] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"bs_{s}_{st['id']}")

    # Linking constraints for built variables
    for s in node_ids:
        node = nodes_dict[s]
        parent = node['parent_id']
        for g in candidate_gens:
            gid = g['id']
            if parent is None:
                model.addConstr(built_gen[(s, gid)] == x_gen[(s, gid)], name=f"build_root_{s}_{gid}")
            else:
                model.addConstr(built_gen[(s, gid)] == built_gen[(parent, gid)] + x_gen[(s, gid)],
                                name=f"build_link_{s}_{gid}")

        for l in candidate_lines:
            lid = l['id']
            if parent is None:
                model.addConstr(built_line[(s, lid)] == x_line[(s, lid)], name=f"buildl_root_{s}_{lid}")
            else:
                model.addConstr(built_line[(s, lid)] == built_line[(parent, lid)] + x_line[(s, lid)],
                                name=f"buildl_link_{s}_{lid}")

        for st in candidate_stor:
            sid = st['id']
            if parent is None:
                model.addConstr(built_stor[(s, sid)] == x_stor[(s, sid)], name=f"builds_root_{s}_{sid}")
            else:
                model.addConstr(built_stor[(s, sid)] == built_stor[(parent, sid)] + x_stor[(s, sid)],
                                name=f"builds_link_{s}_{sid}")

    # At most once constraint (implicitly handled by built ≤ 1, but let's be explicit)
    # built_gen ≤ 1 is already enforced by ub=1.0

    # Operational variables
    # Use compact indexing: (s, d, w, h, unit_id)
    alpha = {}  # commitment
    gamma = {}  # startup
    p_gen = {}  # dispatch
    s_res = {}  # spinning reserve

    for s in node_ids:
        for d_idx in range(D):
            day_key = str(rep_days[d_idx]['id'])
            num_scen = len(lower_scen.get(day_key, []))
            for w_idx in range(num_scen):
                for h in range(H):
                    for g in generators:
                        key = (s, d_idx, w_idx, h, g['id'])
                        alpha[key] = model.addVar(vtype=GRB.BINARY, name=f"a_{s}_{d_idx}_{w_idx}_{h}_{g['id']}")
                        gamma[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"g_{s}_{d_idx}_{w_idx}_{h}_{g['id']}")
                        p_gen[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"p_{s}_{d_idx}_{w_idx}_{h}_{g['id']}")
                        s_res[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"r_{s}_{d_idx}_{w_idx}_{h}_{g['id']}")

    # Transmission flow variables
    f_fwd = {}
    f_bwd = {}
    for s in node_ids:
        for d_idx in range(D):
            day_key = str(rep_days[d_idx]['id'])
            num_scen = len(lower_scen.get(day_key, []))
            for w_idx in range(num_scen):
                for h in range(H):
                    for l in lines:
                        key = (s, d_idx, w_idx, h, l['id'])
                        f_fwd[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"ff_{s}_{d_idx}_{w_idx}_{h}_{l['id']}")
                        f_bwd[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"fb_{s}_{d_idx}_{w_idx}_{h}_{l['id']}")

    # Storage variables
    u_stor = {}  # charging
    v_stor = {}  # discharging
    r_stor = {}  # state of charge
    for s in node_ids:
        for d_idx in range(D):
            day_key = str(rep_days[d_idx]['id'])
            num_scen = len(lower_scen.get(day_key, []))
            for w_idx in range(num_scen):
                for h in range(H):
                    for st in storage_units:
                        key = (s, d_idx, w_idx, h, st['id'])
                        u_stor[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"u_{s}_{d_idx}_{w_idx}_{h}_{st['id']}")
                        v_stor[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"v_{s}_{d_idx}_{w_idx}_{h}_{st['id']}")
                        r_stor[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"rs_{s}_{d_idx}_{w_idx}_{h}_{st['id']}")

    model.update()

    # Constraints
    # 1. Startup definition: gamma[h] >= alpha[h] - alpha[h-1]
    for s in node_ids:
        for d_idx in range(D):
            day_key = str(rep_days[d_idx]['id'])
            num_scen = len(lower_scen.get(day_key, []))
            for w_idx in range(num_scen):
                for g in generators:
                    gid = g['id']
                    for h in range(H):
                        key = (s, d_idx, w_idx, h, gid)
                        if h == 0:
                            model.addConstr(gamma[key] >= alpha[key], name=f"startup_def_{s}_{d_idx}_{w_idx}_{h}_{gid}")
                        else:
                            prev_key = (s, d_idx, w_idx, h - 1, gid)
                            model.addConstr(gamma[key] >= alpha[key] - alpha[prev_key],
                                            name=f"startup_def_{s}_{d_idx}_{w_idx}_{h}_{gid}")

    # 2. Minimum up time
    for s in node_ids:
        for d_idx in range(D):
            day_key = str(rep_days[d_idx]['id'])
            num_scen = len(lower_scen.get(day_key, []))
            for w_idx in range(num_scen):
                for g in generators:
                    gid = g['id']
                    min_on = g.get('min_on_h', 1)
                    if min_on <= 1:
                        continue
                    for h in range(H):
                        key_gamma = (s, d_idx, w_idx, h, gid)
                        for t in range(h + 1, min(H, h + min_on)):
                            key_alpha = (s, d_idx, w_idx, t, gid)
                            model.addConstr(alpha[key_alpha] >= gamma[key_gamma],
                                            name=f"minup_{s}_{d_idx}_{w_idx}_{h}_{t}_{gid}")

    # 3. Minimum down time: alpha[h'] + alpha[h-1] - alpha[h] <= 1
    for s in node_ids:
        for d_idx in range(D):
            day_key = str(rep_days[d_idx]['id'])
            num_scen = len(lower_scen.get(day_key, []))
            for w_idx in range(num_scen):
                for g in generators:
                    gid = g['id']
                    min_off = g.get('min_off_h', 1)
                    if min_off <= 1:
                        continue
                    for h in range(1, H):
                        key_prev = (s, d_idx, w_idx, h - 1, gid)
                        key_cur = (s, d_idx, w_idx, h, gid)
                        for t in range(h, min(H, h + min_off)):
                            key_t = (s, d_idx, w_idx, t, gid)
                            model.addConstr(alpha[key_t] + alpha[key_prev] - alpha[key_cur] <= 1,
                                            name=f"mindown_{s}_{d_idx}_{w_idx}_{h}_{t}_{gid}")

    # 4. Generator output bounds and commitment linking
    for s in node_ids:
        for d_idx in range(D):
            day_key = str(rep_days[d_idx]['id'])
            num_scen = len(lower_scen.get(day_key, []))
            for w_idx in range(num_scen):
                for h in range(H):
                    for g in generators:
                        gid = g['id']
                        key = (s, d_idx, w_idx, h, gid)
                        Pmax = g['Pmax_MW']
                        Pmin = g['Pmin_MW']

                        # Determine effective capacity
                        if g['existing']:
                            # p + s_res <= Pmax * alpha
                            model.addConstr(p_gen[key] + s_res[key] <= Pmax * alpha[key],
                                            name=f"cap_ub_{s}_{d_idx}_{w_idx}_{h}_{gid}")
                            # p >= Pmin * alpha
                            model.addConstr(p_gen[key] >= Pmin * alpha[key],
                                            name=f"cap_lb_{s}_{d_idx}_{w_idx}_{h}_{gid}")
                        else:
                            # Candidate: p + s_res <= Pmax * built
                            model.addConstr(p_gen[key] + s_res[key] <= Pmax * built_gen[(s, gid)],
                                            name=f"cap_ub_{s}_{d_idx}_{w_idx}_{h}_{gid}")
                            model.addConstr(p_gen[key] >= Pmin * alpha[key],
                                            name=f"cap_lb_{s}_{d_idx}_{w_idx}_{h}_{gid}")
                            # Commitment only if built
                            model.addConstr(alpha[key] <= built_gen[(s, gid)],
                                            name=f"commit_built_{s}_{d_idx}_{w_idx}_{h}_{gid}")

                        # Renewable upper bounds
                        if is_solar(g):
                            cf = solar_cf.get(d_idx, {}).get(w_idx, {}).get(h, 1.0)
                            model.addConstr(p_gen[key] <= cf * Pmax * alpha[key],
                                            name=f"solar_ub_{s}_{d_idx}_{w_idx}_{h}_{gid}")
                        elif is_wind(g):
                            wf = wind_pf.get(d_idx, {}).get(w_idx, {}).get(h, 1.0)
                            model.addConstr(p_gen[key] <= wf * Pmax * alpha[key],
                                            name=f"wind_ub_{s}_{d_idx}_{w_idx}_{h}_{gid}")

    # 5. Ramp constraints
    for s in node_ids:
        for d_idx in range(D):
            day_key = str(rep_days[d_idx]['id'])
            num_scen = len(lower_scen.get(day_key, []))
            for w_idx in range(num_scen):
                for g in generators:
                    gid = g['id']
                    ramp = g.get('ramp_MW_per_h', g['Pmax_MW'])
                    for h in range(1, H):
                        key_cur = (s, d_idx, w_idx, h, gid)
                        key_prev = (s, d_idx, w_idx, h - 1, gid)
                        model.addConstr(p_gen[key_cur] - p_gen[key_prev] <= ramp,
                                        name=f"rampup_{s}_{d_idx}_{w_idx}_{h}_{gid}")
                        model.addConstr(p_gen[key_prev] - p_gen[key_cur] <= ramp,
                                        name=f"rampdn_{s}_{d_idx}_{w_idx}_{h}_{gid}")

    # 6. Spinning reserve at each bus
    spinning_pct = gparams.get('spinning_reserve_percent', 0.0)
    if spinning_pct > 0:
        gens_at_bus = {b: [] for b in buses}
        for g in generators:
            gens_at_bus[g['bus']].append(g['id'])

        for s in node_ids:
            for d_idx in range(D):
                day_key = str(rep_days[d_idx]['id'])
                num_scen = len(lower_scen.get(day_key, []))
                for w_idx in range(num_scen):
                    for h in range(H):
                        for b in buses:
                            dem = demand_val[s][d_idx][w_idx][h].get(b, 0.0)
                            req = spinning_pct * dem
                            if req > 0 and gens_at_bus[b]:
                                lhs = gp.quicksum(s_res[(s, d_idx, w_idx, h, gid)] for gid in gens_at_bus[b])
                                model.addConstr(lhs >= req,
                                                name=f"spinres_{s}_{d_idx}_{w_idx}_{h}_{b}")

    # 7. Transmission flow limits
    for s in node_ids:
        for d_idx in range(D):
            day_key = str(rep_days[d_idx]['id'])
            num_scen = len(lower_scen.get(day_key, []))
            for w_idx in range(num_scen):
                for h in range(H):
                    for l in lines:
                        lid = l['id']
                        key = (s, d_idx, w_idx, h, lid)
                        if l['existing']:
                            fl = l['flow_limit_MW']
                            model.addConstr(f_fwd[key] <= fl, name=f"flowlim_f_{s}_{d_idx}_{w_idx}_{h}_{lid}")
                            model.addConstr(f_bwd[key] <= fl, name=f"flowlim_b_{s}_{d_idx}_{w_idx}_{h}_{lid}")
                        else:
                            model.addConstr(f_fwd[key] <= l['flow_limit_MW'] * built_line[(s, lid)],
                                            name=f"flowlim_f_{s}_{d_idx}_{w_idx}_{h}_{lid}")
                            model.addConstr(f_bwd[key] <= l['flow_limit_MW'] * built_line[(s, lid)],
                                            name=f"flowlim_b_{s}_{d_idx}_{w_idx}_{h}_{lid}")

    # 8. Storage constraints
    for s in node_ids:
        for d_idx in range(D):
            day_key = str(rep_days[d_idx]['id'])
            num_scen = len(lower_scen.get(day_key, []))
            for w_idx in range(num_scen):
                for st in storage_units:
                    sid = st['id']
                    cap = st['capacity_MW']
                    eff = st.get('efficiency', gparams.get('storage_efficiency', 0.9))

                    for h in range(H):
                        key = (s, d_idx, w_idx, h, sid)

                        # SOC dynamics
                        if h == 0:
                            model.addConstr(r_stor[key] == 0, name=f"soc_init_{s}_{d_idx}_{w_idx}_{h}_{sid}")
                        else:
                            prev_key = (s, d_idx, w_idx, h - 1, sid)
                            model.addConstr(r_stor[key] == r_stor[prev_key] + eff * u_stor[prev_key] - v_stor[prev_key],
                                            name=f"soc_dyn_{s}_{d_idx}_{w_idx}_{h}_{sid}")

                        # Discharge limit
                        model.addConstr(v_stor[key] <= r_stor[key],
                                        name=f"disch_lim_{s}_{d_idx}_{w_idx}_{h}_{sid}")

                        # Power limits
                        if st['existing']:
                            model.addConstr(u_stor[key] <= cap, name=f"ch_lim_{s}_{d_idx}_{w_idx}_{h}_{sid}")
                            model.addConstr(v_stor[key] <= cap, name=f"disch_cap_{s}_{d_idx}_{w_idx}_{h}_{sid}")
                            model.addConstr(r_stor[key] <= cap, name=f"soc_cap_{s}_{d_idx}_{w_idx}_{h}_{sid}")
                        else:
                            model.addConstr(u_stor[key] <= cap * built_stor[(s, sid)],
                                            name=f"ch_lim_{s}_{d_idx}_{w_idx}_{h}_{sid}")
                            model.addConstr(v_stor[key] <= cap * built_stor[(s, sid)],
                                            name=f"disch_cap_{s}_{d_idx}_{w_idx}_{h}_{sid}")
                            model.addConstr(r_stor[key] <= cap * built_stor[(s, sid)],
                                            name=f"soc_cap_{s}_{d_idx}_{w_idx}_{h}_{sid}")

    # 9. Energy balance at each bus
    # Precompute bus-line adjacency
    lines_from_bus = {b: [] for b in buses}
    lines_to_bus = {b: [] for b in buses}
    for l in lines:
        lines_from_bus[l['from_bus']].append(l)
        lines_to_bus[l['to_bus']].append(l)

    gens_at_bus = {b: [] for b in buses}
    for g in generators:
        gens_at_bus[g['bus']].append(g['id'])

    stors_at_bus = {b: [] for b in buses}
    for st in storage_units:
        stors_at_bus[st['bus']].append(st['id'])

    for s in node_ids:
        for d_idx in range(D):
            day_key = str(rep_days[d_idx]['id'])
            num_scen = len(lower_scen.get(day_key, []))
            for w_idx in range(num_scen):
                for h in range(H):
                    for b in buses:
                        dem = demand_val[s][d_idx][w_idx][h].get(b, 0.0)

                        # Generation
                        gen_sum = gp.quicksum(p_gen[(s, d_idx, w_idx, h, gid)] for gid in gens_at_bus[b]) if gens_at_bus[b] else 0

                        # Storage net withdrawal
                        stor_sum = gp.quicksum(v_stor[(s, d_idx, w_idx, h, sid)] - u_stor[(s, d_idx, w_idx, h, sid)]
                                               for sid in stors_at_bus[b]) if stors_at_bus[b] else 0

                        # Line flows
                        outflow = 0
                        inflow = 0
                        for l in lines_from_bus[b]:
                            lid = l['id']
                            loss = l.get('loss_factor', 0.0)
                            key = (s, d_idx, w_idx, h, lid)
                            outflow += f_fwd[key]
                            inflow += f_bwd[key] * (1 - loss)

                        for l in lines_to_bus[b]:
                            lid = l['id']
                            loss = l.get('loss_factor', 0.0)
                            key = (s, d_idx, w_idx, h, lid)
                            inflow += f_fwd[key] * (1 - loss)
                            outflow += f_bwd[key]

                        model.addConstr(gen_sum + stor_sum == dem + outflow - inflow,
                                        name=f"balance_{s}_{d_idx}_{w_idx}_{h}_{b}")

    # 10. Reserve margin constraint
    reserve_req = gparams.get('reserve_margin_requirement', 0.0)
    if reserve_req > 0:
        for s in node_ids:
            # Sum of derated capacities
            derated_cap_expr = 0
            for g in generators:
                dr = get_derating(g)
                if g['existing']:
                    derated_cap_expr += dr * g['Pmax_MW']
                else:
                    derated_cap_expr += dr * g['Pmax_MW'] * built_gen[(s, g['id'])]

            peak = total_peak.get(s, 0.0)
            model.addConstr(derated_cap_expr >= (1 + reserve_req) * peak,
                            name=f"reserve_margin_{s}")

    # Objective function
    obj = gp.LinExpr()

    # Investment costs
    for s in node_ids:
        prob_s = nodes_dict[s]['probability']
        cm = cost_mult.get(s, 1.0)

        for g in candidate_gens:
            cost = g.get('capital_cost_per_kW', 0.0) * g['Pmax_MW'] * 1000 * cm
            obj += prob_s * cost * x_gen[(s, g['id'])]

        for l in candidate_lines:
            cost = l.get('capital_cost_per_kW', 0.0) * l['flow_limit_MW'] * 1000 * cm
            obj += prob_s * cost * x_line[(s, l['id'])]

        for st in candidate_stor:
            cost = st.get('capital_cost_per_kW', 0.0) * st['capacity_MW'] * 1000 * cm
            obj += prob_s * cost * x_stor[(s, st['id'])]

    # Operating costs
    for s in node_ids:
        prob_s = nodes_dict[s]['probability']
        for d_idx in range(D):
            day_key = str(rep_days[d_idx]['id'])
            num_scen = len(lower_scen.get(day_key, []))
            for w_idx in range(num_scen):
                weight = prob_s / num_scen
                for h in range(H):
                    for g in generators:
                        gid = g['id']
                        key = (s, d_idx, w_idx, h, gid)
                        a_cost = g.get('a_cost', 0.0)
                        b_cost = g.get('b_cost', 0.0)
                        c_cost = g.get('c_cost', 0.0)
                        Pmax = g['Pmax_MW']
                        startup = g.get('startup_cost', 0.0)

                        # Secant-linearized cost: a * alpha + (b + c * Pmax) * p
                        marginal_rate = b_cost + c_cost * Pmax
                        obj += weight * (a_cost * alpha[key] + marginal_rate * p_gen[key] + startup * gamma[key])

    model.setObjective(obj, GRB.MINIMIZE)
    model.update()

    # Solve
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed - 2
    if remaining > 0:
        model.Params.TimeLimit = remaining
    else:
        model.Params.TimeLimit = 1

    model.optimize()

    # Extract solution
    solution = {
        "objective_value": 0.0,
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

    if model.SolCount > 0:
        solution["objective_value"] = model.ObjVal

        # Extract investments
        for s in node_ids:
            solution["investments"][str(s)] = {
                "generators": {},
                "lines": {},
                "storage": {}
            }
            for g in candidate_gens:
                val = x_gen[(s, g['id'])].X
                solution["investments"][str(s)]["generators"][g['id']] = int(round(val))
            for l in candidate_lines:
                val = x_line[(s, l['id'])].X
                solution["investments"][str(s)]["lines"][l['id']] = int(round(val))
            for st in candidate_stor:
                val = x_stor[(s, st['id'])].X
                solution["investments"][str(s)]["storage"][st['id']] = int(round(val))

        # Extract operational variables
        for s in node_ids:
            for d_idx in range(D):
                day_key = str(rep_days[d_idx]['id'])
                num_scen = len(lower_scen.get(day_key, []))
                for w_idx in range(num_scen):
                    for h in range(H):
                        for g in generators:
                            gid = g['id']
                            key_str = f"{s},{d_idx},{w_idx},{h},{gid}"
                            key = (s, d_idx, w_idx, h, gid)
                            solution["alpha"][key_str] = round(alpha[key].X, 6)
                            solution["gamma"][key_str] = round(gamma[key].X, 6)
                            solution["p_gen"][key_str] = round(p_gen[key].X, 6)
                            solution["s_res"][key_str] = round(s_res[key].X, 6)

                        for l in lines:
                            lid = l['id']
                            key_str = f"{s},{d_idx},{w_idx},{h},{lid}"
                            key = (s, d_idx, w_idx, h, lid)
                            solution["f_fwd"][key_str] = round(f_fwd[key].X, 6)
                            solution["f_bwd"][key_str] = round(f_bwd[key].X, 6)

                        for st in storage_units:
                            sid = st['id']
                            key_str = f"{s},{d_idx},{w_idx},{h},{sid}"
                            key = (s, d_idx, w_idx, h, sid)
                            solution["u_stor"][key_str] = round(u_stor[key].X, 6)
                            solution["v_stor"][key_str] = round(v_stor[key].X, 6)
                            solution["r_stor"][key_str] = round(r_stor[key].X, 6)

        if logger:
            logger.log_solution(solution["objective_value"], solution)
    else:
        # No solution found - output zeros
        solution["objective_value"] = 0.0
        for s in node_ids:
            solution["investments"][str(s)] = {
                "generators": {g['id']: 0 for g in candidate_gens},
                "lines": {l['id']: 0 for l in candidate_lines},
                "storage": {st['id']: 0 for st in candidate_stor}
            }
        if logger:
            logger.log(0.0)

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

    print(f"Objective value: {solution['objective_value']}")
    print(f"Total time: {time.time() - start_time:.2f}s")


if __name__ == '__main__':
    main()