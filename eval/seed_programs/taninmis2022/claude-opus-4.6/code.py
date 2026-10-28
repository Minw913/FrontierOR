import argparse
import json
import sys
import time
from collections import defaultdict

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    import gurobipy as gp
    from gurobipy import GRB

    start_time = time.time()

    with open(instance_path, 'r') as f:
        data = json.load(f)

    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    system = data['system']
    scenario_params = data['scenario_parameters']
    scenario_tree = data['scenario_tree']
    rep_days = data['representative_days']
    lower_scenarios = data['lower_level_scenarios']
    global_params = data['global_parameters']

    buses = system['buses']
    generators = system['generators']
    lines = system['transmission_lines']
    storages = system['storage']
    loads = system['loads']

    num_hours = scenario_params['hours_per_day']
    hours = list(range(num_hours))
    num_rep_days = scenario_params['num_representative_days']
    num_ll_scenarios = scenario_params['num_lower_level_scenarios_per_day']

    # Build scenario tree lookup
    tree_nodes = {n['id']: n for n in scenario_tree}
    root_id = None
    for n in scenario_tree:
        if n['parent_id'] is None:
            root_id = n['id']
            break

    # Generator lookups
    gen_by_id = {g['id']: g for g in generators}
    existing_gens = [g for g in generators if g['existing']]
    candidate_gens = [g for g in generators if not g['existing']]

    existing_lines = [l for l in lines if l['existing']]
    candidate_lines = [l for l in lines if not l['existing']]

    existing_storage = [s for s in storages if s['existing']]
    candidate_storage = [s for s in storages if not s['existing']]

    # Load by bus
    load_by_bus = {}
    for ld in loads:
        load_by_bus[ld['bus']] = ld['annual_peak_load_MW']

    # Generators at each bus
    gens_at_bus = defaultdict(list)
    for g in generators:
        gens_at_bus[g['bus']].append(g)

    # Storage at each bus
    stor_at_bus = defaultdict(list)
    for s in storages:
        stor_at_bus[s['bus']].append(s)

    # Lines from/to bus
    lines_from_bus = defaultdict(list)
    lines_to_bus = defaultdict(list)
    for l in lines:
        lines_from_bus[l['from_bus']].append(l)
        lines_to_bus[l['to_bus']].append(l)

    # Derating factors
    def get_derating(g):
        gtype = g['type'].lower()
        if 'wind' in gtype:
            return global_params['derating_factors']['wind']
        elif 'solar' in gtype:
            return global_params['derating_factors']['solar']
        else:
            return global_params['derating_factors']['fossil']

    # Demand growth factors per node
    def get_demand_growth(node_id):
        node = tree_nodes[node_id]
        path = []
        curr = node_id
        while curr is not None:
            path.append(curr)
            curr = tree_nodes[curr]['parent_id']
        path.reverse()
        # Growth is cumulative through stages
        # We need to determine at each node whether it's high or low growth
        # The scenario tree encodes this implicitly
        # For simplicity, compute based on stage and branch pattern
        # Actually the demand growth is encoded in the tree structure
        # We need to trace the path and determine growth at each stage
        growth = 1.0
        high_growth = scenario_params['demand_growth_high_percent']
        low_growth = scenario_params['demand_growth_low_percent']
        
        for i in range(1, len(path)):
            parent = tree_nodes[path[i-1]]
            child_id = path[i]
            children = parent['children']
            if len(children) > 0:
                # First child is typically high growth, second is low
                child_idx = children.index(child_id)
                if child_idx == 0:
                    growth *= (1.0 + high_growth)
                else:
                    growth *= (1.0 + low_growth)
        return growth

    # Investment cost variation per node
    def get_cost_factor(node_id):
        node = tree_nodes[node_id]
        path = []
        curr = node_id
        while curr is not None:
            path.append(curr)
            curr = tree_nodes[curr]['parent_id']
        path.reverse()
        factor = 1.0
        variation = scenario_params['cost_variation_percent']
        for i in range(1, len(path)):
            parent = tree_nodes[path[i-1]]
            child_id = path[i]
            children = parent['children']
            if len(children) > 0:
                child_idx = children.index(child_id)
                if child_idx == 0:
                    factor *= (1.0 + variation)
                else:
                    factor *= (1.0 - variation)
        return factor

    # Precompute demand growth and cost factors
    demand_growth = {n['id']: get_demand_growth(n['id']) for n in scenario_tree}
    cost_factors = {n['id']: get_cost_factor(n['id']) for n in scenario_tree}

    # Get ancestor path for each node
    def get_ancestors(node_id):
        """Return list of node ids from root to node_id (inclusive)"""
        path = []
        curr = node_id
        while curr is not None:
            path.append(curr)
            curr = tree_nodes[curr]['parent_id']
        path.reverse()
        return path

    # Representative day info
    rep_day_ids = [rd['id'] for rd in rep_days]
    rep_day_by_id = {rd['id']: rd for rd in rep_days}

    # Lower level scenarios
    ll_scenario_ids = {}
    for rd_id_str, scens in lower_scenarios.items():
        rd_id = int(rd_id_str) if rd_id_str.isdigit() else rd_id_str
        ll_scenario_ids[rd_id] = [s['scenario_id'] for s in scens]

    ll_scenario_data = {}
    for rd_id_str, scens in lower_scenarios.items():
        rd_id = int(rd_id_str) if rd_id_str.isdigit() else rd_id_str
        for s in scens:
            ll_scenario_data[(rd_id, s['scenario_id'])] = s

    # Weight per representative day (assume equal weight, scaled to represent a year)
    # With num_representative_days days, each day represents 365/num_rep_days days
    day_weight = 365.0 / num_rep_days
    # Weight per lower-level scenario
    ll_weight = 1.0 / num_ll_scenarios

    # Build the model
    model = gp.Model("GEP")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    remaining_time = time_limit - (time.time() - start_time)
    model.setParam('TimeLimit', max(remaining_time - 10, 10))

    # ========== INVESTMENT VARIABLES ==========
    # x_gen[node_id, gen_id] = 1 if candidate gen is built at this node
    x_gen = {}
    for n in scenario_tree:
        for g in candidate_gens:
            x_gen[n['id'], g['id']] = model.addVar(vtype=GRB.BINARY, name=f"xg_{n['id']}_{g['id']}")

    # x_line[node_id, line_id]
    x_line = {}
    for n in scenario_tree:
        for l in candidate_lines:
            x_line[n['id'], l['id']] = model.addVar(vtype=GRB.BINARY, name=f"xl_{n['id']}_{l['id']}")

    # x_stor[node_id, stor_id]
    x_stor = {}
    for n in scenario_tree:
        for s in candidate_storage:
            x_stor[n['id'], s['id']] = model.addVar(vtype=GRB.BINARY, name=f"xs_{n['id']}_{s['id']}")

    # Available capacity variables (cumulative)
    # cap_gen[node_id, gen_id] = available capacity (0 or Pmax)
    # Instead of explicit cap variables, we use the sum of x along ancestor path
    # built_gen[node_id, gen_id] = 1 if gen has been built at or before node
    # This equals sum of x_gen along ancestors (but at most 1)

    # For efficiency, define built variables
    built_gen = {}
    for n in scenario_tree:
        ancestors = get_ancestors(n['id'])
        for g in candidate_gens:
            built_gen[n['id'], g['id']] = model.addVar(vtype=GRB.BINARY, name=f"bg_{n['id']}_{g['id']}")

    built_line = {}
    for n in scenario_tree:
        for l in candidate_lines:
            built_line[n['id'], l['id']] = model.addVar(vtype=GRB.BINARY, name=f"bl_{n['id']}_{l['id']}")

    built_stor = {}
    for n in scenario_tree:
        for s in candidate_storage:
            built_stor[n['id'], s['id']] = model.addVar(vtype=GRB.BINARY, name=f"bs_{n['id']}_{s['id']}")

    # Link built to x along ancestors
    for n in scenario_tree:
        ancestors = get_ancestors(n['id'])
        for g in candidate_gens:
            model.addConstr(built_gen[n['id'], g['id']] == gp.quicksum(x_gen[a, g['id']] for a in ancestors))
            model.addConstr(built_gen[n['id'], g['id']] <= 1)
        for l in candidate_lines:
            model.addConstr(built_line[n['id'], l['id']] == gp.quicksum(x_line[a, l['id']] for a in ancestors))
            model.addConstr(built_line[n['id'], l['id']] <= 1)
        for s in candidate_storage:
            model.addConstr(built_stor[n['id'], s['id']] == gp.quicksum(x_stor[a, s['id']] for a in ancestors))
            model.addConstr(built_stor[n['id'], s['id']] <= 1)

    # ========== OPERATIONAL VARIABLES ==========
    # For each (node, rep_day, ll_scenario, hour, generator/line/storage)
    
    alpha = {}  # commitment
    gamma = {}  # startup
    p_gen = {}  # generation
    s_res = {}  # spinning reserve
    f_fwd = {}  # forward flow
    f_bwd = {}  # backward flow
    u_stor = {}  # storage charge (injection)
    v_stor = {}  # storage discharge (withdrawal)
    r_stor = {}  # storage state of charge

    all_ops_keys = []
    for n in scenario_tree:
        nid = n['id']
        for rd in rep_day_ids:
            rd_key = rd
            if rd_key not in ll_scenario_ids and str(rd_key) in lower_scenarios:
                rd_key_lookup = str(rd_key)
            else:
                rd_key_lookup = rd_key
            
            for sc in ll_scenario_ids.get(rd_key, ll_scenario_ids.get(str(rd_key), [])):
                for t in hours:
                    key_prefix = (nid, rd, sc, t)
                    
                    for g in generators:
                        gid = g['id']
                        k = (*key_prefix, gid)
                        
                        is_renewable = g['type'].lower() in ['wind', 'solar']
                        
                        if is_renewable:
                            alpha[k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"a_{nid}_{rd}_{sc}_{t}_{gid}")
                            gamma[k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"g_{nid}_{rd}_{sc}_{t}_{gid}")
                        else:
                            alpha[k] = model.addVar(vtype=GRB.BINARY, name=f"a_{nid}_{rd}_{sc}_{t}_{gid}")
                            gamma[k] = model.addVar(vtype=GRB.BINARY, name=f"g_{nid}_{rd}_{sc}_{t}_{gid}")
                        
                        p_gen[k] = model.addVar(lb=0, name=f"p_{nid}_{rd}_{sc}_{t}_{gid}")
                        s_res[k] = model.addVar(lb=0, name=f"sr_{nid}_{rd}_{sc}_{t}_{gid}")

                    for l in lines:
                        lid = l['id']
                        k = (*key_prefix, lid)
                        f_fwd[k] = model.addVar(lb=0, name=f"ff_{nid}_{rd}_{sc}_{t}_{lid}")
                        f_bwd[k] = model.addVar(lb=0, name=f"fb_{nid}_{rd}_{sc}_{t}_{lid}")

                    for s in storages:
                        sid = s['id']
                        k = (*key_prefix, sid)
                        u_stor[k] = model.addVar(lb=0, name=f"u_{nid}_{rd}_{sc}_{t}_{sid}")
                        v_stor[k] = model.addVar(lb=0, name=f"v_{nid}_{rd}_{sc}_{t}_{sid}")
                        r_stor[k] = model.addVar(lb=0, name=f"r_{nid}_{rd}_{sc}_{t}_{sid}")

    model.update()

    # ========== CONSTRAINTS ==========

    # Helper to get demand at bus b, node n, rep_day rd, ll_scenario sc, hour t
    def get_demand(bus, nid, rd, sc, t):
        peak = load_by_bus.get(bus, 0.0)
        if peak == 0:
            return 0.0
        rd_data = rep_day_by_id[rd]
        load_frac = rd_data['load_profile'][t]
        
        rd_key = rd
        if (rd_key, sc) not in ll_scenario_data:
            rd_key = str(rd)
        sc_data = ll_scenario_data.get((rd_key, sc), ll_scenario_data.get((rd, sc), None))
        if sc_data is None:
            # Try int conversion
            for k, v in ll_scenario_data.items():
                if str(k[0]) == str(rd) and k[1] == sc:
                    sc_data = v
                    break
        
        dm = sc_data['demand_multipliers'][t] if sc_data else 1.0
        growth = demand_growth[nid]
        return peak * load_frac * dm * growth

    def get_solar_cf(rd, sc, t):
        rd_data = rep_day_by_id[rd]
        base_solar = rd_data['solar_profile'][t]
        
        rd_key = rd
        if (rd_key, sc) not in ll_scenario_data:
            rd_key = str(rd)
        sc_data = ll_scenario_data.get((rd_key, sc), ll_scenario_data.get((rd, sc), None))
        if sc_data is None:
            for k, v in ll_scenario_data.items():
                if str(k[0]) == str(rd) and k[1] == sc:
                    sc_data = v
                    break
        
        sm = sc_data['solar_multipliers'][t] if sc_data else 1.0
        return base_solar * sm

    def get_wind_cf(rd, sc, t):
        rd_key = rd
        if (rd_key, sc) not in ll_scenario_data:
            rd_key = str(rd)
        sc_data = ll_scenario_data.get((rd_key, sc), ll_scenario_data.get((rd, sc), None))
        if sc_data is None:
            for k, v in ll_scenario_data.items():
                if str(k[0]) == str(rd) and k[1] == sc:
                    sc_data = v
                    break
        return sc_data['wind_power_factors'][t] if sc_data else 0.0

    # Precompute ll_scenario_data lookup
    ll_data_cache = {}
    for rd_id_str, scens in lower_scenarios.items():
        rd_id = int(rd_id_str) if rd_id_str.isdigit() else rd_id_str
        for s in scens:
            ll_data_cache[(rd_id, s['scenario_id'])] = s

    def get_ll_data(rd, sc):
        if (rd, sc) in ll_data_cache:
            return ll_data_cache[(rd, sc)]
        if (str(rd), sc) in ll_data_cache:
            return ll_data_cache[(str(rd), sc)]
        if (int(rd) if isinstance(rd, str) and rd.isdigit() else rd, sc) in ll_data_cache:
            return ll_data_cache[(int(rd), sc)]
        return None

    spinning_reserve_pct = global_params['spinning_reserve_percent']

    # Add constraints for each scenario tree node
    for n in scenario_tree:
        nid = n['id']
        prob = n['probability']

        for rd in rep_day_ids:
            rd_data = rep_day_by_id[rd]
            
            sc_list = []
            for k in ll_data_cache:
                if str(k[0]) == str(rd):
                    sc_list.append(k[1])
            sc_list = sorted(set(sc_list))
            
            for sc in sc_list:
                sc_data = get_ll_data(rd, sc)
                
                for t in hours:
                    # === Generator constraints ===
                    for g in generators:
                        gid = g['id']
                        k = (nid, rd, sc, t, gid)
                        pmax = g['Pmax_MW']
                        pmin = g['Pmin_MW']
                        
                        gtype = g['type'].lower()
                        is_solar = 'solar' in gtype
                        is_wind = 'wind' in gtype
                        
                        # Capacity factor for renewables
                        if is_solar:
                            cf = get_solar_cf(rd, sc, t)
                            effective_pmax = pmax * cf
                        elif is_wind:
                            cf = get_wind_cf(rd, sc, t)
                            effective_pmax = pmax * cf
                        else:
                            effective_pmax = pmax

                        if g['existing']:
                            # p + s_res <= effective_pmax * alpha
                            model.addConstr(p_gen[k] + s_res[k] <= effective_pmax * alpha[k])
                            # p >= pmin * alpha
                            if not (is_solar or is_wind):
                                model.addConstr(p_gen[k] >= pmin * alpha[k])
                            # alpha <= 1
                            model.addConstr(alpha[k] <= 1)
                        else:
                            # Candidate: commitment limited by built status
                            model.addConstr(alpha[k] <= built_gen[nid, gid])
                            # p + s_res <= effective_pmax * alpha (using alpha which is <=1 when built)
                            model.addConstr(p_gen[k] + s_res[k] <= effective_pmax * alpha[k])
                            if not (is_solar or is_wind):
                                model.addConstr(p_gen[k] >= pmin * alpha[k])

                        # Startup constraint
                        if t == 0:
                            model.addConstr(gamma[k] >= alpha[k])
                        else:
                            k_prev = (nid, rd, sc, t-1, gid)
                            model.addConstr(gamma[k] >= alpha[k] - alpha[k_prev])

                        # Min up/down time
                        min_ud = g.get('min_on_h', 1)
                        if not (is_solar or is_wind) and min_ud > 1:
                            # Min up time: if started at t, must be on for min_ud hours
                            end_up = min(t + min_ud, num_hours)
                            if t > 0:
                                k_prev = (nid, rd, sc, t-1, gid)
                                for tt in range(t, end_up):
                                    k_tt = (nid, rd, sc, tt, gid)
                                    model.addConstr(alpha[k_tt] >= alpha[k] - alpha[k_prev])
                            
                            # Min down time: if shut down at t, must be off for min_ud hours
                            if t > 0:
                                k_prev = (nid, rd, sc, t-1, gid)
                                end_down = min(t + min_ud, num_hours)
                                for tt in range(t, end_down):
                                    k_tt = (nid, rd, sc, tt, gid)
                                    model.addConstr(alpha[k_tt] <= 1 - (alpha[k_prev] - alpha[k]))

                        # Ramp constraints
                        ramp = g['ramp_MW_per_h']
                        if t > 0 and ramp < pmax:
                            k_prev = (nid, rd, sc, t-1, gid)
                            model.addConstr(p_gen[k] - p_gen[k_prev] <= ramp)
                            model.addConstr(p_gen[k_prev] - p_gen[k] <= ramp)

                    # === Transmission line constraints ===
                    for l in lines:
                        lid = l['id']
                        k = (nid, rd, sc, t, lid)
                        if l['existing']:
                            model.addConstr(f_fwd[k] <= l['flow_limit_MW'])
                            model.addConstr(f_bwd[k] <= l['flow_limit_MW'])
                        else:
                            cap = l['flow_limit_MW']
                            model.addConstr(f_fwd[k] <= cap * built_line[nid, lid])
                            model.addConstr(f_bwd[k] <= cap * built_line[nid, lid])

                    # === Storage constraints ===
                    for s in storages:
                        sid = s['id']
                        k = (nid, rd, sc, t, sid)
                        eff = s['efficiency']
                        cap = s['capacity_MW']
                        
                        if t == 0:
                            # Start with zero stored energy
                            model.addConstr(r_stor[k] == 0)
                        else:
                            k_prev = (nid, rd, sc, t-1, sid)
                            model.addConstr(r_stor[k] == r_stor[k_prev] + eff * u_stor[k_prev] - v_stor[k_prev])
                        
                        # Withdrawal <= stored energy
                        model.addConstr(v_stor[k] <= r_stor[k])
                        
                        # Charge/discharge limits
                        model.addConstr(u_stor[k] <= cap)
                        model.addConstr(v_stor[k] <= cap)
                        
                        if s['existing']:
                            model.addConstr(r_stor[k] <= cap)
                        else:
                            model.addConstr(r_stor[k] <= cap * built_stor[nid, sid])
                            model.addConstr(u_stor[k] <= cap * built_stor[nid, sid])
                            model.addConstr(v_stor[k] <= cap * built_stor[nid, sid])

                    # === Energy balance at each bus ===
                    for b in buses:
                        demand = get_demand(b, nid, rd, sc, t)
                        
                        # Generation at bus
                        gen_expr = gp.quicksum(p_gen[(nid, rd, sc, t, g['id'])] for g in gens_at_bus[b])
                        
                        # Storage net withdrawal at bus
                        stor_expr = gp.quicksum(v_stor[(nid, rd, sc, t, s['id'])] - u_stor[(nid, rd, sc, t, s['id'])] for s in stor_at_bus[b])
                        
                        # Lines: flow out from bus b
                        flow_out = gp.LinExpr()
                        flow_in = gp.LinExpr()
                        
                        for l in lines_from_bus[b]:
                            lid = l['id']
                            lk = (nid, rd, sc, t, lid)
                            loss = l['loss_factor']
                            flow_out.add(f_fwd[lk])
                            flow_in.add(f_bwd[lk], 1.0 - loss)
                        
                        for l in lines_to_bus[b]:
                            lid = l['id']
                            lk = (nid, rd, sc, t, lid)
                            loss = l['loss_factor']
                            flow_out.add(f_bwd[lk])
                            flow_in.add(f_fwd[lk], 1.0 - loss)
                        
                        model.addConstr(gen_expr + stor_expr + flow_in - flow_out == demand)

                    # === Spinning reserve at each bus ===
                    for b in buses:
                        demand = get_demand(b, nid, rd, sc, t)
                        sr_req = spinning_reserve_pct * demand
                        if sr_req > 0:
                            model.addConstr(
                                gp.quicksum(s_res[(nid, rd, sc, t, g['id'])] for g in gens_at_bus[b]) >= sr_req
                            )

        # === Reserve margin constraint ===
        # Sum of derated capacities >= (1 + reserve_margin) * peak_demand * growth
        total_peak = sum(load_by_bus.get(b, 0) for b in buses) * demand_growth[nid]
        rm_req = global_params['reserve_margin_requirement']
        
        derated_cap = gp.LinExpr()
        for g in existing_gens:
            derated_cap.add(g['Pmax_MW'] * get_derating(g))
        for g in candidate_gens:
            derated_cap.add(built_gen[nid, g['id']], g['Pmax_MW'] * get_derating(g))
        
        model.addConstr(derated_cap >= (1.0 + rm_req) * total_peak)

    # ========== OBJECTIVE ==========
    obj = gp.LinExpr()

    # Investment costs
    for n in scenario_tree:
        nid = n['id']
        prob = n['probability']
        cf = cost_factors[nid]
        
        for g in candidate_gens:
            cost = g['capital_cost_per_kW'] * g['Pmax_MW'] * 1000.0 * cf  # capital_cost_per_kW * kW
            # Actually capital_cost_per_kW * Pmax_MW * 1000 converts MW to kW
            # Wait: Pmax is in MW, capital_cost is $/kW, so cost = capital_cost_per_kW * Pmax_MW * 1000
            obj.add(x_gen[nid, g['id']], prob * g['capital_cost_per_kW'] * g['Pmax_MW'] * 1000.0 * cf)

        for l in candidate_lines:
            obj.add(x_line[nid, l['id']], prob * l['capital_cost_per_kW'] * l['flow_limit_MW'] * 1000.0 * cf)

        for s in candidate_storage:
            obj.add(x_stor[nid, s['id']], prob * s['capital_cost_per_kW'] * s['capacity_MW'] * 1000.0 * cf)

    # Operating costs
    for n in scenario_tree:
        nid = n['id']
        prob = n['probability']
        
        for rd in rep_day_ids:
            sc_list = []
            for k in ll_data_cache:
                if str(k[0]) == str(rd):
                    sc_list.append(k[1])
            sc_list = sorted(set(sc_list))
            
            for sc in sc_list:
                weight = prob * day_weight * ll_weight
                
                for t in hours:
                    for g in generators:
                        gid = g['id']
                        k = (nid, rd, sc, t, gid)
                        
                        # Startup cost
                        obj.add(gamma[k], weight * g['startup_cost'])
                        
                        # Generation cost: a * alpha + (b + c * Pmax) * p
                        a = g['a_cost']
                        b = g['b_cost']
                        c = g['c_cost']
                        pmax = g['Pmax_MW']
                        
                        obj.add(alpha[k], weight * a)
                        obj.add(p_gen[k], weight * (b + c * pmax))

    model.setObjective(obj, GRB.MINIMIZE)
    
    # Use lazy updates for large models
    model.update()

    # Callback to log solutions
    best_obj = [float('inf')]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0]:
                best_obj[0] = obj_val
                if logger:
                    logger.log(obj_val)

    model.optimize(callback)

    # ========== EXTRACT SOLUTION ==========
    sol = {
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
        sol["objective_value"] = model.ObjVal

        # Extract investments
        for n in scenario_tree:
            nid = str(n['id'])
            sol["investments"][nid] = {
                "generators": {},
                "lines": {},
                "storage": {}
            }
            for g in candidate_gens:
                val = x_gen[n['id'], g['id']].X
                sol["investments"][nid]["generators"][g['id']] = int(round(val))
            for l in candidate_lines:
                val = x_line[n['id'], l['id']].X
                sol["investments"][nid]["lines"][l['id']] = int(round(val))
            for s in candidate_storage:
                val = x_stor[n['id'], s['id']].X
                sol["investments"][nid]["storage"][s['id']] = int(round(val))

        # Extract operational variables
        for key, var in alpha.items():
            nid, rd, sc, t, gid = key
            str_key = f"{nid}_{rd}_{sc}_{t}_{gid}"
            sol["alpha"][str_key] = var.X

        for key, var in gamma.items():
            nid, rd, sc, t, gid = key
            str_key = f"{nid}_{rd}_{sc}_{t}_{gid}"
            sol["gamma"][str_key] = var.X

        for key, var in p_gen.items():
            nid, rd, sc, t, gid = key
            str_key = f"{nid}_{rd}_{sc}_{t}_{gid}"
            sol["p_gen"][str_key] = var.X

        for key, var in s_res.items():
            nid, rd, sc, t, gid = key
            str_key = f"{nid}_{rd}_{sc}_{t}_{gid}"
            sol["s_res"][str_key] = var.X

        for key, var in f_fwd.items():
            nid, rd, sc, t, lid = key
            str_key = f"{nid}_{rd}_{sc}_{t}_{lid}"
            sol["f_fwd"][str_key] = var.X

        for key, var in f_bwd.items():
            nid, rd, sc, t, lid = key
            str_key = f"{nid}_{rd}_{sc}_{t}_{lid}"
            sol["f_bwd"][str_key] = var.X

        for key, var in u_stor.items():
            nid, rd, sc, t, sid = key
            str_key = f"{nid}_{rd}_{sc}_{t}_{sid}"
            sol["u_stor"][str_key] = var.X

        for key, var in v_stor.items():
            nid, rd, sc, t, sid = key
            str_key = f"{nid}_{rd}_{sc}_{t}_{sid}"
            sol["v_stor"][str_key] = var.X

        for key, var in r_stor.items():
            nid, rd, sc, t, sid = key
            str_key = f"{nid}_{rd}_{sc}_{t}_{sid}"
            sol["r_stor"][str_key] = var.X

        if logger:
            logger.log_solution(sol["objective_value"], sol)

    with open(solution_path, 'w') as f:
        json.dump(sol, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)