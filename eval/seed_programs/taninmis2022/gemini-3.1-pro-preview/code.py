import sys
import json
import argparse
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def main():
    parser = argparse.ArgumentParser(description="Multi-stage Power System Expansion and Operation")
    parser.add_argument('--instance_path', type=str, required=True, help="Path to the instance JSON file")
    parser.add_argument('--solution_path', type=str, required=True, help="Path to write the solution JSON file")
    parser.add_argument('--time_limit', type=int, required=True, help="Maximum runtime in seconds")
    parser.add_argument('--log_path', type=str, default=None, help="Path to the intermediate solution log file")
    
    args = parser.parse_args()
    
    logger = None
    if SolutionLogger and args.log_path:
        logger = SolutionLogger(args.log_path, sense="minimize")
        
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
        
    sys_data = data['system']
    scen_params = data['scenario_parameters']
    tree = data['scenario_tree']
    rep_days = data['representative_days']
    ll_scens = data['lower_level_scenarios']
    globs = data['global_parameters']

    buses = sys_data['buses']
    gens = sys_data['generators']
    existing_gens = [g for g in gens if g['existing']]
    cand_gens = [g for g in gens if not g['existing']]
    
    lines = sys_data['transmission_lines']
    existing_lines = [l for l in lines if l['existing']]
    cand_lines = [l for l in lines if not l['existing']]
    
    stors = sys_data['storage']
    existing_stors = [s for s in stors if s['existing']]
    cand_stors = [s for s in stors if not s['existing']]
    
    bus_loads = {b: 0.0 for b in buses}
    for l in sys_data.get('loads', []):
        bus_loads[l['bus']] = l['annual_peak_load_MW']
        
    model = gp.Model("PowerSystemExpansion")
    model.setParam("TimeLimit", args.time_limit)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam("Seed", 0)
    
    # Precalculate path from root to node for cumulative capacity constraints
    path = {}
    def get_path(nid):
        if nid not in path:
            parent = next((n['parent_id'] for n in tree if n['id'] == nid), None)
            if parent is not None:
                path[nid] = get_path(parent) + [nid]
            else:
                path[nid] = [nid]
        return path[nid]
    
    for n in tree:
        get_path(n['id'])
        
    # Sort tree for BFS multiplier traversal
    sorted_tree = []
    q = [n for n in tree if n['parent_id'] is None]
    while q:
        curr = q.pop(0)
        sorted_tree.append(curr)
        q.extend([n for n in tree if n['parent_id'] == curr['id']])

    node_demand_mult = {}
    node_cost_mult = {}
    for node in sorted_tree:
        nid = node['id']
        if node['parent_id'] is None:
            node_demand_mult[nid] = 1.0
            node_cost_mult[nid] = 1.0
        else:
            pid = node['parent_id']
            parent_node = next(n for n in tree if n['id'] == pid)
            idx = parent_node['children'].index(nid)
            if idx == 0:  # High growth branch
                node_cost_mult[nid] = node_cost_mult[pid] * (1 + scen_params['cost_variation_percent'])
                node_demand_mult[nid] = node_demand_mult[pid] * (1 + scen_params['demand_growth_high_percent'])
            else:  # Low growth branch
                node_cost_mult[nid] = node_cost_mult[pid] * (1 - scen_params['cost_variation_percent'])
                node_demand_mult[nid] = node_demand_mult[pid] * (1 + scen_params['demand_growth_low_percent'])

    # ------------------ Investment Variables ------------------
    x_gen, x_line, x_stor = {}, {}, {}
    for n in tree:
        nid = n['id']
        for g in cand_gens:
            x_gen[nid, g['id']] = model.addVar(vtype=GRB.BINARY)
        for l in cand_lines:
            x_line[nid, l['id']] = model.addVar(vtype=GRB.BINARY)
        for s in cand_stors:
            x_stor[nid, s['id']] = model.addVar(vtype=GRB.BINARY)
            
    X_gen, X_line, X_stor = {}, {}, {}
    for n in tree:
        nid = n['id']
        for g in cand_gens:
            X_gen[nid, g['id']] = gp.quicksum(x_gen[m, g['id']] for m in path[nid])
            model.addConstr(X_gen[nid, g['id']] <= 1)
        for l in cand_lines:
            X_line[nid, l['id']] = gp.quicksum(x_line[m, l['id']] for m in path[nid])
            model.addConstr(X_line[nid, l['id']] <= 1)
        for s in cand_stors:
            X_stor[nid, s['id']] = gp.quicksum(x_stor[m, s['id']] for m in path[nid])
            model.addConstr(X_stor[nid, s['id']] <= 1)
            
        for g in existing_gens: X_gen[nid, g['id']] = 1.0
        for l in existing_lines: X_line[nid, l['id']] = 1.0
        for s in existing_stors: X_stor[nid, s['id']] = 1.0

    # ------------------ Precomputations ------------------
    hours = scen_params['hours_per_day']
    CF = {}
    for d in rep_days:
        did = d['id']
        for s_dict in ll_scens[str(did)]:
            sid = s_dict['scenario_id']
            for h in range(hours):
                for g in gens:
                    gtype = g['type'].lower()
                    if 'solar' in gtype:
                        CF[did, sid, h, g['id']] = d['solar_profile'][h] * s_dict['solar_multipliers'][h]
                    elif 'wind' in gtype:
                        CF[did, sid, h, g['id']] = s_dict['wind_power_factors'][h]
                    else:
                        CF[did, sid, h, g['id']] = 1.0

    Dem = {}
    for n in tree:
        nid = n['id']
        for d in rep_days:
            did = d['id']
            for s_dict in ll_scens[str(did)]:
                sid = s_dict['scenario_id']
                for h in range(hours):
                    for b in buses:
                        Dem[nid, did, sid, h, b] = (bus_loads[b] * node_demand_mult[nid] *
                                                    d['load_profile'][h] * s_dict['demand_multipliers'][h])

    # ------------------ Operational Variables ------------------
    alpha, gamma, SD, p_gen, s_res = {}, {}, {}, {}, {}
    f_fwd, f_bwd, u_stor, v_stor, r_stor = {}, {}, {}, {}, {}
    
    for n in tree:
        nid = n['id']
        for d in rep_days:
            did = d['id']
            for s_dict in ll_scens[str(did)]:
                sid = s_dict['scenario_id']
                for h in range(hours):
                    for g in gens:
                        gid = g['id']
                        alpha[nid, did, sid, h, gid] = model.addVar(vtype=GRB.BINARY)
                        gamma[nid, did, sid, h, gid] = model.addVar(lb=0)
                        SD[nid, did, sid, h, gid] = model.addVar(lb=0)
                        p_gen[nid, did, sid, h, gid] = model.addVar(lb=0)
                        s_res[nid, did, sid, h, gid] = model.addVar(lb=0)
                    for l in lines:
                        lid = l['id']
                        f_fwd[nid, did, sid, h, lid] = model.addVar(lb=0)
                        f_bwd[nid, did, sid, h, lid] = model.addVar(lb=0)
                    for st in stors:
                        stid = st['id']
                        u_stor[nid, did, sid, h, stid] = model.addVar(lb=0)
                        v_stor[nid, did, sid, h, stid] = model.addVar(lb=0)
                        r_stor[nid, did, sid, h, stid] = model.addVar(lb=0)

    # ------------------ Operational Constraints ------------------
    for n in tree:
        nid = n['id']
        for d in rep_days:
            did = d['id']
            for s_dict in ll_scens[str(did)]:
                sid = s_dict['scenario_id']
                for h in range(hours):
                    # Generators
                    for g in gens:
                        gid = g['id']
                        a = alpha[nid, did, sid, h, gid]
                        if not g['existing']:
                            model.addConstr(a <= X_gen[nid, gid])
                        
                        model.addConstr(p_gen[nid, did, sid, h, gid] + s_res[nid, did, sid, h, gid] <= 
                                        g['Pmax_MW'] * CF[did, sid, h, gid] * a)
                        model.addConstr(p_gen[nid, did, sid, h, gid] >= g['Pmin_MW'] * a)
                        
                        if h == 0:
                            model.addConstr(gamma[nid, did, sid, h, gid] >= a)
                        else:
                            a_prev = alpha[nid, did, sid, h-1, gid]
                            model.addConstr(gamma[nid, did, sid, h, gid] >= a - a_prev)
                            model.addConstr(SD[nid, did, sid, h, gid] >= a_prev - a)
                            
                            ramp = g['ramp_MW_per_h']
                            p_curr, p_prev = p_gen[nid, did, sid, h, gid], p_gen[nid, did, sid, h-1, gid]
                            model.addConstr(p_curr - p_prev <= ramp)
                            model.addConstr(p_prev - p_curr <= ramp)
                            
                        # Min Up Time
                        start_t_on = max(0, h - g['min_on_h'] + 1)
                        model.addConstr(gp.quicksum(gamma[nid, did, sid, t, gid] for t in range(start_t_on, h+1)) <= a)
                        
                        # Min Down Time
                        start_t_off = max(1, h - g['min_off_h'] + 1)
                        if start_t_off <= h:
                            model.addConstr(gp.quicksum(SD[nid, did, sid, t, gid] for t in range(start_t_off, h+1)) <= 1 - a)

                    # Storage
                    for st in stors:
                        stid = st['id']
                        cap = st['capacity_MW']
                        x_val = X_stor[nid, stid]
                        
                        u, v, r = u_stor[nid, did, sid, h, stid], v_stor[nid, did, sid, h, stid], r_stor[nid, did, sid, h, stid]
                        if not st['existing']:
                            model.addConstr(u <= cap * x_val)
                            model.addConstr(v <= cap * x_val)
                            model.addConstr(r <= cap * x_val)
                        else:
                            model.addConstr(u <= cap)
                            model.addConstr(v <= cap)
                            model.addConstr(r <= cap)
                            
                        if h == 0:
                            model.addConstr(r == 0)
                            model.addConstr(v == 0)
                        else:
                            r_prev, u_prev, v_prev = r_stor[nid, did, sid, h-1, stid], u_stor[nid, did, sid, h-1, stid], v_stor[nid, did, sid, h-1, stid]
                            model.addConstr(r == r_prev + st['efficiency'] * u_prev - v_prev)
                            model.addConstr(v <= r)

                    # Transmission Lines
                    for l in lines:
                        lid = l['id']
                        x_val = X_line[nid, lid]
                        limit = l['flow_limit_MW']
                        if not l['existing']:
                            model.addConstr(f_fwd[nid, did, sid, h, lid] <= limit * x_val)
                            model.addConstr(f_bwd[nid, did, sid, h, lid] <= limit * x_val)
                        else:
                            model.addConstr(f_fwd[nid, did, sid, h, lid] <= limit)
                            model.addConstr(f_bwd[nid, did, sid, h, lid] <= limit)
                            
    # Energy Balance and Spinning Reserve
    gen_by_bus = {b: [] for b in buses}
    stor_by_bus = {b: [] for b in buses}
    lines_from = {b: [] for b in buses}
    lines_to = {b: [] for b in buses}
    line_dict = {}
    
    for g in gens: gen_by_bus[g['bus']].append(g['id'])
    for st in stors: stor_by_bus[st['bus']].append(st['id'])
    for l in lines:
        lines_from[l['from_bus']].append(l['id'])
        lines_to[l['to_bus']].append(l['id'])
        line_dict[l['id']] = l

    for n in tree:
        nid = n['id']
        for d in rep_days:
            did = d['id']
            for s_dict in ll_scens[str(did)]:
                sid = s_dict['scenario_id']
                for h in range(hours):
                    for b in buses:
                        gen_sum = gp.quicksum(p_gen[nid, did, sid, h, gid] for gid in gen_by_bus[b])
                        net_with = gp.quicksum(v_stor[nid, did, sid, h, stid] - u_stor[nid, did, sid, h, stid] for stid in stor_by_bus[b])
                        sent = gp.quicksum(f_fwd[nid, did, sid, h, lid] for lid in lines_from[b]) + \
                               gp.quicksum(f_bwd[nid, did, sid, h, lid] for lid in lines_to[b])
                        recv = gp.quicksum(f_fwd[nid, did, sid, h, lid] * (1 - line_dict[lid]['loss_factor']) for lid in lines_to[b]) + \
                               gp.quicksum(f_bwd[nid, did, sid, h, lid] * (1 - line_dict[lid]['loss_factor']) for lid in lines_from[b])
                        demand = Dem[nid, did, sid, h, b]
                        
                        model.addConstr(gen_sum + net_with == demand + sent - recv)
                        
                        res_sum = gp.quicksum(s_res[nid, did, sid, h, gid] for gid in gen_by_bus[b])
                        model.addConstr(res_sum >= globs['spinning_reserve_percent'] * demand)

    # ------------------ Reserve Margin Requirement ------------------
    deratings = globs.get('derating_factors', {})
    def get_derating(g):
        typ = g['type'].lower()
        if typ in deratings: return deratings[typ]
        if typ in ['coal', 'natural_gas'] and 'fossil' in deratings: return deratings['fossil']
        return 1.0

    for n in tree:
        nid = n['id']
        total_derated_cap = gp.quicksum(get_derating(g) * g['Pmax_MW'] * X_gen[nid, g['id']] for g in gens)
        peak_demand_req = (1 + globs['reserve_margin_requirement']) * sum(bus_loads[b] * node_demand_mult[nid] for b in buses)
        model.addConstr(total_derated_cap >= peak_demand_req)

    # ------------------ Objective ------------------
    inv_cost_n = {}
    for n in tree:
        nid = n['id']
        cost, mult = 0.0, node_cost_mult[nid]
        for g in cand_gens: cost += mult * g['capital_cost_per_kW'] * 1000 * g['Pmax_MW'] * x_gen[nid, g['id']]
        for l in cand_lines: cost += mult * l['capital_cost_per_kW'] * 1000 * l['flow_limit_MW'] * x_line[nid, l['id']]
        for st in cand_stors: cost += mult * st['capital_cost_per_kW'] * 1000 * st['capacity_MW'] * x_stor[nid, st['id']]
        inv_cost_n[nid] = cost

    op_weight = scen_params['investment_period_years'] * 365.0 / (scen_params['num_representative_days'] * scen_params['num_lower_level_scenarios_per_day'])
    op_cost_n = {n['id']: 0.0 for n in tree}
    for n in tree:
        nid = n['id']
        for d in rep_days:
            did = d['id']
            for s_dict in ll_scens[str(did)]:
                sid = s_dict['scenario_id']
                for h in range(hours):
                    for g in gens:
                        gid = g['id']
                        a, p = alpha[nid, did, sid, h, gid], p_gen[nid, did, sid, h, gid]
                        st_cost = gamma[nid, did, sid, h, gid] * g['startup_cost']
                        gen_cost = g['a_cost'] * a + (g['b_cost'] + g['c_cost'] * g['Pmax_MW']) * p
                        op_cost_n[nid] += op_weight * (st_cost + gen_cost)

    total_obj = gp.quicksum(n['probability'] * (inv_cost_n[n['id']] + op_cost_n[n['id']]) for n in tree)
    model.setObjective(total_obj, GRB.MINIMIZE)
    
    # ------------------ Logging and Optimization ------------------
    def my_callback(model_cb, where):
        if where == GRB.Callback.MIPSOL and logger:
            obj = model_cb.cbGet(GRB.Callback.MIPSOL_OBJ)
            logger.log(obj)
            
    model.optimize(my_callback)
    
    # ------------------ Output Formatting ------------------
    if model.SolCount > 0:
        def extract_dict(var_dict):
            return {f"{k[0]}_{k[1]}_{k[2]}_{k[3]}_{k[4]}": float(v.X) for k, v in var_dict.items()}
        
        investments = {}
        for n in tree:
            nid = str(n['id'])
            inv_n = {"generators": {}, "lines": {}, "storage": {}}
            for g in cand_gens: inv_n["generators"][str(g['id'])] = int(round(x_gen[n['id'], g['id']].X))
            for l in cand_lines: inv_n["lines"][str(l['id'])] = int(round(x_line[n['id'], l['id']].X))
            for st in cand_stors: inv_n["storage"][str(st['id'])] = int(round(x_stor[n['id'], st['id']].X))
            investments[nid] = inv_n

        solution_dict = {
            "objective_value": float(model.ObjVal),
            "alpha": extract_dict(alpha),
            "gamma": extract_dict(gamma),
            "p_gen": extract_dict(p_gen),
            "s_res": extract_dict(s_res),
            "f_fwd": extract_dict(f_fwd),
            "f_bwd": extract_dict(f_bwd),
            "u_stor": extract_dict(u_stor),
            "v_stor": extract_dict(v_stor),
            "r_stor": extract_dict(r_stor),
            "investments": investments
        }
        
        if logger:
            logger.log_solution(solution_dict["objective_value"], solution_dict)

        with open(args.solution_path, 'w') as f:
            json.dump(solution_dict, f, indent=2)

if __name__ == "__main__":
    main()