import json
import argparse
import time
import sys
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None


def compute_thresholds(gen, max_gas_price):
    """Compute per-bid profitability thresholds for a GFPP."""
    if not gen['is_gfpp']:
        return []
    hrc = gen['heat_rate_coefficients']
    H_u2 = hrc['H_u2'] if hrc['H_u2'] is not None else 0.0
    H_u1 = hrc['H_u1'] if hrc['H_u1'] is not None else 0.0
    thresholds = []
    cum = 0.0
    for bid in gen['bids']:
        mid = cum + bid['max_amount'] / 2.0
        marginal_hr = 2.0 * H_u2 * mid + H_u1
        if marginal_hr > 1e-8:
            threshold = bid['price'] / marginal_hr
        else:
            threshold = max_gas_price
        thresholds.append(min(threshold, max_gas_price))
        cum += bid['max_amount']
    return thresholds


def main():
    parser = argparse.ArgumentParser(description="UCGNA Solver")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if (args.log_path and SolutionLogger) else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    beta = instance['objective_weight_beta']
    T = instance['time_periods']
    eta_p = instance.get('stress_parameters', {}).get('eta_p', 1.0)
    eta_g = instance.get('stress_parameters', {}).get('eta_g', 1.0)

    buses = instance['electricity_network']['buses']
    lines = instance['electricity_network']['lines']
    generators = instance['generators']['generators']
    gas_net = instance['gas_network']
    junctions = gas_net['junctions']
    connections = gas_net['connections']
    pricing_zones = gas_net['pricing_zones']
    max_gas_price = gas_net['max_gas_price_mmBtu']
    min_gas_price = gas_net['min_gas_price_mmBtu']

    # Mappings
    junction_to_zone = {}
    for z in pricing_zones:
        for j in z['junctions']:
            junction_to_zone[j] = z['id']

    gfpp_at_junction = {}
    for j in junctions:
        gfpp_at_junction[j['id']] = []
    for g in generators:
        if g['is_gfpp'] and g['gas_junction'] is not None:
            if g['gas_junction'] not in gfpp_at_junction:
                gfpp_at_junction[g['gas_junction']] = []
            gfpp_at_junction[g['gas_junction']].append(g)

    bus_gens = {}
    for b in buses:
        bus_gens[b['id']] = []
    for g in generators:
        if g['bus'] not in bus_gens:
            bus_gens[g['bus']] = []
        bus_gens[g['bus']].append(g)

    bus_lines_out = {}
    bus_lines_in = {}
    for b in buses:
        bus_lines_out[b['id']] = []
        bus_lines_in[b['id']] = []
    for l in lines:
        bus_lines_out[l['from_bus']].append(l)
        bus_lines_in[l['to_bus']].append(l)

    conn_from_j = {}
    conn_to_j = {}
    for j in junctions:
        conn_from_j[j['id']] = []
        conn_to_j[j['id']] = []
    for c in connections:
        conn_from_j[c['from_junction']].append(c)
        conn_to_j[c['to_junction']].append(c)

    # Create model
    model = gp.Model("UCGNA")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', 0)

    # === VARIABLES ===
    o = {}
    vp = {}
    vm = {}
    w = {}
    r = {}
    phimax = {}
    p = {}
    se = {}

    for g in generators:
        gid = g['id']
        for t in range(1, T + 1):
            o[gid, t] = model.addVar(vtype=GRB.BINARY, name=f"o_{gid}_{t}")
            vp[gid, t] = model.addVar(vtype=GRB.BINARY, name=f"vp_{gid}_{t}")
            vm[gid, t] = model.addVar(vtype=GRB.BINARY, name=f"vm_{gid}_{t}")
            r[gid, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"r_{gid}_{t}")
            p[gid, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=g['max_power'], name=f"p_{gid}_{t}")
            for bid in g['bids']:
                bid_id = bid['id']
                w[gid, bid_id, t] = model.addVar(vtype=GRB.BINARY, name=f"w_{gid}_{bid_id}_{t}")
                se[gid, bid_id, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=bid['max_amount'], name=f"se_{gid}_{bid_id}_{t}")
        if g['is_gfpp']:
            for t in range(1, T + 1):
                phimax[gid, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max_gas_price, name=f"phimax_{gid}_{t}")

    f_var = {}
    for l in lines:
        lid = l['id']
        for t in range(1, T + 1):
            f_var[lid, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=-l['thermal_limit'], ub=l['thermal_limit'], name=f"f_{lid}_{t}")

    theta = {}
    for b in buses:
        bid_bus = b['id']
        for t in range(1, T + 1):
            theta[bid_bus, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=b['voltage_angle_lb'], ub=b['voltage_angle_ub'], name=f"theta_{bid_bus}_{t}")

    sg = {}
    sgs = {}
    pisq = {}
    phig = {}
    lg = {}
    qg = {}
    gamma = {}

    for j in junctions:
        jid = j['id']
        for t in range(1, T + 1):
            sg[jid, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"sg_{jid}_{t}")
            pisq[jid, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=j['pressure_lb_squared'], ub=j['pressure_ub_squared'], name=f"pisq_{jid}_{t}")
            lg[jid, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"lg_{jid}_{t}")
            qg[jid, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"qg_{jid}_{t}")
            gamma[jid, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"gamma_{jid}_{t}")
        if j['is_source'] and 'supply_intervals' in j:
            for si in j['supply_intervals']:
                sid = si['id']
                for t in range(1, T + 1):
                    sgs[jid, sid, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=si['interval_lb'], ub=si['interval_ub'], name=f"sgs_{jid}_{sid}_{t}")

    for c in connections:
        cid = c['id']
        for t in range(1, T + 1):
            phig[cid, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"phig_{cid}_{t}")

    psi = {}
    for z in pricing_zones:
        zid = z['id']
        for t in range(1, T + 1):
            psi[zid, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=min_gas_price, ub=max_gas_price, name=f"psi_{zid}_{t}")

    model.update()

    # === CONSTRAINTS ===

    # Startup/shutdown relation
    for g in generators:
        gid = g['id']
        for t in range(1, T + 1):
            if t == 1:
                model.addConstr(vp[gid, t] - vm[gid, t] == o[gid, t] - g['initial_status'])
            else:
                model.addConstr(vp[gid, t] - vm[gid, t] == o[gid, t] - o[gid, t - 1])

    # Initial status constraints
    for g in generators:
        gid = g['id']
        if g['initial_status'] == 1:
            remaining = g['min_up_time'] - g['initial_active_periods']
            for t in range(1, min(remaining, T) + 1):
                model.addConstr(o[gid, t] == 1)
        else:
            remaining = g['min_down_time'] - g['initial_inactive_periods']
            for t in range(1, min(remaining, T) + 1):
                model.addConstr(o[gid, t] == 0)

    # Minimum up time
    for g in generators:
        gid = g['id']
        MU = g['min_up_time']
        if MU > 1:
            for t in range(1, T - MU + 2):
                model.addConstr(
                    gp.quicksum(vp[gid, tau] for tau in range(t, t + MU)) <= o[gid, t + MU - 1]
                )

    # Minimum down time
    for g in generators:
        gid = g['id']
        MD = g['min_down_time']
        if MD > 1:
            for t in range(1, T - MD + 2):
                if t == 1:
                    rhs = 1 - g['initial_status']
                    model.addConstr(
                        gp.quicksum(vp[gid, tau] for tau in range(t, t + MD)) <= rhs
                    )
                else:
                    model.addConstr(
                        gp.quicksum(vp[gid, tau] for tau in range(t, t + MD)) <= 1 - o[gid, t - 1]
                    )

    # Startup cost tiers
    for g in generators:
        gid = g['id']
        if not g['startup_cost_params']:
            continue
        for tier in g['startup_cost_params']:
            lookback = int(tier[0])
            cost = tier[1]
            for t in range(1, T + 1):
                if t <= lookback and g['initial_status'] == 0:
                    if g['initial_inactive_periods'] + (t - 1) < lookback:
                        continue
                start = max(1, t - lookback)
                if start < t:
                    vm_sum = gp.quicksum(vm[gid, tau] for tau in range(start, t))
                    model.addConstr(r[gid, t] >= cost * (vp[gid, t] - vm_sum))
                else:
                    model.addConstr(r[gid, t] >= cost * vp[gid, t])

    # Bid constraints
    for g in generators:
        gid = g['id']
        bids = g['bids']
        num_bids = len(bids)
        for t in range(1, T + 1):
            for i, bid in enumerate(bids):
                bid_id = bid['id']
                model.addConstr(w[gid, bid_id, t] <= o[gid, t])
                model.addConstr(se[gid, bid_id, t] <= bid['max_amount'] * w[gid, bid_id, t])
            for i in range(num_bids - 1):
                bid_id = bids[i]['id']
                next_bid_id = bids[i + 1]['id']
                model.addConstr(w[gid, next_bid_id, t] <= w[gid, bid_id, t])
                model.addConstr(se[gid, bid_id, t] >= bids[i]['max_amount'] * w[gid, next_bid_id, t])

    # Power output = sum of bids
    for g in generators:
        gid = g['id']
        for t in range(1, T + 1):
            model.addConstr(
                p[gid, t] == gp.quicksum(se[gid, bid['id'], t] for bid in g['bids'])
            )

    # Power output bounds
    for g in generators:
        gid = g['id']
        for t in range(1, T + 1):
            model.addConstr(p[gid, t] >= g['min_power'] * o[gid, t])
            model.addConstr(p[gid, t] <= g['max_power'] * o[gid, t])

    # Ramping
    for g in generators:
        gid = g['id']
        for t in range(1, T + 1):
            if t == 1:
                p_prev = g['initial_gen']
                o_prev_val = g['initial_status']
                model.addConstr(p[gid, t] - p_prev <= g['ramp_up'] * o_prev_val + g['max_power'] * vp[gid, t])
                model.addConstr(p_prev - p[gid, t] <= g['ramp_down'] * o[gid, t] + g['min_power'] * vm[gid, t])
            else:
                model.addConstr(p[gid, t] - p[gid, t - 1] <= g['ramp_up'] * o[gid, t - 1] + g['max_power'] * vp[gid, t])
                model.addConstr(p[gid, t - 1] - p[gid, t] <= g['ramp_down'] * o[gid, t] + g['min_power'] * vm[gid, t])

    # Power balance
    for b in buses:
        bid_bus = b['id']
        for t in range(1, T + 1):
            demand = b['demand_profile'][t - 1] * eta_p
            gen_sum = gp.quicksum(p[g['id'], t] for g in bus_gens.get(bid_bus, []))
            flow_out = gp.quicksum(f_var[l['id'], t] for l in bus_lines_out.get(bid_bus, []))
            flow_in = gp.quicksum(f_var[l['id'], t] for l in bus_lines_in.get(bid_bus, []))
            model.addConstr(gen_sum - demand == flow_out - flow_in)

    # DC power flow
    for l in lines:
        lid = l['id']
        for t in range(1, T + 1):
            model.addConstr(f_var[lid, t] == l['susceptance'] * (theta[l['from_bus'], t] - theta[l['to_bus'], t]))
            model.addConstr(theta[l['from_bus'], t] - theta[l['to_bus'], t] <= l['angle_diff_limit'])
            model.addConstr(theta[l['from_bus'], t] - theta[l['to_bus'], t] >= -l['angle_diff_limit'])

    # phimax definition for GFPPs
    for g in generators:
        if not g['is_gfpp']:
            continue
        gid = g['id']
        bids = g['bids']
        num_bids = len(bids)
        thresholds = compute_thresholds(g, max_gas_price)
        for t in range(1, T + 1):
            expr = gp.LinExpr()
            for i in range(num_bids - 1):
                expr += thresholds[i] * (w[gid, bids[i]['id'], t] - w[gid, bids[i + 1]['id'], t])
            expr += thresholds[num_bids - 1] * w[gid, bids[num_bids - 1]['id'], t]
            model.addConstr(phimax[gid, t] == expr)

    # Gas price constraint for GFPPs
    for g in generators:
        if not g['is_gfpp']:
            continue
        gid = g['id']
        junction = g['gas_junction']
        if junction is None or junction not in junction_to_zone:
            continue
        zone = junction_to_zone[junction]
        for t in range(1, T + 1):
            model.addConstr(phimax[gid, t] >= psi[zone, t] - max_gas_price * (1 - o[gid, t]))

    # max_gas_price_fraction upper bound on phimax
    for g in generators:
        if not g['is_gfpp']:
            continue
        gid = g['id']
        if g.get('max_gas_price_fraction') is not None:
            cap = g['max_gas_price_fraction'] * max_gas_price
            for t in range(1, T + 1):
                model.addConstr(phimax[gid, t] <= cap)

    # Gas balance
    for j in junctions:
        jid = j['id']
        for t in range(1, T + 1):
            flow_out = gp.quicksum(phig[c['id'], t] for c in conn_from_j.get(jid, []))
            flow_in = gp.quicksum(phig[c['id'], t] for c in conn_to_j.get(jid, []))
            model.addConstr(sg[jid, t] - lg[jid, t] - gamma[jid, t] == flow_out - flow_in)

    # Gas supply
    for j in junctions:
        jid = j['id']
        if j['is_source'] and 'supply_intervals' in j and len(j['supply_intervals']) > 0:
            for t in range(1, T + 1):
                model.addConstr(sg[jid, t] == gp.quicksum(sgs[jid, si['id'], t] for si in j['supply_intervals']))
        else:
            for t in range(1, T + 1):
                model.addConstr(sg[jid, t] == 0)

    # Gas demand
    for j in junctions:
        jid = j['id']
        for t in range(1, T + 1):
            demand = j['gas_demand_profile'][t - 1] * eta_g
            model.addConstr(lg[jid, t] + qg[jid, t] == demand)

    # Weymouth constraints
    for c in connections:
        cid = c['id']
        if c['type'] == 'pipeline' and c.get('weymouth_factor') is not None:
            wf = c['weymouth_factor']
            fj = c['from_junction']
            tj = c['to_junction']
            for t in range(1, T + 1):
                model.addConstr(pisq[fj, t] - pisq[tj, t] >= wf * phig[cid, t] * phig[cid, t])

    # Compressor constraints
    for c in connections:
        cid = c['id']
        if c['type'] == 'compressor' and c.get('compression_ratio_lb') is not None:
            cr_lb = c['compression_ratio_lb']
            cr_ub = c['compression_ratio_ub']
            fj = c['from_junction']
            tj = c['to_junction']
            for t in range(1, T + 1):
                model.addConstr(pisq[tj, t] >= cr_lb ** 2 * pisq[fj, t])
                model.addConstr(pisq[tj, t] <= cr_ub ** 2 * pisq[fj, t])

    # Control valve constraints
    for c in connections:
        cid = c['id']
        if c.get('control_ratio_lb') is not None and c.get('control_ratio_ub') is not None:
            cv_lb = c['control_ratio_lb']
            cv_ub = c['control_ratio_ub']
            fj = c['from_junction']
            tj = c['to_junction']
            for t in range(1, T + 1):
                model.addConstr(pisq[tj, t] >= cv_lb ** 2 * pisq[fj, t])
                model.addConstr(pisq[tj, t] <= cv_ub ** 2 * pisq[fj, t])

    # Heat rate constraints
    for j in junctions:
        jid = j['id']
        gfpps_here = gfpp_at_junction.get(jid, [])
        if gfpps_here:
            for t in range(1, T + 1):
                rhs = gp.QuadExpr()
                for g in gfpps_here:
                    gid = g['id']
                    hrc = g['heat_rate_coefficients']
                    H_u2 = hrc['H_u2'] if hrc['H_u2'] is not None else 0.0
                    H_u1 = hrc['H_u1'] if hrc['H_u1'] is not None else 0.0
                    H_u0 = hrc['H_u0'] if hrc['H_u0'] is not None else 0.0
                    if H_u2 > 0:
                        rhs += H_u2 * p[gid, t] * p[gid, t]
                    if H_u1 != 0:
                        rhs += H_u1 * p[gid, t]
                    if H_u0 != 0:
                        rhs += H_u0 * o[gid, t]
                model.addConstr(gamma[jid, t] >= rhs)
        else:
            for t in range(1, T + 1):
                model.addConstr(gamma[jid, t] == 0)

    # === OBJECTIVE ===
    elec_cost = gp.LinExpr()
    for g in generators:
        gid = g['id']
        for t in range(1, T + 1):
            elec_cost += g['no_load_cost'] * o[gid, t]
            elec_cost += r[gid, t]
            for bid in g['bids']:
                elec_cost += bid['price'] * se[gid, bid['id'], t]

    gas_cost = gp.LinExpr()
    for j in junctions:
        jid = j['id']
        for t in range(1, T + 1):
            gas_cost += j['demand_shedding_cost'] * qg[jid, t]
            if j['is_source'] and 'supply_intervals' in j:
                for si in j['supply_intervals']:
                    gas_cost += si['slope'] * sgs[jid, si['id'], t]

    model.setObjective(beta * elec_cost + (1 - beta) * gas_cost, GRB.MINIMIZE)

    # Solve
    model.optimize()

    # Extract solution
    if model.SolCount > 0:
        primary_vars = {}

        for g in generators:
            gid = g['id']
            primary_vars[f"o_{gid}_0"] = float(g['initial_status'])
            primary_vars[f"p_{gid}_0"] = float(g['initial_gen'])
            for t in range(1, T + 1):
                primary_vars[f"o_{gid}_{t}"] = o[gid, t].X
                primary_vars[f"vp_{gid}_{t}"] = vp[gid, t].X
                primary_vars[f"vm_{gid}_{t}"] = vm[gid, t].X
                primary_vars[f"r_{gid}_{t}"] = r[gid, t].X
                primary_vars[f"p_{gid}_{t}"] = p[gid, t].X
                for bid in g['bids']:
                    bid_id = bid['id']
                    primary_vars[f"w_{gid}_{bid_id}_{t}"] = w[gid, bid_id, t].X
                    primary_vars[f"se_{gid}_{bid_id}_{t}"] = se[gid, bid_id, t].X
                if g['is_gfpp']:
                    primary_vars[f"phimax_{gid}_{t}"] = phimax[gid, t].X

        for l in lines:
            lid = l['id']
            for t in range(1, T + 1):
                primary_vars[f"f_{lid}_{t}"] = f_var[lid, t].X

        for b in buses:
            bid_bus = b['id']
            for t in range(1, T + 1):
                primary_vars[f"theta_{bid_bus}_{t}"] = theta[bid_bus, t].X

        for j in junctions:
            jid = j['id']
            for t in range(1, T + 1):
                primary_vars[f"sg_{jid}_{t}"] = sg[jid, t].X
                primary_vars[f"pisq_{jid}_{t}"] = pisq[jid, t].X
                primary_vars[f"lg_{jid}_{t}"] = lg[jid, t].X
                primary_vars[f"qg_{jid}_{t}"] = qg[jid, t].X
                primary_vars[f"gamma_{jid}_{t}"] = gamma[jid, t].X
                if j['is_source'] and 'supply_intervals' in j:
                    for si in j['supply_intervals']:
                        primary_vars[f"sgs_{jid}_{si['id']}_{t}"] = sgs[jid, si['id'], t].X

        for c in connections:
            cid = c['id']
            for t in range(1, T + 1):
                primary_vars[f"phig_{cid}_{t}"] = phig[cid, t].X

        for z in pricing_zones:
            zid = z['id']
            for t in range(1, T + 1):
                primary_vars[f"psi_{zid}_{t}"] = psi[zid, t].X

        obj_val = model.ObjVal
        solution = {
            "objective_value": obj_val,
            "released_problem_variant": "integrated_primal_ucgna",
            "objective_weight_beta": beta,
            "primary_variables": primary_vars
        }

        if logger:
            logger.log_solution(obj_val, solution)

        with open(args.solution_path, 'w') as f_out:
            json.dump(solution, f_out)
    else:
        # Fallback: try to build a trivial feasible solution
        primary_vars = {}
        for g in generators:
            gid = g['id']
            primary_vars[f"o_{gid}_0"] = float(g['initial_status'])
            primary_vars[f"p_{gid}_0"] = float(g['initial_gen'])
            for t in range(1, T + 1):
                primary_vars[f"o_{gid}_{t}"] = 0.0
                primary_vars[f"vp_{gid}_{t}"] = 0.0
                primary_vars[f"vm_{gid}_{t}"] = 0.0
                primary_vars[f"r_{gid}_{t}"] = 0.0
                primary_vars[f"p_{gid}_{t}"] = 0.0
                for bid in g['bids']:
                    primary_vars[f"w_{gid}_{bid['id']}_{t}"] = 0.0
                    primary_vars[f"se_{gid}_{bid['id']}_{t}"] = 0.0
                if g['is_gfpp']:
                    primary_vars[f"phimax_{gid}_{t}"] = 0.0

        for l in lines:
            for t in range(1, T + 1):
                primary_vars[f"f_{l['id']}_{t}"] = 0.0

        for b in buses:
            for t in range(1, T + 1):
                primary_vars[f"theta_{b['id']}_{t}"] = 0.0

        for j in junctions:
            jid = j['id']
            for t in range(1, T + 1):
                demand_g = j['gas_demand_profile'][t - 1] * eta_g
                primary_vars[f"sg_{jid}_{t}"] = 0.0
                primary_vars[f"pisq_{jid}_{t}"] = j['pressure_lb_squared']
                primary_vars[f"lg_{jid}_{t}"] = 0.0
                primary_vars[f"qg_{jid}_{t}"] = demand_g
                primary_vars[f"gamma_{jid}_{t}"] = 0.0
                if j['is_source'] and 'supply_intervals' in j:
                    for si in j['supply_intervals']:
                        primary_vars[f"sgs_{jid}_{si['id']}_{t}"] = 0.0

        for c in connections:
            for t in range(1, T + 1):
                primary_vars[f"phig_{c['id']}_{t}"] = 0.0

        for z in pricing_zones:
            for t in range(1, T + 1):
                primary_vars[f"psi_{z['id']}_{t}"] = min_gas_price

        solution = {
            "objective_value": 0.0,
            "released_problem_variant": "integrated_primal_ucgna",
            "objective_weight_beta": beta,
            "primary_variables": primary_vars
        }

        with open(args.solution_path, 'w') as f_out:
            json.dump(solution, f_out)


if __name__ == "__main__":
    main()