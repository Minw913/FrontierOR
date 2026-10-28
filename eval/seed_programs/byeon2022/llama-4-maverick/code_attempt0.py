import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    model = gp.Model()
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = args.time_limit

    # Extract data from instance
    time_periods = instance['time_periods']
    electricity_network = instance['electricity_network']
    generators = instance['generators']['generators']
    gas_network = instance['gas_network']
    objective_weight_beta = instance['objective_weight_beta']

    # Create variables
    o = {(g['id'], t): model.addVar(vtype=GRB.BINARY) for g in generators for t in range(time_periods)}
    vp = {(g['id'], t): model.addVar(vtype=GRB.BINARY) for g in generators for t in range(time_periods)}
    vm = {(g['id'], t): model.addVar(vtype=GRB.BINARY) for g in generators for t in range(time_periods)}
    w = {(g['id'], b['id'], t): model.addVar(vtype=GRB.BINARY) for g in generators for b in g['bids'] for t in range(time_periods)}
    r = {(g['id'], t): model.addVar() for g in generators for t in range(time_periods)}
    phimax = {(g['id'], t): model.addVar() for g in generators if g['is_gfpp'] for t in range(time_periods)}
    p = {(g['id'], t): model.addVar() for g in generators for t in range(time_periods + 1)}
    se = {(g['id'], b['id'], t): model.addVar() for g in generators for b in g['bids'] for t in range(time_periods)}
    f = {(l['id'], t): model.addVar() for l in electricity_network['lines'] for t in range(time_periods)}
    theta = {(b['id'], t): model.addVar() for b in electricity_network['buses'] for t in range(time_periods)}
    sg = {(j['id'], t): model.addVar() for j in gas_network['junctions'] for t in range(time_periods)}
    sgs = {(j['id'], si['id'], t): model.addVar() for j in gas_network['junctions'] for si in j.get('supply_intervals', []) for t in range(time_periods)}
    pisq = {(j['id'], t): model.addVar() for j in gas_network['junctions'] for t in range(time_periods)}
    phig = {(c['id'], t): model.addVar() for c in gas_network['connections'] for t in range(time_periods)}
    lg = {(j['id'], t): model.addVar() for j in gas_network['junctions'] for t in range(time_periods)}
    qg = {(j['id'], t): model.addVar() for j in gas_network['junctions'] for t in range(time_periods)}
    gamma = {(j['id'], t): model.addVar() for j in gas_network['junctions'] for t in range(time_periods)}
    psi = {(pz['id'], t): model.addVar(lb=gas_network['min_gas_price_mmBtu'], ub=gas_network['max_gas_price_mmBtu']) for pz in gas_network['pricing_zones'] for t in range(time_periods)}

    # Add constraints
    for g in generators:
        for t in range(time_periods):
            # Initial status and minimum up/down time constraints
            if t == 0:
                model.addConstr(o[g['id'], t] == g['initial_status'])
            model.addConstr(vp[g['id'], t] - vm[g['id'], t] == o[g['id'], t] - (o[g['id'], t-1] if t > 0 else g['initial_status']))
            model.addConstr(vp[g['id'], t] + vm[g['id'], t] <= 1)
            for dt in range(1, g['min_up_time']):
                if t + dt < time_periods:
                    model.addConstr(o[g['id'], t + dt] >= vp[g['id'], t])
            for dt in range(1, g['min_down_time']):
                if t + dt < time_periods:
                    model.addConstr(1 - o[g['id'], t + dt] >= vm[g['id'], t])

            # Bid selection and generation constraints
            for b in g['bids']:
                model.addConstr(se[g['id'], b['id'], t] <= b['max_amount'] * w[g['id'], b['id'], t])
                if b['id'] > 0:
                    model.addConstr(w[g['id'], b['id'], t] <= w[g['id'], b['id']-1, t])
            model.addConstr(p[g['id'], t] == sum(se[g['id'], b['id'], t] for b in g['bids']))
            model.addConstr(p[g['id'], t] <= g['max_power'] * o[g['id'], t])
            model.addConstr(p[g['id'], t] >= g['min_power'] * o[g['id'], t])

            # Ramping constraints
            if t > 0:
                model.addConstr(p[g['id'], t] - p[g['id'], t-1] <= g['ramp_up'] * o[g['id'], t-1] + g['max_power'] * vp[g['id'], t])
                model.addConstr(p[g['id'], t-1] - p[g['id'], t] <= g['ramp_down'] * o[g['id'], t] + g['min_power'] * vm[g['id'], t])

            # GFPP specific constraints
            if g['is_gfpp']:
                model.addConstr(phimax[g['id'], t] == sum(b.get('profitability_threshold', 0) * (w[g['id'], b['id'], t] - w.get((g['id'], b['id']+1, t), 0)) for b in g['bids']))
                model.addConstr(phimax[g['id'], t] >= psi[pz['id'], t] * o[g['id'], t] for pz in gas_network['pricing_zones'] if g['gas_junction'] in pz['junctions'])

    for b in electricity_network['buses']:
        for t in range(time_periods):
            model.addConstr(sum(p[g['id'], t] for g in generators if g['bus'] == b['id']) - b['demand_profile'][t] == sum(f[l['id'], t] for l in electricity_network['lines'] if l['from_bus'] == b['id']) - sum(f[l['id'], t] for l in electricity_network['lines'] if l['to_bus'] == b['id']))

    for l in electricity_network['lines']:
        for t in range(time_periods):
            model.addConstr(f[l['id'], t] == l['susceptance'] * (theta[l['from_bus'], t] - theta[l['to_bus'], t]))
            model.addConstr(f[l['id'], t] <= l['thermal_limit'])
            model.addConstr(theta[l['from_bus'], t] - theta[l['to_bus'], t] <= l['angle_diff_limit'])

    for j in gas_network['junctions']:
        for t in range(time_periods):
            model.addConstr(sg[j['id'], t] - lg[j['id'], t] - gamma[j['id'], t] == sum(phig[c['id'], t] for c in gas_network['connections'] if c['to_junction'] == j['id']) - sum(phig[c['id'], t] for c in gas_network['connections'] if c['from_junction'] == j['id']))
            if j['is_source']:
                model.addConstr(sg[j['id'], t] == sum(sgs[j['id'], si['id'], t] for si in j['supply_intervals']))
                for si in j['supply_intervals']:
                    model.addConstr(sgs[j['id'], si['id'], t] <= si['interval_ub'])
            else:
                model.addConstr(sg[j['id'], t] == 0)
            model.addConstr(lg[j['id'], t] + qg[j['id'], t] == j['gas_demand_profile'][t])
            model.addConstr(qg[j['id'], t] <= j['gas_demand_profile'][t])

    for c in gas_network['connections']:
        for t in range(time_periods):
            if c['type'] == 'pipeline':
                model.addConstr(pisq[c['from_junction'], t] - pisq[c['to_junction'], t] >= c['weymouth_factor'] * phig[c['id'], t]**2)
            elif c['type'] == 'compressor':
                model.addConstr(pisq[c['to_junction'], t] >= c['compression_ratio_lb']**2 * pisq[c['from_junction'], t])
                model.addConstr(pisq[c['to_junction'], t] <= c['compression_ratio_ub']**2 * pisq[c['from_junction'], t])

    for g in generators:
        if g['is_gfpp']:
            for t in range(time_periods):
                model.addConstr(gamma[g['gas_junction'], t] >= g['heat_rate_coefficients']['H_u2'] * p[g['id'], t]**2 + g['heat_rate_coefficients']['H_u1'] * p[g['id'], t] + g['heat_rate_coefficients']['H_u0'] * o[g['id'], t])

    # Objective function
    electricity_cost = sum(g['no_load_cost'] * o[g['id'], t] + r[g['id'], t] + sum(b['price'] * se[g['id'], b['id'], t] for b in g['bids']) for g in generators for t in range(time_periods))
    gas_cost = sum(j['demand_shedding_cost'] * qg[j['id'], t] + sum(si['slope'] * sgs[j['id'], si['id'], t] for si in j.get('supply_intervals', [])) for j in gas_network['junctions'] for t in range(time_periods))
    model.setObjective(objective_weight_beta * electricity_cost + (1 - objective_weight_beta) * gas_cost, GRB.MINIMIZE)

    model.optimize()
    if logger:
        logger.log_solution(model.ObjVal, get_solution_dict(instance, o, vp, vm, w, r, phimax, p, se, f, theta, sg, sgs, pisq, phig, lg, qg, gamma, psi))

    with open(args.solution_path, 'w') as f:
        json.dump(get_solution_dict(instance, o, vp, vm, w, r, phimax, p, se, f, theta, sg, sgs, pisq, phig, lg, qg, gamma, psi), f)

def get_solution_dict(instance, o, vp, vm, w, r, phimax, p, se, f, theta, sg, sgs, pisq, phig, lg, qg, gamma, psi):
    solution = {
        'objective_value': o[0].getAttr('X') * instance['objective_weight_beta'] + (1 - instance['objective_weight_beta']) * sum(j['demand_shedding_cost'] * qg[j['id'], 0].getAttr('X') for j in instance['gas_network']['junctions']),
        'released_problem_variant': 'integrated_primal_ucgna',
        'objective_weight_beta': instance['objective_weight_beta'],
        'primary_variables': {}
    }
    time_periods = instance['time_periods']
    generators = instance['generators']['generators']
    electricity_network = instance['electricity_network']
    gas_network = instance['gas_network']

    for g in generators:
        for t in range(time_periods):
            solution['primary_variables'][f'o_{g["id"]}_{t}'] = o[g['id'], t].getAttr('X')
            solution['primary_variables'][f'vp_{g["id"]}_{t}'] = vp[g['id'], t].getAttr('X')
            solution['primary_variables'][f'vm_{g["id"]}_{t}'] = vm[g['id'], t].getAttr('X')
            for b in g['bids']:
                solution['primary_variables'][f'w_{g["id"]}_{b["id"]}_{t}'] = w[g['id'], b['id'], t].getAttr('X')
            solution['primary_variables'][f'r_{g["id"]}_{t}'] = r[g['id'], t].getAttr('X')
            solution['primary_variables'][f'p_{g["id"]}_{t}'] = p[g['id'], t].getAttr('X')
            for b in g['bids']:
                solution['primary_variables'][f'se_{g["id"]}_{b["id"]}_{t}'] = se[g['id'], b['id'], t].getAttr('X')
            if g['is_gfpp']:
                solution['primary_variables'][f'phimax_{g["id"]}_{t}'] = phimax[g['id'], t].getAttr('X')

    for l in electricity_network['lines']:
        for t in range(time_periods):
            solution['primary_variables'][f'f_{l["id"]}_{t}'] = f[l['id'], t].getAttr('X')

    for b in electricity_network['buses']:
        for t in range(time_periods):
            solution['primary_variables'][f'theta_{b["id"]}_{t}'] = theta[b['id'], t].getAttr('X')

    for j in gas_network['junctions']:
        for t in range(time_periods):
            solution['primary_variables'][f'sg_{j["id"]}_{t}'] = sg[j['id'], t].getAttr('X')
            for si in j.get('supply_intervals', []):
                solution['primary_variables'][f'sgs_{j["id"]}_{si["id"]}_{t}'] = sgs[j['id'], si['id'], t].getAttr('X')
            solution['primary_variables'][f'pisq['j["id"]}_{t}'] = pisq[j['id'], t].getAttr('X')
            solution['primary_variables'][f'lg_{j["id"]}_{t}'] = lg[j['id'], t].getAttr('X')
            solution['primary_variables'][f'qg_{j["id"]}_{t}'] = qg[j['id'], t].getAttr('X')
            solution['primary_variables'][f'gamma_{j["id"]}_{t}'] = gamma[j['id'], t].getAttr('X')

    for c in gas_network['connections']:
        for t in range(time_periods):
            solution['primary_variables'][f'phig_{c["id"]}_{t}'] = phig[c['id'], t].getAttr('X')

    for pz in gas_network['pricing_zones']:
        for t in range(time_periods):
            solution['primary_variables'][f'psi_{pz["id"]}_{t}'] = psi[pz['id'], t].getAttr('X')

    return solution

if __name__ == "__main__":
    main()