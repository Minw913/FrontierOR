import json
import argparse
import time
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger


def build_arcs(vessels, port_ids, T, travel_times):
    all_vessel_arcs = {}
    for v in vessels:
        vid = v['id']
        as_ = max(1, v['availability_start'])
        ae = min(T, v['availability_end'])
        tc = v['transport_cost_per_day']
        dc_cost = v['demurrage_cost_per_day']
        il = v['initial_load']
        arcs = []
        # source arcs
        for p in port_ids:
            arcs.append({'tail': f'src,{vid}', 'head': f'{p},{as_}', 'type': 'source',
                         'cost': 0.0, 'head_port': p, 'fixed_cargo': il})
        # sink arcs
        for p in port_ids:
            for t in range(as_, ae + 1):
                arcs.append({'tail': f'{p},{t}', 'head': f'snk,{vid}', 'type': 'sink',
                             'cost': 0.0, 'head_port': None, 'fixed_cargo': None})
        # demurrage arcs
        for p in port_ids:
            for t in range(1, T):
                arcs.append({'tail': f'{p},{t}', 'head': f'{p},{t+1}', 'type': 'demurrage',
                             'cost': dc_cost, 'head_port': p, 'fixed_cargo': None})
        # travel arcs
        for p in port_ids:
            for q in port_ids:
                if p == q:
                    continue
                key = f'{p}->{q}'
                if key not in travel_times:
                    continue
                tt = travel_times[key]
                if not tt or tt < 1:
                    continue
                for t in range(1, T + 1):
                    if t + tt <= T:
                        arcs.append({'tail': f'{p},{t}', 'head': f'{q},{t+tt}', 'type': 'travel',
                                     'cost': tc * tt, 'head_port': q, 'fixed_cargo': None})
        all_vessel_arcs[vid] = arcs
    return all_vessel_arcs


def build_solution_from_values(xv, fv, lv, dv, all_vessel_arcs, vessels, ports_info, port_ids, is_supply, T):
    vessel_routes = {}
    vessel_operations = {}
    for v in vessels:
        vid = v['id']
        arcs = all_vessel_arcs[vid]
        selected = [i for i in range(len(arcs)) if xv.get((vid, i), 0.0) > 0.5]
        out_map = {}
        for i in selected:
            a = arcs[i]
            out_map.setdefault(a['tail'], []).append(i)
        route = []
        cur = f'src,{vid}'
        guard = 0
        while cur != f'snk,{vid}' and guard < len(arcs) + 10:
            guard += 1
            nexts = out_map.get(cur, [])
            if not nexts:
                break
            i = nexts[0]
            a = arcs[i]
            route.append({
                'tail': a['tail'],
                'head': a['head'],
                'type': a['type'],
                'cost': a['cost'],
                'product_flow': float(fv.get((vid, i), 0.0))
            })
            cur = a['head']
        vessel_routes[vid] = route
        ops = []
        for p in port_ids:
            for t in range(1, T + 1):
                if is_supply[p]:
                    val = lv.get((vid, p, t), 0.0)
                    if val > 1e-6:
                        ops.append({'port': p, 'time': t, 'quantity': float(val), 'type': 'load'})
                else:
                    val = dv.get((vid, p, t), 0.0)
                    if val > 1e-6:
                        ops.append({'port': p, 'time': t, 'quantity': -float(val), 'type': 'discharge'})
        vessel_operations[vid] = ops

    port_inventories = {}
    for p in port_ids:
        inv_list = []
        I = float(ports_info[p]['initial_inventory'])
        for t in range(1, T + 1):
            if is_supply[p]:
                prod = ports_info[p]['production_rate'][t - 1]
                tl = sum(lv.get((vv['id'], p, t), 0.0) for vv in vessels)
                I = I + prod - tl
            else:
                cons = ports_info[p]['consumption_rate'][t - 1]
                td = sum(dv.get((vv['id'], p, t), 0.0) for vv in vessels)
                I = I - cons + td
            inv_list.append(float(I))
        port_inventories[p] = inv_list

    transport_cost = 0.0
    demurrage_cost = 0.0
    procurement_cost = 0.0
    supply_revenue = 0.0
    for v in vessels:
        vid = v['id']
        arcs = all_vessel_arcs[vid]
        for i, a in enumerate(arcs):
            if xv.get((vid, i), 0.0) > 0.5:
                if a['type'] == 'travel':
                    transport_cost += a['cost']
                elif a['type'] == 'demurrage':
                    demurrage_cost += a['cost']
    for p in port_ids:
        if is_supply[p]:
            uc = ports_info[p]['unit_procurement_cost']
            for vv in vessels:
                for t in range(1, T + 1):
                    procurement_cost += lv.get((vv['id'], p, t), 0.0) * uc
        else:
            ur = ports_info[p]['unit_supply_revenue']
            for vv in vessels:
                for t in range(1, T + 1):
                    supply_revenue += dv.get((vv['id'], p, t), 0.0) * ur
    total_cost = transport_cost + demurrage_cost + procurement_cost - supply_revenue
    return {
        'objective_value': float(total_cost),
        'vessel_routes': vessel_routes,
        'port_inventories': port_inventories,
        'vessel_operations': vessel_operations,
        'cost_breakdown': {
            'transport_cost': float(transport_cost),
            'demurrage_cost': float(demurrage_cost),
            'procurement_cost': float(procurement_cost),
            'supply_revenue': float(supply_revenue),
            'total_cost': float(total_cost)
        }
    }


def build_trivial_solution(vessels, ports_info, port_ids, is_supply, T, all_vessel_arcs):
    vessel_routes = {}
    vessel_operations = {}
    for v in vessels:
        vid = v['id']
        arcs = all_vessel_arcs[vid]
        as_ = max(1, v['availability_start'])
        ae = min(T, v['availability_end'])
        p0 = port_ids[0] if port_ids else None
        route = []
        if p0 is not None:
            src_idx = None
            snk_idx = None
            for i, a in enumerate(arcs):
                if a['type'] == 'source' and a['head'] == f'{p0},{as_}':
                    src_idx = i
                if a['type'] == 'sink' and a['tail'] == f'{p0},{as_}' and as_ <= ae:
                    snk_idx = i
            if src_idx is not None and snk_idx is not None:
                il = v['initial_load']
                route = [
                    {'tail': arcs[src_idx]['tail'], 'head': arcs[src_idx]['head'], 'type': 'source',
                     'cost': 0.0, 'product_flow': float(il)},
                    {'tail': arcs[snk_idx]['tail'], 'head': arcs[snk_idx]['head'], 'type': 'sink',
                     'cost': 0.0, 'product_flow': float(il)}
                ]
        vessel_routes[vid] = route
        vessel_operations[vid] = []
    port_inventories = {}
    for p in port_ids:
        inv_list = []
        I = float(ports_info[p]['initial_inventory'])
        for t in range(1, T + 1):
            if is_supply[p]:
                I = I + ports_info[p]['production_rate'][t - 1]
            else:
                I = I - ports_info[p]['consumption_rate'][t - 1]
            inv_list.append(float(I))
        port_inventories[p] = inv_list
    return {
        'objective_value': 0.0,
        'vessel_routes': vessel_routes,
        'port_inventories': port_inventories,
        'vessel_operations': vessel_operations,
        'cost_breakdown': {
            'transport_cost': 0.0,
            'demurrage_cost': 0.0,
            'procurement_cost': 0.0,
            'supply_revenue': 0.0,
            'total_cost': 0.0
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    with open(args.instance_path) as fp:
        inst = json.load(fp)

    T = inst['planning_horizon']
    vessels = inst['vessels']
    supply_ports = inst['supply_ports']
    demand_ports = inst['demand_ports']

    ports_info = {}
    for p in supply_ports:
        ports_info[p['port_id']] = p
    for p in demand_ports:
        ports_info[p['port_id']] = p
    port_ids = list(ports_info.keys())
    is_supply = {pid: ports_info[pid]['type'] == 'supply' for pid in port_ids}

    travel_times = inst.get('travel_times', {})
    draft_limits = inst.get('draft_limits', {})

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    t_start = time.time()
    all_vessel_arcs = build_arcs(vessels, port_ids, T, travel_times)

    model = Model("mirp")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0
    elapsed = time.time() - t_start
    model.Params.TimeLimit = max(1, args.time_limit - elapsed - 5)

    # Variables
    x = {}
    f = {}
    for v in vessels:
        vid = v['id']
        cap = v['capacity']
        for i, a in enumerate(all_vessel_arcs[vid]):
            x[(vid, i)] = model.addVar(vtype=GRB.BINARY, name=f'x_{vid}_{i}')
            f[(vid, i)] = model.addVar(lb=0, ub=cap, name=f'f_{vid}_{i}')

    visit = {}
    load = {}
    disch = {}
    op = {}
    for v in vessels:
        vid = v['id']
        for p in port_ids:
            for t in range(1, T + 1):
                visit[(vid, p, t)] = model.addVar(vtype=GRB.BINARY, name=f'vis_{vid}_{p}_{t}')
                op[(vid, p, t)] = model.addVar(vtype=GRB.BINARY, name=f'op_{vid}_{p}_{t}')
                if is_supply[p]:
                    load[(vid, p, t)] = model.addVar(lb=0, name=f'ld_{vid}_{p}_{t}')
                else:
                    disch[(vid, p, t)] = model.addVar(lb=0, name=f'dc_{vid}_{p}_{t}')

    port_inv = {}
    for p in port_ids:
        port_inv[p] = model.addVars(range(T + 1), lb=0, name=f'inv_{p}')

    model.update()

    # Source / sink
    for v in vessels:
        vid = v['id']
        arcs = all_vessel_arcs[vid]
        source_idxs = [i for i, a in enumerate(arcs) if a['type'] == 'source']
        sink_idxs = [i for i, a in enumerate(arcs) if a['type'] == 'sink']
        model.addConstr(quicksum(x[(vid, i)] for i in source_idxs) == 1, name=f'src_{vid}')
        model.addConstr(quicksum(x[(vid, i)] for i in sink_idxs) == 1, name=f'snk_{vid}')
        for i in source_idxs:
            il = arcs[i]['fixed_cargo']
            model.addConstr(f[(vid, i)] == il * x[(vid, i)], name=f'sf_{vid}_{i}')

    # Flow conservation and cargo continuity
    for v in vessels:
        vid = v['id']
        arcs = all_vessel_arcs[vid]
        nin = {}
        nout = {}
        for i, a in enumerate(arcs):
            nin.setdefault(a['head'], []).append(i)
            nout.setdefault(a['tail'], []).append(i)
        for t in range(1, T + 1):
            for p in port_ids:
                node = f'{p},{t}'
                ins = nin.get(node, [])
                outs = nout.get(node, [])
                model.addConstr(quicksum(x[(vid, i)] for i in ins) == visit[(vid, p, t)], name=f'vin_{vid}_{p}_{t}')
                model.addConstr(quicksum(x[(vid, i)] for i in outs) == visit[(vid, p, t)], name=f'vout_{vid}_{p}_{t}')
                fin = quicksum(f[(vid, i)] for i in ins)
                fout = quicksum(f[(vid, i)] for i in outs)
                if is_supply[p]:
                    ld = load[(vid, p, t)]
                    dc_term = 0
                else:
                    ld = 0
                    dc_term = disch[(vid, p, t)]
                model.addConstr(fin + ld - dc_term == fout, name=f'cargo_{vid}_{p}_{t}')

    # Capacity and draft limits
    for v in vessels:
        vid = v['id']
        cap = v['capacity']
        arcs = all_vessel_arcs[vid]
        for i, a in enumerate(arcs):
            model.addConstr(f[(vid, i)] <= cap * x[(vid, i)], name=f'fcap_{vid}_{i}')
            if a['type'] in ('source', 'travel'):
                p = a['head_port']
                dl = draft_limits.get(f'{vid}->{p}', cap)
                if dl < cap:
                    model.addConstr(f[(vid, i)] <= dl * x[(vid, i)], name=f'draft_{vid}_{i}')

    # Operation linking
    for v in vessels:
        vid = v['id']
        for p in port_ids:
            fmin = ports_info[p]['F_min']
            fmax = ports_info[p]['F_max']
            for t in range(1, T + 1):
                model.addConstr(op[(vid, p, t)] <= visit[(vid, p, t)], name=f'opv_{vid}_{p}_{t}')
                if is_supply[p]:
                    ld = load[(vid, p, t)]
                    if fmax > 0:
                        model.addConstr(ld <= fmax * op[(vid, p, t)], name=f'ldm_{vid}_{p}_{t}')
                    model.addConstr(ld >= fmin * op[(vid, p, t)], name=f'ldn_{vid}_{p}_{t}')
                else:
                    dc = disch[(vid, p, t)]
                    if fmax > 0:
                        model.addConstr(dc <= fmax * op[(vid, p, t)], name=f'dcm_{vid}_{p}_{t}')
                    model.addConstr(dc >= fmin * op[(vid, p, t)], name=f'dcn_{vid}_{p}_{t}')

    # Port inventory
    for p in port_ids:
        model.addConstr(port_inv[p][0] == ports_info[p]['initial_inventory'], name=f'inv0_{p}')
        if is_supply[p]:
            for t in range(1, T + 1):
                prod = ports_info[p]['production_rate'][t - 1]
                total_load = quicksum(load[(vv['id'], p, t)] for vv in vessels)
                model.addConstr(port_inv[p][t] == port_inv[p][t - 1] + prod - total_load, name=f'invb_{p}_{t}')
        else:
            for t in range(1, T + 1):
                cons = ports_info[p]['consumption_rate'][t - 1]
                total_dc = quicksum(disch[(vv['id'], p, t)] for vv in vessels)
                model.addConstr(port_inv[p][t] == port_inv[p][t - 1] - cons + total_dc, name=f'invb_{p}_{t}')
        for t in range(1, T + 1):
            sc = ports_info[p]['storage_capacity'][t - 1]
            model.addConstr(port_inv[p][t] <= sc, name=f'invc_{p}_{t}')

    # Objective
    transport = quicksum()
    demurr = quicksum()
    for v in vessels:
        vid = v['id']
        arcs = all_vessel_arcs[vid]
        for i, a in enumerate(arcs):
            if a['type'] == 'travel':
                transport += a['cost'] * x[(vid, i)]
            elif a['type'] == 'demurrage':
                demurr += a['cost'] * x[(vid, i)]
    proc = quicksum()
    rev = quicksum()
    for p in port_ids:
        if is_supply[p]:
            uc = ports_info[p]['unit_procurement_cost']
            for vv in vessels:
                for t in range(1, T + 1):
                    proc += uc * load[(vv['id'], p, t)]
        else:
            ur = ports_info[p]['unit_supply_revenue']
            for vv in vessels:
                for t in range(1, T + 1):
                    rev += ur * disch[(vv['id'], p, t)]
    model.setObjective(transport + demurr + proc - rev, GRB.MINIMIZE)

    # Prepare var lists for callback
    x_items = list(x.items())
    f_items = list(f.items())
    load_items = list(load.items())
    disch_items = list(disch.items())
    x_vars_list = [var for _, var in x_items]
    f_vars_list = [var for _, var in f_items]
    load_vars_list = [var for _, var in load_items]
    disch_vars_list = [var for _, var in disch_items]
    x_keys = [k for k, _ in x_items]
    f_keys = [k for k, _ in f_items]
    load_keys = [k for k, _ in load_items]
    disch_keys = [k for k, _ in disch_items]

    best_obj = [float('inf')]
    best_solution = [None]

    def callback(cb_model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj < best_obj[0]:
                    x_vals = cb_model.cbGetSolution(x_vars_list)
                    f_vals = cb_model.cbGetSolution(f_vars_list)
                    lv_vals = cb_model.cbGetSolution(load_vars_list) if load_vars_list else []
                    dv_vals = cb_model.cbGetSolution(disch_vars_list) if disch_vars_list else []
                    xv = {k: v for k, v in zip(x_keys, x_vals)}
                    fv = {k: v for k, v in zip(f_keys, f_vals)}
                    lv = {k: v for k, v in zip(load_keys, lv_vals)} if load_keys else {}
                    dv = {k: v for k, v in zip(disch_keys, dv_vals)} if disch_keys else {}
                    sol = build_solution_from_values(xv, fv, lv, dv, all_vessel_arcs, vessels,
                                                     ports_info, port_ids, is_supply, T)
                    best_obj[0] = obj
                    best_solution[0] = sol
                    if logger:
                        logger.log_solution(obj, sol)
            except Exception:
                pass

    model.optimize(callback)

    # Extract final solution
    if model.SolCount > 0:
        xv = {}
        fv = {}
        lv = {}
        dv = {}
        for k, var in x.items():
            xv[k] = var.X
        for k, var in f.items():
            fv[k] = var.X
        for k, var in load.items():
            lv[k] = var.X
        for k, var in disch.items():
            dv[k] = var.X
        sol = build_solution_from_values(xv, fv, lv, dv, all_vessel_arcs, vessels,
                                         ports_info, port_ids, is_supply, T)
        if sol['objective_value'] < best_obj[0]:
            best_solution[0] = sol
            best_obj[0] = sol['objective_value']
    elif best_solution[0] is not None:
        sol = best_solution[0]
    else:
        sol = build_trivial_solution(vessels, ports_info, port_ids, is_supply, T, all_vessel_arcs)

    if logger and best_solution[0] is not None:
        logger.log_solution(best_obj[0], best_solution[0])

    with open(args.solution_path, 'w') as fp:
        json.dump(sol, fp, indent=2)


if __name__ == '__main__':
    main()