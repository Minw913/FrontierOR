import argparse
import json
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=300)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()
    t_start = time.time()

    import gurobipy as gp
    from gurobipy import GRB

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    with open(args.instance_path) as fh:
        inst = json.load(fh)

    T = int(inst['planning_horizon'])
    vessels = inst['vessels']
    V = len(vessels)

    pinfo = {}
    ports = []
    for p in inst['supply_ports']:
        pid = p['port_id']
        ports.append(pid)
        pinfo[pid] = {'type': 'supply', 'init': float(p['initial_inventory']),
                      'rate': p['production_rate'], 'cap': p['storage_capacity'],
                      'fmin': float(p['F_min']), 'fmax': float(p['F_max']),
                      'price': float(p['unit_procurement_cost'])}
    for p in inst['demand_ports']:
        pid = p['port_id']
        ports.append(pid)
        pinfo[pid] = {'type': 'demand', 'init': float(p['initial_inventory']),
                      'rate': p['consumption_rate'], 'cap': p['storage_capacity'],
                      'fmin': float(p['F_min']), 'fmax': float(p['F_max']),
                      'price': float(p['unit_supply_revenue'])}

    tt_raw = inst.get('travel_times', {})
    drafts = inst.get('draft_limits', {})

    def travel_time(i, j):
        key = "{}->{}".format(i, j)
        if key in tt_raw:
            return max(1, int(round(tt_raw[key])))
        key2 = "{}->{}".format(j, i)
        if key2 in tt_raw:
            return max(1, int(round(tt_raw[key2])))
        return 1

    # ---------------- Build MIP (time-space network per vessel) ----------------
    m = gp.Model('mirp')
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1

    varcs = []          # per vessel: list of arc dicts
    qvar = {}           # (vi, port, t) -> quantity variable
    port_q = {}         # (port, t) -> list of q vars (all vessels)

    for vi, v in enumerate(vessels):
        vid = v['id']
        cap = float(v['capacity'])
        sv = int(v['availability_start'])
        ev = int(v['availability_end'])
        if ev < sv:
            ev = sv
        il = float(v['initial_load'])
        tc = float(v['transport_cost_per_day'])
        dc = float(v['demurrage_cost_per_day'])
        dmap = {i: min(cap, float(drafts.get("{}->{}".format(vid, i), cap))) for i in ports}

        A = []
        nin = {}
        nout = {}

        def add_arc(tail, head, typ, cost, ub, fix):
            ai = len(A)
            A.append({'tail': tail, 'head': head, 'type': typ,
                      'cost': float(cost), 'ub': float(ub), 'fix': fix})
            nout.setdefault(tail, []).append(ai)
            nin.setdefault(head, []).append(ai)

        # unused-vessel escape arc
        add_arc('src', 'snk', 'source', 0.0, max(cap, il), True)
        # start arcs (respect draft with initial load)
        for i in ports:
            if dmap[i] >= il - 1e-9:
                add_arc('src', (i, sv), 'source', 0.0, max(cap, il), True)
        # demurrage and travel arcs
        for t in range(sv, ev + 1):
            for i in ports:
                if t < ev:
                    add_arc((i, t), (i, t + 1), 'demurrage', dc, cap, False)
                    for j in ports:
                        if j != i:
                            d = travel_time(i, j)
                            if t + d <= ev:
                                add_arc((i, t), (j, t + d), 'travel', tc * d, dmap[j], False)
        # sink arcs
        for i in ports:
            add_arc((i, ev), 'snk', 'sink', 0.0, cap, False)

        for a in A:
            a['x'] = m.addVar(vtype=GRB.BINARY, obj=a['cost'])
            a['f'] = m.addVar(lb=0.0, ub=a['ub'])
            if a['fix']:
                m.addConstr(a['f'] == il * a['x'])
            else:
                m.addConstr(a['f'] <= a['ub'] * a['x'])

        # vessel path conservation
        m.addConstr(gp.quicksum(A[ai]['x'] for ai in nout.get('src', [])) == 1)
        m.addConstr(gp.quicksum(A[ai]['x'] for ai in nin.get('snk', [])) == 1)

        nodes = set(nin.keys()) | set(nout.keys())
        for nd in nodes:
            if nd == 'src' or nd == 'snk':
                continue
            ins = nin.get(nd, [])
            outs = nout.get(nd, [])
            m.addConstr(gp.quicksum(A[ai]['x'] for ai in ins) ==
                        gp.quicksum(A[ai]['x'] for ai in outs))
            i, t = nd
            fexpr = gp.quicksum(A[ai]['f'] for ai in outs) - \
                    gp.quicksum(A[ai]['f'] for ai in ins)
            if 0 <= t < T:
                info = pinfo[i]
                qub = min(info['fmax'], cap)
                if qub <= 1e-9 or info['fmin'] > qub + 1e-9:
                    m.addConstr(fexpr == 0)
                    continue
                sign = 1.0 if info['type'] == 'supply' else -1.0
                obj_coef = info['price'] * (1.0 if info['type'] == 'supply' else -1.0)
                q = m.addVar(lb=0.0, ub=qub, obj=obj_coef)
                z = m.addVar(vtype=GRB.BINARY)
                m.addConstr(q <= qub * z)
                m.addConstr(q >= info['fmin'] * z)
                m.addConstr(z <= gp.quicksum(A[ai]['x'] for ai in outs))
                m.addConstr(fexpr == sign * q)
                qvar[(vi, i, t)] = q
                port_q.setdefault((i, t), []).append(q)
            else:
                m.addConstr(fexpr == 0)
        varcs.append(A)

    # port inventory dynamics
    for i in ports:
        info = pinfo[i]
        prev_var = None
        for t in range(T):
            ub = float(info['cap'][t])
            s = m.addVar(lb=0.0, ub=ub)
            qs = gp.quicksum(port_q.get((i, t), []))
            rate = float(info['rate'][t])
            base = prev_var if prev_var is not None else info['init']
            if info['type'] == 'supply':
                m.addConstr(s == base + rate - qs)
            else:
                m.addConstr(s == base - rate + qs)
            prev_var = s

    m.ModelSense = GRB.MINIMIZE

    # ---------------- Solution extraction ----------------
    def build_solution(xd, fd, qd):
        vessel_routes = {}
        vessel_ops = {}
        transport_cost = 0.0
        demurrage_cost = 0.0
        proc_cost = 0.0
        revenue = 0.0
        qsum = {}

        for vi, v in enumerate(vessels):
            vid = v['id']
            A = varcs[vi]
            used = {}
            for ai, a in enumerate(A):
                if xd.get((vi, ai), 0.0) > 0.5:
                    used[a['tail']] = (ai, a)

            def fmt(nd):
                if nd == 'src':
                    return "(src,{})".format(vid)
                if nd == 'snk':
                    return "(snk,{})".format(vid)
                return "({},{})".format(nd[0], nd[1])

            route = []
            cur = 'src'
            steps = 0
            while cur != 'snk' and steps < len(A) + 2:
                steps += 1
                if cur not in used:
                    break
                ai, a = used[cur]
                flow = max(0.0, fd.get((vi, ai), 0.0))
                route.append({'tail': fmt(a['tail']), 'head': fmt(a['head']),
                              'type': a['type'], 'cost': round(a['cost'], 6),
                              'product_flow': round(flow, 6)})
                if a['type'] == 'travel':
                    transport_cost += a['cost']
                elif a['type'] == 'demurrage':
                    demurrage_cost += a['cost']
                cur = a['head']
            vessel_routes[vid] = route

            ops = []
            keys = sorted([k for k in qd if k[0] == vi], key=lambda k: (k[2], k[1]))
            for (vv, i, t) in keys:
                val = qd[(vv, i, t)]
                if val > 1e-6:
                    info = pinfo[i]
                    qsum[(i, t)] = qsum.get((i, t), 0.0) + val
                    if info['type'] == 'supply':
                        ops.append({'port': i, 'time': t,
                                    'quantity': round(val, 6), 'type': 'load'})
                        proc_cost += info['price'] * val
                    else:
                        ops.append({'port': i, 'time': t,
                                    'quantity': round(-val, 6), 'type': 'discharge'})
                        revenue += info['price'] * val
            vessel_ops[vid] = ops

        port_inv = {}
        for i in ports:
            info = pinfo[i]
            s = info['init']
            lst = []
            for t in range(T):
                qv = qsum.get((i, t), 0.0)
                if info['type'] == 'supply':
                    s = s + float(info['rate'][t]) - qv
                else:
                    s = s - float(info['rate'][t]) + qv
                lst.append({'time': t, 'inventory': round(s, 6)})
            port_inv[i] = lst

        total = transport_cost + demurrage_cost + proc_cost - revenue
        return {
            'objective_value': round(total, 6),
            'vessel_routes': vessel_routes,
            'port_inventories': port_inv,
            'vessel_operations': vessel_ops,
            'cost_breakdown': {
                'transport_cost': round(transport_cost, 6),
                'demurrage_cost': round(demurrage_cost, 6),
                'procurement_cost': round(proc_cost, 6),
                'supply_revenue': round(revenue, 6),
                'total_cost': round(total, 6)
            }
        }

    # ---------------- Callback for incumbent logging ----------------
    x_keys = [(vi, ai) for vi in range(V) for ai in range(len(varcs[vi]))]
    x_vars = [varcs[vi][ai]['x'] for (vi, ai) in x_keys]
    f_vars = [varcs[vi][ai]['f'] for (vi, ai) in x_keys]
    q_keys = list(qvar.keys())
    q_vars = [qvar[k] for k in q_keys]

    best = {'obj': float('inf'), 'sol': None}

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best['obj'] - 1e-6:
                best['obj'] = obj
                try:
                    xv = model.cbGetSolution(x_vars)
                    fv = model.cbGetSolution(f_vars)
                    qv = model.cbGetSolution(q_vars)
                    xd = {x_keys[k]: xv[k] for k in range(len(x_keys))}
                    fd = {x_keys[k]: fv[k] for k in range(len(x_keys))}
                    qd = {q_keys[k]: qv[k] for k in range(len(q_keys))}
                    sol = build_solution(xd, fd, qd)
                    best['sol'] = sol
                    if logger:
                        logger.log_solution(sol['objective_value'], sol)
                except Exception:
                    if logger:
                        try:
                            logger.log(obj)
                        except Exception:
                            pass

    remaining = args.time_limit - (time.time() - t_start) - 3.0
    m.Params.TimeLimit = max(5.0, remaining)

    try:
        m.optimize(cb)
    except Exception:
        pass

    sol = None
    if m.SolCount > 0:
        try:
            xd = {x_keys[k]: x_vars[k].X for k in range(len(x_keys))}
            fd = {x_keys[k]: f_vars[k].X for k in range(len(x_keys))}
            qd = {q_keys[k]: q_vars[k].X for k in range(len(q_keys))}
            sol = build_solution(xd, fd, qd)
            if logger:
                logger.log_solution(sol['objective_value'], sol)
        except Exception:
            sol = best['sol']
    if sol is None:
        sol = best['sol']
    if sol is None:
        # trivial fallback: all vessels unused (arc 0 is src->snk)
        xd = {}
        fd = {}
        for vi in range(V):
            for ai in range(len(varcs[vi])):
                xd[(vi, ai)] = 1.0 if ai == 0 else 0.0
                fd[(vi, ai)] = float(vessels[vi]['initial_load']) if ai == 0 else 0.0
        sol = build_solution(xd, fd, {})
        if logger:
            try:
                logger.log_solution(sol['objective_value'], sol)
            except Exception:
                pass

    with open(args.solution_path, 'w') as fh:
        json.dump(sol, fh, indent=2)


if __name__ == '__main__':
    main()