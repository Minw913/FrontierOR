import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=600)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    t0 = time.time()

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="maximize")

    with open(args.instance_path, 'r') as f:
        inst = json.load(f)

    products = inst['products']
    facilities = inst['facilities']
    fac_ids = [fc['id'] for fc in facilities]
    cap = {fc['id']: fc['capacity'] for fc in facilities}
    price = {p['id']: p['sale_price'] for p in products}
    disc = {p['id']: p['discounted_price'] for p in products}
    pids = [p['id'] for p in products]

    pf = {}
    for e in inst['product_facility_data']:
        pf[(e['product_id'], e['facility_id'])] = e

    # Only yield_and_demand distributions are selectable (they carry level combos)
    dists = {pid: [] for pid in pids}
    for pd in inst['products_distributions']:
        pid = pd['product_id']
        if pid not in dists:
            dists[pid] = []
        for d in pd['distributions']:
            if d.get('type') == 'yield_and_demand' and d.get('level_combination') is not None:
                dists[pid].append(d)

    # ------------------------------------------------------------------
    # Build MIP
    # ------------------------------------------------------------------
    m = gp.Model('endogenous_production')
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1

    x = {}
    y = {}
    delta = {}
    z = {}
    w = {}
    o = {}
    xmax = {}

    for pid in pids:
        for fid in fac_ids:
            e = pf[(pid, fid)]
            ubmax = max(l['upper_bound'] for l in e['production_levels'])
            xm = min(cap[fid], ubmax)
            xmax[(pid, fid)] = xm
            x[(pid, fid)] = m.addVar(lb=0.0, ub=xm, name=f'x_{pid}_{fid}')
            for l in e['production_levels']:
                y[(pid, fid, l['level_id'])] = m.addVar(
                    vtype=GRB.BINARY, name=f'y_{pid}_{fid}_{l["level_id"]}')

    for pid in pids:
        for d in dists[pid]:
            did = d['distribution_id']
            delta[(pid, did)] = m.addVar(vtype=GRB.BINARY, name=f'd_{pid}_{did}')
            for s in d['scenarios']:
                sid = s['scenario_id']
                dem = max(0.0, float(s['demand']))
                z[(pid, did, sid)] = m.addVar(lb=0.0, name=f'z_{pid}_{did}_{sid}')
                w[(pid, did, sid)] = m.addVar(lb=0.0, ub=dem, name=f'w_{pid}_{did}_{sid}')
                o[(pid, did, sid)] = m.addVar(lb=0.0, name=f'o_{pid}_{did}_{sid}')

    # Level selection & bounds on x
    for pid in pids:
        for fid in fac_ids:
            e = pf[(pid, fid)]
            lvls = e['production_levels']
            m.addConstr(gp.quicksum(y[(pid, fid, l['level_id'])] for l in lvls) == 1)
            m.addConstr(x[(pid, fid)] <= gp.quicksum(
                l['upper_bound'] * y[(pid, fid, l['level_id'])] for l in lvls))
            m.addConstr(x[(pid, fid)] >= gp.quicksum(
                l['lower_bound'] * y[(pid, fid, l['level_id'])] for l in lvls))

    # Capacity
    for fid in fac_ids:
        m.addConstr(gp.quicksum(x[(pid, fid)] for pid in pids) <= cap[fid])

    # Distribution selection & linkage with levels
    for pid in pids:
        m.addConstr(gp.quicksum(delta[(pid, d['distribution_id'])] for d in dists[pid]) == 1)
        for d in dists[pid]:
            did = d['distribution_id']
            comb = d['level_combination']
            for i, fid in enumerate(fac_ids):
                lev = comb[i]
                m.addConstr(delta[(pid, did)] <= y[(pid, fid, lev)])

    # Scenario constraints
    for pid in pids:
        for d in dists[pid]:
            did = d['distribution_id']
            for s in d['scenarios']:
                sid = s['scenario_id']
                yl = s.get('yields', {}) or {}
                inv_expr = gp.LinExpr()
                M = 0.0
                for fid in fac_ids:
                    yv = float(yl.get(str(fid), 0.0))
                    if yv != 0.0:
                        inv_expr.add(x[(pid, fid)], yv)
                        M += abs(yv) * xmax[(pid, fid)]
                M = max(M, 1e-6)
                zv = z[(pid, did, sid)]
                dv = delta[(pid, did)]
                m.addConstr(zv <= inv_expr)
                m.addConstr(zv >= inv_expr - M * (1 - dv))
                m.addConstr(zv <= M * dv)
                m.addConstr(w[(pid, did, sid)] + o[(pid, did, sid)] == zv)

    # Objective
    rev = gp.LinExpr()
    for pid in pids:
        for d in dists[pid]:
            did = d['distribution_id']
            for s in d['scenarios']:
                sid = s['scenario_id']
                prob = float(s['probability'])
                rev.add(w[(pid, did, sid)], prob * price[pid])
                rev.add(o[(pid, did, sid)], prob * disc[pid])
    cost = gp.LinExpr()
    for pid in pids:
        for fid in fac_ids:
            cost.add(x[(pid, fid)], pf[(pid, fid)]['manufacturing_cost'])
    m.setObjective(rev - cost, GRB.MAXIMIZE)

    # ------------------------------------------------------------------
    # Solution extraction helpers
    # ------------------------------------------------------------------
    def build_solution(val):
        sol = {'x': {}, 'y': {}, 'delta': {}, 'z': {}, 'w': {}, 'o': {}}
        for (pid, fid), v in x.items():
            sol['x'][f'{pid}_{fid}'] = max(0.0, float(val(v)))
        for (pid, fid, lid), v in y.items():
            sol['y'][f'{pid}_{fid}_{lid}'] = int(round(val(v)))
        for (pid, did), v in delta.items():
            sol['delta'][f'{pid}_{did}'] = int(round(val(v)))
        for (pid, did, sid), v in z.items():
            sol['z'][f'{pid}_{did}_{sid}'] = max(0.0, float(val(v)))
        for (pid, did, sid), v in w.items():
            sol['w'][f'{pid}_{did}_{sid}'] = max(0.0, float(val(v)))
        for (pid, did, sid), v in o.items():
            sol['o'][f'{pid}_{did}_{sid}'] = max(0.0, float(val(v)))
        return sol

    def trivial_solution():
        sol = {'x': {}, 'y': {}, 'delta': {}, 'z': {}, 'w': {}, 'o': {}}
        for pid in pids:
            zero_levels = {}
            for fid in fac_ids:
                e = pf[(pid, fid)]
                zl = None
                for l in e['production_levels']:
                    if l['lower_bound'] == 0 and l['upper_bound'] == 0:
                        zl = l['level_id']
                        break
                if zl is None:
                    zl = e['production_levels'][0]['level_id']
                zero_levels[fid] = zl
                sol['x'][f'{pid}_{fid}'] = 0.0
                for l in e['production_levels']:
                    sol['y'][f'{pid}_{fid}_{l["level_id"]}'] = 1 if l['level_id'] == zl else 0
            chosen = None
            for d in dists[pid]:
                comb = d['level_combination']
                if all(comb[i] == zero_levels[fac_ids[i]] for i in range(len(fac_ids))):
                    chosen = d['distribution_id']
                    break
            if chosen is None and dists[pid]:
                chosen = dists[pid][0]['distribution_id']
            for d in dists[pid]:
                did = d['distribution_id']
                sol['delta'][f'{pid}_{did}'] = 1 if did == chosen else 0
                for s in d['scenarios']:
                    key = f'{pid}_{did}_{s["scenario_id"]}'
                    sol['z'][key] = 0.0
                    sol['w'][key] = 0.0
                    sol['o'][key] = 0.0
        return 0.0, sol

    var_list = m.getVars()
    best_obj = [-float('inf')]

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj > best_obj[0] + 1e-9:
                best_obj[0] = obj
                if logger:
                    try:
                        vals = model.cbGetSolution(var_list)
                        vmap = {vv: val for vv, val in zip(var_list, vals)}
                        sol = build_solution(lambda v: vmap[v])
                        logger.log_solution(obj, sol)
                    except Exception:
                        try:
                            logger.log(obj)
                        except Exception:
                            pass

    remaining = args.time_limit - (time.time() - t0) - 2.0
    m.Params.TimeLimit = max(1.0, remaining)

    m.optimize(cb)

    if m.SolCount > 0:
        obj_val = float(m.ObjVal)
        sol = build_solution(lambda v: v.X)
        if logger and obj_val > best_obj[0] + 1e-9:
            try:
                logger.log_solution(obj_val, sol)
            except Exception:
                pass
    else:
        obj_val, sol = trivial_solution()
        if logger:
            try:
                logger.log_solution(obj_val, sol)
            except Exception:
                pass

    out = {'objective_value': obj_val, 'solution': sol}
    with open(args.solution_path, 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()