import json
import argparse
import time
import random
from collections import defaultdict, deque


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    t_start = time.time()
    time_limit = float(args.time_limit)

    with open(args.instance_path) as f:
        inst = json.load(f)

    containers = inst.get('containers', [])
    trades = inst.get('trades', [])

    cont_ids = [c['id'] for c in containers]
    cont_id_set = set(cont_ids)
    avail_start = {c['id']: c['availability_start'] for c in containers}
    avail_end = {c['id']: c['availability_end'] for c in containers}

    # Symmetrized adjacency
    adj_sets = defaultdict(set)
    for c in containers:
        cid = c['id']
        for a in (c.get('adjacent_containers') or []):
            if a == cid or a not in cont_id_set:
                continue
            adj_sets[cid].add(a)
            adj_sets[a].add(cid)
    adj = {cid: sorted(adj_sets.get(cid, ())) for cid in cont_ids}
    # Heuristic adjacency ordered by long remaining availability (stability)
    adj_h = {cid: sorted(adj[cid], key=lambda v: (-avail_end.get(v, 0), v)) for cid in cont_ids}

    trade_periods = {t['id']: list(range(t['start_period'], t['end_period'] + 1)) for t in trades}
    trade_dem = {t['id']: t['container_demand'] for t in trades}
    trade_k = {t['id']: min(t['max_dispersion'], t['container_demand']) for t in trades}

    all_periods = sorted({p for ps in trade_periods.values() for p in ps})
    avail_per_period = {
        p: [cid for cid in cont_ids if avail_start[cid] <= p <= avail_end[cid]]
        for p in all_periods
    }
    arcs_per_period = {}
    incoming_per_period = {}
    for p in all_periods:
        nodes = avail_per_period[p]
        nset = set(nodes)
        arcs = []
        incoming = defaultdict(list)
        for a in nodes:
            for b in adj[a]:
                if b in nset:
                    arcs.append((a, b))
                    incoming[b].append(a)
        arcs_per_period[p] = arcs
        incoming_per_period[p] = dict(incoming)

    # ---------------- helpers ----------------
    def count_components(sel):
        sel = set(sel)
        seen = set()
        comps = 0
        for c0 in sel:
            if c0 in seen:
                continue
            comps += 1
            dq = deque([c0])
            seen.add(c0)
            while dq:
                u = dq.popleft()
                for v in adj.get(u, ()):
                    if v in sel and v not in seen:
                        seen.add(v)
                        dq.append(v)
        return comps

    def check_feasible(assignments):
        for t in trades:
            i = t['id']
            dem = trade_dem[i]
            k = trade_k[i]
            for p in trade_periods[i]:
                sel = assignments.get(i, {}).get(p, [])
                if len(sel) != dem or len(set(sel)) != dem:
                    return False
                for c in sel:
                    if not (avail_start[c] <= p <= avail_end[c]):
                        return False
                if count_components(sel) > k:
                    return False
        for p in all_periods:
            used = set()
            for t in trades:
                if t['start_period'] <= p <= t['end_period']:
                    for c in assignments.get(t['id'], {}).get(p, []):
                        if c in used:
                            return False
                        used.add(c)
        return True

    def compute_objective(assignments):
        obj = 0
        for t in trades:
            i = t['id']
            prev = None
            for p in trade_periods[i]:
                cur = tuple(assignments.get(i, {}).get(p, []))
                if prev is not None and cur != prev:
                    obj += 1
                prev = cur
        return obj

    def to_output(assignments, obj):
        return {
            "objective_value": float(obj),
            "assignments": {
                str(t['id']): {
                    str(p): list(assignments.get(t['id'], {}).get(p, []))
                    for p in trade_periods[t['id']]
                }
                for t in trades
            }
        }

    def assignments_from_vals(x_index, vals):
        assignments = {t['id']: {p: [] for p in trade_periods[t['id']]} for t in trades}
        for (i, c, p), v in zip(x_index, vals):
            if v > 0.5:
                assignments[i][p].append(c)
        for i in assignments:
            for p in assignments[i]:
                assignments[i][p].sort()
        return assignments

    log_state = {'last': float('inf')}

    def do_log(obj, assignments):
        if logger is None:
            return
        try:
            if obj < log_state['last'] - 1e-9:
                logger.log_solution(float(obj), to_output(assignments, obj))
                log_state['last'] = obj
        except Exception:
            pass

    # ---------------- constructive heuristic ----------------
    def construct(rng):
        assignments = {t['id']: {} for t in trades}
        prev_set = {t['id']: set() for t in trades}
        for p in all_periods:
            active = [t for t in trades if t['start_period'] <= p <= t['end_period']]
            if rng is not None:
                rng.shuffle(active)
                active.sort(key=lambda t: -t['container_demand'])
            else:
                active.sort(key=lambda t: (-t['container_demand'], t['max_dispersion'], t['id']))
            free = set(avail_per_period[p])
            for t in active:
                i = t['id']
                dem = trade_dem[i]
                k = trade_k[i]
                chosen = []
                chosen_set = set()
                # keep previous containers that are still available
                for c in sorted(prev_set[i]):
                    if len(chosen) >= dem:
                        break
                    if c in free:
                        chosen.append(c)
                        chosen_set.add(c)
                # grow around kept containers (no new components)
                frontier = list(chosen)
                while len(chosen) < dem and frontier:
                    nxt = []
                    for u in frontier:
                        for v in adj_h.get(u, ()):
                            if v in free and v not in chosen_set:
                                chosen.append(v)
                                chosen_set.add(v)
                                nxt.append(v)
                                if len(chosen) >= dem:
                                    break
                        if len(chosen) >= dem:
                            break
                    frontier = nxt
                # open new components while allowed
                while len(chosen) < dem:
                    if count_components(chosen_set) >= k:
                        break
                    cand = [c for c in free if c not in chosen_set]
                    if not cand:
                        break
                    if rng is not None:
                        seed = rng.choice(cand)
                    else:
                        seed = max(
                            cand,
                            key=lambda c: (
                                sum(1 for v in adj.get(c, ()) if v in free and v not in chosen_set),
                                avail_end.get(c, 0),
                            ),
                        )
                    stack = [seed]
                    while len(chosen) < dem and stack:
                        u = stack.pop()
                        if u in free and u not in chosen_set:
                            chosen.append(u)
                            chosen_set.add(u)
                            for v in adj_h.get(u, ()):
                                if v in free and v not in chosen_set:
                                    stack.append(v)
                # last-resort fill (may violate dispersion; only as fallback)
                if len(chosen) < dem:
                    for c in sorted(free):
                        if c not in chosen_set:
                            chosen.append(c)
                            chosen_set.add(c)
                            if len(chosen) >= dem:
                                break
                for c in chosen:
                    free.discard(c)
                assignments[i][p] = sorted(chosen)
                prev_set[i] = set(chosen)
        return assignments

    best_assign = None
    best_obj = float('inf')
    best_feasible = False

    cand = construct(None)
    last_cand = cand
    if check_feasible(cand):
        o = compute_objective(cand)
        best_assign, best_obj, best_feasible = cand, o, True

    heur_deadline = t_start + min(8.0, max(1.0, 0.05 * time_limit))
    rng = random.Random(0)
    trials = 0
    while time.time() < heur_deadline and trials < 300:
        trials += 1
        cand = construct(rng)
        last_cand = cand
        if check_feasible(cand):
            o = compute_objective(cand)
            if o < best_obj:
                best_assign, best_obj, best_feasible = cand, o, True
        if best_feasible and best_obj == 0:
            break

    if best_assign is None:
        best_assign = last_cand

    if best_feasible:
        do_log(best_obj, best_assign)

    # ---------------- MIP ----------------
    mip_assign = None
    remaining = time_limit - (time.time() - t_start) - 2.0
    if remaining > 1.0 and trades and cont_ids:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            model = gp.Model('container_unit_assignment')
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1

            x_index = []
            for t in trades:
                i = t['id']
                for p in trade_periods[i]:
                    for c in avail_per_period[p]:
                        x_index.append((i, c, p))
            x = model.addVars(x_index, vtype=GRB.BINARY, name='x')

            y_index = []
            for t in trades:
                i = t['id']
                for p in trade_periods[i][1:]:
                    y_index.append((i, p))
            y = model.addVars(y_index, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name='y')

            by_ip = defaultdict(list)
            by_cp = defaultdict(list)
            for key in x_index:
                by_ip[(key[0], key[2])].append(key)
                by_cp[(key[1], key[2])].append(key)

            # demand constraints
            for (i, p), keys in by_ip.items():
                model.addConstr(gp.quicksum(x[kk] for kk in keys) == trade_dem[i],
                                name='dem_%s_%s' % (i, p))
            # container capacity constraints
            for (c, p), keys in by_cp.items():
                if len(keys) > 1:
                    model.addConstr(gp.quicksum(x[kk] for kk in keys) <= 1,
                                    name='cap_%s_%s' % (c, p))
            # re-assignment detection: y=1 iff a new container enters the trade
            x_key_set = set(x_index)
            for (i, p) in y_index:
                yv = y[(i, p)]
                for key in by_ip[(i, p)]:
                    c = key[1]
                    pk = (i, c, p - 1)
                    if pk in x_key_set:
                        model.addConstr(yv >= x[key] - x[pk])
                    else:
                        model.addConstr(yv >= x[key])

            model.setObjective(gp.quicksum(y[kk] for kk in y_index), GRB.MINIMIZE)

            # which (trade, period) need explicit dispersion constraints
            need_pairs = []
            total_d = 0
            for t in trades:
                i = t['id']
                dem = trade_dem[i]
                k = trade_k[i]
                if k >= dem:
                    continue
                for p in trade_periods[i]:
                    nodes = avail_per_period[p]
                    if len(nodes) <= k:
                        continue
                    need_pairs.append((i, p))
                    total_d += len(arcs_per_period[p])

            rem_now = time_limit - (time.time() - t_start) - 2.0
            use_lazy = (total_d > 120000) or (len(x_index) + 4 * total_d > 700000)
            if not use_lazy and total_d > 0:
                est = (3.0 * total_d + 2.0 * len(x_index)) / 40000.0
                if est > max(5.0, 0.4 * rem_now):
                    use_lazy = True

            if not use_lazy:
                # Forest formulation of "<= k connected components":
                # a spanning forest of the selected induced subgraph with
                # >= dem - k edges exists iff components <= k.
                d_index = []
                for (i, p) in need_pairs:
                    for (a, b) in arcs_per_period[p]:
                        d_index.append((i, p, a, b))
                d = model.addVars(d_index, vtype=GRB.BINARY, name='d')
                arcs_by_ip = defaultdict(list)
                for (i, p, a, b) in d_index:
                    arcs_by_ip[(i, p)].append((a, b))
                for (i, p), arcs in arcs_by_ip.items():
                    nodes = avail_per_period[p]
                    n = len(nodes)
                    lev = model.addVars(nodes, vtype=GRB.CONTINUOUS, lb=0.0,
                                        ub=float(max(n - 1, 0)),
                                        name='lev_%s_%s' % (i, p))
                    incoming = incoming_per_period[p]
                    dem = trade_dem[i]
                    k = trade_k[i]
                    for (a, b) in arcs:
                        dv = d[(i, p, a, b)]
                        model.addConstr(dv <= x[(i, a, p)])
                        model.addConstr(dv <= x[(i, b, p)])
                        model.addConstr(lev[b] - lev[a] >= 1 - n * (1 - dv))
                    for a in nodes:
                        ins = incoming.get(a)
                        if ins:
                            model.addConstr(
                                gp.quicksum(d[(i, p, u, a)] for u in ins) <= x[(i, a, p)]
                            )
                    model.addConstr(
                        gp.quicksum(d[(i, p, a, b)] for (a, b) in arcs) >= dem - k
                    )
            else:
                model.Params.LazyConstraints = 1

            # MIP start from heuristic
            if best_feasible:
                heur_sets = {
                    t['id']: {p: set(best_assign[t['id']].get(p, [])) for p in trade_periods[t['id']]}
                    for t in trades
                }
                for key in x_index:
                    i, c, p = key
                    x[key].Start = 1.0 if c in heur_sets[i].get(p, ()) else 0.0
                for (i, p) in y_index:
                    cur = heur_sets[i].get(p, set())
                    prev = heur_sets[i].get(p - 1, set())
                    y[(i, p)].Start = 1.0 if cur != prev else 0.0

            x_var_list = [x[kk] for kk in x_index]
            cb_best = {'best': best_obj if best_feasible else float('inf')}

            def cb(m, where):
                if where != GRB.Callback.MIPSOL:
                    return
                try:
                    try:
                        vals = list(m.cbGetSolution(x_var_list))
                    except Exception:
                        vals = [m.cbGetSolution(v) for v in x_var_list]
                    if use_lazy:
                        sel_map = defaultdict(list)
                        for (i, c, p), v in zip(x_index, vals):
                            if v > 0.5:
                                sel_map[(i, p)].append(c)
                        violated = False
                        for (i, p), sel in sel_map.items():
                            kk = trade_k[i]
                            dem = trade_dem[i]
                            if kk >= dem or len(sel) <= kk:
                                continue
                            if count_components(sel) > kk:
                                m.cbLazy(gp.quicksum(x[(i, c, p)] for c in sel) <= dem - 1)
                                violated = True
                        if violated:
                            return
                    obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj < cb_best['best'] - 1e-6:
                        cb_best['best'] = obj
                        sol = assignments_from_vals(x_index, vals)
                        o2 = compute_objective(sol)
                        do_log(o2, sol)
                except Exception:
                    pass

            rem2 = time_limit - (time.time() - t_start) - 1.5
            if rem2 > 0.3:
                model.Params.TimeLimit = rem2
                model.optimize(cb)
                if model.SolCount > 0:
                    if x_var_list:
                        vals = list(model.getAttr('X', x_var_list))
                    else:
                        vals = []
                    mip_assign = assignments_from_vals(x_index, vals)
        except Exception:
            mip_assign = None

    # ---------------- final selection ----------------
    final_assign = best_assign
    final_obj = best_obj
    if mip_assign is not None and check_feasible(mip_assign):
        mo = compute_objective(mip_assign)
        if (not best_feasible) or mo < best_obj:
            final_assign = mip_assign
            final_obj = mo
            best_feasible = True
    if not best_feasible:
        final_obj = compute_objective(final_assign)

    do_log(final_obj, final_assign)

    out = to_output(final_assign, final_obj)
    with open(args.solution_path, 'w') as f:
        json.dump(out, f)

    if logger is not None:
        for meth in ('flush', 'close'):
            fn = getattr(logger, meth, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass


if __name__ == '__main__':
    main()