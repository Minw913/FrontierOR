import argparse
import json
import sys
import math


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        data = json.load(f)

    params = data['parameters']
    T = int(params['T'])
    W = int(params['W'])
    L_str = params['L']
    L = None if L_str == 'infinity' else int(L_str)

    source_id = int(data['source_item_id'])
    sink_id = int(data['sink_item_id'])

    items = data['items']
    real_items = [it for it in items if it['type'] == 'treatment']
    num_real_items = len(real_items)

    arcs = data['arcs']

    # Build node index
    item_ids = set()
    for it in items:
        item_ids.add(int(it['item_id']))
    for a in arcs:
        item_ids.add(int(a['from']))
        item_ids.add(int(a['to']))
    item_ids = sorted(item_ids)
    id_to_idx = {iid: i for i, iid in enumerate(item_ids)}
    n = len(item_ids)

    src_idx = id_to_idx[source_id]
    sink_idx = id_to_idx[sink_id]

    # Build adjacency for longest-path (earliest) and reverse (latest)
    adj = {i: [] for i in range(n)}
    radj = {i: [] for i in range(n)}
    for a in arcs:
        u = id_to_idx[int(a['from'])]
        v = id_to_idx[int(a['to'])]
        lag = int(a['lag'])
        adj[u].append((v, lag))
        radj[v].append((u, lag))

    INF = float('inf')
    earliest = [-INF] * n
    earliest[src_idx] = 0
    for _ in range(n + 1):
        updated = False
        for u in range(n):
            if earliest[u] == -INF:
                continue
            for v, lag in adj[u]:
                if earliest[u] + lag > earliest[v]:
                    earliest[v] = earliest[u] + lag
                    updated = True
        if not updated:
            break

    latest = [INF] * n
    latest[sink_idx] = T + 1
    for _ in range(n + 1):
        updated = False
        for v in range(n):
            if latest[v] == INF:
                continue
            for u, lag in radj[v]:
                if latest[v] - lag < latest[u]:
                    latest[u] = latest[v] - lag
                    updated = True
        if not updated:
            break

    # Feasible periods per real item
    feasible_periods = {}
    for it in real_items:
        iid = int(it['item_id'])
        idx = id_to_idx[iid]
        e = max(1, int(earliest[idx])) if earliest[idx] != -INF else 1
        l = min(T, int(latest[idx])) if latest[idx] != INF else T
        if e > l:
            e, l = 1, T
        feasible_periods[iid] = list(range(e, l + 1))

    # Compute B_max: upper bound on bins per period
    # For each period t, sum weights of items feasible in t; B_max = max ceil(sum/W)
    period_feasible_weight = [0] * (T + 2)
    for it in real_items:
        iid = int(it['item_id'])
        w = int(it['weight'])
        for t in feasible_periods[iid]:
            period_feasible_weight[t] += w
    B_max = 1
    for t in range(1, T + 1):
        if period_feasible_weight[t] > 0:
            b_t = math.ceil(period_feasible_weight[t] / W)
            B_max = max(B_max, b_t)
    if L is not None:
        B_max = min(B_max, L)
    B_max = max(B_max, 1)

    # Greedy initial solution: assign each item to earliest feasible period, first-fit bin packing
    init_period = {}
    init_bin = {}
    bin_usage = {}
    for it in real_items:
        iid = int(it['item_id'])
        e = feasible_periods[iid][0]
        w = int(it['weight'])
        placed = False
        for b in range(B_max):
            key = (b, e)
            cur = bin_usage.get(key, 0)
            if cur + w <= W:
                init_period[iid] = e
                init_bin[iid] = b
                bin_usage[key] = cur + w
                placed = True
                break
        if not placed:
            init_period[iid] = e
            init_bin[iid] = 0  # fallback (may violate capacity)

    def build_solution_from_period_bin(period_map, bin_map):
        assignments = []
        bin_periods = {}
        for it in real_items:
            iid = int(it['item_id'])
            w = int(it['weight'])
            t = period_map[iid]
            b = bin_map[iid]
            assignments.append({
                'item_id': iid, 'bin': b, 'period': t, 'weight': w
            })
            key = (b, t)
            if key not in bin_periods:
                bin_periods[key] = []
            bin_periods[key].append((iid, w))
        active = []
        for (b, t) in sorted(bin_periods.keys()):
            lst = bin_periods[(b, t)]
            active.append({
                'bin': b, 'period': t,
                'items': [iid for iid, _ in lst],
                'total_weight': sum(w for _, w in lst)
            })
        obj = len(active)
        return {
            'objective_value': float(obj),
            'assignments': assignments,
            'active_bin_periods': active,
            'num_active_bin_periods': int(obj)
        }

    init_sol = build_solution_from_period_bin(init_period, init_bin)
    best_obj = init_sol['objective_value']
    best_sol = init_sol

    if logger:
        logger.log_solution(best_obj, best_sol)

    # Now build MIP
    import gurobipy as gp
    from gurobipy import GRB

    env = gp.Env(empty=True)
    env.setParam('Seed', 0)
    env.setParam('Threads', 1)
    env.setParam('OutputFlag', 0)
    env.start()

    model = gp.Model(env=env)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('TimeLimit', max(1, args.time_limit))

    x = {}
    z = {}
    for it in real_items:
        iid = int(it['item_id'])
        for t in feasible_periods[iid]:
            z[iid, t] = model.addVar(vtype=GRB.BINARY, name=f'z_{iid}_{t}')
            for b in range(B_max):
                x[iid, b, t] = model.addVar(vtype=GRB.BINARY, name=f'x_{iid}_{b}_{t}')

    y = {}
    for b in range(B_max):
        for t in range(1, T + 1):
            y[b, t] = model.addVar(vtype=GRB.BINARY, name=f'y_{b}_{t}')

    model.update()

    # Each item assigned one period
    for it in real_items:
        iid = int(it['item_id'])
        model.addConstr(gp.quicksum(z[iid, t] for t in feasible_periods[iid]) == 1)

    # Link x and z
    for it in real_items:
        iid = int(it['item_id'])
        for t in feasible_periods[iid]:
            for b in range(B_max):
                model.addConstr(x[iid, b, t] <= z[iid, t])
            model.addConstr(gp.quicksum(x[iid, b, t] for b in range(B_max)) == z[iid, t])

    # Capacity and linking y
    for b in range(B_max):
        for t in range(1, T + 1):
            terms = []
            for it in real_items:
                iid = int(it['item_id'])
                if (iid, b, t) in x:
                    terms.append(int(it['weight']) * x[iid, b, t])
            if terms:
                model.addConstr(gp.quicksum(terms) <= W * y[b, t])

    # x <= y
    for it in real_items:
        iid = int(it['item_id'])
        for t in feasible_periods[iid]:
            for b in range(B_max):
                model.addConstr(x[iid, b, t] <= y[b, t])

    # Max bins per period
    if L is not None:
        for t in range(1, T + 1):
            model.addConstr(gp.quicksum(y[b, t] for b in range(B_max)) <= L)

    # Symmetry breaking
    for b in range(1, B_max):
        for t in range(1, T + 1):
            model.addConstr(y[b, t] <= y[b - 1, t])

    # Precedence
    period_expr = {}
    for it in real_items:
        iid = int(it['item_id'])
        period_expr[id_to_idx[iid]] = gp.quicksum(t * z[iid, t] for t in feasible_periods[iid])

    def get_period_expr(node_id):
        if node_id == source_id:
            return 0
        if node_id == sink_id:
            return T + 1
        return period_expr[id_to_idx[node_id]]

    for a in arcs:
        u = int(a['from'])
        v = int(a['to'])
        lag = int(a['lag'])
        u_expr = get_period_expr(u)
        v_expr = get_period_expr(v)
        if isinstance(u_expr, (int, float)) and isinstance(v_expr, (int, float)):
            continue
        if isinstance(u_expr, (int, float)):
            model.addConstr(v_expr >= u_expr + lag)
        elif isinstance(v_expr, (int, float)):
            model.addConstr(u_expr <= v_expr - lag)
        else:
            model.addConstr(v_expr >= u_expr + lag)

    model.setObjective(gp.quicksum(y[b, t] for b in range(B_max) for t in range(1, T + 1)), GRB.MINIMIZE)

    # MIP start
    try:
        for it in real_items:
            iid = int(it['item_id'])
            t0 = init_period[iid]
            b0 = init_bin[iid]
            for t in feasible_periods[iid]:
                z[iid, t].Start = 1.0 if t == t0 else 0.0
                for b in range(B_max):
                    x[iid, b, t].Start = 1.0 if (t == t0 and b == b0) else 0.0
        for b in range(B_max):
            for t in range(1, T + 1):
                y[b, t].Start = 1.0 if (b, t) in bin_usage else 0.0
    except Exception:
        pass

    # Callback for logging incumbents
    state = {'best_obj': best_obj, 'best_sol': best_sol}

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < state['best_obj'] - 1e-9:
                # Build solution
                period_map = {}
                bin_map = {}
                for it in real_items:
                    iid = int(it['item_id'])
                    found = False
                    for t in feasible_periods[iid]:
                        for b in range(B_max):
                            if (iid, b, t) in x and model.cbGetSolution(x[iid, b, t]) > 0.5:
                                period_map[iid] = t
                                bin_map[iid] = b
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        period_map[iid] = init_period[iid]
                        bin_map[iid] = init_bin[iid]
                sol = build_solution_from_period_bin(period_map, bin_map)
                state['best_obj'] = obj
                state['best_sol'] = sol
                if logger:
                    logger.log_solution(obj, sol)

    model.optimize(callback)

    # Extract best solution
    if model.SolCount > 0:
        obj = model.ObjVal
        if obj < state['best_obj'] - 1e-9:
            period_map = {}
            bin_map = {}
            for it in real_items:
                iid = int(it['item_id'])
                found = False
                for t in feasible_periods[iid]:
                    for b in range(B_max):
                        if (iid, b, t) in x and x[iid, b, t].X > 0.5:
                            period_map[iid] = t
                            bin_map[iid] = b
                            found = True
                            break
                    if found:
                        break
                if not found:
                    period_map[iid] = init_period[iid]
                    bin_map[iid] = init_bin[iid]
            sol = build_solution_from_period_bin(period_map, bin_map)
            state['best_obj'] = obj
            state['best_sol'] = sol

    final_sol = state['best_sol']
    with open(args.solution_path, 'w') as f:
        json.dump(final_sol, f, indent=2)


if __name__ == '__main__':
    main()