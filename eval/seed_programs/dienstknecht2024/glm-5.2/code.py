import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def count_components(nodes, adj):
    nodes = set(nodes)
    visited = set()
    comps = 0
    for n in nodes:
        if n not in visited:
            comps += 1
            stack = [n]
            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u)
                for v in adj[u]:
                    if v in nodes and v not in visited:
                        stack.append(v)
    return comps


def find_connected_set(available, adj, demand):
    available = set(available)
    for start in available:
        visited = set()
        stack = [start]
        while stack and len(visited) < demand:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            for v in adj[u]:
                if v in available and v not in visited:
                    stack.append(v)
        if len(visited) >= demand:
            return list(visited)[:demand]
    return None


def solve_period_model(t, avail_c, trade_list, adj, C, time_limit):
    env = gp.Env(empty=True)
    env.setParam('Seed', 0)
    env.setParam('Threads', 1)
    env.start()
    m = gp.Model(f'period_{t}', env=env)
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.OutputFlag = 0

    x = {}
    z = {}
    g = {}
    f = {}

    for k in trade_list:
        for c in avail_c:
            x[k['idx'], c] = m.addVar(vtype=GRB.BINARY)
            z[k['idx'], c] = m.addVar(vtype=GRB.BINARY)
            g[k['idx'], c] = m.addVar(lb=0, ub=k['demand'])
        for c in avail_c:
            for nb in adj[c]:
                if nb in avail_c and c < nb:
                    f[k['idx'], c, nb] = m.addVar(lb=0, ub=k['demand'])
                    f[k['idx'], nb, c] = m.addVar(lb=0, ub=k['demand'])
    m.update()

    for k in trade_list:
        m.addConstr(gp.quicksum(x[k['idx'], c] for c in avail_c) == k['demand'])
        m.addConstr(gp.quicksum(g[k['idx'], c] for c in avail_c) == k['demand'])
        for c in avail_c:
            m.addConstr(g[k['idx'], c] <= k['demand'] * z[k['idx'], c])
            m.addConstr(z[k['idx'], c] <= x[k['idx'], c])
        m.addConstr(gp.quicksum(z[k['idx'], c] for c in avail_c) <= k['max_disp'])
        for c in avail_c:
            inflow = g[k['idx'], c]
            outflow = 0
            for nb in adj[c]:
                if nb in avail_c:
                    if (k['idx'], nb, c) in f:
                        inflow += f[k['idx'], nb, c]
                    if (k['idx'], c, nb) in f:
                        outflow += f[k['idx'], c, nb]
            m.addConstr(inflow - outflow == x[k['idx'], c])
        for c in avail_c:
            for nb in adj[c]:
                if nb in avail_c and c < nb:
                    m.addConstr(f[k['idx'], c, nb] <= k['demand'] * x[k['idx'], c])
                    m.addConstr(f[k['idx'], c, nb] <= k['demand'] * x[k['idx'], nb])
                    m.addConstr(f[k['idx'], nb, c] <= k['demand'] * x[k['idx'], nb])
                    m.addConstr(f[k['idx'], nb, c] <= k['demand'] * x[k['idx'], c])
    for c in avail_c:
        m.addConstr(gp.quicksum(x[k['idx'], c] for k in trade_list) <= 1)

    m.optimize()
    result = None
    if m.SolCount > 0:
        result = {}
        for k in trade_list:
            assigned = []
            for c in avail_c:
                if x[k['idx'], c].X > 0.5:
                    assigned.append(c)
            result[k['idx']] = assigned
    env.dispose()
    return result


def fallback_solution(containers, avail, adj, trade_data, C, num_periods, time_limit, logger):
    cont_id_to_idx = {c['id']: i for i, c in enumerate(containers)}
    assignments = {}
    for k in trade_data:
        assignments[k['idx']] = {}

    remaining_time = time_limit
    for t in range(num_periods):
        avail_c = [i for i in range(C) if avail[t][i]]
        active_trades = [k for k in trade_data if k['start'] <= t <= k['end']]
        if not active_trades:
            continue
        per_time = max(1, remaining_time / max(1, num_periods - t))
        sol = solve_period_model(t, avail_c, active_trades, adj, C, per_time)
        if sol is not None:
            for k in active_trades:
                assignments[k['idx']][t] = [containers[c]['id'] for c in sol[k['idx']]]
        else:
            available = set(avail_c)
            for k in active_trades:
                assigned = []
                cand = list(available)[:k['demand']]
                comps = count_components(cand, adj)
                if comps <= k['max_disp']:
                    assigned = cand
                else:
                    conn = find_connected_set(available, adj, k['demand'])
                    if conn:
                        assigned = conn
                    else:
                        assigned = cand
                for c in assigned:
                    available.discard(c)
                assignments[k['idx']][t] = [containers[c]['id'] for c in assigned]
        remaining_time = max(0, remaining_time - per_time)

    obj = 0
    for k in trade_data:
        for t in range(k['start'] + 1, k['end'] + 1):
            if t >= num_periods:
                continue
            prev = set(assignments[k['idx']].get(t - 1, []))
            curr = set(assignments[k['idx']].get(t, []))
            obj += len(curr - prev)

    solution = {"objective_value": obj, "assignments": assignments}
    if logger:
        logger.log_solution(obj, solution)
    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    with open(args.instance_path) as f:
        instance = json.load(f)

    num_periods = instance['problem_parameters']['num_periods']
    num_containers = instance['problem_parameters']['num_containers']
    num_trades = instance['problem_parameters']['num_trades']
    containers = instance['containers']
    trades = instance['trades']

    C = len(containers)
    K = len(trades)
    cont_id_to_idx = {c['id']: i for i, c in enumerate(containers)}

    avail = [[False] * C for _ in range(num_periods)]
    for c in containers:
        idx = cont_id_to_idx[c['id']]
        for t in range(c['availability_start'], c['availability_end'] + 1):
            if 0 <= t < num_periods:
                avail[t][idx] = True

    adj = [[] for _ in range(C)]
    for c in containers:
        idx = cont_id_to_idx[c['id']]
        for nb_id in c['adjacent_containers']:
            nb_idx = cont_id_to_idx[nb_id]
            if nb_idx not in adj[idx]:
                adj[idx].append(nb_idx)
                adj[nb_idx].append(idx)
    for i in range(C):
        adj[i] = list(set(adj[i]))

    trade_data = []
    for k_idx, trade in enumerate(trades):
        trade_data.append({
            'idx': k_idx,
            'start': trade['start_period'],
            'end': trade['end_period'],
            'demand': trade['container_demand'],
            'max_disp': trade['max_dispersion']
        })

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start_time = time.time()

    # Full MILP
    env = gp.Env(empty=True)
    env.setParam('Seed', 0)
    env.setParam('Threads', 1)
    env.start()
    model = gp.Model('container_assignment', env=env)
    model.Params.TimeLimit = args.time_limit
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.OutputFlag = 0

    x = {}
    z = {}
    g = {}
    f = {}
    r = {}

    for k in trade_data:
        for t in range(k['start'], k['end'] + 1):
            if t >= num_periods:
                continue
            avail_c = [i for i in range(C) if avail[t][i]]
            for c in avail_c:
                x[k['idx'], c, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{k['idx']}_{c}_{t}")
                z[k['idx'], c, t] = model.addVar(vtype=GRB.BINARY, name=f"z_{k['idx']}_{c}_{t}")
                g[k['idx'], c, t] = model.addVar(lb=0, ub=k['demand'], name=f"g_{k['idx']}_{c}_{t}")
            for c in avail_c:
                for nb in adj[c]:
                    if nb in avail_c and c < nb:
                        f[k['idx'], c, nb, t] = model.addVar(lb=0, ub=k['demand'], name=f"f_{k['idx']}_{c}_{nb}_{t}")
                        f[k['idx'], nb, c, t] = model.addVar(lb=0, ub=k['demand'], name=f"f_{k['idx']}_{nb}_{c}_{t}")
            if t > k['start']:
                for c in avail_c:
                    r[k['idx'], c, t] = model.addVar(lb=0, ub=1, name=f"r_{k['idx']}_{c}_{t}")
    model.update()

    for k in trade_data:
        for t in range(k['start'], k['end'] + 1):
            if t >= num_periods:
                continue
            avail_c = [i for i in range(C) if avail[t][i]]
            model.addConstr(gp.quicksum(x[k['idx'], c, t] for c in avail_c) == k['demand'])
            model.addConstr(gp.quicksum(g[k['idx'], c, t] for c in avail_c) == k['demand'])
            for c in avail_c:
                model.addConstr(g[k['idx'], c, t] <= k['demand'] * z[k['idx'], c, t])
                model.addConstr(z[k['idx'], c, t] <= x[k['idx'], c, t])
            model.addConstr(gp.quicksum(z[k['idx'], c, t] for c in avail_c) <= k['max_disp'])
            for c in avail_c:
                inflow = g[k['idx'], c, t]
                outflow = 0
                for nb in adj[c]:
                    if avail[t][nb]:
                        if (k['idx'], nb, c, t) in f:
                            inflow += f[k['idx'], nb, c, t]
                        if (k['idx'], c, nb, t) in f:
                            outflow += f[k['idx'], c, nb, t]
                model.addConstr(inflow - outflow == x[k['idx'], c, t])
            for c in avail_c:
                for nb in adj[c]:
                    if avail[t][nb] and c < nb:
                        model.addConstr(f[k['idx'], c, nb, t] <= k['demand'] * x[k['idx'], c, t])
                        model.addConstr(f[k['idx'], c, nb, t] <= k['demand'] * x[k['idx'], nb, t])
                        model.addConstr(f[k['idx'], nb, c, t] <= k['demand'] * x[k['idx'], nb, t])
                        model.addConstr(f[k['idx'], nb, c, t] <= k['demand'] * x[k['idx'], c, t])
            if t > k['start']:
                for c in avail_c:
                    if avail[t - 1][c] and t - 1 >= k['start'] and t - 1 <= k['end']:
                        x_prev = x[k['idx'], c, t - 1]
                    else:
                        x_prev = 0
                    model.addConstr(r[k['idx'], c, t] >= x[k['idx'], c, t] - x_prev)

    for t in range(num_periods):
        for c in range(C):
            if not avail[t][c]:
                continue
            terms = []
            for k in trade_data:
                if k['start'] <= t <= k['end'] and avail[t][c]:
                    terms.append(x[k['idx'], c, t])
            if terms:
                model.addConstr(gp.quicksum(terms) <= 1)

    if r:
        model.setObjective(gp.quicksum(r.values()), GRB.MINIMIZE)
    else:
        model.setObjective(0, GRB.MINIMIZE)

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    model.optimize(callback=callback)

    if model.SolCount > 0:
        obj_val = model.ObjVal
        assignments = {}
        for k in trade_data:
            assignments[k['idx']] = {}
            for t in range(k['start'], k['end'] + 1):
                if t >= num_periods:
                    continue
                assigned = []
                for c in range(C):
                    if avail[t][c] and (k['idx'], c, t) in x:
                        if x[k['idx'], c, t].X > 0.5:
                            assigned.append(containers[c]['id'])
                assignments[k['idx']][t] = assigned
        solution = {
            "objective_value": obj_val,
            "assignments": assignments
        }
        if logger:
            logger.log_solution(obj_val, solution)
    else:
        remaining = max(0, args.time_limit - (time.time() - start_time))
        solution = fallback_solution(containers, avail, adj, trade_data, C, num_periods, remaining, logger)

    env.dispose()

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()