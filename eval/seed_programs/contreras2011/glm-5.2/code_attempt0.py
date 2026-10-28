import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def compute_objective(n, flow, dist, outgoing, incoming, setup, chi, alpha, delta, assignment, hubs):
    cost = sum(setup[k] for k in hubs)
    for i in range(n):
        k = assignment[i]
        cost += chi * outgoing[i] * dist[i][k]
    for j in range(n):
        m = assignment[j]
        cost += delta * incoming[j] * dist[m][j]
    for i in range(n):
        ai = assignment[i]
        aj_list = [assignment[j] for j in range(n)]
        for j in range(n):
            cost += alpha * flow[i][j] * dist[ai][aj_list[j]]
    return cost


def greedy_warmstart(n, candidates, cap, setup, outgoing, incoming, dist, D, chi, delta):
    p = len(candidates)
    open_hubs = []
    remaining = {}
    sorted_cand = sorted(range(p), key=lambda ki: setup[candidates[ki]] / max(cap[candidates[ki]], 1e-9))
    total_cap = 0.0
    for ki in sorted_cand:
        open_hubs.append(ki)
        remaining[ki] = cap[candidates[ki]]
        total_cap += cap[candidates[ki]]
        if total_cap >= D:
            break
    open_set = set(open_hubs)
    assigned = {}
    # Hub nodes assigned to themselves
    for ki in open_set:
        k = candidates[ki]
        assigned[k] = ki
        remaining[ki] -= outgoing[k]
    # Assign remaining nodes
    for i in range(n):
        if i in assigned:
            continue
        best_ki = None
        best_score = float('inf')
        for ki in open_set:
            if remaining[ki] >= outgoing[i] - 1e-9:
                s = chi * outgoing[i] * dist[i][candidates[ki]] + delta * incoming[i] * dist[candidates[ki]][i]
                if s < best_score:
                    best_score = s
                    best_ki = ki
        if best_ki is None:
            best_ki = min(open_set, key=lambda ki: dist[i][candidates[ki]])
        assigned[i] = best_ki
        remaining[best_ki] -= outgoing[i]
    return open_set, assigned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    start_time = time.time()

    with open(args.instance_path) as f:
        inst = json.load(f)

    n = inst['n']
    chi = inst['cost_parameters']['collection_cost_chi']
    alpha = inst['cost_parameters']['transfer_cost_alpha']
    delta = inst['cost_parameters']['distribution_cost_delta']
    D = inst['total_flow_D']
    flow = inst['flow_matrix']
    dist = inst['distance_matrix']
    outgoing = inst['outgoing_flow']
    setup = inst['setup_costs']
    cap = inst['capacities']

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    incoming = [sum(flow[i][j] for i in range(n)) for j in range(n)]

    # Candidate hub selection
    if n <= 40:
        candidates = list(range(n))
    else:
        scores = []
        for k in range(n):
            centrality = (sum(outgoing[i] * dist[i][k] for i in range(n)) +
                          sum(incoming[j] * dist[k][j] for j in range(n)))
            score = setup[k] + 0.001 * centrality - 0.0001 * cap[k]
            scores.append((score, k))
        scores.sort()
        max_p = int((100000.0 / n) ** 0.5)
        max_p = max(15, min(max_p, 40))
        candidates = [k for _, k in scores[:max_p]]
        total_cap = sum(cap[k] for k in candidates)
        idx = max_p
        while total_cap < D and idx < len(scores):
            k = scores[idx][1]
            if k not in candidates:
                candidates.append(k)
                total_cap += cap[k]
            idx += 1

    candidates = list(dict.fromkeys(candidates))
    p = len(candidates)

    env = gp.Env(empty=True)
    env.setParam('Seed', 0)
    env.setParam('Threads', 1)
    env.setParam('MIPGap', 1e-4)
    env.setParam('NumericFocus', 0)
    env.setParam('OutputFlag', 0)
    env.start()

    model = gp.Model(env=env)

    z = model.addVars(p, vtype=GRB.BINARY, name='z')
    x = model.addVars(n, p, vtype=GRB.BINARY, name='x')
    t = model.addVars(n, p, p, lb=0, name='t')

    model.addConstrs((gp.quicksum(x[i, k] for k in range(p)) == 1 for i in range(n)), name='assign')
    model.addConstrs((x[i, k] <= z[k] for i in range(n) for k in range(p)), name='link')
    for ki, k in enumerate(candidates):
        model.addConstr(x[k, ki] == z[ki])
    model.addConstrs(
        (gp.quicksum(outgoing[i] * x[i, ki] for i in range(n)) <= cap[candidates[ki]] * z[ki] for ki in range(p)),
        name='cap')
    model.addConstr(gp.quicksum(cap[candidates[ki]] * z[ki] for ki in range(p)) >= D, name='totcap')

    # w expressions: w[i][m] = sum_j flow[i][j] * x[j,m]
    wexpr = [[gp.quicksum(flow[i][j] * x[j, m] for j in range(n)) for m in range(p)] for i in range(n)]

    model.addConstrs((t[i, k, m] <= outgoing[i] * x[i, k]
                     for i in range(n) for k in range(p) for m in range(p)), name='tub1')
    model.addConstrs((t[i, k, m] <= wexpr[i][m]
                     for i in range(n) for k in range(p) for m in range(p)), name='tub2')
    model.addConstrs((t[i, k, m] >= wexpr[i][m] - outgoing[i] * (1 - x[i, k])
                     for i in range(n) for k in range(p) for m in range(p)), name='tlb')

    obj = gp.LinExpr()
    obj += gp.quicksum(setup[candidates[ki]] * z[ki] for ki in range(p))
    obj += chi * gp.quicksum(outgoing[i] * dist[i][candidates[k]] * x[i, k]
                            for i in range(n) for k in range(p))
    obj += delta * gp.quicksum(incoming[j] * dist[candidates[m]][j] * x[j, m]
                               for j in range(n) for m in range(p))
    obj += alpha * gp.quicksum(dist[candidates[k]][candidates[m]] * t[i, k, m]
                               for i in range(n) for k in range(p) for m in range(p))
    model.setObjective(obj, GRB.MINIMIZE)

    # Warm start
    try:
        open_set, assigned = greedy_warmstart(n, candidates, cap, setup, outgoing, incoming, dist, D, chi, delta)
        for ki in range(p):
            z[ki].Start = 1 if ki in open_set else 0
        for i in range(n):
            for ki in range(p):
                x[i, ki].Start = 1 if assigned.get(i) == ki else 0
    except Exception:
        pass

    best_solution = {'objective_value': float('inf'), 'hubs': [], 'assignment': []}

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            objval = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if objval < best_solution['objective_value'] - 1e-9:
                zvals = [model.cbGetSolution(z[ki]) for ki in range(p)]
                hubs = [candidates[ki] for ki in range(p) if zvals[ki] > 0.5]
                assignment = []
                for i in range(n):
                    found = None
                    for ki in range(p):
                        xv = model.cbGetSolution(x[i, ki])
                        if xv > 0.5:
                            found = candidates[ki]
                            break
                    if found is None:
                        found = hubs[0] if hubs else candidates[0]
                    assignment.append(found)
                best_solution['objective_value'] = objval
                best_solution['hubs'] = hubs
                best_solution['assignment'] = assignment
                if logger:
                    logger.log_solution(objval, dict(best_solution))

    elapsed = time.time() - start_time
    remaining_time = max(args.time_limit - elapsed - 1.0, 1.0)
    model.Params.TimeLimit = remaining_time
    model.optimize(callback)

    if model.SolCount > 0:
        zvals = [z[ki].X for ki in range(p)]
        hubs = [candidates[ki] for ki in range(p) if zvals[ki] > 0.5]
        assignment = []
        for i in range(n):
            found = None
            for ki in range(p):
                if x[i, ki].X > 0.5:
                    found = candidates[ki]
                    break
            if found is None:
                found = hubs[0] if hubs else candidates[0]
            assignment.append(found)
        recomputed = compute_objective(n, flow, dist, outgoing, incoming, setup, chi, alpha, delta, assignment, hubs)
        best_solution['objective_value'] = recomputed
        best_solution['hubs'] = hubs
        best_solution['assignment'] = assignment
    elif best_solution['objective_value'] == float('inf'):
        # Fallback heuristic over ALL nodes
        sorted_all = sorted(range(n), key=lambda k: setup[k] / max(cap[k], 1e-9))
        open_hubs_all = []
        total = 0.0
        for k in sorted_all:
            open_hubs_all.append(k)
            total += cap[k]
            if total >= D:
                break
        remaining = {k: cap[k] for k in open_hubs_all}
        assignment = [-1] * n
        for k in open_hubs_all:
            assignment[k] = k
            remaining[k] -= outgoing[k]
        for i in sorted(range(n), key=lambda i: -outgoing[i]):
            if assignment[i] != -1:
                continue
            best_k = None
            best_s = float('inf')
            for k in open_hubs_all:
                if remaining[k] >= outgoing[i] - 1e-9:
                    s = chi * outgoing[i] * dist[i][k] + delta * incoming[i] * dist[k][i]
                    if s < best_s:
                        best_s = s
                        best_k = k
            if best_k is None:
                best_k = min(open_hubs_all, key=lambda k: dist[i][k])
            assignment[i] = best_k
            remaining[best_k] -= outgoing[i]
        obj = compute_objective(n, flow, dist, outgoing, incoming, setup, chi, alpha, delta, assignment, open_hubs_all)
        best_solution = {'objective_value': obj, 'hubs': open_hubs_all, 'assignment': assignment}

    if logger:
        logger.log_solution(best_solution['objective_value'], best_solution)

    with open(args.solution_path, 'w') as f:
        json.dump(best_solution, f)


if __name__ == '__main__':
    main()