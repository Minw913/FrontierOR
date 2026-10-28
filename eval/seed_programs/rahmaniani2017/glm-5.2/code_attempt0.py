import argparse
import json
import numpy as np
from itertools import combinations_with_replacement
import gurobipy as gp
from gurobipy import GRB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        instance = json.load(f)

    num_nodes = instance['num_nodes']
    num_commodities = instance['num_commodities']
    params = instance['parameters']
    tau = params['tau']
    chi = params['chi']
    delta = params['delta']
    nodes = instance['nodes']
    distances = np.array(instance['distances'], dtype=float)
    commodities = instance['commodities']
    fixed_costs = np.array([float(nodes[i]['fixed_cost']) for i in range(num_nodes)])

    origins = [c['origin'] for c in commodities]
    dests = [c['destination'] for c in commodities]
    flows = [float(c['flow']) for c in commodities]

    def compute_cost(c, i, j):
        o = origins[c]
        d = dests[c]
        w = flows[c]
        if i == j:
            return w * (chi * distances[o][i] + delta * distances[i][d])
        cost_ij = w * (chi * distances[o][i] + tau * distances[i][j] + delta * distances[j][d])
        cost_ji = w * (chi * distances[o][j] + tau * distances[j][i] + delta * distances[i][d])
        return min(cost_ij, cost_ji)

    all_pairs = list(combinations_with_replacement(range(num_nodes), 2))
    num_pairs = len(all_pairs)
    num_two_hub_pairs = num_pairs - num_nodes

    MAX_VARS = 200000
    total_vars = num_commodities * num_pairs

    if total_vars <= MAX_VARS:
        com_candidate_pairs = []
        for c in range(num_commodities):
            if origins[c] == dests[c]:
                pairs = [(i, i) for i in range(num_nodes)]
            else:
                pairs = all_pairs
            com_candidate_pairs.append(pairs)
    else:
        max_pairs_per_com = MAX_VARS // max(1, num_commodities)
        K_pairs = max(0, min(num_two_hub_pairs, max_pairs_per_com - num_nodes))

        com_candidate_pairs = []
        all_pairs_set = set()

        for c in range(num_commodities):
            single_pairs = [(i, i) for i in range(num_nodes)]
            for p in single_pairs:
                all_pairs_set.add(p)

            if origins[c] != dests[c] and K_pairs > 0:
                o = origins[c]
                d = dests[c]
                w = flows[c]

                term1 = chi * distances[o, :][:, None] + delta * distances[:, d][None, :]
                term2 = chi * distances[o, :][None, :] + delta * distances[:, d][:, None]
                min_term = np.minimum(term1, term2)
                cost_matrix = w * (tau * distances + min_term)

                two_hub_list = []
                for i in range(num_nodes):
                    for j in range(i + 1, num_nodes):
                        two_hub_list.append((cost_matrix[i, j], (i, j)))

                two_hub_list.sort()
                top_pairs = [p for _, p in two_hub_list[:K_pairs]]
                for p in top_pairs:
                    all_pairs_set.add(p)

                com_candidate_pairs.append(single_pairs + top_pairs)
            else:
                com_candidate_pairs.append(single_pairs)

        all_pairs = sorted(all_pairs_set)

    # Heuristic: best single hub (vectorized)
    single_hub_transport = np.zeros(num_nodes)
    for c in range(num_commodities):
        o = origins[c]
        d = dests[c]
        w = flows[c]
        single_hub_transport += w * (chi * distances[o, :] + delta * distances[:, d])

    best_h_k = int(np.argmin(fixed_costs + single_hub_transport))
    best_h_obj = float((fixed_costs + single_hub_transport)[best_h_k])

    # Build heuristic solution
    heuristic_assignments = []
    for c in range(num_commodities):
        cost = compute_cost(c, best_h_k, best_h_k)
        heuristic_assignments.append({
            'commodity': c,
            'origin': origins[c],
            'destination': dests[c],
            'hub_edge': [best_h_k],
            'cost': cost
        })

    heuristic_solution = {
        'objective_value': best_h_obj,
        'open_hubs': [best_h_k],
        'assignments': heuristic_assignments
    }

    if logger:
        logger.log_solution(best_h_obj, heuristic_solution)

    # Build MIP
    model = gp.Model('hub')
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', max(1, args.time_limit - 3))
    model.setParam('OutputFlag', 0)

    y = model.addVars(num_nodes, vtype=GRB.BINARY, name='y')

    x = {}
    for c in range(num_commodities):
        for p in com_candidate_pairs[c]:
            x[c, p] = model.addVar(vtype=GRB.BINARY, name=f'x_{c}_{p[0]}_{p[1]}')

    model.update()

    # Objective
    obj = gp.LinExpr()
    for k in range(num_nodes):
        obj += fixed_costs[k] * y[k]
    for c in range(num_commodities):
        for p in com_candidate_pairs[c]:
            obj += compute_cost(c, p[0], p[1]) * x[c, p]
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints: each commodity assigned to exactly one pair
    for c in range(num_commodities):
        model.addConstr(gp.quicksum(x[c, p] for p in com_candidate_pairs[c]) == 1)

    # Constraints: hubs must be open
    for c in range(num_commodities):
        for p in com_candidate_pairs[c]:
            i, j = p
            model.addConstr(x[c, p] <= y[i])
            model.addConstr(x[c, p] <= y[j])

    model.addConstr(gp.quicksum(y[k] for k in range(num_nodes)) >= 1)

    # Warm start
    for k in range(num_nodes):
        y[k].Start = 1 if k == best_h_k else 0

    for c in range(num_commodities):
        assigned = False
        for p in com_candidate_pairs[c]:
            if p == (best_h_k, best_h_k):
                x[c, p].Start = 1
                assigned = True
            else:
                x[c, p].Start = 0
        if not assigned:
            best_p = None
            best_cost = float('inf')
            for p in com_candidate_pairs[c]:
                if best_h_k in p:
                    c_cost = compute_cost(c, p[0], p[1])
                    if c_cost < best_cost:
                        best_cost = c_cost
                        best_p = p
            if best_p:
                x[c, best_p].Start = 1

    # Callback for logging
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)

    model.optimize(callback)

    # Extract solution
    if model.SolCount == 0:
        solution = heuristic_solution
    else:
        obj_val = model.ObjVal
        open_hubs = [k for k in range(num_nodes) if y[k].X > 0.5]
        assignments = []
        for c in range(num_commodities):
            for p in com_candidate_pairs[c]:
                if x[c, p].X > 0.5:
                    i, j = p
                    hub_edge = [i] if i == j else [i, j]
                    assignments.append({
                        'commodity': c,
                        'origin': origins[c],
                        'destination': dests[c],
                        'hub_edge': hub_edge,
                        'cost': compute_cost(c, i, j)
                    })
                    break

        solution = {
            'objective_value': float(obj_val),
            'open_hubs': open_hubs,
            'assignments': assignments
        }

    if logger:
        logger.log_solution(solution['objective_value'], solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()