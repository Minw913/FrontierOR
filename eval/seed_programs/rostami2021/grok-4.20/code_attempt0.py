import json
import argparse
import time
import sys
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    num_scenarios = instance['num_scenarios']
    chi = instance['cost_parameters']['chi']
    alpha = instance['cost_parameters']['alpha']
    delta = instance['cost_parameters']['delta']
    p_hubs = instance['p_hubs']
    distances = np.array(instance['distances'])
    fixed_costs = instance['fixed_costs']
    hub_capacities = instance.get('hub_capacities')
    is_capacitated = hub_capacities is not None and hub_capacities[0] is not None
    scenarios = instance['scenarios']
    probabilities = [s['probability'] for s in scenarios]
    demands = [np.array(s['demands']) for s in scenarios]
    outgoing = []
    incoming = []
    for s_idx in range(num_scenarios):
        dem = demands[s_idx]
        outg = dem.sum(axis=1)
        inc = dem.sum(axis=0)
        outgoing.append(outg)
        incoming.append(inc)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model = Model("HubLocation")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 0.01)
    model.setParam('LogToConsole', 0)

    z = model.addVars(n, vtype=GRB.BINARY, name="z")
    x = {}
    for s in range(num_scenarios):
        for i in range(n):
            for k in range(n):
                if i != k:
                    x[s, i, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{s}_{i}_{k}")

    # At least one hub or exactly p
    if p_hubs is not None and p_hubs > 0:
        model.addConstr(quicksum(z[i] for i in range(n)) == p_hubs)
    else:
        model.addConstr(quicksum(z[i] for i in range(n)) >= 1)

    # Each non-hub assigned to exactly one hub
    for s in range(num_scenarios):
        for i in range(n):
            model.addConstr(
                quicksum(x[s, i, k] for k in range(n) if k != i) + z[i] == 1,
                name=f"assign_{s}_{i}"
            )
            for k in range(n):
                if i != k:
                    model.addConstr(x[s, i, k] <= z[k])

    # Capacity constraints
    if is_capacitated:
        for s in range(num_scenarios):
            for k in range(n):
                assigned_out = quicksum(
                    (outgoing[s][i] * x[s, i, k]) for i in range(n) if i != k
                ) + (outgoing[s][k] * z[k])
                model.addConstr(assigned_out <= hub_capacities[k] * z[k])

    # Objective
    fixed_cost = quicksum(fixed_costs[i] * z[i] for i in range(n))

    coll_dist_cost = 0.0
    transfer_cost = 0.0

    for s in range(num_scenarios):
        prob = probabilities[s]
        outg = outgoing[s]
        inc = incoming[s]
        dem = demands[s]

        # Collection + distribution for spokes
        for i in range(n):
            for k in range(n):
                if i != k:
                    coeff = (chi * outg[i] + delta * inc[i]) * distances[i][k]
                    coll_dist_cost += prob * coeff * x[s, i, k]

        # Inter-hub transfer
        for i in range(n):
            for j in range(n):
                if dem[i][j] > 0:
                    flow = dem[i][j]
                    for k in range(n):
                        for m in range(n):
                            if k != m:
                                coeff = alpha * flow * distances[k][m] * prob
                                transfer_cost += coeff * x[s, i, k] * x[s, j, m]
                            elif k == m and k != i and k != j:
                                coeff = alpha * flow * 0.0 * prob
                                pass
        # When i is hub: x_ii effectively 1 via z[i]
        for i in range(n):
            for j in range(n):
                if dem[i][j] > 0:
                    flow = dem[i][j]
                    prob_flow = prob * flow * alpha
                    for k in range(n):
                        for m in range(n):
                            if k == m:
                                term = z[k] * (x[s, j, m] if j != m else 0)
                                if i != k:
                                    transfer_cost += prob_flow * distances[k][k] * 0
                                continue
                            # Full bilinear
                            if i != k and j != m:
                                transfer_cost += prob_flow * distances[k][m] * x[s, i, k] * x[s, j, m]
                            elif i == k and j != m:
                                transfer_cost += prob_flow * distances[i][m] * z[i] * x[s, j, m]
                            elif i != k and j == m:
                                transfer_cost += prob_flow * distances[k][j] * x[s, i, k] * z[j]
                            else:
                                # both hubs, but i==k, j==m
                                transfer_cost += prob_flow * distances[i][j] * z[i] * z[j]

    objective = fixed_cost + coll_dist_cost + transfer_cost
    model.setObjective(objective, GRB.MINIMIZE)

    # Callback for logging
    best_obj = float('inf')
    def callback(m, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj_val = m.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj_val < best_obj - 1e-6:
                best_obj = obj_val
                if logger:
                    logger.log(float(obj_val))

    start_time = time.time()
    model.optimize(callback)

    # Extract solution
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        hubs = [i for i in range(n) if z[i].X > 0.5]
        z_vals = [1.0 if z[i].X > 0.5 else 0.0 for i in range(n)]

        allocations = {}
        x_values = {}
        for s_idx in range(num_scenarios):
            alloc_s = {}
            x_s = {}
            for i in range(n):
                assigned = False
                for k in range(n):
                    if i == k:
                        if z[k].X > 0.5:
                            alloc_s[str(i)] = i
                            assigned = True
                            break
                    elif x[s_idx, i, k].X > 0.5:
                        alloc_s[str(i)] = k
                        x_s.setdefault(str(i), {})
                        for kk in range(n):
                            if i != kk:
                                key = str(kk)
                                x_s[str(i)][key] = 1.0 if kk == k else 0.0
                        assigned = True
                        break
                if not assigned and z[i].X > 0.5:
                    alloc_s[str(i)] = i
            allocations[str(s_idx)] = alloc_s
            x_values[str(s_idx)] = x_s

        obj_val = model.objVal if model.objVal < GRB.INFINITY else best_obj

        solution = {
            "objective_value": float(obj_val),
            "hubs": [int(h) for h in hubs],
            "allocations": allocations,
            "z_values": [float(v) for v in z_vals],
            "x_values": x_values
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # Fallback: open cheapest hub and assign all to it
        hub_idx = int(np.argmin(fixed_costs))
        hubs = [hub_idx]
        z_vals = [0.0] * n
        z_vals[hub_idx] = 1.0
        allocations = {}
        x_values = {}
        total_obj = sum(fixed_costs[hub_idx])
        for s_idx in range(num_scenarios):
            alloc_s = {}
            x_s = {}
            prob = probabilities[s_idx]
            dem = demands[s_idx]
            outg = outgoing[s_idx]
            inc = incoming[s_idx]
            for i in range(n):
                alloc_s[str(i)] = hub_idx
                if i != hub_idx:
                    x_s.setdefault(str(i), {})
                    for k in range(n):
                        if i != k:
                            x_s[str(i)][str(k)] = 1.0 if k == hub_idx else 0.0
            allocations[str(s_idx)] = alloc_s
            x_values[str(s_idx)] = x_s

            # compute rough cost
            coll_dist = 0.0
            for i in range(n):
                if i != hub_idx:
                    coll_dist += (chi * outg[i] + delta * inc[i]) * distances[i][hub_idx]
            transfer = 0.0
            for i_idx in range(n):
                for j_idx in range(n):
                    if dem[i_idx][j_idx] > 0:
                        transfer += alpha * dem[i_idx][j_idx] * distances[hub_idx][hub_idx]
            total_obj += prob * (coll_dist + transfer)
        solution = {
            "objective_value": float(total_obj),
            "hubs": [int(hub_idx)],
            "allocations": allocations,
            "z_values": [float(v) for v in z_vals],
            "x_values": x_values
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()