import json
import argparse
import time
import sys
import numpy as np
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

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
    distances = np.array(instance['distances'], dtype=float)
    fixed_costs = instance['fixed_costs']
    hub_capacities = instance.get('hub_capacities')
    is_capacitated = hub_capacities is not None and hub_capacities[0] is not None
    scenarios = instance['scenarios']
    probabilities = [float(s['probability']) for s in scenarios]
    demands = [np.array(s['demands'], dtype=float) for s in scenarios]
    outgoing = [dem.sum(axis=1) for dem in demands]
    incoming = [dem.sum(axis=0) for dem in demands]

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model = Model("HubLocation")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 1e-4)
    model.setParam('LogToConsole', 0)
    model.setParam('Presolve', 1)

    z = model.addVars(n, vtype=GRB.BINARY, name="z")
    x = {}
    for s in range(num_scenarios):
        for i in range(n):
            for k in range(n):
                if i != k:
                    x[s, i, k] = model.addVar(vtype=GRB.BINARY, name=f"x{s}_{i}_{k}")

    # Hub cardinality
    if p_hubs is not None and isinstance(p_hubs, int) and p_hubs > 0:
        model.addConstr(quicksum(z[i] for i in range(n)) == p_hubs, "exactly_p")
    else:
        model.addConstr(quicksum(z[i] for i in range(n)) >= 1, "at_least_one")

    # Assignment constraints
    for s in range(num_scenarios):
        for i in range(n):
            lhs = z[i]
            for k in range(n):
                if i != k:
                    lhs += x[s, i, k]
            model.addConstr(lhs == 1, name=f"assign_s{s}_i{i}")
            for k in range(n):
                if i != k:
                    model.addConstr(x[s, i, k] <= z[k], name=f"link_s{s}_i{i}_k{k}")

    # Capacity constraints
    if is_capacitated:
        for s in range(num_scenarios):
            for k in range(n):
                assigned_out = outgoing[s][k] * z[k]
                for i in range(n):
                    if i != k:
                        assigned_out += outgoing[s][i] * x[s, i, k]
                model.addConstr(assigned_out <= hub_capacities[k] * z[k], name=f"cap_s{s}_k{k}")

    # Objective components
    fixed_cost = quicksum(fixed_costs[i] * z[i] for i in range(n))

    coll_dist = 0.0
    for s in range(num_scenarios):
        prob = probabilities[s]
        for i in range(n):
            for k in range(n):
                if i != k:
                    coeff = prob * (chi * outgoing[s][i] + delta * incoming[s][i]) * distances[i, k]
                    coll_dist += coeff * x[s, i, k]

    interhub = 0.0
    for s in range(num_scenarios):
        prob = probabilities[s]
        dem = demands[s]
        for i in range(n):
            for j in range(n):
                flow = dem[i, j]
                if flow > 0.0:
                    pflow_alpha = prob * flow * alpha
                    # Case 1: i uses hub k, j uses hub m (k and m can be equal)
                    for k_idx in range(n):
                        for m_idx in range(n):
                            if k_idx == i and m_idx == j:
                                interhub += pflow_alpha * distances[i, j] * z[i] * z[j]
                            elif k_idx == i and m_idx != j:
                                interhub += pflow_alpha * distances[i, m_idx] * z[i] * x[s, j, m_idx]
                            elif k_idx != i and m_idx == j:
                                interhub += pflow_alpha * distances[k_idx, j] * x[s, i, k_idx] * z[j]
                            else:
                                interhub += pflow_alpha * distances[k_idx, m_idx] * x[s, i, k_idx] * x[s, j, m_idx]

    model.setObjective(fixed_cost + coll_dist + interhub, GRB.MINIMIZE)

    best_obj = float('inf')
    def callback(m, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj_val = m.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj_val < best_obj - 1e-4:
                best_obj = obj_val
                if logger:
                    logger.log(float(obj_val))

    model.optimize(callback)

    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) and model.SolCount > 0:
        obj_val = model.objVal
        hubs = [i for i in range(n) if z[i].X > 0.5]
        z_vals = [1.0 if z[i].X > 0.5 else 0.0 for i in range(n)]

        allocations = {}
        x_values = {}
        for s_idx in range(num_scenarios):
            alloc_s = {}
            x_dict_s = {}
            for i in range(n):
                assigned_hub = None
                if z[i].X > 0.5:
                    assigned_hub = i
                else:
                    for k in range(n):
                        if i != k and x[s_idx, i, k].X > 0.5:
                            assigned_hub = k
                            break
                if assigned_hub is None:
                    assigned_hub = int(np.argmax([z[k].X for k in range(n)]))
                alloc_s[str(i)] = int(assigned_hub)
                if i != assigned_hub:
                    x_dict_s[str(i)] = {}
                    for kk in range(n):
                        if i != kk:
                            x_dict_s[str(i)][str(kk)] = 1.0 if kk == assigned_hub else 0.0
            allocations[str(s_idx)] = alloc_s
            x_values[str(s_idx)] = x_dict_s

        solution = {
            "objective_value": float(obj_val),
            "hubs": [int(h) for h in hubs],
            "allocations": allocations,
            "z_values": [float(v) for v in z_vals],
            "x_values": x_values
        }
    else:
        # Fallback solution: open the node with smallest fixed cost as single hub
        hub_idx = int(np.argmin(fixed_costs))
        obj_val = float(fixed_costs[hub_idx])
        z_vals = [0.0] * n
        z_vals[hub_idx] = 1.0
        hubs = [hub_idx]
        allocations = {}
        x_values = {}
        for s_idx in range(num_scenarios):
            alloc_s = {}
            x_dict_s = {}
            prob = probabilities[s_idx]
            dem = demands[s_idx]
            outg = outgoing[s_idx]
            inc = incoming[s_idx]
            coll_dist_s = 0.0
            for i in range(n):
                if i != hub_idx:
                    d_ik = distances[i, hub_idx]
                    coll_dist_s += (chi * outg[i] + delta * inc[i]) * d_ik
                    x_dict_s[str(i)] = {}
                    for k in range(n):
                        if i != k:
                            x_dict_s[str(i)][str(k)] = 1.0 if k == hub_idx else 0.0
                alloc_s[str(i)] = hub_idx
            transfer_s = 0.0
            for i in range(n):
                for j in range(n):
                    if dem[i, j] > 0:
                        transfer_s += alpha * dem[i, j] * distances[hub_idx, hub_idx]
            obj_val += prob * (coll_dist_s + transfer_s)
            allocations[str(s_idx)] = alloc_s
            x_values[str(s_idx)] = x_dict_s

        solution = {
            "objective_value": float(obj_val),
            "hubs": [int(hub_idx)],
            "allocations": allocations,
            "z_values": [float(v) for v in z_vals],
            "x_values": x_values
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()