import argparse
import json
import gurobipy as grb
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    n = data['n']
    cost_params = data['cost_parameters']
    chi = cost_params['collection_cost_chi']
    alpha = cost_params['transfer_cost_alpha']
    delta = cost_params['distribution_cost_delta']
    total_flow_D = data['total_flow_D']
    coordinates = data['coordinates']
    flow_matrix = data['flow_matrix']
    distances = data['distance_matrix']
    outgoing_flow_list = data['outgoing_flow']
    setup_costs = data['setup_costs']
    capacities = data['capacities']
    
    incoming_flow = [0.0] * n
    for i in range(n):
        for j in range(n):
            incoming_flow[j] += flow_matrix[i][j]
    
    model = grb.Model()
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', 1)
    
    y = model.addVars(n, vtype=grb.GRB.BINARY, name="y")
    x = model.addVars(n, n, vtype=grb.GRB.BINARY, name="x")
    t = model.addVars(n, n, lb=0.0, name="t")
    u = model.addVars(n, n, n, lb=0.0, name="u")
    
    for i in range(n):
        model.addConstr(grb.quicksum(x[i, k] for k in range(n)) == 1, f"assign_{i}")
    
    for i in range(n):
        for k in range(n):
            model.addConstr(x[i, k] <= y[k], f"open_{i}_{k}")
    
    for k in range(n):
        model.addConstr(
            grb.quicksum(outgoing_flow_list[i] * x[i, k] for i in range(n)) <= capacities[k] * y[k],
            f"cap_{k}"
        )
    
    model.addConstr(
        grb.quicksum(capacities[k] * y[k] for k in range(n)) >= total_flow_D,
        "total_cap"
    )
    
    for i in range(n):
        for m in range(n):
            model.addConstr(
                t[i, m] == grb.quicksum(flow_matrix[i][j] * x[j, m] for j in range(n)),
                f"t_{i}_{m}"
            )
    
    for i in range(n):
        for k in range(n):
            for m in range(n):
                model.addConstr(u[i, k, m] <= outgoing_flow_list[i] * x[i, k], f"u1_{i}_{k}_{m}")
                model.addConstr(u[i, k, m] <= t[i, m], f"u2_{i}_{k}_{m}")
                model.addConstr(
                    u[i, k, m] >= t[i, m] - outgoing_flow_list[i] * (1 - x[i, k]),
                    f"u3_{i}_{k}_{m}"
                )
    
    collection_cost = grb.quicksum(
        chi * distances[i][k] * outgoing_flow_list[i] * x[i, k]
        for i in range(n) for k in range(n)
    )
    
    distribution_cost = grb.quicksum(
        delta * distances[m][j] * incoming_flow[j] * x[j, m]
        for j in range(n) for m in range(n)
    )
    
    transfer_cost = grb.quicksum(
        alpha * distances[k][m] * u[i, k, m]
        for i in range(n) for k in range(n) for m in range(n)
    )
    
    setup_cost = grb.quicksum(setup_costs[k] * y[k] for k in range(n))
    
    model.setObjective(collection_cost + distribution_cost + transfer_cost + setup_cost, grb.GRB.MINIMIZE)
    
    if logger:
        def callback(model, where):
            if where == grb.GRB.Callback.MIPSOL:
                obj = model.cbGet(grb.GRB.Callback.MIPSOL_OBJ)
                logger.log(obj)
        model.optimize(callback)
    else:
        model.optimize()
    
    if model.status == grb.GRB.OPTIMAL or model.status == grb.GRB.TIME_LIMIT:
        if model.SolCount > 0:
            hubs = [k for k in range(n) if y[k].X > 0.5]
            assignment = []
            for i in range(n):
                for k in range(n):
                    if x[i, k].X > 0.5:
                        assignment.append(k)
                        break
            objective_value = model.ObjVal
            solution = {
                "objective_value": objective_value,
                "hubs": hubs,
                "assignment": assignment
            }
            with open(args.solution_path, 'w') as f:
                json.dump(solution, f, indent=2)
            return
    
    nodes_sorted = sorted(range(n), key=lambda i: outgoing_flow_list[i], reverse=True)
    open_hubs = []
    residual_capacities = {}
    assignment = [-1] * n
    
    for i in nodes_sorted:
        flow_i = outgoing_flow_list[i]
        candidate_hub = None
        for hub in open_hubs:
            if residual_capacities[hub] >= flow_i:
                if candidate_hub is None or residual_capacities[hub] < residual_capacities[candidate_hub]:
                    candidate_hub = hub
        if candidate_hub is not None:
            assignment[i] = candidate_hub
            residual_capacities[candidate_hub] -= flow_i
        else:
            candidate = None
            min_cost = float('inf')
            for k in range(n):
                if k not in open_hubs and capacities[k] >= flow_i:
                    if setup_costs[k] < min_cost:
                        min_cost = setup_costs[k]
                        candidate = k
            if candidate is None:
                candidate = max(range(n), key=lambda k: capacities[k] if k not in open_hubs else -1)
            open_hubs.append(candidate)
            residual_capacities[candidate] = capacities[candidate] - flow_i
            assignment[i] = candidate
    
    setup_cost_val = sum(setup_costs[k] for k in open_hubs)
    routing_cost_val = 0.0
    for i in range(n):
        hub_i = assignment[i]
        for j in range(n):
            hub_j = assignment[j]
            routing_cost_val += flow_matrix[i][j] * (
                chi * distances[i][hub_i] +
                alpha * distances[hub_i][hub_j] +
                delta * distances[hub_j][j]
            )
    
    objective_value = setup_cost_val + routing_cost_val
    solution = {
        "objective_value": objective_value,
        "hubs": sorted(open_hubs),
        "assignment": assignment
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()