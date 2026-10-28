import json
import argparse
import time
from gurobipy import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    # Load instance
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    n = data['n']
    num_scenarios = data['num_scenarios']
    chi = data['cost_parameters']['chi']
    alpha = data['cost_parameters']['alpha']
    delta = data['cost_parameters']['delta']
    p_hubs = data['p_hubs']
    distances = data['distances']
    fixed_costs = data['fixed_costs']
    hub_capacities = data['hub_capacities']
    scenarios = data['scenarios']
    
    # Precompute outgoing and incoming flows for each scenario
    out_flows = []  # out_flows[s][i] = total outgoing from i in scenario s
    in_flows = []   # in_flows[s][j] = total incoming to j in scenario s
    for s in range(num_scenarios):
        demands = scenarios[s]['demands']
        out_s = [sum(demands[i]) for i in range(n)]
        in_s = [sum(demands[i][j] for i in range(n)) for j in range(n)]
        out_flows.append(out_s)
        in_flows.append(in_s)
    
    # Build model
    model = Model("stochastic_hub_location")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    
    # Decision variables
    z = model.addVars(n, vtype=GRB.BINARY, name="z")
    
    # x[s, i, k] = 1 if node i is assigned to hub k in scenario s
    x = {}
    for s in range(num_scenarios):
        for i in range(n):
            for k in range(n):
                x[s, i, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{s}_{i}_{k}")
    
    # Objective
    obj = LinExpr()
    
    # Fixed costs
    for i in range(n):
        obj += fixed_costs[i] * z[i]
    
    # Collection and distribution costs (linear part)
    for s in range(num_scenarios):
        prob_s = scenarios[s]['probability']
        for i in range(n):
            for k in range(n):
                cost_coeff = distances[i][k] * (chi * out_flows[s][i] + delta * in_flows[s][i])
                obj += prob_s * cost_coeff * x[s, i, k]
    
    # Inter-hub transfer costs (quadratic part)
    for s in range(num_scenarios):
        prob_s = scenarios[s]['probability']
        demands = scenarios[s]['demands']
        for i in range(n):
            for j in range(n):
                if demands[i][j] > 0:
                    for k in range(n):
                        for l in range(n):
                            coeff = prob_s * alpha * demands[i][j] * distances[k][l]
                            if coeff != 0:
                                obj += coeff * x[s, i, k] * x[s, j, l]
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Constraints
    # Each node assigned to exactly one hub in each scenario
    for s in range(num_scenarios):
        for i in range(n):
            model.addConstr(quicksum(x[s, i, k] for k in range(n)) == 1, name=f"assign_{s}_{i}")
    
    # Can only assign to open hubs
    for s in range(num_scenarios):
        for i in range(n):
            for k in range(n):
                model.addConstr(x[s, i, k] <= z[k], name=f"link_{s}_{i}_{k}")
    
    # Hub nodes are assigned to themselves
    for s in range(num_scenarios):
        for k in range(n):
            model.addConstr(x[s, k, k] == z[k], name=f"hub_self_{s}_{k}")
    
    # At least one hub
    model.addConstr(quicksum(z[i] for i in range(n)) >= 1, name="min_hub")
    
    # p-hub median variant
    if p_hubs is not None:
        model.addConstr(quicksum(z[i] for i in range(n)) == p_hubs, name="p_hub")
    
    # Capacitated variant
    if hub_capacities is not None:
        for s in range(num_scenarios):
            for k in range(n):
                model.addConstr(
                    quicksum(out_flows[s][i] * x[s, i, k] for i in range(n)) <= hub_capacities[k],
                    name=f"cap_{s}_{k}"
                )
    
    # Optimize
    model.optimize()
    
    # Extract solution
    ...