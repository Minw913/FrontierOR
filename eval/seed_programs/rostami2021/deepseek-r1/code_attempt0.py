import json
import gurobipy as gp
from gurobipy import GRB, quicksum
import sys
import os
from utils import SolutionLogger

def main(instance_path, solution_path, time_limit, log_file=None):
    logger = SolutionLogger(log_file) if log_file else None
    with open(instance_path) as f:
        data = json.load(f)
    
    n = data['n']
    num_scenarios = data['num_scenarios']
    cost_parameters = data['cost_parameters']
    chi = cost_parameters['chi']
    alpha = cost_parameters['alpha']
    delta = cost_parameters['delta']
    p_hubs = data.get('p_hubs')
    distances = data['distances']
    fixed_costs = data['fixed_costs']
    hub_capacities = data.get('hub_capacities')
    scenarios = data['scenarios']
    
    outgoing_flow = []
    incoming_flow = []
    for s in range(num_scenarios):
        scen = scenarios[s]
        demands = scen['demands']
        out_flow = [0] * n
        in_flow = [0] * n
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                out_flow[i] += demands[i][j]
                in_flow[j] += demands[i][j]
        outgoing_flow.append(out_flow)
        incoming_flow.append(in_flow)
    
    model = gp.Model()
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('Threads', 1)
    model.setParam('NonConvex', 2)
    model.setParam('OutputFlag', 0)
    
    z = model.addVars(n, vtype=GRB.BINARY, name="z")
    x = {}
    for s in range(num_scenarios):
        x[s] = model.addVars(n, n, vtype=GRB.BINARY, name=f"x_{s}")
    
    if p_hubs is not None:
        model.addConstr(quicksum(z[i] for i in range(n)) == p_hubs)
    else:
        model.addConstr(quicksum(z[i] for i in range(n)) >= 1)
    
    for s in range(num_scenarios):
        for i in range(n):
            model.addConstr(quicksum(x[s][i, j] for j in range(n)) == 1)
    
    for s in range(num_scenarios):
        for i in range(n):
            for j in range(n):
                model.addConstr(x[s][i, j] <= z[j])
        for i in range(n):
            model.addConstr(x[s][i, i] >= z[i])
    
    if hub_capacities is not None:
        for s in range(num_scenarios):
            for j in range(n):
                total_flow = quicksum(outgoing_flow[s][i] * x[s][i, j] for i in range(n))
                model.addConstr(total_flow <= hub_capacities[j] * z[j])
    
    A = {}
    for s in range(num_scenarios):
        A[s] = model.addVars(n, n, lb=0, name=f"A_{s}")
        for k in range(n):
            for j in range(n):
                model.addConstr(A[s][k, j] == quicksum(scenarios[s]['demands'][k][l] * x[s][l, j] for l in range(n)))
    
    fixed_cost = quicksum(fixed_costs[i] * z[i] for i in range(n))
    scenario_cost = 0
    for s in range(num_scenarios):
        prob = scenarios[s]['probability']
        linear_part = quicksum((chi * outgoing_flow[s][i] + delta * incoming_flow[s][i]) * distances[i][j] * x[s][i, j] 
                              for i in range(n) for j in range(n))
        quad_part = alpha * quicksum(distances[i][j] * x[s][k, i] * A[s][k, j] 
                                    for k in range(n) for i in range(n) for j in range(n))
        scenario_cost += prob * (linear_part + quad_part)
    total_cost = fixed_cost + scenario_cost
    model.setObjective(total_cost, GRB.MINIMIZE)
    
    model.setParam('TimeLimit', time_limit)
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            z_vals = [model.cbGetSolution(z[i]) for i in range(n)]
            hubs = [i for i in range(n) if z_vals[i] > 0.5]
            allocations = {}
            x_values = {}
            for s in range(num_scenarios):
                alloc_dict = {}
                x_dict = {}
                x_vals = model.cbGetSolution(x[s])
                for i in range(n):
                    j0 = None
                    for j in range(n):
                        if x_vals.get((i, j), 0) > 0.5:
                            j0 = j
                            break
                    alloc_dict[str(i)] = j0
                    if z_vals[i] > 0.5:
                        x_dict[str(i)] = {}
                    else:
                        x_dict[str(i)] = {str(j0): 1.0}
                allocations[str(s)] = alloc_dict
                x_values[str(s)] = x_dict
            solution_dict = {
                "objective_value": obj,
                "hubs": hubs,
                "allocations": allocations,
                "z_values": z_vals,
                "x_values": x_values
            }
            if logger:
                logger.log_solution(obj, solution_dict)
    
    model.optimize(callback if logger else None)
    
    if model.status == GRB.INFEASIBLE:
        hubs = [0]
        z_vals = [1.0] + [0.0] * (n - 1)
        allocations = {}
        x_values = {}
        obj_val = fixed_costs[0]
        for s in range(num_scenarios):
            alloc_dict = {str(i): 0 for i in range(n)}
            allocations[str(s)] = alloc_dict
            x_dict = {}
            for i in range(n):
                if i == 0:
                    x_dict[str(i)] = {}
                else:
                    x_dict[str(i)] = {"0": 1.0}
            x_values[str(s)] = x_dict
            for i in range(n):
                obj_val += scenarios[s]['probability'] * (chi * outgoing_flow[s][i] + delta * incoming_flow[s][i]) * distances[i][0]
        solution_dict = {
            "objective_value": obj_val,
            "hubs": hubs,
            "allocations": allocations,
            "z_values": z_vals,
            "x_values": x_values
        }
    else:
        z_vals = [z[i].X for i in range(n)]
        hubs = [i for i in range(n) if z_vals[i] > 0.5]
        allocations = {}
        x_values = {}
        for s in range(num_scenarios):
            alloc_dict = {}
            x_dict = {}
            for i in range(n):
                j0 = None
                for j in range(n):
                    if x[s][i, j].X > 0.5:
                        j0 = j
                        break
                alloc_dict[str(i)] = j0
                if z_vals[i] > 0.5:
                    x_dict[str(i)] = {}
                else:
                    x_dict[str(i)] = {str(j0): 1.0}
            allocations[str(s)] = alloc_dict
            x_values[str(s)] = x_dict
        solution_dict = {
            "objective_value": model.ObjVal,
            "hubs": hubs,
            "allocations": allocations,
            "z_values": z_vals,
            "x_values": x_values
        }
    
    with open(solution_path, 'w') as f:
        json.dump(solution_dict, f)

if __name__ == "__main__":
    instance_path = sys.argv[1]
    solution_path = sys.argv[2]
    time_limit = float(sys.argv[3])
    log_file = sys.argv[4] if len(sys.argv) > 4 else None
    main(instance_path, solution_path, time_limit, log_file)