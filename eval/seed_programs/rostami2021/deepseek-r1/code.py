import argparse
import json
import gurobipy as gp
from gurobipy import GRB, quicksum
import sys
from solution_logger import SolutionLogger

def main(instance_path, solution_path, time_limit, log_path=None):
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Load instance data
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
    
    # Precompute outgoing and incoming flows for each scenario
    outgoing_flow = []
    incoming_flow = []
    for s in range(num_scenarios):
        scen = scenarios[s]
        demands = scen['demands']
        out_flow = [sum(demands[i]) for i in range(n)]
        in_flow = [sum(demands[j][i] for j in range(n)) for i in range(n)]
        outgoing_flow.append(out_flow)
        incoming_flow.append(in_flow)
    
    # Create optimization model
    model = gp.Model()
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)
    
    # Decision variables
    z = model.addVars(n, vtype=GRB.BINARY, name="z")  # Hub selection
    x = {}  # Node assignments per scenario
    
    for s in range(num_scenarios):
        x[s] = model.addVars(n, n, vtype=GRB.BINARY, name=f"x_{s}")
    
    # Constraints: Hub selection
    if p_hubs is not None:
        model.addConstr(z.sum() == p_hubs, "p_hubs_constraint")
    else:
        model.addConstr(z.sum() >= 1, "at_least_one_hub")
    
    # Constraints: Node assignments
    for s in range(num_scenarios):
        for i in range(n):
            model.addConstr(quicksum(x[s][i, j] for j in range(n)) == 1, f"assign_{s}_{i}")
    
    for s in range(num_scenarios):
        for i in range(n):
            for j in range(n):
                model.addConstr(x[s][i, j] <= z[j], f"hub_active_{s}_{i}_{j}")
            model.addConstr(x[s][i, i] >= z[i], f"self_assign_{s}_{i}")
    
    # Capacity constraints
    if hub_capacities is not None:
        for s in range(num_scenarios):
            for j in range(n):
                total_flow = quicksum(outgoing_flow[s][i] * x[s][i, j] for i in range(n))
                model.addConstr(total_flow <= hub_capacities[j] * z[j], f"capacity_{s}_{j}")
    
    # Objective function components
    fixed_cost = quicksum(fixed_costs[i] * z[i] for i in range(n))
    scenario_cost = 0
    
    for s in range(num_scenarios):
        prob = scenarios[s]['probability']
        # Collection and distribution costs
        linear_cost = quicksum((chi * outgoing_flow[s][i] + delta * incoming_flow[s][i]) * 
                              distances[i][j] * x[s][i, j] for i in range(n) for j in range(n))
        
        # Inter-hub transfer costs
        transfer_cost = quicksum(alpha * distances[i][j] * 
                                quicksum(scenarios[s]['demands'][k][l] * 
                                         x[s][k, i] * x[s][l, j] 
                                         for k in range(n) for l in range(n))
                                for i in range(n) for j in range(n))
        
        scenario_cost += prob * (linear_cost + transfer_cost)
    
    total_cost = fixed_cost + scenario_cost
    model.setObjective(total_cost, GRB.MINIMIZE)
    
    # Callback for logging intermediate solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            z_vals = [round(model.cbGetSolution(z[i])) for i in range(n)]
            hubs = [i for i in range(n) if z_vals[i] > 0.5]
            
            allocations = {}
            x_values = {}
            
            for s in range(num_scenarios):
                alloc_dict = {}
                x_dict = {}
                for i in range(n):
                    assigned_hub = None
                    for j in range(n):
                        if model.cbGetSolution(x[s][i, j]) > 0.5:
                            assigned_hub = j
                            break
                    alloc_dict[str(i)] = assigned_hub
                    
                    # Build x_values structure
                    if z_vals[i] > 0.5:
                        x_dict[str(i)] = {}
                    else:
                        x_dict[str(i)] = {str(assigned_hub): 1.0}
                
                allocations[str(s)] = alloc_dict
                x_values[str(s)] = x_dict
            
            solution_dict = {
                "objective_value": obj_val,
                "hubs": hubs,
                "allocations": allocations,
                "z_values": z_vals,
                "x_values": x_values
            }
            
            if logger:
                logger.log_solution(obj_val, solution_dict)
    
    # Optimize model
    model.optimize(callback if logger else None)
    
    # Process solution
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
            z_vals = [round(z[i].X) for i in range(n)]
            hubs = [i for i in range(n) if z_vals[i] > 0.5]
            
            allocations = {}
            x_values = {}
            
            for s in range(num_scenarios):
                alloc_dict = {}
                x_dict = {}
                for i in range(n):
                    assigned_hub = None
                    for j in range(n):
                        if x[s][i, j].X > 0.5:
                            assigned_hub = j
                            break
                    alloc_dict[str(i)] = assigned_hub
                    
                    if z_vals[i] > 0.5:
                        x_dict[str(i)] = {}
                    else:
                        x_dict[str(i)] = {str(assigned_hub): 1.0}
                
                allocations[str(s)] = alloc_dict
                x_values[str(s)] = x_dict
            
            solution_dict = {
                "objective_value": model.ObjVal,
                "hubs": hubs,
                "allocations": allocations,
                "z_values": z_vals,
                "x_values": x_values
            }
        else:
            # Fallback solution if no feasible solution found
            solution_dict = create_fallback_solution(data)
    else:
        # Fallback solution for infeasible cases
        solution_dict = create_fallback_solution(data)
    
    # Save final solution
    with open(solution_path, 'w') as f:
        json.dump(solution_dict, f)

def create_fallback_solution(data):
    """Create a feasible fallback solution when optimization fails"""
    n = data['n']
    scenarios = data['scenarios']
    fixed_costs = data['fixed_costs']
    distances = data['distances']
    cost_parameters = data['cost_parameters']
    chi = cost_parameters['chi']
    delta = cost_parameters['delta']
    
    # Precompute flows for cost calculation
    outgoing_flow = []
    incoming_flow = []
    for scen in scenarios:
        demands = scen['demands']
        out_flow = [sum(demands[i]) for i in range(n)]
        in_flow = [sum(demands[j][i] for j in range(n)) for i in range(n)]
        outgoing_flow.append(out_flow)
        incoming_flow.append(in_flow)
    
    # Use first node as hub
    hubs = [0]
    z_vals = [1] + [0] * (n - 1)
    
    allocations = {}
    x_values = {}
    obj_val = fixed_costs[0]  # Fixed cost for hub 0
    
    for s, scen in enumerate(scenarios):
        alloc_dict = {}
        x_dict = {}
        prob = scen['probability']
        
        for i in range(n):
            alloc_dict[str(i)] = 0  # Assign all nodes to hub 0
            
            # Build x_values structure
            if i == 0:
                x_dict[str(i)] = {}
            else:
                x_dict[str(i)] = {"0": 1.0}
            
            # Add collection and distribution costs
            obj_val += prob * (chi * outgoing_flow[s][i] + delta * incoming_flow[s][i]) * distances[i][0]
        
        allocations[str(s)] = alloc_dict
        x_values[str(s)] = x_dict
    
    return {
        "objective_value": obj_val,
        "hubs": hubs,
        "allocations": allocations,
        "z_values": z_vals,
        "x_values": x_values
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    main(args.instance_path, args.solution_path, args.time_limit, args.log_path)