import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
def solve_logistics_problem(instance_data, time_limit, log_path):
    # Extract instance data
    num_sources = instance_data['num_sources']
    num_sinks = instance_data['num_sinks']
    supply = instance_data['supply']
    demand = instance_data['demand']
    fixed_cost = instance_data['fixed_cost']
    variable_cost = instance_data['variable_cost']
    capacity = instance_data['capacity']

    # Create model
    model = gp.Model("Logistics_Network")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam(GRB.Param.TimeLimit, time_limit)
    
    # Define decision variables
    flow = []
    arcs_used = []
    for i in range(num_sources):
        flow_row = []
        arcs_row = []
        for j in range(num_sinks):
            # Flow variable (continuous)
            x = model.addVar(lb=0, ub=capacity[i][j], name=f"flow_{i}_{j}")
            flow_row.append(x)
            
            # Binary indicator if arc is used
            y = model.addVar(vtype=GRB.BINARY, name=f"arc_used_{i}_{j}")
            arcs_row.append(y)
        flow.append(flow_row)
        arcs_used.append(arcs_row)
    
    # Add constraints
    # Supply constraint for each source
    for i in range(num_sources):
        model.addConstr(gp.quicksum(flow[i][j] for j in range(num_sinks)) == supply[i], f"supply_{i}")
    
    # Demand constraint for each sink
    for j in range(num_sinks):
        model.addConstr(gp.quicksum(flow[i][j] for i in range(num_sources)) == demand[j], f"demand_{j}")
    
    # Capacity/usage constraints: if flow > 0, then arc is used
    for i in range(num_sources):
        for j in range(num_sinks):
            # Model flow <= capacity * arcs_used
            model.addConstr(flow[i][j] <= capacity[i][j] * arcs_used[i][j], f"capacity_flow_{i}_{j}")
    
    # Set objective function
    total_var_cost = gp.quicksum(variable_cost[i][j] * flow[i][j] 
                                 for i in range(num_sources) for j in range(num_sinks))
    total_fixed_cost = gp.quicksum(fixed_cost[i][j] * arcs_used[i][j] 
                                   for i in range(num_sources) for j in range(num_sinks))
    model.setObjective(total_var_cost + total_fixed_cost, GRB.MINIMIZE)
    
    # Solve the model
    if log_path:
        logger = SolutionLogger(log_path, sense="minimize")
        
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj_val)
                
        model.optimize(callback)
    else:
        model.optimize()

    # Prepare solution
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        # Extract flow values
        flow_values = [[flow[i][j].x for j in range(num_sinks)] for i in range(num_sources)]
        arcs_values = [[int(round(arcs_used[i][j].x)) for j in range(num_sinks)] for i in range(num_sources)]
        
        # Calculate actual objective value
        total_var_cost = sum(
            variable_cost[i][j] * flow_values[i][j] 
            for i in range(num_sources) for j in range(num_sinks)
        )
        total_fixed_cost = sum(
            fixed_cost[i][j] * arcs_values[i][j] 
            for i in range(num_sources) for j in range(num_sinks)
        )
        objective_value = total_var_cost + total_fixed_cost
        
        solution = {
            "objective_value": objective_value,
            "flow": flow_values,
            "arcs_used": arcs_values
        }
    else:
        # In case the model is infeasible or another issue
        solution = {
            "objective_value": float('inf'),
            "flow": [[0 for _ in range(num_sinks)] for _ in range(num_sources)],
            "arcs_used": [[0 for _ in range(num_sinks)] for _ in range(num_sources)]
        }

    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    
    args = parser.parse_args()
    
    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Solve the problem
    solution = solve_logistics_problem(instance_data, args.time_limit, args.log_path)
    
    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()