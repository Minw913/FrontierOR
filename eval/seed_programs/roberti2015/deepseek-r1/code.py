import argparse
import json
import gurobipy as grb
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser(description='Solve Fixed-Charge Transportation Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the JSON file containing the problem instance')
    parser.add_argument('--solution_path', type=str, required=True, help='Path where the final solution JSON file must be written')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to a JSONL file for logging intermediate solutions')
    args = parser.parse_args()

    # Read instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    num_sources = data['num_sources']
    num_sinks = data['num_sinks']
    supply = data['supply']
    demand = data['demand']
    fixed_cost = data['fixed_cost']
    variable_cost = data['variable_cost']
    capacity = data['capacity']

    # Initialize solution logger if log_path provided
    logger = None
    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="minimize")

    # Create Gurobi model
    model = grb.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)

    # Create variables
    x = []  # Flow variables
    y = []  # Binary usage variables
    for i in range(num_sources):
        x_row = []
        y_row = []
        for j in range(num_sinks):
            x_var = model.addVar(lb=0.0, ub=capacity[i][j], vtype=grb.GRB.CONTINUOUS, name=f'x_{i}_{j}')
            y_var = model.addVar(vtype=grb.GRB.BINARY, name=f'y_{i}_{j}')
            x_row.append(x_var)
            y_row.append(y_var)
        x.append(x_row)
        y.append(y_row)

    # Set objective: minimize total cost (variable + fixed)
    obj = grb.quicksum(
        variable_cost[i][j] * x[i][j] + fixed_cost[i][j] * y[i][j]
        for i in range(num_sources) for j in range(num_sinks)
    )
    model.setObjective(obj, grb.GRB.MINIMIZE)

    # Supply constraints
    for i in range(num_sources):
        model.addConstr(
            grb.quicksum(x[i][j] for j in range(num_sinks)) == supply[i],
            f'supply_{i}'
        )

    # Demand constraints
    for j in range(num_sinks):
        model.addConstr(
            grb.quicksum(x[i][j] for i in range(num_sources)) == demand[j],
            f'demand_{j}'
        )

    # Linking constraints: if flow > 0 then y[i][j] = 1
    for i in range(num_sources):
        for j in range(num_sinks):
            model.addConstr(
                x[i][j] <= capacity[i][j] * y[i][j],
                f'link_{i}_{j}'
            )

    # Callback function for logging intermediate solutions
    def callback_wrapper(logger_ref):
        best_obj = float('inf')
        def callback(model, where):
            nonlocal best_obj
            if where == grb.GRB.Callback.MIPSOL:
                obj_val = model.cbGet(grb.GRB.Callback.MIPSOL_OBJ)
                if obj_val < best_obj:
                    best_obj = obj_val
                    logger_ref.log(obj_val)
        return callback

    # Optimize model
    if logger:
        model.optimize(callback_wrapper(logger))
    else:
        model.optimize()

    # Process solution
    if model.SolCount > 0:
        objective_value = model.ObjVal
        flow = []
        arcs_used = []
        for i in range(num_sources):
            flow_row = []
            arcs_row = []
            for j in range(num_sinks):
                flow_row.append(x[i][j].X)
                arcs_row.append(int(round(y[i][j].X)))
            flow.append(flow_row)
            arcs_used.append(arcs_row)
    else:
        # Fallback: construct a feasible solution if solver fails
        # This should not happen due to problem feasibility, but included for robustness
        flow = [[0.0] * num_sinks for _ in range(num_sources)]
        arcs_used = [[0] * num_sinks for _ in range(num_sources)]
        
        # Simple greedy assignment to satisfy constraints
        s_idx = 0
        d_idx = 0
        remaining_supply = supply.copy()
        remaining_demand = demand.copy()
        
        while s_idx < num_sources and d_idx < num_sinks:
            amt = min(remaining_supply[s_idx], remaining_demand[d_idx], capacity[s_idx][d_idx])
            if amt > 0:
                flow[s_idx][d_idx] = amt
                arcs_used[s_idx][d_idx] = 1
                remaining_supply[s_idx] -= amt
                remaining_demand[d_idx] -= amt
            
            if remaining_supply[s_idx] == 0:
                s_idx += 1
            if remaining_demand[d_idx] == 0:
                d_idx += 1
        
        # Calculate objective value for the fallback solution
        objective_value = 0.0
        for i in range(num_sources):
            for j in range(num_sinks):
                objective_value += variable_cost[i][j] * flow[i][j]
                if arcs_used[i][j]:
                    objective_value += fixed_cost[i][j]
        
        # Log fallback solution if logger exists
        if logger:
            logger.log(objective_value)

    # Write solution
    solution = {
        "objective_value": objective_value,
        "flow": flow,
        "arcs_used": arcs_used
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()