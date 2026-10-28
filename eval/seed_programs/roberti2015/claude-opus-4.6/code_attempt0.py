import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    
    # Read instance data
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    num_sources = instance['num_sources']
    num_sinks = instance['num_sinks']
    supply = instance['supply']
    demand = instance['demand']
    fixed_cost = instance['fixed_cost']
    variable_cost = instance['variable_cost']
    capacity = instance['capacity']
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Build MIP model
    model = gp.Model("FCTP")
    model.setParam('OutputFlag', 1)
    model.setParam('TimeLimit', max(1, time_limit - (time.time() - start_time) - 1))
    model.setParam('Threads', 1)
    # Tuning parameters for better performance on fixed-charge problems
    model.setParam('MIPFocus', 1)  # Focus on finding good feasible solutions
    
    # Decision variables
    x = {}  # flow variables
    y = {}  # binary arc usage variables
    
    for i in range(num_sources):
        for j in range(num_sinks):
            cap = capacity[i][j]
            x[i, j] = model.addVar(lb=0, ub=cap, vtype=GRB.CONTINUOUS, 
                                     name=f'x_{i}_{j}',
                                     obj=variable_cost[i][j])
            y[i, j] = model.addVar(vtype=GRB.BINARY, name=f'y_{i}_{j}',
                                     obj=fixed_cost[i][j])
    
    model.update()
    
    # Supply constraints: sum over sinks of flow from source i = supply[i]
    for i in range(num_sources):
        model.addConstr(
            gp.quicksum(x[i, j] for j in range(num_sinks)) == supply[i],
            name=f'supply_{i}'
        )
    
    # Demand constraints: sum over sources of flow to sink j = demand[j]
    for j in range(num_sinks):
        model.addConstr(
            gp.quicksum(x[i, j] for i in range(num_sources)) == demand[j],
            name=f'demand_{j}'
        )
    
    # Linking constraints: x[i,j] <= capacity[i][j] * y[i,j]
    for i in range(num_sources):
        for j in range(num_sinks):
            cap = capacity[i][j]
            model.addConstr(
                x[i, j] <= cap * y[i, j],
                name=f'link_{i}_{j}'
            )
    
    model.setAttr('ModelSense', GRB.MINIMIZE)
    
    # Callback to log incumbent solutions
    best_obj = [float('inf')]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_obj[0] - 1e-6:
                best_obj[0] = obj
                if logger:
                    logger.log(obj)
    
    # Optimize
    model.optimize(callback)
    
    # Extract solution
    if model.SolCount > 0:
        obj_val = model.ObjVal
        flow = [[0.0] * num_sinks for _ in range(num_sources)]
        arcs_used = [[0] * num_sinks for _ in range(num_sources)]
        
        for i in range(num_sources):
            for j in range(num_sinks):
                f = x[i, j].X
                flow[i][j] = f
                arcs_used[i][j] = 1 if f > 1e-6 else 0
        
        # Recompute objective to be precise
        total_cost = 0.0
        for i in range(num_sources):
            for j in range(num_sinks):
                if arcs_used[i][j]:
                    total_cost += variable_cost[i][j] * flow[i][j] + fixed_cost[i][j]
        
        solution = {
            "objective_value": obj_val,
            "flow": flow,
            "arcs_used": arcs_used
        }
    else:
        # Return empty/infeasible solution
        solution = {
            "objective_value": float('inf'),
            "flow": [[0.0] * num_sinks for _ in range(num_sources)],
            "arcs_used": [[0] * num_sinks for _ in range(num_sources)]
        }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)