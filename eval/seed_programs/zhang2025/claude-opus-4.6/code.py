import argparse
import json
import time
import sys

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    # Read instance
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    num_trips = instance['parameters']['num_trips_M']
    num_duties = instance['parameters']['num_duties_N']
    duties = instance['duties']
    costs = instance['costs']
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    start_time = time.time()
    
    # This is a Set Cover / Set Partitioning problem
    # We need to cover each trip exactly once (partitioning)
    # Minimize total cost of selected duties
    
    # Build trip-to-duties mapping
    trip_to_duties = [[] for _ in range(num_trips)]
    for j, duty_trips in enumerate(duties):
        for t in duty_trips:
            trip_to_duties[t].append(j)
    
    # Use Gurobi to solve the set partitioning problem
    import gurobipy as gp
    from gurobipy import GRB
    
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time <= 0:
        remaining_time = 1
    
    model = gp.Model("set_partitioning")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', max(1, remaining_time - 2))
    model.setParam('Threads', 1)
    
    # Decision variables: x[j] = 1 if duty j is selected
    x = model.addVars(num_duties, vtype=GRB.BINARY, name="x")
    
    # Objective: minimize total cost
    model.setObjective(
        gp.quicksum(costs[j] * x[j] for j in range(num_duties)),
        GRB.MINIMIZE
    )
    
    # Constraints: each trip must be covered exactly once
    for t in range(num_trips):
        model.addConstr(
            gp.quicksum(x[j] for j in trip_to_duties[t]) == 1,
            name=f"trip_{t}"
        )
    
    # Callback to log incumbent solutions
    best_obj = [float('inf')]
    best_solution = [None]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_obj[0] - 1e-9:
                best_obj[0] = obj
                vals = model.cbGetSolution(x)
                selected = [j for j in range(num_duties) if vals[j] > 0.5]
                best_solution[0] = selected
                if logger:
                    logger.log(obj)
    
    model.optimize(callback)
    
    # Extract solution
    if model.SolCount > 0:
        selected_duties = [j for j in range(num_duties) if x[j].X > 0.5]
        objective_value = sum(costs[j] for j in selected_duties)
    elif best_solution[0] is not None:
        selected_duties = best_solution[0]
        objective_value = best_obj[0]
    else:
        # Fallback: try greedy approach
        selected_duties = []
        objective_value = 0.0
        covered = set()
        
        # Sort duties by cost per trip (efficiency)
        duty_efficiency = []
        for j in range(num_duties):
            uncovered_trips = [t for t in duties[j] if t not in covered]
            if len(uncovered_trips) > 0:
                duty_efficiency.append((costs[j] / len(uncovered_trips), j))
        
        duty_efficiency.sort()
        
        while len(covered) < num_trips:
            best_ratio = float('inf')
            best_duty = -1
            for j in range(num_duties):
                uncovered_trips = [t for t in duties[j] if t not in covered]
                if len(uncovered_trips) > 0:
                    ratio = costs[j] / len(uncovered_trips)
                    if ratio < best_ratio:
                        best_ratio = ratio
                        best_duty = j
            
            if best_duty == -1:
                break
            
            selected_duties.append(best_duty)
            objective_value += costs[best_duty]
            for t in duties[best_duty]:
                covered.add(t)
    
    # Write solution
    solution = {
        "objective_value": objective_value,
        "selected_duties": selected_duties
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)