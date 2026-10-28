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
    
    num_machines = instance['num_machines']
    num_jobs = instance['num_jobs']
    cost_matrix = instance['cost_matrix']
    resource_matrix = instance['resource_matrix']
    capacities = instance['capacities']
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Create Gurobi model
    model = gp.Model("GAP")
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', 1)
    
    elapsed = time.time() - start_time
    remaining = max(1, time_limit - elapsed)
    model.setParam('TimeLimit', remaining)
    
    # Decision variables: x[i][j] = 1 if job j is assigned to machine i
    x = {}
    for i in range(num_machines):
        for j in range(num_jobs):
            x[i, j] = model.addVar(vtype=GRB.BINARY, obj=cost_matrix[i][j],
                                    name=f"x_{i}_{j}")
    
    model.update()
    
    # Each job must be assigned to exactly one machine
    for j in range(num_jobs):
        model.addConstr(
            gp.quicksum(x[i, j] for i in range(num_machines)) == 1,
            name=f"job_{j}"
        )
    
    # Capacity constraints for each machine
    for i in range(num_machines):
        model.addConstr(
            gp.quicksum(resource_matrix[i][j] * x[i, j] for j in range(num_jobs)) <= capacities[i],
            name=f"cap_{i}"
        )
    
    model.setAttr('ModelSense', GRB.MINIMIZE)
    
    # Store best solution found
    best_solution = None
    best_obj = float('inf')
    
    # Callback to log incumbent solutions
    def callback(model, where):
        nonlocal best_solution, best_obj
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_obj - 1e-6:
                best_obj = obj
                # Extract solution
                vals = model.cbGetSolution([x[i, j] for i in range(num_machines) for j in range(num_jobs)])
                assignments = {}
                idx = 0
                for i in range(num_machines):
                    for j in range(num_jobs):
                        if vals[idx] > 0.5:
                            assignments[j] = i
                        idx += 1
                best_solution = assignments
                if logger:
                    logger.log(obj)
    
    model.optimize(callback)
    
    # Extract final solution
    if model.SolCount > 0:
        obj_val = model.ObjVal
        assignments = {}
        for j in range(num_jobs):
            for i in range(num_machines):
                if x[i, j].X > 0.5:
                    assignments[j] = i
                    break
        
        if obj_val < best_obj - 1e-6:
            best_obj = obj_val
            best_solution = assignments
            if logger:
                logger.log(obj_val)
    
    if best_solution is None:
        # Fallback: try to find a greedy feasible solution
        remaining_cap = list(capacities)
        assignments = {}
        total_cost = 0
        feasible = True
        
        for j in range(num_jobs):
            best_machine = -1
            best_cost_j = float('inf')
            for i in range(num_machines):
                if resource_matrix[i][j] <= remaining_cap[i] and cost_matrix[i][j] < best_cost_j:
                    best_cost_j = cost_matrix[i][j]
                    best_machine = i
            if best_machine == -1:
                feasible = False
                break
            assignments[j] = best_machine
            remaining_cap[best_machine] -= resource_matrix[best_machine][j]
            total_cost += best_cost_j
        
        if feasible:
            best_obj = total_cost
            best_solution = assignments
            if logger:
                logger.log(total_cost)
    
    # Write solution
    solution = {
        "objective_value": best_obj,
        "assignments": {str(j): best_solution[j] for j in best_solution}
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