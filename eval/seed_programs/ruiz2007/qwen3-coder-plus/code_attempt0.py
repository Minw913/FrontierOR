import argparse
import json
import math
from itertools import permutations
import gurobipy as gp
from gurobipy import GRB
import time
from solution_logger import SolutionLogger

def solve_flow_shop_scheduling(instance_data, time_limit):
    n_jobs = instance_data["n_jobs"]
    n_machines = instance_data["n_machines"]
    no_idle_machines = set(instance_data["no_idle_machines"])  # 1-indexed
    no_idle_flags = instance_data["no_idle_flags"]
    processing_times = instance_data["processing_times"]

    # Create model
    model = gp.Model("flow_shop_scheduling")
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
    
    # Decision variables
    # x[i, j] = 1 if job i is scheduled in position j
    x = {}
    for i in range(n_jobs):
        for j in range(n_jobs):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # c[j, k] = completion time of job in position j on machine k
    c = {}
    for j in range(n_jobs):
        for k in range(n_machines):
            c[j, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"c_{j}_{k}")

    # Add constraints
    # Each job appears exactly once in the sequence
    for i in range(n_jobs):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n_jobs)) == 1, name=f"job_assigned_{i}")

    # Each position has exactly one job
    for j in range(n_jobs):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n_jobs)) == 1, name=f"pos_filled_{j}")

    # Calculate completion times
    # Machine 0 constraints
    for j in range(n_jobs):
        # Sum over all jobs placed in position j
        expr = gp.LinExpr()
        for i in range(n_jobs):
            expr += x[i, j] * processing_times[i][0]
        model.addConstr(c[j, 0] >= expr, name=f"comp_time_machine0_pos{j}")
        
        if j > 0:
            # For regular machines, we can have idle time, so c[j,k] >= c[j-1,k] + p[current_job,k]
            # For no-idle machines, no idle time allowed: c[j,k] = c[j-1,k] + p[current_job,k]
            if no_idle_flags[0]:
                model.addConstr(c[j, 0] == c[j-1, 0] + expr, name=f"no_idle_machine0_pos{j}")
            else:
                model.addConstr(c[j, 0] >= c[j-1, 0] + expr, name=f"reg_machine0_pos{j}")

    # Machine 1 to m-1 constraints
    for k in range(1, n_machines):
        for j in range(n_jobs):
            # Sum over all jobs placed in position j
            expr = gp.LinExpr()
            for i in range(n_jobs):
                expr += x[i, j] * processing_times[i][k]
            
            # c[j, k] >= c[j, k-1] + processing time of job in position j on machine k (job dependent)
            model.addConstr(c[j, k] >= c[j, k-1] + expr, name=f"job_transfer_pos{j}_mach{k}")

            # c[j, k] >= c[j-1, k] + processing time of job in position j on machine k (machine dependent) 
            if j > 0:
                if no_idle_flags[k]:  # No-idle machine
                    model.addConstr(c[j, k] == c[j-1, k] + expr, name=f"no_idle_pos{j}_mach{k}")
                else:  # Regular machine
                    model.addConstr(c[j, k] >= c[j-1, k] + expr, name=f"reg_pos{j}_mach{k}")

    # Minimize the makespan (last job's completion time on last machine)
    makespan = c[n_jobs - 1, n_machines - 1]
    model.setObjective(makespan, GRB.MINIMIZE)

    # Optimize
    model.optimize()

    # Extract solution
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        # Get the sequence by checking the x variables
        sequence = [None] * n_jobs
        for i in range(n_jobs):
            for j in range(n_jobs):
                if x[i, j].x > 0.5:
                    sequence[j] = i
                    
        # Compute final completion times
        completion_times = {}
        for k in range(n_machines):
            completion_times[f"machine_{k}"] = [
                round(c[j, k].x) for j in range(n_jobs)
            ]
        
        return {
            "objective_value": model.objVal,
            "sequence": sequence,
            "completion_times": completion_times
        }
    
    return None    

def main():
    parser = argparse.ArgumentParser(description='Solve Flow Shop Scheduling with No-Idle Constraint')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to output solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log file for intermediate solutions')

    args = parser.parse_args()
    
    # Initialize logger if provided
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    solution = solve_flow_shop_scheduling(instance_data, args.time_limit)
    
    if solution:
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
    else:
        print("No solution found within the given time limit")


if __name__ == "__main__":
    main()