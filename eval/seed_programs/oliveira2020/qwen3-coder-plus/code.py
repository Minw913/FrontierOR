import argparse
import json
import math
import time
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB


def solve_parallel_machine_scheduling(instance_data, time_limit):
    """
    Solve the parallel machine scheduling problem to minimize total weighted tardiness.
    
    Args:
        instance_data: Dictionary containing the instance data
        time_limit: Maximum allowed runtime in seconds
    
    Returns:
        Dictionary with the solution in the required format
    """
    # Extract instance data
    time_horizon = instance_data["time_horizon"]
    jobs = instance_data["jobs"]
    m = instance_data["parameters"]["m"]
    n = instance_data["parameters"]["n"]

    # Create mapping from job id to index for easier handling
    job_id_to_idx = {job['id']: i for i, job in enumerate(jobs)}
    
    # Prepare job data arrays
    p = [job['processing_time'] for job in jobs]  # processing times
    w = [job['weight'] for job in jobs]          # weights
    d = [job['due_date'] for job in jobs]        # due dates
    
    # Initialize Gurobi model
    model = gp.Model("ParallelMachineScheduling")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    
    # Decision variables
    # x[i, j] = 1 if job i is assigned to machine j
    x = {}
    for i in range(n):
        for j in range(m):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    
    # Start time variables for each job
    start = {}
    for i in range(n):
        start[i] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=time_horizon, name=f"start_{i}")
    
    # Completion time variables for each job
    comp = {}
    for i in range(n):
        comp[i] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=time_horizon + max(p), name=f"comp_{i}")
    
    # Tardiness variables for each job
    tardiness = {}
    for i in range(n):
        tardiness[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"tardiness_{i}")
    
    # Binary variable to indicate if job i comes before job k on machine j
    z = {}
    for j in range(m):
        for i in range(n):
            for k in range(n):
                if i != k:
                    z[i, k, j] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{k}_{j}")
    
    # Big-M constant for disjunctive constraints
    M = time_horizon + max(p)
    
    # Constraints
    # Each job must be assigned to exactly one machine
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(m)) == 1)
    
    # Define completion time
    for i in range(n):
        model.addConstr(comp[i] == start[i] + p[i])
    
    # Calculate tardiness
    for i in range(n):
        model.addConstr(tardiness[i] >= comp[i] - d[i])
        model.addConstr(tardiness[i] >= 0)
    
    # Non-overlap constraints for jobs on the same machine
    for j in range(m):
        for i in range(n):
            for k in range(i + 1, n):
                # If both jobs i and k are on machine j, they must not overlap
                model.addConstr(
                    comp[i] <= start[k] + M * (3 - x[i, j] - x[k, j] - z[i, k, j])
                )
                model.addConstr(
                    comp[k] <= start[i] + M * (2 + z[i, k, j] - x[i, j] - x[k, j])
                )
    
    # Objective function: minimize total weighted tardiness
    obj_expr = gp.quicksum(w[i] * tardiness[i] for i in range(n))
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    # Optimize
    model.optimize()
    
    # Extract solution
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        # Build schedule from the solution
        schedule = []
        machine_assignments = [[] for _ in range(m)]
        
        # Find job assignments
        for i in range(n):
            for j in range(m):
                if x[i, j].x > 0.5:
                    # Determine start time and completion time for this job
                    start_time = int(start[i].x)
                    comp_time = int(comp[i].x)
                    tard = int(tardiness[i].x)
                    wtard = w[i] * tard
                    
                    # Add to machine assignment list
                    machine_assignments[j].append({
                        'job_idx': i,
                        'job_id': jobs[i]['id'],
                        'start_time': start_time,
                        'comp_time': comp_time,
                        'tardiness': tard,
                        'weighted_tardiness': wtard
                    })
                    break
        
        # Sort each machine's jobs by start time
        for j in range(m):
            machine_assignments[j].sort(key=lambda x: x['start_time'])
        
        # Create a mapping from job id to schedule details
        job_schedule_map = {}
        for j in range(m):
            prev_job_id = 0
            for idx, job_info in enumerate(machine_assignments[j]):
                job_id = job_info['job_id']
                job_schedule_map[job_id] = {
                    'machine_id': j,
                    'prev_job': prev_job_id,
                    'job': job_id,
                    'start_time': job_info['start_time'],
                    'completion_time': job_info['comp_time'],
                    'weighted_tardiness': job_info['weighted_tardiness']
                }
                prev_job_id = job_id
        
        # Build the schedule list in the order of original job IDs
        schedule = [job_schedule_map[job['id']] for job in jobs]
        
        # Compute the objective value
        obj_val = sum(w[i] * tardiness[i].x for i in range(n))
        
        return {
            'objective_value': obj_val,
            'schedule': schedule
        }
    
    else:
        # Infeasible or other status
        print(f"Model status: {model.status}")
        raise Exception(f"Model could not be solved. Status: {model.status}")


def main():
    parser = argparse.ArgumentParser(description='Solve Parallel Machine Scheduling Problem')
    parser.add_argument('--instance_path', required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', required=True, help='Path to save the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', help='Path to log intermediate solutions (optional)')
    
    args = parser.parse_args()
    
    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Initialize logger if provided
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
    
    # Solve the problem
    start_time = time.time()
    try:
        solution = solve_parallel_machine_scheduling(instance_data, args.time_limit)
        
        # Log final solution if logger is active
        if logger:
            logger.log_solution(solution['objective_value'], solution)
        
        # Write solution to file
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
            
    except Exception as e:
        print(f"Error solving the problem: {e}")
        raise


if __name__ == '__main__':
    main()