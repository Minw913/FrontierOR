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
    # x[i, j, k] = 1 if job i is assigned to machine j at position k
    x = {}
    for i in range(n):
        for j in range(m):
            for k in range(n):
                x[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")
    
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
    
    # Big-M constant for disjunctive constraints
    M = time_horizon + max(p)
    
    # Constraints
    # Each job must be assigned to exactly one machine and one position
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j, k] for j in range(m) for k in range(n)) == 1)
    
    # Each machine-position pair can have at most one job
    for j in range(m):
        for k in range(n):
            model.addConstr(gp.quicksum(x[i, j, k] for i in range(n)) <= 1)
    
    # Define start and completion times
    for i in range(n):
        model.addConstr(start[i] >= 0)
        model.addConstr(comp[i] == start[i] + p[i])
        
        # Link start time to assignment
        for j in range(m):
            for k in range(n):
                # If job i is assigned to machine j at position k, its start time is determined accordingly
                # We need to ensure that if x[i,j,k]=1, start[i] is properly positioned
                pass  # Will handle this through ordering constraints below
    
    # Ordering constraints: if job i1 is before job i2 on machine j, then comp[i1] <= start[i2]
    for j in range(m):
        for k in range(n):
            for i1 in range(n):
                for i2 in range(n):
                    if i1 != i2:
                        # If both jobs i1 and i2 are on machine j, with i1 at position k and i2 at position >k
                        # Then i1 completes before i2 starts
                        model.addConstr(
                            comp[i1] - start[i2] 
                            <= M * (3 - x[i1, j, k] - x[i2, j, k+1] if k < n-1 else 0 - (gp.quicksum(x[i2, j, kp] for kp in range(k+1, n)) if k < n-1 else 0)),
                            name=f"order_{i1}_{i2}_{j}_{k}"
                        )
    
    # Simplified ordering using binary variables
    # For each pair of jobs on the same machine, they must not overlap
    # Introduce binary variables y[i1, i2] to indicate if job i1 comes before job i2
    y = {}
    for i1 in range(n):
        for i2 in range(n):
            if i1 != i2:
                y[i1, i2] = model.addVar(vtype=GRB.BINARY, name=f"y_{i1}_{i2}")
    
    # Disjunctive constraints for non-overlapping jobs on the same machine
    for i1 in range(n):
        for i2 in range(n):
            if i1 != i2:
                # Determine if jobs are on the same machine
                same_machine_expr = gp.quicksum(x[i1, j, k] * x[i2, j, k_prime] 
                                               for j in range(m) 
                                               for k in range(n) 
                                               for k_prime in range(n))
                
                # If on the same machine, one must come before the other
                model.addConstr(comp[i1] <= start[i2] + M * (2 - same_machine_expr + y[i1, i2]))
                model.addConstr(comp[i2] <= start[i1] + M * (2 - same_machine_expr + (1 - y[i1, i2])))
    
    # Calculate tardiness
    for i in range(n):
        model.addConstr(tardiness[i] >= comp[i] - d[i])
        model.addConstr(tardiness[i] >= 0)
    
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
                for k in range(n):
                    if x[i, j, k].x > 0.5:
                        machine_assignments[j].append((k, i))
                        break
        
        # Sort each machine's jobs by position
        for j in range(m):
            machine_assignments[j].sort()
        
        # Create a mapping from job id to schedule details
        job_details = {}
        for j in range(m):
            prev_job_id = 0
            for pos, job_idx in machine_assignments[j]:
                job_id = jobs[job_idx]['id']
                s_time = int(start[job_idx].x)
                c_time = int(comp[job_idx].x)
                tard = int(tardiness[job_idx].x)
                wtard = w[job_idx] * tard
                
                job_details[job_id] = {
                    'machine_id': j,
                    'prev_job': prev_job_id,
                    'job': job_id,
                    'start_time': s_time,
                    'completion_time': c_time,
                    'weighted_tardiness': wtard
                }
                
                prev_job_id = job_id
        
        # Build the schedule list in the order of job IDs
        schedule = [job_details[job['id']] for job in jobs]
        
        # Compute the objective value
        obj_val = sum(w[i] * tardiness[i].x for i in range(n))
        
        return {
            'objective_value': obj_val,
            'schedule': schedule
        }
    
    else:
        # Infeasible or other status
        raise Exception(f"Model status: {model.status}")


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