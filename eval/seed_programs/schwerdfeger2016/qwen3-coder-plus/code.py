import argparse
import json
import math
from typing import Dict, List
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
def solve_instance(num_machines: int, num_jobs: int, processing_times: List[int]):
    """
    Solve the scheduling problem with Gurobi to minimize the sum of squared completion times.
    """
    # Create model
    model = gp.Model("ParallelMachineScheduling")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam('TimeLimit', args.time_limit)
    
    avg_load = sum(processing_times) / num_machines
    
    # Variables: x[machine][job] binary variables indicating assignment
    x = {}
    for m in range(num_machines):
        for j in range(num_jobs):
            x[m, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{m}_{j}")
    
    # Variables: completion_time[machine] continuous variables for machine loads
    completion_time = {}
    for m in range(num_machines):
        completion_time[m] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"completion_time_{m}")
    
    # Constraints: Each job assigned to exactly one machine
    for j in range(num_jobs):
        model.addConstr(gp.quicksum(x[m, j] for m in range(num_machines)) == 1)
    
    # Define the completion time variables
    for m in range(num_machines):
        model.addConstr(completion_time[m] == gp.quicksum(processing_times[j] * x[m, j] for j in range(num_jobs)))
    
    # Objective: minimize sum of squared completion times
    model.setObjective(gp.quicksum(completion_time[m] * completion_time[m] for m in range(num_machines)), GRB.MINIMIZE)
    
    # Set callback to log incumbent solutions
    def callback(cb_model, cb_where):
        if cb_where == GRB.Callback.MIPSOL:
            obj_val = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                current_assignment = {}
                for m in range(num_machines):
                    assigned_jobs = []
                    for j in range(num_jobs):
                        if cb_model.cbGetSolution(x[m, j]) > 0.5:
                            assigned_jobs.append(j)
                    current_assignment[str(m)] = assigned_jobs
                
                current_machine_times = {}
                for m in range(num_machines):
                    time = sum(processing_times[j] for j in current_assignment[str(m)])
                    current_machine_times[str(m)] = time
                
                avg_load = sum(processing_times) / num_machines
                sum_sq_deviation = sum((current_machine_times[str(m)] - avg_load)**2 for m in range(num_machines))
                final_objective = math.sqrt(sum_sq_deviation) / avg_load

                logger.log(final_objective)
                
    # Optimize
    model.optimize(callback)
    
    # Extract solution
    assignment = {}
    machine_completion_times = {}
    
    for m in range(num_machines):
        assigned_jobs = []
        for j in range(num_jobs):
            if x[m, j].X > 0.5:
                assigned_jobs.append(j)
        assignment[str(m)] = assigned_jobs
        machine_completion_times[str(m)] = sum(processing_times[j] for j in assigned_jobs)
    
    # Calculate objective value
    avg_load = sum(processing_times) / num_machines
    sum_sq_deviation = sum((machine_completion_times[str(m)] - avg_load)**2 for m in range(num_machines))
    final_objective = math.sqrt(sum_sq_deviation) / avg_load

    return {
        "objective_value": final_objective,
        "assignment": assignment,
        "machine_completion_times": machine_completion_times
    }


def solve_heuristic_first(num_machines: int, num_jobs: int, processing_times: List[int]):
    """
    First fit decreasing heuristic for initial improvement.
    This assigns jobs in descending order of size to the machine with the smallest current load.
    """
    assignment = {str(m): [] for m in range(num_machines)}
    machine_completion_times = {str(m): 0 for m in range(num_machines)}
    
    # Sort jobs by processing time in decreasing order (already done per input specification)
    for j in range(num_jobs):
        # Find the machine with the minimum completion time so far
        min_machine_idx = min(range(num_machines), 
                             key=lambda m: machine_completion_times[str(m)])
        
        assignment[str(min_machine_idx)].append(j)
        machine_completion_times[str(min_machine_idx)] += processing_times[j]
    
    # Calculate objective
    avg_load = sum(processing_times) / num_machines
    sum_sq_deviation = sum((machine_completion_times[str(m)] - avg_load)**2 
                          for m in range(num_machines))
    final_objective = math.sqrt(sum_sq_deviation) / avg_load

    return {
        "objective_value": final_objective,
        "assignment": assignment,
        "machine_completion_times": machine_completion_times
    }


def main(args):
    global logger
    # Load the instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    num_machines = instance_data["num_machines"]
    num_jobs = instance_data["num_jobs"]
    processing_times = instance_data["processing_times"]
    
    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Start with a heuristic solution as fallback if needed
    best_solution = solve_heuristic_first(num_machines, num_jobs, processing_times)
    best_obj = best_solution["objective_value"]

    # Update logger with the heuristic solution
    if logger:
        logger.log(best_obj)

    try:
        # Solve with MIP
        mip_solution = solve_instance(num_machines, num_jobs, processing_times)
        if mip_solution and mip_solution["objective_value"] < best_obj:
            best_solution = mip_solution
    except Exception as e:
        print(f"Gurobi solver error: {e}")
        # Fallback to best known solution (heuristic)
        pass

    # Write the final solution
    with open(args.solution_path, 'w') as f:
        json.dump(best_solution, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve parallel machine scheduling problem.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, required=False, help="Path to log file")
    
    args = parser.parse_args()
    
    main(args)