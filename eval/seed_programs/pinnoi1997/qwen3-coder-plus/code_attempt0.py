import argparse
import json
import math
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB

def solve_assembly_line_balance(instance_data, time_limit):
    # Extract parameters
    num_tasks = instance_data["num_tasks"]
    cycle_time = instance_data["cycle_time"]
    num_stations = instance_data["num_stations"]
    task_processing_times = instance_data["task_processing_times"]
    precedence_arcs = instance_data["precedence_arcs"]
    earliest_station = instance_data["earliest_station"]
    latest_station = instance_data["latest_station"]

    # Create model
    m = gp.Model()
    
    # Turn off verbose output (only print warnings/errors)
    m.setParam('OutputFlag', 0)
    
    # Set time limit
    m.setParam('TimeLimit', time_limit)

    # Decision variables: binary x[task, station] = 1 if task is assigned to station
    x = {}
    valid_assignments = []
    for i in range(num_tasks):
        for j in range(earliest_station[str(i)], min(latest_station[str(i)] + 1, num_stations)):
            x[i, j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
            valid_assignments.append((i, j))
            
    # Objective: Minimize the max idle time across all stations (which is min max unused capacity)
    # This can be modeled by a var "idle" that is >= cycle_time - actual_total_per_station
    idle = m.addVar(vtype=GRB.CONTINUOUS, name="idle")
    
    # Each task is assigned to exactly one station
    for i in range(num_tasks):
        m.addConstr(
            gp.quicksum(x[i, j] for (t, s) in valid_assignments if t == i) == 1,
            f"assign_task_{i}"
        )
    
    # Precedence constraints: If (i,j) is an arc, then station(i) <= station(j)
    for pred, succ in precedence_arcs:
        # For each possible assignment of i and j, ensure station order
        for (pred_t, station_p) in valid_assignments:
            if pred_t != pred:
                continue
            # Check all assignments of successor to later or same stations
            for (succ_t, station_s) in valid_assignments:
                if succ_t == succ and station_s < station_p:
                    m.addConstr(x[pred_t, station_p] + x[succ_t, station_s] <= 1)
    
    # Capacity constraint: total time per station should not exceed cycle_time
    for station in range(num_stations):
        # Get all tasks that could possibly be assigned to this station
        tasks_for_station = [(t, p) for (t, s) in valid_assignments if s == station]
        if tasks_for_station:
            m.addConstr(
                gp.quicksum(task_processing_times[str(t)] * x[t, station] for (t, _) in tasks_for_station) 
                <= cycle_time,
                f"capacity_station_{station}"
            )
    
    # Define the relationship for idle time: idle >= max(cycle_time - used_capacity_in_s)
    for station in range(num_stations):
        tasks_for_station = [(t, p) for (t, s) in valid_assignments if s == station]
        if tasks_for_station:
            total_used = gp.quicksum(task_processing_times[str(t)] * x[t, station] for (t, _) in tasks_for_station)
            m.addConstr(idle >= cycle_time - total_used, f"idle_def_station_{station}")
        else:
            # If no tasks can go to this station, idle at least equals cycle_time
            m.addConstr(idle >= cycle_time, f"idle_empty_station_{station}")
    
    # Set objective
    m.setObjective(idle, GRB.MINIMIZE)

    # Optimize
    m.optimize()

    # Extract solution
    if m.SolCount > 0:
        task_assignments = {}
        for i in range(num_tasks):
            for j in range(earliest_station[str(i)], min(latest_station[str(i)] + 1, num_stations)):
                if x[i, j].X > 0.5:
                    task_assignments[str(i)] = j
                    break
        
        # Calculate actual idle value (for verification)
        # Recompute based on station loads (this is just for consistency check)
        station_loads = [0 for _ in range(num_stations)]
        for task_str, station in task_assignments.items():
            task_idx = int(task_str)
            station_loads[station] += task_processing_times[str(task_idx)]
        
        idle_times = [cycle_time - load for load in station_loads]
        actual_idle = max(idle_times)
        
        # Use the variable value to handle potential numerical differences
        obj_val = m.objVal
        
        return {
            "objective_value": float(obj_val),
            "task_assignments": task_assignments
        }
    else:
        print("No solution found!")
        return None


def main():
    parser = argparse.ArgumentParser(description='Solve Assembly Line Balancing Problem')
    parser.add_argument('--instance_path', type=str, required=True, 
                        help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, 
                        help='Path where the solution JSON file must be written')
    parser.add_argument('--time_limit', type=int, default=float('inf'), 
                        help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, help='Path for logging intermediate solutions')
    args = parser.parse_args()
    
    # Import the logger after arg parsing
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None
    
    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Solve the problem
    solution = solve_assembly_line_balance(instance_data, args.time_limit)
    
    # If we have a solution, log it and save it
    if solution:
        if logger:
            logger.log(solution['objective_value'])
        
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
    else:
        with open(args.solution_path, 'w') as f:
            json.dump({"error": "No feasible solution found"}, f)
    

if __name__ == "__main__":
    main()