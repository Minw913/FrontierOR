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
    total_processing_time = instance_data["total_processing_time"] if "total_processing_time" in instance_data else sum(task_processing_times.values())

    # Create model (disable all output)
    m = gp.Model("AssemblyLineBalance")
    m.setParam("Threads", 1)
    m.setParam('OutputFlag', 0)
    m.setParam('LogToConsole', 0)
    m.setParam('LogFile', '')
    m.setParam('TimeLimit', time_limit)

    # Decision variables: binary x[task, station] = 1 if task is assigned to station
    x = {}
    possible_assignments = []  # store possible assignments to iterate later
    for i in range(num_tasks):
        for j in range(earliest_station.get(str(i), 0), min(latest_station.get(str(i), num_stations - 1) + 1, num_stations)):
            x[i, j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
            possible_assignments.append((i, j))

    # Variable for the maximum station idle time we want to minimize
    idle = m.addVar(vtype=GRB.CONTINUOUS, name="idle")

    # Each task is assigned to exactly one station
    for i in range(num_tasks):
        m.addConstr(
            gp.quicksum(x[i, j] for (t, s) in possible_assignments if t == i) == 1,
            f"assign_task_{i}"
        )

    # Precedence constraints: if (i,j) is an arc -> task i must be processed at station <= station of task j
    for pred, succ in precedence_arcs:
        # Get valid stations for both tasks
        possible_stations_pred = [
            s for s in range(
                earliest_station.get(str(pred), 0),
                min(latest_station.get(str(pred), num_stations - 1) + 1, num_stations)
            )
        ]
        possible_stations_succ = [
            s for s in range(
                earliest_station.get(str(succ), 0),
                min(latest_station.get(str(succ), num_stations - 1) + 1, num_stations)
            )
        ]

        # For each invalid pair s_pred > s_succ, add a constraint  
        for station_pred in possible_stations_pred:
            for station_succ in possible_stations_succ:
                if station_pred > station_succ:
                    # Ensure we don't assign pred to station_pred and succ to station_succ simultaneously
                    if (pred, station_pred) in x and (succ, station_succ) in x:
                        m.addConstr(
                            x[pred, station_pred] + x[succ, station_succ] <= 1,
                            f"precedence_violation_{pred}_{succ}_{station_pred}_{station_succ}"
                        )

    # Capacity constraints: total time at each station ≤ cycle_time
    for station in range(num_stations):
        relevant_tasks = [(t, s) for (t, s) in possible_assignments if s == station]
        if relevant_tasks:
            total_load_expr = gp.quicksum(
                task_processing_times.get(str(task), 0) * x[task, station]
                for (task, st) in relevant_tasks
            )
            m.addConstr(total_load_expr <= cycle_time, f"capacity_station_{station}")

    # Define idle time: idle ≥ (cycle_time - load_of_each_station) for every station
    # This captures the maximum idle time among all stations
    for station in range(num_stations):
        relevant_tasks = [(t, s) for (t, s) in possible_assignments if s == station]
        if relevant_tasks:
            total_load = gp.quicksum(
                task_processing_times.get(str(task), 0) * x[task, station]
                for (task, st) in relevant_tasks
            )
            # idle >= cycle_time - total_load
            m.addConstr(idle >= cycle_time - total_load, f"idle_def_{station}")

    # Set objective to minimize the maximum idle time
    m.setObjective(idle, GRB.MINIMIZE)

    # Optimize
    m.optimize()

    # Handle solution extraction
    status = m.Status
    if status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and m.SolCount > 0:
        # Get the assignments
        task_assignments = {}
        for i in range(num_tasks):
            for j in range(earliest_station.get(str(i), 0), min(latest_station.get(str(i), num_stations - 1) + 1, num_stations)):
                if x[i, j].x > 0.5:
                    task_assignments[str(i)] = j
                    break
        
        return {
            "objective_value": m.objVal,
            "task_assignments": task_assignments
        }
    
    return {"objective_value": float('inf'), "task_assignments": {}}


def main():
    parser = argparse.ArgumentParser(description='Solve Assembly Line Balancing Problem')
    parser.add_argument('--instance_path', type=str, required=True, 
                        help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, 
                        help='Path where the solution JSON file must be written')
    parser.add_argument('--time_limit', type=int, default=7200, 
                        help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, help='Path for logging intermediate solutions')
    args = parser.parse_args()
    
    # Initialize logger for intermediate solutions
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")  # Since we want to minimize idle time
    else:
        logger = None
    
    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Solve the problem
    solution = solve_assembly_line_balance(instance_data, args.time_limit)
    
    # Log the solution if we got one
    if logger and solution["objective_value"] != float('inf'):
        logger.log(solution["objective_value"])
    
    # Save solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()