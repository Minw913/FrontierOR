import json
import argparse
import time
from typing import Dict, List, Any
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_tasks = instance['num_tasks']
    cycle_time = instance['cycle_time']
    num_stations = instance['num_stations']
    task_processing_times = instance['task_processing_times']
    precedence_arcs = instance['precedence_arcs']
    earliest_station = instance['earliest_station']
    latest_station = instance['latest_station']
    total_processing_time = instance['total_processing_time']

    tasks = list(task_processing_times.keys())
    stations = list(range(1, num_stations + 1))

    # Compute lower bound on max idle time
    total_capacity = num_stations * cycle_time
    total_slack = total_capacity - total_processing_time
    lower_bound = total_slack // num_stations

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model = gp.Model("Assembly_Line_Balancing")
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1
    model.Params.LogToConsole = 0

    # x[task, station] = 1 if task assigned to station
    x = {}
    for task in tasks:
        e = int(earliest_station[task])
        l = int(latest_station[task])
        for s in range(e, l + 1):
            x[task, s] = model.addVar(vtype=GRB.BINARY, name=f"x_{task}_{s}")

    # y[s] = idle time at station s (continuous)
    y = {}
    for s in stations:
        y[s] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"y_{s}")

    # z = max idle time (objective)
    z = model.addVar(lb=lower_bound, vtype=GRB.CONTINUOUS, name="z")

    # Each task assigned to exactly one station
    for task in tasks:
        e = int(earliest_station[task])
        l = int(latest_station[task])
        model.addConstr(gp.quicksum(x[task, s] for s in range(e, l + 1)) == 1, name=f"assign_{task}")

    # Station workload + idle >= cycle_time, and z >= y[s] for all s
    for s in stations:
        load = gp.quicksum(int(task_processing_times[t]) * x[t, s] for t in tasks if (t, s) in x)
        model.addConstr(load + y[s] == cycle_time, name=f"workload_{s}")
        model.addConstr(y[s] <= z, name=f"maxidle_{s}")

    # Precedence constraints
    for pred, succ in precedence_arcs:
        pred = str(pred)
        succ = str(succ)
        e_pred = int(earliest_station[pred])
        l_pred = int(latest_station[pred])
        e_succ = int(earliest_station[succ])
        l_succ = int(latest_station[succ])
        possible_s_pred = list(range(max(e_pred, 1), min(l_pred, num_stations) + 1))
        possible_s_succ = list(range(max(e_succ, 1), min(l_succ, num_stations) + 1))
        
        # For every possible pair where s_pred > s_succ, forbid it
        for sp in possible_s_pred:
            for ss in possible_s_succ:
                if sp > ss:
                    if (pred, sp) in x and (succ, ss) in x:
                        model.addConstr(x[pred, sp] + x[succ, ss] <= 1, 
                                      name=f"prec_{pred}_{succ}_{sp}_{ss}")

    # Objective: minimize z (max idle time)
    model.setObjective(z, GRB.MINIMIZE)

    # Callback to log incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(float(obj_val))

    start_time = time.time()
    model.optimize(callback)

    # Extract solution
    task_assignments = {}
    if model.SolCount > 0:
        for task in tasks:
            e = int(earliest_station[task])
            l = int(latest_station[task])
            for s in range(e, l + 1):
                if (task, s) in x and abs(x[task, s].X) > 0.5:
                    task_assignments[task] = int(s)
                    break
        objective_value = float(z.X)
    else:
        # Fallback: assign each task to its earliest station (may be infeasible)
        objective_value = 0.0
        for task in tasks:
            task_assignments[task] = int(earliest_station[task])

    # Write solution
    solution = {
        "objective_value": objective_value,
        "task_assignments": task_assignments
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    elapsed = time.time() - start_time
    if elapsed < args.time_limit * 0.95 and model.SolCount > 0 and model.Status == GRB.OPTIMAL:
        if logger:
            logger.log(objective_value)

if __name__ == "__main__":
    main()