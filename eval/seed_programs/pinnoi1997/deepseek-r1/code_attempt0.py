import argparse
import json
from collections import defaultdict
from gurobipy import GRB, Model, quicksum
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    # Read instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    tasks = list(data['task_processing_times'].keys())
    cycle_time = data['cycle_time']
    num_stations = data['num_stations']
    precedence_arcs = data['precedence_arcs']
    earliest_station = data['earliest_station']
    latest_station = data['latest_station']
    task_processing = data['task_processing_times']

    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model = Model("AssemblyLine")
    model.setParam('OutputFlag', 0)

    # Create variables x[t, s]
    x = {}
    for t in tasks:
        es = earliest_station[t]
        ls = latest_station[t]
        for s in range(es, ls + 1):
            x[t, s] = model.addVar(vtype=GRB.BINARY, name=f"x_{t}_{s}")

    # Assignment constraints
    for t in tasks:
        es = earliest_station[t]
        ls = latest_station[t]
        model.addConstr(quicksum(x[t, s] for s in range(es, ls + 1)) == 1, f"assign_{t}")

    # Precedence constraints
    for pred, succ in precedence_arcs:
        succ_es = earliest_station[succ]
        succ_ls = latest_station[succ]
        for s in range(succ_es, succ_ls + 1):
            pred_es = earliest_station[pred]
            pred_ls = latest_station[pred]
            s_upper = min(pred_ls, s)
            if pred_es > s_upper:
                model.addConstr(x[succ, s] == 0, f"prec_{pred}_{succ}_s{s}")
            else:
                model.addConstr(x[succ, s] <= quicksum(x[pred, sp] for sp in range(pred_es, s_upper + 1)), f"prec_{pred}_{succ}_s{s}")

    # Objective and station constraints
    z = model.addVar(vtype=GRB.CONTINUOUS, name="z")
    model.setObjective(z, GRB.MINIMIZE)

    for s in range(1, num_stations + 1):
        pt_sum = quicksum(task_processing[t] * x[t, s] for t in tasks if (t, s) in x)
        model.addConstr(pt_sum <= cycle_time, f"cycle_s{s}")
        model.addConstr(z >= cycle_time - pt_sum, f"idle_s{s}")

    model.setParam('TimeLimit', args.time_limit)

    # Solve with logging support
    def callback(model_, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            z_val = model_.cbGet(GRB.Callback.MIPSOL_OBJ)
            logger.log(z_val)

    model.optimize(callback)

    # Extract solution
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and model.SolCount > 0:
        task_assignments = {}
        for t in tasks:
            for s in range(earliest_station[t], latest_station[t] + 1):
                if x[t, s].X > 0.5:
                    task_assignments[t] = s
                    break

        sum_pt = [0] * num_stations
        for t, s in task_assignments.items():
            sum_pt[s - 1] += task_processing[t]
        max_idle = max(cycle_time - pt for pt in sum_pt)
    else:
        task_assignments = {t: earliest_station[t] for t in tasks}
        max_idle = cycle_time * num_stations - data['total_processing_time']

    solution = {
        "objective_value": float(max_idle),
        "task_assignments": task_assignments
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()