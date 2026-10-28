import argparse
import json
from gurobipy import GRB, Model, quicksum
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    tasks = list(data['task_processing_times'].keys())
    cycle_time = data['cycle_time']
    num_stations = data['num_stations']
    precedence_arcs = [[str(arc[0]), str(arc[1])] for arc in data['precedence_arcs']]
    earliest_station = data['earliest_station']
    latest_station = data['latest_station']
    task_processing = data['task_processing_times']

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model = Model("AssemblyLine")
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', 1)

    x = {}
    for t in tasks:
        es = earliest_station[t]
        ls = latest_station[t]
        for s in range(es, ls + 1):
            x[t, s] = model.addVar(vtype=GRB.BINARY, name=f"x_{t}_{s}")

    for t in tasks:
        es = earliest_station[t]
        ls = latest_station[t]
        model.addConstr(quicksum(x[t, s] for s in range(es, ls + 1)) == 1, f"assign_{t}")

    for pred, succ in precedence_arcs:
        for s in range(earliest_station[succ], latest_station[succ] + 1):
            s_upper = min(latest_station[pred], s)
            if earliest_station[pred] > s_upper:
                model.addConstr(x[succ, s] == 0, f"prec_{pred}_{succ}_s{s}")
            else:
                model.addConstr(
                    x[succ, s] <= quicksum(x[pred, sp] for sp in range(earliest_station[pred], s_upper + 1)),
                    f"prec_{pred}_{succ}_s{s}"
                )

    z = model.addVar(vtype=GRB.CONTINUOUS, name="z")
    model.setObjective(z, GRB.MINIMIZE)

    for s in range(1, num_stations + 1):
        station_tasks = [t for t in tasks if (t, s) in x]
        if station_tasks:
            pt_sum = quicksum(task_processing[t] * x[t, s] for t in station_tasks)
            model.addConstr(pt_sum <= cycle_time, f"cycle_s{s}")
            model.addConstr(z >= cycle_time - pt_sum, f"idle_s{s}")

    model.setParam('TimeLimit', args.time_limit)

    def callback(model_, where):
        if where == GRB.Callback.MIPSOL:
            if logger is not None:
                z_val = model_.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(z_val)

    model.optimize(callback)

    if model.status == GRB.OPTIMAL or (model.status == GRB.TIME_LIMIT and model.SolCount > 0):
        task_assignments = {}
        for t in tasks:
            for s in range(earliest_station[t], latest_station[t] + 1):
                if (t, s) in x and x[t, s].X > 0.5:
                    task_assignments[t] = s
                    break
        station_times = [0] * num_stations
        for t, s in task_assignments.items():
            station_times[s-1] += task_processing[t]
        max_idle = max(cycle_time - pt for pt in station_times)
    else:
        task_assignments = {t: earliest_station[t] for t in tasks}
        max_idle = cycle_time

    solution = {
        "objective_value": float(max_idle),
        "task_assignments": task_assignments
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()