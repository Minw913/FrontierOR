import json
import argparse
from collections import deque
from gurobipy import *

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_tasks = instance['num_tasks']
    cycle_time = instance['cycle_time']
    num_stations = instance['num_stations']
    total_processing_time = instance.get('total_processing_time', 0)
    task_processing_times = instance['task_processing_times']
    precedence_arcs = instance['precedence_arcs']
    earliest_station = instance['earliest_station']
    latest_station = instance['latest_station']

    # Determine task keys from the data
    task_keys = sorted(task_processing_times.keys(), key=lambda x: int(x))
    tasks = [int(k) for k in task_keys]

    # Helper to get value from dict with flexible key types
    def get_val(d, task):
        if str(task) in d:
            return d[str(task)]
        elif task in d:
            return d[task]
        else:
            raise KeyError(f"Task {task} not found in dict with keys: {list(d.keys())[:5]}")

    # Determine station range from data
    min_station = min(get_val(earliest_station, i) for i in tasks)
    max_station = max(get_val(latest_station, i) for i in tasks)
    stations = list(range(min_station, max_station + 1))

    # Build predecessor/successor lists
    predecessors = {i: [] for i in tasks}
    successors = {i: [] for i in tasks}
    for arc in precedence_arcs:
        i, j = int(arc[0]), int(arc[1])
        predecessors[j].append(i)
        successors[i].append(j)

    # Greedy heuristic for warm start
    def greedy_heuristic():
        in_degree = {i: len(predecessors[i]) for i in tasks}
        queue = deque([i for i in tasks if in_degree[i] == 0])
        topo_order = []
        temp_in_degree = dict(in_degree)
        while queue:
            node = queue.popleft()
            topo_order.append(node)
            for succ in successors[node]:
                temp_in_degree[succ] -= 1
                if temp_in_degree[succ] == 0:
                    queue.append(succ)

        assignment = {}
        station_load = {k: 0 for k in stations}

        for task in topo_order:
            es = get_val(earliest_station, task)
            ls = get_val(latest_station, task)
            pt = get_val(task_processing_times, task)

            min_st = es
            for pred in predecessors[task]:
                if pred in assignment:
                    min_st = max(min_st, assignment[pred])

            best_station = None
            for k in range(min_st, ls + 1):
                if station_load.get(k, 0) + pt <= cycle_time:
                    best_station = k
                    break

            if best_station is not None:
                assignment[task] = best_station
                station_load[best_station] = station_load.get(best_station, 0) + pt

        if len(assignment) == num_tasks:
            return assignment
        return None

    warm_start = greedy_heuristic()

    # Build MIP model
    model = Model("ALBP")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = args.time_limit
    model.Params.OutputFlag = 0

    # Decision variables
    x = {}
    for i in tasks:
        es = get_val(earliest_station, i)
        ls = get_val(latest_station, i)
        for k in range(es, ls + 1):
            x[i, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{k}")

    z = model.addVar(vtype=GRB.INTEGER, lb=0, name="z")

    model.update()

    # Objective: minimize maximum idle time
    model.setObjective(z, GRB.MINIMIZE)

    # Each task assigned to exactly one station
    for i in tasks:
        es = get_val(earliest_station, i)
        ls = get_val(latest_station, i)
        model.addConstr(quicksum(x[i, k] for k in range(es, ls + 1)) == 1)

    # Precedence constraints
    for arc in precedence_arcs:
        i, j = int(arc[0]), int(arc[1])
        es_i = get_val(earliest_station, i)
        ls_i = get_val(latest_station, i)
        es_j = get_val(earliest_station, j)
        ls_j = get_val(latest_station, j)
        model.addConstr(
            quicksum(k * x[i, k] for k in range(es_i, ls_i + 1)) <=
            quicksum(k * x[j, k] for k in range(es_j, ls_j + 1))
        )

    # Cycle time and idle time constraints for each station
    for k in stations:
        tasks_at_k = [i for i in tasks if (i, k) in x]
        if tasks_at_k:
            load_expr = quicksum(get_val(task_processing_times, i) * x[i, k] for i in tasks_at_k)
            model.addConstr(load_expr <= cycle_time)
            model.addConstr(z >= cycle_time - load_expr)
        else:
            model.addConstr(z >= cycle_time)

    # Lower bound on z
    if total_processing_time > 0:
        total_slack = num_stations * cycle_time - total_processing_time
        lb_z = total_slack // cycle_time
        if lb_z > 0:
            model.addConstr(z >= lb_z)

    # Set warm start if available
    if warm_start:
        for i in tasks:
            if i in warm_start:
                k_assigned = warm_start[i]
                es = get_val(earliest_station, i)
                ls = get_val(latest_station, i)
                for k in range(es, ls + 1):
                    if (i, k) in x:
                        x[i, k].Start = 1.0 if k == k_assigned else 0.0

        station_loads = {k: 0 for k in stations}
        for i, k in warm_start.items():
            station_loads[k] = station_loads.get(k, 0) + get_val(task_processing_times, i)
        max_idle = max(cycle_time - station_loads.get(k, 0) for k in stations)
        z.Start = max_idle

    # Callback for logging incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            task_assignments = {}
            for i in tasks:
                es = get_val(earliest_station, i)
                ls = get_val(latest_station, i)
                for k in range(es, ls + 1):
                    if model.cbGetSolution(x[i, k]) > 0.5:
                        task_assignments[str(i)] = int(k)
                        break

            solution_dict = {
                "objective_value": float(obj_val),
                "task_assignments": task_assignments
            }
            if logger:
                logger.log_solution(float(obj_val), solution_dict)

    model.optimize(callback)

    # Extract final solution
    if model.SolCount > 0:
        task_assignments = {}
        for i in tasks:
            es = get_val(earliest_station, i)
            ls = get_val(latest_station, i)
            for k in range(es, ls + 1):
                if x[i, k].X > 0.5:
                    task_assignments[str(i)] = int(k)
                    break

        objective_value = float(z.X)
    else:
        # Fallback
        task_assignments = {str(i): int(get_val(earliest_station, i)) for i in tasks}
        objective_value = float(cycle_time)

    solution = {
        "objective_value": objective_value,
        "task_assignments": task_assignments
    }

    if logger:
        logger.log_solution(objective_value, solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    solve()