import argparse
import json
import math
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    num_tasks = data['num_tasks']
    cycle_time = data['cycle_time']
    num_stations = data['num_stations']
    total_processing_time = data['total_processing_time']
    task_times = data['task_processing_times']
    arcs = data['precedence_arcs']
    earliest = data['earliest_station']
    latest = data['latest_station']

    # Task names as strings, sorted numerically
    task_names = sorted(task_times.keys(), key=lambda x: int(x))

    import gurobipy as gp
    from gurobipy import GRB

    env = gp.Env(empty=True)
    env.setParam('Seed', 0)
    env.setParam('Threads', 1)
    env.start()

    model = gp.Model(env=env)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('TimeLimit', float(args.time_limit))
    model.setParam('OutputFlag', 0)

    # Decision variables: x[t, k] = 1 if task t assigned to station k
    x = {}
    for t in task_names:
        e = earliest[t]
        l = latest[t]
        for k in range(e, l + 1):
            x[t, k] = model.addVar(vtype=GRB.BINARY, name=f'x_{t}_{k}')

    # Objective variable: max idle time
    total_slack = num_stations * cycle_time - total_processing_time
    lb1 = int(total_slack // cycle_time)
    lb2 = math.ceil(total_slack / num_stations) if num_stations > 0 else 0
    z_lb = max(0, lb1, lb2)

    z = model.addVar(vtype=GRB.INTEGER, lb=z_lb, ub=cycle_time, name='z')
    model.update()

    # Constraint: each task assigned to exactly one station
    for t in task_names:
        e = earliest[t]
        l = latest[t]
        model.addConstr(gp.quicksum(x[t, k] for k in range(e, l + 1)) == 1)

    # Constraint: precedence — station(pred) <= station(succ)
    for arc in arcs:
        i, j = int(arc[0]), int(arc[1])
        ti, tj = str(i), str(j)
        ei, li = earliest[ti], latest[ti]
        ej, lj = earliest[tj], latest[tj]
        model.addConstr(
            gp.quicksum(k * x[ti, k] for k in range(ei, li + 1)) <=
            gp.quicksum(k * x[tj, k] for k in range(ej, lj + 1))
        )

    # Constraint: station capacity and max idle time
    for k in range(1, num_stations + 1):
        load = gp.LinExpr()
        for t in task_names:
            e = earliest[t]
            l = latest[t]
            if e <= k <= l:
                load += task_times[t] * x[t, k]
        model.addConstr(load <= cycle_time)
        model.addConstr(load + z >= cycle_time)

    model.setObjective(z, GRB.MINIMIZE)

    # Callback to log incumbents
    best_obj = [float('inf')]
    best_sol = [None]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_obj[0]:
                best_obj[0] = obj
                sol = {}
                for t in task_names:
                    e = earliest[t]
                    l = latest[t]
                    for k in range(e, l + 1):
                        if model.cbGet(x[t, k]) > 0.5:
                            sol[t] = k
                            break
                best_sol[0] = sol
                if logger:
                    logger.log_solution(obj, sol)

    model.optimize(callback=callback)

    # Retrieve final solution
    if model.SolCount > 0:
        obj_val = model.ObjVal
        if best_sol[0] is not None and best_obj[0] <= obj_val + 1e-6:
            sol = best_sol[0]
        else:
            sol = {}
            for t in task_names:
                e = earliest[t]
                l = latest[t]
                for k in range(e, l + 1):
                    if x[t, k].X > 0.5:
                        sol[t] = k
                        break
            if logger and best_obj[0] == float('inf'):
                logger.log_solution(obj_val, sol)
    else:
        # Fallback: trivial assignment (may not satisfy all constraints)
        obj_val = float(cycle_time)
        sol = {t: earliest[t] for t in task_names}

    output = {
        "objective_value": float(obj_val),
        "task_assignments": sol
    }

    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == '__main__':
    main()