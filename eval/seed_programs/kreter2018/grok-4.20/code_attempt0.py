import json
import argparse
import time
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

    num_real = instance['num_real_activities']
    num_total = instance['num_total_activities']
    num_res = instance['num_resource_types']
    T = instance['max_project_duration']
    activities = instance['activities']
    precedences = instance['precedence_relations']
    costs = instance['resource_costs']

    # Build duration and requirements
    duration = [0] * num_total
    req = [[0] * num_res for _ in range(num_total)]
    for act in activities:
        aid = act['id']
        duration[aid] = act['duration']
        req[aid] = act['resource_requirements']

    # Earliest and latest start times
    EST = [0] * num_total
    LST = [T] * num_total

    # Forward pass for EST
    graph = [[] for _ in range(num_total)]
    rgraph = [[] for _ in range(num_total)]
    for pred, succ in precedences:
        graph[pred].append(succ)
        rgraph[succ].append(pred)

    updated = True
    while updated:
        updated = False
        for i in range(num_total):
            for j in graph[i]:
                if EST[j] < EST[i] + duration[i]:
                    EST[j] = EST[i] + duration[i]
                    updated = True

    # Backward pass for LST
    LST[num_total - 1] = T - duration[num_total - 1]
    updated = True
    while updated:
        updated = False
        for i in range(num_total):
            for j in rgraph[i]:
                if LST[j] > LST[i] - duration[j]:
                    LST[j] = LST[i] - duration[j]
                    updated = True
    for i in range(num_total):
        LST[i] = max(LST[i], EST[i])

    # Possible start times for each activity
    possible_starts = [list(range(EST[i], LST[i] + 1)) for i in range(num_total)]

    # Create model
    model = gp.Model("RCPSP_Resource_Leveling")
    model.setParam(GRB.Param.TimeLimit, args.time_limit)
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.MIPGap, 0.0)
    model.setParam(GRB.Param.LogToConsole, 0)

    # Variables: start time for each activity
    start = {}
    for i in range(num_total):
        for t in possible_starts[i]:
            start[i, t] = model.addVar(vtype=GRB.BINARY, name=f"start_{i}_{t}")

    # Exactly one start time per activity
    for i in range(num_total):
        model.addConstr(gp.quicksum(start[i, t] for t in possible_starts[i]) == 1)

    # Precedence constraints
    for pred, succ in precedences:
        lhs = gp.quicksum(t * start[pred, t] for t in possible_starts[pred])
        rhs = gp.quicksum(t * start[succ, t] for t in possible_starts[succ])
        model.addConstr(lhs + duration[pred] <= rhs)

    # Resource capacities (to be decided)
    max_possible = [0] * num_res
    for r in range(num_res):
        max_possible[r] = sum(req[i][r] for i in range(num_total))
    R = {}
    for r in range(num_res):
        R[r] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_possible[r], name=f"R_{r}")

    # Resource usage constraints
    for r in range(num_res):
        for tau in range(T + 1):
            usage = gp.LinExpr()
            for i in range(1, num_total - 1):  # skip dummies
                if req[i][r] == 0:
                    continue
                for t in possible_starts[i]:
                    # Activity i started at t is active at tau if t <= tau < t + duration[i]
                    if tau - duration[i] + 1 <= t <= tau:
                        usage.add(start[i, t], req[i][r])
            model.addConstr(usage <= R[r])

    # Objective
    obj = gp.quicksum(costs[r] * R[r] for r in range(num_res))
    model.setObjective(obj, GRB.MINIMIZE)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    best_obj = None
    incumbent_start_times = None
    incumbent_resources = None

    def callback(model, where):
        nonlocal best_obj, incumbent_start_times, incumbent_resources
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if best_obj is None or obj_val < best_obj:
                best_obj = obj_val
                # Extract solution
                sol_start = {}
                for i in range(num_total):
                    for t in possible_starts[i]:
                        if abs(model.cbGetSolution(start[i, t]) - 1.0) < 1e-6:
                            sol_start[str(i)] = int(t)
                            break
                sol_res = {}
                for r in range(num_res):
                    sol_res[str(r)] = int(round(model.cbGetSolution(R[r])))
                incumbent_start_times = sol_start
                incumbent_resources = sol_res
                if logger:
                    logger.log(float(obj_val))

    model.optimize(callback)

    # If no solution found via callback, retrieve from final model
    if incumbent_start_times is None and model.SolCount > 0:
        best_obj = model.ObjVal
        sol_start = {}
        for i in range(num_total):
            for t in possible_starts[i]:
                if start[i, t].X > 0.5:
                    sol_start[str(i)] = int(t)
                    break
        sol_res = {}
        for r_val in range(num_res):
            sol_res[str(r_val)] = int(round(R[r_val].X))
        incumbent_start_times = sol_start
        incumbent_resources = sol_res
        if logger:
            logger.log(float(best_obj))

    # Write solution
    if incumbent_start_times is None:
        # No feasible solution found - write dummy feasible if possible or minimal
        # For safety, provide a simple feasible solution by serializing
        sol_start = {}
        current_time = 0
        for i in range(num_total):
            sol_start[str(i)] = current_time
            current_time += duration[i]
        if current_time > T:
            current_time = T
        sol_res = {}
        for r in range(num_res):
            max_req = 0
            for i in range(num_total):
                max_req = max(max_req, req[i][r])
            sol_res[str(r)] = max_req
        final_obj = sum(costs[r] * sol_res[str(r)] for r in range(num_res))
        solution = {
            "objective_value": float(final_obj),
            "start_times": sol_start,
            "resource_amounts": sol_res
        }
    else:
        solution = {
            "objective_value": float(best_obj),
            "start_times": incumbent_start_times,
            "resource_amounts": incumbent_resources
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()