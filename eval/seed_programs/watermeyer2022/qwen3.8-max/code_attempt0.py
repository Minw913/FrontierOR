import json
import argparse
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from gurobipy import Model, GRB, quicksum, LinExpr
    from solution_logger import SolutionLogger

    start_time = time.time()

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_real = instance['num_real_activities']
    num_total = instance['num_activities_total']
    num_resources = instance['num_resources']
    max_duration = instance['max_project_duration']

    activities = instance['activities']
    temporal_constraints = instance['temporal_constraints']
    resources = instance['resources']

    # Processing times
    proc_time = {}
    for act in activities:
        proc_time[act['id']] = act['processing_time']

    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Compute bounds on start times using Bellman-Ford style propagation
    # Earliest start times (forward pass - longest path from start)
    earliest = [0] * num_total
    for _ in range(num_total + 1):
        changed = False
        for tc in temporal_constraints:
            i, j, lag = tc['from_activity'], tc['to_activity'], tc['time_lag']
            if earliest[i] + lag > earliest[j]:
                earliest[j] = earliest[i] + lag
                changed = True
        if not changed:
            break

    # Latest start times (backward pass)
    latest = [max_duration] * num_total
    for _ in range(num_total + 1):
        changed = False
        for tc in temporal_constraints:
            i, j, lag = tc['from_activity'], tc['to_activity'], tc['time_lag']
            if latest[j] - lag < latest[i]:
                latest[i] = latest[j] - lag
                changed = True
        if not changed:
            break

    # Fix activity 0 (project start) at time 0
    earliest[0] = 0
    latest[0] = 0

    # Clamp bounds to valid range
    for i in range(num_total):
        earliest[i] = max(0, earliest[i])
        latest[i] = min(max_duration, latest[i])
        if earliest[i] > latest[i]:
            earliest[i] = latest[i]

    # Create Gurobi model
    model = Model("project_scheduling")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.LogToConsole = 0

    elapsed = time.time() - start_time
    remaining_time = max(1.0, float(args.time_limit) - elapsed - 1.0)
    model.Params.TimeLimit = remaining_time

    # Decision variables: x[i,t] = 1 if activity i starts at time t
    x = {}
    possible_starts = {}
    for i in range(num_total):
        possible_starts[i] = list(range(earliest[i], latest[i] + 1))
        for t in possible_starts[i]:
            x[i, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{t}")

    model.update()

    # Each activity starts exactly once
    for i in range(num_total):
        model.addConstr(quicksum(x[i, t] for t in possible_starts[i]) == 1)

    # Objective: minimize start time of project-end activity
    end_activity = num_real + 1
    model.setObjective(
        quicksum(t * x[end_activity, t] for t in possible_starts[end_activity]),
        GRB.MINIMIZE
    )

    # Temporal constraints: s_j - s_i >= lag
    for tc in temporal_constraints:
        i = tc['from_activity']
        j = tc['to_activity']
        lag = tc['time_lag']
        model.addConstr(
            quicksum(t * x[j, t] for t in possible_starts[j]) -
            quicksum(t * x[i, t] for t in possible_starts[i]) >= lag
        )

    # Resource constraints using prefix sum for efficient counting
    for res in resources:
        Pi_k_list = res['Pi_k']
        capacity = res['capacity']
        demands = res['demands']

        # Build prefix sum array for Pi_k
        # prefix[t] = number of periods in Pi_k that are <= t
        prefix = [0] * (max_duration + 2)
        for p in Pi_k_list:
            if 1 <= p <= max_duration:
                prefix[p] += 1
        for t in range(1, max_duration + 2):
            prefix[t] += prefix[t - 1]

        resource_expr = LinExpr()
        for i in range(num_total):
            demand_ik = demands.get(str(i), 0)
            if demand_ik == 0:
                continue
            p_i = proc_time[i]
            if p_i == 0:
                continue
            for t in possible_starts[i]:
                # Count periods in Pi_k ∩ {t+1, ..., t+p_i}
                upper = min(t + p_i, max_duration)
                count = prefix[upper] - prefix[t]
                if count > 0:
                    resource_expr += demand_ik * count * x[i, t]

        model.addConstr(resource_expr <= capacity)

    model.update()

    # Callback to log incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            schedule = {}
            for i in range(num_total):
                for t in possible_starts[i]:
                    if model.cbGetSolution(x[i, t]) > 0.5:
                        schedule[str(i)] = t
                        break
            solution_dict = {
                "objective_value": float(obj),
                "schedule": schedule
            }
            if logger:
                logger.log_solution(float(obj), solution_dict)

    model.optimize(callback)

    # Extract final solution
    if model.SolCount > 0:
        schedule = {}
        for i in range(num_total):
            for t in possible_starts[i]:
                if x[i, t].X > 0.5:
                    schedule[str(i)] = t
                    break

        objective_value = float(schedule[str(end_activity)])
        solution = {
            "objective_value": objective_value,
            "schedule": schedule
        }
    else:
        # Fallback: use earliest start times (may not satisfy resource constraints)
        schedule = {str(i): int(earliest[i]) for i in range(num_total)}
        objective_value = float(earliest[end_activity])
        solution = {
            "objective_value": objective_value,
            "schedule": schedule
        }

    # Log final solution
    if logger:
        logger.log_solution(objective_value, solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()