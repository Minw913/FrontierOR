import json
import argparse
import sys
from collections import deque


def compute_earliest_start(activity_ids, durations, successors, predecessors):
    """Forward pass to compute earliest start times."""
    earliest = {aid: 0 for aid in activity_ids}
    in_degree = {aid: len(predecessors[aid]) for aid in activity_ids}
    queue = deque([aid for aid in activity_ids if in_degree[aid] == 0])

    while queue:
        node = queue.popleft()
        for succ in successors[node]:
            earliest[succ] = max(earliest[succ], earliest[node] + durations[node])
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    return earliest


def compute_latest_start(activity_ids, durations, successors, predecessors, deadline):
    """Backward pass to compute latest start times."""
    latest = {aid: deadline for aid in activity_ids}
    out_degree = {aid: len(successors[aid]) for aid in activity_ids}
    queue = deque([aid for aid in activity_ids if out_degree[aid] == 0])

    while queue:
        node = queue.popleft()
        for pred in predecessors[node]:
            latest[pred] = min(latest[pred], latest[node] - durations[pred])
            out_degree[pred] -= 1
            if out_degree[pred] == 0:
                queue.append(pred)

    return latest


def greedy_schedule(activity_ids, durations, requirements, successors, predecessors,
                    earliest, latest, num_resources, max_duration):
    """Serial schedule generation: schedule activities as early as possible respecting precedence."""
    in_degree = {aid: len(predecessors[aid]) for aid in activity_ids}
    queue = deque([aid for aid in activity_ids if in_degree[aid] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    start_times = {}
    for aid in topo_order:
        start_times[aid] = earliest[aid]

    # Compute peak resource usage
    resource_amounts = [0] * num_resources
    for t in range(max_duration):
        for r in range(num_resources):
            demand = 0
            for aid in activity_ids:
                if durations[aid] > 0:
                    s = start_times[aid]
                    if s <= t < s + durations[aid]:
                        demand += requirements[aid][r]
            resource_amounts[r] = max(resource_amounts[r], demand)

    return start_times, resource_amounts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        instance = json.load(f)

    num_real = instance["num_real_activities"]
    num_total = instance["num_total_activities"]
    num_resources = instance["num_resource_types"]
    max_duration = instance["max_project_duration"]
    activities = instance["activities"]
    precedences = instance["precedence_relations"]
    resource_costs = instance["resource_costs"]

    activity_ids = [act["id"] for act in activities]
    durations = {act["id"]: act["duration"] for act in activities}
    requirements = {act["id"]: act["resource_requirements"] for act in activities}

    successors = {aid: [] for aid in activity_ids}
    predecessors = {aid: [] for aid in activity_ids}
    for pred, succ in precedences:
        successors[pred].append(succ)
        predecessors[succ].append(pred)

    earliest = compute_earliest_start(activity_ids, durations, successors, predecessors)
    latest = compute_latest_start(activity_ids, durations, successors, predecessors, max_duration)

    # Greedy solution as initial feasible solution
    greedy_starts, greedy_resources = greedy_schedule(
        activity_ids, durations, requirements, successors, predecessors,
        earliest, latest, num_resources, max_duration
    )

    greedy_obj = sum(resource_costs[r] * greedy_resources[r] for r in range(num_resources))

    best_solution = {
        "objective_value": greedy_obj,
        "start_times": {str(aid): greedy_starts[aid] for aid in activity_ids},
        "resource_amounts": {str(r): greedy_resources[r] for r in range(num_resources)}
    }

    if logger:
        logger.log_solution(greedy_obj, best_solution)

    # MIP formulation using Gurobi
    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model("RCPSP_Resource_Cost")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", args.time_limit)
    model.setParam("OutputFlag", 0)

    # Decision variables: x[aid, t] = 1 if activity aid starts at time t
    x = {}
    for aid in activity_ids:
        es = earliest[aid]
        ls = latest[aid]
        if es > ls:
            # Infeasible window - shouldn't happen for valid instances
            es = ls = earliest[aid]
        for t in range(es, ls + 1):
            x[aid, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{aid}_{t}")

    # Resource provision variables
    R = {}
    for r in range(num_resources):
        ub = sum(requirements[aid][r] for aid in activity_ids if durations[aid] > 0)
        R[r] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=ub, name=f"R_{r}")

    model.update()

    # Objective: minimize total resource provision cost
    model.setObjective(gp.quicksum(resource_costs[r] * R[r] for r in range(num_resources)), GRB.MINIMIZE)

    # Constraint: Each activity starts exactly once
    for aid in activity_ids:
        es = earliest[aid]
        ls = latest[aid]
        if es > ls:
            es = ls = earliest[aid]
        model.addConstr(gp.quicksum(x[aid, t] for t in range(es, ls + 1)) == 1)

    # Precedence constraints
    for pred, succ in precedences:
        es_p, ls_p = earliest[pred], latest[pred]
        es_s, ls_s = earliest[succ], latest[succ]
        if es_p > ls_p:
            es_p = ls_p = earliest[pred]
        if es_s > ls_s:
            es_s = ls_s = earliest[succ]
        S_pred = gp.quicksum(t * x[pred, t] for t in range(es_p, ls_p + 1))
        S_succ = gp.quicksum(t * x[succ, t] for t in range(es_s, ls_s + 1))
        model.addConstr(S_succ >= S_pred + durations[pred])

    # Resource constraints: for each time period and resource type
    for t in range(max_duration):
        for r in range(num_resources):
            terms = []
            for aid in activity_ids:
                if durations[aid] == 0 or requirements[aid][r] == 0:
                    continue
                es = earliest[aid]
                ls = latest[aid]
                if es > ls:
                    es = ls = earliest[aid]
                start_low = max(es, t - durations[aid] + 1)
                start_high = min(ls, t)
                if start_low <= start_high:
                    for t_prime in range(start_low, start_high + 1):
                        terms.append(requirements[aid][r] * x[aid, t_prime])
            if terms:
                model.addConstr(gp.quicksum(terms) <= R[r])

    # Set initial solution from greedy
    for aid in activity_ids:
        es = earliest[aid]
        ls = latest[aid]
        if es > ls:
            es = ls = earliest[aid]
        for t in range(es, ls + 1):
            x[aid, t].Start = 1.0 if t == greedy_starts[aid] else 0.0
    for r in range(num_resources):
        R[r].Start = float(greedy_resources[r])

    # Callback for logging intermediate solutions
    best_obj_cb = [greedy_obj]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_obj_cb[0] - 1e-6:
                best_obj_cb[0] = obj
                start_times_sol = {}
                for aid in activity_ids:
                    es = earliest[aid]
                    ls = latest[aid]
                    if es > ls:
                        es = ls = earliest[aid]
                    for t in range(es, ls + 1):
                        val = model.cbGetSolution(x[aid, t])
                        if val > 0.5:
                            start_times_sol[str(aid)] = t
                            break

                resource_amounts_sol = {}
                for r in range(num_resources):
                    resource_amounts_sol[str(r)] = int(round(model.cbGetSolution(R[r])))

                solution = {
                    "objective_value": obj,
                    "start_times": start_times_sol,
                    "resource_amounts": resource_amounts_sol
                }

                if logger:
                    logger.log_solution(obj, solution)

    model.optimize(callback)

    # Extract final solution
    if model.SolCount > 0:
        start_times_sol = {}
        for aid in activity_ids:
            es = earliest[aid]
            ls = latest[aid]
            if es > ls:
                es = ls = earliest[aid]
            found = False
            for t in range(es, ls + 1):
                if x[aid, t].X > 0.5:
                    start_times_sol[str(aid)] = t
                    found = True
                    break
            if not found:
                start_times_sol[str(aid)] = es

        resource_amounts_sol = {}
        for r in range(num_resources):
            resource_amounts_sol[str(r)] = int(round(R[r].X))

        objective_value = model.ObjVal

        solution = {
            "objective_value": objective_value,
            "start_times": start_times_sol,
            "resource_amounts": resource_amounts_sol
        }

        if logger and objective_value < best_obj_cb[0] - 1e-6:
            logger.log_solution(objective_value, solution)
    else:
        solution = best_solution

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()