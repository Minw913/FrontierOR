import argparse
import json
import math
import random
import time
from typing import List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def validate_instance(data: dict) -> Tuple[int, int, List[List[int]], List[List[int]], List[int]]:
    num_machines = int(data["num_machines"])
    num_jobs = int(data["num_jobs"])
    cost_matrix = data["cost_matrix"]
    resource_matrix = data["resource_matrix"]
    capacities = data["capacities"]

    if len(cost_matrix) != num_machines:
        raise ValueError("cost_matrix first dimension does not match num_machines.")
    if len(resource_matrix) != num_machines:
        raise ValueError("resource_matrix first dimension does not match num_machines.")
    if len(capacities) != num_machines:
        raise ValueError("capacities length does not match num_machines.")

    for m in range(num_machines):
        if len(cost_matrix[m]) != num_jobs:
            raise ValueError(f"cost_matrix[{m}] length does not match num_jobs.")
        if len(resource_matrix[m]) != num_jobs:
            raise ValueError(f"resource_matrix[{m}] length does not match num_jobs.")

    return num_machines, num_jobs, cost_matrix, resource_matrix, capacities


def greedy_initial_solution(
    cost: List[List[int]],
    resource: List[List[int]],
    capacities: List[int],
    time_budget_sec: float,
    rng_seed: int = 0,
) -> Tuple[Optional[float], Optional[List[int]]]:
    start = time.time()
    rng = random.Random(rng_seed)

    M = len(cost)
    J = len(cost[0]) if M > 0 else 0
    jobs = list(range(J))

    best_obj = math.inf
    best_assign = None

    # Trial count scales with size but respects time budget
    max_trials = min(200, max(20, 2 * J))

    # Difficulty proxy: jobs with larger minimum resource first
    difficult_order = sorted(
        jobs, key=lambda j: min(resource[m][j] for m in range(M)), reverse=True
    )

    for t in range(max_trials):
        if time.time() - start > time_budget_sec:
            break

        residual = capacities[:]
        assign = [-1] * J
        feasible = True

        if t == 0:
            order = difficult_order
            lam = 0.0
        else:
            order = jobs[:]
            rng.shuffle(order)
            lam = rng.uniform(0.0, 5.0)

        for j in order:
            best_m = -1
            best_score = math.inf
            for m in range(M):
                r = resource[m][j]
                if residual[m] >= r:
                    # Penalize tighter residual capacities lightly
                    score = cost[m][j] + lam * (r / max(1.0, residual[m]))
                    if score < best_score:
                        best_score = score
                        best_m = m

            if best_m == -1:
                feasible = False
                break

            assign[j] = best_m
            residual[best_m] -= resource[best_m][j]

        if not feasible:
            continue

        # Simple 1-move improvement (cost-reducing reassignments if capacity allows)
        improved = True
        while improved and (time.time() - start <= time_budget_sec):
            improved = False
            for j in jobs:
                cur = assign[j]
                cur_cost = cost[cur][j]
                cur_res = resource[cur][j]
                for m in range(M):
                    if m == cur:
                        continue
                    new_cost = cost[m][j]
                    if new_cost >= cur_cost:
                        continue
                    needed = resource[m][j]
                    if residual[m] >= needed:
                        # Move j from cur -> m
                        residual[cur] += cur_res
                        residual[m] -= needed
                        assign[j] = m
                        improved = True
                        cur = m
                        cur_cost = new_cost
                        cur_res = needed

        obj = float(sum(cost[assign[j]][j] for j in jobs))
        if obj < best_obj:
            best_obj = obj
            best_assign = assign[:]

    if best_assign is None:
        return None, None
    return best_obj, best_assign


def build_and_solve(
    num_machines: int,
    num_jobs: int,
    cost: List[List[int]],
    resource: List[List[int]],
    capacities: List[int],
    time_limit: int,
    logger: Optional[SolutionLogger],
) -> Tuple[Optional[List[int]], Optional[float]]:
    global_start = time.time()

    # Heuristic warm start
    heuristic_budget = min(5.0, max(0.2, 0.1 * float(time_limit)))
    h_obj, h_assign = greedy_initial_solution(cost, resource, capacities, heuristic_budget)

    best_logged = [math.inf]
    if logger and h_obj is not None and h_obj < best_logged[0]:
        logger.log(float(h_obj))
        best_logged[0] = float(h_obj)

    model = gp.Model("generalized_assignment")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.MIPFocus = 1

    elapsed = time.time() - global_start
    remaining = max(1, int(time_limit - elapsed))
    model.Params.TimeLimit = remaining

    x = model.addVars(num_machines, num_jobs, vtype=GRB.BINARY, name="x")

    # Each job assigned to exactly one machine
    for j in range(num_jobs):
        model.addConstr(gp.quicksum(x[m, j] for m in range(num_machines)) == 1, name=f"assign_{j}")

    # Capacity constraints per machine
    for m in range(num_machines):
        model.addConstr(
            gp.quicksum(resource[m][j] * x[m, j] for j in range(num_jobs)) <= capacities[m],
            name=f"cap_{m}",
        )

    model.setObjective(
        gp.quicksum(cost[m][j] * x[m, j] for m in range(num_machines) for j in range(num_jobs)),
        GRB.MINIMIZE,
    )

    # Apply warm start if found
    if h_assign is not None:
        for m in range(num_machines):
            for j in range(num_jobs):
                x[m, j].Start = 1.0 if h_assign[j] == m else 0.0

    def cb(mdl, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            obj = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < best_logged[0]:
                best_logged[0] = float(obj)
                logger.log(float(obj))

    model.optimize(cb)

    # Extract best available solution
    final_assign = None
    if model.SolCount > 0:
        final_assign = []
        for j in range(num_jobs):
            chosen = None
            for m in range(num_machines):
                if x[m, j].X > 0.5:
                    chosen = m
                    break
            if chosen is None:
                # Numerical fallback
                chosen = max(range(num_machines), key=lambda mm: x[mm, j].X)
            final_assign.append(int(chosen))
    elif h_assign is not None:
        final_assign = h_assign[:]

    if final_assign is None:
        return None, None

    final_obj = float(sum(cost[final_assign[j]][j] for j in range(num_jobs)))
    return final_assign, final_obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    num_machines, num_jobs, cost, resource, capacities = validate_instance(data)

    assignment, objective = build_and_solve(
        num_machines=num_machines,
        num_jobs=num_jobs,
        cost=cost,
        resource=resource,
        capacities=capacities,
        time_limit=args.time_limit,
        logger=logger,
    )

    if assignment is None:
        # No feasible solution found/proved.
        solution = {
            "objective_value": 1e100,
            "assignments": {},
        }
    else:
        solution = {
            "objective_value": float(objective),
            "assignments": {str(j): int(assignment[j]) for j in range(num_jobs)},
        }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()