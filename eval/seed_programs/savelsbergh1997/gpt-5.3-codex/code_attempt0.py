import argparse
import json
import math
import random
import time
from typing import List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def load_instance(path: str):
    with open(path, "r") as f:
        data = json.load(f)

    num_agents = int(data["num_agents"])
    num_jobs = int(data["num_jobs"])
    profits = data["profits"]
    weights = data["weights"]
    capacities = data["capacities"]

    # Basic validation
    if len(profits) != num_agents or len(weights) != num_agents or len(capacities) != num_agents:
        raise ValueError("Dimension mismatch in agents-related fields.")
    for a in range(num_agents):
        if len(profits[a]) != num_jobs or len(weights[a]) != num_jobs:
            raise ValueError(f"Dimension mismatch for agent {a} row length vs num_jobs.")

    return num_agents, num_jobs, profits, weights, capacities


def greedy_heuristic(
    num_agents: int,
    num_jobs: int,
    profits: List[List[int]],
    weights: List[List[int]],
    capacities: List[int],
    feasible_agents_by_job: List[List[int]],
    time_budget: float,
    rng_seed: int = 0,
) -> Tuple[Optional[List[int]], float]:
    """
    Multi-start randomized greedy constructor for a feasible assignment.
    Returns (assignment, objective). assignment[j] = agent for job j.
    """
    start = time.time()
    rng = random.Random(rng_seed)

    best_assign = None
    best_obj = -math.inf

    # If impossible at single-job level, fail fast.
    for j in range(num_jobs):
        if not feasible_agents_by_job[j]:
            return None, -math.inf

    trial = 0
    while time.time() - start < time_budget:
        trial += 1
        remaining = capacities[:]
        assign = [-1] * num_jobs

        # Harder jobs first: fewer feasible agents, then larger max weight.
        jobs = list(range(num_jobs))
        rng.shuffle(jobs)
        jobs.sort(key=lambda j: (len(feasible_agents_by_job[j]), -max(weights[a][j] for a in feasible_agents_by_job[j])))

        feasible = True
        for j in jobs:
            candidates = []
            for a in feasible_agents_by_job[j]:
                w = weights[a][j]
                if w <= remaining[a]:
                    # Score: prioritize profit with mild penalty for capacity use + random noise
                    score = profits[a][j] - 0.02 * (w / max(1, capacities[a])) + rng.uniform(0.0, 1e-4)
                    candidates.append((score, a))
            if not candidates:
                feasible = False
                break
            candidates.sort(reverse=True)
            # Randomized pick among top few candidates
            top_k = min(3, len(candidates))
            _, chosen_a = candidates[rng.randrange(top_k)]
            assign[j] = chosen_a
            remaining[chosen_a] -= weights[chosen_a][j]

        if feasible:
            obj = float(sum(profits[assign[j]][j] for j in range(num_jobs)))
            if obj > best_obj:
                best_obj = obj
                best_assign = assign

    return best_assign, best_obj


class IncumbentCallback:
    def __init__(self, logger: Optional[SolutionLogger]):
        self.logger = logger
        self.best_logged = -math.inf

    def __call__(self, model, where):
        if where == GRB.Callback.MIPSOL:
            obj = float(model.cbGet(GRB.Callback.MIPSOL_OBJ))
            if obj > self.best_logged + 1e-9:
                self.best_logged = obj
                if self.logger:
                    self.logger.log(obj)


def solve(instance_path: str, solution_path: str, time_limit: int, log_path: Optional[str]):
    t0 = time.time()
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None

    num_agents, num_jobs, profits, weights, capacities = load_instance(instance_path)

    feasible_agents_by_job = []
    for j in range(num_jobs):
        feas = [a for a in range(num_agents) if weights[a][j] <= capacities[a]]
        feasible_agents_by_job.append(feas)

    # Quick heuristic for an initial incumbent
    heuristic_budget = max(0.0, min(2.0, 0.15 * max(0, time_limit)))
    h_assign, h_obj = greedy_heuristic(
        num_agents, num_jobs, profits, weights, capacities, feasible_agents_by_job, heuristic_budget, rng_seed=42
    )

    if h_assign is not None and logger:
        logger.log(float(h_obj))

    # Build MIP model
    model = gp.Model("generalized_assignment")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.MIPFocus = 1

    elapsed = time.time() - t0
    remaining_time = max(0.0, float(time_limit) - elapsed)
    model.Params.TimeLimit = remaining_time

    # Variables only for feasible pairs
    x = {}
    for a in range(num_agents):
        for j in range(num_jobs):
            if weights[a][j] <= capacities[a]:
                x[a, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{a}_{j}")

    model.update()

    # Each job assigned exactly one agent
    for j in range(num_jobs):
        vars_j = [x[a, j] for a in feasible_agents_by_job[j] if (a, j) in x]
        if not vars_j:
            # Infeasible at structural level
            assignment = {str(jj): -1 for jj in range(num_jobs)}
            result = {"objective_value": 0.0, "assignment": assignment}
            with open(solution_path, "w") as f:
                json.dump(result, f)
            return
        model.addConstr(gp.quicksum(vars_j) == 1, name=f"assign_{j}")

    # Capacity constraints
    for a in range(num_agents):
        model.addConstr(
            gp.quicksum(weights[a][j] * x[a, j] for j in range(num_jobs) if (a, j) in x) <= capacities[a],
            name=f"cap_{a}",
        )

    # Objective
    model.setObjective(
        gp.quicksum(profits[a][j] * x[a, j] for (a, j) in x.keys()),
        GRB.MAXIMIZE,
    )

    # Provide MIP start from heuristic if available
    if h_assign is not None:
        for (a, j), var in x.items():
            var.Start = 1.0 if h_assign[j] == a else 0.0

    cb = IncumbentCallback(logger)
    model.optimize(cb)

    final_assignment = None
    final_obj = -math.inf

    if model.SolCount > 0:
        assign = [-1] * num_jobs
        for j in range(num_jobs):
            chosen = None
            best_val = -1.0
            for a in feasible_agents_by_job[j]:
                if (a, j) in x:
                    val = x[a, j].X
                    if val > best_val:
                        best_val = val
                        chosen = a
            # Numerical safety
            if chosen is None:
                chosen = feasible_agents_by_job[j][0]
            assign[j] = int(chosen)

        final_assignment = assign
        final_obj = float(model.ObjVal)
    elif h_assign is not None:
        final_assignment = h_assign
        final_obj = float(h_obj)

    if final_assignment is None:
        # No feasible solution found
        assignment_dict = {str(j): -1 for j in range(num_jobs)}
        result = {"objective_value": 0.0, "assignment": assignment_dict}
    else:
        assignment_dict = {str(j): int(final_assignment[j]) for j in range(num_jobs)}
        # Recompute objective from assignment to ensure consistency
        recomputed_obj = float(sum(profits[final_assignment[j]][j] for j in range(num_jobs)))
        result = {"objective_value": recomputed_obj, "assignment": assignment_dict}

    with open(solution_path, "w") as f:
        json.dump(result, f)


def main():
    parser = argparse.ArgumentParser(description="Solve generalized assignment with capacities (maximize profit).")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input instance JSON.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON.")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to JSONL convergence log (optional).")
    args = parser.parse_args()

    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == "__main__":
    main()