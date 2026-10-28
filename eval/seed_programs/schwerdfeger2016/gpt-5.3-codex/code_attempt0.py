import argparse
import json
import math
import time
from typing import List, Dict, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def normalized_objective(loads: List[float], total_processing: float, num_machines: int) -> float:
    avg = total_processing / num_machines
    dev_sq = sum((l - avg) ** 2 for l in loads)
    return math.sqrt(dev_sq) / avg if avg > 0 else 0.0


def sumsq(loads: List[float]) -> float:
    return sum(l * l for l in loads)


def reorder_machines_by_load(
    assignment: List[List[int]], loads: List[int]
) -> Tuple[List[List[int]], List[int]]:
    m = len(loads)
    order = sorted(range(m), key=lambda i: loads[i], reverse=True)
    new_assignment = [assignment[i] for i in order]
    new_loads = [loads[i] for i in order]
    return new_assignment, new_loads


def lpt_initial_assignment(processing_times: List[int], m: int) -> Tuple[List[List[int]], List[int], List[int]]:
    # Greedy: assign each job to currently least loaded machine
    import heapq

    n = len(processing_times)
    assignment = [[] for _ in range(m)]
    loads = [0] * m
    job_to_machine = [-1] * n

    heap = [(0, i) for i in range(m)]  # (load, machine)
    heapq.heapify(heap)

    for j, p in enumerate(processing_times):
        load, i = heapq.heappop(heap)
        assignment[i].append(j)
        loads[i] += p
        job_to_machine[j] = i
        heapq.heappush(heap, (loads[i], i))

    return assignment, loads, job_to_machine


def local_search_relocate(
    processing_times: List[int],
    assignment: List[List[int]],
    loads: List[int],
    job_to_machine: List[int],
    time_end: float,
    logger: Optional[SolutionLogger],
    best_logged_obj: float,
    max_iterations: int = 300,
) -> Tuple[List[List[int]], List[int], List[int], float]:
    """
    Best-improvement single-job relocation local search for minimizing sum(load_i^2).
    """
    n = len(processing_times)
    m = len(loads)
    total_p = sum(processing_times)

    current_sumsq = sumsq(loads)
    current_norm = normalized_objective(loads, total_p, m)

    if current_norm + 1e-12 < best_logged_obj:
        best_logged_obj = current_norm
        if logger:
            logger.log(current_norm)

    it = 0
    while it < max_iterations and time.time() < time_end:
        it += 1
        best_delta = 0.0
        best_move = None  # (job, from_m, to_m)

        # Evaluate all relocations
        for j in range(n):
            a = job_to_machine[j]
            p = processing_times[j]
            la = loads[a]
            base_a = la * la
            for b in range(m):
                if b == a:
                    continue
                lb = loads[b]
                delta = (la - p) * (la - p) + (lb + p) * (lb + p) - base_a - lb * lb
                if delta < best_delta - 1e-12:
                    best_delta = delta
                    best_move = (j, a, b)

        if best_move is None:
            break

        j, a, b = best_move
        p = processing_times[j]

        # Apply move
        assignment[a].remove(j)
        assignment[b].append(j)
        loads[a] -= p
        loads[b] += p
        job_to_machine[j] = b
        current_sumsq += best_delta

        current_norm = normalized_objective(loads, total_p, m)
        if current_norm + 1e-12 < best_logged_obj:
            best_logged_obj = current_norm
            if logger:
                logger.log(current_norm)

    return assignment, loads, job_to_machine, best_logged_obj


class MIPIncumbentCallback:
    def __init__(self, L_vars, total_processing, num_machines, logger, best_logged_obj):
        self.L_vars = L_vars
        self.total_processing = total_processing
        self.num_machines = num_machines
        self.logger = logger
        self.best_logged_obj = best_logged_obj

    def __call__(self, model, where):
        if where == GRB.Callback.MIPSOL and self.logger is not None:
            loads = [model.cbGetSolution(v) for v in self.L_vars]
            obj = normalized_objective(loads, self.total_processing, self.num_machines)
            if obj + 1e-12 < self.best_logged_obj:
                self.best_logged_obj = obj
                self.logger.log(obj)


def solve_instance(
    num_machines: int,
    num_jobs: int,
    processing_times: List[int],
    time_limit: int,
    logger: Optional[SolutionLogger] = None,
):
    start = time.time()
    hard_end = start + time_limit

    m = num_machines
    n = num_jobs
    p = processing_times
    total_p = sum(p)

    # ----------------------
    # Heuristic construction
    # ----------------------
    assignment, loads, job_to_machine = lpt_initial_assignment(p, m)
    best_logged_obj = float("inf")

    init_norm = normalized_objective(loads, total_p, m)
    best_logged_obj = init_norm
    if logger:
        logger.log(init_norm)

    # Local search (leave room for MIP)
    heuristic_end = min(hard_end - 1.0, start + max(0.2, 0.35 * time_limit))
    if heuristic_end > time.time():
        assignment, loads, job_to_machine, best_logged_obj = local_search_relocate(
            p, assignment, loads, job_to_machine, heuristic_end, logger, best_logged_obj
        )

    # Reorder for symmetry-breaking warm start
    assignment, loads = reorder_machines_by_load(assignment, loads)
    job_to_machine = [-1] * n
    for i in range(m):
        for j in assignment[i]:
            job_to_machine[j] = i

    best_assignment = [lst[:] for lst in assignment]
    best_loads = loads[:]
    best_norm = normalized_objective(best_loads, total_p, m)

    # ----------------------
    # MIQP with Gurobi
    # ----------------------
    remaining = hard_end - time.time()
    if remaining > 0.2:
        model = gp.Model("parallel_machine_balance")
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = remaining
        model.Params.MIPFocus = 1

        x = [[model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{i}") for i in range(m)] for j in range(n)]
        L = [model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"L_{i}") for i in range(m)]

        # Each job assigned exactly once
        for j in range(n):
            model.addConstr(gp.quicksum(x[j][i] for i in range(m)) == 1, name=f"assign_{j}")

        # Load definitions
        for i in range(m):
            model.addConstr(L[i] == gp.quicksum(p[j] * x[j][i] for j in range(n)), name=f"load_{i}")

        # Symmetry breaking (machines interchangeable)
        for i in range(m - 1):
            model.addConstr(L[i] >= L[i + 1], name=f"sym_{i}")

        model.setObjective(gp.quicksum(L[i] * L[i] for i in range(m)), GRB.MINIMIZE)

        # Warm start from heuristic
        for j in range(n):
            ai = job_to_machine[j]
            for i in range(m):
                x[j][i].Start = 1.0 if i == ai else 0.0

        callback = MIPIncumbentCallback(L, total_p, m, logger, best_logged_obj)

        try:
            model.optimize(callback if logger else None)
        except gp.GurobiError:
            model.optimize()

        if model.SolCount > 0:
            mip_assignment = [[] for _ in range(m)]
            for j in range(n):
                chosen = None
                for i in range(m):
                    val = x[j][i].X
                    if val > 0.5:
                        chosen = i
                        break
                if chosen is None:
                    # Fallback: pick largest x value
                    chosen = max(range(m), key=lambda i: x[j][i].X)
                mip_assignment[chosen].append(j)

            mip_loads = [0] * m
            for i in range(m):
                mip_loads[i] = sum(p[j] for j in mip_assignment[i])

            mip_norm = normalized_objective(mip_loads, total_p, m)
            if mip_norm < best_norm - 1e-12:
                best_norm = mip_norm
                best_assignment = mip_assignment
                best_loads = mip_loads

    # Prepare output dicts
    assignment_out: Dict[str, List[int]] = {}
    loads_out: Dict[str, int] = {}
    for i in range(m):
        assignment_out[str(i)] = sorted(best_assignment[i])
        loads_out[str(i)] = int(best_loads[i])

    return {
        "objective_value": float(best_norm),
        "assignment": assignment_out,
        "machine_completion_times": loads_out,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    num_machines = int(instance["num_machines"])
    num_jobs = int(instance["num_jobs"])
    processing_times = list(map(int, instance["processing_times"]))

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    solution = solve_instance(
        num_machines=num_machines,
        num_jobs=num_jobs,
        processing_times=processing_times,
        time_limit=args.time_limit,
        logger=logger,
    )

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()