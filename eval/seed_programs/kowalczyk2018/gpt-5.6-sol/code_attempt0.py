import argparse
import json
import os
import time
from functools import cmp_to_key

from solution_logger import SolutionLogger


def smith_compare(a, b):
    """Order jobs by nonincreasing w/p without floating-point division."""
    left = a["w"] * b["p"]
    right = b["w"] * a["p"]
    if left > right:
        return -1
    if left < right:
        return 1
    # Deterministic tie-breaking; any order is optimal for equal ratios.
    if a["_input_pos"] < b["_input_pos"]:
        return -1
    if a["_input_pos"] > b["_input_pos"]:
        return 1
    return 0


def canonicalize_assignment(assignment, m):
    """Relabel used identical machines by order of first appearance."""
    mapping = {}
    next_label = 0
    result = []
    for machine in assignment:
        if machine not in mapping:
            mapping[machine] = next_label
            next_label += 1
        result.append(mapping[machine])
    return result


def build_solution(jobs, assignment, m):
    machine_jobs = [[] for _ in range(m)]
    for idx, machine in enumerate(assignment):
        machine_jobs[machine].append(idx)

    machines_output = []
    total_cost = 0

    for indices in machine_jobs:
        completion = 0
        schedule_cost = 0
        output_jobs = []

        for idx in indices:
            job = jobs[idx]
            completion += job["p"]
            schedule_cost += job["w"] * completion
            output_jobs.append(
                {
                    "j_orig": job["j"],
                    "p": job["p"],
                    "w": job["w"],
                    "C_j": completion,
                }
            )

        total_cost += schedule_cost
        machines_output.append(
            {
                "schedule_cost": schedule_cost,
                "jobs": output_jobs,
            }
        )

    return {
        "objective_value": float(total_cost),
        "machines": machines_output,
    }, total_cost


def greedy_assignment(jobs, m):
    loads = [0] * m
    assignment = []

    for job in jobs:
        machine = min(range(m), key=lambda k: (loads[k], k))
        assignment.append(machine)
        loads[machine] += job["p"]

    return canonicalize_assignment(assignment, m)


def improve_by_single_moves(
    jobs,
    assignment,
    m,
    deadline,
    logger,
    best_solution,
    best_cost,
):
    """
    Best-improvement local search over all one-job machine moves.

    With jobs in Smith order, removing or inserting job j changes only its
    interactions with jobs before and after it on the affected machine.
    """
    n = len(jobs)
    if n == 0 or m <= 1:
        return assignment, best_solution, best_cost

    max_iterations = min(5000, max(50, 5 * n))
    iteration = 0

    while iteration < max_iterations and time.monotonic() < deadline:
        iteration += 1

        total_weight = [0] * m
        for j, machine in enumerate(assignment):
            total_weight[machine] += jobs[j]["w"]

        prefix_p = [[0] * n for _ in range(m)]
        prefix_w = [[0] * n for _ in range(m)]
        running_p = [0] * m
        running_w = [0] * m

        for j in range(n):
            for k in range(m):
                prefix_p[k][j] = running_p[k]
                prefix_w[k][j] = running_w[k]

            machine = assignment[j]
            running_p[machine] += jobs[j]["p"]
            running_w[machine] += jobs[j]["w"]

        best_delta = 0
        best_move = None

        for j, job in enumerate(jobs):
            source = assignment[j]
            p_j = job["p"]
            w_j = job["w"]

            removal = (
                w_j * prefix_p[source][j]
                + p_j
                * (
                    total_weight[source]
                    - prefix_w[source][j]
                    - w_j
                )
            )

            for target in range(m):
                if target == source:
                    continue

                insertion = (
                    w_j * prefix_p[target][j]
                    + p_j
                    * (
                        total_weight[target]
                        - prefix_w[target][j]
                    )
                )
                delta = insertion - removal

                if delta < best_delta:
                    best_delta = delta
                    best_move = (j, target)

        if best_move is None:
            break

        job_idx, target = best_move
        assignment[job_idx] = target
        assignment = canonicalize_assignment(assignment, m)

        solution, cost = build_solution(jobs, assignment, m)
        if cost < best_cost:
            best_cost = cost
            best_solution = solution
            if logger:
                logger.log_solution(float(cost), solution)

    return assignment, best_solution, best_cost


def solve_with_gurobi(
    jobs,
    m,
    initial_assignment,
    deadline,
    logger,
    best_solution,
    best_cost,
):
    n = len(jobs)
    remaining = deadline - time.monotonic()
    if n == 0 or remaining <= 0.1:
        return best_solution, best_cost

    # Avoid excessive model-construction time and memory on very large inputs.
    variable_grid = n * m
    if variable_grid > 100000:
        return best_solution, best_cost

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return best_solution, best_cost

    try:
        model = gp.Model("parallel_machine_weighted_completion")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1

        # x[j,k] assigns job j to machine k. Machine labels are canonical:
        # machine k cannot first appear before machine k-1 has appeared.
        x = {}
        for j in range(n):
            for k in range(m):
                ub = 0.0 if k > j else 1.0
                x[j, k] = model.addVar(
                    vtype=GRB.BINARY,
                    lb=0.0,
                    ub=ub,
                    name=f"x_{j}_{k}",
                )

        # load[j,k] is the cumulative processing load on machine k after
        # considering jobs 0..j in Smith order.
        load = {}
        completion = {}

        prefix_processing = 0
        for j, job in enumerate(jobs):
            prefix_processing += job["p"]
            completion[j] = model.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0.0,
                ub=float(prefix_processing),
                name=f"C_{j}",
            )
            for k in range(m):
                load[j, k] = model.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=0.0,
                    ub=float(prefix_processing),
                    name=f"L_{j}_{k}",
                )

        model.update()

        for j in range(n):
            model.addConstr(
                gp.quicksum(x[j, k] for k in range(m)) == 1,
                name=f"assign_{j}",
            )

        prefix_processing = 0
        for j, job in enumerate(jobs):
            prefix_processing += job["p"]

            for k in range(m):
                if j == 0:
                    model.addConstr(
                        load[j, k] == job["p"] * x[j, k],
                        name=f"load_{j}_{k}",
                    )
                else:
                    model.addConstr(
                        load[j, k]
                        == load[j - 1, k] + job["p"] * x[j, k],
                        name=f"load_{j}_{k}",
                    )

                # Active when j is assigned to k; otherwise the big-M term
                # makes the constraint redundant.
                model.addConstr(
                    completion[j]
                    >= load[j, k]
                    - prefix_processing * (1 - x[j, k]),
                    name=f"completion_{j}_{k}",
                )

        # Symmetry breaking: machines are labeled in order of first use.
        if n > 0:
            model.addConstr(x[0, 0] == 1, name="first_machine")

        for j in range(1, n):
            upper_k = min(m - 1, j)
            for k in range(1, upper_k + 1):
                model.addConstr(
                    x[j, k]
                    <= gp.quicksum(x[i, k - 1] for i in range(j)),
                    name=f"machine_order_{j}_{k}",
                )

        model.setObjective(
            gp.quicksum(jobs[j]["w"] * completion[j] for j in range(n)),
            GRB.MINIMIZE,
        )

        # MIP start.
        cumulative_loads = [0] * m
        for j, job in enumerate(jobs):
            assigned = initial_assignment[j]
            for k in range(m):
                x[j, k].Start = 1.0 if k == assigned else 0.0
                cumulative_loads[k] += job["p"] if k == assigned else 0
                load[j, k].Start = float(cumulative_loads[k])
            completion[j].Start = float(cumulative_loads[assigned])

        model.update()

        remaining = deadline - time.monotonic()
        if remaining <= 0.05:
            return best_solution, best_cost
        model.Params.TimeLimit = max(0.01, remaining)

        x_list = [x[j, k] for j in range(n) for k in range(m)]
        incumbent_state = {
            "cost": best_cost,
            "solution": best_solution,
        }

        def callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return

            try:
                values = cb_model.cbGetSolution(x_list)
                candidate_assignment = [0] * n

                offset = 0
                for j in range(n):
                    best_k = 0
                    best_value = -1.0
                    for k in range(m):
                        value = values[offset]
                        offset += 1
                        if value > best_value:
                            best_value = value
                            best_k = k
                    candidate_assignment[j] = best_k

                candidate_assignment = canonicalize_assignment(
                    candidate_assignment, m
                )
                solution, cost = build_solution(
                    jobs, candidate_assignment, m
                )

                if cost < incumbent_state["cost"]:
                    incumbent_state["cost"] = cost
                    incumbent_state["solution"] = solution
                    if logger:
                        logger.log_solution(float(cost), solution)
            except Exception:
                # A logging or callback extraction failure must not abort the
                # optimization run.
                pass

        model.optimize(callback)

        if model.SolCount > 0:
            candidate_assignment = []
            for j in range(n):
                machine = max(range(m), key=lambda k: x[j, k].X)
                candidate_assignment.append(machine)

            candidate_assignment = canonicalize_assignment(
                candidate_assignment, m
            )
            solution, cost = build_solution(
                jobs, candidate_assignment, m
            )
            if cost < incumbent_state["cost"]:
                incumbent_state["cost"] = cost
                incumbent_state["solution"] = solution
                if logger:
                    logger.log_solution(float(cost), solution)

        return incumbent_state["solution"], incumbent_state["cost"]

    except Exception:
        # Return the best heuristic incumbent if Gurobi is unavailable,
        # unlicensed, or model construction/optimization fails.
        return best_solution, best_cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path
        else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as file:
        instance = json.load(file)

    n = int(instance["n"])
    m = int(instance["m"])
    if m <= 0:
        raise ValueError("The number of machines must be positive.")

    raw_jobs = instance.get("jobs", [])
    if len(raw_jobs) != n:
        raise ValueError("The jobs array length does not match n.")

    jobs = []
    for pos, raw_job in enumerate(raw_jobs):
        p = int(raw_job["p"])
        w = int(raw_job["w"])
        if p < 0 or w < 0:
            raise ValueError("Processing times and weights must be nonnegative.")
        jobs.append(
            {
                "j": int(raw_job["j"]),
                "p": p,
                "w": w,
                "_input_pos": pos,
            }
        )

    # Smith's rule is optimal for the sequence on each individual machine.
    jobs.sort(key=cmp_to_key(smith_compare))

    if n == 0:
        solution = {
            "objective_value": 0.0,
            "machines": [
                {"schedule_cost": 0, "jobs": []} for _ in range(m)
            ],
        }
        if logger:
            logger.log_solution(0.0, solution)
    else:
        assignment = greedy_assignment(jobs, m)
        solution, cost = build_solution(jobs, assignment, m)

        if logger:
            logger.log_solution(float(cost), solution)

        # Reserve most of the runtime for the exact MIP search.
        total_budget = max(0.0, deadline - time.monotonic())
        heuristic_budget = min(5.0, 0.20 * total_budget)
        local_deadline = min(deadline, time.monotonic() + heuristic_budget)

        assignment, solution, cost = improve_by_single_moves(
            jobs=jobs,
            assignment=assignment,
            m=m,
            deadline=local_deadline,
            logger=logger,
            best_solution=solution,
            best_cost=cost,
        )

        if time.monotonic() < deadline:
            solution, cost = solve_with_gurobi(
                jobs=jobs,
                m=m,
                initial_assignment=assignment,
                deadline=deadline,
                logger=logger,
                best_solution=solution,
                best_cost=cost,
            )

    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as file:
        json.dump(solution, file, indent=2)


if __name__ == "__main__":
    main()