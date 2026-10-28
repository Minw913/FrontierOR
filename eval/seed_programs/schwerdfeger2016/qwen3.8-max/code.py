import json
import argparse
import time
import math
import random
import heapq
from solution_logger import SolutionLogger


def compute_objective(machine_loads, avg):
    """Compute normalized sum of squared workload deviations."""
    if avg == 0:
        return 0.0
    sum_sq_dev = sum((load - avg) ** 2 for load in machine_loads)
    return math.sqrt(sum_sq_dev) / avg


def compute_sum_sq(machine_loads):
    """Sum of squared loads (equivalent objective to minimize)."""
    return sum(load * load for load in machine_loads)


def lpt_heuristic(processing_times, num_machines):
    """Longest Processing Time first heuristic."""
    num_jobs = len(processing_times)
    machine_loads = [0] * num_machines
    assignment = [[] for _ in range(num_machines)]

    heap = [(0, m) for m in range(num_machines)]
    heapq.heapify(heap)

    for job_idx in range(num_jobs):
        load, m = heapq.heappop(heap)
        machine_loads[m] += processing_times[job_idx]
        assignment[m].append(job_idx)
        heapq.heappush(heap, (machine_loads[m], m))

    return assignment, machine_loads


def build_solution_dict(assignment, machine_loads, obj_value):
    """Build the solution dictionary matching the output schema."""
    assignment_dict = {}
    completion_dict = {}
    for m in range(len(assignment)):
        key = str(m)
        assignment_dict[key] = list(assignment[m])
        completion_dict[key] = machine_loads[m]

    return {
        "objective_value": obj_value,
        "assignment": assignment_dict,
        "machine_completion_times": completion_dict
    }


def local_search(processing_times, num_machines, assignment, machine_loads,
                 time_limit, start_time, logger, avg):
    """Local search with moves and swaps."""
    best_assignment = [list(a) for a in assignment]
    best_loads = list(machine_loads)
    best_sum_sq = compute_sum_sq(best_loads)

    max_no_improve_rounds = 3

    while time.time() - start_time < time_limit * 0.92:
        improved_this_round = False

        # Phase 1: Try single job moves
        for src in range(num_machines):
            for dst in range(num_machines):
                if src == dst:
                    continue
                # Only consider moves that could help
                if best_loads[src] <= best_loads[dst]:
                    continue
                for job_idx in list(best_assignment[src]):
                    p = processing_times[job_idx]
                    new_src = best_loads[src] - p
                    new_dst = best_loads[dst] + p
                    old_ss = best_loads[src] ** 2 + best_loads[dst] ** 2
                    new_ss = new_src ** 2 + new_dst ** 2
                    if new_ss < old_ss - 1e-12:
                        best_loads[src] = new_src
                        best_loads[dst] = new_dst
                        best_assignment[src].remove(job_idx)
                        best_assignment[dst].append(job_idx)
                        best_sum_sq = compute_sum_sq(best_loads)
                        improved_this_round = True

                        if logger:
                            obj = compute_objective(best_loads, avg)
                            sol_dict = build_solution_dict(best_assignment, best_loads, obj)
                            logger.log_solution(obj, sol_dict)

        # Phase 2: Try pairwise swaps
        for i in range(num_machines):
            for j in range(i + 1, num_machines):
                if best_loads[i] < best_loads[j]:
                    i, j = j, i
                # i is heavier, j is lighter
                for pos_i, job_i in enumerate(best_assignment[i]):
                    for pos_j, job_j in enumerate(best_assignment[j]):
                        p_i = processing_times[job_i]
                        p_j = processing_times[job_j]
                        if p_i <= p_j:
                            continue

                        new_load_i = best_loads[i] - p_i + p_j
                        new_load_j = best_loads[j] - p_j + p_i
                        old_ss = best_loads[i] ** 2 + best_loads[j] ** 2
                        new_ss = new_load_i ** 2 + new_load_j ** 2

                        if new_ss < old_ss - 1e-12:
                            best_loads[i] = new_load_i
                            best_loads[j] = new_load_j
                            best_assignment[i][pos_i] = job_j
                            best_assignment[j][pos_j] = job_i
                            best_sum_sq = compute_sum_sq(best_loads)
                            improved_this_round = True

                            if logger:
                                obj = compute_objective(best_loads, avg)
                                sol_dict = build_solution_dict(best_assignment, best_loads, obj)
                                logger.log_solution(obj, sol_dict)

        if not improved_this_round:
            break

    obj = compute_objective(best_loads, avg)
    return best_assignment, best_loads, obj


def randomized_local_search(processing_times, num_machines, num_jobs,
                            time_limit, start_time, logger, avg,
                            current_assignment, current_loads, current_obj):
    """Multiple random restarts with perturbation + local search."""
    best_assignment = [list(a) for a in current_assignment]
    best_loads = list(current_loads)
    best_obj = current_obj

    random.seed(42)

    while time.time() - start_time < time_limit * 0.85:
        # Perturbation: randomly reassign some jobs
        perturbed_assignment = [list(a) for a in best_assignment]
        perturbed_loads = list(best_loads)

        num_perturb = max(1, num_jobs // 10)
        for _ in range(num_perturb):
            # Pick a random job and move it to a random machine
            src = random.randint(0, num_machines - 1)
            if len(perturbed_assignment[src]) == 0:
                continue
            dst = random.randint(0, num_machines - 1)
            if src == dst:
                continue
            job_pos = random.randint(0, len(perturbed_assignment[src]) - 1)
            job_idx = perturbed_assignment[src][job_pos]
            p = processing_times[job_idx]
            perturbed_loads[src] -= p
            perturbed_loads[dst] += p
            perturbed_assignment[src].pop(job_pos)
            perturbed_assignment[dst].append(job_idx)

        # Local search from perturbed solution
        new_assignment, new_loads, new_obj = local_search(
            processing_times, num_machines, perturbed_assignment, perturbed_loads,
            time_limit, start_time, logger, avg
        )

        if new_obj < best_obj - 1e-12:
            best_obj = new_obj
            best_assignment = new_assignment
            best_loads = new_loads
            if logger:
                sol_dict = build_solution_dict(best_assignment, best_loads, best_obj)
                logger.log_solution(best_obj, sol_dict)

    return best_assignment, best_loads, best_obj


def solve_with_gurobi(processing_times, num_machines, num_jobs,
                      time_limit, start_time, logger, avg):
    """Solve using Gurobi MIQP."""
    try:
        import gurobipy as gp
        from gurobipy import GRB

        remaining_time = max(1, time_limit - (time.time() - start_time) - 2)

        model = gp.Model("load_balancing")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = remaining_time
        model.Params.OutputFlag = 0

        # Binary assignment variables
        x = {}
        for i in range(num_jobs):
            for j in range(num_machines):
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

        # Continuous completion time variables
        C = {}
        for j in range(num_machines):
            C[j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"C_{j}")

        model.update()

        # Each job assigned to exactly one machine
        for i in range(num_jobs):
            model.addConstr(gp.quicksum(x[i, j] for j in range(num_machines)) == 1,
                            name=f"assign_{i}")

        # Completion time definition
        for j in range(num_machines):
            model.addConstr(C[j] == gp.quicksum(processing_times[i] * x[i, j]
                                                for i in range(num_jobs)),
                            name=f"completion_{j}")

        # Objective: minimize sum of C[j]^2
        model.setObjective(gp.quicksum(C[j] * C[j] for j in range(num_machines)),
                           GRB.MINIMIZE)

        model.optimize()

        if model.SolCount > 0:
            assignment = [[] for _ in range(num_machines)]
            machine_loads = [0] * num_machines

            for i in range(num_jobs):
                for j in range(num_machines):
                    if x[i, j].X > 0.5:
                        assignment[j].append(i)
                        machine_loads[j] += processing_times[i]
                        break

            obj_value = compute_objective(machine_loads, avg)
            return assignment, machine_loads, obj_value

        return None, None, None
    except Exception:
        return None, None, None


def main():
    parser = argparse.ArgumentParser(description="Parallel machine load balancing")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_machines = instance["num_machines"]
    num_jobs = instance["num_jobs"]
    processing_times = instance["processing_times"]

    total_time = sum(processing_times)
    avg = total_time / num_machines

    # Step 1: LPT heuristic for initial solution
    assignment, machine_loads = lpt_heuristic(processing_times, num_machines)
    obj_value = compute_objective(machine_loads, avg)

    if logger:
        sol_dict = build_solution_dict(assignment, machine_loads, obj_value)
        logger.log_solution(obj_value, sol_dict)

    # Step 2: Local search improvement
    assignment, machine_loads, obj_value = local_search(
        processing_times, num_machines, assignment, machine_loads,
        args.time_limit, start_time, logger, avg
    )

    # Step 3: Try Gurobi for small/medium instances
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed

    use_gurobi = (num_jobs * num_machines <= 5000 and remaining > 5)

    if use_gurobi:
        g_assignment, g_loads, g_obj = solve_with_gurobi(
            processing_times, num_machines, num_jobs,
            args.time_limit, start_time, logger, avg
        )
        if g_assignment is not None and g_obj < obj_value - 1e-12:
            assignment, machine_loads, obj_value = g_assignment, g_loads, g_obj
            if logger:
                sol_dict = build_solution_dict(assignment, machine_loads, obj_value)
                logger.log_solution(obj_value, sol_dict)

    # Step 4: Randomized restarts if time permits
    elapsed = time.time() - start_time
    if elapsed < args.time_limit * 0.8 and num_jobs > num_machines:
        assignment, machine_loads, obj_value = randomized_local_search(
            processing_times, num_machines, num_jobs,
            args.time_limit, start_time, logger, avg,
            assignment, machine_loads, obj_value
        )

    # Build and write final solution
    solution = build_solution_dict(assignment, machine_loads, obj_value)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()