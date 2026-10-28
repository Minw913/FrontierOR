import argparse
import json
import math
import os
import random
import sys
import time

from solution_logger import SolutionLogger

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None


EPS = 1e-9


def build_solution(costs, assignment):
    objective = float(
        sum(costs[assignment[j]][j] for j in range(len(assignment)))
    )
    return {
        "objective_value": objective,
        "assignments": {
            str(j): int(assignment[j]) for j in range(len(assignment))
        },
    }


def read_instance(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    num_machines = int(data["num_machines"])
    num_jobs = int(data["num_jobs"])
    costs = data["cost_matrix"]
    resources = data["resource_matrix"]
    capacities = data["capacities"]

    if num_machines < 0 or num_jobs < 0:
        raise ValueError("Machine and job counts must be nonnegative")
    if len(costs) != num_machines:
        raise ValueError("Invalid cost_matrix row count")
    if any(len(row) != num_jobs for row in costs):
        raise ValueError("Invalid cost_matrix column count")
    if len(resources) != num_machines:
        raise ValueError("Invalid resource_matrix row count")
    if any(len(row) != num_jobs for row in resources):
        raise ValueError("Invalid resource_matrix column count")
    if len(capacities) != num_machines:
        raise ValueError("Invalid capacities length")
    if num_jobs > 0 and num_machines == 0:
        raise ValueError("No machines are available")

    return num_machines, num_jobs, costs, resources, capacities


def local_search(assignment, costs, resources, capacities, deadline):
    num_jobs = len(assignment)
    num_machines = len(capacities)
    loads = [0] * num_machines

    for job, machine in enumerate(assignment):
        loads[machine] += resources[machine][job]

    for _ in range(8):
        if time.monotonic() >= deadline:
            break

        improved = False

        # Best-improvement single relocations.
        moves = []
        for job in range(num_jobs):
            old_machine = assignment[job]
            for new_machine in range(num_machines):
                if new_machine == old_machine:
                    continue
                saving = (
                    costs[old_machine][job] - costs[new_machine][job]
                )
                if saving > 0:
                    moves.append(
                        (saving, job, old_machine, new_machine)
                    )

        moves.sort(reverse=True)

        for _, job, old_machine, new_machine in moves:
            if assignment[job] != old_machine:
                continue
            if (
                loads[new_machine] + resources[new_machine][job]
                <= capacities[new_machine]
            ):
                loads[old_machine] -= resources[old_machine][job]
                loads[new_machine] += resources[new_machine][job]
                assignment[job] = new_machine
                improved = True

        if not improved:
            break

    return assignment


def greedy_search(
    num_machines,
    num_jobs,
    costs,
    resources,
    capacities,
    deadline,
    logger,
):
    eligible = []
    for job in range(num_jobs):
        choices = [
            machine
            for machine in range(num_machines)
            if resources[machine][job] <= capacities[machine]
        ]
        if not choices:
            return None, None
        eligible.append(choices)

    if num_jobs == 0:
        solution = {"objective_value": 0.0, "assignments": {}}
        if logger:
            logger.log_solution(0.0, solution)
        return [], solution

    def relative_size(job):
        values = []
        for machine in eligible[job]:
            capacity = capacities[machine]
            usage = resources[machine][job]
            if capacity > 0:
                values.append(usage / capacity)
            else:
                values.append(0.0 if usage <= 0 else math.inf)
        return min(values)

    min_sizes = [relative_size(job) for job in range(num_jobs)]

    def regret(job):
        values = sorted(costs[machine][job] for machine in eligible[job])
        if len(values) == 1:
            return float("inf")
        return values[1] - values[0]

    base_orders = [
        sorted(
            range(num_jobs),
            key=lambda job: (
                len(eligible[job]),
                -min_sizes[job],
                -regret(job),
            ),
        ),
        sorted(
            range(num_jobs),
            key=lambda job: (
                len(eligible[job]),
                -regret(job),
                -min_sizes[job],
            ),
        ),
        sorted(
            range(num_jobs),
            key=lambda job: (-min_sizes[job], -regret(job)),
        ),
    ]

    best_assignment = None
    best_solution = None
    best_value = math.inf
    rng = random.Random(0)
    alphas = [1.0, 0.85, 0.65, 0.45, 0.25, 0.0]

    # Always perform a few very cheap attempts, even for a zero time limit.
    max_trials = 96
    for trial in range(max_trials):
        if trial >= len(base_orders) and time.monotonic() >= deadline:
            break

        if trial < len(base_orders):
            order = list(base_orders[trial])
        else:
            order = list(range(num_jobs))
            order.sort(
                key=lambda job: (
                    len(eligible[job]),
                    -min_sizes[job],
                    -regret(job),
                    rng.random(),
                )
            )

        alpha = alphas[trial % len(alphas)]
        remaining = list(capacities)
        assignment = [-1] * num_jobs
        feasible = True

        for job in order:
            choices = [
                machine
                for machine in eligible[job]
                if resources[machine][job] <= remaining[machine]
            ]
            if not choices:
                feasible = False
                break

            minimum_cost = min(costs[machine][job] for machine in choices)
            maximum_cost = max(costs[machine][job] for machine in choices)
            cost_range = max(1.0, float(maximum_cost - minimum_cost))

            def score(machine):
                normalized_cost = (
                    costs[machine][job] - minimum_cost
                ) / cost_range
                remaining_capacity = remaining[machine]
                if remaining_capacity > 0:
                    packing_score = (
                        resources[machine][job] / remaining_capacity
                    )
                else:
                    packing_score = (
                        0.0
                        if resources[machine][job] <= 0
                        else math.inf
                    )
                noise = rng.random() * 1e-8
                return (
                    alpha * normalized_cost
                    + (1.0 - alpha) * packing_score
                    + noise
                )

            selected = min(choices, key=score)
            assignment[job] = selected
            remaining[selected] -= resources[selected][job]

        if not feasible:
            continue

        assignment = local_search(
            assignment, costs, resources, capacities, deadline
        )
        solution = build_solution(costs, assignment)
        value = solution["objective_value"]

        if value < best_value - EPS:
            best_value = value
            best_assignment = list(assignment)
            best_solution = solution
            if logger:
                logger.log_solution(value, solution)

    return best_assignment, best_solution


def branch_and_bound_search(
    num_machines,
    num_jobs,
    costs,
    resources,
    capacities,
    deadline,
    initial_assignment,
    initial_solution,
    logger,
):
    """
    Solver-independent fallback. It is exact if it finishes, and otherwise
    retains the best feasible assignment found before the deadline.
    """
    if num_jobs == 0:
        return [], {"objective_value": 0.0, "assignments": {}}

    eligible = []
    for job in range(num_jobs):
        choices = [
            machine
            for machine in range(num_machines)
            if resources[machine][job] <= capacities[machine]
        ]
        if not choices:
            return initial_assignment, initial_solution
        eligible.append(choices)

    def difficulty(job):
        choices = eligible[job]
        minimum_fraction = min(
            (
                resources[machine][job] / capacities[machine]
                if capacities[machine] > 0
                else 0.0
            )
            for machine in choices
        )
        ordered_costs = sorted(costs[machine][job] for machine in choices)
        regret = (
            ordered_costs[1] - ordered_costs[0]
            if len(ordered_costs) > 1
            else float("inf")
        )
        return (len(choices), -minimum_fraction, -regret)

    order = sorted(range(num_jobs), key=difficulty)

    # Cost ordering quickly produces good incumbents.
    choices_by_job = {
        job: sorted(
            eligible[job],
            key=lambda machine: (
                costs[machine][job],
                resources[machine][job],
            ),
        )
        for job in range(num_jobs)
    }

    suffix_lower_bound = [0.0] * (num_jobs + 1)
    for position in range(num_jobs - 1, -1, -1):
        job = order[position]
        suffix_lower_bound[position] = (
            suffix_lower_bound[position + 1]
            + min(costs[machine][job] for machine in eligible[job])
        )

    best_assignment = (
        list(initial_assignment)
        if initial_assignment is not None
        else None
    )
    best_solution = initial_solution
    best_value = (
        initial_solution["objective_value"]
        if initial_solution is not None
        else math.inf
    )

    remaining = list(capacities)
    current_assignment = [-1] * num_jobs
    nodes = 0
    stopped = False

    # Raise the recursion limit for instances with many jobs.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), num_jobs + 100))

    def search(position, current_cost):
        nonlocal best_assignment, best_solution, best_value, nodes, stopped

        nodes += 1
        if (nodes & 1023) == 0 and time.monotonic() >= deadline:
            stopped = True
            return

        if current_cost + suffix_lower_bound[position] >= best_value - EPS:
            return

        if position == num_jobs:
            assignment = list(current_assignment)
            solution = build_solution(costs, assignment)
            value = solution["objective_value"]
            if value < best_value - EPS:
                best_value = value
                best_assignment = assignment
                best_solution = solution
                if logger:
                    logger.log_solution(value, solution)
            return

        job = order[position]
        for machine in choices_by_job[job]:
            usage = resources[machine][job]
            if usage > remaining[machine]:
                continue

            next_cost = current_cost + costs[machine][job]
            if (
                next_cost + suffix_lower_bound[position + 1]
                >= best_value - EPS
            ):
                continue

            remaining[machine] -= usage
            current_assignment[job] = machine
            search(position + 1, next_cost)
            current_assignment[job] = -1
            remaining[machine] += usage

            if stopped:
                return

    if time.monotonic() < deadline:
        try:
            search(0, 0.0)
        except RecursionError:
            pass

    return best_assignment, best_solution


def gurobi_search(
    num_machines,
    num_jobs,
    costs,
    resources,
    capacities,
    deadline,
    initial_assignment,
    initial_solution,
    logger,
):
    if gp is None or time.monotonic() >= deadline:
        return initial_assignment, initial_solution

    best_assignment = (
        list(initial_assignment)
        if initial_assignment is not None
        else None
    )
    best_solution = initial_solution
    best_value = (
        initial_solution["objective_value"]
        if initial_solution is not None
        else math.inf
    )

    model = None
    environment = None

    try:
        # An explicit environment avoids failures involving Gurobi's global
        # default environment and permits clean fallback when no license exists.
        environment = gp.Env(empty=True)
        environment.setParam("OutputFlag", 0)
        environment.start()

        model = gp.Model("generalized_assignment", env=environment)
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1

        variables = model.addVars(
            num_machines, num_jobs, vtype=GRB.BINARY, name="x"
        )

        model.setObjective(
            gp.quicksum(
                costs[machine][job] * variables[machine, job]
                for machine in range(num_machines)
                for job in range(num_jobs)
            ),
            GRB.MINIMIZE,
        )

        model.addConstrs(
            (
                gp.quicksum(
                    variables[machine, job]
                    for machine in range(num_machines)
                )
                == 1
                for job in range(num_jobs)
            ),
            name="assignment",
        )

        model.addConstrs(
            (
                gp.quicksum(
                    resources[machine][job] * variables[machine, job]
                    for job in range(num_jobs)
                )
                <= capacities[machine]
                for machine in range(num_machines)
            ),
            name="capacity",
        )

        if initial_assignment is not None:
            for machine in range(num_machines):
                for job in range(num_jobs):
                    variables[machine, job].Start = (
                        1.0
                        if initial_assignment[job] == machine
                        else 0.0
                    )

        state = {
            "value": best_value,
            "assignment": best_assignment,
            "solution": best_solution,
        }

        def callback(callback_model, where):
            if where != GRB.Callback.MIPSOL:
                return

            callback_objective = callback_model.cbGet(
                GRB.Callback.MIPSOL_OBJ
            )
            if callback_objective >= state["value"] - EPS:
                return

            values = callback_model.cbGetSolution(variables)
            assignment = [
                max(
                    range(num_machines),
                    key=lambda machine: values[machine, job],
                )
                for job in range(num_jobs)
            ]
            solution = build_solution(costs, assignment)
            value = solution["objective_value"]

            if value < state["value"] - EPS:
                state["value"] = value
                state["assignment"] = assignment
                state["solution"] = solution
                if logger:
                    logger.log_solution(value, solution)

        remaining_time = deadline - time.monotonic()
        if remaining_time > 0:
            model.Params.TimeLimit = max(0.001, remaining_time)
            model.optimize(callback)

        if model.SolCount > 0:
            assignment = [
                max(
                    range(num_machines),
                    key=lambda machine: variables[machine, job].X,
                )
                for job in range(num_jobs)
            ]
            solution = build_solution(costs, assignment)
            value = solution["objective_value"]

            if value < state["value"] - EPS:
                state["value"] = value
                state["assignment"] = assignment
                state["solution"] = solution
                if logger:
                    logger.log_solution(value, solution)

        return state["assignment"], state["solution"]

    except Exception:
        # Gurobi may be installed without an available license. In that case,
        # retain the heuristic incumbent and continue with the independent
        # branch-and-bound fallback.
        return best_assignment, best_solution

    finally:
        if model is not None:
            try:
                model.dispose()
            except Exception:
                pass
        if environment is not None:
            try:
                environment.dispose()
            except Exception:
                pass


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

    (
        num_machines,
        num_jobs,
        costs,
        resources,
        capacities,
    ) = read_instance(args.instance_path)

    if num_jobs == 0:
        final_solution = {
            "objective_value": 0.0,
            "assignments": {},
        }
        if logger:
            logger.log_solution(0.0, final_solution)
    else:
        best_assignment, best_solution = greedy_search(
            num_machines,
            num_jobs,
            costs,
            resources,
            capacities,
            deadline,
            logger,
        )

        # Prefer Gurobi when its runtime and license are available.
        best_assignment, best_solution = gurobi_search(
            num_machines,
            num_jobs,
            costs,
            resources,
            capacities,
            deadline,
            best_assignment,
            best_solution,
            logger,
        )

        # This also handles installations where Gurobi cannot initialize.
        if time.monotonic() < deadline:
            best_assignment, best_solution = branch_and_bound_search(
                num_machines,
                num_jobs,
                costs,
                resources,
                capacities,
                deadline,
                best_assignment,
                best_solution,
                logger,
            )

        if best_solution is None:
            raise RuntimeError(
                "No feasible assignment was found within the time limit; "
                "the instance may be infeasible."
            )

        final_solution = best_solution

    output_directory = os.path.dirname(
        os.path.abspath(args.solution_path)
    )
    os.makedirs(output_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as handle:
        json.dump(final_solution, handle)


if __name__ == "__main__":
    main()