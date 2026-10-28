import argparse
import json
import math
import random
import time

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def make_solution(profits, assignment):
    objective = sum(profits[agent][job] for job, agent in enumerate(assignment))
    return {
        "objective_value": float(objective),
        "assignment": {str(job): int(agent) for job, agent in enumerate(assignment)},
    }


def construct_greedy(options, profits, weights, capacities, order, alpha, profit_scale):
    num_agents = len(capacities)
    num_jobs = len(order)
    remaining = list(capacities)
    assignment = [-1] * num_jobs

    for job in order:
        best_agent = -1
        best_score = -math.inf
        best_tie = None

        for agent in options[job]:
            weight = weights[agent][job]
            if weight > remaining[agent]:
                continue

            normalized_weight = weight / max(1, remaining[agent])
            score = profits[agent][job] - alpha * profit_scale * normalized_weight
            tie = (
                profits[agent][job],
                -weight / max(1, capacities[agent]),
                remaining[agent] - weight,
                -agent,
            )

            if score > best_score or (
                score == best_score and (best_tie is None or tie > best_tie)
            ):
                best_score = score
                best_tie = tie
                best_agent = agent

        if best_agent < 0:
            return None

        assignment[job] = best_agent
        remaining[best_agent] -= weights[best_agent][job]

    return assignment


def improve_direct_moves(assignment, profits, weights, capacities, deadline):
    num_agents = len(capacities)
    num_jobs = len(assignment)
    loads = [0] * num_agents

    for job, agent in enumerate(assignment):
        loads[agent] += weights[agent][job]

    # A few improving passes provide a better MIP start without spending too
    # much of the solver's time budget.
    for _ in range(3):
        improved = False
        for job in range(num_jobs):
            if time.monotonic() >= deadline:
                return assignment

            old_agent = assignment[job]
            old_profit = profits[old_agent][job]
            best_agent = old_agent
            best_gain = 0

            for agent in range(num_agents):
                if agent == old_agent:
                    continue
                if loads[agent] + weights[agent][job] > capacities[agent]:
                    continue

                gain = profits[agent][job] - old_profit
                if gain > best_gain:
                    best_gain = gain
                    best_agent = agent

            if best_agent != old_agent:
                loads[old_agent] -= weights[old_agent][job]
                loads[best_agent] += weights[best_agent][job]
                assignment[job] = best_agent
                improved = True

        if not improved:
            break

    return assignment


def write_solution(path, solution):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(solution, file, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    num_agents = int(data["num_agents"])
    num_jobs = int(data["num_jobs"])
    profits = data["profits"]
    weights = data["weights"]
    capacities = data["capacities"]

    options = []
    for job in range(num_jobs):
        feasible_agents = [
            agent
            for agent in range(num_agents)
            if weights[agent][job] <= capacities[agent]
        ]
        if not feasible_agents:
            raise RuntimeError(f"Instance is infeasible: job {job} has no feasible agent")
        options.append(feasible_agents)

    incumbent = {
        "objective": -math.inf,
        "assignment": None,
        "solution": None,
    }

    def adopt(assignment):
        solution = make_solution(profits, assignment)
        objective = int(round(solution["objective_value"]))
        if objective > incumbent["objective"]:
            incumbent["objective"] = objective
            incumbent["assignment"] = list(assignment)
            incumbent["solution"] = solution
            if logger:
                logger.log_solution(float(objective), solution)

    # Generate a deterministic feasible MIP start when possible.
    if num_jobs == 0:
        adopt([])
    else:
        profit_scale = max(
            1,
            max(profits[agent][job]
                for agent in range(num_agents)
                for job in range(num_jobs)),
        )

        difficulty_order = sorted(
            range(num_jobs),
            key=lambda job: (
                len(options[job]),
                -min(
                    weights[agent][job] / max(1, capacities[agent])
                    for agent in options[job]
                ),
                -max(profits[agent][job] for agent in options[job]),
            ),
        )

        regret_order = sorted(
            range(num_jobs),
            key=lambda job: (
                -(
                    sorted(
                        (profits[agent][job] for agent in options[job]),
                        reverse=True,
                    )[0]
                    - (
                        sorted(
                            (profits[agent][job] for agent in options[job]),
                            reverse=True,
                        )[1]
                        if len(options[job]) > 1
                        else -profit_scale
                    )
                ),
                len(options[job]),
            ),
        )

        heavy_order = sorted(
            range(num_jobs),
            key=lambda job: -min(
                weights[agent][job] / max(1, capacities[agent])
                for agent in options[job]
            ),
        )

        heuristic_budget = min(
            0.5,
            max(0.03, max(0, args.time_limit) * 0.05),
        )
        heuristic_deadline = min(deadline, time.monotonic() + heuristic_budget)

        base_orders = [
            difficulty_order,
            regret_order,
            heavy_order,
            list(range(num_jobs)),
        ]
        alphas = (0.0, 0.1, 0.5, 1.0, 2.0)

        # Always perform at least one inexpensive deterministic attempt.
        attempted = False
        for order in base_orders:
            for alpha in alphas:
                if attempted and time.monotonic() >= heuristic_deadline:
                    break
                attempted = True
                candidate = construct_greedy(
                    options,
                    profits,
                    weights,
                    capacities,
                    order,
                    alpha,
                    profit_scale,
                )
                if candidate is not None:
                    candidate = improve_direct_moves(
                        candidate,
                        profits,
                        weights,
                        capacities,
                        heuristic_deadline,
                    )
                    adopt(candidate)
            if time.monotonic() >= heuristic_deadline:
                break

        rng = random.Random(0)
        random_attempts = 0
        while (
            time.monotonic() < heuristic_deadline
            and random_attempts < 20
            and num_jobs > 1
        ):
            order = difficulty_order[:]
            # Perturb blocks while retaining some preference for difficult jobs.
            block = max(2, int(math.sqrt(num_jobs)))
            for first in range(0, num_jobs, block):
                part = order[first:first + block]
                rng.shuffle(part)
                order[first:first + block] = part

            alpha = alphas[random_attempts % len(alphas)]
            candidate = construct_greedy(
                options,
                profits,
                weights,
                capacities,
                order,
                alpha,
                profit_scale,
            )
            if candidate is not None:
                candidate = improve_direct_moves(
                    candidate,
                    profits,
                    weights,
                    capacities,
                    heuristic_deadline,
                )
                adopt(candidate)
            random_attempts += 1

    # If the full time budget has already elapsed, return the heuristic solution.
    if time.monotonic() >= deadline and incumbent["solution"] is not None:
        write_solution(args.solution_path, incumbent["solution"])
        return

    model = gp.Model()
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.ModelSense = GRB.MAXIMIZE

    variables = []
    variable_jobs = []
    variable_agents = []
    vars_by_job = [[] for _ in range(num_jobs)]
    vars_by_agent = [[] for _ in range(num_agents)]

    for job in range(num_jobs):
        for agent in options[job]:
            var = model.addVar(
                vtype=GRB.BINARY,
                obj=float(profits[agent][job]),
            )
            variables.append(var)
            variable_jobs.append(job)
            variable_agents.append(agent)
            vars_by_job[job].append(var)
            vars_by_agent[agent].append((job, var))

    for job in range(num_jobs):
        model.addConstr(gp.quicksum(vars_by_job[job]) == 1)

    for agent in range(num_agents):
        model.addConstr(
            gp.quicksum(
                weights[agent][job] * var
                for job, var in vars_by_agent[agent]
            )
            <= capacities[agent]
        )

    model.update()

    if incumbent["assignment"] is not None:
        start_assignment = incumbent["assignment"]
        for index, var in enumerate(variables):
            var.Start = (
                1.0
                if start_assignment[variable_jobs[index]] == variable_agents[index]
                else 0.0
            )

    remaining = max(0.0, deadline - time.monotonic())
    model.Params.TimeLimit = remaining

    def callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            values = cb_model.cbGetSolution(variables)
            assignment = [-1] * num_jobs
            selected_value = [-math.inf] * num_jobs

            for index, value in enumerate(values):
                job = variable_jobs[index]
                if value > selected_value[job]:
                    selected_value[job] = value
                    assignment[job] = variable_agents[index]

            if all(agent >= 0 for agent in assignment):
                adopt(assignment)
        except Exception:
            # Logging or callback extraction must not abort the optimization.
            return

    model.optimize(callback)

    if model.SolCount > 0:
        assignment = [-1] * num_jobs
        selected_value = [-math.inf] * num_jobs

        for index, var in enumerate(variables):
            job = variable_jobs[index]
            value = var.X
            if value > selected_value[job]:
                selected_value[job] = value
                assignment[job] = variable_agents[index]

        if all(agent >= 0 for agent in assignment):
            adopt(assignment)

    if incumbent["solution"] is None:
        if model.Status == GRB.INFEASIBLE:
            raise RuntimeError("The instance is infeasible")
        raise RuntimeError("No feasible solution was found within the time limit")

    write_solution(args.solution_path, incumbent["solution"])


if __name__ == "__main__":
    main()