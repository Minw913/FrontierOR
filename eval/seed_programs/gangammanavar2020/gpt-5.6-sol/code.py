import argparse
import json
import os
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def get_random_seed(instance, second_stage, demand_distribution):
    for container in (instance, second_stage, demand_distribution):
        for key in ("random_seed", "seed", "cost_random_seed"):
            if key in container:
                return int(container[key])
    return 0


def generate_scenario_costs(instance):
    second_stage = instance["second_stage"]
    demand_distribution = second_stage["demand_distribution"]

    means = np.asarray(second_stage["transshipment_costs_mean"], dtype=float)
    stds = np.asarray(
        second_stage.get("transshipment_costs_std", np.zeros_like(means)),
        dtype=float,
    )
    random_links = second_stage.get("random_cost_links", [])
    num_scenarios = int(demand_distribution["num_scenarios"])

    seed = get_random_seed(instance, second_stage, demand_distribution)
    rng = np.random.RandomState(seed)

    scenario_costs = np.repeat(means[np.newaxis, :, :], num_scenarios, axis=0)

    std_fraction = float(second_stage.get("random_cost_std_fraction", 0.0))
    for k in range(num_scenarios):
        for link in random_links:
            i, j = int(link[0]), int(link[1])
            std = float(stds[i, j])
            if std == 0.0 and std_fraction != 0.0:
                std = abs(float(means[i, j])) * std_fraction
            scenario_costs[k, i, j] = rng.normal(float(means[i, j]), std)

    return scenario_costs


def local_only_solution(s_values, demands, probabilities, holding, penalty):
    num_scenarios, n = demands.shape
    scenarios = []
    objective = 0.0

    for k in range(num_scenarios):
        e = [0.0] * n
        f = [0.0] * n
        q = [0.0] * n
        r = [0.0] * n
        t = [[0.0] * n for _ in range(n)]

        scenario_cost = 0.0
        for i in range(n):
            stock = float(s_values[i])
            demand = float(demands[k, i])
            served = min(stock, demand)
            ending = max(0.0, stock - demand)
            backlog = max(0.0, demand - stock)

            f[i] = served
            e[i] = ending
            q[i] = served
            r[i] = backlog
            scenario_cost += holding[i] * ending + penalty[i] * backlog

        objective += probabilities[k] * scenario_cost
        scenarios.append({"e": e, "f": f, "q": q, "r": r, "t": t})

    return {
        "objective_value": float(objective),
        "s": [float(x) for x in s_values],
        "scenarios": scenarios,
    }


def weighted_newsvendor_levels(demands, probabilities, holding, penalty, lower_bounds):
    num_scenarios, n = demands.shape
    total_probability = float(np.sum(probabilities))
    levels = np.asarray(lower_bounds, dtype=float).copy()

    if total_probability <= 0.0:
        return levels

    for i in range(n):
        h = float(holding[i])
        p = float(penalty[i])

        if p <= 0.0:
            candidate = 0.0
        elif h <= 0.0:
            candidate = float(np.max(demands[:, i]))
        else:
            target = (p / (p + h)) * total_probability
            order = np.argsort(demands[:, i], kind="mergesort")
            cumulative = 0.0
            candidate = float(demands[order[-1], i])
            for k in order:
                cumulative += float(probabilities[k])
                if cumulative + 1e-15 >= target:
                    candidate = float(demands[k, i])
                    break

        levels[i] = max(float(lower_bounds[i]), candidate)

    return levels


def clean_nonnegative(value, tolerance=1e-8):
    value = float(value)
    if -tolerance <= value <= tolerance:
        return 0.0
    return max(0.0, value)


def extract_solution(model, s, f, t_vars, demands, probabilities,
                     holding, penalty, scenario_costs):
    num_scenarios, n = demands.shape

    s_values = [clean_nonnegative(s[i].X) for i in range(n)]
    scenarios = []
    objective = 0.0

    for k in range(num_scenarios):
        f_values = [clean_nonnegative(f[k, i].X) for i in range(n)]
        t_values = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i != j:
                    t_values[i][j] = clean_nonnegative(t_vars[k, i, j].X)

        e_values = [0.0] * n
        q_values = [0.0] * n
        r_values = [0.0] * n

        for i in range(n):
            outbound = sum(t_values[i][j] for j in range(n) if j != i)
            inbound = sum(t_values[j][i] for j in range(n) if j != i)

            q_values[i] = clean_nonnegative(f_values[i] + outbound)
            e_values[i] = clean_nonnegative(s_values[i] - q_values[i])
            r_values[i] = clean_nonnegative(
                float(demands[k, i]) - f_values[i] - inbound
            )

        scenario_cost = 0.0
        for i in range(n):
            scenario_cost += holding[i] * e_values[i]
            scenario_cost += penalty[i] * r_values[i]
            for j in range(n):
                if i != j:
                    scenario_cost += scenario_costs[k, i, j] * t_values[i][j]

        objective += probabilities[k] * scenario_cost
        scenarios.append({
            "e": e_values,
            "f": f_values,
            "q": q_values,
            "r": r_values,
            "t": t_values,
        })

    return {
        "objective_value": float(objective),
        "s": s_values,
        "scenarios": scenarios,
    }


def write_solution(path, solution):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(solution, file, separators=(",", ":"), allow_nan=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as file:
        instance = json.load(file)

    n = int(instance["N"])
    second_stage = instance["second_stage"]
    demand_distribution = second_stage["demand_distribution"]

    demands = np.asarray(demand_distribution["scenarios"], dtype=float)
    probabilities = np.asarray(demand_distribution["probabilities"], dtype=float)
    num_scenarios = int(demand_distribution["num_scenarios"])

    if demands.shape != (num_scenarios, n):
        raise ValueError("Demand scenario array has an invalid shape.")

    holding = np.asarray(second_stage["holding_costs"], dtype=float)
    penalty = np.asarray(second_stage["penalty_costs"], dtype=float)
    lower_bounds = np.asarray(
        instance["first_stage"]["variable_lower_bounds"], dtype=float
    )
    scenario_costs = generate_scenario_costs(instance)

    # Always construct an immediate feasible incumbent.
    best_solution = local_only_solution(
        lower_bounds, demands, probabilities, holding, penalty
    )
    best_objective = float(best_solution["objective_value"])
    if logger:
        logger.log_solution(best_objective, best_solution)

    # A stronger no-transshipment newsvendor incumbent.
    newsvendor_levels = weighted_newsvendor_levels(
        demands, probabilities, holding, penalty, lower_bounds
    )
    newsvendor_solution = local_only_solution(
        newsvendor_levels, demands, probabilities, holding, penalty
    )
    if newsvendor_solution["objective_value"] < best_objective - 1e-9:
        best_solution = newsvendor_solution
        best_objective = float(newsvendor_solution["objective_value"])
        if logger:
            logger.log_solution(best_objective, best_solution)

    remaining = float(args.time_limit) - (time.monotonic() - start_time)
    if remaining <= 0.01:
        write_solution(args.solution_path, best_solution)
        return

    model = gp.Model("stochastic_retail_transshipment")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0

    max_total_demand = float(np.max(np.sum(demands, axis=1))) if num_scenarios else 0.0
    upper_bounds = [
        max(float(lower_bounds[i]), max_total_demand) for i in range(n)
    ]

    s = model.addVars(
        n,
        lb={i: float(lower_bounds[i]) for i in range(n)},
        ub={i: upper_bounds[i] for i in range(n)},
        name="s",
    )
    e = model.addVars(num_scenarios, n, lb=0.0, name="e")
    f = model.addVars(num_scenarios, n, lb=0.0, name="f")
    r = model.addVars(num_scenarios, n, lb=0.0, name="r")

    transshipment_keys = [
        (k, i, j)
        for k in range(num_scenarios)
        for i in range(n)
        for j in range(n)
        if i != j
    ]
    t_vars = model.addVars(transshipment_keys, lb=0.0, name="t")

    for k in range(num_scenarios):
        for i in range(n):
            outbound = gp.quicksum(
                t_vars[k, i, j] for j in range(n) if j != i
            )
            inbound = gp.quicksum(
                t_vars[k, j, i] for j in range(n) if j != i
            )

            model.addConstr(
                f[k, i] + outbound + e[k, i] == s[i],
                name=f"stock_balance_{k}_{i}",
            )
            model.addConstr(
                f[k, i] + inbound + r[k, i] == float(demands[k, i]),
                name=f"demand_balance_{k}_{i}",
            )

    objective = gp.LinExpr()
    for k in range(num_scenarios):
        probability = float(probabilities[k])
        for i in range(n):
            objective += probability * float(holding[i]) * e[k, i]
            objective += probability * float(penalty[i]) * r[k, i]
            for j in range(n):
                if i != j:
                    objective += (
                        probability
                        * float(scenario_costs[k, i, j])
                        * t_vars[k, i, j]
                    )

    model.setObjective(objective, GRB.MINIMIZE)

    # Supply the best heuristic as a primal start.
    for i in range(n):
        s[i].Start = float(best_solution["s"][i])

    for k in range(num_scenarios):
        scenario = best_solution["scenarios"][k]
        for i in range(n):
            e[k, i].Start = float(scenario["e"][i])
            f[k, i].Start = float(scenario["f"][i])
            r[k, i].Start = float(scenario["r"][i])
            for j in range(n):
                if i != j:
                    t_vars[k, i, j].Start = float(scenario["t"][i][j])

    remaining = float(args.time_limit) - (time.monotonic() - start_time)
    if remaining > 0.001:
        model.Params.TimeLimit = max(0.001, remaining)
        model.optimize()

        if model.SolCount > 0:
            candidate = extract_solution(
                model,
                s,
                f,
                t_vars,
                demands,
                probabilities,
                holding,
                penalty,
                scenario_costs,
            )
            candidate_objective = float(candidate["objective_value"])

            if candidate_objective < best_objective - 1e-8:
                best_solution = candidate
                best_objective = candidate_objective
                if logger:
                    logger.log_solution(best_objective, best_solution)

    write_solution(args.solution_path, best_solution)


if __name__ == "__main__":
    main()