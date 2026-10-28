import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def clean_nonnegative(value):
    value = float(value)
    return 0.0 if value < 0.0 else value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    NI = int(data["parameters"]["NI"])
    NK = int(data["parameters"]["NK"])
    NT = int(data["parameters"]["NT"])

    setup_cost = float(data["costs"]["setup_cost_f"])
    startup_cost = float(data["costs"]["startup_cost_g"])
    holding_cost = [float(x) for x in data["costs"]["holding_cost_h"]]
    backlog_cost = [float(x) for x in data["costs"]["backlogging_cost_e"]]

    capacity = [float(x) for x in data["machine_data"]["capacity_C"]]
    startup_loss = [
        float(x) for x in data["machine_data"]["startup_time_sigma"]
    ]

    demand = [
        [float(data["demand"][i][t]) for t in range(NT)]
        for i in range(NI)
    ]
    initial_stock = [
        float(x) for x in data["initial_conditions"]["initial_stock_s0"]
    ]
    initial_backlog = [
        float(x) for x in data["initial_conditions"]["initial_backlog_r0"]
    ]

    triples = [
        (i, k, t)
        for i in range(NI)
        for k in range(NK)
        for t in range(NT)
    ]
    pairs = [(i, t) for i in range(NI) for t in range(NT)]

    def calculate_objective(stock_values, backlog_values,
                            setup_values, startup_values):
        value = 0.0
        value += setup_cost * sum(setup_values)
        value += startup_cost * sum(startup_values)
        for index, (i, _) in enumerate(pairs):
            value += holding_cost[i] * stock_values[index]
            value += backlog_cost[i] * backlog_values[index]
        return float(value)

    def make_solution(production_values, stock_values, backlog_values,
                      setup_values, startup_values):
        production_values = [
            clean_nonnegative(x) for x in production_values
        ]
        stock_values = [clean_nonnegative(x) for x in stock_values]
        backlog_values = [clean_nonnegative(x) for x in backlog_values]
        setup_values = [1 if float(x) >= 0.5 else 0 for x in setup_values]
        startup_values = [1 if float(x) >= 0.5 else 0 for x in startup_values]

        objective = calculate_objective(
            stock_values, backlog_values, setup_values, startup_values
        )

        solution = {
            "objective_value": objective,
            "production": {
                f"{i}_{k}_{t}": production_values[j]
                for j, (i, k, t) in enumerate(triples)
            },
            "stock": {
                f"{i}_{t}": stock_values[j]
                for j, (i, t) in enumerate(pairs)
            },
            "backlog": {
                f"{i}_{t}": backlog_values[j]
                for j, (i, t) in enumerate(pairs)
            },
            "setup": {
                f"{i}_{k}_{t}": setup_values[j]
                for j, (i, k, t) in enumerate(triples)
            },
            "startup": {
                f"{i}_{k}_{t}": startup_values[j]
                for j, (i, k, t) in enumerate(triples)
            },
        }
        return solution

    # Always-available feasible plan: produce nothing and carry the resulting
    # net requirements as either inventory or backlog.
    fallback_production = [0.0] * len(triples)
    fallback_setup = [0] * len(triples)
    fallback_startup = [0] * len(triples)
    fallback_stock_map = {}
    fallback_backlog_map = {}

    for i in range(NI):
        net_stock = initial_stock[i] - initial_backlog[i]
        for t in range(NT):
            net_stock -= demand[i][t]
            fallback_stock_map[(i, t)] = max(net_stock, 0.0)
            fallback_backlog_map[(i, t)] = max(-net_stock, 0.0)

    fallback_stock = [fallback_stock_map[p] for p in pairs]
    fallback_backlog = [fallback_backlog_map[p] for p in pairs]

    best_solution = make_solution(
        fallback_production,
        fallback_stock,
        fallback_backlog,
        fallback_setup,
        fallback_startup,
    )
    best_objective = best_solution["objective_value"]

    if logger:
        logger.log_solution(best_objective, best_solution)

    # If no optimization time is available, return the feasible fallback.
    if time.monotonic() >= deadline:
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(best_solution, f)
        return

    model = gp.Model("multi_item_multi_machine_lot_sizing")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    q = model.addVars(
        triples, lb=0.0, vtype=GRB.CONTINUOUS, name="production"
    )
    stock = model.addVars(
        pairs, lb=0.0, vtype=GRB.CONTINUOUS, name="stock"
    )
    backlog = model.addVars(
        pairs, lb=0.0, vtype=GRB.CONTINUOUS, name="backlog"
    )
    setup = model.addVars(
        triples, vtype=GRB.BINARY, name="setup"
    )
    startup = model.addVars(
        triples, vtype=GRB.BINARY, name="startup"
    )

    # Inventory/backlog flow balance.
    for i in range(NI):
        for t in range(NT):
            previous_stock = initial_stock[i] if t == 0 else stock[i, t - 1]
            previous_backlog = (
                initial_backlog[i] if t == 0 else backlog[i, t - 1]
            )
            production_total = gp.quicksum(q[i, k, t] for k in range(NK))

            model.addConstr(
                previous_stock
                + production_total
                - previous_backlog
                == demand[i][t] + stock[i, t] - backlog[i, t]
            )

    for k in range(NK):
        for t in range(NT):
            # Each machine can be assigned to at most one item per period.
            model.addConstr(
                gp.quicksum(setup[i, k, t] for i in range(NI)) <= 1
            )

            # Production plus startup capacity consumption.
            model.addConstr(
                gp.quicksum(q[i, k, t] for i in range(NI))
                + startup_loss[k]
                * gp.quicksum(startup[i, k, t] for i in range(NI))
                <= capacity[k]
            )

            for i in range(NI):
                # Production is permitted only under the corresponding setup.
                model.addConstr(q[i, k, t] <= capacity[k] * setup[i, k, t])

                if t == 0:
                    # No setup is present before the first period.
                    model.addConstr(startup[i, k, t] == setup[i, k, t])
                else:
                    # Exact logical relation:
                    # startup = setup_now AND NOT setup_previous.
                    model.addConstr(
                        startup[i, k, t]
                        >= setup[i, k, t] - setup[i, k, t - 1]
                    )
                    model.addConstr(
                        startup[i, k, t] <= setup[i, k, t]
                    )
                    model.addConstr(
                        startup[i, k, t] <= 1 - setup[i, k, t - 1]
                    )

    objective = (
        setup_cost * gp.quicksum(setup[idx] for idx in triples)
        + startup_cost * gp.quicksum(startup[idx] for idx in triples)
        + gp.quicksum(
            holding_cost[i] * stock[i, t]
            + backlog_cost[i] * backlog[i, t]
            for i, t in pairs
        )
    )
    model.setObjective(objective, GRB.MINIMIZE)

    # Warm start with the no-production feasible plan.
    for idx in triples:
        q[idx].Start = 0.0
        setup[idx].Start = 0.0
        startup[idx].Start = 0.0
    for i, t in pairs:
        stock[i, t].Start = fallback_stock_map[(i, t)]
        backlog[i, t].Start = fallback_backlog_map[(i, t)]

    q_vars = [q[idx] for idx in triples]
    stock_vars = [stock[idx] for idx in pairs]
    backlog_vars = [backlog[idx] for idx in pairs]
    setup_vars = [setup[idx] for idx in triples]
    startup_vars = [startup[idx] for idx in triples]

    state = {
        "best_objective": best_objective,
        "best_solution": best_solution,
    }

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return

        try:
            production_values = cb_model.cbGetSolution(q_vars)
            stock_values = cb_model.cbGetSolution(stock_vars)
            backlog_values = cb_model.cbGetSolution(backlog_vars)
            setup_values = cb_model.cbGetSolution(setup_vars)
            startup_values = cb_model.cbGetSolution(startup_vars)

            solution = make_solution(
                production_values,
                stock_values,
                backlog_values,
                setup_values,
                startup_values,
            )
            value = solution["objective_value"]

            tolerance = 1e-7 * max(1.0, abs(state["best_objective"]))
            if value < state["best_objective"] - tolerance:
                state["best_objective"] = value
                state["best_solution"] = solution
                if logger:
                    logger.log_solution(value, solution)
        except Exception:
            # Logging or callback extraction must not prevent the optimizer
            # from returning its best feasible solution.
            return

    remaining = deadline - time.monotonic()
    if remaining > 0.0:
        model.Params.TimeLimit = max(0.001, remaining)
        model.optimize(incumbent_callback)

        if model.SolCount > 0:
            final_solution = make_solution(
                [v.X for v in q_vars],
                [v.X for v in stock_vars],
                [v.X for v in backlog_vars],
                [v.X for v in setup_vars],
                [v.X for v in startup_vars],
            )
            final_value = final_solution["objective_value"]

            tolerance = 1e-7 * max(1.0, abs(state["best_objective"]))
            if final_value < state["best_objective"] - tolerance:
                state["best_objective"] = final_value
                state["best_solution"] = final_solution
                if logger:
                    logger.log_solution(final_value, final_solution)
            elif final_value <= state["best_objective"] + tolerance:
                state["best_solution"] = final_solution
                state["best_objective"] = final_value

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(state["best_solution"], f)


if __name__ == "__main__":
    main()