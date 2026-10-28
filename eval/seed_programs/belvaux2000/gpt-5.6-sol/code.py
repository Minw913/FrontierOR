import argparse
import json
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger

    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path
        else None
    )

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    NI = int(data["parameters"]["NI"])
    NK = int(data["parameters"]["NK"])
    NT = int(data["parameters"]["NT"])

    setup_cost = float(data["costs"]["setup_cost_f"])
    startup_cost = float(data["costs"]["startup_cost_g"])
    holding_cost = [
        float(x) for x in data["costs"]["holding_cost_h"]
    ]
    backlog_cost = [
        float(x) for x in data["costs"]["backlogging_cost_e"]
    ]

    capacity = [
        float(x) for x in data["machine_data"]["capacity_C"]
    ]
    startup_loss = [
        float(x) for x in data["machine_data"]["startup_time_sigma"]
    ]

    demand = [
        [float(data["demand"][i][t]) for t in range(NT)]
        for i in range(NI)
    ]
    initial_stock = [
        float(x)
        for x in data["initial_conditions"]["initial_stock_s0"]
    ]
    initial_backlog = [
        float(x)
        for x in data["initial_conditions"]["initial_backlog_r0"]
    ]

    triples = [
        (i, k, t)
        for i in range(NI)
        for k in range(NK)
        for t in range(NT)
    ]
    pairs = [
        (i, t)
        for i in range(NI)
        for t in range(NT)
    ]

    def normalized_nonnegative(value):
        value = float(value)
        if abs(value) < 1e-8:
            return 0.0
        return max(0.0, value)

    def build_solution(
        production_values,
        stock_values,
        backlog_values,
        setup_values,
        startup_values,
    ):
        production_values = {
            idx: normalized_nonnegative(production_values.get(idx, 0.0))
            for idx in triples
        }
        stock_values = {
            idx: normalized_nonnegative(stock_values.get(idx, 0.0))
            for idx in pairs
        }
        backlog_values = {
            idx: normalized_nonnegative(backlog_values.get(idx, 0.0))
            for idx in pairs
        }
        setup_values = {
            idx: 1 if float(setup_values.get(idx, 0.0)) >= 0.5 else 0
            for idx in triples
        }
        startup_values = {
            idx: 1 if float(startup_values.get(idx, 0.0)) >= 0.5 else 0
            for idx in triples
        }

        objective = 0.0
        for i, k, t in triples:
            objective += setup_cost * setup_values[i, k, t]
            objective += startup_cost * startup_values[i, k, t]

        for i, t in pairs:
            objective += holding_cost[i] * stock_values[i, t]
            objective += backlog_cost[i] * backlog_values[i, t]

        if abs(objective) < 1e-8:
            objective = 0.0

        return {
            "objective_value": float(objective),
            "production": {
                f"{i}_{k}_{t}": production_values[i, k, t]
                for i, k, t in triples
            },
            "stock": {
                f"{i}_{t}": stock_values[i, t]
                for i, t in pairs
            },
            "backlog": {
                f"{i}_{t}": backlog_values[i, t]
                for i, t in pairs
            },
            "setup": {
                f"{i}_{k}_{t}": setup_values[i, k, t]
                for i, k, t in triples
            },
            "startup": {
                f"{i}_{k}_{t}": startup_values[i, k, t]
                for i, k, t in triples
            },
        }

    def write_solution(solution):
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(solution, f)

    def log_if_possible(solution):
        if logger:
            try:
                logger.log_solution(
                    solution["objective_value"], solution
                )
            except Exception:
                pass

    # An always-feasible no-production solution.
    zero_production = {idx: 0.0 for idx in triples}
    zero_setup = {idx: 0 for idx in triples}
    zero_startup = {idx: 0 for idx in triples}
    zero_stock = {}
    zero_backlog = {}

    for i in range(NI):
        net = initial_stock[i] - initial_backlog[i]
        for t in range(NT):
            net -= demand[i][t]
            zero_stock[i, t] = max(net, 0.0)
            zero_backlog[i, t] = max(-net, 0.0)

    best_solution = build_solution(
        zero_production,
        zero_stock,
        zero_backlog,
        zero_setup,
        zero_startup,
    )
    best_objective = best_solution["objective_value"]
    log_if_possible(best_solution)

    # Construct a second feasible incumbent without relying on Gurobi.
    # This also provides a useful answer when no usable Gurobi license exists.
    greedy_production = {idx: 0.0 for idx in triples}
    greedy_setup = {idx: 0 for idx in triples}
    greedy_startup = {idx: 0 for idx in triples}
    greedy_stock = {}
    greedy_backlog = {}

    net_balance = [
        initial_stock[i] - initial_backlog[i]
        for i in range(NI)
    ]
    previous_item = [None] * NK

    machine_order = sorted(
        range(NK),
        key=lambda k: (capacity[k], -startup_loss[k]),
        reverse=True,
    )

    for t in range(NT):
        for i in range(NI):
            net_balance[i] -= demand[i][t]

        current_item = [None] * NK

        for k in machine_order:
            best_item = None
            best_quantity = 0.0
            best_score = 0.0

            for i in range(NI):
                shortage = max(-net_balance[i], 0.0)
                if shortage <= 1e-9:
                    continue

                continuation = previous_item[k] == i
                available = capacity[k]
                if not continuation:
                    available -= startup_loss[k]

                if available <= 1e-9:
                    continue

                quantity = min(shortage, available)

                # A unit of backlog left now may incur penalties in several
                # remaining periods. This score favors expensive backlogs.
                periods_remaining = NT - t
                avoided_penalty = (
                    quantity * backlog_cost[i] * periods_remaining
                )
                fixed_cost = setup_cost
                if not continuation:
                    fixed_cost += startup_cost

                score = avoided_penalty - fixed_cost

                if (
                    score > best_score + 1e-9
                    or (
                        abs(score - best_score) <= 1e-9
                        and quantity > best_quantity + 1e-9
                    )
                ):
                    best_score = score
                    best_item = i
                    best_quantity = quantity

            if best_item is not None:
                i = best_item
                greedy_production[i, k, t] = best_quantity
                greedy_setup[i, k, t] = 1
                if previous_item[k] != i:
                    greedy_startup[i, k, t] = 1
                current_item[k] = i
                net_balance[i] += best_quantity

        previous_item = current_item

        for i in range(NI):
            greedy_stock[i, t] = max(net_balance[i], 0.0)
            greedy_backlog[i, t] = max(-net_balance[i], 0.0)

    greedy_solution = build_solution(
        greedy_production,
        greedy_stock,
        greedy_backlog,
        greedy_setup,
        greedy_startup,
    )

    if greedy_solution["objective_value"] < best_objective - 1e-7:
        best_solution = greedy_solution
        best_objective = greedy_solution["objective_value"]
        log_if_possible(best_solution)

    # Write a feasible solution immediately. It remains available even if
    # Gurobi initialization, licensing, model construction, or optimization
    # subsequently fails.
    write_solution(best_solution)

    if time.monotonic() >= deadline:
        return

    # Gurobi may be installed but unavailable because of a license or
    # environment initialization problem. The whole Gurobi section is guarded
    # so such a problem does not prevent a feasible output from being returned.
    try:
        import gurobipy as gp
        from gurobipy import GRB

        model = gp.Model("multi_item_multi_machine_lot_sizing")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            write_solution(best_solution)
            return
        model.Params.TimeLimit = max(0.001, remaining)

        production = model.addVars(
            triples,
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            name="production",
        )
        stock = model.addVars(
            pairs,
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            name="stock",
        )
        backlog = model.addVars(
            pairs,
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            name="backlog",
        )
        setup = model.addVars(
            triples,
            vtype=GRB.BINARY,
            name="setup",
        )
        startup = model.addVars(
            triples,
            vtype=GRB.BINARY,
            name="startup",
        )

        for i in range(NI):
            for t in range(NT):
                if t == 0:
                    previous_stock = initial_stock[i]
                    previous_backlog = initial_backlog[i]
                else:
                    previous_stock = stock[i, t - 1]
                    previous_backlog = backlog[i, t - 1]

                model.addConstr(
                    previous_stock
                    + gp.quicksum(
                        production[i, k, t] for k in range(NK)
                    )
                    - previous_backlog
                    == demand[i][t]
                    + stock[i, t]
                    - backlog[i, t],
                    name=f"balance_{i}_{t}",
                )

        for k in range(NK):
            for t in range(NT):
                model.addConstr(
                    gp.quicksum(
                        setup[i, k, t] for i in range(NI)
                    ) <= 1,
                    name=f"one_item_{k}_{t}",
                )

                model.addConstr(
                    gp.quicksum(
                        production[i, k, t] for i in range(NI)
                    )
                    + startup_loss[k]
                    * gp.quicksum(
                        startup[i, k, t] for i in range(NI)
                    )
                    <= capacity[k],
                    name=f"capacity_{k}_{t}",
                )

                for i in range(NI):
                    model.addConstr(
                        production[i, k, t]
                        <= capacity[k] * setup[i, k, t],
                        name=f"production_setup_{i}_{k}_{t}",
                    )

                    if t == 0:
                        model.addConstr(
                            startup[i, k, t] == setup[i, k, t],
                            name=f"initial_startup_{i}_{k}_{t}",
                        )
                    else:
                        model.addConstr(
                            startup[i, k, t]
                            >= setup[i, k, t]
                            - setup[i, k, t - 1],
                            name=f"startup_lb_{i}_{k}_{t}",
                        )
                        model.addConstr(
                            startup[i, k, t] <= setup[i, k, t],
                            name=f"startup_setup_ub_{i}_{k}_{t}",
                        )
                        model.addConstr(
                            startup[i, k, t]
                            <= 1 - setup[i, k, t - 1],
                            name=f"startup_previous_ub_{i}_{k}_{t}",
                        )

        model.setObjective(
            setup_cost
            * gp.quicksum(setup[idx] for idx in triples)
            + startup_cost
            * gp.quicksum(startup[idx] for idx in triples)
            + gp.quicksum(
                holding_cost[i] * stock[i, t]
                + backlog_cost[i] * backlog[i, t]
                for i, t in pairs
            ),
            GRB.MINIMIZE,
        )

        # Warm-start from the best heuristic solution.
        for i, k, t in triples:
            key = f"{i}_{k}_{t}"
            production[i, k, t].Start = best_solution["production"][key]
            setup[i, k, t].Start = best_solution["setup"][key]
            startup[i, k, t].Start = best_solution["startup"][key]

        for i, t in pairs:
            key = f"{i}_{t}"
            stock[i, t].Start = best_solution["stock"][key]
            backlog[i, t].Start = best_solution["backlog"][key]

        production_vars = [production[idx] for idx in triples]
        stock_vars = [stock[idx] for idx in pairs]
        backlog_vars = [backlog[idx] for idx in pairs]
        setup_vars = [setup[idx] for idx in triples]
        startup_vars = [startup[idx] for idx in triples]

        incumbent_state = {
            "objective": best_objective,
            "solution": best_solution,
        }

        def values_to_solution(
            production_list,
            stock_list,
            backlog_list,
            setup_list,
            startup_list,
        ):
            return build_solution(
                {
                    idx: production_list[pos]
                    for pos, idx in enumerate(triples)
                },
                {
                    idx: stock_list[pos]
                    for pos, idx in enumerate(pairs)
                },
                {
                    idx: backlog_list[pos]
                    for pos, idx in enumerate(pairs)
                },
                {
                    idx: setup_list[pos]
                    for pos, idx in enumerate(triples)
                },
                {
                    idx: startup_list[pos]
                    for pos, idx in enumerate(triples)
                },
            )

        def callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return

            try:
                candidate = values_to_solution(
                    cb_model.cbGetSolution(production_vars),
                    cb_model.cbGetSolution(stock_vars),
                    cb_model.cbGetSolution(backlog_vars),
                    cb_model.cbGetSolution(setup_vars),
                    cb_model.cbGetSolution(startup_vars),
                )

                value = candidate["objective_value"]
                tolerance = 1e-7 * max(
                    1.0, abs(incumbent_state["objective"])
                )

                if value < incumbent_state["objective"] - tolerance:
                    incumbent_state["objective"] = value
                    incumbent_state["solution"] = candidate
                    log_if_possible(candidate)
            except Exception:
                # Callback-side output/logging failures must not stop Gurobi.
                pass

        remaining = deadline - time.monotonic()
        if remaining > 0:
            model.Params.TimeLimit = max(0.001, remaining)
            model.optimize(callback)

        if model.SolCount > 0:
            candidate = values_to_solution(
                [var.X for var in production_vars],
                [var.X for var in stock_vars],
                [var.X for var in backlog_vars],
                [var.X for var in setup_vars],
                [var.X for var in startup_vars],
            )

            tolerance = 1e-7 * max(
                1.0, abs(incumbent_state["objective"])
            )
            if candidate["objective_value"] < (
                incumbent_state["objective"] - tolerance
            ):
                incumbent_state["objective"] = candidate["objective_value"]
                incumbent_state["solution"] = candidate
                log_if_possible(candidate)
            elif candidate["objective_value"] <= (
                incumbent_state["objective"] + tolerance
            ):
                incumbent_state["objective"] = candidate["objective_value"]
                incumbent_state["solution"] = candidate

        best_solution = incumbent_state["solution"]
        best_objective = incumbent_state["objective"]

        try:
            model.dispose()
        except Exception:
            pass

    except Exception:
        # This includes missing/invalid Gurobi licenses and failures during
        # model creation. The previously built heuristic solution is feasible.
        pass

    write_solution(best_solution)


if __name__ == "__main__":
    main()