import argparse
import json
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


TOL = 1e-7


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def write_json(path, data):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, separators=(",", ":"), allow_nan=False)
    os.replace(temporary_path, path)


def normalized_number(value):
    value = float(value)
    nearest = round(value)
    if abs(value - nearest) <= 1e-7:
        return int(nearest)
    return value


def main():
    start_time = time.monotonic()
    args = parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    n = int(data["dimensions"]["n"])
    m = int(data["dimensions"]["m"])
    p = int(data["dimensions"]["p"])

    demands = data["demands"]
    holding_costs = data["inventory_costs"]
    processing_time_data = data["processing_time"]
    setup_times = data["setup_times"]
    setup_costs = data["setup_costs"]
    capacities = data["machine_capacities"]
    max_production = data["max_production"]
    max_setups = data["max_setups_per_item"]
    minimum_lots = data["minimum_lot_sizes"]
    eligibility = data["machine_eligibility"]

    def processing_time(i, machine, period):
        if not isinstance(processing_time_data, list):
            return float(processing_time_data)
        value = processing_time_data
        try:
            if isinstance(value[0], list):
                if isinstance(value[0][0], list):
                    return float(value[i][machine][period])
                return float(value[i][machine])
            return float(value[i])
        except (IndexError, TypeError):
            return float(value[0])

    model = gp.Model("parallel_machine_lot_sizing_and_scheduling")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.MIPFocus = 1

    remaining_time = args.time_limit - (time.monotonic() - start_time)
    model.Params.TimeLimit = max(0.01, remaining_time)

    # x[i, machine, production period, demand period]
    x = {}
    production_terms = {
        (i, machine, period): []
        for i in range(n)
        for machine in range(m)
        for period in range(p)
    }
    demand_terms = {
        (i, demand_period): []
        for i in range(n)
        for demand_period in range(p)
    }

    objective = gp.LinExpr()

    for i in range(n):
        for production_period in range(p):
            period_limit = max(0.0, float(max_production[i][production_period]))
            if period_limit <= TOL:
                continue

            for machine in range(m):
                allowed = (
                    int(eligibility[i][machine]) != 0
                    and int(max_setups[i][machine][production_period]) > 0
                )
                if not allowed:
                    continue

                for demand_period in range(production_period, p):
                    demand = float(demands[i][demand_period])
                    if demand <= TOL:
                        continue

                    upper_bound = min(demand, period_limit)
                    var = model.addVar(
                        lb=0.0,
                        ub=upper_bound,
                        vtype=GRB.CONTINUOUS,
                        name=f"x_{i}_{machine}_{production_period}_{demand_period}",
                    )
                    x[i, machine, production_period, demand_period] = var
                    production_terms[i, machine, production_period].append(var)
                    demand_terms[i, demand_period].append(var)

                    holding_coefficient = (
                        float(holding_costs[i])
                        * (demand_period - production_period)
                    )
                    if holding_coefficient:
                        objective.addTerms(holding_coefficient, var)

    # z is the setup state at the beginning of a period.
    z = {}
    for i in range(n):
        for machine in range(m):
            for period in range(p):
                ub = 1.0 if int(eligibility[i][machine]) != 0 else 0.0
                z[i, machine, period] = model.addVar(
                    lb=0.0,
                    ub=ub,
                    vtype=GRB.BINARY,
                    name=f"z_{i}_{machine}_{period}",
                )

    # w indicates that an item appears in the within-period route.
    w = {}
    for i in range(n):
        for machine in range(m):
            for period in range(p):
                ub = 1.0 if int(eligibility[i][machine]) != 0 else 0.0
                w[i, machine, period] = model.addVar(
                    lb=0.0,
                    ub=ub,
                    vtype=GRB.BINARY,
                    name=f"w_{i}_{machine}_{period}",
                )

    # y is the number of directed changeovers.
    y = {}
    for source in range(n):
        for target in range(n):
            if source == target:
                continue
            for machine in range(m):
                if (
                    int(eligibility[source][machine]) == 0
                    or int(eligibility[target][machine]) == 0
                ):
                    continue
                for period in range(p):
                    upper_bound = int(max_setups[target][machine][period])
                    if upper_bound <= 0:
                        continue
                    var = model.addVar(
                        lb=0.0,
                        ub=upper_bound,
                        vtype=GRB.INTEGER,
                        name=f"y_{source}_{target}_{machine}_{period}",
                    )
                    y[source, target, machine, period] = var
                    cost = float(setup_costs[source][target][machine])
                    if cost:
                        objective.addTerms(cost, var)

    # Final state after the last period.
    final_state = {}
    for i in range(n):
        for machine in range(m):
            ub = 1.0 if int(eligibility[i][machine]) != 0 else 0.0
            final_state[i, machine] = model.addVar(
                lb=0.0,
                ub=ub,
                vtype=GRB.BINARY,
                name=f"end_{i}_{machine}",
            )

    # Connectivity flow. It prevents disconnected setup cycles.
    flow = {}
    root_flow = {}
    for machine in range(m):
        for period in range(p):
            for i in range(n):
                if int(eligibility[i][machine]) != 0:
                    root_flow[i, machine, period] = model.addVar(
                        lb=0.0,
                        ub=n,
                        vtype=GRB.CONTINUOUS,
                        name=f"rootflow_{i}_{machine}_{period}",
                    )

            for source in range(n):
                for target in range(n):
                    if source == target:
                        continue
                    if (source, target, machine, period) not in y:
                        continue
                    flow[source, target, machine, period] = model.addVar(
                        lb=0.0,
                        ub=n,
                        vtype=GRB.CONTINUOUS,
                        name=f"flow_{source}_{target}_{machine}_{period}",
                    )

    model.update()

    # Every demand must be supplied.
    for i in range(n):
        for demand_period in range(p):
            required = float(demands[i][demand_period])
            terms = demand_terms[i, demand_period]
            model.addConstr(
                gp.quicksum(terms) == required,
                name=f"demand_{i}_{demand_period}",
            )

    # Per-item, per-period production limits.
    for i in range(n):
        for period in range(p):
            terms = []
            for machine in range(m):
                terms.extend(production_terms[i, machine, period])
            model.addConstr(
                gp.quicksum(terms) <= float(max_production[i][period]),
                name=f"maxprod_{i}_{period}",
            )

    # One beginning state per machine and period, and one final state.
    for machine in range(m):
        for period in range(p):
            model.addConstr(
                gp.quicksum(z[i, machine, period] for i in range(n)) == 1,
                name=f"one_start_{machine}_{period}",
            )
        model.addConstr(
            gp.quicksum(final_state[i, machine] for i in range(n)) == 1,
            name=f"one_final_{machine}",
        )

    for machine in range(m):
        for period in range(p):
            total_setup_cap = sum(
                max(0, int(max_setups[i][machine][period]))
                for i in range(n)
                if int(eligibility[i][machine]) != 0
            )
            route_big_m = max(1, total_setup_cap)

            for i in range(n):
                incoming = gp.quicksum(
                    y[j, i, machine, period]
                    for j in range(n)
                    if (j, i, machine, period) in y
                )
                outgoing = gp.quicksum(
                    y[i, j, machine, period]
                    for j in range(n)
                    if (i, j, machine, period) in y
                )

                if period + 1 < p:
                    ending = z[i, machine, period + 1]
                else:
                    ending = final_state[i, machine]

                # Euler-trail degree balance between start and end states.
                model.addConstr(
                    z[i, machine, period] + incoming
                    == ending + outgoing,
                    name=f"route_balance_{i}_{machine}_{period}",
                )

                model.addConstr(
                    z[i, machine, period] <= w[i, machine, period],
                    name=f"start_visited_{i}_{machine}_{period}",
                )
                model.addConstr(
                    w[i, machine, period]
                    <= z[i, machine, period] + incoming,
                    name=f"visited_entered_{i}_{machine}_{period}",
                )
                model.addConstr(
                    outgoing <= route_big_m * w[i, machine, period],
                    name=f"outgoing_visited_{i}_{machine}_{period}",
                )

                setup_limit = max(0, int(max_setups[i][machine][period]))
                model.addConstr(
                    incoming <= setup_limit,
                    name=f"setup_limit_{i}_{machine}_{period}",
                )

                quantity = gp.quicksum(
                    production_terms[i, machine, period]
                )
                future_demand = sum(
                    float(demands[i][k]) for k in range(period, p)
                )
                quantity_upper_bound = min(
                    future_demand,
                    max(0.0, float(max_production[i][period])),
                )

                allowed = (
                    int(eligibility[i][machine]) != 0
                    and setup_limit > 0
                )
                if allowed and quantity_upper_bound > TOL:
                    model.addConstr(
                        quantity
                        <= quantity_upper_bound * w[i, machine, period],
                        name=f"production_activation_{i}_{machine}_{period}",
                    )
                else:
                    model.addConstr(
                        quantity == 0,
                        name=f"production_forbidden_{i}_{machine}_{period}",
                    )

                # Every setup into an item accounts for a minimum lot.
                lot_size = float(minimum_lots[i])
                if lot_size > 0:
                    model.addConstr(
                        quantity >= lot_size * incoming,
                        name=f"minimum_lot_{i}_{machine}_{period}",
                    )

            # Single-commodity connectivity from the beginning state.
            model.addConstr(
                gp.quicksum(
                    root_flow[i, machine, period]
                    for i in range(n)
                    if (i, machine, period) in root_flow
                )
                == gp.quicksum(w[i, machine, period] for i in range(n)),
                name=f"root_supply_{machine}_{period}",
            )

            for i in range(n):
                if (i, machine, period) not in root_flow:
                    continue

                model.addConstr(
                    root_flow[i, machine, period]
                    <= n * z[i, machine, period],
                    name=f"root_select_{i}_{machine}_{period}",
                )

                incoming_flow = gp.quicksum(
                    flow[j, i, machine, period]
                    for j in range(n)
                    if (j, i, machine, period) in flow
                )
                outgoing_flow = gp.quicksum(
                    flow[i, j, machine, period]
                    for j in range(n)
                    if (i, j, machine, period) in flow
                )
                model.addConstr(
                    root_flow[i, machine, period]
                    + incoming_flow
                    - outgoing_flow
                    == w[i, machine, period],
                    name=f"connectivity_balance_{i}_{machine}_{period}",
                )

            for source in range(n):
                for target in range(n):
                    key = (source, target, machine, period)
                    if key in flow:
                        model.addConstr(
                            flow[key] <= n * y[key],
                            name=f"flow_arc_{source}_{target}_{machine}_{period}",
                        )

            # Machine time capacity.
            processing_expr = gp.LinExpr()
            for i in range(n):
                pt = processing_time(i, machine, period)
                if pt:
                    for var in production_terms[i, machine, period]:
                        processing_expr.addTerms(pt, var)

            setup_expr = gp.LinExpr()
            for source in range(n):
                for target in range(n):
                    key = (source, target, machine, period)
                    if key in y:
                        st = float(setup_times[source][target][machine])
                        if st:
                            setup_expr.addTerms(st, y[key])

            model.addConstr(
                processing_expr + setup_expr
                <= float(capacities[machine][period]),
                name=f"capacity_{machine}_{period}",
            )

    model.setObjective(objective, GRB.MINIMIZE)

    x_records = [
        (
            var,
            f"x_{i}_{machine}_{production_period}_{demand_period}",
        )
        for (i, machine, production_period, demand_period), var in x.items()
    ]
    y_records = [
        (
            var,
            f"y_{source}_{target}_{machine}_{period}",
        )
        for (source, target, machine, period), var in y.items()
    ]
    z_records = [
        (
            var,
            f"z_{i}_{machine}_{period}",
        )
        for (i, machine, period), var in z.items()
    ]

    all_output_vars = (
        [record[0] for record in x_records]
        + [record[0] for record in y_records]
        + [record[0] for record in z_records]
    )
    x_count = len(x_records)
    y_count = len(y_records)

    def construct_solution(values, objective_value):
        production = {}
        setups_output = {}
        carryover = {}

        for idx, (_, key) in enumerate(x_records):
            value = values[idx]
            if value > TOL:
                production[key] = normalized_number(value)

        offset = x_count
        for idx, (_, key) in enumerate(y_records):
            value = values[offset + idx]
            if value > 0.5:
                setups_output[key] = int(round(value))

        offset += y_count
        for idx, (_, key) in enumerate(z_records):
            value = values[offset + idx]
            if value > 0.5:
                carryover[key] = 1

        return {
            "objective_value": float(objective_value),
            "production": production,
            "setups": setups_output,
            "carryover": carryover,
        }

    def incumbent_callback(callback_model, where):
        if where != GRB.Callback.MIPSOL or logger is None:
            return
        try:
            objective_value = callback_model.cbGet(
                GRB.Callback.MIPSOL_OBJ
            )
            values = callback_model.cbGetSolution(all_output_vars)
            solution = construct_solution(values, objective_value)
            logger.log_solution(float(objective_value), solution)
        except Exception:
            # Logging must not interrupt optimization.
            pass

    remaining_time = args.time_limit - (time.monotonic() - start_time)
    model.Params.TimeLimit = max(0.01, remaining_time)
    model.optimize(incumbent_callback)

    if model.SolCount > 0:
        final_values = [var.X for var in all_output_vars]
        solution = construct_solution(final_values, model.ObjVal)
        if logger:
            try:
                logger.log_solution(float(model.ObjVal), solution)
            except Exception:
                pass
    else:
        # This is used only if the supplied instance has no feasible incumbent
        # within the allowed runtime.
        solution = {
            "objective_value": 1e100,
            "production": {},
            "setups": {},
            "carryover": {},
        }

    write_json(args.solution_path, solution)


if __name__ == "__main__":
    main()