import argparse
import json
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def clean_float(value, tolerance=1e-9):
    value = float(value)
    if abs(value) <= tolerance:
        return 0.0
    return value


def main():
    args = parse_args()
    start_time = time.monotonic()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    products = data["products"]
    facilities = data["facilities"]
    product_ids = [p["id"] for p in products]
    facility_ids = [f["id"] for f in facilities]

    product_by_id = {p["id"]: p for p in products}
    facility_by_id = {f["id"]: f for f in facilities}

    pfd_by_key = {}
    levels_by_key = {}
    level_by_id = {}

    for entry in data["product_facility_data"]:
        p = entry["product_id"]
        f = entry["facility_id"]
        pfd_by_key[p, f] = entry
        levels_by_key[p, f] = entry["production_levels"]
        for level in entry["production_levels"]:
            level_by_id[p, f, level["level_id"]] = level

    distributions_by_product = {p: [] for p in product_ids}
    all_distributions_by_product = {p: [] for p in product_ids}

    for product_distribution_data in data["products_distributions"]:
        p = product_distribution_data["product_id"]
        all_distributions_by_product[p] = product_distribution_data["distributions"]
        distributions_by_product[p] = [
            d
            for d in product_distribution_data["distributions"]
            if d.get("type") == "yield_and_demand"
            and d.get("level_combination") is not None
        ]

    distribution_by_key = {}
    for p in product_ids:
        for d in all_distributions_by_product[p]:
            distribution_by_key[p, d["distribution_id"]] = d

    model = gp.Model("endogenous_yield_production")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    delta = {}
    q = {}
    w = {}

    # Distribution selection variables.
    for p in product_ids:
        for d in distributions_by_product[p]:
            did = d["distribution_id"]
            delta[p, did] = model.addVar(vtype=GRB.BINARY)

    # Distribution-disaggregated allocations. This avoids big-M products
    # between allocation quantities and selected distributions.
    for p in product_ids:
        for d in distributions_by_product[p]:
            did = d["distribution_id"]
            combination = d["level_combination"]

            for facility_index, f in enumerate(facility_ids):
                level_id = combination[facility_index]
                level = level_by_id[p, f, level_id]
                upper = float(level["upper_bound"])

                q[p, did, f] = model.addVar(
                    lb=0.0,
                    ub=max(0.0, upper),
                    vtype=GRB.CONTINUOUS,
                )

            for scenario in d["scenarios"]:
                sid = scenario["scenario_id"]
                w[p, did, sid] = model.addVar(
                    lb=0.0,
                    vtype=GRB.CONTINUOUS,
                )

    model.update()

    # Exactly one endogenous yield-and-demand distribution per product.
    for p in product_ids:
        options = distributions_by_product[p]
        if not options:
            raise ValueError(
                f"Product {p} has no yield_and_demand distribution with a "
                "level combination."
            )
        model.addConstr(
            gp.quicksum(delta[p, d["distribution_id"]] for d in options) == 1
        )

    # Enforce the selected production-level interval.
    for p in product_ids:
        for d in distributions_by_product[p]:
            did = d["distribution_id"]
            combination = d["level_combination"]

            for facility_index, f in enumerate(facility_ids):
                level_id = combination[facility_index]
                level = level_by_id[p, f, level_id]
                lower = float(level["lower_bound"])
                upper = float(level["upper_bound"])

                model.addConstr(q[p, did, f] >= lower * delta[p, did])
                model.addConstr(q[p, did, f] <= upper * delta[p, did])

    # Facility capacities.
    for f in facility_ids:
        capacity = float(facility_by_id[f]["capacity"])
        model.addConstr(
            gp.quicksum(
                q[p, d["distribution_id"], f]
                for p in product_ids
                for d in distributions_by_product[p]
            )
            <= capacity
        )

    objective = gp.LinExpr()

    # Manufacturing costs.
    for p in product_ids:
        for d in distributions_by_product[p]:
            did = d["distribution_id"]
            for f in facility_ids:
                cost = float(pfd_by_key[p, f]["manufacturing_cost"])
                objective.add(q[p, did, f], -cost)

    # Scenario revenue:
    # discounted_price * inventory
    # + (sale_price - discounted_price) * full-price sales.
    for p in product_ids:
        product = product_by_id[p]
        sale_price = float(product["sale_price"])
        discounted_price = float(product["discounted_price"])
        premium = sale_price - discounted_price

        for d in distributions_by_product[p]:
            did = d["distribution_id"]

            for scenario in d["scenarios"]:
                sid = scenario["scenario_id"]
                probability = float(scenario["probability"])
                demand = max(0.0, float(scenario["demand"]))
                yields = scenario.get("yields", {})

                inventory_expression = gp.LinExpr()
                for f in facility_ids:
                    yield_value = float(yields.get(str(f), yields.get(f, 0.0)))
                    inventory_expression.add(q[p, did, f], yield_value)

                    if discounted_price != 0.0 and probability != 0.0:
                        objective.add(
                            q[p, did, f],
                            probability * discounted_price * yield_value,
                        )

                model.addConstr(w[p, did, sid] <= inventory_expression)
                model.addConstr(w[p, did, sid] <= demand * delta[p, did])

                if premium != 0.0 and probability != 0.0:
                    objective.add(w[p, did, sid], probability * premium)

    model.setObjective(objective, GRB.MAXIMIZE)

    # Construct a guaranteed zero-production MIP start when the corresponding
    # all-zero level combination is available.
    zero_distribution = {}
    for p in product_ids:
        chosen = None
        for d in distributions_by_product[p]:
            combination = d["level_combination"]
            is_zero = True
            for facility_index, f in enumerate(facility_ids):
                level = level_by_id[p, f, combination[facility_index]]
                if (
                    abs(float(level["lower_bound"])) > 1e-12
                    or abs(float(level["upper_bound"])) > 1e-12
                ):
                    is_zero = False
                    break
            if is_zero:
                chosen = d["distribution_id"]
                break
        zero_distribution[p] = chosen

    if all(zero_distribution[p] is not None for p in product_ids):
        for p in product_ids:
            chosen = zero_distribution[p]
            for d in distributions_by_product[p]:
                did = d["distribution_id"]
                delta[p, did].Start = 1.0 if did == chosen else 0.0
                for f in facility_ids:
                    q[p, did, f].Start = 0.0
                for scenario in d["scenarios"]:
                    w[p, did, scenario["scenario_id"]].Start = 0.0

    q_keys = list(q.keys())
    q_variables = [q[key] for key in q_keys]
    delta_keys = list(delta.keys())
    delta_variables = [delta[key] for key in delta_keys]

    logged_best = [-float("inf")]

    def build_output(q_values, delta_values):
        selected_distribution = {}

        for p in product_ids:
            options = distributions_by_product[p]
            selected = max(
                options,
                key=lambda d: delta_values.get(
                    (p, d["distribution_id"]), 0.0
                ),
            )
            selected_distribution[p] = selected["distribution_id"]

        x_output = {}
        y_output = {}
        delta_output = {}
        z_output = {}
        w_output = {}
        o_output = {}

        selected_allocations = {}

        for p in product_ids:
            selected_did = selected_distribution[p]
            selected_d = distribution_by_key[p, selected_did]
            combination = selected_d["level_combination"]

            for facility_index, f in enumerate(facility_ids):
                raw_value = q_values.get((p, selected_did, f), 0.0)
                raw_value = max(0.0, float(raw_value))

                level_id = combination[facility_index]
                level = level_by_id[p, f, level_id]
                lower = float(level["lower_bound"])
                upper = float(level["upper_bound"])

                # Remove tiny solver-bound violations in the serialized result.
                if raw_value < lower and lower - raw_value <= 1e-6:
                    raw_value = lower
                if raw_value > upper and raw_value - upper <= 1e-6:
                    raw_value = upper

                raw_value = clean_float(raw_value)
                selected_allocations[p, f] = raw_value
                x_output[f"{p}_{f}"] = raw_value

        for p in product_ids:
            selected_did = selected_distribution[p]
            selected_d = distribution_by_key[p, selected_did]
            selected_combination = selected_d["level_combination"]

            for facility_index, f in enumerate(facility_ids):
                active_level = selected_combination[facility_index]
                for level in levels_by_key[p, f]:
                    lid = level["level_id"]
                    y_output[f"{p}_{f}_{lid}"] = (
                        1 if lid == active_level else 0
                    )

            # Include every listed distribution key. Demand-only distributions
            # are not selectable in this endogenous-yield formulation and are
            # therefore reported as zero.
            for d in all_distributions_by_product[p]:
                did = d["distribution_id"]
                is_selected = (
                    d.get("type") == "yield_and_demand"
                    and did == selected_did
                )
                delta_output[f"{p}_{did}"] = 1 if is_selected else 0

                for scenario in d.get("scenarios", []):
                    sid = scenario["scenario_id"]

                    if is_selected:
                        yields = scenario.get("yields", {})
                        inventory = sum(
                            float(yields.get(str(f), yields.get(f, 0.0)))
                            * selected_allocations[p, f]
                            for f in facility_ids
                        )
                        inventory = max(0.0, inventory)
                        demand = max(0.0, float(scenario["demand"]))
                        full_price_sales = min(inventory, demand)
                        discounted_sales = max(
                            0.0, inventory - full_price_sales
                        )
                    else:
                        inventory = 0.0
                        full_price_sales = 0.0
                        discounted_sales = 0.0

                    z_output[f"{p}_{did}_{sid}"] = clean_float(inventory)
                    w_output[f"{p}_{did}_{sid}"] = clean_float(
                        full_price_sales
                    )
                    o_output[f"{p}_{did}_{sid}"] = clean_float(
                        discounted_sales
                    )

        manufacturing_cost = 0.0
        expected_revenue = 0.0

        for p in product_ids:
            product = product_by_id[p]
            sale_price = float(product["sale_price"])
            discounted_price = float(product["discounted_price"])
            selected_did = selected_distribution[p]
            selected_d = distribution_by_key[p, selected_did]

            for f in facility_ids:
                manufacturing_cost += (
                    float(pfd_by_key[p, f]["manufacturing_cost"])
                    * selected_allocations[p, f]
                )

            for scenario in selected_d["scenarios"]:
                sid = scenario["scenario_id"]
                probability = float(scenario["probability"])
                expected_revenue += probability * (
                    sale_price * w_output[f"{p}_{selected_did}_{sid}"]
                    + discounted_price
                    * o_output[f"{p}_{selected_did}_{sid}"]
                )

        objective_value = clean_float(
            expected_revenue - manufacturing_cost, tolerance=1e-8
        )

        solution_section = {
            "x": x_output,
            "y": y_output,
            "delta": delta_output,
            "z": z_output,
            "w": w_output,
            "o": o_output,
        }

        return {
            "objective_value": objective_value,
            "solution": solution_section,
        }

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL or logger is None:
            return

        try:
            incumbent_q = cb_model.cbGetSolution(q_variables)
            incumbent_delta = cb_model.cbGetSolution(delta_variables)

            q_values = dict(zip(q_keys, incumbent_q))
            delta_values = dict(zip(delta_keys, incumbent_delta))
            result = build_output(q_values, delta_values)
            objective_value = result["objective_value"]

            if objective_value > logged_best[0] + 1e-8:
                logger.log_solution(
                    objective_value,
                    result["solution"],
                )
                logged_best[0] = objective_value
        except Exception:
            # Logging must not interrupt optimization.
            pass

    elapsed = time.monotonic() - start_time
    remaining = max(0.01, float(args.time_limit) - elapsed)
    model.Params.TimeLimit = remaining

    model.optimize(incumbent_callback if logger is not None else None)

    if model.SolCount > 0:
        final_q_values = {key: var.X for key, var in q.items()}
        final_delta_values = {key: var.X for key, var in delta.items()}
        final_result = build_output(final_q_values, final_delta_values)
    elif all(zero_distribution[p] is not None for p in product_ids):
        # Guaranteed feasible fallback based on the required zero levels.
        fallback_q = {key: 0.0 for key in q_keys}
        fallback_delta = {}
        for key in delta_keys:
            p, did = key
            fallback_delta[key] = (
                1.0 if did == zero_distribution[p] else 0.0
            )
        final_result = build_output(fallback_q, fallback_delta)
    else:
        raise RuntimeError("No feasible solution was found within the time limit.")

    if logger is not None:
        final_objective = final_result["objective_value"]
        if final_objective > logged_best[0] + 1e-8:
            logger.log_solution(final_objective, final_result["solution"])
            logged_best[0] = final_objective

    solution_directory = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(solution_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()