import argparse
import json
import os
import time

import gurobipy as gp
from gurobipy import GRB, quicksum

from solution_logger import SolutionLogger


def same_id(a, b):
    """Robust comparison for identifiers represented either as ints or strings."""
    try:
        return int(a) == int(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


def lookup_id(mapping, key, default=None):
    if not isinstance(mapping, dict):
        return default
    if key in mapping:
        return mapping[key]
    return mapping.get(str(key), default)


def numeric(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.time()

    logger = None
    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="maximize")

    with open(args.instance_path, "r", encoding="utf-8") as handle:
        instance = json.load(handle)

    products = instance.get("products", [])
    facilities = instance.get("facilities", [])
    product_ids = [p["id"] for p in products]
    facility_ids = [f["id"] for f in facilities]

    product_info = {}
    facility_capacity = {}

    for product in products:
        product_info[product["id"]] = {
            "sale_price": numeric(product.get("sale_price")),
            "discounted_price": numeric(product.get("discounted_price")),
            "expected_demand": numeric(product.get("expected_demand")),
        }

    for facility in facilities:
        facility_capacity[facility["id"]] = numeric(facility.get("capacity"))

    # Index product/facility production data.
    pf_data = {}
    level_ids = {}
    bounds = {}
    costs = {}
    level_means = {}

    for entry in instance.get("product_facility_data", []):
        p_id = entry["product_id"]
        f_id = entry["facility_id"]
        pair = (p_id, f_id)
        levels = entry.get("production_levels", [])

        pf_data[pair] = entry
        level_ids[pair] = [level["level_id"] for level in levels]
        costs[pair] = numeric(entry.get("manufacturing_cost"))

        for level in levels:
            l_id = level["level_id"]
            bounds[(p_id, f_id, l_id)] = (
                numeric(level.get("lower_bound")),
                numeric(level.get("upper_bound")),
            )
            level_means[(p_id, f_id, l_id)] = numeric(level.get("yield_mean"))

    # Index all distributions available for each product.
    distributions_by_product = {}
    for product in product_ids:
        found = []
        for block in instance.get("products_distributions", []):
            if block.get("product_id") == product:
                found = block.get("distributions", [])
                break
        distributions_by_product[product] = found

    # ------------------------------------------------------------------
    # Helper: choose an all-zero feasible incumbent. The problem schema
    # guarantees that a zero production level exists for every pair and
    # that level combinations enumerate all possible facility choices.
    # ------------------------------------------------------------------
    fallback_levels = {}
    for pair in pf_data:
        candidates = level_ids[pair]
        zero_level = None
        for level_id in candidates:
            lower, upper = bounds[(pair[0], pair[1], level_id)]
            if abs(lower) <= 1e-9 and abs(upper) <= 1e-9:
                zero_level = level_id
                break
        if zero_level is None:
            # Defensive fallback for malformed data; schema promises zero.
            zero_level = min(
                candidates,
                key=lambda lid: abs(bounds[(pair[0], pair[1], lid)][0])
                + abs(bounds[(pair[0], pair[1], lid)][1]),
            )
        fallback_levels[pair] = zero_level

    def distribution_matches(product, distribution, selected_levels):
        mapping = distribution.get("level_combination")
        if not isinstance(mapping, list):
            return False
        if len(mapping) != len(facility_ids):
            return False
        return all(
            same_id(
                mapping[idx],
                selected_levels[(product, facility_ids[idx])],
            )
            for idx in range(len(facility_ids))
        )

    fallback_distributions = {}
    for product in product_ids:
        distributions = distributions_by_product.get(product, [])
        selected = None

        # First prefer a fully specified matching yield-and-demand mapping.
        for distribution in distributions:
            if distribution_matches(product, distribution, fallback_levels):
                selected = distribution.get("distribution_id")
                break

        # A demand-only distribution with a null mapping is unrestricted.
        if selected is None:
            for distribution in distributions:
                if distribution.get("level_combination") is None:
                    selected = distribution.get("distribution_id")
                    break

        if selected is None and distributions:
            selected = distributions[0].get("distribution_id")

        fallback_distributions[product] = selected

    # ------------------------------------------------------------------
    # Construct a complete output payload from selected levels,
    # selected distributions, and an allocation lookup.
    # ------------------------------------------------------------------
    def assemble_solution(selected_levels, selected_distributions, x_lookup):
        payload = {"x": {}, "y": {}, "delta": {}, "z": {}, "w": {}, "o": {}}
        objective = 0.0

        for product in product_ids:
            for facility in facility_ids:
                pair = (product, facility)
                allocation = max(0.0, x_lookup(pair))
                payload["x"][f"{product}_{facility}"] = float(allocation)
                objective -= costs[pair] * allocation

                selected_level = selected_levels.get(pair)
                for level_id in level_ids[pair]:
                    value = 1 if level_id == selected_level else 0
                    payload["y"][f"{product}_{facility}_{level_id}"] = value

            selected_distribution = selected_distributions.get(product)
            distributions = distributions_by_product.get(product, [])

            for distribution in distributions:
                dist_id = distribution.get("distribution_id")
                is_selected = dist_id == selected_distribution
                payload["delta"][f"{product}_{dist_id}"] = 1 if is_selected else 0

                for scenario in distribution.get("scenarios", []):
                    scen_id = scenario.get("scenario_id")
                    key = f"{product}_{dist_id}_{scen_id}"
                    payload["z"][key] = 0.0
                    payload["w"][key] = 0.0
                    payload["o"][key] = 0.0

                    if is_selected:
                        yields = scenario.get("yields")
                        availability = 0.0

                        if yields is not None:
                            # Joint yield-and-demand scenario.
                            for facility in facility_ids:
                                coefficient = numeric(
                                    lookup_id(yields, facility, 0.0)
                                )
                                availability += coefficient * max(
                                    0.0, x_lookup((product, facility))
                                )
                        else:
                            # Demand-only distributions have no explicit yield
                            # realization. Use the deterministic mean yield of
                            # the selected production level at each facility.
                            for facility in facility_ids:
                                level_id = selected_levels.get(
                                    (product, facility)
                                )
                                mean_yield = level_means.get(
                                    (product, facility, level_id), 0.0
                                )
                                availability += mean_yield * max(
                                    0.0, x_lookup((product, facility))
                                )

                        availability = max(0.0, availability)
                        demand = numeric(scenario.get("demand"), 0.0)
                        full_sales = min(max(0.0, demand), availability)
                        discounted_sales = max(0.0, availability - full_sales)

                        probability = numeric(scenario.get("probability"), 0.0)
                        price = product_info[product]["sale_price"]
                        discount = product_info[product]["discounted_price"]

                        payload["z"][key] = float(availability)
                        payload["w"][key] = float(full_sales)
                        payload["o"][key] = float(discounted_sales)

                        objective += probability * (
                            price * full_sales + discount * discounted_sales
                        )

        return float(objective), payload

    fallback_x_lookup = lambda pair: 0.0
    best_objective, best_payload = assemble_solution(
        fallback_levels, fallback_distributions, fallback_x_lookup
    )

    if logger:
        logger.log_solution(best_objective, best_payload)

    # ------------------------------------------------------------------
    # Build and solve a MILP. Disaggregate per-level allocation so bounds
    # tie allocation directly to the chosen binary production level.
    # ------------------------------------------------------------------
    model = gp.Model("endogenous_inventory_allocation")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0

    y_vars = {}
    a_vars = {}
    x_vars = {}
    delta_vars = {}
    z_vars = {}
    w_vars = {}
    o_vars = {}

    for product in product_ids:
        for facility in facility_ids:
            pair = (product, facility)
            cap_limited = min(
                max(0.0, facility_capacity.get(facility, 0.0)),
                max(
                    max(bounds[(product, facility, lid)][1] for lid in level_ids[pair]),
                    0.0,
                ),
            )
            x_vars[pair] = model.addVar(
                lb=0.0, ub=max(0.0, cap_limited), vtype=GRB.CONTINUOUS
            )
            for level_id in level_ids[pair]:
                y_vars[(product, facility, level_id)] = model.addVar(
                    lb=0.0, ub=1.0, vtype=GRB.BINARY
                )
                upper = max(0.0, bounds[(product, facility, level_id)][1])
                a_vars[(product, facility, level_id)] = model.addVar(
                    lb=0.0, ub=upper, vtype=GRB.CONTINUOUS
                )

    for product in product_ids:
        for distribution in distributions_by_product.get(product, []):
            dist_id = distribution.get("distribution_id")
            delta_vars[(product, dist_id)] = model.addVar(
                lb=0.0, ub=1.0, vtype=GRB.BINARY
            )

    # Base allocation upper bound for a single product at a facility.
    base_allocation = {}
    for product in product_ids:
        for facility in facility_ids:
            pair = (product, facility)
            max_upper = max(
                max(bounds[(product, facility, lid)][1] for lid in level_ids[pair]),
                0.0,
            )
            base_allocation[pair] = min(
                max(0.0, facility_capacity.get(facility, 0.0)),
                max(0.0, max_upper),
            )

    # Exactly one production level per product and facility; allocation
    # is disaggregated so only the selected level can carry allocation.
    for product in product_ids:
        for facility in facility_ids:
            pair = (product, facility)
            ids = level_ids[pair]

            model.addConstr(
                quicksum(
                    y_vars[(product, facility, level_id)] for level_id in ids
                )
                == 1.0
            )
            model.addConstr(
                x_vars[pair]
                == quicksum(
                    a_vars[(product, facility, level_id)] for level_id in ids
                )
            )

            for level_id in ids:
                lower, upper = bounds[(product, facility, level_id)]
                y_var = y_vars[(product, facility, level_id)]
                a_var = a_vars[(product, facility, level_id)]
                model.addConstr(a_var >= lower * y_var)
                model.addConstr(a_var <= upper * y_var)

    # Facility capacities couple allocations across products.
    for facility in facility_ids:
        model.addConstr(
            quicksum(
                x_vars[(product, facility)] for product in product_ids
            )
            <= max(0.0, facility_capacity.get(facility, 0.0))
        )

    # Exactly one distribution per product.
    for product in product_ids:
        dist_ids = [
            item.get("distribution_id")
            for item in distributions_by_product.get(product, [])
        ]
        if dist_ids:
            model.addConstr(
                quicksum(delta_vars[(product, dist_id)] for dist_id in dist_ids)
                == 1.0
            )

    # Distribution-to-production-level implications.
    for product in product_ids:
        for distribution in distributions_by_product.get(product, []):
            dist_id = distribution.get("distribution_id")
            mapping = distribution.get("level_combination")
            if not isinstance(mapping, list):
                continue
            limit = min(len(mapping), len(facility_ids))
            for idx in range(limit):
                facility = facility_ids[idx]
                mapped_level = mapping[idx]
                key = (product, facility, mapped_level)
                if key in y_vars:
                    model.addConstr(
                        delta_vars[(product, dist_id)] <= y_vars[key]
                    )

    # Inventory and recourse variables. Only the selected distribution
    # has nonzero recourse; conditional inventory equality is imposed
    # by two big-M constraints.
    inventory_upper = {}
    for product in product_ids:
        for distribution in distributions_by_product.get(product, []):
            dist_id = distribution.get("distribution_id")
            for scenario in distribution.get("scenarios", []):
                scen_id = scenario.get("scenario_id")
                upper = 0.0
                yields = scenario.get("yields")

                if yields is not None:
                    for facility in facility_ids:
                        coefficient = numeric(
                            lookup_id(yields, facility, 0.0)
                        )
                        upper += abs(coefficient) * base_allocation[
                            (product, facility)
                        ]
                else:
                    for facility in facility_ids:
                        max_mean = max(
                            max(
                                abs(
                                    level_means.get(
                                        (product, facility, level_id), 0.0
                                    )
                                )
                                for level_id in level_ids[
                                    (product, facility)
                                ]
                            ),
                            0.0,
                        )
                        upper += max_mean * base_allocation[
                            (product, facility)
                        ]

                key = (product, dist_id, scen_id)
                inventory_upper[key] = max(0.0, upper)
                demand = max(0.0, numeric(scenario.get("demand"), 0.0))

                z_vars[key] = model.addVar(
                    lb=0.0, ub=max(0.0, upper), vtype=GRB.CONTINUOUS
                )
                w_vars[key] = model.addVar(
                    lb=0.0,
                    ub=min(max(0.0, upper), demand),
                    vtype=GRB.CONTINUOUS,
                )
                o_vars[key] = model.addVar(
                    lb=0.0, ub=max(0.0, upper), vtype=GRB.CONTINUOUS
                )

                delta_var = delta_vars[(product, dist_id)]
                z_var = z_vars[key]

                if yields is not None:
                    inventory_expression = quicksum(
                        numeric(lookup_id(yields, facility, 0.0))
                        * x_vars[(product, facility)]
                        for facility in facility_ids
                    )
                else:
                    inventory_expression = quicksum(
                        level_means.get(
                            (product, facility, level_id), 0.0
                        )
                        * a_vars[(product, facility, level_id)]
                        for facility in facility_ids
                        for level_id in level_ids[(product, facility)]
                    )

                upper_bound = max(0.0, upper)
                model.addConstr(
                    z_var - inventory_expression
                    <= upper_bound - upper_bound * delta_var
                )
                model.addConstr(
                    z_var - inventory_expression
                    >= -upper_bound + upper_bound * delta_var
                )
                model.addConstr(z_var <= upper_bound * delta_var)

                # Scenario recourse: all available inventory is sold, and
                # full-price sales are limited by realized demand.
                model.addConstr(
                    w_vars[key] <= max(
                        0.0, numeric(scenario.get("demand"), 0.0)
                    )
                )
                model.addConstr(w_vars[key] <= z_var)
                model.addConstr(z_var == w_vars[key] + o_vars[key])

    objective_expression = -quicksum(
        costs[(product, facility)] * x_vars[(product, facility)]
        for product in product_ids
        for facility in facility_ids
    )

    for product in product_ids:
        sale_price = product_info[product]["sale_price"]
        discounted_price = product_info[product]["discounted_price"]

        for distribution in distributions_by_product.get(product, []):
            dist_id = distribution.get("distribution_id")
            for scenario in distribution.get("scenarios", []):
                scen_id = scenario.get("scenario_id")
                key = (product, dist_id, scen_id)
                probability = numeric(scenario.get("probability"), 0.0)
                objective_expression += probability * (
                    sale_price * w_vars[key]
                    + discounted_price * o_vars[key]
                )

    model.setObjective(objective_expression, GRB.MAXIMIZE)

    # MIP start corresponding to the logged zero-allocation incumbent.
    for key, variable in y_vars.items():
        variable.Start = 1 if fallback_levels.get((key[0], key[1])) == key[2] else 0
    for key, variable in delta_vars.items():
        variable.Start = 1 if fallback_distributions.get(key[0]) == key[1] else 0
    for variable in list(x_vars.values()) + list(a_vars.values()):
        variable.Start = 0.0
    for variable in list(z_vars.values()) + list(w_vars.values()) + list(o_vars.values()):
        variable.Start = 0.0

    all_variables = (
        list(x_vars.values())
        + list(a_vars.values())
        + list(y_vars.values())
        + list(delta_vars.values())
        + list(z_vars.values())
        + list(w_vars.values())
        + list(o_vars.values())
    )

    convergence_state = {
        "best_objective": best_objective,
        "best_payload": best_payload,
    }
    improvement_tolerance = 1e-7

    def solution_from_raw(raw_values):
        selected_levels = {}
        selected_distributions = {}
        allocation_lookup = {}

        for product in product_ids:
            for facility in facility_ids:
                pair = (product, facility)
                raw_x = raw_values.get(x_vars[pair], 0.0)
                allocation_lookup[pair] = max(0.0, raw_x)

                ids_for_pair = level_ids[pair]
                selected_levels[pair] = max(
                    ids_for_pair,
                    key=lambda level_id: raw_values.get(
                        y_vars[(product, facility, level_id)], 0.0
                    ),
                )

            distributions = distributions_by_product.get(product, [])
            dist_ids = [item.get("distribution_id") for item in distributions]
            if dist_ids:
                selected_distributions[product] = max(
                    dist_ids,
                    key=lambda dist_id: raw_values.get(
                        delta_vars[(product, dist_id)], 0.0
                    ),
                )
            else:
                selected_distributions[product] = None

        return assemble_solution(
            selected_levels, selected_distributions, allocation_lookup
        )

    def incumbent_callback(current_model, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            values = dict(
                zip(
                    all_variables,
                    current_model.cbGetSolution(all_variables),
                )
            )
            objective, payload = solution_from_raw(values)
        except gp.GurobiError:
            return

        if objective > convergence_state["best_objective"] + improvement_tolerance:
            convergence_state["best_objective"] = objective
            convergence_state["best_payload"] = payload
            if logger:
                logger.log_solution(objective, payload)

    remaining_time = max(
        0.0, float(args.time_limit) - (time.time() - start_time)
    )
    model.Params.TimeLimit = remaining_time

    if remaining_time > 0.0 and all_variables:
        try:
            model.optimize(incumbent_callback)
        except gp.GurobiError:
            # If Gurobi is unable to run to completion, retain the already
            # constructed feasible zero-allocation solution.
            pass

        if model.SolCount > 0:
            try:
                raw_values = dict(
                    zip(
                        all_variables,
                        model.getAttr("X", all_variables),
                    )
                )
                final_objective, final_payload = solution_from_raw(raw_values)
                if (
                    final_objective
                    > convergence_state["best_objective"]
                    + improvement_tolerance
                ):
                    convergence_state["best_objective"] = final_objective
                    convergence_state["best_payload"] = final_payload
                    if logger:
                        logger.log_solution(final_objective, final_payload)
            except (gp.GurobiError, ValueError):
                pass

    output = {
        "objective_value": float(convergence_state["best_objective"]),
        "solution": convergence_state["best_payload"],
    }

    destination_directory = os.path.dirname(args.solution_path)
    if destination_directory:
        os.makedirs(destination_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle)


if __name__ == "__main__":
    main()