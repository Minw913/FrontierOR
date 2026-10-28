import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    # Parse instance data
    products = {p['id']: p for p in instance['products']}
    facilities = {f['id']: f for f in instance['facilities']}
    product_ids = sorted(products.keys())
    facility_ids = sorted(facilities.keys())

    pf_data = {}
    for entry in instance['product_facility_data']:
        pf_data[(entry['product_id'], entry['facility_id'])] = entry

    prod_dists = {}
    for entry in instance['products_distributions']:
        prod_dists[entry['product_id']] = entry['distributions']

    # Build Gurobi model
    model = gp.Model("endogenous_uncertainty_production")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', 0)

    # Decision variables
    # x[p,f]: raw materials allocated to product p at facility f
    x = {}
    for p in product_ids:
        for f in facility_ids:
            x[p, f] = model.addVar(lb=0.0, name=f"x_{p}_{f}")

    # y[p,f,l]: binary - production level l selected for product p at facility f
    y = {}
    for p in product_ids:
        for f in facility_ids:
            for level in pf_data[(p, f)]['production_levels']:
                l = level['level_id']
                y[p, f, l] = model.addVar(vtype=GRB.BINARY, name=f"y_{p}_{f}_{l}")

    # delta[p,d]: binary - distribution d selected for product p
    delta = {}
    for p in product_ids:
        for dist in prod_dists[p]:
            d = dist['distribution_id']
            delta[p, d] = model.addVar(vtype=GRB.BINARY, name=f"delta_{p}_{d}")

    # u[p,d,f]: linearization variable = delta[p,d] * x[p,f]
    u = {}
    for p in product_ids:
        for dist in prod_dists[p]:
            d = dist['distribution_id']
            for f in facility_ids:
                u[p, d, f] = model.addVar(lb=0.0, name=f"u_{p}_{d}_{f}")

    # w[p,d,s]: amount sold at full price
    # o[p,d,s]: amount sold at discounted price
    w = {}
    o = {}
    for p in product_ids:
        for dist in prod_dists[p]:
            d = dist['distribution_id']
            for scenario in dist['scenarios']:
                s = scenario['scenario_id']
                w[p, d, s] = model.addVar(lb=0.0, name=f"w_{p}_{d}_{s}")
                o[p, d, s] = model.addVar(lb=0.0, name=f"o_{p}_{d}_{s}")

    model.update()

    # --- Constraints ---

    # 1. Exactly one production level per product-facility pair
    for p in product_ids:
        for f in facility_ids:
            levels = pf_data[(p, f)]['production_levels']
            model.addConstr(
                gp.quicksum(y[p, f, lev['level_id']] for lev in levels) == 1,
                name=f"one_level_{p}_{f}"
            )

    # 2. Exactly one distribution per product
    for p in product_ids:
        model.addConstr(
            gp.quicksum(delta[p, dist['distribution_id']] for dist in prod_dists[p]) == 1,
            name=f"one_dist_{p}"
        )

    # 3. Production level bounds on x
    for p in product_ids:
        for f in facility_ids:
            levels = pf_data[(p, f)]['production_levels']
            model.addConstr(
                x[p, f] >= gp.quicksum(
                    lev['lower_bound'] * y[p, f, lev['level_id']] for lev in levels
                ),
                name=f"lb_{p}_{f}"
            )
            model.addConstr(
                x[p, f] <= gp.quicksum(
                    lev['upper_bound'] * y[p, f, lev['level_id']] for lev in levels
                ),
                name=f"ub_{p}_{f}"
            )

    # 4. Facility capacity constraints
    for f in facility_ids:
        model.addConstr(
            gp.quicksum(x[p, f] for p in product_ids) <= facilities[f]['capacity'],
            name=f"cap_{f}"
        )

    # 5. Distribution-level linkage constraints
    for p in product_ids:
        for dist in prod_dists[p]:
            d = dist['distribution_id']
            if dist['type'] == 'yield_and_demand' and dist.get('level_combination') is not None:
                lc = dist['level_combination']
                for idx, f in enumerate(facility_ids):
                    req_level = lc[idx]
                    model.addConstr(
                        delta[p, d] <= y[p, f, req_level],
                        name=f"link_{p}_{d}_{f}"
                    )

    # 6. Linearization constraints: u[p,d,f] = delta[p,d] * x[p,f]
    for p in product_ids:
        for f in facility_ids:
            M_pf = max(lev['upper_bound'] for lev in pf_data[(p, f)]['production_levels'])
            if M_pf < 1e-10:
                M_pf = 1.0  # avoid zero big-M
            for dist in prod_dists[p]:
                d = dist['distribution_id']
                model.addConstr(u[p, d, f] <= x[p, f], name=f"u1_{p}_{d}_{f}")
                model.addConstr(u[p, d, f] <= M_pf * delta[p, d], name=f"u2_{p}_{d}_{f}")
                model.addConstr(
                    u[p, d, f] >= x[p, f] - M_pf * (1 - delta[p, d]),
                    name=f"u3_{p}_{d}_{f}"
                )

    # 7. Sum of u over distributions equals x (since exactly one delta is 1)
    for p in product_ids:
        for f in facility_ids:
            model.addConstr(
                gp.quicksum(u[p, dist['distribution_id'], f] for dist in prod_dists[p]) == x[p, f],
                name=f"sum_u_{p}_{f}"
            )

    # 8. Sales balance and demand constraints
    for p in product_ids:
        for dist in prod_dists[p]:
            d = dist['distribution_id']
            for scenario in dist['scenarios']:
                s = scenario['scenario_id']
                if dist['type'] == 'yield_and_demand':
                    z_expr = gp.quicksum(
                        scenario['yields'].get(str(f), 0.0) * u[p, d, f]
                        for f in facility_ids
                    )
                else:
                    # demand_only: yield = 1 for all facilities
                    z_expr = gp.quicksum(u[p, d, f] for f in facility_ids)

                model.addConstr(
                    w[p, d, s] + o[p, d, s] == z_expr,
                    name=f"bal_{p}_{d}_{s}"
                )
                model.addConstr(
                    w[p, d, s] <= scenario['demand'],
                    name=f"dem_{p}_{d}_{s}"
                )

    # --- Objective ---
    obj = gp.LinExpr()
    for p in product_ids:
        sp = products[p]['sale_price']
        dp = products[p]['discounted_price']
        for dist in prod_dists[p]:
            d = dist['distribution_id']
            for scenario in dist['scenarios']:
                s = scenario['scenario_id']
                prob = scenario['probability']
                obj += prob * sp * w[p, d, s]
                obj += prob * dp * o[p, d, s]
        for f in facility_ids:
            obj -= pf_data[(p, f)]['manufacturing_cost'] * x[p, f]

    model.setObjective(obj, GRB.MAXIMIZE)

    # --- Callback for logging incumbents ---
    def my_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                solution = extract_solution(model, instance, x, y, delta, u, w, o, use_cb=True)
                if logger:
                    logger.log_solution(obj_val, solution)
            except Exception:
                pass

    # Solve
    model.optimize(my_callback)

    # Extract final solution
    if model.SolCount > 0:
        solution = extract_solution(model, instance, x, y, delta, u, w, o, use_cb=False)
        obj_val = model.ObjVal
        if logger:
            logger.log_solution(obj_val, solution)
    else:
        solution = build_zero_solution(instance)
        obj_val = 0.0
        if logger:
            logger.log_solution(obj_val, solution)

    output = {"objective_value": obj_val, "solution": solution}
    with open(args.solution_path, 'w') as f:
        json.dump(output, f)


def extract_solution(model, instance, x, y, delta, u, w, o, use_cb=False):
    """Extract solution dictionary from model."""

    def get_val(var):
        if use_cb:
            return model.cbGetSolution(var)
        else:
            return var.X

    product_ids = sorted(p['id'] for p in instance['products'])
    facility_ids = sorted(f['id'] for f in instance['facilities'])

    pf_data = {}
    for entry in instance['product_facility_data']:
        pf_data[(entry['product_id'], entry['facility_id'])] = entry

    prod_dists = {}
    for entry in instance['products_distributions']:
        prod_dists[entry['product_id']] = entry['distributions']

    solution = {"x": {}, "y": {}, "delta": {}, "z": {}, "w": {}, "o": {}}

    # x values
    for p in product_ids:
        for f in facility_ids:
            solution["x"][f"{p}_{f}"] = get_val(x[p, f])

    # y values
    for p in product_ids:
        for f in facility_ids:
            for level in pf_data[(p, f)]['production_levels']:
                l = level['level_id']
                solution["y"][f"{p}_{f}_{l}"] = int(round(get_val(y[p, f, l])))

    # delta values
    for p in product_ids:
        for dist in prod_dists[p]:
            d = dist['distribution_id']
            solution["delta"][f"{p}_{d}"] = int(round(get_val(delta[p, d])))

    # z, w, o values
    for p in product_ids:
        for dist in prod_dists[p]:
            d = dist['distribution_id']
            for scenario in dist['scenarios']:
                s = scenario['scenario_id']
                if dist['type'] == 'yield_and_demand':
                    z_val = sum(
                        scenario['yields'].get(str(f), 0.0) * get_val(u[p, d, f])
                        for f in facility_ids
                    )
                else:
                    z_val = sum(get_val(u[p, d, f]) for f in facility_ids)
                solution["z"][f"{p}_{d}_{s}"] = z_val
                solution["w"][f"{p}_{d}_{s}"] = get_val(w[p, d, s])
                solution["o"][f"{p}_{d}_{s}"] = get_val(o[p, d, s])

    return solution


def build_zero_solution(instance):
    """Build a zero solution (all production levels at zero, no allocation)."""
    product_ids = sorted(p['id'] for p in instance['products'])
    facility_ids = sorted(f['id'] for f in instance['facilities'])

    pf_data = {}
    for entry in instance['product_facility_data']:
        pf_data[(entry['product_id'], entry['facility_id'])] = entry

    prod_dists = {}
    for entry in instance['products_distributions']:
        prod_dists[entry['product_id']] = entry['distributions']

    solution = {"x": {}, "y": {}, "delta": {}, "z": {}, "w": {}, "o": {}}

    for p in product_ids:
        for f in facility_ids:
            solution["x"][f"{p}_{f}"] = 0.0
            for level in pf_data[(p, f)]['production_levels']:
                l = level['level_id']
                # Select the zero level if available
                if level['lower_bound'] == 0 and level['upper_bound'] == 0:
                    solution["y"][f"{p}_{f}_{l}"] = 1
                else:
                    solution["y"][f"{p}_{f}_{l}"] = 0

    for p in product_ids:
        dists = prod_dists[p]
        # Select the first distribution as default
        if dists:
            first_d = dists[0]['distribution_id']
            for dist in dists:
                d = dist['distribution_id']
                solution["delta"][f"{p}_{d}"] = 1 if d == first_d else 0
                for scenario in dist['scenarios']:
                    s = scenario['scenario_id']
                    solution["z"][f"{p}_{d}_{s}"] = 0.0
                    solution["w"][f"{p}_{d}_{s}"] = 0.0
                    solution["o"][f"{p}_{d}_{s}"] = 0.0

    return solution


if __name__ == "__main__":
    main()