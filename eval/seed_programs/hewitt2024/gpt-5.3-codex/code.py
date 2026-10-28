import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Solve endogenous-yield production planning problem.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input instance JSON.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON.")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Optional JSONL path for incumbent logging.")
    return parser.parse_args()


def safe_get_yield(yields_dict, facility_id, default=1.0):
    if yields_dict is None:
        return default
    if str(facility_id) in yields_dict:
        return float(yields_dict[str(facility_id)])
    if facility_id in yields_dict:
        return float(yields_dict[facility_id])
    return default


def main():
    args = parse_args()

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    products = inst["products"]
    facilities = inst["facilities"]
    pf_data = inst["product_facility_data"]
    prod_dists = inst["products_distributions"]

    product_ids = [p["id"] for p in products]
    facility_ids = [f["id"] for f in facilities]

    sale_price = {p["id"]: float(p["sale_price"]) for p in products}
    disc_price = {p["id"]: float(p["discounted_price"]) for p in products}
    capacity = {f["id"]: float(f["capacity"]) for f in facilities}

    # product-facility info
    manuf_cost = {}
    levels = {}          # (p,f) -> list of level dicts
    level_ids = {}       # (p,f) -> list of level ids
    lb = {}
    ub = {}
    yield_mean = {}
    x_ub_pf = {}

    for rec in pf_data:
        p = rec["product_id"]
        f_id = rec["facility_id"]
        manuf_cost[(p, f_id)] = float(rec["manufacturing_cost"])
        lvls = rec["production_levels"]
        levels[(p, f_id)] = lvls
        level_ids[(p, f_id)] = [lv["level_id"] for lv in lvls]
        for lv in lvls:
            l = lv["level_id"]
            lb[(p, f_id, l)] = float(lv["lower_bound"])
            ub[(p, f_id, l)] = float(lv["upper_bound"])
            yield_mean[(p, f_id, l)] = float(lv.get("yield_mean", 1.0))
        x_ub_pf[(p, f_id)] = max(float(lv["upper_bound"]) for lv in lvls)

    # distributions by product
    distributions = {}   # p -> list of drec
    for pd in prod_dists:
        p = pd["product_id"]
        distributions[p] = pd["distributions"]

    m = Model("endogenous_yield_planning")
    m.Params.TimeLimit = args.time_limit
    m.Params.Threads = 1
    m.Params.OutputFlag = 0

    # Decision variables
    x = {}
    y = {}
    delta = {}
    z = {}
    w = {}
    o = {}

    # x and y
    for p in product_ids:
        for f_id in facility_ids:
            x[(p, f_id)] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"x_{p}_{f_id}")
            for l in level_ids[(p, f_id)]:
                y[(p, f_id, l)] = m.addVar(vtype=GRB.BINARY, name=f"y_{p}_{f_id}_{l}")

    # delta, z, w, o
    scenario_info = {}  # (p,d,s) -> dict(prob, demand, type, coeff_by_f, inv_ub)
    for p in product_ids:
        for drec in distributions[p]:
            d = drec["distribution_id"]
            delta[(p, d)] = m.addVar(vtype=GRB.BINARY, name=f"delta_{p}_{d}")

            d_type = drec.get("type", "yield_and_demand")
            level_comb = drec.get("level_combination", None)

            for srec in drec["scenarios"]:
                s = srec["scenario_id"]
                prob = float(srec["probability"])
                dem = float(srec["demand"])

                coeff = {}
                if d_type == "yield_and_demand":
                    ydict = srec.get("yields", {})
                    for f_id in facility_ids:
                        coeff[f_id] = safe_get_yield(ydict, f_id, default=1.0)
                else:
                    # demand_only fallback:
                    # if level combination exists, use corresponding level mean yields;
                    # else assume unit conversion (1.0) from raw material to finished goods.
                    if level_comb is not None:
                        for idx, f_id in enumerate(facility_ids):
                            l = level_comb[idx]
                            coeff[f_id] = float(yield_mean.get((p, f_id, l), 1.0))
                    else:
                        for f_id in facility_ids:
                            coeff[f_id] = 1.0

                inv_ub = 0.0
                for f_id in facility_ids:
                    c = max(0.0, float(coeff[f_id]))
                    inv_ub += c * x_ub_pf[(p, f_id)]

                z[(p, d, s)] = m.addVar(lb=0.0, ub=max(inv_ub, 0.0), vtype=GRB.CONTINUOUS, name=f"z_{p}_{d}_{s}")
                w[(p, d, s)] = m.addVar(lb=0.0, ub=dem, vtype=GRB.CONTINUOUS, name=f"w_{p}_{d}_{s}")
                o[(p, d, s)] = m.addVar(lb=0.0, ub=max(inv_ub, 0.0), vtype=GRB.CONTINUOUS, name=f"o_{p}_{d}_{s}")

                scenario_info[(p, d, s)] = {
                    "prob": prob,
                    "demand": dem,
                    "type": d_type,
                    "coeff": coeff,
                    "inv_ub": inv_ub,
                    "level_combination": level_comb
                }

    m.update()

    # Constraints

    # Facility capacities
    for f_id in facility_ids:
        m.addConstr(quicksum(x[(p, f_id)] for p in product_ids) <= capacity[f_id], name=f"cap_{f_id}")

    # One level per product-facility, and x within selected interval
    for p in product_ids:
        for f_id in facility_ids:
            m.addConstr(quicksum(y[(p, f_id, l)] for l in level_ids[(p, f_id)]) == 1, name=f"one_level_{p}_{f_id}")
            m.addConstr(
                x[(p, f_id)] >= quicksum(lb[(p, f_id, l)] * y[(p, f_id, l)] for l in level_ids[(p, f_id)]),
                name=f"lb_link_{p}_{f_id}"
            )
            m.addConstr(
                x[(p, f_id)] <= quicksum(ub[(p, f_id, l)] * y[(p, f_id, l)] for l in level_ids[(p, f_id)]),
                name=f"ub_link_{p}_{f_id}"
            )

    # One distribution per product
    for p in product_ids:
        d_ids = [drec["distribution_id"] for drec in distributions[p]]
        m.addConstr(quicksum(delta[(p, d)] for d in d_ids) == 1, name=f"one_dist_{p}")

        # Endogenous mapping constraints when level_combination is specified
        for drec in distributions[p]:
            d = drec["distribution_id"]
            level_comb = drec.get("level_combination", None)
            if level_comb is not None:
                for idx, f_id in enumerate(facility_ids):
                    req_level = level_comb[idx]
                    m.addConstr(
                        delta[(p, d)] <= y[(p, f_id, req_level)],
                        name=f"map_{p}_{d}_{f_id}"
                    )

    # Scenario constraints
    for (p, d, s), info in scenario_info.items():
        dem = info["demand"]
        inv_ub = max(0.0, info["inv_ub"])
        coeff = info["coeff"]

        # Activate/Deactivate second-stage variables by selected distribution
        m.addConstr(w[(p, d, s)] <= dem * delta[(p, d)], name=f"w_act_{p}_{d}_{s}")
        m.addConstr(o[(p, d, s)] <= inv_ub * delta[(p, d)], name=f"o_act_{p}_{d}_{s}")
        m.addConstr(z[(p, d, s)] <= inv_ub * delta[(p, d)], name=f"z_act_{p}_{d}_{s}")

        # Inventory sold split
        m.addConstr(w[(p, d, s)] + o[(p, d, s)] == z[(p, d, s)], name=f"balance_{p}_{d}_{s}")

        # Inventory definition when distribution selected (big-M)
        inv_expr = quicksum(coeff[f_id] * x[(p, f_id)] for f_id in facility_ids)
        m.addConstr(
            z[(p, d, s)] - inv_expr <= inv_ub * (1 - delta[(p, d)]),
            name=f"inv_up_{p}_{d}_{s}"
        )
        m.addConstr(
            z[(p, d, s)] - inv_expr >= -inv_ub * (1 - delta[(p, d)]),
            name=f"inv_lo_{p}_{d}_{s}"
        )

    # Objective
    revenue = quicksum(
        scenario_info[(p, d, s)]["prob"] * (
            sale_price[p] * w[(p, d, s)] + disc_price[p] * o[(p, d, s)]
        )
        for (p, d, s) in scenario_info.keys()
    )
    cost = quicksum(manuf_cost[(p, f_id)] * x[(p, f_id)] for p in product_ids for f_id in facility_ids)
    m.setObjective(revenue - cost, GRB.MAXIMIZE)

    # Incumbent logging callback
    def cb(model, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            logger.log(float(obj))

    m.optimize(cb)

    feasible = m.SolCount > 0

    # Build output dictionaries (all keys required by schema)
    x_out = {}
    y_out = {}
    delta_out = {}
    z_out = {}
    w_out = {}
    o_out = {}

    for p in product_ids:
        for f_id in facility_ids:
            x_key = f"{p}_{f_id}"
            x_out[x_key] = float(x[(p, f_id)].X) if feasible else 0.0
            for l in level_ids[(p, f_id)]:
                y_key = f"{p}_{f_id}_{l}"
                y_val = y[(p, f_id, l)].X if feasible else 0.0
                y_out[y_key] = int(round(y_val))

    for p in product_ids:
        for drec in distributions[p]:
            d = drec["distribution_id"]
            d_key = f"{p}_{d}"
            d_val = delta[(p, d)].X if feasible else 0.0
            delta_out[d_key] = int(round(d_val))

            for srec in drec["scenarios"]:
                s = srec["scenario_id"]
                key = f"{p}_{d}_{s}"
                z_out[key] = float(z[(p, d, s)].X) if feasible else 0.0
                w_out[key] = float(w[(p, d, s)].X) if feasible else 0.0
                o_out[key] = float(o[(p, d, s)].X) if feasible else 0.0

    objective_value = float(m.ObjVal) if feasible else 0.0

    solution = {
        "objective_value": objective_value,
        "solution": {
            "x": x_out,
            "y": y_out,
            "delta": delta_out,
            "z": z_out,
            "w": w_out,
            "o": o_out
        }
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()