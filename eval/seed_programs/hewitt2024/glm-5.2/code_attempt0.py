import argparse
import json
import sys
import time
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    start_time = time.time()

    products = instance['products']
    facilities = instance['facilities']

    product_ids = [p['id'] for p in products]
    facility_ids = [f['id'] for f in facilities]

    sale_price = {p['id']: p['sale_price'] for p in products}
    discounted_price = {p['id']: p['discounted_price'] for p in products}
    capacity = {f['id']: f['capacity'] for f in facilities}

    pf_data = {}
    for entry in instance['product_facility_data']:
        pf_data[(entry['product_id'], entry['facility_id'])] = entry

    product_distributions = {}
    for pd in instance['products_distributions']:
        product_distributions[pd['product_id']] = pd['distributions']

    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model('production_planning')
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0

    # --- Variables ---
    y = {}
    x_pfl = {}
    level_data = {}

    for pid in product_ids:
        for fid in facility_ids:
            entry = pf_data[(pid, fid)]
            levels = entry['production_levels']
            level_data[pid, fid] = levels
            for level in levels:
                lid = level['level_id']
                y[pid, fid, lid] = model.addVar(vtype=GRB.BINARY, name=f'y_{pid}_{fid}_{lid}')
                x_pfl[pid, fid, lid] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=level['upper_bound'], name=f'x_{pid}_{fid}_{lid}')

    delta = {}
    for pid in product_ids:
        for d in product_distributions[pid]:
            did = d['distribution_id']
            delta[pid, did] = model.addVar(vtype=GRB.BINARY, name=f'd_{pid}_{did}')

    xi_yd = {}
    xi_do = {}

    for pid in product_ids:
        for d in product_distributions[pid]:
            did = d['distribution_id']
            if d['type'] == 'yield_and_demand':
                for fid in facility_ids:
                    xi_yd[pid, fid, did] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'xiy_{pid}_{fid}_{did}')
            else:
                for fid in facility_ids:
                    for level in level_data[pid, fid]:
                        lid = level['level_id']
                        xi_do[pid, fid, lid, did] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'xid_{pid}_{fid}_{lid}_{did}')

    w = {}
    for pid in product_ids:
        for d in product_distributions[pid]:
            did = d['distribution_id']
            for s in d['scenarios']:
                sid = s['scenario_id']
                w[pid, did, sid] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'w_{pid}_{did}_{sid}')

    model.update()

    # --- Constraints ---

    # 1. Exactly one level per product-facility
    for pid in product_ids:
        for fid in facility_ids:
            levels = level_data[pid, fid]
            model.addConstr(gp.quicksum(y[pid, fid, l['level_id']] for l in levels) == 1)

    # 2. Bounds on x_pfl
    for pid in product_ids:
        for fid in facility_ids:
            for level in level_data[pid, fid]:
                lid = level['level_id']
                lb = level['lower_bound']
                ub = level['upper_bound']
                if lb > 0:
                    model.addConstr(x_pfl[pid, fid, lid] >= lb * y[pid, fid, lid])
                model.addConstr(x_pfl[pid, fid, lid] <= ub * y[pid, fid, lid])

    # 3. Facility capacity
    for fid in facility_ids:
        model.addConstr(
            gp.quicksum(x_pfl[pid, fid, l['level_id']]
                        for pid in product_ids
                        for l in level_data[pid, fid]) <= capacity[fid]
        )

    # 4. Exactly one distribution per product
    for pid in product_ids:
        dists = product_distributions[pid]
        model.addConstr(gp.quicksum(delta[pid, d['distribution_id']] for d in dists) == 1)

    # 5. Link delta to y for yield_and_demand
    for pid in product_ids:
        for d in product_distributions[pid]:
            did = d['distribution_id']
            if d['type'] == 'yield_and_demand':
                level_combo = d['level_combination']
                for fi, fid in enumerate(facility_ids):
                    req_level = level_combo[fi]
                    model.addConstr(delta[pid, did] <= y[pid, fid, req_level])

    # 6. Linearization of xi
    for pid in product_ids:
        for fid in facility_ids:
            cap = capacity[fid]
            M = min(max(l['upper_bound'] for l in level_data[pid, fid]), cap)
            x_sum = gp.quicksum(x_pfl[pid, fid, l['level_id']] for l in level_data[pid, fid])

            for d in product_distributions[pid]:
                did = d['distribution_id']
                if d['type'] == 'yield_and_demand':
                    model.addConstr(xi_yd[pid, fid, did] <= delta[pid, did] * M)
                    model.addConstr(xi_yd[pid, fid, did] <= x_sum)
                    model.addConstr(xi_yd[pid, fid, did] >= x_sum - M * (1 - delta[pid, did]))
                else:
                    for level in level_data[pid, fid]:
                        lid = level['level_id']
                        Ml = min(level['upper_bound'], cap)
                        model.addConstr(xi_do[pid, fid, lid, did] <= delta[pid, did] * Ml)
                        model.addConstr(xi_do[pid, fid, lid, did] <= x_pfl[pid, fid, lid])
                        model.addConstr(xi_do[pid, fid, lid, did] >= x_pfl[pid, fid, lid] - Ml * (1 - delta[pid, did]))

    # 7. w <= z and w <= delta * demand
    for pid in product_ids:
        for d in product_distributions[pid]:
            did = d['distribution_id']
            dtype = d['type']

            for s in d['scenarios']:
                sid = s['scenario_id']
                demand = s['demand']

                if dtype == 'yield_and_demand':
                    yields = s.get('yields', {})
                    z_expr = gp.LinExpr()
                    for fid in facility_ids:
                        yld = yields.get(str(fid), yields.get(fid, 0))
                        z_expr += yld * xi_yd[pid, fid, did]
                else:
                    z_expr = gp.LinExpr()
                    for fid in facility_ids:
                        for level in level_data[pid, fid]:
                            lid = level['level_id']
                            z_expr += level['yield_mean'] * xi_do[pid, fid, lid, did]

                model.addConstr(w[pid, did, sid] <= z_expr)
                model.addConstr(w[pid, did, sid] <= delta[pid, did] * demand)

    # --- Objective ---
    obj = gp.LinExpr()

    for pid in product_ids:
        sp = sale_price[pid]
        dp = discounted_price[pid]
        diff = sp - dp
        for d in product_distributions[pid]:
            did = d['distribution_id']
            dtype = d['type']
            for s in d['scenarios']:
                sid = s['scenario_id']
                prob = s['probability']

                if dtype == 'yield_and_demand':
                    yields = s.get('yields', {})
                    z_expr = gp.LinExpr()
                    for fid in facility_ids:
                        yld = yields.get(str(fid), yields.get(fid, 0))
                        z_expr += yld * xi_yd[pid, fid, did]
                else:
                    z_expr = gp.LinExpr()
                    for fid in facility_ids:
                        for level in level_data[pid, fid]:
                            lid = level['level_id']
                            z_expr += level['yield_mean'] * xi_do[pid, fid, lid, did]

                obj += prob * (dp * z_expr + diff * w[pid, did, sid])

    for pid in product_ids:
        for fid in facility_ids:
            mc = pf_data[(pid, fid)]['manufacturing_cost']
            for level in level_data[pid, fid]:
                lid = level['level_id']
                obj -= mc * x_pfl[pid, fid, lid]

    model.setObjective(obj, GRB.MAXIMIZE)

    # Set time limit
    build_time = time.time() - start_time
    remaining_time = max(1, args.time_limit - build_time - 3)
    model.Params.TimeLimit = remaining_time

    # Callback for logging incumbents
    logger_ref = logger
    all_vars = {
        'y': y, 'x_pfl': x_pfl, 'delta': delta, 'xi_yd': xi_yd, 'xi_do': xi_do, 'w': w
    }
    data_refs = {
        'product_ids': product_ids, 'facility_ids': facility_ids,
        'level_data': level_data, 'product_distributions': product_distributions,
        'sale_price': sale_price, 'discounted_price': discounted_price,
        'pf_data': pf_data
    }

    def log_callback(cb_model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger_ref:
                logger_ref.log(obj_val)

    model.optimize(log_callback)

    # --- Extract solution ---
    x_dict = {}
    y_dict = {}
    delta_dict = {}
    z_dict = {}
    w_dict = {}
    o_dict = {}

    has_sol = model.SolCount > 0

    # Find zero levels for fallback
    zero_level_id = {}
    for pid in product_ids:
        for fid in facility_ids:
            for level in level_data[pid, fid]:
                if level['lower_bound'] == 0 and level['upper_bound'] == 0:
                    zero_level_id[pid, fid] = level['level_id']
                    break

    # Find compatible distribution for fallback
    fallback_delta = {}
    for pid in product_ids:
        found = False
        for d in product_distributions[pid]:
            if d['type'] == 'demand_only':
                fallback_delta[pid] = d['distribution_id']
                found = True
                break
        if not found:
            for d in product_distributions[pid]:
                if d['type'] == 'yield_and_demand':
                    lc = d['level_combination']
                    if all(lc[fi] == zero_level_id.get((pid, fid), -1) for fi, fid in enumerate(facility_ids)):
                        fallback_delta[pid] = d['distribution_id']
                        found = True
                        break
        if not found:
            fallback_delta[pid] = product_distributions[pid][0]['distribution_id']

    obj_val = 0.0
    if has_sol:
        obj_val = model.ObjVal

    for pid in product_ids:
        for fid in facility_ids:
            x_total = 0.0
            for level in level_data[pid, fid]:
                lid = level['level_id']
                if has_sol:
                    y_val = int(round(y[pid, fid, lid].X))
                    x_val = x_pfl[pid, fid, lid].X
                else:
                    zl = zero_level_id.get((pid, fid), level['level_id'])
                    y_val = 1 if lid == zl else 0
                    x_val = 0.0
                y_dict[f"{pid}_{fid}_{lid}"] = y_val
                x_total += x_val
            x_dict[f"{pid}_{fid}"] = x_total

        for d in product_distributions[pid]:
            did = d['distribution_id']
            if has_sol:
                delta_val = int(round(delta[pid, did].X))
            else:
                delta_val = 1 if did == fallback_delta[pid] else 0
            delta_dict[f"{pid}_{did}"] = delta_val

            for s in d['scenarios']:
                sid = s['scenario_id']
                dtype = d['type']

                if has_sol and delta_val == 1:
                    if dtype == 'yield_and_demand':
                        yields = s.get('yields', {})
                        z_val = 0.0
                        for fid in facility_ids:
                            yld = yields.get(str(fid), yields.get(fid, 0))
                            z_val += yld * xi_yd[pid, fid, did].X
                    else:
                        z_val = 0.0
                        for fid in facility_ids:
                            for level in level_data[pid, fid]:
                                lid = level['level_id']
                                z_val += level['yield_mean'] * xi_do[pid, fid, lid, did].X
                    w_val = w[pid, did, sid].X
                    o_val = max(0.0, z_val - w_val)
                else:
                    z_val = 0.0
                    w_val = 0.0
                    o_val = 0.0

                z_dict[f"{pid}_{did}_{sid}"] = z_val
                w_dict[f"{pid}_{did}_{sid}"] = w_val
                o_dict[f"{pid}_{did}_{sid}"] = o_val

    solution = {
        "objective_value": obj_val,
        "solution": {
            "x": x_dict,
            "y": y_dict,
            "delta": delta_dict,
            "z": z_dict,
            "w": w_dict,
            "o": o_dict
        }
    }

    if logger:
        logger.log_solution(obj_val, solution["solution"])

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()