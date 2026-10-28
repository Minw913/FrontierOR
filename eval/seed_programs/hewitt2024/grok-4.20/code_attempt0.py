import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    num_products = data['num_products']
    num_facilities = data['num_facilities']
    products = {p['id']: p for p in data['products']}
    facilities = {f['id']: f for f in data['facilities']}

    # Build product_facility_data lookup
    prod_fac = {}
    for entry in data['product_facility_data']:
        pid = entry['product_id']
        fid = entry['facility_id']
        if pid not in prod_fac:
            prod_fac[pid] = {}
        prod_fac[pid][fid] = entry

    # Build distributions per product
    distributions = {}
    for pdist in data['products_distributions']:
        pid = pdist['product_id']
        distributions[pid] = pdist['distributions']

    # Create model
    model = gp.Model("Production_Planning")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('MIPGap', 0.01)  # Allow some gap for speed if needed
    model.setParam('Threads', 1)

    # Variables
    # x[pid,fid]: raw materials allocated
    x = {}
    for pid in products:
        for fid in facilities:
            x[pid, fid] = model.addVar(lb=0.0, name=f"x_{pid}_{fid}")

    # y[pid,fid,lid]: select production level
    y = {}
    for pid in products:
        for fid in facilities:
            entry = prod_fac[pid][fid]
            for level in entry['production_levels']:
                lid = level['level_id']
                y[pid, fid, lid] = model.addVar(vtype=GRB.BINARY, name=f"y_{pid}_{fid}_{lid}")

    # delta[pid, did]: select distribution (only yield_and_demand ones matter)
    delta = {}
    for pid in products:
        for dist in distributions[pid]:
            if dist['type'] == 'yield_and_demand':
                did = dist['distribution_id']
                delta[pid, did] = model.addVar(vtype=GRB.BINARY, name=f"delta_{pid}_{did}")

    # For each product, we need to link levels to selected distribution
    # Also z, w, o per distribution and scenario (only for selected ones)
    z = {}
    w = {}
    o = {}
    for pid in products:
        for dist in distributions[pid]:
            did = dist['distribution_id']
            for scen in dist['scenarios']:
                sid = scen['scenario_id']
                z[pid, did, sid] = model.addVar(lb=0.0, name=f"z_{pid}_{did}_{sid}")
                w[pid, did, sid] = model.addVar(lb=0.0, name=f"w_{pid}_{did}_{sid}")
                o[pid, did, sid] = model.addVar(lb=0.0, name=f"o_{pid}_{did}_{sid}")

    # Objective: maximize expected profit = expected revenue - manufacturing cost
    manuf_cost = gp.LinExpr()
    for pid in products:
        for fid in facilities:
            entry = prod_fac[pid][fid]
            manuf_cost += entry['manufacturing_cost'] * x[pid, fid]

    expected_rev = gp.LinExpr()
    bigM = 1e9  # for bounding

    for pid in products:
        prod_info = products[pid]
        sale_p = prod_info['sale_price']
        disc_p = prod_info['discounted_price']
        for dist_dict in distributions[pid]:
            did = dist_dict['distribution_id']
            if dist_dict['type'] == 'demand_only':
                continue  # we only use yield_and_demand as per problem
            is_yield_demand = dist_dict['type'] == 'yield_and_demand'
            prob_sum = 0.0
            for scen in dist_dict['scenarios']:
                sid = scen['scenario_id']
                prob = scen['probability']
                prob_sum += prob
                # w <= demand, w + o = z, w,o >=0
                dem = scen['demand']
                expected_rev += prob * (sale_p * w[pid, did, sid] + disc_p * o[pid, did, sid])
                model.addConstr(w[pid, did, sid] <= dem, name=f"wlim_{pid}_{did}_{sid}")
                model.addConstr(w[pid, did, sid] + o[pid, did, sid] == z[pid, did, sid], name=f"bal_{pid}_{did}_{sid}")
                model.addConstr(o[pid, did, sid] >= 0, name=f"o_nonneg_{pid}_{did}_{sid}")

            if is_yield_demand:
                # Link delta to this distribution being chosen
                # z is only active if delta selected
                for scen in dist_dict['scenarios']:
                    sid = scen['scenario_id']
                    model.addConstr(z[pid, did, sid] <= bigM * delta[pid, did], name=f"zlink_{pid}_{did}_{sid}")

    # Constraints

    # Exactly one distribution selected per product (among yield_and_demand)
    for pid in products:
        yield_dists = [d for d in distributions[pid] if d['type'] == 'yield_and_demand']
        model.addConstr(gp.quicksum(delta[pid, d['distribution_id']] for d in yield_dists) == 1,
                        name=f"one_delta_{pid}")

    # Production level selection: exactly one level per product-facility
    for pid in products:
        for fid in facilities:
            entry = prod_fac[pid][fid]
            levels = entry['production_levels']
            model.addConstr(gp.quicksum(y[pid, fid, lev['level_id']] for lev in levels) == 1,
                            name=f"one_level_{pid}_{fid}")

    # Link production levels to chosen distribution
    for pid in products:
        for dist_dict in distributions[pid]:
            if dist_dict['type'] != 'yield_and_demand':
                continue
            did = dist_dict['distribution_id']
            level_comb = dist_dict['level_combination']
            delta_var = delta[pid, did]
            for f_idx, fid in enumerate(facilities.keys()):  # assume order is consistent with list index
                required_level = level_comb[f_idx]
                # If delta=1 then y for that level must be 1
                model.addConstr(y[pid, fid, required_level] >= delta_var,
                                name=f"level_force_{pid}_{did}_{fid}_{required_level}")

    # Linking x to selected level bounds
    for pid in products:
        for fid in facilities:
            entry = prod_fac[pid][fid]
            for level in entry['production_levels']:
                lid = level['level_id']
                lb = level['lower_bound']
                ub = level['upper_bound']
                yvar = y[pid, fid, lid]
                model.addConstr(x[pid, fid] >= lb * yvar, name=f"lb_{pid}_{fid}_{lid}")
                model.addConstr(x[pid, fid] <= ub * yvar + bigM * (1 - yvar), name=f"ub_{pid}_{fid}_{lid}")

    # Capacity constraints
    for fid in facilities:
        cap = facilities[fid]['capacity']
        total_alloc = gp.quicksum(x[pid, fid] for pid in products)
        model.addConstr(total_alloc <= cap, name=f"cap_{fid}")

    # Yield linking: z = sum over facilities of (yield * x)
    for pid in products:
        for dist_dict in distributions[pid]:
            if dist_dict['type'] != 'yield_and_demand':
                continue
            did = dist_dict['distribution_id']
            for scen in dist_dict['scenarios']:
                sid = scen['scenario_id']
                yields = scen['yields']
                z_expr = gp.LinExpr()
                for fid_str, yld in yields.items():
                    fid = int(fid_str)
                    z_expr += yld * x[pid, fid]
                model.addConstr(z[pid, did, sid] == z_expr, name=f"yielddef_{pid}_{did}_{sid}")

    # Objective
    model.setObjective(expected_rev - manuf_cost, GRB.MAXIMIZE)

    # Callback for logging incumbents
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    # Optimize
    start_time = time.time()
    model.optimize(callback)

    # Extract solution
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT]:
        objective_value = model.objVal

        sol_x = {}
        sol_y = {}
        sol_delta = {}
        sol_z = {}
        sol_w = {}
        sol_o = {}

        for pid in products:
            for fid in facilities:
                val = x[pid, fid].X
                if abs(val) > 1e-7:
                    sol_x[f"{pid}_{fid}"] = float(val)
                for level in prod_fac[pid][fid]['production_levels']:
                    lid = level['level_id']
                    yval = y[pid, fid, lid].X
                    sol_y[f"{pid}_{fid}_{lid}"] = 1 if yval > 0.5 else 0

        for pid in products:
            for dist in distributions[pid]:
                if dist['type'] != 'yield_and_demand':
                    continue
                did = dist['distribution_id']
                dval = delta[pid, did].X if (pid, did) in delta else 0
                sol_delta[f"{pid}_{did}"] = 1 if dval > 0.5 else 0
                for scen in dist['scenarios']:
                    sid = scen['scenario_id']
                    zkey = f"{pid}_{did}_{sid}"
                    sol_z[zkey] = float(z[pid, did, sid].X)
                    sol_w[zkey] = float(w[pid, did, sid].X)
                    sol_o[zkey] = float(o[pid, did, sid].X)

        solution = {
            "x": sol_x,
            "y": sol_y,
            "delta": sol_delta,
            "z": sol_z,
            "w": sol_w,
            "o": sol_o
        }

        result = {
            "objective_value": float(objective_value),
            "solution": solution
        }

        with open(args.solution_path, 'w') as f:
            json.dump(result, f, indent=2)
    else:
        # Infeasible or other error - write empty feasible if possible
        result = {
            "objective_value": 0.0,
            "solution": {
                "x": {},
                "y": {},
                "delta": {},
                "z": {},
                "w": {},
                "o": {}
            }
        }
        with open(args.solution_path, 'w') as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()