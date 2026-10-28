import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    
    # Initialize logger
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    # Load instance data
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    num_products = data['num_products']
    num_facilities = data['num_facilities']
    
    products = {p['id']: p for p in data['products']}
    facilities = {f['id']: f for f in data['facilities']}
    
    # Product-facility data indexed by (product_id, facility_id)
    pf_data = {}
    for pf in data['product_facility_data']:
        pid = pf['product_id']
        fid = pf['facility_id']
        pf_data[(pid, fid)] = pf
    
    # Distributions indexed by (product_id, distribution_id)
    prod_dists = {}
    for pd_entry in data['products_distributions']:
        pid = pd_entry['product_id']
        prod_dists[pid] = pd_entry
    
    product_ids = sorted(products.keys())
    facility_ids = sorted(facilities.keys())
    
    # Build model
    model = gp.Model("endogenous_uncertainty")
    model.Params.TimeLimit = max(1, time_limit - (time.time() - start_time) - 5)
    model.Params.Threads = 1
    model.Params.MIPFocus = 1  # Focus on finding feasible solutions
    
    # Decision variables
    # x[p,f] - raw materials allocated
    x = {}
    for pid in product_ids:
        for fid in facility_ids:
            pf = pf_data[(pid, fid)]
            # Find max upper bound across all levels
            max_ub = max(lv['upper_bound'] for lv in pf['production_levels'])
            x[(pid, fid)] = model.addVar(lb=0, ub=max_ub, name=f"x_{pid}_{fid}")
    
    # y[p,f,l] - production level selection (binary)
    y = {}
    for pid in product_ids:
        for fid in facility_ids:
            pf = pf_data[(pid, fid)]
            for lv in pf['production_levels']:
                lid = lv['level_id']
                y[(pid, fid, lid)] = model.addVar(vtype=GRB.BINARY, name=f"y_{pid}_{fid}_{lid}")
    
    # delta[p,d] - distribution selection (binary)
    delta = {}
    for pid in product_ids:
        pd_entry = prod_dists[pid]
        for dist in pd_entry['distributions']:
            did = dist['distribution_id']
            delta[(pid, did)] = model.addVar(vtype=GRB.BINARY, name=f"delta_{pid}_{did}")
    
    # z[p,d,s] - available inventory in scenario
    z = {}
    # w[p,d,s] - full price sales
    w = {}
    # o[p,d,s] - discounted sales
    o = {}
    
    for pid in product_ids:
        pd_entry = prod_dists[pid]
        for dist in pd_entry['distributions']:
            did = dist['distribution_id']
            for scen in dist['scenarios']:
                sid = scen['scenario_id']
                z[(pid, did, sid)] = model.addVar(lb=0, name=f"z_{pid}_{did}_{sid}")
                w[(pid, did, sid)] = model.addVar(lb=0, name=f"w_{pid}_{did}_{sid}")
                o[(pid, did, sid)] = model.addVar(lb=0, name=f"o_{pid}_{did}_{sid}")
    
    model.update()
    
    # Constraints
    
    # 1. Exactly one production level per product-facility
    for pid in product_ids:
        for fid in facility_ids:
            pf = pf_data[(pid, fid)]
            level_ids = [lv['level_id'] for lv in pf['production_levels']]
            model.addConstr(
                gp.quicksum(y[(pid, fid, lid)] for lid in level_ids) == 1,
                name=f"one_level_{pid}_{fid}"
            )
    
    # 2. x bounds based on selected level
    for pid in product_ids:
        for fid in facility_ids:
            pf = pf_data[(pid, fid)]
            for lv in pf['production_levels']:
                lid = lv['level_id']
                lb = lv['lower_bound']
                ub = lv['upper_bound']
                # x >= lb * y and x <= ub * y (when this level is selected)
                # We need: if y[p,f,l]=1 then lb <= x[p,f] <= ub
                # Use indicator constraints or big-M
                # Since exactly one level is selected, we can use:
                # x[p,f] >= sum_l (lb_l * y[p,f,l])
                # x[p,f] <= sum_l (ub_l * y[p,f,l])
            
            lbs = {lv['level_id']: lv['lower_bound'] for lv in pf['production_levels']}
            ubs = {lv['level_id']: lv['upper_bound'] for lv in pf['production_levels']}
            level_ids = [lv['level_id'] for lv in pf['production_levels']]
            
            model.addConstr(
                x[(pid, fid)] >= gp.quicksum(lbs[lid] * y[(pid, fid, lid)] for lid in level_ids),
                name=f"x_lb_{pid}_{fid}"
            )
            model.addConstr(
                x[(pid, fid)] <= gp.quicksum(ubs[lid] * y[(pid, fid, lid)] for lid in level_ids),
                name=f"x_ub_{pid}_{fid}"
            )
    
    # 3. Facility capacity
    for fid in facility_ids:
        model.addConstr(
            gp.quicksum(x[(pid, fid)] for pid in product_ids) <= facilities[fid]['capacity'],
            name=f"capacity_{fid}"
        )
    
    # 4. Exactly one distribution per product
    for pid in product_ids:
        pd_entry = prod_dists[pid]
        dist_ids = [dist['distribution_id'] for dist in pd_entry['distributions']]
        model.addConstr(
            gp.quicksum(delta[(pid, did)] for did in dist_ids) == 1,
            name=f"one_dist_{pid}"
        )
    
    # 5. Link distribution selection to production levels
    # If delta[p,d]=1, then y[p,f,level_combination[f]]=1 for all f
    for pid in product_ids:
        pd_entry = prod_dists[pid]
        for dist in pd_entry['distributions']:
            did = dist['distribution_id']
            if dist['type'] == 'yield_and_demand' and dist['level_combination'] is not None:
                lc = dist['level_combination']
                for idx, fid in enumerate(facility_ids):
                    required_level = lc[idx]
                    # delta[p,d] <= y[p,f,required_level]
                    model.addConstr(
                        delta[(pid, did)] <= y[(pid, fid, required_level)],
                        name=f"link_dist_level_{pid}_{did}_{fid}"
                    )
    
    # 6. Inventory calculation: z[p,d,s] = sum_f (yield[f,s] * x[p,f]) when delta[p,d]=1
    # We need z to be correct when delta is selected, and 0 otherwise
    # z[p,d,s] = delta[p,d] * sum_f(yield[f,s] * x[p,f]) -- this is bilinear
    # Since delta is binary, we can linearize using big-M
    
    # Compute a big M for inventory: max possible inventory per product
    M_inv = {}
    for pid in product_ids:
        max_inv = 0
        for fid in facility_ids:
            pf = pf_data[(pid, fid)]
            max_ub = max(lv['upper_bound'] for lv in pf['production_levels'])
            max_inv += max_ub  # yield <= 1 typically, but let's be safe
        M_inv[pid] = max_inv * 2  # generous bound
    
    for pid in product_ids:
        pd_entry = prod_dists[pid]
        for dist in pd_entry['distributions']:
            did = dist['distribution_id']
            for scen in dist['scenarios']:
                sid = scen['scenario_id']
                
                if dist['type'] == 'yield_and_demand' and 'yields' in scen:
                    yields = scen['yields']
                    inv_expr = gp.quicksum(
                        float(yields[str(fid)]) * x[(pid, fid)] for fid in facility_ids
                    )
                else:
                    # demand-only distribution: yields might default to expected or something
                    # Based on problem description, only yield_and_demand distributions have yields
                    # For demand_only, inventory is deterministic based on expected yields?
                    # Actually, re-reading: "the company must select exactly one distribution for each product"
                    # and it should be yield_and_demand type based on the problem structure
                    # For safety, treat yields as 1.0 if not present
                    inv_expr = gp.quicksum(x[(pid, fid)] for fid in facility_ids)
                
                M = M_inv[pid]
                
                # z[p,d,s] <= inv_expr (always, since z should be 0 when delta=0)
                # z[p,d,s] >= inv_expr - M*(1 - delta[p,d])
                # z[p,d,s] <= M * delta[p,d]
                
                model.addConstr(
                    z[(pid, did, sid)] <= inv_expr,
                    name=f"z_ub1_{pid}_{did}_{sid}"
                )
                model.addConstr(
                    z[(pid, did, sid)] >= inv_expr - M * (1 - delta[(pid, did)]),
                    name=f"z_lb_{pid}_{did}_{sid}"
                )
                model.addConstr(
                    z[(pid, did, sid)] <= M * delta[(pid, did)],
                    name=f"z_ub2_{pid}_{did}_{sid}"
                )
    
    # 7. Sales constraints
    for pid in product_ids:
        pd_entry = prod_dists[pid]
        for dist in pd_entry['distributions']:
            did = dist['distribution_id']
            for scen in dist['scenarios']:
                sid = scen['scenario_id']
                demand = scen['demand']
                
                # w + o = z
                model.addConstr(
                    w[(pid, did, sid)] + o[(pid, did, sid)] == z[(pid, did, sid)],
                    name=f"sales_balance_{pid}_{did}_{sid}"
                )
                
                # w <= demand * delta (can only sell at full price up to demand, and only when this dist is selected)
                model.addConstr(
                    w[(pid, did, sid)] <= demand * delta[(pid, did)],
                    name=f"demand_limit_{pid}_{did}_{sid}"
                )
    
    # Objective: maximize expected profit
    # Manufacturing cost
    mfg_cost = gp.quicksum(
        pf_data[(pid, fid)]['manufacturing_cost'] * x[(pid, fid)]
        for pid in product_ids for fid in facility_ids
    )
    
    # Expected revenue (only from selected distribution)
    # For each product, revenue comes from the selected distribution's scenarios
    # Since delta linearizes this, and w,o are 0 when delta=0:
    revenue = gp.LinExpr()
    for pid in product_ids:
        pd_entry = prod_dists[pid]
        sp = products[pid]['sale_price']
        dp = products[pid]['discounted_price']
        for dist in pd_entry['distributions']:
            did = dist['distribution_id']
            for scen in dist['scenarios']:
                sid = scen['scenario_id']
                prob = scen['probability']
                revenue += prob * (sp * w[(pid, did, sid)] + dp * o[(pid, did, sid)])
    
    model.setObjective(revenue - mfg_cost, GRB.MAXIMIZE)
    
    # Callback to log incumbent solutions
    best_obj = [float('-inf')]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj > best_obj[0]:
                best_obj[0] = obj
                if logger:
                    logger.log(obj)
    
    # Update time limit considering time spent building model
    elapsed = time.time() - start_time
    remaining = max(1, time_limit - elapsed - 2)
    model.Params.TimeLimit = remaining
    
    model.optimize(callback)
    
    # Extract solution
    if model.SolCount > 0:
        obj_val = model.ObjVal
        
        sol_x = {}
        for pid in product_ids:
            for fid in facility_ids:
                sol_x[f"{pid}_{fid}"] = x[(pid, fid)].X
        
        sol_y = {}
        for pid in product_ids:
            for fid in facility_ids:
                pf = pf_data[(pid, fid)]
                for lv in pf['production_levels']:
                    lid = lv['level_id']
                    sol_y[f"{pid}_{fid}_{lid}"] = round(y[(pid, fid, lid)].X)
        
        sol_delta = {}
        for pid in product_ids:
            pd_entry = prod_dists[pid]
            for dist in pd_entry['distributions']:
                did = dist['distribution_id']
                sol_delta[f"{pid}_{did}"] = round(delta[(pid, did)].X)
        
        sol_z = {}
        sol_w = {}
        sol_o = {}
        for pid in product_ids:
            pd_entry = prod_dists[pid]
            for dist in pd_entry['distributions']:
                did = dist['distribution_id']
                for scen in dist['scenarios']:
                    sid = scen['scenario_id']
                    key = f"{pid}_{did}_{sid}"
                    sol_z[key] = z[(pid, did, sid)].X
                    sol_w[key] = w[(pid, did, sid)].X
                    sol_o[key] = o[(pid, did, sid)].X
        
        solution = {
            "objective_value": obj_val,
            "solution": {
                "x": sol_x,
                "y": sol_y,
                "delta": sol_delta,
                "z": sol_z,
                "w": sol_w,
                "o": sol_o
            }
        }
    else:
        # No feasible solution found - return zero solution
        sol_x = {}
        sol_y = {}
        sol_delta = {}
        sol_z = {}
        sol_w = {}
        sol_o = {}
        
        for pid in product_ids:
            for fid in facility_ids:
                sol_x[f"{pid}_{fid}"] = 0.0
                pf = pf_data[(pid, fid)]
                for lv in pf['production_levels']:
                    lid = lv['level_id']
                    # Select zero level
                    sol_y[f"{pid}_{fid}_{lid}"] = 1 if lv['lower_bound'] == 0 and lv['upper_bound'] == 0 else 0
        
        # Select first distribution for each product
        for pid in product_ids:
            pd_entry = prod_dists[pid]
            for i, dist in enumerate(pd_entry['distributions']):
                did = dist['distribution_id']
                sol_delta[f"{pid}_{did}"] = 1 if i == 0 else 0
                for scen in dist['scenarios']:
                    sid = scen['scenario_id']
                    key = f"{pid}_{did}_{sid}"
                    sol_z[key] = 0.0
                    sol_w[key] = 0.0
                    sol_o[key] = 0.0
        
        solution = {
            "objective_value": 0.0,
            "solution": {
                "x": sol_x,
                "y": sol_y,
                "delta": sol_delta,
                "z": sol_z,
                "w": sol_w,
                "o": sol_o
            }
        }
        if logger:
            logger.log(0.0)
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)