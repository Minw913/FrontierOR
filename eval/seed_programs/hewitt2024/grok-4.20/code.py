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

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    num_products = data['num_products']
    num_facilities = data['num_facilities']
    products = data['products']
    facilities = data['facilities']
    product_facility_data = data['product_facility_data']
    products_distributions = data['products_distributions']

    # Build useful lookups
    prod_list = [p['id'] for p in products]
    fac_list = [f['id'] for f in facilities]
    
    # product -> list of distributions
    prod_to_dists = {}
    for pd in products_distributions:
        pid = pd['product_id']
        prod_to_dists[pid] = pd['distributions']
    
    # product, facility -> manufacturing cost, production levels
    pf_data = {}
    for entry in product_facility_data:
        pid = entry['product_id']
        fid = entry['facility_id']
        key = (pid, fid)
        pf_data[key] = {
            'cost': entry['manufacturing_cost'],
            'levels': entry['production_levels'],
            'capacity': next(f['capacity'] for f in facilities if f['id'] == fid)
        }
    
    # For each product, map (level_tuple) -> distribution for yield_and_demand
    prod_level_to_dist = {}
    for pid in prod_list:
        prod_level_to_dist[pid] = {}
        for dist in prod_to_dists[pid]:
            if dist['type'] == 'yield_and_demand':
                level_tuple = tuple(dist['level_combination'])
                prod_level_to_dist[pid][level_tuple] = dist

    # Create model
    model = gp.Model("Production_Planning")
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1
    model.Params.LogToConsole = 0
    model.Params.MIPGap = 1e-4  # Allow 1% gap for larger instances

    # Decision variables
    # x[pid,fid]: raw materials allocated
    x = {}
    for pid in prod_list:
        for fid in fac_list:
            x[pid, fid] = model.addVar(lb=0, name=f"x_{pid}_{fid}")
    
    # y[pid,fid,lid]: binary for selecting production level
    y = {}
    level_dict = {}  # (p,f) -> list of levels
    for entry in product_facility_data:
        pid = entry['product_id']
        fid = entry['facility_id']
        level_dict[(pid, fid)] = entry['production_levels']
    
    for pid in prod_list:
        for fid in fac_list:
            levels = level_dict[(pid, fid)]
            for lev in levels:
                lid = lev['level_id']
                y[pid, fid, lid] = model.addVar(vtype=GRB.BINARY, name=f"y_{pid}_{fid}_{lid}")
    
    # delta[pid, did]: binary selecting distribution (only yield_and_demand ones)
    delta = {}
    dist_dict = {}  # pid -> list of yield_and_demand dists
    for pid in prod_list:
        dist_dict[pid] = [d for d in prod_to_dists[pid] if d['type'] == 'yield_and_demand']
        for d in dist_dict[pid]:
            did = d['distribution_id']
            delta[pid, did] = model.addVar(vtype=GRB.BINARY, name=f"delta_{pid}_{did}")
    
    # For each product, exactly one distribution and compatible levels
    for pid in prod_list:
        model.addConstr(gp.quicksum(delta[pid, d['distribution_id']] for d in dist_dict[pid]) == 1)
        
        # Link levels to chosen distribution
        levels_for_p = level_dict[(pid, fac_list[0])]  # assume same levels across facilities
        for lev in levels_for_p:
            lid = lev['level_id']
            # Find distributions that require this level at each facility
            for fid_idx, fid in enumerate(fac_list):
                relevant_deltas = []
                for d in dist_dict[pid]:
                    level_comb = d['level_combination']
                    if level_comb[fid_idx] == lid:
                        relevant_deltas.append(delta[pid, d['distribution_id']])
                if relevant_deltas:
                    model.addConstr(y[pid, fid, lid] == gp.quicksum(relevant_deltas))
    
    # For each product-facility, exactly one level selected
    for pid in prod_list:
        for fid in fac_list:
            levels = level_dict[(pid, fid)]
            lids = [lev['level_id'] for lev in levels]
            model.addConstr(gp.quicksum(y[pid, fid, lid] for lid in lids) == 1)
    
    # Production level bounds on x
    for pid in prod_list:
        for fid in fac_list:
            levels = level_dict[(pid, fid)]
            for lev in levels:
                lid = lev['level_id']
                lb = lev['lower_bound']
                ub = lev['upper_bound']
                model.addConstr(x[pid, fid] >= lb * y[pid, fid, lid])
                model.addConstr(x[pid, fid] <= ub * y[pid, fid, lid])
    
    # Capacity constraints
    for fid in fac_list:
        cap = next(f['capacity'] for f in facilities if f['id'] == fid)
        model.addConstr(gp.quicksum(x[pid, fid] for pid in prod_list) <= cap)
    
    # Now for second stage: for each product and each possible distribution, we need z, w, o per scenario
    # But to avoid too many variables, we only activate variables for the chosen distribution via big-M or use auxiliary
    
    # Since delta selects exactly one distribution, we can create variables per distribution and link them
    
    z_vars = {}  # (pid, did, sid) -> inventory
    w_vars = {}  # full price sales
    o_vars = {}  # discounted sales
    
    # Also keep track of revenue contribution per product per distribution
    revenue_terms = {}
    
    M = 1e6  # Big M, assuming quantities won't exceed this (can be tightened but for now)
    
    for pidx, pid in enumerate(prod_list):
        prod_info = next(p for p in products if p['id'] == pid)
        sale_price = prod_info['sale_price']
        disc_price = prod_info['discounted_price']
        
        for dist in prod_to_dists[pid]:
            did = dist['distribution_id']
            is_yield = dist['type'] == 'yield_and_demand'
            scenarios = dist['scenarios']
            
            # Only create variables for yield_and_demand as per schema, but demand_only ignored in revenue?
            # According to problem, only selected yield_and_demand contribute. Demand-only not used in delta.
            
            if not is_yield:
                continue
                
            for scen in scenarios:
                sid = scen['scenario_id']
                prob = scen['probability']
                
                z_var = model.addVar(lb=0, name=f"z_{pid}_{did}_{sid}")
                w_var = model.addVar(lb=0, name=f"w_{pid}_{did}_{sid}")
                o_var = model.addVar(lb=0, name=f"o_{pid}_{did}_{sid}")
                
                z_vars[pid, did, sid] = z_var
                w_vars[pid, did, sid] = w_var
                o_vars[pid, did, sid] = o_var
                
                # Link to delta
                model.addConstr(z_var <= M * delta[pid, did])
                model.addConstr(w_var <= M * delta[pid, did])
                model.addConstr(o_var <= M * delta[pid, did])
                
                # Inventory definition when selected
                if is_yield:
                    yield_sum = 0
                    for fidx, fid in enumerate(fac_list):
                        yld = scen['yields'][str(fid)]
                        yield_sum += yld * x[pid, fid]
                    model.addConstr(z_var == yield_sum - M * (1 - delta[pid, did]))
                
                # Sales constraints
                demand = scen['demand']
                model.addConstr(w_var <= demand)
                model.addConstr(w_var + o_var == z_var)
                model.addConstr(w_var <= z_var)
                model.addConstr(o_var <= z_var)
            
            # Expected revenue for this distribution
            exp_rev = 0
            for scen in scenarios:
                sid = scen['scenario_id']
                prob = scen['probability']
                exp_rev += prob * (sale_price * w_vars[pid, did, sid] + disc_price * o_vars[pid, did, sid])
            revenue_terms[pid, did] = exp_rev
    
    # Objective: sum expected revenue - manufacturing cost
    total_exp_rev = gp.quicksum(revenue_terms[pid, did] for pid in prod_list for did in [d['distribution_id'] for d in dist_dict[pid]])
    manuf_cost = gp.quicksum(pf_data[(pid, fid)]['cost'] * x[pid, fid] for pid in prod_list for fid in fac_list)
    model.setObjective(total_exp_rev - manuf_cost, GRB.MAXIMIZE)
    
    # Callback for logging
    best_obj = -float('inf')
    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj > best_obj + 1e-6:
                best_obj = obj
                if logger:
                    logger.log(float(obj))
    
    model.optimize(callback)
    
    # Extract solution
    if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
        sol_x = {}
        sol_y = {}
        sol_delta = {}
        sol_z = {}
        sol_w = {}
        sol_o = {}
        
        for pid in prod_list:
            for fid in fac_list:
                val = x[pid, fid].X
                if val > 1e-8:
                    sol_x[f"{pid}_{fid}"] = float(val)
                
                levels = level_dict[(pid, fid)]
                for lev in levels:
                    lid = lev['level_id']
                    if y[pid, fid, lid].X > 0.5:
                        sol_y[f"{pid}_{fid}_{lid}"] = 1
                    else:
                        sol_y[f"{pid}_{fid}_{lid}"] = 0
        
        for pid in prod_list:
            for d in dist_dict[pid]:
                did = d['distribution_id']
                if delta[pid, did].X > 0.5:
                    sol_delta[f"{pid}_{did}"] = 1
                else:
                    sol_delta[f"{pid}_{did}"] = 0
        
        for key in z_vars:
            pid, did, sid = key
            sol_z[f"{pid}_{did}_{sid}"] = float(z_vars[key].X)
            sol_w[f"{pid}_{did}_{sid}"] = float(w_vars[key].X)
            sol_o[f"{pid}_{did}_{sid}"] = float(o_vars[key].X)
        
        objective = model.ObjVal if model.SolCount > 0 else 0.0
        
        solution = {
            "objective_value": float(objective),
            "solution": {
                "x": sol_x,
                "y": sol_y,
                "delta": sol_delta,
                "z": sol_z,
                "w": sol_w,
                "o": sol_o
            }
        }
        
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # Infeasible or no solution
        solution = {
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
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()