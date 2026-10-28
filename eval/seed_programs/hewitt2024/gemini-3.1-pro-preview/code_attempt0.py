import json
import argparse
import sys
import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser(description="Solve manufacturing allocation and distribution selection problem.")
    parser.add_argument('--instance_path', type=str, required=True, help="Path to the instance JSON file.")
    parser.add_argument('--solution_path', type=str, required=True, help="Path to save the solution JSON file.")
    parser.add_argument('--time_limit', type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument('--log_path', type=str, default=None, help="Path to the log file for solution logger.")
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="maximize")
        except ImportError:
            pass

    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
        
    products = instance['products']
    facilities = instance['facilities']
    product_facility_data = instance['product_facility_data']
    products_distributions = instance['products_distributions']
    
    product_list = [p['id'] for p in products]
    facility_list = [f['id'] for f in facilities]
    
    products_dict = {p['id']: p for p in products}
    facilities_dict = {f['id']: f for f in facilities}
    facility_indices = {facilities[i]['id']: i for i in range(len(facilities))}
    
    # Organize product-facility data
    levels_dict = {p: {f: [] for f in facility_list} for p in product_list}
    cost_dict = {p: {f: 0.0 for f in facility_list} for p in product_list}
    
    for pf in product_facility_data:
        p = pf['product_id']
        f = pf['facility_id']
        cost_dict[p][f] = pf['manufacturing_cost']
        levels_dict[p][f] = pf['production_levels']
        
    dists = {p: [] for p in product_list}
    for pd in products_distributions:
        p = pd['product_id']
        dists[p] = pd['distributions']
        
    # Gurobi Model
    model = gp.Model("Production_Planning")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    # model.setParam('OutputFlag', 0) # Optionally silence output
    
    # Callback for logger
    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)
                
    # Variables definition
    x_vars = {}
    y_vars = {}
    delta_vars = {}
    R_rev_vars = {}
    
    for p in product_list:
        R_rev_vars[p] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"R_rev_{p}")
        for f in facility_list:
            x_vars[p, f] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_{p}_{f}")
            for lvl in levels_dict[p][f]:
                y_vars[p, f, lvl['level_id']] = model.addVar(vtype=GRB.BINARY, name=f"y_{p}_{f}_{lvl['level_id']}")
                
        for d in dists[p]:
            delta_vars[p, d['distribution_id']] = model.addVar(vtype=GRB.BINARY, name=f"delta_{p}_{d['distribution_id']}")
            
    # Constraints
    
    # 1. Total Facility Capacity constraint
    for f in facility_list:
        model.addConstr(gp.quicksum(x_vars[p, f] for p in product_list) <= facilities_dict[f]['capacity'], name=f"cap_{f}")
        
    # 2. Allocation Bounds and Distribution mapping
    for p in product_list:
        # Exactly one distribution must be selected
        model.addConstr(gp.quicksum(delta_vars[p, d['distribution_id']] for d in dists[p]) == 1, name=f"one_dist_{p}")
        
        # Identify zero level for demand_only handling
        zero_levels = {}
        for f in facility_list:
            z_id = None
            for lvl in levels_dict[p][f]:
                if lvl['lower_bound'] == 0 and lvl['upper_bound'] == 0:
                    z_id = lvl['level_id']
                    break
            zero_levels[f] = z_id
            
        for f in facility_list:
            f_idx = facility_indices[f]
            for lvl in levels_dict[p][f]:
                l_id = lvl['level_id']
                matching_d = []
                for d in dists[p]:
                    if d['type'] == 'yield_and_demand':
                        if d['level_combination'][f_idx] == l_id:
                            matching_d.append(delta_vars[p, d['distribution_id']])
                    elif d['type'] == 'demand_only':
                        if l_id == zero_levels[f]:
                            matching_d.append(delta_vars[p, d['distribution_id']])
                
                # Link chosen distribution mapped-levels logic
                model.addConstr(y_vars[p, f, l_id] == gp.quicksum(matching_d), name=f"map_{p}_{f}_{l_id}")
                
            # x bound linking logic
            lb_expr = gp.quicksum(lvl['lower_bound'] * y_vars[p, f, lvl['level_id']] for lvl in levels_dict[p][f])
            ub_expr = gp.quicksum(lvl['upper_bound'] * y_vars[p, f, lvl['level_id']] for lvl in levels_dict[p][f])
            
            model.addConstr(x_vars[p, f] >= lb_expr, name=f"lb_{p}_{f}")
            model.addConstr(x_vars[p, f] <= ub_expr, name=f"ub_{p}_{f}")

    # 3. Revenue links & Scenario handling
    for p in product_list:
        price = products_dict[p]['sale_price']
        discount = products_dict[p]['discounted_price']
        
        # Calculate optimal big-M to safely bound R_rev constraints
        max_UB = {}
        for f in facility_list:
            max_UB[f] = max(lvl['upper_bound'] for lvl in levels_dict[p][f])
            
        M_p = 0.0
        for d in dists[p]:
            expected_max_rev = 0.0
            for s in d['scenarios']:
                if d['type'] == 'yield_and_demand':
                    z_max = sum(s['yields'].get(str(f), 0.0) * max_UB[f] for f in facility_list)
                else:
                    z_max = 0.0
                w_max = min(z_max, float(s['demand']))
                o_max = z_max - w_max
                rev = price * w_max + discount * o_max
                expected_max_rev += s['probability'] * rev
            if expected_max_rev > M_p:
                M_p = expected_max_rev
                
        M_p = M_p * 1.01 + 10.0 # Buffer safely
        
        for d in dists[p]:
            rev_expr = gp.LinExpr()
            d_id = d['distribution_id']
            for s in d['scenarios']:
                s_id = s['scenario_id']
                if d['type'] == 'yield_and_demand':
                    Z = gp.quicksum(s['yields'].get(str(f), 0.0) * x_vars[p, f] for f in facility_list)
                else:
                    Z = 0.0
                    
                w_var = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=float(s['demand']), name=f"w_{p}_{d_id}_{s_id}")
                
                if type(Z) is float and Z == 0.0:
                    model.addConstr(w_var <= 0.0, name=f"w_Z_{p}_{d_id}_{s_id}")
                else:
                    model.addConstr(w_var <= Z, name=f"w_Z_{p}_{d_id}_{s_id}")
                
                rev_expr += s['probability'] * ( (price - discount) * w_var + discount * Z )
                
            model.addConstr(R_rev_vars[p] <= rev_expr + M_p * (1 - delta_vars[p, d_id]), name=f"R_rev_{p}_{d_id}")

    # Objective logic
    total_cost = gp.quicksum(cost_dict[p][f] * x_vars[p, f] for p in product_list for f in facility_list)
    total_revenue = gp.quicksum(R_rev_vars[p] for p in product_list)
    
    model.setObjective(total_revenue - total_cost, GRB.MAXIMIZE)
    
    # Solve Problem
    model.optimize(cb)
    
    # Store solutions & Post-parse logic matches exactly Output Schema Specification
    if model.SolCount > 0:
        sol_x = {}
        sol_y = {}
        sol_delta = {}
        sol_z = {}
        sol_w = {}
        sol_o = {}

        # Capture decisions
        for p in product_list:
            for f in facility_list:
                val = max(0.0, float(x_vars[p, f].X))
                sol_x[f"{p}_{f}"] = val
                for l in levels_dict[p][f]:
                    y_val = int(round(y_vars[p, f, l['level_id']].X))
                    sol_y[f"{p}_{f}_{l['level_id']}"] = y_val

        true_rev = 0.0
        true_cost = 0.0
        
        for p in product_list:
            selected_d_id = None
            for d in dists[p]:
                d_id = d['distribution_id']
                delta_val = int(round(delta_vars[p, d_id].X))
                sol_delta[f"{p}_{d_id}"] = delta_val
                if delta_val == 1:
                    selected_d_id = d_id
                    
            price = products_dict[p]['sale_price']
            discount = products_dict[p]['discounted_price']
            
            # Formally calculate exactly realized Z, W, O to eliminate tiny numerical ambiguities and fill Schema
            for d in dists[p]:
                d_id = d['distribution_id']
                for s in d['scenarios']:
                    s_id = s['scenario_id']
                    if d['type'] == 'yield_and_demand':
                        z_val = sum(s['yields'].get(str(f), 0.0) * sol_x[f"{p}_{f}"] for f in facility_list)
                    else:
                        z_val = 0.0
                        
                    w_val = min(z_val, float(s['demand']))
                    o_val = z_val - w_val
                    
                    sol_z[f"{p}_{d_id}_{s_id}"] = float(z_val)
                    sol_w[f"{p}_{d_id}_{s_id}"] = float(w_val)
                    sol_o[f"{p}_{d_id}_{s_id}"] = float(o_val)
                    
                    if d_id == selected_d_id:
                        true_rev += float(s['probability']) * (price * w_val + discount * o_val)

        for p in product_list:
            for f in facility_list:
                true_cost += cost_dict[p][f] * sol_x[f"{p}_{f}"]
                
        expected_profit = true_rev - true_cost

        # Save formal formatted projection matching output schema strict constraints
        solution = {
            "objective_value": float(expected_profit),
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
            json.dump(solution, f, indent=4)

if __name__ == '__main__':
    main()