import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main(instance_path, solution_path, time_limit, log_path):
    # Read instance data
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    # Create mappings for products and facilities
    products_by_id = {p['id']: p for p in data['products']}
    facilities_by_id = {f['id']: f for f in data['facilities']}
    
    # Create a list of facility ids in the order they appear in the input
    facility_ids_ordered = [f['id'] for f in data['facilities']]
    
    # Create a mapping for product-facility data
    product_facility_by_pf = {}
    for entry in data['product_facility_data']:
        key = (entry['product_id'], entry['facility_id'])
        product_facility_by_pf[key] = entry
    
    # Create a mapping for distributions per product
    products_distributions_by_product = {}
    for dist_set in data['products_distributions']:
        product_id = dist_set['product_id']
        products_distributions_by_product[product_id] = dist_set
    
    # Precompute big M for each product
    product_big_M = {}
    for product_id in products_by_id:
        M_i = 0
        distributions = products_distributions_by_product[product_id]['distributions']
        for dist in distributions:
            if dist['type'] == 'yield_and_demand':
                for scenario in dist['scenarios']:
                    z_temp = 0.0
                    for fac_id in facility_ids_ordered:
                        # Get yield for facility j in this scenario
                        yield_js = scenario['yields'].get(str(fac_id), 0.0)
                        cap = facilities_by_id[fac_id]['capacity']
                        z_temp += yield_js * cap
                    if z_temp > M_i:
                        M_i = z_temp
            else:  # 'demand_only'
                z_temp = 0.0
                for fac_id in facility_ids_ordered:
                    key = (product_id, fac_id)
                    if key in product_facility_by_pf:
                        levels = product_facility_by_pf[key]['production_levels']
                        max_yield = max(level['yield_mean'] for level in levels)
                        cap = facilities_by_id[fac_id]['capacity']
                        z_temp += max_yield * cap
                    else:
                        # If no data, assume 0
                        pass
                if z_temp > M_i:
                    M_i = z_temp
        product_big_M[product_id] = M_i
    
    # Initialize Gurobi model
    model = gp.Model('ProductionPlanning')
    model.setParam('OutputFlag', 1)
    model.setParam('TimeLimit', time_limit)
    
    # Create variables
    x_vars = {}  # (product_id, facility_id, level_id) -> var
    y_vars = {}  # (product_id, facility_id, level_id) -> binary var
    delta_vars = {}  # (product_id, distribution_id) -> binary var
    w_vars = {}  # (product_id, distribution_id, scenario_id) -> var
    o_vars = {}  # (product_id, distribution_id, scenario_id) -> var
    aux_vars = {}  # (product_id, distribution_id, scenario_id) -> var
    
    # Create variables for x and y
    for product_id in products_by_id:
        for facility_id in facility_ids_ordered:
            key = (product_id, facility_id)
            if key in product_facility_by_pf:
                levels = product_facility_by_pf[key]['production_levels']
                for level in levels:
                    level_id = level['level_id']
                    x_key = (product_id, facility_id, level_id)
                    x_vars[x_key] = model.addVar(lb=0.0, name=f'x_{product_id}_{facility_id}_{level_id}')
                    y_vars[x_key] = model.addVar(vtype=GRB.BINARY, name=f'y_{product_id}_{facility_id}_{level_id}')
    
    # Create delta variables
    for product_id in products_by_id:
        distributions = products_distributions_by_product[product_id]['distributions']
        for dist in distributions:
            dist_id = dist['distribution_id']
            delta_vars[(product_id, dist_id)] = model.addVar(vtype=GRB.BINARY, name=f'delta_{product_id}_{dist_id}')
    
    # Create w, o, aux variables for each product, distribution, and scenario
    for product_id in products_by_id:
        distributions = products_distributions_by_product[product_id]['distributions']
        for dist in distributions:
            dist_id = dist['distribution_id']
            for scenario in dist['scenarios']:
                scenario_id = scenario['scenario_id']
                w_key = (product_id, dist_id, scenario_id)
                w_vars[w_key] = model.addVar(lb=0.0, name=f'w_{product_id}_{dist_id}_{scenario_id}')
                o_vars[w_key] = model.addVar(lb=0.0, name=f'o_{product_id}_{dist_id}_{scenario_id}')
                aux_vars[w_key] = model.addVar(lb=0.0, name=f'aux_{product_id}_{dist_id}_{scenario_id}')
    
    model.update()
    
    # Add constraints
    # 1. Exactly one production level per product-facility
    for product_id in products_by_id:
        for facility_id in facility_ids_ordered:
            key = (product_id, facility_id)
            if key in product_facility_by_pf:
                levels = product_facility_by_pf[key]['production_levels']
                y_list = []
                for level in levels:
                    level_id = level['level_id']
                    y_key = (product_id, facility_id, level_id)
                    y_list.append(y_vars[y_key])
                model.addConstr(gp.quicksum(y_list) == 1, name=f'y_sum_{product_id}_{facility_id}')
    
    # 2. Bound constraints for x and link to y
    for (product_id, facility_id, level_id), x_var in x_vars.items():
        key = (product_id, facility_id)
        levels = product_facility_by_pf[key]['production_levels']
        level_info = next(lev for lev in levels if lev['level_id'] == level_id)
        lb = level_info['lower_bound']
        ub = level_info['upper_bound']
        y_var = y_vars[(product_id, facility_id, level_id)]
        model.addConstr(x_var >= lb * y_var, name=f'x_lb_{product_id}_{facility_id}_{level_id}')
        model.addConstr(x_var <= ub * y_var, name=f'x_ub_{product_id}_{facility_id}_{level_id}')
    
    # 3. For yield_and_demand distributions, force production levels
    for product_id in products_by_id:
        distributions = products_distributions_by_product[product_id]['distributions']
        for dist in distributions:
            if dist['type'] == 'yield_and_demand':
                dist_id = dist['distribution_id']
                level_combination = dist['level_combination']
                for idx, facility_id in enumerate(facility_ids_ordered):
                    required_level_id = level_combination[idx]
                    y_key = (product_id, facility_id, required_level_id)
                    if y_key in y_vars:
                        model.addConstr(y_vars[y_key] >= delta_vars[(product_id, dist_id)],
                                       name=f'delta_force_{product_id}_{dist_id}_{facility_id}')
    
    # 4. Exactly one distribution per product
    for product_id in products_by_id:
        distributions = products_distributions_by_product[product_id]['distributions']
        delta_list = [delta_vars[(product_id, dist['distribution_id'])] for dist in distributions]
        model.addConstr(gp.quicksum(delta_list) == 1, name=f'delta_sum_{product_id}')
    
    # 5. Facility capacity constraints
    for facility_id in facility_ids_ordered:
        cap = facilities_by_id[facility_id]['capacity']
        x_sum = 0.0
        for product_id in products_by_id:
            key = (product_id, facility_id)
            if key in product_facility_by_pf:
                levels = product_facility_by_pf[key]['production_levels']
                for level in levels:
                    level_id = level['level_id']
                    x_key = (product_id, facility_id, level_id)
                    if x_key in x_vars:
                        x_sum += x_vars[x_key]
        model.addConstr(x_sum <= cap, name=f'capacity_{facility_id}')
    
    # 6. Define aux_{ids} = z_{ids} * delta_{id} using big-M
    for product_id in products_by_id:
        M_i = product_big_M[product_id]
        distributions = products_distributions_by_product[product_id]['distributions']
        for dist in distributions:
            dist_id = dist['distribution_id']
            for scenario in dist['scenarios']:
                scenario_id = scenario['scenario_id']
                aux_key = (product_id, dist_id, scenario_id)
                delta_var = delta_vars[(product_id, dist_id)]
                aux_var = aux_vars[aux_key]
                # Compute z_{ids} expression
                z_expr = 0.0
                if dist['type'] == 'yield_and_demand':
                    for facility_id in facility_ids_ordered:
                        key = (product_id, facility_id)
                        if key in product_facility_by_pf:
                            # Sum x over levels for this facility
                            x_fac_sum = 0.0
                            levels = product_facility_by_pf[key]['production_levels']
                            for level in levels:
                                level_id = level['level_id']
                                x_key = (product_id, facility_id, level_id)
                                if x_key in x_vars:
                                    x_fac_sum += x_vars[x_key]
                            # Get yield for this facility and scenario
                            yield_js = scenario['yields'].get(str(facility_id), 0.0)
                            z_expr += yield_js * x_fac_sum
                else:  # 'demand_only'
                    for facility_id in facility_ids_ordered:
                        key = (product_id, facility_id)
                        if key in product_facility_by_pf:
                            levels = product_facility_by_pf[key]['production_levels']
                            for level in levels:
                                level_id = level['level_id']
                                x_key = (product_id, facility_id, level_id)
                                if x_key in x_vars:
                                    yield_mean = level['yield_mean']
                                    z_expr += yield_mean * x_vars[x_key]
                # Add constraints for aux
                model.addConstr(aux_var <= M_i * delta_var, name=f'aux_ub_{aux_key}')
                model.addConstr(aux_var <= z_expr, name=f'aux_z_ub_{aux_key}')
                model.addConstr(aux_var >= z_expr - M_i * (1 - delta_var), name=f'aux_z_lb_{aux_key}')
    
    # 7. Sales constraints
    for product_id in products_by_id:
        sale_price = products_by_id[product_id]['sale_price']
        discounted_price = products_by_id[product_id]['discounted_price']
        distributions = products_distributions_by_product[product_id]['distributions']
        for dist in distributions:
            dist_id = dist['distribution_id']
            for scenario in dist['scenarios']:
                scenario_id = scenario['scenario_id']
                w_key = (product_id, dist_id, scenario_id)
                w_var = w_vars[w_key]
                o_var = o_vars[w_key]
                aux_var = aux_vars[w_key]
                demand = scenario['demand']
                # w + o = aux
                model.addConstr(w_var + o_var == aux_var, name=f'sales_eq_{w_key}')
                # w <= demand
                model.addConstr(w_var <= demand, name=f'w_ub_{w_key}')
    
    # Set objective
    revenue_expr = 0.0
    for product_id in products_by_id:
        distributions = products_distributions_by_product[product_id]['distributions']
        for dist in distributions:
            dist_id = dist['distribution_id']
            delta_var = delta_vars[(product_id, dist_id)]
            for scenario in dist['scenarios']:
                scenario_id = scenario['scenario_id']
                prob = scenario['probability']
                w_var = w_vars[(product_id, dist_id, scenario_id)]
                o_var = o_vars[(product_id, dist_id, scenario_id)]
                sale_price = products_by_id[product_id]['sale_price']
                discounted_price = products_by_id[product_id]['discounted_price']
                revenue_expr += prob * (sale_price * w_var + discounted_price * o_var)
    
    cost_expr = 0.0
    for (product_id, facility_id, level_id), x_var in x_vars.items():
        key = (product_id, facility_id)
        manufacturing_cost = product_facility_by_pf[key]['manufacturing_cost']
        cost_expr += manufacturing_cost * x_var
    
    model.setObjective(revenue_expr - cost_expr, GRB.MAXIMIZE)
    
    # Initialize logger and callback for intermediate solutions
    if log_path:
        logger = SolutionLogger(log_path, sense="maximize")
    else:
        logger = None
    
    last_logged_obj = None
    
    def log_callback(model, where):
        nonlocal last_logged_obj
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if last_logged_obj is None or obj > last_logged_obj:
                last_logged_obj = obj
                if logger:
                    logger.log(obj)
    
    # Optimize
    if logger:
        model._last_logged_obj = None
        model.optimize(log_callback)
    else:
        model.optimize()
    
    # Check status and save solution
    solution = {}
    solution['x'] = {}
    solution['y'] = {}
    solution['delta'] = {}
    solution['z'] = {}
    solution['w'] = {}
    solution['o'] = {}
    
    # Extract x: total allocation per product-facility
    for product_id in products_by_id:
        for facility_id in facility_ids_ordered:
            key = (product_id, facility_id)
            if key in product_facility_by_pf:
                x_sum = 0.0
                levels = product_facility_by_pf[key]['production_levels']
                for level in levels:
                    level_id = level['level_id']
                    x_key = (product_id, facility_id, level_id)
                    if x_key in x_vars:
                        x_val = x_vars[x_key].X
                        x_sum += x_val
                dict_key = f'{product_id}_{facility_id}'
                solution['x'][dict_key] = x_sum
    
    # Extract y: selected level per product-facility
    for (product_id, facility_id, level_id), y_var in y_vars.items():
        y_val = y_var.X
        dict_key = f'{product_id}_{facility_id}_{level_id}'
        solution['y'][dict_key] = int(round(y_val))
    
    # Extract delta: selected distribution per product
    for (product_id, dist_id), delta_var in delta_vars.items():
        delta_val = delta_var.X
        dict_key = f'{product_id}_{dist_id}'
        solution['delta'][dict_key] = int(round(delta_val))
    
    # Extract z, w, o: for every product, distribution, scenario
    for product_id in products_by_id:
        distributions = products_distributions_by_product[product_id]['distributions']
        for dist in distributions:
            dist_id = dist['distribution_id']
            for scenario in dist['scenarios']:
                scenario_id = scenario['scenario_id']
                dict_key = f'{product_id}_{dist_id}_{scenario_id}'
                # Compute z from the solution
                z_val = 0.0
                if dist['type'] == 'yield_and_demand':
                    for facility_id in facility_ids_ordered:
                        key = (product_id, facility_id)
                        if key in product_facility_by_pf:
                            # Total allocation for product i at facility j
                            x_fac_sum = 0.0
                            levels = product_facility_by_pf[key]['production_levels']
                            for level in levels:
                                level_id = level['level_id']
                                x_key = (product_id, facility_id, level_id)
                                if x_key in x_vars:
                                    x_val = x_vars[x_key].X
                                    x_fac_sum += x_val
                            yield_js = scenario['yields'].get(str(facility_id), 0.0)
                            z_val += yield_js * x_fac_sum
                else:  # 'demand_only'
                    for facility_id in facility_ids_ordered:
                        key = (product_id, facility_id)
                        if key in product_facility_by_pf:
                            levels = product_facility_by_pf[key]['production_levels']
                            for level in levels:
                                level_id = level['level_id']
                                x_key = (product_id, facility_id, level_id)
                                if x_key in x_vars:
                                    x_val = x_vars[x_key].X
                                    yield_mean = level['yield_mean']
                                    z_val += yield_mean * x_val
                solution['z'][dict_key] = z_val
                w_val = w_vars.get((product_id, dist_id, scenario_id), None)
                if w_val is not None:
                    solution['w'][dict_key] = w_val.X
                else:
                    solution['w'][dict_key] = 0.0
                o_val = o_vars.get((product_id, dist_id, scenario_id), None)
                if o_val is not None:
                    solution['o'][dict_key] = o_val.X
                else:
                    solution['o'][dict_key] = 0.0
    
    # Write solution
    output = {
        "objective_value": model.ObjVal if model.Status == GRB.OPTIMAL else model.ObjBound,
        "solution": solution
    }
    
    with open(solution_path, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    main(args.instance_path, args.solution_path, args.time_limit, args.log_path)