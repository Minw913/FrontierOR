import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance


def solve_instance(instance, time_limit):
    # Extract data from instance
    num_products = instance['num_products']
    num_facilities = instance['num_facilities']

    products = {p['id']: p for p in instance['products']}
    facilities = {f['id']: f for f in instance['facilities']}

    # Create mappings for product-facility data
    product_facility_data = {}  # Maps (product_id, facility_id) to product_facility_data entry
    # Initialize all combinations for safety
    for prod_data in instance['product_facility_data']:
        p_id = prod_data['product_id']
        f_id = prod_data['facility_id']
        product_facility_data[(p_id, f_id)] = prod_data

    # Distribution data
    distribution_map = {}  # Maps (prod_id, dist_id) to distribution object for easy lookup
    distribution_lookup = {}  # Maps product_id to list of its distributions
    for prod_dist in instance['products_distributions']:
        prod_id = prod_dist['product_id']
        distribution_lookup[prod_id] = []
        for dist in prod_dist['distributions']:
            dist_id = dist['distribution_id']
            distribution_map[(prod_id, dist_id)] = dist
            distribution_lookup[prod_id].append(dist)
    
    # Create model
    model = gp.Model("Stochastic Production Planning")
    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 1e-4)  # Ensure better optimality gap
    
    # Decision Variables
    x = {}  # Raw material allocation: (p, f) -> continuous
    y = {}  # Production level selection: (p, f, l) -> binary
    delta = {}  # Distribution selection: (p, d) -> binary
    
    # Scenario-dependent variables
    z = {}  # Available inventory: (p, d, s) -> continuous
    w = {}  # Full price sales: (p, d, s) -> continuous
    o = {}  # Discounted sales: (p, d, s) -> continuous

    # Initialize variables
    for p_id in range(num_products):
        for f_id in range(num_facilities):
            prod_fac_data = product_facility_data.get((p_id, f_id))
            if prod_fac_data:
                for level in prod_fac_data['production_levels']:
                    l = level['level_id']
                    y[(p_id, f_id, l)] = model.addVar(vtype=GRB.BINARY, name=f"y_{p_id}_{f_id}_{l}")

            x[(p_id, f_id)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"x_{p_id}_{f_id}")

    for p_id in range(num_products):
        for dist in distribution_lookup[p_id]:
            d = dist['distribution_id']
            delta[(p_id, d)] = model.addVar(vtype=GRB.BINARY, name=f"delta_{p_id}_{d}")

            for scenario in dist['scenarios']:
                s = scenario['scenario_id']
                z[(p_id, d, s)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"z_{p_id}_{d}_{s}")
                w[(p_id, d, s)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"w_{p_id}_{d}_{s}")
                o[(p_id, d, s)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"o_{p_id}_{d}_{s}")

    # Constraints to tie allocation variables with production level selection
    for p_id in range(num_products):
        for f_id in range(num_facilities):
            prod_fac_data = product_facility_data.get((p_id, f_id), None)
            if prod_fac_data:
                # Exactly one production level per facility-product pair
                model.addConstr(
                    gp.quicksum(y[(p_id, f_id, level['level_id'])] for level in prod_fac_data['production_levels']) == 1,
                    name=f"one_level_{p_id}_{f_id}")

                # Link allocation with level selection
                for level in prod_fac_data['production_levels']:
                    l = level['level_id']
                    # Ensure x[p,f] respects bounds based on selected level
                    model.addConstr(
                        x[(p_id, f_id)] >= level['lower_bound'] * y[(p_id, f_id, l)],
                        name=f"alloc_lb_{p_id}_{f_id}_{l}")

                    upper_bound = level['upper_bound']
                    # Handle unbounded case - assume some high value if upper bound is infinity
                    if upper_bound == float('inf') or upper_bound >= 1e100:
                        # Use facility capacity as an upper bound (or total product demand?)
                        effective_upper_bound = facilities[f_id]['capacity']
                    else:
                        effective_upper_bound = upper_bound
                            
                    model.addConstr(
                        x[(p_id, f_id)] <= effective_upper_bound * y[(p_id, f_id, l)] + (1-y[(p_id, f_id, l)]) * max(facilities[ff]['capacity'] for ff in facilities),
                        name=f"alloc_ub_{p_id}_{f_id}_{l}")

    # Capacity constraints
    for f_id in range(num_facilities):
        model.addConstr(gp.quicksum(x[(p_id, f_id)] for p_id in range(num_products)) <= facilities[f_id]['capacity'],
                       name=f"capacity_{f_id}")

    # Select exactly one distribution for each product
    for p_id in range(num_products):
        model.addConstr(gp.quicksum(delta[(p_id, d['distribution_id'])] for d in distribution_lookup[p_id]) == 1,
                       name=f"select_one_dist_{p_id}")

    # Connect allocation decisions to distribution eligibility
    for p_id in range(num_products):
        for dist in distribution_lookup[p_id]:
            d_id = dist['distribution_id']
            if dist['type'] == 'yield_and_demand' and dist.get('level_combination'):
                # Verify that level combination matches the number of facilities
                if dist.get('level_combination') is not None and len(dist['level_combination']) == num_facilities:
                    level_combo = dist['level_combination']
                    for f_idx, l in enumerate(level_combo):
                        # Get corresponding binary variable
                        if (p_id, f_idx, l) in y:
                            model.addConstr(delta[(p_id, d_id)] <= y[(p_id, f_idx, l)],
                                           name=f"link_dist_level_{p_id}_{d_id}_{f_idx}_{l}")

    # Link inventory with allocation and yield for 'yield_and_demand' distributions
    for p_id in range(num_products):
        for dist in distribution_lookup[p_id]:
            d_id = dist['distribution_id']
            if dist['type'] == 'yield_and_demand':
                for scenario in dist['scenarios']:
                    s_id = scenario['scenario_id']
                        
                    # Build the expression for the total yield
                    inventory_expr = gp.LinExpr()
                    # Yields keys might be string representations of IDs
                    for f_idx in range(num_facilities):
                        # Key in the yields dict is string representation of facility_id
                        fac_key = str(f_idx)
                        if fac_key in scenario.get('yields', {}):
                            yield_factor = scenario['yields'][fac_key]
                            inventory_expr += x[(p_id, f_idx)] * yield_factor
                    
                    # For this scenario, express that inventory is sum of yields if distribution is selected
                    # The inventory should only exist if the distribution is activated
                    # Inventory is the realized inventory in this scenario based on actual allocated and yields
                    # So we need inventory = delta*distribution_selected * (sum of realized yields) 
                    model.addConstr(z[(p_id, d_id, s_id)] <= (
                        gp.quicksum(x[(p_id, f_idx)] * scenario['yields'].get(str(f_idx), 0)
                                    for f_idx in range(num_facilities)) + 
                        (1 - delta[(p_id, d_id)]) * 1e6  # Big M when distribution not selected
                    ), name=f"inv_calc_ub_{p_id}_{d_id}_{s_id}")
                    
                    model.addConstr(z[(p_id, d_id, s_id)] >= (
                        gp.quicksum(x[(p_id, f_idx)] * scenario['yields'].get(str(f_idx), 0)
                                    for f_idx in range(num_facilities)) - 
                        (1 - delta[(p_id, d_id)]) * 1e6  # Big M when distribution not selected
                    ), name=f"inv_calc_lb_{p_id}_{d_id}_{s_id}")

    # Sales constraints
    for p_id in range(num_products):
        for dist in distribution_lookup[p_id]:
            d_id = dist['distribution_id']
            for scenario in dist['scenarios']:
                s_id = scenario['scenario_id']

                demand = scenario['demand']
                # Full price sales cannot exceed demand
                model.addConstr(w[(p_id, d_id, s_id)] <= demand, name=f"demand_limit_{p_id}_{d_id}_{s_id}")

                # Full price sales cannot exceed available inventory (only if dist is selected)
                model.addConstr(w[(p_id, d_id, s_id)] <= z[(p_id, d_id, s_id)], 
                               name=f"inventory_limit_{p_id}_{d_id}_{s_id}")

                # Total sales = available inventory (only if dist is selected)
                model.addConstr(w[(p_id, d_id, s_id)] + o[(p_id, d_id, s_id)] == z[(p_id, d_id, s_id)],
                               name=f"total_sales_{p_id}_{d_id}_{s_id}")

    # Objective function
    # Maximize expected profit = E[revenue] - total manufacturing cost
    revenue_expr = gp.LinExpr(0)
    cost_expr = gp.LinExpr(0)

    # Add revenue calculation
    for p_id in range(num_products):
        for dist in distribution_lookup[p_id]:
            d_id = dist['distribution_id']
            for scenario in dist['scenarios']:
                s_id = scenario['scenario_id']

                prob = scenario['probability']
                sale_price = products[p_id]['sale_price']
                disc_price = products[p_id]['discounted_price']

                revenue_expr.addTerms([prob * sale_price], [w[(p_id, d_id, s_id)]])
                revenue_expr.addTerms([prob * disc_price], [o[(p_id, d_id, s_id)]])

    # Add manufacturing cost calculation
    for p_id in range(num_products):
        for f_id in range(num_facilities):
            pfd_entry = product_facility_data.get((p_id, f_id))
            if pfd_entry:
                manufacturing_cost = pfd_entry['manufacturing_cost']
                cost_expr.addTerms([manufacturing_cost], [x[(p_id, f_id)]])

    model.setObjective(revenue_expr - cost_expr, GRB.MAXIMIZE)

    # Solve
    model.optimize()

    # Extract solution
    if model.Status == GRB.INFEASIBLE:
        print("Model is infeasible!")
        model.computeIIS()
        model.write("model.ilp")
        raise ValueError("Model is infeasible")

    sol_x = {}
    sol_y = {}
    sol_delta = {}
    sol_z = {}
    sol_w = {}
    sol_o = {}

    for p_id in range(num_products):
        for f_id in range(num_facilities):
            key = f"{p_id}_{f_id}"
            sol_x[key] = x[(p_id, f_id)].x if x[(p_id, f_id)].x > 1e-8 else 0.0
            
            pfd = product_facility_data.get((p_id, f_id))
            if pfd:
                for level in pfd['production_levels']:
                    l_id = level['level_id']
                    y_key = f"{p_id}_{f_id}_{l_id}"
                    if (p_id, f_id, l_id) in y:
                        sol_y[y_key] = int(round(y[(p_id, f_id, l_id)].x))
                    else:
                        sol_y[y_key] = 0

    for p_id in range(num_products):
        for dist in distribution_lookup[p_id]:
            d_id = dist['distribution_id']
            delta_key = f"{p_id}_{d_id}"
            sol_delta[delta_key] = int(round(delta[(p_id, d_id)].x))
            
            for scenario in dist['scenarios']:
                s_id = scenario['scenario_id']
                z_key = f"{p_id}_{d_id}_{s_id}"
                w_key = f"{p_id}_{d_id}_{s_id}"
                o_key = f"{p_id}_{d_id}_{s_id}"
                
                sol_z[z_key] = z[(p_id, d_id, s_id)].x if z[(p_id, d_id, s_id)].x > 1e-8 else 0.0
                sol_w[w_key] = w[(p_id, d_id, s_id)].x if w[(p_id, d_id, s_id)].x > 1e-8 else 0.0
                sol_o[o_key] = o[(p_id, d_id, s_id)].x if o[(p_id, d_id, s_id)].x > 1e-8 else 0.0

    objective_value = model.objVal if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] else float('-inf')

    # Ensure all variables are populated even when empty to avoid missing keys
    all_keys_needed = {
        'x': [(f"{p_id}_{f_id}") for p_id in range(num_products) for f_id in range(num_facilities)],
        'y': [(f"{p_id}_{f_id}_{level['level_id']}") 
              for p_id in range(num_products) 
              for f_id in range(num_facilities) 
              if (p_id, f_id) in product_facility_data
              for level in product_facility_data[(p_id, f_id)]['production_levels']],
        'delta': [(f"{p_id}_{dist['distribution_id']}") 
                  for p_id in range(num_products) 
                  for dist in distribution_lookup[p_id]],
        'z': [(f"{p_id}_{d['distribution_id']}_{s['scenario_id']}") 
              for prod_dist in instance['products_distributions']
              for d in prod_dist['distributions']
              for s in d['scenarios']],
        'w': [(f"{p_id}_{d['distribution_id']}_{s['scenario_id']}") 
              for prod_dist in instance['products_distributions']
              for d in prod_dist['distributions']
              for s in d['scenarios']],
        'o': [(f"{p_id}_{d['distribution_id']}_{s['scenario_id']}") 
              for prod_dist in instance['products_distributions']
              for d in prod_dist['distributions']
              for s in d['scenarios']]
    }

    all_solutions = {"x": sol_x, "y": sol_y, "delta": sol_delta, "z": sol_z, "w": sol_w, "o": sol_o}
    
    for var_type, required_keys in all_keys_needed.items():
        for k in required_keys:
            if k not in all_solutions[var_type]:
                all_solutions[var_type][k] = 0.0 if var_type in ['x', 'z', 'w', 'o'] else 0

    return {
        "objective_value": objective_value,
        "solution": {
            "x": sol_x,
            "y": sol_y,
            "delta": sol_delta,
            "z": sol_z,
            "w": sol_w,
            "o": sol_o
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Solve stochastic production planning problem.')
    parser.add_argument('--instance_path', required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', required=True, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', default=None, help='Path to log file for incumbent solutions')
    args = parser.parse_args()

    # Initialize logger if requested
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    # Function to handle callbacks for logging
    def callback(model, where):
        if where == GRB.Callback.MIPSOL and model.SolCount > 0:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    try:
        # Read instance
        instance = read_instance(args.instance_path)
        
        # Solve instance
        solution = solve_instance(instance, args.time_limit)

        # Write solution
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
    
    except Exception as e:
        print(f"Encountered error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()