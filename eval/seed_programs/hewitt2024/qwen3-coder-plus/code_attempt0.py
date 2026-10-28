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
    num_scenarios_per_distribution = instance['num_scenarios_per_distribution']
    
    products = {p['id']: p for p in instance['products']}
    facilities = {f['id']: f for f in instance['facilities']}
    product_facility_data = {}
    for pfd in instance['product_facility_data']:
        product_id = pfd['product_id']
        facility_id = pfd['facility_id']
        if (product_id, facility_id) not in product_facility_data:
            product_facility_data[(product_id, facility_id)] = pfd

    distribution_map = {}  # Maps (prod_id, dist_id) to distribution object for easy lookup
    for prod_dist in instance['products_distributions']:
        prod_id = prod_dist['product_id']
        for dist in prod_dist['distributions']:
            dist_id = dist['distribution_id']
            distribution_map[(prod_id, dist_id)] = dist
    
    # Create model
    model = gp.Model("Stochastic Production Planning")
    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 1e-6)  # Ensure better optimality gap
    
    # Decision Variables
    x = {}  # Raw material allocation: (p, f) -> continuous
    y = {}  # Production level selection: (p, f, l) -> binary
    delta = {}  # Distribution selection: (p, d) -> binary
    
    # Scenario-dependent variables
    z = {}  # Available inventory: (p, d, s) -> continuous
    w = {}  # Full price sales: (p, d, s) -> continuous
    o = {}  # Discounted sales: (p, d, s) -> continuous
    
    # Initialize variables
    for p in range(num_products):
        for f in range(num_facilities):
            pfd = product_facility_data[(p, f)]
            for level in pfd['production_levels']:
                l = level['level_id']
                y[(p, f, l)] = model.addVar(vtype=GRB.BINARY, name=f"y_{p}_{f}_{l}")
            
            x[(p, f)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"x_{p}_{f}")
        
        for prod_dist in [pd for pd in instance['products_distributions'] if pd['product_id'] == p]:
            for dist in prod_dist['distributions']:
                d = dist['distribution_id']
                delta[(p, d)] = model.addVar(vtype=GRB.BINARY, name=f"delta_{p}_{d}")
                
                for scenario in dist['scenarios']:
                    s = scenario['scenario_id']
                    z[(p, d, s)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"z_{p}_{d}_{s}")
                    w[(p, d, s)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"w_{p}_{d}_{s}")
                    o[(p, d, s)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"o_{p}_{d}_{s}")
    
    # Constraints to tie allocation variables with production level selection
    for p in range(num_products):
        for f in range(num_facilities):
            pfd = product_facility_data[(p, f)]
            
            # Exactly one production level per facility-product pair
            model.addConstr(gp.quicksum(y[(p, f, level['level_id'])] for level in pfd['production_levels']) == 1, 
                           name=f"one_level_{p}_{f}")
            
            # Link allocation with level selection
            for level in pfd['production_levels']:
                l = level['level_id']
                x[(p, f)].ub = max(x[(p, f)].ub, level['upper_bound'])  # Set upper bound based on highest possible
                model.addConstr(
                    x[(p, f)] >= level['lower_bound'] * y[(p, f, l)],
                    name=f"alloc_lb_{p}_{f}_{l}"
                )
                model.addConstr(
                    x[(p, f)] <= level['upper_bound'] * y[(p, f, l)] + (1 - y[(p, f, l)]) * max([lvl['upper_bound'] for lvl in pfd['production_levels']]),
                    name=f"alloc_ub_{p}_{f}_{l}"
                )
    
    # Capacity constraints
    for f in range(num_facilities):
        model.addConstr(gp.quicksum(x[(p, f)] for p in range(num_products)) <= facilities[f]['capacity'], 
                       name=f"capacity_{f}")
    
    # Select exactly one distribution for each product
    for p in range(num_products):
        prod_dist = next(pd for pd in instance['products_distributions'] if pd['product_id'] == p)
        model.addConstr(gp.quicksum(delta[(p, d['distribution_id'])] for d in prod_dist['distributions']) == 1,
                       name=f"select_one_dist_{p}")
    
    # Connect allocation decisions to distribution eligibility 
    for p in range(num_products):
        prod_dist = next(pd for pd in instance['products_distributions'] if pd['product_id'] == p)
        for dist in prod_dist['distributions']:
            d = dist['distribution_id']
            if dist['type'] == 'yield_and_demand' and dist['level_combination'] is not None:
                for f in range(len(dist['level_combination'])):
                    l = dist['level_combination'][f]
                    model.addConstr(delta[(p, d)] <= y[(p, f, l)], 
                                   name=f"link_dist_level_{p}_{d}_{f}_{l}")
    
    # Link inventory with allocation and yield
    for p in range(num_products):
        prod_dist = next(pd for pd in instance['products_distributions'] if pd['product_id'] == p)
        for dist in prod_dist['distributions']:
            d = dist['distribution_id']
            for scenario in dist['scenarios']:
                s = scenario['scenario_id']
                
                # Calculate inventory based on yield and allocation
                inventory_expr = gp.LinExpr()
                for f_idx in range(num_facilities):
                    fac_id = f_idx  # Assuming facility IDs are consecutive starting from 0
                    if 'yields' in scenario:
                        yield_factor = scenario['yields'].get(str(fac_id))
                        if yield_factor is not None:
                            inventory_expr.addTerms([yield_factor], [x[(p, fac_id)]])
                
                model.addConstr(z[(p, d, s)] == inventory_expr, name=f"inv_calc_{p}_{d}_{s}")
    
    # Sales constraints
    for p in range(num_products):
        prod_dist = next(pd for pd in instance['products_distributions'] if pd['product_id'] == p)
        for dist in prod_dist['distributions']:
            d = dist['distribution_id']
            for scenario in dist['scenarios']:
                s = scenario['scenario_id']
                
                demand = scenario['demand']
                # Full price sales cannot exceed demand
                model.addConstr(w[(p, d, s)] <= demand, name=f"demand_limit_{p}_{d}_{s}")
                
                # Full price sales cannot exceed available inventory
                model.addConstr(w[(p, d, s)] <= z[(p, d, s)], name=f"inventory_limit_{p}_{d}_{s}")
                
                # Total sales = available inventory
                model.addConstr(w[(p, d, s)] + o[(p, d, s)] == z[(p, d, s)], 
                               name=f"total_sales_{p}_{d}_{s}")
                
                # Enforce that full price sales occur before discounted sales (not needed due to price relationship)
    
    # Objective function
    # Maximize expected profit = E[revenue] - total manufacturing cost
    revenue = gp.QuadExpr(0)
    cost = gp.QuadExpr(0)
    
    for p in range(num_products):
        prod_dist = next(pd for pd in instance['products_distributions'] if pd['product_id'] == p)
        for dist in prod_dist['distributions']:
            d = dist['distribution_id']
            for scenario in dist['scenarios']:
                s = scenario['scenario_id']
                
                prob = scenario['probability']
                sale_price = products[p]['sale_price']
                disc_price = products[p]['discounted_price']
                
                revenue += delta[(p, d)] * prob * (sale_price * w[(p, d, s)] + disc_price * o[(p, d, s)])
    
    for p in range(num_products):
        for f in range(num_facilities):
            manufacturing_cost = product_facility_data[(p, f)]['manufacturing_cost']
            cost += manufacturing_cost * x[(p, f)]
    
    model.setObjective(revenue - cost, GRB.MAXIMIZE)
    
    # Solve
    model.optimize()
    
    # Extract solution
    sol_x = {}
    sol_y = {}
    sol_delta = {}
    sol_z = {}
    sol_w = {}
    sol_o = {}
    
    for p in range(num_products):
        for f in range(num_facilities):
            sol_x[f"{p}_{f}"] = x[(p, f)].x if x[(p, f)].x > 1e-6 else 0.0
            pfd = product_facility_data[(p, f)]
            for level in pfd['production_levels']:
                l = level['level_id']
                sol_y[f"{p}_{f}_{l}"] = int(round(y[(p, f, l)].x))
                
        prod_dist = next(pd for pd in instance['products_distributions'] if pd['product_id'] == p)
        for dist in prod_dist['distributions']:
            d = dist['distribution_id']
            sol_delta[f"{p}_{d}"] = int(round(delta[(p, d)].x))
            for scenario in dist['scenarios']:
                s = scenario['scenario_id']
                sol_z[f"{p}_{d}_{s}"] = z[(p, d, s)].x
                sol_w[f"{p}_{d}_{s}"] = w[(p, d, s)].x
                sol_o[f"{p}_{d}_{s}"] = o[(p, d, s)].x
    
    objective_value = model.objVal if model.Status not in [GRB.INFEASIBLE, GRB.UNBOUNDED, GRB.INF_OR_UNBD] else float('-inf')
    
    return {
        "objective_value": objective_value if objective_value != float('-inf') else 0.0,
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
    
    # Read instance
    instance = read_instance(args.instance_path)
    
    # Solve instance
    solution = solve_instance(instance, args.time_limit)
    
    # Log the final objective value if found better solution
    if solver.is_mip and solution["objective_value"] != float('-inf'):
        if logger:
            logger.log(float(solution["objective_value"]) if solution["objective_value"] != float('-inf') else float('-1e8'))
    
    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()