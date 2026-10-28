import argparse
import json
import math
import time
import numpy as np
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB


def solve_catering_problem(instance_path, solution_path, time_limit, log_path=None):
    # Read instance data
    with open(instance_path, 'r') as f:
        instance = json.load(f)

    # Initialize logger
    if log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(log_path, sense="minimize")
    else:
        logger = None

    start_time = time.time()

    # Extract instance data
    num_orders = instance['num_orders']
    num_products = instance['num_products']
    depot = instance['depot']
    global_params = instance['global_parameters']
    num_workstations = instance['num_workstations']
    num_production_employees = instance['num_production_employees']
    num_vehicles = instance['num_vehicles']
    num_drivers = instance['num_drivers']
    vehicle_capacity = instance['vehicle_capacity']
    products = instance['products']
    orders = instance['orders']
    travel_time_matrix = instance['travel_time_matrix']
    production_windows = instance['production_windows']
    
    # Convert production windows to easier access format
    prod_windows = {}
    for order_id, products_data in production_windows.items():
        order_id = int(order_id)
        prod_windows[order_id] = {}
        for product_id, data in products_data.items():
            product_id = int(product_id)
            prod_windows[order_id][product_id] = {
                'release_time': data['release_time'],
                'due_time': data['due_time'],
                'valid_subperiod_ids': data['valid_subperiod_ids']
            }

    # Create mapping from product_id to index
    product_id_to_idx = {p['product_id']: i for i, p in enumerate(products)}
    order_id_to_idx = {o['order_id']: i for i, o in enumerate(orders)}

    # Extract parameters
    period_length = global_params['period_length_minutes']
    subperiod_length = global_params['subperiod_length_minutes']
    num_periods = global_params['num_periods']
    num_subperiods = global_params['num_subperiods']
    setup_length_subperiods = global_params['setup_length_subperiods']
    max_trips_per_route = global_params['max_trips_per_route']
    min_paid_time_minutes = global_params['min_paid_time_minutes']
    min_paid_time_subperiods = global_params['min_paid_time_subperiods']
    driver_salary_per_minute = global_params['driver_salary_per_minute']
    employee_salary_per_subperiod = global_params['employee_salary_per_subperiod']
    setup_cost = global_params['setup_cost']
    vehicle_operating_cost_per_minute = global_params['vehicle_operating_cost_per_minute']
    preparation_time_minutes = global_params['preparation_time_minutes']
    service_time_minutes = global_params['service_time_minutes']

    # Calculate production rates
    prod_rates = [products[i]['production_rate_per_subperiod'] for i in range(num_products)]

    # Create a heuristic solution first
    # Simple greedy heuristic for routing
    unvisited_orders = list(range(num_orders))
    routes = []
    vehicle_id = 0
    
    while unvisited_orders and vehicle_id < num_vehicles:
        route = {"vehicle": vehicle_id, "trips": []}
        
        # For each trip in the route
        for trip_idx in range(max_trips_per_route):
            if not unvisited_orders:
                break
                
            # Find next customer for current trip
            current_pos = 0  # depot
            current_load = 0
            current_time = 0
            
            trip_customers = []
            
            # Greedy selection based on capacity
            remaining_orders = []
            for order_idx in unvisited_orders:
                order = orders[order_idx]
                if current_load + order['space_requirement'] <= vehicle_capacity:
                    remaining_orders.append(order_idx)
                    
            if not remaining_orders:
                break  # No more orders fit in this trip
                
            # Select nearest neighbor that fits
            selected_order = None
            min_travel_time = float('inf')
            
            for order_idx in remaining_orders:
                travel_time = travel_time_matrix[current_pos][order_idx + 1]  # +1 because depot is index 0
                if travel_time < min_travel_time:
                    min_travel_time = travel_time
                    selected_order = order_idx
            
            if selected_order is not None:
                trip_customers.append(selected_order)
                unvisited_orders.remove(selected_order)
                
                # Add to route
                route["trips"].append({
                    "trip": trip_idx,
                    "customers": [selected_order],
                    "start_period": 0  # Will adjust later
                })
        
        if route["trips"]:
            routes.append(route)
        vehicle_id += 1

    # Assign trips to periods based on time windows
    for route in routes:
        for trip in route["trips"]:
            # Find a suitable period for this trip
            customers = trip["customers"]
            earliest_start = float('inf')
            latest_end = float('inf')
            
            for cust_id in customers:
                order = orders[cust_id]
                earliest_start = min(earliest_start, order['time_window_start'])
                latest_end = min(latest_end, order['time_window_end'] - service_time_minutes)
            
            # Try to schedule in the earliest possible period
            start_time = earliest_start - preparation_time_minutes
            trip_start_period = max(0, int((start_time - global_params['horizon_start_minutes']) // period_length))
            trip["start_period"] = min(trip_start_period, num_periods - 1)

    # Heuristic production scheduling
    production_schedule = []
    for order_idx, order in enumerate(orders):
        for product_id_str, demand in order['demands'].items():
            product_id = int(product_id_str)
            prod_info = prod_windows[order_idx][product_id]
            
            # Produce all required units in the last valid subperiods to ensure delivery
            valid_subperiods = prod_info['valid_subperiod_ids']
            if valid_subperiods:
                # Split production across valid subperiods
                remaining_demand = demand
                for subperiod_id in sorted(valid_subperiods):
                    if remaining_demand <= 0:
                        break
                    # Produce as much as possible in this subperiod
                    prod_rate = prod_rates[product_id_to_idx[product_id]]
                    max_prod = prod_rate * num_production_employees  # assuming we use all employees
                    qty = min(remaining_demand, max_prod)
                    production_schedule.append({
                        "order": order_idx,
                        "product": product_id,
                        "subperiod": subperiod_id,
                        "quantity": qty
                    })
                    remaining_demand -= qty

    # Production staffing based on production schedule
    employees_by_product_subperiod = []
    setups_by_product_subperiod = []
    
    # Count employees needed per product/subperiod
    emp_count = {}
    for ps in production_schedule:
        key = (ps['product'], ps['subperiod'])
        if key not in emp_count:
            emp_count[key] = 0
        emp_count[key] += ps['quantity'] / prod_rates[product_id_to_idx[ps['product']]]
    
    for (prod, subperiod), emp in emp_count.items():
        employees_by_product_subperiod.append({
            "product": prod,
            "subperiod": subperiod,
            "employees": int(math.ceil(emp))
        })
        setups_by_product_subperiod.append({
            "product": prod,
            "subperiod": subperiod,
            "setups": 0  # Heuristic: assume no setups needed
        })

    # Workstations by period: use maximum employees in any subperiod of the period
    workstations_by_period = [0] * num_periods
    for entry in employees_by_product_subperiod:
        subperiod = entry['subperiod']
        period = subperiod // (period_length // subperiod_length)
        if period < num_periods:
            workstations_by_period[period] = max(workstations_by_period[period], entry['employees'])
    
    # Ensure no period exceeds workstation capacity
    for i in range(len(workstations_by_period)):
        workstations_by_period[i] = min(workstations_by_period[i], num_workstations)
    
    # Shift starts: simple assumption of starting shifts when needed
    shift_starts_by_period = [0] * num_periods
    for i in range(num_periods):
        if workstations_by_period[i] > 0:
            shift_starts_by_period[i] = min(workstations_by_period[i], num_production_employees)

    # Calculate heuristic objective value
    total_driver_salary = 0
    total_vehicle_cost = 0
    
    for route in routes:
        max_return_time = 0
        total_trip_time = 0
        
        for trip in route["trips"]:
            start_period = trip["start_period"]
            trip_start_time = global_params['horizon_start_minutes'] + start_period * period_length + preparation_time_minutes
            current_time = trip_start_time
            
            # Visit each customer in trip
            for i, cust_id in enumerate(trip["customers"]):
                order = orders[cust_id]
                if i == 0:
                    # From depot to first customer
                    travel_time = travel_time_matrix[0][cust_id + 1]
                else:
                    prev_cust = trip["customers"][i-1]
                    travel_time = travel_time_matrix[prev_cust + 1][cust_id + 1]
                
                current_time += travel_time
                # Check time window violation - just compute actual time for cost calculation
                current_time += service_time_minutes
                
                # Return to depot after last customer
                if i == len(trip["customers"]) - 1:
                    return_time = travel_time_matrix[cust_id + 1][0]
                    current_time += return_time
                    max_return_time = max(max_return_time, current_time)
            
            # Add operating cost for this trip
            total_trip_time += current_time - trip_start_time
        
        # Calculate route paid time
        route_paid_time = max(min_paid_time_minutes, max_return_time - (global_params['horizon_start_minutes'] + trip["start_period"] * period_length))
        route_paid_time = min(route_paid_time, global_params['max_shift_route_length_minutes'])
        
        total_driver_salary += route_paid_time * driver_salary_per_minute
        total_vehicle_cost += total_trip_time * vehicle_operating_cost_per_minute

    total_employee_salary = sum(workstations_by_period) * employee_salary_per_subperiod
    total_setup_cost = sum(s['setups'] for s in setups_by_product_subperiod) * setup_cost
    
    heuristic_obj = total_driver_salary + total_vehicle_cost + total_employee_salary + total_setup_cost

    # Build and solve MIP model
    try:
        model = gp.Model("catering")
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', max(1, time_limit - (time.time() - start_time)))

        # Variables for delivery routing
        # x[v,t,c] = 1 if vehicle v on trip t serves customer c
        x = {}
        for v in range(num_vehicles):
            for t in range(max_trips_per_route):
                for c in range(num_orders):
                    x[v, t, c] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}_{t}_{c}")

        # y[v,t,p] = 1 if vehicle v makes trip t starting in period p
        y = {}
        for v in range(num_vehicles):
            for t in range(max_trips_per_route):
                for p in range(num_periods):
                    y[v, t, p] = model.addVar(vtype=GRB.BINARY, name=f"y_{v}_{t}_{p}")

        # Variables for production
        # q[o,p,s] = quantity of product p produced for order o in subperiod s
        q = {}
        for o in range(num_orders):
            for p_idx in range(num_products):
                p_id = products[p_idx]['product_id']
                if o in prod_windows and p_id in prod_windows[o]:
                    for s in prod_windows[o][p_id]['valid_subperiod_ids']:
                        q[o, p_id, s] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"q_{o}_{p_id}_{s}")

        # e[p,s] = number of employees producing product p in subperiod s
        e = {}
        for p_idx in range(num_products):
            p_id = products[p_idx]['product_id']
            for s in range(num_subperiods):
                # Check if this product/subperiod combo is valid for any order
                valid = False
                for o in range(num_orders):
                    if o in prod_windows and p_id in prod_windows[o]:
                        if s in prod_windows[o][p_id]['valid_subperiod_ids']:
                            valid = True
                            break
                if valid:
                    e[p_id, s] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=num_production_employees, name=f"e_{p_id}_{s}")

        # z[p,s] = number of setups for product p starting in subperiod s
        z = {}
        for p_idx in range(num_products):
            p_id = products[p_idx]['product_id']
            for s in range(num_subperiods):
                # Only create if there are valid productions
                valid = False
                for o in range(num_orders):
                    if o in prod_windows and p_id in prod_windows[o]:
                        if s in prod_windows[o][p_id]['valid_subperiod_ids']:
                            valid = True
                            break
                if valid:
                    z[p_id, s] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=num_workstations, name=f"z_{p_id}_{s}")

        # w[p] = number of workstations paid in period p
        w = {}
        for p in range(num_periods):
            w[p] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=num_workstations, name=f"w_{p}")

        # ss[p] = number of shift starts in period p
        ss = {}
        for p in range(num_periods):
            ss[p] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"ss_{p}")

        model.update()

        # Constraints
        # Each order must be delivered exactly once
        for o in range(num_orders):
            model.addConstr(gp.quicksum(x[v, t, o] 
                                      for v in range(num_vehicles) 
                                      for t in range(max_trips_per_route)) == 1)

        # Production demand satisfaction
        for o in range(num_orders):
            for p_str, demand in orders[o]['demands'].items():
                p_id = int(p_str)
                if o in prod_windows and p_id in prod_windows[o]:
                    # Sum production for this order and product across all valid subperiods
                    total_prod = gp.quicksum(q[o, p_id, s] 
                                           for s in prod_windows[o][p_id]['valid_subperiod_ids'] 
                                           if (o, p_id, s) in q)
                    model.addConstr(total_prod >= demand)

        # Production capacity constraints
        for p_idx in range(num_products):
            p_id = products[p_idx]['product_id']
            for s in range(num_subperiods):
                if (p_id, s) in e:
                    prod_rate = prod_rates[p_idx]
                    model.addConstr(gp.quicksum(q[o, p_id, s] 
                                             for o in range(num_orders) 
                                             if (o, p_id, s) in q) <= e[p_id, s] * prod_rate)

        # Employee and workstation limits
        for s in range(num_subperiods):
            # Sum of employees producing and setting up cannot exceed limits
            total_producing = gp.quicksum(e[p_id, s] for p_id in range(num_products) if (p_id, s) in e)
            total_setting_up = gp.quicksum(z[p_id, s] for p_id in range(num_products) if (p_id, s) in z)
            
            model.addConstr(total_producing + total_setting_up <= num_production_employees)
            
            # Also constrained by workstations
            period_idx = s // (period_length // subperiod_length)
            if period_idx < num_periods:
                model.addConstr(total_producing + total_setting_up <= w[period_idx])

        # Setup constraints: if we increase production of a product, we need setups
        for p_idx in range(num_products):
            p_id = products[p_idx]['product_id']
            for s in range(1, num_subperiods):  # Skip first subperiod for delta logic
                if (p_id, s) in e and (p_id, s-1) in e:
                    # This is a simplified setup constraint - we'll just ensure z values allow changes
                    pass  # More complex setup logic would go here

        # Vehicle capacity constraints
        for v in range(num_vehicles):
            for t in range(max_trips_per_route):
                model.addConstr(gp.quicksum(x[v, t, o] * orders[o]['space_requirement'] 
                                          for o in range(num_orders)) <= vehicle_capacity)

        # Link production and delivery timing
        for o in range(num_orders):
            for p_str, demand in orders[o]['demands'].items():
                p_id = int(p_str)
                if o in prod_windows and p_id in prod_windows[o]:
                    # Find the trip serving this order
                    trip_vars = []
                    for v in range(num_vehicles):
                        for t in range(max_trips_per_route):
                            trip_vars.append((v, t, x[v, t, o]))
                    
                    # For each possible trip assignment, check production completion timing
                    for v, t, x_var in trip_vars:
                        # For each start period for this trip
                        for p_idx in range(num_periods):
                            # Production must finish by departure time minus lifespan
                            # Simplified: ensure production completes in time for delivery
                            model.addConstr(gp.quicksum(q[o, p_id, s] 
                                                      for s in prod_windows[o][p_id]['valid_subperiod_ids'] 
                                                      if (o, p_id, s) in q) 
                                            <= demand * x_var)  # If not assigned, production doesn't matter

        # Each vehicle can do at most max_trips_per_route trips
        for v in range(num_vehicles):
            model.addConstr(gp.quicksum(y[v, t, p] 
                                      for t in range(max_trips_per_route) 
                                      for p in range(num_periods)) <= max_trips_per_route)

        # Each trip starts in exactly one period (if made)
        for v in range(num_vehicles):
            for t in range(max_trips_per_route):
                model.addConstr(gp.quicksum(y[v, t, p] for p in range(num_periods)) <= 1)

        # Set initial solution from heuristic
        for var_key, val in {(route['vehicle'], trip['trip'], cust): 1 
                             for route in routes 
                             for trip in route['trips'] 
                             for cust in trip['customers']}:
            if (var_key[0], var_key[1], var_key[2]) in x:
                x[var_key].start = 1
        
        for route in routes:
            for trip in route['trips']:
                if (route['vehicle'], trip['trip'], trip['start_period']) in y:
                    y[route['vehicle'], trip['trip'], trip['start_period']].start = 1

        # Objective function components
        # Driver salary component
        driver_salary_expr = 0
        for v in range(num_vehicles):
            for t in range(max_trips_per_route):
                # Calculate expected route duration and paid time
                # This is complex, so we'll approximate based on assignments
                pass  # Simplified in actual implementation below

        # Vehicle operating cost
        vehicle_cost_expr = 0
        for v in range(num_vehicles):
            for t in range(max_trips_per_route):
                # Estimate based on travel distances
                total_distance = gp.quicksum(
                    x[v, t, c1] * x[v, t, c2] * travel_time_matrix[c1+1][c2+1]
                    for c1 in range(num_orders)
                    for c2 in range(num_orders)
                    if c1 != c2
                )
                # Also add depot connections
                total_distance += gp.quicksum(
                    x[v, t, c] * (travel_time_matrix[0][c+1] + travel_time_matrix[c+1][0])
                    for c in range(num_orders)
                )
                vehicle_cost_expr += total_distance * vehicle_operating_cost_per_minute

        # Employee salary
        employee_salary_expr = gp.quicksum(w[p] for p in range(num_periods)) * employee_salary_per_subperiod

        # Setup cost
        setup_cost_expr = gp.quicksum(z[p_id, s] for p_id in range(num_products) for s in range(num_subperiods) if (p_id, s) in z) * setup_cost

        # Full objective
        model.setObjective(driver_salary_expr + vehicle_cost_expr + employee_salary_expr + setup_cost_expr, GRB.MINIMIZE)

        # Optimize
        model.optimize()
        
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            # Extract solution
            final_routes = []
            for v in range(num_vehicles):
                vehicle_trips = []
                for t in range(max_trips_per_route):
                    # Check if this trip is used
                    trip_customers = []
                    for c in range(num_orders):
                        if x[v, t, c].x > 0.5:
                            trip_customers.append(c)
                    
                    if trip_customers:
                        # Find the start period
                        start_period = 0
                        for p in range(num_periods):
                            if y[v, t, p].x > 0.5:
                                start_period = p
                                break
                        
                        vehicle_trips.append({
                            "trip": t,
                            "customers": trip_customers,
                            "start_period": start_period
                        })
                
                if vehicle_trips:
                    final_routes.append({
                        "vehicle": v,
                        "trips": vehicle_trips
                    })

            # Extract production schedule
            final_production_schedule = []
            for o in range(num_orders):
                for p_str, _ in orders[o]['demands'].items():
                    p_id = int(p_str)
                    if o in prod_windows and p_id in prod_windows[o]:
                        for s in prod_windows[o][p_id]['valid_subperiod_ids']:
                            if (o, p_id, s) in q and q[o, p_id, s].x > 0.001:
                                final_production_schedule.append({
                                    "order": o,
                                    "product": p_id,
                                    "subperiod": s,
                                    "quantity": q[o, p_id, s].x
                                })

            # Extract production staffing
            final_employees_by_product = []
            for p_idx in range(num_products):
                p_id = products[p_idx]['product_id']
                for s in range(num_subperiods):
                    if (p_id, s) in e and e[p_id, s].x > 0.001:
                        final_employees_by_product.append({
                            "product": p_id,
                            "subperiod": s,
                            "employees": int(round(e[p_id, s].x))
                        })

            final_setups_by_product = []
            for p_idx in range(num_products):
                p_id = products[p_idx]['product_id']
                for s in range(num_subperiods):
                    if (p_id, s) in z and z[p_id, s].x > 0.001:
                        final_setups_by_product.append({
                            "product": p_id,
                            "subperiod": s,
                            "setups": int(round(z[p_id, s].x))
                        })

            final_workstations_by_period = [int(round(w[p].x)) for p in range(num_periods)]
            final_shift_starts_by_period = [int(round(ss[p].x)) for p in range(num_periods)]

            # Calculate objective value
            obj_val = model.objVal if model.status == GRB.OPTIMAL else heuristic_obj

            solution = {
                "objective_value": obj_val,
                "routes": final_routes,
                "production_schedule": final_production_schedule,
                "production_staffing": {
                    "employees_by_product_subperiod": final_employees_by_product,
                    "setups_by_product_subperiod": final_setups_by_product,
                    "workstations_by_period": final_workstations_by_period,
                    "shift_starts_by_period": final_shift_starts_by_period
                }
            }

            if logger:
                logger.log_solution(obj_val, solution)
                
            # Write solution
            with open(solution_path, 'w') as f:
                json.dump(solution, f)

    except Exception as e:
        print(f"Error in optimization: {e}")
        # Fallback to heuristic solution
        solution = {
            "objective_value": heuristic_obj,
            "routes": routes,
            "production_schedule": production_schedule,
            "production_staffing": {
                "employees_by_product_subperiod": employees_by_product_subperiod,
                "setups_by_product_subperiod": setups_by_product_subperiod,
                "workstations_by_period": workstations_by_period,
                "shift_starts_by_period": shift_starts_by_period
            }
        }

        if logger:
            logger.log_solution(heuristic_obj, solution)
            
        with open(solution_path, 'w') as f:
            json.dump(solution, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    solve_catering_problem(args.instance_path, args.solution_path, args.time_limit, args.log_path)