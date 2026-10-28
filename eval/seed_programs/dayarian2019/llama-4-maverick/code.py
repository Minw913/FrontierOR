import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def solve(instance):
    num_orders = instance['num_orders']
    num_products = instance['num_products']
    num_vehicles = instance['num_vehicles']
    num_workstations = instance['num_workstations']
    num_subperiods = instance['global_parameters']['num_subperiods']
    num_periods = instance['global_parameters']['num_periods']
    subperiods_per_period = instance['global_parameters']['subperiods_per_period']
    vehicle_capacity = instance['vehicle_capacity']
    driver_salary_per_minute = instance['global_parameters']['driver_salary_per_minute']
    employee_salary_per_subperiod = instance['global_parameters']['employee_salary_per_subperiod']
    setup_cost = instance['global_parameters']['setup_cost']
    vehicle_operating_cost_per_minute = instance['global_parameters']['vehicle_operating_cost_per_minute']
    max_trips_per_route = instance['global_parameters']['max_trips_per_route']
    preparation_time_minutes = instance['global_parameters']['preparation_time_minutes']
    service_time_minutes = instance['global_parameters']['service_time_minutes']
    travel_time_matrix = instance['travel_time_matrix']
    orders = instance['orders']
    products = instance['products']
    production_windows = instance['production_windows']
    periods = instance['periods']
    subperiods = instance['subperiods']

    try:
        model = gp.Model("CateringServices")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1

        # Production Variables
        production_quantity = model.addVars(num_orders, num_products, num_subperiods, vtype=GRB.CONTINUOUS, name="production_quantity")
        employees_by_product_subperiod = model.addVars(num_products, num_subperiods, vtype=GRB.INTEGER, name="employees_by_product_subperiod")
        setups_by_product_subperiod = model.addVars(num_products, num_subperiods, vtype=GRB.INTEGER, name="setups_by_product_subperiod")
        workstations_by_period = model.addVars(num_periods, vtype=GRB.INTEGER, name="workstations_by_period")
        shift_starts_by_period = model.addVars(num_periods, vtype=GRB.INTEGER, name="shift_starts_by_period")

        # Routing Variables
        vehicle_used = model.addVars(num_vehicles, vtype=GRB.BINARY, name="vehicle_used")
        trip_start_period = model.addVars(num_vehicles, max_trips_per_route, vtype=GRB.INTEGER, name="trip_start_period")
        customers_in_trip = model.addVars(num_vehicles, max_trips_per_route, num_orders, vtype=GRB.BINARY, name="customers_in_trip")

        # Objective Function
        production_cost = gp.quicksum(employee_salary_per_subperiod * workstations_by_period[p] for p in range(num_periods))
        setup_total_cost = gp.quicksum(setup_cost * setups_by_product_subperiod[prod, sp] for prod in range(num_products) for sp in range(num_subperiods))
        routing_cost = gp.quicksum(driver_salary_per_minute * vehicle_used[v] * instance['global_parameters']['max_shift_route_length_minutes'] for v in range(num_vehicles))
        
        operating_cost = gp.quicksum(vehicle_operating_cost_per_minute * _calculate_trip_time(instance, v, t, customers_in_trip, travel_time_matrix, preparation_time_minutes, orders, model) for v in range(num_vehicles) for t in range(max_trips_per_route))
        
        model.setObjective(production_cost + setup_total_cost + routing_cost + operating_cost, GRB.MINIMIZE)

        # Production Constraints
        for o in range(num_orders):
            for p in range(num_products):
                if str(o+1) in production_windows and str(p) in production_windows[str(o+1)]:
                    valid_subperiods = production_windows[str(o+1)][str(p)]['valid_subperiod_ids']
                    demand = orders[o]['demands'].get(str(p), 0)
                    model.addConstr(gp.quicksum(production_quantity[o, p, sp] for sp in valid_subperiods) >= demand)

        for p in range(num_products):
            for sp in range(num_subperiods):
                model.addConstr(employees_by_product_subperiod[p, sp] * products[p]['production_rate_per_subperiod'] >= gp.quicksum(production_quantity[o, p, sp] for o in range(num_orders)))

        for p in range(num_periods):
            model.addConstr(workstations_by_period[p] >= gp.quicksum(employees_by_product_subperiod[prod, sp] for prod in range(num_products) for sp in range(p*subperiods_per_period, (p+1)*subperiods_per_period)))

        # Routing Constraints
        for v in range(num_vehicles):
            for t in range(max_trips_per_route):
                model.addConstr(gp.quicksum(customers_in_trip[v, t, o] for o in range(num_orders)) <= num_orders)

        for o in range(num_orders):
            model.addConstr(gp.quicksum(customers_in_trip[v, t, o] for v in range(num_vehicles) for t in range(max_trips_per_route)) == 1)

        # Capacity Constraints
        for v in range(num_vehicles):
            for t in range(max_trips_per_route):
                model.addConstr(gp.quicksum(customers_in_trip[v, t, o] * orders[o]['space_requirement'] for o in range(num_orders)) <= vehicle_capacity)

        model.optimize()

        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            solution = _extract_solution(instance, model)
            return solution
        else:
            return None

    except gp.GurobiError as e:
        print(f'Gurobi Error: {e}')
        return None

def _calculate_trip_time(instance, v, t, customers_in_trip, travel_time_matrix, preparation_time_minutes, orders, model):
    trip_time = preparation_time_minutes
    previous_customer = 0
    customers = []
    for o in range(instance['num_orders']):
        if model.getVarByName(f'customers_in_trip[{v},{t},{o}]').x > 0.5:
            customers.append(o)
    if not customers:
        return 0
    customers.sort(key=lambda x: orders[x]['time_window_start'])
    for o in customers:
        trip_time += travel_time_matrix[previous_customer][o+1]
        trip_time += orders[o]['service_time'] 
        previous_customer = o+1
    trip_time += travel_time_matrix[previous_customer][0]
    return trip_time

def _extract_solution(instance, model):
    num_orders = instance['num_orders']
    num_products = instance['num_products']
    num_vehicles = instance['num_vehicles']
    max_trips_per_route = instance['global_parameters']['max_trips_per_route']
    num_subperiods = instance['global_parameters']['num_subperiods']
    num_periods = instance['global_parameters']['num_periods']

    solution = {}
    solution['objective_value'] = model.ObjVal

    routes = []
    for v in range(num_vehicles):
        if model.getVarByName(f'vehicle_used[{v}]').x > 0.5:
            vehicle_route = {'vehicle': v, 'trips': []}
            for t in range(max_trips_per_route):
                trip = {'trip': t, 'customers': [], 'start_period': int(model.getVarByName(f'trip_start_period[{v},{t}]').x)}
                customers = []
                for o in range(num_orders):
                    if model.getVarByName(f'customers_in_trip[{v},{t},{o}]').x > 0.5:
                        customers.append(o+1)
                if customers:
                    trip['customers'] = customers
                    vehicle_route['trips'].append(trip)
            routes.append(vehicle_route)
    solution['routes'] = routes

    production_schedule = []
    for o in range(num_orders):
        for p in range(num_products):
            for sp in range(num_subperiods):
                quantity = model.getVarByName(f'production_quantity[{o},{p},{sp}]').x
                if quantity > 0:
                    production_schedule.append({'order': o+1, 'product': p, 'subperiod': sp, 'quantity': quantity})
    solution['production_schedule'] = production_schedule

    production_staffing = {'employees_by_product_subperiod': [], 'setups_by_product_subperiod': [], 'workstations_by_period': [], 'shift_starts_by_period': []}
    for p in range(num_products):
        for sp in range(num_subperiods):
            employees = model.getVarByName(f'employees_by_product_subperiod[{p},{sp}]').x
            if employees > 0:
                production_staffing['employees_by_product_subperiod'].append({'product': p, 'subperiod': sp, 'employees': int(employees)})
            setups = model.getVarByName(f'setups_by_product_subperiod[{p},{sp}]').x
            if setups > 0:
                production_staffing['setups_by_product_subperiod'].append({'product': p, 'subperiod': sp, 'setups': int(setups)})
    for p in range(num_periods):
        workstations = model.getVarByName(f'workstations_by_period[{p}]').x
        production_staffing['workstations_by_period'].append(int(workstations))
        shift_starts = model.getVarByName(f'shift_starts_by_period[{p}]').x
        production_staffing['shift_starts_by_period'].append(int(shift_starts))
    solution['production_staffing'] = production_staffing

    return solution

def main():
    parser = argparse.ArgumentParser(description='Solve Catering Services Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions')
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model = gp.Model("CateringServices")
    model.Params.TimeLimit = args.time_limit

    solution = solve(instance)

    if solution is not None:
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
        if logger:
            logger.log_solution(solution['objective_value'], solution)

if __name__ == "__main__":
    main()