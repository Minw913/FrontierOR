import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def compute_distances(instance):
    site_to_site_distances = instance['site_to_site_distances']
    customer_to_site_distances = instance['customer_to_site_distances']
    num_sites = instance['num_sites']
    num_customers = instance['num_customers']
    return site_to_site_distances, customer_to_site_distances

def compute_attraction(instance):
    num_sites = instance['num_sites']
    num_customers = instance['num_customers']
    num_periods = instance['num_periods']
    parameters = instance['parameters']
    q = parameters['q']
    weekly_attraction_cycle = parameters['weekly_attraction_cycle']
    avg_businesses_customers = parameters['avg_businesses_customers']
    avg_businesses_sites = parameters['avg_businesses_sites']
    customers = instance['customers']
    sites = instance['sites']
    customer_to_site_distances = instance['customer_to_site_distances']
    
    customer_self_attraction = {}
    for t in range(num_periods):
        for c in range(num_customers):
            customer = customers[c]
            day_of_week = t % 7
            relative_attractiveness = weekly_attraction_cycle[day_of_week]
            self_attraction = (customer['num_businesses'] / avg_businesses_customers) * relative_attractiveness
            customer_self_attraction[f'{c}_{t}'] = self_attraction
    
    site_attraction = {}
    for t in range(num_periods):
        for s in range(1, num_sites + 1): # Site IDs are 1-indexed
            site = next((site for site in sites if site['id'] == s), None)
            if site is None:
                raise ValueError(f"Site {s} not found in sites list.")
            day_of_week = t % 7
            relative_attractiveness = weekly_attraction_cycle[day_of_week]
            attraction = (site['num_businesses'] / avg_businesses_sites) * relative_attractiveness
            site_attraction[f'{s}_{t}'] = attraction
    
    customer_to_event_attraction = {}
    for c in range(num_customers):
        for s in range(1, num_sites + 1):
            for t in range(num_periods):
                distance = float(customer_to_site_distances[f'{c}_{s}'])
                if distance == 0:
                    attraction = 0  # Avoid division by zero
                else:
                    attraction = (site_attraction[f'{s}_{t}'] * weekly_attraction_cycle[t % 7]) / (distance ** q)
                customer_to_event_attraction[f'{c}_{s}_{t}'] = attraction
    
    return customer_self_attraction, site_attraction, customer_to_event_attraction

def formulate_model(instance, site_to_site_distances, customer_self_attraction, customer_to_event_attraction):
    num_sites = instance['num_sites']
    num_customers = instance['num_customers']
    num_periods = instance['num_periods']
    parameters = instance['parameters']
    T = parameters['T']
    delta = parameters['delta']
    epsilon = parameters['epsilon']
    L = parameters['L']
    c = parameters['c']
    sites = instance['sites']
    customers = instance['customers']
    
    model = gp.Model('TouringEntertainment')
    
    # Decision Variables
    x = model.addVars(num_periods, num_sites + 1, num_sites + 1, vtype=GRB.BINARY, name='x') # x[t, i, j] = 1 if move from site i to j at period t
    y = model.addVars(num_periods, num_sites + 1, vtype=GRB.BINARY, name='y') # y[t, j] = 1 if event held at site j in period t
    z = model.addVars(num_sites + 1, num_periods, vtype=GRB.INTEGER, name='z', lb=0) # z[j, t] = attendance at site j in period t
    p = model.addVars(num_customers, num_sites + 1, num_periods, vtype=GRB.CONTINUOUS, name='p', lb=0, ub=1) # p[c, j, t] = proportion of customers from c attending event at j in t
    
    # Objective Function
    revenue = gp.quicksum(sites[s-1]['revenue_per_customer'] * z[s, t] for s in range(1, num_sites+1) for t in range(num_periods))
    fixed_cost = gp.quicksum(sites[s-1]['fixed_cost'] * y[t, s] for s in range(1, num_sites+1) for t in range(num_periods))
    travel_cost = c * gp.quicksum(float(site_to_site_distances[f'{i}_{j}']) * x[t, i, j] for t in range(num_periods) for i in range(num_sites+1) for j in range(num_sites+1))
    model.setObjective(revenue - fixed_cost - travel_cost, GRB.MAXIMIZE)
    
    # Constraints
    # 1. Routing and Event Constraints
    for t in range(num_periods):
        for j in range(num_sites + 1):
            model.addConstr(y[t, j] <= gp.quicksum(x[t, i, j] for i in range(num_sites + 1)), name=f'event_routing_{t}_{j}')
    
    for t in range(num_periods):
        model.addConstr(gp.quicksum(x[t, i, j] for i in range(num_sites + 1) for j in range(num_sites + 1)) == 1, name=f'unique_move_{t}')
    
    for t in range(num_periods - 1):
        for j in range(num_sites + 1):
            model.addConstr(gp.quicksum(x[t, i, j] for i in range(num_sites + 1)) == gp.quicksum(x[t+1, j, k] for k in range(num_sites + 1)), name=f'flow_conservation_{t}_{j}')
    
    model.addConstr(gp.quicksum(x[0, 0, j] for j in range(num_sites + 1)) == 1, name='initial_move')
    
    for t in range(num_periods):
        model.addConstr(gp.quicksum(float(site_to_site_distances[f'{i}_{j}']) * x[t, i, j] for i in range(num_sites + 1) for j in range(num_sites + 1)) <= L, name=f'max_distance_{t}')
    
    # 2. Event and Attendance Constraints
    for s in range(1, num_sites + 1):
        model.addConstr(gp.quicksum(y[t, s] for t in range(num_periods)) <= sites[s-1]['max_events'], name=f'max_events_{s}')
    
    for t in range(num_periods - delta - 1):
        model.addConstr(gp.quicksum(y[tau, s] for tau in range(t, t + delta + 2) for s in range(1, num_sites + 1)) <= delta + 1, name=f'max_consecutive_events_{t}')
    
    for c in range(num_customers):
        for s in range(1, num_sites + 1):
            for t in range(num_periods):
                attraction_numerator = customer_to_event_attraction[f'{c}_{s}_{t}'] * y[t, s]
                attraction_denominator = customer_self_attraction[f'{c}_{t}']
                for s_prime in range(1, num_sites + 1):
                    for tau in range(max(0, t - epsilon), min(num_periods, t + epsilon + 1)):
                        attraction_denominator += customer_to_event_attraction[f'{c}_{s_prime}_{tau}'] * y[tau, s_prime]
                model.addConstr(p[c, s, t] * attraction_denominator <= attraction_numerator, name=f'attendance_proportion_{c}_{s}_{t}')
                model.addConstr(p[c, s, t] <= 1, name=f'attendance_proportion_ub_{c}_{s}_{t}')
    
    for s in range(1, num_sites + 1):
        for t in range(num_periods):
            model.addConstr(z[s, t] <= gp.quicksum(customers[c]['population'] * p[c, s, t] for c in range(num_customers)), name=f'attendance_ub_{s}_{t}')
            model.addConstr(z[s, t] <= sites[s-1]['capacity'] * y[t, s], name=f'capacity_{s}_{t}')
    
    return model, x, y, z, p

def extract_solution(instance, x, y, z):
    num_sites = instance['num_sites']
    num_periods = instance['num_periods']
    sites = instance['sites']
    
    tour = {}
    events = {}
    event_sites = {}
    attendance = {}
    
    for t in range(num_periods):
        for i in range(num_sites + 1):
            for j in range(num_sites + 1):
                if x[t, i, j].X > 0.5:
                    tour[t] = j
    
    for t in range(num_periods):
        for j in range(1, num_sites + 1):
            if y[t, j].X > 0.5:
                events[t] = 1
                event_sites[t] = j
    
    for s in range(1, num_sites + 1):
        for t in range(num_periods):
            attendance[f'{s}_{t}'] = z[s, t].X
    
    objective_value = sum(sites[s-1]['revenue_per_customer'] * z[s, t].X for s in range(1, num_sites+1) for t in range(num_periods)) - \
                      sum(sites[s-1]['fixed_cost'] * y[t, s].X for s in range(1, num_sites+1) for t in range(num_periods))
    
    # Compute travel cost
    site_to_site_distances = instance['site_to_site_distances']
    c = instance['parameters']['c']
    travel_cost = c * sum(float(site_to_site_distances[f'{i}_{j}']) * x[t, i, j].X for t in range(num_periods) for i in range(num_sites+1) for j in range(num_sites+1))
    objective_value -= travel_cost
    
    return {
        'objective_value': objective_value,
        'tour': tour,
        'events': events,
        'event_sites': event_sites,
        'attendance': attendance
    }

def main():
    parser = argparse.ArgumentParser(description='Solve Touring Entertainment Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to the output solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to the log file for intermediate solutions')
    args = parser.parse_args()
    
    instance = read_instance(args.instance_path)
    site_to_site_distances, customer_to_site_distances = compute_distances(instance)
    customer_self_attraction, site_attraction, customer_to_event_attraction = compute_attraction(instance)
    
    model, x, y, z, p = formulate_model(instance, site_to_site_distances, customer_self_attraction, customer_to_event_attraction)
    
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    model.setParam('TimeLimit', args.time_limit)
    model.optimize()
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        if logger and model.status == GRB.TIME_LIMIT:
            logger.log(model.ObjVal)
        solution = extract_solution(instance, x, y, z)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
    else:
        print(f'Model status: {model.status}')
        # Optionally, handle other statuses or log them

if __name__ == "__main__":
    main()