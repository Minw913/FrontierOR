import json
import argparse
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve(instance, time_limit, log_path=None):
    n_trips = instance['parameters']['n_trips']
    m_depots = instance['parameters']['m_depots']
    trips = instance['trips']
    compatible_arcs = instance['compatible_arcs']
    depot_to_trip_costs = instance['depot_to_trip_costs']
    trip_to_depot_costs = instance['trip_to_depot_costs']
    vehicle_capacities = instance['vehicle_capacities']

    model = gp.Model('vehicle_routing')
    model.Params.LogToConsole = 0
    model.Params.TimeLimit = time_limit

    # Variables
    x = model.addVars([(from_trip['from_trip'], to_trip['to_trip']) for from_trip in compatible_arcs for to_trip in compatible_arcs if from_trip['to_trip'] == to_trip['from_trip']], vtype=GRB.BINARY, name='x')
    y = model.addVars([(depot['depot'], trip['trip']) for depot in depot_to_trip_costs for trip in depot_to_trip_costs if depot['trip'] == trip['trip']], vtype=GRB.BINARY, name='y')
    z = model.addVars([(trip['trip'], depot['depot']) for trip in trip_to_depot_costs for depot in trip_to_depot_costs if trip['trip'] == trip['trip'] and depot['depot'] == depot['depot']], vtype=GRB.BINARY, name='z')
    flow = model.addVars(range(m_depots), vtype=GRB.INTEGER, name='flow')

    # Objective
    model.setObjective(
        gp.quicksum(x[from_trip, to_trip] * next((arc['cost'] for arc in compatible_arcs if arc['from_trip'] == from_trip and arc['to_trip'] == to_trip), 0) for from_trip in range(1, n_trips + 1) for to_trip in range(1, n_trips + 1) if (from_trip, to_trip) in x) +
        gp.quicksum(y[depot, trip] * next((cost['cost'] for cost in depot_to_trip_costs if cost['depot'] == depot and cost['trip'] == trip), 0) for depot in range(1, m_depots + 1) for trip in range(1, n_trips + 1) if (depot, trip) in y) +
        gp.quicksum(z[trip, depot] * next((cost['cost'] for cost in trip_to_depot_costs if cost['trip'] == trip and cost['depot'] == depot), 0) for trip in range(1, n_trips + 1) for depot in range(1, m_depots + 1) if (trip, depot) in z),
        GRB.MINIMIZE
    )

    # Constraints
    for trip in range(1, n_trips + 1):
        model.addConstr(gp.quicksum(y[depot, trip] for depot in range(1, m_depots + 1) if (depot, trip) in y) == 1, name=f'trip_start_{trip}')
        model.addConstr(gp.quicksum(z[trip, depot] for depot in range(1, m_depots + 1) if (trip, depot) in z) == 1, name=f'trip_end_{trip}')

    for depot in range(1, m_depots + 1):
        model.addConstr(flow[depot - 1] == gp.quicksum(y[depot, trip] for trip in range(1, n_trips + 1) if (depot, trip) in y), name=f'flow_out_{depot}')
        model.addConstr(flow[depot - 1] == gp.quicksum(z[trip, depot] for trip in range(1, n_trips + 1) if (trip, depot) in z), name=f'flow_in_{depot}')
        model.addConstr(flow[depot - 1] <= vehicle_capacities[depot - 1], name=f'vehicle_capacity_{depot}')

    for trip in range(1, n_trips + 1):
        model.addConstr(gp.quicksum(x[from_trip, trip] for from_trip in range(1, n_trips + 1) if (from_trip, trip) in x) + gp.quicksum(y[depot, trip] for depot in range(1, m_depots + 1) if (depot, trip) in y) == 
                        gp.quicksum(x[trip, to_trip] for to_trip in range(1, n_trips + 1) if (trip, to_trip) in x) + gp.quicksum(z[trip, depot] for depot in range(1, m_depots + 1) if (trip, depot) in z), 
                        name=f'flow_conservation_{trip}')

    model.optimize()

    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    if logger:
        for sol_count in range(model.SolCount):
            model.Params.SolutionNumber = sol_count
            objective_value = model.PoolObjVal
            logger.log(objective_value)

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        return model, x, y, z
    else:
        raise Exception(f"Model status: {model.status}")

def construct_solution(instance, model, x, y, z):
    n_trips = instance['parameters']['n_trips']
    m_depots = instance['parameters']['m_depots']
    trips = instance['trips']
    depot_to_trip_costs = instance['depot_to_trip_costs']
    trip_to_depot_costs = instance['trip_to_depot_costs']

    solution = {'routes': [], 'objective_value': model.ObjVal, 'num_vehicles': 0}

    for depot in range(1, m_depots + 1):
        depot_trips = []
        for trip in range(1, n_trips + 1):
            if (depot, trip) in y and y[depot, trip].X > 0.5:
                route = [trip]
                current_trip = trip
                while True:
                    next_trip = None
                    for next_trip_candidate in range(1, n_trips + 1):
                        if (current_trip, next_trip_candidate) in x and x[current_trip, next_trip_candidate].X > 0.5:
                            next_trip = next_trip_candidate
                            break
                    if next_trip is not None:
                        route.append(next_trip)
                        current_trip = next_trip
                    else:
                        break
                depot_trips.append(route)

        for route in depot_trips:
            if len(route) > 0 and any((route[-1], depot) in z and z[route[-1], depot].X > 0.5 for _ in range(1)):
                solution['routes'].append({'depot': depot, 'trips': route})
                solution['num_vehicles'] += 1

    solution['objective_value'] = model.ObjVal
    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    model, x, y, z = solve(instance, args.time_limit, args.log_path)
    solution = construct_solution(instance, model, x, y, z)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()