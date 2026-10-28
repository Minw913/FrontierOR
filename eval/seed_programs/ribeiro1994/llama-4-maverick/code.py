import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve(instance, time_limit, log_path=None):
    try:
        n_trips = instance['parameters']['n_trips']
        m_depots = instance['parameters']['m_depots']
        compatible_arcs = instance['compatible_arcs']
        depot_to_trip_costs = instance['depot_to_trip_costs']
        trip_to_depot_costs = instance['trip_to_depot_costs']
        vehicle_capacities = instance['vehicle_capacities']

        model = gp.Model('vehicle_routing')
        model.setParam("Threads", 1)
        model.Params.LogToConsole = 0
        model.Params.TimeLimit = time_limit

        x_keys = [(arc['from_trip'], arc['to_trip']) for arc in compatible_arcs]
        y_keys = [(cost['depot'], cost['trip']) for cost in depot_to_trip_costs]
        z_keys = [(cost['trip'], cost['depot']) for cost in trip_to_depot_costs]

        x = model.addVars(x_keys, vtype=GRB.BINARY, name='x')
        y = model.addVars(y_keys, vtype=GRB.BINARY, name='y')
        z = model.addVars(z_keys, vtype=GRB.BINARY, name='z')

        model.setObjective(
            gp.quicksum(x[from_trip, to_trip] * next((arc['cost'] for arc in compatible_arcs if arc['from_trip'] == from_trip and arc['to_trip'] == to_trip), 0) for from_trip, to_trip in x_keys) +
            gp.quicksum(y[depot, trip] * next((cost['cost'] for cost in depot_to_trip_costs if cost['depot'] == depot and cost['trip'] == trip), 0) for depot, trip in y_keys) +
            gp.quicksum(z[trip, depot] * next((cost['cost'] for cost in trip_to_depot_costs if cost['trip'] == trip and cost['depot'] == depot), 0) for trip, depot in z_keys),
            GRB.MINIMIZE
        )

        for trip in range(1, n_trips + 1):
            model.addConstr(gp.quicksum(y[depot, trip] for depot in range(1, m_depots + 1) if (depot, trip) in y_keys) == 1, name=f'trip_start_{trip}')
            model.addConstr(gp.quicksum(z[trip, depot] for depot in range(1, m_depots + 1) if (trip, depot) in z_keys) == 1, name=f'trip_end_{trip}')

        for depot in range(1, m_depots + 1):
            depot_y_keys = [key for key in y_keys if key[0] == depot]
            depot_z_keys = [key for key in z_keys if key[1] == depot]
            model.addConstr(gp.quicksum(y[key] for key in depot_y_keys) == gp.quicksum(z[key] for key in depot_z_keys), name=f'flow_balance_{depot}')
            model.addConstr(gp.quicksum(y[key] for key in depot_y_keys) <= vehicle_capacities[depot - 1], name=f'vehicle_capacity_{depot}')

        for trip in range(1, n_trips + 1):
            incoming_keys = [key for key in x_keys if key[1] == trip]
            outgoing_keys = [key for key in x_keys if key[0] == trip]
            y_key = [key for key in y_keys if key[1] == trip]
            z_key = [key for key in z_keys if key[0] == trip]
            model.addConstr(gp.quicksum(x[key] for key in incoming_keys) + (y[y_key[0]] if y_key else 0) == 
                            gp.quicksum(x[key] for key in outgoing_keys) + (z[z_key[0]] if z_key else 0), 
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
            return None, None, None, None
    except gp.GurobiError as e:
        print(f'Gurobi error: {e}')
        return None, None, None, None

def construct_solution(instance, model, x, y, z):
    if model is None:
        return {'routes': [], 'objective_value': float('inf'), 'num_vehicles': 0}

    n_trips = instance['parameters']['n_trips']
    m_depots = instance['parameters']['m_depots']
    y_keys = [(cost['depot'], cost['trip']) for cost in instance['depot_to_trip_costs']]
    z_keys = [(cost['trip'], cost['depot']) for cost in instance['trip_to_depot_costs']]
    x_keys = [(arc['from_trip'], arc['to_trip']) for arc in instance['compatible_arcs']]

    solution = {'routes': [], 'objective_value': model.ObjVal, 'num_vehicles': 0}

    for depot in range(1, m_depots + 1):
        depot_trips = []
        for trip in range(1, n_trips + 1):
            if (depot, trip) in y_keys and y[depot, trip].X > 0.5:
                route = [trip]
                current_trip = trip
                while True:
                    next_trip = None
                    for next_trip_candidate in range(1, n_trips + 1):
                        if (current_trip, next_trip_candidate) in x_keys and x[current_trip, next_trip_candidate].X > 0.5:
                            next_trip = next_trip_candidate
                            break
                    if next_trip is not None:
                        route.append(next_trip)
                        current_trip = next_trip
                    else:
                        break
                depot_trips.append(route)

        for route in depot_trips:
            if len(route) > 0 and (route[-1], depot) in z_keys and z[route[-1], depot].X > 0.5:
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
    if model is not None:
        solution = construct_solution(instance, model, x, y, z)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()