import argparse
import json
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve(instance, time_limit):
    try:
        # Extract relevant data from the instance
        parameters = instance['parameters']
        fleet = instance['fleet']
        depot = instance['depot']
        truck_customers = instance['truck_customers']
        trailer_customers = instance['trailer_customers']
        transshipment_locations = instance['transshipment_locations']
        distance_matrix = instance['distance_matrix']
        travel_time_matrix = instance['travel_time_matrix']

        # Define sets and parameters
        truck_types = fleet['truck_types']
        trailer_types = fleet['trailer_types']
        vehicle_classes = fleet['vehicle_classes']

        customers = truck_customers + trailer_customers
        customer_ids = [c['id'] for c in customers]
        truck_customer_ids = [c['id'] for c in truck_customers]
        trailer_customer_ids = [c['id'] for c in trailer_customers]
        transshipment_location_ids = [l['id'] for l in transshipment_locations]
        parking_location_ids = trailer_customer_ids + transshipment_location_ids

        num_locations = len(customers) + len(transshipment_locations) + 1  # Including depot
        location_ids = list(range(num_locations))

        # Create Gurobi model
        model = gp.Model('VRPTW')
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = time_limit

        # Decision variables
        x = model.addVars([(v['class_id'], i, j) for v in vehicle_classes for i in location_ids for j in location_ids], vtype=GRB.BINARY, name='x')
        y = model.addVars([(v['class_id'], c) for v in vehicle_classes for c in customer_ids], vtype=GRB.BINARY, name='y')
        u = model.addVars([(v['class_id'], i) for v in vehicle_classes for i in location_ids], vtype=GRB.CONTINUOUS, name='u')
        t = model.addVars([(v['class_id'], i) for v in vehicle_classes for i in location_ids], vtype=GRB.CONTINUOUS, name='t')
        decouple = model.addVars([(v['class_id'], i) for v in vehicle_classes for i in parking_location_ids], vtype=GRB.BINARY, name='decouple')
        couple = model.addVars([(v['class_id'], i) for v in vehicle_classes for i in parking_location_ids], vtype=GRB.BINARY, name='couple')
        transfer = model.addVars([(v['class_id'], i) for v in vehicle_classes for i in parking_location_ids], vtype=GRB.BINARY, name='transfer')

        # Objective function
        model.setObjective(
            gp.quicksum(v['truck_fixed_cost'] * gp.quicksum(x[v['class_id'], depot['id'], j] for j in location_ids) for v in vehicle_classes) +
            gp.quicksum((v['truck_cost_factor'] + (v['trailer_towing_cost_multiplier'] if v['trailer_type_id'] != 0 else 0)) * distance_matrix[i][j] * x[v['class_id'], i, j] for v in vehicle_classes for i in location_ids for j in location_ids) +
            parameters['decoupling_cost'] * gp.quicksum(decouple[v['class_id'], i] for v in vehicle_classes for i in parking_location_ids) +
            parameters['coupling_cost'] * gp.quicksum(couple[v['class_id'], i] for v in vehicle_classes for i in parking_location_ids) +
            parameters['transfer_cost'] * gp.quicksum(transfer[v['class_id'], i] for v in vehicle_classes for i in parking_location_ids),
            GRB.MINIMIZE
        )

        # Constraints
        # Each customer is served exactly once
        model.addConstrs((gp.quicksum(y[v['class_id'], c] for v in vehicle_classes) == 1 for c in customer_ids), name='customer_served_once')

        # Flow conservation
        model.addConstrs((gp.quicksum(x[v['class_id'], i, j] for j in location_ids) == gp.quicksum(x[v['class_id'], j, i] for j in location_ids) for v in vehicle_classes for i in location_ids), name='flow_conservation')

        # Capacity and load constraints
        model.addConstrs((u[v['class_id'], j] >= u[v['class_id'], i] + customers[c]['supply'] - (1 - x[v['class_id'], i, j]) * v['total_capacity'] for v in vehicle_classes for i in location_ids for j in location_ids if i in customer_ids and j in location_ids), name='capacity_update')
        model.addConstrs((u[v['class_id'], i] <= v['total_capacity'] for v in vehicle_classes for i in location_ids), name='capacity_limit')

        # Time window constraints
        model.addConstrs((t[v['class_id'], j] >= t[v['class_id'], i] + travel_time_matrix[i][j] + customers[j]['service_time'] - (1 - x[v['class_id'], i, j]) * parameters['max_route_duration'] for v in vehicle_classes for i in customer_ids for j in customer_ids), name='time_update_customer')
        model.addConstrs((t[v['class_id'], j] >= t[v['class_id'], i] + travel_time_matrix[i][j] - (1 - x[v['class_id'], i, j]) * parameters['max_route_duration'] for v in vehicle_classes for i in location_ids if i not in customer_ids for j in location_ids), name='time_update_non_customer')

        model.optimize()

        # Construct solution
        solution = {
            'objective_value': model.ObjVal if model.status == GRB.OPTIMAL else float('inf'),
            'routes': []
        }

        return solution
    except gp.GurobiError as e:
        print(f'Gurobi error: {e}')
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)
    solution = solve(instance, args.time_limit)

    if solution is not None:
        if logger:
            logger.log_solution(solution['objective_value'], solution)

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()