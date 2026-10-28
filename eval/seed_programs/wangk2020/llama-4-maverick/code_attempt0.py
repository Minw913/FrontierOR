import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        return json.load(file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model = Model("AirTrafficManagement")

    # Extract data from instance
    num_time_periods = instance['num_time_periods']
    maximum_displacement_delta = instance['maximum_displacement_delta']
    airports = instance['airports']
    flights = instance['flights']
    aircraft_connections = instance['aircraft_connections']
    passenger_connections = instance['passenger_connections']
    capacity_envelopes = instance['capacity_envelopes']
    scenarios = instance['scenarios']
    rho = 0.5  # Assuming rho is given or fixed, adjust as necessary

    # Create dictionaries for easy access
    flight_dict = {flight['id']: flight for flight in flights}

    # First-stage variables
    y_dep = {(flight['id'], t): model.addVar(vtype=GRB.BINARY) for flight in flights for t in range(max(0, flight['dep_period'] - maximum_displacement_delta), min(num_time_periods, flight['dep_period'] + maximum_displacement_delta + 1))}
    y_arr = {(flight['id'], t): model.addVar(vtype=GRB.BINARY) for flight in flights for t in range(max(0, flight['arr_period'] - maximum_displacement_delta), min(num_time_periods, flight['arr_period'] + maximum_displacement_delta + 1))}

    # Second-stage variables
    x_dep = {(scenario['id'], flight['id'], t): model.addVar(vtype=GRB.BINARY) for scenario in scenarios for flight in flights for t in range(num_time_periods)}
    x_arr = {(scenario['id'], flight['id'], t): model.addVar(vtype=GRB.BINARY) for scenario in scenarios for flight in flights for t in range(num_time_periods)}
    v_dep = {(scenario['id'], flight['id']): model.addVar(vtype=GRB.CONTINUOUS, lb=0) for scenario in scenarios for flight in flights}
    v_arr = {(scenario['id'], flight['id']): model.addVar(vtype=GRB.CONTINUOUS, lb=0) for scenario in scenarios for flight in flights}

    # Objective function
    schedule_displacement_cost = quicksum(rho * flight['cost_dep_delay'] * (quicksum(t * (y_dep[flight['id'], t] - y_dep[flight['id'], t-1]) for t in range(max(0, flight['dep_period'] - maximum_displacement_delta), min(num_time_periods, flight['dep_period'] + maximum_displacement_delta + 1))) - flight['dep_period']) for flight in flights)
    expected_operational_delay_cost = quicksum(scenario['probability'] * (1 - rho) * quicksum(flight['cost_dep_delay'] * v_dep[scenario['id'], flight['id']] + flight['cost_arr_delay'] * v_arr[scenario['id'], flight['id']] for flight in flights) for scenario in scenarios)
    model.setObjective(schedule_displacement_cost + expected_operational_delay_cost, GRB.MINIMIZE)

    # Constraints
    for flight in flights:
        # Scheduled departure and arrival indicators are non-increasing
        for t in range(max(0, flight['dep_period'] - maximum_displacement_delta) + 1, min(num_time_periods, flight['dep_period'] + maximum_displacement_delta + 1)):
            model.addConstr(y_dep[flight['id'], t] <= y_dep[flight['id'], t-1])
        for t in range(max(0, flight['arr_period'] - maximum_displacement_delta) + 1, min(num_time_periods, flight['arr_period'] + maximum_displacement_delta + 1)):
            model.addConstr(y_arr[flight['id'], t] <= y_arr[flight['id'], t-1])

        # En-route duration constraint for scheduled times
        model.addConstr(quicksum(y_arr[flight['id'], t] - y_dep[flight['id'], t] for t in range(max(0, flight['dep_period'] - maximum_displacement_delta), min(num_time_periods, flight['dep_period'] + maximum_displacement_delta + 1))) == flight['delta_sch'])

        # Connection constraints for passenger and aircraft connections
        for connection in passenger_connections:
            if connection['flight_i'] == flight['id']:
                flight_j = next((f for f in flights if f['id'] == connection['flight_j']), None)
                if flight_j:
                    model.addConstr(quicksum(y_dep[flight_j['id'], t] - y_arr[flight['id'], t] for t in range(max(0, flight['arr_period'] - maximum_displacement_delta), min(num_time_periods, flight_j['dep_period'] + maximum_displacement_delta + 1))) >= connection['min_connection_time'])
        for connection in aircraft_connections:
            if connection['flight_i'] == flight['id']:
                flight_j = next((f for f in flights if f['id'] == connection['flight_j']), None)
                if flight_j:
                    for scenario in scenarios:
                        model.addConstr(quicksum(x_dep[scenario['id'], flight_j['id'], t] - x_arr[scenario['id'], flight['id'], t] for t in range(num_time_periods)) >= connection['min_connection_time'])

        # Operated departure and arrival indicators are non-increasing
        for scenario in scenarios:
            for t in range(1, num_time_periods):
                model.addConstr(x_dep[scenario['id'], flight['id'], t] <= x_dep[scenario['id'], flight['id'], t-1])
                model.addConstr(x_arr[scenario['id'], flight['id'], t] <= x_arr[scenario['id'], flight['id'], t-1])

            # Departure and arrival delay calculations
            model.addConstr(v_dep[scenario['id'], flight['id']] >= quicksum(x_dep[scenario['id'], flight['id'], t] - y_dep[flight['id'], t] for t in range(max(0, flight['dep_period'] - maximum_displacement_delta), min(num_time_periods, flight['dep_period'] + maximum_displacement_delta + 1))))
            model.addConstr(v_arr[scenario['id'], flight['id']] >= quicksum(x_arr[scenario['id'], flight['id'], t] - y_arr[flight['id'], t] for t in range(max(0, flight['arr_period'] - maximum_displacement_delta), min(num_time_periods, flight['arr_period'] + maximum_displacement_delta + 1))))

            # En-route duration constraints for operated times
            model.addConstr(quicksum(x_arr[scenario['id'], flight['id'], t] - x_dep[scenario['id'], flight['id'], t] for t in range(num_time_periods)) >= flight['delta_min'])
            model.addConstr(quicksum(x_arr[scenario['id'], flight['id'], t] - x_dep[scenario['id'], flight['id'], t] for t in range(num_time_periods)) <= flight['delta_max'])

            # Maximum delay constraints
            model.addConstr(v_dep[scenario['id'], flight['id']] <= flight['max_dep_delay'])
            model.addConstr(v_arr[scenario['id'], flight['id']] <= flight['max_arr_delay'])

    # Airport capacity envelope constraints
    for scenario in scenarios:
        for airport in airports:
            for t in range(num_time_periods):
                dep_count = quicksum(x_dep[scenario['id'], flight['id'], t] - x_dep[scenario['id'], flight['id'], t-1] if t > 0 else x_dep[scenario['id'], flight['id'], t] for flight in flights if flight['dep_airport'] == airport)
                arr_count = quicksum(x_arr[scenario['id'], flight['id'], t] - x_arr[scenario['id'], flight['id'], t-1] if t > 0 else x_arr[scenario['id'], flight['id'], t] for flight in flights if flight['arr_airport'] == airport)
                condition = scenario['operating_conditions'][airport][t]
                for envelope in capacity_envelopes[airport][condition]:
                    model.addConstr(envelope['a'] * dep_count + envelope['b'] * arr_count <= envelope['Q'])

    model.setParam('TimeLimit', args.time_limit)
    model.optimize()

    if logger and model.SolCount > 0:
        logger.log(model.ObjVal)

    if model.SolCount > 0:
        solution = {
            "objective_value": model.ObjVal,
            "schedule": {},
            "scenario_solutions": []
        }
        for flight in flights:
            scheduled_dep = next((t for t in range(max(0, flight['dep_period'] - maximum_displacement_delta), min(num_time_periods, flight['dep_period'] + maximum_displacement_delta + 1)) if y_dep[flight['id'], t].x < 0.5), None)
            scheduled_arr = next((t for t in range(max(0, flight['arr_period'] - maximum_displacement_delta), min(num_time_periods, flight['arr_period'] + maximum_displacement_delta + 1)) if y_arr[flight['id'], t].x < 0.5), None)
            solution["schedule"][str(flight['id'])] = {'scheduled_dep': scheduled_dep, 'scheduled_arr': scheduled_arr}
        for scenario in scenarios:
            scenario_solution = {
                "scenario_id": scenario['id'],
                "x_dep": {},
                "x_arr": {},
                "v_dep": {},
                "v_arr": {}
            }
            for flight in flights:
                scenario_solution["v_dep"][str(flight['id'])] = v_dep[scenario['id'], flight['id']].x
                scenario_solution["v_arr"][str(flight['id'])] = v_arr[scenario['id'], flight['id']].x
                for t in range(num_time_periods):
                    if x_dep[scenario['id'], flight['id'], t].x > 0.5:
                        scenario_solution["x_dep"][f"{flight['id']}_{t}"] = 1
                    if x_arr[scenario['id'], flight['id'], t].x > 0.5:
                        scenario_solution["x_arr"][f"{flight['id']}_{t}"] = 1
            solution["scenario_solutions"].append(scenario_solution)
        with open(args.solution_path, 'w') as file:
            json.dump(solution, file, indent=4)
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()