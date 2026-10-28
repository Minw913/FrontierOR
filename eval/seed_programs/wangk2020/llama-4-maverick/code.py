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
    model.setParam("Threads", 1)

    num_time_periods = instance['num_time_periods']
    maximum_displacement_delta = instance['maximum_displacement_delta']
    airports = instance['airports']
    flights = instance['flights']
    aircraft_connections = instance['aircraft_connections']
    passenger_connections = instance['passenger_connections']
    capacity_envelopes = instance['capacity_envelopes']
    scenarios = instance['scenarios']
    rho = 0.5  

    flight_dict = {flight['id']: flight for flight in flights}

    y_dep = {(flight['id'], t): model.addVar(vtype=GRB.BINARY) for flight in flights for t in range(max(0, flight['dep_period'] - maximum_displacement_delta), min(num_time_periods, flight['dep_period'] + maximum_displacement_delta + 1))}
    y_arr = {(flight['id'], t): model.addVar(vtype=GRB.BINARY) for flight in flights for t in range(max(0, flight['arr_period'] - maximum_displacement_delta), min(num_time_periods, flight['arr_period'] + maximum_displacement_delta + 1))}

    x_dep = {(scenario['id'], flight['id'], t): model.addVar(vtype=GRB.BINARY) for scenario in scenarios for flight in flights for t in range(num_time_periods + 1)}
    x_arr = {(scenario['id'], flight['id'], t): model.addVar(vtype=GRB.BINARY) for scenario in scenarios for flight in flights for t in range(num_time_periods + 1)}
    v_dep = {(scenario['id'], flight['id']): model.addVar(vtype=GRB.CONTINUOUS, lb=0) for scenario in scenarios for flight in flights}
    v_arr = {(scenario['id'], flight['id']): model.addVar(vtype=GRB.CONTINUOUS, lb=0) for scenario in scenarios for flight in flights}

    scheduled_dep = {flight['id']: model.addVar(vtype=GRB.INTEGER) for flight in flights}
    scheduled_arr = {flight['id']: model.addVar(vtype=GRB.INTEGER) for flight in flights}

    displacement_cost = {flight['id']: model.addVar(vtype=GRB.CONTINUOUS) for flight in flights}
    for flight in flights:
        model.addConstr(displacement_cost[flight['id']] >= scheduled_dep[flight['id']] - flight['dep_period'])
        model.addConstr(displacement_cost[flight['id']] >= flight['dep_period'] - scheduled_dep[flight['id']])

    schedule_displacement_cost = quicksum(rho * displacement_cost[flight['id']] for flight in flights)
    expected_operational_delay_cost = quicksum(scenario['probability'] * (1 - rho) * quicksum(flight['cost_dep_delay'] * v_dep[scenario['id'], flight['id']] + flight['cost_arr_delay'] * v_arr[scenario['id'], flight['id']] for flight in flights) for scenario in scenarios)
    model.setObjective(schedule_displacement_cost + expected_operational_delay_cost, GRB.MINIMIZE)

    for flight in flights:
        dep_range = range(max(0, flight['dep_period'] - maximum_displacement_delta), min(num_time_periods, flight['dep_period'] + maximum_displacement_delta + 1))
        arr_range = range(max(0, flight['arr_period'] - maximum_displacement_delta), min(num_time_periods, flight['arr_period'] + maximum_displacement_delta + 1))

        model.addConstr(scheduled_dep[flight['id']] == quicksum(t * (y_dep.get((flight['id'], t), 0) - y_dep.get((flight['id'], t-1), 0)) for t in dep_range))
        model.addConstr(scheduled_arr[flight['id']] == quicksum(t * (y_arr.get((flight['id'], t), 0) - y_arr.get((flight['id'], t-1), 0)) for t in arr_range))

        if (flight['id'], dep_range[0]-1) in y_dep:
            model.addConstr(y_dep[flight['id'], dep_range[0]-1] == 1)
        if (flight['id'], dep_range[-1]) in y_dep:
            model.addConstr(y_dep[flight['id'], dep_range[-1]] == 0)
        if (flight['id'], arr_range[0]-1) in y_arr:
            model.addConstr(y_arr[flight['id'], arr_range[0]-1] == 1)
        if (flight['id'], arr_range[-1]) in y_arr:
            model.addConstr(y_arr[flight['id'], arr_range[-1]] == 0)

        for t in dep_range[1:]:
            model.addConstr(y_dep[flight['id'], t] <= y_dep[flight['id'], t-1])
        for t in arr_range[1:]:
            model.addConstr(y_arr[flight['id'], t] <= y_arr[flight['id'], t-1])

        model.addConstr(scheduled_arr[flight['id']] - scheduled_dep[flight['id']] == flight['delta_sch'])

        for scenario in scenarios:
            for t in range(1, num_time_periods + 1):
                model.addConstr(x_dep[scenario['id'], flight['id'], t] <= x_dep[scenario['id'], flight['id'], t-1])
                model.addConstr(x_arr[scenario['id'], flight['id'], t] <= x_arr[scenario['id'], flight['id'], t-1])

            model.addConstr(x_dep[scenario['id'], flight['id'], 0] == 1)
            model.addConstr(x_arr[scenario['id'], flight['id'], 0] == 1)

            model.addConstr(v_dep[scenario['id'], flight['id']] == quicksum(x_dep[scenario['id'], flight['id'], t] for t in range(num_time_periods + 1)) - scheduled_dep[flight['id']])
            model.addConstr(v_arr[scenario['id'], flight['id']] == quicksum(x_arr[scenario['id'], flight['id'], t] for t in range(num_time_periods + 1)) - scheduled_arr[flight['id']])

            model.addConstr(quicksum(x_arr[scenario['id'], flight['id'], t] for t in range(num_time_periods + 1)) - quicksum(x_dep[scenario['id'], flight['id'], t] for t in range(num_time_periods + 1)) >= flight['delta_min'])
            model.addConstr(quicksum(x_arr[scenario['id'], flight['id'], t] for t in range(num_time_periods + 1)) - quicksum(x_dep[scenario['id'], flight['id'], t] for t in range(num_time_periods + 1)) <= flight['delta_max'])

            model.addConstr(v_dep[scenario['id'], flight['id']] <= flight['max_dep_delay'])
            model.addConstr(v_arr[scenario['id'], flight['id']] <= flight['max_arr_delay'])

    for connection in passenger_connections:
        flight_i = next((f for f in flights if f['id'] == connection['flight_i']), None)
        flight_j = next((f for f in flights if f['id'] == connection['flight_j']), None)
        if flight_i and flight_j:
            model.addConstr(scheduled_dep[flight_j['id']] - scheduled_arr[flight_i['id']] >= connection['min_connection_time'])

    for scenario in scenarios:
        for connection in aircraft_connections:
            flight_i = next((f for f in flights if f['id'] == connection['flight_i']), None)
            flight_j = next((f for f in flights if f['id'] == connection['flight_j']), None)
            if flight_i and flight_j:
                model.addConstr(quicksum(x_dep[scenario['id'], flight_j['id'], t] for t in range(num_time_periods + 1)) - quicksum(x_arr[scenario['id'], flight_i['id'], t] for t in range(num_time_periods + 1)) >= connection['min_connection_time'])

    for scenario in scenarios:
        for airport in airports:
            for t in range(num_time_periods):
                dep_count = quicksum(x_dep[scenario['id'], flight['id'], t+1] - x_dep[scenario['id'], flight['id'], t] for flight in flights if flight['dep_airport'] == airport)
                arr_count = quicksum(x_arr[scenario['id'], flight['id'], t+1] - x_arr[scenario['id'], flight['id'], t] for flight in flights if flight['arr_airport'] == airport)
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
            solution["schedule"][str(flight['id'])] = {'scheduled_dep': int(scheduled_dep[flight['id']].x), 'scheduled_arr': int(scheduled_arr[flight['id']].x)}
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
                for t in range(num_time_periods + 1):
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