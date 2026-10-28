import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def solve(instance):
    model = Model("Berth Allocation and Speed Optimization")
    model.Params.LogToConsole = 0
    
    # Extract data
    ships = instance['ships']
    ports = {port['id']: port for port in instance['ports']}
    external_ships = instance['external_ships']
    cost_params = instance['cost_parameters']
    speeds = instance['speeds_knots']
    distances = instance['distances_nautical_miles']
    travel_time_hours_per_nm_by_speed = instance['travel_time_hours_per_nm_by_speed']
    fuel_consumption_per_nm_by_speed = {float(k): v for k, v in instance['ships'][0]['fuel_consumption_per_nm_by_speed'].items()}
    handling_time_deviation_factor_beta = instance['handling_time_deviation_factor_beta_per_meter']
    planning_horizon_hours = instance['planning_horizon_hours']

    # Decision Variables
    berthing_positions = {(ship['id'], visit['port_id']): model.addVar(lb=0, ub=ports[visit['port_id']]['quay_length_meters'] - ship['length_meters'], name=f"berthing_position_{ship['id']}_{visit['port_id']}") for ship in ships for visit in ship['port_visits']}
    berthing_start_times = {(ship['id'], visit['port_id']): model.addVar(lb=visit['earliest_start_time_hours'], ub=planning_horizon_hours, name=f"berthing_start_time_{ship['id']}_{visit['port_id']}") for ship in ships for visit in ship['port_visits']}
    sailing_speeds = {(ship['id'], i): model.addVar(vtype=GRB.INTEGER, lb=0, ub=len(speeds)-1, name=f"sailing_speed_index_{ship['id']}_{i}") for ship in ships for i in range(len(ship['route'])-1)}
    handling_times = {(ship['id'], visit['port_id']): model.addVar(name=f"handling_time_{ship['id']}_{visit['port_id']}") for ship in ships for visit in ship['port_visits']}
    arrival_times = {(ship['id'], visit['port_id']): model.addVar(name=f"arrival_time_{ship['id']}_{visit['port_id']}") for ship in ships for visit in ship['port_visits']}
    delays = {(ship['id'], visit['port_id']): model.addVar(lb=0, name=f"delay_{ship['id']}_{visit['port_id']}") for ship in ships for visit in ship['port_visits']}
    lft_violations = {(ship['id'], visit['port_id']): model.addVar(lb=0, name=f"lft_violation_{ship['id']}_{visit['port_id']}") for ship in ships for visit in ship['port_visits']}
    waiting_times = {(ship['id'], visit['port_id']): model.addVar(lb=0, name=f"waiting_time_{ship['id']}_{visit['port_id']}") for ship in ships for visit in ship['port_visits']}

    # Constraints
    for ship in ships:
        for i, visit in enumerate(ship['port_visits']):
            port_id = visit['port_id']
            ship_id = ship['id']
            model.addConstr(handling_times[ship_id, port_id] == visit['min_handling_time_hours'] * (1 + handling_time_deviation_factor_beta * abs(berthing_positions[ship_id, port_id] - visit['ideal_berthing_position'])))
            model.addConstr(arrival_times[ship_id, port_id] <= berthing_start_times[ship_id, port_id])
            model.addConstr(berthing_start_times[ship_id, port_id] >= visit['earliest_start_time_hours'])
            model.addConstr(delays[ship_id, port_id] == GRB.max(0, berthing_start_times[ship_id, port_id] + handling_times[ship_id, port_id] - visit['expected_finish_time_hours']))
            model.addConstr(lft_violations[ship_id, port_id] == GRB.max(0, berthing_start_times[ship_id, port_id] + handling_times[ship_id, port_id] - visit['latest_finish_time_hours']))
            model.addConstr(waiting_times[ship_id, port_id] == berthing_start_times[ship_id, port_id] - arrival_times[ship_id, port_id])
            if i > 0:
                prev_visit = ship['port_visits'][i-1]
                prev_port_id = prev_visit['port_id']
                distance_nm = distances[f"{prev_port_id}_{port_id}"]
                speed_index = sailing_speeds[ship_id, i-1]
                model.addConstr(arrival_times[ship_id, port_id] == berthing_start_times[ship_id, prev_port_id] + handling_times[ship_id, prev_port_id] + distance_nm * travel_time_hours_per_nm_by_speed[speeds[int(speed_index)]])

    for port in external_ships:
        port_id = port['port_id']
        for external_ship in port['ships']:
            for ship in ships:
                for visit in ship['port_visits']:
                    if visit['port_id'] == port_id:
                        ship_id = ship['id']
                        model.addConstr(berthing_start_times[ship_id, port_id] + handling_times[ship_id, port_id] <= external_ship['start_time_hours'] or berthing_start_times[ship_id, port_id] >= external_ship['end_time_hours'] or berthing_positions[ship_id, port_id] + ship['length_meters'] <= external_ship['berthing_position'] or berthing_positions[ship_id, port_id] >= external_ship['berthing_position'] + external_ship['length_meters'])

    for ship1 in ships:
        for ship2 in ships:
            if ship1['id'] != ship2['id']:
                common_ports = set([visit['port_id'] for visit in ship1['port_visits']]) & set([visit['port_id'] for visit in ship2['port_visits']])
                for port_id in common_ports:
                    visit1 = next(visit for visit in ship1['port_visits'] if visit['port_id'] == port_id)
                    visit2 = next(visit for visit in ship2['port_visits'] if visit['port_id'] == port_id)
                    model.addConstr(berthing_start_times[ship1['id'], port_id] + handling_times[ship1['id'], port_id] <= berthing_start_times[ship2['id'], port_id] or berthing_start_times[ship2['id'], port_id] + handling_times[ship2['id'], port_id] <= berthing_start_times[ship1['id'], port_id] or berthing_positions[ship1['id'], port_id] + ship1['length_meters'] <= berthing_positions[ship2['id'], port_id] or berthing_positions[ship2['id'], port_id] + ship2['length_meters'] <= berthing_positions[ship1['id'], port_id])

    # Objective
    total_cost = quicksum([
        waiting_times[ship['id'], visit['port_id']] * cost_params['waiting_cost_usd_per_hour'] +
        handling_times[ship['id'], visit['port_id']] * cost_params['handling_cost_usd_per_hour'] +
        delays[ship['id'], visit['port_id']] * cost_params['delay_cost_usd_per_hour'] +
        lft_violations[ship['id'], visit['port_id']] * cost_params['lft_penalty_usd'] +
        (distances[f"{ship['route'][i]}_{ship['route'][i+1]}"] * fuel_consumption_per_nm_by_speed[speeds[int(sailing_speeds[ship['id'], i])]] * cost_params['fuel_price_usd_per_tonne'] if i < len(ship['route'])-1 else 0)
        for ship in ships for i, visit in enumerate(ship['port_visits'])
    ])
    model.setObjective(total_cost, GRB.MINIMIZE)

    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    model = solve(instance)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model.Params.TimeLimit = args.time_limit
    model.optimize()

    if logger and model.SolCount > 0:
        logger.log(model.ObjVal)

    if model.SolCount > 0:
        solution = {
            "objective_value": model.ObjVal,
            "ships": []
        }
        for ship in instance['ships']:
            ship_solution = {
                "ship_id": ship['id'],
                "port_visits": [],
                "speed_selections": []
            }
            for i, visit in enumerate(ship['port_visits']):
                port_id = visit['port_id']
                ship_id = ship['id']
                ship_solution['port_visits'].append({
                    "port_call_index": i+1,
                    "port_id": port_id,
                    "berthing_position": model.getVarByName(f"berthing_position_{ship_id}_{port_id}").X,
                    "start_time": model.getVarByName(f"berthing_start_time_{ship_id}_{port_id}").X,
                    "handling_time": model.getVarByName(f"handling_time_{ship_id}_{port_id}").X,
                    "arrival_time": model.getVarByName(f"arrival_time_{ship_id}_{port_id}").X,
                    "delay": model.getVarByName(f"delay_{ship_id}_{port_id}").X,
                    "lft_violation": model.getVarByName(f"lft_violation_{ship_id}_{port_id}").X,
                    "position_deviation": abs(model.getVarByName(f"berthing_position_{ship_id}_{port_id}").X - visit['ideal_berthing_position'])
                })
                if i < len(ship['route']) - 1:
                    ship_solution['speed_selections'].append({
                        "from_call": i+1,
                        "to_call": i+2,
                        "speed_knots": instance['speeds_knots'][int(model.getVarByName(f"sailing_speed_index_{ship_id}_{i}").X)]
                    })
            solution['ships'].append(ship_solution)
    else:
        solution = {
            "objective_value": float('inf'),
            "ships": []
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()