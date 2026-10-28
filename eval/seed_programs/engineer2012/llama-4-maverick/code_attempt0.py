import argparse
import json
from gurobipy import Model, GRB, quicksum
import numpy as np
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        instance = json.load(file)
    return instance

def solve(instance, time_limit):
    # Extract data from instance
    planning_horizon = instance['planning_horizon']
    num_vessels = instance['num_vessels']
    num_supply_ports = instance['num_supply_ports']
    num_demand_ports = instance['num_demand_ports']
    vessels = instance['vessels']
    supply_ports = instance['supply_ports']
    demand_ports = instance['demand_ports']
    travel_times = instance['travel_times']
    draft_limits = instance['draft_limits']

    # Combine supply and demand ports into a single list for easier access
    ports = supply_ports + demand_ports
    port_dict = {port['port_id']: port for port in ports}

    # Create a Gurobi model
    model = Model("Shipping_Scheduling")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit

    # Define sets for vessels, ports, and time periods
    vessel_ids = [vessel['id'] for vessel in vessels]
    port_ids = [port['port_id'] for port in ports]
    time_periods = range(planning_horizon + 1)

    # Create variables
    # Binary variable indicating whether a vessel visits a port at a given time
    visit = model.addVars(vessel_ids, port_ids, time_periods, vtype=GRB.BINARY, name='visit')
    # Quantity of product loaded or discharged by a vessel at a port and time
    quantity = model.addVars(vessel_ids, port_ids, time_periods, vtype=GRB.CONTINUOUS, name='quantity', lb=-GRB.INFINITY)
    # Amount of cargo on board a vessel at a given time
    cargo = model.addVars(vessel_ids, time_periods, vtype=GRB.CONTINUOUS, name='cargo')
    # Inventory level at a port and time
    inventory = model.addVars(port_ids, time_periods, vtype=GRB.CONTINUOUS, name='inventory')

    # Objective: Minimize total cost
    transport_cost = quicksum(vessels[i]['transport_cost_per_day'] * travel_times[f"{port1}->{port2}"] * visit[vessel_ids[i], port2, t + travel_times[f"{port1}->{port2}"]] 
                              for i in range(num_vessels) for port1 in port_ids for port2 in port_ids for t in time_periods if t + travel_times[f"{port1}->{port2}"] <= planning_horizon)
    demurrage_cost = quicksum(vessels[i]['demurrage_cost_per_day'] * visit[vessel_ids[i], port, t] 
                              for i in range(num_vessels) for port in port_ids for t in time_periods)
    procurement_cost = quicksum(port_dict[port]['unit_procurement_cost'] * quantity[vessel_ids[i], port, t] 
                                for i in range(num_vessels) for port in port_ids if port_dict[port]['type'] == 'supply' for t in time_periods)
    supply_revenue = quicksum(port_dict[port]['unit_supply_revenue'] * -quantity[vessel_ids[i], port, t] 
                              for i in range(num_vessels) for port in port_ids if port_dict[port]['type'] == 'demand' for t in time_periods)
    model.setObjective(transport_cost + demurrage_cost + procurement_cost - supply_revenue, GRB.MINIMIZE)

    # Constraints
    # 1. Inventory constraints at ports
    for port in port_ids:
        for t in time_periods[1:]:
            production_or_consumption = port_dict[port]['production_rate'][t-1] if port_dict[port]['type'] == 'supply' else -port_dict[port]['consumption_rate'][t-1]
            model.addConstr(inventory[port, t] == inventory[port, t-1] + production_or_consumption + quicksum(quantity[vessel, port, t-1] for vessel in vessel_ids), name=f'Inventory_balance_{port}_{t}')

    # 2. Inventory bounds at ports
    for port in port_ids:
        for t in time_periods:
            model.addConstr(inventory[port, t] >= 0, name=f'Inventory_lower_bound_{port}_{t}')
            model.addConstr(inventory[port, t] <= port_dict[port]['storage_capacity'][t-1], name=f'Inventory_upper_bound_{port}_{t}')

    # 3. Quantity loaded/discharged within allowed range
    for vessel in vessel_ids:
        for port in port_ids:
            for t in time_periods:
                min_quantity = port_dict[port]['F_min'] if port_dict[port]['type'] == 'supply' else -port_dict[port]['F_max']
                max_quantity = port_dict[port]['F_max'] if port_dict[port]['type'] == 'supply' else -port_dict[port]['F_min']
                model.addConstr(quantity[vessel, port, t] >= min_quantity * visit[vessel, port, t], name=f'Quantity_lower_bound_{vessel}_{port}_{t}')
                model.addConstr(quantity[vessel, port, t] <= max_quantity * visit[vessel, port, t], name=f'Quantity_upper_bound_{vessel}_{port}_{t}')

    # 4. Cargo capacity constraints for vessels
    for vessel in vessel_ids:
        vessel_capacity = next(v['capacity'] for v in vessels if v['id'] == vessel)
        for t in time_periods:
            model.addConstr(cargo[vessel, t] >= 0, name=f'Cargo_lower_bound_{vessel}_{t}')
            model.addConstr(cargo[vessel, t] <= vessel_capacity, name=f'Cargo_upper_bound_{vessel}_{t}')

    # 5. Update cargo on board vessels
    for vessel in vessel_ids:
        initial_load = next(v['initial_load'] for v in vessels if v['id'] == vessel)
        model.addConstr(cargo[vessel, 0] == initial_load, name=f'Initial_cargo_{vessel}')
        for t in time_periods[1:]:
            model.addConstr(cargo[vessel, t] == cargo[vessel, t-1] + quicksum(quantity[vessel, port, t-1] for port in port_ids), name=f'Cargo_balance_{vessel}_{t}')

    # 6. Draft limit constraints
    for vessel in vessel_ids:
        for port in port_ids:
            for t in time_periods:
                draft_limit = draft_limits.get(f'{vessel}->{port}', float('inf'))
                model.addConstr(cargo[vessel, t] <= draft_limit, name=f'Draft_limit_{vessel}_{port}_{t}')

    # Optimize the model
    model.optimize()

    # Extract solution
    objective_value = model.ObjVal if model.status == GRB.OPTIMAL else float('inf')
    vessel_routes = {}
    port_inventories = {}
    vessel_operations = {}

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        for vessel in vessel_ids:
            vessel_routes[vessel] = []
            vessel_operations[vessel] = []
            for port in port_ids:
                for t in time_periods:
                    if visit[vessel, port, t].x > 0.5:
                        # Record operation
                        vessel_operations[vessel].append({
                            'port': port,
                            'time': t,
                            'quantity': quantity[vessel, port, t].x,
                            'type': 'load' if port_dict[port]['type'] == 'supply' else 'discharge'
                        })
                        # Construct route
                        for next_port in port_ids:
                            if travel_times.get(f'{port}->{next_port}', None) is not None:
                                travel_time = travel_times[f'{port}->{next_port}']
                                if t + travel_time <= planning_horizon and visit[vessel, next_port, t + travel_time].x > 0.5:
                                    vessel_routes[vessel].append({
                                        'tail': f'({port}, {t})',
                                        'head': f'({next_port}, {t + travel_time})',
                                        'type': 'travel',
                                        'cost': travel_time * next(v['transport_cost_per_day'] for v in vessels if v['id'] == vessel),
                                        'product_flow': cargo[vessel, t].x
                                    })

        for port in port_ids:
            port_inventories[port] = {t: inventory[port, t].x for t in time_periods[1:]}

        # Compute cost breakdown
        transport_cost = sum(vessels[i]['transport_cost_per_day'] * travel_times.get(f"{port1}->{port2}", 0) * visit[vessel_ids[i], port2, t + travel_times.get(f"{port1}->{port2}", 0)].x 
                             for i in range(num_vessels) for port1 in port_ids for port2 in port_ids for t in time_periods if t + travel_times.get(f"{port1}->{port2}", 0) <= planning_horizon)
        demurrage_cost = sum(vessels[i]['demurrage_cost_per_day'] * visit[vessel_ids[i], port, t].x 
                             for i in range(num_vessels) for port in port_ids for t in time_periods)
        procurement_cost = sum(port_dict[port]['unit_procurement_cost'] * quantity[vessel_ids[i], port, t].x 
                               for i in range(num_vessels) for port in port_ids if port_dict[port]['type'] == 'supply' for t in time_periods)
        supply_revenue = sum(port_dict[port]['unit_supply_revenue'] * -quantity[vessel_ids[i], port, t].x 
                             for i in range(num_vessels) for port in port_ids if port_dict[port]['type'] == 'demand' for t in time_periods)

        cost_breakdown = {
            'transport_cost': transport_cost,
            'demurrage_cost': demurrage_cost,
            'procurement_cost': procurement_cost,
            'supply_revenue': supply_revenue,
            'total_cost': transport_cost + demurrage_cost + procurement_cost - supply_revenue
        }

        solution = {
            'objective_value': objective_value,
            'vessel_routes': vessel_routes,
            'port_inventories': port_inventories,
            'vessel_operations': vessel_operations,
            'cost_breakdown': cost_breakdown
        }

        return solution

    return None

def main():
    parser = argparse.ArgumentParser(description='Solve the shipping scheduling problem.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file.')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file.')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds.')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions (optional).')
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)
    solution = solve(instance, args.time_limit)

    if solution is not None:
        with open(args.solution_path, 'w') as file:
            json.dump(solution, file, indent=4)
        if logger:
            logger.log_solution(solution['objective_value'], solution)

if __name__ == "__main__":
    main()