import argparse
import json
import math
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_shipment_planning(instance_path, solution_path, time_limit, log_path):
    # Read instance data
    with open(instance_path, 'r') as f:
        data = json.load(f)

    # Extract data
    T = data['planning_horizon']
    num_vessels = data['num_vessels']
    num_supply_ports = data['num_supply_ports']
    num_demand_ports = data['num_demand_ports']

    vessels = {v['id']: v for v in data['vessels']}
    vessel_list = list(vessels.keys())

    supply_ports = {p['port_id']: p for p in data['supply_ports']}
    demand_ports = {p['port_id']: p for p in data['demand_ports']}
    all_ports = {**supply_ports, **demand_ports}
    port_list = list(all_ports.keys())
    supply_port_list = list(supply_ports.keys())
    demand_port_list = list(demand_ports.keys())

    travel_times = data['travel_times']  # format: "{origin_port}->{destination_port}"
    draft_limits = data['draft_limits']  # format: "{vessel_id}->{port_id}"

    # Preprocess travel times into dict
    travel_time_dict = {}
    for key, value in travel_times.items():
        origin, dest = key.split('->')
        travel_time_dict[(origin, dest)] = value

    # Preprocess draft limits
    draft_limit_dict = {}
    for vessel_key, ports in draft_limits.items():
        for port_key, limit in ports.items():
            draft_limit_dict[(vessel_key, port_key)] = limit

    # Initialize logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    # Create model
    model = gp.Model("ship_planning")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)

    # Variables
    # x[v, i, j, t] = 1 if vessel v travels from port i to port j starting at time t
    x = {}
    for v_id in vessel_list:
        v_data = vessels[v_id]
        for i in port_list:
            for j in port_list:
                if i != j:
                    for t in range(T):
                        if (i, j) in travel_time_dict and t + travel_time_dict[(i, j)] <= T and v_data['availability_start'] <= t <= v_data['availability_end']:
                            x[v_id, i, j, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{v_id}_{i}_{j}_{t}")

    # y[v, i, t] = 1 if vessel v stays at port i during time period t
    y = {}
    for v_id in vessel_list:
        v_data = vessels[v_id]
        for i in port_list:
            for t in range(T):
                if v_data['availability_start'] <= t <= v_data['availability_end']:
                    y[v_id, i, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{v_id}_{i}_{t}")

    # q_load[v, i, t] = quantity loaded by vessel v at port i during time period t
    q_load = {}
    for v_id in vessel_list:
        v_data = vessels[v_id]
        for i in supply_port_list:
            for t in range(T):
                if v_data['availability_start'] <= t <= v_data['availability_end']:
                    q_load[v_id, i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=min(supply_ports[i]['F_max'], v_data['capacity']), 
                                                      name=f"q_load_{v_id}_{i}_{t}")

    # q_discharge[v, i, t] = quantity discharged by vessel v at port i during time period t
    q_discharge = {}
    for v_id in vessel_list:
        v_data = vessels[v_id]
        for i in demand_port_list:
            for t in range(T):
                if v_data['availability_start'] <= t <= v_data['availability_end']:
                    q_discharge[v_id, i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=min(demand_ports[i]['F_max'], v_data['capacity']), 
                                                           name=f"q_discharge_{v_id}_{i}_{t}")

    # z_load[v, i, t] = 1 if vessel v loads at port i during time period t
    z_load = {}
    for v_id in vessel_list:
        v_data = vessels[v_id]
        for i in supply_port_list:
            for t in range(T):
                if v_data['availability_start'] <= t <= v_data['availability_end']:
                    z_load[v_id, i, t] = model.addVar(vtype=GRB.BINARY, name=f"z_load_{v_id}_{i}_{t}")

    # z_discharge[v, i, t] = 1 if vessel v discharges at port i during time period t
    z_discharge = {}
    for v_id in vessel_list:
        v_data = vessels[v_id]
        for i in demand_port_list:
            for t in range(T):
                if v_data['availability_start'] <= t <= v_data['availability_end']:
                    z_discharge[v_id, i, t] = model.addVar(vtype=GRB.BINARY, name=f"z_discharge_{v_id}_{i}_{t}")

    # c[v, i, t] = cargo on board vessel v when arriving at port i at time t
    c = {}
    for v_id in vessel_list:
        v_data = vessels[v_id]
        for i in port_list:
            for t in range(T):
                if v_data['availability_start'] <= t <= v_data['availability_end']:
                    c[v_id, i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=v_data['capacity'],
                                                 name=f"c_{v_id}_{i}_{t}")

    # Port inventory variables
    port_inventory = {}
    for port_id in port_list:
        for t in range(1, T+1):  # Inventory after period t
            if all_ports[port_id]['type'] == 'supply':
                port_inventory[port_id, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,
                                                          ub=all_ports[port_id]['storage_capacity'][t-1],
                                                          name=f"inv_{port_id}_{t}")
            else:  # demand port
                port_inventory[port_id, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,
                                                          ub=all_ports[port_id]['storage_capacity'][t-1],
                                                          name=f"inv_{port_id}_{t}")

    # Constraints
    # Flow conservation for each vessel at each time period
    for v_id in vessel_list:
        v_data = vessels[v_id]
        for t in range(v_data['availability_start'], min(v_data['availability_end']+1, T)):
            # Create variables for flow balance at time t
            incoming_flow = gp.LinExpr()
            outgoing_flow = gp.LinExpr()
            
            # Incoming flows: arrivals from other ports
            for i in port_list:
                for j in port_list:
                    if i != j and (i, j) in travel_time_dict:
                        # Check if vessel could arrive at j at time t from i
                        start_t = t - travel_time_dict[(i, j)]
                        if 0 <= start_t < T and start_t >= v_data['availability_start']:
                            if (v_id, i, j, start_t) in x:
                                incoming_flow.addTerms(1, x[v_id, i, j, start_t])
            
            # Add staying at port in previous period (if t > 0)
            if t > 0:
                for j in port_list:
                    if (v_id, j, t-1) in y:
                        incoming_flow.addTerms(1, y[v_id, j, t-1])
            
            # Outgoing flows: departures to other ports
            for i in port_list:
                for j in port_list:
                    if i != j and (i, j) in travel_time_dict:
                        if (v_id, i, j, t) in x:
                            outgoing_flow.addTerms(1, x[v_id, i, j, t])
                
                # Staying at same port for next period
                if (v_id, i, t) in y:
                    outgoing_flow.addTerms(1, y[v_id, i, t])
            
            model.addConstr(incoming_flow == outgoing_flow, name=f"flow_conservation_{v_id}_{t}")

    # Loading and discharging bounds
    for v_id in vessel_list:
        for i in supply_port_list:
            for t in range(T):
                if (v_id, i, t) in q_load and (v_id, i, t) in z_load:
                    F_min = supply_ports[i]['F_min']
                    F_max = supply_ports[i]['F_max']
                    model.addConstr(q_load[v_id, i, t] >= F_min * z_load[v_id, i, t], 
                                    name=f"load_min_{v_id}_{i}_{t}")
                    model.addConstr(q_load[v_id, i, t] <= F_max * z_load[v_id, i, t], 
                                    name=f"load_max_{v_id}_{i}_{t}")

    for v_id in vessel_list:
        for i in demand_port_list:
            for t in range(T):
                if (v_id, i, t) in q_discharge and (v_id, i, t) in z_discharge:
                    F_min = demand_ports[i]['F_min']
                    F_max = demand_ports[i]['F_max']
                    model.addConstr(q_discharge[v_id, i, t] >= F_min * z_discharge[v_id, i, t], 
                                    name=f"discharge_min_{v_id}_{i}_{t}")
                    model.addConstr(q_discharge[v_id, i, t] <= F_max * z_discharge[v_id, i, t], 
                                    name=f"discharge_max_{v_id}_{i}_{t}")

    # Only load if at port and available
    for v_id in vessel_list:
        for i in supply_port_list:
            for t in range(T):
                if (v_id, i, t) in z_load and (v_id, i, t) in y:
                    model.addConstr(z_load[v_id, i, t] <= y[v_id, i, t], 
                                    name=f"load_only_at_port_{v_id}_{i}_{t}")

    # Only discharge if at port and available
    for v_id in vessel_list:
        for i in demand_port_list:
            for t in range(T):
                if (v_id, i, t) in z_discharge and (v_id, i, t) in y:
                    model.addConstr(z_discharge[v_id, i, t] <= y[v_id, i, t], 
                                    name=f"discharge_only_at_port_{v_id}_{i}_{t}")

    # Draft limits
    for v_id in vessel_list:
        for i in port_list:
            for t in range(T):
                if (v_id, i, t) in c and (v_id, i, t) in y:
                    draft_limit = draft_limit_dict.get((v_id, i), vessels[v_id]['capacity'])
                    model.addConstr(c[v_id, i, t] <= draft_limit * y[v_id, i, t], 
                                    name=f"draft_limit_{v_id}_{i}_{t}")

    # Cargo flow constraints - simplified
    for v_id in vessel_list:
        v_data = vessels[v_id]
        for t in range(v_data['availability_start'], min(v_data['availability_end'], T-1)+1):
            for j in port_list:
                if (v_id, j, t+1) in c and (v_id, j, t+1) in y:
                    # Calculate cargo based on previous state
                    prev_cargo_expr = gp.LinExpr()
                    
                    # Check if vessel was at the same port in the previous period
                    if (v_id, j, t) in y and (v_id, j, t) in c:
                        prev_cargo_expr.addTerms(1, c[v_id, j, t])
                        
                    # Check if vessel arrived at j at time t+1 from another port
                    for i in port_list:
                        if i != j and (i, j) in travel_time_dict:
                            start_time = (t+1) - travel_time_dict[(i, j)]
                            if start_time >= 0 and (v_id, i, j, start_time) in x and (v_id, i, start_time) in c:
                                prev_cargo_expr.addTerms(1, c[v_id, i, start_time])

                    # Add loading and subtract discharging at current time t
                    if j in supply_port_list and (v_id, j, t) in q_load:
                        prev_cargo_expr.addTerms(1, q_load[v_id, j, t])
                    elif j in demand_port_list and (v_id, j, t) in q_discharge:
                        prev_cargo_expr.addTerms(-1, q_discharge[v_id, j, t])

                    model.addConstr(c[v_id, j, t+1] == prev_cargo_expr, name=f"cargo_flow_{v_id}_{j}_{t}")

    # Initial cargo constraint - simplified
    for v_id in vessel_list:
        v_data = vessels[v_id]
        if v_data['availability_start'] < T:
            # Find where the vessel starts - this is a simplification
            initial_port = None
            for i in port_list:
                if (v_id, i, v_data['availability_start']) in y:
                    initial_port = i
                    break
            
            if initial_port and (v_id, initial_port, v_data['availability_start']) in c:
                model.addConstr(c[v_id, initial_port, v_data['availability_start']] == v_data['initial_load'], 
                                name=f"initial_cargo_{v_id}")

    # Port inventory evolution
    for port_id in supply_port_list:
        for t in range(1, T+1):
            # Inventory at end of period t = inventory at end of period t-1 + production - total loaded
            prev_inv = all_ports[port_id]['initial_inventory'] if t == 1 else port_inventory[port_id, t-1]
            prod = all_ports[port_id]['production_rate'][t-1]
            
            total_loaded = gp.LinExpr()
            for v_id in vessel_list:
                if (v_id, port_id, t-1) in q_load:
                    total_loaded.addTerms(1, q_load[v_id, port_id, t-1])
            
            model.addConstr(port_inventory[port_id, t] == prev_inv + prod - total_loaded,
                            name=f"inventory_supply_{port_id}_{t}")

    for port_id in demand_port_list:
        for t in range(1, T+1):
            # Inventory at end of period t = inventory at end of period t-1 - consumption + total discharged
            prev_inv = all_ports[port_id]['initial_inventory'] if t == 1 else port_inventory[port_id, t-1]
            cons = all_ports[port_id]['consumption_rate'][t-1]
            
            total_discharged = gp.LinExpr()
            for v_id in vessel_list:
                if (v_id, port_id, t-1) in q_discharge:
                    total_discharged.addTerms(1, q_discharge[v_id, port_id, t-1])
            
            model.addConstr(port_inventory[port_id, t] == prev_inv - cons + total_discharged,
                            name=f"inventory_demand_{port_id}_{t}")

    # Objective function
    obj_expr = gp.LinExpr()

    # Transportation costs
    for (v_id, i, j, t), var in x.items():
        transport_cost = vessels[v_id]['transport_cost_per_day'] * travel_time_dict[(i, j)]
        obj_expr.addTerms(transport_cost, var)

    # Demurrage costs
    for (v_id, i, t), var in y.items():
        demurrage_cost = vessels[v_id]['demurrage_cost_per_day']
        obj_expr.addTerms(demurrage_cost, var)

    # Procurement costs
    for (v_id, i, t), var in q_load.items():
        unit_cost = supply_ports[i]['unit_procurement_cost']
        obj_expr.addTerms(unit_cost, var)

    # Supply revenue (negative because we want to maximize profit -> minimize negative profit)
    for (v_id, i, t), var in q_discharge.items():
        unit_revenue = demand_ports[i]['unit_supply_revenue']
        obj_expr.addTerms(-unit_revenue, var)

    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Optimize
    model.optimize()

    # Prepare solution
    solution = {
        "objective_value": model.ObjVal if model.SolCount > 0 else float('inf'),
        "vessel_routes": {},
        "port_inventories": {},
        "vessel_operations": {},
        "cost_breakdown": {
            "transport_cost": 0,
            "demurrage_cost": 0,
            "procurement_cost": 0,
            "supply_revenue": 0,
            "total_cost": model.ObjVal if model.SolCount > 0 else float('inf')
        }
    }

    if model.SolCount > 0:
        # Build routes
        for v_id in vessel_list:
            solution["vessel_routes"][v_id] = []
            solution["vessel_operations"][v_id] = []

            # Track vessel movements
            for t in range(T):
                for i in port_list:
                    if (v_id, i, t) in y and y[v_id, i, t].X > 0.9:
                        # Vessel stays at port - check for loading/discharging
                        if i in supply_port_list and (v_id, i, t) in z_load and z_load[v_id, i, t].X > 0.9:
                            quantity = q_load[v_id, i, t].X
                            solution["vessel_operations"][v_id].append({
                                "port": i,
                                "time": t,
                                "quantity": quantity,
                                "type": "load"
                            })
                            solution["cost_breakdown"]["procurement_cost"] += \
                                supply_ports[i]['unit_procurement_cost'] * quantity
                        elif i in demand_port_list and (v_id, i, t) in z_discharge and z_discharge[v_id, i, t].X > 0.9:
                            quantity = q_discharge[v_id, i, t].X
                            solution["vessel_operations"][v_id].append({
                                "port": i,
                                "time": t,
                                "quantity": -quantity,
                                "type": "discharge"
                            })
                            solution["cost_breakdown"]["supply_revenue"] += \
                                demand_ports[i]['unit_supply_revenue'] * quantity

                    # Check for travel
                    for j in port_list:
                        if i != j and (i, j) in travel_time_dict and (v_id, i, j, t) in x and x[v_id, i, j, t].X > 0.9:
                            travel_time = travel_time_dict[(i, j)]
                            solution["vessel_routes"][v_id].append({
                                "tail": f"({i},{t})",
                                "head": f"({j},{t + travel_time})",
                                "type": "travel",
                                "cost": vessels[v_id]['transport_cost_per_day'] * travel_time,
                                "product_flow": c[v_id, i, t].X if (v_id, i, t) in c else 0
                            })
                            solution["cost_breakdown"]["transport_cost"] += \
                                vessels[v_id]['transport_cost_per_day'] * travel_time

            # Add demurrage costs
            for t in range(T):
                for i in port_list:
                    if (v_id, i, t) in y and y[v_id, i, t].X > 0.9:
                        solution["vessel_routes"][v_id].append({
                            "tail": f"({i},{t})",
                            "head": f"({i},{t+1})",
                            "type": "demurrage",
                            "cost": vessels[v_id]['demurrage_cost_per_day'],
                            "product_flow": c[v_id, i, t].X if (v_id, i, t) in c else 0
                        })
                        solution["cost_breakdown"]["demurrage_cost"] += vessels[v_id]['demurrage_cost_per_day']

        # Build port inventories
        for port_id in port_list:
            solution["port_inventories"][port_id] = {}
            for t in range(1, T+1):
                solution["port_inventories"][port_id][str(t)] = port_inventory[port_id, t].X

    # Write solution
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    # Log solution if logger exists
    if logger and model.SolCount > 0:
        logger.log_solution(model.ObjVal, solution)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Solve shipment planning problem.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to save the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log file for intermediate solutions')

    args = parser.parse_args()

    solve_shipment_planning(args.instance_path, args.solution_path, args.time_limit, args.log_path)