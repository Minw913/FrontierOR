import json
import argparse
import gurobipy as gp
from gurobipy import GRB
import os
from solution_logger import SolutionLogger

def main(instance_path, solution_path, time_limit, log_path=None):
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    with open(instance_path, 'r') as f:
        instance_data = json.load(f)
    
    num_customers = instance_data['num_customers']
    num_nodes = instance_data['num_nodes']
    nodes = instance_data['nodes']
    truck_travel_times = instance_data['truck_travel_times']
    drone_travel_times = instance_data['drone_travel_times']
    
    if num_nodes != num_customers + 2:
        print(f"Warning: num_nodes ({num_nodes}) is not equal to num_customers+2 ({num_customers+2}). Proceeding with the first node as depot_origin and the last as depot_destination.")
    
    V = list(range(num_nodes))
    depot_origin_idx = 0
    depot_destination_idx = num_nodes - 1
    customer_indices = list(range(1, num_nodes - 1))
    
    allowed_arcs = []
    for i in V:
        for j in V:
            if i == j:
                continue
            if i == depot_origin_idx and j == depot_destination_idx:
                continue
            if i == depot_destination_idx:
                continue
            if j == depot_origin_idx:
                continue
            allowed_arcs.append((i, j))
    
    model = gp.Model("TSP_Drone")
    
    x = {}
    y = {}
    for arc in allowed_arcs:
        i, j = arc
        x[arc] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
        y[arc] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")
    
    a = model.addVars(V, vtype=GRB.CONTINUOUS, name="a")
    
    truck_only = model.addVars(customer_indices, vtype=GRB.BINARY, name="truck_only")
    drone_only = model.addVars(customer_indices, vtype=GRB.BINARY, name="drone_only")
    combined = model.addVars(customer_indices, vtype=GRB.BINARY, name="combined")
    
    u = model.addVars(V, vtype=GRB.CONTINUOUS, name="u_truck")
    v = model.addVars(V, vtype=GRB.CONTINUOUS, name="v_drone")
    
    for i in customer_indices:
        model.addConstr(truck_only[i] + drone_only[i] + combined[i] == 1, f"type_{i}")
    
    model.addConstr(gp.quicksum(x.get((depot_origin_idx, j), 0) for j in customer_indices) == 1, "truck_start")
    model.addConstr(gp.quicksum(x.get((i, depot_destination_idx), 0) for i in customer_indices) == 1, "truck_end")
    
    for i in customer_indices:
        out_arcs = [j for j in V if (i, j) in x]
        model.addConstr(gp.quicksum(x.get((i, j), 0) for j in out_arcs) == truck_only[i] + combined[i], f"truck_out_{i}")
        in_arcs = [j for j in V if (j, i) in x]
        model.addConstr(gp.quicksum(x.get((j, i), 0) for j in in_arcs) == truck_only[i] + combined[i], f"truck_in_{i}")
    
    model.addConstr(gp.quicksum(y.get((depot_origin_idx, j), 0) for j in customer_indices) == 1, "drone_start")
    model.addConstr(gp.quicksum(y.get((i, depot_destination_idx), 0) for i in customer_indices) == 1, "drone_end")
    
    for i in customer_indices:
        out_arcs = [j for j in V if (i, j) in y]
        model.addConstr(gp.quicksum(y.get((i, j), 0) for j in out_arcs) == drone_only[i] + combined[i], f"drone_out_{i}")
        in_arcs = [j for j in V if (j, i) in y]
        model.addConstr(gp.quicksum(y.get((j, i), 0) for j in in_arcs) == drone_only[i] + combined[i], f"drone_in_{i}")
    
    model.addConstr(a[depot_origin_idx] == 0, "a0")
    
    M = 1000000
    for (i, j) in allowed_arcs:
        if (i, j) in x:
            model.addConstr(a[j] >= a[i] + truck_travel_times[i][j] - M * (1 - x[i, j]), f"truck_time_{i}_{j}")
        if (i, j) in y:
            model.addConstr(a[j] >= a[i] + drone_travel_times[i][j] - M * (1 - y[i, j]), f"drone_time_{i}_{j}")
    
    for i in customer_indices:
        arc1 = (depot_origin_idx, i)
        arc2 = (i, depot_destination_idx)
        if arc1 in y and arc2 in y:
            model.addConstr(y[arc1] + y[arc2] <= 1, f"drone_no_single_{i}")
    
    for i in customer_indices:
        for j in customer_indices:
            if i != j:
                arc1 = (i, j)
                arc2 = (j, i)
                expr = gp.LinExpr()
                if arc1 in y:
                    expr += y[arc1]
                if arc2 in y:
                    expr += y[arc2]
                if expr.size() > 0:
                    model.addConstr(expr <= combined[i] + combined[j], f"drone_pair_{i}_{j}")
    
    model.addConstr(u[depot_origin_idx] == 0, "u0")
    for (i, j) in allowed_arcs:
        if (i, j) in x:
            model.addConstr(u[j] >= u[i] + 1 - M * (1 - x[i, j]), f"truck_mtz_{i}_{j}")
    
    model.addConstr(v[depot_origin_idx] == 0, "v0")
    for (i, j) in allowed_arcs:
        if (i, j) in y:
            model.addConstr(v[j] >= v[i] + 1 - M * (1 - y[i, j]), f"drone_mtz_{i}_{j}")
    
    for i in V:
        u[i].LB = 0
        u[i].UB = num_nodes
        v[i].LB = 0
        v[i].UB = num_nodes
    
    model.setObjective(a[depot_destination_idx], GRB.MINIMIZE)
    
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 1)
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)
    
    model.optimize(callback if logger else None)
    
    if model.SolCount > 0:
        truck_route_list = [depot_origin_idx]
        current = depot_origin_idx
        while current != depot_destination_idx:
            next_found = False
            for j in V:
                arc = (current, j)
                if arc in x and x[arc].X > 0.5:
                    truck_route_list.append(j)
                    current = j
                    next_found = True
                    break
            if not next_found:
                break
        
        drone_route_list = [depot_origin_idx]
        current = depot_origin_idx
        while current != depot_destination_idx:
            next_found = False
            for j in V:
                arc = (current, j)
                if arc in y and y[arc].X > 0.5:
                    drone_route_list.append(j)
                    current = j
                    next_found = True
                    break
            if not next_found:
                break
        
        customer_types_dict = {}
        for i in customer_indices:
            if truck_only[i].X > 0.5:
                customer_types_dict[str(nodes[i]['id'])] = 'truck'
            elif drone_only[i].X > 0.5:
                customer_types_dict[str(nodes[i]['id'])] = 'drone'
            elif combined[i].X > 0.5:
                customer_types_dict[str(nodes[i]['id'])] = 'combined'
        
        arrival_times_dict = {}
        for i in V:
            arrival_times_dict[str(nodes[i]['id'])] = float(a[i].X)
        
        solution = {
            "objective_value": float(a[depot_destination_idx].X),
            "truck_route": [nodes[i]['id'] for i in truck_route_list],
            "drone_route": [nodes[i]['id'] for i in drone_route_list],
            "customer_types": customer_types_dict,
            "arrival_times": arrival_times_dict
        }
        
        with open(solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        print("No solution found.")
        solution = {
            "objective_value": None,
            "truck_route": [],
            "drone_route": [],
            "customer_types": {},
            "arrival_times": {}
        }
        with open(solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Solve the truck-drone routing problem.")
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file.')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to the output solution JSON file.')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds.')
    parser.add_argument('--log_path', type=str, default=None, help='Path to the log file for intermediate solutions.')
    args = parser.parse_args()
    
    main(args.instance_path, args.solution_path, args.time_limit, args.log_path)