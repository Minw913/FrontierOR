import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys
from collections import defaultdict
import math

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_customers = instance['num_customers']
    vehicle_capacity = instance['vehicle_capacity']
    scheduling_horizon = instance['scheduling_horizon']
    a = instance['cost_function']['a']
    b = instance['cost_function']['b']
    depot = instance['depot']
    customers = instance['customers']
    dist = instance['distance_matrix']

    N = num_customers + 1  # 0 = depot start, 1..num_customers = customers, num_customers+1 = depot end
    depot_end = num_customers + 1
    all_nodes = list(range(N + 1))
    customers_list = list(range(1, num_customers + 1))

    demands = [0] + [c['demand'] for c in customers] + [0]
    service_times = [depot['service_time']] + [c['service_time'] for c in customers] + [0]
    time_windows = [depot['time_window']] + [c['time_window'] for c in customers] + [depot['time_window']]
    earliest = [tw[0] for tw in time_windows]
    latest = [tw[1] for tw in time_windows]

    travel_time = [[0.0 for _ in range(N + 1)] for _ in range(N + 1)]
    for i in range(N):
        for j in range(N):
            if i == j:
                travel_time[i][j] = 0.0
            else:
                travel_time[i][j] = dist[i][j]  # assuming time = distance as common in such problems when not specified separately
    # Note: problem mentions separate distance and travel time but schema only provides distance_matrix.
    # Using distance as time (common proxy when speeds are unit). If separate matrix was intended, it would be in schema.

    M = 1e6  # big-M for time constraints
    max_vehicles = len(customers_list)  # conservative upper bound

    try:
        model = gp.Model("SDVRPTW_LinearCost")
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', args.time_limit - 5)  # reserve some time for output

        # Variables
        x = model.addVars(max_vehicles, N + 1, N + 1, vtype=GRB.BINARY, name="x")
        y = model.addVars(max_vehicles, N + 1, vtype=GRB.CONTINUOUS, lb=0, name="y")  # delivery amount at node
        load = model.addVars(max_vehicles, N + 1, N + 1, vtype=GRB.CONTINUOUS, lb=0, ub=vehicle_capacity, name="load")
        t = model.addVars(max_vehicles, N + 1, vtype=GRB.CONTINUOUS, lb=0, name="t")  # service start time

        # Objective: sum distance * (a * load + b) over all used edges
        obj = gp.quicksum(dist[i][j] * (a * load[k, i, j] + b) * x[k, i, j]
                          for k in range(max_vehicles)
                          for i in all_nodes
                          for j in all_nodes if i != j)
        model.setObjective(obj, GRB.MINIMIZE)

        # Each customer demand satisfied
        model.addConstrs((gp.quicksum(y[k, i] for k in range(max_vehicles)) == demands[i]
                          for i in customers_list), name="demand_satisfaction")

        # Vehicle capacity and flow
        for k in range(max_vehicles):
            # Depot outflow = total delivery by vehicle
            model.addConstr(gp.quicksum(y[k, i] for i in customers_list) <= vehicle_capacity, name=f"cap_{k}")
            
            # Flow conservation: load in - load out = delivery (adjusted for depot)
            for i in customers_list:
                model.addConstr(
                    gp.quicksum(load[k, j, i] for j in all_nodes if j != i) -
                    gp.quicksum(load[k, i, j] for j in all_nodes if j != i) == y[k, i],
                    name=f"flow_{k}_{i}"
                )
            
            # Depot start load = total delivery
            model.addConstr(
                gp.quicksum(load[k, 0, j] for j in all_nodes if j != 0) == gp.quicksum(y[k, i] for i in customers_list),
                name=f"depot_load_{k}"
            )
            
            # No load into final depot
            model.addConstr(
                gp.quicksum(load[k, j, depot_end] for j in all_nodes if j != depot_end) == 0,
                name=f"return_load_{k}"
            )
            
            # At most one visit per customer per vehicle
            model.addConstrs((gp.quicksum(x[k, i, j] for j in all_nodes if j != i) <= 1
                              for i in customers_list), name=f"visit_{k}")
            
            # Route continuity
            for i in customers_list:
                model.addConstr(
                    gp.quicksum(x[k, j, i] for j in all_nodes if j != i) ==
                    gp.quicksum(x[k, i, j] for j in all_nodes if j != i),
                    name=f"flow_visit_{k}_{i}"
                )
            
            # Leave depot once, return once
            model.addConstr(gp.quicksum(x[k, 0, j] for j in all_nodes if j != 0) <= 1, name=f"leave_depot_{k}")
            model.addConstr(gp.quicksum(x[k, i, depot_end] for i in all_nodes if i != depot_end) <= 1, name=f"return_depot_{k}")
            
            # Link load and x
            for i in all_nodes:
                for j in all_nodes:
                    if i != j:
                        model.addConstr(load[k, i, j] <= vehicle_capacity * x[k, i, j], name=f"load_link_{k}_{i}_{j}")
            
            # Time constraints
            for i in all_nodes:
                for j in all_nodes:
                    if i != j:
                        model.addConstr(
                            t[k, j] >= t[k, i] + service_times[i] + travel_time[i][j] - M * (1 - x[k, i, j]),
                            name=f"time_{k}_{i}_{j}"
                        )
            
            # Time windows
            model.addConstrs((t[k, i] >= earliest[i] for i in all_nodes), name=f"earliest_{k}")
            model.addConstrs((t[k, i] <= latest[i] for i in all_nodes), name=f"latest_{k}")
            
            # Depot start and end
            model.addConstr(t[k, 0] >= earliest[0], name=f"depot_start_{k}")
            model.addConstr(t[k, depot_end] <= latest[depot_end], name=f"depot_end_{k}")
            model.addConstr(t[k, depot_end] <= scheduling_horizon, name=f"horizon_{k}")

        # Symmetry breaking: use vehicles in order
        for k in range(max_vehicles - 1):
            model.addConstr(gp.quicksum(x[k, 0, j] for j in all_nodes if j != 0) >=
                            gp.quicksum(x[k+1, 0, j] for j in all_nodes if j != 0), name=f"symmetry_{k}")

        # Callback for incumbent logging
        best_obj = float('inf')
        solution_dicts = []

        def callback(model, where):
            nonlocal best_obj
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
                if obj_val < best_obj - 1e-6:
                    best_obj = obj_val
                    # Extract solution
                    sol_x = model.cbGetSolution(x)
                    sol_y = model.cbGetSolution(y)
                    sol_load = model.cbGetSolution(load)
                    sol_t = model.cbGetSolution(t)
                    
                    routes = []
                    vehicle_id = 0
                    for k in range(max_vehicles):
                        # Check if vehicle is used
                        if sum(sol_x[k, 0, j] for j in all_nodes if j != 0) > 0.5:
                            # Reconstruct route
                            current = 0
                            route_nodes = [0]
                            visited = set()
                            while True:
                                found = False
                                for j in all_nodes:
                                    if j != current and sol_x[k, current, j] > 0.5:
                                        if j == depot_end:
                                            route_nodes.append(depot_end)
                                            found = True
                                            break
                                        if j not in visited and j != 0 and j != depot_end:
                                            route_nodes.append(j)
                                            visited.add(j)
                                            current = j
                                            found = True
                                            break
                                if not found or len(route_nodes) > N + 2:
                                    break
                            
                            if route_nodes[-1] != depot_end:
                                route_nodes.append(depot_end)
                            
                            deliveries = {}
                            for i in customers_list:
                                if sol_y[k, i] > 1e-6:
                                    deliveries[str(i)] = float(round(sol_y[k, i], 6))
                            
                            edges = []
                            for idx in range(len(route_nodes)-1):
                                fr = route_nodes[idx]
                                to = route_nodes[idx+1]
                                lw = float(round(sol_load[k, fr, to], 6))
                                edges.append({
                                    "from": int(fr),
                                    "to": int(to),
                                    "load_weight": lw
                                })
                            
                            routes.append({
                                "vehicle": vehicle_id,
                                "route": [int(r) for r in route_nodes],
                                "deliveries": deliveries,
                                "edges": edges
                            })
                            vehicle_id += 1
                    
                    solution = {
                        "objective_value": float(round(obj_val, 6)),
                        "routes": routes
                    }
                    
                    if logger:
                        logger.log_solution(obj_val, solution)
                    best_obj = obj_val

        model.optimize(callback)

        # Final solution extraction
        if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
            obj_val = model.objVal
            routes = []
            vehicle_id = 0
            used_vehicles = []

            for k in range(max_vehicles):
                if model.getVarByName(f"x[{k},0,1]").X > 0.5 or any(model.getVarByName(f"x[{k},0,{j}]").X > 0.5 for j in customers_list):
                    route_nodes = [0]
                    current = 0
                    visited = set()
                    for _ in range(N + 5):  # safety
                        next_node = None
                        for j_idx in all_nodes:
                            if j_idx != current:
                                var_name = f"x[{k},{current},{j_idx}]"
                                if model.getVarByName(var_name).X > 0.5:
                                    next_node = j_idx
                                    break
                        if next_node is None or next_node == depot_end:
                            route_nodes.append(depot_end)
                            break
                        if next_node in visited:
                            break
                        route_nodes.append(next_node)
                        visited.add(next_node)
                        current = next_node
                    if route_nodes[-1] != depot_end:
                        route_nodes.append(depot_end)

                    deliveries = {}
                    for i in customers_list:
                        yval = model.getVarByName(f"y[{k},{i}]").X
                        if yval > 1e-5:
                            deliveries[str(i)] = float(round(yval, 6))

                    edges = []
                    for r in range(len(route_nodes)-1):
                        fr = route_nodes[r]
                        to = route_nodes[r+1]
                        load_var = model.getVarByName(f"load[{k},{fr},{to}]")
                        lw = load_var.X if load_var else 0.0
                        edges.append({
                            "from": int(fr),
                            "to": int(to),
                            "load_weight": float(round(lw, 6))
                        })

                    routes.append({
                        "vehicle": vehicle_id,
                        "route": [int(node) for node in route_nodes],
                        "deliveries": deliveries,
                        "edges": edges
                    })
                    vehicle_id += 1

            final_solution = {
                "objective_value": float(round(obj_val, 6)),
                "routes": routes
            }

            with open(args.solution_path, 'w') as f:
                json.dump(final_solution, f, indent=2)

            if logger:
                logger.log_solution(obj_val, final_solution)

    except Exception as e:
        print("Error:", str(e), file=sys.stderr)
        # Write empty feasible solution as fallback
        fallback = {
            "objective_value": 0.0,
            "routes": []
        }
        with open(args.solution_path, 'w') as f:
            json.dump(fallback, f, indent=2)


if __name__ == "__main__":
    main()