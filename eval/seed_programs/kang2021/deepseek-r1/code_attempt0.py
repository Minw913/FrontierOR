import json
import argparse
import gurobipy as gp
from gurobipy import GRB

def main(instance_file, solution_file, time_limit, logger):
    with open(instance_file) as f:
        data = json.load(f)
    
    parameters = data["parameters"]
    num_nodes = parameters["num_nodes"]
    depot_index = parameters["depot_index"]
    customer_indices = parameters["customer_indices"]
    num_drones = parameters["num_drones"]
    truck_travel_times = parameters["truck_travel_times"]
    truck_service_time = parameters["truck_service_time"]
    drone_battery_capacities = parameters["drone_battery_capacities"]
    drone_battery_consumption = parameters["drone_battery_consumption"]
    drone_delivery_time = parameters["drone_delivery_time"]
    
    sink = num_nodes
    V = list(range(num_nodes)) + [sink]
    N = list(range(num_nodes))
    C = customer_indices
    D = list(range(num_drones))
    
    T = [[0] * (num_nodes + 1) for _ in range(num_nodes + 1)]
    for i in range(num_nodes):
        for j in range(num_nodes):
            T[i][j] = truck_travel_times[i][j]
        T[i][sink] = max(0, truck_travel_times[i][depot_index] - truck_service_time)
    for j in range(num_nodes + 1):
        T[sink][j] = 0
    
    model = gp.Model("Truck_Drone")
    
    x = {}
    for i in V:
        for j in V:
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    
    z = {}
    for d in D:
        for i in N:
            for k in C:
                z[d, i, k] = model.addVar(vtype=GRB.BINARY, name=f"z_{d}_{i}_{k}")
    
    t = {}
    for k in C:
        t[k] = model.addVar(vtype=GRB.BINARY, name=f"t_{k}")
    
    w = {}
    for i in N:
        w[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"w_{i}")
    
    for k in C:
        model.addConstr(t[k] + gp.quicksum(z[d, i, k] for d in D for i in N) == 1, f"serve_{k}")
    
    model.addConstr(gp.quicksum(x[depot_index, j] for j in V) == 1, "start")
    model.addConstr(gp.quicksum(x[i, sink] for i in V) == 1, "end")
    model.addConstr(gp.quicksum(x[sink, j] for j in V) == 0, "sink_out")
    model.addConstr(gp.quicksum(x[i, depot_index] for i in V) == 0, "depot_in")
    
    for u in N:
        if u == depot_index:
            model.addConstr(gp.quicksum(x[u, j] for j in V) == 1, f"out_{u}")
            model.addConstr(gp.quicksum(x[i, u] for i in V) == 0, f"in_{u}")
        else:
            model.addConstr(gp.quicksum(x[i, u] for i in V) == t[u], f"in_{u}")
            model.addConstr(gp.quicksum(x[u, j] for j in V) == t[u], f"out_{u}")
    
    for d in D:
        for i in N:
            if i != depot_index:
                for k in C:
                    model.addConstr(z[d, i, k] <= t[i], f"drone_node_{d}_{i}_{k}")
    
    for d in D:
        model.addConstr(
            gp.quicksum(
                drone_battery_consumption[d][i][k] * z[d, i, k]
                for i in N for k in C
            ) <= drone_battery_capacities[d],
            f"battery_{d}"
        )
    
    for d in D:
        for i in N:
            model.addConstr(
                w[i] >= gp.quicksum(
                    drone_delivery_time[d][i][k] * z[d, i, k] for k in C
                ),
                f"wait_{d}_{i}"
            )
    
    truck_travel = gp.quicksum(T[i][j] * x[i, j] for i in V for j in V)
    waiting_time_sum = gp.quicksum(w[i] for i in N)
    model.setObjective(truck_travel + waiting_time_sum, GRB.MINIMIZE)
    
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    
    def callback_fn(model_param, where):
        if where == GRB.Callback.MIPSOL:
            obj = model_param.cbGet(GRB.Callback.MIPSOL_OBJ)
            x_vals = {}
            for i, j in x:
                x_vals[i, j] = model_param.cbGetSolution(x[i, j])
            z_vals = {}
            for d, i, k in z:
                z_vals[d, i, k] = model_param.cbGetSolution(z[d, i, k])
            t_vals = {}
            for k in t:
                t_vals[k] = model_param.cbGetSolution(t[k])
            w_vals = {}
            for i in w:
                w_vals[i] = model_param.cbGetSolution(w[i])
            
            truck_route = []
            for i, j in x:
                if x_vals[i, j] > 0.5:
                    truck_route.append([int(i), int(j)])
            
            drone_assignments = []
            for d, i, k in z:
                if z_vals[d, i, k] > 0.5:
                    drone_assignments.append({
                        "drone": int(d),
                        "dispatch_node": int(i),
                        "delivery_node": int(k)
                    })
            
            visited_nodes = [depot_index]
            for k in C:
                if t_vals[k] > 0.5:
                    visited_nodes.append(k)
            waiting_times = {}
            for i in visited_nodes:
                waiting_times[str(i)] = float(w_vals[i])
            
            solution_dict = {
                "truck_route": truck_route,
                "drone_assignments": drone_assignments,
                "waiting_times": waiting_times
            }
            if logger:
                logger.log_solution(obj, solution_dict)
    
    model.optimize(callback_fn if logger else None)
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        truck_route = []
        for i, j in x:
            if x[i, j].X > 0.5:
                truck_route.append([int(i), int(j)])
        
        drone_assignments = []
        for d, i, k in z:
            if z[d, i, k].X > 0.5:
                drone_assignments.append({
                    "drone": int(d),
                    "dispatch_node": int(i),
                    "delivery_node": int(k)
                })
        
        visited_nodes = [depot_index]
        for k in C:
            if t[k].X > 0.5:
                visited_nodes.append(k)
        waiting_times = {}
        for i in visited_nodes:
            waiting_times[str(i)] = float(w[i].X)
        
        solution_dict = {
            "objective_value": float(model.ObjVal),
            "truck_route": truck_route,
            "drone_assignments": drone_assignments,
            "waiting_times": waiting_times
        }
        
        with open(solution_file, 'w') as f:
            json.dump(solution_dict, f)
    else:
        raise RuntimeError("No solution found")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=str, required=True)
    parser.add_argument('--solution', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    args = parser.parse_args()
    main(args.instance, args.solution, args.time_limit, None)