import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(instance, time_limit, logger=None):
    # Extract data
    num_nodes = instance['num_nodes']
    depot_idx = instance['depot_index']
    customers = instance['customer_indices']
    num_customers = len(customers)
    num_drones = instance['parameters']['num_drones']
    
    travel_times = instance['truck_travel_times']
    battery = instance['drone_battery_consumption']
    del_time = instance['drone_delivery_time']
    drone_cap = instance['parameters']['drone_battery_capacities']
    
    N = num_nodes
    S = N # virtual sink node index
    
    model = gp.Model("TruckDrone")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)
    
    # Variables
    x = {}
    for i in range(N):
        if i == depot_idx:
            for j in customers:
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
            x[i, S] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{S}")
        else:
            for j in customers:
                if i != j:
                    x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
            x[i, S] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{S}")
            
    y = {}
    for i in range(N):
        if i == depot_idx:
            y[i] = model.addVar(vtype=GRB.BINARY, ub=1.0, name=f"y_{i}")
        else:
            y[i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}")
            
    z = {}
    for i in range(N):
        for c in customers:
            for k in range(num_drones):
                z[i, c, k] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{c}_{k}")
                
    W = {}
    for i in range(N):
        W[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"W_{i}")
        
    u = {}
    for c in customers:
        u[c] = model.addVar(vtype=GRB.CONTINUOUS, lb=1, ub=num_customers, name=f"u_{c}")
        
    model.update()
    
    # 1. Depot is visited
    model.addConstr(y[depot_idx] == 1)
    
    # 2. Every customer served exactly once
    for c in customers:
        model.addConstr(y[c] + gp.quicksum(z[i, c, k] for i in range(N) for k in range(num_drones)) == 1)
        
    # 3. Drone dispatch only from visited nodes
    for i in range(N):
        for c in customers:
            for k in range(num_drones):
                model.addConstr(z[i, c, k] <= y[i])
                
    # 4. Drone battery limits
    for k in range(num_drones):
        model.addConstr(gp.quicksum(battery[k][i][c] * z[i, c, k] for i in range(N) for c in customers) <= drone_cap[k])
        
    # 5. Waiting times
    M_W = 0
    for i in range(N):
        for k in range(num_drones):
            s = sum(del_time[k][i][c] for c in customers)
            if s > M_W:
                M_W = s
            model.addConstr(W[i] >= gp.quicksum(del_time[k][i][c] * z[i, c, k] for c in customers))
        model.addConstr(W[i] <= M_W * y[i])
        
    # 6. Routing - Degrees
    model.addConstr(gp.quicksum(x[depot_idx, j] for j in list(customers) + [S]) == 1)
    model.addConstr(gp.quicksum(x[i, S] for i in range(N)) == 1)
    
    for j in customers:
        model.addConstr(gp.quicksum(x[i, j] for i in range(N) if (i, j) in x) == y[j])
        
    for i in customers:
        model.addConstr(gp.quicksum(x[i, j] for j in list(customers) + [S] if (i, j) in x) == y[i])
        
    # 7. MTZ Subtour Elimination
    for i in customers:
        for j in customers:
            if i != j:
                model.addConstr(u[i] - u[j] + num_customers * x[i, j] <= num_customers - 1)
                
    # 8. Strengthening
    for i in customers:
        for j in list(customers) + [S]:
            if (i, j) in x:
                model.addConstr(x[i, j] <= y[i])
                if j != S:
                    model.addConstr(x[i, j] <= y[j])
                    
    for i in customers:
        for j in customers:
            if i != j and (i, j) in x and (j, i) in x:
                model.addConstr(x[i, j] + x[j, i] <= 1)
                
    # Objective
    obj = gp.LinExpr()
    for i in range(N):
        for j in list(customers) + [S]:
            if (i, j) in x:
                if j == S:
                    obj += travel_times[i][depot_idx] * x[i, j]
                else:
                    obj += travel_times[i][j] * x[i, j]
    for i in range(N):
        obj += W[i]
        
    model.setObjective(obj, GRB.MINIMIZE)
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)
                
    model.optimize(callback)
    
    if model.SolCount == 0:
        return None
        
    obj_val = model.ObjVal
    truck_route = []
    for i in range(N):
        for j in list(customers) + [S]:
            if (i, j) in x and x[i, j].X > 0.5:
                truck_route.append([i, j])
                
    drone_assignments = []
    for i in range(N):
        for c in customers:
            for k in range(num_drones):
                if z[i, c, k].X > 0.5:
                    drone_assignments.append({
                        "drone": k,
                        "dispatch_node": i,
                        "delivery_node": c
                    })
                    
    waiting_times = {}
    for i in range(N):
        if W[i].X > 1e-6:
            waiting_times[str(i)] = W[i].X
            
    solution = {
        "objective_value": obj_val,
        "truck_route": truck_route,
        "drone_assignments": drone_assignments,
        "waiting_times": waiting_times
    }
    
    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
        
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    solution = solve(instance, args.time_limit, logger)
    
    if solution is not None:
        if logger:
            logger.log_solution(solution["objective_value"], solution)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()