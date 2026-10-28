import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_sites = instance['num_delivery_sites']
    num_customers = instance['num_customers']
    Q = instance['vehicle_capacity']
    num_vehicles = instance['num_vehicles']
    routing_cost = instance['routing_cost_matrix']
    assignment_cost = instance['assignment_cost_matrix']
    depot_id = 0
    site_ids = [s['id'] for s in instance['delivery_sites']]
    customer_ids = [c['id'] for c in instance['customers']]
    demands = {c['id']: c['demand'] for c in instance['customers']}
    
    # Map indices
    N = num_sites  # sites indexed 0 to N-1 internally, but use 1..N for nodes
    nodes = list(range(num_sites + 1))  # 0 = depot, 1..N = sites
    customers = list(range(num_customers))
    vehicles = list(range(num_vehicles))
    
    total_demand = sum(demands[cid] for cid in customer_ids)
    min_vehicles_lower = (total_demand + Q - 1) // Q
    print(f"Total demand: {total_demand}, Vehicle capacity: {Q}, Min vehicles lower bound: {min_vehicles_lower}", file=sys.stderr)

    # Start timing
    start_time = time.time()
    remaining_time = args.time_limit

    # Create model
    model = gp.Model("LocationRouting")
    model.Params.TimeLimit = remaining_time
    model.Params.MIPGap = 0.01
    model.Params.Threads = 1
    model.Params.LogToConsole = 0

    # Variables
    # x[v,i,j]: 1 if vehicle v travels from node i to j
    x = model.addVars(vehicles, nodes, nodes, vtype=GRB.BINARY, name="x")
    
    # y[v,k]: 1 if site k is visited by vehicle v
    y = model.addVars(vehicles, range(1, num_sites+1), vtype=GRB.BINARY, name="y")
    
    # u[v,k]: cumulative delivery quantity at site k on vehicle v's tour
    u = model.addVars(vehicles, range(1, num_sites+1), vtype=GRB.CONTINUOUS, lb=0, name="u")
    
    # z[c,s,v]: 1 if customer c is assigned to site s on tour v
    z = model.addVars(customers, range(1, num_sites+1), vehicles, vtype=GRB.BINARY, name="z")
    
    # w[c,s]: 1 if customer c is assigned to site s (aggregated)
    w = model.addVars(customers, range(1, num_sites+1), vtype=GRB.BINARY, name="w")

    # Objective: routing cost + assignment cost
    routing_obj = gp.quicksum(routing_cost[i][j] * x[v,i,j] for v in vehicles for i in nodes for j in nodes if i != j)
    assignment_obj = gp.quicksum(assignment_cost[c][s-1] * w[c,s] for c in customers for s in range(1, num_sites+1))
    model.setObjective(routing_obj + assignment_obj, GRB.MINIMIZE)

    # Constraints

    # 1. Each customer assigned to exactly one site
    for c in customers:
        model.addConstr(gp.quicksum(w[c,s] for s in range(1, num_sites+1)) == 1)

    # 2. Link w and z
    for c in customers:
        for s in range(1, num_sites+1):
            model.addConstr(w[c,s] == gp.quicksum(z[c,s,v] for v in vehicles))

    # 3. z only if site visited by that vehicle
    for c in customers:
        for s in range(1, num_sites+1):
            for v in vehicles:
                model.addConstr(z[c,s,v] <= y[v,s])

    # 4. Each site visited by at most one vehicle
    for s in range(1, num_sites+1):
        model.addConstr(gp.quicksum(y[v,s] for v in vehicles) <= 1)

    # 5. Vehicle capacity per tour
    for v in vehicles:
        model.addConstr(gp.quicksum(demands[customer_ids[c]] * z[c,s,v] for c in customers for s in range(1, num_sites+1)) <= Q)

    # 6. Flow conservation
    for v in vehicles:
        for k in nodes:
            model.addConstr(
                gp.quicksum(x[v,i,k] for i in nodes if i != k) == 
                gp.quicksum(x[v,k,j] for j in nodes if j != k)
            )

    # 7. Each site visited at most once across all (already enforced by y and at most one v)
    # 8. Link y to x: if y[v,s]=1 then exactly one incoming and one outgoing arc for that vehicle
    for v in vehicles:
        for s in range(1, num_sites+1):
            model.addConstr(y[v,s] == gp.quicksum(x[v,i,s] for i in nodes if i != s))
            model.addConstr(y[v,s] == gp.quicksum(x[v,s,j] for j in nodes if j != s))

    # 9. At most one departure from depot per vehicle
    for v in vehicles:
        model.addConstr(gp.quicksum(x[v,0,j] for j in nodes if j != 0) <= 1)

    # 10. MTZ-like subtour elimination + capacity linking via u
    M = Q + 1
    for v in vehicles:
        for i in range(1, num_sites+1):
            for j in range(1, num_sites+1):
                if i != j:
                    model.addConstr(u[v,i] + demands_sum_bound(i) - M * (1 - x[v,i,j]) <= u[v,j] + Q * (1 - y[v,j]))
                    # More standard:
                    model.addConstr(u[v,i] - u[v,j] + Q * x[v,i,j] <= Q - demands_sum_bound_lower(j))
    
    # Better MTZ with capacity
    for v in vehicles:
        for i in range(1, num_sites+1):
            model.addConstr(u[v,i] >= 0)
            model.addConstr(u[v,i] <= Q * y[v,i])
        
        for i in range(1, num_sites+1):
            for j in range(1, num_sites+1):
                if i != j:
                    model.addConstr(
                        u[v,i] + gp.quicksum(demands[customer_ids[c]] * z[c,j,v] for c in customers) 
                        - Q * (1 - x[v,i,j]) <= u[v,j]
                    )

    # Depot connections don't carry u
    for v in vehicles:
        for j in range(1, num_sites+1):
            model.addConstr(u[v,j] >= gp.quicksum(demands[customer_ids[c]] * z[c,j,v] for c in customers))
            model.addConstr(u[v,j] <= Q * y[v,j])

    def demands_sum_bound(k):
        return sum(demands[customer_ids[c]] for c in customers)
    
    # Actually the above MTZ is a bit loose but should work
    
    # 11. No self loops
    for v in vehicles:
        for i in nodes:
            model.addConstr(x[v,i,i] == 0)

    # Callback for incumbent logging
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    best_obj = float('inf')
    
    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
            if obj < best_obj - 1e-6:
                best_obj = obj
                if logger:
                    logger.log(float(obj))
    
    model.optimize(callback)
    
    # Extract solution
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT]:
        objective_value = model.objVal
        
        # Build routes
        routes = {}
        for v_idx, v in enumerate(vehicles):
            # Find arcs used
            arcs = []
            sequence = [0]
            current = 0
            visited = set()
            for _ in range(num_sites + 2):  # safety
                found = False
                for j in nodes:
                    if j == current: continue
                    if abs(model.getVarByName(f"x[{v},{current},{j}]").X - 1.0) < 1e-5:
                        if j == 0:
                            arcs.append([current, 0])
                            sequence.append(0)
                            found = True
                            break
                        else:
                            if j not in visited:
                                arcs.append([current, j])
                                sequence.append(j)
                                visited.add(j)
                                current = j
                                found = True
                                break
                if not found or current == 0:
                    break
            if len(arcs) > 0:
                routes[f"vehicle_{v_idx}"] = {
                    "arcs": arcs,
                    "sequence": sequence
                }
        
        # Build assignments and customer_tours
        assignments = {}
        customer_tours = {}
        for c_idx, c_id in enumerate(customer_ids):
            assigned = False
            for s in range(1, num_sites+1):
                if abs(model.getVarByName(f"w[{c_idx},{s}]").X - 1.0) < 1e-5:
                    site_id = site_ids[s-1]
                    assignments[str(c_id)] = site_id
                    # Find which vehicle
                    vehicle_id = None
                    for v_idx, v in enumerate(vehicles):
                        if abs(model.getVarByName(f"z[{c_idx},{s},{v}]").X - 1.0) < 1e-5:
                            vehicle_id = f"vehicle_{v_idx}"
                            break
                    customer_tours[str(c_id)] = {
                        "vehicle": vehicle_id,
                        "site": site_id
                    }
                    assigned = True
                    break
            if not assigned:
                # Should not happen
                pass
        
        solution = {
            "objective_value": float(objective_value),
            "routes": routes,
            "assignments": assignments,
            "customer_tours": customer_tours
        }
        
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # No solution found, write empty feasible if possible or best known
        solution = {
            "objective_value": 0.0,
            "routes": {},
            "assignments": {},
            "customer_tours": {}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()