import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

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
    demands = [c['demand'] for c in instance['customers']]
    
    # Map indices
    N = num_sites
    nodes = list(range(N + 1))  # 0 = depot, 1..N = sites
    customers = list(range(num_customers))
    vehicles = list(range(num_vehicles))
    
    total_demand = sum(demands)
    print(f"Total demand: {total_demand}, Vehicle capacity: {Q}, Min vehicles lower bound: {(total_demand + Q - 1) // Q}", file=sys.stderr)

    start_time = time.time()

    model = gp.Model("LocationRouting")
    model.Params.TimeLimit = args.time_limit - 5
    model.Params.Threads = 1
    model.Params.LogToConsole = 0
    model.Params.MIPGap = 1e-4

    # Variables
    x = model.addVars(vehicles, nodes, nodes, vtype=GRB.BINARY, name="x")
    y = model.addVars(vehicles, range(1, N+1), vtype=GRB.BINARY, name="y")
    u = model.addVars(vehicles, range(1, N+1), vtype=GRB.CONTINUOUS, lb=0, ub=Q, name="u")
    z = model.addVars(customers, range(1, N+1), vehicles, vtype=GRB.BINARY, name="z")
    w = model.addVars(customers, range(1, N+1), vtype=GRB.BINARY, name="w")

    # Objective
    routing_cost_expr = gp.quicksum(routing_cost[i][j] * x[v, i, j] for v in vehicles for i in nodes for j in nodes if i != j)
    assign_cost_expr = gp.quicksum(assignment_cost[c][s-1] * w[c, s] for c in customers for s in range(1, N+1))
    model.setObjective(routing_cost_expr + assign_cost_expr, GRB.MINIMIZE)

    # Constraints
    # Customer assignment
    for c in customers:
        model.addConstr(gp.quicksum(w[c, s] for s in range(1, N+1)) == 1, name=f"assign_c{c}")

    # Link w and z
    for c in customers:
        for s in range(1, N+1):
            model.addConstr(w[c, s] == gp.quicksum(z[c, s, v] for v in vehicles), name=f"linkw{c}_{s}")

    # z only if site visited by vehicle
    for c in customers:
        for s in range(1, N+1):
            for v in vehicles:
                model.addConstr(z[c, s, v] <= y[v, s], name=f"zlink_{c}_{s}_{v}")

    # Each site at most one vehicle
    for s in range(1, N+1):
        model.addConstr(gp.quicksum(y[v, s] for v in vehicles) <= 1, name=f"site_use_{s}")

    # Capacity per tour
    for v in vehicles:
        model.addConstr(
            gp.quicksum(demands[c] * z[c, s, v] for c in customers for s in range(1, N+1)) <= Q,
            name=f"cap_v{v}"
        )

    # Flow conservation
    for v in vehicles:
        for k in nodes:
            inflow = gp.quicksum(x[v, i, k] for i in nodes if i != k)
            outflow = gp.quicksum(x[v, k, j] for j in nodes if j != k)
            model.addConstr(inflow == outflow, name=f"flow_v{v}_n{k}")

    # Link y to x
    for v in vehicles:
        for s in range(1, N+1):
            model.addConstr(y[v, s] == gp.quicksum(x[v, i, s] for i in nodes if i != s), name=f"y_in_{v}_{s}")
            model.addConstr(y[v, s] == gp.quicksum(x[v, s, j] for j in nodes if j != s), name=f"y_out_{v}_{s}")

    # At most one departure from depot per vehicle
    for v in vehicles:
        model.addConstr(gp.quicksum(x[v, 0, j] for j in nodes if j != 0) <= 1, name=f"depot_out_{v}")

    # No self-loops
    for v in vehicles:
        for i in nodes:
            model.addConstr(x[v, i, i] == 0, name=f"noloop_v{v}_{i}")

    # Delivery quantity linking
    for v in vehicles:
        for s in range(1, N+1):
            model.addConstr(u[v, s] <= Q * y[v, s], name=f"u_bound_{v}_{s}")
            demand_at_s = gp.quicksum(demands[c] * z[c, s, v] for c in customers)
            model.addConstr(u[v, s] >= demand_at_s, name=f"u_min_{v}_{s}")

    # MTZ subtour elimination and capacity
    for v in vehicles:
        for i in range(1, N+1):
            for j in range(1, N+1):
                if i != j:
                    model.addConstr(
                        u[v, i] - u[v, j] + Q * x[v, i, j] <= Q - gp.quicksum(demands[c] * z[c, j, v] for c in customers),
                        name=f"mtz_{v}_{i}_{j}"
                    )

    # Callback
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    best_obj = float('inf')

    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj - 1e-4:
                best_obj = obj_val
                if logger:
                    logger.log(float(obj_val))

    model.optimize(callback)

    # Extract solution
    if model.SolCount > 0:
        objective_value = model.objVal
        
        routes = {}
        for v_idx in vehicles:
            arcs = []
            sequence = [0]
            current = 0
            visited = set()
            for _step in range(N + 2):
                found_next = False
                for j in nodes:
                    if j == current:
                        continue
                    if model.getVarByName(f"x[{v_idx},{current},{j}]").X > 0.5:
                        arcs.append([current, j])
                        if j != 0:
                            sequence.append(site_ids[j-1])
                            current = j
                            visited.add(j)
                        else:
                            sequence.append(0)
                        found_next = True
                        break
                if not found_next or current == 0:
                    break
            if len(arcs) > 1:
                routes[f"vehicle_{v_idx}"] = {"arcs": arcs, "sequence": sequence}
        
        assignments = {}
        customer_tours = {}
        for c_idx, cid in enumerate(customer_ids):
            for s in range(1, N+1):
                if model.getVarByName(f"w[{c_idx},{s}]").X > 0.5:
                    site_id = site_ids[s-1]
                    assignments[str(cid)] = site_id
                    
                    vehicle_str = None
                    for v_idx in vehicles:
                        if model.getVarByName(f"z[{c_idx},{s},{v_idx}]").X > 0.5:
                            vehicle_str = f"vehicle_{v_idx}"
                            break
                    customer_tours[str(cid)] = {
                        "vehicle": vehicle_str,
                        "site": site_id
                    }
                    break

        solution = {
            "objective_value": float(objective_value),
            "routes": routes,
            "assignments": assignments,
            "customer_tours": customer_tours
        }
    else:
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