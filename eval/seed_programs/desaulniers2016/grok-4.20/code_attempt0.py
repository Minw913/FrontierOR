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

    num_customers = instance['num_customers']
    num_periods = instance['num_periods']
    num_vehicles = instance['num_vehicles']
    vehicle_capacity = instance['vehicle_capacity']
    num_nodes = instance['num_nodes']
    distance_matrix = instance['distance_matrix']
    holding_costs = instance['holding_costs']
    customer_demands = instance['customer_demands']
    supplier_production = instance['supplier_production']
    inventory_capacities = instance['inventory_capacities']
    initial_inventories = instance['initial_inventories']

    # Customers indexed from 1 to num_customers
    customers = list(range(1, num_customers + 1))
    periods = list(range(num_periods))
    nodes = list(range(num_nodes))
    vehicles = list(range(num_vehicles))

    # Big-M for routing
    M = vehicle_capacity + 1

    model = gp.Model("IRP")

    # Decision variables
    # x[p][i][j][k] = 1 if vehicle k travels from i to j in period p
    x = {}
    for p in periods:
        for i in nodes:
            for j in nodes:
                if i != j:
                    for k in vehicles:
                        x[p, i, j, k] = model.addVar(vtype=GRB.BINARY, name=f'x_{p}_{i}_{j}_{k}')

    # y[p][i][k] = quantity delivered by vehicle k to customer i in period p
    y = {}
    for p in periods:
        for i in customers:
            for k in vehicles:
                y[p, i, k] = model.addVar(lb=0, ub=vehicle_capacity, vtype=GRB.CONTINUOUS, name=f'y_{p}_{i}_{k}')

    # q[p][i] = inventory at location i at END of period p
    q = {}
    for p in periods:
        for i in nodes:
            q[p, i] = model.addVar(lb=0, ub=inventory_capacities[i], vtype=GRB.CONTINUOUS, name=f'q_{p}_{i}')

    # z[p][i][k] = 1 if vehicle k visits customer i in period p
    z = {}
    for p in periods:
        for i in customers:
            for k in vehicles:
                z[p, i, k] = model.addVar(vtype=GRB.BINARY, name=f'z_{p}_{i}_{k}')

    # Objective: travel cost + holding cost
    travel_cost = gp.quicksum(distance_matrix[i][j] * x[p, i, j, k]
                              for p in periods for i in nodes for j in nodes if i != j
                              for k in vehicles)
    holding_cost = gp.quicksum(holding_costs[i] * q[p, i]
                               for p in periods for i in nodes)
    model.setObjective(travel_cost + holding_cost, GRB.MINIMIZE)

    # Inventory balance and capacity constraints
    for i in nodes:
        # Initial inventory
        if i == 0:  # depot
            model.addConstr(q[0, 0] == initial_inventories[0] + supplier_production[0] - gp.quicksum(y[0, c, k] for c in customers for k in vehicles))
        else:
            c_idx = i - 1
            model.addConstr(q[0, i] == initial_inventories[i] + gp.quicksum(y[0, i, k] for k in vehicles) - customer_demands[c_idx][0])

    for p in range(1, num_periods):
        for i_idx, i in enumerate(nodes):
            if i == 0:  # depot
                model.addConstr(q[p, 0] == q[p-1, 0] + supplier_production[p] - gp.quicksum(y[p, c, k] for c in customers for k in vehicles))
            else:
                c_idx = i - 1
                model.addConstr(q[p, i] == q[p-1, i] + gp.quicksum(y[p, i, k] for k in vehicles) - customer_demands[c_idx][p])

    # Inventory bounds already in variable bounds, but enforce non-negativity explicitly if needed (already in lb=0)

    # Delivery capacity at customer upon receipt
    for p in periods:
        for i in customers:
            prev_q = initial_inventories[i] if p == 0 else q[p-1, i]
            model.addConstr(prev_q + gp.quicksum(y[p, i, k] for k in vehicles) <= inventory_capacities[i])

    # Link y and z
    for p in periods:
        for i in customers:
            for k in vehicles:
                model.addConstr(y[p, i, k] <= vehicle_capacity * z[p, i, k])

    # Each customer visited at most once per period
    for p in periods:
        for i in customers:
            model.addConstr(gp.quicksum(z[p, i, k] for k in vehicles) <= 1)

    # Vehicle capacity per route
    for p in periods:
        for k in vehicles:
            model.addConstr(gp.quicksum(y[p, i, k] for i in customers) <= vehicle_capacity)

    # Routing constraints
    for p in periods:
        for k in vehicles:
            # Flow conservation at customers
            for i in customers:
                model.addConstr(gp.quicksum(x[p, i, j, k] for j in nodes if j != i) == z[p, i, k])
                model.addConstr(gp.quicksum(x[p, j, i, k] for j in nodes if j != i) == z[p, i, k])

            # Flow out of depot
            model.addConstr(gp.quicksum(x[p, 0, j, k] for j in customers) == gp.quicksum(z[p, i, k] for i in customers))

            # Flow into depot
            model.addConstr(gp.quicksum(x[p, j, 0, k] for j in customers) == gp.quicksum(z[p, i, k] for i in customers))

            # Subtour elimination (MTZ formulation)
            u = {}
            for i in customers:
                u[i] = model.addVar(lb=1, ub=num_customers, vtype=GRB.CONTINUOUS, name=f'u_{p}_{i}_{k}')
            for i in customers:
                for j in customers:
                    if i != j:
                        model.addConstr(u[i] - u[j] + (num_customers + 1) * x[p, i, j, k] <= num_customers)

    # Number of vehicles used <= available (implicit since we have num_vehicles)

    # Symmetry breaking: prevent empty vehicles from being used before others (optional)
    for p in periods:
        for k in range(1, num_vehicles):
            model.addConstr(gp.quicksum(z[p, i, k] for i in customers) <= gp.quicksum(z[p, i, k-1] for i in customers))

    model.Params.TimeLimit = args.time_limit
    model.Params.MIPGap = 0.01  # Allow some gap for larger instances
    model.Params.Threads = 1
    model.Params.LogToConsole = 0

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Callback to log incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj < model._best_obj or model._best_obj == float('inf'):
                model._best_obj = obj
                if logger:
                    logger.log(float(obj))

    model._best_obj = float('inf')
    model.optimize(callback)

    if model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED]:
        print("No feasible solution found.")
        sys.exit(1)

    # Extract solution
    objective_value = model.objVal

    # Build routes and deliveries
    routes = {}
    for p in periods:
        routes[str(p)] = {}
        for k in vehicles:
            # Find route by following x variables
            route = [0]
            current = 0
            visited = set()
            while True:
                found = False
                for j in nodes:
                    if j != current and x[p, current, j, k].X > 0.5:
                        if j == 0:
                            route.append(0)
                            found = True
                            break
                        if j not in visited:
                            route.append(j)
                            visited.add(j)
                            current = j
                            found = True
                            break
                if not found or j == 0:
                    break
            # Only include if route has customers
            if len(route) > 2:
                deliveries = {}
                for c in customers:
                    if z[p, c, k].X > 0.5:
                        deliveries[str(c)] = float(y[p, c, k].X)
                routes[str(p)][str(k)] = {
                    "route": route,
                    "deliveries": deliveries
                }

    # Inventories
    inventories = {}
    for i in nodes:
        inv_dict = {}
        for p in periods:
            inv_dict[str(p)] = float(q[p, i].X)
        inventories[str(i)] = inv_dict

    # If no routes for a period/vehicle, do not include or include empty
    # The schema expects routes only for used ones, but to be safe we can omit empty
    solution = {
        "objective_value": float(objective_value),
        "routes": routes,
        "inventories": inventories
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()