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
        data = json.load(f)

    N = data['num_nodes']
    C = data['num_customers']
    K = data['parameters']['num_drones']
    depot = data['depot_index']
    sink = N  # virtual sink
    customers = data['customer_indices']
    truck_travel = data['truck_travel_times']
    drone_battery = data['drone_battery_consumption']
    drone_time = data['drone_delivery_time']
    battery_cap = data['parameters']['drone_battery_capacities']

    # Precompute possible launch nodes for each customer (those with finite battery)
    possible_launches = {}
    for c in customers:
        possible_launches[c] = []
        for i in range(N):
            feasible = False
            for d in range(K):
                if drone_battery[d][i][c] < battery_cap[d]:
                    feasible = True
                    break
            if feasible:
                possible_launches[c].append(i)

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    try:
        m = gp.Model("DroneTruckDelivery")
        m.setParam('Seed', 0)
        m.setParam('MIPGap', 1e-4)
        m.setParam('NumericFocus', 0)
        m.setParam('Threads', 1)
        m.setParam('TimeLimit', args.time_limit - 5)  # leave margin

        # Variables
        x = m.addVars(range(N), range(N+1), vtype=GRB.BINARY, name="x")  # to sink allowed
        y = m.addVars(range(K), range(N), customers, vtype=GRB.BINARY, name="y")  # drone, launch, customer
        w = m.addVars(range(N), vtype=GRB.CONTINUOUS, lb=0, name="w")  # wait time at node
        u = m.addVars(range(N), vtype=GRB.CONTINUOUS, lb=0, ub=N, name="u")  # MTZ

        # Objective: truck travel + waiting
        obj = gp.quicksum(truck_travel[i][j] * x[i,j] for i in range(N) for j in range(N+1) if j != i)
        obj += gp.quicksum(w[i] for i in range(N))
        m.setObjective(obj, GRB.MINIMIZE)

        # Truck leaves depot exactly once
        m.addConstr(gp.quicksum(x[depot, j] for j in range(N+1) if j != depot) == 1)

        # Flow conservation - every visited node has indeg=outdeg=1 (except sink)
        for i in range(N):
            if i == depot:
                continue
            m.addConstr(gp.quicksum(x[j,i] for j in range(N) if j != i) == 
                       gp.quicksum(x[i,j] for j in range(N+1) if j != i))

        # Exactly one arc into sink
        m.addConstr(gp.quicksum(x[i, sink] for i in range(N)) == 1)

        # No self-loops
        for i in range(N):
            m.addConstr(x[i,i] == 0)
            if i < N:
                m.addConstr(x[i,sink] <= 1)

        # MTZ to prevent subtours
        for i in range(N):
            for j in range(N):
                if i == j or i == depot:
                    continue
                m.addConstr(u[i] - u[j] + (N-1)*(1 - x[i,j]) >= 0)
        for i in range(N):
            if i != depot:
                m.addConstr(u[i] >= 1)
                m.addConstr(u[i] <= N-1)

        # Each customer served exactly once (truck or drone)
        for c in customers:
            truck_served = gp.quicksum(x[c,j] for j in range(N+1) if j != c)
            drone_served = gp.quicksum(y[d,i,c] for d in range(K) for i in possible_launches[c])
            m.addConstr(truck_served + drone_served == 1)

        # Drone battery capacity
        for d in range(K):
            m.addConstr(gp.quicksum(drone_battery[d][i][c] * y[d,i,c] 
                                   for i in range(N) for c in customers) <= battery_cap[d])

        # Waiting time at each node: max over drones of sum of their delivery times from that node
        # Linearized using big-M (M = 1e6 should be safe)
        M = 1e6
        for i in range(N):
            for d in range(K):
                total_drone_time = gp.quicksum(drone_time[d][i][c] * y[d,i,c] for c in customers)
                m.addConstr(w[i] >= total_drone_time)
                # Also link to truck visit: if any drone launched from i, truck must visit i
                m.addConstr(total_drone_time <= M * gp.quicksum(x[j,i] for j in range(N) if j != i) +
                            M * gp.quicksum(x[i,j] for j in range(N+1) if j != i))

        # If truck visits a customer, it serves it (already enforced by above)

        # Symmetry breaking: order first visited nodes
        for i in range(1, len(customers)//2):
            m.addConstr(u[customers[i]] >= u[customers[i-1]])

        m.optimize()

        if m.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT]:
            # Try to find any feasible - fallback to heuristic
            print("MIP failed, using heuristic fallback", file=sys.stderr)
            solution = heuristic_fallback(data, args.time_limit - (time.time()-start_time))
            obj_val = solution["objective_value"]
        else:
            obj_val = m.objVal
            solution = extract_solution(m, x, y, w, data, obj_val, depot, sink, N, K, customers)

        # Log final
        if logger:
            logger.log_solution(obj_val, solution)

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

    except Exception as e:
        print("Exception occurred:", str(e), file=sys.stderr)
        # Last resort: very simple heuristic
        solution = trivial_solution(data)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)


def extract_solution(m, x, y, w, data, obj_val, depot, sink, N, K, customers):
    truck_route = []
    for i in range(N):
        for j in range(N+1):
            if x[i,j].X > 0.5:
                truck_route.append([i, j if j < N else sink])

    drone_assignments = []
    for d_idx in range(K):
        for i in range(N):
            for c in customers:
                if y[d_idx, i, c].X > 0.5:
                    drone_assignments.append({
                        "drone": d_idx,
                        "dispatch_node": i,
                        "delivery_node": c
                    })

    waiting_times = {}
    for i in range(N):
        val = w[i].X
        if val > 1e-4:  # only report if positive
            waiting_times[str(i)] = float(val)

    # If depot has no wait, include 0 explicitly if visited
    if str(depot) not in waiting_times:
        waiting_times[str(depot)] = 0.0

    solution = {
        "objective_value": float(obj_val),
        "truck_route": truck_route,
        "drone_assignments": drone_assignments,
        "waiting_times": waiting_times
    }
    return solution


def heuristic_fallback(data, remaining_time):
    """Simple greedy heuristic: truck visits all, no drone usage"""
    N = data['num_nodes']
    depot = data['depot_index']
    sink = N
    customers = data['customer_indices']
    truck_travel = data['truck_travel_times']

    # TSP-like: just go depot -> sorted customers -> sink
    route = [depot]
    cust_sorted = sorted(customers, key=lambda c: (data['node_positions']['x'][c], data['node_positions']['y'][c]))
    route.extend(cust_sorted)
    truck_route = []
    for i in range(len(route)-1):
        truck_route.append([route[i], route[i+1]])
    truck_route.append([route[-1], sink])

    total_travel = sum(truck_travel[a[0]][a[1]] for a in truck_route if a[1] != sink)
    # Add return from last to depot
    last = route[-1]
    total_travel += truck_travel[last][depot]

    obj = float(total_travel)

    solution = {
        "objective_value": obj,
        "truck_route": truck_route,
        "drone_assignments": [],
        "waiting_times": {str(depot): 0.0}
    }
    for c in cust_sorted:
        solution["waiting_times"][str(c)] = 0.0
    return solution


def trivial_solution(data):
    """Absolute fallback"""
    N = data['num_nodes']
    depot = data['depot_index']
    sink = N
    customers = data['customer_indices']
    truck_travel = data['truck_travel_times']

    route_nodes = [depot] + customers
    truck_route = []
    total = 0
    for i in range(len(route_nodes)-1):
        a = route_nodes[i]
        b = route_nodes[i+1]
        truck_route.append([a, b])
        total += truck_travel[a][b]
    truck_route.append([route_nodes[-1], sink])
    total += truck_travel[route_nodes[-1]][depot]

    solution = {
        "objective_value": float(total),
        "truck_route": truck_route,
        "drone_assignments": [],
        "waiting_times": {str(i): 0.0 for i in route_nodes}
    }
    return solution


if __name__ == "__main__":
    main()