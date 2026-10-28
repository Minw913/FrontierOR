import json
import math
import argparse
import sys
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None


def main():
    parser = argparse.ArgumentParser(description="Solve Dial-a-Ride Problem (DARP)")
    parser.add_argument('--instance_path', type=str, required=True, help="Path to the instance JSON file.")
    parser.add_argument('--solution_path', type=str, required=True, help="Path to write the solution JSON.")
    parser.add_argument('--time_limit', type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument('--log_path', type=str, default=None, help="Path to log incumbent solutions.")
    args = parser.parse_args()

    # Load instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    nodes = data['nodes']
    num_nodes = data['num_nodes']
    num_vehicles = data['num_vehicles']
    capacity = data['vehicle_capacity']
    max_ride_time = data['maximum_ride_time']
    max_route_duration = data['maximum_route_duration']

    origin_depot = 0
    dest_depot = num_nodes - 1

    P = []
    D = []
    N_nodes = []
    pairs = {}
    
    for node in nodes:
        nid = node['node_id']
        ntype = node['node_type']
        if ntype == 'pickup':
            P.append(nid)
            N_nodes.append(nid)
            pairs[nid] = node['paired_node']
        elif ntype == 'dropoff':
            D.append(nid)
            N_nodes.append(nid)

    def dist(i, j):
        dx = nodes[i]['x'] - nodes[j]['x']
        dy = nodes[i]['y'] - nodes[j]['y']
        return math.hypot(dx, dy)

    # Precompute arcs, filter by basic time feasibility
    arcs = []
    
    # Depot to Pickups
    for j in P:
        if nodes[0]['earliest_time'] + nodes[0]['service_duration'] + dist(0, j) <= nodes[j]['latest_time']:
            arcs.append((0, j))

    # User nodes to user nodes
    for i in N_nodes:
        for j in N_nodes:
            if i != j:
                if nodes[i]['earliest_time'] + nodes[i]['service_duration'] + dist(i, j) <= nodes[j]['latest_time']:
                    # Reject immediate impossible combinations based on paired logic
                    if nodes[i]['node_type'] == 'dropoff' and nodes[j]['node_type'] == 'pickup':
                        if nodes[i]['user_id'] == nodes[j]['user_id']:
                            continue  # Cannot dropoff before pickup
                    arcs.append((i, j))

    # Dropoffs to Dest Depot
    for i in D:
        if nodes[i]['earliest_time'] + nodes[i]['service_duration'] + dist(i, dest_depot) <= nodes[dest_depot]['latest_time']:
            arcs.append((i, dest_depot))

    # Depot to Depot (unused vehicles)
    arcs.append((0, dest_depot))

    K = list(range(num_vehicles))

    # Initialize model
    model = gp.Model("DARP")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)  # Restrict to single CPU core
    model.setParam('OutputFlag', 0)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Decision variables
    x = {}
    for k in K:
        for (i, j) in arcs:
            x[k, i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{k}_{i}_{j}")

    B = {}
    L = {}
    for i in N_nodes:
        B[i] = model.addVar(lb=nodes[i]['earliest_time'], ub=nodes[i]['latest_time'], 
                            vtype=GRB.CONTINUOUS, name=f"B_{i}")
        lb_val = max(0, nodes[i]['load'])
        ub_val = min(capacity, capacity + nodes[i]['load'])
        L[i] = model.addVar(lb=lb_val, ub=ub_val, vtype=GRB.CONTINUOUS, name=f"L_{i}")

    B_start = {}
    B_end = {}
    for k in K:
        B_start[k] = model.addVar(lb=nodes[0]['earliest_time'], ub=nodes[0]['latest_time'], 
                                  vtype=GRB.CONTINUOUS, name=f"B0_{k}")
        B_end[k] = model.addVar(lb=nodes[dest_depot]['earliest_time'], ub=nodes[dest_depot]['latest_time'], 
                                vtype=GRB.CONTINUOUS, name=f"Be_{k}")

    # Objective
    model.setObjective(
        gp.quicksum(x[k, i, j] * dist(i, j) for k in K for (i, j) in arcs), 
        GRB.MINIMIZE
    )

    # 1. Exact visit for each request
    for p in P:
        model.addConstr(gp.quicksum(x[k, p, j] for k in K for j in N_nodes if (p, j) in arcs) == 1)

    # 2. Flow conservation
    for k in K:
        for i in N_nodes:
            in_flow = gp.quicksum(x[k, j, i] for j in [0] + N_nodes if (j, i) in arcs)
            out_flow = gp.quicksum(x[k, i, j] for j in N_nodes + [dest_depot] if (i, j) in arcs)
            model.addConstr(in_flow == out_flow)

    # 3. Vehicle origin/dest links
    for k in K:
        model.addConstr(gp.quicksum(x[k, 0, j] for j in P + [dest_depot] if (0, j) in arcs) == 1)
        model.addConstr(gp.quicksum(x[k, i, dest_depot] for i in D + [0] if (i, dest_depot) in arcs) == 1)

    # 4. Same route logic logic
    for k in K:
        for p in P:
            d = pairs[p]
            v_p = gp.quicksum(x[k, p, j] for j in N_nodes if (p, j) in arcs)
            v_d = gp.quicksum(x[k, d, j] for j in N_nodes + [dest_depot] if (d, j) in arcs)
            model.addConstr(v_p == v_d)

    # 5. Time consistency & Time windows tracking via Big-M
    for (i, j) in arcs:
        if i == 0 and j == dest_depot:
            M = nodes[0]['latest_time'] + nodes[0]['service_duration'] + dist(0, dest_depot) - nodes[dest_depot]['earliest_time']
            M = max(0, M)
            for k in K:
                model.addConstr(B_end[k] >= B_start[k] + nodes[0]['service_duration'] + dist(0, dest_depot) - M * (1 - x[k, 0, dest_depot]))
        
        elif i == 0:
            M = nodes[0]['latest_time'] + nodes[0]['service_duration'] + dist(0, j) - nodes[j]['earliest_time']
            M = max(0, M)
            for k in K:
                model.addConstr(B[j] >= B_start[k] + nodes[0]['service_duration'] + dist(0, j) - M * (1 - x[k, 0, j]))
        
        elif j == dest_depot:
            M = nodes[i]['latest_time'] + nodes[i]['service_duration'] + dist(i, dest_depot) - nodes[dest_depot]['earliest_time']
            M = max(0, M)
            for k in K:
                model.addConstr(B_end[k] >= B[i] + nodes[i]['service_duration'] + dist(i, dest_depot) - M * (1 - x[k, i, dest_depot]))
        
        else:
            M = nodes[i]['latest_time'] + nodes[i]['service_duration'] + dist(i, j) - nodes[j]['earliest_time']
            M = max(0, M)
            z_ij = gp.quicksum(x[k, i, j] for k in K)
            model.addConstr(B[j] >= B[i] + nodes[i]['service_duration'] + dist(i, j) - M * (1 - z_ij))

    # 6. Load continuity constraints via Big-M
    for (i, j) in arcs:
        if i in N_nodes and j in N_nodes:
            z_ij = gp.quicksum(x[k, i, j] for k in K)
            model.addConstr(L[j] >= L[i] + nodes[j]['load'] - capacity * (1 - z_ij))
            model.addConstr(L[j] <= L[i] + nodes[j]['load'] + capacity * (1 - z_ij))
            
    for j in P:
        if (0, j) in arcs:
            z_0j = gp.quicksum(x[k, 0, j] for k in K)
            model.addConstr(L[j] >= nodes[j]['load'] - capacity * (1 - z_0j))
            model.addConstr(L[j] <= nodes[j]['load'] + capacity * (1 - z_0j))

    # 7. Ride Time bounding Constraints
    ride_times = {}
    for p in P:
        orig = p
        dest = pairs[p]
        uid = nodes[p]['user_id']
        lb = max(dist(orig, dest), 0.0)
        ride_times[p] = model.addVar(lb=lb, ub=max_ride_time, vtype=GRB.CONTINUOUS, name=f"ride_{uid}")
        model.addConstr(ride_times[p] == B[dest] - (B[orig] + nodes[orig]['service_duration']))

    # 8. Constraints for Max Route Duration limit
    for k in K:
        model.addConstr(B_end[k] - B_start[k] <= max_route_duration)

    # 9. Optional Symmetry Breaking constraint (Unused vehicles sequenced at the end)
    for k in range(1, len(K)):
        if (0, dest_depot) in arcs:
            model.addConstr(x[k, 0, dest_depot] >= x[k-1, 0, dest_depot])

    # Connect callback for logging best intermediate outcomes 
    def callback(cb_model, where):
        if where == GRB.Callback.MIPSOL:
            obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(float(obj))

    model.optimize(callback)

    if model.SolCount > 0:
        routes_out = {}
        service_times_out = {}
        ride_times_out = {}

        # Fill route chains mappings
        for k in K:
            curr = 0
            route = [curr]
            while curr != dest_depot:
                moved = False
                for j in N_nodes + [dest_depot]:
                    if (curr, j) in arcs:
                        if x[k, curr, j].X > 0.5:
                            route.append(j)
                            curr = j
                            moved = True
                            break
                if not moved:
                    break   # fail-safe edgebreak
            routes_out[str(k)] = route

        # Collect assigned schedule intervals
        for i in N_nodes:
            service_times_out[str(i)] = float(B[i].X)
            
        for k in K:
            service_times_out[f"depot_start_{k}"] = float(B_start[k].X)
            service_times_out[f"depot_end_{k}"] = float(B_end[k].X)

        # Ride duration assignment outputs    
        for p in P:
            uid = nodes[p]['user_id']
            ride_times_out[str(uid)] = float(ride_times[p].X)

        solution = {
            "objective_value": float(model.ObjVal),
            "routes": routes_out,
            "service_times": service_times_out,
            "ride_times": ride_times_out
        }
    else:
        solution = {"error": "No optimal or feasible solution found within the specified time limit."}

    # Store derived output
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()