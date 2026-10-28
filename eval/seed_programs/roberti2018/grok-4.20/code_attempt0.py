import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
import math
from solution_logger import SolutionLogger

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    # Extract data
    depot = instance['depot']
    satellites = instance['satellites']
    customers = instance['customers']
    groups = instance['groups']
    Q1 = instance['Q1']
    Q2 = instance['Q2']
    T1 = instance['T1']
    T2 = instance['T2']
    num_sat = instance['num_satellites']
    num_cust = instance['num_customers']
    num_groups = instance['num_groups']

    # Node mappings
    sat_ids = [s['id'] for s in satellites]
    cust_ids = [c['id'] for c in customers]
    all_nodes = [depot['id']] + sat_ids + cust_ids
    id_to_idx = {nid: i for i, nid in enumerate(all_nodes)}
    D = 0  # depot index
    S = list(range(1, 1 + num_sat))
    C = list(range(1 + num_sat, 1 + num_sat + num_cust))
    V2 = S + C  # second echelon nodes excluding depot

    # Coordinates and demands
    coords = {}
    coords[depot['id']] = (depot['x'], depot['y'])
    for s in satellites:
        coords[s['id']] = (s['x'], s['y'])
    demand_dict = {}
    for c in customers:
        coords[c['id']] = (c['x'], c['y'])
        demand_dict[c['id']] = c['demand']
    demand = [0.0] * len(all_nodes)
    for cid in cust_ids:
        demand[id_to_idx[cid]] = demand_dict[cid]

    # Satellite data
    sat_capacity = {}
    sat_max_veh = {}
    sat_handling = {}
    sat_idx_to_id = {}
    for i, s in enumerate(satellites):
        sid = s['id']
        sat_idx_to_id[i] = sid
        sat_capacity[sid] = s['capacity']
        sat_max_veh[sid] = s['max_vehicles']
        sat_handling[sid] = s['handling_cost']

    # Group data
    group_to_customers = {}
    group_list = []
    for g in groups:
        gid = g['group_id']
        group_list.append(gid)
        group_to_customers[gid] = g['customer_ids']

    # Precompute distances
    dist1 = {}  # first echelon: depot and sats
    for i in [D] + S:
        nid1 = all_nodes[i]
        for j in [D] + S:
            nid2 = all_nodes[j]
            dist1[(i, j)] = euclidean_distance(coords[nid1][0], coords[nid1][1], coords[nid2][0], coords[nid2][1])

    dist2 = {}
    for i in V2:
        nid1 = all_nodes[i]
        for j in V2:
            nid2 = all_nodes[j]
            dist2[(i, j)] = euclidean_distance(coords[nid1][0], coords[nid1][1], coords[nid2][0], coords[nid2][1])

    # Build MIP
    m = gp.Model("TwoEchelon")

    # Time limit
    m.Params.TimeLimit = args.time_limit
    m.Params.Threads = 1
    m.Params.MIPGap = 0.01  # allow some gap for larger instances

    # Variables

    # First echelon: which vehicle uses which arc (depot + sats)
    x1 = m.addVars(T1, [D] + S, [D] + S, vtype=GRB.BINARY, name="x1")

    # Delivery quantities from first echelon vehicles to satellites
    y1 = m.addVars(T1, S, vtype=GRB.CONTINUOUS, lb=0, name="y1")

    # Second echelon: which satellite serves which customer/group
    # Use group assignment
    z = m.addVars(group_list, S, vtype=GRB.BINARY, name="z")  # group to satellite

    # Second echelon vehicle usage per satellite
    num_veh_s = m.addVars(S, vtype=GRB.INTEGER, lb=0, ub=T2, name="num_veh_s")

    # Second echelon arcs: from satellite or customer to customer/satellite
    x2 = m.addVars(S, V2, V2, vtype=GRB.BINARY, name="x2")

    # Load on second echelon arcs
    f2 = m.addVars(S, V2, V2, vtype=GRB.CONTINUOUS, lb=0, name="f2")

    # Auxiliary: total flow through each satellite
    flow_sat = m.addVars(S, vtype=GRB.CONTINUOUS, lb=0, name="flow_sat")

    # Objective: travel1 + handling + travel2
    travel1 = gp.quicksum(dist1[(i,j)] * x1[v,i,j] for v in range(T1) for i in [D]+S for j in [D]+S if i != j)
    handling = gp.quicksum(sat_handling[sat_idx_to_id[s-1]] * flow_sat[s] for s in S)
    travel2 = gp.quicksum(dist2[(i,j)] * x2[s,i,j] for s in S for i in V2 for j in V2 if i != j)
    m.setObjective(travel1 + handling + travel2, GRB.MINIMIZE)

    # Constraints

    # 1. First echelon capacity per vehicle
    for v in range(T1):
        m.addConstr(gp.quicksum(y1[v,s] for s in S) <= Q1)

    # 2. Satellite throughput capacity
    for s_idx in range(num_sat):
        s = S[s_idx]
        sid = sat_idx_to_id[s_idx]
        m.addConstr(flow_sat[s] <= sat_capacity[sid])

    # 3. First echelon vehicle limit (at most T1 used, but since indexed 0..T1-1 it's ok)
    # Each vehicle can be used or not (no fixed cost, so always allow)

    # 4. Flow conservation for first echelon: each satellite inflow = total delivered
    for s_idx in range(num_sat):
        s = S[s_idx]
        m.addConstr(flow_sat[s] == gp.quicksum(y1[v, s] for v in range(T1)))

    # 5. Link deliveries to visits in first echelon
    for v in range(T1):
        for s in S:
            m.addConstr(y1[v,s] <= Q1 * gp.quicksum(x1[v, i, s] for i in [D]+S if i != s))
            m.addConstr(y1[v,s] <= sat_capacity[sat_idx_to_id[s-1]])

    # 6. First echelon degree constraints and subtour elimination (MTZ)
    u1 = m.addVars(T1, [D]+S, vtype=GRB.CONTINUOUS, lb=0, ub=len(S)+1, name="u1")
    for v in range(T1):
        # Outdegree from depot = indegree to depot
        m.addConstr(gp.quicksum(x1[v, D, j] for j in S) == gp.quicksum(x1[v, i, D] for i in S))
        # Degree at satellites
        for s in S:
            m.addConstr(gp.quicksum(x1[v, i, s] for i in [D]+S if i != s) == gp.quicksum(x1[v, s, j] for j in [D]+S if j != s))
            m.addConstr(gp.quicksum(x1[v, i, s] for i in [D]+S if i != s) <= 1)  # visit at most once per vehicle
        # MTZ
        for i in S:
            for j in S:
                if i != j:
                    m.addConstr(u1[v,i] - u1[v,j] + (len(S)+1)*x1[v,i,j] <= len(S))

    # 7. Group to satellite assignment (exactly one satellite per group)
    for g in group_list:
        m.addConstr(gp.quicksum(z[g, s] for s in S) == 1)

    # 8. Link second echelon routes to satellite assignment
    # All customers in a group assigned to same satellite via z
    for g_idx, g in enumerate(group_list):
        custs = group_to_customers[g]
        for cust_id in custs:
            c_idx = id_to_idx[cust_id]
            c = c_idx  # index in V2
            for s in S:
                m.addConstr(gp.quicksum(x2[s, c, j] for j in V2 if j != c) == z[g, s])
                m.addConstr(gp.quicksum(x2[s, i, c] for i in V2 if i != c) == z[g, s])

    # 9. Second echelon: each satellite has exactly one self-loop if used, but actually we allow no self loop since not visiting self as customer
    # Better: each used satellite must have proper degree for its customers
    # Each customer has indegree and outdegree exactly 1 from its satellite's routes
    for c_idx_global in C:
        c = c_idx_global
        # will be covered by above via z

    # 10. Satellite vehicle limits and total vehicles
    for s_idx in range(num_sat):
        s = S[s_idx]
        sid = sat_idx_to_id[s_idx]
        m.addConstr(num_veh_s[s] <= sat_max_veh[sid])

    m.addConstr(gp.quicksum(num_veh_s[s] for s in S) <= T2)

    # 11. Link number of vehicles to number of routes per satellite
    # Number of vehicles from satellite = number of outgoing arcs from satellite
    for s in S:
        m.addConstr(num_veh_s[s] == gp.quicksum(x2[s, s, j] for j in V2 if j != s))

    # 12. Second echelon capacity and flow constraints
    for s in S:
        sid = sat_idx_to_id[s-1]
        for i in V2:
            for j in V2:
                if i == j:
                    continue
                # Load bounds
                if j in C:  # going to customer
                    m.addConstr(f2[s, i, j] >= demand[j] * x2[s, i, j])
                else:
                    m.addConstr(f2[s, i, j] >= 0)
                if i in C:  # leaving customer
                    m.addConstr(f2[s, i, j] <= (Q2 - demand[i]) * x2[s, i, j])
                else:
                    m.addConstr(f2[s, i, j] <= Q2 * x2[s, i, j])

    # 13. Flow conservation at customers for each satellite's subgraph
    for s in S:
        for k in C:
            if k == s: continue  # shouldn't be
            inflow = gp.quicksum(f2[s, i, k] for i in V2 if i != k)
            outflow = gp.quicksum(f2[s, k, j] for j in V2 if j != k)
            m.addConstr(inflow - outflow == demand[k])

    # 14. Flow out of satellite equals total demand assigned to it
    for s in S:
        assigned_demand = 0
        for g in group_list:
            custs_g = group_to_customers[g]
            for cid in custs_g:
                assigned_demand += demand_dict[cid] * z[g, s]
        m.addConstr(gp.quicksum(f2[s, s, j] for j in V2 if j != s) == assigned_demand)
        m.addConstr(flow_sat[s] == assigned_demand)

    # 15. Subtour elimination for second echelon (MTZ per satellite)
    u2 = m.addVars(S, V2, vtype=GRB.CONTINUOUS, lb=0, name="u2")
    for s in S:
        for i in C:
            m.addConstr(u2[s, i] >= 1)
            for j in C:
                if i != j:
                    m.addConstr(u2[s, i] - u2[s, j] + (num_cust + 1) * x2[s, i, j] <= num_cust)

    # No self loops
    for s in S:
        for i in V2:
            m.addConstr(x2[s, i, i] == 0)
    for v in range(T1):
        for i in [D]+S:
            m.addConstr(x1[v, i, i] == 0)

    # Symmetry breaking: order vehicles
    for v in range(1, T1):
        m.addConstr(gp.quicksum(x1[v, D, j] for j in S) <= gp.quicksum(x1[v-1, D, j] for j in S))

    m.Params.LogFile = ""  # suppress gurobi log if needed

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Solve
    start_time = time.time()
    m.optimize()

    if m.status == GRB.INFEASIBLE:
        print("Model is infeasible")
        # Write empty or basic
        result = {
            "objective_value": 0.0,
            "first_echelon_routes": [],
            "second_echelon_routes": [],
            "group_assignments": {}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(result, f, indent=2)
        return

    # Extract solution
    obj_val = m.objVal if m.objVal is not None else 0.0

    # Log the final objective
    if logger:
        logger.log(obj_val)

    # Reconstruct first echelon routes
    first_routes = []
    for v in range(T1):
        used = False
        for j in S:
            if x1[v, D, j].X > 0.5:
                used = True
                break
        if not used:
            continue
        # Build route
        route = [depot['id']]
        current = D
        visited = set()
        while True:
            found = False
            for nxt in [D] + S:
                if nxt == current or nxt in visited:
                    continue
                if x1[v, current, nxt].X > 0.5:
                    next_node_id = all_nodes[nxt]
                    route.append(next_node_id)
                    visited.add(nxt)
                    current = nxt
                    found = True
                    break
            if not found or current == D:
                break
        if route[-1] != depot['id']:
            route.append(depot['id'])
        # deliveries
        deliveries = {}
        for s in S:
            deliv = y1[v, s].X
            if deliv > 1e-6:
                sid = all_nodes[s]
                deliveries[str(sid)] = float(deliv)
        if len(deliveries) > 0 or len(route) > 2:
            first_routes.append({
                "vehicle": v,
                "route": route,
                "deliveries": deliveries
            })

    # Reconstruct second echelon routes and group assignments
    second_routes = []
    group_assign = {}
    sat_to_group = {}
    for g in group_list:
        for s_idx in range(num_sat):
            s = S[s_idx]
            if z[g, s].X > 0.5:
                sid = sat_idx_to_id[s_idx]
                group_assign[str(g)] = int(sid)
                break

    for s_idx in range(num_sat):
        s = S[s_idx]
        sid = sat_idx_to_id[s_idx]
        # Find all routes from this satellite
        # We find each cycle starting from satellite
        visited_cust = set()
        for start_c in C:
            if x2[s, s, start_c].X < 0.5:
                continue
            # start a new route
            route = [sid]
            current = start_c
            while True:
                cid = all_nodes[current]
                route.append(cid)
                visited_cust.add(current)
                next_found = False
                for nxt in V2:
                    if nxt == current:
                        continue
                    if x2[s, current, nxt].X > 0.5:
                        if nxt == s:
                            route.append(sid)
                            next_found = True
                            break
                        else:
                            current = nxt
                            next_found = True
                            break
                if not next_found or nxt == s:
                    break
            second_routes.append({
                "satellite": int(sid),
                "route": route
            })

    result = {
        "objective_value": float(obj_val),
        "first_echelon_routes": first_routes,
        "second_echelon_routes": second_routes,
        "group_assignments": group_assign
    }

    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()