import argparse
import json
import math
import gurobipy as gp
from gurobipy import GRB

class SolutionLoggerCallback:
    def __init__(self, logger):
        self.logger = logger
        self.best_obj = float('inf')

    def __call__(self, model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < self.best_obj - 1e-4:
                self.best_obj = obj
                if self.logger:
                    self.logger.log(obj)

def compute_dist(coords, n1, n2):
    return math.hypot(coords[n1][0] - coords[n2][0], coords[n1][1] - coords[n2][1])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    logger = None
    if getattr(args, 'log_path', None):
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except ImportError:
            pass

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    depot_id = data['depot']['id']
    coords = {depot_id: (data['depot']['x'], data['depot']['y'])}

    sat_caps = {}
    sat_max_v = {}
    sat_hc = {}
    sat_ids = []

    for s_data in data['satellites']:
        s = s_data['id']
        sat_ids.append(s)
        coords[s] = (s_data['x'], s_data['y'])
        sat_caps[s] = s_data['capacity']
        sat_max_v[s] = s_data['max_vehicles']
        sat_hc[s] = s_data['handling_cost']

    cust_ids = []
    demands = {}
    for c_data in data['customers']:
        c = c_data['id']
        cust_ids.append(c)
        coords[c] = (c_data['x'], c_data['y'])
        demands[c] = c_data['demand']

    g_of = {}
    group_ids = []
    for g_data in data['groups']:
        g = g_data['group_id']
        group_ids.append(g)
        for c in g_data['customer_ids']:
            g_of[c] = g

    Q1 = data['Q1']
    Q2 = data['Q2']
    T1 = data['T1']
    T2 = data['T2']

    dist = {}
    N1 = [depot_id] + sat_ids
    for i in N1:
        for j in N1:
            if i != j:
                dist[(i, j)] = compute_dist(coords, i, j)

    for s in sat_ids:
        for c in cust_ids:
            dist[(s, c)] = compute_dist(coords, s, c)
            dist[(c, s)] = compute_dist(coords, c, s)

    for i in cust_ids:
        for j in cust_ids:
            if i != j:
                dist[(i, j)] = compute_dist(coords, i, j)

    # Initialize model
    model = gp.Model("2E-VRP-Grouped")
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1
    model.Params.MIPFocus = 1
    model.Params.Heuristics = 0.2
    
    # ---------------- Variables ---------------- 
    # Group assignment
    z = model.addVars(group_ids, sat_ids, vtype=GRB.BINARY, name="z")

    # Second echelon arcs
    A2 = []
    for s in sat_ids:
        for c in cust_ids:
            A2.append((s, c))
            A2.append((c, s))
    for c1 in cust_ids:
        for c2 in cust_ids:
            if c1 != c2:
                A2.append((c1, c2))

    x2 = model.addVars(A2, vtype=GRB.BINARY, name="x2")
    f2 = model.addVars(A2, vtype=GRB.CONTINUOUS, name="f2")
    u = model.addVars(sat_ids, vtype=GRB.INTEGER, name="u")
    p2 = model.addVars(cust_ids, lb=1, ub=len(cust_ids), vtype=GRB.CONTINUOUS, name="p2")

    # First echelon arcs
    A1 = []
    for i in N1:
        for j in N1:
            if i != j:
                A1.append((i, j))

    VEH1 = list(range(1, 1 + T1))
    x1 = model.addVars(VEH1, A1, vtype=GRB.BINARY, name="x1")
    q1 = model.addVars(VEH1, sat_ids, vtype=GRB.INTEGER, name="q1", lb=0)
    f1 = model.addVars(VEH1, A1, vtype=GRB.CONTINUOUS, name="f1")
    p1 = model.addVars(VEH1, sat_ids, lb=1, ub=len(sat_ids), vtype=GRB.CONTINUOUS, name="p1")
    
    sat_used = model.addVars(sat_ids, vtype=GRB.BINARY, name="sat_used")

    # ---------------- Constraints ----------------

    # 1. Group to Satellite assignment constraints
    for g in group_ids:
        model.addConstr(gp.quicksum(z[g, s] for s in sat_ids) == 1)

    # 2. 2E Consistency constraints
    for s in sat_ids:
        for c in cust_ids:
            model.addConstr(x2[s, c] <= z[g_of[c], s])
            model.addConstr(x2[c, s] <= z[g_of[c], s])

    for c1 in cust_ids:
        for c2 in cust_ids:
            if c1 != c2:
                for s in sat_ids:
                    model.addConstr(z[g_of[c1], s] - z[g_of[c2], s] <= 1 - x2[c1, c2])
                    model.addConstr(z[g_of[c2], s] - z[g_of[c1], s] <= 1 - x2[c1, c2])

    # 3. 2E Degree constraints
    for c in cust_ids:
        arcs_in = [(i, c) for i in sat_ids] + [(j, c) for j in cust_ids if j != c]
        arcs_out = [(c, j) for j in sat_ids] + [(c, k) for k in cust_ids if k != c]
        model.addConstr(gp.quicksum(x2[i, j] for i, j in arcs_in) == 1)
        model.addConstr(gp.quicksum(x2[i, j] for i, j in arcs_out) == 1)

    # 4. 2E Vehicle limits, capacities and subtour
    for s in sat_ids:
        model.addConstr(u[s] == gp.quicksum(x2[s, c] for c in cust_ids))
        model.addConstr(u[s] == gp.quicksum(x2[c, s] for c in cust_ids))
        model.addConstr(u[s] <= sat_max_v[s] * sat_used[s])
        
        # Redundant valid inequality to tighten bound
        model.addConstr(u[s] * Q2 >= gp.quicksum(demands[c] * z[g_of[c], s] for c in cust_ids))

    model.addConstr(gp.quicksum(u[s] for s in sat_ids) <= T2)

    # 2E MTZ
    for i in cust_ids:
        for j in cust_ids:
            if i != j:
                model.addConstr(p2[j] >= p2[i] + 1 - (len(cust_ids) + 1) * (1 - x2[i, j]))

    # 2E Flow
    for s in sat_ids:
        for c in cust_ids:
            model.addConstr(f2[s, c] <= Q2 * x2[s, c])
            model.addConstr(f2[s, c] >= demands[c] * x2[s, c])
            model.addConstr(f2[c, s] <= (Q2 - demands[c]) * x2[c, s])
            model.addConstr(f2[c, s] >= 0 * x2[c, s])

    for c1 in cust_ids:
        for c2 in cust_ids:
            if c1 != c2:
                model.addConstr(f2[c1, c2] <= (Q2 - demands[c1]) * x2[c1, c2])
                model.addConstr(f2[c1, c2] >= demands[c2] * x2[c1, c2])

    for c in cust_ids:
        arcs_in = [(i, c) for i in sat_ids] + [(j, c) for j in cust_ids if j != c]
        arcs_out = [(c, j) for j in sat_ids] + [(c, k) for k in cust_ids if k != c]
        model.addConstr(gp.quicksum(f2[i, j] for i, j in arcs_in) - 
                        gp.quicksum(f2[j, k] for j, k in arcs_out) == demands[c])

    # 5. 1E Degree, Capacity, and Flow
    total_demands = sum(demands.values())
    for s in sat_ids:
        max_dem_s = min(sat_caps[s], total_demands)
        model.addConstr(gp.quicksum(demands[c] * z[g_of[c], s] for c in cust_ids) <= max_dem_s * sat_used[s])

        # Throughput limit & demand match
        model.addConstr(gp.quicksum(q1[v, s] for v in VEH1) == gp.quicksum(demands[c] * z[g_of[c], s] for c in cust_ids))
        model.addConstr(gp.quicksum(demands[c] * z[g_of[c], s] for c in cust_ids) <= sat_caps[s])

    for v in VEH1:
        # Degree
        for i in N1:
            arcs_in = [(j, i) for j in N1 if j != i]
            arcs_out = [(i, k) for k in N1 if k != i]
            model.addConstr(gp.quicksum(x1[v, j, i] for j, i in arcs_in) == 
                            gp.quicksum(x1[v, i, k] for i, k in arcs_out))

        model.addConstr(gp.quicksum(x1[v, depot_id, j] for j in sat_ids) <= 1)
        
        for s in sat_ids:
            arcs_in_s = [(j, s) for j in N1 if j != s]
            model.addConstr(gp.quicksum(x1[v, j, s] for j, s in arcs_in_s) <= sat_used[s])
            model.addConstr(gp.quicksum(x1[v, j, s] for j, s in arcs_in_s) <= 1)
            model.addConstr(q1[v, s] <= min(sat_caps[s], Q1) * gp.quicksum(x1[v, j, s] for j, s in arcs_in_s))

        model.addConstr(gp.quicksum(q1[v, s] for s in sat_ids) <= Q1)
        
        # Flow
        model.addConstr(gp.quicksum(f1[v, depot_id, j] for j in sat_ids) == gp.quicksum(q1[v, s] for s in sat_ids))
        for s in sat_ids:
            arcs_in_s = [(j, s) for j in N1 if j != s]
            arcs_out_s = [(s, k) for k in N1 if k != s]
            model.addConstr(gp.quicksum(f1[v, j, s] for j, s in arcs_in_s) - 
                            gp.quicksum(f1[v, s, k] for s, k in arcs_out_s) == q1[v, s])
            
            model.addConstr(f1[v, depot_id, s] <= Q1 * x1[v, depot_id, s])
            model.addConstr(f1[v, s, depot_id] == 0)
            
            for j in sat_ids:
                if s != j:
                    model.addConstr(f1[v, s, j] <= Q1 * x1[v, s, j])
                    model.addConstr(p1[v, j] >= p1[v, s] + 1 - (len(sat_ids) + 1) * (1 - x1[v, s, j]))

    # 6. 1E Symmetry breaking
    for v_idx in range(len(VEH1) - 1):
        model.addConstr(gp.quicksum(x1[VEH1[v_idx], depot_id, j] for j in sat_ids) >= 
                        gp.quicksum(x1[VEH1[v_idx+1], depot_id, j] for j in sat_ids))

    # ---------------- Objective ----------------
    travel1 = gp.quicksum(dist[i, j] * x1[v, i, j] for v in VEH1 for i, j in A1)
    travel2 = gp.quicksum(dist[i, j] * x2[i, j] for i, j in A2)
    handling = gp.quicksum(sat_hc[s] * demands[c] * z[g_of[c], s] for s in sat_ids for c in cust_ids)

    model.setObjective(travel1 + travel2 + handling, GRB.MINIMIZE)

    # Solve
    cb = SolutionLoggerCallback(logger)
    model.optimize(cb)

    # ---------------- Extraction ----------------
    if model.SolCount == 0:
        res = {
            "objective_value": 0.0,
            "first_echelon_routes": [],
            "second_echelon_routes": [],
            "group_assignments": {}
        }
    else:
        res = {
            "objective_value": float(model.ObjVal),
            "first_echelon_routes": [],
            "second_echelon_routes": [],
            "group_assignments": {}
        }

        # Extract group assignments
        for g in group_ids:
            for s in sat_ids:
                if z[g, s].X > 0.5:
                    res["group_assignments"][str(g)] = s

        # Extract First Echelon routes
        for v in VEH1:
            used = False
            for j in sat_ids:
                if x1[v, depot_id, j].X > 0.5:
                    used = True
                    break
            if used:
                route = [depot_id]
                curr = depot_id
                deliveries = {}
                while True:
                    nxt = None
                    for j in N1:
                        if j != curr and x1[v, curr, j].X > 0.5:
                            nxt = j
                            break
                    if nxt is None or nxt == depot_id:
                        route.append(depot_id)
                        break
                    route.append(nxt)
                    q_val = int(round(q1[v, nxt].X))
                    if q_val > 0:
                        deliveries[str(nxt)] = q_val
                    curr = nxt
                res["first_echelon_routes"].append({
                    "vehicle": v,
                    "route": route,
                    "deliveries": deliveries
                })

        # Extract Second Echelon routes
        for s in sat_ids:
            start_nodes = [c for c in cust_ids if x2[s, c].X > 0.5]
            for start_node in start_nodes:
                route = [s, start_node]
                curr = start_node
                while True:
                    if x2[curr, s].X > 0.5:
                        route.append(s)
                        break
                    nxt = None
                    for j in cust_ids:
                        if j != curr and x2[curr, j].X > 0.5:
                            nxt = j
                            break
                    if nxt is not None:
                        route.append(nxt)
                        curr = nxt
                    else:
                        route.append(s)
                        break
                res["second_echelon_routes"].append({
                    "satellite": s,
                    "route": route
                })

    with open(args.solution_path, 'w') as f:
        json.dump(res, f, indent=2)

if __name__ == '__main__':
    main()