import argparse
import json
import math
from itertools import permutations
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import time


def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)


def solve_freight_distribution(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()

    # Load instance data
    with open(instance_path, 'r') as f:
        data = json.load(f)

    # Extract instance data
    # Basic parameters
    num_satellites = data['num_satellites']
    num_customers = data['num_customers']
    num_groups = data['num_groups']
    Q1 = data['Q1']  # First-echelon vehicle capacity
    Q2 = data['Q2']  # Second-echelon vehicle capacity
    T1 = data['T1']  # First-echelon vehicles available
    T2 = data['T2']  # Second-echelon vehicles available
    depot = data['depot']
    
    satellites = {s['id']: s for s in data['satellites']}
    customers = {c['id']: c for c in data['customers']}
    groups = {g['group_id']: g for g in data['groups']}

    # IDs
    sat_ids = [s['id'] for s in data['satellites']]  # List of satellite IDs
    cust_ids = [c['id'] for c in data['customers']]  # List of customer IDs
    group_ids = [g['group_id'] for g in data['groups']]  # List of group IDs

    all_nodes = [depot['id']] + sat_ids + cust_ids

    # Location mappings
    coords = {}
    coords[depot['id']] = depot
    for s in data['satellites']:
        coords[s['id']] = s
    for c in data['customers']:
        coords[c['id']] = c

    # Precompute distance matrix
    dist = {}
    for i in all_nodes:
        for j in all_nodes:
            dist[i, j] = euclidean_distance(coords[i], coords[j])

    # Groups to customers mapping
    group_customers = {}  # {groupId: [custIds]}
    for g in groups.values():
        group_customers[g['group_id']] = g['customer_ids']

    # Initialize logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    # Create model
    model = gp.Model("FreightDistribution")
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)  # Suppress Gurobi output to avoid conflicts
    model.setParam('Threads', 1)

    # Variables for first-echelon
    # Binary variable: route v visits satellite s
    x1 = model.addVars(range(T1), sat_ids, vtype=GRB.BINARY, name="x1")

    # Continuous: total load delivered by vehicle v to satellite s
    q1 = model.addVars(range(T1), sat_ids, vtype=GRB.INTEGER, lb=0, name="q1")

    # Binary: satellite s is used (visited by any first echelon vehicle)
    y1 = model.addVars(sat_ids, vtype=GRB.BINARY, name="y1")

    # Load delivered to satellite S from first echelon
    load_to_sat = model.addVars(sat_ids, vtype=GRB.INTEGER, lb=0, name="load_to_sat")

    # Total delivered through satellite s (including handling)
    total_delivered_at_sat = model.addVars(sat_ids, vtype=GRB.INTEGER, lb=0, name="total_delivered_at_sat")

    # Variables for second-echelon
    # For simplicity in modeling, we'll create binary vars x2[i,j,s] for each possible edge and satellite
    # but only allow connections between nodes in same cluster (cluster = group assigned to satellite)
    x2_routes = {}  # x2_routes[(i,j,v,sat)] = is edge (i,j) used by vehicle v at satellite sat
    x2 = {}  # Will contain vars grouped by sat
    for sat in sat_ids:
        x2[(sat,)] = {}  # Group variables by satellite
    
    # We'll only create a subset of arcs since full enumeration can be too large
    # For each satellite, create vars for all customer pairs + sat-to-cust and cust-to-sat
    for sat_id in sat_ids:
        # All possible arcs for vehicles starting from this satellite
        # Sat to customer: start of tour
        for c_id in cust_ids:
            x2[(sat_id,)][(sat_id, c_id)] = model.addVar(vtype=GRB.BINARY, name=f"x2_sat{sat_id}_s{sat_id}_e{c_id}")
        
        # Customer to customer
        for c1_id in cust_ids:
            for c2_id in cust_ids:
                if c1_id != c2_id:
                    x2[(sat_id,)][(c1_id, c2_id)] = model.addVar(vtype=GRB.BINARY, name=f"x2_sat{sat_id}_s{c1_id}_e{c2_id}")
                    
        # Customer to sat: end of tour
        for c_id in cust_ids:
            x2[(sat_id,)][(c_id, sat_id)] = model.addVar(vtype=GRB.BINARY, name=f"x2_sat{sat_id}_s{c_id}_e{sat_id}")


    # Flow variables for load on each edge
    load2 = {}
    for sat_id in sat_ids:
        load2[(sat_id,)] = {}
        for (i, j), var in x2[(sat_id,)].items():
            load2[(sat_id,)][(i, j)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=Q2, name=f"load2_sat{sat_id}_arc{i}_{j}")


    # Group assignment variables: z[group_id, sat_id] = 1 if group assigned to satellite
    z = model.addVars(group_ids, sat_ids, vtype=GRB.BINARY, name="z")

    # First-Echelon Constraints:
    # 1. Vehicle capacity limitation
    for v in range(T1):
        model.addConstr(gp.quicksum(q1[v, s] for s in sat_ids) <= Q1)

    # 2. Linking x1 to whether deliveries occur
    M_big = max(c['demand'] for c in customers.values()) * len(cust_ids)  # A sufficient big M
    for v in range(T1):
        for s in sat_ids:
            model.addConstr(q1[v, s] <= M_big * x1[v, s])

    # 3. Linking y1 to whether satellite is actually served
    for s in sat_ids:
        model.addConstr(gp.quicksum(x1[v, s] for v in range(T1)) >= y1[s])

    # 4. Throughput limit on satellites
    for s in sat_ids:
        model.addConstr(gp.quicksum(q1[v, s] for v in range(T1)) <= satellites[s]['capacity'])

    # Define total delivered to satellite
    for s in sat_ids:
        model.addConstr(load_to_sat[s] == gp.quicksum(q1[v, s] for v in range(T1)))

    # Second-Echelon Constraints:
    # 1. Each customer is visited exactly once
    for c in cust_ids:
        # Which group does c belong to?
        c_group = next(g_id for g_id in group_ids if c in group_customers[g_id])
        
        model.addConstr(gp.quicksum(
            gp.quicksum(x2[(s,)][(i, c)] for i in [s] + cust_ids if (i, c) in x2[(s,)])
            for s in sat_ids
        ) == 1)
        
        # Same number of outcoming arcs as incoming
        model.addConstr(gp.quicksum(
            gp.quicksum(x2[(s,)][(c, j)] for j in [s] + cust_ids if (c, j) in x2[(s,)])
            for s in sat_ids  
        ) == 1)

    # 2. Satellite capacity on handling load
    for s in sat_ids:
        model.addConstr(total_delivered_at_sat[s] <= satellites[s]['capacity'])

    # 3. Assign each customer to satellite via group assignment
    for g_id in group_ids:
        model.addConstr(gp.quicksum(z[g_id, s] for s in sat_ids) == 1)
        
        # All customers in group assigned together
        for c_id in group_customers[g_id]:
            for s in sat_ids:
                # x2 from s can go to c if s is assigned to group of c
                for (i, j), _ in x2[(s,)].items():
                    if j == c:
                        # j must be served by sat assigned to group of j=c
                        model.addConstr(x2[(s,)][(i, j)] <= z[customers[j]['group'], s])

    # 4. Flow conservation at each satellite and customer in second echelon
    for sat_id in sat_ids:
        # Flow conservation: in flow = out flow + load served
        for c_id in cust_ids:
            # For customer c
            inflow = gp.quicksum(x2[(sat_id,)][(i, c_id)] for i in [sat_id] + cust_ids if (i, c_id) in x2[(sat_id,)])
            outflow = gp.quicksum(x2[(sat_id,)][(c_id, j)] for j in [sat_id] + cust_ids if (c_id, j) in x2[(sat_id,)])

            model.addConstr(inflow == outflow) 

        # Flow conservation at satellite itself for second echelon
        sat_inflow = gp.quicksum(x2[(sat_id,)][(i, sat_id)] for i in cust_ids if (i, sat_id) in x2[(sat_id,)])
        sat_outflow = gp.quicksum(x2[(sat_id,)][(sat_id, j)] for j in cust_ids if (sat_id, j) in x2[(sat_id,)])
        
        # This should be zero, since vehicle leaves and comes back the same - maybe redundant
    
        # Load constraints
        for (i, j), var in x2[(sat_id,)].items():
            if (i, j) in load2[(sat_id,)]:
                model.addConstr(load2[(sat_id,)][(i, j)] <= Q2 * var)
                
                # If customer is destination
                if j in cust_ids:
                    # Load going into j must cover its demand if arc is used
                    model.addConstr(load2[(sat_id,)][(i, j)] >= customers[j]['demand'] * var)
                    
                    # If customer is origin
                    if i in cust_ids:
                        model.addConstr(load2[(sat_id,)][(i, j)] <= (Q2 - customers[i]['demand']) * var)

    # 5. Ensure second echelon delivers the first echelon receives
    for s in sat_ids:
        # Load delivered by FE = Load taken by SE
        model.addConstr(    
            gp.quicksum(q1[v, s] for v in range(T1)) == 
            gp.quicksum(
                load2[(s,)][(s, c)] for c in cust_ids 
                if (s, c) in load2[(s,)]
            )
        )

    # 6. Vehicle count constraints:
    # Max number of vehicles per satellite
    for s in sat_ids:
        # Number of vehicles departing from satellite s
        num_vehs_s = gp.quicksum(x2[(s,)][(s, c)] for c in cust_ids if (s, c) in x2[(s,)])
        model.addConstr(num_vehs_s <= satellites[s]['max_vehicles'])

    # Total second echelon vehicles used
    total_vehs = gp.quicksum(x2[(s,)][(s, c)] for s in sat_ids for c in cust_ids if (s, c) in x2[(s,)])
    model.addConstr(total_vehs <= T2)

    # OBJECTIVE FUNCTION
    # 1. First echelon travel cost
    first_ech_tcost = gp.quicksum(
        dist[depot['id'], s] * 2 * x1[v, s]  # Round trip for each satellite
        for v in range(T1) for s in sat_ids 
    )

    # 2. Handling cost at satellites
    handling_cost = gp.quicksum(satellites[s]['handling_cost'] * total_delivered_at_sat[s] for s in sat_ids)

    # 3. Second echelon travel cost
    second_ech_tcost = gp.quicksum(
        gp.quicksum(
            dist[i, j] * x2[(s,)][(i, j)]
            for (i, j) in x2[(s,)].keys()
        )
        for s in sat_ids
    )

    total_cost = first_ech_tcost + handling_cost + second_ech_tcost
    model.setObjective(total_cost, GRB.MINIMIZE)

    # Callback to handle logging of feasible solutions
    def my_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)

    # Optimize
    model.optimize(my_callback)

    # Build solution output
    solution = {
        "objective_value": float('inf'),
        "first_echelon_routes": [],
        "second_echelon_routes": [],
        "group_assignments": {}
    }

    # If solution found
    if model.SolCount > 0 and model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED]:
        if model.Status != GRB.INF_OR_UNBD and model.Status != GRB.UNBOUNDED:
            solution["objective_value"] = model.ObjVal
            
            # Build first echelon routes - approximate from active x1 variables
            fe_routes = {}
            for v in range(T1):
                route = [depot['id']]
                deliveries = {}
                for s in sat_ids:
                    if x1[v, s].X > 0.5:  # Visited and potentially delivered
                        # Simulate simple round-trip for this vehicle
                        quantity = round(q1[v, s].X)
                        if quantity > 0:
                            deliveries[s] = quantity
                        route.extend([s, depot['id']])
                        
                        # Save as separate route if has deliveries
                        if deliveries:
                            solution["first_echelon_routes"].append({
                                "vehicle": v,
                                "route": route,
                                "deliveries": deliveries
                            })
                        break  # Just one delivery per vehicle for now due to complexity of subtour elimination
                else:
                    route = [depot['id']]  # Reset for next satellite (or just keep as 2-stop route per satellite)

            # More realistic extraction of first echelon routes:
            solution["first_echelon_routes"] = []  # Recalculate
            for v in range(T1):
                visited = [s for s in sat_ids if x1[v, s].X > 0.5]
                if visited:
                    # Create tour visiting all of this vehicle's satellites
                    route = [depot['id']] + visited + [depot['id']]
                    deliveries = {}
                    for s in sat_ids:
                        if x1[v, s].X > 0.5:
                            deliveries[s] = int(round(q1[v, s].X))
                    if route:
                        solution["first_echelon_routes"].append({
                            "vehicle": v,
                            "route": route,
                            "deliveries": deliveries
                        })

            # Group assignments
            for g_id in group_ids:
                for s in sat_ids:
                    if z[g_id, s].X > 0.5:
                        solution["group_assignments"][g_id] = s
                        break  # Only one assignment

            # Second echelon routes construction - this is complex, since finding paths from variables
            # We'll simulate reconstruction by following the flow variables
            solution["second_echelon_routes"] = []
            
            for s in sat_ids:
                # Find all routes starting from satellite s
                # Check how many vehicles started from s based on x2 values
                num_vehicles_used_for_sat = 0
                vehicle_routes = []
                
                # Iterate until no unprocessed x2 edge exists from satellite to unused customer
                temp_remaining_x2_vars = {}
                for (start, end), var in x2[(s,)].items():
                    if var.X > 0.5:
                        if (start, end) not in temp_remaining_x2_vars:
                            temp_remaining_x2_vars[(start, end)] = 0
                        temp_remaining_x2_vars[(start, end)] += var.X
                
                while len(temp_remaining_x2_vars) > 0:
                    # Start new route
                    route = [s]
                    current_pos = s
                    while current_pos != s or len(route) == 1:
                        found_next_edge = False
                        for (start, end), val in temp_remaining_x2_vars.items():
                            if start == current_pos:
                                current_pos = end
                                route.append(current_pos)
                                del temp_remaining_x2_vars[(start, end)]
                                found_next_edge = True 
                                if current_pos == s:  # Completed a circuit
                                    break
                                break
                        if not found_next_edge:
                            break  # Dead end
                    vehicle_routes.append(route)
                    num_vehicles_used_for_sat += 1
                    if num_vehicles_used_for_sat >= satellites[s]['max_vehicles']:
                        break  # Don't exceed satellite vehicle limits
                
                for r in vehicle_routes:
                    if len(r) > 2:
                        solution["second_echelon_routes"].append({
                            "satellite": s,
                            "route": r
                        })

        else:
            solution["objective_value"] = float('inf')

    # Write the solution to the specified path
    try:
        with open(solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    except Exception as e:
        print(f"Error writing solution to {solution_path}: {e}")
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Solve the Two-Echelon Freight Distribution Problem.")
    parser.add_argument('--instance_path', required=True, type=str, help="Path to the input instance file")
    parser.add_argument('--solution_path', required=True, type=str, help="Path to the output solution file")
    parser.add_argument('--time_limit', required=True, type=int, help="Time limit in seconds")
    parser.add_argument('--log_path', required=False, type=str, help="Path to the log file (optional)")

    args = parser.parse_args()

    solve_freight_distribution(
        instance_path=args.instance_path,
        solution_path=args.solution_path,
        time_limit=args.time_limit,
        log_path=args.log_path
    )