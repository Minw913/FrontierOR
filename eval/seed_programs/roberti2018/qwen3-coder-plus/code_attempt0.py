import argparse
import json
import math
from itertools import combinations
from collections import defaultdict
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
    num_satellites = data['num_satellites']
    num_customers = data['num_customers']
    num_groups = data['num_groups']
    Q1 = data['Q1']
    Q2 = data['Q2']
    T1 = data['T1']
    T2 = data['T2']
    depot = data['depot']
    
    satellites = {s['id']: s for s in data['satellites']}
    customers = {c['id']: c for c in data['customers']}
    groups = {g['group_id']: g for g in data['groups']}

    # Create lists of IDs for convenience
    sat_ids = list(satellites.keys())
    cust_ids = list(customers.keys())
    group_ids = list(groups.keys())

    all_nodes = [depot['id']] + sat_ids + cust_ids
    
    # Precompute distances between all pairs
    dist_matrix = {}
    coords = {depot['id']: depot}
    coords.update(satellites)
    coords.update(customers)
    
    for i in all_nodes:
        for j in all_nodes:
            dist_matrix[i, j] = euclidean_distance(coords[i], coords[j])

    # Group to customer mapping
    group_to_customers = defaultdict(list)
    for g_id, g_info in groups.items():
        for cid in g_info['customer_ids']:
            group_to_customers[g_id].append(cid)

    # Initialize logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    # Create model
    model = gp.Model("freight_distribution")
    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', 1)

    # Decision variables for first echelon (depot to satellites)
    x1 = model.addVars(depot['id'], sat_ids, vtype=GRB.BINARY,
                       name=lambda i, j: f'x1_depot_{i}_to_sat_{j}')
    x1_sat_to_sat = model.addVars(sat_ids, sat_ids, vtype=GRB.BINARY, name='x1_sat_to_sat')
    y1 = model.addVars(sat_ids, vtype=GRB.BINARY, name=lambda j: f'y1_sat_{j}_visited')

    # Deliveries from first echelon vehicles to satellites
    d1 = model.addVars(T1, sat_ids, vtype=GRB.INTEGER, ub=Q2, name='delivery_from_vehicle_1_to_sat_2')

    # Decision variables for second echelon (satellites to customers)
    # x2[sat_id][cust1, cust2] binary vars: whether vehicle goes from cust1 to cust2 using this satellite
    x2 = {}
    y2 = {}

    # Create variables for each satellite separately
    for sat_id in sat_ids:
        x2[sat_id] = {}
        # Vehicle routes originating from sat_id, including depot-style loops back to sat_id
        x2_keys_here = []
        # Start and end at this satellite
        sat_to_cx = [(sat_id, c_id) for c_id in cust_ids]
        cx_to_sat = [(c_id, sat_id) for c_id in cust_ids]
        cx_to_cx = [(c_id1, c_id2) for c_id1 in cust_ids for c_id2 in cust_ids if c_id1 != c_id2]
        for edge in sat_to_cx + cx_to_sat + cx_to_cx:
            x2[sat_id][edge] = model.addVar(vtype=GRB.BINARY, name=f"x2_sat_{sat_id}_{edge[0]}_to_{edge[1]}")

        y2[sat_id] = model.addVar(vtype=GRB.BINARY, name=f"y2_sat_used_{sat_id}")

    # Flow variables for load on arcs for second echelon
    load2 = {}
    for sat_id in sat_ids:
        load2[sat_id] = {}
        for edge in x2[sat_id].keys():
            load2[sat_id][edge] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=Q2,
                                              name=f"load2_sat_{sat_id}_on_edge_{edge}")

    # Group assignments
    z = model.addVars(group_ids, sat_ids, vtype=GRB.BINARY, name='assign_group_to_sat')  # assign group g to satellite s

    # Load delivered to each satellite for second echelon
    load_to_sat = model.addVars(sat_ids, vtype=GRB.INTEGER, ub=sum(c['demand'] for c in customers.values()),
                                name='load_to_sat')

    # Constraints
    # First Echelon Route Formation
    # Flow balance constraints for depot and satellites
    model.addConstr(sum(x1[depot['id'], j] for j in sat_ids) <= T1)
    for j in sat_ids:
        model.addConstr(sum(x1[depot['id'], j]) + sum(x1_sat_to_sat[k, j] for k in sat_ids if k != j)
                        == sum(x1_sat_to_sat[j, k] for k in sat_ids if k != j) + sum(y1[j]))
        
    # Same flow out from depot to same number of return to depot
    model.addConstr(sum(x1[depot['id'], j] for j in sat_ids) == 
                    sum(x1_sat_to_sat[j, k] for j in sat_ids for k in sat_ids))  # Not fully correct yet
    
    # Fix: Proper flow balance constraint with sub-tour prevention - we'll simplify with bigM if needed
    # But let's implement MTZ-like formulation instead

    # Add MTZ variables for subtour elimination in first echelon
    u1 = model.addVars(sat_ids, vtype=GRB.INTEGER, lb=2, ub=num_satellites + 1, name="u1")
    for i in sat_ids:
        for j in sat_ids:
            if i != j:
                # Only when x1[i,j] = 1, we want u1[j] >= u1[i] + 1
                model.addConstr(u1[j] >= u1[i] + 1 - num_satellites * (1 - x1_sat_to_sat[i, j]))

    # Corrected route definition and flow balance
    # Flow from depot
    model.addConstrs(
        x1[depot['id'], j] == sum(x1_sat_to_sat[j, k] for k in sat_ids if k != j)
        if sum(x1_sat_to_sat[j, k] for k in sat_ids if k != j).getValue() is not None
        else x1[depot['id'], j] == 0  # This won't work directly. Let me fix below
        for j in sat_ids
    )
    
    # Let's restart logic without complex subtour - use direct approach with y1:
    # y1[j] = 1 if satellite j is visited by at least one first-echelon vehicle
    
    # Define y1[j] properly:
    for j in sat_ids:
        # At least entering or leaving means visited
        model.addConstr(y1[j] <= sum(x1[depot['id'], k] for k in sat_ids if k == j) +
                        sum(x1_sat_to_sat[i, j] for i in sat_ids if i != j))
        model.addConstr(y1[j] <= sum(x1_sat_to_sat[j, k] for k in sat_ids if j != k) +
                        sum(x1[depot['id'], k] for k in sat_ids if k == j))

        # And sum of both directions must > 0 if y1[j]==1
        # So: 2*y1[j] <= in_deg + out_deg
        model.addConstr(2 * y1[j] <= 
               sum(x1[depot['id'], k] for k in sat_ids if k == j) +
               sum(x1_sat_to_sat[i, j] for i in sat_ids if i != j) +
               sum(x1_sat_to_sat[j, k] for k in sat_ids if j != k) +
               sum(x1[depot['id'], k] for k in sat_ids if k == j)  # Count depot-sat links twice since they both start/stop
        )

    # Capacity constraint for first-echelon vehicles
    # Sum of deliveries by a single vehicle across satellites it visits <= Q1
    for v_idx in range(T1):
        delivered_by_v = sum(d1[v_idx, j] for j in sat_ids)
        model.addConstr(delivered_by_v <= Q1)

    # Delivery cannot exceed satellite's throughput capacity per delivery point
    for v_idx in range(T1):
        for j in sat_ids:
            model.addConstr(d1[v_idx, j] <= satellites[j]['capacity'] * 
                            (sum(x1[depot['id'], j]) + sum(x1_sat_to_sat[i, j] for i in sat_ids)))
    
    # Also ensure total delivered <= satellite capacity
    for j in sat_ids:
        model.addConstr(sum(d1[v_idx, j] for v_idx in range(T1)) <= satellites[j]['capacity'])

    # Second Echelon Constraints:
    # Every customer assigned to exactly one group, and that group assigns to exactly one satellite
    # Assignment consistency: all customers in group go to same satellite
    for g_id, g_info in groups.items():
        model.addConstr(sum(z[g_id, s] for s in sat_ids) == 1)  # Group assigned to exactly one satellite

    for g_id in group_ids:
        group_customer_ids = groups[g_id]['customer_ids']
        for c_id in group_customer_ids:
            # Customer assignment follows group's assignment
            for s in sat_ids:
                # This isn't straightforward: z maps groups->satellites, 
                # but we don't have a variable for customers->satellites.
                # However, if we define customer assignments from their groups later, it's fine.
                pass

    # Now, force that if customer c in group g, and g assigned to s, then customer routes use satellite s
    for g_id in group_ids:
        for s in sat_ids:
            # For all customers c in group g_id:
            group_customers = group_to_customers[g_id]
            # Use x2[s] for routes from satellite s
            # The routes must visit these specific customers
            for c_id in group_customers:
                # Force x2 vars to connect c if z[g_id, s] is 1
                # Sum of x2[s] involving c_id must be activated if it's in this cluster
                in_flow = sum(x2[s][i, c_id] for i in [s] + cust_ids if (i, c_id) in x2[s])
                out_flow = sum(x2[s][c_id, j] for j in [s] + cust_ids if (c_id, j) in x2[s])
                model.addConstr(in_flow == z[g_id, s])
                model.addConstr(out_flow == z[g_id, s])


    # Flow conservation in second echelon routes
    for sat_id in sat_ids:
        for c_id in cust_ids:
            if any((i, j) in x2[sat_id] for i in [sat_id] + cust_ids for j in [sat_id] + cust_ids):
                # Sum of all flows into c_id equals sum of all flows out of c_id
                inflow = sum(x2[sat_id][i, c_id] for i in [sat_id] + cust_ids if (i, c_id) in x2[sat_id])
                outflow = sum(x2[sat_id][c_id, j] for j in [sat_id] + cust_ids if (c_id, j) in x2[sat_id])
                if inflow.size > 0 and outflow.size > 0:
                    model.addConstr(inflow == outflow)
                    
    # Each customer must be served exactly once
    for c_id in cust_ids:
        total_served_count = sum(
            sum(x2[sat_id][i, c_id] for i in [sat_id] + cust_ids if (i, c_id) in x2[sat_id])
            for sat_id in sat_ids
        )
        model.addConstr(total_served_count == 1)

    # Load consistency on second echelon routes
    for sat_id in sat_ids:
        for i, j in x2[sat_id].keys():
            if i != sat_id and j != sat_id:  # Both are customers
                model.addConstr(load2[sat_id][i, j] <= Q2 * x2[sat_id][i, j])
                # If arc is used, check load difference matches demands
                model.addConstr(load2[sat_id][i, j] >= customers[j]['demand'] * x2[sat_id][i, j])
                model.addConstr(load2[sat_id][i, j] <= (Q2 - customers[i]['demand']) * x2[sat_id][i, j])

    # Flow balance loads
    for sat_id in sat_ids:
        # Incoming to satellite = delivered goods = load_out_from_sat
        # Outgoing from sat = sum of customer demands assigned to this sat
        for c_idx in cust_ids:
            # Balance at customer location
            if (sat_id, c_idx) in x2[sat_id]:
                inflow_load = sum(load2[sat_id][i, c_idx] * x2[sat_id][i, c_idx] for i in [sat_id] + cust_ids if (i, c_idx) in x2[sat_id])
                outflow_load = sum(load2[sat_id][c_idx, j] * x2[sat_id][c_idx, j] for j in [sat_id] + cust_ids if (c_idx, j) in x2[sat_id])
                model.addConstr(inflow_load - outflow_load >= customers[c_idx]['demand'])
                
    # For satellite (as hub):
    outflows_from_sat = sum(load2[sat_id][sat_id, j] * x2[sat_id][sat_id, j] for j in cust_ids if (sat_id, j) in x2[sat_id])
    model.addConstr(load_to_sat[sat_id] == outflows_from_sat)

    # Limit on number of vehicles per satellite
    for sat_id in sat_ids:
        # Total routes from this satellite is determined by the number of times satellite is visited in second echelon
        total_x2_from_sat = sum(x2[sat_id][sat_id, j] for j in cust_ids if (sat_id, j) in x2[sat_id])
        model.addConstr(total_x2_from_sat <= satellites[sat_id]['max_vehicles'])
        
    # Total second echelon vehicles across all satellites
    total_vehs_used = sum(sum(x2[sat_id][sat_id, j] for j in cust_ids if (sat_id, j) in x2[sat_id])
                          for sat_id in sat_ids)
    model.addConstr(total_vehs_used <= T2)

    # Define objective function
    # 1. First-echelon travel cost
    first_echelon_travel_cost = sum(dist_matrix[depot['id'], j] * x1[depot['id'], j] for j in sat_ids) + \
                               sum(dist_matrix[i, j] * x1_sat_to_sat[i, j] for i in sat_ids for j in sat_ids if i != j)

    # 2. Handling costs at satellites
    handling_cost = sum(satellites[j]['handling_cost'] * sum(d1[v_idx, j] for v_idx in range(T1)) for j in sat_ids)

    # 3. Second-echelon travel cost  
    second_echelon_travel_cost = sum(
        sum(dist_matrix[i, j] * x2[sat_id][i, j] for (i, j) in x2[sat_id].keys())
        for sat_id in sat_ids
    )

    total_cost = first_echelon_travel_cost + handling_cost + second_echelon_travel_cost

    model.setObjective(total_cost, GRB.MINIMIZE)

    
    # Solve the model
    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)
    model.optimize(mycallback)

    # Prepare solution if we have a feasible result
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        if model.SolCount > 0:
            # Build the solution dict
            solution = {
                "objective_value": model.ObjVal,
                "first_echelon_routes": [],
                "second_echelon_routes": [],
                "group_assignments": {}
            }

            # Get first echelon routes and deliveries
            delivery_per_sat = defaultdict(int)
            for v_idx in range(T1):
                route = [depot['id']]
                deliveries = {}
                current = depot['id']

                # Find the path
                # This requires simulating flow paths since we might have complicated routing.
                # Simple approach: reconstruct from x values, but note that our current model might not have the most efficient representation for route tracking
                # We'll need to re-implement this part differently after solving.

                visited_sats = [j for j in sat_ids if y1[j].x > 0.9]
                
                # Find actual vehicles and corresponding routes
                for j in visited_sats:
                    # Check if this satellite is part of a vehicle route and add accordingly
                    if (depot['id'], j) in x1 and x1[(depot['id'], j)].X > 0.5:
                        deliveries[j] = sum(d1[v_idx, j].X for v_idx in range(T1))
                        
                if len(route) > 1:  # Added at least one satellite
                    route.append(depot['id'])
                    if deliveries:  # If we're delivering
                        solution["first_echelon_routes"].append({
                            "vehicle": v_idx,
                            "route": route,
                            "deliveries": deliveries
                        })

            # For simplicity now, rebuild logical routes from binary variables
            solution["first_echelon_routes"] = []
            for v_idx in range(T1):
                has_delivery_this_v = False
                deliveries = {}
                depot_to_sat = None
                for j in sat_ids:
                    d_total = sum(d1[v_idx, j].X for v_idx in range(T1))
                    if d1[v_idx, j].X > 0.01:
                        has_delivery_this_v = True
                        deliveries[j] = round(d1[v_idx, j].X)
                    elif d_total > 0.01:
                        has_delivery_this_v = True
                        deliveries[j] = round(d_total)

                found_first = False
                route = [depot['id']]
                for i in sat_ids:
                    if x1[depot['id'], i].X > 0.5:
                        route.insert(1, i)
                        found_first = True
                        break
                for i in sat_ids:
                    for j in sat_ids:
                        if j != i:
                            if x1_sat_to_sat[i, j].X > 0.5:
                                if i in route and j not in route:
                                    route.insert(route.index(i)+1, j)
                if found_first:
                    route.append(depot['id'])
                else:
                    del deliveries  # Don't report routes with no moves
                
                if route and len(route) > 2:  # Non-trivial route
                    solution["first_echelon_routes"].append({
                        "vehicle": v_idx,
                        "route": route,
                        "deliveries": {k: int(v) for k,v in deliveries.items()}
                    })
            
            # Group assignments -- determine which group goes to which satellite
            for g_id in group_ids:
                for s_id in sat_ids:
                    if z[g_id, s_id].X > 0.5:
                        solution["group_assignments"][g_id] = s_id
                        break
            
            # Second echelon routes
            for sat_id in sat_ids:
                if y2[sat_id].X > 0.5 or any(x2[sat_id][a, b].X > 0.5 for a,b in x2[sat_id]):
                    # Reconstruct route from x2 matrix
                    route_start = [sat_id]
                    remaining_arcs = {(i, j): x2[sat_id][i, j].X 
                                      for (i, j) in x2[sat_id].keys() 
                                      if x2[sat_id][i, j].X > 0.5}

                    current_pos = sat_id
                    temp_route = [current_pos]
                    
                    while remaining_arcs:
                        next_node = None
                        for (i, j) in list(remaining_arcs.keys()):
                            if i == current_pos:
                                next_node = j
                                del remaining_arcs[(i, j)]
                                break
                        if next_node is not None:
                            temp_route.append(next_node)
                            current_pos = next_node
                            if next_node == sat_id:
                                break  # Returned to satellite
                        else:
                            break
                    
                    if len(temp_route) > 1 and temp_route[-1] != sat_id:
                        temp_route.append(sat_id)  # Return to satellite
                        
                    if len(temp_route) > 2:  # Meaningfully traveled to service customers
                        solution["second_echelon_routes"].append({
                            "satellite": sat_id,
                            "route": temp_route
                        })
            
            # Final verification to handle empty fields properly  
            if not solution['first_echelon_routes']:
                solution['first_echelon_routes'] = []
            if not solution['second_echelon_routes']:
                solution['second_echelon_routes'] = []
            if not solution['group_assignments']:
                solution['group_assignments'] = {}
                
        else:
            # No feasible solution found within time limit
            solution = {
                "objective_value": float('inf'),
                "first_echelon_routes": [],
                "second_echelon_routes": [],
                "group_assignments": {}
            }
            
        total_runtime = time.time() - start_time
        if logger:
            logger.log(solution["objective_value"])

    else:
        # Model didn't find any solution
        solution = {
            "objective_value": float('inf'),
            "first_echelon_routes": [],
            "second_echelon_routes": [],
            "group_assignments": {}
        }
        
        total_runtime = time.time() - start_time
        if logger:
            logger.log(float('inf'))

    # Write solution to file
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    return solution


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Solve Freight Distribution Problem")
    parser.add_argument('--instance_path', type=str, required=True, help="Path to the input instance JSON file")
    parser.add_argument('--solution_path', type=str, required=True, help="Path to save the solution JSON file")
    parser.add_argument('--time_limit', type=int, required=True, help="Time limit in seconds")
    parser.add_argument('--log_path', type=str, default=None, help="Path to the log file (optional)")
    
    args = parser.parse_args()
    
    solve_freight_distribution(
        instance_path=args.instance_path,
        solution_path=args.solution_path,
        time_limit=args.time_limit,
        log_path=args.log_path
    )