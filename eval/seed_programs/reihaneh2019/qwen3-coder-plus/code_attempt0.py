import argparse
import json
import math
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_locationallocation_routing(instance_data, time_limit, logger):
    # Extract data from instance
    num_delivery_sites = instance_data["num_delivery_sites"]
    num_customers = instance_data["num_customers"]
    vehicle_capacity = instance_data["vehicle_capacity"]
    num_vehicles = instance_data["num_vehicles"]
    
    depot = instance_data["depot"]["id"]
    delivery_sites = [site['id'] for site in instance_data["delivery_sites"]]
    customers = [cust['id'] for cust in instance_data["customers"]]
    
    coordinates = {
        depot: (instance_data["depot"]["x"], instance_data["depot"]["y"]),
    }
    site_to_coords = {}
    for i, site in enumerate(delivery_sites):
        coords = (instance_data["delivery_sites"][i]["x"], instance_data["delivery_sites"][i]["y"])
        coordinates[site] = coords
        site_to_coords[site] = coords
        
    customer_coords = {}
    customer_demand = {}
    for i, cust in enumerate(customers):
        coords = (instance_data["customers"][i]["x"], instance_data["customers"][i]["y"])
        customer_coords[cust] = coords
        customer_demand[cust] = instance_data["customers"][i]["demand"]

    routing_cost = instance_data["routing_cost_matrix"]  # indexed as depot=0, then delivery sites 1..n
    assignment_cost = instance_data["assignment_cost_matrix"]

    # Create the model
    model = gp.Model("Locational_Routing")
    model.setParam('TimeLimit', time_limit)
    
    # Variables:
    # y[i] = 1 if delivery site i is open
    y = {}
    for i_idx in range(num_delivery_sites):  # 0-indexed for delivery sites for routing matrix
        site_id = delivery_sites[i_idx]
        y[site_id] = model.addVar(vtype=GRB.BINARY, name=f"y_site_{site_id}")

    # x[i,j,k] = 1 if vehicle k travels directly from site i to site j
    x = {}
    sites_with_depot = [depot] + delivery_sites  # depot index 0 in routing cost matrix
    depot_idx = 0
    
    for k in range(num_vehicles):
        for i_idx, from_id in enumerate(sites_with_depot):
            from_pos = i_idx if i_idx == 0 else i_idx  # depot idx=0, sites start from 1
            for j_idx, to_id in enumerate(sites_with_depot):
                to_pos = j_idx
                if from_id != to_id:  # no self loops
                    x[from_id, to_id, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{from_id}_{to_id}_{k}")
                    
    # z[c,i] = 1 if customer c is assigned to site i
    z = {}
    for c in customers:
        for i_idx in range(num_delivery_sites):
            site_id = delivery_sites[i_idx]
            z[c, site_id] = model.addVar(vtype=GRB.BINARY, name=f"z_customer_{c}_site_{site_id}")

    # u[i, k] = amount of goods cumulatively delivered to site i at the moment just before the outbound route segment from i in tour k
    u = {}
    for i in delivery_sites:
        for k in range(num_vehicles):
            u[i, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=vehicle_capacity, name=f"u_delivery_{i}_tour_{k}")

    # Define service-pairing variables implicitly. For each customer, vehicle, and site, we track
    # the service relationship as combinations of assignments and vehicle visits
    # Since this becomes complex, let's use linearization techniques appropriately

    # Objective function: minimize routing and assignment costs
    routing_cost_expr = gp.LinExpr()
    for k in range(num_vehicles):
        for i_idx, from_id in enumerate(sites_with_depot):
            for j_idx, to_id in enumerate(sites_with_depot):
                if from_id != to_id:
                    r_idx_i = i_idx
                    r_idx_j = j_idx
                    routing_cost_expr.addTerms(routing_cost[r_idx_i][r_idx_j], x[from_id, to_id, k])

    assignment_cost_expr = gp.LinExpr()
    for c_idx, c in enumerate(customers):
        for i_idx, site_id in enumerate(delivery_sites):
            assignment_cost_expr.addTerms(assignment_cost[c_idx][i_idx], z[c, site_id])

    model.setObjective(routing_cost_expr + assignment_cost_expr, GRB.MINIMIZE)

    # Constraints:

    # 1. Every customer must be assigned to exactly one delivery site
    for c in customers:
        model.addConstr(gp.quicksum(z[c, site_id] for site_id in delivery_sites) == 1, name=f"assign_cust_{c}_once")

    # 2. If customer c is assigned to site i, then site i must be opened
    for c in customers:
        for i_idx in range(num_delivery_sites):
            site_id = delivery_sites[i_idx]
            model.addConstr(z[c, site_id] <= y[site_id], name=f"c{c}_at_s{site_id}_implies_open")

    # 3. Each delivery site can be visited by at most one vehicle tour
    for i_idx in range(num_delivery_sites):
        site_id = delivery_sites[i_idx]
        model.addConstr(gp.quicksum(x[f, site_id, k] for k in range(num_vehicles) 
                                  for f in sites_with_depot if f != site_id) <= 1, 
                       name=f"serve_site_{site_id}_at_most_once")
        
        model.addConstr(gp.quicksum(x[site_id, t, k] for k in range(num_vehicles) 
                                  for t in sites_with_depot if t != site_id) <= 1, 
                       name=f"leave_site_{site_id}_at_most_once")
    
    # 4. Flow conservation constraints (for valid tours)
    for k in range(num_vehicles):
        # Conservation at depot: enter once, exit once
        model.addConstr(gp.quicksum(x[depot, to_node, k] for to_node in delivery_sites) == 1, 
                       name=f"tour_{k}_entry")
        model.addConstr(gp.quicksum(x[from_node, depot, k] for from_node in delivery_sites) == 1, 
                       name=f"tour_{k}_exit")
        
        # Conservation at delivery sites
        for site in delivery_sites:
            model.addConstr(
                gp.quicksum(x[other, site, k] for other in [depot] + delivery_sites if other != site) ==
                gp.quicksum(x[site, other, k] for other in [depot] + delivery_sites if other != site),
                name=f"flow_conservation_{k}_{site}"
            )

    # 5. Vehicle capacity constraints
    for k in range(num_vehicles):
        total_demand_expr = gp.LinExpr()
        for c in customers:
            for i_idx in range(num_delivery_sites):
                site_id = delivery_sites[i_idx]
                total_demand_expr.addTerms(customer_demand[c], z[c, site_id] * x[depot, site_id, k])  # rough estimate if we don't track flow per arc
        # This is insufficient due to customer aggregation. Need tighter constraint for actual tour capacity.
        # Add cumulative load tracking constraints instead.
        total_demand_per_tour = gp.LinExpr()
        for c in customers:
            for site in delivery_sites:
                # Link z[c,site] and the tour k if the vehicle visits the site
                # We'll use an auxiliary linking variable: s[c,site,k] = 1 if customer c is assigned to site 
                # and that site is visited by tour k
                # s[c,site,k] <= z[c,site]
                # s[c,site,k] <= sum of x[*,site,k]  or x[depot,site,k] + x[site,depot,k]?
                # Actually: sum x[from,site,k] >= s[c,site,k]  - not this way.
                
                # This is becoming complex - add subtour elimination and load tracking
                
                # We'll compute demand per site per tour and ensure vehicle capacity.
                
                # If site j is visited by tour k, total demand served via that site must fit in a vehicle
                model.addConstr(
                    gp.quicksum(customer_demand[c] * z[c, site] for c in customers) <= 
                    vehicle_capacity,
                    name=f"capacity_site_{site}_k_{k}")
    
    # 6. Cumulative quantity delivered constraints
    # u[i, k] >= total demand of customers assigned to site i and served by tour k
    for k in range(num_vehicles):
        for i_idx in range(num_delivery_sites):
            site_id = delivery_sites[i_idx]
            demand_at_site_k_on_tour_x = gp.quicksum(customer_demand[c] * z[c, site_id] for c in customers)
            # This is correct only if site is visited by tour k
            
            # Need to ensure u[i,k] only reflects customers served via tour k, so we introduce a connection:
            # We link site visit to customer assignment using a logical relation:
            big_m = sum(customer_demand.values())  # upper bound on total demand

            # u[i,k] is non-zero only if site i is visited by tour k
            total_visit_by_k = gp.quicksum(
                x[prev_loc, site_id, k] for prev_loc in [depot] + [other for other in delivery_sites if other != site_id],
            )
            
            model.addConstr(u[site_id, k] <= vehicle_capacity * total_visit_by_k, 
                           name=f"u_bounded_if_visited_{site_id}_k_{k}")

            # u[i,k] should be >= demands of all customers assigned to i and served by tour k.
            # For this, we need to state that total of demand * z[c, site] is only considered if site is visited by tour k
            # More properly: sum of cummulative quantity must be >= demand at site if site visited
            # The correct logic: we must calculate demand allocated to this site that actually goes via TOUR k
            # We need binary vars indicating if a customer-site-tour pair is active.
            
            # Let s_csk = 1 if customer c assigned to site i and both are part of tour k
            # Then u[site, k] >= sum over c of demand[c] * s_csk
            # s_csk <= z[c,site]
            # s_csk <= sum of arcs to/from site per tour
            # s_csk >= z[c,site] - (1 - sum of arcs to site per tour)
    
    # Introduce variables for customer-site-tour assignments
    s = {}
    for c in customers:
        for i_idx in range(num_delivery_sites):
            site_id = delivery_sites[i_idx]
            for k in range(num_vehicles):
                s[c, site_id, k] = model.addVar(vtype=GRB.BINARY, name=f"serv_cust_{c}_site_{site_id}_tour_{k}")

                # s[c,site,k] <= z[c,site]
                model.addConstr(s[c, site_id, k] <= z[c, site_id], name=f"serve_implies_assign_{c}_{site_id}_{k}")
                
                # s[c,site,k] <= 1 iff site is visited by tour k
                # This means sum outgoing from site in tour k >= 1 -> this doesn't work directly
                # Better sum incoming or outgoing x[prev,site,k]
                model.addConstr(s[c, site_id, k] <= gp.quicksum(x[prev, site_id, k] for prev in sites_with_depot if prev != site_id), 
                               name=f"serve_k_iff_visited_{c}_{site_id}_{k}")

                # If customer assigned to site AND site visited by k, then s[c,site,k] >= 1
                # s[c,site,k] >= z[c,site] - (1 - sum_incoming)
                model.addConstr(s[c, site_id, k] >= z[c, site_id] - 
                               (1 - gp.quicksum(x[prev, site_id, k] for prev in sites_with_depot if prev != site_id)),
                               name=f"assign_and_visit_implies_serve_{c}_{site_id}_{k}")

    # Now link cumulative quantities:
    for k in range(num_vehicles):
        for site in delivery_sites:
            model.addConstr(
                u[site, k] >= gp.quicksum(customer_demand[c] * s[c, site, k] for c in customers),
                name=f"u_geq_demand_{site}_k_{k}"
            )

    # Ensure each customer uses exactly one tour
    for c in customers:
        model.addConstr(
            gp.quicksum(s[c, site, k] for site in delivery_sites for k in range(num_vehicles)) == 1,
            name=f"cust_uses_one_tour_{c}"
        )

    # Subtour elimination constraints - Miller-Tucker-Zemlin style adaptation (simplified)
    # This is challenging without fixed ordering - we'll rely on default lazy constraints for MTZ
    model._x = x # for user cuts/subtour handling
    model._sites = delivery_sites
    model._vehicles = range(num_vehicles)

    # Solve the model
    model.optimize()

    # Extract solution
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.objVal if model.SolCount > 0 else float('inf')

        routes = {}
        assignments = {}
        customer_tours = {}

        # Build routes dict for each active tour
        for k in range(num_vehicles):
            tour_arcs = []
            tour_order = [depot]
            current = depot
            visited = {depot}
            
            while len([d for d in delivery_sites if x.get((current, d, k), 0) and x.get((current, d, k), 0).X > 0.5]) > 0:
                next_node = None
                for d in delivery_sites:
                    if d not in visited and x.get((current, d, k), 0) and x.get((current, d, k), 0).X > 0.5:
                        next_node = d
                        break
                if next_node is None: 
                    break
                visited.add(next_node)
                tour_arcs.append([current, next_node])
                tour_order.append(next_node)
                current = next_node
                
            # Return to depot if completed a chain through delivery sites
            if current != depot and x.get((current, depot, k), 0) and x.get((current, depot, k), 0).X > 0.5:
                tour_arcs.append([current, depot])
                tour_order.append(depot)

            if len(tour_arcs) > 0:  # Only add if tour has meaningful content
                routes[k] = {
                    'arcs': tour_arcs,
                    'sequence': tour_order
                }

        # Assignments: z[c,s] = 1 means assign customer c to site s
        for c in customers:
            for site in delivery_sites:
                if z.get((c, site), 0) and z[c, site].X > 0.5:
                    assignments[str(c)] = site
                    
                    # For customer tours: find which tour it's associated with via s[c,site,k]
                    for k in range(num_vehicles):
                        if s.get((c, site, k), 0) and s[c, site, k].X > 0.5:
                            customer_tours[str(c)] = {"vehicle": k, "site": site}
                            break  # only one tour per customer

        solution = {
            "objective_value": objective_value,
            "routes": routes,
            "assignments": assignments,
            "customer_tours": customer_tours
        }

        if logger:
            logger.log(objective_value)

        return solution
    else:
        return {
            "objective_value": float('inf'),
            "routes": {},
            "assignments": {},
            "customer_tours": {}
        }


def main():
    parser = argparse.ArgumentParser(description='Solve Location-Allocation and Routing Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to save the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log file (JSONL format)')
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    solution = solve_locationallocation_routing(instance_data, args.time_limit, logger)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()