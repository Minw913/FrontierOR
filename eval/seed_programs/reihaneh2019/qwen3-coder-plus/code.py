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
    model.setParam("Threads", 1)
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

    # Define service-pairing variables
    s = {}
    for c in customers:
        for i_idx in range(num_delivery_sites):
            site_id = delivery_sites[i_idx]
            for k in range(num_vehicles):
                s[c, site_id, k] = model.addVar(vtype=GRB.BINARY, name=f"serv_cust_{c}_site_{site_id}_tour_{k}")

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
        # Sum over k of whether the site is visited by tour k
        total_visits = gp.quicksum(x[f, site_id, k] for k in range(num_vehicles) 
                                  for f in [depot] + [j for j in delivery_sites if j != site_id])
        model.addConstr(total_visits <= 1, name=f"serve_site_{site_id}_at_most_once")
    
    # 4. Flow conservation constraints (for valid tours)
    for k in range(num_vehicles):
        # Conservation at depot: enter once, exit once
        model.addConstr(gp.quicksum(x[depot, to_node, k] for to_node in delivery_sites) == 
                       gp.quicksum(x[from_node, depot, k] for from_node in delivery_sites), 
                       name=f"tour_{k}_flow_balance_at_depot")
        
        # Entry equals exit and also both equal whether tour k exists (1 if used, 0 otherwise)
        model.addConstr(gp.quicksum(x[depot, to_node, k] for to_node in delivery_sites) <= 1,
                       name=f"tour_{k}_uses_at_most_one_vehicle")
        
        # Conservation at delivery sites
        for site in delivery_sites:
            model.addConstr(
                gp.quicksum(x[other, site, k] for other in [depot] + delivery_sites if other != site) ==
                gp.quicksum(x[site, other, k] for other in [depot] + delivery_sites if other != site),
                name=f"flow_conservation_{k}_{site}"
            )
    
    # 5. Assignment to visited site constraint
    # If a customer is assigned to a site, that site must be visited by some tour
    for c in customers:
        for i_idx in range(num_delivery_sites):
            site_id = delivery_sites[i_idx]
            # The site is visited by some tour if sum x[_, site, k] over any k > 0
            total_visited_by_any_tour = gp.quicksum(
                x[from_site, site_id, k] 
                for k in range(num_vehicles) 
                for from_site in [depot] + [j for j in delivery_sites if j != site_id]
            )
            # If z[c,site]=1 then total_visited_by_any_tour >= 1
            model.addConstr(z[c, site_id] <= total_visited_by_any_tour,
                           name=f"assign_implies_visit_{c}_{site_id}")

    # Constraints for customer-site-tour assignments
    for c in customers:
        for i_idx in range(num_delivery_sites):
            site_id = delivery_sites[i_idx]
            for k in range(num_vehicles):
                # s[c,site,k] <= z[c,site]
                model.addConstr(s[c, site_id, k] <= z[c, site_id], name=f"serve_implies_assign_{c}_{site_id}_{k}")
                
                # s[c,site,k] <= whether this site is visited by tour k
                site_visited_by_k = gp.quicksum(x[prev, site_id, k] for prev in [depot] + [other for other in delivery_sites if other != site_id])
                model.addConstr(s[c, site_id, k] <= site_visited_by_k, 
                               name=f"serve_k_iff_visited_{c}_{site_id}_{k}")

                # s[c,site,k] >= z[c,site] - (1 - visited_by_k)
                model.addConstr(s[c, site_id, k] >= z[c, site_id] - (1 - site_visited_by_k),
                               name=f"assign_and_visit_implies_serve_{c}_{site_id}_{k}")

    # Link cumulative quantities to actual assignments:
    for k in range(num_vehicles):
        for site in delivery_sites:
            # u[site, k] should reflect demand loaded for this tour up to site
            model.addConstr(
                u[site, k] >= gp.quicksum(customer_demand[c] * s[c, site, k] for c in customers),
                name=f"u_geq_assigned_demand_{site}_k_{k}"
            )
            # u values are bounded by vehicle capacity and only meaningful if site visited
            max_possible_here = sum(customer_demand[c] for c in customers)  # Upper bound
            total_visits_to_site = gp.quicksum(x[prev, site, k] for prev in [depot] + [other for other in delivery_sites if other != site])
            model.addConstr(u[site, k] <= vehicle_capacity * total_visits_to_site,
                           name=f"u_zero_unless_visited_{site}_k_{k}")

    # Ensure vehicle capacity is respected
    for k in range(num_vehicles):
        # Total demand assigned via tours with k should not exceed capacity
        demand_on_tour_k = gp.quicksum(customer_demand[c] * s[c, site, k] for c in customers for site in delivery_sites)
        model.addConstr(demand_on_tour_k <= vehicle_capacity, name=f"capacity_tour_{k}")

    # Ensure each customer uses exactly one tour
    for c in customers:
        model.addConstr(
            gp.quicksum(s[c, site, k] for site in delivery_sites for k in range(num_vehicles)) == 1,
            name=f"cust_uses_one_tour_{c}"
        )

    # Solve the model
    model.optimize()

    # Extract solution
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT and model.SolCount > 0:
        objective_value = model.objVal

        routes = {}
        assignments = {}
        customer_tours = {}

        # Build routes dict for each active tour
        for k in range(num_vehicles):
            tour_arcs = []
            tour_order = [depot]
            current = depot
            
            # Check which delivery sites are visited first in this tour
            visited_this_tour = set()
            while True:
                found_next = False
                for next_site in delivery_sites:
                    if next_site not in visited_this_tour and x.get((current, next_site, k), 0) and x[current, next_site, k].X > 0.5:
                        tour_arcs.append([current, next_site])
                        tour_order.append(next_site)
                        visited_this_tour.add(next_site)
                        current = next_site
                        found_next = True
                        break
                if not found_next:
                    # Go back to depot if we've visited sites but haven't returned yet
                    if current != depot and x.get((current, depot, k), 0) and x[current, depot, k].X > 0.5:
                        tour_arcs.append([current, depot])
                        tour_order.append(depot)
                        break
                    else:
                        # Check if there's another unvisited site accessible from depot
                        next_start_found = False
                        for start_site in delivery_sites:
                            if start_site not in visited_this_tour and x.get((depot, start_site, k), 0) and x[depot, start_site, k].X > 0.5:
                                next_start_found = True
                                # This might indicate a disconnected subtour, which shouldn't happen with proper constraints
                                # But if it does, continue processing
                                break
                        if not next_start_found:
                            break

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
                    
                    # Find which tour serves this customer through this site
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