import json
import argparse
import math
import time
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

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()
    solution = solve(data, args.time_limit, logger, start_time)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def solve(data, time_limit, logger, start_time):
    S = data['num_delivery_sites']
    C = data['num_customers']
    K = data['num_vehicles']

    # Estimate problem size for MIP
    num_vars_approx = (S + 1) * S * K + C * S + C * S * K + S * K

    # Try MIP for reasonable sizes
    if num_vars_approx <= 300000:
        try:
            solution = solve_mip(data, time_limit, logger, start_time)
            if solution is not None:
                return solution
        except Exception as e:
            print(f"MIP failed: {e}", file=sys.stderr)

    # Fallback to heuristic
    elapsed = time.time() - start_time
    remaining = max(1, time_limit - elapsed)
    return solve_heuristic(data, remaining, logger, start_time)


def build_solution_from_values(data, x_val, z_val, w_val):
    """Build solution dict from variable values."""
    S = data['num_delivery_sites']
    C = data['num_customers']
    K = data['num_vehicles']
    routing_cost = data['routing_cost_matrix']
    assignment_cost = data['assignment_cost_matrix']

    routes = {}
    assignments = {}
    customer_tours = {}

    # Extract routes
    for k in range(K):
        # Check if vehicle k is used
        used = False
        for j in range(1, S + 1):
            if x_val.get((0, j, k), 0) > 0.5:
                used = True
                break

        if used:
            sequence = [0]
            current = 0
            arcs = []
            visited_count = 0
            max_steps = S + 2

            while visited_count < max_steps:
                next_node = None
                for j in range(S + 1):
                    if j != current and x_val.get((current, j, k), 0) > 0.5:
                        next_node = j
                        break
                if next_node is None or next_node == 0:
                    arcs.append([current, 0])
                    sequence.append(0)
                    break
                arcs.append([current, next_node])
                sequence.append(next_node)
                current = next_node
                visited_count += 1

            if len(sequence) > 2:  # At least depot -> site -> depot
                routes[str(k)] = {
                    "arcs": arcs,
                    "sequence": sequence
                }

    # Extract assignments and customer tours
    for c in range(C):
        cust_id = str(data['customers'][c]['id'])
        for j in range(1, S + 1):
            if z_val.get((c, j), 0) > 0.5:
                assignments[cust_id] = data['delivery_sites'][j - 1]['id']
                # Find which tour serves this
                for k in range(K):
                    if w_val.get((c, j, k), 0) > 0.5:
                        customer_tours[cust_id] = {
                            "vehicle": k,
                            "site": data['delivery_sites'][j - 1]['id']
                        }
                        break
                break

    # Calculate objective
    total_routing = 0
    for k_str, route in routes.items():
        for arc in route['arcs']:
            total_routing += routing_cost[arc[0]][arc[1]]

    total_assignment = 0
    for c in range(C):
        for j in range(1, S + 1):
            if z_val.get((c, j), 0) > 0.5:
                total_assignment += assignment_cost[c][j - 1]
                break

    objective = total_routing + total_assignment

    return {
        "objective_value": objective,
        "routes": routes,
        "assignments": assignments,
        "customer_tours": customer_tours
    }


def solve_mip(data, time_limit, logger, start_time):
    import gurobipy as gp
    from gurobipy import GRB

    S = data['num_delivery_sites']
    C = data['num_customers']
    Q = data['vehicle_capacity']
    K = data['num_vehicles']
    routing_cost = data['routing_cost_matrix']
    assignment_cost = data['assignment_cost_matrix']
    demands = [data['customers'][c]['demand'] for c in range(C)]

    model = gp.Model("LRP")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit
    model.Params.OutputFlag = 0

    # Decision variables
    x = {}
    for i in range(S + 1):
        for j in range(S + 1):
            if i != j:
                for k in range(K):
                    x[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")

    z = {}
    for c in range(C):
        for j in range(1, S + 1):
            z[c, j] = model.addVar(vtype=GRB.BINARY, name=f"z_{c}_{j}")

    w = {}
    for c in range(C):
        for j in range(1, S + 1):
            for k in range(K):
                w[c, j, k] = model.addVar(vtype=GRB.BINARY, name=f"w_{c}_{j}_{k}")

    u = {}
    for j in range(1, S + 1):
        for k in range(K):
            u[j, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=S, name=f"u_{j}_{k}")

    model.update()

    # Objective
    obj = gp.LinExpr()
    for i in range(S + 1):
        for j in range(S + 1):
            if i != j:
                for k in range(K):
                    obj += routing_cost[i][j] * x[i, j, k]
    for c in range(C):
        for j in range(1, S + 1):
            obj += assignment_cost[c][j - 1] * z[c, j]
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints
    # Each vehicle leaves depot at most once
    for k in range(K):
        model.addConstr(gp.quicksum(x[0, j, k] for j in range(1, S + 1)) <= 1)

    # Flow conservation at each site for each vehicle
    for k in range(K):
        for j in range(1, S + 1):
            model.addConstr(
                gp.quicksum(x[i, j, k] for i in range(S + 1) if i != j) ==
                gp.quicksum(x[j, i, k] for i in range(S + 1) if i != j)
            )

    # Each site visited by at most one vehicle
    for j in range(1, S + 1):
        model.addConstr(
            gp.quicksum(x[i, j, k] for i in range(S + 1) if i != j for k in range(K)) <= 1
        )

    # Each customer assigned to exactly one site
    for c in range(C):
        model.addConstr(gp.quicksum(z[c, j] for j in range(1, S + 1)) == 1)

    # Can only assign to visited sites
    for c in range(C):
        for j in range(1, S + 1):
            model.addConstr(
                z[c, j] <= gp.quicksum(x[i, j, k] for i in range(S + 1) if i != j for k in range(K))
            )

    # Service pairing: w <= z
    for c in range(C):
        for j in range(1, S + 1):
            for k in range(K):
                model.addConstr(w[c, j, k] <= z[c, j])

    # Service pairing: w <= visited(j, k)
    for c in range(C):
        for j in range(1, S + 1):
            for k in range(K):
                model.addConstr(
                    w[c, j, k] <= gp.quicksum(x[i, j, k] for i in range(S + 1) if i != j)
                )

    # Each customer covered exactly once via w
    for c in range(C):
        model.addConstr(
            gp.quicksum(w[c, j, k] for j in range(1, S + 1) for k in range(K)) == 1
        )

    # Capacity constraint
    for k in range(K):
        model.addConstr(
            gp.quicksum(demands[c] * w[c, j, k] for c in range(C) for j in range(1, S + 1)) <= Q
        )

    # MTZ subtour elimination
    for k in range(K):
        for i in range(1, S + 1):
            for j in range(1, S + 1):
                if i != j:
                    model.addConstr(u[i, k] - u[j, k] + S * x[i, j, k] <= S - 1)

    # u >= 1 if site visited
    for j in range(1, S + 1):
        for k in range(K):
            model.addConstr(
                u[j, k] >= gp.quicksum(x[i, j, k] for i in range(S + 1) if i != j)
            )

    # Symmetry breaking
    for k in range(1, K):
        model.addConstr(
            gp.quicksum(x[0, j, k] for j in range(1, S + 1)) <=
            gp.quicksum(x[0, j, k - 1] for j in range(1, S + 1))
        )

    model.update()

    # Callback for logging incumbents
    best_solution = [None]
    best_obj = [float('inf')]

    def my_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0] - 1e-6:
                best_obj[0] = obj_val
                # Extract variable values
                x_val = {}
                for key, var in x.items():
                    x_val[key] = model.cbGetSolution(var)
                z_val = {}
                for key, var in z.items():
                    z_val[key] = model.cbGetSolution(var)
                w_val = {}
                for key, var in w.items():
                    w_val[key] = model.cbGetSolution(var)

                sol = build_solution_from_values(data, x_val, z_val, w_val)
                best_solution[0] = sol
                if logger:
                    logger.log_solution(obj_val, sol)

    model.optimize(my_callback)

    if best_solution[0] is not None:
        return best_solution[0]

    # Try to extract from model directly
    if model.SolCount > 0:
        x_val = {key: var.X for key, var in x.items()}
        z_val = {key: var.X for key, var in z.items()}
        w_val = {key: var.X for key, var in w.items()}
        sol = build_solution_from_values(data, x_val, z_val, w_val)
        if logger:
            logger.log_solution(sol["objective_value"], sol)
        return sol

    return None


def solve_heuristic(data, time_limit, logger, start_time):
    """Constructive heuristic with local search improvement."""
    S = data['num_delivery_sites']
    C = data['num_customers']
    Q = data['vehicle_capacity']
    K = data['num_vehicles']
    routing_cost = data['routing_cost_matrix']
    assignment_cost = data['assignment_cost_matrix']
    demands = [data['customers'][c]['demand'] for c in range(C)]

    best_solution = None
    best_obj = float('inf')

    # Strategy 1: Assign to nearest site, then route
    sol1 = construct_solution_nearest(data)
    if sol1 is not None:
        obj1 = sol1["objective_value"]
        if obj1 < best_obj:
            best_obj = obj1
            best_solution = sol1
            if logger:
                logger.log_solution(obj1, sol1)

    # Strategy 2: Try different site selections
    if time.time() - start_time < time_limit * 0.5:
        sol2 = construct_solution_greedy(data)
        if sol2 is not None:
            obj2 = sol2["objective_value"]
            if obj2 < best_obj:
                best_obj = obj2
                best_solution = sol2
                if logger:
                    logger.log_solution(obj2, sol2)

    # Strategy 3: Try with reduced site set
    if time.time() - start_time < time_limit * 0.7:
        sol3 = construct_solution_reduced(data)
        if sol3 is not None:
            obj3 = sol3["objective_value"]
            if obj3 < best_obj:
                best_obj = obj3
                best_solution = sol3
                if logger:
                    logger.log_solution(obj3, sol3)

    # Local search improvement
    if best_solution is not None and time.time() - start_time < time_limit * 0.8:
        improved = local_search(data, best_solution, time_limit, start_time)
        if improved is not None and improved["objective_value"] < best_obj:
            best_obj = improved["objective_value"]
            best_solution = improved
            if logger:
                logger.log_solution(best_obj, best_solution)

    if best_solution is None:
        best_solution = construct_solution_nearest(data)

    return best_solution


def construct_solution_nearest(data):
    """Assign each customer to nearest site, then route."""
    S = data['num_delivery_sites']
    C = data['num_customers']
    Q = data['vehicle_capacity']
    K = data['num_vehicles']
    routing_cost = data['routing_cost_matrix']
    assignment_cost = data['assignment_cost_matrix']
    demands = [data['customers'][c]['demand'] for c in range(C)]

    # Assign each customer to nearest site
    assign = []
    for c in range(C):
        best_j = 0
        best_cost = float('inf')
        for j in range(S):
            if assignment_cost[c][j] < best_cost:
                best_cost = assignment_cost[c][j]
                best_j = j
        assign.append(best_j + 1)  # 1-indexed

    return build_routes_and_solution(data, assign)


def construct_solution_greedy(data):
    """Greedy: consider opening cost vs assignment savings."""
    S = data['num_delivery_sites']
    C = data['num_customers']
    Q = data['vehicle_capacity']
    K = data['num_vehicles']
    routing_cost = data['routing_cost_matrix']
    assignment_cost = data['assignment_cost_matrix']
    demands = [data['customers'][c]['demand'] for c in range(C)]

    # Start with all sites, compute assignment
    # Then try removing sites that are expensive to route to
    assign = []
    for c in range(C):
        best_j = 0
        best_cost = float('inf')
        for j in range(S):
            if assignment_cost[c][j] < best_cost:
                best_cost = assignment_cost[c][j]
                best_j = j
        assign.append(best_j + 1)

    # Try to improve by considering second-best assignments
    # For sites with high routing cost, try reassigning customers
    opened = set(assign)

    # Compute routing cost for each site (distance from depot)
    site_routing_cost = {}
    for j in range(1, S + 1):
        site_routing_cost[j] = routing_cost[0][j] + routing_cost[j][0]

    # Try closing expensive sites
    sorted_sites = sorted(opened, key=lambda j: -site_routing_cost.get(j, 0))

    for j in sorted_sites:
        if len(opened) <= 1:
            break
        # Try closing site j
        customers_at_j = [c for c in range(C) if assign[c] == j]
        if not customers_at_j:
            opened.discard(j)
            continue

        # Check if we can reassign these customers
        can_reassign = True
        new_assign = assign[:]
        for c in customers_at_j:
            best_alt = None
            best_alt_cost = float('inf')
            for jj in opened:
                if jj != j:
                    if assignment_cost[c][jj - 1] < best_alt_cost:
                        best_alt_cost = assignment_cost[c][jj - 1]
                        best_alt = jj
            if best_alt is None:
                can_reassign = False
                break
            new_assign[c] = best_alt

        if can_reassign:
            # Check if closing saves cost
            savings = site_routing_cost.get(j, 0)
            extra_assignment = sum(
                assignment_cost[c][new_assign[c] - 1] - assignment_cost[c][assign[c] - 1]
                for c in customers_at_j
            )
            if savings > extra_assignment:
                assign = new_assign
                opened.discard(j)

    return build_routes_and_solution(data, assign)


def construct_solution_reduced(data):
    """Try opening fewer sites to reduce routing cost."""
    S = data['num_delivery_sites']
    C = data['num_customers']
    Q = data['vehicle_capacity']
    K = data['num_vehicles']
    routing_cost = data['routing_cost_matrix']
    assignment_cost = data['assignment_cost_matrix']
    demands = [data['customers'][c]['demand'] for c in range(C)]

    # Score each site by total assignment cost if it's the only option
    site_scores = []
    for j in range(S):
        total_assign = sum(assignment_cost[c][j] for c in range(C))
        site_scores.append((total_assign, j + 1))

    # Try opening top-k sites for various k
    best_sol = None
    best_obj = float('inf')

    site_scores.sort()
    for num_sites in range(max(1, S // 3), S + 1):
        candidate_sites = set(s[1] for s in site_scores[:num_sites])
        assign = []
        for c in range(C):
            best_j = None
            best_cost = float('inf')
            for j in candidate_sites:
                if assignment_cost[c][j - 1] < best_cost:
                    best_cost = assignment_cost[c][j - 1]
                    best_j = j
            assign.append(best_j)

        sol = build_routes_and_solution(data, assign)
        if sol is not None and sol["objective_value"] < best_obj:
            best_obj = sol["objective_value"]
            best_sol = sol

    return best_sol


def build_routes_and_solution(data, assign):
    """Given assignments, build routes and compute solution."""
    S = data['num_delivery_sites']
    C = data['num_customers']
    Q = data['vehicle_capacity']
    K = data['num_vehicles']
    routing_cost = data['routing_cost_matrix']
    assignment_cost = data['assignment_cost_matrix']
    demands = [data['customers'][c]['demand'] for c in range(C)]

    # Group customers by site
    site_customers = {j: [] for j in range(1, S + 1)}
    for c in range(C):
        site_customers[assign[c]].append(c)

    # Compute demand at each site
    site_demand = {}
    for j in range(1, S + 1):
        site_demand[j] = sum(demands[c] for c in site_customers[j])

    # Sites that need to be visited
    opened_sites = [j for j in range(1, S + 1) if site_customers[j]]

    if not opened_sites:
        return None

    # Build routes using nearest neighbor with capacity
    routes = {}
    customer_tours_map = {}
    remaining = set(opened_sites)
    vehicle_id = 0

    while remaining and vehicle_id < K:
        current = 0
        tour_sites = []
        tour_demand = 0

        while remaining:
            # Find nearest feasible site
            best_next = None
            best_dist = float('inf')
            for j in remaining:
                if tour_demand + site_demand[j] <= Q:
                    if routing_cost[current][j] < best_dist:
                        best_dist = routing_cost[current][j]
                        best_next = j

            if best_next is None:
                break

            tour_sites.append(best_next)
            tour_demand += site_demand[best_next]
            remaining.remove(best_next)
            current = best_next

        if tour_sites:
            # Apply 2-opt improvement
            tour_sites = two_opt(routing_cost, tour_sites)

            sequence = [0] + tour_sites + [0]
            arcs = [[sequence[i], sequence[i + 1]] for i in range(len(sequence) - 1)]
            routes[str(vehicle_id)] = {
                "arcs": arcs,
                "sequence": sequence
            }

            for j in tour_sites:
                for c in site_customers[j]:
                    cust_id = str(data['customers'][c]['id'])
                    customer_tours_map[cust_id] = {
                        "vehicle": vehicle_id,
                        "site": data['delivery_sites'][j - 1]['id']
                    }

            vehicle_id += 1
        else:
            break

    # Handle remaining sites
    while remaining and vehicle_id < K:
        j = min(remaining, key=lambda jj: site_demand[jj])
        remaining.remove(j)
        sequence = [0, j, 0]
        arcs = [[0, j], [j, 0]]
        routes[str(vehicle_id)] = {
            "arcs": arcs,
            "sequence": sequence
        }
        for c in site_customers[j]:
            cust_id = str(data['customers'][c]['id'])
            customer_tours_map[cust_id] = {
                "vehicle": vehicle_id,
                "site": data['delivery_sites'][j - 1]['id']
            }
        vehicle_id += 1

    if remaining:
        return None  # Can't serve all sites

    # Build assignments
    assignments = {}
    for c in range(C):
        cust_id = str(data['customers'][c]['id'])
        assignments[cust_id] = data['delivery_sites'][assign[c] - 1]['id']

    # Calculate objective
    total_routing = sum(
        routing_cost[arc[0]][arc[1]]
        for route in routes.values()
        for arc in route['arcs']
    )
    total_assignment = sum(
        assignment_cost[c][assign[c] - 1] for c in range(C)
    )

    return {
        "objective_value": total_routing + total_assignment,
        "routes": routes,
        "assignments": assignments,
        "customer_tours": customer_tours_map
    }


def two_opt(routing_cost, tour_sites):
    """Apply 2-opt improvement to a sequence of sites."""
    if len(tour_sites) <= 2:
        return tour_sites

    improved = True
    best = tour_sites[:]

    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                # Compute cost of current edges
                node_before_i = 0 if i == 0 else best[i - 1]
                node_i = best[i]
                node_j = best[j]
                node_after_j = 0 if j == len(best) - 1 else best[j + 1]

                old_cost = routing_cost[node_before_i][node_i] + routing_cost[node_j][node_after_j]
                new_cost = routing_cost[node_before_i][node_j] + routing_cost[node_i][node_after_j]

                if new_cost < old_cost - 1e-10:
                    best[i:j + 1] = best[i:j + 1][::-1]
                    improved = True

    return best


def local_search(data, solution, time_limit, start_time):
    """Try to improve solution by reassigning customers."""
    S = data['num_delivery_sites']
    C = data['num_customers']
    Q = data['vehicle_capacity']
    K = data['num_vehicles']
    routing_cost = data['routing_cost_matrix']
    assignment_cost = data['assignment_cost_matrix']
    demands = [data['customers'][c]['demand'] for c in range(C)]

    # Reconstruct assignment from solution
    assign = [0] * C
    for c in range(C):
        cust_id = str(data['customers'][c]['id'])
        if cust_id in solution['assignments']:
            site_id = solution['assignments'][cust_id]
            # Find site index
            for j in range(S):
                if data['delivery_sites'][j]['id'] == site_id:
                    assign[c] = j + 1
                    break

    best_assign = assign[:]
    best_obj = solution['objective_value']
    improved = True

    while improved and time.time() - start_time < time_limit * 0.95:
        improved = False
        for c in range(C):
            current_site = best_assign[c]
            current_cost = assignment_cost[c][current_site - 1]

            for j in range(1, S + 1):
                if j == current_site:
                    continue
                new_cost = assignment_cost[c][j - 1]
                if new_cost < current_cost - 1e-10:
                    # Try reassigning
                    trial_assign = best_assign[:]
                    trial_assign[c] = j
                    sol = build_routes_and_solution(data, trial_assign)
                    if sol is not None and sol['objective_value'] < best_obj:
                        best_obj = sol['objective_value']
                        best_assign = trial_assign
                        improved = True
                        break

    if best_obj < solution['objective_value']:
        return build_routes_and_solution(data, best_assign)

    return solution


if __name__ == '__main__':
    main()