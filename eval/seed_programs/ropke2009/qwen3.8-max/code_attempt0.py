import json
import argparse
import time
import math
import random
import sys
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, default=3600)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    time_limit = args.time_limit

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_requests = instance['num_requests']
    num_vehicles = instance['num_vehicles']
    vehicle_capacity = instance['vehicle_capacity']
    planning_horizon = instance['planning_horizon']

    nodes = instance['nodes']
    n = len(nodes)

    # Distance matrix (routing cost = Euclidean distance)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dx = nodes[i]['x'] - nodes[j]['x']
            dy = nodes[i]['y'] - nodes[j]['y']
            dist[i][j] = math.sqrt(dx * dx + dy * dy)

    # Travel time matrix (includes service time at origin)
    travel_time = [[0.0] * n for _ in range(n)]
    for i in range(n):
        st = nodes[i].get('service_time', 0)
        for j in range(n):
            travel_time[i][j] = dist[i][j] + st

    # Identify depot nodes
    depot_origin = None
    depot_dest = None
    for node in nodes:
        if node['type'] == 'depot_origin':
            depot_origin = node['id']
        elif node['type'] == 'depot_destination':
            depot_dest = node['id']

    # Request info
    requests = instance['requests']
    pickup_to_delivery = {}
    delivery_to_pickup = {}
    for req in requests:
        p = req['pickup_node']
        d = req['delivery_node']
        pickup_to_delivery[p] = d
        delivery_to_pickup[d] = p

    def evaluate_route(route):
        """Returns (feasible, cost)"""
        if len(route) < 2:
            return True, 0.0
        if len(route) == 2:
            return True, dist[route[0]][route[1]]

        cost = 0.0
        current_time = nodes[route[0]]['time_window'][0]
        current_load = 0

        for idx in range(1, len(route)):
            prev = route[idx - 1]
            curr = route[idx]

            cost += dist[prev][curr]
            arrival = current_time + travel_time[prev][curr]

            tw = nodes[curr]['time_window']
            if arrival > tw[1] + 1e-6:
                return False, float('inf')

            service_start = max(arrival, tw[0])

            current_load += nodes[curr]['demand']
            if current_load > vehicle_capacity + 1e-6 or current_load < -1e-6:
                return False, float('inf')

            current_time = service_start

        return True, cost

    def route_cost(route):
        f, c = evaluate_route(route)
        return c if f else float('inf')

    def total_cost(routes):
        return sum(route_cost(r) for r in routes)

    def format_solution(routes, obj):
        return {
            "objective_value": obj,
            "routes": [
                {"vehicle": v, "nodes": list(route)}
                for v, route in enumerate(routes)
            ]
        }

    # ============ GUROBI MIP FOR SMALL INSTANCES ============
    def solve_gurobi():
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError:
            return None

        remaining = time_limit - (time.time() - start_time)
        if remaining <= 5:
            return None

        try:
            model = gp.Model("pdptw")
            model.setParam('OutputFlag', 0)
            model.setParam('Seed', 0)
            model.setParam('MIPGap', 1e-4)
            model.setParam('NumericFocus', 0)
            model.setParam('Threads', 1)
            model.setParam('TimeLimit', min(remaining * 0.6, 300))

            # Sets
            all_nodes = list(range(n))
            pickup_nodes_list = [req['pickup_node'] for req in requests]
            delivery_nodes_list = [req['delivery_node'] for req in requests]
            vehicles = list(range(num_vehicles))

            # Decision variables
            x = {}
            for i in all_nodes:
                for j in all_nodes:
                    if i == j:
                        continue
                    for k in vehicles:
                        x[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")

            t = {}
            for i in all_nodes:
                for k in vehicles:
                    lb = nodes[i]['time_window'][0]
                    ub = nodes[i]['time_window'][1]
                    t[i, k] = model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f"t_{i}_{k}")

            # Objective: minimize total routing cost
            obj_expr = gp.quicksum(dist[i][j] * x[i, j, k]
                                   for i in all_nodes for j in all_nodes
                                   if i != j for k in vehicles)
            model.setObjective(obj_expr, GRB.MINIMIZE)

            # Constraints
            # 1. Each pickup visited exactly once
            for p in pickup_nodes_list:
                model.addConstr(
                    gp.quicksum(x[p, j, k] for j in all_nodes if j != p for k in vehicles) == 1
                )

            # 2. Each delivery visited exactly once
            for d in delivery_nodes_list:
                model.addConstr(
                    gp.quicksum(x[d, j, k] for j in all_nodes if j != d for k in vehicles) == 1
                )

            # 3. Flow conservation
            for k in vehicles:
                for i in all_nodes:
                    model.addConstr(
                        gp.quicksum(x[i, j, k] for j in all_nodes if j != i) ==
                        gp.quicksum(x[j, i, k] for j in all_nodes if j != i)
                    )

            # 4. Each vehicle leaves depot_origin exactly once
            for k in vehicles:
                model.addConstr(
                    gp.quicksum(x[depot_origin, j, k] for j in all_nodes if j != depot_origin) == 1
                )

            # 5. Each vehicle enters depot_dest exactly once
            for k in vehicles:
                model.addConstr(
                    gp.quicksum(x[j, depot_dest, k] for j in all_nodes if j != depot_dest) == 1
                )

            # 6. No vehicle enters depot_origin or leaves depot_dest
            for k in vehicles:
                model.addConstr(
                    gp.quicksum(x[j, depot_origin, k] for j in all_nodes if j != depot_origin) == 0
                )
                model.addConstr(
                    gp.quicksum(x[depot_dest, j, k] for j in all_nodes if j != depot_dest) == 0
                )

            # 7. Same vehicle for pickup and delivery of each request
            for req in requests:
                p = req['pickup_node']
                d = req['delivery_node']
                for k in vehicles:
                    model.addConstr(
                        gp.quicksum(x[p, j, k] for j in all_nodes if j != p) ==
                        gp.quicksum(x[d, j, k] for j in all_nodes if j != d)
                    )

            # 8. Time propagation
            M_time = planning_horizon + 1000
            for i in all_nodes:
                for j in all_nodes:
                    if i == j:
                        continue
                    for k in vehicles:
                        model.addConstr(
                            t[j, k] >= t[i, k] + travel_time[i][j] - M_time * (1 - x[i, j, k])
                        )

            # 9. Precedence: pickup before delivery
            for req in requests:
                p = req['pickup_node']
                d = req['delivery_node']
                for k in vehicles:
                    model.addConstr(t[d, k] >= t[p, k])

            # 10. Capacity constraints using load variables
            load = {}
            for i in all_nodes:
                for k in vehicles:
                    load[i, k] = model.addVar(lb=0, ub=vehicle_capacity, vtype=GRB.CONTINUOUS,
                                              name=f"load_{i}_{k}")

            # Load at depot_origin is 0
            for k in vehicles:
                model.addConstr(load[depot_origin, k] == 0)

            # Load propagation
            M_load = vehicle_capacity
            for i in all_nodes:
                for j in all_nodes:
                    if i == j:
                        continue
                    for k in vehicles:
                        model.addConstr(
                            load[j, k] >= load[i, k] + nodes[j]['demand'] - M_load * (1 - x[i, j, k])
                        )
                        model.addConstr(
                            load[j, k] <= load[i, k] + nodes[j]['demand'] + M_load * (1 - x[i, j, k])
                        )

            model.optimize()

            if model.Status == GRB.OPTIMAL or (model.Status == GRB.TIME_LIMIT and model.SolCount > 0):
                # Extract solution
                routes = []
                for k in vehicles:
                    # Reconstruct route for vehicle k
                    route = [depot_origin]
                    current = depot_origin
                    visited = {depot_origin}
                    while current != depot_dest:
                        found_next = False
                        for j in all_nodes:
                            if j != current and j not in visited:
                                if x[current, j, k].X > 0.5:
                                    route.append(j)
                                    visited.add(j)
                                    current = j
                                    found_next = True
                                    break
                        if not found_next:
                            # Try depot_dest
                            if current != depot_dest:
                                route.append(depot_dest)
                            break
                    if route[-1] != depot_dest:
                        route.append(depot_dest)
                    routes.append(route)

                # Verify solution
                obj = total_cost(routes)
                if obj < float('inf'):
                    return routes, obj

            return None

        except Exception as e:
            return None

    # ============ CONSTRUCTION HEURISTICS ============
    def cheapest_insertion(order=None):
        routes = [[depot_origin, depot_dest] for _ in range(num_vehicles)]
        route_costs = [dist[depot_origin][depot_dest]] * num_vehicles

        if order is None:
            order = list(range(num_requests))

        for req_idx in order:
            req = requests[req_idx]
            pickup = req['pickup_node']
            delivery = req['delivery_node']

            best_increase = float('inf')
            best_v = -1
            best_new_route = None

            for v in range(num_vehicles):
                route = routes[v]
                for pi in range(1, len(route)):
                    temp = route[:pi] + [pickup] + route[pi:]
                    for di in range(pi + 1, len(temp)):
                        new_route = temp[:di] + [delivery] + temp[di:]
                        f, c = evaluate_route(new_route)
                        if f:
                            increase = c - route_costs[v]
                            if increase < best_increase - 1e-10:
                                best_increase = increase
                                best_v = v
                                best_new_route = new_route

            if best_v >= 0:
                routes[best_v] = best_new_route
                route_costs[best_v] += best_increase
            else:
                return None

        return routes

    def regret_insertion(order=None):
        """Regret-2 insertion heuristic."""
        routes = [[depot_origin, depot_dest] for _ in range(num_vehicles)]
        route_costs = [dist[depot_origin][depot_dest]] * num_vehicles
        unassigned = set(range(num_requests))

        while unassigned:
            best_regret = -float('inf')
            best_req = -1
            best_v = -1
            best_new_route = None

            for req_idx in unassigned:
                req = requests[req_idx]
                pickup = req['pickup_node']
                delivery = req['delivery_node']

                # Find best and second best insertion costs
                insertion_costs = []
                for v in range(num_vehicles):
                    route = routes[v]
                    best_cost_v = float('inf')
                    best_route_v = None
                    for pi in range(1, len(route)):
                        temp = route[:pi] + [pickup] + route[pi:]
                        for di in range(pi + 1, len(temp)):
                            new_route = temp[:di] + [delivery] + temp[di:]
                            f, c = evaluate_route(new_route)
                            if f:
                                increase = c - route_costs[v]
                                if increase < best_cost_v:
                                    best_cost_v = increase
                                    best_route_v = new_route
                    insertion_costs.append((best_cost_v, v, best_route_v))

                insertion_costs.sort(key=lambda x: x[0])

                if len(insertion_costs) >= 2 and insertion_costs[0][0] < float('inf'):
                    regret = insertion_costs[1][0] - insertion_costs[0][0]
                elif len(insertion_costs) >= 1 and insertion_costs[0][0] < float('inf'):
                    regret = insertion_costs[0][0]
                else:
                    regret = -float('inf')

                if regret > best_regret or (regret == best_regret and
                                            insertion_costs[0][0] < best_regret):
                    best_regret = regret
                    best_req = req_idx
                    if insertion_costs[0][0] < float('inf'):
                        best_v = insertion_costs[0][1]
                        best_new_route = insertion_costs[0][2]

            if best_req >= 0 and best_v >= 0 and best_new_route is not None:
                routes[best_v] = best_new_route
                req = requests[best_req]
                f, c = evaluate_route(best_new_route)
                route_costs[best_v] = c
                unassigned.remove(best_req)
            else:
                return None

        return routes

    # ============ LOCAL SEARCH ============
    def find_best_insertion(route, pickup, delivery):
        """Find best position to insert a pickup-delivery pair into a route."""
        best_cost = float('inf')
        best_route = None
        for pi in range(1, len(route)):
            temp = route[:pi] + [pickup] + route[pi:]
            for di in range(pi + 1, len(temp)):
                new_route = temp[:di] + [delivery] + temp[di:]
                f, c = evaluate_route(new_route)
                if f and c < best_cost:
                    best_cost = c
                    best_route = new_route
        return best_cost, best_route

    def intra_route_search(routes):
        """Try moving pickup-delivery pairs within each route."""
        improved = True
        while improved:
            improved = False
            for v in range(num_vehicles):
                route = routes[v]
                if len(route) <= 4:
                    continue

                pairs = [(node, pickup_to_delivery[node]) for node in route if node in pickup_to_delivery]
                current_cost = route_cost(route)

                for pickup, delivery in pairs:
                    reduced = [x for x in route if x != pickup and x != delivery]
                    best_c, best_r = find_best_insertion(reduced, pickup, delivery)
                    if best_r is not None and best_c < current_cost - 1e-9:
                        routes[v] = best_r
                        current_cost = best_c
                        improved = True
        return routes

    def inter_route_relocate(routes):
        """Try moving pickup-delivery pairs between routes."""
        improved = True
        while improved:
            improved = False
            for v1 in range(num_vehicles):
                if improved:
                    break
                route1 = routes[v1]
                pairs_in_r1 = [(node, pickup_to_delivery[node]) for node in route1 if node in pickup_to_delivery]

                for pickup, delivery in pairs_in_r1:
                    if improved:
                        break
                    new_route1 = [x for x in route1 if x != pickup and x != delivery]
                    f1, c1 = evaluate_route(new_route1)
                    if not f1:
                        continue

                    cost1 = route_cost(route1)

                    for v2 in range(num_vehicles):
                        if v2 == v1:
                            continue
                        route2 = routes[v2]
                        cost2 = route_cost(route2)

                        best_c2, best_r2 = find_best_insertion(route2, pickup, delivery)
                        if best_r2 is not None and c1 + best_c2 < cost1 + cost2 - 1e-9:
                            routes[v1] = new_route1
                            routes[v2] = best_r2
                            improved = True
                            break
        return routes

    def swap_pairs(routes):
        """Try swapping pickup-delivery pairs between routes."""
        improved = True
        while improved:
            improved = False
            for v1 in range(num_vehicles):
                if improved:
                    break
                for v2 in range(v1 + 1, num_vehicles):
                    if improved:
                        break
                    route1 = routes[v1]
                    route2 = routes[v2]

                    pairs1 = [(node, pickup_to_delivery[node]) for node in route1 if node in pickup_to_delivery]
                    pairs2 = [(node, pickup_to_delivery[node]) for node in route2 if node in pickup_to_delivery]

                    if not pairs1 or not pairs2:
                        continue

                    cost1 = route_cost(route1)
                    cost2 = route_cost(route2)

                    for p1, d1 in pairs1:
                        if improved:
                            break
                        for p2, d2 in pairs2:
                            # Remove pair1 from route1, insert pair2
                            reduced1 = [x for x in route1 if x != p1 and x != d1]
                            c_new1, r_new1 = find_best_insertion(reduced1, p2, d2)
                            if r_new1 is None:
                                continue

                            # Remove pair2 from route2, insert pair1
                            reduced2 = [x for x in route2 if x != p2 and x != d2]
                            c_new2, r_new2 = find_best_insertion(reduced2, p1, d1)
                            if r_new2 is None:
                                continue

                            if c_new1 + c_new2 < cost1 + cost2 - 1e-9:
                                routes[v1] = r_new1
                                routes[v2] = r_new2
                                improved = True
                                break
        return routes

    def two_opt_intra(routes):
        """Try 2-opt style improvements within routes (reversing segments)."""
        improved = True
        while improved:
            improved = False
            for v in range(num_vehicles):
                route = routes[v]
                if len(route) <= 4:
                    continue
                current_cost = route_cost(route)
                # Try reversing segments (but must maintain pickup before delivery)
                for i in range(1, len(route) - 2):
                    for j in range(i + 1, len(route) - 1):
                        new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                        # Check precedence
                        pos = {node: idx for idx, node in enumerate(new_route)}
                        valid = True
                        for p, d in pickup_to_delivery.items():
                            if p in pos and d in pos:
                                if pos[p] > pos[d]:
                                    valid = False
                                    break
                        if not valid:
                            continue
                        f, c = evaluate_route(new_route)
                        if f and c < current_cost - 1e-9:
                            routes[v] = new_route
                            current_cost = c
                            improved = True
                            break
                    if improved:
                        break
        return routes

    def full_local_search(routes):
        """Run all local search procedures until no improvement."""
        prev_cost = total_cost(routes)
        for _ in range(10):  # Max iterations
            if time.time() - start_time > time_limit * 0.85:
                break
            routes = intra_route_search(routes)
            routes = inter_route_relocate(routes)
            routes = swap_pairs(routes)
            routes = two_opt_intra(routes)
            routes = intra_route_search(routes)
            new_cost = total_cost(routes)
            if new_cost >= prev_cost - 1e-9:
                break
            prev_cost = new_cost
        return routes

    # ============ MAIN SOLVER LOGIC ============
    best_routes = None
    best_obj = float('inf')

    # Try Gurobi for small instances
    if num_requests <= 7:
        gurobi_result = solve_gurobi()
        if gurobi_result is not None:
            routes, obj = gurobi_result
            if obj < best_obj:
                best_obj = obj
                best_routes = [r[:] for r in routes]
                sol = format_solution(best_routes, best_obj)
                if logger:
                    logger.log_solution(best_obj, sol)

    # Heuristic approach
    random.seed(42)
    num_attempts = max(10, min(50, time_limit // 5))

    for attempt in range(num_attempts):
        if time.time() - start_time > time_limit * 0.75:
            break

        if attempt == 0:
            # Tightest time window first
            order = sorted(range(num_requests),
                           key=lambda i: requests[i]['pickup_time_window'][1] - requests[i]['pickup_time_window'][0])
            routes = cheapest_insertion(order)
        elif attempt == 1:
            # Largest load first
            order = sorted(range(num_requests), key=lambda i: -requests[i]['load'])
            routes = cheapest_insertion(order)
        elif attempt == 2:
            # Earliest pickup time first
            order = sorted(range(num_requests), key=lambda i: requests[i]['pickup_time_window'][0])
            routes = cheapest_insertion(order)
        elif attempt == 3:
            # Regret insertion
            routes = regret_insertion()
        elif attempt == 4:
            # Default order
            routes = cheapest_insertion()
        else:
            # Random order
            order = list(range(num_requests))
            random.shuffle(order)
            routes = cheapest_insertion(order)

        if routes is None:
            continue

        # Local search improvement
        routes = full_local_search(routes)

        obj = total_cost(routes)
        if obj < best_obj - 1e-9:
            best_obj = obj
            best_routes = [r[:] for r in routes]
            sol = format_solution(best_routes, best_obj)
            if logger:
                logger.log_solution(best_obj, sol)

    # If still no solution, try basic insertion
    if best_routes is None:
        routes = cheapest_insertion()
        if routes is not None:
            best_routes = routes
            best_obj = total_cost(routes)
            sol = format_solution(best_routes, best_obj)
            if logger:
                logger.log_solution(best_obj, sol)

    # Final output
    if best_routes is not None:
        sol = format_solution(best_routes, best_obj)
    else:
        # Fallback (should not happen for feasible instances)
        fallback_routes = [[depot_origin, depot_dest] for _ in range(num_vehicles)]
        sol = format_solution(fallback_routes, num_vehicles * dist[depot_origin][depot_dest])

    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)


if __name__ == '__main__':
    main()