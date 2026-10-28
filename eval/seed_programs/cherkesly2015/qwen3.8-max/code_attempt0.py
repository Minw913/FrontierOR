import json
import argparse
import time
import random
from copy import deepcopy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()
    time_limit = args.time_limit

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n_requests = instance['n_requests']
    n_nodes = instance['n_nodes']
    capacity = instance['vehicle_capacity']
    nodes_data = {n['node_id']: n for n in instance['nodes']}

    arc_cost = {}
    arc_time = {}
    for a in instance['arcs']:
        arc_cost[(a['from'], a['to'])] = a['cost']
        arc_time[(a['from'], a['to'])] = a['travel_time']

    depot_o = 0
    depot_d = 2 * n_requests + 1

    def tt(i, j):
        return arc_time.get((i, j), float('inf'))

    def tc(i, j):
        return arc_cost.get((i, j), float('inf'))

    def has_arc(i, j):
        return (i, j) in arc_cost

    def check_lifo(route):
        stack = []
        for node in route[1:-1]:
            if 1 <= node <= n_requests:
                stack.append(node)
            elif n_requests + 1 <= node <= 2 * n_requests:
                req = node - n_requests
                if not stack or stack[-1] != req:
                    return False
                stack.pop()
        return len(stack) == 0

    def evaluate_route(route):
        if len(route) < 2:
            return False, None, None, float('inf')
        if route[0] != depot_o or route[-1] != depot_d:
            return False, None, None, float('inf')
        if not check_lifo(route):
            return False, None, None, float('inf')

        times = [0.0] * len(route)
        loads = [0] * len(route)
        cost = 0.0

        times[0] = nodes_data[route[0]]['tw_early']
        loads[0] = 0

        for i in range(1, len(route)):
            prev, curr = route[i - 1], route[i]
            if not has_arc(prev, curr):
                return False, None, None, float('inf')
            cost += tc(prev, curr)
            earliest = times[i - 1] + nodes_data[prev]['service_time'] + tt(prev, curr)
            times[i] = max(earliest, nodes_data[curr]['tw_early'])
            if times[i] > nodes_data[curr]['tw_late']:
                return False, None, None, float('inf')
            loads[i] = loads[i - 1] + nodes_data[curr]['load']
            if loads[i] < 0 or loads[i] > capacity:
                return False, None, None, float('inf')

        return True, times, loads, cost

    def route_cost_only(route):
        cost = 0.0
        for i in range(len(route) - 1):
            if not has_arc(route[i], route[i + 1]):
                return float('inf')
            cost += tc(route[i], route[i + 1])
        return cost

    def total_cost(routes):
        return sum(route_cost_only(r) for r in routes)

    def find_best_insertion(route, pickup, delivery):
        n = len(route)
        best_cost = float('inf')
        best_route = None

        for i in range(1, n):
            for j in range(i + 1, n + 1):
                new_route = route[:i] + [pickup] + route[i:j - 1] + [delivery] + route[j - 1:]
                feasible, _, _, cost = evaluate_route(new_route)
                if feasible and cost < best_cost:
                    best_cost = cost
                    best_route = new_route

        return best_cost, best_route

    def construct_solution(order):
        routes = []
        for req_id in order:
            pickup = req_id
            delivery = req_id + n_requests

            best_cost = float('inf')
            best_route = None
            best_idx = -1

            for idx, route in enumerate(routes):
                cost, new_route = find_best_insertion(route, pickup, delivery)
                if new_route is not None and cost < best_cost:
                    best_cost = cost
                    best_route = new_route
                    best_idx = idx

            # Try new route
            new_route = [depot_o, pickup, delivery, depot_d]
            feasible, _, _, cost = evaluate_route(new_route)
            if feasible and cost < best_cost:
                best_cost = cost
                best_route = new_route
                best_idx = -1

            if best_route is not None:
                if best_idx >= 0:
                    routes[best_idx] = best_route
                else:
                    routes.append(best_route)
            else:
                routes.append([depot_o, pickup, delivery, depot_d])

        return routes

    def local_search_relocate(routes):
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > time_limit - 2:
                break

            for r1_idx in range(len(routes)):
                route1 = routes[r1_idx]
                reqs_in_route = [n for n in route1[1:-1] if 1 <= n <= n_requests]

                for req_id in reqs_in_route:
                    pickup = req_id
                    delivery = req_id + n_requests

                    reduced = [n for n in route1 if n != pickup and n != delivery]
                    old_cost_r1 = route_cost_only(route1)
                    reduced_cost = route_cost_only(reduced)

                    best_saving = 1e-10
                    best_action = None

                    # Try other routes
                    for r2_idx in range(len(routes)):
                        if r2_idx == r1_idx:
                            continue
                        route2 = routes[r2_idx]
                        cost2, new_route2 = find_best_insertion(route2, pickup, delivery)
                        if new_route2 is not None:
                            old_total = old_cost_r1 + route_cost_only(route2)
                            new_total = reduced_cost + cost2
                            saving = old_total - new_total
                            if saving > best_saving:
                                best_saving = saving
                                best_action = ('move', r1_idx, reduced, r2_idx, new_route2)

                    # Try reinserting into same route at different position
                    cost_same, new_route_same = find_best_insertion(reduced, pickup, delivery)
                    if new_route_same is not None:
                        saving = old_cost_r1 - cost_same
                        if saving > best_saving:
                            best_saving = saving
                            best_action = ('reinsert', r1_idx, new_route_same, -1, None)

                    if best_action:
                        if best_action[0] == 'move':
                            _, r1, new_r1, r2, new_r2 = best_action
                            routes[r1] = new_r1
                            routes[r2] = new_r2
                        else:
                            _, r1, new_r1, _, _ = best_action
                            routes[r1] = new_r1
                        improved = True
                        break

                if improved:
                    break

            routes = [r for r in routes if len(r) > 2]

        return routes

    def local_search_swap(routes):
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > time_limit - 2:
                break

            for r1_idx in range(len(routes)):
                route1 = routes[r1_idx]
                reqs1 = [n for n in route1[1:-1] if 1 <= n <= n_requests]

                for r2_idx in range(r1_idx + 1, len(routes)):
                    route2 = routes[r2_idx]
                    reqs2 = [n for n in route2[1:-1] if 1 <= n <= n_requests]

                    for req_a in reqs1:
                        for req_b in reqs2:
                            # Remove req_a from route1, req_b from route2
                            red1 = [n for n in route1 if n != req_a and n != req_a + n_requests]
                            red2 = [n for n in route2 if n != req_b and n != req_b + n_requests]

                            # Insert req_b into red1
                            cost1, new_r1 = find_best_insertion(red1, req_b, req_b + n_requests)
                            if new_r1 is None:
                                continue

                            # Insert req_a into red2
                            cost2, new_r2 = find_best_insertion(red2, req_a, req_a + n_requests)
                            if new_r2 is None:
                                continue

                            old_total = route_cost_only(route1) + route_cost_only(route2)
                            new_total = cost1 + cost2

                            if new_total < old_total - 1e-10:
                                routes[r1_idx] = new_r1
                                routes[r2_idx] = new_r2
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        return routes

    def full_local_search(routes):
        routes = local_search_relocate(routes)
        routes = local_search_swap(routes)
        routes = local_search_relocate(routes)
        return routes

    def perturb(routes, strength=2):
        routes = deepcopy(routes)
        for _ in range(strength):
            if len(routes) <= 1:
                break
            # Pick a random route with at least one request
            candidates = [i for i, r in enumerate(routes) if len(r) > 2]
            if not candidates:
                break
            r_idx = random.choice(candidates)
            route = routes[r_idx]
            reqs = [n for n in route[1:-1] if 1 <= n <= n_requests]
            if not reqs:
                break
            req_id = random.choice(reqs)
            pickup = req_id
            delivery = req_id + n_requests

            reduced = [n for n in route if n != pickup and n != delivery]

            # Try to insert into another random route
            other_indices = [i for i in range(len(routes)) if i != r_idx and len(routes[i]) > 2]
            if other_indices:
                target_idx = random.choice(other_indices)
                cost, new_route = find_best_insertion(routes[target_idx], pickup, delivery)
                if new_route is not None:
                    routes[target_idx] = new_route
                    routes[r_idx] = reduced
                else:
                    # Create new route
                    new_r = [depot_o, pickup, delivery, depot_d]
                    if evaluate_route(new_r)[0]:
                        routes.append(new_r)
                        routes[r_idx] = reduced
            else:
                # Create new route
                new_r = [depot_o, pickup, delivery, depot_d]
                if evaluate_route(new_r)[0]:
                    routes.append(new_r)
                    routes[r_idx] = reduced

        routes = [r for r in routes if len(r) > 2]
        return routes

    def format_solution(routes):
        obj = total_cost(routes)
        solution = {
            "objective_value": obj,
            "routes": []
        }
        for vid, route in enumerate(routes):
            feasible, times, loads, _ = evaluate_route(route)
            route_details = []
            for i, node in enumerate(route):
                route_details.append({
                    "node_id": node,
                    "arrival_time": times[i] if times else 0.0,
                    "load_after": loads[i] if loads else 0
                })
            solution["routes"].append({
                "vehicle_id": vid + 1,
                "route_sequence": route,
                "route_details": route_details
            })
        return solution

    # ============ MAIN SOLVING LOGIC ============
    random.seed(42)

    best_routes = None
    best_obj = float('inf')

    def update_best(routes):
        nonlocal best_routes, best_obj
        # Verify all routes are feasible
        for r in routes:
            if not evaluate_route(r)[0]:
                return
        obj = total_cost(routes)
        if obj < best_obj:
            best_obj = obj
            best_routes = deepcopy(routes)
            sol = format_solution(best_routes)
            if logger:
                logger.log_solution(best_obj, sol)

    # Construction strategies
    construction_orders = []

    # Sequential
    construction_orders.append(list(range(1, n_requests + 1)))

    # Reverse
    construction_orders.append(list(range(n_requests, 0, -1)))

    # By time window tightness (pickup)
    construction_orders.append(sorted(range(1, n_requests + 1),
                                      key=lambda r: nodes_data[r]['tw_late'] - nodes_data[r]['tw_early']))

    # By time window tightness (delivery)
    construction_orders.append(sorted(range(1, n_requests + 1),
                                      key=lambda r: nodes_data[r + n_requests]['tw_late'] - nodes_data[r + n_requests]['tw_early']))

    # By load (descending)
    construction_orders.append(sorted(range(1, n_requests + 1),
                                      key=lambda r: -nodes_data[r]['load']))

    # By distance from depot
    construction_orders.append(sorted(range(1, n_requests + 1),
                                      key=lambda r: (nodes_data[r]['x'] - nodes_data[depot_o]['x']) ** 2 +
                                                    (nodes_data[r]['y'] - nodes_data[depot_o]['y']) ** 2))

    # Random orderings
    n_random_trials = max(10, min(100, 200 // max(1, n_requests // 5)))
    for _ in range(n_random_trials):
        order = list(range(1, n_requests + 1))
        random.shuffle(order)
        construction_orders.append(order)

    # Phase 1: Construction + local search
    for order in construction_orders:
        if time.time() - start_time > time_limit * 0.4:
            break
        routes = construct_solution(order)
        routes = full_local_search(routes)
        update_best(routes)

    # Phase 2: Iterated local search
    if best_routes is not None and time.time() - start_time < time_limit - 5:
        current_routes = deepcopy(best_routes)
        current_obj = best_obj
        temp = best_obj * 0.05 if best_obj > 0 else 100

        iteration = 0
        while time.time() - start_time < time_limit - 3:
            iteration += 1

            # Perturb
            strength = random.randint(1, min(3, max(1, len(current_routes) // 2)))
            perturbed = perturb(current_routes, strength)

            if not perturbed:
                continue

            # Local search
            improved_routes = full_local_search(perturbed)

            # Verify feasibility
            all_feasible = all(evaluate_route(r)[0] for r in improved_routes)
            if not all_feasible:
                continue

            new_obj = total_cost(improved_routes)

            # Accept or reject
            delta = new_obj - current_obj
            if delta < 0 or random.random() < max(0, 1.0 - delta / max(temp, 1e-10)):
                current_routes = improved_routes
                current_obj = new_obj
                if new_obj < best_obj:
                    update_best(current_routes)

            # Cooling
            temp *= 0.9995

            if iteration % 50 == 0:
                temp = max(temp, best_obj * 0.01 if best_obj > 0 else 10)

    # Phase 3: Try MIP for small instances
    if n_requests <= 6 and time.time() - start_time < time_limit - 10:
        try:
            mip_routes = solve_mip(instance, arc_cost, arc_time, nodes_data,
                                   n_requests, capacity, depot_o, depot_d,
                                   time_limit - (time.time() - start_time) - 2)
            if mip_routes is not None:
                all_feasible = all(evaluate_route(r)[0] for r in mip_routes)
                if all_feasible:
                    update_best(mip_routes)
        except Exception:
            pass

    # Ensure we have a valid solution
    if best_routes is None:
        # Fallback: one route per request
        best_routes = []
        for r in range(1, n_requests + 1):
            route = [depot_o, r, r + n_requests, depot_d]
            best_routes.append(route)
        best_obj = total_cost(best_routes)

    # Final output
    sol = format_solution(best_routes)
    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)

    if logger:
        logger.log_solution(best_obj, sol)


def solve_mip(instance, arc_cost, arc_time, nodes_data, n_requests, capacity,
              depot_o, depot_d, remaining_time):
    """Solve using Gurobi MIP for small instances."""
    import gurobipy as gp
    from gurobipy import GRB

    n_nodes = instance['n_nodes']
    arcs_list = instance['arcs']

    model = gp.Model("pdptw_lifo")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', max(1, remaining_time))
    model.setParam('OutputFlag', 0)

    # Decision variables
    x = {}
    for a in arcs_list:
        x[(a['from'], a['to'])] = model.addVar(vtype=GRB.BINARY, name=f"x_{a['from']}_{a['to']}")

    # Time variables
    t = {}
    for nid in nodes_data:
        if nid != depot_o and nid != depot_d:
            t[nid] = model.addVar(lb=nodes_data[nid]['tw_early'],
                                  ub=nodes_data[nid]['tw_late'],
                                  name=f"t_{nid}")

    # Load variables
    l = {}
    for nid in nodes_data:
        if nid != depot_o and nid != depot_d:
            l[nid] = model.addVar(lb=0, ub=capacity, name=f"l_{nid}")

    model.update()

    # Objective
    obj = gp.quicksum(arc_cost[(a['from'], a['to'])] * x[(a['from'], a['to'])]
                      for a in arcs_list)
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints
    # Each pickup visited exactly once
    for r in range(1, n_requests + 1):
        p = r
        model.addConstr(gp.quicksum(x[(a['from'], a['to'])] for a in arcs_list
                                    if a['to'] == p) == 1, name=f"visit_p_{r}")

    # Each delivery visited exactly once
    for r in range(1, n_requests + 1):
        d = r + n_requests
        model.addConstr(gp.quicksum(x[(a['from'], a['to'])] for a in arcs_list
                                    if a['to'] == d) == 1, name=f"visit_d_{r}")

    # Flow conservation at non-depot nodes
    for nid in nodes_data:
        if nid == depot_o or nid == depot_d:
            continue
        inflow = gp.quicksum(x[(a['from'], a['to'])] for a in arcs_list if a['to'] == nid)
        outflow = gp.quicksum(x[(a['from'], a['to'])] for a in arcs_list if a['from'] == nid)
        model.addConstr(inflow == outflow, name=f"flow_{nid}")

    # Depot balance
    out_depot = gp.quicksum(x[(a['from'], a['to'])] for a in arcs_list if a['from'] == depot_o)
    in_depot = gp.quicksum(x[(a['from'], a['to'])] for a in arcs_list if a['to'] == depot_d)
    model.addConstr(out_depot == in_depot, name="depot_balance")
    model.addConstr(out_depot >= 1, name="at_least_one_vehicle")

    # Time linking constraints
    M_time = max(nodes_data[n]['tw_late'] for n in nodes_data) + 1000
    for a in arcs_list:
        i, j = a['from'], a['to']
        if i == depot_o or j == depot_d:
            continue
        if i == depot_o or j == depot_d:
            continue
        if i in t and j in t:
            travel = arc_time[(i, j)]
            service = nodes_data[i]['service_time']
            model.addConstr(t[j] >= t[i] + service + travel - M_time * (1 - x[(i, j)]),
                            name=f"time_{i}_{j}")

    # Time from depot
    for a in arcs_list:
        if a['from'] == depot_o and a['to'] in t:
            j = a['to']
            travel = arc_time[(depot_o, j)]
            model.addConstr(t[j] >= nodes_data[depot_o]['tw_early'] + travel - M_time * (1 - x[(depot_o, j)]),
                            name=f"time_depot_{j}")

    # Load linking constraints
    M_load = capacity + 1
    for a in arcs_list:
        i, j = a['from'], a['to']
        if j == depot_o or j == depot_d or i == depot_d:
            continue
        if i == depot_o:
            # Load after first node = demand of that node
            if j in l:
                model.addConstr(l[j] >= nodes_data[j]['load'] - M_load * (1 - x[(i, j)]),
                                name=f"load_start_{j}")
        elif i in l and j in l:
            model.addConstr(l[j] >= l[i] + nodes_data[j]['load'] - M_load * (1 - x[(i, j)]),
                            name=f"load_{i}_{j}")

    # Load must be zero before returning to depot
    for a in arcs_list:
        if a['to'] == depot_d and a['from'] in l:
            i = a['from']
            model.addConstr(l[i] <= capacity * (1 - x[(i, depot_d)]),
                            name=f"load_end_{i}")

    # Precedence: pickup before delivery
    for r in range(1, n_requests + 1):
        p = r
        d = r + n_requests
        if p in t and d in t:
            direct_tt = arc_time.get((p, d), 0)
            model.addConstr(t[d] >= t[p] + nodes_data[p]['service_time'] + direct_tt,
                            name=f"prec_{r}")

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
            # Extract routes
            selected_arcs = [(a['from'], a['to']) for a in arcs_list
                             if x[(a['from'], a['to'])].X > 0.5]

            # Build adjacency
            adj = {}
            for i, j in selected_arcs:
                if i not in adj:
                    adj[i] = []
                adj[i].append(j)

            # Trace routes from depot
            routes = []
            visited_from_depot = set()

            for start_target in adj.get(depot_o, []):
                if start_target in visited_from_depot:
                    continue
                route = [depot_o]
                current = start_target
                visited_from_depot.add(current)
                while current != depot_d:
                    route.append(current)
                    nexts = adj.get(current, [])
                    if not nexts:
                        break
                    current = nexts[0]
                route.append(depot_d)
                routes.append(route)

            # Verify all requests are covered
            covered = set()
            for route in routes:
                for node in route[1:-1]:
                    if 1 <= node <= n_requests:
                        covered.add(node)

            if len(covered) == n_requests:
                return routes

    return None


if __name__ == '__main__':
    main()