import json
import argparse
import time
import sys

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

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_vertices = instance['num_vertices']
    num_markets = instance['num_markets']
    num_products = instance['num_products']
    depot = instance['depot']
    travel_costs = instance['travel_costs']
    product_demands = instance['product_demands']
    product_markets = instance['product_markets']
    purchase_costs = instance['purchase_costs']
    supplies = instance['supplies']

    markets = sorted([v for v in range(num_vertices) if v != depot])

    # Determine mandatory markets
    mandatory = set()
    for k in range(num_products):
        demand_k = product_demands[k]
        mkts = product_markets[str(k)]
        for m in mkts:
            total_without = sum(supplies[f"{k},{o}"] for o in mkts if o != m)
            if total_without < demand_k:
                mandatory.add(m)

    from gurobipy import *

    env = Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    model = Model("purchasing_tour", env=env)
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = args.time_limit

    # Edge variables
    x = {}
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # Visit variables
    y = {}
    for m in markets:
        y[m] = model.addVar(vtype=GRB.BINARY, name=f"y_{m}")

    # Purchase variables (integer)
    p = {}
    for k in range(num_products):
        for m in product_markets[str(k)]:
            sup = supplies[f"{k},{m}"]
            p[k, m] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=sup, name=f"p_{k}_{m}")

    # MTZ position variables
    u = {}
    for m in markets:
        u[m] = model.addVar(vtype=GRB.CONTINUOUS, lb=1, ub=num_markets, name=f"u_{m}")

    model.update()

    # Objective
    obj = LinExpr()
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            obj += travel_costs[f"{i},{j}"] * x[i, j]
    for k in range(num_products):
        for m in product_markets[str(k)]:
            obj += purchase_costs[f"{k},{m}"] * p[k, m]
    model.setObjective(obj, GRB.MINIMIZE)

    # Degree constraints
    for v in range(num_vertices):
        deg = LinExpr()
        for j in range(num_vertices):
            if j != v:
                deg += x[min(v, j), max(v, j)]
        if v == depot:
            model.addConstr(deg == 2, f"deg_{v}")
        else:
            model.addConstr(deg == 2 * y[v], f"deg_{v}")

    # Mandatory markets
    for m in mandatory:
        model.addConstr(y[m] == 1, f"mand_{m}")

    # Demand constraints
    for k in range(num_products):
        model.addConstr(
            quicksum(p[k, m] for m in product_markets[str(k)]) == product_demands[k],
            f"demand_{k}"
        )

    # Link purchases to visits
    for k in range(num_products):
        for m in product_markets[str(k)]:
            model.addConstr(p[k, m] <= supplies[f"{k},{m}"] * y[m], f"link_{k}_{m}")

    # MTZ subtour elimination
    for i in markets:
        for j in markets:
            if i != j:
                model.addConstr(
                    u[i] - u[j] + num_markets * x[min(i, j), max(i, j)] <= num_markets - 1,
                    f"mtz_{i}_{j}"
                )

    model.update()

    # Build a greedy initial solution to warm-start
    init_solution = build_greedy_solution(instance, mandatory, markets, depot,
                                          num_vertices, num_products,
                                          product_markets, product_demands,
                                          travel_costs, purchase_costs, supplies)
    if init_solution is not None:
        try:
            visited_set = set(init_solution["visited_markets"])
            for m in markets:
                y[m].Start = 1.0 if m in visited_set else 0.0
            for i in range(num_vertices):
                for j in range(i + 1, num_vertices):
                    edge_used = False
                    tour = init_solution["tour"]
                    for idx in range(len(tour) - 1):
                        a, b = tour[idx], tour[idx + 1]
                        if (min(a, b), max(a, b)) == (i, j):
                            edge_used = True
                            break
                    x[i, j].Start = 1.0 if edge_used else 0.0
            for k in range(num_products):
                for m in product_markets[str(k)]:
                    key = str(k)
                    mkey = str(m)
                    if key in init_solution["purchases"] and mkey in init_solution["purchases"][key]:
                        p[k, m].Start = float(init_solution["purchases"][key][mkey])
                    else:
                        p[k, m].Start = 0.0
            # Set u starts based on tour order
            tour = init_solution["tour"]
            for idx, v in enumerate(tour[1:-1], 1):
                if v in u:
                    u[v].Start = float(idx)
        except:
            pass

    model.optimize()

    # Extract solution
    solution = None
    if model.SolCount > 0:
        solution = extract_solution(model, instance, markets, depot, num_vertices,
                                    num_products, product_markets, x, y, p)

    if solution is None:
        solution = init_solution

    if solution is None:
        solution = build_greedy_solution(instance, mandatory, markets, depot,
                                         num_vertices, num_products,
                                         product_markets, product_demands,
                                         travel_costs, purchase_costs, supplies)

    # Try to improve with local search if time permits
    elapsed = time.time() - start_time
    if solution is not None and elapsed < args.time_limit - 5:
        solution = local_search_improve(solution, instance, mandatory, markets, depot,
                                        num_vertices, num_products,
                                        product_markets, product_demands,
                                        travel_costs, purchase_costs, supplies,
                                        args.time_limit, start_time)

    if logger and solution:
        logger.log_solution(solution["objective_value"], solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def extract_solution(model, instance, markets, depot, num_vertices,
                     num_products, product_markets, x, y, p):
    try:
        visited = sorted([m for m in markets if y[m].X > 0.5])

        adj = {v: [] for v in range(num_vertices)}
        tour_edges = []
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                if x[i, j].X > 0.5:
                    tour_edges.append([i, j])
                    adj[i].append(j)
                    adj[j].append(i)

        # Trace tour
        tour = [depot]
        current = depot
        used = {depot}
        while len(tour) < len(visited) + 1:
            nexts = [n for n in adj.get(current, []) if n not in used]
            if not nexts:
                break
            current = nexts[0]
            tour.append(current)
            used.add(current)
        tour.append(depot)

        purchases = {}
        for k in range(num_products):
            purchases[str(k)] = {}
            for m in product_markets[str(k)]:
                val = p[k, m].X
                if val > 0.5:
                    purchases[str(k)][str(m)] = int(round(val))

        obj_val = model.ObjVal

        return {
            "objective_value": float(obj_val),
            "visited_markets": visited,
            "tour": tour,
            "tour_edges": tour_edges,
            "purchases": purchases
        }
    except:
        return None


def build_greedy_solution(instance, mandatory, markets, depot, num_vertices,
                          num_products, product_markets, product_demands,
                          travel_costs, purchase_costs, supplies):
    """Build a feasible solution greedily."""
    try:
        # Determine which markets we need to visit
        needed = set(mandatory)

        # For each product, ensure we can meet demand
        for k in range(num_products):
            demand_k = product_demands[k]
            mkts = product_markets[str(k)]
            # Sort markets by purchase cost
            sorted_mkts = sorted(mkts, key=lambda m: purchase_costs[f"{k},{m}"])
            remaining = demand_k
            for m in sorted_mkts:
                if remaining <= 0:
                    break
                needed.add(m)
                remaining -= supplies[f"{k},{m}"]

        visited = sorted(needed)
        if not visited:
            return None

        # Build tour using nearest neighbor
        tour = [depot]
        unvisited = set(visited)
        current = depot
        while unvisited:
            nearest = min(unvisited, key=lambda m: travel_costs[f"{min(current,m)},{max(current,m)}"])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        tour.append(depot)

        # Compute tour edges
        tour_edges = []
        for idx in range(len(tour) - 1):
            a, b = tour[idx], tour[idx + 1]
            tour_edges.append([min(a, b), max(a, b)])

        # Assign purchases greedily (cheapest first)
        purchases = {}
        for k in range(num_products):
            purchases[str(k)] = {}
            demand_k = product_demands[k]
            mkts = product_markets[str(k)]
            visited_mkts = [m for m in mkts if m in set(visited)]
            sorted_mkts = sorted(visited_mkts, key=lambda m: purchase_costs[f"{k},{m}"])
            remaining = demand_k
            for m in sorted_mkts:
                if remaining <= 0:
                    break
                buy = min(remaining, supplies[f"{k},{m}"])
                if buy > 0:
                    purchases[str(k)][str(m)] = buy
                    remaining -= buy

        # Verify all demands met
        for k in range(num_products):
            total = sum(purchases[str(k)].values()) if str(k) in purchases else 0
            if total < product_demands[k]:
                return None

        # Compute objective
        obj_val = 0.0
        for i in range(len(tour) - 1):
            a, b = tour[i], tour[i + 1]
            obj_val += travel_costs[f"{min(a,b)},{max(a,b)}"]
        for k in range(num_products):
            for m_str, qty in purchases[str(k)].items():
                obj_val += purchase_costs[f"{k},{m_str}"] * qty

        return {
            "objective_value": float(obj_val),
            "visited_markets": visited,
            "tour": tour,
            "tour_edges": tour_edges,
            "purchases": purchases
        }
    except:
        return None


def local_search_improve(solution, instance, mandatory, markets, depot,
                         num_vertices, num_products, product_markets,
                         product_demands, travel_costs, purchase_costs,
                         supplies, time_limit, start_time):
    """Try to improve solution with 2-opt on tour and market removal/addition."""
    import random
    random.seed(42)

    best_solution = solution
    best_obj = solution["objective_value"]

    # 2-opt improvement on tour
    tour = solution["tour"][:]
    visited_set = set(solution["visited_markets"])

    improved = True
    iterations = 0
    max_iterations = 1000

    while improved and iterations < max_iterations:
        if time.time() - start_time > time_limit - 2:
            break
        improved = False
        iterations += 1
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour) - 1):
                # Try reversing segment between i and j
                new_tour = tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]
                # Compute cost difference
                old_cost = (travel_costs[f"{min(tour[i-1],tour[i])},{max(tour[i-1],tour[i])}"] +
                           travel_costs[f"{min(tour[j],tour[j+1])},{max(tour[j],tour[j+1])}"])
                new_cost = (travel_costs[f"{min(new_tour[i-1],new_tour[i])},{max(new_tour[i-1],new_tour[i])}"] +
                           travel_costs[f"{min(new_tour[j],new_tour[j+1])},{max(new_tour[j],new_tour[j+1])}"])
                if new_cost < old_cost - 1e-10:
                    tour = new_tour
                    improved = True
                    break
            if improved:
                break

    if improved or iterations > 0:
        # Rebuild solution with improved tour
        tour_edges = []
        for idx in range(len(tour) - 1):
            a, b = tour[idx], tour[idx + 1]
            tour_edges.append([min(a, b), max(a, b)])

        obj_val = 0.0
        for i in range(len(tour) - 1):
            a, b = tour[i], tour[i + 1]
            obj_val += travel_costs[f"{min(a,b)},{max(a,b)}"]
        for k in range(num_products):
            for m_str, qty in solution["purchases"][str(k)].items():
                obj_val += purchase_costs[f"{k},{m_str}"] * qty

        new_solution = {
            "objective_value": float(obj_val),
            "visited_markets": solution["visited_markets"],
            "tour": tour,
            "tour_edges": tour_edges,
            "purchases": solution["purchases"]
        }

        if obj_val < best_obj:
            best_solution = new_solution
            best_obj = obj_val

    # Try removing non-mandatory markets
    for m in solution["visited_markets"]:
        if m in mandatory:
            continue
        if time.time() - start_time > time_limit - 2:
            break

        # Check if we can still meet all demands without market m
        can_remove = True
        for k in range(num_products):
            mkts_k = product_markets[str(k)]
            if m not in mkts_k:
                continue
            # Check remaining supply
            total_supply = sum(supplies[f"{k},{o}"] for o in mkts_k
                             if o != m and o in visited_set)
            if total_supply < product_demands[k]:
                can_remove = False
                break

        if not can_remove:
            continue

        # Try removing market m
        new_visited = [v for v in solution["visited_markets"] if v != m]
        if not new_visited:
            continue

        # Rebuild tour without m
        new_tour_nodes = [depot] + new_visited + [depot]
        # Use nearest neighbor for new tour
        new_tour = [depot]
        unvisited = set(new_visited)
        current = depot
        while unvisited:
            nearest = min(unvisited, key=lambda v: travel_costs[f"{min(current,v)},{max(current,v)}"])
            new_tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        new_tour.append(depot)

        # Reassign purchases
        new_purchases = {}
        feasible = True
        for k in range(num_products):
            new_purchases[str(k)] = {}
            demand_k = product_demands[k]
            mkts_k = [mm for mm in product_markets[str(k)] if mm in set(new_visited)]
            sorted_mkts = sorted(mkts_k, key=lambda mm: purchase_costs[f"{k},{mm}"])
            remaining = demand_k
            for mm in sorted_mkts:
                if remaining <= 0:
                    break
                buy = min(remaining, supplies[f"{k},{mm}"])
                if buy > 0:
                    new_purchases[str(k)][str(mm)] = buy
                    remaining -= buy
            if remaining > 0:
                feasible = False
                break

        if not feasible:
            continue

        # Compute new objective
        new_obj = 0.0
        for i in range(len(new_tour) - 1):
            a, b = new_tour[i], new_tour[i + 1]
            new_obj += travel_costs[f"{min(a,b)},{max(a,b)}"]
        for k in range(num_products):
            for m_str, qty in new_purchases[str(k)].items():
                new_obj += purchase_costs[f"{k},{m_str}"] * qty

        if new_obj < best_obj:
            tour_edges = []
            for idx in range(len(new_tour) - 1):
                a, b = new_tour[idx], new_tour[idx + 1]
                tour_edges.append([min(a, b), max(a, b)])

            best_solution = {
                "objective_value": float(new_obj),
                "visited_markets": new_visited,
                "tour": new_tour,
                "tour_edges": tour_edges,
                "purchases": new_purchases
            }
            best_obj = new_obj
            visited_set = set(new_visited)

    return best_solution


if __name__ == '__main__':
    main()