import argparse
import json
import time
from itertools import combinations
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def parse_instance(path):
    with open(path, "r") as f:
        data = json.load(f)

    instance = {}
    instance["num_vertices"] = data["num_vertices"]
    instance["num_markets"] = data["num_markets"]
    instance["num_products"] = data["num_products"]
    instance["depot"] = data["depot"]
    instance["coordinates"] = data.get("coordinates")
    instance["tau"] = data.get("tau")

    travel_costs = {}
    for key, val in data["travel_costs"].items():
        i, j = map(int, key.split(","))
        travel_costs[(min(i, j), max(i, j))] = val
    instance["travel_costs"] = travel_costs

    instance["product_demands"] = data["product_demands"]

    product_markets = {}
    for key, val in data["product_markets"].items():
        p = int(key)
        product_markets[p] = val
    instance["product_markets"] = product_markets

    purchase_costs = {}
    for key, val in data["purchase_costs"].items():
        p, i = map(int, key.split(","))
        purchase_costs[(p, i)] = val
    instance["purchase_costs"] = purchase_costs

    supplies = {}
    for key, val in data["supplies"].items():
        p, i = map(int, key.split(","))
        supplies[(p, i)] = val
    instance["supplies"] = supplies

    return instance


def get_cost(i, j, travel_costs):
    return travel_costs[(min(i, j), max(i, j))]


def compute_mandatory(data):
    mandatory = set()
    for p, markets in data["product_markets"].items():
        demand = data["product_demands"][p]
        total_supply = sum(data["supplies"][(p, i)] for i in markets)
        for i in markets:
            if total_supply - data["supplies"][(p, i)] < demand:
                mandatory.add(i)
    return mandatory


def nearest_neighbor(nodes, travel_costs):
    depot = nodes[0]
    unvisited = set(nodes[1:])
    tour = [depot]
    current = depot
    while unvisited:
        next_node = min(unvisited, key=lambda x: get_cost(current, x, travel_costs))
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    tour.append(depot)
    return tour


def two_opt(tour, travel_costs):
    n = len(tour) - 1
    if n <= 3:
        return tour
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a = tour[i - 1]
                b = tour[i]
                c = tour[j]
                d = tour[j + 1]
                cost_ab = get_cost(a, b, travel_costs)
                cost_cd = get_cost(c, d, travel_costs)
                cost_ac = get_cost(a, c, travel_costs)
                cost_bd = get_cost(b, d, travel_costs)
                if cost_ac + cost_bd < cost_ab + cost_cd - 1e-9:
                    tour[i : j + 1] = reversed(tour[i : j + 1])
                    improved = True
    return tour


def used_markets_and_purchases(data, available_markets):
    P = range(data["num_products"])
    purchases = {p: {} for p in P}
    used = set()
    purchase_cost = 0.0
    for p in P:
        demand = data["product_demands"][p]
        markets = [i for i in data["product_markets"][p] if i in available_markets]
        markets.sort(key=lambda i: data["purchase_costs"][(p, i)])
        remaining = demand
        for i in markets:
            if remaining <= 0:
                break
            cap = data["supplies"][(p, i)]
            buy = min(remaining, cap)
            if buy > 0:
                purchases[p][i] = buy
                used.add(i)
                remaining -= buy
                purchase_cost += buy * data["purchase_costs"][(p, i)]
    return used, purchases, purchase_cost


def total_cost_for_set(data, available_markets):
    used, purchases, pcost = used_markets_and_purchases(data, available_markets)
    if not used:
        return float("inf"), None
    nodes = [data["depot"]] + sorted(used)
    tour = nearest_neighbor(nodes, data["travel_costs"])
    tour = two_opt(tour, data["travel_costs"])
    travel_cost = sum(
        get_cost(tour[k], tour[k + 1], data["travel_costs"])
        for k in range(len(tour) - 1)
    )
    total = pcost + travel_cost
    sol = {
        "used": used,
        "purchases": purchases,
        "tour": tour,
        "travel_cost": travel_cost,
        "purchase_cost": pcost,
    }
    return total, sol


def heuristic_local_search(data, time_limit, start_time):
    all_markets = set()
    for p in range(data["num_products"]):
        for i in data["product_markets"][p]:
            all_markets.add(i)
    mandatory = compute_mandatory(data)
    S = set(all_markets)
    best_cost, best_sol = total_cost_for_set(data, S)
    if best_sol is None:
        # Fallback: just try mandatory markets (should be feasible)
        S = set(mandatory)
        best_cost, best_sol = total_cost_for_set(data, S)
        if best_sol is None:
            return float("inf"), None

    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        used = best_sol["used"]
        # Try removing a market
        for i in sorted(used):
            if i in mandatory:
                continue
            S_new = S - {i}
            cost, sol = total_cost_for_set(data, S_new)
            if cost < best_cost - 1e-6:
                best_cost, best_sol, S = cost, sol, S_new
                improved = True
                break
        if improved:
            continue
        # Try adding a market
        for i in sorted(all_markets - used):
            S_new = S | {i}
            cost, sol = total_cost_for_set(data, S_new)
            if cost < best_cost - 1e-6:
                best_cost, best_sol, S = cost, sol, S_new
                improved = True
                break
    return best_cost, best_sol


def build_solution_dict(obj, visited_markets, tour, tour_edges, purchases):
    return {
        "objective_value": float(obj),
        "visited_markets": sorted(visited_markets),
        "tour": tour,
        "tour_edges": tour_edges,
        "purchases": purchases,
    }


def find_components(vertices, edges):
    adj = {v: set() for v in vertices}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    visited = set()
    comps = []
    for v in vertices:
        if v not in visited:
            comp = set()
            stack = [v]
            while stack:
                node = stack.pop()
                if node not in comp:
                    comp.add(node)
                    visited.add(node)
                    for nb in adj[node]:
                        if nb not in comp:
                            stack.append(nb)
            comps.append(comp)
    return comps


def build_tour_from_x(depot, V, x_vals):
    adj = {v: [] for v in V}
    for (i, j) in x_vals:
        if x_vals[(i, j)] > 0.5:
            adj[i].append(j)
            adj[j].append(i)
    for v in V:
        if len(adj[v]) not in (0, 2):
            return None
    if len(adj[depot]) != 2:
        return None
    tour = [depot]
    prev = depot
    curr = adj[depot][0]
    while curr != depot:
        tour.append(curr)
        next_node = None
        for nb in adj[curr]:
            if nb != prev:
                next_node = nb
                break
        if next_node is None:
            return None
        prev, curr = curr, next_node
    tour.append(depot)
    return tour


def solve_mip(data, heur_sol, time_limit, logger, start_time):
    depot = data["depot"]
    P = range(data["num_products"])
    markets = set()
    for p in P:
        for i in data["product_markets"][p]:
            markets.add(i)
    markets = sorted(markets)
    V = [depot] + markets
    n_V = len(V)

    mandatory = compute_mandatory(data)

    model = gp.Model("TPP")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.LazyConstraints = 1

    # Variables
    y = {i: model.addVar(vtype=GRB.BINARY, name=f"y_{i}") for i in markets}
    x = {}
    for i_idx, i in enumerate(V):
        for j_idx in range(i_idx + 1, n_V):
            j = V[j_idx]
            x[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    q = {}
    for p in P:
        for i in data["product_markets"][p]:
            q[(p, i)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"q_{p}_{i}")

    model.update()

    # Objective
    travel_obj = gp.quicksum(data["travel_costs"][(i, j)] * x[(i, j)] for (i, j) in x)
    purchase_obj = gp.quicksum(
        data["purchase_costs"][(p, i)] * q[(p, i)] for (p, i) in q
    )
    model.setObjective(travel_obj + purchase_obj, GRB.MINIMIZE)

    # Degree constraints
    for i in V:
        degree = gp.quicksum(x[(min(i, j), max(i, j))] for j in V if j != i)
        if i == depot:
            model.addConstr(degree == 2, name=f"degree_{i}")
        else:
            model.addConstr(degree == 2 * y[i], name=f"degree_{i}")

    # Mandatory markets
    for i in mandatory:
        if i in y:
            model.addConstr(y[i] == 1, name=f"mandatory_{i}")

    # Demand constraints
    for p in P:
        model.addConstr(
            gp.quicksum(q[(p, i)] for i in data["product_markets"][p]) == data["product_demands"][p],
            name=f"demand_{p}",
        )

    # Supply constraints
    for p in P:
        for i in data["product_markets"][p]:
            model.addConstr(
                q[(p, i)] <= data["supplies"][(p, i)] * y[i],
                name=f"supply_{p}_{i}",
            )

    # Warm start
    if heur_sol is not None:
        heur_dict = heur_sol["solution_dict"]
        heur_visited = set(heur_dict["visited_markets"])
        heur_tour_edges = [tuple(e) for e in heur_dict["tour_edges"]]
        heur_purchases = heur_dict["purchases"]
        for i in markets:
            y[i].Start = 1 if i in heur_visited else 0
        for (i, j) in x:
            edge = (min(i, j), max(i, j))
            if edge in heur_tour_edges:
                x[(i, j)].Start = 1
            else:
                x[(i, j)].Start = 0
        for (p, i) in q:
            if str(p) in heur_purchases and str(i) in heur_purchases[str(p)]:
                q[(p, i)].Start = heur_purchases[str(p)][str(i)]
            else:
                q[(p, i)].Start = 0

    # Prepare lists for callback
    x_vars_list = [x[(i, j)] for (i, j) in x]
    x_keys = list(x.keys())
    y_vars_list = [y[i] for i in markets]
    y_keys = list(y.keys())
    q_vars_list = [q[(p, i)] for (p, i) in q]
    q_keys = list(q.keys())

    def subtour_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            x_vals_list = model.cbGetSolution(model._x_vars_list)
            x_vals = dict(zip(model._x_keys, x_vals_list))
            y_vals_list = model.cbGetSolution(model._y_vars_list)
            y_vals = dict(zip(model._y_keys, y_vals_list))
            q_vals_list = model.cbGetSolution(model._q_vars_list)
            q_vals = dict(zip(model._q_keys, q_vals_list))

            edges = []
            for (i, j) in model._x_keys:
                if x_vals[(i, j)] > 0.5:
                    edges.append((i, j))

            comps = find_components(model._V, edges)
            for comp in comps:
                if model._depot not in comp:
                    model.cbLazy(
                        gp.quicksum(
                            model._x[(min(i, j), max(i, j))]
                            for i in comp
                            for j in comp
                            if i < j
                        )
                        <= len(comp) - 1
                    )

            # Try to build full tour for logging
            visited_markets = [i for i in model._markets if y_vals[i] > 0.5]
            depot_comp = None
            for comp in comps:
                if model._depot in comp:
                    depot_comp = comp
                    break
            can_log_full = False
            if depot_comp is not None and set(visited_markets).issubset(depot_comp):
                tour = build_tour_from_x(model._depot, model._V, x_vals)
                if tour is not None and len(tour) == len(visited_markets) + 2:
                    can_log_full = True

            if can_log_full and model._logger is not None:
                tour_edges = []
                for k in range(len(tour) - 1):
                    u, v = tour[k], tour[k + 1]
                    if u > v:
                        u, v = v, u
                    tour_edges.append([u, v])
                purchases = {}
                for p in model._P:
                    purchases[str(p)] = {}
                    for i in model._product_markets[p]:
                        val = q_vals[(p, i)]
                        if val > 1e-6:
                            purchases[str(p)][str(i)] = float(val)
                sol_dict = {
                    "objective_value": float(obj_val),
                    "visited_markets": sorted(visited_markets),
                    "tour": tour,
                    "tour_edges": tour_edges,
                    "purchases": purchases,
                }
                model._logger.log_solution(float(obj_val), sol_dict)
            elif model._logger is not None:
                model._logger.log(float(obj_val))

    model._x = x
    model._x_vars_list = x_vars_list
    model._x_keys = x_keys
    model._y = y
    model._y_vars_list = y_vars_list
    model._y_keys = y_keys
    model._q = q
    model._q_vars_list = q_vars_list
    model._q_keys = q_keys
    model._V = V
    model._depot = depot
    model._markets = markets
    model._P = P
    model._product_markets = data["product_markets"]
    model._logger = logger

    time_left = max(1.0, time_limit - (time.time() - start_time))
    model.Params.TimeLimit = time_left
    model.optimize(subtour_callback)

    if model.SolCount > 0:
        obj_val = model.ObjVal
        y_vals = {i: y[i].X for i in markets}
        x_vals = {(i, j): x[(i, j)].X for (i, j) in x}
        q_vals = {(p, i): q[(p, i)].X for (p, i) in q}
        visited_markets = [i for i in markets if y_vals[i] > 0.5]
        tour = build_tour_from_x(depot, V, x_vals)
        if tour is not None and len(tour) == len(visited_markets) + 2:
            tour_edges = []
            for k in range(len(tour) - 1):
                u, v = tour[k], tour[k + 1]
                if u > v:
                    u, v = v, u
                tour_edges.append([u, v])
            purchases = {}
            for p in P:
                purchases[str(p)] = {}
                for i in data["product_markets"][p]:
                    val = q_vals[(p, i)]
                    if val > 1e-6:
                        purchases[str(p)][str(i)] = float(val)
            return {
                "objective_value": float(obj_val),
                "visited_markets": sorted(visited_markets),
                "tour": tour,
                "tour_edges": tour_edges,
                "purchases": purchases,
            }
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    data = parse_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start_time = time.time()

    # Heuristic solution
    heur_cost, heur_sol = heuristic_local_search(data, args.time_limit, start_time)
    if heur_sol is not None:
        heur_dict = build_solution_dict(
            heur_cost,
            heur_sol["used"],
            heur_sol["tour"],
            [
                [min(heur_sol["tour"][k], heur_sol["tour"][k + 1]),
                 max(heur_sol["tour"][k], heur_sol["tour"][k + 1])]
                for k in range(len(heur_sol["tour"]) - 1)
            ],
            {str(p): {str(i): float(v) for i, v in heur_sol["purchases"][p].items()}
             for p in range(data["num_products"])},
        )
        if logger:
            logger.log_solution(float(heur_cost), heur_dict)
    else:
        heur_dict = None

    # Exact MIP (if time permits)
    try:
        mip_sol = solve_mip(data, {"solution_dict": heur_dict}, args.time_limit, logger, start_time)
    except Exception:
        mip_sol = None

    if mip_sol is not None:
        final_sol = mip_sol
    elif heur_dict is not None:
        final_sol = heur_dict
    else:
        # Absolute fallback (should not happen)
        final_sol = {
            "objective_value": 0.0,
            "visited_markets": [],
            "tour": [data["depot"]],
            "tour_edges": [],
            "purchases": {},
        }

    with open(args.solution_path, "w") as f:
        json.dump(final_sol, f, indent=2)


if __name__ == "__main__":
    main()