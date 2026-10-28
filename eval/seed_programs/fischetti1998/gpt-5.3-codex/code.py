import argparse
import json
import time
from itertools import combinations

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_pair_key(key: str):
    s = key.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    a, b = s.split(",")
    return int(a.strip()), int(b.strip())


def build_distance_lookup(travel_times_raw):
    d = {}
    for k, v in travel_times_raw.items():
        i, j = parse_pair_key(k)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        d[(i, j)] = float(v)
    return d


def dist(d, i, j):
    if i == j:
        return 0.0
    a, b = (i, j) if i < j else (j, i)
    return d[(a, b)]


def greedy_initial_tour(cities, depot, prizes, d, t0):
    non_depot = [i for i in cities if i != depot]
    if len(non_depot) < 2:
        return None, None, None

    # Best 2-city cycle: depot -> i -> j -> depot
    best_pair = None
    best_prize = -1.0
    best_cost = None

    for i, j in combinations(non_depot, 2):
        c = dist(d, depot, i) + dist(d, i, j) + dist(d, j, depot)
        if c <= t0 + 1e-9:
            p = prizes.get(i, 0.0) + prizes.get(j, 0.0)
            if p > best_prize or (abs(p - best_prize) <= 1e-9 and (best_cost is None or c < best_cost)):
                best_prize = p
                best_pair = (i, j)
                best_cost = c

    if best_pair is None:
        return None, None, None

    route = [depot, best_pair[0], best_pair[1], depot]
    in_route = set(route[:-1])
    total_cost = best_cost
    total_prize = prizes.get(best_pair[0], 0.0) + prizes.get(best_pair[1], 0.0)

    # Greedy insertion by prize / added_cost
    while True:
        best_move = None
        best_score = -1e18

        for k in non_depot:
            if k in in_route:
                continue
            pk = prizes.get(k, 0.0)
            if pk <= 0:
                continue

            for pos in range(len(route) - 1):
                a, b = route[pos], route[pos + 1]
                add = dist(d, a, k) + dist(d, k, b) - dist(d, a, b)
                new_cost = total_cost + add
                if new_cost <= t0 + 1e-9:
                    score = pk / max(add, 1e-6)
                    if score > best_score:
                        best_score = score
                        best_move = (k, pos + 1, add, pk)

        if best_move is None:
            break

        k, ins_pos, add, pk = best_move
        route.insert(ins_pos, k)
        in_route.add(k)
        total_cost += add
        total_prize += pk

    return route, total_cost, total_prize


def extract_solution(model, x_vars, y_vars, cities, depot, prizes):
    if model.SolCount == 0:
        return {
            "objective_value": 0.0,
            "visited_nodes": [depot],
            "edges": [],
            "tour": [depot, depot],
        }

    selected_edges = []
    for (i, j), var in x_vars.items():
        if var.X > 0.5:
            selected_edges.append((i, j))

    visited = [i for i in cities if y_vars[i].X > 0.5]
    visited_set = set(visited)

    # Build adjacency from selected edges
    adj = {i: [] for i in visited_set}
    for i, j in selected_edges:
        if i in adj and j in adj:
            adj[i].append(j)
            adj[j].append(i)

    # Reconstruct cycle starting at depot
    tour = [depot]
    if depot in adj and len(adj[depot]) >= 1:
        prev = None
        curr = depot
        max_steps = len(visited_set) + 5
        steps = 0
        while steps < max_steps:
            nbrs = adj.get(curr, [])
            if not nbrs:
                break
            if prev is None:
                nxt = nbrs[0]
            else:
                if len(nbrs) == 1:
                    nxt = nbrs[0]
                else:
                    nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]

            tour.append(nxt)
            if nxt == depot:
                break
            prev, curr = curr, nxt
            steps += 1

    if len(tour) < 2 or tour[-1] != depot:
        # Fallback if reconstruction fails
        ordered = [depot] + sorted([i for i in visited if i != depot]) + [depot]
        tour = ordered

    visited_nodes = tour[:-1] if len(tour) >= 2 else visited
    if depot not in visited_nodes:
        visited_nodes = [depot] + [i for i in visited_nodes if i != depot]

    objective_value = sum(prizes.get(i, 0.0) for i in set(visited_nodes) if i != depot)

    return {
        "objective_value": float(objective_value),
        "visited_nodes": [int(i) for i in visited_nodes],
        "edges": [[int(i), int(j)] for (i, j) in selected_edges],
        "tour": [int(i) for i in tour],
    }


def solve(instance, time_limit, logger=None):
    start_time = time.time()

    n = int(instance["n"])
    depot = int(instance["depot"])
    t0 = float(instance["t0"])

    prizes_raw = instance["prizes"]
    prizes = {int(k): float(v) for k, v in prizes_raw.items()}
    if depot not in prizes:
        prizes[depot] = 0.0

    cities = list(range(1, n + 1))
    for i in cities:
        prizes.setdefault(i, 0.0)

    d = build_distance_lookup(instance["travel_times"])
    edge_list = sorted(d.keys())

    incident = {i: [] for i in cities}
    for e in edge_list:
        i, j = e
        incident[i].append(e)
        incident[j].append(e)

    model = gp.Model("orienteering_cycle")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.MIPFocus = 1
    model.Params.LazyConstraints = 1

    # Keep global time cap including model building
    elapsed = time.time() - start_time
    remaining = max(0.0, time_limit - elapsed)
    model.Params.TimeLimit = remaining

    x = model.addVars(edge_list, vtype=GRB.BINARY, name="x")
    y = model.addVars(cities, vtype=GRB.BINARY, name="y")

    # Objective: maximize total collected prize (exclude depot prize if any)
    model.setObjective(gp.quicksum(prizes[i] * y[i] for i in cities if i != depot), GRB.MAXIMIZE)

    # Depot must be visited
    model.addConstr(y[depot] == 1, name="depot_visited")

    # Degree constraints: visited nodes have degree 2, others 0
    for i in cities:
        model.addConstr(gp.quicksum(x[e] for e in incident[i]) == 2 * y[i], name=f"deg_{i}")

    # Time budget
    model.addConstr(gp.quicksum(d[e] * x[e] for e in edge_list) <= t0, name="time_budget")

    # At least 3 visited cities including depot
    model.addConstr(gp.quicksum(y[i] for i in cities) >= 3, name="min_nodes")

    # Warm start from greedy heuristic
    best_logged = {"val": -1e100}
    route, _, route_prize = greedy_initial_tour(cities, depot, prizes, d, t0)
    if route is not None:
        start_edges = set()
        start_nodes = set(route[:-1])
        for idx in range(len(route) - 1):
            a, b = route[idx], route[idx + 1]
            if a > b:
                a, b = b, a
            start_edges.add((a, b))

        for i in cities:
            y[i].Start = 1.0 if i in start_nodes else 0.0
        for e in edge_list:
            x[e].Start = 1.0 if e in start_edges else 0.0

        if logger and route_prize is not None:
            logger.log(float(route_prize))
            best_logged["val"] = float(route_prize)

    x_keys_set = set(edge_list)

    def callback(cb_model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if logger and obj > best_logged["val"] + 1e-9:
                    logger.log(float(obj))
                    best_logged["val"] = float(obj)

                y_sol = cb_model.cbGetSolution(y)
                x_sol = cb_model.cbGetSolution(x)

                visited = [i for i in cities if y_sol[i] > 0.5]
                if depot not in visited:
                    return
                if len(visited) <= 1:
                    return

                visited_set = set(visited)
                adj = {i: [] for i in visited}
                for (i, j), val in x_sol.items():
                    if val > 0.5 and i in visited_set and j in visited_set:
                        adj[i].append(j)
                        adj[j].append(i)

                # Find connected components among visited nodes
                comps = []
                unseen = set(visited)
                while unseen:
                    s = unseen.pop()
                    stack = [s]
                    comp = {s}
                    while stack:
                        u = stack.pop()
                        for v in adj.get(u, []):
                            if v in unseen:
                                unseen.remove(v)
                                comp.add(v)
                                stack.append(v)
                    comps.append(comp)

                if len(comps) <= 1:
                    return

                for comp in comps:
                    if depot in comp:
                        continue
                    comp_list = sorted(comp)
                    if len(comp_list) <= 1:
                        continue

                    lhs_terms = []
                    for idx_i in range(len(comp_list)):
                        i = comp_list[idx_i]
                        for idx_j in range(idx_i + 1, len(comp_list)):
                            j = comp_list[idx_j]
                            e = (i, j) if i < j else (j, i)
                            if e in x_keys_set:
                                lhs_terms.append(x[e])

                    if lhs_terms:
                        cb_model.cbLazy(
                            gp.quicksum(lhs_terms) <= gp.quicksum(y[i] for i in comp_list) - 1
                        )
            except gp.GurobiError:
                return

    # Update remaining time right before optimize
    elapsed = time.time() - start_time
    remaining = max(0.0, time_limit - elapsed)
    model.Params.TimeLimit = remaining

    if remaining > 0:
        model.optimize(callback)
    else:
        model.update()

    return extract_solution(model, x, y, cities, depot, prizes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    solution = solve(instance, time_limit=int(args.time_limit), logger=logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()