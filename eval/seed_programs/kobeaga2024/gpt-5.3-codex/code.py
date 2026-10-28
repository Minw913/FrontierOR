import argparse
import json
from typing import List, Tuple, Set

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def build_greedy_insertion_tour(
    n: int,
    depot: int,
    budget: int,
    scores: List[int],
    dist: List[List[int]],
) -> Tuple[List[int], int, int]:
    """
    Greedy insertion heuristic for a feasible cycle:
    route is represented as [depot, ..., depot].
    Returns (route, total_distance, total_score).
    """
    route = [depot, depot]
    visited: Set[int] = {depot}
    total_dist = 0
    total_score = 0

    while True:
        best_node = None
        best_pos = None
        best_delta = None
        best_value = -1.0

        for v in range(n):
            if v in visited:
                continue
            if scores[v] <= 0:
                continue

            local_best_delta = None
            local_best_pos = None

            for p in range(len(route) - 1):
                a = route[p]
                b = route[p + 1]
                delta = dist[a][v] + dist[v][b] - dist[a][b]
                if local_best_delta is None or delta < local_best_delta:
                    local_best_delta = delta
                    local_best_pos = p + 1

            if local_best_delta is None:
                continue
            if total_dist + local_best_delta > budget:
                continue

            # score per extra distance
            value = scores[v] / (local_best_delta + 1e-6)
            if value > best_value or (
                abs(value - best_value) <= 1e-12
                and (best_node is None or scores[v] > scores[best_node])
            ):
                best_value = value
                best_node = v
                best_pos = local_best_pos
                best_delta = local_best_delta

        if best_node is None:
            break

        route.insert(best_pos, best_node)
        visited.add(best_node)
        total_dist += best_delta
        total_score += scores[best_node]

    return route, total_dist, total_score


def incumbent_callback(model, where):
    if where == GRB.Callback.MIPSOL and model._logger is not None:
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        if model._best_inc is None or obj > model._best_inc + 1e-6:
            model._best_inc = obj
            model._logger.log(float(obj))


def solve_instance(instance: dict, time_limit: int, logger: SolutionLogger = None) -> dict:
    n = int(instance["num_nodes"])
    budget = int(instance["distance_limitation_d0"])
    depot = int(instance["depot_vertex"])
    scores = [int(v) for v in instance["scores"]]
    dist = instance["distance_matrix"]

    # Greedy heuristic (also used as fallback and MIP start)
    heuristic_route, heuristic_len, heuristic_obj = build_greedy_insertion_tour(
        n, depot, budget, scores, dist
    )

    if logger and heuristic_obj > 0:
        logger.log(float(heuristic_obj))

    model = gp.Model("profitable_tour")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = max(1, int(time_limit))
    model.Params.Threads = 1

    nodes = range(n)
    non_depot = [i for i in nodes if i != depot]
    arcs = [(i, j) for i in nodes for j in nodes if i != j]

    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")  # directed tour arc usage
    y = model.addVars(nodes, vtype=GRB.BINARY, name="y")  # visit indicator
    f = model.addVars(arcs, lb=0.0, vtype=GRB.CONTINUOUS, name="f")  # flow for connectivity

    # Objective: maximize collected score
    model.setObjective(gp.quicksum(scores[i] * y[i] for i in nodes), GRB.MAXIMIZE)

    # Depot must be visited
    model.addConstr(y[depot] == 1, name="depot_visit")

    # In/out degree constraints
    for i in non_depot:
        model.addConstr(gp.quicksum(x[i, j] for j in nodes if j != i) == y[i], name=f"out_{i}")
        model.addConstr(gp.quicksum(x[j, i] for j in nodes if j != i) == y[i], name=f"in_{i}")

    model.addConstr(
        gp.quicksum(x[depot, j] for j in nodes if j != depot) == 1, name="depot_out"
    )
    model.addConstr(
        gp.quicksum(x[j, depot] for j in nodes if j != depot) == 1, name="depot_in"
    )

    # Distance budget
    model.addConstr(
        gp.quicksum(dist[i][j] * x[i, j] for (i, j) in arcs) <= budget,
        name="distance_budget",
    )

    # Flow-based connectivity: depot sends one unit to every visited non-depot node
    for i in non_depot:
        model.addConstr(
            gp.quicksum(f[j, i] for j in nodes if j != i)
            - gp.quicksum(f[i, j] for j in nodes if j != i)
            == y[i],
            name=f"flow_bal_{i}",
        )

    model.addConstr(
        gp.quicksum(f[depot, j] for j in nodes if j != depot)
        - gp.quicksum(f[j, depot] for j in nodes if j != depot)
        == gp.quicksum(y[i] for i in non_depot),
        name="flow_bal_depot",
    )

    cap = max(1, n - 1)
    for (i, j) in arcs:
        model.addConstr(f[i, j] <= cap * x[i, j], name=f"cap_{i}_{j}")

    # MIP start from heuristic if it includes at least one non-depot node
    if len(heuristic_route) >= 3:
        visited_h = set(heuristic_route[:-1])  # exclude repeated last depot
        for i in nodes:
            y[i].Start = 1.0 if i in visited_h else 0.0
        for (a, b) in zip(heuristic_route[:-1], heuristic_route[1:]):
            if a != b:
                x[a, b].Start = 1.0

    model._logger = logger
    model._best_inc = float(heuristic_obj) if heuristic_obj > 0 else None

    model.optimize(incumbent_callback)

    if model.SolCount > 0:
        visited_vertices = [i for i in nodes if y[i].X > 0.5]
        if depot not in visited_vertices:
            visited_vertices.append(depot)
        visited_vertices.sort()

        tour_edges = []
        for (i, j) in arcs:
            if x[i, j].X > 0.5:
                tour_edges.append([int(i), int(j)])

        objective_value = float(sum(scores[i] for i in visited_vertices))
        return {
            "objective_value": objective_value,
            "visited_vertices": visited_vertices,
            "tour_edges": tour_edges,
        }

    # Fallback if no feasible MIP solution returned
    if len(heuristic_route) >= 3:
        visited_vertices = sorted(set(heuristic_route[:-1]))
        tour_edges = [[int(a), int(b)] for a, b in zip(heuristic_route[:-1], heuristic_route[1:])]
        objective_value = float(sum(scores[i] for i in visited_vertices))
    else:
        visited_vertices = [int(depot)]
        tour_edges = []
        objective_value = 0.0

    return {
        "objective_value": objective_value,
        "visited_vertices": visited_vertices,
        "tour_edges": tour_edges,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    solution = solve_instance(instance, args.time_limit, logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()