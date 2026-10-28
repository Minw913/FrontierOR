import json
import argparse
import time
import random
import sys
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def build_solution(tour, dist, scores, depot):
    """
    Construct the solution dictionary from a tour (list of nodes starting at depot).
    """
    obj = sum(scores[i] for i in tour if i != depot)
    visited_vertices = list(tour)
    edges = []
    for i in range(len(tour) - 1):
        a, b = tour[i], tour[i + 1]
        edges.append([min(a, b), max(a, b)])
    a, b = tour[-1], tour[0]
    edges.append([min(a, b), max(a, b)])
    return {
        "objective_value": float(obj),
        "visited_vertices": visited_vertices,
        "tour_edges": edges
    }


def greedy_heuristic(n, depot, dist, scores, D, time_budget):
    """
    Simple greedy insertion heuristic for the Orienteering Problem.
    Returns the best tour found (list starting at depot), its score and distance.
    """
    best_tour = [depot]
    best_score = 0
    best_sol = build_solution(best_tour, dist, scores, depot)

    if time_budget <= 0:
        return best_tour, best_score, best_sol

    start_time = time.time()

    # Strategy 1: best score / extra distance ratio
    if time.time() - start_time < time_budget:
        tour = [depot]
        visited = {depot}
        current_dist = 0
        while time.time() - start_time < time_budget:
            best_node = None
            best_pos = None
            best_extra = 0
            best_val = -1.0
            for i in range(n):
                if i in visited:
                    continue
                for k in range(len(tour)):
                    u = tour[k]
                    v = tour[(k + 1) % len(tour)]
                    extra = -dist[u][v] + dist[u][i] + dist[i][v]
                    if current_dist + extra <= D:
                        val = scores[i] / extra if extra > 0 else float('inf')
                        if val > best_val:
                            best_val = val
                            best_node = i
                            best_pos = k
                            best_extra = extra
            if best_node is None:
                break
            tour.insert(best_pos + 1, best_node)
            current_dist += best_extra
            visited.add(best_node)

        score = sum(scores[i] for i in tour if i != depot)
        if score > best_score:
            best_score = score
            best_tour = tour
            best_sol = build_solution(best_tour, dist, scores, depot)

    # Strategy 2: highest absolute score first
    if time.time() - start_time < time_budget:
        tour = [depot]
        visited = {depot}
        current_dist = 0
        while True:
            best_node = None
            best_pos = None
            best_extra = 0
            best_val = -1
            for i in range(n):
                if i in visited:
                    continue
                for k in range(len(tour)):
                    u = tour[k]
                    v = tour[(k + 1) % len(tour)]
                    extra = -dist[u][v] + dist[u][i] + dist[i][v]
                    if current_dist + extra <= D:
                        val = scores[i]
                        if val > best_val:
                            best_val = val
                            best_node = i
                            best_pos = k
                            best_extra = extra
            if best_node is None:
                break
            tour.insert(best_pos + 1, best_node)
            current_dist += best_extra
            visited.add(best_node)

        score = sum(scores[i] for i in tour if i != depot)
        if score > best_score:
            best_score = score
            best_tour = tour
            best_sol = build_solution(best_tour, dist, scores, depot)

    # Strategy 3: random restarts with first-feasible insertion
    random.seed(0)
    while time.time() - start_time < time_budget:
        nodes = [i for i in range(n) if i != depot]
        random.shuffle(nodes)
        tour = [depot]
        visited = {depot}
        current_dist = 0
        for i in nodes:
            if i in visited:
                continue
            for k in range(len(tour)):
                u = tour[k]
                v = tour[(k + 1) % len(tour)]
                extra = -dist[u][v] + dist[u][i] + dist[i][v]
                if current_dist + extra <= D:
                    tour.insert(k + 1, i)
                    current_dist += extra
                    visited.add(i)
                    break
        score = sum(scores[i] for i in tour if i != depot)
        if score > best_score:
            best_score = score
            best_tour = tour
            best_sol = build_solution(best_tour, dist, scores, depot)

    return best_tour, best_score, best_sol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['num_nodes']
    D = instance['distance_limitation_d0']
    depot = instance['depot_vertex']
    scores = instance['scores']
    dist = instance['distance_matrix']

    logger = None
    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="maximize")

    start_time = time.time()
    time_limit = args.time_limit

    # Trivial case: only the depot exists
    if n == 1:
        sol = build_solution([depot], dist, scores, depot)
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f)
        if logger:
            logger.log_solution(0.0, sol)
        return

    # Run heuristic for up to 20% of time limit or 2 seconds
    heuristic_time = min(2.0, time_limit * 0.2)
    best_tour, best_score, best_sol = greedy_heuristic(
        n, depot, dist, scores, D, heuristic_time
    )
    if logger:
        logger.log_solution(float(best_score), best_sol)

    # Solve with Gurobi if time remains
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 0.5:
        try:
            model = gp.Model("OP")
            model.setParam('Seed', 0)
            model.setParam('MIPGap', 1e-4)
            model.setParam('NumericFocus', 0)
            model.setParam('Threads', 1)
            model.setParam('OutputFlag', 0)

            # Variables
            y = model.addVars(range(n), vtype=GRB.BINARY, name="y")
            x = model.addVars([(i, j) for i in range(n) for j in range(n) if i != j],
                              vtype=GRB.BINARY, name="x")
            f = model.addVars([(i, j) for i in range(n) for j in range(n) if i != j],
                              vtype=GRB.CONTINUOUS, name="f")

            # Depot must be visited
            model.addConstr(y[depot] == 1, name="depot_visit")

            # Degree constraints
            model.addConstrs(
                (sum(x[i, j] for j in range(n) if i != j) == y[i] for i in range(n)),
                name="out"
            )
            model.addConstrs(
                (sum(x[j, i] for j in range(n) if i != j) == y[i] for i in range(n)),
                name="in"
            )

            # Distance budget
            model.addConstr(
                sum(dist[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j) <= D,
                name="distance"
            )

            # Flow constraints to enforce a single cycle through the depot
            depot_flow = sum(y[i] for i in range(n) if i != depot)
            model.addConstr(
                sum(f[depot, j] for j in range(n) if j != depot) -
                sum(f[j, depot] for j in range(n) if j != depot) == depot_flow,
                name="flow_depot"
            )
            for i in range(n):
                if i != depot:
                    model.addConstr(
                        sum(f[i, j] for j in range(n) if i != j) -
                        sum(f[j, i] for j in range(n) if j != i) == -y[i],
                        name=f"flow_{i}"
                    )
            model.addConstrs(
                (f[i, j] <= (n - 1) * x[i, j] for i in range(n) for j in range(n) if i != j),
                name="cap"
            )
            model.addConstrs(
                (f[i, j] >= 0 for i in range(n) for j in range(n) if i != j),
                name="f_pos"
            )

            # Objective: maximize collected scores
            model.setObjective(
                sum(scores[i] * y[i] for i in range(n) if i != depot),
                GRB.MAXIMIZE
            )

            # Provide MIP start from heuristic
            if best_tour is not None:
                for i in range(n):
                    y[i].Start = 1.0 if i in best_tour else 0.0
                for (i, j) in x:
                    x[i, j].Start = 0.0
                for k in range(len(best_tour) - 1):
                    x[best_tour[k], best_tour[k + 1]].Start = 1.0
                x[best_tour[-1], best_tour[0]].Start = 1.0
                # Set flow values along the tour
                num_visited = len(best_tour) - 1
                flow_val = float(num_visited)
                for k in range(len(best_tour) - 1):
                    i = best_tour[k]
                    j = best_tour[k + 1]
                    f[i, j].Start = flow_val
                    flow_val -= 1.0
                f[best_tour[-1], best_tour[0]].Start = 0.0

            model.setParam('TimeLimit', max(0.0, remaining_time - 0.1))

            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if logger:
                        y_vals = model.cbGetSolution(y)
                        x_vals = model.cbGetSolution(x)
                        visited = [i for i in range(n) if y_vals[i] > 0.5]
                        next_node = {}
                        for (i, j) in x:
                            if x_vals[i, j] > 0.5:
                                next_node[i] = j
                        tour = [depot]
                        curr = depot
                        while True:
                            curr = next_node.get(curr, None)
                            if curr is None or curr == depot:
                                break
                            tour.append(curr)
                        if len(tour) != len(visited):
                            tour = [depot] + [i for i in visited if i != depot]
                        sol = build_solution(tour, dist, scores, depot)
                        logger.log_solution(obj, sol)

            model.optimize(callback)

            if model.SolCount > 0:
                y_vals = model.getAttr('X', y)
                x_vals = model.getAttr('X', x)
                visited = [i for i in range(n) if y_vals[i] > 0.5]
                next_node = {}
                for (i, j) in x:
                    if x_vals[i, j] > 0.5:
                        next_node[i] = j
                tour = [depot]
                curr = depot
                while True:
                    curr = next_node.get(curr, None)
                    if curr is None or curr == depot:
                        break
                    tour.append(curr)
                if len(tour) != len(visited):
                    tour = [depot] + [i for i in visited if i != depot]
                obj = model.ObjVal
                if obj > best_score:
                    best_score = obj
                    best_tour = tour
                    best_sol = build_solution(best_tour, dist, scores, depot)
                    if logger:
                        logger.log_solution(float(best_score), best_sol)
        except Exception:
            # If Gurobi fails for any reason, fall back to heuristic solution
            pass

    # Write final solution
    with open(args.solution_path, 'w') as f:
        json.dump(best_sol, f)


if __name__ == "__main__":
    main()