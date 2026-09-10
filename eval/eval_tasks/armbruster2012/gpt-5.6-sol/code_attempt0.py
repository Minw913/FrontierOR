import argparse
import json
import math
import random
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = int(data["num_nodes"])
    capacity = int(data["bisection_capacity_F"])
    weights = [int(w) for w in data["node_weights"]]
    total_weight = sum(weights)
    lower_weight = total_weight - capacity

    if n <= 0:
        raise ValueError("The instance must contain at least one node")

    # Aggregate parallel edges. Self-loops never contribute to a cut.
    edge_costs = defaultdict(float)
    for endpoints, cost in zip(data["edges"], data["edge_weights"]):
        u, v = int(endpoints[0]), int(endpoints[1])
        if u == v:
            continue
        if u > v:
            u, v = v, u
        edge_costs[(u, v)] += float(cost)

    aggregated_edges = [
        (u, v, c) for (u, v), c in edge_costs.items() if abs(c) > 1e-15
    ]

    adjacency = [[] for _ in range(n)]
    for u, v, c in aggregated_edges:
        adjacency[u].append((v, c))
        adjacency[v].append((u, c))

    def objective_from_assignment(x):
        return float(sum(c for u, v, c in aggregated_edges if x[u] != x[v]))

    def is_feasible(x):
        if len(x) != n or x[0] != 1:
            return False
        side_weight = sum(weights[i] for i in range(n) if x[i])
        return lower_weight <= side_weight <= capacity

    def make_solution(x, objective=None):
        if objective is None:
            objective = objective_from_assignment(x)
        partition_s = [i for i in range(n) if x[i]]
        partition_complement = [i for i in range(n) if not x[i]]
        return {
            "objective_value": float(objective),
            "partition_S": partition_s,
            "partition_complement": partition_complement,
        }

    state = {
        "best_obj": math.inf,
        "best_x": None,
    }

    def consider_assignment(x, known_objective=None):
        if not is_feasible(x):
            return False

        objective = (
            objective_from_assignment(x)
            if known_objective is None
            else float(known_objective)
        )

        tolerance = 1e-9 * max(1.0, abs(state["best_obj"])) if math.isfinite(
            state["best_obj"]
        ) else 0.0

        if state["best_x"] is None or objective < state["best_obj"] - tolerance:
            candidate = list(map(int, x))
            # Recompute to keep the reported value exactly consistent with the cut.
            objective = objective_from_assignment(candidate)
            state["best_obj"] = objective
            state["best_x"] = candidate
            solution = make_solution(candidate, objective)
            if logger:
                logger.log_solution(objective, solution)
            return True
        return False

    # Fast deterministic/randomized feasible-solution construction.
    heuristic_budget = min(
        0.5,
        max(0.01, 0.03 * max(1, args.time_limit)),
    )
    heuristic_deadline = min(deadline, start_time + heuristic_budget)

    def build_from_order(order):
        if weights[0] > capacity:
            return None

        x = [0] * n
        x[0] = 1
        side_weight = weights[0]

        if side_weight > capacity:
            return None

        if side_weight < lower_weight:
            for v in order:
                if v == 0:
                    continue
                if side_weight + weights[v] <= capacity:
                    x[v] = 1
                    side_weight += weights[v]
                    if side_weight >= lower_weight:
                        break

        if lower_weight <= side_weight <= capacity:
            return x
        return None

    vertices = list(range(1, n))
    orders = [
        vertices,
        sorted(vertices, key=lambda v: (-weights[v], v)),
        sorted(vertices, key=lambda v: (weights[v], v)),
        sorted(
            vertices,
            key=lambda v: (
                -sum(abs(c) for _, c in adjacency[v]),
                -weights[v],
                v,
            ),
        ),
    ]

    rng = random.Random(0)
    random_order_count = 16 if n <= 100000 else 6
    for _ in range(random_order_count):
        shuffled = vertices.copy()
        rng.shuffle(shuffled)
        orders.append(shuffled)

    for order in orders:
        if time.monotonic() >= heuristic_deadline:
            break
        candidate = build_from_order(order)
        if candidate is not None:
            consider_assignment(candidate)

    # Improve the best constructed solution with feasible single-node flips.
    if state["best_x"] is not None and time.monotonic() < heuristic_deadline:
        x = state["best_x"].copy()
        side_weight = sum(weights[i] for i in range(n) if x[i])

        # Delta[v] is the objective change caused by flipping v.
        delta = [0.0] * n
        for u, v, c in aggregated_edges:
            contribution = c if x[u] == x[v] else -c
            delta[u] += contribution
            delta[v] += contribution

        max_iterations = min(1000, max(50, 2 * n))
        for _ in range(max_iterations):
            if time.monotonic() >= heuristic_deadline:
                break

            best_v = -1
            best_delta = -1e-12
            for v in range(1, n):  # Node 0 remains in partition S.
                new_weight = (
                    side_weight - weights[v]
                    if x[v]
                    else side_weight + weights[v]
                )
                if lower_weight <= new_weight <= capacity and delta[v] < best_delta:
                    best_delta = delta[v]
                    best_v = v

            if best_v < 0:
                break

            v = best_v
            old_value = x[v]
            old_delta_v = delta[v]

            # Update neighboring flip deltas using the relation before flipping v.
            for u, c in adjacency[v]:
                old_contribution_to_u = c if x[u] == old_value else -c
                delta[u] -= 2.0 * old_contribution_to_u

            x[v] = 1 - old_value
            delta[v] = -old_delta_v
            side_weight += weights[v] if x[v] else -weights[v]

            consider_assignment(x)

    # Build an exact mixed-integer formulation.
    model = gp.Model("weighted_graph_bisection")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    x_vars = [
        model.addVar(vtype=GRB.BINARY, name=f"x_{i}")
        for i in range(n)
    ]
    y_vars = [
        model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"y_{k}")
        for k in range(len(aggregated_edges))
    ]

    model.addConstr(x_vars[0] == 1, name="reference_node")

    partition_weight = gp.quicksum(weights[i] * x_vars[i] for i in range(n))
    model.addConstr(partition_weight >= lower_weight, name="partition_lower")
    model.addConstr(partition_weight <= capacity, name="partition_upper")

    for k, (u, v, _) in enumerate(aggregated_edges):
        y = y_vars[k]
        xu = x_vars[u]
        xv = x_vars[v]

        # Convex-hull description of y = |xu - xv| for binary xu and xv.
        model.addConstr(y >= xu - xv)
        model.addConstr(y >= xv - xu)
        model.addConstr(y <= xu + xv)
        model.addConstr(y <= 2 - xu - xv)

    model.setObjective(
        gp.quicksum(
            aggregated_edges[k][2] * y_vars[k]
            for k in range(len(aggregated_edges))
        ),
        GRB.MINIMIZE,
    )

    # Supply the heuristic incumbent as a MIP start.
    if state["best_x"] is not None:
        warm_start = state["best_x"]
        for i, var in enumerate(x_vars):
            var.Start = warm_start[i]
        for k, (u, v, _) in enumerate(aggregated_edges):
            y_vars[k].Start = int(warm_start[u] != warm_start[v])

    model._x_vars = x_vars

    def incumbent_callback(callback_model, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            values = callback_model.cbGetSolution(callback_model._x_vars)
            assignment = [1 if value >= 0.5 else 0 for value in values]
            consider_assignment(assignment)
        except Exception:
            # A logging or numerical issue should not abort the optimization.
            pass

    remaining = deadline - time.monotonic()
    if remaining > 0:
        model.Params.TimeLimit = max(0.001, remaining)
        model.optimize(incumbent_callback)

        # Retrieve the solver's final incumbent as a safeguard.
        if model.SolCount > 0:
            final_assignment = [1 if var.X >= 0.5 else 0 for var in x_vars]
            consider_assignment(final_assignment)

    if state["best_x"] is None:
        raise RuntimeError(
            "No feasible partition was found within the supplied time limit"
        )

    final_solution = make_solution(state["best_x"], state["best_obj"])
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, separators=(",", ":"))


if __name__ == "__main__":
    main()