import argparse
import json
import math
import os
import time
from collections import deque

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_feasible_flow(supplies, demands, arc_capacity):
    """Construct a feasible transportation flow using Dinic's max-flow algorithm."""
    n = len(supplies)
    m = len(demands)
    source = n + m
    sink = source + 1
    num_nodes = sink + 1
    graph = [[] for _ in range(num_nodes)]

    def add_edge(u, v, capacity):
        forward = [v, len(graph[v]), float(capacity)]
        reverse = [u, len(graph[u]), 0.0]
        graph[u].append(forward)
        graph[v].append(reverse)
        return forward

    for i, supply in enumerate(supplies):
        add_edge(source, i, supply)

    arc_edges = [[None] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            arc_edges[i][j] = add_edge(i, n + j, arc_capacity)

    for j, demand in enumerate(demands):
        add_edge(n + j, sink, demand)

    required_flow = float(sum(demands))
    total_flow = 0.0
    eps = 1e-12

    while total_flow + eps < required_flow:
        level = [-1] * num_nodes
        level[source] = 0
        queue = deque([source])

        while queue:
            u = queue.popleft()
            for edge in graph[u]:
                if edge[2] > eps and level[edge[0]] < 0:
                    level[edge[0]] = level[u] + 1
                    queue.append(edge[0])

        if level[sink] < 0:
            break

        next_edge = [0] * num_nodes

        def dfs(u, pushed):
            if u == sink:
                return pushed

            while next_edge[u] < len(graph[u]):
                edge = graph[u][next_edge[u]]
                v = edge[0]
                if edge[2] > eps and level[v] == level[u] + 1:
                    amount = dfs(v, min(pushed, edge[2]))
                    if amount > eps:
                        edge[2] -= amount
                        graph[v][edge[1]][2] += amount
                        return amount
                next_edge[u] += 1
            return 0.0

        while total_flow + eps < required_flow:
            pushed = dfs(source, required_flow - total_flow)
            if pushed <= eps:
                break
            total_flow += pushed

    if total_flow + 1e-7 < required_flow:
        raise ValueError(
            "The instance is infeasible: customer demands cannot be met under "
            "the supplier and arc capacities."
        )

    flows = [[0.0] * m for _ in range(n)]
    capacity = float(arc_capacity)
    for i in range(n):
        for j in range(m):
            value = capacity - arc_edges[i][j][2]
            if abs(value) < 1e-10:
                value = 0.0
            elif abs(value - capacity) < 1e-10:
                value = capacity
            flows[i][j] = value

    return flows


def calculate_objective(flows, linear_costs, quadratic_costs, is_quadratic):
    objective = 0.0
    for i, row in enumerate(flows):
        for j, value in enumerate(row):
            objective += float(linear_costs[i][j]) * value
            if is_quadratic:
                objective += float(quadratic_costs[i][j]) * value * value
    return float(objective)


def make_solution(flows, objective_value):
    flow_dict = {}
    for i, row in enumerate(flows):
        for j, value in enumerate(row):
            flow_dict[f"x_{i}_{j}"] = float(value)

    return {
        "objective_value": float(objective_value),
        "flows": flow_dict,
    }


def is_feasible(flows, supplies, demands, arc_capacity, tolerance=1e-5):
    n = len(supplies)
    m = len(demands)
    upper_bound = float(arc_capacity)

    for i in range(n):
        supplier_total = 0.0
        for j in range(m):
            value = flows[i][j]
            if not math.isfinite(value):
                return False
            if value < -tolerance or value > upper_bound + tolerance:
                return False
            supplier_total += value

        if supplier_total > float(supplies[i]) + tolerance:
            return False

    for j in range(m):
        customer_total = sum(flows[i][j] for i in range(n))
        if abs(customer_total - float(demands[j])) > tolerance:
            return False

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as file:
        instance = json.load(file)

    n = int(instance["n"])
    m = int(instance["m"])
    supplies = [float(value) for value in instance["supplies"]]
    demands = [float(value) for value in instance["demands"]]
    arc_capacity = float(instance["arc_capacity"])
    linear_costs = instance["linear_costs"]
    cost_type = str(instance["cost_type"])
    is_quadratic = cost_type == "quadratic"

    if is_quadratic:
        quadratic_costs = instance["quadratic_costs"]
    else:
        quadratic_costs = [[0.0] * m for _ in range(n)]

    best_flows = build_feasible_flow(supplies, demands, arc_capacity)
    best_objective = calculate_objective(
        best_flows, linear_costs, quadratic_costs, is_quadratic
    )
    best_solution = make_solution(best_flows, best_objective)

    if logger:
        logger.log_solution(best_objective, best_solution)

    model = None
    remaining_time = deadline - time.monotonic()

    if remaining_time > 0.01 and n > 0 and m > 0:
        try:
            model = gp.Model("capacitated_transportation")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(0.01, deadline - time.monotonic())

            x = model.addVars(
                n,
                m,
                lb=0.0,
                ub=arc_capacity,
                vtype=GRB.CONTINUOUS,
                name="x",
            )

            model.addConstrs(
                (gp.quicksum(x[i, j] for i in range(n)) == demands[j]
                 for j in range(m)),
                name="demand",
            )
            model.addConstrs(
                (gp.quicksum(x[i, j] for j in range(m)) <= supplies[i]
                 for i in range(n)),
                name="supply",
            )

            variables = [x[i, j] for i in range(n) for j in range(m)]
            linear_coefficients = [
                float(linear_costs[i][j]) for i in range(n) for j in range(m)
            ]
            linear_objective = gp.LinExpr(linear_coefficients, variables)

            if is_quadratic:
                quadratic_objective = gp.QuadExpr(linear_objective)
                quadratic_coefficients = []
                quadratic_variables = []

                for i in range(n):
                    for j in range(m):
                        coefficient = float(quadratic_costs[i][j])
                        if coefficient != 0.0:
                            quadratic_coefficients.append(coefficient)
                            quadratic_variables.append(x[i, j])

                if quadratic_coefficients:
                    quadratic_objective.addTerms(
                        quadratic_coefficients,
                        quadratic_variables,
                        quadratic_variables,
                    )
                model.setObjective(quadratic_objective, GRB.MINIMIZE)
            else:
                model.setObjective(linear_objective, GRB.MINIMIZE)

            # Supply a complete feasible primal starting point.
            for i in range(n):
                for j in range(m):
                    x[i, j].PStart = best_flows[i][j]

            remaining_time = deadline - time.monotonic()
            if remaining_time > 0.0:
                model.Params.TimeLimit = max(0.01, remaining_time)
                model.optimize()

                if model.SolCount > 0:
                    candidate_flows = [[0.0] * m for _ in range(n)]
                    for i in range(n):
                        for j in range(m):
                            value = float(x[i, j].X)
                            if value < 0.0 and value > -1e-7:
                                value = 0.0
                            elif value > arc_capacity and value < arc_capacity + 1e-7:
                                value = arc_capacity
                            candidate_flows[i][j] = value

                    if is_feasible(
                        candidate_flows, supplies, demands, arc_capacity
                    ):
                        candidate_objective = calculate_objective(
                            candidate_flows,
                            linear_costs,
                            quadratic_costs,
                            is_quadratic,
                        )

                        improvement_tolerance = (
                            1e-9 * max(1.0, abs(best_objective))
                        )
                        if candidate_objective < best_objective - improvement_tolerance:
                            best_flows = candidate_flows
                            best_objective = candidate_objective
                            best_solution = make_solution(
                                best_flows, best_objective
                            )
                            if logger:
                                logger.log_solution(
                                    best_objective, best_solution
                                )

        except gp.GurobiError:
            # Preserve and return the independently constructed feasible solution.
            pass
        finally:
            if model is not None:
                model.dispose()

    solution_directory = os.path.dirname(os.path.abspath(args.solution_path))
    if solution_directory:
        os.makedirs(solution_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as file:
        json.dump(best_solution, file, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()