import argparse
import json
import math
import os
import time

import numpy as np
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = int(data["num_nodes"])
    commodities = data["commodities"]
    k_count = len(commodities)

    params = data["parameters"]
    tau = float(params["tau"])
    chi = float(params["chi"])
    delta = float(params["delta"])

    # The schema states that distances have already been scaled by TC.
    dist = np.asarray(data["distances"], dtype=np.float64)
    fixed = np.asarray(
        [float(node["fixed_cost"]) for node in data["nodes"]],
        dtype=np.float64,
    )

    origins = np.asarray(
        [int(c["origin"]) for c in commodities], dtype=np.int32
    )
    destinations = np.asarray(
        [int(c["destination"]) for c in commodities], dtype=np.int32
    )
    flows = np.asarray(
        [float(c["flow"]) for c in commodities], dtype=np.float64
    )
    self_commodity = origins == destinations

    best_objective = math.inf
    best_solution = None
    logged_objective = math.inf

    def evaluate_hubs(open_hubs):
        """Return the optimal routing and complete solution for a fixed hub set."""
        hubs = sorted(set(int(h) for h in open_hubs))

        if k_count == 0:
            objective = float(np.sum(fixed[hubs])) if hubs else 0.0
            return {
                "objective_value": objective,
                "open_hubs": hubs,
                "assignments": [],
            }

        if not hubs:
            return None

        route_cost = np.full(k_count, np.inf, dtype=np.float64)
        first_hub = np.full(k_count, -1, dtype=np.int32)
        second_hub = np.full(k_count, -1, dtype=np.int32)

        # Enumerating both orientations is equivalent to taking the best
        # orientation of each unordered hub edge.
        for i in hubs:
            oi = dist[origins, i]
            for j in hubs:
                if i != j:
                    valid = ~self_commodity
                    if not np.any(valid):
                        continue
                else:
                    valid = None

                leg_cost = (
                    chi * oi
                    + tau * dist[i, j]
                    + delta * dist[j, destinations]
                )
                total = flows * leg_cost

                if valid is None:
                    improve = total < route_cost
                else:
                    improve = valid & (total < route_cost)

                if np.any(improve):
                    route_cost[improve] = total[improve]
                    first_hub[improve] = i
                    second_hub[improve] = j

        if not np.all(np.isfinite(route_cost)):
            return None

        assignments = []
        for k in range(k_count):
            i = int(first_hub[k])
            j = int(second_hub[k])
            edge = [i] if i == j else [i, j]
            assignments.append(
                {
                    "commodity": k,
                    "origin": int(origins[k]),
                    "destination": int(destinations[k]),
                    "hub_edge": edge,
                    "cost": float(route_cost[k]),
                }
            )

        objective = float(np.sum(fixed[hubs]) + np.sum(route_cost))
        return {
            "objective_value": objective,
            "open_hubs": hubs,
            "assignments": assignments,
        }

    def accept_solution(solution, force_log=False):
        nonlocal best_objective, best_solution, logged_objective
        if solution is None:
            return False

        value = float(solution["objective_value"])
        tolerance = 1e-8 * max(1.0, abs(best_objective)) if math.isfinite(best_objective) else 0.0
        improved = value < best_objective - tolerance

        if improved or best_solution is None:
            best_objective = value
            best_solution = solution

        log_tol = 1e-8 * max(1.0, abs(logged_objective)) if math.isfinite(logged_objective) else 0.0
        if logger and (value < logged_objective - log_tol or force_log):
            if value <= logged_objective + log_tol:
                logger.log_solution(value, solution)
                logged_objective = min(logged_objective, value)

        return improved

    if k_count == 0:
        # With no commodities, open only negative-cost hubs.
        hubs = [i for i in range(n) if fixed[i] < 0.0]
        accept_solution(evaluate_hubs(hubs), force_log=True)
    elif n == 0:
        raise ValueError("An instance with commodities must contain at least one node.")
    else:
        # Best one-hub solution.
        single_route_costs = np.empty((n, k_count), dtype=np.float64)
        best_single_node = 0
        best_single_value = math.inf

        for i in range(n):
            costs = flows * (
                chi * dist[origins, i] + delta * dist[i, destinations]
            )
            single_route_costs[i, :] = costs
            value = float(fixed[i] + np.sum(costs))
            if value < best_single_value:
                best_single_value = value
                best_single_node = i

        current_hubs = [best_single_node]
        accept_solution(evaluate_hubs(current_hubs), force_log=True)

        # Search all two-hub starts when the available heuristic budget permits.
        heuristic_deadline = min(
            deadline,
            start_time + min(10.0, max(0.2, 0.18 * max(1, args.time_limit))),
        )
        best_pair = None
        best_pair_value = best_single_value

        for i in range(n):
            if time.monotonic() >= heuristic_deadline:
                break
            base_i = single_route_costs[i]
            for j in range(i + 1, n):
                if (j & 31) == 0 and time.monotonic() >= heuristic_deadline:
                    break

                ij = flows * (
                    chi * dist[origins, i]
                    + tau * dist[i, j]
                    + delta * dist[j, destinations]
                )
                ji = flows * (
                    chi * dist[origins, j]
                    + tau * dist[j, i]
                    + delta * dist[i, destinations]
                )

                # Same-origin/destination commodities must use one hub.
                if np.any(self_commodity):
                    ij = ij.copy()
                    ji = ji.copy()
                    ij[self_commodity] = np.inf
                    ji[self_commodity] = np.inf

                route = np.minimum(base_i, single_route_costs[j])
                route = np.minimum(route, ij)
                route = np.minimum(route, ji)
                value = float(fixed[i] + fixed[j] + np.sum(route))

                if value < best_pair_value:
                    best_pair_value = value
                    best_pair = [i, j]

        if best_pair is not None:
            pair_solution = evaluate_hubs(best_pair)
            if pair_solution is not None and pair_solution["objective_value"] < best_objective:
                current_hubs = best_pair
                accept_solution(pair_solution)

        # Greedy hub additions, evaluating all routes involving the new hub.
        current_solution = evaluate_hubs(current_hubs)
        current_cost_vector = np.asarray(
            [a["cost"] for a in current_solution["assignments"]],
            dtype=np.float64,
        )

        while time.monotonic() < heuristic_deadline:
            open_set = set(current_hubs)
            fixed_current = float(np.sum(fixed[current_hubs]))
            selected = None
            selected_value = float(
                fixed_current + np.sum(current_cost_vector)
            )
            selected_cost_vector = None

            for c in range(n):
                if c in open_set:
                    continue
                if time.monotonic() >= heuristic_deadline:
                    break

                involving = single_route_costs[c].copy()
                for h in current_hubs:
                    ch = flows * (
                        chi * dist[origins, c]
                        + tau * dist[c, h]
                        + delta * dist[h, destinations]
                    )
                    hc = flows * (
                        chi * dist[origins, h]
                        + tau * dist[h, c]
                        + delta * dist[c, destinations]
                    )
                    if np.any(self_commodity):
                        ch = ch.copy()
                        hc = hc.copy()
                        ch[self_commodity] = np.inf
                        hc[self_commodity] = np.inf
                    involving = np.minimum(involving, ch)
                    involving = np.minimum(involving, hc)

                candidate_costs = np.minimum(current_cost_vector, involving)
                value = float(
                    fixed_current + fixed[c] + np.sum(candidate_costs)
                )
                if value < selected_value - 1e-9 * max(1.0, abs(selected_value)):
                    selected = c
                    selected_value = value
                    selected_cost_vector = candidate_costs

            if selected is None:
                break

            current_hubs.append(selected)
            current_hubs.sort()
            current_cost_vector = selected_cost_vector
            accept_solution(evaluate_hubs(current_hubs))

        # Try deleting redundant hubs.
        deletion_improved = True
        while deletion_improved and len(current_hubs) > 1 and time.monotonic() < heuristic_deadline:
            deletion_improved = False
            chosen_solution = None
            chosen_hubs = None

            for h in list(current_hubs):
                if time.monotonic() >= heuristic_deadline:
                    break
                trial = [x for x in current_hubs if x != h]
                solution = evaluate_hubs(trial)
                if (
                    solution is not None
                    and solution["objective_value"]
                    < current_solution["objective_value"]
                    - 1e-9 * max(1.0, abs(current_solution["objective_value"]))
                ):
                    if (
                        chosen_solution is None
                        or solution["objective_value"]
                        < chosen_solution["objective_value"]
                    ):
                        chosen_solution = solution
                        chosen_hubs = trial

            if chosen_solution is not None:
                current_hubs = chosen_hubs
                current_solution = chosen_solution
                current_cost_vector = np.asarray(
                    [a["cost"] for a in current_solution["assignments"]],
                    dtype=np.float64,
                )
                accept_solution(current_solution)
                deletion_improved = True

        # Exact MIP for instances whose assignment formulation is a manageable size.
        edge_count = n * (n + 1) // 2
        x_count = k_count * edge_count
        remaining = deadline - time.monotonic()

        if x_count <= 350000 and remaining >= 1.0:
            try:
                import gurobipy as gp
                from gurobipy import GRB

                edges = []
                edge_index = np.full((n, n), -1, dtype=np.int32)
                for i in range(n):
                    for j in range(i, n):
                        p = len(edges)
                        edges.append((i, j))
                        edge_index[i, j] = p
                        edge_index[j, i] = p

                costs = np.empty((k_count, edge_count), dtype=np.float64)
                aborted = False

                for p, (i, j) in enumerate(edges):
                    if (p & 127) == 0 and time.monotonic() >= deadline - 0.4:
                        aborted = True
                        break

                    if i == j:
                        costs[:, p] = single_route_costs[i]
                    else:
                        ij = flows * (
                            chi * dist[origins, i]
                            + tau * dist[i, j]
                            + delta * dist[j, destinations]
                        )
                        ji = flows * (
                            chi * dist[origins, j]
                            + tau * dist[j, i]
                            + delta * dist[i, destinations]
                        )
                        pair_cost = np.minimum(ij, ji)

                        # Such assignments are dominated for proper metric data.
                        # A large finite coefficient additionally discourages
                        # degenerate zero-distance ties for self commodities.
                        if np.any(self_commodity):
                            pair_cost = pair_cost.copy()
                            pair_cost[self_commodity] = (
                                np.minimum(
                                    single_route_costs[i, self_commodity],
                                    single_route_costs[j, self_commodity],
                                )
                                + 1e6
                                * max(
                                    1.0,
                                    float(np.max(np.abs(single_route_costs)))
                                    if single_route_costs.size
                                    else 1.0,
                                )
                            )
                        costs[:, p] = pair_cost

                if not aborted and time.monotonic() < deadline - 0.25:
                    model = gp.Model("hub_location")
                    model.Params.OutputFlag = 0
                    model.Params.Seed = 0
                    model.Params.MIPGap = 1e-4
                    model.Params.NumericFocus = 0
                    model.Params.Threads = 1
                    model.Params.TimeLimit = max(
                        0.05, deadline - time.monotonic()
                    )

                    y = model.addMVar(n, vtype=GRB.BINARY, name="open")
                    x = model.addMVar(
                        (k_count, edge_count),
                        vtype=GRB.BINARY,
                        name="assign",
                    )

                    model.setObjective(
                        fixed @ y + (costs * x).sum(),
                        GRB.MINIMIZE,
                    )
                    model.addConstr(x.sum(axis=1) == 1.0)

                    for p, (i, j) in enumerate(edges):
                        model.addConstr(x[:, p] <= y[i])
                        if j != i:
                            model.addConstr(x[:, p] <= y[j])

                    # Complete MIP start from the best heuristic solution.
                    warm_hubs = set(best_solution["open_hubs"])
                    y.Start = np.asarray(
                        [1.0 if i in warm_hubs else 0.0 for i in range(n)],
                        dtype=np.float64,
                    )
                    x_start = np.zeros((k_count, edge_count), dtype=np.float64)
                    for assignment in best_solution["assignments"]:
                        edge = assignment["hub_edge"]
                        if len(edge) == 1:
                            i = j = int(edge[0])
                        else:
                            i, j = int(edge[0]), int(edge[1])
                        x_start[int(assignment["commodity"]), edge_index[i, j]] = 1.0
                    x.Start = x_start

                    def incumbent_callback(cb_model, where):
                        if where != GRB.Callback.MIPSOL:
                            return
                        try:
                            y_values = cb_model.cbGetSolution(y)
                            hubs = [
                                i for i, value in enumerate(y_values)
                                if value > 0.5
                            ]
                            if not hubs:
                                return
                            solution = evaluate_hubs(hubs)
                            accept_solution(solution)
                        except Exception:
                            # Logging must not interrupt optimization.
                            pass

                    model.optimize(incumbent_callback)

                    if model.SolCount > 0:
                        y_values = y.X
                        hubs = [
                            i for i, value in enumerate(y_values)
                            if value > 0.5
                        ]
                        if hubs:
                            accept_solution(evaluate_hubs(hubs))

            except Exception:
                # Preserve and output the best heuristic incumbent if Gurobi is
                # unavailable, the license fails, or model construction is cut short.
                pass

    if best_solution is None:
        # Defensive fallback for malformed edge cases.
        if n > 0:
            best_solution = evaluate_hubs([0])
        else:
            best_solution = {
                "objective_value": 0.0,
                "open_hubs": [],
                "assignments": [],
            }

    # Normalize all JSON numbers to built-in Python types.
    final_solution = {
        "objective_value": float(best_solution["objective_value"]),
        "open_hubs": [int(h) for h in best_solution["open_hubs"]],
        "assignments": [
            {
                "commodity": int(a["commodity"]),
                "origin": int(a["origin"]),
                "destination": int(a["destination"]),
                "hub_edge": [int(h) for h in a["hub_edge"]],
                "cost": float(a["cost"]),
            }
            for a in best_solution["assignments"]
        ],
    }

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, indent=2)


if __name__ == "__main__":
    main()