import argparse
import json
import math
import os
import time

import numpy as np

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def route_utility_matrix(
    distance,
    waiting,
    first_hub,
    second_hub,
    alpha,
    gamma,
    beta,
    fare_exp,
    cost_scalar,
    attractiveness,
):
    """Utility for every ordered origin-destination pair for one hub order."""
    interhub_distance = distance[first_hub, second_hub]

    total_time = (
        distance[:, first_hub][:, None]
        + interhub_distance
        + distance[second_hub, :][None, :]
        + waiting[:, None]
        + waiting[first_hub]
        + waiting[second_hub]
    )

    total_fare = cost_scalar * (
        distance[:, first_hub][:, None]
        + alpha * interhub_distance
        + distance[second_hub, :][None, :]
    )

    time_term = gamma * np.power(np.maximum(total_time, 0.0), beta)
    fare_term = (1.0 - gamma) * np.power(np.maximum(total_fare, 0.0), fare_exp)
    denominator = np.maximum(time_term + fare_term, 1e-12)

    utility = attractiveness / denominator
    return np.minimum(utility, 1e15)


def build_features(instance):
    n = int(instance["n_nodes"])
    nodes = instance["nodes"]
    waiting = np.asarray([float(node["waiting_time"]) for node in nodes], dtype=float)
    distance = np.asarray(instance["distance_matrix"], dtype=float)
    demand_full = np.asarray(instance["demand_matrix"], dtype=float)

    params = instance["parameters"]
    alpha = float(params["alpha"])
    gamma = float(params["gamma"])
    beta = float(params["beta"])
    fare_exp = float(params["lambda"])
    cost_scalar = float(params["cost_scalar"])
    attr_two = float(params["attractiveness_2hub"])
    attr_one = float(params["attractiveness_1hub"])

    positive_mask = demand_full.ravel() > 0.0
    q_indices = np.flatnonzero(positive_mask)
    demand = demand_full.ravel()[q_indices].astype(float, copy=False)
    q_count = len(q_indices)

    edge_a = []
    edge_b = []
    for a in range(n):
        for b in range(a + 1, n):
            edge_a.append(a)
            edge_b.append(b)
    edge_a = np.asarray(edge_a, dtype=np.int32)
    edge_b = np.asarray(edge_b, dtype=np.int32)
    edge_count = len(edge_a)

    node_features = np.empty((q_count, n), dtype=float)
    for hub in range(n):
        utility = route_utility_matrix(
            distance,
            waiting,
            hub,
            hub,
            alpha,
            gamma,
            beta,
            fare_exp,
            cost_scalar,
            attr_one,
        )
        node_features[:, hub] = utility.ravel()[q_indices]

    edge_features = np.empty((q_count, edge_count), dtype=float)
    for e, (a, b) in enumerate(zip(edge_a, edge_b)):
        utility_ab = route_utility_matrix(
            distance,
            waiting,
            int(a),
            int(b),
            alpha,
            gamma,
            beta,
            fare_exp,
            cost_scalar,
            attr_two,
        )
        utility_ba = route_utility_matrix(
            distance,
            waiting,
            int(b),
            int(a),
            alpha,
            gamma,
            beta,
            fare_exp,
            cost_scalar,
            attr_two,
        )
        edge_features[:, e] = np.maximum(utility_ab, utility_ba).ravel()[q_indices]

    edge_index = np.full((n, n), -1, dtype=np.int32)
    for e, (a, b) in enumerate(zip(edge_a, edge_b)):
        edge_index[a, b] = e
        edge_index[b, a] = e

    competitor_hubs = sorted(set(int(h) for h in instance["competitor_hub_locations"]))
    competitor_utility = np.zeros(q_count, dtype=float)

    for hub in competitor_hubs:
        competitor_utility += node_features[:, hub]

    for pos, a in enumerate(competitor_hubs):
        for b in competitor_hubs[pos + 1:]:
            competitor_utility += edge_features[:, edge_index[a, b]]

    return (
        demand,
        competitor_utility,
        node_features,
        edge_features,
        edge_a,
        edge_b,
        edge_index,
    )


def entrant_utility(selected, node_features, edge_features, edge_index):
    q_count = node_features.shape[0]
    total = np.zeros(q_count, dtype=float)

    selected = sorted(int(v) for v in selected)
    for hub in selected:
        total += node_features[:, hub]

    for pos, a in enumerate(selected):
        for b in selected[pos + 1:]:
            total += edge_features[:, edge_index[a, b]]

    return total


def objective_from_utility(utility, demand, competitor_utility):
    denominator = utility + competitor_utility
    shares = np.divide(
        utility,
        denominator,
        out=np.ones_like(utility),
        where=denominator > 1e-15,
    )
    return float(np.dot(demand, shares))


def make_solution(objective, selected):
    return {
        "objective_value": float(objective),
        "hub_locations": sorted(int(v) for v in selected),
    }


def heuristic_search(
    n,
    p,
    demand,
    competitor_utility,
    node_features,
    edge_features,
    edge_index,
    deadline,
    logger,
    rng,
    initial_best=None,
):
    def evaluate_set(selected):
        util = entrant_utility(
            selected, node_features, edge_features, edge_index
        )
        return objective_from_utility(util, demand, competitor_utility), util

    if initial_best is None:
        selected = list(range(p))
        best_obj, best_util = evaluate_set(selected)
        best_selected = sorted(selected)
        if logger:
            logger.log_solution(best_obj, make_solution(best_obj, best_selected))
    else:
        best_obj, best_selected = initial_best
        best_selected = sorted(best_selected)
        best_util = entrant_utility(
            best_selected, node_features, edge_features, edge_index
        )

    # Deterministic greedy construction.
    if time.monotonic() < deadline:
        greedy = []
        greedy_util = np.zeros_like(demand)
        unused = set(range(n))

        while len(greedy) < p:
            if time.monotonic() >= deadline:
                fill = sorted(unused)[: p - len(greedy)]
                for v in fill:
                    addition = node_features[:, v].copy()
                    for h in greedy:
                        addition += edge_features[:, edge_index[v, h]]
                    greedy_util += addition
                    greedy.append(v)
                    unused.remove(v)
                break

            chosen = None
            chosen_obj = -math.inf
            chosen_util = None

            for v in unused:
                candidate_util = greedy_util + node_features[:, v]
                if greedy:
                    candidate_util = candidate_util.copy()
                    for h in greedy:
                        candidate_util += edge_features[:, edge_index[v, h]]
                value = objective_from_utility(
                    candidate_util, demand, competitor_utility
                )
                if value > chosen_obj:
                    chosen = v
                    chosen_obj = value
                    chosen_util = candidate_util

            greedy.append(chosen)
            unused.remove(chosen)
            greedy_util = chosen_util

        greedy_obj = objective_from_utility(
            greedy_util, demand, competitor_utility
        )
        if greedy_obj > best_obj + 1e-9:
            best_obj = greedy_obj
            best_selected = sorted(greedy)
            best_util = greedy_util.copy()
            if logger:
                logger.log_solution(
                    best_obj, make_solution(best_obj, best_selected)
                )

    def local_improvement(start_selected):
        nonlocal best_obj, best_selected, best_util

        current = sorted(start_selected)
        current_obj, current_util = evaluate_set(current)

        if current_obj > best_obj + 1e-9:
            best_obj = current_obj
            best_selected = current.copy()
            best_util = current_util.copy()
            if logger:
                logger.log_solution(
                    best_obj, make_solution(best_obj, best_selected)
                )

        while time.monotonic() < deadline:
            current_set = set(current)
            outside = [v for v in range(n) if v not in current_set]
            best_swap = None
            best_swap_obj = current_obj
            best_swap_util = None

            for removed in current:
                if time.monotonic() >= deadline:
                    break

                remaining = [h for h in current if h != removed]
                reduced_util = current_util - node_features[:, removed]
                for h in remaining:
                    reduced_util = (
                        reduced_util
                        - edge_features[:, edge_index[removed, h]]
                    )

                for added in outside:
                    candidate_util = reduced_util + node_features[:, added]
                    if remaining:
                        candidate_util = candidate_util.copy()
                        for h in remaining:
                            candidate_util += edge_features[
                                :, edge_index[added, h]
                            ]

                    value = objective_from_utility(
                        candidate_util, demand, competitor_utility
                    )
                    if value > best_swap_obj + 1e-9:
                        best_swap_obj = value
                        best_swap = (removed, added)
                        best_swap_util = candidate_util

            if best_swap is None:
                break

            removed, added = best_swap
            current.remove(removed)
            current.append(added)
            current.sort()
            current_util = best_swap_util
            current_obj = best_swap_obj

            if current_obj > best_obj + 1e-9:
                best_obj = current_obj
                best_selected = current.copy()
                best_util = current_util.copy()
                if logger:
                    logger.log_solution(
                        best_obj, make_solution(best_obj, best_selected)
                    )

        return current

    if time.monotonic() < deadline:
        local_improvement(best_selected)

    # Randomized restarts are mainly used when an exact/outer-approximation
    # model is too large to build.
    while time.monotonic() < deadline:
        if n == p:
            break

        if rng.random() < 0.7:
            start = best_selected.copy()
            swaps = max(1, min(3, p))
            for _ in range(swaps):
                selected_set = set(start)
                outside = [v for v in range(n) if v not in selected_set]
                if not outside:
                    break
                removed = int(rng.choice(start))
                added = int(rng.choice(outside))
                start.remove(removed)
                start.append(added)
                start.sort()
        else:
            start = sorted(
                int(v) for v in rng.choice(n, size=p, replace=False)
            )

        local_improvement(start)

    return best_obj, best_selected


def solve_with_outer_approximation(
    n,
    p,
    demand,
    competitor_utility,
    node_features,
    edge_features,
    edge_a,
    edge_b,
    edge_index,
    deadline,
    logger,
    incumbent,
):
    import gurobipy as gp
    from gurobipy import GRB

    best_holder = {
        "objective": float(incumbent[0]),
        "selected": sorted(incumbent[1]),
    }

    q_count = len(demand)
    edge_count = len(edge_a)

    # OD pairs without incumbent utility have a constant entrant share of one,
    # provided the entrant opens at least one hub. Utilities are positive for
    # ordinary data, so these terms can be removed from the model.
    variable_mask = competitor_utility > 1e-14
    constant_objective = float(np.sum(demand[~variable_mask]))

    d = demand[variable_mask]
    b = competitor_utility[variable_mask]
    nf = node_features[variable_mask, :]
    ef = edge_features[variable_mask, :]
    active_q = len(d)

    if active_q == 0:
        value = float(np.sum(demand))
        selected = best_holder["selected"]
        if value > best_holder["objective"] + 1e-9:
            best_holder["objective"] = value
            if logger:
                logger.log_solution(value, make_solution(value, selected))
        return best_holder["objective"], best_holder["selected"]

    model = gp.Model("entrant_hub_location")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    x = model.addMVar(n, vtype=GRB.BINARY, name="x")
    z = model.addMVar(edge_count, lb=0.0, ub=1.0, name="z")

    model.addConstr(x.sum() == p)
    if edge_count:
        model.addConstr(z <= x[edge_a])
        model.addConstr(z <= x[edge_b])
        model.addConstr(z >= x[edge_a] + x[edge_b] - 1.0)

    # A valid independent upper bound: at most p node terms and C(p,2)
    # pair terms can be active for any OD pair.
    if p >= n:
        node_upper = np.sum(nf, axis=1)
    else:
        kth = n - p
        node_upper = np.sum(np.partition(nf, kth, axis=1)[:, kth:], axis=1)

    selected_edge_count = p * (p - 1) // 2
    if selected_edge_count == 0 or edge_count == 0:
        edge_upper = np.zeros(active_q, dtype=float)
    elif selected_edge_count >= edge_count:
        edge_upper = np.sum(ef, axis=1)
    else:
        kth = edge_count - selected_edge_count
        edge_upper = np.sum(
            np.partition(ef, kth, axis=1)[:, kth:], axis=1
        )

    utility_upper = (node_upper + edge_upper) * (1.0 + 1e-9) + 1e-12

    a_var = model.addMVar(
        active_q, lb=0.0, ub=utility_upper, name="entrant_utility"
    )
    captured = model.addMVar(active_q, lb=0.0, ub=d, name="captured")

    utility_expression = nf @ x
    if edge_count:
        utility_expression = utility_expression + ef @ z
    model.addConstr(a_var == utility_expression)

    def add_tangent(points):
        points = np.minimum(np.maximum(points, 0.0), utility_upper)
        function_value = d * points / (b + points)
        derivative = d * b / np.square(b + points)
        intercept = function_value - derivative * points
        model.addConstr(captured <= intercept + derivative * a_var)

    # Tangents globally upper-bound the concave capture functions. Several
    # initial tangent locations substantially tighten the first master MIP.
    add_tangent(np.zeros(active_q))
    for ratio in (0.1, 0.3, 1.0, 3.0, 10.0):
        add_tangent(np.minimum(utility_upper, ratio * b))
    add_tangent(utility_upper)

    model.setObjective(constant_objective + captured.sum(), GRB.MAXIMIZE)

    incumbent_selected = best_holder["selected"]
    x.Start = np.asarray(
        [1.0 if i in set(incumbent_selected) else 0.0 for i in range(n)]
    )

    def callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            values = cb_model.cbGetSolution(x)
            selected = np.flatnonzero(values > 0.5).tolist()
            if len(selected) != p:
                return

            util = entrant_utility(
                selected, node_features, edge_features, edge_index
            )
            value = objective_from_utility(
                util, demand, competitor_utility
            )

            if value > best_holder["objective"] + 1e-8:
                best_holder["objective"] = value
                best_holder["selected"] = sorted(selected)
                if logger:
                    logger.log_solution(
                        value, make_solution(value, selected)
                    )
        except Exception:
            pass

    # Repeatedly solve the tangent master and add a tangent at its selected
    # solution. If convergence occurs before the deadline, the master supplies
    # a tight global bound for the discrete problem.
    for _ in range(20):
        remaining = deadline - time.monotonic()
        if remaining <= 0.05:
            break

        model.Params.TimeLimit = max(0.01, remaining)
        model.optimize(callback)

        if model.SolCount <= 0:
            break

        selected = np.flatnonzero(x.X > 0.5).tolist()
        if len(selected) != p:
            break

        full_utility = entrant_utility(
            selected, node_features, edge_features, edge_index
        )
        true_value = objective_from_utility(
            full_utility, demand, competitor_utility
        )

        if true_value > best_holder["objective"] + 1e-8:
            best_holder["objective"] = true_value
            best_holder["selected"] = sorted(selected)
            if logger:
                logger.log_solution(
                    true_value, make_solution(true_value, selected)
                )

        active_utility = full_utility[variable_mask]
        add_tangent(active_utility)
        model.update()

        # If the solved master bound agrees with the actual value, the current
        # incumbent is effectively optimal within the requested MIP tolerance.
        if model.Status == GRB.OPTIMAL:
            master_bound = float(model.ObjVal)
            if master_bound <= best_holder["objective"] + max(
                1e-6, 1e-5 * max(1.0, abs(best_holder["objective"]))
            ):
                break
        elif model.Status in (GRB.TIME_LIMIT, GRB.INTERRUPTED):
            break

    return best_holder["objective"], best_holder["selected"]


def main():
    args = parse_args()
    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = (
        SolutionLogger(args.log_path, sense="maximize")
        if args.log_path
        else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    n = int(instance["n_nodes"])
    p = int(instance["p"])
    if p < 0 or p > n:
        raise ValueError("p must satisfy 0 <= p <= n")

    if p == 0:
        solution = make_solution(0.0, [])
        if logger:
            logger.log_solution(0.0, solution)
        os.makedirs(
            os.path.dirname(os.path.abspath(args.solution_path)),
            exist_ok=True,
        )
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(solution, f)
        return

    (
        demand,
        competitor_utility,
        node_features,
        edge_features,
        edge_a,
        edge_b,
        edge_index,
    ) = build_features(instance)

    if len(demand) == 0:
        selected = list(range(p))
        solution = make_solution(0.0, selected)
        if logger:
            logger.log_solution(0.0, solution)
        os.makedirs(
            os.path.dirname(os.path.abspath(args.solution_path)),
            exist_ok=True,
        )
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(solution, f)
        return

    rng = np.random.default_rng(0)

    # Always establish a valid incumbent first.
    initial_selected = list(range(p))
    initial_utility = entrant_utility(
        initial_selected, node_features, edge_features, edge_index
    )
    initial_objective = objective_from_utility(
        initial_utility, demand, competitor_utility
    )
    best = (initial_objective, initial_selected)

    if logger:
        logger.log_solution(
            initial_objective,
            make_solution(initial_objective, initial_selected),
        )

    # Reserve most of the available time for the global outer-approximation
    # model while still producing a strong warm start.
    remaining = max(0.0, deadline - time.monotonic())
    heuristic_budget = min(5.0, 0.12 * remaining)
    heuristic_deadline = min(deadline, time.monotonic() + heuristic_budget)

    best = heuristic_search(
        n,
        p,
        demand,
        competitor_utility,
        node_features,
        edge_features,
        edge_index,
        heuristic_deadline,
        logger,
        rng,
        initial_best=best,
    )

    feature_entries = (
        node_features.size + edge_features.size
    )
    use_model = (
        time.monotonic() < deadline - 0.25
        and feature_entries <= 10_000_000
        and n <= 70
    )

    if use_model:
        try:
            best = solve_with_outer_approximation(
                n,
                p,
                demand,
                competitor_utility,
                node_features,
                edge_features,
                edge_a,
                edge_b,
                edge_index,
                deadline,
                logger,
                best,
            )
        except Exception:
            # Retain the valid heuristic incumbent and use any remaining time
            # for local/randomized improvement if model construction fails.
            best = heuristic_search(
                n,
                p,
                demand,
                competitor_utility,
                node_features,
                edge_features,
                edge_index,
                deadline,
                logger,
                rng,
                initial_best=best,
            )
    else:
        best = heuristic_search(
            n,
            p,
            demand,
            competitor_utility,
            node_features,
            edge_features,
            edge_index,
            deadline,
            logger,
            rng,
            initial_best=best,
        )

    final_objective, final_selected = best
    final_solution = make_solution(final_objective, final_selected)

    os.makedirs(
        os.path.dirname(os.path.abspath(args.solution_path)),
        exist_ok=True,
    )
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f)


if __name__ == "__main__":
    main()