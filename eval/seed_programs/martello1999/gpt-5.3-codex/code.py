import argparse
import json
import math
import time
from typing import List, Dict, Tuple

from gurobipy import Model, GRB
from solution_logger import SolutionLogger


def load_instance(path: str) -> Tuple[int, int, List[Dict]]:
    with open(path, "r") as f:
        data = json.load(f)
    n = int(data["n"])
    capacity = int(data["capacity"])
    items = data["items"]
    return n, capacity, items


def greedy_solutions(items: List[Dict], capacity: int):
    """
    Generate several fast feasible heuristics; yield (objective, chosen_indices).
    """
    n = len(items)
    profits = [int(it["profit"]) for it in items]
    weights = [int(it["weight"]) for it in items]

    def pack(order):
        rem = capacity
        chosen = []
        obj = 0
        for i in order:
            if weights[i] <= rem:
                rem -= weights[i]
                chosen.append(i)
                obj += profits[i]
        return obj, chosen

    # 1) by profit/weight ratio (descending)
    order_ratio = sorted(
        range(n),
        key=lambda i: (profits[i] / weights[i], profits[i], -weights[i]),
        reverse=True,
    )
    yield pack(order_ratio)

    # 2) by pure profit
    order_profit = sorted(range(n), key=lambda i: (profits[i], -weights[i]), reverse=True)
    yield pack(order_profit)

    # 3) by low weight first
    order_weight = sorted(range(n), key=lambda i: (weights[i], -profits[i]))
    yield pack(order_weight)

    # 4) best single item
    best_single_obj = 0
    best_single_idx = None
    for i in range(n):
        if weights[i] <= capacity and profits[i] > best_single_obj:
            best_single_obj = profits[i]
            best_single_idx = i
    if best_single_idx is not None:
        yield best_single_obj, [best_single_idx]
    else:
        yield 0, []


def solve_knapsack(
    n: int,
    capacity: int,
    items: List[Dict],
    time_limit: int,
    logger: SolutionLogger = None,
):
    start_time = time.time()

    profits = [int(it["profit"]) for it in items]
    weights = [int(it["weight"]) for it in items]
    item_ids = [int(it["id"]) for it in items]

    # Build a strong feasible incumbent via heuristics
    best_obj = 0
    best_chosen_indices = []
    for obj, chosen in greedy_solutions(items, capacity):
        if obj > best_obj:
            best_obj = obj
            best_chosen_indices = chosen
            if logger:
                logger.log(int(best_obj))

    # Respect global time limit
    elapsed = time.time() - start_time
    remaining = max(0.0, float(time_limit) - elapsed)
    if remaining <= 0.0:
        selected_ids = sorted(item_ids[i] for i in best_chosen_indices)
        return int(best_obj), selected_ids

    try:
        model = Model("knapsack")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = remaining
        model.Params.Threads = 1

        x = model.addVars(n, vtype=GRB.BINARY, name="x")
        model.setObjective(sum(profits[i] * x[i] for i in range(n)), GRB.MAXIMIZE)
        model.addConstr(sum(weights[i] * x[i] for i in range(n)) <= capacity, name="capacity")

        # Warm start with heuristic incumbent
        chosen_set = set(best_chosen_indices)
        for i in range(n):
            x[i].Start = 1.0 if i in chosen_set else 0.0

        model._logger = logger
        model._best_logged = float(best_obj)

        def callback(m, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = m.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj_val > m._best_logged + 1e-9:
                    m._best_logged = obj_val
                    if m._logger:
                        m._logger.log(int(round(obj_val)))

        model.optimize(callback)

        # Use the best between solver result and heuristic
        if model.SolCount > 0:
            sol_obj = int(round(model.ObjVal))
            if sol_obj >= best_obj:
                selected_ids = sorted(item_ids[i] for i in range(n) if x[i].X > 0.5)
                return sol_obj, selected_ids

        # Fallback to heuristic if solver has no better solution
        selected_ids = sorted(item_ids[i] for i in best_chosen_indices)
        return int(best_obj), selected_ids

    except Exception:
        # Robust fallback
        selected_ids = sorted(item_ids[i] for i in best_chosen_indices)
        return int(best_obj), selected_ids


def main():
    parser = argparse.ArgumentParser(description="0/1 Knapsack Solver")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input instance JSON")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Optional JSONL log path for incumbents")
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    n, capacity, items = load_instance(args.instance_path)
    objective_value, selected_items = solve_knapsack(
        n=n,
        capacity=capacity,
        items=items,
        time_limit=args.time_limit,
        logger=logger,
    )

    solution = {
        "objective_value": int(objective_value),
        "selected_items": [int(i) for i in selected_items],
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()