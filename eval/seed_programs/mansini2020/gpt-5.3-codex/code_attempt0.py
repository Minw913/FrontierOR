import argparse
import json
from typing import Any, Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def is_dominated(item_a: Dict[str, Any], item_b: Dict[str, Any]) -> bool:
    """
    Returns True if item_a is dominated by item_b:
      - profit_b >= profit_a
      - weights_b <= weights_a component-wise
      - and at least one strict improvement
    """
    pa, pb = item_a["profit"], item_b["profit"]
    wa, wb = item_a["weights"], item_b["weights"]

    if pb < pa:
        return False
    if any(wb_r > wa_r for wa_r, wb_r in zip(wa, wb)):
        return False

    strict = (pb > pa) or any(wb_r < wa_r for wa_r, wb_r in zip(wa, wb))
    return strict


def preprocess_groups(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove dominated items inside each group to reduce model size.
    """
    reduced_groups = []
    for g in groups:
        items = g["items"]
        keep = []
        for i, it_i in enumerate(items):
            dominated = False
            for j, it_j in enumerate(items):
                if i == j:
                    continue
                if is_dominated(it_i, it_j):
                    dominated = True
                    break
            if not dominated:
                keep.append(it_i)

        # Fallback safety: keep at least one item if something unexpected happens
        if not keep:
            keep = items[:1]

        reduced_groups.append({"group_id": g["group_id"], "items": keep})
    return reduced_groups


def solve_instance(instance: Dict[str, Any], time_limit: int, logger: SolutionLogger = None) -> Dict[str, Any]:
    capacities: List[int] = instance["capacities"]
    m_resources = len(capacities)

    groups_in = instance["groups"]
    groups = preprocess_groups(groups_in)

    model = gp.Model("mmkp")
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0, int(time_limit))
    model.Params.OutputFlag = 0
    model.Params.Seed = 0

    # Decision variables: x[g_idx, i_idx] = 1 if item i from group g is selected
    x = {}
    for g_idx, g in enumerate(groups):
        for i_idx, item in enumerate(g["items"]):
            x[(g_idx, i_idx)] = model.addVar(vtype=GRB.BINARY, name=f"x_{g_idx}_{i_idx}")

    model.update()

    # Exactly one item per group
    for g_idx, g in enumerate(groups):
        model.addConstr(
            gp.quicksum(x[(g_idx, i_idx)] for i_idx in range(len(g["items"]))) == 1,
            name=f"one_item_group_{g_idx}",
        )

    # Resource capacity constraints
    for r in range(m_resources):
        model.addConstr(
            gp.quicksum(
                g["items"][i_idx]["weights"][r] * x[(g_idx, i_idx)]
                for g_idx, g in enumerate(groups)
                for i_idx in range(len(g["items"]))
            )
            <= capacities[r],
            name=f"capacity_{r}",
        )

    # Objective: maximize profit
    model.setObjective(
        gp.quicksum(
            g["items"][i_idx]["profit"] * x[(g_idx, i_idx)]
            for g_idx, g in enumerate(groups)
            for i_idx in range(len(g["items"]))
        ),
        GRB.MAXIMIZE,
    )

    best_logged = {"obj": float("-inf")}

    def incumbent_callback(cb_model, where):
        if logger is None:
            return
        if where == GRB.Callback.MIPSOL:
            obj_val = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val > best_logged["obj"] + 1e-9:
                best_logged["obj"] = obj_val
                logger.log(float(obj_val))

    model.optimize(incumbent_callback if logger is not None else None)

    # Build output
    if model.SolCount == 0:
        # No feasible solution found
        return {"objective_value": 0.0, "selected_items": []}

    selected_items = []
    total_profit = 0.0

    for g_idx, g in enumerate(groups):
        chosen_i = None
        for i_idx, item in enumerate(g["items"]):
            if x[(g_idx, i_idx)].X > 0.5:
                chosen_i = i_idx
                break

        # Safety fallback in case of numerical oddity
        if chosen_i is None:
            chosen_i = max(range(len(g["items"])), key=lambda i: x[(g_idx, i)].X)

        item = g["items"][chosen_i]
        selected_items.append(
            {
                "group_id": int(g["group_id"]),
                "item_id": int(item["item_id"]),
                "profit": int(item["profit"]),
                "weights": [int(w) for w in item["weights"]],
            }
        )
        total_profit += float(item["profit"])

    return {"objective_value": float(total_profit), "selected_items": selected_items}


def main():
    parser = argparse.ArgumentParser(description="Solve multiple-choice multidimensional knapsack instance.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Optional JSONL path for incumbent logging.")
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    solution = solve_instance(instance, args.time_limit, logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()