import argparse
import json
import time

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", required=True,
                        help="Path where the final solution JSON will be written.")
    parser.add_argument("--time_limit", type=int, default=60,
                        help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", default=None,
                        help="Optional path to a JSONL convergence log file.")
    return parser.parse_args()


def load_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    capacity = int(data["capacity"])
    items = data["items"]
    return capacity, items


def greedy_solution(capacity, items):
    """Fast density-based greedy with best-single-item fallback."""
    order = sorted(items, key=lambda it: (-(it["profit"] / it["weight"]),
                                          -it["profit"]))
    total_w = 0
    total_p = 0
    chosen = []
    for it in order:
        w = it["weight"]
        if total_w + w <= capacity:
            chosen.append(it["id"])
            total_w += w
            total_p += it["profit"]
    best_single = None
    for it in items:
        if it["weight"] <= capacity:
            if best_single is None or it["profit"] > best_single["profit"]:
                best_single = it
    if best_single is not None and best_single["profit"] > total_p:
        return best_single["profit"], [best_single["id"]]
    return total_p, chosen


def solve_with_gurobi(capacity, items, incumbent, deadline, logger):
    """Solve the 0-1 knapsack with Gurobi, tracking incumbents via callback."""
    import gurobipy as gp
    from gurobipy import GRB

    n = len(items)
    profits = [int(it["profit"]) for it in items]
    weights = [int(it["weight"]) for it in items]
    ids = [int(it["id"]) for it in items]

    model = gp.Model("knapsack")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam("OutputFlag", 0)
    remaining = max(0.1, deadline - time.time())
    model.setParam("TimeLimit", remaining)

    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    model.addConstr(
        gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity,
        name="capacity",
    )
    model.setObjective(
        gp.quicksum(profits[i] * x[i] for i in range(n)),
        GRB.MAXIMIZE,
    )

    # Warm start with the greedy incumbent.
    chosen_set = set(incumbent["items"])
    for i in range(n):
        x[i].Start = 1 if ids[i] in chosen_set else 0

    best = {"obj": int(incumbent["obj"]), "items": list(incumbent["items"])}

    def callback(m, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj = int(round(m.cbGet(GRB.Callback.MIPSOL_OBJ)))
                if obj > best["obj"]:
                    sol = m.cbGetSolution(x)
                    chosen = [ids[i] for i in range(n) if sol[i] > 0.5]
                    wsum = sum(weights[i] for i in range(n) if sol[i] > 0.5)
                    if wsum <= capacity:
                        best["obj"] = obj
                        best["items"] = chosen
                        if logger:
                            logger.log_solution(obj, {
                                "objective_value": obj,
                                "selected_items": list(chosen),
                            })
            except Exception:
                pass

    model.optimize(callback)

    if model.SolCount > 0:
        obj = int(round(model.ObjVal))
        vals = model.getAttr("X", x)
        chosen = [ids[i] for i in range(n) if vals[i] > 0.5]
        if obj > best["obj"]:
            best["obj"] = obj
            best["items"] = chosen
            if logger:
                logger.log_solution(obj, {
                    "objective_value": obj,
                    "selected_items": list(chosen),
                })

    return best["obj"], best["items"]


def main():
    args = parse_args()
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    start = time.time()
    deadline = start + max(0, args.time_limit)

    capacity, items = load_instance(args.instance_path)

    best_obj, best_items = greedy_solution(capacity, items)
    best_obj = int(best_obj)

    if logger:
        logger.log_solution(best_obj, {
            "objective_value": best_obj,
            "selected_items": list(best_items),
        })

    if time.time() < deadline:
        try:
            incumbent = {"obj": best_obj, "items": best_items}
            obj, chosen = solve_with_gurobi(
                capacity, items, incumbent, deadline, logger
            )
            if obj > best_obj:
                best_obj, best_items = obj, chosen
        except Exception:
            # Fall back to the greedy incumbent if the solver is unavailable.
            pass

    result = {
        "objective_value": int(best_obj),
        "selected_items": [int(i) for i in best_items],
    }
    with open(args.solution_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()