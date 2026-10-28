import argparse
import json
import time
import sys
from solution_logger import SolutionLogger


def solve_with_gurobi(instance, time_limit, logger):
    """Solve using Gurobi MIP formulation."""
    import gurobipy as gp
    from gurobipy import GRB

    n_groups = instance["parameters"]["n_groups"]
    m_resources = instance["parameters"]["m_resources"]
    capacities = instance["capacities"]
    groups = instance["groups"]

    model = gp.Model("MCMKP")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", max(time_limit, 1))
    model.setParam("OutputFlag", 0)

    # Decision variables
    x = {}
    for g in range(n_groups):
        items = groups[g]["items"]
        for i in range(len(items)):
            x[g, i] = model.addVar(vtype=GRB.BINARY, name=f"x_{g}_{i}")

    model.update()

    # Objective: maximize total profit
    obj = gp.quicksum(
        groups[g]["items"][i]["profit"] * x[g, i]
        for g in range(n_groups)
        for i in range(len(groups[g]["items"]))
    )
    model.setObjective(obj, GRB.MAXIMIZE)

    # Constraint: exactly one item per group
    for g in range(n_groups):
        model.addConstr(
            gp.quicksum(x[g, i] for i in range(len(groups[g]["items"]))) == 1,
            name=f"group_{g}"
        )

    # Constraint: resource capacities
    for r in range(m_resources):
        model.addConstr(
            gp.quicksum(
                groups[g]["items"][i]["weights"][r] * x[g, i]
                for g in range(n_groups)
                for i in range(len(groups[g]["items"]))
            ) <= capacities[r],
            name=f"resource_{r}"
        )

    # Track best solution found via callback
    best_solution = [None]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            selected = []
            total_profit = 0
            for g in range(n_groups):
                for i in range(len(groups[g]["items"])):
                    if model.cbGetSolution(x[g, i]) > 0.5:
                        item = groups[g]["items"][i]
                        selected.append({
                            "group_id": g,
                            "item_id": item["item_id"],
                            "profit": item["profit"],
                            "weights": item["weights"]
                        })
                        total_profit += item["profit"]
                        break

            solution_dict = {
                "objective_value": float(total_profit),
                "selected_items": selected
            }
            best_solution[0] = solution_dict

            if logger:
                logger.log_solution(float(total_profit), solution_dict)

    model.optimize(callback)

    # Extract final solution
    if model.SolCount > 0:
        selected = []
        total_profit = 0
        for g in range(n_groups):
            for i in range(len(groups[g]["items"])):
                if x[g, i].X > 0.5:
                    item = groups[g]["items"][i]
                    selected.append({
                        "group_id": g,
                        "item_id": item["item_id"],
                        "profit": item["profit"],
                        "weights": item["weights"]
                    })
                    total_profit += item["profit"]
                    break

        return {
            "objective_value": float(total_profit),
            "selected_items": selected
        }

    return best_solution[0]


def greedy_heuristic(instance):
    """Greedy heuristic based on profit density."""
    n_groups = instance["parameters"]["n_groups"]
    m_resources = instance["parameters"]["m_resources"]
    capacities = instance["capacities"]
    groups = instance["groups"]

    remaining_cap = list(capacities)
    selected = []
    total_profit = 0

    for g in range(n_groups):
        items = groups[g]["items"]
        best_item = None
        best_score = -1

        for item in items:
            # Check feasibility
            feasible = True
            for r in range(m_resources):
                if item["weights"][r] > remaining_cap[r]:
                    feasible = False
                    break

            if feasible:
                # Score: profit / (1 + sum of normalized weights)
                weight_sum = sum(
                    item["weights"][r] / max(capacities[r], 1)
                    for r in range(m_resources)
                )
                score = item["profit"] / (1.0 + weight_sum)
                if score > best_score:
                    best_score = score
                    best_item = item

        if best_item is None:
            # Pick item with minimum total weight as fallback
            best_item = min(items, key=lambda it: sum(it["weights"]))

        selected.append({
            "group_id": g,
            "item_id": best_item["item_id"],
            "profit": best_item["profit"],
            "weights": best_item["weights"]
        })
        total_profit += best_item["profit"]
        for r in range(m_resources):
            remaining_cap[r] -= best_item["weights"][r]

    return {
        "objective_value": float(total_profit),
        "selected_items": selected
    }


def local_search_improve(instance, initial_solution, time_remaining, logger):
    """Try to improve solution via single-item swaps."""
    n_groups = instance["parameters"]["n_groups"]
    m_resources = instance["parameters"]["m_resources"]
    capacities = instance["capacities"]
    groups = instance["groups"]

    selected = [None] * n_groups
    for item in initial_solution["selected_items"]:
        selected[item["group_id"]] = item

    # Compute current resource usage
    usage = [0] * m_resources
    total_profit = 0
    for g in range(n_groups):
        for r in range(m_resources):
            usage[r] += selected[g]["weights"][r]
        total_profit += selected[g]["profit"]

    improved = True
    start = time.time()

    while improved and (time.time() - start) < time_remaining:
        improved = False
        for g in range(n_groups):
            if time.time() - start >= time_remaining:
                break
            current_item = selected[g]
            for candidate in groups[g]["items"]:
                if candidate["item_id"] == current_item["item_id"]:
                    continue
                # Check if swap is feasible
                feasible = True
                for r in range(m_resources):
                    new_usage = usage[r] - current_item["weights"][r] + candidate["weights"][r]
                    if new_usage > capacities[r]:
                        feasible = False
                        break
                if feasible and candidate["profit"] > current_item["profit"]:
                    # Perform swap
                    for r in range(m_resources):
                        usage[r] = usage[r] - current_item["weights"][r] + candidate["weights"][r]
                    total_profit = total_profit - current_item["profit"] + candidate["profit"]
                    selected[g] = {
                        "group_id": g,
                        "item_id": candidate["item_id"],
                        "profit": candidate["profit"],
                        "weights": candidate["weights"]
                    }
                    current_item = selected[g]
                    improved = True

    solution = {
        "objective_value": float(total_profit),
        "selected_items": [selected[g] for g in range(n_groups)]
    }

    if logger:
        logger.log_solution(solution["objective_value"], solution)

    return solution


def main():
    parser = argparse.ArgumentParser(description="MCMKP Solver")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    start_time = time.time()

    # Step 1: Get a quick greedy solution as initial incumbent
    greedy_sol = greedy_heuristic(instance)
    if logger:
        logger.log_solution(greedy_sol["objective_value"], greedy_sol)

    best_solution = greedy_sol

    # Step 2: Try local search improvement on greedy solution
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed
    if remaining > 1:
        improved_sol = local_search_improve(instance, greedy_sol, min(remaining * 0.3, 10), logger)
        if improved_sol["objective_value"] > best_solution["objective_value"]:
            best_solution = improved_sol

    # Step 3: Try Gurobi for optimal/near-optimal solution
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed
    if remaining > 0.5:
        try:
            gurobi_sol = solve_with_gurobi(instance, remaining, logger)
            if gurobi_sol is not None and gurobi_sol["objective_value"] > best_solution["objective_value"]:
                best_solution = gurobi_sol
        except Exception as e:
            print(f"Gurobi failed: {e}", file=sys.stderr)

    # Write final solution
    with open(args.solution_path, "w") as f:
        json.dump(best_solution, f, indent=2)


if __name__ == "__main__":
    main()