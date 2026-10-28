import argparse
import json
import math
import os
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


EPS = 1e-7


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def determine_primary_problem(data):
    valid = {"PSCLP", "MCLP"}

    value = data.get("primary_problem_type")
    if value in valid:
        return value

    value = data.get("problem_type")
    if value in valid:
        return value

    values = data.get("problem_types")
    if isinstance(values, str) and values in valid:
        return values
    if isinstance(values, list):
        for value in values:
            if value in valid:
                return value

    return "PSCLP"


def build_coverage(data, num_facilities, num_customers):
    customer_neighbors = [[] for _ in range(num_customers)]
    supplied = data.get("customer_to_facilities_coverage")

    if isinstance(supplied, dict):
        for j in range(num_customers):
            values = supplied.get(str(j), supplied.get(j, []))
            customer_neighbors[j] = sorted({
                int(i) for i in values
                if 0 <= int(i) < num_facilities
            })
    elif isinstance(supplied, list) and len(supplied) >= num_customers:
        for j in range(num_customers):
            customer_neighbors[j] = sorted({
                int(i) for i in supplied[j]
                if 0 <= int(i) < num_facilities
            })
    else:
        supplied_facility = data.get("facility_to_customers_coverage")
        if isinstance(supplied_facility, dict):
            for i in range(num_facilities):
                customers = supplied_facility.get(str(i), supplied_facility.get(i, []))
                for j in customers:
                    j = int(j)
                    if 0 <= j < num_customers:
                        customer_neighbors[j].append(i)
            customer_neighbors = [sorted(set(v)) for v in customer_neighbors]
        else:
            facility_coordinates = data["facility_coordinates"]
            customer_coordinates = data["customer_coordinates"]
            radius = float(data["radius_of_coverage"])
            radius_squared = radius * radius
            tolerance = max(1.0, radius_squared) * 1e-12

            for j, (cx, cy) in enumerate(customer_coordinates):
                neighbors = []
                for i, (fx, fy) in enumerate(facility_coordinates):
                    dx = float(fx) - float(cx)
                    dy = float(fy) - float(cy)
                    if dx * dx + dy * dy <= radius_squared + tolerance:
                        neighbors.append(i)
                customer_neighbors[j] = neighbors

    facility_customers = [[] for _ in range(num_facilities)]
    for j, neighbors in enumerate(customer_neighbors):
        for i in neighbors:
            facility_customers[i].append(j)

    return customer_neighbors, facility_customers


def selected_cost(selected, costs):
    return float(sum(costs[i] for i in selected))


def covered_demand(selected, facility_customers, demands):
    covered = set()
    for i in selected:
        covered.update(facility_customers[i])
    return float(sum(demands[j] for j in covered))


def make_variant_result(problem_type, selected, costs, facility_customers, demands):
    selected = sorted(set(int(i) for i in selected))
    demand = covered_demand(selected, facility_customers, demands)
    objective = selected_cost(selected, costs) if problem_type == "PSCLP" else demand
    return {
        "objective_value": float(objective),
        "open_facilities": selected,
        "covered_demand": float(demand),
    }


def make_solution(primary_problem_type, results):
    return {
        "objective_value": float(results[primary_problem_type]["objective_value"]),
        "primary_problem_type": primary_problem_type,
        "results": {
            "PSCLP": results["PSCLP"],
            "MCLP": results["MCLP"],
        },
    }


def greedy_psclp(costs, demands, facility_customers, threshold):
    n = len(costs)
    m = len(demands)

    if threshold <= EPS:
        return []

    selected = set()
    covered = [False] * m
    current_demand = 0.0

    while current_demand + EPS < threshold:
        best_i = None
        best_key = None

        for i in range(n):
            if i in selected:
                continue
            gain = sum(
                demands[j] for j in facility_customers[i]
                if not covered[j]
            )
            if gain <= EPS:
                continue

            if costs[i] <= EPS:
                key = (1, gain, -costs[i])
            else:
                key = (0, gain / costs[i], gain, -costs[i])

            if best_key is None or key > best_key:
                best_key = key
                best_i = i

        if best_i is None:
            break

        selected.add(best_i)
        for j in facility_customers[best_i]:
            if not covered[j]:
                covered[j] = True
                current_demand += demands[j]

    # Compare against any single facility that satisfies the threshold.
    for i in range(n):
        single_demand = sum(demands[j] for j in facility_customers[i])
        if single_demand + EPS >= threshold:
            if selected_cost([i], costs) + EPS < selected_cost(selected, costs):
                selected = {i}
                break

    # Remove redundant selected facilities, considering expensive ones first.
    counts = [0] * m
    for i in selected:
        for j in facility_customers[i]:
            counts[j] += 1

    current_demand = sum(demands[j] for j in range(m) if counts[j] > 0)
    for i in sorted(selected, key=lambda k: (costs[k], k), reverse=True):
        loss = sum(
            demands[j] for j in facility_customers[i]
            if counts[j] == 1
        )
        if current_demand - loss + EPS >= threshold:
            selected.remove(i)
            current_demand -= loss
            for j in facility_customers[i]:
                counts[j] -= 1

    return sorted(selected)


def greedy_mclp(costs, demands, facility_customers, budget):
    n = len(costs)
    m = len(demands)

    selected = set()
    covered = [False] * m
    spent = 0.0

    # Every useful zero-cost facility can safely be selected.
    for i in range(n):
        if costs[i] <= EPS and facility_customers[i]:
            selected.add(i)
            for j in facility_customers[i]:
                covered[j] = True

    while True:
        best_i = None
        best_key = None

        for i in range(n):
            if i in selected or spent + costs[i] > budget + EPS:
                continue

            gain = sum(
                demands[j] for j in facility_customers[i]
                if not covered[j]
            )
            if gain <= EPS:
                continue

            if costs[i] <= EPS:
                key = (float("inf"), gain, -i)
            else:
                key = (gain / costs[i], gain, -costs[i], -i)

            if best_key is None or key > best_key:
                best_key = key
                best_i = i

        if best_i is None:
            break

        selected.add(best_i)
        spent += costs[best_i]
        for j in facility_customers[best_i]:
            covered[j] = True

    # A high-value singleton may outperform ratio-greedy selection.
    greedy_value = sum(demands[j] for j in range(m) if covered[j])
    best_single = None
    best_single_value = greedy_value
    zero_cost = {i for i in range(n) if costs[i] <= EPS and facility_customers[i]}

    zero_covered = set()
    for i in zero_cost:
        zero_covered.update(facility_customers[i])

    for i in range(n):
        if i in zero_cost or costs[i] > budget + EPS:
            continue
        candidate_covered = set(zero_covered)
        candidate_covered.update(facility_customers[i])
        value = sum(demands[j] for j in candidate_covered)
        if value > best_single_value + EPS:
            best_single_value = value
            best_single = i

    if best_single is not None:
        selected = set(zero_cost)
        selected.add(best_single)

    return sorted(selected)


def aggregate_customer_groups(customer_neighbors, demands):
    groups = defaultdict(float)
    for neighbors, demand in zip(customer_neighbors, demands):
        if demand > EPS and neighbors:
            groups[tuple(neighbors)] += float(demand)
    return [(neighbors, demand) for neighbors, demand in groups.items()]


def solve_variant(
    problem_type,
    costs,
    demands,
    customer_neighbors,
    facility_customers,
    threshold,
    budget,
    warm_start,
    time_limit,
    callback_logger=None,
):
    if time_limit <= 0.02:
        return warm_start

    n = len(costs)
    groups = aggregate_customer_groups(customer_neighbors, demands)

    if problem_type == "PSCLP" and threshold > sum(w for _, w in groups) + EPS:
        return warm_start

    try:
        model = gp.Model()
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(0.01, float(time_limit))

        x = model.addVars(n, vtype=GRB.BINARY, name="open")
        y = model.addVars(len(groups), vtype=GRB.BINARY, name="covered")

        if problem_type == "MCLP":
            for i in range(n):
                if costs[i] > budget + EPS:
                    x[i].UB = 0.0

        for g, (neighbors, _) in enumerate(groups):
            model.addConstr(y[g] <= gp.quicksum(x[i] for i in neighbors))

        if problem_type == "PSCLP":
            model.addConstr(
                gp.quicksum(groups[g][1] * y[g] for g in range(len(groups)))
                >= threshold
            )
            model.setObjective(
                gp.quicksum(costs[i] * x[i] for i in range(n)),
                GRB.MINIMIZE,
            )
        else:
            model.addConstr(
                gp.quicksum(costs[i] * x[i] for i in range(n)) <= budget
            )
            model.setObjective(
                gp.quicksum(groups[g][1] * y[g] for g in range(len(groups))),
                GRB.MAXIMIZE,
            )

        warm_set = set(warm_start)
        for i in range(n):
            x[i].Start = 1.0 if i in warm_set else 0.0

        for g, (neighbors, _) in enumerate(groups):
            y[g].Start = 1.0 if any(i in warm_set for i in neighbors) else 0.0

        x_vars = [x[i] for i in range(n)]

        def callback(model_cb, where):
            if where != GRB.Callback.MIPSOL or callback_logger is None:
                return
            try:
                values = model_cb.cbGetSolution(x_vars)
                selected = [i for i, value in enumerate(values) if value > 0.5]
                callback_logger(selected)
            except Exception:
                # Logging must never interrupt the optimization.
                pass

        model.optimize(callback if callback_logger is not None else None)

        if model.SolCount > 0:
            selected = [i for i in range(n) if x[i].X > 0.5]

            if problem_type == "PSCLP":
                if covered_demand(selected, facility_customers, demands) + EPS >= threshold:
                    return sorted(selected)
            else:
                if selected_cost(selected, costs) <= budget + 1e-5:
                    return sorted(selected)

    except gp.GurobiError:
        pass

    return warm_start


def main():
    args = parse_args()
    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    num_facilities = int(data["num_facilities"])
    num_customers = int(data["num_customers"])
    costs = [float(v) for v in data["facility_cost"]]
    demands = [float(v) for v in data["customer_demands"]]

    customer_neighbors, facility_customers = build_coverage(
        data, num_facilities, num_customers
    )

    threshold = float(
        data.get(
            "covering_demand_D",
            float(data.get("covering_demand_fraction", 0.0))
            * float(data.get("total_demand_D_bar", sum(demands))),
        )
    )
    budget = float(data.get("budget_B", 0.0))

    primary = determine_primary_problem(data)
    logger = (
        SolutionLogger(
            args.log_path,
            sense="minimize" if primary == "PSCLP" else "maximize",
        )
        if args.log_path
        else None
    )

    ps_selected = greedy_psclp(costs, demands, facility_customers, threshold)
    mc_selected = greedy_mclp(costs, demands, facility_customers, budget)

    results = {
        "PSCLP": make_variant_result(
            "PSCLP", ps_selected, costs, facility_customers, demands
        ),
        "MCLP": make_variant_result(
            "MCLP", mc_selected, costs, facility_customers, demands
        ),
    }

    if primary == "PSCLP":
        best_logged = results["PSCLP"]["objective_value"]
    else:
        best_logged = results["MCLP"]["objective_value"]

    if logger:
        logger.log_solution(best_logged, make_solution(primary, results))

    def primary_incumbent_logger(selected):
        nonlocal best_logged

        candidate = make_variant_result(
            primary, selected, costs, facility_customers, demands
        )

        if primary == "PSCLP":
            if candidate["covered_demand"] + EPS < threshold:
                return
            improved = candidate["objective_value"] < best_logged - EPS
        else:
            if selected_cost(selected, costs) > budget + 1e-5:
                return
            improved = candidate["objective_value"] > best_logged + EPS

        if not improved:
            return

        snapshot_results = {
            "PSCLP": dict(results["PSCLP"]),
            "MCLP": dict(results["MCLP"]),
        }
        snapshot_results[primary] = candidate
        best_logged = candidate["objective_value"]

        if logger:
            logger.log_solution(
                best_logged,
                make_solution(primary, snapshot_results),
            )

    order = [primary, "MCLP" if primary == "PSCLP" else "PSCLP"]

    for position, problem_type in enumerate(order):
        remaining = deadline - time.monotonic()
        if remaining <= 0.02:
            break

        if position == 0:
            # Reserve time for the other independently optimized variant.
            allocated = max(0.01, remaining * 0.65)
        else:
            allocated = max(0.01, remaining - 0.01)

        warm_start = ps_selected if problem_type == "PSCLP" else mc_selected
        selected = solve_variant(
            problem_type=problem_type,
            costs=costs,
            demands=demands,
            customer_neighbors=customer_neighbors,
            facility_customers=facility_customers,
            threshold=threshold,
            budget=budget,
            warm_start=warm_start,
            time_limit=allocated,
            callback_logger=(
                primary_incumbent_logger if problem_type == primary else None
            ),
        )

        if problem_type == "PSCLP":
            ps_selected = selected
        else:
            mc_selected = selected

        results[problem_type] = make_variant_result(
            problem_type, selected, costs, facility_customers, demands
        )

    final_solution = make_solution(primary, results)
    final_objective = final_solution["objective_value"]

    if logger:
        improved = (
            final_objective < best_logged - EPS
            if primary == "PSCLP"
            else final_objective > best_logged + EPS
        )
        if improved:
            logger.log_solution(final_objective, final_solution)

    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, indent=2)


if __name__ == "__main__":
    main()