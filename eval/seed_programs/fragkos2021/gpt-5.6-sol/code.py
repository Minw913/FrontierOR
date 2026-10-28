import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def read_instance(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_solution(path, solution):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(solution, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    data = read_instance(args.instance_path)

    num_facilities = int(data["num_facilities"])
    num_customers = int(data["num_customers"])
    nodes = data["nodes"]

    fixed_cost = [
        float(nodes[i].get("fixed_setup_cost", 0.0))
        for i in range(num_facilities)
    ]
    demand = [
        float(nodes[j].get("demand", 0.0))
        for j in range(num_customers)
    ]

    survival = []
    for i in range(num_facilities):
        node = nodes[i]
        if "survival_probability" in node:
            value = float(node["survival_probability"])
        else:
            value = 1.0 - float(node.get("disruption_probability", 0.0))
        survival.append(min(1.0, max(0.0, value)))

    failure = [1.0 - value for value in survival]

    transportation_cost = [
        [float(v) for v in data["transportation_cost_matrix"][j][:num_facilities]]
        for j in range(num_customers)
    ]
    penalty_cost = [
        float(v) for v in data["penalty_cost"][:num_customers]
    ]

    # Facilities whose cost is not below the penalty option can never be the
    # selected service option for that customer.
    customer_orders = []
    for j in range(num_customers):
        order = [
            i for i in range(num_facilities)
            if transportation_cost[j][i] < penalty_cost[j] - 1e-12
        ]
        order.sort(key=lambda i: (transportation_cost[j][i], i))
        customer_orders.append(order)

    def evaluate(open_vector):
        """Exact expectation under independent facility disruptions."""
        total = math.fsum(
            fixed_cost[i] for i in range(num_facilities) if open_vector[i]
        )

        transport_total = 0.0
        for j in range(num_customers):
            expected_unit_cost = penalty_cost[j]
            order = customer_orders[j]

            # Backward recursion:
            # E = s_i*c_ij + (1-s_i)*E if facility i is opened.
            for i in reversed(order):
                if open_vector[i]:
                    expected_unit_cost = (
                        survival[i] * transportation_cost[j][i]
                        + failure[i] * expected_unit_cost
                    )
            transport_total += demand[j] * expected_unit_cost

        return float(total + transport_total)

    best_open = [0] * num_facilities
    best_value = evaluate(best_open)
    best_logged_value = [best_value]

    initial_solution = {
        "objective_value": float(best_value),
        "facility_locations": list(best_open),
    }
    if logger:
        logger.log_solution(best_value, initial_solution)

    # Greedy construction supplies Gurobi with a useful feasible MIP start.
    pair_count = sum(len(order) for order in customer_orders)
    greedy_budget = min(5.0, max(0.0, 0.10 * args.time_limit))
    greedy_deadline = min(deadline, time.monotonic() + greedy_budget)

    # Avoid spending most of a short run on a very large greedy pass.
    run_greedy = (
        num_facilities > 0
        and pair_count <= max(100000, int(max(1, args.time_limit) * 150000))
    )

    if run_greedy:
        while time.monotonic() < greedy_deadline:
            gains = [0.0] * num_facilities

            for j in range(num_customers):
                order = customer_orders[j]
                if not order or demand[j] == 0.0:
                    continue

                prefix_no_service = [1.0] * len(order)
                probability = 1.0
                for r, i in enumerate(order):
                    prefix_no_service[r] = probability
                    if best_open[i]:
                        probability *= failure[i]

                expected_more_expensive = penalty_cost[j]
                for r in range(len(order) - 1, -1, -1):
                    i = order[r]
                    if not best_open[i]:
                        improvement = (
                            survival[i]
                            * prefix_no_service[r]
                            * (
                                expected_more_expensive
                                - transportation_cost[j][i]
                            )
                        )
                        if improvement > 0.0:
                            gains[i] += demand[j] * improvement
                    else:
                        expected_more_expensive = (
                            survival[i] * transportation_cost[j][i]
                            + failure[i] * expected_more_expensive
                        )

            chosen = -1
            chosen_net_improvement = 0.0
            for i in range(num_facilities):
                if not best_open[i]:
                    net_improvement = gains[i] - fixed_cost[i]
                    if net_improvement > chosen_net_improvement + 1e-10:
                        chosen = i
                        chosen_net_improvement = net_improvement

            if chosen < 0:
                break

            candidate = list(best_open)
            candidate[chosen] = 1
            candidate_value = evaluate(candidate)

            if candidate_value < best_value - 1e-8:
                best_open = candidate
                best_value = candidate_value
                solution = {
                    "objective_value": float(best_value),
                    "facility_locations": list(best_open),
                }
                if logger:
                    logger.log_solution(best_value, solution)
                best_logged_value[0] = best_value
            else:
                break

    remaining = deadline - time.monotonic()

    if num_facilities == 0 or remaining <= 0.02:
        final_solution = {
            "objective_value": float(best_value),
            "facility_locations": list(best_open),
        }
        write_solution(args.solution_path, final_solution)
        return

    # If constructing the full MIP would itself consume an unreasonable
    # fraction of the remaining time, retain the feasible heuristic solution.
    estimated_constraints = 2 * pair_count
    if estimated_constraints > max(250000, int(remaining * 120000)):
        final_solution = {
            "objective_value": float(best_value),
            "facility_locations": list(best_open),
        }
        write_solution(args.solution_path, final_solution)
        return

    try:
        model = gp.Model("reliable_uncapacitated_facility_location")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1

        x = model.addVars(num_facilities, vtype=GRB.BINARY, name="open")

        objective = gp.LinExpr()
        for i in range(num_facilities):
            objective += fixed_cost[i] * x[i]

        z_by_customer = []

        for j in range(num_customers):
            order = customer_orders[j]

            if not order:
                objective += demand[j] * penalty_cost[j]
                z_by_customer.append([])
                continue

            z_vars = []
            for r, i in enumerate(order):
                z = model.addVar(
                    lb=0.0,
                    ub=1.0,
                    vtype=GRB.CONTINUOUS,
                    name=f"z_{j}_{r}",
                )
                z_vars.append(z)

                if r == 0:
                    # z = 1 if closed, and z = failure probability if open.
                    model.addConstr(z >= failure[i])
                    model.addConstr(z >= 1.0 - x[i])
                else:
                    previous_z = z_vars[r - 1]
                    model.addConstr(z >= failure[i] * previous_z)
                    model.addConstr(z >= previous_z - x[i])

            first_cost = transportation_cost[j][order[0]]
            objective += demand[j] * first_cost

            for r in range(len(order) - 1):
                current_cost = transportation_cost[j][order[r]]
                next_cost = transportation_cost[j][order[r + 1]]
                coefficient = demand[j] * (next_cost - current_cost)
                if coefficient != 0.0:
                    objective += coefficient * z_vars[r]

            last_cost = transportation_cost[j][order[-1]]
            last_coefficient = demand[j] * (penalty_cost[j] - last_cost)
            if last_coefficient != 0.0:
                objective += last_coefficient * z_vars[-1]

            z_by_customer.append(z_vars)

        model.setObjective(objective, GRB.MINIMIZE)

        # Complete feasible MIP start.
        for i in range(num_facilities):
            x[i].Start = float(best_open[i])

        for j in range(num_customers):
            no_available_probability = 1.0
            for r, i in enumerate(customer_orders[j]):
                if best_open[i]:
                    no_available_probability *= failure[i]
                z_by_customer[j][r].Start = no_available_probability

        model.update()
        remaining = max(0.01, deadline - time.monotonic())
        model.Params.TimeLimit = remaining

        callback_best = {
            "value": best_value,
            "open": list(best_open),
        }

        def incumbent_callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return
            try:
                values = cb_model.cbGetSolution([x[i] for i in range(num_facilities)])
                open_vector = [1 if value >= 0.5 else 0 for value in values]
                true_value = evaluate(open_vector)

                if true_value < callback_best["value"] - 1e-8:
                    callback_best["value"] = true_value
                    callback_best["open"] = open_vector

                if logger and true_value < best_logged_value[0] - 1e-8:
                    solution = {
                        "objective_value": float(true_value),
                        "facility_locations": list(open_vector),
                    }
                    logger.log_solution(true_value, solution)
                    best_logged_value[0] = true_value
            except Exception:
                # Logging must not terminate the optimization search.
                pass

        model.optimize(incumbent_callback)

        if callback_best["value"] < best_value - 1e-8:
            best_value = callback_best["value"]
            best_open = list(callback_best["open"])

        if model.SolCount > 0:
            mip_open = [
                1 if x[i].X >= 0.5 else 0
                for i in range(num_facilities)
            ]
            mip_value = evaluate(mip_open)
            if mip_value < best_value - 1e-8:
                best_value = mip_value
                best_open = mip_open

    except gp.GurobiError:
        # The already constructed heuristic solution remains feasible.
        pass

    final_solution = {
        "objective_value": float(best_value),
        "facility_locations": [int(v) for v in best_open],
    }

    if logger and best_value < best_logged_value[0] - 1e-8:
        logger.log_solution(best_value, final_solution)

    write_solution(args.solution_path, final_solution)


if __name__ == "__main__":
    main()