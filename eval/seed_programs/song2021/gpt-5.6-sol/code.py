import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


EPS = 1e-9


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def project_to_bounded_simplex(values, lower, upper):
    """Repair small numerical errors while preserving lower/upper bounds and sum 1."""
    x = [min(upper, max(lower, float(v))) for v in values]

    for _ in range(3):
        delta = 1.0 - sum(x)
        if abs(delta) <= 1e-12:
            break

        if delta > 0.0:
            for k in range(len(x)):
                add = min(delta, max(0.0, upper - x[k]))
                x[k] += add
                delta -= add
                if delta <= 1e-12:
                    break
        else:
            need = -delta
            for k in range(len(x)):
                sub = min(need, max(0.0, x[k] - lower))
                x[k] -= sub
                need -= sub
                if need <= 1e-12:
                    break

    # Apply any final floating-point residual to a variable with sufficient slack.
    residual = 1.0 - sum(x)
    if abs(residual) > 1e-12:
        if residual > 0:
            for k in range(len(x)):
                if upper - x[k] >= residual - 1e-12:
                    x[k] += residual
                    break
        else:
            need = -residual
            for k in range(len(x)):
                if x[k] - lower >= need - 1e-12:
                    x[k] -= need
                    break

    return x


def make_intervals(instance):
    H = int(instance["planning_horizon_H"])
    intervals = {}

    for i, order in enumerate(instance["orders"]):
        ub = float(order["intensity_upper_bound_UB"])
        lb = float(order["intensity_lower_bound_LB"])
        release = int(order["release_date_r"])

        schema_min_len = int(order["lower_exec_interval_length_l"])
        schema_max_len = int(order["upper_exec_interval_length_l_bar"])

        bound_min_len = int(math.ceil((1.0 / ub) - 1e-10))
        bound_max_len = int(math.floor((1.0 / lb) + 1e-10))

        min_len = max(1, schema_min_len, bound_min_len)
        max_len = min(H, schema_max_len, bound_max_len)

        order_intervals = []
        first_start = max(1, release)
        for length in range(min_len, max_len + 1):
            last_start = H - length + 1
            for start in range(first_start, last_start + 1):
                finish = start + length - 1
                order_intervals.append((start, finish))

        if not order_intervals:
            raise ValueError(
                f"Order {order['order_id']} has no feasible execution interval"
            )

        intervals[i] = order_intervals

    return intervals


def allocate_for_interval(order, start, finish, current_load, capacities):
    """
    Construct an intensity profile for one interval. Lower bounds are assigned
    first; remaining work is preferentially placed in unused regular capacity.
    """
    p = float(order["work_content_p"])
    lb = float(order["intensity_lower_bound_LB"])
    ub = float(order["intensity_upper_bound_UB"])
    periods = list(range(start, finish + 1))

    q = {t: lb for t in periods}
    remaining = max(0.0, 1.0 - lb * len(periods))

    # First use regular-capacity headroom.
    free_candidates = []
    for t in periods:
        load_after_lb = current_load[t] + p * lb
        free_fraction = max(0.0, capacities[t] - load_after_lb) / p if p > 0 else 0.0
        free_candidates.append((free_fraction, t))

    free_candidates.sort(reverse=True)
    for free_fraction, t in free_candidates:
        if remaining <= 1e-12:
            break
        add = min(remaining, ub - q[t], free_fraction)
        if add > 0:
            q[t] += add
            remaining -= add

    # Any unavoidable overtime is spread toward the least loaded periods.
    if remaining > 1e-12:
        periods_by_load = sorted(
            periods,
            key=lambda t: (
                (current_load[t] + p * q[t]) / max(1.0, capacities[t]),
                current_load[t] + p * q[t],
                t,
            ),
        )
        for t in periods_by_load:
            if remaining <= 1e-12:
                break
            add = min(remaining, ub - q[t])
            if add > 0:
                q[t] += add
                remaining -= add

    repaired = project_to_bounded_simplex(
        [q[t] for t in periods], lb, ub
    )
    return {t: repaired[k] for k, t in enumerate(periods)}


def build_greedy_solution_data(instance, intervals):
    H = int(instance["planning_horizon_H"])
    capacities = [0.0] + [float(v) for v in instance["capacities"]]
    sigma = float(instance["unit_cost_nonregular_capacity_sigma"])
    orders = instance["orders"]

    current_load = [0.0] * (H + 1)
    chosen = {}
    intensities = {}

    order_sequence = sorted(
        range(len(orders)),
        key=lambda i: (
            int(orders[i]["due_date_d"]),
            int(orders[i]["release_date_r"]),
            -int(orders[i]["tardiness_cost_w"]),
            -int(orders[i]["work_content_p"]),
        ),
    )

    for i in order_sequence:
        order = orders[i]
        p = float(order["work_content_p"])
        due = int(order["due_date_d"])
        weight = float(order["tardiness_cost_w"])

        old_overtime = sum(
            max(0.0, current_load[t] - capacities[t])
            for t in range(1, H + 1)
        )

        best_score = float("inf")
        best_interval = None
        best_q = None

        for start, finish in intervals[i]:
            q = allocate_for_interval(
                order, start, finish, current_load, capacities
            )

            new_overtime = old_overtime
            for t in range(start, finish + 1):
                before = max(0.0, current_load[t] - capacities[t])
                after = max(
                    0.0, current_load[t] + p * q[t] - capacities[t]
                )
                new_overtime += after - before

            incremental_cost = (
                weight * max(0, finish - due)
                + sigma * (new_overtime - old_overtime)
            )

            if (
                incremental_cost < best_score - 1e-10
                or (
                    abs(incremental_cost - best_score) <= 1e-10
                    and (finish, start) < (best_interval[1], best_interval[0])
                )
            ):
                best_score = incremental_cost
                best_interval = (start, finish)
                best_q = q

        chosen[i] = best_interval
        for t, value in best_q.items():
            intensities[(i, t)] = value
            current_load[t] += p * value

    return chosen, intensities


def construct_solution(instance, chosen, raw_intensities):
    H = int(instance["planning_horizon_H"])
    capacities = [0.0] + [float(v) for v in instance["capacities"]]
    sigma = float(instance["unit_cost_nonregular_capacity_sigma"])
    orders = instance["orders"]

    loads = [0.0] * (H + 1)
    orders_solution = []
    tardiness_cost = 0.0

    for i, order in enumerate(orders):
        start, finish = chosen[i]
        lb = float(order["intensity_lower_bound_LB"])
        ub = float(order["intensity_upper_bound_UB"])
        p = float(order["work_content_p"])

        periods = list(range(start, finish + 1))
        values = [
            float(raw_intensities.get((i, t), 0.0))
            for t in periods
        ]
        values = project_to_bounded_simplex(values, lb, ub)

        intensity_output = {}
        for k, t in enumerate(periods):
            value = float(values[k])
            intensity_output[str(t)] = value
            loads[t] += p * value

        tardiness = max(0, finish - int(order["due_date_d"]))
        tardiness_cost += float(order["tardiness_cost_w"]) * tardiness

        orders_solution.append(
            {
                "order_id": int(order["order_id"]),
                "chosen_interval": [int(start), int(finish)],
                "intensities": intensity_output,
            }
        )

    nonregular = {}
    total_nonregular = 0.0
    for t in range(1, H + 1):
        overtime = max(0.0, loads[t] - capacities[t])
        if overtime > 1e-10:
            nonregular[str(t)] = float(overtime)
            total_nonregular += overtime

    objective = tardiness_cost + sigma * total_nonregular

    return {
        "objective_value": float(objective),
        "orders_solution": orders_solution,
        "nonregular_capacity_by_period": nonregular,
    }


def main():
    program_start = time.monotonic()
    args = parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    H = int(instance["planning_horizon_H"])
    orders = instance["orders"]
    capacities = [float(v) for v in instance["capacities"]]
    sigma = float(instance["unit_cost_nonregular_capacity_sigma"])

    intervals = make_intervals(instance)

    # Generate a guaranteed feasible incumbent and use it as the MIP start.
    greedy_chosen, greedy_q = build_greedy_solution_data(instance, intervals)
    best_solution = construct_solution(instance, greedy_chosen, greedy_q)
    best_objective = best_solution["objective_value"]

    if logger:
        logger.log_solution(best_objective, best_solution)

    remaining_time = args.time_limit - (time.monotonic() - program_start)

    if remaining_time > 0.05:
        model = gp.Model("nonpreemptive_order_scheduling")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(0.01, remaining_time)

        z = {}
        z_keys = []
        z_vars = []

        for i in range(len(orders)):
            for start, finish in intervals[i]:
                key = (i, start, finish)
                var = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"z_{i}_{start}_{finish}",
                )
                z[key] = var
                z_keys.append(key)
                z_vars.append(var)

        q = {}
        q_keys = []
        q_vars = []

        for i in range(len(orders)):
            covered_periods = sorted(
                {
                    t
                    for start, finish in intervals[i]
                    for t in range(start, finish + 1)
                }
            )
            for t in covered_periods:
                key = (i, t)
                var = model.addVar(
                    lb=0.0,
                    ub=1.0,
                    vtype=GRB.CONTINUOUS,
                    name=f"q_{i}_{t}",
                )
                q[key] = var
                q_keys.append(key)
                q_vars.append(var)

        overtime = {
            t: model.addVar(
                lb=0.0,
                vtype=GRB.CONTINUOUS,
                name=f"overtime_{t}",
            )
            for t in range(1, H + 1)
        }

        model.update()

        # Select exactly one execution interval per order.
        for i in range(len(orders)):
            model.addConstr(
                gp.quicksum(
                    z[(i, start, finish)]
                    for start, finish in intervals[i]
                )
                == 1.0,
                name=f"select_interval_{i}",
            )

        # Intensity bounds conditional on the selected interval.
        for i, order in enumerate(orders):
            lb = float(order["intensity_lower_bound_LB"])
            ub = float(order["intensity_upper_bound_UB"])

            model.addConstr(
                gp.quicksum(q[(i, t)] for ii, t in q if ii == i) == 1.0,
                name=f"complete_{i}",
            )

            periods_i = [t for ii, t in q if ii == i]
            for t in periods_i:
                covering = gp.quicksum(
                    z[(i, start, finish)]
                    for start, finish in intervals[i]
                    if start <= t <= finish
                )
                model.addConstr(
                    q[(i, t)] >= lb * covering,
                    name=f"lower_intensity_{i}_{t}",
                )
                model.addConstr(
                    q[(i, t)] <= ub * covering,
                    name=f"upper_intensity_{i}_{t}",
                )

        # Capacity and overtime constraints.
        for t in range(1, H + 1):
            usage = gp.quicksum(
                float(orders[i]["work_content_p"]) * q[(i, t)]
                for i in range(len(orders))
                if (i, t) in q
            )
            model.addConstr(
                overtime[t] >= usage - capacities[t - 1],
                name=f"capacity_{t}",
            )

        tardiness_expr = gp.quicksum(
            float(orders[i]["tardiness_cost_w"])
            * max(0, finish - int(orders[i]["due_date_d"]))
            * z[(i, start, finish)]
            for i in range(len(orders))
            for start, finish in intervals[i]
        )

        overtime_expr = sigma * gp.quicksum(
            overtime[t] for t in range(1, H + 1)
        )

        model.setObjective(tardiness_expr + overtime_expr, GRB.MINIMIZE)

        # Install the greedy MIP start.
        for key, var in z.items():
            i, start, finish = key
            var.Start = 1.0 if greedy_chosen[i] == (start, finish) else 0.0

        for key, var in q.items():
            var.Start = float(greedy_q.get(key, 0.0))

        greedy_load = [0.0] * (H + 1)
        for (i, t), value in greedy_q.items():
            greedy_load[t] += float(orders[i]["work_content_p"]) * value
        for t in range(1, H + 1):
            overtime[t].Start = max(0.0, greedy_load[t] - capacities[t - 1])

        logged_best = [best_objective]

        def extract_callback_solution(cb_model):
            z_values = cb_model.cbGetSolution(z_vars)
            q_values = cb_model.cbGetSolution(q_vars)

            selected = {}
            best_z_by_order = {}
            for key, value in zip(z_keys, z_values):
                i, start, finish = key
                if i not in best_z_by_order or value > best_z_by_order[i][0]:
                    best_z_by_order[i] = (value, start, finish)

            for i in range(len(orders)):
                _, start, finish = best_z_by_order[i]
                selected[i] = (start, finish)

            intensity_values = {
                key: float(value)
                for key, value in zip(q_keys, q_values)
            }
            return construct_solution(instance, selected, intensity_values)

        def incumbent_callback(cb_model, where):
            if where != GRB.Callback.MIPSOL or logger is None:
                return

            candidate_obj = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
            if candidate_obj >= logged_best[0] - 1e-8:
                return

            candidate_solution = extract_callback_solution(cb_model)
            projected_obj = candidate_solution["objective_value"]

            if projected_obj < logged_best[0] - 1e-8:
                logger.log_solution(projected_obj, candidate_solution)
                logged_best[0] = projected_obj

        try:
            if logger:
                model.optimize(incumbent_callback)
            else:
                model.optimize()

            if model.SolCount > 0:
                selected = {}
                for i in range(len(orders)):
                    best_value = -float("inf")
                    best_interval = None
                    for start, finish in intervals[i]:
                        value = z[(i, start, finish)].X
                        if value > best_value:
                            best_value = value
                            best_interval = (start, finish)
                    selected[i] = best_interval

                final_q = {
                    key: float(var.X)
                    for key, var in q.items()
                }
                final_solution = construct_solution(instance, selected, final_q)

                if final_solution["objective_value"] < best_objective - 1e-8:
                    best_solution = final_solution
                    best_objective = final_solution["objective_value"]

                    if logger and best_objective < logged_best[0] - 1e-8:
                        logger.log_solution(best_objective, best_solution)
                        logged_best[0] = best_objective

        except gp.GurobiError:
            # The initial greedy solution remains available if optimization fails.
            pass

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution, f, indent=2)


if __name__ == "__main__":
    main()