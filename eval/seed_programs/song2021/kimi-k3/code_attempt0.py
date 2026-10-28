import argparse
import json
import math
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def ceil_with_tolerance(x, tol=1e-9):
    return int(math.ceil(x - tol))


def floor_with_tolerance(x, tol=1e-9):
    return int(math.floor(x + tol))


def project_to_box_sum(raw_values, lb, ub):
    """
    Projects values approximately onto [lb, ub] and adjusts them so that their
    sum is very close to one.  Used to clean up tiny solver numerical noise.
    """
    n = len(raw_values)
    if n == 0:
        return []
    if n * lb > 1.0 + 1e-6 or n * ub < 1.0 - 1e-6:
        raise ValueError("Intensity bounds are incompatible with a unit sum.")

    vals = []
    for v in raw_values:
        if not math.isfinite(v):
            v = 1.0 / n
        vals.append(min(max(float(v), lb), ub))

    for _ in range(200):
        residual = 1.0 - sum(vals)
        if abs(residual) <= 1e-11:
            return vals

        if residual > 0:
            idx = max(range(n), key=lambda j: ub - vals[j])
            room = ub - vals[idx]
            if room <= 1e-14:
                break
            delta = min(room, residual)
            vals[idx] += delta
            residual -= delta
        else:
            idx = max(range(n), key=lambda j: vals[j] - lb)
            room = vals[idx] - lb
            if room <= 1e-14:
                break
            delta = max(-room, residual)
            vals[idx] += delta
            residual -= delta

        if abs(residual) <= 1e-11:
            return vals

    # Last-resort uniform profile; feasible interval lengths guarantee that
    # this lies between lb and ub up to harmless numerical tolerance.
    uniform = 1.0 / n
    if uniform < lb - 1e-8 or uniform > ub + 1e-8:
        raise ValueError("Could not construct a feasible intensity profile.")
    return [uniform] * n


def load_and_prepare(instance_path):
    with open(instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    horizon = int(data["planning_horizon_H"])
    if horizon <= 0:
        raise ValueError("The planning horizon must be positive.")

    capacities = [float(x) for x in data["capacities"]]
    if len(capacities) != horizon:
        raise ValueError("The capacities list length must equal the horizon.")

    raw_orders = data["orders"]
    releases = [int(o["release_date_r"]) for o in raw_orders]

    if releases and all(0 <= r <= horizon - 1 for r in releases) and any(r == 0 for r in releases):
        first_period = 0
    elif all(1 <= r <= horizon for r in releases):
        first_period = 1
    elif not releases:
        first_period = 1
    else:
        raise ValueError("Release dates do not consistently fit a 0-based or 1-based horizon.")

    period_labels = list(range(first_period, first_period + horizon))

    orders = []
    for o in raw_orders:
        lb = float(o["intensity_lower_bound_LB"])
        ub = float(o["intensity_upper_bound_UB"])
        if lb <= 0 or ub <= 0:
            raise ValueError(f"Order {o.get('order_id')} has nonpositive intensity bounds.")

        if lb > ub + 1e-9:
            raise ValueError(f"Order {o.get('order_id')} has intensity lower bound above upper bound.")

        derived_min_len = ceil_with_tolerance(1.0 / ub)
        derived_max_len = floor_with_tolerance(1.0 / lb)

        explicit_min = o.get("lower_exec_interval_length_l")
        explicit_max = o.get("upper_exec_interval_length_l_bar")

        min_len = derived_min_len
        max_len = min(derived_max_len, horizon)

        if explicit_min is not None:
            min_len = max(min_len, int(explicit_min))
        if explicit_max is not None:
            max_len = min(max_len, int(explicit_max))

        if min_len > max_len:
            raise ValueError(
                f"Order {o.get('order_id')} has incompatible intensity and execution-length bounds."
            )

        valid_starts = [
            idx
            for idx, label in enumerate(period_labels)
            if label >= int(o["release_date_r"])
            and label + min_len - 1 <= period_labels[-1]
        ]

        if not valid_starts:
            raise ValueError(f"Order {o.get('order_id')} has no feasible execution interval.")

        info = {
            "order_id": int(o["order_id"]),
            "p": float(o["work_content_p"]),
            "lb": lb,
            "ub": ub,
            "release": int(o["release_date_r"]),
            "due": int(o["due_date_d"]),
            "w": float(o["tardiness_cost_w"]),
            "min_len": min_len,
            "max_len": max_len,
            "valid_starts": valid_starts,
        }
        orders.append(info)

    return {
        "horizon": horizon,
        "periods": period_labels,
        "capacities": capacities,
        "orders": orders,
        "sigma": float(data["unit_cost_nonregular_capacity_sigma"]),
    }


def assemble_solution(orders, period_labels, capacities, sigma, assignments):
    """
    assignments[i] is a tuple (start_period_label, finish_period_label,
    raw_intensity_values_for_the_window).
    """
    horizon = len(period_labels)
    label_to_index = {label: idx for idx, label in enumerate(period_labels)}
    loads = [0.0] * horizon
    fixed_tardiness = 0.0
    solution_entries = []

    for idx, info in enumerate(orders):
        start, finish, raw_values = assignments[idx]
        start_idx = label_to_index[start]
        finish_idx = label_to_index[finish]

        if finish_idx < start_idx:
            raise ValueError("Finish period precedes start period.")

        raw_values = list(raw_values)
        if len(raw_values) != finish_idx - start_idx + 1:
            raise ValueError("Intensity profile does not match the chosen execution window.")

        values = project_to_box_sum(raw_values, info["lb"], info["ub"])

        intensities = {}
        for offset, k in enumerate(range(start_idx, finish_idx + 1)):
            val = values[offset]
            intensities[str(period_labels[k])] = float(val)
            loads[k] += info["p"] * val

        fixed_tardiness += info["w"] * max(finish - info["due"], 0.0)
        solution_entries.append(
            {
                "order_id": info["order_id"],
                "chosen_interval": [int(start), int(finish)],
                "intensities": intensities,
            }
        )

    overtime = {}
    overtime_cost = 0.0
    for k, label in enumerate(period_labels):
        ot = loads[k] - capacities[k]
        if ot > 1e-9:
            overtime[str(label)] = float(ot)
            overtime_cost += ot

    objective = fixed_tardiness + sigma * overtime_cost
    solution = {
        "objective_value": float(objective),
        "orders_solution": solution_entries,
        "nonregular_capacity_by_period": overtime,
    }
    return objective, solution


def build_fallback_solution(instance):
    orders = instance["orders"]
    periods = instance["periods"]
    capacities = instance["capacities"]
    sigma = instance["sigma"]
    horizon = len(periods)
    assignments = [None] * len(orders)

    candidate_count = 0
    for info in orders:
        for s in info["valid_starts"]:
            usable_max_len = min(info["max_len"], horizon - s)
            candidate_count += max(0, usable_max_len - info["min_len"] + 1)

    # A light greedy pass is worthwhile for modest interval sets.  For very
    # large ones, an immediate feasible first-interval solution preserves time
    # for the exact MIP.
    use_greedy = candidate_count <= 300000

    if use_greedy:
        loads = [0.0] * horizon
        sort_keys = sorted(
            range(len(orders)),
            key=lambda i: (
                orders[i]["due"],
                -orders[i]["w"],
                -orders[i]["p"],
                orders[i]["order_id"],
            ),
        )
        for i in sort_keys:
            info = orders[i]
            best = None
            for s in info["valid_starts"]:
                usable_max_len = min(info["max_len"], horizon - s)
                for length in range(info["min_len"], usable_max_len + 1):
                    f = s + length - 1
                    uniform = 1.0 / length
                    added = info["p"] * uniform
                    overtime_increase = 0.0
                    for k in range(s, f + 1):
                        before = max(loads[k] - capacities[k], 0.0)
                        after = max(loads[k] + added - capacities[k], 0.0)
                        overtime_increase += after - before

                    finish_label = periods[f]
                    tardiness = info["w"] * max(finish_label - info["due"], 0.0)
                    score = tardiness + sigma * overtime_increase
                    key = (score, finish_label, periods[s])
                    if best is None or key < best[0]:
                        best = (key, s, f, uniform)

            if best is None:
                raise ValueError(f"Order {info['order_id']} has no feasible execution interval.")

            _, s, f, uniform = best
            raw_values = [uniform] * (f - s + 1)
            assignments[i] = (periods[s], periods[f], raw_values)
            for k in range(s, f + 1):
                loads[k] += info["p"] * uniform
    else:
        for i, info in enumerate(orders):
            s = info["valid_starts"][0]
            length = min(info["min_len"], horizon - s)
            f = s + length - 1
            assignments[i] = (
                periods[s],
                periods[f],
                [1.0 / length] * length,
            )

    return assemble_solution(orders, periods, capacities, sigma, assignments)


def build_mip(instance, deadline):
    horizon = instance["horizon"]
    periods = instance["periods"]
    capacities = instance["capacities"]
    orders = instance["orders"]
    sigma = instance["sigma"]
    n = len(orders)

    model = gp.Model("continuous_execution_scheduling")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = max(1e-4, deadline - time.monotonic())

    total_work = sum(info["p"] for info in orders)
    objective = gp.LinExpr()

    overtime_vars = []
    for k in range(horizon):
        upper = max(0.0, total_work - capacities[k])
        v = model.addVar(lb=0.0, ub=upper, name=f"overtime_{periods[k]}")
        overtime_vars.append(v)
        objective.add(float(sigma) * v, 1.0)

    x_vars = []
    start_vars = []
    start_positions = []
    end_vars = []
    active_vars = []

    for i, info in enumerate(orders):
        x_row = [
            model.addVar(lb=0.0, ub=1.0, name=f"x_{i}_{periods[k]}")
            for k in range(horizon)
        ]
        x_vars.append(x_row)
        model.addConstr(gp.quicksum(x_row), GRB.EQUAL, 1.0, name=f"complete_order_{i}")

        s_dict = {}
        positions = []
        for k in info["valid_starts"]:
            v = model.addVar(vtype=GRB.BINARY, name=f"start_{i}_{periods[k]}")
            s_dict[k] = v
            positions.append(k)
        start_vars.append(s_dict)
        start_positions.append(positions)

        e_row = [
            model.addVar(vtype=GRB.BINARY, name=f"finish_{i}_{periods[k]}")
            for k in range(horizon)
        ]
        end_vars.append(e_row)

        a_row = [
            model.addVar(lb=0.0, ub=1.0, name=f"active_{i}_{periods[k]}")
            for k in range(horizon)
        ]
        active_vars.append(a_row)

        model.addConstr(gp.quicksum(s_dict.values()), GRB.EQUAL, 1.0, name=f"one_start_{i}")
        model.addConstr(gp.quicksum(e_row), GRB.EQUAL, 1.0, name=f"one_finish_{i}")

        duration_difference = gp.LinExpr()
        for k, v in e_row:
            duration_difference.add(v, float(k))
        for k, v in s_dict.items():
            duration_difference.add(v, -float(k))

        model.addConstr(
            duration_difference,
            GRB.GREATER_EQUAL,
            float(info["min_len"] - 1),
            name=f"min_duration_{i}",
        )
        model.addConstr(
            duration_difference,
            GRB.LESS_EQUAL,
            float(info["max_len"] - 1),
            name=f"max_duration_{i}",
        )

        for k in range(horizon):
            if k == 0:
                activation_expr = s_dict.get(0, 0.0)
            else:
                activation_expr = (
                    a_row[k - 1]
                    + s_dict.get(k, 0.0)
                    - e_row[k - 1]
                )
            model.addConstr(
                a_row[k],
                GRB.EQUAL,
                activation_expr,
                name=f"activity_{i}_{k}",
            )
            model.addConstr(
                x_row[k],
                GRB.LESS_EQUAL,
                info["ub"] * a_row[k],
                name=f"upper_intensity_{i}_{k}",
            )
            model.addConstr(
                x_row[k],
                GRB.GREATER_EQUAL,
                info["lb"] * a_row[k],
                name=f"lower_intensity_{i}_{k}",
            )

        for k, finish_var in enumerate(e_row):
            tardiness_periods = max(periods[k] - info["due"], 0)
            coefficient = info["w"] * tardiness_periods
            if coefficient != 0.0:
                objective.add(finish_var, float(coefficient))

    for k in range(horizon):
        load_expr = gp.LinExpr()
        for i, info in enumerate(orders):
            load_expr.add(x_vars[i][k], info["p"])
        model.addConstr(
            overtime_vars[k],
            GRB.GREATER_EQUAL,
            load_expr - capacities[k],
            name=f"overtime_def_{k}",
        )

    model.setObjective(objective, GRB.MINIMIZE)

    solution_vars = []
    metadata = []
    for i in range(n):
        for k, v in enumerate(x_vars[i]):
            solution_vars.append(v)
            metadata.append(("x", i, k))
    for i in range(n):
        for k, v in start_vars[i].items():
            solution_vars.append(v)
            metadata.append(("s", i, k))
    for i in range(n):
        for k, v in enumerate(end_vars[i]):
            solution_vars.append(v)
            metadata.append(("e", i, k))
    for k, v in enumerate(overtime_vars):
        solution_vars.append(v)
        metadata.append(("o", -1, k))

    structure = {
        "x": x_vars,
        "s": start_vars,
        "s_positions": start_positions,
        "e": end_vars,
        "a": active_vars,
        "o": overtime_vars,
        "solution_vars": solution_vars,
        "metadata": metadata,
    }
    return model, structure


def parse_mip_values(values, instance, structure):
    orders = instance["orders"]
    periods = instance["periods"]
    capacities = instance["capacities"]
    sigma = instance["sigma"]

    x_data = [[0.0] * len(periods) for _ in orders]
    start_data = [[0.0] * len(periods) for _ in orders]
    end_data = [[0.0] * len(periods) for _ in orders]

    metadata = structure["metadata"]
    for idx, (kind, i, k) in enumerate(metadata):
        val = float(values[idx])
        if kind == "x":
            x_data[i][k] = val
        elif kind == "s":
            start_data[i][k] = val
        elif kind == "e":
            end_data[i][k] = val

    assignments = []
    for i, info in enumerate(orders):
        possible_starts = structure["s_positions"][i]
        if not possible_starts:
            raise ValueError("MIP parser encountered an order with no start variable.")

        start_idx = max(possible_starts, key=lambda k: start_data[i][k])
        finish_idx = max(range(len(periods)), key=lambda k: end_data[i][k])

        if finish_idx < start_idx:
            # Numerically inconsistent incumbent; repair by using the start and
            # the minimum feasible duration.  This should not normally happen.
            finish_idx = min(start_idx + info["min_len"] - 1, len(periods) - 1)

        raw_values = x_data[i][start_idx : finish_idx + 1]
        try:
            normalized = project_to_box_sum(raw_values, info["lb"], info["ub"])
        except Exception:
            length = finish_idx - start_idx + 1
            normalized = [1.0 / length] * length

        assignments.append(
            (
                periods[start_idx],
                periods[finish_idx],
                normalized,
            )
        )

    return assemble_solution(orders, periods, capacities, sigma, assignments)


def apply_mip_start(instance, structure, solution):
    periods = instance["periods"]
    label_to_index = {label: idx for idx, label in enumerate(periods)}
    by_id = {
        entry["order_id"]: entry for entry in solution["orders_solution"]
    }

    for i, info in enumerate(instance["orders"]):
        entry = by_id[info["order_id"]]
        start, finish = entry["chosen_interval"]
        start_idx = label_to_index[start]
        finish_idx = label_to_index[finish]

        for k, var in structure["s"][i].items():
            var.Start = 1.0 if k == start_idx else 0.0
        for k, var in enumerate(structure["e"][i]):
            var.Start = 1.0 if k == finish_idx else 0.0

        loads = 0.0
        for k, var in enumerate(structure["x"][i]):
            label = str(periods[k])
            val = float(entry["intensities"].get(label, 0.0))
            var.Start = val
            loads += info["p"] * val

        for k, var in enumerate(structure["a"][i]):
            var.Start = 1.0 if start_idx <= k <= finish_idx else 0.0

    # Reconstruct period loads for an overtime warm start.
    horizon = instance["horizon"]
    period_loads = [0.0] * horizon
    for i, info in enumerate(instance["orders"]):
        entry = by_id[info["order_id"]]
        for k, label in enumerate(periods):
            period_loads[k] += info["p"] * float(entry["intensities"].get(str(label), 0.0))

    for k, var in enumerate(structure["o"]):
        var.Start = max(period_loads[k] - instance["capacities"][k], 0.0)


def main():
    args = parse_args()
    deadline = time.monotonic() + max(0, args.time_limit)

    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path
        else None
    )

    instance = load_and_prepare(args.instance_path)

    best_objective, fallback_solution = build_fallback_solution(instance)
    best_solution = fallback_solution
    if logger:
        logger.log_solution(best_objective, best_solution)

    if time.monotonic() >= deadline:
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(best_solution, f, indent=2)
        return

    model = None
    try:
        model, structure = build_mip(instance, deadline)
        apply_mip_start(instance, structure, best_solution)

        best_holder = [best_objective, best_solution]

        def incumbent_callback(m, where):
            if where != GRB.Callback.MIPSOL:
                return
            try:
                values = m.cbGetSolution(structure["solution_vars"])
                objective, solution = parse_mip_values(values, instance, structure)
            except Exception:
                try:
                    raw_objective = float(m.cbGet(GRB.Callback.MIPSOL_OBJ))
                except Exception:
                    return
                if raw_objective < best_holder[0] - 1e-7:
                    best_holder[0] = raw_objective
                    if logger:
                        logger.log(raw_objective)
                return

            if objective < best_holder[0] - 1e-7:
                best_holder[0] = objective
                best_holder[1] = solution
                if logger:
                    logger.log_solution(objective, solution)

        remaining = deadline - time.monotonic()
        if remaining > 0:
            model.Params.TimeLimit = max(1e-4, remaining)
            model.optimize(incumbent_callback)

            if model.SolCount > 0:
                values = [v.X for v in structure["solution_vars"]]
                objective, solution = parse_mip_values(values, instance, structure)
                if objective < best_holder[0] - 1e-7:
                    best_holder[0] = objective
                    best_holder[1] = solution
                    if logger:
                        logger.log_solution(objective, solution)

            if best_holder[1] is not None:
                best_objective, best_solution = best_holder[0], best_holder[1]

    except gp.GurobiError:
        # Keep the already-logged feasible fallback under licensing, model,
        # or numerical failures.
        pass
    finally:
        if model is not None:
            try:
                model.dispose()
            except Exception:
                pass

    best_solution["objective_value"] = float(best_objective)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution, f, indent=2)


if __name__ == "__main__":
    main()