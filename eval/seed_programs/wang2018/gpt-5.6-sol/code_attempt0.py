import argparse
import json
import math
import os
import random
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def load_instance(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path, data):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(temporary, path)


def prepare_data(instance):
    vessels = instance["vessels"]
    V = len(vessels)
    B = int(instance["num_berths"])
    S = int(instance["num_subblocks"])
    H = int(instance["H"])
    E = int(instance["E"])
    T = H + E
    num_qcs = int(instance["num_QCs"])

    raw_capacity = instance["QC_capacity_per_timestep"]
    capacity = {
        t: min(num_qcs, int(raw_capacity[t - 1]))
        for t in range(1, T + 1)
    }

    # Each option is (berth, profile, alpha, beta, handling_time, qc_tuple,
    #                 penalty_cost, unloading_cost, loading_estimate).
    options = [[] for _ in range(V)]
    for v, vessel in enumerate(vessels):
        feasible_start, feasible_end = map(int, vessel["feasible_time_window"])
        expected_start, expected_end = map(int, vessel["expected_time_window"])
        penalty_coeff = float(vessel["penalty_cost_coefficient"])
        unload = int(vessel["unloading_containers"])
        load = int(vessel["loading_containers"])
        reserved = int(vessel["reserved_subblocks"])

        for p, profile in enumerate(vessel["qc_profiles"]):
            duration = int(profile["handling_time"])
            qc = tuple(int(q) for q in profile["qc_per_timestep"][:duration])
            if duration <= 0 or len(qc) != duration:
                continue

            first_alpha = max(1, feasible_start)
            last_alpha = min(H, feasible_end - duration + 1, T - duration + 1)
            if first_alpha > last_alpha:
                continue

            for alpha in range(first_alpha, last_alpha + 1):
                beta = alpha + duration - 1
                penalty = penalty_coeff * (
                    max(0, expected_start - alpha)
                    + max(0, beta - expected_end)
                )

                for b in range(B):
                    unloading_cost = (
                        float(instance["penalty_cost_coefficient_co"])
                        * float(instance["average_unloading_route_lengths_D_U"][b])
                        * unload
                    )

                    if reserved > 0 and load > 0:
                        route_costs = sorted(
                            float(instance["loading_route_lengths_D_L"][s][b])
                            for s in range(S)
                        )
                        k = min(reserved, len(route_costs))
                        loading_estimate = (
                            float(instance["penalty_cost_coefficient_co"])
                            * load
                            / reserved
                            * sum(route_costs[:k])
                        )
                    else:
                        loading_estimate = 0.0

                    options[v].append(
                        (
                            b,
                            p,
                            alpha,
                            beta,
                            duration,
                            qc,
                            penalty,
                            unloading_cost,
                            loading_estimate,
                        )
                    )

    return {
        "V": V,
        "B": B,
        "S": S,
        "H": H,
        "E": E,
        "T": T,
        "capacity": capacity,
        "options": options,
    }


def cyclic_slot(t, H):
    return t if t <= H else t - H


def build_solution(instance, prepared, selected_options, assigned_subblocks):
    vessels = instance["vessels"]
    B = prepared["B"]
    H = prepared["H"]

    result_vessels = []
    berth_members = [[] for _ in range(B)]
    qc_usage = {}

    penalty_total = 0.0
    loading_total = 0.0
    unloading_total = 0.0
    co = float(instance["penalty_cost_coefficient_co"])

    for v, option_index in enumerate(selected_options):
        option = prepared["options"][v][option_index]
        b, p, alpha, beta, duration, qc, penalty, unload_cost, _ = option
        vessel = vessels[v]
        subs = sorted(int(s) for s in assigned_subblocks[v])

        penalty_total += penalty
        unloading_total += unload_cost

        reserved = int(vessel["reserved_subblocks"])
        load = int(vessel["loading_containers"])
        if reserved > 0:
            unit_load = co * load / reserved
            for s in subs:
                loading_total += (
                    unit_load
                    * float(instance["loading_route_lengths_D_L"][s][b])
                )

        berth_members[b].append((alpha, beta))
        for offset, q in enumerate(qc):
            t = alpha + offset
            qc_usage[t] = qc_usage.get(t, 0) + int(q)

        result_vessels.append(
            {
                "vessel_id": int(vessel["vessel_id"]),
                "alpha": int(alpha),
                "beta": int(beta),
                "berth": int(b),
                "profile": int(p),
                "subblocks": subs,
                "handling_time": int(duration),
                "qc_per_timestep": [int(q) for q in qc],
            }
        )

    result_berths = []
    for b in range(B):
        if berth_members[b]:
            rho = min(a for a, _ in berth_members[b])
            varsigma = max(e for _, e in berth_members[b])
        else:
            rho = 0
            varsigma = 0
        result_berths.append(
            {
                "berth": int(b),
                "rho": int(rho),
                "varsigma": int(varsigma),
            }
        )

    objective = penalty_total + loading_total + unloading_total
    return {
        "objective_value": float(objective),
        "vessels": result_vessels,
        "berths": result_berths,
        "qc_usage": {
            str(t): int(qc_usage[t])
            for t in sorted(qc_usage)
            if qc_usage[t] != 0
        },
    }


def greedy_subblock_assignment(instance, prepared, selected_options):
    V, S = prepared["V"], prepared["S"]
    vessels = instance["vessels"]
    total_required = sum(int(v["reserved_subblocks"]) for v in vessels)
    if total_required > S:
        return None

    # Vessels for which berth choice causes more differentiated costs are
    # handled first, reducing the damage caused by the greedy allocation.
    priorities = []
    for v in range(V):
        r = int(vessels[v]["reserved_subblocks"])
        if r <= 0:
            priorities.append((0.0, v))
            continue
        b = prepared["options"][v][selected_options[v]][0]
        costs = sorted(
            float(instance["loading_route_lengths_D_L"][s][b])
            for s in range(S)
        )
        spread = costs[min(r, S) - 1] - costs[0] if S else 0.0
        scale = int(vessels[v]["loading_containers"]) / r
        priorities.append((-spread * scale, v))

    available = set(range(S))
    assignment = [[] for _ in range(V)]
    for _, v in sorted(priorities):
        r = int(vessels[v]["reserved_subblocks"])
        if r == 0:
            continue
        b = prepared["options"][v][selected_options[v]][0]
        candidates = sorted(
            available,
            key=lambda s: (
                float(instance["loading_route_lengths_D_L"][s][b]),
                s,
            ),
        )
        if len(candidates) < r:
            return None
        chosen = candidates[:r]
        assignment[v] = chosen
        available.difference_update(chosen)

    return assignment


def find_greedy_solution(instance, prepared, deadline):
    V = prepared["V"]
    B = prepared["B"]
    H = prepared["H"]
    capacity = prepared["capacity"]
    options = prepared["options"]

    if any(not opts for opts in options):
        return None

    option_counts = [len(opts) for opts in options]
    latest = [
        int(instance["vessels"][v]["feasible_time_window"][1])
        for v in range(V)
    ]
    max_duration = [
        max(o[4] for o in options[v])
        for v in range(V)
    ]

    base_orders = [
        sorted(range(V), key=lambda v: (option_counts[v], latest[v])),
        sorted(range(V), key=lambda v: (latest[v], option_counts[v])),
        sorted(range(V), key=lambda v: (-max_duration[v], option_counts[v])),
    ]

    rng = random.Random(0)
    best = None
    attempts = 0

    while attempts < 30 and time.monotonic() < deadline:
        if attempts < len(base_orders):
            order = list(base_orders[attempts])
        else:
            order = list(range(V))
            rng.shuffle(order)
            order.sort(
                key=lambda v: option_counts[v] * (0.8 + 0.4 * rng.random())
            )

        selected = [-1] * V
        berth_intervals = [[] for _ in range(B)]
        slot_usage = {t: 0 for t in range(1, H + 1)}
        failed = False

        for v in order:
            feasible_candidates = []
            for oi, option in enumerate(options[v]):
                b, _, alpha, beta, _, qc, penalty, unload_cost, load_est = option

                intervals = berth_intervals[b]
                if intervals:
                    new_min = min(alpha, min(a for a, _ in intervals))
                    new_max = max(beta, max(e for _, e in intervals))
                    if new_max - new_min > H - 1:
                        continue

                overlap = any(
                    not (beta < other_alpha or other_beta < alpha)
                    for other_alpha, other_beta in intervals
                )
                if overlap:
                    continue

                added = {}
                qc_ok = True
                for offset, q in enumerate(qc):
                    slot = cyclic_slot(alpha + offset, H)
                    added[slot] = added.get(slot, 0) + q
                for slot, amount in added.items():
                    if (
                        slot not in capacity
                        or slot_usage.get(slot, 0) + amount > capacity[slot]
                    ):
                        qc_ok = False
                        break
                if not qc_ok:
                    continue

                congestion = sum(
                    (slot_usage.get(slot, 0) + amount)
                    / max(1, capacity[slot])
                    for slot, amount in added.items()
                )
                score = penalty + unload_cost + load_est + 0.01 * congestion
                feasible_candidates.append((score, oi, added))

            if not feasible_candidates:
                failed = True
                break

            feasible_candidates.sort(key=lambda x: (x[0], x[1]))
            if attempts >= len(base_orders) and len(feasible_candidates) > 1:
                top = min(3, len(feasible_candidates))
                chosen = feasible_candidates[rng.randrange(top)]
            else:
                chosen = feasible_candidates[0]

            _, oi, added = chosen
            selected[v] = oi
            option = options[v][oi]
            b, _, alpha, beta = option[:4]
            berth_intervals[b].append((alpha, beta))
            for slot, amount in added.items():
                slot_usage[slot] = slot_usage.get(slot, 0) + amount

        if not failed:
            subblocks = greedy_subblock_assignment(instance, prepared, selected)
            if subblocks is not None:
                solution = build_solution(
                    instance, prepared, selected, subblocks
                )
                if (
                    best is None
                    or solution["objective_value"]
                    < best["solution"]["objective_value"] - 1e-8
                ):
                    best = {
                        "selected": selected,
                        "subblocks": subblocks,
                        "solution": solution,
                    }

        attempts += 1

    return best


def main():
    args = parse_args()
    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path
        else None
    )

    instance = load_instance(args.instance_path)
    prepared = prepare_data(instance)

    V = prepared["V"]
    B = prepared["B"]
    S = prepared["S"]
    H = prepared["H"]
    E = prepared["E"]
    T = prepared["T"]
    options = prepared["options"]
    vessels = instance["vessels"]

    if any(len(options[v]) == 0 for v in range(V)):
        raise RuntimeError("Instance is infeasible: a vessel has no valid option.")

    heuristic_deadline = min(
        deadline,
        time.monotonic() + min(2.0, max(0.05, args.time_limit * 0.08)),
    )
    heuristic = find_greedy_solution(
        instance, prepared, heuristic_deadline
    )

    incumbent_solution = None
    best_logged_objective = math.inf
    if heuristic is not None:
        incumbent_solution = heuristic["solution"]
        best_logged_objective = incumbent_solution["objective_value"]
        if logger:
            logger.log_solution(
                incumbent_solution["objective_value"], incumbent_solution
            )

    model = gp.Model("cyclic_terminal_planning")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.Heuristics = 0.2

    x = []
    flat_x_vars = []
    flat_x_meta = []
    for v in range(V):
        xv = []
        for oi in range(len(options[v])):
            var = model.addVar(vtype=GRB.BINARY, name=f"x_{v}_{oi}")
            xv.append(var)
            flat_x_vars.append(var)
            flat_x_meta.append((v, oi))
        x.append(xv)

    z = [
        [
            model.addVar(vtype=GRB.BINARY, name=f"z_{v}_{b}")
            for b in range(B)
        ]
        for v in range(V)
    ]

    alpha_var = [
        model.addVar(lb=1, ub=H, vtype=GRB.CONTINUOUS, name=f"alpha_{v}")
        for v in range(V)
    ]
    beta_var = [
        model.addVar(lb=1, ub=T, vtype=GRB.CONTINUOUS, name=f"beta_{v}")
        for v in range(V)
    ]

    rho = [
        model.addVar(lb=0, ub=T, vtype=GRB.CONTINUOUS, name=f"rho_{b}")
        for b in range(B)
    ]
    varsigma = [
        model.addVar(
            lb=0, ub=T, vtype=GRB.CONTINUOUS, name=f"varsigma_{b}"
        )
        for b in range(B)
    ]

    y = {}
    flat_y_vars = []
    flat_y_meta = []
    for v in range(V):
        if int(vessels[v]["reserved_subblocks"]) <= 0:
            continue
        for s in range(S):
            for b in range(B):
                var = model.addVar(
                    vtype=GRB.BINARY, name=f"y_{v}_{s}_{b}"
                )
                y[v, s, b] = var
                flat_y_vars.append(var)
                flat_y_meta.append((v, s, b))

    model.update()

    for v in range(V):
        model.addConstr(gp.quicksum(x[v]) == 1, name=f"choose_{v}")
        for b in range(B):
            model.addConstr(
                gp.quicksum(
                    x[v][oi]
                    for oi, option in enumerate(options[v])
                    if option[0] == b
                )
                == z[v][b],
                name=f"berth_link_{v}_{b}",
            )

        model.addConstr(
            alpha_var[v]
            == gp.quicksum(
                option[2] * x[v][oi]
                for oi, option in enumerate(options[v])
            ),
            name=f"alpha_def_{v}",
        )
        model.addConstr(
            beta_var[v]
            == gp.quicksum(
                option[3] * x[v][oi]
                for oi, option in enumerate(options[v])
            ),
            name=f"beta_def_{v}",
        )

    occupancy_terms = {
        (b, t): [] for b in range(B) for t in range(1, T + 1)
    }
    qc_terms = {t: [] for t in range(1, T + 1)}

    for v in range(V):
        for oi, option in enumerate(options[v]):
            b, _, alpha, beta, _, qc = option[:6]
            var = x[v][oi]
            for t in range(alpha, beta + 1):
                occupancy_terms[b, t].append(var)
            for offset, q in enumerate(qc):
                if q:
                    qc_terms[alpha + offset].append(q * var)

    for b in range(B):
        for t in range(1, T + 1):
            terms = occupancy_terms[b, t]
            if terms:
                model.addConstr(
                    gp.quicksum(terms) <= 1,
                    name=f"berth_occ_{b}_{t}",
                )

    q_expr = {}
    for t in range(1, T + 1):
        q_expr[t] = (
            gp.quicksum(qc_terms[t]) if qc_terms[t] else gp.LinExpr(0.0)
        )

    capacity = prepared["capacity"]
    for t in range(E + 1, H + 1):
        model.addConstr(
            q_expr[t] <= capacity[t], name=f"qc_capacity_{t}"
        )
    for t in range(1, E + 1):
        twin = t + H
        if twin <= T:
            model.addConstr(
                q_expr[t] + q_expr[twin] <= capacity[t],
                name=f"qc_periodic_{t}",
            )

    big_m = max(1, 2 * T + 1)
    for b in range(B):
        for v in range(V):
            model.addConstr(
                rho[b] <= alpha_var[v] + big_m * (1 - z[v][b]),
                name=f"rho_link_{b}_{v}",
            )
            model.addConstr(
                varsigma[b] >= beta_var[v] - big_m * (1 - z[v][b]),
                name=f"varsigma_link_{b}_{v}",
            )
        model.addConstr(
            varsigma[b] - rho[b] <= H - 1,
            name=f"cyclic_span_{b}",
        )

    for s in range(S):
        terms = [
            y[v, s, b]
            for v in range(V)
            if int(vessels[v]["reserved_subblocks"]) > 0
            for b in range(B)
        ]
        if terms:
            model.addConstr(
                gp.quicksum(terms) <= 1, name=f"subblock_unique_{s}"
            )

    for v in range(V):
        r = int(vessels[v]["reserved_subblocks"])
        if r <= 0:
            continue
        for b in range(B):
            model.addConstr(
                gp.quicksum(y[v, s, b] for s in range(S))
                == r * z[v][b],
                name=f"subblock_count_{v}_{b}",
            )

    objective_terms = []
    for v in range(V):
        for oi, option in enumerate(options[v]):
            option_cost = option[6] + option[7]
            if option_cost:
                objective_terms.append(option_cost * x[v][oi])

    co = float(instance["penalty_cost_coefficient_co"])
    for (v, s, b), var in y.items():
        r = int(vessels[v]["reserved_subblocks"])
        load = int(vessels[v]["loading_containers"])
        coefficient = (
            co
            * float(instance["loading_route_lengths_D_L"][s][b])
            * load
            / r
        )
        if coefficient:
            objective_terms.append(coefficient * var)

    model.setObjective(gp.quicksum(objective_terms), GRB.MINIMIZE)

    if heuristic is not None:
        selected = heuristic["selected"]
        subblock_sets = [
            set(items) for items in heuristic["subblocks"]
        ]

        for v in range(V):
            for oi, var in enumerate(x[v]):
                var.Start = 1.0 if oi == selected[v] else 0.0
            selected_berth = options[v][selected[v]][0]
            for b in range(B):
                z[v][b].Start = 1.0 if b == selected_berth else 0.0

        for (v, s, b), var in y.items():
            selected_berth = options[v][selected[v]][0]
            var.Start = (
                1.0
                if b == selected_berth and s in subblock_sets[v]
                else 0.0
            )

    model.update()

    callback_state = {
        "best": best_logged_objective,
        "solution": incumbent_solution,
    }

    def extract_callback_solution(cb_model):
        x_values = cb_model.cbGetSolution(flat_x_vars)
        y_values = (
            cb_model.cbGetSolution(flat_y_vars) if flat_y_vars else []
        )

        selected = [-1] * V
        selected_value = [-math.inf] * V
        for value, (v, oi) in zip(x_values, flat_x_meta):
            if value > selected_value[v]:
                selected_value[v] = value
                selected[v] = oi

        if any(oi < 0 for oi in selected):
            return None

        subblocks = [[] for _ in range(V)]
        for value, (v, s, b) in zip(y_values, flat_y_meta):
            if value > 0.5:
                selected_berth = options[v][selected[v]][0]
                if b == selected_berth:
                    subblocks[v].append(s)

        for v in range(V):
            required = int(vessels[v]["reserved_subblocks"])
            if len(subblocks[v]) != required:
                return None

        return build_solution(instance, prepared, selected, subblocks)

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            objective = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
            if objective >= callback_state["best"] - 1e-7:
                return

            solution = extract_callback_solution(cb_model)
            if solution is None:
                return

            actual_objective = solution["objective_value"]
            if actual_objective < callback_state["best"] - 1e-7:
                callback_state["best"] = actual_objective
                callback_state["solution"] = solution
                if logger:
                    logger.log_solution(actual_objective, solution)
        except Exception:
            # Logging or snapshot extraction must never interrupt optimization.
            return

    remaining = deadline - time.monotonic()
    if remaining > 0:
        model.Params.TimeLimit = max(0.01, remaining)
        model.optimize(incumbent_callback)

    if model.SolCount > 0:
        selected = []
        for v in range(V):
            oi = max(range(len(x[v])), key=lambda j: x[v][j].X)
            selected.append(oi)

        subblocks = [[] for _ in range(V)]
        for (v, s, b), var in y.items():
            if var.X > 0.5 and b == options[v][selected[v]][0]:
                subblocks[v].append(s)

        final_solution = build_solution(
            instance, prepared, selected, subblocks
        )
        if (
            callback_state["solution"] is None
            or final_solution["objective_value"]
            <= callback_state["solution"]["objective_value"] + 1e-6
        ):
            callback_state["solution"] = final_solution

        if (
            logger
            and final_solution["objective_value"]
            < callback_state["best"] - 1e-7
        ):
            logger.log_solution(
                final_solution["objective_value"], final_solution
            )
            callback_state["best"] = final_solution["objective_value"]

    if callback_state["solution"] is None:
        raise RuntimeError(
            "No feasible solution was found within the time limit."
        )

    atomic_write_json(args.solution_path, callback_state["solution"])


if __name__ == "__main__":
    main()