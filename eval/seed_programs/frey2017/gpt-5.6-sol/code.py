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


def write_json(path, data):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(temp_path, path)


def get_type_bounds(flight, type_id):
    bounds = flight.get("working_station_bounds_by_carousel_type", {})
    entry = bounds.get(str(type_id), bounds.get(type_id))
    if entry is None:
        return None
    return int(entry["Wmin"]), int(entry["Wmax"])


def create_options(instance):
    """
    Create assignment columns. Each column fixes:
      carousel, working-station count, handling start, depletion start.

    Period convention:
      - the arrival profile ends at end_period;
      - arrivals before handling starts enter central storage;
      - arrivals from the handling start onward go directly to the carousel;
      - stored bags are depleted at the configured constant rate;
      - workload is the number of bags on the belt immediately before station
        processing in that period.
    """
    horizon = int(instance["planning_horizon"]["num_periods"])
    storage_rate = int(instance["central_storage"]["depletion_rate_bags_per_period"])
    depletion_offset = int(instance["central_storage"]["depletion_offset_periods"])
    working_rate = int(instance["working_rate_bags_per_period"])

    carousels = instance["carousels"]
    flights = instance["flights"]

    options = []
    options_by_flight = [[] for _ in flights]

    for fi, flight in enumerate(flights):
        end = min(int(flight["end_period"]), horizon - 1)
        earliest = max(0, int(flight["earliest_start_period"]))
        latest = min(int(flight["latest_start_period"]), end)
        profile = [int(v) for v in flight["arrival_profile"]]

        # The profile describes successive periods ending at the fixed handling
        # end. This also handles profiles beginning before the planning horizon.
        profile_origin = int(flight["end_period"]) - len(profile) + 1
        arrivals = defaultdict(int)
        initial_early_bags = 0
        for k, amount in enumerate(profile):
            period = profile_origin + k
            if period < 0:
                initial_early_bags += amount
            elif period < horizon:
                arrivals[period] += amount

        # If expected_baggage exceeds the represented profile, conservatively
        # regard the difference as baggage already in storage.
        represented = sum(profile)
        initial_early_bags += max(0, int(flight["expected_baggage"]) - represented)

        for carousel_index, carousel in enumerate(carousels):
            type_id = int(carousel["type_r"])
            ws_bounds = get_type_bounds(flight, type_id)
            if ws_bounds is None:
                continue

            wmin, wmax = ws_bounds
            wmin = max(1, wmin)
            wmax = min(wmax, int(carousel["Kws"]))
            if wmin > wmax:
                continue

            for stations in range(wmin, wmax + 1):
                processing_rate = stations * working_rate

                for handling_start in range(earliest, latest + 1):
                    latest_depletion_start = end - depletion_offset
                    if latest_depletion_start < handling_start:
                        continue

                    # No pre-start baggage means every depletion choice is
                    # operationally equivalent; retain one representative.
                    pre_start_total = initial_early_bags + sum(
                        amount for t, amount in arrivals.items()
                        if t < handling_start
                    )
                    if pre_start_total <= 0:
                        depletion_starts = [handling_start]
                    else:
                        depletion_starts = range(
                            handling_start, latest_depletion_start + 1
                        )

                    seen_signatures = set()
                    for depletion_start in depletion_starts:
                        central_stock = initial_early_bags
                        belt_queue = 0
                        storage = {}
                        workload = {}

                        for t in range(0, end + 1):
                            amount = arrivals.get(t, 0)
                            direct_inflow = 0

                            if t < handling_start:
                                central_stock += amount
                            else:
                                direct_inflow += amount

                            if t >= depletion_start and central_stock > 0:
                                moved = min(storage_rate, central_stock)
                                central_stock -= moved
                                direct_inflow += moved

                            if t >= handling_start:
                                belt_before_processing = belt_queue + direct_inflow
                                if belt_before_processing:
                                    workload[t] = belt_before_processing
                                belt_queue = max(
                                    0, belt_before_processing - processing_rate
                                )

                            if central_stock:
                                storage[t] = central_stock

                        # All represented baggage must have left storage and
                        # have been processed by the fixed end period.
                        if central_stock > 0 or belt_queue > 0:
                            continue

                        signature = (
                            tuple(sorted(storage.items())),
                            tuple(sorted(workload.items())),
                        )
                        if signature in seen_signatures:
                            continue
                        seen_signatures.add(signature)

                        option = {
                            "flight_index": fi,
                            "flight_id": int(flight["flight_id"]),
                            "carousel_index": carousel_index,
                            "carousel_id": int(carousel["carousel_id"]),
                            "carousel_type": type_id,
                            "working_stations": stations,
                            "handling_start": handling_start,
                            "depletion_start": depletion_start,
                            "end": end,
                            "containers": int(flight["num_containers"]),
                            "storage": storage,
                            "workload": workload,
                        }
                        oi = len(options)
                        options.append(option)
                        options_by_flight[fi].append(oi)

    return options, options_by_flight


def choose_threshold(excess, thresholds, penalties):
    if excess <= EPS:
        return None
    candidates = [
        i for i, threshold in enumerate(thresholds)
        if threshold + EPS >= excess
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda i: (penalties[i], thresholds[i], i))


def build_solution(instance, options, selected_option_indices,
                   violation_levels=None):
    thresholds = [float(x) for x in instance["objective"]["threshold_values"]]
    penalties = [int(x) for x in instance["objective"]["penalties"]]
    target = float(instance["objective"]["target_utilization"])
    carousels = instance["carousels"]
    peaks = instance["peak_intervals"]

    assignments = []
    loads = defaultdict(float)

    selected = sorted(
        (options[oi] for oi in selected_option_indices),
        key=lambda o: o["flight_id"]
    )

    for option in selected:
        assignments.append({
            "flight_id": int(option["flight_id"]),
            "carousel_id": int(option["carousel_id"]),
            "carousel_type": int(option["carousel_type"]),
            "working_stations": int(option["working_stations"]),
            "handling_start_period": int(option["handling_start"]),
            "depletion_start_period": int(option["depletion_start"]),
            "end_period": int(option["end"]),
        })
        ci = option["carousel_index"]
        for t, value in option["workload"].items():
            loads[(ci, t)] += value

    utilization_violations = []
    objective = 0.0

    for ci, carousel in enumerate(carousels):
        belt_capacity = float(carousel["Kcb"])
        for pi, peak in enumerate(peaks):
            level = None
            if violation_levels is not None:
                level = violation_levels.get((ci, pi))
            else:
                peak_max = 0.0
                start = int(peak["start_period"])
                end = int(peak["end_period"])
                for t in range(start, end + 1):
                    peak_max = max(peak_max, loads.get((ci, t), 0.0))
                excess = peak_max / belt_capacity - target
                level = choose_threshold(excess, thresholds, penalties)

            if level is not None:
                objective += penalties[level]
                utilization_violations.append({
                    "carousel_id": int(carousel["carousel_id"]),
                    "peak_interval": int(peak["peak_id"]),
                    "threshold_level": int(level),
                    "threshold_value": float(thresholds[level]),
                })

    return {
        "objective_value": float(objective),
        "assignments": assignments,
        "utilization_violations": utilization_violations,
    }


def greedy_incumbent(instance, options, options_by_flight):
    horizon = int(instance["planning_horizon"]["num_periods"])
    storage_capacity = int(instance["central_storage"]["capacity_bags"])
    target = float(instance["objective"]["target_utilization"])
    thresholds = [float(x) for x in instance["objective"]["threshold_values"]]
    penalties = [int(x) for x in instance["objective"]["penalties"]]
    max_allowance = max(thresholds) if thresholds else 0.0

    carousels = instance["carousels"]
    peaks = instance["peak_intervals"]

    station_use = [[0] * horizon for _ in carousels]
    parking_use = [[0] * horizon for _ in carousels]
    belt_load = [[0.0] * horizon for _ in carousels]
    storage_use = [0] * horizon
    selected = []

    period_to_peak = {}
    for pi, peak in enumerate(peaks):
        for t in range(
            max(0, int(peak["start_period"])),
            min(horizon - 1, int(peak["end_period"])) + 1
        ):
            period_to_peak[t] = pi

    order = sorted(
        range(len(options_by_flight)),
        key=lambda fi: (len(options_by_flight[fi]), fi)
    )

    for fi in order:
        if not options_by_flight[fi]:
            return None

        best = None
        best_score = None

        for oi in options_by_flight[fi]:
            option = options[oi]
            ci = option["carousel_index"]
            carousel = carousels[ci]
            feasible = True

            for t in range(option["handling_start"], option["end"] + 1):
                if t < 0 or t >= horizon:
                    continue
                if (station_use[ci][t] + option["working_stations"]
                        > int(carousel["Kws"])):
                    feasible = False
                    break
                if (parking_use[ci][t] + option["containers"]
                        > int(carousel["Kpp"])):
                    feasible = False
                    break
            if not feasible:
                continue

            for t, amount in option["storage"].items():
                if 0 <= t < horizon and storage_use[t] + amount > storage_capacity:
                    feasible = False
                    break
            if not feasible:
                continue

            for t, amount in option["workload"].items():
                if t in period_to_peak:
                    utilization = (
                        belt_load[ci][t] + amount
                    ) / float(carousel["Kcb"])
                    if utilization > target + max_allowance + EPS:
                        feasible = False
                        break
            if not feasible:
                continue

            # Estimate the resulting threshold objective for the affected
            # carousel and use a smooth load-balancing term as a tie-breaker.
            penalty_score = 0.0
            smooth_score = 0.0
            for pi, peak in enumerate(peaks):
                maximum = 0.0
                for t in range(
                    max(0, int(peak["start_period"])),
                    min(horizon - 1, int(peak["end_period"])) + 1
                ):
                    added = option["workload"].get(t, 0)
                    utilization = (
                        belt_load[ci][t] + added
                    ) / float(carousel["Kcb"])
                    maximum = max(maximum, utilization)
                    smooth_score += utilization * utilization

                level = choose_threshold(
                    maximum - target, thresholds, penalties
                )
                if maximum > target + EPS and level is None:
                    feasible = False
                    break
                if level is not None:
                    penalty_score += penalties[level]

            if not feasible:
                continue

            score = (
                penalty_score,
                smooth_score,
                option["working_stations"],
                option["handling_start"],
                option["carousel_id"],
                option["depletion_start"],
            )
            if best_score is None or score < best_score:
                best_score = score
                best = oi

        if best is None:
            return None

        selected.append(best)
        option = options[best]
        ci = option["carousel_index"]
        for t in range(option["handling_start"], option["end"] + 1):
            if 0 <= t < horizon:
                station_use[ci][t] += option["working_stations"]
                parking_use[ci][t] += option["containers"]
        for t, amount in option["storage"].items():
            if 0 <= t < horizon:
                storage_use[t] += amount
        for t, amount in option["workload"].items():
            if 0 <= t < horizon:
                belt_load[ci][t] += amount

    return selected


def main():
    args = parse_args()
    logger = SolutionLogger(
        args.log_path, sense="minimize"
    ) if args.log_path else None

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    options, options_by_flight = create_options(instance)

    missing = [
        int(instance["flights"][fi]["flight_id"])
        for fi, indices in enumerate(options_by_flight)
        if not indices
    ]
    if missing:
        raise RuntimeError(
            "No feasible assignment columns could be generated for flights: "
            + ", ".join(map(str, missing))
        )

    greedy_selected = greedy_incumbent(
        instance, options, options_by_flight
    )
    greedy_solution = None
    if greedy_selected is not None:
        greedy_solution = build_solution(
            instance, options, greedy_selected
        )
        if logger:
            logger.log_solution(
                greedy_solution["objective_value"], greedy_solution
            )

    horizon = int(instance["planning_horizon"]["num_periods"])
    storage_capacity = int(instance["central_storage"]["capacity_bags"])
    target = float(instance["objective"]["target_utilization"])
    thresholds = [float(x) for x in instance["objective"]["threshold_values"]]
    penalties = [int(x) for x in instance["objective"]["penalties"]]
    carousels = instance["carousels"]
    peaks = instance["peak_intervals"]

    model = gp.Model("airport_baggage_scheduling")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = max(0.01, deadline - time.monotonic())

    x = [
        model.addVar(vtype=GRB.BINARY, name=f"x_{oi}")
        for oi in range(len(options))
    ]

    y = {}
    for ci in range(len(carousels)):
        for pi in range(len(peaks)):
            for li in range(len(thresholds)):
                y[(ci, pi, li)] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"y_{ci}_{pi}_{li}"
                )

    model.update()

    # Exactly one combined assignment per flight.
    for fi, indices in enumerate(options_by_flight):
        model.addConstr(
            gp.quicksum(x[oi] for oi in indices) == 1,
            name=f"assign_{fi}"
        )

    station_terms = defaultdict(list)
    parking_terms = defaultdict(list)
    storage_terms = defaultdict(list)
    workload_terms = defaultdict(list)

    for oi, option in enumerate(options):
        ci = option["carousel_index"]

        for t in range(option["handling_start"], option["end"] + 1):
            if 0 <= t < horizon:
                station_terms[(ci, t)].append(
                    option["working_stations"] * x[oi]
                )
                parking_terms[(ci, t)].append(
                    option["containers"] * x[oi]
                )

        for t, amount in option["storage"].items():
            if amount and 0 <= t < horizon:
                storage_terms[t].append(amount * x[oi])

        for t, amount in option["workload"].items():
            if amount and 0 <= t < horizon:
                workload_terms[(ci, t)].append(amount * x[oi])

    for ci, carousel in enumerate(carousels):
        for t in range(horizon):
            terms = station_terms.get((ci, t))
            if terms:
                model.addConstr(
                    gp.quicksum(terms) <= int(carousel["Kws"]),
                    name=f"stations_{ci}_{t}"
                )
            terms = parking_terms.get((ci, t))
            if terms:
                model.addConstr(
                    gp.quicksum(terms) <= int(carousel["Kpp"]),
                    name=f"parking_{ci}_{t}"
                )

    for t in range(horizon):
        terms = storage_terms.get(t)
        if terms:
            model.addConstr(
                gp.quicksum(terms) <= storage_capacity,
                name=f"storage_{t}"
            )

    # At most one utilization threshold can be selected for each
    # carousel/peak pair.
    for ci in range(len(carousels)):
        for pi in range(len(peaks)):
            model.addConstr(
                gp.quicksum(
                    y[(ci, pi, li)] for li in range(len(thresholds))
                ) <= 1,
                name=f"one_threshold_{ci}_{pi}"
            )

    # Workload/Kcb <= target + selected threshold.
    for ci, carousel in enumerate(carousels):
        belt_capacity = float(carousel["Kcb"])
        for pi, peak in enumerate(peaks):
            threshold_expr = gp.quicksum(
                belt_capacity * thresholds[li] * y[(ci, pi, li)]
                for li in range(len(thresholds))
            )
            start = max(0, int(peak["start_period"]))
            end = min(horizon - 1, int(peak["end_period"]))
            for t in range(start, end + 1):
                load_expr = gp.quicksum(
                    workload_terms.get((ci, t), [])
                )
                model.addConstr(
                    load_expr - threshold_expr
                    <= target * belt_capacity,
                    name=f"utilization_{ci}_{pi}_{t}"
                )

    model.setObjective(
        gp.quicksum(
            penalties[li] * y[(ci, pi, li)]
            for ci in range(len(carousels))
            for pi in range(len(peaks))
            for li in range(len(thresholds))
        ),
        GRB.MINIMIZE
    )

    # Seed Gurobi with the complete greedy solution.
    if greedy_selected is not None:
        greedy_set = set(greedy_selected)
        for oi, var in enumerate(x):
            var.Start = 1.0 if oi in greedy_set else 0.0

        greedy_levels = {}
        for violation in greedy_solution["utilization_violations"]:
            carousel_id = int(violation["carousel_id"])
            peak_id = int(violation["peak_interval"])
            level = int(violation["threshold_level"])
            ci = next(
                i for i, c in enumerate(carousels)
                if int(c["carousel_id"]) == carousel_id
            )
            pi = next(
                i for i, p in enumerate(peaks)
                if int(p["peak_id"]) == peak_id
            )
            greedy_levels[(ci, pi)] = level

        for key, var in y.items():
            ci, pi, li = key
            var.Start = (
                1.0 if greedy_levels.get((ci, pi)) == li else 0.0
            )

    best_logged_objective = (
        greedy_solution["objective_value"]
        if greedy_solution is not None else math.inf
    )

    def incumbent_callback(cb_model, where):
        nonlocal best_logged_objective
        if where != GRB.Callback.MIPSOL:
            return
        try:
            selected_indices = [
                oi for oi, var in enumerate(x)
                if cb_model.cbGetSolution(var) > 0.5
            ]
            if len(selected_indices) != len(instance["flights"]):
                return

            levels = {}
            for (ci, pi, li), var in y.items():
                if cb_model.cbGetSolution(var) > 0.5:
                    levels[(ci, pi)] = li

            solution = build_solution(
                instance, options, selected_indices, levels
            )
            objective = solution["objective_value"]
            if objective + EPS < best_logged_objective:
                best_logged_objective = objective
                if logger:
                    logger.log_solution(objective, solution)
        except Exception:
            # Logging must never interrupt the optimization.
            pass

    model.optimize(incumbent_callback)

    if model.SolCount > 0:
        selected_indices = [
            oi for oi, var in enumerate(x) if var.X > 0.5
        ]
        levels = {}
        for (ci, pi, li), var in y.items():
            if var.X > 0.5:
                levels[(ci, pi)] = li

        final_solution = build_solution(
            instance, options, selected_indices, levels
        )
    elif greedy_solution is not None:
        final_solution = greedy_solution
    else:
        raise RuntimeError(
            "No feasible solution was found within the time limit."
        )

    if (logger and
            final_solution["objective_value"] + EPS < best_logged_objective):
        logger.log_solution(
            final_solution["objective_value"], final_solution
        )

    write_json(args.solution_path, final_solution)


if __name__ == "__main__":
    main()