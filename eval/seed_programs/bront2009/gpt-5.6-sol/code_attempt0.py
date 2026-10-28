import argparse
import heapq
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


def write_json(path, data):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)


def solve(instance, time_limit, logger):
    start_time = time.monotonic()
    deadline = start_time + max(0, time_limit)

    products = instance.get("products", [])
    segments_data = instance.get("segments", [])
    legs_data = instance.get("network", {}).get("legs", [])

    n = len(products)
    m = len(legs_data)
    global_lambda = float(instance.get("lambda", 0.0))
    horizon = max(0.0, float(instance.get("booking_horizon", {}).get("T", 0)))
    alpha = float(instance.get("alpha", 1.0))

    zero_solution = {
        "objective_value": 0.0,
        "active_columns": []
    }

    if logger:
        logger.log_solution(0.0, zero_solution)

    if n == 0 or horizon <= 0.0 or global_lambda <= 0.0 or time_limit <= 0:
        return zero_solution

    product_ids = [p["product_id"] for p in products]
    product_index = {pid: j for j, pid in enumerate(product_ids)}
    fares = [float(p.get("fare", 0.0)) for p in products]

    leg_ids = [leg["leg_id"] for leg in legs_data]
    leg_index = {leg_id: i for i, leg_id in enumerate(leg_ids)}
    capacities = [
        max(0.0, alpha * float(leg.get("capacity", 0.0)))
        for leg in legs_data
    ]

    product_legs = []
    for product in products:
        used = []
        seen = set()
        for leg_id in product.get("legs_used", []):
            if leg_id in leg_index:
                i = leg_index[leg_id]
                if i not in seen:
                    seen.add(i)
                    used.append(i)
        product_legs.append(used)

    # Each segment is represented by:
    # (probability share, no-purchase weight, [(product_index, preference), ...])
    segments = []
    memberships = [[] for _ in range(n)]
    for raw_segment in segments_data:
        lambda_l = float(raw_segment.get("lambda_l", 0.0))
        share = lambda_l / global_lambda if global_lambda > 0.0 else 0.0
        if share <= 0.0:
            continue

        no_purchase = float(raw_segment.get("no_purchase_preference", 0.0))
        no_purchase = max(no_purchase, 1e-12)

        consideration = raw_segment.get("consideration_set", [])
        preferences = raw_segment.get("preference_vector", [])

        # Combine repeated product identifiers defensively.
        combined = {}
        for pid, pref in zip(consideration, preferences):
            if pid not in product_index:
                continue
            v = float(pref)
            if v > 0.0:
                j = product_index[pid]
                combined[j] = combined.get(j, 0.0) + v

        entries = sorted(combined.items())
        seg_idx = len(segments)
        segments.append((share, no_purchase, entries))
        for j, v in entries:
            memberships[j].append((seg_idx, v))

    if not segments:
        return zero_solution

    def mask_to_offer(mask):
        return [product_ids[j] for j in range(n) if (mask >> j) & 1]

    column_cache = {}

    def compute_column(mask):
        """Return (expected revenue per arrival, expected leg consumptions)."""
        cached = column_cache.get(mask)
        if cached is not None:
            return cached

        probabilities = [0.0] * n
        for share, no_purchase, entries in segments:
            denominator = no_purchase
            selected = []
            for j, v in entries:
                if (mask >> j) & 1:
                    denominator += v
                    selected.append((j, v))
            if not selected:
                continue
            factor = share / denominator
            for j, v in selected:
                probabilities[j] += factor * v

        revenue = 0.0
        leg_consumption = [0.0] * m
        for j, probability in enumerate(probabilities):
            if probability == 0.0:
                continue
            revenue += fares[j] * probability
            for i in product_legs[j]:
                leg_consumption[i] += probability

        result = (revenue, leg_consumption)
        column_cache[mask] = result
        return result

    def assortment_value(mask, adjusted_revenues):
        value = 0.0
        for share, no_purchase, entries in segments:
            denominator = no_purchase
            numerator = 0.0
            for j, v in entries:
                if (mask >> j) & 1:
                    denominator += v
                    numerator += v * adjusted_revenues[j]
            value += share * numerator / denominator
        return value

    def local_search(adjusted_revenues, starts, max_results=40):
        """Best-toggle local search for the mixed-MNL pricing problem."""
        results = {}
        max_steps = max(10, min(4 * max(n, 1), 400))

        for initial_mask in starts:
            if time.monotonic() >= deadline:
                break

            mask = initial_mask
            denominators = [segment[1] for segment in segments]
            numerators = [0.0] * len(segments)

            selected_bits = mask
            while selected_bits:
                lowbit = selected_bits & -selected_bits
                j = lowbit.bit_length() - 1
                selected_bits ^= lowbit
                rj = adjusted_revenues[j]
                for seg_idx, v in memberships[j]:
                    denominators[seg_idx] += v
                    numerators[seg_idx] += v * rj

            current_value = sum(
                segments[s][0] * numerators[s] / denominators[s]
                for s in range(len(segments))
            )

            for _ in range(max_steps):
                if time.monotonic() >= deadline:
                    break

                best_delta = 1e-11
                best_j = -1

                for j in range(n):
                    selected = ((mask >> j) & 1) != 0
                    sign = -1.0 if selected else 1.0
                    delta = 0.0
                    rj = adjusted_revenues[j]

                    for seg_idx, v in memberships[j]:
                        old_d = denominators[seg_idx]
                        old_n = numerators[seg_idx]
                        new_d = old_d + sign * v
                        new_n = old_n + sign * v * rj
                        share = segments[seg_idx][0]
                        delta += share * (
                            new_n / new_d - old_n / old_d
                        )

                    if delta > best_delta:
                        best_delta = delta
                        best_j = j

                if best_j < 0:
                    break

                if (mask >> best_j) & 1:
                    sign = -1.0
                    mask &= ~(1 << best_j)
                else:
                    sign = 1.0
                    mask |= 1 << best_j

                rj = adjusted_revenues[best_j]
                for seg_idx, v in memberships[best_j]:
                    denominators[seg_idx] += sign * v
                    numerators[seg_idx] += sign * v * rj
                current_value += best_delta

            if mask:
                previous = results.get(mask)
                if previous is None or current_value > previous:
                    results[mask] = current_value

        ranked = sorted(
            results.items(),
            key=lambda item: item[1],
            reverse=True
        )
        return ranked[:max_results]

    rng = random.Random(0)

    def make_search_starts(adjusted_revenues, extra_masks=None):
        starts = {0}
        if extra_masks:
            starts.update(extra_masks)

        positive_mask = 0
        for j, value in enumerate(adjusted_revenues):
            if value > 0.0 and memberships[j]:
                positive_mask |= 1 << j
        starts.add(positive_mask)

        order = sorted(
            range(n),
            key=lambda j: adjusted_revenues[j],
            reverse=True
        )
        running = 0
        if order:
            prefix_points = set()
            for k in range(1, min(n, 20) + 1):
                prefix_points.add(k)
            if n > 20:
                for q in range(1, 11):
                    prefix_points.add(max(1, int(round(q * n / 10.0))))

            for k, j in enumerate(order, 1):
                if adjusted_revenues[j] > 0.0:
                    running |= 1 << j
                if k in prefix_points:
                    starts.add(running)

            for j in order[:min(10, n)]:
                starts.add(1 << j)

        for _, _, entries in segments:
            mask = 0
            for j, _ in entries:
                if adjusted_revenues[j] > 0.0:
                    mask |= 1 << j
            starts.add(mask)

        random_start_count = 12 if n <= 200 else 5
        for _ in range(random_start_count):
            mask = 0
            inclusion_probability = rng.choice((0.15, 0.3, 0.5, 0.7))
            for j in range(n):
                if adjusted_revenues[j] > 0.0 and rng.random() < inclusion_probability:
                    mask |= 1 << j
            starts.add(mask)

        return list(starts)

    # Initial assortment pool.
    initial_masks = set()

    if n <= 500:
        for j in range(n):
            if memberships[j]:
                initial_masks.add(1 << j)
    else:
        top_products = sorted(
            range(n), key=lambda j: fares[j], reverse=True
        )[:500]
        for j in top_products:
            if memberships[j]:
                initial_masks.add(1 << j)

    all_relevant = 0
    for j in range(n):
        if memberships[j]:
            all_relevant |= 1 << j
    if all_relevant:
        initial_masks.add(all_relevant)

    for _, _, entries in segments:
        mask = 0
        for j, _ in entries:
            mask |= 1 << j
        if mask:
            initial_masks.add(mask)

    fare_order = sorted(range(n), key=lambda j: fares[j], reverse=True)
    running = 0
    prefix_limit = min(n, 100)
    for k, j in enumerate(fare_order[:prefix_limit], 1):
        if memberships[j]:
            running |= 1 << j
        if running and (k <= 20 or k % 5 == 0):
            initial_masks.add(running)

    if time.monotonic() < deadline:
        starts = make_search_starts(fares, initial_masks)
        for mask, _ in local_search(fares, starts, max_results=60):
            initial_masks.add(mask)

    model = gp.Model("choice_based_deterministic_lp")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.ModelSense = GRB.MAXIMIZE

    capacity_constraints = [
        model.addConstr(gp.LinExpr() <= capacities[i], name=f"capacity_{i}")
        for i in range(m)
    ]
    time_constraint = model.addConstr(
        gp.LinExpr() <= horizon, name="booking_time"
    )

    variables = {}
    column_data = {}

    def add_offer_column(mask):
        if mask == 0 or mask in variables:
            return False

        revenue, consumption = compute_column(mask)
        column = gp.Column()
        for i, amount in enumerate(consumption):
            coefficient = global_lambda * amount
            if coefficient != 0.0:
                column.addTerms(coefficient, capacity_constraints[i])
        column.addTerms(1.0, time_constraint)

        var = model.addVar(
            lb=0.0,
            obj=global_lambda * revenue,
            column=column,
            name=f"offer_{len(variables)}"
        )
        variables[mask] = var
        column_data[mask] = (revenue, consumption)
        return True

    for mask in initial_masks:
        if time.monotonic() >= deadline:
            break
        add_offer_column(mask)

    if not variables and all_relevant:
        add_offer_column(all_relevant)

    best_solution = zero_solution
    best_objective = 0.0
    solved_once = False

    def extract_solution():
        active = []
        objective = 0.0

        for mask, var in variables.items():
            value = float(var.X)
            if value > 1e-8:
                revenue = column_data[mask][0]
                objective += global_lambda * revenue * value
                active.append({
                    "offer_set": mask_to_offer(mask),
                    "time_allocated": value
                })

        active.sort(
            key=lambda entry: (
                -entry["time_allocated"],
                entry["offer_set"]
            )
        )

        if abs(objective) < 1e-10:
            objective = 0.0

        return {
            "objective_value": float(objective),
            "active_columns": active
        }

    # Exact pricing by Gray-code enumeration for moderately sized instances.
    exact_pricing_limit = 18

    def exact_price(adjusted_revenues, time_dual, max_columns=80):
        top = []
        completed = True

        denominators = [segment[1] for segment in segments]
        numerators = [0.0] * len(segments)
        previous_gray = 0
        total_masks = 1 << n

        for integer in range(1, total_masks):
            if (integer & 4095) == 0 and time.monotonic() >= deadline:
                completed = False
                break

            gray = integer ^ (integer >> 1)
            diff = gray ^ previous_gray
            j = diff.bit_length() - 1
            adding = (gray & diff) != 0
            sign = 1.0 if adding else -1.0
            rj = adjusted_revenues[j]

            for seg_idx, v in memberships[j]:
                denominators[seg_idx] += sign * v
                numerators[seg_idx] += sign * v * rj

            value = 0.0
            for seg_idx, segment in enumerate(segments):
                value += (
                    segment[0]
                    * numerators[seg_idx]
                    / denominators[seg_idx]
                )

            reduced_cost = global_lambda * value - time_dual
            if gray not in variables and reduced_cost > 1e-8:
                item = (reduced_cost, gray)
                if len(top) < max_columns:
                    heapq.heappush(top, item)
                elif reduced_cost > top[0][0]:
                    heapq.heapreplace(top, item)

            previous_gray = gray

        top.sort(reverse=True)
        return top, completed

    iteration = 0
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0.01:
            break

        model.Params.TimeLimit = max(0.01, remaining)
        model.optimize()

        if model.SolCount <= 0:
            break

        solved_once = True
        current_solution = extract_solution()
        current_objective = current_solution["objective_value"]

        if current_objective > best_objective + 1e-7:
            best_objective = current_objective
            best_solution = current_solution
            if logger:
                logger.log_solution(best_objective, best_solution)

        if time.monotonic() >= deadline:
            break

        capacity_duals = [
            max(0.0, float(constraint.Pi))
            for constraint in capacity_constraints
        ]
        time_dual = max(0.0, float(time_constraint.Pi))

        adjusted_revenues = []
        for j in range(n):
            displacement_cost = sum(
                capacity_duals[i] for i in product_legs[j]
            )
            adjusted_revenues.append(fares[j] - displacement_cost)

        candidate_masks = []
        exact_completed = False

        if n <= exact_pricing_limit:
            priced, exact_completed = exact_price(
                adjusted_revenues, time_dual, max_columns=100
            )
            candidate_masks.extend(mask for _, mask in priced)
        else:
            active_starts = []
            try:
                active_starts = [
                    mask for mask, var in variables.items()
                    if var.X > 1e-8
                ]
            except gp.GurobiError:
                active_starts = []

            starts = make_search_starts(
                adjusted_revenues,
                active_starts
            )
            priced = local_search(
                adjusted_revenues,
                starts,
                max_results=80
            )
            for mask, value in priced:
                reduced_cost = global_lambda * value - time_dual
                if reduced_cost > 1e-8 and mask not in variables:
                    candidate_masks.append(mask)

        if not candidate_masks:
            # For exact completed pricing, this certifies LP optimality.
            # For large instances, it indicates convergence of the heuristic.
            break

        if deadline - time.monotonic() <= 0.03:
            break

        added = 0
        for mask in candidate_masks:
            if time.monotonic() >= deadline:
                break
            if add_offer_column(mask):
                added += 1

        if added == 0:
            break

        iteration += 1
        if iteration >= 500:
            break

    # If columns were added after the last solve, best_solution still contains
    # the last fully solved and feasible restricted-master incumbent.
    if not solved_once and time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining > 0.01:
            model.Params.TimeLimit = remaining
            model.optimize()
            if model.SolCount > 0:
                candidate = extract_solution()
                if candidate["objective_value"] > best_objective + 1e-7:
                    best_solution = candidate
                    best_objective = candidate["objective_value"]
                    if logger:
                        logger.log_solution(best_objective, best_solution)

    return best_solution


def main():
    args = parse_args()

    logger = (
        SolutionLogger(args.log_path, sense="maximize")
        if args.log_path
        else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    solution = solve(instance, args.time_limit, logger)
    write_json(args.solution_path, solution)


if __name__ == "__main__":
    main()