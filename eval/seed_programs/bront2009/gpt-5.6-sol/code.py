import argparse
import heapq
import json
import os
import random
import time

import numpy as np

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
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, allow_nan=False)


class PackingLPSolver:
    """
    Small custom revised-simplex solver for

        maximize c'x
        subject to Ax <= b
                   x >= 0.

    The number of constraints is the number of flight legs plus one, which is
    usually much smaller than the number of assortment columns.
    """

    def __init__(self, rhs, deadline):
        self.rhs = np.asarray(rhs, dtype=float)
        self.num_rows = len(rhs)
        self.deadline = deadline

    def solve(self, columns, objectives, max_iterations=10000):
        num_variables = len(columns)
        num_rows = self.num_rows

        if num_variables == 0:
            return {
                "values": np.zeros(0, dtype=float),
                "objective": 0.0,
                "duals": np.zeros(num_rows, dtype=float),
                "optimal": True,
            }

        matrix = np.column_stack(columns).astype(float, copy=False)
        objective = np.asarray(objectives, dtype=float)

        # Slack variables form the initial basis.
        basis = [num_variables + row for row in range(num_rows)]
        basic_values = self.rhs.copy()

        tolerance = 1e-9
        optimal = False
        duals = np.zeros(num_rows, dtype=float)

        def get_column(index):
            if index < num_variables:
                return matrix[:, index]
            result = np.zeros(num_rows, dtype=float)
            result[index - num_variables] = 1.0
            return result

        def get_cost(index):
            return objective[index] if index < num_variables else 0.0

        for iteration in range(max_iterations):
            if iteration % 20 == 0 and time.monotonic() >= self.deadline:
                break

            basis_matrix = np.column_stack([get_column(index) for index in basis])
            basis_costs = np.asarray([get_cost(index) for index in basis])

            try:
                basic_values = np.linalg.solve(basis_matrix, self.rhs)
                duals = np.linalg.solve(basis_matrix.T, basis_costs)
            except np.linalg.LinAlgError:
                # This should be rare. The current basis is still associated
                # with the last feasible solution, so stop safely.
                break

            basic_values[np.abs(basic_values) < 1e-11] = 0.0
            basis_set = set(basis)

            best_entering = -1
            best_reduced_cost = tolerance

            # Original assortment variables.
            reduced_costs = objective - matrix.T.dot(duals)
            for index in range(num_variables):
                if index not in basis_set:
                    reduced_cost = reduced_costs[index]
                    if reduced_cost > best_reduced_cost:
                        best_reduced_cost = reduced_cost
                        best_entering = index

            # Nonbasic slack variables can also re-enter.
            for row in range(num_rows):
                index = num_variables + row
                if index not in basis_set:
                    reduced_cost = -duals[row]
                    if reduced_cost > best_reduced_cost:
                        best_reduced_cost = reduced_cost
                        best_entering = index

            if best_entering < 0:
                optimal = True
                break

            entering_column = get_column(best_entering)
            try:
                direction = np.linalg.solve(basis_matrix, entering_column)
            except np.linalg.LinAlgError:
                break

            leaving_row = -1
            best_ratio = float("inf")
            best_basis_index = float("inf")

            for row in range(num_rows):
                if direction[row] > tolerance:
                    value = max(0.0, basic_values[row])
                    ratio = value / direction[row]
                    if (
                        ratio < best_ratio - 1e-11
                        or (
                            abs(ratio - best_ratio) <= 1e-11
                            and basis[row] < best_basis_index
                        )
                    ):
                        best_ratio = ratio
                        best_basis_index = basis[row]
                        leaving_row = row

            if leaving_row < 0:
                # The model would be unbounded. In this problem the booking
                # time row has coefficient one in every assortment column, so
                # unboundedness cannot occur for valid inputs.
                break

            basis[leaving_row] = best_entering

        # Recover the last basis solution.
        basis_matrix = np.column_stack([get_column(index) for index in basis])
        basis_costs = np.asarray([get_cost(index) for index in basis])
        try:
            basic_values = np.linalg.solve(basis_matrix, self.rhs)
            duals = np.linalg.solve(basis_matrix.T, basis_costs)
        except np.linalg.LinAlgError:
            basic_values = np.zeros(num_rows, dtype=float)
            duals = np.zeros(num_rows, dtype=float)
            optimal = False

        values = np.zeros(num_variables, dtype=float)
        for row, index in enumerate(basis):
            if index < num_variables and basic_values[row] > 1e-9:
                values[index] = basic_values[row]

        values[values < 1e-9] = 0.0
        achieved_objective = float(objective.dot(values))

        # Clamp tiny negative dual values caused by numerical noise.
        duals[np.abs(duals) < 1e-10] = 0.0

        return {
            "values": values,
            "objective": achieved_objective,
            "duals": duals,
            "optimal": optimal,
        }


def solve(instance, time_limit, logger):
    start_time = time.monotonic()
    deadline = start_time + max(0, time_limit)

    zero_solution = {
        "objective_value": 0.0,
        "active_columns": [],
    }
    if logger:
        logger.log_solution(0.0, zero_solution)

    products_data = instance.get("products", [])
    segments_data = instance.get("segments", [])
    legs_data = instance.get("network", {}).get("legs", [])

    num_products = len(products_data)
    num_legs = len(legs_data)
    arrival_rate = max(0.0, float(instance.get("lambda", 0.0)))
    horizon = max(
        0.0, float(instance.get("booking_horizon", {}).get("T", 0.0))
    )
    alpha = max(0.0, float(instance.get("alpha", 1.0)))

    if (
        num_products == 0
        or horizon <= 0.0
        or arrival_rate <= 0.0
        or time_limit <= 0
    ):
        return zero_solution

    product_ids = [product["product_id"] for product in products_data]
    product_index = {
        product_id: index for index, product_id in enumerate(product_ids)
    }
    fares = np.asarray(
        [float(product.get("fare", 0.0)) for product in products_data],
        dtype=float,
    )

    leg_index = {
        leg["leg_id"]: index for index, leg in enumerate(legs_data)
    }
    capacities = [
        max(0.0, alpha * float(leg.get("capacity", 0.0)))
        for leg in legs_data
    ]

    product_legs = []
    for product in products_data:
        used = []
        seen = set()
        for leg_id in product.get("legs_used", []):
            if leg_id in leg_index:
                row = leg_index[leg_id]
                if row not in seen:
                    seen.add(row)
                    used.append(row)
        product_legs.append(used)

    # Segment format:
    # (segment probability, no-purchase weight, [(product, weight), ...])
    segments = []
    memberships = [[] for _ in range(num_products)]

    for raw_segment in segments_data:
        segment_rate = max(0.0, float(raw_segment.get("lambda_l", 0.0)))
        segment_probability = segment_rate / arrival_rate
        if segment_probability <= 0.0:
            continue

        no_purchase = max(
            1e-12, float(raw_segment.get("no_purchase_preference", 0.0))
        )
        consideration = raw_segment.get("consideration_set", [])
        preferences = raw_segment.get("preference_vector", [])

        combined = {}
        for product_id, preference in zip(consideration, preferences):
            if product_id not in product_index:
                continue
            weight = float(preference)
            if weight > 0.0:
                index = product_index[product_id]
                combined[index] = combined.get(index, 0.0) + weight

        entries = sorted(combined.items())
        segment_index = len(segments)
        segments.append((segment_probability, no_purchase, entries))

        for product, weight in entries:
            memberships[product].append((segment_index, weight))

    if not segments:
        return zero_solution

    def mask_to_offer(mask):
        result = []
        while mask:
            bit = mask & -mask
            index = bit.bit_length() - 1
            result.append(product_ids[index])
            mask ^= bit
        return result

    column_cache = {}

    def compute_column(mask):
        cached = column_cache.get(mask)
        if cached is not None:
            return cached

        probabilities = np.zeros(num_products, dtype=float)

        for probability, no_purchase, entries in segments:
            denominator = no_purchase
            selected = []

            for product, weight in entries:
                if mask & (1 << product):
                    denominator += weight
                    selected.append((product, weight))

            if selected:
                factor = probability / denominator
                for product, weight in selected:
                    probabilities[product] += factor * weight

        revenue = float(fares.dot(probabilities))
        consumption = np.zeros(num_legs, dtype=float)

        nonzero_products = np.flatnonzero(probabilities > 0.0)
        for product in nonzero_products:
            probability = probabilities[product]
            for leg in product_legs[product]:
                consumption[leg] += probability

        result = revenue, consumption
        column_cache[mask] = result
        return result

    def adjusted_assortment_value(mask, adjusted_revenues):
        value = 0.0
        for probability, no_purchase, entries in segments:
            denominator = no_purchase
            numerator = 0.0
            for product, weight in entries:
                if mask & (1 << product):
                    denominator += weight
                    numerator += weight * adjusted_revenues[product]
            value += probability * numerator / denominator
        return value

    rng = random.Random(0)

    def local_search(adjusted_revenues, starts, max_results=50):
        results = {}
        max_steps = max(10, min(4 * num_products, 500))

        for initial_mask in starts:
            if time.monotonic() >= deadline:
                break

            mask = initial_mask
            denominators = np.asarray(
                [segment[1] for segment in segments], dtype=float
            )
            numerators = np.zeros(len(segments), dtype=float)

            remaining = mask
            while remaining:
                bit = remaining & -remaining
                product = bit.bit_length() - 1
                remaining ^= bit
                adjusted = adjusted_revenues[product]
                for segment_index, weight in memberships[product]:
                    denominators[segment_index] += weight
                    numerators[segment_index] += weight * adjusted

            current_value = 0.0
            for segment_index, segment in enumerate(segments):
                current_value += (
                    segment[0]
                    * numerators[segment_index]
                    / denominators[segment_index]
                )

            for step in range(max_steps):
                if step % 5 == 0 and time.monotonic() >= deadline:
                    break

                best_product = -1
                best_improvement = 1e-10

                for product in range(num_products):
                    selected = bool(mask & (1 << product))
                    sign = -1.0 if selected else 1.0
                    adjusted = adjusted_revenues[product]
                    improvement = 0.0

                    for segment_index, weight in memberships[product]:
                        old_denominator = denominators[segment_index]
                        old_numerator = numerators[segment_index]
                        new_denominator = old_denominator + sign * weight
                        new_numerator = (
                            old_numerator + sign * weight * adjusted
                        )
                        probability = segments[segment_index][0]
                        improvement += probability * (
                            new_numerator / new_denominator
                            - old_numerator / old_denominator
                        )

                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_product = product

                if best_product < 0:
                    break

                if mask & (1 << best_product):
                    sign = -1.0
                    mask &= ~(1 << best_product)
                else:
                    sign = 1.0
                    mask |= 1 << best_product

                adjusted = adjusted_revenues[best_product]
                for segment_index, weight in memberships[best_product]:
                    denominators[segment_index] += sign * weight
                    numerators[segment_index] += sign * weight * adjusted

                current_value += best_improvement

            if mask:
                old_value = results.get(mask)
                if old_value is None or current_value > old_value:
                    results[mask] = current_value

        return sorted(
            results.items(), key=lambda item: item[1], reverse=True
        )[:max_results]

    def make_search_starts(adjusted_revenues, extra_masks=None):
        starts = {0}

        if extra_masks:
            starts.update(extra_masks)

        positive_mask = 0
        for product in range(num_products):
            if adjusted_revenues[product] > 0.0 and memberships[product]:
                positive_mask |= 1 << product
        starts.add(positive_mask)

        order = sorted(
            range(num_products),
            key=lambda product: adjusted_revenues[product],
            reverse=True,
        )

        running = 0
        prefix_points = set(range(1, min(20, num_products) + 1))
        if num_products > 20:
            for quantile in range(1, 11):
                prefix_points.add(
                    max(1, int(round(quantile * num_products / 10.0)))
                )

        for position, product in enumerate(order, 1):
            if adjusted_revenues[product] > 0.0 and memberships[product]:
                running |= 1 << product
            if position in prefix_points:
                starts.add(running)

        for product in order[: min(15, num_products)]:
            if memberships[product]:
                starts.add(1 << product)

        for _, _, entries in segments:
            mask = 0
            for product, _ in entries:
                if adjusted_revenues[product] > 0.0:
                    mask |= 1 << product
            starts.add(mask)

        random_count = 12 if num_products <= 200 else 5
        for _ in range(random_count):
            probability = rng.choice((0.15, 0.3, 0.5, 0.7))
            mask = 0
            for product in range(num_products):
                if (
                    adjusted_revenues[product] > 0.0
                    and memberships[product]
                    and rng.random() < probability
                ):
                    mask |= 1 << product
            starts.add(mask)

        return list(starts)

    assortment_masks = []
    assortment_set = set()
    master_columns = []
    master_objectives = []

    def add_assortment(mask):
        if mask == 0 or mask in assortment_set:
            return False

        revenue, consumption = compute_column(mask)
        column = np.empty(num_legs + 1, dtype=float)
        if num_legs:
            column[:num_legs] = arrival_rate * consumption
        column[num_legs] = 1.0

        assortment_set.add(mask)
        assortment_masks.append(mask)
        master_columns.append(column)
        master_objectives.append(arrival_rate * revenue)
        return True

    # Initial restricted-master columns.
    if num_products <= 1000:
        for product in range(num_products):
            if memberships[product]:
                add_assortment(1 << product)
    else:
        top_products = sorted(
            range(num_products), key=lambda product: fares[product], reverse=True
        )[:1000]
        for product in top_products:
            if memberships[product]:
                add_assortment(1 << product)

    all_products_mask = 0
    for product in range(num_products):
        if memberships[product]:
            all_products_mask |= 1 << product
    add_assortment(all_products_mask)

    segment_masks = []
    for _, _, entries in segments:
        mask = 0
        for product, _ in entries:
            mask |= 1 << product
        if mask:
            segment_masks.append(mask)
            add_assortment(mask)

    fare_order = sorted(
        range(num_products), key=lambda product: fares[product], reverse=True
    )
    running = 0
    for position, product in enumerate(fare_order[: min(150, num_products)], 1):
        if memberships[product]:
            running |= 1 << product
        if position <= 20 or position % 5 == 0:
            add_assortment(running)

    if time.monotonic() < deadline:
        starts = make_search_starts(fares, segment_masks)
        for mask, _ in local_search(fares, starts, max_results=80):
            add_assortment(mask)

    if not assortment_masks:
        return zero_solution

    rhs = capacities + [horizon]
    master_solver = PackingLPSolver(rhs, deadline)

    best_solution = zero_solution
    best_objective = 0.0
    last_active_masks = []

    def build_solution(values):
        active_columns = []
        calculated_objective = 0.0

        for index, value in enumerate(values):
            value = float(value)
            if value <= 1e-8:
                continue

            mask = assortment_masks[index]
            revenue, _ = compute_column(mask)
            calculated_objective += arrival_rate * revenue * value
            active_columns.append(
                {
                    "offer_set": mask_to_offer(mask),
                    "time_allocated": value,
                }
            )

        active_columns.sort(
            key=lambda entry: (-entry["time_allocated"], entry["offer_set"])
        )

        if abs(calculated_objective) < 1e-10:
            calculated_objective = 0.0

        return {
            "objective_value": float(calculated_objective),
            "active_columns": active_columns,
        }

    def exact_pricing(adjusted_revenues, time_dual, max_columns=100):
        candidates = []
        completed = True

        denominators = np.asarray(
            [segment[1] for segment in segments], dtype=float
        )
        numerators = np.zeros(len(segments), dtype=float)

        previous_gray = 0
        total = 1 << num_products

        for integer in range(1, total):
            if integer % 2048 == 0 and time.monotonic() >= deadline:
                completed = False
                break

            gray = integer ^ (integer >> 1)
            changed_bit = gray ^ previous_gray
            product = changed_bit.bit_length() - 1
            sign = 1.0 if gray & changed_bit else -1.0
            adjusted = adjusted_revenues[product]

            for segment_index, weight in memberships[product]:
                denominators[segment_index] += sign * weight
                numerators[segment_index] += sign * weight * adjusted

            value = 0.0
            for segment_index, segment in enumerate(segments):
                value += (
                    segment[0]
                    * numerators[segment_index]
                    / denominators[segment_index]
                )

            reduced_cost = arrival_rate * value - time_dual
            if gray not in assortment_set and reduced_cost > 1e-8:
                item = (reduced_cost, gray)
                if len(candidates) < max_columns:
                    heapq.heappush(candidates, item)
                elif reduced_cost > candidates[0][0]:
                    heapq.heapreplace(candidates, item)

            previous_gray = gray

        candidates.sort(reverse=True)
        return [mask for _, mask in candidates], completed

    exact_limit = 20
    max_rounds = 500

    for _ in range(max_rounds):
        if time.monotonic() >= deadline:
            break

        result = master_solver.solve(master_columns, master_objectives)

        solution = build_solution(result["values"])
        objective = solution["objective_value"]

        if objective > best_objective + 1e-7:
            best_objective = objective
            best_solution = solution
            if logger:
                logger.log_solution(best_objective, best_solution)

        last_active_masks = [
            assortment_masks[index]
            for index, value in enumerate(result["values"])
            if value > 1e-8
        ]

        if not result["optimal"] or time.monotonic() >= deadline:
            break

        duals = result["duals"]
        capacity_duals = duals[:num_legs]
        time_dual = duals[num_legs]

        adjusted_revenues = fares.copy()
        for product in range(num_products):
            displacement = 0.0
            for leg in product_legs[product]:
                displacement += capacity_duals[leg]
            adjusted_revenues[product] -= displacement

        candidate_masks = []

        if num_products <= exact_limit:
            candidate_masks, completed = exact_pricing(
                adjusted_revenues, time_dual
            )
            if not candidate_masks and completed:
                break
        else:
            starts = make_search_starts(
                adjusted_revenues, last_active_masks
            )
            priced = local_search(
                adjusted_revenues, starts, max_results=100
            )
            for mask, value in priced:
                reduced_cost = arrival_rate * value - time_dual
                if reduced_cost > 1e-8 and mask not in assortment_set:
                    candidate_masks.append(mask)

            if not candidate_masks:
                break

        added = 0
        for mask in candidate_masks:
            if time.monotonic() >= deadline:
                break
            if add_assortment(mask):
                added += 1

        if added == 0:
            break

    return best_solution


def main():
    args = parse_args()

    logger = (
        SolutionLogger(args.log_path, sense="maximize")
        if args.log_path
        else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as file:
        instance = json.load(file)

    solution = solve(instance, args.time_limit, logger)
    write_json(args.solution_path, solution)


if __name__ == "__main__":
    main()