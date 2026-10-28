#!/usr/bin/env python3

import argparse
import json
import math
import random
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimum-cost exact partition of bus trips into pre-enumerated driver duties."
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the input instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path where the output solution JSON file will be written.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None,
                        help="Optional JSONL path for incumbent logging.")
    return parser.parse_args()


def canonicalize_duties(
    duties: Sequence[Sequence[int]],
    costs: Sequence[float],
) -> Tuple[List[Tuple[int, ...]], List[float], List[int]]:
    """
    Remove exact duplicate duties, keeping only the cheapest representative.

    Identical duty trip-sets are interchangeable in any feasible partition, so this
    reduction is safe and often useful when the candidate pool contains duplicates.
    Returns canonical trip tuples, costs, and original duty indices.
    """
    best: Dict[Tuple[int, ...], Tuple[float, int]] = {}

    for idx, raw_trips in enumerate(duties):
        key = tuple(sorted(set(int(t) for t in raw_trips)))
        if not key:
            continue

        cost = float(costs[idx])
        old = best.get(key)

        if (
            old is None
            or cost < old[0] - 1e-12
            or (abs(cost - old[0]) <= 1e-12 and idx < old[1])
        ):
            best[key] = (cost, idx)

    ordered = sorted(best.items(), key=lambda item: item[1][1])
    canonical_trips = [item[0] for item in ordered]
    canonical_costs = [float(item[1][0]) for item in ordered]
    original_indices = [int(item[1][1]) for item in ordered]
    return canonical_trips, canonical_costs, original_indices


def build_incidence(
    canonical_trips: Sequence[Tuple[int, ...]],
    num_trips: int,
) -> List[List[int]]:
    """Build trip -> canonical-duty-indices incidence."""
    by_trip: List[List[int]] = [[] for _ in range(num_trips)]
    for duty_idx, trips in enumerate(canonical_trips):
        for trip in trips:
            if trip < 0 or trip >= num_trips:
                raise ValueError(f"Duty contains invalid trip id {trip}.")
            by_trip[trip].append(duty_idx)
    return by_trip


def is_feasible_active(
    selected: Sequence[int],
    canonical_sets: Sequence[Set[int]],
    num_trips: int,
) -> bool:
    """Verify that active-duty indices cover every trip exactly once."""
    covered = bytearray(num_trips)
    covered_count = 0

    for duty_idx in selected:
        for trip in canonical_sets[duty_idx]:
            if covered[trip]:
                return False
            covered[trip] = 1
            covered_count += 1

    return covered_count == num_trips


def greedy_initial_solution(
    canonical_trips: Sequence[Tuple[int, ...]],
    costs: Sequence[float],
    by_trip: Sequence[Sequence[int]],
    num_trips: int,
    deadline: float,
    max_variants: int = 24,
) -> Optional[List[int]]:
    """
    Fast deterministic/randomized exact-cover construction used as a warm start
    and as a fallback if the MIP solver has not yet produced an incumbent.
    """
    if num_trips == 0:
        return []

    if any(not by_trip[trip] for trip in range(num_trips)):
        return None

    canonical_sets = [set(trips) for trips in canonical_trips]
    constrained_order = sorted(
        range(num_trips),
        key=lambda trip: (len(by_trip[trip]), trip),
    )

    rng = random.Random(0)
    base_orders = [
        constrained_order,
        list(reversed(constrained_order)),
        list(range(num_trips)),
    ]

    for variant in range(max_variants):
        if time.monotonic() >= deadline:
            break

        if variant < len(base_orders):
            trip_order = list(base_orders[variant])
        else:
            trip_order = list(constrained_order)
            rng.shuffle(trip_order)

        mode = variant % 5
        selected: List[int] = []
        covered = bytearray(num_trips)
        feasible = True

        for position, trip in enumerate(trip_order):
            if position & 31 == 0 and time.monotonic() >= deadline:
                feasible = False
                break

            if covered[trip]:
                continue

            best_duty: Optional[int] = None
            best_key = None

            for duty_idx in by_trip[trip]:
                duty_set = canonical_sets[duty_idx]
                if any(covered[t] for t in duty_set):
                    continue

                size = len(duty_set)
                cost = costs[duty_idx]

                if mode == 0:
                    primary = cost / size
                elif mode == 1:
                    primary = cost
                elif mode == 2:
                    primary = cost / float(size) ** 1.5
                elif mode == 3:
                    primary = cost - 0.1 * size
                else:
                    primary = cost / (size + 1)

                key = (primary, -size, cost, duty_idx)
                if best_key is None or key < best_key:
                    best_key = key
                    best_duty = duty_idx

            if best_duty is None:
                feasible = False
                break

            selected.append(best_duty)
            for covered_trip in canonical_sets[best_duty]:
                covered[covered_trip] = 1

            # Sorting only once at the end of the successful variant is enough.
        if feasible and selected:
            selected.sort()
            if is_feasible_active(selected, canonical_sets, num_trips):
                return selected

    return None


def main() -> None:
    args = parse_args()
    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path
        else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as instance_file:
        instance = json.load(instance_file)

    parameters = instance.get("parameters", {})
    trips_data = instance.get("trips", [])
    duties = instance.get("duties", [])
    costs = [float(c) for c in instance.get("costs", [])]

    num_trips = int(parameters.get("num_trips_M", len(trips_data)))
    if len(trips_data) != num_trips:
        num_trips = len(trips_data)

    canonical_trips, canonical_costs, original_indices = canonicalize_duties(
        duties, costs
    )
    canonical_sets = [set(trips) for trips in canonical_trips]
    by_trip = build_incidence(canonical_trips, num_trips)

    state = {
        "best_obj": math.inf,
        "best_active": None,
        "best_original": [],
    }

    def register_solution(
        active_selected: Sequence[int],
        log_improvement: bool = True,
    ) -> bool:
        """Record and optionally log a new globally improved feasible solution."""
        if active_selected is None:
            return False

        selected = sorted(set(int(d) for d in active_selected))
        if not is_feasible_active(selected, canonical_sets, num_trips):
            return False

        objective = float(sum(canonical_costs[d] for d in selected))
        if objective >= state["best_obj"] - 1e-9:
            return False

        original_selected = sorted(original_indices[d] for d in selected)
        solution_dict = {
            "objective_value": objective,
            "selected_duties": original_selected,
        }

        state["best_obj"] = objective
        state["best_active"] = list(selected)
        state["best_original"] = original_selected

        if log_improvement and logger is not None:
            logger.log_solution(objective, solution_dict)

        return True

    # Handle the degenerate zero-trip case immediately.
    if num_trips == 0:
        register_solution([])
    else:
        # A short construction phase gives Gurobi a full MIP start and provides
        # a feasible fallback for very tight time limits.
        heuristic_budget = min(1.0, max(0.0, 0.08 * max(0, args.time_limit)))
        heuristic_deadline = min(deadline, time.monotonic() + heuristic_budget)

        initial = greedy_initial_solution(
            canonical_trips=canonical_trips,
            costs=canonical_costs,
            by_trip=by_trip,
            num_trips=num_trips,
            deadline=heuristic_deadline,
        )
        if initial is not None:
            register_solution(initial)

        # Solve the binary weighted set-partitioning model:
        # minimize sum c_d x_d
        # subject to, for every trip t, sum_{d contains t} x_d = 1.
        remaining = deadline - time.monotonic()
        if remaining > 0 and canonical_trips:
            model = gp.Model("driver_duty_exact_partition")

            # Required solver settings.
            model.setParam("Seed", 0)
            model.setParam("MIPGap", 1e-4)
            model.setParam("NumericFocus", 0)
            model.setParam("Threads", 1)

            model.setParam("OutputFlag", 0)
            model.setParam("Presolve", 2)
            model.setParam("TimeLimit", max(0.0, remaining))

            num_duties = len(canonical_trips)
            x_vars_dict = model.addVars(
                range(num_duties),
                vtype=GRB.BINARY,
                obj=canonical_costs,
                name="x",
            )
            x_vars = [x_vars_dict[i] for i in range(num_duties)]

            for trip in range(num_trips):
                model.addConstr(
                    gp.quicksum(x_vars[d] for d in by_trip[trip]) == 1,
                    name=f"cover_trip_{trip}",
                )

            if state["best_active"] is not None:
                selected_set = set(state["best_active"])
                for duty_idx, var in enumerate(x_vars):
                    var.Start = 1 if duty_idx in selected_set else 0

            def incumbent_callback(m: gp.Model, where: int) -> None:
                if where != GRB.Callback.MIPSOL:
                    return
                try:
                    values = m.cbGetSolution(x_vars)
                    active_selected = [
                        duty_idx
                        for duty_idx, value in enumerate(values)
                        if value > 0.5
                    ]
                    register_solution(active_selected)
                except Exception:
                    # A logging/callback issue must not terminate the MIP search.
                    pass

            try:
                model.optimize(incumbent_callback)

                if model.SolCount > 0:
                    values = model.getAttr("X", x_vars)
                    active_selected = [
                        duty_idx
                        for duty_idx, value in enumerate(values)
                        if value > 0.5
                    ]
                    register_solution(active_selected)

            except gp.GurobiError:
                # Keep the best heuristic/fallback incumbent if solver startup
                # or optimization fails unexpectedly.
                pass

    if state["best_active"] is None:
        # Feasible benchmark instances should always produce an incumbent.  This
        # branch only guarantees a schema-valid file for malformed/infeasible data.
        final_solution = {
            "objective_value": 0.0,
            "selected_duties": [],
        }
    else:
        final_solution = {
            "objective_value": float(state["best_obj"]),
            "selected_duties": list(state["best_original"]),
        }

    with open(args.solution_path, "w", encoding="utf-8") as solution_file:
        json.dump(final_solution, solution_file, indent=2)
        solution_file.write("\n")


if __name__ == "__main__":
    main()