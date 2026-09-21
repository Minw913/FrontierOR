import argparse
import json
import math
import os
import time
from collections import defaultdict, deque

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


def parse_bin_limit(value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"infinity", "inf", "infinite", "unrestricted"}:
            return None
        return int(text)
    return int(value)


def first_fit_decreasing(items, weights, capacity):
    """Return a list of bins, where each bin is a list of item indices."""
    bins = []
    loads = []

    for i in sorted(items, key=lambda x: (-weights[x], x)):
        chosen = None
        for b, load in enumerate(loads):
            if load + weights[i] <= capacity:
                chosen = b
                break

        if chosen is None:
            bins.append([i])
            loads.append(weights[i])
        else:
            bins[chosen].append(i)
            loads[chosen] += weights[i]

    return bins


def pack_period_assignment(periods, weights, T, capacity, max_bins_per_period):
    groups = []
    feasible = True

    by_period = [[] for _ in range(T + 1)]
    for i, period in enumerate(periods):
        if period is None or period < 1 or period > T:
            return None
        by_period[period].append(i)

    for period in range(1, T + 1):
        period_bins = first_fit_decreasing(by_period[period], weights, capacity)
        if max_bins_per_period is not None and len(period_bins) > max_bins_per_period:
            feasible = False
        for item_list in period_bins:
            groups.append((period, item_list))

    return groups if feasible else None


def packing_score(partial_periods, weights, T, capacity, max_bins_per_period):
    by_period = [[] for _ in range(T + 1)]
    for i, period in enumerate(partial_periods):
        if period is not None:
            by_period[period].append(i)

    total_bins = 0
    excess_bins = 0
    peak_bins = 0

    for period in range(1, T + 1):
        count = len(first_fit_decreasing(by_period[period], weights, capacity))
        total_bins += count
        peak_bins = max(peak_bins, count)
        if max_bins_per_period is not None:
            excess_bins += max(0, count - max_bins_per_period)

    return excess_bins, total_bins, peak_bins


def node_value(node_id, real_id_to_index, periods, source_id, sink_id, T):
    if node_id == source_id:
        return 0
    if node_id == sink_id:
        return T + 1
    idx = real_id_to_index.get(node_id)
    if idx is None:
        return None
    return periods[idx]


def verify_periods(periods, arcs, real_id_to_index, source_id, sink_id, T):
    if any(p is None or p < 1 or p > T for p in periods):
        return False

    for arc in arcs:
        tail = node_value(
            arc["from"], real_id_to_index, periods, source_id, sink_id, T
        )
        head = node_value(
            arc["to"], real_id_to_index, periods, source_id, sink_id, T
        )
        if tail is None or head is None:
            continue
        if tail + int(arc["lag"]) > head:
            return False
    return True


def compute_earliest_periods(
    n, arcs, real_id_to_index, source_id, sink_id, T
):
    periods = [1] * n

    # Longest-path style relaxation for lower-bound difference constraints.
    # Source and sink remain fixed anchors.
    for _ in range(n + 2):
        changed = False
        for arc in arcs:
            head_idx = real_id_to_index.get(arc["to"])
            if head_idx is None:
                continue

            tail_id = arc["from"]
            if tail_id == source_id:
                tail_value = 0
            elif tail_id == sink_id:
                tail_value = T + 1
            else:
                tail_idx = real_id_to_index.get(tail_id)
                if tail_idx is None:
                    continue
                tail_value = periods[tail_idx]

            candidate = tail_value + int(arc["lag"])
            if candidate > periods[head_idx]:
                periods[head_idx] = candidate
                changed = True

        if not changed:
            break

    if verify_periods(
        periods, arcs, real_id_to_index, source_id, sink_id, T
    ):
        return periods
    return None


def real_components(n, arcs, real_id_to_index):
    adjacency = [[] for _ in range(n)]

    for arc in arcs:
        u = real_id_to_index.get(arc["from"])
        v = real_id_to_index.get(arc["to"])
        if u is not None and v is not None:
            adjacency[u].append(v)
            adjacency[v].append(u)

    components = []
    seen = [False] * n

    for start in range(n):
        if seen[start]:
            continue
        component = []
        queue = deque([start])
        seen[start] = True

        while queue:
            u = queue.popleft()
            component.append(u)
            for v in adjacency[u]:
                if not seen[v]:
                    seen[v] = True
                    queue.append(v)

        components.append(component)

    return components


def valid_component_shifts(
    base_periods,
    component,
    arcs,
    real_id_to_index,
    source_id,
    sink_id,
    T,
):
    component_set = set(component)
    lower = 1 - min(base_periods[i] for i in component)
    upper = T - max(base_periods[i] for i in component)
    valid = []

    relevant_arcs = []
    for arc in arcs:
        u = real_id_to_index.get(arc["from"])
        v = real_id_to_index.get(arc["to"])
        if (u in component_set) or (v in component_set):
            relevant_arcs.append(arc)

    trial = list(base_periods)
    for shift in range(lower, upper + 1):
        for i in component:
            trial[i] = base_periods[i] + shift

        okay = True
        for arc in relevant_arcs:
            tail = node_value(
                arc["from"], real_id_to_index, trial, source_id, sink_id, T
            )
            head = node_value(
                arc["to"], real_id_to_index, trial, source_id, sink_id, T
            )
            if tail is None or head is None:
                continue
            if tail + int(arc["lag"]) > head:
                okay = False
                break

        if okay:
            valid.append(shift)

    for i in component:
        trial[i] = base_periods[i]

    return valid


def improve_by_component_shifting(
    base_periods,
    components,
    arcs,
    real_id_to_index,
    source_id,
    sink_id,
    T,
    weights,
    capacity,
    max_bins_per_period,
):
    component_data = []
    for component in components:
        shifts = valid_component_shifts(
            base_periods,
            component,
            arcs,
            real_id_to_index,
            source_id,
            sink_id,
            T,
        )
        if not shifts:
            shifts = [0]
        total_weight = sum(weights[i] for i in component)
        component_data.append((component, shifts, total_weight))

    component_data.sort(key=lambda x: (-x[2], -len(x[0]), min(x[0])))

    partial = [None] * len(base_periods)

    for component, shifts, _ in component_data:
        best_shift = shifts[0]
        best_score = None

        # Prefer shifts close to zero when packing scores tie.
        for shift in sorted(shifts, key=lambda d: (abs(d), d)):
            for i in component:
                partial[i] = base_periods[i] + shift

            score = packing_score(
                partial, weights, T, capacity, max_bins_per_period
            )
            tie_break = (score[0], score[1], score[2], abs(shift), shift)

            if best_score is None or tie_break < best_score:
                best_score = tie_break
                best_shift = shift

        for i in component:
            partial[i] = base_periods[i] + best_shift

    return partial


def solution_from_groups(groups, item_ids, weights):
    groups_by_period = defaultdict(list)
    for _, (period, item_list) in enumerate(groups):
        if item_list:
            groups_by_period[period].append(list(item_list))

    assignments = []
    active = []

    for period in sorted(groups_by_period):
        period_groups = groups_by_period[period]
        period_groups.sort(
            key=lambda g: (
                -sum(weights[i] for i in g),
                min(item_ids[i] for i in g),
            )
        )

        for local_bin, item_list in enumerate(period_groups, start=1):
            sorted_items = sorted(item_list, key=lambda i: item_ids[i])
            total_weight = sum(weights[i] for i in sorted_items)

            active.append(
                {
                    "bin": local_bin,
                    "period": period,
                    "items": [item_ids[i] for i in sorted_items],
                    "total_weight": int(total_weight),
                }
            )

            for i in sorted_items:
                assignments.append(
                    {
                        "item_id": item_ids[i],
                        "bin": local_bin,
                        "period": period,
                        "weight": int(weights[i]),
                    }
                )

    assignments.sort(key=lambda x: x["item_id"])
    active.sort(key=lambda x: (x["period"], x["bin"]))

    objective = len(active)
    return {
        "objective_value": float(objective),
        "assignments": assignments,
        "active_bin_periods": active,
        "num_active_bin_periods": objective,
    }


def groups_from_values(a_values, q_values, n, K, T):
    selected_period = [None] * K
    for k in range(K):
        best_t = None
        best_value = 0.5
        offset = k * T
        for t0 in range(T):
            value = q_values[offset + t0]
            if value > best_value:
                best_value = value
                best_t = t0 + 1
        selected_period[k] = best_t

    grouped = [[] for _ in range(K)]
    for i in range(n):
        best_k = None
        best_value = 0.5
        offset = i * K
        for k in range(K):
            value = a_values[offset + k]
            if value > best_value:
                best_value = value
                best_k = k
        if best_k is None:
            best_k = max(range(K), key=lambda k: a_values[offset + k])
        grouped[best_k].append(i)

    groups = []
    for k in range(K):
        if not grouped[k]:
            continue
        period = selected_period[k]
        if period is None:
            return None
        groups.append((period, grouped[k]))

    return groups


def write_solution(path, solution):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(solution, f, indent=2)


def main():
    args = parse_args()
    start_time = time.monotonic()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    T = int(instance["parameters"]["T"])
    capacity = int(instance["parameters"]["W"])
    max_bins_per_period = parse_bin_limit(instance["parameters"].get("L"))

    source_id = int(instance["source_item_id"])
    sink_id = int(instance["sink_item_id"])
    arcs = instance.get("arcs", [])

    treatment_items = [
        item for item in instance["items"] if item.get("type") == "treatment"
    ]
    treatment_items.sort(key=lambda x: int(x["item_id"]))

    n = len(treatment_items)
    item_ids = [int(item["item_id"]) for item in treatment_items]
    weights = [int(item["weight"]) for item in treatment_items]
    real_id_to_index = {item_id: i for i, item_id in enumerate(item_ids)}

    if n == 0:
        empty_solution = {
            "objective_value": 0.0,
            "assignments": [],
            "active_bin_periods": [],
            "num_active_bin_periods": 0,
        }
        if logger:
            logger.log_solution(0.0, empty_solution)
        write_solution(args.solution_path, empty_solution)
        return

    best_solution = None
    best_objective = math.inf
    heuristic_groups = None

    earliest = compute_earliest_periods(
        n, arcs, real_id_to_index, source_id, sink_id, T
    )

    candidates = []
    if earliest is not None:
        candidates.append(earliest)

        components = real_components(n, arcs, real_id_to_index)
        shifted = improve_by_component_shifting(
            earliest,
            components,
            arcs,
            real_id_to_index,
            source_id,
            sink_id,
            T,
            weights,
            capacity,
            max_bins_per_period,
        )
        if verify_periods(
            shifted, arcs, real_id_to_index, source_id, sink_id, T
        ):
            candidates.append(shifted)

    for periods in candidates:
        groups = pack_period_assignment(
            periods, weights, T, capacity, max_bins_per_period
        )
        if groups is None:
            continue
        solution = solution_from_groups(groups, item_ids, weights)
        objective = solution["num_active_bin_periods"]
        if objective < best_objective:
            best_objective = objective
            best_solution = solution
            heuristic_groups = groups
            if logger:
                logger.log_solution(float(objective), solution)

    if max_bins_per_period is None:
        maximum_slots = n
    else:
        maximum_slots = min(n, max_bins_per_period * T)

    if heuristic_groups is not None:
        K = min(maximum_slots, len(heuristic_groups))
    else:
        K = maximum_slots

    if K <= 0:
        if best_solution is not None:
            write_solution(args.solution_path, best_solution)
            return
        raise RuntimeError("No bin slots are available for a nonempty instance.")

    model = gp.Model("capacitated_treatment_scheduling")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.MIPFocus = 1
    model.Params.Heuristics = 0.20

    elapsed = time.monotonic() - start_time
    remaining = max(0.01, float(args.time_limit) - elapsed)
    model.Params.TimeLimit = remaining

    # a[i,k] assigns real item i to global active-slot candidate k.
    a = model.addVars(n, K, vtype=GRB.BINARY, name="assign")
    y = model.addVars(K, vtype=GRB.BINARY, name="used")
    q = model.addVars(K, T, vtype=GRB.BINARY, name="slot_period")
    p = model.addVars(n, lb=1, ub=T, vtype=GRB.INTEGER, name="period")

    for i in range(n):
        model.addConstr(
            gp.quicksum(a[i, k] for k in range(K)) == 1,
            name=f"one_slot_{i}",
        )

    for k in range(K):
        model.addConstr(
            gp.quicksum(weights[i] * a[i, k] for i in range(n))
            <= capacity * y[k],
            name=f"capacity_{k}",
        )
        model.addConstr(
            gp.quicksum(q[k, t0] for t0 in range(T)) == y[k],
            name=f"one_period_{k}",
        )

    # Used slots form a prefix.
    for k in range(K - 1):
        model.addConstr(y[k] >= y[k + 1], name=f"used_prefix_{k}")

    slot_period_expr = [
        gp.quicksum((t0 + 1) * q[k, t0] for t0 in range(T))
        for k in range(K)
    ]

    # Sort used slots by period to reduce slot symmetry.
    for k in range(K - 1):
        model.addConstr(
            slot_period_expr[k]
            <= slot_period_expr[k + 1] + T * (1 - y[k + 1]),
            name=f"period_order_{k}",
        )

    for i in range(n):
        for k in range(K):
            model.addConstr(
                p[i] - slot_period_expr[k] <= T * (1 - a[i, k]),
                name=f"period_link_pos_{i}_{k}",
            )
            model.addConstr(
                slot_period_expr[k] - p[i] <= T * (1 - a[i, k]),
                name=f"period_link_neg_{i}_{k}",
            )

    if max_bins_per_period is not None:
        for t0 in range(T):
            model.addConstr(
                gp.quicksum(q[k, t0] for k in range(K))
                <= max_bins_per_period,
                name=f"period_bin_limit_{t0 + 1}",
            )

    def timing_expression(node_id):
        if node_id == source_id:
            return 0
        if node_id == sink_id:
            return T + 1
        idx = real_id_to_index.get(node_id)
        if idx is None:
            return None
        return p[idx]

    for arc_index, arc in enumerate(arcs):
        tail_expr = timing_expression(int(arc["from"]))
        head_expr = timing_expression(int(arc["to"]))
        if tail_expr is None or head_expr is None:
            continue
        model.addConstr(
            tail_expr + int(arc["lag"]) <= head_expr,
            name=f"lag_{arc_index}",
        )

    model.setObjective(gp.quicksum(y[k] for k in range(K)), GRB.MINIMIZE)

    # Warm start from the best constructive solution.
    if heuristic_groups is not None and len(heuristic_groups) <= K:
        sorted_groups = sorted(
            heuristic_groups,
            key=lambda x: (
                x[0],
                -sum(weights[i] for i in x[1]),
                min(x[1]),
            ),
        )

        item_to_slot = {}
        item_period = {}
        for k, (period, members) in enumerate(sorted_groups):
            y[k].Start = 1.0
            for t0 in range(T):
                q[k, t0].Start = 1.0 if t0 + 1 == period else 0.0
            for i in members:
                item_to_slot[i] = k
                item_period[i] = period

        for k in range(len(sorted_groups), K):
            y[k].Start = 0.0
            for t0 in range(T):
                q[k, t0].Start = 0.0

        for i in range(n):
            p[i].Start = float(item_period[i])
            selected_k = item_to_slot[i]
            for k in range(K):
                a[i, k].Start = 1.0 if k == selected_k else 0.0

    a_list = [a[i, k] for i in range(n) for k in range(K)]
    q_list = [q[k, t0] for k in range(K) for t0 in range(T)]

    callback_state = {
        "best": best_objective,
        "solution": best_solution,
    }

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return

        try:
            objective = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
            if objective >= callback_state["best"] - 1e-6:
                return

            a_values = cb_model.cbGetSolution(a_list)
            q_values = cb_model.cbGetSolution(q_list)
            groups = groups_from_values(a_values, q_values, n, K, T)
            if groups is None:
                return

            solution = solution_from_groups(groups, item_ids, weights)
            actual_objective = solution["num_active_bin_periods"]

            if actual_objective < callback_state["best"]:
                callback_state["best"] = actual_objective
                callback_state["solution"] = solution
                if logger:
                    logger.log_solution(float(actual_objective), solution)
        except Exception:
            # Logging or incumbent extraction must not interrupt optimization.
            pass

    model.optimize(incumbent_callback)

    best_solution = callback_state["solution"]
    best_objective = callback_state["best"]

    if model.SolCount > 0:
        final_a = [var.X for var in a_list]
        final_q = [var.X for var in q_list]
        final_groups = groups_from_values(final_a, final_q, n, K, T)

        if final_groups is not None:
            final_solution = solution_from_groups(
                final_groups, item_ids, weights
            )
            final_objective = final_solution["num_active_bin_periods"]

            if final_objective < best_objective:
                best_objective = final_objective
                best_solution = final_solution
                if logger:
                    logger.log_solution(float(final_objective), final_solution)

    if best_solution is None:
        raise RuntimeError(
            "No feasible solution was found within the supplied time limit."
        )

    write_solution(args.solution_path, best_solution)


if __name__ == "__main__":
    main()