import argparse
import itertools
import json
import math
import os
import random
import time

import numpy as np

from solution_logger import SolutionLogger


def write_json(path, data):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), allow_nan=False)
    os.replace(temp_path, path)


def prune_selection(selected, masks, all_mask):
    selected = list(dict.fromkeys(selected))
    changed = True
    while changed and len(selected) > 1:
        changed = False
        for pos in range(len(selected) - 1, -1, -1):
            union_mask = 0
            for q, idx in enumerate(selected):
                if q != pos:
                    union_mask |= masks[idx]
            if union_mask == all_mask:
                selected.pop(pos)
                changed = True
                break
    return selected


def make_solution(selected, candidates, n):
    selected = list(dict.fromkeys(selected))
    hyperplanes = []
    point_assignments = [[] for _ in range(n)]

    for out_idx, candidate_idx in enumerate(selected):
        w, w0, mask = candidates[candidate_idx]
        assigned = []
        bits = mask
        while bits:
            low_bit = bits & -bits
            point_idx = low_bit.bit_length() - 1
            assigned.append(point_idx)
            point_assignments[point_idx].append(out_idx)
            bits ^= low_bit

        hyperplanes.append({
            "w": [float(v) for v in w],
            "w0": float(w0),
            "assigned_points": assigned,
        })

    return {
        "objective_value": float(len(hyperplanes)),
        "hyperplanes": hyperplanes,
        "point_assignments": point_assignments,
    }


def solve_one_dimensional(points, epsilon):
    values = points[:, 0]
    order = np.argsort(values, kind="mergesort")
    planes = []
    assignments = [[] for _ in range(len(values))]
    i = 0

    while i < len(order):
        left_value = float(values[order[i]])
        center = left_value + epsilon
        plane_index = len(planes)
        assigned = []

        j = i
        upper = left_value + 2.0 * epsilon
        tolerance = max(1e-12, abs(upper) * 1e-13)
        while j < len(order) and float(values[order[j]]) <= upper + tolerance:
            point_idx = int(order[j])
            assigned.append(point_idx)
            assignments[point_idx].append(plane_index)
            j += 1

        planes.append({
            "w": [1.0],
            "w0": float(center),
            "assigned_points": assigned,
        })
        i = j

    return {
        "objective_value": float(len(planes)),
        "hyperplanes": planes,
        "point_assignments": assignments,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    n = int(instance["n"])
    d = int(instance["d"])
    epsilon = max(0.0, float(instance["epsilon"]))
    upper_bound_k = int(instance.get("upper_bound_K", n))
    points = np.asarray(instance["points"], dtype=float)

    if n == 0:
        solution = {
            "objective_value": 0.0,
            "hyperplanes": [],
            "point_assignments": [],
        }
        if logger:
            logger.log_solution(0.0, solution)
        write_json(args.solution_path, solution)
        return

    if d == 1:
        solution = solve_one_dimensional(points, epsilon)
        if logger:
            logger.log_solution(solution["objective_value"], solution)
        write_json(args.solution_path, solution)
        return

    all_mask = (1 << n) - 1
    coordinate_scale = max(
        1.0,
        float(np.max(np.abs(points))) if points.size else 1.0
    )
    coverage_tolerance = max(1e-12, 1e-11 * coordinate_scale)

    # Keep the candidate pool compact enough for a single-core set-cover MIP.
    max_candidates = min(
        20000,
        max(1500, int(2_500_000 / max(1, n)))
    )
    max_orientations = min(
        8000,
        max(500, int(1_500_000 / max(1, n)))
    )

    candidates = []
    masks = []
    mask_to_index = {}

    def add_candidate(w, w0):
        w = np.asarray(w, dtype=float)
        norm = float(np.linalg.norm(w))
        if not math.isfinite(norm) or norm <= 1e-14:
            return None

        w = w / norm
        w0 = float(w0) / norm
        distances = np.abs(points @ w - w0)
        covered_indices = np.flatnonzero(
            distances <= epsilon + coverage_tolerance
        )

        if covered_indices.size == 0:
            return None

        mask = 0
        for idx in covered_indices:
            mask |= 1 << int(idx)

        existing = mask_to_index.get(mask)
        if existing is not None:
            return existing

        if len(candidates) >= max_candidates:
            return None

        idx = len(candidates)
        candidates.append((w.copy(), float(w0), mask))
        masks.append(mask)
        mask_to_index[mask] = idx
        return idx

    def fitted_hyperplane(indices):
        subset = points[np.asarray(indices, dtype=int)]
        center = np.mean(subset, axis=0)
        centered = subset - center
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=True)
            w = vh[-1]
        except np.linalg.LinAlgError:
            w = np.zeros(d)
            w[-1] = 1.0

        norm = float(np.linalg.norm(w))
        if norm <= 1e-14:
            w = np.zeros(d)
            w[-1] = 1.0
        else:
            w = w / norm
        return w, float(np.dot(w, center))

    def add_windows_for_orientation(w):
        if len(candidates) >= max_candidates:
            return
        w = np.asarray(w, dtype=float)
        norm = float(np.linalg.norm(w))
        if norm <= 1e-14 or not math.isfinite(norm):
            return
        w = w / norm

        projections = points @ w
        order = np.argsort(projections, kind="mergesort")
        sorted_proj = projections[order]

        width_limit = 2.0 * epsilon + coverage_tolerance
        windows = []
        right = 0
        for left in range(n):
            if right < left:
                right = left
            while (
                right + 1 < n
                and sorted_proj[right + 1] - sorted_proj[left] <= width_limit
            ):
                right += 1

            size = right - left + 1
            if size >= 2:
                center = 0.5 * (
                    float(sorted_proj[left]) + float(sorted_proj[right])
                )
                windows.append((size, left, right, center))

            if right == left:
                right += 1

        if not windows:
            return

        # Preserve large windows, but also retain windows from different
        # locations so parallel hyperplanes remain available.
        max_per_orientation = 100 if n <= 1000 else 50
        if len(windows) > max_per_orientation:
            by_size = sorted(windows, key=lambda x: (-x[0], x[1]))
            keep_count = max_per_orientation * 3 // 4
            chosen = by_size[:keep_count]

            remaining = max_per_orientation - keep_count
            if remaining > 0:
                positions = np.linspace(
                    0, len(windows) - 1, remaining, dtype=int
                )
                chosen.extend(windows[int(pos)] for pos in positions)
            windows = chosen

        seen_centers = set()
        for _, _, _, center in windows:
            key = round(center, 13)
            if key in seen_centers:
                continue
            seen_centers.add(key)
            add_candidate(w, center)
            if len(candidates) >= max_candidates:
                break

    # A guaranteed feasible cover: every set of at most d points lies on some
    # affine hyperplane in R^d.
    order = np.lexsort(tuple(points[:, j] for j in reversed(range(d))))
    baseline_indices = []
    for begin in range(0, n, d):
        group = order[begin:min(n, begin + d)]
        w, w0 = fitted_hyperplane(group)
        idx = add_candidate(w, w0)
        if idx is not None:
            baseline_indices.append(idx)

    # Numerical fallback candidates through individual points.
    for i in range(n):
        if time.monotonic() >= deadline:
            break
        w = np.zeros(d)
        w[i % d] = 1.0
        add_candidate(w, float(np.dot(w, points[i])))

    baseline_indices = prune_selection(
        baseline_indices, masks, all_mask
    )

    # The individual fallback above should not be needed, but use all
    # candidates greedily if an SVD-rounding issue prevented baseline coverage.
    def greedy_cover(candidate_limit=None):
        limit = len(candidates) if candidate_limit is None else candidate_limit
        uncovered = all_mask
        selected = []

        while uncovered:
            best_idx = -1
            best_count = 0
            for idx in range(limit):
                count = (masks[idx] & uncovered).bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = idx
            if best_idx < 0 or best_count == 0:
                return None
            selected.append(best_idx)
            uncovered &= ~masks[best_idx]

        return prune_selection(selected, masks, all_mask)

    if not baseline_indices:
        baseline_indices = greedy_cover()

    baseline_union = 0
    for idx in baseline_indices or []:
        baseline_union |= masks[idx]
    if baseline_union != all_mask:
        baseline_indices = greedy_cover()

    if baseline_indices is None:
        # Last-resort construction, theoretically unreachable.
        baseline_indices = []
        for i in range(n):
            w = np.zeros(d)
            w[0] = 1.0
            idx = add_candidate(w, float(points[i, 0]))
            if idx is not None:
                baseline_indices.append(idx)
        baseline_indices = prune_selection(
            baseline_indices, masks, all_mask
        )

    best_selected = list(baseline_indices)
    best_solution = make_solution(best_selected, candidates, n)
    if logger:
        logger.log_solution(best_solution["objective_value"], best_solution)
    write_json(args.solution_path, best_solution)

    if len(best_selected) <= 1 or time.monotonic() >= deadline:
        return

    generation_deadline = min(
        deadline,
        start_time + max(0.0, args.time_limit * 0.45)
    )
    orientation_count = 0

    def process_orientation(w, direct_w0=None):
        nonlocal orientation_count
        if (
            orientation_count >= max_orientations
            or len(candidates) >= max_candidates
            or time.monotonic() >= generation_deadline
        ):
            return False
        orientation_count += 1
        if direct_w0 is not None:
            add_candidate(w, direct_w0)
        add_windows_for_orientation(w)
        return True

    # Coordinate-aligned slabs.
    for axis in range(d):
        w = np.zeros(d)
        w[axis] = 1.0
        if not process_orientation(w):
            break

    # Global least-squares hyperplane.
    if time.monotonic() < generation_deadline:
        center = np.mean(points, axis=0)
        try:
            _, _, vh = np.linalg.svd(points - center, full_matrices=False)
            global_w = vh[-1]
            process_orientation(global_w, float(np.dot(global_w, center)))
        except np.linalg.LinAlgError:
            pass

    rng = np.random.default_rng(0)

    # Local exact and least-squares hyperplanes.
    local_seed_count = min(n, max(50, int(300000 / max(1, n))))
    if local_seed_count < n:
        local_seeds = rng.choice(n, size=local_seed_count, replace=False)
    else:
        local_seeds = np.arange(n)

    exact_size = min(d, n)
    neighborhood_size = min(n, max(d + 1, 2 * d))

    for seed in local_seeds:
        if time.monotonic() >= generation_deadline:
            break
        delta = points - points[int(seed)]
        dist2 = np.einsum("ij,ij->i", delta, delta)

        nearest_exact = np.argpartition(
            dist2, exact_size - 1
        )[:exact_size]
        w, w0 = fitted_hyperplane(nearest_exact)
        if not process_orientation(w, w0):
            break

        if neighborhood_size > exact_size:
            nearest = np.argpartition(
                dist2, neighborhood_size - 1
            )[:neighborhood_size]
            w, w0 = fitted_hyperplane(nearest)
            if not process_orientation(w, w0):
                break

    # In 2D, pair-derived line orientations are especially effective.
    if d == 2 and time.monotonic() < generation_deadline:
        pair_total = n * (n - 1) // 2
        if pair_total <= max_orientations - orientation_count:
            pair_iterator = itertools.combinations(range(n), 2)
        else:
            sample_count = max(0, max_orientations - orientation_count)

            def random_pairs():
                seen = set()
                attempts = 0
                while len(seen) < sample_count and attempts < sample_count * 10 + 100:
                    a = int(rng.integers(0, n))
                    b = int(rng.integers(0, n - 1))
                    if b >= a:
                        b += 1
                    if a > b:
                        a, b = b, a
                    attempts += 1
                    if (a, b) not in seen:
                        seen.add((a, b))
                        yield a, b

            pair_iterator = random_pairs()

        for a, b in pair_iterator:
            if time.monotonic() >= generation_deadline:
                break
            direction = points[b] - points[a]
            w = np.array([-direction[1], direction[0]], dtype=float)
            norm = float(np.linalg.norm(w))
            if norm <= 1e-14:
                continue
            w /= norm
            w0 = float(np.dot(w, points[a]))
            if not process_orientation(w, w0):
                break

    # General random d-point hyperplanes.
    while (
        n >= d
        and orientation_count < max_orientations
        and len(candidates) < max_candidates
        and time.monotonic() < generation_deadline
    ):
        sample = rng.choice(n, size=d, replace=False)
        w, w0 = fitted_hyperplane(sample)
        if not process_orientation(w, w0):
            break

    greedy_selected = greedy_cover()
    if greedy_selected is not None and len(greedy_selected) < len(best_selected):
        best_selected = greedy_selected
        best_solution = make_solution(best_selected, candidates, n)
        if logger:
            logger.log_solution(best_solution["objective_value"], best_solution)
        write_json(args.solution_path, best_solution)

    if len(best_selected) <= 1 or time.monotonic() >= deadline:
        return

    # Exact set cover over the generated hyperplane pool.
    try:
        import gurobipy as gp
        from gurobipy import GRB

        remaining = deadline - time.monotonic()
        if remaining <= 0.05:
            return

        model = gp.Model("hyperplane_set_cover")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(0.01, remaining)

        y = model.addVars(
            len(candidates), vtype=GRB.BINARY, name="use"
        )

        incidence = [[] for _ in range(n)]
        for j, mask in enumerate(masks):
            bits = mask
            while bits:
                low_bit = bits & -bits
                i = low_bit.bit_length() - 1
                incidence[i].append(j)
                bits ^= low_bit

        for i in range(n):
            model.addConstr(
                gp.quicksum(y[j] for j in incidence[i]) >= 1,
                name=f"cover_{i}",
            )

        model.setObjective(
            gp.quicksum(y[j] for j in range(len(candidates))),
            GRB.MINIMIZE,
        )

        # The current incumbent supplies both a cutoff and a MIP start.
        incumbent_set = set(best_selected)
        for j in range(len(candidates)):
            y[j].Start = 1.0 if j in incumbent_set else 0.0

        if upper_bound_k > 0 and len(best_selected) <= upper_bound_k:
            model.addConstr(
                gp.quicksum(y[j] for j in range(len(candidates)))
                <= upper_bound_k
            )

        callback_state = {
            "best_count": len(best_selected),
            "selected": list(best_selected),
            "solution": best_solution,
        }

        def callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return
            try:
                values = cb_model.cbGetSolution(
                    [y[j] for j in range(len(candidates))]
                )
                selected = [
                    j for j, value in enumerate(values) if value > 0.5
                ]
                selected = prune_selection(selected, masks, all_mask)
                if len(selected) < callback_state["best_count"]:
                    solution = make_solution(selected, candidates, n)
                    callback_state["best_count"] = len(selected)
                    callback_state["selected"] = selected
                    callback_state["solution"] = solution
                    if logger:
                        logger.log_solution(
                            solution["objective_value"], solution
                        )
                    write_json(args.solution_path, solution)
            except Exception:
                # Logging or snapshot failures must not abort the optimization.
                pass

        model.optimize(callback)

        if model.SolCount > 0:
            selected = [
                j for j in range(len(candidates)) if y[j].X > 0.5
            ]
            selected = prune_selection(selected, masks, all_mask)
            if len(selected) < callback_state["best_count"]:
                solution = make_solution(selected, candidates, n)
                callback_state["best_count"] = len(selected)
                callback_state["selected"] = selected
                callback_state["solution"] = solution
                if logger:
                    logger.log_solution(solution["objective_value"], solution)

        if callback_state["best_count"] < len(best_selected):
            best_selected = callback_state["selected"]
            best_solution = callback_state["solution"]

    except Exception:
        # Retain the best heuristic solution if Gurobi is unavailable, its
        # license cannot be initialized, or the remaining time is exhausted.
        pass

    write_json(args.solution_path, best_solution)


if __name__ == "__main__":
    main()