import argparse
import json
import math
import time
from typing import List, Tuple, Optional

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def canonicalize_plane(w: np.ndarray, w0: float) -> Tuple[np.ndarray, float]:
    norm = np.linalg.norm(w)
    if norm <= 1e-12:
        return None, None
    w = w / norm
    w0 = float(w0) / norm

    # Canonical sign to reduce duplicates
    idx = int(np.argmax(np.abs(w)))
    if w[idx] < 0:
        w = -w
        w0 = -w0
    return w, w0


def fit_plane_pca(points_subset: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    if points_subset.shape[0] == 0:
        return None
    c = points_subset.mean(axis=0)
    A = points_subset - c
    if A.shape[0] == 1:
        # Any unit normal is fine; choose axis 0
        w = np.zeros(points_subset.shape[1], dtype=float)
        w[0] = 1.0
        w0 = -float(np.dot(w, c))
        return canonicalize_plane(w, w0)
    try:
        _, _, vt = np.linalg.svd(A, full_matrices=False)
        w = vt[-1]
        w0 = -float(np.dot(w, c))
        return canonicalize_plane(w, w0)
    except np.linalg.LinAlgError:
        return None


def plane_through_points(sample: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    # sample shape: (m, d), ideally m>=1
    m, d = sample.shape
    if m == 0:
        return None
    if d == 1:
        w = np.array([1.0], dtype=float)
        w0 = -float(sample[0, 0])
        return canonicalize_plane(w, w0)
    if m == 1:
        w = np.zeros(d, dtype=float)
        w[0] = 1.0
        w0 = -float(np.dot(w, sample[0]))
        return canonicalize_plane(w, w0)

    p0 = sample[0]
    D = sample[1:] - p0
    try:
        _, s, vt = np.linalg.svd(D, full_matrices=True)
        # Need at least one-dimensional null space
        rank = int(np.sum(s > 1e-10))
        if rank >= d:
            return None
        w = vt[-1]
        w0 = -float(np.dot(w, p0))
        return canonicalize_plane(w, w0)
    except np.linalg.LinAlgError:
        return None


def solve_instance(data: dict, time_limit: int, logger: Optional[SolutionLogger]):
    t0 = time.time()

    def remaining():
        return max(0.0, time_limit - (time.time() - t0))

    n = int(data["n"])
    d = int(data["d"])
    epsilon = float(data["epsilon"])
    upper_bound_K = int(data.get("upper_bound_K", n))
    points = np.array(data["points"], dtype=float)

    # --- Candidate generation ---
    candidates_w = []
    candidates_b = []
    seen = set()

    def add_candidate(w: np.ndarray, b: float):
        wb = canonicalize_plane(w, b)
        if wb[0] is None:
            return
        ww, bb = wb
        key = tuple(np.round(np.concatenate([ww, [bb]]), 7))
        if key in seen:
            return
        seen.add(key)
        candidates_w.append(ww)
        candidates_b.append(float(bb))

    # 1) Singleton-guarantee candidates: one hyperplane through each point
    # Use axis with largest variance
    var = np.var(points, axis=0) if n > 0 else np.ones(d)
    axis = int(np.argmax(var)) if d > 0 else 0
    e = np.zeros(d, dtype=float)
    if d > 0:
        e[axis] = 1.0
    for i in range(n):
        add_candidate(e, -float(np.dot(e, points[i])))

    # 2) Global PCA candidate
    if remaining() > 0.1:
        fp = fit_plane_pca(points)
        if fp is not None:
            add_candidate(fp[0], fp[1])

    # 3) Random local PCA neighborhoods
    rng = np.random.default_rng(0)
    max_local = min(800, 8 * n + 50)
    neighborhood_size = min(max(d + 2, 5), n)
    for _ in range(max_local):
        if remaining() < 0.5:
            break
        cidx = int(rng.integers(0, n))
        diff = points - points[cidx]
        dist2 = np.einsum("ij,ij->i", diff, diff)
        nn = np.argpartition(dist2, neighborhood_size - 1)[:neighborhood_size]
        fp = fit_plane_pca(points[nn])
        if fp is not None:
            add_candidate(fp[0], fp[1])

    # 4) RANSAC-like exact planes from random samples
    sample_size = min(d, n) if n > 0 else 0
    max_ransac = min(1200, 10 * n + 100)
    if sample_size >= 1:
        for _ in range(max_ransac):
            if remaining() < 0.5:
                break
            idx = rng.choice(n, size=sample_size, replace=False)
            pp = plane_through_points(points[idx])
            if pp is not None:
                add_candidate(pp[0], pp[1])

    m = len(candidates_w)
    W = np.vstack(candidates_w) if m > 0 else np.zeros((0, d))
    b = np.array(candidates_b, dtype=float) if m > 0 else np.zeros(0)

    # Coverage matrix
    if m == 0:
        # extreme fallback
        W = np.zeros((n, d), dtype=float)
        b = np.zeros(n, dtype=float)
        for i in range(n):
            if d > 0:
                W[i, 0] = 1.0
            b[i] = -float(points[i, 0]) if d > 0 else 0.0
        m = n

    resid = np.abs(points @ W.T + b.reshape(1, -1))
    covers = resid <= (epsilon + 1e-9)

    # Ensure each point has at least one covering candidate
    for i in range(n):
        if not np.any(covers[i]):
            # add exact singleton through point
            ww = np.zeros(d, dtype=float)
            if d > 0:
                ww[0] = 1.0
            bb = -float(np.dot(ww, points[i]))
            add_candidate(ww, bb)

    # Rebuild if expanded
    if len(candidates_w) != m:
        m = len(candidates_w)
        W = np.vstack(candidates_w)
        b = np.array(candidates_b, dtype=float)
        resid = np.abs(points @ W.T + b.reshape(1, -1))
        covers = resid <= (epsilon + 1e-9)

    point_to_cands = [np.where(covers[i])[0].tolist() for i in range(n)]

    # --- Greedy initial feasible solution ---
    uncovered = set(range(n))
    selected_greedy = []
    available = set(range(m))

    while uncovered and available:
        if remaining() < 0.2:
            break
        best_k = None
        best_gain = -1
        # sample if huge
        scan_list = list(available)
        if len(scan_list) > 2000:
            scan_list = rng.choice(scan_list, size=2000, replace=False).tolist()

        for k in scan_list:
            cov_idx = np.where(covers[:, k])[0]
            gain = sum((i in uncovered) for i in cov_idx)
            if gain > best_gain:
                best_gain = gain
                best_k = k
        if best_k is None or best_gain <= 0:
            break
        selected_greedy.append(best_k)
        covered_now = np.where(covers[:, best_k])[0]
        for i in covered_now:
            if i in uncovered:
                uncovered.remove(int(i))
        if best_k in available:
            available.remove(best_k)

    # fallback complete with singleton-ish candidates that cover each remaining point
    if uncovered:
        for i in list(uncovered):
            cands_i = point_to_cands[i]
            if cands_i:
                selected_greedy.append(cands_i[0])
                covered_now = np.where(covers[:, cands_i[0]])[0]
                for j in covered_now:
                    uncovered.discard(int(j))

    selected_greedy = sorted(set(selected_greedy))
    best_selected = selected_greedy[:]
    best_obj = float(len(best_selected))
    if logger:
        logger.log(best_obj)

    # --- Set cover MIP improvement ---
    if remaining() > 0.5:
        model = gp.Model("hyperplane_cover")
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(0.1, remaining())
        model.Params.MIPFocus = 1

        z = model.addVars(m, vtype=GRB.BINARY, name="z")

        for i in range(n):
            cands = point_to_cands[i]
            if not cands:
                # should not happen
                continue
            model.addConstr(gp.quicksum(z[k] for k in cands) >= 1, name=f"cover_{i}")

        model.setObjective(gp.quicksum(z[k] for k in range(m)), GRB.MINIMIZE)

        # warm start
        for k in range(m):
            z[k].Start = 1.0 if k in best_selected else 0.0

        # callback logging incumbents
        model._logger = logger
        model._best_cb = best_obj

        def cb(mdl, where):
            if where == GRB.Callback.MIPSOL:
                obj = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj + 1e-9 < mdl._best_cb:
                    mdl._best_cb = obj
                    if mdl._logger:
                        mdl._logger.log(float(obj))

        model.optimize(cb)

        if model.SolCount > 0:
            sel = [k for k in range(m) if z[k].X > 0.5]
            if len(sel) < len(best_selected):
                best_selected = sel
                best_obj = float(len(sel))

    # Respect upper bound K softly (if needed, keep feasibility first)
    if len(best_selected) > upper_bound_K:
        # Try truncation with greedy repair under K (best effort)
        # Since problem instances should be feasible under K, this is a safeguard.
        sel = best_selected[:]
        # sort by coverage descending
        sel.sort(key=lambda k: np.sum(covers[:, k]), reverse=True)
        sel = sel[:upper_bound_K]
        # Check coverage; if not full, keep original feasible solution
        covered_mask = np.any(covers[:, sel], axis=1) if sel else np.zeros(n, dtype=bool)
        if np.all(covered_mask):
            best_selected = sel
            best_obj = float(len(sel))

    # --- Build assignments ---
    selected = best_selected
    if not selected:
        # impossible fallback
        selected = [point_to_cands[i][0] for i in range(n) if point_to_cands[i]]
        selected = sorted(set(selected))

    sel_resid = resid[:, selected]
    sel_covers = covers[:, selected]

    point_assignments_local = [[] for _ in range(n)]
    assigned_points_per_local = [[] for _ in range(len(selected))]

    for i in range(n):
        feasible_js = np.where(sel_covers[i])[0]
        if feasible_js.size > 0:
            # choose closest among feasible
            jloc = feasible_js[np.argmin(sel_resid[i, feasible_js])]
        else:
            # repair: choose closest and still assign (should be rare)
            jloc = int(np.argmin(sel_resid[i]))
        point_assignments_local[i] = [int(jloc)]
        assigned_points_per_local[jloc].append(i)

    # Remove empty hyperplanes and remap
    keep = [j for j in range(len(selected)) if len(assigned_points_per_local[j]) > 0]
    old_to_new = {old: new for new, old in enumerate(keep)}

    hyperplanes = []
    for old_j in keep:
        k = selected[old_j]
        hyperplanes.append({
            "w": [float(x) for x in W[k].tolist()],
            "w0": float(b[k]),
            "assigned_points": [int(i) for i in assigned_points_per_local[old_j]]
        })

    point_assignments = []
    for i in range(n):
        old_j = point_assignments_local[i][0]
        if old_j in old_to_new:
            point_assignments.append([int(old_to_new[old_j])])
        else:
            # Should not happen; fallback to first hyperplane if exists
            point_assignments.append([0] if hyperplanes else [])

    objective_value = float(len(hyperplanes))

    return {
        "objective_value": objective_value,
        "hyperplanes": hyperplanes,
        "point_assignments": point_assignments
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    sol = solve_instance(data, args.time_limit, logger)

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()