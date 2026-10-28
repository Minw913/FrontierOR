import json
import argparse
import numpy as np
import time
from solution_logger import SolutionLogger


def generate_candidates(points, epsilon, time_budget, max_total_candidates=20000):
    """Generate candidate hyperplanes with their coverage sets using slab approach."""
    n, d = points.shape
    two_eps = 2.0 * epsilon
    candidates = []  # list of (w, w0, covered_list)
    seen_sets = set()
    start = time.time()

    def add_candidates_for_direction(w):
        """Generate maximal slab candidates for direction w."""
        projections = points @ w
        sorted_idx = np.argsort(projections)
        sorted_proj = projections[sorted_idx]

        left = 0
        for right in range(n):
            while sorted_proj[right] - sorted_proj[left] > two_eps + 1e-10:
                left += 1
            # Maximal window: can't extend right further
            if right == n - 1 or sorted_proj[right + 1] - sorted_proj[left] > two_eps + 1e-10:
                covered = sorted_idx[left:right + 1].tolist()
                covered_key = frozenset(covered)
                if covered_key not in seen_sets:
                    seen_sets.add(covered_key)
                    mid = (sorted_proj[left] + sorted_proj[right]) / 2.0
                    candidates.append((w.copy(), -mid, covered))
                    if len(candidates) >= max_total_candidates:
                        return True
        return False

    # 1. Coordinate directions
    for coord in range(d):
        w = np.zeros(d)
        w[coord] = 1.0
        if add_candidates_for_direction(w):
            return candidates

    # 2. PCA directions
    if n > 1:
        centered = points - points.mean(axis=0)
        try:
            _, S, Vt = np.linalg.svd(centered, full_matrices=False)
            for idx in range(min(d, Vt.shape[0])):
                if time.time() - start > time_budget:
                    break
                w = Vt[idx].copy()
                if add_candidates_for_direction(w):
                    return candidates
        except Exception:
            pass

    # 3. Pairwise directions
    if n >= 2 and time.time() - start < time_budget:
        total_pairs = n * (n - 1) // 2
        max_pairs = min(total_pairs, 5000)

        if total_pairs <= max_pairs:
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        else:
            rng = np.random.RandomState(42)
            pairs_set = set()
            while len(pairs_set) < max_pairs:
                idx = rng.choice(n, 2, replace=False)
                pairs_set.add((min(idx[0], idx[1]), max(idx[0], idx[1])))
            pairs = list(pairs_set)

        for i, j in pairs:
            if time.time() - start > time_budget:
                break
            if len(candidates) >= max_total_candidates:
                break
            diff = points[i] - points[j]
            norm = np.linalg.norm(diff)
            if norm < 1e-10:
                continue
            w = diff / norm
            if add_candidates_for_direction(w):
                return candidates

    # 4. Random directions
    if time.time() - start < time_budget:
        rng = np.random.RandomState(123)
        n_random = min(300, max(50, d * 15))
        for _ in range(n_random):
            if time.time() - start > time_budget:
                break
            if len(candidates) >= max_total_candidates:
                break
            w = rng.randn(d)
            norm = np.linalg.norm(w)
            if norm < 1e-10:
                continue
            w /= norm
            if add_candidates_for_direction(w):
                return candidates

    # 5. Ensure every point is covered by at least one candidate
    covered_points = set()
    for _, _, covered in candidates:
        covered_points.update(covered)

    for i in range(n):
        if i not in covered_points:
            w = np.zeros(d)
            w[0] = 1.0
            w0 = -points[i][0]
            candidates.append((w, w0, [i]))

    return candidates


def solve_set_cover_gurobi(n, candidates, time_limit):
    """Solve set cover using Gurobi."""
    try:
        import gurobipy as gp
        from gurobipy import GRB

        m = gp.Model("set_cover")
        m.setParam('Seed', 0)
        m.setParam('MIPGap', 1e-4)
        m.setParam('NumericFocus', 0)
        m.setParam('Threads', 1)
        m.setParam('TimeLimit', max(time_limit, 1.0))
        m.setParam('OutputFlag', 0)

        num_cand = len(candidates)
        y = m.addVars(num_cand, vtype=GRB.BINARY, name="y")

        m.setObjective(gp.quicksum(y[j] for j in range(num_cand)), GRB.MINIMIZE)

        # Coverage constraints
        point_to_cands = [[] for _ in range(n)]
        for j, (_, _, covered) in enumerate(candidates):
            for p in covered:
                point_to_cands[p].append(j)

        for i in range(n):
            if point_to_cands[i]:
                m.addConstr(
                    gp.quicksum(y[j] for j in point_to_cands[i]) >= 1,
                    name=f"cover_{i}"
                )

        m.optimize()

        if m.SolCount > 0:
            selected = [j for j in range(num_cand) if y[j].X > 0.5]
            return selected
        return None
    except Exception:
        return None


def greedy_set_cover(n, candidates):
    """Greedy set cover algorithm."""
    uncovered = set(range(n))
    selected = []
    coverage_sets = [set(covered) for _, _, covered in candidates]

    while uncovered:
        best_j = -1
        best_count = 0

        for j in range(len(candidates)):
            count = len(coverage_sets[j] & uncovered)
            if count > best_count:
                best_count = count
                best_j = j

        if best_j == -1 or best_count == 0:
            break

        selected.append(best_j)
        uncovered -= coverage_sets[best_j]

    return selected


def build_solution(points, candidates, selected, d):
    """Build the solution dictionary from selected candidates."""
    hyperplanes = []
    n = points.shape[0]
    point_assignments = [[] for _ in range(n)]

    for idx, j in enumerate(selected):
        w, w0, covered = candidates[j]
        w = np.array(w, dtype=float)
        w_norm = np.linalg.norm(w)
        if w_norm > 1e-10:
            w = w / w_norm
            w0 = w0 / w_norm

        hyperplanes.append({
            'w': w.tolist(),
            'w0': float(w0),
            'assigned_points': sorted(covered)
        })

        for p in covered:
            point_assignments[p].append(idx)

    # Handle any uncovered points (safety net)
    for i in range(n):
        if not point_assignments[i]:
            w = np.zeros(d)
            w[0] = 1.0
            w0 = -points[i][0]
            h_idx = len(hyperplanes)
            hyperplanes.append({
                'w': w.tolist(),
                'w0': float(w0),
                'assigned_points': [i]
            })
            point_assignments[i].append(h_idx)

    return {
        "objective_value": len(hyperplanes),
        "hyperplanes": hyperplanes,
        "point_assignments": point_assignments
    }


def verify_and_fix(points, solution, epsilon, d):
    """Verify all points are covered and fix any issues."""
    n = points.shape[0]
    hyperplanes = solution["hyperplanes"]
    point_assignments = solution["point_assignments"]

    for i in range(n):
        covered = False
        for h_idx in point_assignments[i]:
            if h_idx < len(hyperplanes):
                w = np.array(hyperplanes[h_idx]['w'])
                w0 = hyperplanes[h_idx]['w0']
                dist = abs(np.dot(w, points[i]) + w0)
                if dist <= epsilon + 1e-6:
                    covered = True
                    break
        if not covered:
            # Add a new hyperplane for this point
            w = np.zeros(d)
            w[0] = 1.0
            w0 = -points[i][0]
            h_idx = len(hyperplanes)
            hyperplanes.append({
                'w': w.tolist(),
                'w0': float(w0),
                'assigned_points': [i]
            })
            point_assignments[i].append(h_idx)

    solution["objective_value"] = len(hyperplanes)
    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    d = instance['d']
    epsilon = instance['epsilon']
    points = np.array(instance['points'], dtype=float)

    start_time = time.time()
    time_limit = args.time_limit

    # Phase 1: Generate candidate hyperplanes
    gen_budget = min(time_limit * 0.35, 60.0)
    max_cands = 20000 if n <= 200 else 8000
    candidates = generate_candidates(points, epsilon, gen_budget, max_cands)

    # Phase 2: Quick greedy solution (always produces a feasible solution)
    greedy_selected = greedy_set_cover(n, candidates)
    best_solution = build_solution(points, candidates, greedy_selected, d)
    best_solution = verify_and_fix(points, best_solution, epsilon, d)

    if logger:
        logger.log_solution(best_solution["objective_value"], best_solution)

    # Phase 3: Try Gurobi for improved solution
    remaining_time = time_limit - (time.time() - start_time) - 2.0

    if remaining_time > 3.0 and len(candidates) <= 50000:
        gurobi_selected = solve_set_cover_gurobi(n, candidates, remaining_time)
        if gurobi_selected is not None:
            gurobi_solution = build_solution(points, candidates, gurobi_selected, d)
            gurobi_solution = verify_and_fix(points, gurobi_solution, epsilon, d)
            if gurobi_solution["objective_value"] < best_solution["objective_value"]:
                best_solution = gurobi_solution
                if logger:
                    logger.log_solution(best_solution["objective_value"], best_solution)

    # Write final solution
    with open(args.solution_path, 'w') as f:
        json.dump(best_solution, f)


if __name__ == '__main__':
    main()