import argparse
import json
import time
from typing import List, Tuple, Dict, Optional

from solution_logger import SolutionLogger


def dominates(a: Tuple[int, ...], b: Tuple[int, ...]) -> bool:
    """Return True if vector a dominates vector b (minimization)."""
    leq_all = True
    strict = False
    for x, y in zip(a, b):
        if x > y:
            leq_all = False
            break
        if x < y:
            strict = True
    return leq_all and strict


def prune_nd_vectors(vectors_with_payload):
    """
    Generic Pareto pruning for minimization.
    Input: list of (vec_tuple, payload)
    Output: non-dominated list of (vec_tuple, payload)
    Keeps one payload per retained vector.
    """
    # Deduplicate exact same vector
    dedup: Dict[Tuple[int, ...], object] = {}
    for vec, payload in vectors_with_payload:
        if vec not in dedup:
            dedup[vec] = payload

    items = sorted(dedup.items(), key=lambda kv: kv[0])
    archive: List[Tuple[Tuple[int, ...], object]] = []

    for vec, payload in items:
        dominated = False
        for a_vec, _ in archive:
            if dominates(a_vec, vec):
                dominated = True
                break
        if dominated:
            continue

        # Remove points dominated by new vec
        new_archive = []
        for a_vec, a_payload in archive:
            if not dominates(vec, a_vec):
                new_archive.append((a_vec, a_payload))
        new_archive.append((vec, payload))
        archive = new_archive

    return archive


def mask_to_selection(mask: int, n: int) -> List[int]:
    return [(mask >> i) & 1 for i in range(n)]


def extract_objective_front(states, m: int):
    """
    states: list of (weight, obj_tuple, mask)
    return non-dominated in objective space only: list of (obj_tuple, mask)
    """
    vec_payload = [(obj, mask) for _, obj, mask in states]
    nd = prune_nd_vectors(vec_payload)
    nd.sort(key=lambda x: x[0])
    return nd


def solve(instance: dict, time_limit: int, logger: Optional[SolutionLogger] = None):
    n = int(instance["num_items"])
    m = int(instance["num_objectives"])
    weights = list(map(int, instance["weights"]))
    capacity = int(instance["capacity"])
    coeffs = [list(map(int, row)) for row in instance["objective_coefficients"]]

    start = time.time()
    deadline = start + max(1, int(time_limit))

    def time_exceeded() -> bool:
        return time.time() >= deadline

    # Fast exact shortcut: if all objective coefficients are nonnegative,
    # empty selection achieves (0,...,0) and dominates all others.
    all_nonnegative = all(c >= 0 for row in coeffs for c in row)
    if all_nonnegative:
        result = {
            "objective_value": 1,
            "non_dominated_objective_vectors": [[0 for _ in range(m)]],
            "solutions": [[0 for _ in range(n)]],
        }
        if logger:
            logger.log(1)
        return result

    # General DP with Pareto pruning in (weight + objectives) space.
    # state = (weight, obj_tuple, mask)
    zero_obj = tuple(0 for _ in range(m))
    states: List[Tuple[int, Tuple[int, ...], int]] = [(0, zero_obj, 0)]

    # Incumbent output (for timeout fallback)
    nd_obj = extract_objective_front(states, m)
    best_count = len(nd_obj)
    best_vectors = [list(obj) for obj, _ in nd_obj]
    best_solutions = [mask_to_selection(mask, n) for _, mask in nd_obj]
    if logger:
        logger.log(best_count)

    processed_all_items = True

    for i in range(n):
        if time_exceeded():
            processed_all_items = False
            break

        wi = weights[i]
        item_obj = [coeffs[j][i] for j in range(m)]

        candidates = list(states)  # exclude item i

        # include item i
        for w, obj, mask in states:
            if time_exceeded():
                processed_all_items = False
                break
            nw = w + wi
            if nw <= capacity:
                nobj = tuple(obj[j] + item_obj[j] for j in range(m))
                nmask = mask | (1 << i)
                candidates.append((nw, nobj, nmask))

        # prune in (weight + objectives)
        vec_payload = [((w,) + obj, (w, obj, mask)) for (w, obj, mask) in candidates]
        nd_combined = prune_nd_vectors(vec_payload)
        states = [payload for _, payload in nd_combined]

        # incumbent update from current feasible explored states
        curr_nd = extract_objective_front(states, m)
        curr_count = len(curr_nd)
        if curr_count > best_count:
            best_count = curr_count
            best_vectors = [list(obj) for obj, _ in curr_nd]
            best_solutions = [mask_to_selection(mask, n) for _, mask in curr_nd]
            if logger:
                logger.log(best_count)

    # Final extraction
    if processed_all_items:
        final_nd = extract_objective_front(states, m)
        final_count = len(final_nd)
        if final_count >= best_count:
            best_count = final_count
            best_vectors = [list(obj) for obj, _ in final_nd]
            best_solutions = [mask_to_selection(mask, n) for _, mask in final_nd]
            if logger and final_count > 0:
                # log if strictly better than previously logged value
                # (logger handles monotonic checks internally if needed)
                logger.log(best_count)

    return {
        "objective_value": int(best_count),
        "non_dominated_objective_vectors": best_vectors,
        "solutions": best_solutions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    solution = solve(instance, args.time_limit, logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()