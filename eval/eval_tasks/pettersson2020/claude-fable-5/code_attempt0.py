import argparse
import json
import time
import sys

from solution_logger import SolutionLogger


def pareto_merge(old, new, deadline):
    """Merge two lists of (key_tuple, mask) states and keep only states whose
    key_tuple (weight followed by objective values, all to be minimized) is
    non-dominated. Uses lexicographic sort so a single forward pass suffices:
    a later element can never strictly dominate an earlier one."""
    combined = old + new
    combined.sort(key=lambda s: s[0])
    kept = []
    seen = set()
    check_every = 2048
    cnt = 0
    for key, mask in combined:
        cnt += 1
        if cnt % check_every == 0 and time.time() > deadline:
            # Out of time: keep what we have (still all feasible states).
            break
        if key in seen:
            continue
        dominated = False
        for kkey, _km in kept:
            ok = True
            for a, b in zip(kkey, key):
                if a > b:
                    ok = False
                    break
            if ok:
                dominated = True
                break
        if not dominated:
            kept.append((key, mask))
            seen.add(key)
    return kept


def objectives_front(states):
    """Given DP states (key_tuple=(weight,*objs), mask), return the set of
    non-dominated objective vectors (weight ignored, only feasibility mattered)
    together with one representative selection mask per vector."""
    pts = [(key[1:], mask) for key, mask in states]
    pts.sort(key=lambda s: s[0])
    kept = []
    seen = set()
    for obj, mask in pts:
        if obj in seen:
            continue
        dominated = False
        for kobj, _km in kept:
            ok = True
            for a, b in zip(kobj, obj):
                if a > b:
                    ok = False
                    break
            if ok:
                dominated = True
                break
        if not dominated:
            kept.append((obj, mask))
            seen.add(obj)
    return kept


def build_output(front, n):
    vectors = [list(obj) for obj, _ in front]
    sols = [[(mask >> i) & 1 for i in range(n)] for _, mask in front]
    return {
        "objective_value": len(front),
        "non_dominated_objective_vectors": vectors,
        "solutions": sols,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start = time.time()
    deadline = start + max(1, args.time_limit) - 1.0

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    n = int(data["num_items"])
    m = int(data["num_objectives"])
    weights = [int(w) for w in data["weights"]]
    cap = int(data["capacity"])
    coeffs = [[int(c) for c in row] for row in data["objective_coefficients"]]

    # Initial trivial feasible incumbent: the empty selection (all-zero vector).
    zero_key = (0,) + tuple([0] * m)
    states = [(zero_key, 0)]
    init_front = objectives_front(states)
    best_output = build_output(init_front, n)
    if logger:
        logger.log_solution(best_output["objective_value"], best_output)

    # Dynamic programming over items (Nemhauser-Ullmann style), maintaining the
    # Pareto set in (weight, objectives) space. Weight is treated as an extra
    # minimized pseudo-objective during the DP because lower weight is always at
    # least as good for feasibility; the final front is filtered on objectives
    # only.
    interrupted = False
    for i in range(n):
        if time.time() > deadline:
            interrupted = True
            break
        w = weights[i]
        delta = tuple(coeffs[k][i] for k in range(m))
        bit = 1 << i
        new_states = []
        for key, mask in states:
            nw = key[0] + w
            if nw > cap:
                continue
            nkey = (nw,) + tuple(a + b for a, b in zip(key[1:], delta))
            new_states.append((nkey, mask | bit))
        if new_states:
            states = pareto_merge(states, new_states, deadline)

        # Periodically refresh the incumbent (cheap when the state set is small).
        if len(states) <= 20000 and time.time() <= deadline:
            front = objectives_front(states)
            if len(front) > best_output["objective_value"]:
                best_output = build_output(front, n)
                if logger:
                    logger.log_solution(best_output["objective_value"], best_output)

    # Final filtering: non-dominated objective vectors among all surviving states.
    front = objectives_front(states)
    output = build_output(front, n)
    if output["objective_value"] >= best_output["objective_value"]:
        best_output = output
        if logger:
            logger.log_solution(best_output["objective_value"], best_output)

    with open(args.solution_path, "w") as f:
        json.dump(best_output, f)

    if interrupted:
        sys.stderr.write("Time limit reached; best front found so far was written.\n")


if __name__ == "__main__":
    main()