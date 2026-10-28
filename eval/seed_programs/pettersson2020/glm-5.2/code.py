import argparse
import json
import time

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def obj_dominates(a, b):
    """True if objective vector a dominates b (a <= b and a < b in at least one component)."""
    strict = False
    for x, y in zip(a, b):
        if x > y:
            return False
        if x < y:
            strict = True
    return strict


def dominates(a, b):
    """
    Partial-state dominance for the DP.
    State is (weight, objective_tuple).
    a dominates b if a has no greater weight, no greater objective component,
    and is strictly better in weight or at least one objective component.
    """
    wa, oa = a
    wb, ob = b
    if wa > wb:
        return False
    strict = wa < wb
    for x, y in zip(oa, ob):
        if x > y:
            return False
        if not strict and x < y:
            strict = True
    return strict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    n = int(data.get("num_items", 0))
    k = int(data.get("num_objectives", 0))
    weights = [int(x) for x in data.get("weights", [])]
    capacity = int(data.get("capacity", sum(weights) // 2 if weights else 0))
    objective_coefficients = data.get("objective_coefficients", [])

    # Be robust to small schema mismatches.
    if len(weights) != n:
        n = len(weights)

    obj_coeffs = [[int(x) for x in row] for row in objective_coefficients]
    if len(obj_coeffs) != k:
        k = len(obj_coeffs)

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="maximize")

    start_time = time.time()
    margin = 0.4
    deadline = start_time + max(args.time_limit - margin, 0.1)

    # ------------------------------------------------------------------
    # Important structural observation:
    # If every objective coefficient is nonnegative, the empty selection
    # has objective vector (0,...,0) and weight 0.  Every non-empty
    # feasible selection has each objective >= 0, and if it differs from
    # the zero vector then the empty vector strictly dominates it.
    # Therefore the unique non-dominated objective vector is the zero
    # vector, produced by selecting no items.
    # ------------------------------------------------------------------
    all_nonneg = all(val >= 0 for row in obj_coeffs for val in row)
    if all_nonneg:
        result = {
            "objective_value": 1,
            "non_dominated_objective_vectors": [[0] * k],
            "solutions": [[0] * n],
        }
        if logger:
            logger.log_solution(result["objective_value"], result)
        with open(args.solution_path, "w") as f:
            json.dump(result, f)
        return

    # ------------------------------------------------------------------
    # General exact DP for possibly negative coefficients.
    # Maintains non-dominated partial states (weight, objective_vector).
    # A unique integer ID is used for each accepted state so that parent
    # reconstruction remains correct even if a state key is removed and
    # later recreated by a different item subset.
    # ------------------------------------------------------------------
    zero_obj = tuple([0] * k)
    states = { (0, zero_obj): 0 }   # key -> state_id
    parent_info = { 0: None }      # state_id -> (parent_id, item_index) or None
    next_id = 1

    def insert(key, parent_id, item_index):
        nonlocal next_id
        if key in states:
            return

        # Skip if an existing state dominates this candidate.
        for skey in states:
            if dominates(skey, key):
                return

        # Remove existing states dominated by this candidate.
        to_remove = []
        for skey in states:
            if dominates(key, skey):
                to_remove.append(skey)
        for skey in to_remove:
            del states[skey]

        new_id = next_id
        next_id += 1
        states[key] = new_id
        parent_info[new_id] = (parent_id, item_index)

    timed_out = False
    for i in range(n):
        if time.time() > deadline:
            timed_out = True
            break

        w = weights[i]
        if w > capacity:
            continue

        c = tuple(obj_coeffs[j][i] for j in range(k))
        old_keys = list(states.keys())

        for key in old_keys:
            if key not in states:
                continue
            if time.time() > deadline:
                timed_out = True
                break

            parent_id = states[key]
            wt, obj = key
            nwt = wt + w
            if nwt > capacity:
                continue

            nobj = tuple(obj[j] + c[j] for j in range(k))
            nkey = (nwt, nobj)
            if nkey in states:
                continue

            insert(nkey, parent_id, i)

        if timed_out:
            break

    # ------------------------------------------------------------------
    # Final filtering: collapse duplicate objective vectors and keep only
    # objective-nondominated vectors among all surviving feasible states.
    # ------------------------------------------------------------------
    obj_to_state = {}
    for key, sid in states.items():
        wt, obj = key
        if obj not in obj_to_state or wt < obj_to_state[obj][0]:
            obj_to_state[obj] = (wt, sid)

    items = list(obj_to_state.items())
    nondom = []
    for obj, (wt, sid) in items:
        dominated = False
        for obj2, _ in items:
            if obj2 is obj:
                continue
            if obj_dominates(obj2, obj):
                dominated = True
                break
        if not dominated:
            nondom.append((obj, sid))

    nondom.sort(key=lambda x: x[0])

    def reconstruct(sid):
        sol = [0] * n
        cur = sid
        while True:
            p = parent_info.get(cur)
            if p is None:
                break
            prev_id, item_index = p
            sol[item_index] = 1
            cur = prev_id
        return sol

    result = {
        "objective_value": len(nondom),
        "non_dominated_objective_vectors": [list(obj) for obj, _ in nondom],
        "solutions": [reconstruct(sid) for _, sid in nondom],
    }

    if logger:
        logger.log_solution(result["objective_value"], result)

    with open(args.solution_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()