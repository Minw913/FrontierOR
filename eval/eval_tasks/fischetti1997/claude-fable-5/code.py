import argparse
import json
import time
import random
import math
import numpy as np

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def load_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    n = data["problem_size"]["n"]
    m = data["problem_size"]["m"]
    nodes = data["nodes"]
    clusters = data["clusters"]
    vals = np.array(data["distance_matrix_upper_triangular"]["values"], dtype=np.float64)
    D = np.zeros((n, n), dtype=np.float64)
    iu = np.triu_indices(n, 1)
    D[iu] = vals
    D = D + D.T
    return n, m, nodes, clusters, D


def tour_cost(tour, D):
    m = len(tour)
    if m <= 1:
        return 0.0
    t = np.asarray(tour)
    return float(D[t, np.roll(t, -1)].sum())


def two_opt_pass(arr, D, deadline):
    """One sweep of best-improvement 2-opt on node array (in place). Returns True if improved."""
    m = len(arr)
    improved = False
    for i in range(m - 1):
        if time.time() > deadline:
            return improved
        a = arr[i]
        b = arr[i + 1]
        hi = m - 1 if i == 0 else m
        if i + 2 >= hi:
            continue
        js = np.arange(i + 2, hi)
        c = arr[js]
        d2 = arr[(js + 1) % m]
        gain = D[a, b] + D[c, d2] - D[a, c] - D[b, d2]
        k = int(np.argmax(gain))
        if gain[k] > 1e-9:
            j = int(js[k])
            arr[i + 1:j + 1] = arr[i + 1:j + 1][::-1]
            improved = True
    return improved


def or_opt_pass(arr, D, deadline):
    """Relocate single nodes to best position. Returns (arr, improved)."""
    m = len(arr)
    improved = False
    p = 0
    while p < m:
        if time.time() > deadline:
            return arr, improved
        v = arr[p]
        a = arr[p - 1]
        b = arr[(p + 1) % m]
        rem_gain = D[a, v] + D[v, b] - D[a, b]
        if rem_gain > 1e-9:
            nxt = np.roll(arr, -1)
            ins = D[arr, v] + D[v, nxt] - D[arr, nxt]
            ins[p] = np.inf
            ins[(p - 1) % m] = np.inf
            q = int(np.argmin(ins))
            if rem_gain - ins[q] > 1e-9:
                tmp = np.delete(arr, p)
                # position of node arr[q] in tmp
                if q < p:
                    pos = q
                else:
                    pos = q - 1
                arr = np.insert(tmp, pos + 1, v)
                improved = True
                continue
        p += 1
    return arr, improved


def reselect_pass(arr, D, cluster_of, cluster_nodes, deadline):
    """For each position, choose best node from its cluster given neighbors. In place."""
    m = len(arr)
    improved = False
    for p in range(m):
        if time.time() > deadline:
            return improved
        v = arr[p]
        a = arr[p - 1]
        b = arr[(p + 1) % m]
        cand = cluster_nodes[cluster_of[v]]
        if len(cand) == 1:
            continue
        costs = D[a, cand] + D[cand, b]
        k = int(np.argmin(costs))
        if costs[k] + 1e-9 < D[a, v] + D[v, b]:
            arr[p] = cand[k]
            improved = True
    return improved


def dp_optimize(arr, D, cluster_of, cluster_nodes, deadline):
    """Given cluster order implied by arr, find optimal node selection via DP.
    Returns (new_arr, cost) or (arr, None) if timed out."""
    m = len(arr)
    order = [cluster_of[v] for v in arr]
    sizes = [len(cluster_nodes[c]) for c in order]
    k = int(np.argmin(sizes))
    order = order[k:] + order[:k]
    c0 = cluster_nodes[order[0]]
    best_cost = math.inf
    best_path = None
    for s in c0:
        if time.time() > deadline:
            break
        prev_nodes = np.array([s])
        prev_cost = np.array([0.0])
        back = []
        for c in order[1:]:
            cn = cluster_nodes[c]
            costm = prev_cost[:, None] + D[np.ix_(prev_nodes, cn)]
            bp = costm.argmin(axis=0)
            prev_cost = costm[bp, np.arange(len(cn))]
            back.append((bp, cn))
            prev_nodes = cn
        total = prev_cost + D[prev_nodes, s]
        j = int(np.argmin(total))
        if total[j] < best_cost:
            best_cost = float(total[j])
            path = []
            idx = j
            for bp, cn in reversed(back):
                path.append(int(cn[idx]))
                idx = int(bp[idx])
            path.append(int(s))
            path.reverse()
            best_path = path
    if best_path is None:
        return arr, None
    return np.array(best_path, dtype=arr.dtype), best_cost


def local_search(arr, D, cluster_of, cluster_nodes, deadline, use_dp=True):
    while time.time() < deadline:
        imp = False
        if two_opt_pass(arr, D, deadline):
            imp = True
        if reselect_pass(arr, D, cluster_of, cluster_nodes, deadline):
            imp = True
        arr, imp2 = or_opt_pass(arr, D, deadline)
        if imp2:
            imp = True
        if not imp:
            break
    if use_dp and time.time() < deadline:
        new_arr, c = dp_optimize(arr, D, cluster_of, cluster_nodes, deadline)
        if c is not None:
            arr = new_arr
            # one more quick round of 2-opt after reselection
            while time.time() < deadline:
                imp = two_opt_pass(arr, D, deadline)
                imp = reselect_pass(arr, D, cluster_of, cluster_nodes, deadline) or imp
                if not imp:
                    break
    return arr


def greedy_initial(D, cluster_of, cluster_nodes, m, rng):
    all_nodes = [int(v) for cn in cluster_nodes for v in cn]
    start = rng.choice(all_nodes)
    tour = [start]
    visited_clusters = {cluster_of[start]}
    cur = start
    while len(tour) < m:
        best_v = None
        best_d = math.inf
        for c, cn in enumerate(cluster_nodes):
            if c in visited_clusters:
                continue
            ds = D[cur, cn]
            k = int(np.argmin(ds))
            if ds[k] < best_d:
                best_d = float(ds[k])
                best_v = int(cn[k])
        tour.append(best_v)
        visited_clusters.add(cluster_of[best_v])
        cur = best_v
    return np.array(tour, dtype=np.int64)


def double_bridge(arr, rng):
    m = len(arr)
    if m < 8:
        # simple shuffle-ish perturbation
        a = arr.copy()
        i, j = sorted(rng.sample(range(m), 2))
        a[i:j + 1] = a[i:j + 1][::-1]
        return a
    pts = sorted(rng.sample(range(1, m), 3))
    p1, p2, p3 = pts
    return np.concatenate([arr[:p1], arr[p2:p3], arr[p1:p2], arr[p3:]])


def build_solution(arr, D):
    tour = [int(v) for v in arr]
    return {
        "objective_value": tour_cost(tour, D),
        "tour": tour,
        "visited_nodes": tour[:],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    deadline = start_time + max(1, args.time_limit) - 0.7

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    n, m, nodes_data, clusters_data, D = load_instance(args.instance_path)

    # Map cluster ids to contiguous indices
    cluster_ids = [c["id"] for c in clusters_data]
    cid2idx = {cid: i for i, cid in enumerate(cluster_ids)}
    cluster_nodes = [np.array(sorted(c["node_ids"]), dtype=np.int64) for c in clusters_data]
    cluster_of = np.zeros(n, dtype=np.int64)
    for c in clusters_data:
        for v in c["node_ids"]:
            cluster_of[v] = cid2idx[c["id"]]

    rng = random.Random(0)
    np.random.seed(0)

    # Trivial cases
    if m == 1:
        v = int(cluster_nodes[0][0])
        sol = {"objective_value": 0.0, "tour": [v], "visited_nodes": [v]}
        if logger:
            logger.log_solution(0.0, sol)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return
    if m == 2:
        best = None
        for u in cluster_nodes[0]:
            ds = D[u, cluster_nodes[1]]
            k = int(np.argmin(ds))
            c = 2.0 * float(ds[k])
            if best is None or c < best[0]:
                best = (c, int(u), int(cluster_nodes[1][k]))
        sol = {"objective_value": best[0], "tour": [best[1], best[2]],
               "visited_nodes": [best[1], best[2]]}
        if logger:
            logger.log_solution(best[0], sol)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    # Initial solution
    arr = greedy_initial(D, cluster_of, cluster_nodes, m, rng)
    best_cost = tour_cost(arr, D)
    best_arr = arr.copy()
    if logger:
        logger.log_solution(best_cost, build_solution(best_arr, D))

    # Initial local search
    arr = local_search(arr, D, cluster_of, cluster_nodes, deadline)
    c = tour_cost(arr, D)
    if c < best_cost - 1e-9:
        best_cost = c
        best_arr = arr.copy()
        if logger:
            logger.log_solution(best_cost, build_solution(best_arr, D))

    cur_arr = best_arr.copy()
    cur_cost = best_cost
    no_improve = 0

    # Iterated local search
    while time.time() < deadline:
        cand = double_bridge(cur_arr, rng)
        cand = local_search(cand, D, cluster_of, cluster_nodes, deadline,
                            use_dp=(rng.random() < 0.5))
        c = tour_cost(cand, D)
        if c < cur_cost - 1e-9:
            cur_arr = cand
            cur_cost = c
            no_improve = 0
            if c < best_cost - 1e-9:
                best_cost = c
                best_arr = cand.copy()
                if logger:
                    logger.log_solution(best_cost, build_solution(best_arr, D))
        else:
            no_improve += 1
            if no_improve >= 20:
                # restart from best or fresh greedy
                if rng.random() < 0.3:
                    cur_arr = greedy_initial(D, cluster_of, cluster_nodes, m, rng)
                    cur_arr = local_search(cur_arr, D, cluster_of, cluster_nodes, deadline)
                    cur_cost = tour_cost(cur_arr, D)
                    if cur_cost < best_cost - 1e-9:
                        best_cost = cur_cost
                        best_arr = cur_arr.copy()
                        if logger:
                            logger.log_solution(best_cost, build_solution(best_arr, D))
                else:
                    cur_arr = best_arr.copy()
                    cur_cost = best_cost
                no_improve = 0

    # Final DP polish on best if time allows (small extra margin)
    final_deadline = start_time + max(1, args.time_limit) - 0.3
    if time.time() < final_deadline:
        new_arr, c = dp_optimize(best_arr, D, cluster_of, cluster_nodes, final_deadline)
        if c is not None and c < best_cost - 1e-9:
            best_arr = new_arr
            best_cost = c
            if logger:
                logger.log_solution(best_cost, build_solution(best_arr, D))

    sol = build_solution(best_arr, D)
    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()