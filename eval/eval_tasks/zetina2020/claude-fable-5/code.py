import argparse
import json
import time
import random
import itertools
import math

from solution_logger import SolutionLogger


# ----------------------------------------------------------------------
# Instance loading / detection
# ----------------------------------------------------------------------
def load_instance(path):
    with open(path, "r") as f:
        return json.load(f)


def detect_problem_type(inst):
    params = inst.get("parameters", {}) or {}
    data = inst.get("data", {}) or {}
    text_keys = set(params.keys()) | set(data.keys())
    if "distance_matrix" in data or "n_cities" in params or "coordinates" in data:
        return "TSP"
    if any(k in text_keys for k in ("edges", "edge_list", "adjacency_matrix",
                                    "n_nodes", "num_nodes", "graph")):
        return "MAXCUT"
    pt = str(inst.get("problem_type", "")).upper()
    if "TSP" in pt:
        return "TSP"
    return "MAXCUT"


# ----------------------------------------------------------------------
# TSP
# ----------------------------------------------------------------------
def tour_length(tour, D):
    n = len(tour)
    return sum(D[tour[i]][tour[(i + 1) % n]] for i in range(n))


def solve_tsp(inst, deadline, logger):
    data = inst["data"]
    D = data["distance_matrix"]
    n = len(D)

    best_tour = list(range(n))
    best_len = tour_length(best_tour, D)
    if logger:
        logger.log_solution(best_len, tsp_solution_dict(best_tour, n))

    if n <= 10:
        # exact brute force (fix city 0 first)
        for perm in itertools.permutations(range(1, n)):
            tour = [0] + list(perm)
            L = tour_length(tour, D)
            if L < best_len - 1e-12:
                best_len, best_tour = L, tour
                if logger:
                    logger.log_solution(best_len, tsp_solution_dict(best_tour, n))
            if time.time() > deadline:
                break
        return best_tour, best_len

    # heuristic: nearest neighbor + 2-opt with restarts
    rng = random.Random(0)

    def nearest_neighbor(start):
        unvis = set(range(n))
        unvis.remove(start)
        tour = [start]
        cur = start
        while unvis:
            nxt = min(unvis, key=lambda j: D[cur][j])
            unvis.remove(nxt)
            tour.append(nxt)
            cur = nxt
        return tour

    def two_opt(tour):
        improved = True
        L = tour_length(tour, D)
        while improved and time.time() < deadline:
            improved = False
            for i in range(n - 1):
                a, b = tour[i], tour[i + 1]
                for j in range(i + 2, n):
                    c, d = tour[j], tour[(j + 1) % n]
                    if a == d:
                        continue
                    delta = D[a][c] + D[b][d] - D[a][b] - D[c][d]
                    if delta < -1e-12:
                        tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                        L += delta
                        improved = True
                        break
                if improved:
                    break
        return tour, L

    starts = list(range(n))
    rng.shuffle(starts)
    idx = 0
    while time.time() < deadline:
        if idx < len(starts):
            t = nearest_neighbor(starts[idx])
            idx += 1
        else:
            t = list(range(n))
            rng.shuffle(t)
        t, L = two_opt(t)
        if L < best_len - 1e-12:
            best_len, best_tour = L, t[:]
            if logger:
                logger.log_solution(best_len, tsp_solution_dict(best_tour, n))
        if idx >= len(starts) and n > 200:
            break
    return best_tour, best_len


def tsp_solution_dict(tour, n):
    sol = {}
    pos = [0] * n
    for t, city in enumerate(tour):
        pos[city] = t
    for city in range(n):
        for t in range(n):
            sol["x_%d_%d" % (city, t)] = 1 if pos[city] == t else 0
    return sol


# ----------------------------------------------------------------------
# MaxCut
# ----------------------------------------------------------------------
def parse_graph(inst):
    params = inst.get("parameters", {}) or {}
    data = inst.get("data", {}) or {}

    edges = None
    for key in ("edges", "edge_list"):
        if key in data:
            edges = data[key]
            break
        if key in params:
            edges = params[key]
            break

    n = None
    for key in ("n_nodes", "num_nodes", "n", "nodes"):
        v = params.get(key, data.get(key))
        if isinstance(v, int):
            n = v
            break

    edge_list = []
    if edges is not None:
        for e in edges:
            if len(e) >= 3:
                u, v, w = int(e[0]), int(e[1]), float(e[2])
            else:
                u, v, w = int(e[0]), int(e[1]), 1.0
            edge_list.append((u, v, w))
        if n is None:
            n = max(max(u, v) for u, v, _ in edge_list) + 1
    elif "adjacency_matrix" in data:
        A = data["adjacency_matrix"]
        n = len(A)
        for i in range(n):
            for j in range(i + 1, n):
                if A[i][j]:
                    edge_list.append((i, j, float(A[i][j])))
    else:
        raise ValueError("Cannot parse MaxCut graph from instance")

    weights = data.get("weights", params.get("weights"))
    if weights is not None and len(weights) == len(edge_list):
        edge_list = [(u, v, float(w)) for (u, v, _), w in zip(edge_list, weights)]

    return n, edge_list


def solve_maxcut(inst, deadline, logger):
    n, edges = parse_graph(inst)
    rng = random.Random(0)

    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        if u == v:
            continue
        adj[u].append((v, w))
        adj[v].append((u, w))

    def cut_value(s):
        return sum(w for u, v, w in edges if s[u] != s[v])

    def compute_deltas(s):
        # delta[i]: change in cut value if node i flips
        d = [0.0] * n
        for i in range(n):
            acc = 0.0
            for j, w in adj[i]:
                acc += w if s[i] == s[j] else -w
            d[i] = acc
        return d

    best_s = [rng.randint(0, 1) for _ in range(n)]
    best_cut = cut_value(best_s)
    if logger:
        logger.log_solution(best_cut, maxcut_solution_dict(best_s))

    s = best_s[:]
    cut = best_cut
    delta = compute_deltas(s)

    def flip(i):
        nonlocal cut
        cut += delta[i]
        for j, w in adj[i]:
            if s[i] == s[j]:
                delta[j] -= 2 * w
            else:
                delta[j] += 2 * w
        delta[i] = -delta[i]
        s[i] = 1 - s[i]

    perturb_size = max(2, n // 12)

    while time.time() < deadline:
        # local search: best-improvement one-flip
        improved = True
        while improved:
            improved = False
            best_i, best_d = -1, 1e-9
            for i in range(n):
                if delta[i] > best_d:
                    best_d, best_i = delta[i], i
            if best_i >= 0:
                flip(best_i)
                improved = True
            if time.time() > deadline:
                break

        if cut > best_cut + 1e-9:
            best_cut = cut
            best_s = s[:]
            if logger:
                logger.log_solution(best_cut, maxcut_solution_dict(best_s))

        if time.time() > deadline:
            break

        # perturbation (iterated local search): restart from best + random flips
        s = best_s[:]
        delta = compute_deltas(s)
        cut = best_cut
        for _ in range(perturb_size):
            flip(rng.randrange(n))

    return best_s, best_cut


def maxcut_solution_dict(s):
    return {"x_%d" % i: int(v) for i, v in enumerate(s)}


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=60)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    start = time.time()
    deadline = start + max(1, args.time_limit) - 0.5

    inst = load_instance(args.instance_path)
    ptype = detect_problem_type(inst)

    if ptype == "TSP":
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
        tour, best_len = solve_tsp(inst, deadline, logger)
        n = len(tour)
        out = {
            "objective_value": float(best_len),
            "solution": tsp_solution_dict(tour, n),
            "tour": [int(c) for c in tour],
        }
    else:
        logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
        s, best_cut = solve_maxcut(inst, deadline, logger)
        out = {
            "objective_value": float(best_cut),
            "solution": maxcut_solution_dict(s),
        }

    with open(args.solution_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()