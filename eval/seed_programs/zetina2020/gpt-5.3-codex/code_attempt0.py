import argparse
import json
import math
import random
import time
import itertools
from typing import Any, Dict, List, Tuple

from solution_logger import SolutionLogger


def now() -> float:
    return time.time()


def is_tsp_instance(instance: Dict[str, Any]) -> bool:
    pt = str(instance.get("problem_type", "")).lower()
    if "tsp" in pt:
        return True
    if "maxcut" in pt:
        return False
    params = instance.get("parameters", {})
    data = instance.get("data", {})
    if "n_cities" in params:
        return True
    if isinstance(data, dict) and ("distance_matrix" in data or "coordinates" in data):
        return True
    return False


# ----------------------------- TSP --------------------------------- #

def extract_tsp(instance: Dict[str, Any]) -> Tuple[int, List[List[float]]]:
    params = instance.get("parameters", {})
    data = instance.get("data", instance)

    dist = data.get("distance_matrix")
    coords = data.get("coordinates")

    n = params.get("n_cities")
    if n is None:
        if dist is not None:
            n = len(dist)
        elif coords is not None:
            n = len(coords)
        else:
            raise ValueError("TSP instance missing n_cities / distance_matrix / coordinates.")

    n = int(n)

    if dist is None:
        if coords is None:
            raise ValueError("TSP instance missing both distance_matrix and coordinates.")
        dist = [[0.0] * n for _ in range(n)]
        for i in range(n):
            xi, yi = float(coords[i][0]), float(coords[i][1])
            for j in range(i + 1, n):
                xj, yj = float(coords[j][0]), float(coords[j][1])
                d = math.hypot(xi - xj, yi - yj)
                dist[i][j] = d
                dist[j][i] = d
    else:
        dist = [[float(dist[i][j]) for j in range(n)] for i in range(n)]

    return n, dist


def tour_length(tour: List[int], dist: List[List[float]]) -> float:
    n = len(tour)
    total = 0.0
    for i in range(n):
        total += dist[tour[i]][tour[(i + 1) % n]]
    return total


def build_tsp_solution_vector(tour: List[int], n: int) -> Dict[str, int]:
    pos_of_city = [0] * n
    for t, c in enumerate(tour):
        pos_of_city[c] = t
    sol = {}
    for c in range(n):
        t_c = pos_of_city[c]
        for t in range(n):
            sol[f"x_{c}_{t}"] = 1 if t == t_c else 0
    return sol


def nearest_neighbor_tour(start: int, dist: List[List[float]]) -> List[int]:
    n = len(dist)
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: dist[cur][j])
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return tour


def two_opt(tour: List[int], dist: List[List[float]], deadline: float) -> List[int]:
    n = len(tour)
    improved = True
    while improved and now() < deadline:
        improved = False
        for i in range(1, n - 2):
            if now() >= deadline:
                break
            a, b = tour[i - 1], tour[i]
            for k in range(i + 1, n - 1):
                c, d = tour[k], tour[(k + 1) % n]
                delta = (dist[a][c] + dist[b][d]) - (dist[a][b] + dist[c][d])
                if delta < -1e-12:
                    tour[i:k + 1] = reversed(tour[i:k + 1])
                    improved = True
        # continue if improved
    return tour


def solve_tsp(instance: Dict[str, Any], deadline: float, logger: SolutionLogger = None) -> Dict[str, Any]:
    n, dist = extract_tsp(instance)

    best_tour = list(range(n))
    best_obj = tour_length(best_tour, dist)

    def log_if_better(tour: List[int], obj: float):
        nonlocal best_tour, best_obj
        if obj + 1e-12 < best_obj:
            best_tour = tour[:]
            best_obj = obj
            if logger:
                sol_vec = build_tsp_solution_vector(best_tour, n)
                logger.log_solution(best_obj, {"solution": sol_vec, "tour": best_tour[:]})

    # Exact search for small n (with fixed start city 0)
    if n <= 10 and now() < deadline:
        base = 0
        others = [i for i in range(n) if i != base]
        for perm in itertools.permutations(others):
            if now() >= deadline:
                break
            t = [base] + list(perm)
            obj = tour_length(t, dist)
            log_if_better(t, obj)

    # Heuristic fallback / improvement
    starts = list(range(n))
    random.shuffle(starts)
    idx = 0
    while now() < deadline and idx < len(starts):
        s = starts[idx]
        idx += 1
        t = nearest_neighbor_tour(s, dist)
        t = two_opt(t, dist, deadline)
        obj = tour_length(t, dist)
        log_if_better(t, obj)

    # If still time, random perturbations + 2-opt
    while now() < deadline:
        t = best_tour[:]
        if n >= 4:
            i, j = sorted(random.sample(range(1, n), 2))
            t[i:j + 1] = reversed(t[i:j + 1])
        t = two_opt(t, dist, deadline)
        obj = tour_length(t, dist)
        log_if_better(t, obj)
        if n <= 6:
            break  # already effectively exact for tiny n

    solution_vec = build_tsp_solution_vector(best_tour, n)
    return {
        "objective_value": float(best_obj),
        "solution": solution_vec,
        "tour": [int(x) for x in best_tour],
    }


# ---------------------------- MaxCut -------------------------------- #

def extract_maxcut(instance: Dict[str, Any]) -> Tuple[List[Any], List[Tuple[int, int, float]]]:
    params = instance.get("parameters", {})
    data = instance.get("data", instance)

    # Candidate node labels
    labels = None
    for container in (data, instance, params):
        if isinstance(container, dict) and "nodes" in container:
            raw_nodes = container["nodes"]
            if isinstance(raw_nodes, list):
                if raw_nodes and isinstance(raw_nodes[0], dict):
                    labels = [nd.get("id", i) for i, nd in enumerate(raw_nodes)]
                else:
                    labels = list(raw_nodes)
                break

    n_nodes = None
    for container in (params, data, instance):
        if isinstance(container, dict):
            for k in ("n_nodes", "num_nodes", "node_count"):
                if k in container:
                    n_nodes = int(container[k])
                    break
        if n_nodes is not None:
            break

    edges_raw = None
    for container in (data, instance):
        if isinstance(container, dict):
            for k in ("edges", "edge_list", "links"):
                if k in container:
                    edges_raw = container[k]
                    break
        if edges_raw is not None:
            break

    # Adjacency matrix fallback
    adjacency = None
    for container in (data, instance):
        if isinstance(container, dict) and "adjacency_matrix" in container:
            adjacency = container["adjacency_matrix"]
            break

    edge_tuples_labels = []

    if edges_raw is not None:
        for e in edges_raw:
            if isinstance(e, dict):
                u = e.get("u", e.get("i", e.get("source", e.get("from"))))
                v = e.get("v", e.get("j", e.get("target", e.get("to"))))
                w = e.get("weight", e.get("w", 1.0))
            elif isinstance(e, (list, tuple)) and len(e) >= 2:
                u, v = e[0], e[1]
                w = e[2] if len(e) >= 3 else 1.0
            else:
                continue
            if u is None or v is None:
                continue
            if u == v:
                continue
            edge_tuples_labels.append((u, v, float(w)))

    elif adjacency is not None:
        n_adj = len(adjacency)
        if labels is None:
            labels = list(range(n_adj))
        for i in range(n_adj):
            for j in range(i + 1, n_adj):
                w = float(adjacency[i][j])
                if abs(w) > 0:
                    edge_tuples_labels.append((labels[i], labels[j], w))
    else:
        raise ValueError("MaxCut instance missing edges/edge_list/adjacency_matrix.")

    # Infer labels if needed
    if labels is None:
        seen = set()
        for u, v, _ in edge_tuples_labels:
            seen.add(u)
            seen.add(v)
        if n_nodes is not None and all(isinstance(x, (int, float)) for x in seen):
            seen_int = sorted(int(x) for x in seen)
            if seen_int and min(seen_int) == 1 and max(seen_int) <= n_nodes:
                labels = list(range(1, n_nodes + 1))
            else:
                labels = list(range(n_nodes))
        else:
            labels = sorted(seen)

    # If n_nodes larger than provided labels, pad numeric labels
    if n_nodes is not None and len(labels) < n_nodes and all(isinstance(x, int) for x in labels):
        s = set(labels)
        base0 = set(range(n_nodes))
        base1 = set(range(1, n_nodes + 1))
        if s.issubset(base1) and not s.issubset(base0):
            labels = list(range(1, n_nodes + 1))
        else:
            labels = list(range(n_nodes))

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    edges = []
    for u, v, w in edge_tuples_labels:
        if u not in label_to_idx or v not in label_to_idx:
            continue
        i, j = label_to_idx[u], label_to_idx[v]
        if i == j:
            continue
        edges.append((i, j, float(w)))

    if not edges:
        # still feasible: empty graph
        pass

    return labels, edges


def cut_value(assign: List[int], edges: List[Tuple[int, int, float]]) -> float:
    total = 0.0
    for i, j, w in edges:
        if assign[i] != assign[j]:
            total += w
    return total


def compute_gains(assign: List[int], edges: List[Tuple[int, int, float]], n: int) -> List[float]:
    gains = [0.0] * n
    for i, j, w in edges:
        if assign[i] == assign[j]:
            gains[i] += w
            gains[j] += w
        else:
            gains[i] -= w
            gains[j] -= w
    return gains


def one_flip_local_search(assign: List[int], edges: List[Tuple[int, int, float]], deadline: float) -> Tuple[List[int], float]:
    n = len(assign)
    obj = cut_value(assign, edges)
    while now() < deadline:
        gains = compute_gains(assign, edges, n)
        best_idx = max(range(n), key=lambda x: gains[x]) if n > 0 else -1
        best_gain = gains[best_idx] if best_idx >= 0 else 0.0
        if best_gain <= 1e-12:
            break
        assign[best_idx] ^= 1
        obj += best_gain
    return assign, obj


def build_maxcut_solution_vector(assign: List[int], labels: List[Any]) -> Dict[str, int]:
    sol = {}
    for i, lab in enumerate(labels):
        sol[f"x_{lab}"] = int(assign[i])
    return sol


def solve_maxcut(instance: Dict[str, Any], deadline: float, logger: SolutionLogger = None) -> Dict[str, Any]:
    labels, edges = extract_maxcut(instance)
    n = len(labels)

    # Initial feasible
    best_assign = [0] * n
    best_obj = cut_value(best_assign, edges)

    if n > 0:
        # simple alternating start can be better than all-zero
        alt = [i % 2 for i in range(n)]
        alt_obj = cut_value(alt, edges)
        if alt_obj > best_obj:
            best_obj, best_assign = alt_obj, alt[:]

    if logger:
        logger.log_solution(best_obj, {"solution": build_maxcut_solution_vector(best_assign, labels)})

    # Multi-start local search
    random.seed(0)
    while now() < deadline:
        assign = [random.randint(0, 1) for _ in range(n)]
        assign, obj = one_flip_local_search(assign, edges, deadline)
        if obj > best_obj + 1e-12:
            best_obj = obj
            best_assign = assign[:]
            if logger:
                logger.log_solution(best_obj, {"solution": build_maxcut_solution_vector(best_assign, labels)})

        # light perturbation around incumbent
        if now() < deadline and n > 0:
            assign = best_assign[:]
            flips = max(1, n // 20)
            for _ in range(flips):
                idx = random.randrange(n)
                assign[idx] ^= 1
            assign, obj = one_flip_local_search(assign, edges, deadline)
            if obj > best_obj + 1e-12:
                best_obj = obj
                best_assign = assign[:]
                if logger:
                    logger.log_solution(best_obj, {"solution": build_maxcut_solution_vector(best_assign, labels)})

    solution_vec = build_maxcut_solution_vector(best_assign, labels)
    return {
        "objective_value": float(best_obj),
        "solution": solution_vec,
    }


# ----------------------------- Main --------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    problem_is_tsp = is_tsp_instance(instance)
    logger = SolutionLogger(args.log_path, sense="minimize" if problem_is_tsp else "maximize") if args.log_path else None

    start = now()
    # small safety margin
    deadline = start + max(0, args.time_limit) - 1e-3

    if problem_is_tsp:
        result = solve_tsp(instance, deadline, logger)
    else:
        result = solve_maxcut(instance, deadline, logger)

    with open(args.solution_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()