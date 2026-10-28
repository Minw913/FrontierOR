import argparse
import itertools
import json
import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple


def now() -> float:
    return time.time()


# Safe logger that never depends on DBus/session services.
# It writes JSONL incumbents to --log_path.
class SolutionLogger:
    def __init__(self, log_path: Optional[str], sense: str):
        self.log_path = log_path
        self.sense = sense
        self.best = None
        self.enabled = bool(log_path)
        if self.enabled:
            with open(self.log_path, "w") as f:
                pass

    def _better(self, obj: float) -> bool:
        if self.best is None:
            return True
        if self.sense == "minimize":
            return obj < self.best - 1e-12
        return obj > self.best + 1e-12

    def log_solution(self, objective_value: float, solution_dict: Dict[str, Any]) -> None:
        if not self.enabled or not self._better(objective_value):
            return
        self.best = objective_value
        rec = {"objective_value": float(objective_value), **solution_dict}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def log(self, objective_value: float) -> None:
        if not self.enabled or not self._better(objective_value):
            return
        self.best = objective_value
        rec = {"objective_value": float(objective_value)}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")


def is_tsp_instance(instance: Dict[str, Any]) -> bool:
    pt = str(instance.get("problem_type", "")).lower()
    if "tsp" in pt:
        return True
    if "maxcut" in pt:
        return False
    params = instance.get("parameters", {})
    data = instance.get("data", {})
    if isinstance(params, dict) and "n_cities" in params:
        return True
    if isinstance(data, dict) and ("distance_matrix" in data or "coordinates" in data):
        return True
    return False


# ================================ TSP ===================================

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
    return sum(dist[tour[i]][tour[(i + 1) % n]] for i in range(n))


def build_tsp_solution_vector(tour: List[int], n: int) -> Dict[str, int]:
    pos = [0] * n
    for t, c in enumerate(tour):
        pos[c] = t
    sol = {}
    for c in range(n):
        for t in range(n):
            sol[f"x_{c}_{t}"] = 1 if pos[c] == t else 0
    return sol


def nearest_neighbor_tour(start: int, dist: List[List[float]]) -> List[int]:
    n = len(dist)
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: dist[cur][j])
        unvisited.remove(nxt)
        tour.append(nxt)
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
    return tour


def solve_tsp(instance: Dict[str, Any], deadline: float, logger: SolutionLogger) -> Dict[str, Any]:
    n, dist = extract_tsp(instance)
    best_tour = list(range(n))
    best_obj = tour_length(best_tour, dist)

    def update(tour: List[int], obj: float):
        nonlocal best_tour, best_obj
        if obj + 1e-12 < best_obj:
            best_tour = tour[:]
            best_obj = obj
            logger.log_solution(best_obj, {
                "solution": build_tsp_solution_vector(best_tour, n),
                "tour": best_tour[:]
            })

    if n <= 10:
        base = 0
        others = [i for i in range(n) if i != base]
        for perm in itertools.permutations(others):
            if now() >= deadline:
                break
            t = [base] + list(perm)
            update(t, tour_length(t, dist))
    else:
        starts = list(range(n))
        random.shuffle(starts)
        for s in starts:
            if now() >= deadline:
                break
            t = nearest_neighbor_tour(s, dist)
            t = two_opt(t, dist, deadline)
            update(t, tour_length(t, dist))

        while now() < deadline:
            t = best_tour[:]
            i, j = sorted(random.sample(range(1, n), 2))
            t[i:j + 1] = reversed(t[i:j + 1])
            t = two_opt(t, dist, deadline)
            update(t, tour_length(t, dist))

    sol = build_tsp_solution_vector(best_tour, n)
    return {"objective_value": float(best_obj), "solution": sol, "tour": [int(x) for x in best_tour]}


# ============================== MaxCut ==================================

def extract_maxcut(instance: Dict[str, Any]) -> Tuple[List[Any], List[Tuple[int, int, float]]]:
    params = instance.get("parameters", {})
    data = instance.get("data", instance)

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

    adjacency = None
    for container in (data, instance):
        if isinstance(container, dict) and "adjacency_matrix" in container:
            adjacency = container["adjacency_matrix"]
            break

    edge_label_tuples = []
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
            if u is None or v is None or u == v:
                continue
            edge_label_tuples.append((u, v, float(w)))
    elif adjacency is not None:
        n_adj = len(adjacency)
        if labels is None:
            labels = list(range(n_adj))
        for i in range(n_adj):
            for j in range(i + 1, n_adj):
                w = float(adjacency[i][j])
                if abs(w) > 0:
                    edge_label_tuples.append((labels[i], labels[j], w))
    else:
        raise ValueError("MaxCut instance missing edges/edge_list/adjacency_matrix.")

    if labels is None:
        seen = set()
        for u, v, _ in edge_label_tuples:
            seen.add(u)
            seen.add(v)
        try:
            labels = sorted(seen)
        except Exception:
            labels = sorted(seen, key=lambda x: str(x))

    if n_nodes is not None and len(labels) < n_nodes and all(isinstance(x, int) for x in labels):
        s = set(labels)
        if s.issubset(set(range(1, n_nodes + 1))) and not s.issubset(set(range(n_nodes))):
            labels = list(range(1, n_nodes + 1))
        else:
            labels = list(range(n_nodes))

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    edges = []
    for u, v, w in edge_label_tuples:
        if u in label_to_idx and v in label_to_idx:
            i, j = label_to_idx[u], label_to_idx[v]
            if i != j:
                edges.append((i, j, float(w)))

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
        if n == 0:
            break
        idx = max(range(n), key=lambda k: gains[k])
        g = gains[idx]
        if g <= 1e-12:
            break
        assign[idx] ^= 1
        obj += g
    return assign, obj


def build_maxcut_solution_vector(assign: List[int], labels: List[Any]) -> Dict[str, int]:
    return {f"x_{labels[i]}": int(assign[i]) for i in range(len(labels))}


def solve_maxcut(instance: Dict[str, Any], deadline: float, logger: SolutionLogger) -> Dict[str, Any]:
    labels, edges = extract_maxcut(instance)
    n = len(labels)
    random.seed(0)

    best_assign = [0] * n
    best_obj = cut_value(best_assign, edges)

    if n > 0:
        alt = [i % 2 for i in range(n)]
        alt_obj = cut_value(alt, edges)
        if alt_obj > best_obj:
            best_assign, best_obj = alt[:], alt_obj

    logger.log_solution(best_obj, {"solution": build_maxcut_solution_vector(best_assign, labels)})

    while now() < deadline:
        a = [random.randint(0, 1) for _ in range(n)]
        a, obj = one_flip_local_search(a, edges, deadline)
        if obj > best_obj + 1e-12:
            best_assign, best_obj = a[:], obj
            logger.log_solution(best_obj, {"solution": build_maxcut_solution_vector(best_assign, labels)})

        if n > 0 and now() < deadline:
            p = best_assign[:]
            flips = max(1, n // 20)
            for _ in range(flips):
                p[random.randrange(n)] ^= 1
            p, obj = one_flip_local_search(p, edges, deadline)
            if obj > best_obj + 1e-12:
                best_assign, best_obj = p[:], obj
                logger.log_solution(best_obj, {"solution": build_maxcut_solution_vector(best_assign, labels)})

    return {"objective_value": float(best_obj), "solution": build_maxcut_solution_vector(best_assign, labels)}


# ================================ Main ==================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    tsp = is_tsp_instance(instance)
    logger = SolutionLogger(args.log_path, sense="minimize" if tsp else "maximize")

    random.seed(0)

    start = now()
    deadline = start + max(0, int(args.time_limit)) - 1e-3

    if tsp:
        result = solve_tsp(instance, deadline, logger)
    else:
        result = solve_maxcut(instance, deadline, logger)

    with open(args.solution_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()