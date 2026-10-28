import argparse
import json
import time
import heapq
from typing import List, Dict, Any, Optional

from solution_logger import SolutionLogger


class FacilityLocationOracle:
    def __init__(self, data_matrix: List[List[float]], n: int, m: int):
        self.data = data_matrix
        self.n = n
        self.m = m

    def initial_state(self) -> Dict[str, Any]:
        return {"cur": [0.0] * self.m, "obj": 0.0}

    def marginal_gain(self, state: Dict[str, Any], j: int) -> float:
        cur = state["cur"]
        gain = 0.0
        data = self.data
        for i in range(self.m):
            v = data[i][j]
            c = cur[i]
            if v > c:
                gain += (v - c)
        return gain

    def add(self, state: Dict[str, Any], j: int) -> float:
        cur = state["cur"]
        data = self.data
        gain = 0.0
        for i in range(self.m):
            v = data[i][j]
            c = cur[i]
            if v > c:
                inc = v - c
                gain += inc
                cur[i] = v
        state["obj"] += gain
        return gain


class CoverageOracle:
    def __init__(self, data_matrix: List[List[float]], node_weights: List[float], n: int, m: int):
        self.n = n
        self.m = m
        self.weights = node_weights if node_weights else [0.0] * m

        # Precompute sensor -> covered items
        sensor_to_items = [[] for _ in range(n)]
        for i in range(m):
            row = data_matrix[i]
            for j in range(n):
                if row[j] > 0.5:
                    sensor_to_items[j].append(i)
        self.sensor_to_items = sensor_to_items

    def initial_state(self) -> Dict[str, Any]:
        return {"covered": bytearray(self.m), "obj": 0.0}

    def marginal_gain(self, state: Dict[str, Any], j: int) -> float:
        covered = state["covered"]
        gain = 0.0
        w = self.weights
        for i in self.sensor_to_items[j]:
            if not covered[i]:
                gain += w[i]
        return gain

    def add(self, state: Dict[str, Any], j: int) -> float:
        covered = state["covered"]
        w = self.weights
        gain = 0.0
        for i in self.sensor_to_items[j]:
            if not covered[i]:
                covered[i] = 1
                gain += w[i]
        state["obj"] += gain
        return gain


class InfluenceOracle:
    def __init__(self, data_matrix: List[List[float]], element_weights: List[float], n: int, m: int):
        self.n = n
        self.m = m
        self.act = element_weights if element_weights else [0.0] * n

        # Precompute source -> targets with an edge
        source_to_targets = [[] for _ in range(n)]
        for i in range(m):
            row = data_matrix[i]
            for j in range(n):
                if row[j] > 0.5:
                    source_to_targets[j].append(i)
        self.source_to_targets = source_to_targets

    def initial_state(self) -> Dict[str, Any]:
        # q[i] = probability target i is still NOT reached
        return {"q": [1.0] * self.m, "obj": 0.0}

    def marginal_gain(self, state: Dict[str, Any], j: int) -> float:
        p = self.act[j]
        if p <= 0.0:
            return 0.0
        q = state["q"]
        s = 0.0
        for i in self.source_to_targets[j]:
            s += q[i]
        return p * s

    def add(self, state: Dict[str, Any], j: int) -> float:
        p = self.act[j]
        if p <= 0.0:
            return 0.0
        q = state["q"]
        targets = self.source_to_targets[j]
        s = 0.0
        for i in targets:
            s += q[i]
        gain = p * s
        factor = 1.0 - p
        for i in targets:
            q[i] *= factor
        state["obj"] += gain
        return gain


def greedy_lazy_maximize(oracle, k: int, deadline: float, on_improve=None):
    n = oracle.n
    state = oracle.initial_state()
    selected = [False] * n
    sel = []

    if on_improve is not None:
        on_improve(state["obj"], sel)

    if k <= 0 or n == 0:
        return sel, state["obj"]

    # Initialize lazy heap with exact gains at iteration 0
    heap = []
    iter_tag = 0
    for j in range(n):
        if time.perf_counter() >= deadline:
            return sel, state["obj"]
        g = oracle.marginal_gain(state, j)
        heapq.heappush(heap, (-g, j, iter_tag))

    while len(sel) < k and heap and time.perf_counter() < deadline:
        iter_tag = len(sel)
        chosen = False

        while heap and time.perf_counter() < deadline:
            neg_g, j, tag = heapq.heappop(heap)
            if selected[j]:
                continue

            if tag == iter_tag:
                gain = -neg_g
                if gain <= 1e-15:
                    return sel, state["obj"]  # no further useful improvement
                oracle.add(state, j)
                selected[j] = True
                sel.append(j)
                if on_improve is not None:
                    on_improve(state["obj"], sel)
                chosen = True
                break
            else:
                # Recompute at current iteration and push back
                g = oracle.marginal_gain(state, j)
                heapq.heappush(heap, (-g, j, iter_tag))

        if not chosen:
            break

    return sel, state["obj"]


def build_oracle(instance: Dict[str, Any]):
    itype = instance["instance_type"]
    n = int(instance["n"])
    m = int(instance["m"])
    data_matrix = instance["data_matrix"]

    if itype == "LOC":
        return FacilityLocationOracle(data_matrix, n, m)
    elif itype == "COV":
        node_weights = instance.get("node_weights", [0.0] * m)
        return CoverageOracle(data_matrix, node_weights, n, m)
    elif itype == "INF":
        element_weights = instance.get("element_weights", [0.0] * n)
        return InfluenceOracle(data_matrix, element_weights, n, m)
    else:
        raise ValueError(f"Unknown instance_type: {itype}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start = time.perf_counter()
    # small safety margin for writing output
    deadline = start + max(0, args.time_limit) - 0.02

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    n = int(instance["n"])
    k = int(instance["k"])
    k = max(0, min(k, n))

    oracle = build_oracle(instance)

    best = {"obj": float("-inf"), "sel": []}

    def on_improve(obj_val, sel_list):
        if obj_val > best["obj"] + 1e-12:
            best["obj"] = float(obj_val)
            best["sel"] = list(sel_list)
            if logger:
                logger.log(best["obj"])

    # Main construction
    sel, obj = greedy_lazy_maximize(oracle, k, deadline, on_improve=on_improve)

    # Fallback in extremely tight timing cases
    if best["obj"] == float("-inf"):
        best["obj"] = float(obj)
        best["sel"] = list(sel)
        if logger:
            logger.log(best["obj"])

    solution = {
        "objective_value": float(best["obj"]),
        "selected_elements": [int(x) for x in best["sel"][:k]],
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()