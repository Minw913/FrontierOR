import argparse
import json
import os
import sys
import time

from solution_logger import SolutionLogger


class SearchTimeout(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description="Maximum weighted clique solver")
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)
    # Reserve a small amount of time for constructing and writing the result.
    search_deadline = deadline - min(0.10, max(0.01, args.time_limit * 0.01))

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    graph = instance["graph"]
    vertex_ids = list(graph["vertices"])
    weights_input = list(instance["vertex_weights"])
    n = len(vertex_ids)

    if len(weights_input) != n:
        raise ValueError("vertex_weights length does not match the number of vertices")

    if n == 0:
        final_solution = {
            "objective_value": 0.0,
            "clique_vertices": []
        }
        parent = os.path.dirname(os.path.abspath(args.solution_path))
        os.makedirs(parent, exist_ok=True)
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(final_solution, f)
        return

    id_to_old = {vertex_id: i for i, vertex_id in enumerate(vertex_ids)}
    degrees = [0] * n
    valid_edges = []

    for edge in graph["edges"]:
        if len(edge) != 2:
            continue
        u_id, v_id = edge
        u = id_to_old.get(u_id)
        v = id_to_old.get(v_id)
        if u is None or v is None or u == v:
            continue
        valid_edges.append((u, v))
        degrees[u] += 1
        degrees[v] += 1

    # Bit positions are ordered so that greedy coloring considers high-degree,
    # high-weight vertices first.
    old_order = sorted(
        range(n),
        key=lambda i: (-degrees[i], -weights_input[i], vertex_ids[i])
    )
    old_to_new = [0] * n
    for new_i, old_i in enumerate(old_order):
        old_to_new[old_i] = new_i

    ids = [vertex_ids[old_i] for old_i in old_order]
    weights = [int(weights_input[old_i]) for old_i in old_order]
    adjacency = [0] * n

    for old_u, old_v in valid_edges:
        u = old_to_new[old_u]
        v = old_to_new[old_v]
        adjacency[u] |= 1 << v
        adjacency[v] |= 1 << u

    del valid_edges
    del degrees

    try:
        sys.setrecursionlimit(max(2000, min(n + 500, 100000)))
    except Exception:
        pass

    best_weight = 0
    best_clique = []

    def solution_dict(clique, value):
        chosen = sorted(ids[v] for v in clique)
        return {
            "objective_value": float(value),
            "clique_vertices": chosen
        }

    def improve(value, clique):
        nonlocal best_weight, best_clique
        if value <= best_weight:
            return
        best_weight = int(value)
        best_clique = list(clique)
        if logger:
            sol = solution_dict(best_clique, best_weight)
            logger.log_solution(float(best_weight), sol)

    # Ensure that a feasible nonempty solution always exists.
    max_singleton = max(range(n), key=lambda v: (weights[v], adjacency[v].bit_count()))
    improve(weights[max_singleton], [max_singleton])

    all_vertices = (1 << n) - 1
    total_weight = sum(weights)

    def greedy_clique(start, mode):
        clique = [start]
        value = weights[start]
        candidates = adjacency[start]
        scans = 0

        while candidates:
            if time.monotonic() >= search_deadline:
                break

            best_v = -1
            best_key = None
            bits = candidates

            while bits:
                bit = bits & -bits
                v = bit.bit_length() - 1
                bits ^= bit

                if mode == 0:
                    key = (weights[v], (adjacency[v] & candidates).bit_count())
                elif mode == 1:
                    residual_degree = (adjacency[v] & candidates).bit_count()
                    key = (weights[v] * (residual_degree + 1), weights[v], residual_degree)
                else:
                    residual_degree = (adjacency[v] & candidates).bit_count()
                    key = (residual_degree, weights[v])

                if best_key is None or key > best_key:
                    best_key = key
                    best_v = v

                scans += 1
                if (scans & 511) == 0 and time.monotonic() >= search_deadline:
                    break

            if best_v < 0:
                break

            clique.append(best_v)
            value += weights[best_v]
            candidates &= adjacency[best_v]

        improve(value, clique)

    # A small portfolio of constructive heuristics supplies a good incumbent.
    if time.monotonic() < search_deadline and best_weight < total_weight:
        by_weight = sorted(
            range(n),
            key=lambda v: (-weights[v], -adjacency[v].bit_count())
        )
        by_score = sorted(
            range(n),
            key=lambda v: (
                -(weights[v] * (adjacency[v].bit_count() + 1)),
                -weights[v]
            )
        )

        start_limit = min(n, 48)
        starts = []
        seen = set()
        for sequence in (by_weight[:start_limit], by_score[:start_limit], range(min(n, 16))):
            for v in sequence:
                if v not in seen:
                    seen.add(v)
                    starts.append(v)

        for mode in (0, 1, 2):
            for v in starts:
                if time.monotonic() >= search_deadline:
                    break
                greedy_clique(v, mode)
                if best_weight >= total_weight:
                    break
            if time.monotonic() >= search_deadline or best_weight >= total_weight:
                break

    node_count = 0

    def color_sort(candidates):
        """
        Partition candidates greedily into independent sets. A clique can use
        at most one vertex from each such set, so the cumulative sum of the
        largest weight in each class is a valid weighted upper bound.
        """
        order = []
        bounds = []
        uncolored = candidates
        cumulative = 0
        work = 0

        while uncolored:
            available = uncolored
            color_class = []
            class_max = 0

            while available:
                bit = available & -available
                v = bit.bit_length() - 1
                available &= ~(bit | adjacency[v])
                uncolored ^= bit
                color_class.append(v)
                if weights[v] > class_max:
                    class_max = weights[v]

                work += 1
                if (work & 1023) == 0 and time.monotonic() >= search_deadline:
                    raise SearchTimeout

            cumulative += class_max
            order.extend(color_class)
            bounds.extend([cumulative] * len(color_class))

        return order, bounds

    current_clique = []

    def expand(candidates, current_weight):
        nonlocal node_count
        node_count += 1
        if (node_count & 255) == 0 and time.monotonic() >= search_deadline:
            raise SearchTimeout

        order, bounds = color_sort(candidates)

        for i in range(len(order) - 1, -1, -1):
            if current_weight + bounds[i] <= best_weight:
                return

            if time.monotonic() >= search_deadline:
                raise SearchTimeout

            v = order[i]
            bit = 1 << v

            # This vertex may already have been removed by an earlier branch.
            if not (candidates & bit):
                continue

            new_weight = current_weight + weights[v]
            current_clique.append(v)
            improve(new_weight, current_clique)

            new_candidates = candidates & adjacency[v]
            if new_candidates and new_weight < total_weight:
                expand(new_candidates, new_weight)

            current_clique.pop()
            candidates &= ~bit

    if time.monotonic() < search_deadline and best_weight < total_weight:
        try:
            expand(all_vertices, 0)
        except (SearchTimeout, RecursionError):
            pass

    final_solution = solution_dict(best_clique, best_weight)

    parent = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(parent, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f)


if __name__ == "__main__":
    main()