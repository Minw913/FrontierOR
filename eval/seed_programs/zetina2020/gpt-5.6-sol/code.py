import argparse
import itertools
import json
import math
import os
import random
import time

from solution_logger import SolutionLogger


# ---------------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------------

def get_first(mapping, names, default=None):
    if not isinstance(mapping, dict):
        return default

    normalized = {str(key).lower(): value for key, value in mapping.items()}
    for name in names:
        if name.lower() in normalized:
            return normalized[name.lower()]
    return default


def find_named_value(obj, names):
    """Recursively find the first value whose key matches one of names."""
    target_names = {name.lower() for name in names}

    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in target_names:
                return value
        for value in obj.values():
            result = find_named_value(value, target_names)
            if result is not None:
                return result

    return None


def write_json(path, value):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, separators=(",", ":"), allow_nan=False)
    os.replace(temporary_path, path)


def detect_problem_type(instance):
    declared_type = find_named_value(
        instance,
        [
            "problem_type",
            "problem",
            "problem_name",
            "optimization_problem",
            "type",
        ],
    )

    if isinstance(declared_type, str):
        lowered = declared_type.lower().replace("_", "").replace("-", "")
        if "maxcut" in lowered or "maximumcut" in lowered:
            return "maxcut"
        if "tsp" in lowered or "travelling" in lowered or "traveling" in lowered:
            return "tsp"

    parameters = instance.get("parameters", {}) if isinstance(instance, dict) else {}
    data = instance.get("data", {}) if isinstance(instance, dict) else {}

    if (
        get_first(parameters, ["n_cities", "num_cities", "number_of_cities"]) is not None
        or get_first(data, ["distance_matrix", "distances"]) is not None
        or find_named_value(instance, ["distance_matrix"]) is not None
    ):
        return "tsp"

    return "maxcut"


# ---------------------------------------------------------------------------
# TSP
# ---------------------------------------------------------------------------

def tsp_variables(tour, n):
    city_at_step = list(tour)
    result = {}
    for city in range(n):
        for step in range(n):
            result[f"x_{city}_{step}"] = int(city_at_step[step] == city)
    return result


def tsp_cost(tour, distances):
    n = len(tour)
    if n <= 1:
        return 0.0
    return float(
        sum(
            distances[tour[index]][tour[(index + 1) % n]]
            for index in range(n)
        )
    )


def get_tsp_data(instance):
    parameters = instance.get("parameters", {})
    data = instance.get("data", {})

    n_value = get_first(
        parameters,
        ["n_cities", "num_cities", "number_of_cities", "n"],
    )
    coordinates = get_first(data, ["coordinates", "city_coordinates", "points"])
    matrix = get_first(data, ["distance_matrix", "distances", "cost_matrix"])

    if matrix is None:
        matrix = find_named_value(instance, ["distance_matrix", "cost_matrix"])

    if n_value is None:
        if matrix is not None:
            n_value = len(matrix)
        elif coordinates is not None:
            n_value = len(coordinates)
        else:
            raise ValueError("Could not determine the number of TSP cities")

    n = int(n_value)
    if n <= 0:
        raise ValueError("n_cities must be positive")

    if matrix is None:
        if coordinates is None:
            coordinates = find_named_value(
                instance, ["coordinates", "city_coordinates", "points"]
            )
        if coordinates is None or len(coordinates) != n:
            raise ValueError("TSP instance has neither a valid matrix nor coordinates")

        matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                dx = float(coordinates[i][0]) - float(coordinates[j][0])
                dy = float(coordinates[i][1]) - float(coordinates[j][1])
                row.append(math.hypot(dx, dy))
            matrix.append(row)

    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError("distance_matrix must have shape n_cities by n_cities")

    distances = [[float(value) for value in row] for row in matrix]
    return n, distances


def solve_tsp(instance, deadline, logger):
    n, distances = get_tsp_data(instance)

    best_tour = list(range(n))
    best_value = tsp_cost(best_tour, distances)

    def report(candidate):
        nonlocal best_tour, best_value
        value = tsp_cost(candidate, distances)
        if value < best_value - 1e-12:
            best_tour = list(candidate)
            best_value = value
            if logger:
                logger.log_solution(
                    float(best_value), tsp_variables(best_tour, n)
                )
            return True
        return False

    if logger:
        logger.log_solution(float(best_value), tsp_variables(best_tour, n))

    if n <= 2 or time.monotonic() >= deadline:
        return best_value, best_tour

    # Nearest-neighbor construction from each possible starting city.
    for start in range(n):
        if time.monotonic() >= deadline:
            break

        remaining = set(range(n))
        remaining.remove(start)
        tour = [start]

        while remaining:
            current = tour[-1]
            next_city = min(
                remaining,
                key=lambda city: (distances[current][city], city),
            )
            tour.append(next_city)
            remaining.remove(next_city)

        # Rotate the representation to start at city 0.
        position = tour.index(0)
        tour = tour[position:] + tour[:position]
        report(tour)

    # Exact enumeration is especially appropriate for the benchmark's
    # four-city instances. Fix city 0 to remove rotational symmetry.
    if n <= 11 and time.monotonic() < deadline:
        counter = 0
        for suffix in itertools.permutations(range(1, n)):
            candidate = [0] + list(suffix)
            report(candidate)
            counter += 1
            if counter % 256 == 0 and time.monotonic() >= deadline:
                break
        return best_value, best_tour

    # Generic 2-opt fallback for larger schema-valid TSP instances.
    route = list(best_tour)
    while time.monotonic() < deadline:
        best_delta = -1e-12
        best_move = None
        checks = 0

        for i in range(1, n - 1):
            a = route[i - 1]
            b = route[i]
            for k in range(i + 1, n):
                c = route[k]
                d = route[(k + 1) % n]
                delta = (
                    distances[a][c]
                    + distances[b][d]
                    - distances[a][b]
                    - distances[c][d]
                )
                if delta < best_delta:
                    best_delta = delta
                    best_move = (i, k)

                checks += 1
                if checks % 512 == 0 and time.monotonic() >= deadline:
                    break
            if time.monotonic() >= deadline:
                break

        if best_move is None:
            break

        i, k = best_move
        route[i:k + 1] = reversed(route[i:k + 1])
        report(route)

    return best_value, best_tour


# ---------------------------------------------------------------------------
# MaxCut input handling
# ---------------------------------------------------------------------------

def parse_edge_entry(entry):
    if isinstance(entry, dict):
        u = get_first(
            entry,
            ["u", "source", "from", "node1", "node_1", "i", "tail", "first"],
        )
        v = get_first(
            entry,
            ["v", "target", "to", "node2", "node_2", "j", "head", "second"],
        )
        weight = get_first(entry, ["weight", "w", "cost", "value"], 1.0)

        if u is None or v is None:
            endpoints = get_first(entry, ["nodes", "endpoints", "vertices"])
            if isinstance(endpoints, (list, tuple)) and len(endpoints) >= 2:
                u, v = endpoints[0], endpoints[1]

        if u is None or v is None:
            return None
        return u, v, float(weight)

    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        weight = float(entry[2]) if len(entry) >= 3 else 1.0
        return entry[0], entry[1], weight

    return None


def extract_maxcut_data(instance):
    parameters = instance.get("parameters", {}) if isinstance(instance, dict) else {}
    data = instance.get("data", {}) if isinstance(instance, dict) else {}

    n_value = get_first(
        parameters,
        [
            "n_nodes",
            "num_nodes",
            "number_of_nodes",
            "node_count",
            "n_vertices",
            "num_vertices",
            "number_of_vertices",
            "vertex_count",
            "n",
        ],
    )

    if n_value is None:
        n_value = get_first(
            data,
            [
                "n_nodes",
                "num_nodes",
                "number_of_nodes",
                "node_count",
                "n_vertices",
                "num_vertices",
                "number_of_vertices",
                "vertex_count",
            ],
        )

    raw_edges = get_first(
        data,
        [
            "edges",
            "edge_list",
            "edgelist",
            "weighted_edges",
            "links",
            "arcs",
        ],
    )
    if raw_edges is None:
        raw_edges = find_named_value(
            instance,
            ["edges", "edge_list", "edgelist", "weighted_edges", "links"],
        )

    matrix = get_first(
        data,
        ["adjacency_matrix", "adj_matrix", "weight_matrix", "graph_matrix"],
    )
    if matrix is None:
        matrix = find_named_value(
            instance,
            ["adjacency_matrix", "adj_matrix", "weight_matrix"],
        )

    parsed_edges = []

    if isinstance(raw_edges, list):
        for entry in raw_edges:
            parsed = parse_edge_entry(entry)
            if parsed is not None:
                parsed_edges.append(parsed)

    elif isinstance(raw_edges, dict):
        # Support adjacency-list dictionaries such as {"0": [1, 2], ...}.
        for raw_u, neighbors in raw_edges.items():
            if isinstance(neighbors, dict):
                for raw_v, weight in neighbors.items():
                    if isinstance(weight, dict):
                        weight = get_first(weight, ["weight", "w", "value"], 1.0)
                    parsed_edges.append((raw_u, raw_v, float(weight)))
            elif isinstance(neighbors, list):
                for neighbor in neighbors:
                    if isinstance(neighbor, dict):
                        raw_v = get_first(
                            neighbor,
                            ["v", "target", "to", "node", "neighbor", "id"],
                        )
                        weight = get_first(
                            neighbor, ["weight", "w", "value"], 1.0
                        )
                        if raw_v is not None:
                            parsed_edges.append((raw_u, raw_v, float(weight)))
                    elif isinstance(neighbor, (list, tuple)):
                        if neighbor:
                            weight = float(neighbor[1]) if len(neighbor) > 1 else 1.0
                            parsed_edges.append((raw_u, neighbor[0], weight))
                    else:
                        parsed_edges.append((raw_u, neighbor, 1.0))

    if not parsed_edges and matrix is not None:
        if n_value is None:
            n_value = len(matrix)
        n_matrix = int(n_value)
        if len(matrix) != n_matrix:
            raise ValueError("Invalid MaxCut adjacency matrix")

        for i in range(n_matrix):
            if len(matrix[i]) != n_matrix:
                raise ValueError("Invalid MaxCut adjacency matrix")
            for j in range(i + 1, n_matrix):
                weight = float(matrix[i][j])
                if weight != 0.0:
                    parsed_edges.append((i, j, weight))

    if n_value is not None:
        n = int(n_value)
    elif parsed_edges:
        integer_labels = []
        for u, v, _ in parsed_edges:
            try:
                integer_labels.extend([int(u), int(v)])
            except (TypeError, ValueError):
                integer_labels = []
                break

        if integer_labels:
            minimum = min(integer_labels)
            maximum = max(integer_labels)
            n = maximum + 1 if minimum == 0 else maximum
        else:
            labels = sorted(
                {str(u) for u, _, _ in parsed_edges}
                | {str(v) for _, v, _ in parsed_edges}
            )
            n = len(labels)
    else:
        nodes = get_first(data, ["nodes", "vertices"])
        if nodes is None:
            nodes = find_named_value(instance, ["nodes", "vertices"])
        if isinstance(nodes, list):
            n = len(nodes)
        else:
            raise ValueError("Could not determine MaxCut graph size")

    if n <= 0:
        raise ValueError("MaxCut graph must contain at least one node")

    # Convert arbitrary or one-based node identifiers into zero-based indices.
    labels = []
    integer_conversion_ok = True
    integer_edges = []

    for raw_u, raw_v, weight in parsed_edges:
        try:
            u = int(raw_u)
            v = int(raw_v)
            integer_edges.append((u, v, float(weight)))
            labels.extend((u, v))
        except (TypeError, ValueError):
            integer_conversion_ok = False
            break

    converted_edges = []

    if integer_conversion_ok:
        one_based = (
            bool(labels)
            and 0 not in labels
            and min(labels) >= 1
            and max(labels) <= n
        )

        for u, v, weight in integer_edges:
            if one_based:
                u -= 1
                v -= 1
            if 0 <= u < n and 0 <= v < n and u != v and weight != 0.0:
                converted_edges.append((u, v, weight))
    else:
        unique_labels = []
        seen = set()
        for raw_u, raw_v, _ in parsed_edges:
            for label in (str(raw_u), str(raw_v)):
                if label not in seen:
                    seen.add(label)
                    unique_labels.append(label)

        unique_labels.sort()
        label_to_index = {
            label: index for index, label in enumerate(unique_labels)
        }
        if len(unique_labels) > n:
            n = len(unique_labels)

        for raw_u, raw_v, weight in parsed_edges:
            u = label_to_index[str(raw_u)]
            v = label_to_index[str(raw_v)]
            if u != v and weight != 0.0:
                converted_edges.append((u, v, float(weight)))

    # Merge duplicate edges. If an undirected adjacency list contains both
    # directions, avoid accidentally counting every edge twice.
    directed_weights = {}
    for u, v, weight in converted_edges:
        directed_weights[(u, v)] = directed_weights.get((u, v), 0.0) + weight

    merged = {}
    handled = set()
    for (u, v), weight_uv in directed_weights.items():
        key = (min(u, v), max(u, v))
        if key in handled:
            continue
        handled.add(key)

        reverse_weight = directed_weights.get((v, u))
        if reverse_weight is not None and u != v:
            # A symmetric adjacency list normally repeats each undirected edge.
            if abs(weight_uv - reverse_weight) <= 1e-12:
                weight = weight_uv
            else:
                weight = 0.5 * (weight_uv + reverse_weight)
        else:
            weight = weight_uv

        if weight != 0.0:
            merged[key] = merged.get(key, 0.0) + weight

    edges = [(u, v, weight) for (u, v), weight in merged.items()]
    return n, edges


# ---------------------------------------------------------------------------
# MaxCut solver
# ---------------------------------------------------------------------------

def maxcut_variables(bits):
    return {f"x_{node}": int(bits[node]) for node in range(len(bits))}


def cut_value(bits, edges):
    return float(
        sum(weight for u, v, weight in edges if bits[u] != bits[v])
    )


def solve_maxcut(instance, deadline, logger):
    n, edges = extract_maxcut_data(instance)

    adjacency = [[] for _ in range(n)]
    for u, v, weight in edges:
        adjacency[u].append((v, weight))
        adjacency[v].append((u, weight))

    best_bits = [0] * n
    best_value = 0.0

    def report(bits, value):
        nonlocal best_bits, best_value
        if value > best_value + 1e-9:
            best_bits = list(bits)
            best_value = float(value)
            if logger:
                logger.log_solution(
                    float(best_value), maxcut_variables(best_bits)
                )
            return True
        return False

    if logger:
        logger.log_solution(0.0, maxcut_variables(best_bits))

    if not edges or time.monotonic() >= deadline:
        return best_value, best_bits

    rng = random.Random(0)

    def initialize_assignment(restart):
        if restart == 0:
            # Deterministic greedy insertion.
            bits = [0] * n
            order = sorted(
                range(n),
                key=lambda node: (-len(adjacency[node]), node),
            )
            assigned = [False] * n

            for node in order:
                weight_if_zero = 0.0
                weight_if_one = 0.0
                for neighbor, weight in adjacency[node]:
                    if not assigned[neighbor]:
                        continue
                    if bits[neighbor] != 0:
                        weight_if_zero += weight
                    if bits[neighbor] != 1:
                        weight_if_one += weight
                bits[node] = int(weight_if_one > weight_if_zero)
                assigned[node] = True
            return bits

        return [rng.getrandbits(1) for _ in range(n)]

    restart = 0

    while time.monotonic() < deadline:
        bits = initialize_assignment(restart)
        restart += 1
        current_value = cut_value(bits, edges)
        report(bits, current_value)

        # gain[v] is the objective change caused by flipping v.
        gains = [0.0] * n
        for node in range(n):
            gain = 0.0
            side = bits[node]
            for neighbor, weight in adjacency[node]:
                if bits[neighbor] == side:
                    gain += weight
                else:
                    gain -= weight
            gains[node] = gain

        tabu_until = [0] * n
        iteration = 0
        iterations_without_global_improvement = 0
        iteration_limit = max(500, 30 * n)

        while (
            iteration < iteration_limit
            and iterations_without_global_improvement < max(250, 12 * n)
        ):
            iteration += 1
            if iteration % 32 == 0 and time.monotonic() >= deadline:
                break

            selected = -1
            selected_gain = -float("inf")
            tie_count = 0

            for node in range(n):
                gain = gains[node]
                aspiration = current_value + gain > best_value + 1e-9
                allowed = iteration >= tabu_until[node] or aspiration
                if not allowed:
                    continue

                if gain > selected_gain + 1e-12:
                    selected = node
                    selected_gain = gain
                    tie_count = 1
                elif abs(gain - selected_gain) <= 1e-12:
                    tie_count += 1
                    if rng.randrange(tie_count) == 0:
                        selected = node

            if selected < 0:
                selected = rng.randrange(n)
                selected_gain = gains[selected]

            old_side = bits[selected]
            current_value += selected_gain

            # Update neighbors' flip gains using their relationship to the
            # selected node before the selected node is flipped.
            for neighbor, weight in adjacency[selected]:
                if bits[neighbor] == old_side:
                    gains[neighbor] -= 2.0 * weight
                else:
                    gains[neighbor] += 2.0 * weight

            bits[selected] = 1 - old_side
            gains[selected] = -gains[selected]

            tenure = 5 + rng.randrange(max(2, min(15, n // 10 + 2)))
            tabu_until[selected] = iteration + tenure

            if report(bits, current_value):
                iterations_without_global_improvement = 0
            else:
                iterations_without_global_improvement += 1

        # Continue with a new randomized restart if time remains.

    # Recompute to remove any accumulated floating-point update error.
    best_value = cut_value(best_bits, edges)
    return best_value, best_bits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as input_file:
        instance = json.load(input_file)

    problem_type = detect_problem_type(instance)
    sense = "minimize" if problem_type == "tsp" else "maximize"
    logger = SolutionLogger(args.log_path, sense=sense) if args.log_path else None

    if problem_type == "tsp":
        objective_value, tour = solve_tsp(instance, deadline, logger)
        n, distances = get_tsp_data(instance)
        objective_value = tsp_cost(tour, distances)
        output = {
            "objective_value": float(objective_value),
            "solution": tsp_variables(tour, n),
            "tour": [int(city) for city in tour],
        }
    else:
        objective_value, bits = solve_maxcut(instance, deadline, logger)
        output = {
            "objective_value": float(objective_value),
            "solution": maxcut_variables(bits),
        }

    write_json(args.solution_path, output)


if __name__ == "__main__":
    main()