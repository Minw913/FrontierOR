import argparse
import heapq
import json
import math
import os
import time
from collections import defaultdict

from solution_logger import SolutionLogger


def write_solution(path, solution):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(solution, f, indent=2)


def build_arc_distances(instance):
    distances = {}
    for arc in instance.get("arcs", []):
        u = int(arc["from"])
        v = int(arc["to"])
        d = float(arc["distance"])
        key = (u, v) if u <= v else (v, u)
        if key not in distances or d < distances[key]:
            distances[key] = d
    return distances


def path_edge_distance(u, v, arc_distances, coordinates):
    key = (u, v) if u <= v else (v, u)
    if key in arc_distances:
        return arc_distances[key]

    # Defensive fallback for malformed or incomplete arc data.
    x1, y1 = coordinates[u]
    x2, y2 = coordinates[v]
    return math.hypot(x1 - x2, y1 - y2)


def make_route_requirements(path, vehicle_range, arc_distances,
                            coordinates, node_to_index):
    """
    Return:
      []   if the route needs no stations,
      None if the route is impossible regardless of station choices,
      otherwise a list of distinct frozensets. Every set must contain at
      least one selected station.
    """
    if len(path) <= 1:
        return []

    cumulative = [0.0]
    for u, v in zip(path[:-1], path[1:]):
        cumulative.append(
            cumulative[-1] +
            path_edge_distance(u, v, arc_distances, coordinates)
        )

    one_way_distance = cumulative[-1]

    # Outbound occurrences followed by return occurrences, excluding a
    # duplicate destination occurrence.
    route_nodes = list(path)
    route_positions = list(cumulative)
    for i in range(len(path) - 2, -1, -1):
        route_nodes.append(path[i])
        route_positions.append(2.0 * one_way_distance - cumulative[i])

    requirements = []
    seen_requirements = set()

    # Maintain node multiplicities in the interval of station occurrences
    # that can refuel before the current directed arc and reach its end.
    counts = defaultdict(int)
    left = 0
    eps = 1e-9

    for k in range(len(route_nodes) - 1):
        counts[route_nodes[k]] += 1
        arc_end_position = route_positions[k + 1]

        lower_bound = arc_end_position - vehicle_range
        while left <= k and route_positions[left] < lower_bound - eps:
            old_node = route_nodes[left]
            counts[old_node] -= 1
            if counts[old_node] == 0:
                del counts[old_node]
            left += 1

        # The initial full tank automatically covers these arcs.
        if arc_end_position <= vehicle_range + eps:
            continue

        service_set = frozenset(
            node_to_index[node_id] for node_id in counts.keys()
            if node_id in node_to_index
        )
        if not service_set:
            return None

        if service_set not in seen_requirements:
            seen_requirements.add(service_set)
            requirements.append(service_set)

    return requirements


def evaluate_solution(selected_indices, route_requirements, flows):
    selected = set(selected_indices)
    refueled = []
    objective = 0.0

    for pair_index, requirements in enumerate(route_requirements):
        if requirements is None:
            continue

        feasible = True
        for service_set in requirements:
            if selected.isdisjoint(service_set):
                feasible = False
                break

        if feasible:
            refueled.append(pair_index)
            objective += flows[pair_index]

    return float(objective), refueled


def make_solution(selected_indices, node_ids, route_requirements, flows):
    objective, refueled = evaluate_solution(
        selected_indices, route_requirements, flows
    )
    stations = sorted(node_ids[i] for i in selected_indices)
    return {
        "objective_value": float(objective),
        "stations": stations,
        "refueled_od_pairs": refueled,
    }


def heuristic_station_set(n, p, route_requirements, flows):
    """
    Greedily covers weighted arc-service requirements. A lazy heap is used so
    selecting a station removes the value of requirements already covered.
    """
    if p <= 0:
        return []
    if p >= n:
        return list(range(n))

    station_scores = [0.0] * n
    station_entries = [[] for _ in range(n)]
    entry_nodes = []
    entry_weights = []

    for pair_index, requirements in enumerate(route_requirements):
        if not requirements:
            continue

        route_weight = max(0.0, flows[pair_index])
        if route_weight == 0.0:
            continue

        # Give each route equal total surrogate weight across its requirements.
        per_requirement = route_weight / len(requirements)
        for service_set in requirements:
            weight = per_requirement
            entry_id = len(entry_nodes)
            nodes = tuple(service_set)
            entry_nodes.append(nodes)
            entry_weights.append(weight)

            share = weight / max(1, len(nodes))
            for station in nodes:
                station_scores[station] += share
                station_entries[station].append(entry_id)

    heap = [(-station_scores[i], i, station_scores[i]) for i in range(n)]
    heapq.heapify(heap)
    active_entry = [True] * len(entry_nodes)
    selected = []
    selected_set = set()

    while len(selected) < p and heap:
        neg_score, station, stored_score = heapq.heappop(heap)
        if station in selected_set:
            continue
        if abs(stored_score - station_scores[station]) > 1e-10:
            heapq.heappush(
                heap, (-station_scores[station], station,
                       station_scores[station])
            )
            continue

        selected.append(station)
        selected_set.add(station)

        # Requirements covered by this station no longer contribute to the
        # surrogate scores of any station.
        for entry_id in station_entries[station]:
            if not active_entry[entry_id]:
                continue
            active_entry[entry_id] = False
            nodes = entry_nodes[entry_id]
            deduction = entry_weights[entry_id] / max(1, len(nodes))
            for other in nodes:
                if other not in selected_set:
                    station_scores[other] -= deduction

    if len(selected) < p:
        remaining = sorted(
            (i for i in range(n) if i not in selected_set),
            key=lambda i: (-station_scores[i], i)
        )
        selected.extend(remaining[:p - len(selected)])

    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = (
        SolutionLogger(args.log_path, sense="maximize")
        if args.log_path else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    nodes = instance["nodes"]
    node_ids = [int(node["id"]) for node in nodes]
    node_to_index = {node_id: i for i, node_id in enumerate(node_ids)}
    coordinates = {
        int(node["id"]): (float(node["x"]), float(node["y"]))
        for node in nodes
    }

    n = len(node_ids)
    p = int(instance["parameters"]["p"])
    vehicle_range = float(instance["parameters"]["R"])

    if p < 0 or p > n:
        raise ValueError(
            f"Station count p={p} is invalid for a network with {n} nodes."
        )

    arc_distances = build_arc_distances(instance)
    od_pairs = instance.get("od_pairs", [])
    flows = [float(pair["flow"]) for pair in od_pairs]

    route_requirements = []
    for pair in od_pairs:
        path = [int(v) for v in pair["shortest_path"]]
        route_requirements.append(
            make_route_requirements(
                path,
                vehicle_range,
                arc_distances,
                coordinates,
                node_to_index,
            )
        )

    # Always establish and log a complete feasible incumbent.
    arbitrary_indices = list(range(p))
    best_solution = make_solution(
        arbitrary_indices, node_ids, route_requirements, flows
    )
    best_objective = best_solution["objective_value"]

    if logger:
        logger.log_solution(best_objective, best_solution)

    heuristic_indices = heuristic_station_set(
        n, p, route_requirements, flows
    )
    heuristic_solution = make_solution(
        heuristic_indices, node_ids, route_requirements, flows
    )
    if heuristic_solution["objective_value"] > best_objective + 1e-9:
        best_solution = heuristic_solution
        best_objective = heuristic_solution["objective_value"]
        if logger:
            logger.log_solution(best_objective, best_solution)

    remaining_time = deadline - time.monotonic()
    if remaining_time <= 0.05:
        write_solution(args.solution_path, best_solution)
        return

    try:
        import gurobipy as gp
        from gurobipy import GRB

        model = gp.Model("round_trip_refueling_location")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = max(0.01, remaining_time)

        x = model.addVars(n, vtype=GRB.BINARY, name="station")
        model.addConstr(
            gp.quicksum(x[j] for j in range(n)) == p,
            name="station_count"
        )

        # Group OD pairs with identical station requirements.
        automatic_flow = 0.0
        grouped = {}
        for pair_index, requirements in enumerate(route_requirements):
            if requirements is None:
                continue
            if len(requirements) == 0:
                automatic_flow += flows[pair_index]
                continue

            signature = tuple(sorted(
                (tuple(sorted(service_set))
                 for service_set in requirements),
                key=lambda values: (len(values), values)
            ))
            if signature not in grouped:
                grouped[signature] = 0.0
            grouped[signature] += flows[pair_index]

        y_vars = []
        group_signatures = []
        group_weights = []

        for group_index, (signature, weight) in enumerate(grouped.items()):
            # Continuous y is exact because x is binary and all upper bounds
            # are integer sums: y becomes 1 iff every requirement is covered.
            y = model.addVar(
                lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                name=f"refueled_group_{group_index}"
            )
            y_vars.append(y)
            group_signatures.append(signature)
            group_weights.append(weight)

            for requirement_index, service_tuple in enumerate(signature):
                model.addConstr(
                    y <= gp.quicksum(x[j] for j in service_tuple),
                    name=f"cover_{group_index}_{requirement_index}"
                )

        model.setObjective(
            automatic_flow + gp.quicksum(
                group_weights[g] * y_vars[g]
                for g in range(len(y_vars))
            ),
            GRB.MAXIMIZE,
        )

        # Supply the best heuristic solution as a MIP start.
        warm_set = {
            node_to_index[node_id]
            for node_id in best_solution["stations"]
        }
        for j in range(n):
            x[j].Start = 1.0 if j in warm_set else 0.0

        for g, signature in enumerate(group_signatures):
            covered = all(
                not warm_set.isdisjoint(service_tuple)
                for service_tuple in signature
            )
            y_vars[g].Start = 1.0 if covered else 0.0

        incumbent = {
            "objective": best_objective,
            "solution": best_solution,
        }

        def callback(cb_model, where):
            if time.monotonic() >= deadline:
                cb_model.terminate()
                return

            if where != GRB.Callback.MIPSOL:
                return

            try:
                mip_objective = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if mip_objective <= incumbent["objective"] + 1e-8:
                    return

                values = cb_model.cbGetSolution(x)
                chosen = sorted(
                    range(n), key=lambda j: values[j], reverse=True
                )[:p]
                candidate = make_solution(
                    chosen, node_ids, route_requirements, flows
                )

                if (candidate["objective_value"] >
                        incumbent["objective"] + 1e-9):
                    incumbent["objective"] = candidate["objective_value"]
                    incumbent["solution"] = candidate
                    if logger:
                        logger.log_solution(
                            candidate["objective_value"], candidate
                        )
            except gp.GurobiError:
                return

        model.optimize(callback)

        if model.SolCount > 0:
            chosen = sorted(
                range(n), key=lambda j: x[j].X, reverse=True
            )[:p]
            final_candidate = make_solution(
                chosen, node_ids, route_requirements, flows
            )
            if (final_candidate["objective_value"] >
                    incumbent["objective"] + 1e-9):
                incumbent["objective"] = final_candidate["objective_value"]
                incumbent["solution"] = final_candidate
                if logger:
                    logger.log_solution(
                        final_candidate["objective_value"],
                        final_candidate
                    )

        best_solution = incumbent["solution"]

    except Exception:
        # The heuristic incumbent remains a valid solution if Gurobi is
        # unavailable, its license fails, or optimization is interrupted.
        pass

    write_solution(args.solution_path, best_solution)


if __name__ == "__main__":
    main()