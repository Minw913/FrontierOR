import argparse
import heapq
import json
import math
import random
import time
from collections import defaultdict

from solution_logger import SolutionLogger


INF = float("inf")
EPS = 1e-9


def dijkstra(adjacency, source, target_index=None, need_parent=False):
    dist = {source: 0}
    parent = {} if need_parent else None
    heap = [(0, source)]

    while heap:
        current_dist, node = heapq.heappop(heap)
        if current_dist != dist.get(node, INF):
            continue

        for neighbor, cost, edge_id in adjacency.get(node, ()):
            new_dist = current_dist + cost
            old_dist = dist.get(neighbor, INF)
            if new_dist < old_dist:
                dist[neighbor] = new_dist
                if need_parent:
                    parent[neighbor] = (node, edge_id)
                heapq.heappush(heap, (new_dist, neighbor))
            elif need_parent and new_dist == old_dist:
                old_parent = parent.get(neighbor)
                candidate = (node, edge_id)
                if old_parent is None or candidate < old_parent:
                    parent[neighbor] = candidate

    if target_index is None:
        return dist, parent

    result = [INF] * len(target_index)
    for node, index in target_index.items():
        result[index] = dist.get(node, INF)
    return result, parent


def farthest_point_seeds(tasks, num_seeds, depot, distance):
    if not tasks or num_seeds <= 0:
        return []

    first = max(
        range(len(tasks)),
        key=lambda i: min(
            distance(depot, tasks[i]["u"]),
            distance(depot, tasks[i]["v"]),
        ),
    )
    seeds = [first]

    while len(seeds) < min(num_seeds, len(tasks)):
        best_task = None
        best_value = -1
        for i, task in enumerate(tasks):
            if i in seeds:
                continue
            closest = INF
            for seed_index in seeds:
                seed = tasks[seed_index]
                d = min(
                    distance(task["u"], seed["u"]),
                    distance(task["u"], seed["v"]),
                    distance(task["v"], seed["u"]),
                    distance(task["v"], seed["v"]),
                )
                closest = min(closest, d)
            if closest > best_value:
                best_value = closest
                best_task = i
        if best_task is None:
            break
        seeds.append(best_task)

    while len(seeds) < num_seeds:
        seeds.append(seeds[len(seeds) % len(seeds)])
    return seeds


def solve_capacitated_assignment(
    tasks, num_vehicles, capacity, depot, distance, deadline
):
    n = len(tasks)
    if n == 0:
        return [[] for _ in range(num_vehicles)]

    seeds = farthest_point_seeds(tasks, num_vehicles, depot, distance)

    assignment_cost = [[0.0] * num_vehicles for _ in range(n)]
    for i, task in enumerate(tasks):
        for k in range(num_vehicles):
            seed = tasks[seeds[k]]
            d = min(
                distance(task["u"], seed["u"]),
                distance(task["u"], seed["v"]),
                distance(task["v"], seed["u"]),
                distance(task["v"], seed["v"]),
            )
            if not math.isfinite(d):
                d = 1e12
            assignment_cost[i][k] = float(d) + 1e-7 * (i * num_vehicles + k)

    remaining = max(0.01, deadline - time.monotonic())

    try:
        import gurobipy as gp
        from gurobipy import GRB

        model = gp.Model("capacitated_task_assignment")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(0.01, min(remaining * 0.30, 20.0))

        x = model.addVars(n, num_vehicles, vtype=GRB.BINARY, name="x")

        for i in range(n):
            model.addConstr(gp.quicksum(x[i, k] for k in range(num_vehicles)) == 1)

        for k in range(num_vehicles):
            model.addConstr(
                gp.quicksum(tasks[i]["demand"] * x[i, k] for i in range(n))
                <= capacity
            )

        model.setObjective(
            gp.quicksum(
                assignment_cost[i][k] * x[i, k]
                for i in range(n)
                for k in range(num_vehicles)
            ),
            GRB.MINIMIZE,
        )
        model.optimize()

        if model.SolCount > 0:
            bins = [[] for _ in range(num_vehicles)]
            for i in range(n):
                chosen = max(range(num_vehicles), key=lambda k: x[i, k].X)
                bins[chosen].append(i)
            return bins
    except Exception:
        pass

    # Best-fit-decreasing fallback.
    bins = [[] for _ in range(num_vehicles)]
    loads = [0] * num_vehicles
    order = sorted(range(n), key=lambda i: (-tasks[i]["demand"], i))

    for i in order:
        feasible = [
            k
            for k in range(num_vehicles)
            if loads[k] + tasks[i]["demand"] <= capacity
        ]
        if not feasible:
            return None
        chosen = min(
            feasible,
            key=lambda k: (
                assignment_cost[i][k],
                capacity - loads[k] - tasks[i]["demand"],
                k,
            ),
        )
        bins[chosen].append(i)
        loads[chosen] += tasks[i]["demand"]

    return bins


def route_evaluation(sequence, tasks, depot, distance, return_orientations=False):
    if not sequence:
        return (0, []) if return_orientations else 0

    first = tasks[sequence[0]]
    starts = (first["u"], first["v"])
    ends = (first["v"], first["u"])

    dp = [distance(depot, starts[0]), distance(depot, starts[1])]
    parents = [] if return_orientations else None

    for position in range(1, len(sequence)):
        previous = tasks[sequence[position - 1]]
        current = tasks[sequence[position]]

        previous_ends = (previous["v"], previous["u"])
        current_starts = (current["u"], current["v"])

        new_dp = [INF, INF]
        predecessor = [0, 0]

        for orientation in (0, 1):
            for previous_orientation in (0, 1):
                candidate = (
                    dp[previous_orientation]
                    + distance(
                        previous_ends[previous_orientation],
                        current_starts[orientation],
                    )
                )
                if candidate < new_dp[orientation]:
                    new_dp[orientation] = candidate
                    predecessor[orientation] = previous_orientation

        dp = new_dp
        if return_orientations:
            parents.append(predecessor)

    last = tasks[sequence[-1]]
    last_ends = (last["v"], last["u"])
    total_candidates = [
        dp[orientation] + distance(last_ends[orientation], depot)
        for orientation in (0, 1)
    ]
    final_orientation = 0 if total_candidates[0] <= total_candidates[1] else 1
    best_cost = total_candidates[final_orientation]

    if not return_orientations:
        return best_cost

    orientations = [0] * len(sequence)
    orientations[-1] = final_orientation
    for position in range(len(sequence) - 1, 0, -1):
        orientations[position - 1] = parents[position - 1][orientations[position]]

    return best_cost, orientations


def nearest_order(task_indices, tasks, depot, distance, rng=None, randomized=False):
    if not task_indices:
        return []

    remaining = set(task_indices)
    sequence = []
    current = depot

    while remaining:
        ranked = []
        for index in remaining:
            task = tasks[index]
            d0 = distance(current, task["u"])
            d1 = distance(current, task["v"])
            best = min(d0, d1)
            return_distance = min(
                distance(task["u"], depot), distance(task["v"], depot)
            )
            ranked.append((best, -return_distance, index, 0 if d0 <= d1 else 1))

        ranked.sort()
        if randomized and rng is not None:
            width = min(4, len(ranked))
            chosen = ranked[rng.randrange(width)]
        else:
            chosen = ranked[0]

        _, _, index, orientation = chosen
        sequence.append(index)
        remaining.remove(index)
        task = tasks[index]
        current = task["v"] if orientation == 0 else task["u"]

    return sequence


def split_giant_tour(
    giant_sequence, tasks, num_vehicles, capacity, depot, distance, deadline
):
    n = len(giant_sequence)
    if n == 0:
        return [[] for _ in range(num_vehicles)]

    if n > 800:
        return None

    prefix_demand = [0] * (n + 1)
    for i, task_index in enumerate(giant_sequence):
        prefix_demand[i + 1] = prefix_demand[i] + tasks[task_index]["demand"]

    segment_cost = [[INF] * (n + 1) for _ in range(n)]

    for start in range(n):
        if time.monotonic() >= deadline:
            return None

        first_task = tasks[giant_sequence[start]]
        dp0 = distance(depot, first_task["u"])
        dp1 = distance(depot, first_task["v"])

        for end in range(start, n):
            if prefix_demand[end + 1] - prefix_demand[start] > capacity:
                break

            current = tasks[giant_sequence[end]]
            segment_cost[start][end + 1] = min(
                dp0 + distance(current["v"], depot),
                dp1 + distance(current["u"], depot),
            )

            if end + 1 >= n:
                continue

            nxt = tasks[giant_sequence[end + 1]]
            new_dp0 = min(
                dp0 + distance(current["v"], nxt["u"]),
                dp1 + distance(current["u"], nxt["u"]),
            )
            new_dp1 = min(
                dp0 + distance(current["v"], nxt["v"]),
                dp1 + distance(current["u"], nxt["v"]),
            )
            dp0, dp1 = new_dp0, new_dp1

    dp = [[INF] * (n + 1) for _ in range(num_vehicles + 1)]
    predecessor = [[-1] * (n + 1) for _ in range(num_vehicles + 1)]
    dp[0][0] = 0

    for vehicle_count in range(1, num_vehicles + 1):
        for end in range(1, n + 1):
            for start in range(end):
                cost = segment_cost[start][end]
                if math.isfinite(cost) and math.isfinite(dp[vehicle_count - 1][start]):
                    candidate = dp[vehicle_count - 1][start] + cost
                    if candidate < dp[vehicle_count][end]:
                        dp[vehicle_count][end] = candidate
                        predecessor[vehicle_count][end] = start

    feasible_counts = [
        k for k in range(1, num_vehicles + 1) if math.isfinite(dp[k][n])
    ]
    if not feasible_counts:
        return None

    used = min(feasible_counts, key=lambda k: dp[k][n])
    routes = []
    end = n
    k = used

    while k > 0:
        start = predecessor[k][end]
        if start < 0:
            return None
        routes.append(giant_sequence[start:end])
        end = start
        k -= 1

    routes.reverse()
    routes.extend([[] for _ in range(num_vehicles - len(routes))])
    return routes


def local_improvement(
    routes, tasks, capacity, depot, distance, deadline, rng, on_improvement
):
    loads = [
        sum(tasks[index]["demand"] for index in route)
        for route in routes
    ]
    costs = [route_evaluation(route, tasks, depot, distance) for route in routes]
    total_cost = sum(costs)

    stagnation = 0
    iteration_limit = max(1000, 30 * sum(len(route) for route in routes))

    for _ in range(iteration_limit):
        if time.monotonic() >= deadline:
            break

        nonempty = [k for k, route in enumerate(routes) if route]
        if not nonempty:
            break

        move_type = rng.randrange(3)
        improved = False

        if move_type == 0:
            # Intra-route reversal.
            k = rng.choice(nonempty)
            route = routes[k]
            if len(route) >= 2:
                i = rng.randrange(len(route))
                j = rng.randrange(i + 1, len(route) + 1)
                candidate_route = route[:i] + list(reversed(route[i:j])) + route[j:]
                candidate_cost = route_evaluation(
                    candidate_route, tasks, depot, distance
                )
                if candidate_cost + EPS < costs[k]:
                    routes[k] = candidate_route
                    total_cost += candidate_cost - costs[k]
                    costs[k] = candidate_cost
                    improved = True

        elif move_type == 1 and len(routes) >= 2:
            # Relocate a task and choose its best insertion position.
            source = rng.choice(nonempty)
            target = rng.randrange(len(routes) - 1)
            if target >= source:
                target += 1

            source_route = routes[source]
            position = rng.randrange(len(source_route))
            task_index = source_route[position]
            demand = tasks[task_index]["demand"]

            if loads[target] + demand <= capacity:
                new_source = source_route[:position] + source_route[position + 1 :]
                source_cost = route_evaluation(new_source, tasks, depot, distance)

                best_target = None
                best_target_cost = INF
                for insertion in range(len(routes[target]) + 1):
                    candidate = (
                        routes[target][:insertion]
                        + [task_index]
                        + routes[target][insertion:]
                    )
                    candidate_cost = route_evaluation(
                        candidate, tasks, depot, distance
                    )
                    if candidate_cost < best_target_cost:
                        best_target_cost = candidate_cost
                        best_target = candidate

                old_pair_cost = costs[source] + costs[target]
                new_pair_cost = source_cost + best_target_cost
                if new_pair_cost + EPS < old_pair_cost:
                    routes[source] = new_source
                    routes[target] = best_target
                    loads[source] -= demand
                    loads[target] += demand
                    costs[source] = source_cost
                    costs[target] = best_target_cost
                    total_cost += new_pair_cost - old_pair_cost
                    improved = True

        elif len(nonempty) >= 2:
            # Swap two serviced edges.
            first, second = rng.sample(nonempty, 2)
            i = rng.randrange(len(routes[first]))
            j = rng.randrange(len(routes[second]))
            a = routes[first][i]
            b = routes[second][j]

            new_first_load = loads[first] - tasks[a]["demand"] + tasks[b]["demand"]
            new_second_load = loads[second] - tasks[b]["demand"] + tasks[a]["demand"]

            if new_first_load <= capacity and new_second_load <= capacity:
                candidate_first = routes[first][:]
                candidate_second = routes[second][:]
                candidate_first[i], candidate_second[j] = b, a

                first_cost = route_evaluation(
                    candidate_first, tasks, depot, distance
                )
                second_cost = route_evaluation(
                    candidate_second, tasks, depot, distance
                )

                old_pair_cost = costs[first] + costs[second]
                new_pair_cost = first_cost + second_cost

                if new_pair_cost + EPS < old_pair_cost:
                    routes[first] = candidate_first
                    routes[second] = candidate_second
                    loads[first] = new_first_load
                    loads[second] = new_second_load
                    costs[first] = first_cost
                    costs[second] = second_cost
                    total_cost += new_pair_cost - old_pair_cost
                    improved = True

        if improved:
            stagnation = 0
            on_improvement(routes, total_cost)
        else:
            stagnation += 1
            if stagnation > max(500, 10 * sum(len(route) for route in routes)):
                break

    return routes, total_cost


def build_solution(
    routes,
    tasks,
    depot,
    distance,
    adjacency,
    edge_cost,
    service_cost_total,
):
    route_orientations = []
    path_requests = defaultdict(list)

    for vehicle, sequence in enumerate(routes):
        _, orientations = route_evaluation(
            sequence, tasks, depot, distance, return_orientations=True
        )
        route_orientations.append(orientations)

        if not sequence:
            continue

        current = depot
        for position, task_index in enumerate(sequence):
            task = tasks[task_index]
            orientation = orientations[position]
            start = task["u"] if orientation == 0 else task["v"]
            end = task["v"] if orientation == 0 else task["u"]
            if current != start:
                path_requests[current].append((vehicle, start))
            current = end

        if current != depot:
            path_requests[current].append((vehicle, depot))

    deadhead_counts = [defaultdict(int) for _ in routes]

    for source, requests in path_requests.items():
        _, parent = dijkstra(adjacency, source, need_parent=True)

        for vehicle, target in requests:
            node = target
            while node != source:
                if node not in parent:
                    raise RuntimeError(
                        f"No path exists from node {source} to node {target}"
                    )
                previous, edge_id = parent[node]
                deadhead_counts[vehicle][edge_id] += 1
                node = previous

    route_output = []
    deadhead_total = 0

    for vehicle, sequence in enumerate(routes):
        entries = []
        for edge_id in sorted(deadhead_counts[vehicle]):
            times = deadhead_counts[vehicle][edge_id]
            entries.append({"edge_id": edge_id, "times": times})
            deadhead_total += edge_cost[edge_id] * times

        route_output.append(
            {
                "vehicle": vehicle,
                "serviced_edges": [tasks[index]["edge_id"] for index in sequence],
                "deadheaded_edges": entries,
            }
        )

    objective = service_cost_total + deadhead_total
    return {
        "objective_value": float(objective),
        "routes": route_output,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    rng = random.Random(0)

    with open(args.instance_path, "r", encoding="utf-8") as file:
        instance = json.load(file)

    depot = instance["depot"]
    num_vehicles = instance["fleet"]["num_vehicles"]
    capacity = instance["fleet"]["vehicle_capacity"]

    adjacency = defaultdict(list)
    edge_cost = {}
    tasks = []
    service_cost_total = 0
    nodes = {depot}

    for edge in instance["edges"]:
        edge_id = edge["edge_id"]
        u, v = edge["endpoints"]
        cost = edge["cost"]

        nodes.add(u)
        nodes.add(v)
        edge_cost[edge_id] = cost
        adjacency[u].append((v, cost, edge_id))
        adjacency[v].append((u, cost, edge_id))

        if edge["is_required"]:
            task = {
                "edge_id": edge_id,
                "u": u,
                "v": v,
                "demand": edge["demand"],
                "service_cost": edge["service_cost"],
            }
            tasks.append(task)
            service_cost_total += edge["service_cost"]

    if num_vehicles <= 0:
        if tasks:
            raise RuntimeError("The instance has required edges but no vehicles")
        solution = {"objective_value": 0.0, "routes": []}
        with open(args.solution_path, "w", encoding="utf-8") as file:
            json.dump(solution, file, indent=2)
        if logger:
            logger.log_solution(0.0, solution)
        return

    if any(task["demand"] > capacity for task in tasks):
        raise RuntimeError("A required edge demand exceeds vehicle capacity")

    if sum(task["demand"] for task in tasks) > num_vehicles * capacity:
        raise RuntimeError("Total demand exceeds total fleet capacity")

    relevant_nodes = sorted(
        {depot}
        | {task["u"] for task in tasks}
        | {task["v"] for task in tasks}
    )
    relevant_index = {node: i for i, node in enumerate(relevant_nodes)}
    distance_matrix = []

    for source in relevant_nodes:
        row, _ = dijkstra(adjacency, source, relevant_index, need_parent=False)
        distance_matrix.append(row)

    def distance(a, b):
        return distance_matrix[relevant_index[a]][relevant_index[b]]

    for task in tasks:
        if not math.isfinite(distance(depot, task["u"])) and not math.isfinite(
            distance(depot, task["v"])
        ):
            raise RuntimeError(
                f"Required edge {task['edge_id']} is unreachable from the depot"
            )

    if not tasks:
        routes = [[] for _ in range(num_vehicles)]
        solution = build_solution(
            routes,
            tasks,
            depot,
            distance,
            adjacency,
            edge_cost,
            service_cost_total,
        )
        with open(args.solution_path, "w", encoding="utf-8") as file:
            json.dump(solution, file, indent=2)
        if logger:
            logger.log_solution(solution["objective_value"], solution)
        return

    bins = solve_capacitated_assignment(
        tasks, num_vehicles, capacity, depot, distance, deadline
    )
    if bins is None:
        raise RuntimeError("Unable to construct a capacity-feasible assignment")

    initial_routes = [
        nearest_order(bin_tasks, tasks, depot, distance)
        for bin_tasks in bins
    ]
    initial_deadhead = sum(
        route_evaluation(route, tasks, depot, distance)
        for route in initial_routes
    )

    best_routes = [route[:] for route in initial_routes]
    best_deadhead = initial_deadhead

    initial_solution = build_solution(
        best_routes,
        tasks,
        depot,
        distance,
        adjacency,
        edge_cost,
        service_cost_total,
    )
    if logger:
        logger.log_solution(initial_solution["objective_value"], initial_solution)

    # Route-first, cluster-second candidates using giant tours and exact splitting.
    max_giant_candidates = 1 if len(tasks) > 400 else 6
    all_indices = list(range(len(tasks)))

    for candidate_number in range(max_giant_candidates):
        if time.monotonic() >= deadline:
            break

        giant = nearest_order(
            all_indices,
            tasks,
            depot,
            distance,
            rng=rng,
            randomized=(candidate_number > 0),
        )
        candidate_routes = split_giant_tour(
            giant,
            tasks,
            num_vehicles,
            capacity,
            depot,
            distance,
            deadline,
        )
        if candidate_routes is None:
            continue

        candidate_deadhead = sum(
            route_evaluation(route, tasks, depot, distance)
            for route in candidate_routes
        )

        if candidate_deadhead + EPS < best_deadhead:
            best_routes = [route[:] for route in candidate_routes]
            best_deadhead = candidate_deadhead
            if logger:
                logger.log(service_cost_total + best_deadhead)

    def record_local_improvement(routes, deadhead):
        nonlocal best_routes, best_deadhead
        if deadhead + EPS < best_deadhead:
            best_deadhead = deadhead
            best_routes = [route[:] for route in routes]
            if logger:
                logger.log(service_cost_total + best_deadhead)

    if time.monotonic() < deadline:
        improved_routes, improved_deadhead = local_improvement(
            [route[:] for route in best_routes],
            tasks,
            capacity,
            depot,
            distance,
            deadline,
            rng,
            record_local_improvement,
        )
        if improved_deadhead + EPS < best_deadhead:
            best_routes = [route[:] for route in improved_routes]
            best_deadhead = improved_deadhead

    final_solution = build_solution(
        best_routes,
        tasks,
        depot,
        distance,
        adjacency,
        edge_cost,
        service_cost_total,
    )

    if logger:
        logger.log_solution(final_solution["objective_value"], final_solution)

    with open(args.solution_path, "w", encoding="utf-8") as file:
        json.dump(final_solution, file, indent=2)


if __name__ == "__main__":
    main()