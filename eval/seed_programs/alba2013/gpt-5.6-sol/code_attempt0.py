import argparse
import json
import math
import os
import random
import time

from solution_logger import SolutionLogger


EPS = 1e-9


def route_cost(order, depot, vertex_of, distance):
    if not order:
        return 0
    total = distance[depot][vertex_of[order[0]]]
    for a, b in zip(order, order[1:]):
        total += distance[vertex_of[a]][vertex_of[b]]
    total += distance[vertex_of[order[-1]]][depot]
    return total


def make_solution(pickup_order, delivery_order, pickup_depot, delivery_depot,
                  pickup_vertex, delivery_vertex, objective):
    # Tours are represented cyclically: the final return to the initial depot
    # is implicit, as required by the schema's num_vertices length.
    return {
        "objective_value": float(objective),
        "pickup_tour": [pickup_depot] + [pickup_vertex[r] for r in pickup_order],
        "delivery_tour": [delivery_depot] + [delivery_vertex[r] for r in delivery_order],
    }


def nearest_neighbor_order(requests, depot, vertex_of, distance, rng,
                           randomized=False):
    remaining = set(requests)
    order = []
    current = depot

    while remaining:
        ranked = sorted(remaining, key=lambda r: distance[current][vertex_of[r]])
        if randomized and len(ranked) > 1:
            k = min(3, len(ranked))
            weights = [1.0 / (i + 1) for i in range(k)]
            chosen = rng.choices(ranked[:k], weights=weights, k=1)[0]
        else:
            chosen = ranked[0]
        order.append(chosen)
        remaining.remove(chosen)
        current = vertex_of[chosen]

    return order


def improve_unconstrained(order, depot, vertex_of, distance, symmetric,
                          deadline, max_rounds=100):
    order = list(order)
    n = len(order)
    if n <= 1:
        return order

    rounds = 0
    while rounds < max_rounds and time.monotonic() < deadline:
        rounds += 1
        best_delta = 0
        best_move = None

        # Directed-safe relocate neighborhood.
        for i in range(n):
            if time.monotonic() >= deadline:
                break

            x = order[i]
            xv = vertex_of[x]
            prev_v = depot if i == 0 else vertex_of[order[i - 1]]
            next_v = depot if i == n - 1 else vertex_of[order[i + 1]]
            remove_delta = (
                distance[prev_v][next_v]
                - distance[prev_v][xv]
                - distance[xv][next_v]
            )

            reduced = order[:i] + order[i + 1:]
            for k in range(n):
                left_v = depot if k == 0 else vertex_of[reduced[k - 1]]
                right_v = depot if k == n - 1 else vertex_of[reduced[k]]
                delta = (
                    remove_delta
                    + distance[left_v][xv]
                    + distance[xv][right_v]
                    - distance[left_v][right_v]
                )
                if delta < best_delta - EPS:
                    best_delta = delta
                    best_move = (i, k)

        if best_move is not None:
            i, k = best_move
            x = order.pop(i)
            order.insert(k, x)
            continue

        # For Euclidean/symmetric instances, also use 2-opt.
        if symmetric:
            best_delta = 0
            best_segment = None
            for i in range(n - 1):
                if time.monotonic() >= deadline:
                    break
                a = depot if i == 0 else vertex_of[order[i - 1]]
                b = vertex_of[order[i]]
                for j in range(i + 1, n):
                    c = vertex_of[order[j]]
                    d = depot if j == n - 1 else vertex_of[order[j + 1]]
                    delta = (
                        distance[a][c] + distance[b][d]
                        - distance[a][b] - distance[c][d]
                    )
                    if delta < best_delta - EPS:
                        best_delta = delta
                        best_segment = (i, j)

            if best_segment is not None:
                i, j = best_segment
                order[i:j + 1] = reversed(order[i:j + 1])
                continue

        break

    return order


def is_symmetric(distance):
    m = len(distance)
    for i in range(m):
        for j in range(i + 1, m):
            if distance[i][j] != distance[j][i]:
                return False
    return True


def valid_for_chains(order, chains):
    position = {r: i for i, r in enumerate(order)}
    if len(position) != len(order):
        return False
    for chain in chains:
        for a, b in zip(chain, chain[1:]):
            if position[a] >= position[b]:
                return False
    return True


def greedy_topological_order(chains, depot, vertex_of, distance, rng,
                             randomized=False):
    indices = [0] * len(chains)
    total = sum(len(c) for c in chains)
    result = []
    current = depot

    while len(result) < total:
        available = []
        for stack_idx, chain in enumerate(chains):
            p = indices[stack_idx]
            if p < len(chain):
                r = chain[p]
                available.append((distance[current][vertex_of[r]], stack_idx, r))

        available.sort()
        if randomized and len(available) > 1:
            k = min(3, len(available))
            choice = rng.choices(
                available[:k],
                weights=[1.0 / (i + 1) for i in range(k)],
                k=1
            )[0]
        else:
            choice = available[0]

        _, stack_idx, r = choice
        result.append(r)
        indices[stack_idx] += 1
        current = vertex_of[r]

    return result


def improve_precedence_order(order, chains, depot, vertex_of, distance,
                             symmetric, deadline, max_rounds=80):
    order = list(order)
    n = len(order)
    if n <= 1:
        return order

    stack_of = {}
    rank_of = {}
    predecessor = {}
    successor = {}

    for stack_idx, chain in enumerate(chains):
        for rank, r in enumerate(chain):
            stack_of[r] = stack_idx
            rank_of[r] = rank
            predecessor[r] = chain[rank - 1] if rank > 0 else None
            successor[r] = chain[rank + 1] if rank + 1 < len(chain) else None

    rounds = 0
    while rounds < max_rounds and time.monotonic() < deadline:
        rounds += 1
        best_delta = 0
        best_move = None

        positions = {r: i for i, r in enumerate(order)}

        for i, x in enumerate(order):
            if time.monotonic() >= deadline:
                break

            xv = vertex_of[x]
            prev_v = depot if i == 0 else vertex_of[order[i - 1]]
            next_v = depot if i == n - 1 else vertex_of[order[i + 1]]
            remove_delta = (
                distance[prev_v][next_v]
                - distance[prev_v][xv]
                - distance[xv][next_v]
            )

            reduced = order[:i] + order[i + 1:]
            reduced_pos = {r: p for p, r in enumerate(reduced)}

            pred = predecessor[x]
            succ = successor[x]
            low = 0 if pred is None else reduced_pos[pred] + 1
            high = len(reduced) if succ is None else reduced_pos[succ]

            for k in range(low, high + 1):
                left_v = depot if k == 0 else vertex_of[reduced[k - 1]]
                right_v = depot if k == len(reduced) else vertex_of[reduced[k]]
                delta = (
                    remove_delta
                    + distance[left_v][xv]
                    + distance[xv][right_v]
                    - distance[left_v][right_v]
                )
                if delta < best_delta - EPS:
                    best_delta = delta
                    best_move = (i, k)

        if best_move is not None:
            i, k = best_move
            x = order.pop(i)
            order.insert(k, x)
            continue

        # A reversed segment is precedence-feasible if it contains at most
        # one item from every stack.
        if symmetric:
            best_delta = 0
            best_segment = None
            for i in range(n - 1):
                seen_stacks = set()
                for j in range(i, n):
                    st = stack_of[order[j]]
                    if st in seen_stacks:
                        break
                    seen_stacks.add(st)
                    if j <= i:
                        continue

                    a = depot if i == 0 else vertex_of[order[i - 1]]
                    b = vertex_of[order[i]]
                    c = vertex_of[order[j]]
                    d = depot if j == n - 1 else vertex_of[order[j + 1]]
                    delta = (
                        distance[a][c] + distance[b][d]
                        - distance[a][b] - distance[c][d]
                    )
                    if delta < best_delta - EPS:
                        best_delta = delta
                        best_segment = (i, j)

            if best_segment is not None:
                i, j = best_segment
                order[i:j + 1] = reversed(order[i:j + 1])
                continue

        break

    return order


def optimize_topological_route(chains, depot, vertex_of, distance, symmetric,
                               rng, deadline, initial=None, starts=2):
    candidates = []

    if initial is not None and valid_for_chains(initial, chains):
        candidates.append(list(initial))

    if time.monotonic() < deadline:
        candidates.append(
            greedy_topological_order(
                chains, depot, vertex_of, distance, rng, randomized=False
            )
        )

    for _ in range(max(0, starts - 1)):
        if time.monotonic() >= deadline:
            break
        candidates.append(
            greedy_topological_order(
                chains, depot, vertex_of, distance, rng, randomized=True
            )
        )

    best_order = candidates[0]
    best_cost = route_cost(best_order, depot, vertex_of, distance)

    for candidate in candidates:
        if time.monotonic() >= deadline:
            break
        improved = improve_precedence_order(
            candidate, chains, depot, vertex_of, distance,
            symmetric, deadline,
            max_rounds=min(100, max(10, 3 * len(candidate)))
        )
        value = route_cost(improved, depot, vertex_of, distance)
        if value < best_cost:
            best_cost = value
            best_order = improved

    return best_order


def assign_stacks(anchor_order, target_positions, num_stacks, height,
                  mode, rng):
    stacks = [[] for _ in range(num_stacks)]
    last_target = [math.inf] * num_stacks

    for idx, r in enumerate(anchor_order):
        available = [k for k in range(num_stacks) if len(stacks[k]) < height]
        if not available:
            raise ValueError("Vehicle stack capacity is smaller than request count")

        q = target_positions.get(r, 0)

        if mode == "contiguous":
            chosen = available[0]

        elif mode == "round_robin":
            preferred = idx % num_stacks
            chosen = preferred if preferred in available else available[0]

        elif mode == "random":
            chosen = rng.choice(available)

        else:  # compatibility-aware assignment
            compatible = [k for k in available if last_target[k] > q]
            if compatible:
                compatible.sort(key=lambda k: (last_target[k] - q, len(stacks[k])))
                if mode == "qrandom" and len(compatible) > 1:
                    chosen = rng.choice(compatible[:min(3, len(compatible))])
                else:
                    chosen = compatible[0]
            else:
                # Minimize the amount by which this new precedence conflicts
                # with the target route, while preserving capacity balance.
                available.sort(
                    key=lambda k: (q - last_target[k], len(stacks[k]))
                )
                if mode == "qrandom" and len(available) > 1:
                    chosen = rng.choice(available[:min(3, len(available))])
                else:
                    chosen = available[0]

        stacks[chosen].append(r)
        last_target[chosen] = q

    return stacks


def perturb_order(order, rng):
    result = list(order)
    n = len(result)
    if n < 2:
        return result

    if n >= 8 and rng.random() < 0.5:
        cuts = sorted(rng.sample(range(1, n), 3))
        a, b, c = cuts
        result = (
            result[:a]
            + result[b:c]
            + result[a:b]
            + result[c:]
        )
    else:
        i, j = sorted(rng.sample(range(n), 2))
        result[i:j + 1] = reversed(result[i:j + 1])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.monotonic()
    # Keep a very small reserve for serialization.
    deadline = start_time + max(0, args.time_limit) - 0.02
    rng = random.Random(0)

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    params = data["parameters"]
    num_stacks = int(params["num_stacks_s"])
    height = int(params["stack_height_l"])
    n = int(params["num_requests_n"])

    pickup_depot = int(data["pickup_region"]["depot_index"])
    delivery_depot = int(data["delivery_region"]["depot_index"])
    pickup_distance = data["pickup_region"]["distance_matrix"]
    delivery_distance = data["delivery_region"]["distance_matrix"]

    requests_data = data["requests"]
    requests = list(range(n))
    pickup_vertex = [int(req["pickup_vertex"]) for req in requests_data]
    delivery_vertex = [int(req["delivery_vertex"]) for req in requests_data]

    if n > num_stacks * height:
        raise ValueError(
            f"Infeasible instance: {n} items exceed total capacity "
            f"{num_stacks * height}"
        )

    if n == 0:
        solution = {
            "objective_value": 0.0,
            "pickup_tour": [pickup_depot],
            "delivery_tour": [delivery_depot],
        }
        if logger:
            logger.log_solution(0.0, solution)
        os.makedirs(os.path.dirname(os.path.abspath(args.solution_path)), exist_ok=True)
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(solution, f)
        return

    pickup_symmetric = is_symmetric(pickup_distance)
    delivery_symmetric = is_symmetric(delivery_distance)

    # Obtain good independent routing targets.
    pickup_target = nearest_neighbor_order(
        requests, pickup_depot, pickup_vertex, pickup_distance, rng
    )
    if time.monotonic() < deadline:
        pickup_target = improve_unconstrained(
            pickup_target, pickup_depot, pickup_vertex, pickup_distance,
            pickup_symmetric, deadline,
            max_rounds=min(100, max(10, 3 * n))
        )

    delivery_target = nearest_neighbor_order(
        requests, delivery_depot, delivery_vertex, delivery_distance, rng
    )
    if time.monotonic() < deadline:
        delivery_target = improve_unconstrained(
            delivery_target, delivery_depot, delivery_vertex, delivery_distance,
            delivery_symmetric, deadline,
            max_rounds=min(100, max(10, 3 * n))
        )

    best_value = math.inf
    best_pickup = None
    best_delivery = None
    best_stacks = None

    def consider(stacks, pickup_initial=None, delivery_initial=None, starts=2):
        nonlocal best_value, best_pickup, best_delivery, best_stacks

        pickup_chains = [list(st) for st in stacks]
        delivery_chains = [list(reversed(st)) for st in stacks]

        pickup_order = optimize_topological_route(
            pickup_chains, pickup_depot, pickup_vertex, pickup_distance,
            pickup_symmetric, rng, deadline,
            initial=pickup_initial, starts=starts
        )
        delivery_order = optimize_topological_route(
            delivery_chains, delivery_depot, delivery_vertex, delivery_distance,
            delivery_symmetric, rng, deadline,
            initial=delivery_initial, starts=starts
        )

        value = (
            route_cost(pickup_order, pickup_depot, pickup_vertex, pickup_distance)
            + route_cost(delivery_order, delivery_depot, delivery_vertex,
                         delivery_distance)
        )

        if value < best_value - EPS:
            best_value = value
            best_pickup = list(pickup_order)
            best_delivery = list(delivery_order)
            best_stacks = [list(st) for st in stacks]
            solution = make_solution(
                best_pickup, best_delivery,
                pickup_depot, delivery_depot,
                pickup_vertex, delivery_vertex,
                best_value
            )
            if logger:
                logger.log_solution(best_value, solution)

    pickup_positions = {r: i for i, r in enumerate(pickup_target)}
    delivery_positions = {r: i for i, r in enumerate(delivery_target)}

    # Guarantee a very fast first feasible incumbent.
    initial_stacks = assign_stacks(
        pickup_target, delivery_positions, num_stacks, height,
        "round_robin", rng
    )
    consider(
        initial_stacks,
        pickup_initial=pickup_target,
        delivery_initial=delivery_target,
        starts=1
    )

    modes = ["qgreedy", "contiguous", "qrandom", "random", "round_robin"]

    # Pickup-anchored and delivery-anchored constructions.
    for mode in modes:
        if time.monotonic() >= deadline:
            break
        stacks = assign_stacks(
            pickup_target, delivery_positions, num_stacks, height, mode, rng
        )
        consider(
            stacks,
            pickup_initial=pickup_target,
            delivery_initial=delivery_target,
            starts=2
        )

        if time.monotonic() >= deadline:
            break

        # Processing reversed delivery order constructs bottom-to-top stack
        # sequences whose reversed chains follow the delivery anchor.
        bottom_to_top_anchor = list(reversed(delivery_target))
        stacks = assign_stacks(
            bottom_to_top_anchor, pickup_positions,
            num_stacks, height, mode, rng
        )
        consider(
            stacks,
            pickup_initial=pickup_target,
            delivery_initial=delivery_target,
            starts=2
        )

    iteration = 0
    while time.monotonic() < deadline:
        iteration += 1

        if best_pickup is None:
            break

        # Alternate around the current incumbent and perturbed independent
        # routing targets.
        if iteration % 2:
            anchor_pickup = perturb_order(best_pickup, rng)
            if time.monotonic() < deadline:
                anchor_pickup = improve_unconstrained(
                    anchor_pickup, pickup_depot, pickup_vertex, pickup_distance,
                    pickup_symmetric, deadline,
                    max_rounds=min(20, n)
                )
            target_delivery = best_delivery
            target_pos = {r: i for i, r in enumerate(target_delivery)}
            mode = "qrandom" if iteration % 4 else "qgreedy"
            stacks = assign_stacks(
                anchor_pickup, target_pos, num_stacks, height, mode, rng
            )
            consider(
                stacks,
                pickup_initial=anchor_pickup,
                delivery_initial=target_delivery,
                starts=1
            )
        else:
            anchor_delivery = perturb_order(best_delivery, rng)
            if time.monotonic() < deadline:
                anchor_delivery = improve_unconstrained(
                    anchor_delivery, delivery_depot, delivery_vertex,
                    delivery_distance, delivery_symmetric, deadline,
                    max_rounds=min(20, n)
                )
            target_pickup = best_pickup
            target_pos = {r: i for i, r in enumerate(target_pickup)}
            stacks = assign_stacks(
                list(reversed(anchor_delivery)), target_pos,
                num_stacks, height, "qrandom", rng
            )
            consider(
                stacks,
                pickup_initial=target_pickup,
                delivery_initial=anchor_delivery,
                starts=1
            )

    if best_pickup is None:
        # Defensive fallback; normally the first construction already sets it.
        stacks = assign_stacks(
            requests, {r: r for r in requests},
            num_stacks, height, "round_robin", rng
        )
        pickup_chains = stacks
        delivery_chains = [list(reversed(st)) for st in stacks]
        best_pickup = [r for st in pickup_chains for r in st]
        best_delivery = [r for st in delivery_chains for r in st]
        best_value = (
            route_cost(best_pickup, pickup_depot, pickup_vertex, pickup_distance)
            + route_cost(best_delivery, delivery_depot, delivery_vertex,
                         delivery_distance)
        )

    final_solution = make_solution(
        best_pickup, best_delivery,
        pickup_depot, delivery_depot,
        pickup_vertex, delivery_vertex,
        best_value
    )

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, separators=(",", ":"))


if __name__ == "__main__":
    main()