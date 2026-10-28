import argparse
import json
import random
import time
from typing import Dict, List, Tuple, Optional

from solution_logger import SolutionLogger

# State encoding
TRUCK = 0
DRONE = 1
COMBINED = 2


def nearest_neighbor_order(customers: List[int], truck_tt: List[List[float]]) -> List[int]:
    unvisited = set(customers)
    order = []
    cur = 0
    while unvisited:
        nxt = min(unvisited, key=lambda j: truck_tt[cur][j])
        order.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return order


def route_cost(order: List[int], truck_tt: List[List[float]]) -> float:
    if not order:
        return 0.0
    cost = truck_tt[0][order[0]]
    for i in range(len(order) - 1):
        cost += truck_tt[order[i]][order[i + 1]]
    cost += truck_tt[order[-1]][0]
    return float(cost)


def two_opt(order: List[int], truck_tt: List[List[float]], time_end: float) -> List[int]:
    n = len(order)
    if n <= 3:
        return order[:]
    best = order[:]
    best_cost = route_cost(best, truck_tt)
    improved = True
    while improved and time.time() < time_end:
        improved = False
        for i in range(n - 1):
            if time.time() >= time_end:
                break
            for k in range(i + 1, n):
                if time.time() >= time_end:
                    break
                cand = best[:i] + best[i:k + 1][::-1] + best[k + 1:]
                c = route_cost(cand, truck_tt)
                if c + 1e-9 < best_cost:
                    best = cand
                    best_cost = c
                    improved = True
                    break
            if improved:
                break
    return best


def is_feasible_state(order: List[int], state: Dict[int, int]) -> bool:
    drone_nodes = [c for c in order if state[c] != TRUCK]
    truck_nodes = [c for c in order if state[c] != DRONE]

    # Truck and drone must each depart once and return once through customer arcs
    if len(truck_nodes) < 1:
        return False
    if len(drone_nodes) < 2:  # also enforces "no single out-and-back"
        return False

    # Pairwise drone customer-customer arc rule:
    # consecutive nodes on drone route cannot both be DRONE-only.
    for a, b in zip(drone_nodes, drone_nodes[1:]):
        if state[a] == DRONE and state[b] == DRONE:
            return False

    return True


def evaluate_state(
    order: List[int],
    state: Dict[int, int],
    truck_tt: List[List[float]],
    drone_tt: List[List[float]],
    n_customers: int,
) -> Optional[Tuple[float, Dict[int, float], List[int], List[int], Dict[str, str]]]:
    if not is_feasible_state(order, state):
        return None

    sink = n_customers + 1
    arr: Dict[int, float] = {0: 0.0}

    prev_t = 0
    prev_d = 0

    for c in order:
        st = state[c]
        cand = []

        if st != DRONE:  # truck visits
            cand.append(arr[prev_t] + truck_tt[prev_t][c])
        if st != TRUCK:  # drone visits
            cand.append(arr[prev_d] + drone_tt[prev_d][c])

        if not cand:
            return None

        arr[c] = max(cand)

        if st != DRONE:
            prev_t = c
        if st != TRUCK:
            prev_d = c

    # Return to depot destination (mapped to depot index 0 in input matrices)
    arr[sink] = max(arr[prev_t] + truck_tt[prev_t][0], arr[prev_d] + drone_tt[prev_d][0])

    truck_nodes = [c for c in order if state[c] != DRONE]
    drone_nodes = [c for c in order if state[c] != TRUCK]
    truck_route = [0] + truck_nodes + [0]
    drone_route = [0] + drone_nodes + [0]

    ctype = {}
    for c in order:
        if state[c] == TRUCK:
            ctype[str(c)] = "truck"
        elif state[c] == DRONE:
            ctype[str(c)] = "drone"
        else:
            ctype[str(c)] = "combined"

    return float(arr[sink]), arr, truck_route, drone_route, ctype


def hill_climb(
    order: List[int],
    init_state: Dict[int, int],
    truck_tt: List[List[float]],
    drone_tt: List[List[float]],
    n_customers: int,
    time_end: float,
):
    cur_state = dict(init_state)
    cur_eval = evaluate_state(order, cur_state, truck_tt, drone_tt, n_customers)
    if cur_eval is None:
        return None
    cur_obj = cur_eval[0]

    improved = True
    while improved and time.time() < time_end:
        improved = False
        best_move = None
        best_obj = cur_obj

        for c in order:
            if time.time() >= time_end:
                break
            old = cur_state[c]
            for ns in (TRUCK, DRONE, COMBINED):
                if ns == old:
                    continue
                cur_state[c] = ns
                ev = evaluate_state(order, cur_state, truck_tt, drone_tt, n_customers)
                if ev is not None and ev[0] + 1e-9 < best_obj:
                    best_obj = ev[0]
                    best_move = (c, ns, ev)
            cur_state[c] = old

        if best_move is not None:
            c, ns, ev = best_move
            cur_state[c] = ns
            cur_obj = ev[0]
            improved = True

    return evaluate_state(order, cur_state, truck_tt, drone_tt, n_customers), cur_state


def random_feasible_state(order: List[int], rng: random.Random) -> Dict[int, int]:
    n = len(order)
    state = {c: TRUCK for c in order}

    # Ensure at least two drone-visited customers; start with two combined anchors
    i, j = sorted(rng.sample(range(n), 2))
    state[order[i]] = COMBINED
    state[order[j]] = COMBINED

    # Randomly diversify states while trying to stay feasible
    for c in order:
        r = rng.random()
        if r < 0.18:
            state[c] = DRONE
        elif r < 0.38:
            state[c] = COMBINED
        else:
            state[c] = TRUCK

    # Repair feasibility
    drone_nodes = [c for c in order if state[c] != TRUCK]
    if len(drone_nodes) < 2:
        picks = rng.sample(order, 2)
        state[picks[0]] = COMBINED
        state[picks[1]] = COMBINED

    # Remove consecutive drone-only nodes on drone sequence
    changed = True
    while changed:
        changed = False
        dseq = [c for c in order if state[c] != TRUCK]
        for a, b in zip(dseq, dseq[1:]):
            if state[a] == DRONE and state[b] == DRONE:
                # flip one to combined
                if rng.random() < 0.5:
                    state[a] = COMBINED
                else:
                    state[b] = COMBINED
                changed = True
                break

    # Ensure truck has at least one node
    if all(state[c] == DRONE for c in order):
        state[order[rng.randrange(n)]] = COMBINED

    # Ensure drone has at least two nodes
    dseq = [c for c in order if state[c] != TRUCK]
    if len(dseq) < 2:
        missing = 2 - len(dseq)
        cand = [c for c in order if state[c] == TRUCK]
        rng.shuffle(cand)
        for c in cand[:missing]:
            state[c] = COMBINED

    return state


def solve(instance: dict, time_limit: int, logger: Optional[SolutionLogger]):
    start = time.time()
    time_end = start + max(1, int(time_limit))

    n = int(instance["num_customers"])
    customers = list(range(1, n + 1))
    truck_tt = instance["truck_travel_times"]
    drone_tt = instance["drone_travel_times"]

    # Build customer order for both routes to preserve acyclicity/synchronization consistency
    order = nearest_neighbor_order(customers, truck_tt)
    order = two_opt(order, truck_tt, min(time_end, start + 0.25 * max(1, int(time_limit))))

    best = None
    best_state = None

    def consider(eval_pack, st):
        nonlocal best, best_state
        if eval_pack is None:
            return
        obj = eval_pack[0]
        if best is None or obj + 1e-9 < best[0]:
            best = eval_pack
            best_state = dict(st)
            if logger:
                logger.log(float(obj))

    # Deterministic initial states: all truck + two combined anchors
    for i in range(n):
        if time.time() >= time_end:
            break
        for j in range(i + 1, n):
            if time.time() >= time_end:
                break
            st = {c: TRUCK for c in order}
            st[order[i]] = COMBINED
            st[order[j]] = COMBINED
            ev = evaluate_state(order, st, truck_tt, drone_tt, n)
            consider(ev, st)

    # Local improvement from best current
    if best_state is not None and time.time() < time_end:
        hc = hill_climb(order, best_state, truck_tt, drone_tt, n, time_end)
        if hc is not None:
            consider(hc[0], hc[1])

    # Multi-start random hill climbing until time limit
    rng = random.Random(0)
    while time.time() < time_end:
        st = random_feasible_state(order, rng)
        hc = hill_climb(order, st, truck_tt, drone_tt, n, time_end)
        if hc is not None:
            consider(hc[0], hc[1])

    # Safety fallback (always feasible for n>=2)
    if best is None:
        st = {c: TRUCK for c in order}
        st[order[0]] = COMBINED
        st[order[1]] = COMBINED
        best = evaluate_state(order, st, truck_tt, drone_tt, n)
        best_state = st
        if logger and best is not None:
            logger.log(float(best[0]))

    obj, arr, truck_route, drone_route, ctype = best
    sink = n + 1

    arrival_times = {str(0): float(arr[0])}
    for c in customers:
        arrival_times[str(c)] = float(arr[c])
    arrival_times[str(sink)] = float(arr[sink])

    solution = {
        "objective_value": float(obj),
        "truck_route": [int(x) for x in truck_route],
        "drone_route": [int(x) for x in drone_route],
        "customer_types": ctype,
        "arrival_times": arrival_times,
    }
    return solution


def main():
    parser = argparse.ArgumentParser(description="Truck-Drone synchronized routing solver (heuristic, no external MILP dependency)")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    solution = solve(instance, args.time_limit, logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()