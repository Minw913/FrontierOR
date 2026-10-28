import argparse
import json
import math
import random
import time
import heapq
import bisect

from solution_logger import SolutionLogger


def lpt_assign(p, m, order=None):
    """Longest Processing Time first assignment. Returns (assign, loads)."""
    n = len(p)
    if order is None:
        order = sorted(range(n), key=lambda j: -p[j])
    heap = [(0, i) for i in range(m)]
    heapq.heapify(heap)
    assign = [set() for _ in range(m)]
    loads = [0] * m
    for j in order:
        l, i = heapq.heappop(heap)
        assign[i].add(j)
        loads[i] = l + p[j]
        heapq.heappush(heap, (loads[i], i))
    return assign, loads


def sum_sq(loads):
    return sum(l * l for l in loads)


def best_pair_op(jobs_a, jobs_b, la, lb, p):
    """Find best improving move/swap between machine a (heavier) and b (lighter).

    Returns (delta, op) where delta < 0 means improvement in sum of squares.
    op is ('m', j, None) meaning move job j from a to b,
    or ('s', j, q) meaning swap job j (in a) with job q (in b).
    """
    d = la - lb
    best = 0
    best_op = None
    # Moves from a to b: delta = 2*pj*(pj - d), improving iff pj < d
    for j in jobs_a:
        pj = p[j]
        if 0 < pj < d:
            delta = 2 * pj * (pj - d)
            if delta < best:
                best = delta
                best_op = ('m', j, None)
    # Swaps: delta = 2*diff*(diff - d), improving iff 0 < diff < d
    if jobs_b:
        bs = sorted(jobs_b, key=lambda jj: p[jj])
        bvals = [p[jj] for jj in bs]
        half = d / 2.0
        for j in jobs_a:
            pj = p[j]
            target = pj - half  # ideal q value
            pos = bisect.bisect_left(bvals, target)
            for k in (pos - 1, pos, pos + 1):
                if 0 <= k < len(bs):
                    pq = bvals[k]
                    diff = pj - pq
                    if 0 < diff < d:
                        delta = 2 * diff * (diff - d)
                        if delta < best:
                            best = delta
                            best_op = ('s', j, bs[k])
    return best, best_op


def local_search(assign, loads, p, deadline):
    """First/best-improvement local search over machine pairs (moves + swaps)."""
    m = len(loads)
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        idx = sorted(range(m), key=lambda i: loads[i])
        for x in range(m - 1, 0, -1):
            if time.time() > deadline:
                return
            for y in range(x):
                a, b = idx[x], idx[y]
                # exhaust improvements between this pair
                guard = 0
                while guard < 200:
                    guard += 1
                    if loads[a] < loads[b]:
                        a, b = b, a
                    if loads[a] - loads[b] <= 1:
                        break
                    delta, op = best_pair_op(assign[a], assign[b],
                                             loads[a], loads[b], p)
                    if op is None:
                        break
                    if op[0] == 'm':
                        j = op[1]
                        assign[a].discard(j)
                        assign[b].add(j)
                        loads[a] -= p[j]
                        loads[b] += p[j]
                    else:
                        j, q = op[1], op[2]
                        assign[a].discard(j)
                        assign[b].discard(q)
                        assign[a].add(q)
                        assign[b].add(j)
                        loads[a] += p[q] - p[j]
                        loads[b] += p[j] - p[q]
                    improved = True


def build_solution(assign, loads, avg):
    dev = math.sqrt(sum((l - avg) ** 2 for l in loads))
    obj = dev / avg if avg > 0 else 0.0
    sol = {
        "objective_value": obj,
        "assignment": {str(i): sorted(assign[i]) for i in range(len(assign))},
        "machine_completion_times": {str(i): int(loads[i])
                                     for i in range(len(loads))},
    }
    return obj, sol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start = time.time()
    deadline = start + max(1, args.time_limit) - 0.5

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    m = int(data["num_machines"])
    p = [int(x) for x in data["processing_times"]]
    n = len(p)

    random.seed(0)

    T = sum(p)
    avg = T / m
    q, r = divmod(T, m)
    lb_ss = (m - r) * q * q + r * (q + 1) * (q + 1)  # lower bound on sum of squares

    # Initial LPT solution
    assign, loads = lpt_assign(p, m)
    ss = sum_sq(loads)
    best_assign = [set(s) for s in assign]
    best_loads = list(loads)
    best_ss = ss
    obj, sol = build_solution(best_assign, best_loads, avg)
    if logger:
        logger.log_solution(obj, sol)

    # Local search on LPT
    if best_ss > lb_ss and time.time() < deadline:
        local_search(assign, loads, p, deadline)
        ss = sum_sq(loads)
        if ss < best_ss:
            best_assign = [set(s) for s in assign]
            best_loads = list(loads)
            best_ss = ss
            obj, sol = build_solution(best_assign, best_loads, avg)
            if logger:
                logger.log_solution(obj, sol)

    # Iterated local search with random perturbations
    strength = 3
    no_improve = 0
    while time.time() < deadline and best_ss > lb_ss:
        # start from best
        assign = [set(s) for s in best_assign]
        loads = list(best_loads)

        # perturb: move a few random jobs to random machines
        k = strength + random.randint(0, strength)
        for _ in range(k):
            candidates = [i for i in range(m) if assign[i]]
            if not candidates:
                break
            a = random.choice(candidates)
            j = random.choice(tuple(assign[a]))
            b = random.randrange(m)
            if b == a:
                b = (b + 1) % m
            assign[a].discard(j)
            assign[b].add(j)
            loads[a] -= p[j]
            loads[b] += p[j]

        local_search(assign, loads, p, deadline)
        ss = sum_sq(loads)
        if ss < best_ss:
            best_assign = [set(s) for s in assign]
            best_loads = list(loads)
            best_ss = ss
            no_improve = 0
            strength = 3
            obj, sol = build_solution(best_assign, best_loads, avg)
            if logger:
                logger.log_solution(obj, sol)
        else:
            no_improve += 1
            if no_improve % 50 == 0:
                strength = min(strength + 1, max(3, n // 4))
            # occasional randomized restart
            if no_improve % 200 == 199:
                order = list(range(n))
                random.shuffle(order)
                # blend: mostly LPT order with small shuffles
                order.sort(key=lambda j: -p[j] + random.random() * max(p) * 0.1)
                assign2, loads2 = lpt_assign(p, m, order)
                local_search(assign2, loads2, p, deadline)
                ss2 = sum_sq(loads2)
                if ss2 < best_ss:
                    best_assign = [set(s) for s in assign2]
                    best_loads = list(loads2)
                    best_ss = ss2
                    no_improve = 0
                    strength = 3
                    obj, sol = build_solution(best_assign, best_loads, avg)
                    if logger:
                        logger.log_solution(obj, sol)

    obj, sol = build_solution(best_assign, best_loads, avg)
    if logger:
        logger.log_solution(obj, sol)
    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()