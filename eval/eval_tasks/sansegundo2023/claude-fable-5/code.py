import argparse
import json
import random
import time
import sys

from solution_logger import SolutionLogger


class TimeUp(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    deadline = start_time + max(1, args.time_limit) - 0.5  # safety margin

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n = inst["num_vertices"]
    vertices = inst.get("vertices", list(range(n)))
    edges = inst.get("edges", [])

    # Build adjacency bitsets
    adj = [0] * n
    for e in edges:
        u, v = e[0], e[1]
        if u == v:
            continue
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    best_clique = []
    best_size = 0

    def record(clique):
        nonlocal best_clique, best_size
        if len(clique) > best_size:
            best_clique = list(clique)
            best_size = len(clique)
            if logger:
                logger.log_solution(
                    float(best_size),
                    {"objective_value": float(best_size),
                     "clique_vertices": sorted(best_clique)},
                )

    # Validate the provided greedy clique (if any)
    gc = inst.get("greedy_clique", []) or []
    valid = True
    for i in range(len(gc)):
        for j in range(i + 1, len(gc)):
            if not (adj[gc[i]] >> gc[j]) & 1:
                valid = False
                break
        if not valid:
            break
    if valid and gc:
        record(gc)

    if n == 0:
        with open(args.solution_path, "w") as f:
            json.dump({"objective_value": 0.0, "clique_vertices": []}, f)
        return

    # ---------------- Heuristic phase: randomized greedy restarts ----------------
    rng = random.Random(0)
    heur_deadline = min(deadline, start_time + max(1.0, 0.15 * args.time_limit))

    def greedy_from(v0):
        clique = [v0]
        cand = adj[v0]
        while cand:
            # pick vertex in cand with max connections inside cand (ties random)
            best_v = -1
            best_d = -1
            c = cand
            while c:
                lsb = c & (-c)
                u = lsb.bit_length() - 1
                c ^= lsb
                d = (adj[u] & cand).bit_count()
                if d > best_d or (d == best_d and rng.random() < 0.3):
                    best_d = d
                    best_v = u
            clique.append(best_v)
            cand &= adj[best_v]
        return clique

    # Deterministic pass over high-degree vertices first
    deg_order = sorted(range(n), key=lambda v: -adj[v].bit_count())
    idx = 0
    while time.time() < heur_deadline:
        if idx < min(n, 50):
            v0 = deg_order[idx]
            idx += 1
        else:
            v0 = rng.randrange(n)
        c = greedy_from(v0)
        record(c)
        if idx >= n and idx >= 50 and time.time() - start_time > 0.3 * args.time_limit:
            break

    # ---------------- Exact phase: Tomita-style branch & bound with coloring ----------------
    check_counter = [0]

    def timecheck():
        check_counter[0] += 1
        if (check_counter[0] & 255) == 0:
            if time.time() > deadline:
                raise TimeUp()

    def color_sort(P):
        # Greedy coloring: returns list of (vertex, color) in increasing color order
        order = []
        colors = []
        rem = P
        color = 0
        while rem:
            color += 1
            Q = rem
            while Q:
                lsb = Q & (-Q)
                v = lsb.bit_length() - 1
                Q &= ~adj[v]
                Q ^= lsb
                rem ^= lsb
                order.append(v)
                colors.append(color)
        return order, colors

    sys.setrecursionlimit(10000)
    current = []

    def expand(P):
        nonlocal best_size
        timecheck()
        order, colors = color_sort(P)
        Pmut = P
        for i in range(len(order) - 1, -1, -1):
            if len(current) + colors[i] <= best_size:
                return
            v = order[i]
            bit = 1 << v
            current.append(v)
            newP = Pmut & adj[v]
            if newP:
                expand(newP)
            else:
                if len(current) > best_size:
                    record(current)
            current.pop()
            Pmut &= ~bit

    optimal = False
    try:
        full = (1 << n) - 1
        # Order root vertices by degeneracy-like order to strengthen pruning:
        # process vertices in reverse degree order via coloring inside expand.
        expand(full)
        optimal = True
    except TimeUp:
        pass
    except RecursionError:
        pass

    # Final safety: verify best clique
    ok = True
    for i in range(len(best_clique)):
        for j in range(i + 1, len(best_clique)):
            if not (adj[best_clique[i]] >> best_clique[j]) & 1:
                ok = False
    if not ok and n > 0:
        # fallback: single vertex
        best_clique = [0]
        best_size = 1

    solution = {
        "objective_value": float(best_size),
        "clique_vertices": sorted(best_clique),
    }
    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()