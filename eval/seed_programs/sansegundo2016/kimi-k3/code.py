import argparse
import json
import sys
import time

from solution_logger import SolutionLogger


class _TimeUp(Exception):
    """Internal signal used to abort the search when the time limit is hit."""
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Maximum clique solver (bitset Tomita branch and bound)."
    )
    parser.add_argument("--instance_path", required=True,
                        help="Path to the problem instance JSON file.")
    parser.add_argument("--solution_path", required=True,
                        help="Path where the final solution JSON file is written.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", default=None,
                        help="Optional path to a JSONL file for incumbent logging.")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    time_limit = max(int(args.time_limit), 0)
    deadline = start_time + time_limit

    logger = (SolutionLogger(args.log_path, sense="maximize")
              if args.log_path else None)

    with open(args.instance_path, "r") as fh:
        instance = json.load(fh)

    graph = instance.get("graph", {})
    n = int(graph.get("num_vertices", 0) or 0)
    adjacency_list = graph.get("adjacency_list") or {}

    best = []
    best_size = 0

    def record(clique):
        nonlocal best, best_size
        if len(clique) > best_size:
            best = list(clique)
            best_size = len(clique)
            if logger is not None:
                logger.log_solution(best_size, {
                    "objective_value": best_size,
                    "clique_vertices": sorted(best),
                })

    def write_solution():
        solution = {
            "objective_value": int(best_size),
            "clique_vertices": [int(v) for v in sorted(best)],
        }
        with open(args.solution_path, "w") as fh:
            json.dump(solution, fh)

    if n <= 0:
        write_solution()
        return

    # ------------------------------------------------------------------
    # Memory-safe heuristic path for extremely large graphs.
    # ------------------------------------------------------------------
    if n > 20000:
        adj_set = [set() for _ in range(n)]
        for key, neighbors in adjacency_list.items():
            try:
                v = int(key)
            except (TypeError, ValueError):
                continue
            if not (0 <= v < n):
                continue
            for u in neighbors:
                try:
                    u = int(u)
                except (TypeError, ValueError):
                    continue
                if 0 <= u < n and u != v:
                    adj_set[v].add(u)
                    adj_set[u].add(v)
        degree = [len(s) for s in adj_set]
        by_degree = sorted(range(n), key=lambda v: -degree[v])
        starts = min(n, 3000)
        for i in range(starts):
            if (i & 7) == 0 and time.time() > deadline:
                break
            s = by_degree[i]
            clique = [s]
            cand = set(adj_set[s])
            while cand:
                v = max(cand, key=degree.__getitem__)
                clique.append(v)
                cand &= adj_set[v]
                cand.discard(v)
            record(clique)
        write_solution()
        return

    # ------------------------------------------------------------------
    # Standard path: bitset adjacency + Tomita branch and bound.
    # ------------------------------------------------------------------
    adj = [0] * n
    for key, neighbors in adjacency_list.items():
        try:
            v = int(key)
        except (TypeError, ValueError):
            continue
        if not (0 <= v < n):
            continue
        bits = 0
        for u in neighbors:
            try:
                u = int(u)
            except (TypeError, ValueError):
                continue
            if 0 <= u < n and u != v:
                bits |= (1 << u)
        adj[v] |= bits
    for v in range(n):  # symmetrize
        b = adj[v]
        while b:
            lsb = b & -b
            adj[lsb.bit_length() - 1] |= (1 << v)
            b ^= lsb
    for v in range(n):  # drop any self-loops
        adj[v] &= ~(1 << v)

    if hasattr(int, "bit_count"):
        _popcount = int.bit_count
    else:
        def _popcount(x):
            return bin(x).count("1")

    # ---- greedy lower bounds ----
    degree = [_popcount(adj[v]) for v in range(n)]
    by_degree = sorted(range(n), key=lambda v: -degree[v])

    # Cheap single-pass greedy over the degree ordering.
    clique = []
    cmask = 0
    for v in by_degree:
        if cmask & ~adj[v] == 0:  # v adjacent to every chosen vertex
            clique.append(v)
            cmask |= (1 << v)
    record(clique)

    # Multi-start greedy, bounded by a small time budget.
    greedy_budget = min(2.0, 0.25 * time_limit) if time_limit > 0 else 0.0
    greedy_deadline = start_time + greedy_budget
    max_starts = min(n, 4000)
    for idx in range(max_starts):
        if (idx & 15) == 0 and time.time() > greedy_deadline:
            break
        s = by_degree[idx]
        clique = [s]
        cand = adj[s]
        while cand:
            lsb = cand & -cand
            v = lsb.bit_length() - 1
            clique.append(v)
            cand &= adj[v]  # adj[v] excludes v itself
        record(clique)

    # ---- reduction: drop vertices that cannot extend the incumbent ----
    alive = (1 << n) - 1
    if best_size > 0:
        threshold = best_size
        changed = True
        while changed and alive:
            if time.time() > deadline:
                break
            changed = False
            a = alive
            while a:
                lsb = a & -a
                v = lsb.bit_length() - 1
                if _popcount(adj[v] & alive) < threshold:
                    alive &= ~lsb
                    changed = True
                a ^= lsb

    # ---- Tomita branch and bound with greedy-coloring bounds ----
    sys.setrecursionlimit(max(10000, n + 1000))
    nodes = [0]

    def color_sort(P):
        """Greedy coloring of P; returns vertex order and per-vertex color bounds."""
        order = []
        colors = []
        uncolored = P
        color = 0
        adj_l = adj
        while uncolored:
            color += 1
            avail = uncolored
            while avail:
                lsb = avail & -avail
                v = lsb.bit_length() - 1
                order.append(v)
                colors.append(color)
                uncolored ^= lsb
                avail &= ~lsb
                avail &= ~adj_l[v]
        return order, colors

    def expand(R, P):
        nodes[0] += 1
        if (nodes[0] & 1023) == 0 and time.time() > deadline:
            raise _TimeUp
        if not P:
            record(R)
            return
        order, colors = color_sort(P)
        for i in range(len(order) - 1, -1, -1):
            if len(R) + colors[i] <= best_size:
                return
            v = order[i]
            R.append(v)
            expand(R, P & adj[v])
            R.pop()
            P &= ~(1 << v)

    if alive:
        try:
            expand([], alive)
        except _TimeUp:
            pass
        except RecursionError:
            pass

    write_solution()


if __name__ == "__main__":
    main()