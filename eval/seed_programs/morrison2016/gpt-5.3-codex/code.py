import argparse
import json
import time
import random
from typing import List, Set, Tuple, Dict, Optional

from solution_logger import SolutionLogger


def read_instance(path: str):
    with open(path, "r") as f:
        data = json.load(f)

    g = data["graph"]
    n = int(g["num_vertices"])
    raw_edges = g["edges"]

    edge_set = set()
    for e in raw_edges:
        u, v = int(e[0]), int(e[1])
        if u == v:
            continue
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    edges = sorted(edge_set)

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    bounds = data.get("bounds", {})
    lb = int(bounds.get("clique_lower_bound", 1))

    return n, edges, adj, lb


def compact_coloring(colors: List[int]):
    used = sorted(set(colors))
    remap = {c: i for i, c in enumerate(used)}
    new_colors = [remap[c] for c in colors]
    k = len(used)
    classes = [[] for _ in range(k)]
    for v, c in enumerate(new_colors):
        classes[c].append(v)
    for cls in classes:
        cls.sort()
    return new_colors, classes, k


def build_solution(colors: List[int]):
    colors, classes, k = compact_coloring(colors)
    return {
        "objective_value": int(k),
        "selected_sets": classes,
        "coloring": {str(v): int(c) for v, c in enumerate(colors)},
    }


def dsatur_coloring(n: int, adj: List[Set[int]], rng: random.Random, randomized: bool):
    if n == 0:
        return [], 0

    colors = [-1] * n
    uncolored = set(range(n))
    neighbor_colors = [set() for _ in range(n)]
    deg = [len(adj[v]) for v in range(n)]

    while uncolored:
        best_sat = -1
        best_deg = -1
        candidates = []

        for v in uncolored:
            sat = len(neighbor_colors[v])
            dv = deg[v]
            if sat > best_sat or (sat == best_sat and dv > best_deg):
                best_sat = sat
                best_deg = dv
                candidates = [v]
            elif sat == best_sat and dv == best_deg:
                candidates.append(v)

        v = rng.choice(candidates) if randomized and len(candidates) > 1 else min(candidates)

        forbidden = neighbor_colors[v]
        feasible = []
        for c in range(n + 1):
            if c not in forbidden:
                feasible.append(c)
                if not randomized:
                    break

        if randomized and len(feasible) > 1:
            # bias toward small colors while allowing diversification
            if rng.random() < 0.75:
                c = feasible[0]
            else:
                c = feasible[rng.randrange(min(len(feasible), 6))]
        else:
            c = feasible[0]

        colors[v] = c
        uncolored.remove(v)
        for w in adj[v]:
            if w in uncolored:
                neighbor_colors[w].add(c)

    k = max(colors) + 1
    return colors, k


def try_recolor_subset(
    S: List[int],
    colors: List[int],
    target_k: int,
    adj: List[Set[int]],
    deadline: float,
) -> Optional[Dict[int, int]]:
    """
    Try to color vertices in S using colors [0..target_k-1], keeping all other vertices fixed.
    Returns assignment dict {v: color} if feasible, else None.
    """
    if not S:
        return {}

    Sset = set(S)

    # Initial domains considering fixed neighbors
    domains = {}
    for v in S:
        forbidden = set()
        for u in adj[v]:
            if u not in Sset:
                cu = colors[u]
                if 0 <= cu < target_k:
                    forbidden.add(cu)
        dom = [c for c in range(target_k) if c not in forbidden]
        if not dom:
            return None
        domains[v] = dom

    # adjacency within S
    s_neighbors = {v: [u for u in adj[v] if u in Sset] for v in S}

    assignment = {}

    def available_colors(v):
        if time.time() > deadline:
            return []
        used = set()
        for u in s_neighbors[v]:
            if u in assignment:
                used.add(assignment[u])
        return [c for c in domains[v] if c not in used]

    def select_var(unassigned):
        # MRV + degree tie-break
        best_v = None
        best_len = 10**9
        best_deg = -1
        for v in unassigned:
            av = available_colors(v)
            l = len(av)
            if l < best_len:
                best_len = l
                best_v = v
                best_deg = len(s_neighbors[v])
            elif l == best_len:
                d = len(s_neighbors[v])
                if d > best_deg:
                    best_v = v
                    best_deg = d
        return best_v

    unassigned = set(S)

    def dfs():
        if time.time() > deadline:
            return False
        if not unassigned:
            return True

        v = select_var(unassigned)
        if v is None:
            return False

        vals = available_colors(v)
        if not vals:
            return False

        # least-constraining value
        def impact(c):
            imp = 0
            for u in s_neighbors[v]:
                if u in unassigned and u != v:
                    # if u can use c, then assigning c to v removes one option for u
                    if c in domains[u]:
                        imp += 1
            return imp

        vals.sort(key=impact)

        unassigned.remove(v)
        for c in vals:
            assignment[v] = c
            if dfs():
                return True
            del assignment[v]
        unassigned.add(v)
        return False

    ok = dfs()
    if not ok:
        return None
    return assignment


def attempt_reduce_one_color(
    colors: List[int],
    classes: List[List[int]],
    adj: List[Set[int]],
    deadline: float,
):
    """
    Attempt to reduce number of colors by 1 by eliminating a small color class.
    Returns (new_colors, success).
    """
    k = len(classes)
    if k <= 1:
        return colors, False

    # pick smallest class to eliminate
    sizes = [(len(classes[c]), c) for c in range(k)]
    sizes.sort()
    _, remove_c = sizes[0]
    last_c = k - 1

    work = colors[:]

    # swap remove_c and last_c in coloring to always eliminate last color
    if remove_c != last_c:
        for v in range(len(work)):
            if work[v] == remove_c:
                work[v] = last_c
            elif work[v] == last_c:
                work[v] = remove_c

    S = [v for v, c in enumerate(work) if c == last_c]  # vertices to recolor using 0..k-2
    target_k = k - 1

    # quick greedy pass first
    greedy_assignment = {}
    for v in sorted(S, key=lambda x: len(adj[x]), reverse=True):
        forbidden = set()
        for u in adj[v]:
            cu = work[u] if u not in greedy_assignment else greedy_assignment[u]
            if 0 <= cu < target_k:
                forbidden.add(cu)
        placed = False
        for c in range(target_k):
            if c not in forbidden:
                greedy_assignment[v] = c
                placed = True
                break
        if not placed:
            greedy_assignment = None
            break

    if greedy_assignment is not None:
        for v, c in greedy_assignment.items():
            work[v] = c
        return work, True

    # exact backtracking on S with a time slice
    assign = try_recolor_subset(S, work, target_k, adj, deadline)
    if assign is None:
        return colors, False

    for v, c in assign.items():
        work[v] = c
    return work, True


def improve_coloring(
    best_colors: List[int],
    adj: List[Set[int]],
    lb: int,
    start_time: float,
    time_limit: float,
    logger: Optional[SolutionLogger],
):
    colors, classes, k = compact_coloring(best_colors)

    if logger:
        logger.log(k)

    while k > lb and time.time() - start_time < time_limit:
        remaining = time_limit - (time.time() - start_time)
        if remaining <= 0:
            break

        # give a bounded slice for one reduction attempt
        slice_time = min(remaining, 1.5)
        deadline = time.time() + slice_time

        new_colors, ok = attempt_reduce_one_color(colors, classes, adj, deadline)
        if not ok:
            break

        colors, classes, new_k = compact_coloring(new_colors)
        if new_k < k:
            k = new_k
            if logger:
                logger.log(k)
        else:
            break

    return colors, k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    n, edges, adj, lb = read_instance(args.instance_path)
    _ = edges  # not needed beyond validation/parsing, kept for schema completeness

    start = time.time()
    T = max(1, int(args.time_limit))
    rng = random.Random(1234567)

    # Initial deterministic DSATUR
    best_colors, best_k = dsatur_coloring(n, adj, rng, randomized=False)
    best_colors, _, best_k = compact_coloring(best_colors)
    if logger:
        logger.log(best_k)

    # Multi-start randomized DSATUR (most useful on larger instances)
    # Reserve ~40% time for reduction phase
    construction_budget = max(0.0, 0.6 * T)
    while time.time() - start < construction_budget:
        cand_colors, cand_k = dsatur_coloring(n, adj, rng, randomized=True)
        cand_colors, _, cand_k = compact_coloring(cand_colors)
        if cand_k < best_k:
            best_colors, best_k = cand_colors, cand_k
            if logger:
                logger.log(best_k)
            if best_k <= lb:
                break

    # Try to reduce colors by recoloring one class at a time
    if best_k > lb and time.time() - start < T:
        best_colors, best_k = improve_coloring(
            best_colors, adj, lb, start, T, logger
        )

    solution = build_solution(best_colors)
    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()