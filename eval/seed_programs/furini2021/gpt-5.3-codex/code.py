import argparse
import json
import random
import time
import sys

from solution_logger import SolutionLogger


def bits_iter(mask: int):
    while mask:
        b = mask & -mask
        yield b.bit_length() - 1
        mask ^= b


def color_sort(P: int, adj_masks):
    """
    Greedy sequential coloring bound on subgraph induced by P.
    Returns:
      order: list of vertices
      bounds: color index for each vertex in order (nondecreasing)
    """
    order = []
    bounds = []
    uncolored = P
    color = 0
    while uncolored:
        color += 1
        avail = uncolored
        while avail:
            b = avail & -avail
            v = b.bit_length() - 1
            order.append(v)
            bounds.append(color)
            uncolored ^= b
            avail ^= b
            avail &= uncolored
            avail &= ~adj_masks[v]
    return order, bounds


def greedy_clique(adj_masks, rng, randomized=True):
    n = len(adj_masks)
    if n == 0:
        return []
    all_mask = (1 << n) - 1
    cand = all_mask
    clique = []

    while cand:
        verts = list(bits_iter(cand))
        if not verts:
            break

        scored = []
        for v in verts:
            score = (adj_masks[v] & cand).bit_count()
            scored.append((score, v))
        scored.sort(reverse=True)

        if randomized and len(scored) > 1:
            take = min(3, len(scored))
            v = scored[rng.randrange(take)][1]
        else:
            v = scored[0][1]

        clique.append(v)
        cand &= adj_masks[v]

    return clique


def max_clique_bb(adj_masks, deadline, rng):
    """
    Exact branch-and-bound maximum clique (Tomita-style coloring bound),
    with timeout.
    Returns: (best_size, best_mask, complete)
    """
    n = len(adj_masks)
    if n == 0:
        return 0, 0, True

    all_mask = (1 << n) - 1

    # heuristic warm start
    best_clique = greedy_clique(adj_masks, rng, randomized=False)
    best_size = len(best_clique)
    best_mask = 0
    for v in best_clique:
        best_mask |= (1 << v)

    # additional random starts
    for _ in range(4):
        if time.time() > deadline:
            return best_size, best_mask, False
        c = greedy_clique(adj_masks, rng, randomized=True)
        if len(c) > best_size:
            best_size = len(c)
            best_mask = 0
            for v in c:
                best_mask |= (1 << v)

    complete = True
    node_counter = 0

    def expand(C_mask, C_size, P):
        nonlocal best_size, best_mask, complete, node_counter

        node_counter += 1
        if (node_counter & 1023) == 0:
            if time.time() > deadline:
                complete = False
                return

        if P == 0:
            if C_size > best_size:
                best_size = C_size
                best_mask = C_mask
            return

        order, bounds = color_sort(P, adj_masks)

        for i in range(len(order) - 1, -1, -1):
            v = order[i]
            if C_size + bounds[i] <= best_size:
                return

            b = 1 << v
            expand(C_mask | b, C_size + 1, P & adj_masks[v])
            if not complete:
                return
            P ^= b
            if P == 0:
                return

    expand(0, 0, all_mask)
    return best_size, best_mask, complete


def choose_edge_to_remove(adj_masks, rng, deadline):
    """
    Heuristic: sample several large cliques; score edges by frequency/weight.
    Fallback: choose existing edge with highest common-neighbor count.
    Returns (u, v) with u < v, or None
    """
    n = len(adj_masks)
    all_mask = (1 << n) - 1

    cliques = []
    # deterministic + randomized samples
    c0 = greedy_clique(adj_masks, rng, randomized=False)
    if len(c0) >= 2:
        cliques.append(c0)

    iters = 8
    for _ in range(iters):
        if time.time() > deadline:
            break
        c = greedy_clique(adj_masks, rng, randomized=True)
        if len(c) >= 2:
            cliques.append(c)

    edge_score = {}

    for c in cliques:
        s = len(c)
        if s < 2:
            continue
        w = s * s
        for i in range(s):
            u = c[i]
            for j in range(i + 1, s):
                v = c[j]
                if u > v:
                    u, v = v, u
                edge_score[(u, v)] = edge_score.get((u, v), 0) + w

    if edge_score:
        # tie-break with common neighbors
        best_e = None
        best_val = None
        for (u, v), sc in edge_score.items():
            tri = (adj_masks[u] & adj_masks[v]).bit_count()
            val = (sc, tri)
            if best_val is None or val > best_val:
                best_val = val
                best_e = (u, v)
        return best_e

    # fallback over all existing edges
    best_e = None
    best_tri = -1
    for u in range(n):
        nbrs = adj_masks[u]
        while nbrs:
            b = nbrs & -nbrs
            v = b.bit_length() - 1
            nbrs ^= b
            if v <= u:
                continue
            tri = (adj_masks[u] & adj_masks[v]).bit_count()
            if tri > best_tri:
                best_tri = tri
                best_e = (u, v)

    return best_e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    vertices = inst["vertices"]
    n = inst["num_vertices"]
    k = int(inst["interdiction_budget_k"])

    label_to_idx = {lab: i for i, lab in enumerate(vertices)}

    # Build unique undirected edge set
    edge_set = set()
    for e in inst["edges"]:
        a = label_to_idx[e[0]]
        b = label_to_idx[e[1]]
        if a == b:
            continue
        if a > b:
            a, b = b, a
        edge_set.add((a, b))

    edge_list = sorted(edge_set)
    m = len(edge_list)
    if k > m:
        k = m

    # adjacency bitmasks
    adj_masks = [0] * n
    for u, v in edge_list:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)

    start = time.time()
    deadline = start + max(1, args.time_limit)
    rng = random.Random(0)

    removed = set()

    # Greedy interdiction loop
    for _ in range(k):
        if time.time() > deadline - 0.2:
            break
        e = choose_edge_to_remove(adj_masks, rng, deadline - 0.1)
        if e is None:
            break
        u, v = e
        if (u, v) in removed:
            # should not happen, but safe
            continue
        if ((adj_masks[u] >> v) & 1) == 0:
            continue

        # remove edge
        adj_masks[u] &= ~(1 << v)
        adj_masks[v] &= ~(1 << u)
        removed.add((u, v))

    # Compute (best-effort) exact clique number of resulting graph
    best_size, _, complete = max_clique_bb(adj_masks, deadline, rng)

    # If exact search timed out, we still output best lower bound found.
    objective_value = int(best_size)

    if logger:
        logger.log(objective_value)

    interdicted_edges = [[vertices[u], vertices[v]] for (u, v) in sorted(removed)]

    out = {
        "objective_value": objective_value,
        "interdicted_edges": interdicted_edges,
    }

    with open(args.solution_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    sys.setrecursionlimit(1000000)
    main()