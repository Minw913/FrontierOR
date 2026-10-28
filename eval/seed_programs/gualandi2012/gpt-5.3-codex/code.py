import argparse
import ast
import json
import random
import time
from collections import defaultdict

from solution_logger import SolutionLogger


def _parse_graph_dict(data):
    """Parse JSON-like dict following expected schema variants."""
    if not isinstance(data, dict):
        return None

    if "graph" in data and isinstance(data["graph"], dict):
        g = data["graph"]
    else:
        g = data

    if "num_vertices" not in g:
        return None

    n = int(g["num_vertices"])
    edges_raw = g.get("edges", [])
    edges = []
    for e in edges_raw:
        if isinstance(e, (list, tuple)) and len(e) >= 2:
            u, v = int(e[0]), int(e[1])
            edges.append((u, v))
    return n, edges


def _parse_text_graph(text):
    """
    Fallback parser for non-JSON formats:
    - DIMACS .col: lines with 'p edge n m' and 'e u v'
    - Simple edge list: first numeric line 'n m', following lines 'u v'
    """
    lines = text.splitlines()
    n = None
    edges = []
    first_numeric_line_used_as_header = False

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("c", "#", "//")):
            continue

        parts = line.split()
        if not parts:
            continue

        head = parts[0].lower()

        # DIMACS problem line
        if head == "p" and len(parts) >= 3:
            # Typical: p edge n m
            for tok in reversed(parts):
                if tok.lstrip("-").isdigit():
                    # first from right is m, then n
                    pass
            nums = [int(tok) for tok in parts if tok.lstrip("-").isdigit()]
            if len(nums) >= 2:
                n = nums[0]
            continue

        # DIMACS edge line
        if head == "e" and len(parts) >= 3 and parts[1].lstrip("-").isdigit() and parts[2].lstrip("-").isdigit():
            u, v = int(parts[1]), int(parts[2])
            edges.append((u, v))
            continue

        # Generic numeric line
        nums = []
        ok = True
        for tok in parts:
            if tok.lstrip("-").isdigit():
                nums.append(int(tok))
            else:
                ok = False
                break
        if not ok or len(nums) < 2:
            continue

        if n is None and not first_numeric_line_used_as_header and len(nums) >= 2:
            # Heuristic: first numeric line can be "n m"
            n = nums[0]
            first_numeric_line_used_as_header = True
            continue

        u, v = nums[0], nums[1]
        edges.append((u, v))

    if n is None:
        if not edges:
            return 0, []
        max_idx = max(max(u, v) for u, v in edges)
        min_idx = min(min(u, v) for u, v in edges)
        # guess base
        if min_idx >= 1:
            n = max_idx
        else:
            n = max_idx + 1

    # Detect 1-based indexing and convert to 0-based
    if edges:
        min_idx = min(min(u, v) for u, v in edges)
        max_idx = max(max(u, v) for u, v in edges)
        if min_idx >= 1 and max_idx <= n:
            edges = [(u - 1, v - 1) for u, v in edges]

    return int(n), edges


def load_instance(path):
    with open(path, "rb") as f:
        raw = f.read()

    # Decode robustly
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin1")

    # 1) Strict JSON
    try:
        data = json.loads(text)
        parsed = _parse_graph_dict(data)
        if parsed is not None:
            return parsed
    except Exception:
        pass

    # 2) Python-literal-like dict
    try:
        data = ast.literal_eval(text)
        parsed = _parse_graph_dict(data)
        if parsed is not None:
            return parsed
    except Exception:
        pass

    # 3) Text fallback
    return _parse_text_graph(text)


def build_graph(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if 0 <= u < n and 0 <= v < n and u != v:
            adj[u].add(v)
            adj[v].add(u)
    return [list(nei) for nei in adj]


def normalize_colors(colors):
    used = sorted(set(colors))
    mp = {c: i for i, c in enumerate(used)}
    out = [mp[c] for c in colors]
    return out, len(used)


def greedy_coloring_order(adj, order):
    n = len(adj)
    colors = [-1] * n
    used_stamp = [0] * max(1, n)
    stamp = 0
    max_color = -1

    for v in order:
        stamp += 1
        for nb in adj[v]:
            c = colors[nb]
            if c >= 0 and c < len(used_stamp):
                used_stamp[c] = stamp
        c = 0
        while c <= max_color and used_stamp[c] == stamp:
            c += 1
        colors[v] = c
        if c > max_color:
            max_color = c
            if max_color >= len(used_stamp):
                used_stamp.extend([0] * len(used_stamp))

    return colors, max_color + 1


def dsatur_coloring(adj):
    n = len(adj)
    if n == 0:
        return [], 0

    degrees = [len(adj[v]) for v in range(n)]
    uncolored = set(range(n))
    colors = [-1] * n
    sat_colors = [set() for _ in range(n)]

    max_color = -1
    while uncolored:
        best = None
        best_key = None
        for v in uncolored:
            key = (len(sat_colors[v]), degrees[v], -v)
            if best is None or key > best_key:
                best = v
                best_key = key

        used = sat_colors[best]
        c = 0
        while c in used:
            c += 1
        colors[best] = c
        if c > max_color:
            max_color = c

        uncolored.remove(best)
        for nb in adj[best]:
            if nb in uncolored:
                sat_colors[nb].add(c)

    return colors, max_color + 1


def is_feasible(colors, edges):
    for u, v in edges:
        if u != v and colors[u] == colors[v]:
            return False
    return True


def greedy_clique_lower_bound(adj):
    n = len(adj)
    if n == 0:
        return 0
    neigh_sets = [set(lst) for lst in adj]
    order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    clique = []
    for v in order:
        ok = True
        nv = neigh_sets[v]
        for u in clique:
            if u not in nv:
                ok = False
                break
        if ok:
            clique.append(v)
    return max(1, len(clique))


def build_initial_for_kminus1(adj, colors, k_target, rng):
    k_current = max(colors) + 1
    if k_current == k_target:
        return list(colors)

    classes = [[] for _ in range(k_current)]
    for v, c in enumerate(colors):
        classes[c].append(v)
    remove_c = min(range(k_current), key=lambda c: len(classes[c]))

    mp = {}
    idx = 0
    for c in range(k_current):
        if c == remove_c:
            continue
        if idx < k_target:
            mp[c] = idx
            idx += 1

    n = len(colors)
    assign = [-1] * n
    removed = []
    for v, c in enumerate(colors):
        if c == remove_c:
            removed.append(v)
        else:
            assign[v] = mp[c]

    removed.sort(key=lambda v: len(adj[v]), reverse=True)
    for v in removed:
        cnt = [0] * k_target
        for nb in adj[v]:
            cnb = assign[nb]
            if cnb >= 0:
                cnt[cnb] += 1
        best = min(cnt)
        choices = [c for c in range(k_target) if cnt[c] == best]
        assign[v] = rng.choice(choices)
    return assign


def min_conflicts_k_color(adj, edges, k, init_colors, max_steps, rng, time_end):
    n = len(adj)
    if n == 0:
        return []

    if n * k > 5_000_000:  # memory safety for single-core container
        return None

    colors = list(init_colors)
    neighbor_count = [[0] * k for _ in range(n)]

    for v in range(n):
        cv = colors[v]
        for nb in adj[v]:
            if nb > v:
                cn = colors[nb]
                neighbor_count[v][cn] += 1
                neighbor_count[nb][cv] += 1

    vertex_conf = [neighbor_count[v][colors[v]] for v in range(n)]

    edge_conflicts = 0
    for u, v in edges:
        if u != v and colors[u] == colors[v]:
            edge_conflicts += 1

    conf_list = []
    pos = [-1] * n

    def add_conf(v):
        if pos[v] == -1:
            pos[v] = len(conf_list)
            conf_list.append(v)

    def rem_conf(v):
        p = pos[v]
        if p != -1:
            last = conf_list[-1]
            conf_list[p] = last
            pos[last] = p
            conf_list.pop()
            pos[v] = -1

    for v in range(n):
        if vertex_conf[v] > 0:
            add_conf(v)

    best_edge_conf = edge_conflicts
    no_improve = 0

    for _ in range(max_steps):
        if time.time() >= time_end:
            return None
        if edge_conflicts == 0:
            return colors
        if not conf_list:
            return colors

        v = conf_list[rng.randrange(len(conf_list))]
        old = colors[v]

        best_val = None
        best_cols = []
        for c in range(k):
            if c == old:
                continue
            val = neighbor_count[v][c]
            if best_val is None or val < best_val:
                best_val = val
                best_cols = [c]
            elif val == best_val:
                best_cols.append(c)

        if not best_cols:
            continue

        new = rng.choice(best_cols)
        if no_improve > 1500 and rng.random() < 0.2:
            new = rng.randrange(k)
            while new == old and k > 1:
                new = rng.randrange(k)

        if new == old:
            continue

        for nb in adj[v]:
            if colors[nb] == old:
                edge_conflicts -= 1
            if colors[nb] == new:
                edge_conflicts += 1

            neighbor_count[nb][old] -= 1
            neighbor_count[nb][new] += 1

        colors[v] = new

        vertex_conf[v] = neighbor_count[v][new]
        if vertex_conf[v] > 0:
            add_conf(v)
        else:
            rem_conf(v)

        for nb in adj[v]:
            vertex_conf[nb] = neighbor_count[nb][colors[nb]]
            if vertex_conf[nb] > 0:
                add_conf(nb)
            else:
                rem_conf(nb)

        if edge_conflicts < best_edge_conf:
            best_edge_conf = edge_conflicts
            no_improve = 0
        else:
            no_improve += 1

    return None


def to_output(colors):
    color_classes = defaultdict(list)
    for v, c in enumerate(colors):
        color_classes[str(int(c))].append(v)
    for c in color_classes:
        color_classes[c].sort()

    return {
        "objective_value": int(len(set(colors))),
        "coloring": {str(v): int(colors[v]) for v in range(len(colors))},
        "color_classes": {k: color_classes[k] for k in sorted(color_classes, key=lambda x: int(x))}
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start = time.time()
    total_limit = max(1, int(args.time_limit))
    time_end = start + total_limit - 0.01

    n, edges = load_instance(args.instance_path)
    if n <= 0:
        sol = {"objective_value": 0, "coloring": {}, "color_classes": {}}
        with open(args.solution_path, "w") as f:
            json.dump(sol, f, indent=2)
        return

    adj = build_graph(n, edges)
    degrees = [len(adj[v]) for v in range(n)]
    rng = random.Random(1234567)

    # Initial greedy
    order = sorted(range(n), key=lambda v: degrees[v], reverse=True)
    best_colors, best_k = greedy_coloring_order(adj, order)
    best_colors, best_k = normalize_colors(best_colors)
    if logger:
        logger.log(best_k)

    # DSATUR for moderate sizes
    if n <= 2500 and time.time() < time_end:
        ds_colors, ds_k = dsatur_coloring(adj)
        ds_colors, ds_k = normalize_colors(ds_colors)
        if ds_k < best_k and is_feasible(ds_colors, edges):
            best_colors, best_k = ds_colors, ds_k
            if logger:
                logger.log(best_k)

    # Randomized greedy multisarts (time-budgeted)
    while time.time() < time_end and (time.time() - start) < 0.35 * total_limit:
        verts = list(range(n))
        rng.shuffle(verts)
        verts.sort(key=lambda v: (degrees[v], rng.random()), reverse=True)
        cand, ck = greedy_coloring_order(adj, verts)
        cand, ck = normalize_colors(cand)
        if ck < best_k and is_feasible(cand, edges):
            best_colors, best_k = cand, ck
            if logger:
                logger.log(best_k)

    lb = greedy_clique_lower_bound(adj)
    lb = min(lb, best_k)

    # Try reducing one color at a time with min-conflicts
    while time.time() < time_end and best_k > lb:
        target_k = best_k - 1
        success = False

        remaining = time_end - time.time()
        if remaining <= 0.02:
            break
        target_deadline = min(time_end, time.time() + max(0.2, 0.6 * remaining))

        attempts = 0
        while time.time() < target_deadline:
            attempts += 1
            init = build_initial_for_kminus1(adj, best_colors, target_k, rng)
            max_steps = min(250000, max(5000, 20 * len(edges) // max(1, target_k) + 2000))

            sol = min_conflicts_k_color(
                adj=adj,
                edges=edges,
                k=target_k,
                init_colors=init,
                max_steps=max_steps,
                rng=rng,
                time_end=target_deadline,
            )

            if sol is not None:
                sol, sk = normalize_colors(sol)
                if sk <= target_k and is_feasible(sol, edges):
                    best_colors, best_k = sol, sk
                    if logger:
                        logger.log(best_k)
                    success = True
                    break

            if attempts >= 20:
                break

        if not success:
            break

    # final safety
    if not is_feasible(best_colors, edges):
        order = sorted(range(n), key=lambda v: degrees[v], reverse=True)
        best_colors, best_k = greedy_coloring_order(adj, order)
        best_colors, best_k = normalize_colors(best_colors)

    out = to_output(best_colors)
    with open(args.solution_path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()