import argparse
import json
import random
import time
from collections import deque

from solution_logger import SolutionLogger


def build_graph(graph_data):
    vertices = list(graph_data["vertices"])
    adj = {v: set() for v in vertices}
    for e in graph_data["edges"]:
        if not isinstance(e, list) or len(e) != 2:
            continue
        u, v = e
        if u == v:
            continue
        if u in adj and v in adj:
            adj[u].add(v)
            adj[v].add(u)
    return vertices, adj


def is_connected_subset(subset, adj):
    if len(subset) <= 1:
        return True
    sset = set(subset)
    start = next(iter(sset))
    q = deque([start])
    seen = {start}
    while q:
        u = q.popleft()
        for w in adj[u]:
            if w in sset and w not in seen:
                seen.add(w)
                q.append(w)
    return len(seen) == len(sset)


def is_splex(subset, adj, s):
    k = len(subset)
    if k <= 1:
        return True
    req = k - s
    if req <= 0:
        return True
    sset = set(subset)
    for u in sset:
        deg_in = 0
        nu = adj[u]
        for v in sset:
            if v != u and v in nu:
                deg_in += 1
        if deg_in < req:
            return False
    return True


def can_add_vertex(group_list, group_set, deg_map, v, adj, s, connectivity_required):
    if v in group_set:
        return False, 0
    k = len(group_list)

    if connectivity_required and k > 0:
        if not any((v in adj[u]) for u in group_list):
            return False, 0

    new_size = k + 1
    req = new_size - s

    deg_v = 0
    for u in group_list:
        if v in adj[u]:
            deg_v += 1

    if req > 0 and deg_v < req:
        return False, deg_v

    if req > 0:
        for u in group_list:
            new_deg_u = deg_map[u] + (1 if v in adj[u] else 0)
            if new_deg_u < req:
                return False, deg_v

    return True, deg_v


def expand_group(initial_group, vertex_order, adj, s, connectivity_required, randomize=False):
    group_list = list(initial_group)
    group_set = set(initial_group)

    if not is_splex(group_set, adj, s):
        return frozenset(initial_group)
    if connectivity_required and not is_connected_subset(group_set, adj):
        return frozenset(initial_group)

    deg_map = {u: 0 for u in group_list}
    for i in range(len(group_list)):
        u = group_list[i]
        for j in range(i + 1, len(group_list)):
            v = group_list[j]
            if v in adj[u]:
                deg_map[u] += 1
                deg_map[v] += 1

    while True:
        feasible = []
        for v in vertex_order:
            if v in group_set:
                continue
            ok, deg_v = can_add_vertex(group_list, group_set, deg_map, v, adj, s, connectivity_required)
            if ok:
                # Prefer high internal adjacency first, then global degree
                feasible.append((deg_v, len(adj[v]), v))

        if not feasible:
            break

        feasible.sort(reverse=True)
        if randomize and len(feasible) > 1:
            top = feasible[: min(4, len(feasible))]
            _, _, chosen = random.choice(top)
        else:
            _, _, chosen = feasible[0]

        old = list(group_list)
        group_list.append(chosen)
        group_set.add(chosen)
        deg_map[chosen] = 0
        for u in old:
            if chosen in adj[u]:
                deg_map[u] += 1
                deg_map[chosen] += 1

    return frozenset(group_set)


def generate_candidates(vertices, adj, s, connectivity_required, start_time, time_limit):
    n = len(vertices)
    deg = {v: len(adj[v]) for v in vertices}
    by_deg = sorted(vertices, key=lambda x: deg[x], reverse=True)

    max_candidates = min(10000, max(800, 50 * n))
    candidates = []
    cand_set = set()

    def add_candidate(c):
        if not c or c in cand_set:
            return
        # Safety checks
        if not is_splex(c, adj, s):
            return
        if connectivity_required and not is_connected_subset(c, adj):
            return
        cand_set.add(c)
        candidates.append(c)

    # Always include singletons
    for v in vertices:
        add_candidate(frozenset([v]))

    gen_budget = max(0.5, min(time_limit * 0.45, time_limit - 0.5))
    gen_deadline = start_time + gen_budget

    # Deterministic from seeds
    for seed in by_deg:
        if time.time() >= gen_deadline or len(candidates) >= max_candidates:
            break
        c = expand_group([seed], by_deg, adj, s, connectivity_required, randomize=False)
        add_candidate(c)

    # Deterministic from edge seeds
    for u in by_deg:
        if time.time() >= gen_deadline or len(candidates) >= max_candidates:
            break
        nbrs = sorted(adj[u], key=lambda x: deg[x], reverse=True)
        for v in nbrs[: min(4, len(nbrs))]:
            if time.time() >= gen_deadline or len(candidates) >= max_candidates:
                break
            c = expand_group([u, v], by_deg, adj, s, connectivity_required, randomize=False)
            add_candidate(c)

    # Randomized expansion
    random.seed(0)
    while time.time() < gen_deadline and len(candidates) < max_candidates:
        seeds = vertices[:]
        random.shuffle(seeds)
        for seed in seeds[: min(30, len(seeds))]:
            if time.time() >= gen_deadline or len(candidates) >= max_candidates:
                break
            order = vertices[:]
            random.shuffle(order)
            c = expand_group([seed], order, adj, s, connectivity_required, randomize=True)
            add_candidate(c)

    return candidates


def partition_objective_from_selected(selected_groups, n):
    # objective = #selected_groups + #uncovered_singletons = n - total_gain
    gain = sum(len(g) - 1 for g in selected_groups)
    return n - gain


def construct_partition_grasp(vertices, multi_candidates, rng, alpha=0.3):
    # Greedy randomized maximal disjoint packing maximizing gains (|C|-1)
    used = set()
    selected = []

    # randomized priority list
    scored = []
    for c in multi_candidates:
        gain = len(c) - 1
        # random perturbation
        score = gain + alpha * rng.random()
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)

    for _, c in scored:
        if used.isdisjoint(c):
            selected.append(set(c))
            used.update(c)

    # Augment pass by pure gain (in case randomized order missed obvious additions)
    by_gain = sorted(multi_candidates, key=lambda c: (len(c) - 1, len(c)), reverse=True)
    improved = True
    while improved:
        improved = False
        for c in by_gain:
            if used.isdisjoint(c):
                selected.append(set(c))
                used.update(c)
                improved = True

    return selected


def local_improve_partition(selected, multi_candidates, vertices, vidx, deadline):
    # Improvement via replacing up to 2 intersecting groups by one better group
    selected = [set(g) for g in selected]

    while time.time() < deadline:
        improved = False

        owner = [-1] * len(vertices)
        for gi, g in enumerate(selected):
            for v in g:
                owner[vidx[v]] = gi
        gains = [len(g) - 1 for g in selected]

        for c in multi_candidates:
            if time.time() >= deadline:
                break

            owners = set()
            for v in c:
                oi = owner[vidx[v]]
                if oi != -1:
                    owners.add(oi)
                    if len(owners) > 2:
                        break
            if len(owners) > 2:
                continue

            rem_gain = sum(gains[o] for o in owners)
            add_gain = len(c) - 1

            if add_gain > rem_gain:
                keep = []
                for i, g in enumerate(selected):
                    if i not in owners:
                        keep.append(g)
                keep.append(set(c))
                selected = keep
                improved = True
                break

        if not improved:
            break

    return selected


def build_partition_solution(vertices, selected_groups):
    used = set()
    sol = []
    for g in selected_groups:
        if used.isdisjoint(g):
            sol.append(sorted(g))
            used.update(g)
    for v in vertices:
        if v not in used:
            sol.append([v])
    return sol


def construct_cover_greedy(vertices, candidates, rng, randomized=True):
    uncovered = set(vertices)
    selected = []
    candidate_sets = [set(c) for c in candidates]

    while uncovered:
        best_cov = -1
        best = []
        for i, c in enumerate(candidate_sets):
            cov = len(c & uncovered)
            if cov > best_cov:
                best_cov = cov
                best = [i]
            elif cov == best_cov:
                best.append(i)

        if best_cov <= 0:
            # fallback singleton for one uncovered vertex
            v = next(iter(uncovered))
            selected.append({v})
            uncovered.remove(v)
            continue

        if randomized and len(best) > 1:
            # tie-break among best by size, randomized in top few
            best.sort(key=lambda i: len(candidate_sets[i]), reverse=True)
            rcl = best[: min(5, len(best))]
            chosen_idx = rng.choice(rcl)
        else:
            chosen_idx = max(best, key=lambda i: len(candidate_sets[i]))

        chosen = candidate_sets[chosen_idx]
        selected.append(set(chosen))
        uncovered -= chosen

    # Redundancy elimination
    cover_count = {v: 0 for v in vertices}
    for g in selected:
        for v in g:
            cover_count[v] += 1

    order = list(range(len(selected)))
    rng.shuffle(order)
    keep = [True] * len(selected)

    for i in order:
        g = selected[i]
        if all(cover_count[v] >= 2 for v in g):
            keep[i] = False
            for v in g:
                cover_count[v] -= 1

    pruned = [selected[i] for i in range(len(selected)) if keep[i]]
    return pruned


def ensure_cover_valid(vertices, groups):
    covered = set()
    for g in groups:
        covered.update(g)
    if len(covered) == len(vertices):
        return groups
    # add missing as singletons
    missing = [v for v in vertices if v not in covered]
    for v in missing:
        groups.append({v})
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()
    deadline = start_time + max(1, args.time_limit)

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    graph_data = data["graph"]
    settings = data["problem_settings"]

    vertices, adj = build_graph(graph_data)
    n = len(vertices)

    s = int(settings["parameter_s"])
    decomposition_type = str(settings["decomposition_type"]).strip().lower()
    connectivity_required = bool(settings["connectivity_required"])

    candidates = generate_candidates(
        vertices=vertices,
        adj=adj,
        s=s,
        connectivity_required=connectivity_required,
        start_time=start_time,
        time_limit=args.time_limit,
    )

    # Separate candidates
    multi_candidates = [set(c) for c in candidates if len(c) > 1]
    # Sort once for local search (high gain first)
    multi_candidates.sort(key=lambda c: (len(c) - 1, len(c)), reverse=True)

    best_obj = float("inf")
    best_solution = None

    rng = random.Random(1)

    def update_best(groups):
        nonlocal best_obj, best_solution
        obj = float(len(groups))
        if obj + 1e-9 < best_obj:
            best_obj = obj
            best_solution = [sorted(list(g)) for g in groups]
            if logger:
                logger.log(best_obj)

    # Initial baseline: all singletons (always feasible)
    baseline = [[v] for v in vertices]
    update_best(baseline)

    if decomposition_type == "partitioning":
        vidx = {v: i for i, v in enumerate(vertices)}

        # Multi-start GRASP + local search
        iter_count = 0
        while time.time() < deadline - 0.02:
            iter_count += 1
            alpha = 0.15 + 0.5 * rng.random()
            selected = construct_partition_grasp(vertices, multi_candidates, rng, alpha=alpha)

            # local improve with a per-iteration mini-budget
            rem = deadline - time.time()
            if rem <= 0.02:
                break
            local_deadline = min(deadline - 0.01, time.time() + min(0.2, rem * 0.5))
            selected = local_improve_partition(selected, multi_candidates, vertices, vidx, local_deadline)

            sol = build_partition_solution(vertices, selected)
            update_best(sol)

            # if very good theoretical lower bound reached (1 group), stop
            if best_obj <= 1.0:
                break

            # Avoid too many loops for tiny instances
            if n <= 30 and iter_count >= 100:
                break

    else:  # covering
        # Multi-start greedy cover + pruning
        iter_count = 0
        while time.time() < deadline - 0.02:
            iter_count += 1
            selected = construct_cover_greedy(vertices, candidates, rng, randomized=True)
            selected = ensure_cover_valid(vertices, selected)
            update_best(selected)

            if best_obj <= 1.0:
                break
            if n <= 30 and iter_count >= 120:
                break

    # Final safety fallback
    if best_solution is None:
        best_solution = [[v] for v in vertices]
        best_obj = float(len(best_solution))
        if logger:
            logger.log(best_obj)

    output = {
        "objective_value": float(best_obj),
        "solution": best_solution,
    }

    with open(args.solution_path, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()