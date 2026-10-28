import argparse
import json
import time
import random
from collections import defaultdict, deque

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_graph(graph_data):
    vertices = list(graph_data["vertices"])
    edges = graph_data["edges"]
    adj = {v: set() for v in vertices}
    for e in edges:
        if len(e) != 2:
            continue
        u, v = e
        if u == v:
            continue
        if u not in adj or v not in adj:
            continue
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
        for v in sset:
            if v != u and v in adj[u]:
                deg_in += 1
        if deg_in < req:
            return False
    return True


def can_add_vertex(group_list, group_set, deg_map, v, adj, s, connectivity_required):
    k = len(group_list)
    if v in group_set:
        return False, 0
    if connectivity_required and k > 0:
        has_link = False
        for u in group_list:
            if v in adj[u]:
                has_link = True
                break
        if not has_link:
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


def expand_group(initial_group, vertices, adj, s, connectivity_required, randomize=False):
    group_list = list(initial_group)
    group_set = set(initial_group)

    # Build current internal degree map
    deg_map = {u: 0 for u in group_list}
    for i in range(len(group_list)):
        u = group_list[i]
        for j in range(i + 1, len(group_list)):
            v = group_list[j]
            if v in adj[u]:
                deg_map[u] += 1
                deg_map[v] += 1

    # Validate initial
    if not is_splex(group_set, adj, s):
        return frozenset(initial_group)
    if connectivity_required and not is_connected_subset(group_set, adj):
        return frozenset(initial_group)

    while True:
        feasible = []
        for v in vertices:
            if v in group_set:
                continue
            ok, deg_v = can_add_vertex(group_list, group_set, deg_map, v, adj, s, connectivity_required)
            if ok:
                score = (deg_v, len(adj[v]))
                feasible.append((score, v))

        if not feasible:
            break

        feasible.sort(reverse=True)
        if randomize and len(feasible) > 1:
            top_k = min(3, len(feasible))
            _, chosen_v = random.choice(feasible[:top_k])
        else:
            _, chosen_v = feasible[0]

        # update
        old_group = list(group_list)
        group_list.append(chosen_v)
        group_set.add(chosen_v)
        deg_map[chosen_v] = 0
        for u in old_group:
            if chosen_v in adj[u]:
                deg_map[u] += 1
                deg_map[chosen_v] += 1

    return frozenset(group_set)


def generate_candidates(vertices, adj, s, connectivity_required, time_limit, start_time):
    n = len(vertices)
    degrees = {v: len(adj[v]) for v in vertices}
    vertices_by_deg = sorted(vertices, key=lambda x: degrees[x], reverse=True)

    max_candidates = min(8000, max(500, 40 * n))
    candidates = []
    cand_set = set()
    singleton_idx = {}

    def add_candidate(c):
        if not c:
            return
        if c in cand_set:
            return
        # Safety checks (cheap enough at insertion scale)
        if not is_splex(c, adj, s):
            return
        if connectivity_required and not is_connected_subset(c, adj):
            return
        cand_set.add(c)
        candidates.append(c)

    # Always add singletons
    for v in vertices:
        c = frozenset([v])
        add_candidate(c)

    # Save singleton indices after initial insertion
    for i, c in enumerate(candidates):
        if len(c) == 1:
            v = next(iter(c))
            singleton_idx[v] = i

    # Candidate generation budget
    gen_budget = max(0.0, min(time_limit * 0.4, max(1.0, time_limit - 1.0)))
    gen_deadline = start_time + gen_budget

    # Deterministic expansions from single seeds
    for seed in vertices_by_deg:
        if time.time() >= gen_deadline or len(candidates) >= max_candidates:
            break
        c = expand_group([seed], vertices_by_deg, adj, s, connectivity_required, randomize=False)
        add_candidate(c)

    # Expansions from edge seeds for diversity
    for u in vertices_by_deg:
        if time.time() >= gen_deadline or len(candidates) >= max_candidates:
            break
        nbrs = sorted(adj[u], key=lambda x: degrees[x], reverse=True)
        for v in nbrs[: min(3, len(nbrs))]:
            if time.time() >= gen_deadline or len(candidates) >= max_candidates:
                break
            init = [u, v]
            c = expand_group(init, vertices_by_deg, adj, s, connectivity_required, randomize=False)
            add_candidate(c)

    # Randomized expansions while time remains
    random.seed(0)
    while time.time() < gen_deadline and len(candidates) < max_candidates:
        seeds = vertices[:]
        random.shuffle(seeds)
        for seed in seeds[: min(len(seeds), 25)]:
            if time.time() >= gen_deadline or len(candidates) >= max_candidates:
                break
            rand_order = vertices[:]
            random.shuffle(rand_order)
            c = expand_group([seed], rand_order, adj, s, connectivity_required, randomize=True)
            add_candidate(c)

    # Ensure singleton map complete
    for i, c in enumerate(candidates):
        if len(c) == 1:
            v = next(iter(c))
            singleton_idx[v] = i

    return candidates, singleton_idx


def greedy_initial_solution(vertices, candidates, singleton_idx, decomposition_type):
    csets = [set(c) for c in candidates]
    selected = []

    if decomposition_type == "partitioning":
        used = set()
        idxs = list(range(len(candidates)))
        idxs.sort(key=lambda i: (len(csets[i]),), reverse=True)

        for i in idxs:
            if len(csets[i]) <= 1:
                continue
            if csets[i].isdisjoint(used):
                selected.append(i)
                used.update(csets[i])

        for v in vertices:
            if v not in used:
                selected.append(singleton_idx[v])
                used.add(v)

    else:  # covering
        uncovered = set(vertices)
        available = set(range(len(candidates)))
        while uncovered:
            best_i = None
            best_cov = -1
            best_size = -1
            for i in available:
                cov = len(csets[i] & uncovered)
                if cov > best_cov or (cov == best_cov and len(csets[i]) > best_size):
                    best_cov = cov
                    best_size = len(csets[i])
                    best_i = i
            if best_i is None or best_cov <= 0:
                # Fallback to singletons
                v = next(iter(uncovered))
                selected.append(singleton_idx[v])
                uncovered.remove(v)
            else:
                selected.append(best_i)
                uncovered -= csets[best_i]
                available.remove(best_i)

    return selected


def solve_mip(vertices, candidates, decomposition_type, time_limit, start_time, init_selected, logger=None):
    remaining = max(0.0, time_limit - (time.time() - start_time))
    if remaining <= 0.05:
        return None, None

    m = len(candidates)
    csets = [set(c) for c in candidates]
    inc = defaultdict(list)
    for i, c in enumerate(csets):
        for v in c:
            inc[v].append(i)

    model = gp.Model("splex_decomposition")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = remaining
    model.Params.Threads = 1

    x = model.addVars(m, vtype=GRB.BINARY, name="x")
    model.setObjective(gp.quicksum(x[i] for i in range(m)), GRB.MINIMIZE)

    for v in vertices:
        if decomposition_type == "partitioning":
            model.addConstr(gp.quicksum(x[i] for i in inc[v]) == 1, name=f"cover_{v}")
        else:
            model.addConstr(gp.quicksum(x[i] for i in inc[v]) >= 1, name=f"cover_{v}")

    init_set = set(init_selected)
    for i in range(m):
        x[i].Start = 1.0 if i in init_set else 0.0

    model._logger = logger
    model._best_logged = float(len(init_selected)) if init_selected is not None else float("inf")

    def cb(mdl, where):
        if where == GRB.Callback.MIPSOL and mdl._logger is not None:
            obj = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < mdl._best_logged:
                mdl._best_logged = obj
                mdl._logger.log(float(obj))

    model.optimize(cb)

    if model.SolCount > 0:
        selected = [i for i in range(m) if x[i].X > 0.5]
        obj = float(len(selected))
        return selected, obj

    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    graph_data = data["graph"]
    settings = data["problem_settings"]

    vertices, adj = build_graph(graph_data)
    s = int(settings["parameter_s"])
    decomposition_type = settings["decomposition_type"].strip().lower()
    connectivity_required = bool(settings["connectivity_required"])

    candidates, singleton_idx = generate_candidates(
        vertices=vertices,
        adj=adj,
        s=s,
        connectivity_required=connectivity_required,
        time_limit=args.time_limit,
        start_time=start_time,
    )

    init_selected = greedy_initial_solution(vertices, candidates, singleton_idx, decomposition_type)
    best_selected = init_selected
    best_obj = float(len(init_selected))

    if logger:
        logger.log(best_obj)

    mip_selected, mip_obj = solve_mip(
        vertices=vertices,
        candidates=candidates,
        decomposition_type=decomposition_type,
        time_limit=args.time_limit,
        start_time=start_time,
        init_selected=init_selected,
        logger=logger,
    )

    if mip_selected is not None and mip_obj is not None and mip_obj < best_obj + 1e-9:
        best_selected = mip_selected
        best_obj = mip_obj
        # callback likely logged; still safe to ensure final best is logged
        if logger and best_obj < getattr(solve_mip, "_last_logged", float("inf")):
            pass

    solution_groups = [sorted(list(candidates[i])) for i in best_selected]

    out = {
        "objective_value": float(len(solution_groups)),
        "solution": solution_groups,
    }

    with open(args.solution_path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()