import argparse
import json
import time
import random
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    with open(args.instance_path) as f:
        data = json.load(f)

    graph = data['graph']
    settings = data['problem_settings']
    vertices = list(graph['vertices'])
    num_vertices = graph['num_vertices']
    edges = graph['edges']
    s = settings['parameter_s']
    decomposition_type = settings['decomposition_type']
    connectivity_required = settings.get('connectivity_required', False)

    # Build adjacency
    adj = {v: set() for v in vertices}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()
    deadline = start_time + args.time_limit

    # ----- Helper functions -----
    def valid_group(S):
        Sset = set(S)
        n = len(Sset)
        if n == 0:
            return False
        for v in Sset:
            deg = len(adj[v] & Sset)
            if n - 1 - deg > s - 1:
                return False
        if connectivity_required and n > 1:
            start = next(iter(Sset))
            visited = set()
            stack = [start]
            while stack:
                x = stack.pop()
                if x not in visited:
                    visited.add(x)
                    for y in adj[x]:
                        if y in Sset and y not in visited:
                            stack.append(y)
            if len(visited) != n:
                return False
        return True

    def grow(seed, order=None, max_size=None):
        S = set([seed])
        nondeg = {seed: 0}
        if order is None:
            cand = vertices[:]
            random.shuffle(cand)
        else:
            cand = order[:]
        for v in cand:
            if v in S:
                continue
            v_non = 0
            for u in S:
                if u not in adj[v]:
                    v_non += 1
            if v_non > s - 1:
                continue
            ok = True
            for u in S:
                if v not in adj[u]:
                    if nondeg[u] + 1 > s - 1:
                        ok = False
                        break
            if not ok:
                continue
            if connectivity_required:
                if not any(u in adj[v] for u in S):
                    continue
            S.add(v)
            nondeg[v] = v_non
            for u in S:
                if u != v and v not in adj[u]:
                    nondeg[u] += 1
            if max_size and len(S) >= max_size:
                break
        return list(S)

    def greedy_partition(order):
        groups = []  # each: [list_of_vertices, nondeg_dict]
        for v in order:
            placed = False
            groups.sort(key=lambda x: len(x[0]), reverse=True)
            for g_list, g_nondeg in groups:
                v_non = 0
                for u in g_list:
                    if u not in adj[v]:
                        v_non += 1
                if v_non > s - 1:
                    continue
                ok = True
                for u in g_list:
                    if v not in adj[u]:
                        if g_nondeg[u] + 1 > s - 1:
                            ok = False
                            break
                if not ok:
                    continue
                if connectivity_required:
                    if not any(u in adj[v] for u in g_list):
                        continue
                g_list.append(v)
                g_nondeg[v] = v_non
                for u in g_list:
                    if u != v and v not in adj[u]:
                        g_nondeg[u] += 1
                placed = True
                break
            if not placed:
                groups.append(([v], {v: 0}))
        return [g[0] for g in groups]

    def merge_groups(groups):
        changed = True
        while changed:
            changed = False
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    if valid_group(set(groups[i]) | set(groups[j])):
                        groups[i].extend(groups[j])
                        groups.pop(j)
                        changed = True
                        break
                if changed:
                    break
        return groups

    def move_vertices(groups):
        changed = True
        while changed:
            changed = False
            for i, g in enumerate(groups):
                for v in g[:]:
                    for j, g2 in enumerate(groups):
                        if i == j:
                            continue
                        if valid_group(set(g2) | {v}):
                            remaining = [x for x in g if x != v]
                            if not remaining or valid_group(remaining):
                                g.remove(v)
                                g2.append(v)
                                if not g:
                                    groups.pop(i)
                                changed = True
                                break
                    if changed:
                        break
                if changed:
                    break
        return groups

    def improve_partition(groups):
        groups = merge_groups(groups)
        groups = move_vertices(groups)
        groups = merge_groups(groups)
        return groups

    def greedy_cover():
        uncovered = set(vertices)
        groups = []
        while uncovered:
            best_g = None
            best_cover = 0
            seeds = list(uncovered)
            random.shuffle(seeds)
            for seed in seeds[:min(20, len(seeds))]:
                g = grow(seed)
                cover = len(set(g) & uncovered)
                if cover > best_cover:
                    best_cover = cover
                    best_g = g
            if best_g is None:
                best_g = [next(iter(uncovered))]
            groups.append(best_g)
            uncovered -= set(best_g)
        # remove redundant
        changed = True
        while changed:
            changed = False
            for i in range(len(groups)):
                covered_by_others = set()
                for j in range(len(groups)):
                    if i != j:
                        covered_by_others.update(groups[j])
                if set(groups[i]).issubset(covered_by_others):
                    groups.pop(i)
                    changed = True
                    break
        return groups

    # ----- Heuristics -----
    best_obj = float('inf')
    best_solution = []

    if num_vertices > 0:
        if decomposition_type == 'partitioning':
            for _ in range(5):
                if time.time() > deadline - 2:
                    break
                order = vertices[:]
                random.shuffle(order)
                groups = greedy_partition(order)
                groups = improve_partition(groups)
                obj = len(groups)
                if obj < best_obj:
                    best_obj = obj
                    best_solution = groups
                    if logger:
                        logger.log_solution(best_obj, {"objective_value": float(best_obj), "solution": best_solution})
        else:
            for _ in range(5):
                if time.time() > deadline - 2:
                    break
                groups = greedy_cover()
                obj = len(groups)
                if obj < best_obj:
                    best_obj = obj
                    best_solution = groups
                    if logger:
                        logger.log_solution(best_obj, {"objective_value": float(best_obj), "solution": best_solution})

    if num_vertices == 0:
        out = {"objective_value": 0.0, "solution": []}
        with open(args.solution_path, 'w') as f:
            json.dump(out, f)
        return

    # ----- Candidate pool -----
    candidates = []
    seen = set()

    def add_candidate(c):
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            candidates.append(list(c))

    for v in vertices:
        add_candidate([v])

    if best_solution:
        for g in best_solution:
            add_candidate(g)

    max_candidates = 5000
    if num_vertices <= 50:
        max_candidates = 10000
    elif num_vertices <= 100:
        max_candidates = 5000
    else:
        max_candidates = 2000

    # Generate diverse candidates
    for seed in vertices:
        if len(candidates) >= max_candidates or time.time() > deadline - 3:
            break
        g = grow(seed, order=None)
        add_candidate(g)

    for _ in range(50):
        if len(candidates) >= max_candidates or time.time() > deadline - 3:
            break
        seed = random.choice(vertices)
        g = grow(seed, order=None)
        add_candidate(g)

    order_deg = sorted(vertices, key=lambda v: len(adj[v]), reverse=True)
    for seed in vertices:
        if len(candidates) >= max_candidates or time.time() > deadline - 3:
            break
        g = grow(seed, order=order_deg)
        add_candidate(g)

    # ----- ILP over candidate pool -----
    def solve_ilp(cand_list, warm_start=None):
        v_idx = {v: i for i, v in enumerate(vertices)}
        idx_to_v = {i: v for v, i in v_idx.items()}
        cand_sets = [set(v_idx[v] for v in c) for c in cand_list]
        n = len(vertices)
        m = len(cand_sets)

        env = gp.Env(empty=True)
        env.setParam('Seed', 0)
        env.setParam('Threads', 1)
        env.setParam('OutputFlag', 0)
        env.start()
        model = gp.Model(env=env)
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.TimeLimit = max(1.0, deadline - time.time() - 1.0)

        x = model.addVars(m, vtype=GRB.BINARY, name='x')

        for i in range(n):
            if decomposition_type == 'partitioning':
                model.addConstr(gp.quicksum(x[j] for j in range(m) if i in cand_sets[j]) == 1)
            else:
                model.addConstr(gp.quicksum(x[j] for j in range(m) if i in cand_sets[j]) >= 1)

        model.setObjective(gp.quicksum(x[j] for j in range(m)), GRB.MINIMIZE)

        if warm_start:
            ws_sets = [set(v_idx[v] for v in g) for g in warm_start]
            for j in range(m):
                if cand_sets[j] in ws_sets:
                    x[j].Start = 1.0
                else:
                    x[j].Start = 0.0

        model.optimize()

        if model.SolCount > 0:
            chosen = [j for j in range(m) if x[j].X > 0.5]
            solution = [[idx_to_v[i] for i in cand_sets[j]] for j in chosen]
            obj = model.ObjVal
            return obj, solution
        return None, None

    if time.time() < deadline and candidates:
        try:
            ilp_obj, ilp_solution = solve_ilp(candidates, warm_start=best_solution)
            if ilp_solution is not None and ilp_obj < best_obj:
                best_obj = ilp_obj
                best_solution = ilp_solution
                if logger:
                    logger.log_solution(best_obj, {"objective_value": float(best_obj), "solution": best_solution})
        except Exception:
            pass

    if not best_solution and num_vertices > 0:
        best_solution = [[v] for v in vertices]
        best_obj = num_vertices
        if logger:
            logger.log_solution(best_obj, {"objective_value": float(best_obj), "solution": best_solution})

    out = {
        "objective_value": float(best_obj),
        "solution": best_solution
    }
    with open(args.solution_path, 'w') as f:
        json.dump(out, f)


if __name__ == '__main__':
    main()