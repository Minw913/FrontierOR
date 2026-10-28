import argparse
import json
import time
from itertools import combinations

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def bits_iter(x: int):
    while x:
        b = x & -x
        yield b.bit_length() - 1
        x ^= b


def choose_pivot(P: int, X: int, adj_masks):
    UX = P | X
    if UX == 0:
        return -1
    best_u = -1
    best_cnt = -1
    tmp = UX
    while tmp:
        b = tmp & -tmp
        u = b.bit_length() - 1
        cnt = (P & adj_masks[u]).bit_count()
        if cnt > best_cnt:
            best_cnt = cnt
            best_u = u
        tmp ^= b
    return best_u


def find_clique_of_size_r(adj_masks, r):
    n = len(adj_masks)
    if r <= 0:
        return 0
    if r == 1:
        return 1 if n > 0 else 0
    all_mask = (1 << n) - 1
    found_mask = 0

    def dfs(R_mask, R_size, P, X):
        nonlocal found_mask
        if found_mask:
            return True
        if R_size >= r:
            found_mask = R_mask
            return True
        if R_size + P.bit_count() < r:
            return False
        if P == 0:
            return False

        u = choose_pivot(P, X, adj_masks)
        cand = P if u == -1 else (P & ~adj_masks[u])

        while cand:
            b = cand & -cand
            cand ^= b
            v = b.bit_length() - 1

            if dfs(R_mask | b, R_size + 1, P & adj_masks[v], X & adj_masks[v]):
                return True

            P ^= b
            X |= b
            if R_size + P.bit_count() < r:
                return False
        return False

    dfs(0, 0, all_mask, 0)
    return found_mask


def max_clique_size_exact_or_timeout(adj_masks, deadline):
    n = len(adj_masks)
    if n == 0:
        return 0, True
    all_mask = (1 << n) - 1
    best = 0
    timed_out = False
    calls = 0

    def bk(R_size, P, X):
        nonlocal best, timed_out, calls
        if timed_out:
            return
        calls += 1
        if (calls & 1023) == 0 and time.time() > deadline:
            timed_out = True
            return

        if P == 0 and X == 0:
            if R_size > best:
                best = R_size
            return

        if R_size + P.bit_count() <= best:
            return

        u = choose_pivot(P, X, adj_masks)
        cand = P if u == -1 else (P & ~adj_masks[u])

        while cand:
            b = cand & -cand
            cand ^= b
            v = b.bit_length() - 1

            bk(R_size + 1, P & adj_masks[v], X & adj_masks[v])
            if timed_out:
                return

            P ^= b
            X |= b

            if R_size + P.bit_count() <= best:
                return

    bk(0, all_mask, 0)
    return best, (not timed_out)


class CliqueLazyCallback:
    def __init__(self, y_vars, edge_list_idx, n, r, pair_to_eid, deadline):
        self.y_vars = y_vars
        self.edge_list_idx = edge_list_idx
        self.n = n
        self.r = r
        self.pair_to_eid = pair_to_eid
        self.deadline = deadline

    def __call__(self, model, where):
        if where != GRB.Callback.MIPSOL:
            return
        if time.time() > self.deadline:
            model.terminate()
            return

        vals = model.cbGetSolution(self.y_vars)
        rem_adj = [0] * self.n
        for eid, (u, v) in enumerate(self.edge_list_idx):
            if vals[eid] < 0.5:
                rem_adj[u] |= (1 << v)
                rem_adj[v] |= (1 << u)

        clique_mask = find_clique_of_size_r(rem_adj, self.r)
        if clique_mask:
            verts = list(bits_iter(clique_mask))
            if len(verts) > self.r:
                verts = verts[:self.r]
            edge_ids = []
            for i in range(len(verts)):
                ui = verts[i]
                for j in range(i + 1, len(verts)):
                    vj = verts[j]
                    a, b = (ui, vj) if ui < vj else (vj, ui)
                    eid = self.pair_to_eid.get((a, b))
                    if eid is not None:
                        edge_ids.append(eid)
            if edge_ids:
                model.cbLazy(gp.quicksum(self.y_vars[e] for e in edge_ids) >= 1)


def build_remaining_adj(n, edge_list_idx, removed_set):
    rem_adj = [0] * n
    for eid, (u, v) in enumerate(edge_list_idx):
        if eid not in removed_set:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)
    return rem_adj


def solve_feasibility_for_t(
    t,
    n,
    m,
    k,
    edge_list_idx,
    pair_to_eid,
    start_removed,
    deadline,
):
    # Need no (t+1)-clique in remaining graph
    r = t + 1
    if n == 0:
        return True, set()
    if r > n:
        return True, set()
    if r <= 1:
        # Impossible unless n == 0, handled above
        return False, None
    if r == 2:
        # No edges allowed
        if m <= k:
            return True, set(range(m))
        return False, None

    if time.time() >= deadline:
        return None, None

    model = gp.Model("clique_interdiction_feas")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.LazyConstraints = 1
    model.Params.TimeLimit = max(0.0, deadline - time.time())

    y = model.addVars(m, vtype=GRB.BINARY, name="y")
    model.addConstr(gp.quicksum(y[e] for e in range(m)) <= k, name="budget")
    model.setObjective(0.0, GRB.MINIMIZE)

    if start_removed is not None:
        for e in range(m):
            y[e].Start = 1.0 if e in start_removed else 0.0

    cb = CliqueLazyCallback(
        y_vars=[y[e] for e in range(m)],
        edge_list_idx=edge_list_idx,
        n=n,
        r=r,
        pair_to_eid=pair_to_eid,
        deadline=deadline,
    )

    model.optimize(cb)

    status = model.Status
    if model.SolCount > 0:
        removed = set()
        for e in range(m):
            if y[e].X > 0.5:
                removed.add(e)
        return True, removed

    if status == GRB.INFEASIBLE:
        return False, None

    # TIME_LIMIT / INTERRUPTED / other unknown
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    args = parser.parse_args()

    start_time = time.time()
    deadline = start_time + max(1, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    vertices = inst["vertices"]
    n = inst["num_vertices"]
    k = inst["interdiction_budget_k"]

    # Map labels to internal indices
    label_to_idx = {v: i for i, v in enumerate(vertices)}

    # Build unique undirected edge set
    pair_to_eid = {}
    edge_list_idx = []
    for e in inst["edges"]:
        a_lbl, b_lbl = e
        u = label_to_idx[a_lbl]
        v = label_to_idx[b_lbl]
        if u == v:
            continue
        if u > v:
            u, v = v, u
        if (u, v) not in pair_to_eid:
            pair_to_eid[(u, v)] = len(edge_list_idx)
            edge_list_idx.append((u, v))

    m = len(edge_list_idx)
    k = min(k, m)

    # Trivial cases
    if n == 0:
        out = {"objective_value": 0, "interdicted_edges": []}
        with open(args.solution_path, "w") as f:
            json.dump(out, f)
        return

    # Best known feasible solution
    best_t = n  # proven upper bound on max clique after deletion
    best_removed = set()
    if logger:
        logger.log(best_t)

    # Binary search on t using feasibility oracle
    lo = 0      # known infeasible (t=0)
    hi = n      # known feasible (trivial)
    current_start = set()

    while lo + 1 < hi and time.time() < deadline:
        mid = (lo + hi) // 2
        feas, removed = solve_feasibility_for_t(
            t=mid,
            n=n,
            m=m,
            k=k,
            edge_list_idx=edge_list_idx,
            pair_to_eid=pair_to_eid,
            start_removed=current_start,
            deadline=deadline,
        )

        if feas is True:
            hi = mid
            current_start = removed if removed is not None else set()
            if mid < best_t:
                best_t = mid
                best_removed = current_start.copy()
                if logger:
                    logger.log(best_t)
        elif feas is False:
            lo = mid
        else:
            # Unknown due time/interruption
            break

    # Optional local descent if time remains
    t_try = best_t - 1
    while t_try >= 1 and time.time() < deadline:
        feas, removed = solve_feasibility_for_t(
            t=t_try,
            n=n,
            m=m,
            k=k,
            edge_list_idx=edge_list_idx,
            pair_to_eid=pair_to_eid,
            start_removed=best_removed,
            deadline=deadline,
        )
        if feas is True:
            best_t = t_try
            best_removed = removed if removed is not None else set()
            if logger:
                logger.log(best_t)
            t_try -= 1
        elif feas is False:
            break
        else:
            break

    # Try to compute exact clique number of chosen solution with remaining time.
    rem_adj = build_remaining_adj(n, edge_list_idx, best_removed)
    exact_size, complete = max_clique_size_exact_or_timeout(rem_adj, deadline)

    if complete:
        objective_value = int(exact_size)
    else:
        # Safe fallback: proven upper bound from feasibility model
        objective_value = int(best_t)

    interdicted_edges = []
    for eid in sorted(best_removed):
        u, v = edge_list_idx[eid]
        interdicted_edges.append([vertices[u], vertices[v]])

    sol = {
        "objective_value": objective_value,
        "interdicted_edges": interdicted_edges,
    }

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()