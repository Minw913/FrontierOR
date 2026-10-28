import argparse
import json
import math
import time
from typing import List, Dict, Set, Tuple

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def load_instance(path: str):
    with open(path, "r") as f:
        data = json.load(f)

    num_items = int(data["num_items"])
    capacity = int(data["bin_capacity"])
    items_data = data["items"]
    conflict_edges = data.get("conflict_edges", [])

    item_ids = [int(it["id"]) for it in items_data]
    weights = [int(it["weight"]) for it in items_data]

    # Build id <-> index maps (robust even if IDs are not 0..n-1)
    id_to_idx = {item_ids[i]: i for i in range(len(item_ids))}
    idx_to_id = {i: item_ids[i] for i in range(len(item_ids))}

    n = len(item_ids)
    if n != num_items:
        # Trust actual list length if mismatch
        num_items = n

    # Build adjacency sets
    adj: List[Set[int]] = [set() for _ in range(num_items)]
    clean_edges = []
    for e in conflict_edges:
        if not isinstance(e, list) or len(e) != 2:
            continue
        a_id, b_id = int(e[0]), int(e[1])
        if a_id not in id_to_idx or b_id not in id_to_idx:
            continue
        u, v = id_to_idx[a_id], id_to_idx[b_id]
        if u == v:
            continue
        if v not in adj[u]:
            adj[u].add(v)
            adj[v].add(u)
            clean_edges.append((u, v))

    return {
        "num_items": num_items,
        "capacity": capacity,
        "weights": weights,
        "item_ids": item_ids,
        "id_to_idx": id_to_idx,
        "idx_to_id": idx_to_id,
        "adj": adj,
        "edges": clean_edges,
    }


def first_fit_best_conflict(weights: List[int], capacity: int, adj: List[Set[int]]) -> List[List[int]]:
    n = len(weights)
    order = list(range(n))
    order.sort(key=lambda i: (-weights[i], -len(adj[i]), i))

    bins_items: List[List[int]] = []
    bins_sets: List[Set[int]] = []
    bins_rem: List[int] = []

    for i in order:
        w = weights[i]
        best_b = -1
        best_leftover = None

        for b in range(len(bins_items)):
            if bins_rem[b] < w:
                continue
            # conflict check: none of i's neighbors is in bin b
            conflict = False
            for nb in adj[i]:
                if nb in bins_sets[b]:
                    conflict = True
                    break
            if conflict:
                continue

            leftover = bins_rem[b] - w
            if best_leftover is None or leftover < best_leftover:
                best_leftover = leftover
                best_b = b

        if best_b == -1:
            bins_items.append([i])
            bins_sets.append({i})
            bins_rem.append(capacity - w)
        else:
            bins_items[best_b].append(i)
            bins_sets[best_b].add(i)
            bins_rem[best_b] -= w

    return bins_items


def try_reduce_bins(
    bins_items: List[List[int]],
    weights: List[int],
    capacity: int,
    adj: List[Set[int]],
    max_passes: int = 3
) -> List[List[int]]:
    # Prepare mutable structures
    bins_items = [list(b) for b in bins_items if b]
    bins_sets = [set(b) for b in bins_items]
    bins_load = [sum(weights[i] for i in b) for b in bins_items]
    bins_rem = [capacity - ld for ld in bins_load]

    if len(bins_items) <= 1:
        return bins_items

    for _ in range(max_passes):
        improved = False
        # Try to eliminate lightly loaded bins first
        idxs = list(range(len(bins_items)))
        idxs.sort(key=lambda b: (len(bins_items[b]), bins_load[b]))

        for src in idxs:
            if src >= len(bins_items) or not bins_items[src]:
                continue

            src_items = sorted(bins_items[src], key=lambda i: (-weights[i], -len(adj[i]), i))
            tentative_moves: List[Tuple[int, int]] = []  # (item, dest_bin)
            possible = True

            for i in src_items:
                w = weights[i]
                best_dest = -1
                best_left = None

                for b in range(len(bins_items)):
                    if b == src:
                        continue
                    if bins_rem[b] < w:
                        continue
                    conflict = False
                    for nb in adj[i]:
                        if nb in bins_sets[b]:
                            conflict = True
                            break
                    if conflict:
                        continue
                    left = bins_rem[b] - w
                    if best_left is None or left < best_left:
                        best_left = left
                        best_dest = b

                if best_dest == -1:
                    possible = False
                    break

                # apply tentatively
                bins_items[best_dest].append(i)
                bins_sets[best_dest].add(i)
                bins_load[best_dest] += w
                bins_rem[best_dest] -= w
                tentative_moves.append((i, best_dest))

            if possible:
                # clear src bin
                for i in src_items:
                    bins_sets[src].discard(i)
                bins_items[src] = []
                bins_load[src] = 0
                bins_rem[src] = capacity
                improved = True
                break
            else:
                # rollback tentative
                for i, b in tentative_moves:
                    bins_items[b].pop()
                    bins_sets[b].discard(i)
                    bins_load[b] -= weights[i]
                    bins_rem[b] += weights[i]

        if not improved:
            break

        # remove empties after each successful elimination
        new_bins = [b for b in bins_items if b]
        bins_items = new_bins
        bins_sets = [set(b) for b in bins_items]
        bins_load = [sum(weights[i] for i in b) for b in bins_items]
        bins_rem = [capacity - ld for ld in bins_load]

    return bins_items


def bins_to_assignment(bins_items: List[List[int]], n: int) -> List[int]:
    assign = [-1] * n
    for b, items in enumerate(bins_items):
        for i in items:
            assign[i] = b
    return assign


def build_output_from_assignment(
    assignment: List[int],
    weights: List[int],
    idx_to_id: Dict[int, int]
):
    bin_map: Dict[int, List[int]] = {}
    for i, b in enumerate(assignment):
        if b < 0:
            continue
        bin_map.setdefault(b, []).append(i)

    # Renumber bins consecutively in output
    used_bins = sorted(bin_map.keys())
    out_bins = {}
    new_id = 0
    for b in used_bins:
        items_idx = bin_map[b]
        total_w = int(sum(weights[i] for i in items_idx))
        out_bins[str(new_id)] = {
            "items": [int(idx_to_id[i]) for i in items_idx],
            "total_weight": total_w
        }
        new_id += 1

    return {
        "objective_value": len(used_bins),
        "bins": out_bins
    }


def solve_with_mip(
    n: int,
    capacity: int,
    weights: List[int],
    edges: List[Tuple[int, int]],
    adj: List[Set[int]],
    heuristic_bins: List[List[int]],
    time_limit: int,
    logger=None
):
    start = time.time()

    UB = len(heuristic_bins)
    if UB <= 0:
        return None

    # Size guard: skip MIP if model would be too large
    est_x = n * UB
    est_conf = len(edges) * UB
    if est_x > 350000 or est_conf > 1200000:
        return None

    model = gp.Model("bin_packing_conflicts")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1, time_limit)
    model.Params.MIPFocus = 1

    B = range(UB)
    I = range(n)

    x = model.addVars(n, UB, vtype=GRB.BINARY, name="x")
    y = model.addVars(UB, vtype=GRB.BINARY, name="y")

    # Each item exactly once
    for i in I:
        model.addConstr(gp.quicksum(x[i, b] for b in B) == 1)

    # Capacity
    for b in B:
        model.addConstr(gp.quicksum(weights[i] * x[i, b] for i in I) <= capacity * y[b])

    # Linking (important especially for zero-weight items)
    for i in I:
        for b in B:
            model.addConstr(x[i, b] <= y[b])

    # Conflict constraints
    for (u, v) in edges:
        for b in B:
            model.addConstr(x[u, b] + x[v, b] <= 1)

    # Symmetry breaking
    for b in range(UB - 1):
        model.addConstr(y[b] >= y[b + 1])

    model.setObjective(gp.quicksum(y[b] for b in B), GRB.MINIMIZE)

    # MIP start from heuristic
    h_assign = bins_to_assignment(heuristic_bins, n)
    for i in I:
        hb = h_assign[i]
        if hb >= 0:
            x[i, hb].Start = 1.0
    for b in B:
        y[b].Start = 1.0 if b < UB else 0.0

    best_logged = [UB]

    def cb(m, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            val = int(round(obj))
            if val < best_logged[0]:
                best_logged[0] = val
                logger.log(val)

    try:
        model.optimize(cb)
    except gp.GurobiError:
        return None

    if model.SolCount == 0:
        return None

    assignment = [-1] * n
    for i in I:
        for b in B:
            if x[i, b].X > 0.5:
                assignment[i] = b
                break

    # Safety fallback (shouldn't happen)
    if any(a < 0 for a in assignment):
        return None

    elapsed = time.time() - start
    _ = elapsed  # reserved if needed later
    return assignment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    t0 = time.time()

    inst = load_instance(args.instance_path)
    n = inst["num_items"]
    capacity = inst["capacity"]
    weights = inst["weights"]
    adj = inst["adj"]
    edges = inst["edges"]
    idx_to_id = inst["idx_to_id"]

    # Heuristic construction + light improvement
    h_bins = first_fit_best_conflict(weights, capacity, adj)
    h_bins = try_reduce_bins(h_bins, weights, capacity, adj, max_passes=3)
    h_assign = bins_to_assignment(h_bins, n)
    h_obj = len(h_bins)

    if logger:
        logger.log(h_obj)

    # Remaining time for MIP
    spent = time.time() - t0
    rem = args.time_limit - int(spent)
    final_assign = h_assign

    if rem > 1:
        mip_assign = solve_with_mip(
            n=n,
            capacity=capacity,
            weights=weights,
            edges=edges,
            adj=adj,
            heuristic_bins=h_bins,
            time_limit=rem,
            logger=logger
        )
        if mip_assign is not None:
            # Compare objectives
            mip_obj = len(set(mip_assign))
            if mip_obj <= h_obj:
                final_assign = mip_assign

    solution = build_output_from_assignment(final_assign, weights, idx_to_id)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()