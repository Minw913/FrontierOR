import argparse
import json
import math
import random
import time
from typing import List, Tuple, Dict, Set

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Bin Packing with optional incompatibility constraints")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input instance JSON")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Optional path for incumbent log JSONL")
    return parser.parse_args()


def read_instance(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    n = int(data["n"])
    c = int(data["c"])
    weights = list(map(int, data["weights"]))
    if n != len(weights):
        n = len(weights)

    # Optional incompatibility keys (robust parsing)
    raw_pairs = None
    for k in ["incompatible_pairs", "incompatibilities", "conflicts", "edges", "forbidden_pairs"]:
        if k in data:
            raw_pairs = data[k]
            break

    pairs = []
    if raw_pairs is not None:
        for p in raw_pairs:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                i, j = int(p[0]), int(p[1])
                if 0 <= i < n and 0 <= j < n and i != j:
                    if i > j:
                        i, j = j, i
                    pairs.append((i, j))

    # unique
    pairs = sorted(list(set(pairs)))

    return n, c, weights, pairs


def build_adjacency(n: int, pairs: List[Tuple[int, int]]) -> List[Set[int]]:
    adj = [set() for _ in range(n)]
    for i, j in pairs:
        adj[i].add(j)
        adj[j].add(i)
    return adj


def can_place(item: int, bin_items: List[int], load: int, weights: List[int], c: int, adj: List[Set[int]]) -> bool:
    if load + weights[item] > c:
        return False
    ai = adj[item]
    for u in bin_items:
        if u in ai:
            return False
    return True


def construct_solution(order: List[int], weights: List[int], c: int, adj: List[Set[int]]) -> List[List[int]]:
    bins: List[List[int]] = []
    loads: List[int] = []

    for i in order:
        best_b = -1
        best_rem = None
        wi = weights[i]
        for b in range(len(bins)):
            if loads[b] + wi <= c:
                conflict = False
                ai = adj[i]
                for u in bins[b]:
                    if u in ai:
                        conflict = True
                        break
                if not conflict:
                    rem = c - (loads[b] + wi)
                    if best_rem is None or rem < best_rem:
                        best_rem = rem
                        best_b = b

        if best_b == -1:
            bins.append([i])
            loads.append(wi)
        else:
            bins[best_b].append(i)
            loads[best_b] += wi

    return bins


def try_close_bins(bins: List[List[int]], weights: List[int], c: int, adj: List[Set[int]], time_deadline: float) -> List[List[int]]:
    # Simple improvement: try removing light bins by reassigning their items.
    bins = [list(b) for b in bins]
    improved = True
    while improved and time.time() < time_deadline:
        improved = False
        loads = [sum(weights[i] for i in b) for b in bins]
        order_bins = sorted(range(len(bins)), key=lambda x: loads[x])  # try emptier bins first
        for bidx in order_bins:
            if bidx >= len(bins):
                continue
            if time.time() >= time_deadline:
                break

            items_to_move = list(bins[bidx])
            if not items_to_move:
                continue

            # Prepare target bins without bidx
            target_bins = [bins[k][:] for k in range(len(bins)) if k != bidx]
            target_loads = [sum(weights[i] for i in b) for b in target_bins]

            # harder items first
            items_to_move.sort(key=lambda i: (weights[i], len(adj[i])), reverse=True)

            feasible = True
            for it in items_to_move:
                placed = False
                best_t = -1
                best_rem = None
                for t in range(len(target_bins)):
                    if target_loads[t] + weights[it] > c:
                        continue
                    ai = adj[it]
                    conf = False
                    for u in target_bins[t]:
                        if u in ai:
                            conf = True
                            break
                    if conf:
                        continue
                    rem = c - (target_loads[t] + weights[it])
                    if best_rem is None or rem < best_rem:
                        best_rem = rem
                        best_t = t
                if best_t == -1:
                    feasible = False
                    break
                target_bins[best_t].append(it)
                target_loads[best_t] += weights[it]
                placed = True
                if not placed:
                    feasible = False
                    break

            if feasible:
                bins = [b for b in target_bins if b]
                improved = True
                break

    return bins


def evaluate_and_log(candidate_bins, best_bins, logger):
    if best_bins is None or len(candidate_bins) < len(best_bins):
        if logger:
            logger.log(float(len(candidate_bins)))
        return candidate_bins
    return best_bins


def heuristic_search(n, c, weights, adj, time_limit, start_time, logger):
    degrees = [len(adj[i]) for i in range(n)]
    items = list(range(n))

    # Deterministic good order
    base_order = sorted(items, key=lambda i: (weights[i], degrees[i]), reverse=True)
    best_bins = construct_solution(base_order, weights, c, adj)
    best_bins = try_close_bins(best_bins, weights, c, adj, min(start_time + 0.2 * time_limit, start_time + 2.0))
    if logger:
        logger.log(float(len(best_bins)))

    # Multi-start randomized construction (time-bounded)
    rnd = random.Random(0)
    heur_deadline = start_time + max(0.5, min(time_limit * 0.45, time_limit - 0.2))
    if heur_deadline <= time.time():
        return best_bins

    while time.time() < heur_deadline:
        # randomized score around weight/conflict
        shuffled = items[:]
        rnd.shuffle(shuffled)
        shuffled.sort(
            key=lambda i: (
                3.0 * weights[i] / max(1, c) + 1.5 * degrees[i] / max(1, n - 1) + 0.15 * rnd.random()
            ),
            reverse=True,
        )
        cand = construct_solution(shuffled, weights, c, adj)

        # occasional local improvement
        if rnd.random() < 0.7:
            cand = try_close_bins(cand, weights, c, adj, min(time.time() + 0.15, heur_deadline))

        if len(cand) < len(best_bins):
            best_bins = cand
            if logger:
                logger.log(float(len(best_bins)))

    return best_bins


def solve_mip(
    n: int,
    c: int,
    weights: List[int],
    pairs: List[Tuple[int, int]],
    initial_bins: List[List[int]],
    time_remaining: float,
    logger,
):
    UB = len(initial_bins)
    if UB <= 1 or time_remaining <= 0.05:
        return initial_bins

    # Size guard
    est_conf_cons = len(pairs) * UB
    est_vars = n * UB + UB
    if est_conf_cons > 3_000_000 or est_vars > 2_000_000:
        return initial_bins

    model = gp.Model("bin_packing_conflicts")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.05, time_remaining)
    model.Params.MIPFocus = 1

    x = model.addVars(n, UB, vtype=GRB.BINARY, name="x")
    y = model.addVars(UB, vtype=GRB.BINARY, name="y")

    model.setObjective(gp.quicksum(y[b] for b in range(UB)), GRB.MINIMIZE)

    # each item assigned exactly once
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, b] for b in range(UB)) == 1, name=f"assign_{i}")

    # capacity
    for b in range(UB):
        model.addConstr(gp.quicksum(weights[i] * x[i, b] for i in range(n)) <= c * y[b], name=f"cap_{b}")

    # link
    for i in range(n):
        for b in range(UB):
            model.addConstr(x[i, b] <= y[b], name=f"link_{i}_{b}")

    # incompatibilities
    if pairs:
        for b in range(UB):
            for (i, j) in pairs:
                model.addConstr(x[i, b] + x[j, b] <= 1, name=f"conf_{i}_{j}_{b}")

    # symmetry breaking
    for b in range(UB - 1):
        model.addConstr(y[b] >= y[b + 1], name=f"sym_{b}")

    # warm start
    for b in range(UB):
        y[b].Start = 1.0 if b < len(initial_bins) else 0.0
    for b, items in enumerate(initial_bins):
        for i in items:
            x[i, b].Start = 1.0

    best_logged = [float(len(initial_bins))]

    def cb(m, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < best_logged[0]:
                best_logged[0] = obj
                logger.log(float(obj))

    model.optimize(cb)

    if model.SolCount == 0:
        return initial_bins

    bins = []
    for b in range(UB):
        if y[b].X > 0.5:
            items_b = [i for i in range(n) if x[i, b].X > 0.5]
            if items_b:
                bins.append(items_b)

    # Fallback safety: if something odd, keep initial
    assigned = sum(len(b) for b in bins)
    if assigned != n:
        return initial_bins

    return bins


def main():
    args = parse_args()
    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    n, c, weights, pairs = read_instance(args.instance_path)

    # Basic infeasibility check (single item too heavy)
    if any(w > c for w in weights):
        sol = {"objective_value": float(1e30), "bins": []}
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    adj = build_adjacency(n, pairs)

    # Heuristic phase
    best_bins = heuristic_search(n, c, weights, adj, args.time_limit, start_time, logger)

    # MIP polishing with remaining time
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed
    if remaining > 0.1:
        best_bins = solve_mip(n, c, weights, pairs, best_bins, remaining, logger)

    # Normalize output bins
    out_bins = [sorted(b) for b in best_bins if b]
    objective_value = float(len(out_bins))

    solution = {
        "objective_value": objective_value,
        "bins": out_bins,
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()