import argparse
import json
import time
import random
from typing import List, Tuple, Dict, Optional

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    return parser.parse_args()


def detect_and_convert_precedence(precedence_relations: List[List[int]], n: int) -> List[Tuple[int, int]]:
    if not precedence_relations:
        return []
    mx = max(max(p) for p in precedence_relations)
    mn = min(min(p) for p in precedence_relations)

    one_based = (mx == n and mn >= 1) or (mn >= 1 and mx > n - 1)
    rel = []
    if one_based:
        for i, j in precedence_relations:
            rel.append((i - 1, j - 1))
    else:
        for i, j in precedence_relations:
            rel.append((i, j))
    return rel


def build_graph(n: int, rel: List[Tuple[int, int]]):
    succ = [set() for _ in range(n)]
    pred = [set() for _ in range(n)]
    indeg = [0] * n
    for u, v in rel:
        if v not in succ[u]:
            succ[u].add(v)
            pred[v].add(u)
            indeg[v] += 1
    return succ, pred, indeg


def topo_sort(n: int, succ: List[set], indeg0: List[int], rng: random.Random, randomized: bool) -> List[int]:
    indeg = indeg0[:]
    available = [i for i in range(n) if indeg[i] == 0]
    order = []

    if not randomized:
        available.sort()

    while available:
        if randomized:
            idx = rng.randrange(len(available))
            u = available[idx]
            available[idx] = available[-1]
            available.pop()
        else:
            u = available.pop(0)

        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                available.append(v)
                if not randomized:
                    available.sort()

    if len(order) != n:
        # Fallback if cycle (unexpected): natural order
        return list(range(n))
    return order


def station_cost(seq: List[int], p: List[int], fwd: List[List[int]], bwd: List[List[int]]) -> float:
    if not seq:
        return float("inf")
    total = 0.0
    for t in seq:
        total += p[t]
    for i in range(len(seq) - 1):
        total += fwd[seq[i]][seq[i + 1]]
    total += bwd[seq[-1]][seq[0]]
    return total


def optimize_station_sequence(
    tasks: List[int],
    p: List[int],
    fwd: List[List[int]],
    bwd: List[List[int]],
    succ: List[set],
    pred: List[set],
    deadline: float,
) -> Tuple[List[int], float]:
    # Start with given order (already topological slice), then local improvements.
    seq = tasks[:]
    if len(seq) <= 1:
        return seq, station_cost(seq, p, fwd, bwd)

    cur_cost = station_cost(seq, p, fwd, bwd)
    pos = {t: i for i, t in enumerate(seq)}
    in_station = set(seq)

    # Adjacent swap hill-climb
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            # Cannot swap if precedence requires a before b
            if b in succ[a]:
                continue
            cand = seq[:]
            cand[i], cand[i + 1] = cand[i + 1], cand[i]
            c = station_cost(cand, p, fwd, bwd)
            if c + 1e-9 < cur_cost:
                seq = cand
                cur_cost = c
                improved = True
                break

    # One-pass insertion improvements (limited effort)
    if time.time() < deadline:
        made_move = True
        while made_move and time.time() < deadline:
            made_move = False
            pos = {t: i for i, t in enumerate(seq)}
            for i, t in enumerate(seq):
                if time.time() >= deadline:
                    break

                earliest = 0
                for pr in pred[t]:
                    if pr in in_station:
                        earliest = max(earliest, pos[pr] + 1)
                latest = len(seq) - 1
                for sc in succ[t]:
                    if sc in in_station:
                        latest = min(latest, pos[sc] - 1)

                best_j = i
                best_cost = cur_cost
                for j in range(earliest, latest + 1):
                    if j == i:
                        continue
                    cand = seq[:]
                    x = cand.pop(i)
                    cand.insert(j, x)
                    c = station_cost(cand, p, fwd, bwd)
                    if c + 1e-9 < best_cost:
                        best_cost = c
                        best_j = j

                if best_j != i:
                    x = seq.pop(i)
                    seq.insert(best_j, x)
                    cur_cost = best_cost
                    made_move = True
                    break

    return seq, cur_cost


def partition_dp_fixed_order(
    order: List[int], m: int, p: List[int], fwd: List[List[int]], bwd: List[List[int]]
) -> Optional[List[int]]:
    # Returns station end indices (inclusive), length m
    n = len(order)
    if n < m:
        return None

    pref_p = [0.0] * (n + 1)
    for i in range(n):
        pref_p[i + 1] = pref_p[i] + p[order[i]]

    pref_f = [0.0] * n  # pref_f[k] = sum forward on edges 0..k-1
    for k in range(1, n):
        pref_f[k] = pref_f[k - 1] + fwd[order[k - 1]][order[k]]

    def seg_cost(i: int, j: int) -> float:
        # i..j inclusive
        proc = pref_p[j + 1] - pref_p[i]
        fw = pref_f[j] - pref_f[i] if j > i else 0.0
        bw = bwd[order[j]][order[i]]
        return proc + fw + bw

    INF = 1e30
    dp_prev = [INF] * n
    parent = [[-1] * n for _ in range(m)]

    for j in range(n):
        dp_prev[j] = seg_cost(0, j)
        parent[0][j] = -1

    for k in range(1, m):
        dp_cur = [INF] * n
        for j in range(k, n):
            best = INF
            best_i = -1
            i_min = k - 1
            i_max = j - 1
            for i in range(i_min, i_max + 1):
                val = max(dp_prev[i], seg_cost(i + 1, j))
                if val < best:
                    best = val
                    best_i = i
            dp_cur[j] = best
            parent[k][j] = best_i
        dp_prev = dp_cur

    ends = [0] * m
    j = n - 1
    for k in range(m - 1, -1, -1):
        ends[k] = j
        j = parent[k][j]
    return ends


def get_station_range(ends: List[int], s: int) -> Tuple[int, int]:
    start = 0 if s == 0 else ends[s - 1] + 1
    end = ends[s]
    return start, end


def evaluate_solution(
    order: List[int],
    ends: List[int],
    p: List[int],
    fwd: List[List[int]],
    bwd: List[List[int]],
    succ: List[set],
    pred: List[set],
    deadline: float,
):
    m = len(ends)
    seqs = []
    costs = []
    for s in range(m):
        a, b = get_station_range(ends, s)
        tasks = order[a : b + 1]
        seq, c = optimize_station_sequence(tasks, p, fwd, bwd, succ, pred, deadline)
        seqs.append(seq)
        costs.append(c)
    obj = max(costs) if costs else float("inf")
    return seqs, costs, obj


def improve_boundaries(
    order: List[int],
    ends: List[int],
    seqs: List[List[int]],
    costs: List[float],
    p: List[int],
    fwd: List[List[int]],
    bwd: List[List[int]],
    succ: List[set],
    pred: List[set],
    deadline: float,
):
    m = len(ends)
    cur_obj = max(costs)

    while time.time() < deadline:
        max_cost = max(costs)
        bottlenecks = [i for i, c in enumerate(costs) if abs(c - max_cost) <= 1e-9]
        best_move = None
        best_new_obj = cur_obj

        for s in bottlenecks:
            if time.time() >= deadline:
                break
            a, b = get_station_range(ends, s)
            size_s = b - a + 1

            # Move first task of s to s-1 (boundary between s-1 and s moves right)
            if s > 0 and size_s >= 2:
                new_ends = ends[:]
                new_ends[s - 1] += 1

                # Recompute affected stations s-1 and s
                a1, b1 = get_station_range(new_ends, s - 1)
                a2, b2 = get_station_range(new_ends, s)
                t1 = order[a1 : b1 + 1]
                t2 = order[a2 : b2 + 1]

                seq1, c1 = optimize_station_sequence(t1, p, fwd, bwd, succ, pred, deadline)
                seq2, c2 = optimize_station_sequence(t2, p, fwd, bwd, succ, pred, deadline)

                new_costs = costs[:]
                new_costs[s - 1] = c1
                new_costs[s] = c2
                new_obj = max(new_costs)
                if new_obj + 1e-9 < best_new_obj:
                    best_new_obj = new_obj
                    best_move = ("left", s, new_ends, seq1, c1, seq2, c2)

            # Move last task of s to s+1 (boundary between s and s+1 moves left)
            if s < m - 1 and size_s >= 2 and time.time() < deadline:
                new_ends = ends[:]
                new_ends[s] -= 1

                a1, b1 = get_station_range(new_ends, s)
                a2, b2 = get_station_range(new_ends, s + 1)
                t1 = order[a1 : b1 + 1]
                t2 = order[a2 : b2 + 1]

                seq1, c1 = optimize_station_sequence(t1, p, fwd, bwd, succ, pred, deadline)
                seq2, c2 = optimize_station_sequence(t2, p, fwd, bwd, succ, pred, deadline)

                new_costs = costs[:]
                new_costs[s] = c1
                new_costs[s + 1] = c2
                new_obj = max(new_costs)
                if new_obj + 1e-9 < best_new_obj:
                    best_new_obj = new_obj
                    best_move = ("right", s, new_ends, seq1, c1, seq2, c2)

        if best_move is None:
            break

        direction, s, new_ends, seq_a, c_a, seq_b, c_b = best_move
        ends[:] = new_ends
        if direction == "left":
            seqs[s - 1] = seq_a
            costs[s - 1] = c_a
            seqs[s] = seq_b
            costs[s] = c_b
        else:
            seqs[s] = seq_a
            costs[s] = c_a
            seqs[s + 1] = seq_b
            costs[s + 1] = c_b

        cur_obj = max(costs)

    return ends, seqs, costs, max(costs)


def build_assignment_from_sequences(seqs: List[List[int]], n: int) -> Dict[int, int]:
    assign = {}
    for s, lst in enumerate(seqs, start=1):
        for t in lst:
            assign[t] = s
    # If any task missing (shouldn't happen), assign to station 1
    for t in range(n):
        if t not in assign:
            assign[t] = 1
    return assign


def main():
    args = parse_args()
    start_time = time.time()
    deadline = start_time + max(1, args.time_limit) - 0.02

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n = int(inst["num_tasks"])
    m = int(inst["num_stations"])
    p = list(inst["processing_times"])
    fwd = inst["forward_setup_times"]
    bwd = inst["backward_setup_times"]
    precedence_raw = inst.get("precedence_relations", [])
    precedence = detect_and_convert_precedence(precedence_raw, n)

    succ, pred, indeg = build_graph(n, precedence)

    rng = random.Random(0)

    best_obj = float("inf")
    best_seqs = None

    # Fallback deterministic order
    deterministic_order = topo_sort(n, succ, indeg, rng, randomized=False)

    if n >= m:
        initial_ends = partition_dp_fixed_order(deterministic_order, m, p, fwd, bwd)
        if initial_ends is None:
            initial_ends = [i for i in range(m - 1)] + [n - 1]
    else:
        # Infeasible to keep all stations non-empty; fallback best-effort
        initial_ends = [i for i in range(max(0, n - 1))]

    # Multi-start search
    it = 0
    while time.time() < deadline:
        randomized = (it > 0)
        order = topo_sort(n, succ, indeg, rng, randomized=randomized)

        if n >= m:
            ends = partition_dp_fixed_order(order, m, p, fwd, bwd)
            if ends is None:
                it += 1
                continue
        else:
            # Best effort infeasible case
            ends = [i for i in range(n - 1)] + [n - 1] * (m - n + 1)

        seqs, costs, obj = evaluate_solution(order, ends[:], p, fwd, bwd, succ, pred, deadline)
        ends, seqs, costs, obj = improve_boundaries(
            order, ends[:], seqs, costs, p, fwd, bwd, succ, pred, deadline
        )

        if obj + 1e-9 < best_obj:
            best_obj = obj
            best_seqs = [lst[:] for lst in seqs]
            if logger:
                logger.log(float(best_obj))

        it += 1
        # If very little time left, stop
        if deadline - time.time() < 0.02:
            break

    # Final fallback if nothing found
    if best_seqs is None:
        if n >= m:
            ends = partition_dp_fixed_order(deterministic_order, m, p, fwd, bwd)
            if ends is None:
                ends = [i for i in range(m - 1)] + [n - 1]
            seqs, costs, best_obj = evaluate_solution(deterministic_order, ends, p, fwd, bwd, succ, pred, deadline)
            best_seqs = seqs
        else:
            best_seqs = [[] for _ in range(m)]
            for t in range(n):
                best_seqs[t].append(t)
            best_obj = 0.0
            for s in range(m):
                if best_seqs[s]:
                    best_obj = max(best_obj, station_cost(best_seqs[s], p, fwd, bwd))

    assignment0 = build_assignment_from_sequences(best_seqs, n)

    assignment_out = {str(t + 1): int(assignment0[t]) for t in range(n)}
    station_sequences_out = {str(s + 1): [t + 1 for t in best_seqs[s]] for s in range(m)}

    sol = {
        "objective_value": float(best_obj),
        "assignment": assignment_out,
        "station_sequences": station_sequences_out,
        "cycle_time": float(best_obj),
    }

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()