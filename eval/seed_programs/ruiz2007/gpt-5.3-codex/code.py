import argparse
import json
import math
import random
import time
from typing import List, Tuple, Dict, Optional

from solution_logger import SolutionLogger


def build_no_idle_flags(instance: dict) -> List[bool]:
    m = int(instance["n_machines"])
    flags = instance.get("no_idle_flags", None)
    if isinstance(flags, list) and len(flags) == m:
        return [bool(x) for x in flags]

    flags = [False] * m
    no_idle_machines = instance.get("no_idle_machines", [])
    for mach in no_idle_machines:
        idx = int(mach) - 1  # input is 1-indexed
        if 0 <= idx < m:
            flags[idx] = True
    return flags


def solve(instance: dict, time_limit: int, logger: Optional[SolutionLogger] = None) -> dict:
    start_time = time.time()
    n = int(instance["n_jobs"])
    m = int(instance["n_machines"])
    p = instance["processing_times"]
    no_idle = build_no_idle_flags(instance)

    random.seed(0)

    def time_exceeded() -> bool:
        return (time.time() - start_time) >= time_limit

    def eval_makespan(seq: List[int]) -> int:
        nn = len(seq)
        if nn == 0:
            return 0

        prev = None
        for i in range(m):
            if i == 0:
                # First machine (regular/no-idle both produce cumulative earliest schedule)
                curr = [0] * nn
                c = 0
                for k, job in enumerate(seq):
                    c += p[job][i]
                    curr[k] = c
            else:
                if no_idle[i]:
                    # Continuous processing with unknown start offset s >= 0
                    s = 0
                    run = 0  # sum of processing times before position k on machine i
                    for k, job in enumerate(seq):
                        req = prev[k] - run
                        if req > s:
                            s = req
                        run += p[job][i]
                    if s < 0:
                        s = 0

                    curr = [0] * nn
                    c = s
                    for k, job in enumerate(seq):
                        c += p[job][i]
                        curr[k] = c
                else:
                    # Regular flow-shop recurrence
                    curr = [0] * nn
                    c_prev_pos = 0
                    for k, job in enumerate(seq):
                        c = max(prev[k], c_prev_pos) + p[job][i]
                        curr[k] = c
                        c_prev_pos = c
            prev = curr
        return prev[-1]

    def eval_full(seq: List[int]) -> Tuple[int, List[List[int]]]:
        nn = len(seq)
        if nn == 0:
            return 0, [[] for _ in range(m)]

        all_c = []
        prev = None
        for i in range(m):
            if i == 0:
                curr = [0] * nn
                c = 0
                for k, job in enumerate(seq):
                    c += p[job][i]
                    curr[k] = c
            else:
                if no_idle[i]:
                    s = 0
                    run = 0
                    for k, job in enumerate(seq):
                        req = prev[k] - run
                        if req > s:
                            s = req
                        run += p[job][i]
                    if s < 0:
                        s = 0

                    curr = [0] * nn
                    c = s
                    for k, job in enumerate(seq):
                        c += p[job][i]
                        curr[k] = c
                else:
                    curr = [0] * nn
                    c_prev_pos = 0
                    for k, job in enumerate(seq):
                        c = max(prev[k], c_prev_pos) + p[job][i]
                        curr[k] = c
                        c_prev_pos = c

            all_c.append(curr)
            prev = curr
        return all_c[-1][-1], all_c

    best_seq = list(range(n))
    best_obj = eval_makespan(best_seq) if n > 0 else 0
    if logger:
        logger.log(float(best_obj))

    def try_update(seq: List[int], obj: int):
        nonlocal best_seq, best_obj
        if obj < best_obj:
            best_obj = obj
            best_seq = seq[:]
            if logger:
                logger.log(float(best_obj))

    def neh(order: List[int]) -> List[int]:
        seq = []
        for idx, job in enumerate(order):
            if time_exceeded():
                # Finish quickly with remaining jobs appended
                seq.extend(order[idx:])
                return seq

            best_local_seq = None
            best_local_obj = math.inf
            L = len(seq)

            for pos in range(L + 1):
                if time_exceeded():
                    break
                cand = seq[:pos] + [job] + seq[pos:]
                obj = eval_makespan(cand)
                if obj < best_local_obj:
                    best_local_obj = obj
                    best_local_seq = cand

            if best_local_seq is None:
                seq.append(job)
            else:
                seq = best_local_seq
        return seq

    # Initial constructions
    if n > 0 and not time_exceeded():
        totals = [sum(p[j][i] for i in range(m)) for j in range(n)]
        order1 = sorted(range(n), key=lambda j: -totals[j])
        seq1 = neh(order1)
        obj1 = eval_makespan(seq1)
        try_update(seq1, obj1)

    if n > 0 and not time_exceeded():
        # Alternative ordering emphasizing later machines
        tails = [sum(p[j][i] * (i + 1) for i in range(m)) for j in range(n)]
        order2 = sorted(range(n), key=lambda j: -tails[j])
        seq2 = neh(order2)
        obj2 = eval_makespan(seq2)
        try_update(seq2, obj2)

    current_seq = best_seq[:]
    current_obj = best_obj

    # Iterated greedy + light local search
    if n >= 2:
        T = max(1.0, 0.01 * best_obj)
        while not time_exceeded():
            d = min(max(2, n // 10), 8)
            d = min(d, n)

            # Destruction
            positions = sorted(random.sample(range(n), d), reverse=True)
            partial = current_seq[:]
            removed = []
            for pos in positions:
                removed.append(partial.pop(pos))
            random.shuffle(removed)

            # Reconstruction by best insertion
            for job in removed:
                if time_exceeded():
                    break
                best_pos = 0
                best_ins_obj = math.inf
                L = len(partial)
                for pos in range(L + 1):
                    cand = partial[:pos] + [job] + partial[pos:]
                    obj = eval_makespan(cand)
                    if obj < best_ins_obj:
                        best_ins_obj = obj
                        best_pos = pos
                partial.insert(best_pos, job)

            if time_exceeded():
                break

            candidate = partial
            cand_obj = eval_makespan(candidate)

            # Light adjacent-swap improvement (one pass)
            i = 0
            while i < n - 1 and not time_exceeded():
                neighbor = candidate[:]
                neighbor[i], neighbor[i + 1] = neighbor[i + 1], neighbor[i]
                obj_n = eval_makespan(neighbor)
                if obj_n < cand_obj:
                    candidate = neighbor
                    cand_obj = obj_n
                    if i > 0:
                        i -= 1
                    continue
                i += 1

            if cand_obj < best_obj:
                try_update(candidate, cand_obj)

            # Acceptance criterion
            delta = cand_obj - current_obj
            if delta <= 0:
                current_seq, current_obj = candidate, cand_obj
            else:
                prob = math.exp(-delta / max(T, 1e-9))
                if random.random() < prob:
                    current_seq, current_obj = candidate, cand_obj

            T = max(1e-6, T * 0.999)

    final_obj, final_c = eval_full(best_seq)
    completion_times = {f"machine_{i}": final_c[i] for i in range(m)}

    return {
        "objective_value": float(final_obj),
        "sequence": best_seq,
        "completion_times": completion_times,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    solution = solve(instance, args.time_limit, logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()