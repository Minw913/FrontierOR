import argparse
import json
import math
import random
import time

import numpy as np

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", type=str, required=True)
    ap.add_argument("--solution_path", type=str, required=True)
    ap.add_argument("--time_limit", type=int, default=60)
    ap.add_argument("--log_path", type=str, default=None)
    return ap.parse_args()


def load_processing_times(inst, n, m):
    """Robustly extract an (n x m) processing time matrix.

    The raw data may be exactly n x m, may be a larger matrix (take the
    top-left n x m submatrix), may be flat of length n*m, or may be
    transposed (m x n)."""
    raw = inst["processing_times"]
    # Flat list case
    if raw and not isinstance(raw[0], (list, tuple)):
        arr = np.array(raw, dtype=np.float64)
        if arr.size == n * m:
            return arr.reshape(n, m)
        if arr.size >= n * m:
            return arr[: n * m].reshape(n, m)
        raise ValueError("processing_times too small")
    # Nested list case (possibly ragged / oversized)
    nrows = len(raw)
    ncols = min(len(r) for r in raw) if nrows > 0 else 0
    if nrows >= n and ncols >= m:
        return np.array([list(r[:m]) for r in raw[:n]], dtype=np.float64)
    if nrows >= m and ncols >= n:
        # transposed
        arr = np.array([list(r[:n]) for r in raw[:m]], dtype=np.float64)
        return arr.T
    # Last resort: flatten and reshape
    flat = np.array([x for r in raw for x in r], dtype=np.float64)
    if flat.size >= n * m:
        return flat[: n * m].reshape(n, m)
    raise ValueError("Cannot interpret processing_times shape")


def compute_completions(seq, P, flags):
    """Return (m x L) matrix of completion times by machine and position."""
    seq = np.asarray(seq, dtype=np.int64)
    L = len(seq)
    m = P.shape[1]
    comps = np.empty((m, L), dtype=np.float64)
    A = np.zeros(L, dtype=np.float64)
    for mach in range(m):
        p = P[seq, mach]
        cp = np.cumsum(p)
        B = A - (cp - p)  # A[k] - prefix[k-1]
        if flags[mach]:
            A = B.max() + cp
        else:
            A = np.maximum.accumulate(B) + cp
        comps[mach] = A
    return comps


def makespan(seq, P, flags):
    return compute_completions(seq, P, flags)[-1, -1]


def best_insertion(partial, job, P, flags):
    """Evaluate inserting `job` into every position of `partial`.
    Returns (best_position, best_makespan)."""
    L = len(partial)
    if L == 0:
        return 0, makespan([job], P, flags)
    s = np.asarray(partial, dtype=np.int64)
    K = L + 1
    ii = np.arange(K)[:, None]
    kk = np.arange(K)[None, :]
    sa = s[np.clip(kk, 0, L - 1)]
    sb = s[np.clip(kk - 1, 0, L - 1)]
    J = np.where(kk < ii, sa, np.where(kk == ii, job, sb))  # (K, K) job matrix
    m = P.shape[1]
    A = np.zeros((K, K), dtype=np.float64)
    for mach in range(m):
        p = P[:, mach][J]
        cp = np.cumsum(p, axis=1)
        B = A - (cp - p)
        if flags[mach]:
            A = B.max(axis=1, keepdims=True) + cp
        else:
            A = np.maximum.accumulate(B, axis=1) + cp
    mks = A[:, -1]
    pos = int(np.argmin(mks))
    return pos, float(mks[pos])


def neh(P, flags, deadline):
    n = P.shape[0]
    order = sorted(range(n), key=lambda j: -float(P[j].sum()))
    seq = [order[0]]
    for j in order[1:]:
        if time.time() >= deadline:
            seq.append(j)
            continue
        pos, _ = best_insertion(seq, j, P, flags)
        seq = seq[:pos] + [j] + seq[pos:]
    return seq, makespan(seq, P, flags)


def local_search(seq, val, P, flags, deadline, rng):
    """Insertion neighborhood local search."""
    seq = list(seq)
    improved = True
    while improved:
        if time.time() >= deadline:
            break
        improved = False
        jobs = list(seq)
        rng.shuffle(jobs)
        for job in jobs:
            if time.time() >= deadline:
                return seq, val
            pos = seq.index(job)
            partial = seq[:pos] + seq[pos + 1:]
            bpos, bval = best_insertion(partial, job, P, flags)
            if bval < val - 1e-9:
                improved = True
            seq = partial[:bpos] + [job] + partial[bpos:]
            val = bval
    return seq, val


def build_solution(seq, P, flags):
    comps = compute_completions(seq, P, flags)
    m = P.shape[1]
    return {
        "objective_value": float(comps[-1, -1]),
        "sequence": [int(x) for x in seq],
        "completion_times": {
            f"machine_{i}": [float(c) for c in comps[i]] for i in range(m)
        },
    }


def main():
    args = parse_args()
    start = time.time()
    deadline = start + max(1, args.time_limit) - 0.5

    logger = SolutionLogger(args.log_path, sense="minimize") if (args.log_path and SolutionLogger) else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n = int(inst["n_jobs"])
    m = int(inst["n_machines"])
    P = load_processing_times(inst, n, m)

    if "no_idle_flags" in inst and inst["no_idle_flags"] is not None:
        raw_flags = [bool(b) for b in inst["no_idle_flags"]]
        flags = (raw_flags + [False] * m)[:m]
    else:
        ni = set(int(x) for x in inst.get("no_idle_machines", []))
        flags = [(i + 1) in ni for i in range(m)]

    rng = random.Random(0)
    np.random.seed(0)

    if n == 1:
        best_seq = [0]
        best_val = makespan(best_seq, P, flags)
        sol = build_solution(best_seq, P, flags)
        if logger:
            logger.log_solution(best_val, sol)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    # ---- NEH construction ----
    best_seq, best_val = neh(P, flags, deadline)
    sol = build_solution(best_seq, P, flags)
    if logger:
        logger.log_solution(best_val, sol)

    # ---- Initial local search ----
    if time.time() < deadline:
        best_seq, best_val = local_search(best_seq, best_val, P, flags, deadline, rng)
        sol = build_solution(best_seq, P, flags)
        if logger:
            logger.log_solution(best_val, sol)

    # ---- Iterated Greedy ----
    total_p = float(P.sum())
    T = 0.4 * total_p / (n * m * 10.0)
    if T <= 0:
        T = 1.0
    d = min(4, n - 1)

    cur_seq = list(best_seq)
    cur_val = best_val

    while time.time() < deadline:
        # Destruction
        new_seq = list(cur_seq)
        removed = []
        for _ in range(d):
            idx = rng.randrange(len(new_seq))
            removed.append(new_seq.pop(idx))
        # Reconstruction (greedy best insertion)
        timed_out = False
        for job in removed:
            if time.time() >= deadline:
                new_seq.append(job)
                timed_out = True
                continue
            pos, _ = best_insertion(new_seq, job, P, flags)
            new_seq = new_seq[:pos] + [job] + new_seq[pos:]
        new_val = makespan(new_seq, P, flags)

        # Local search on reconstructed solution
        if not timed_out and time.time() < deadline:
            new_seq, new_val = local_search(new_seq, new_val, P, flags, deadline, rng)

        # Acceptance
        if new_val < cur_val - 1e-9:
            cur_seq, cur_val = new_seq, new_val
            if cur_val < best_val - 1e-9:
                best_seq, best_val = list(cur_seq), cur_val
                sol = build_solution(best_seq, P, flags)
                if logger:
                    logger.log_solution(best_val, sol)
        else:
            diff = new_val - cur_val
            try:
                accept = rng.random() < math.exp(-diff / T)
            except OverflowError:
                accept = False
            if accept:
                cur_seq, cur_val = new_seq, new_val

    # ---- Final output ----
    sol = build_solution(best_seq, P, flags)
    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()