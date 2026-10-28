#!/usr/bin/env python3
"""
Parallel-machine weighted tardiness scheduler.

Approach:
  1. Construct greedy initial solutions (several orderings), then improve by
     local search (relocations / swaps).
  2. If the instance is small enough, solve a MIP (assignment + pairwise
     precedence binaries + continuous start times) with Gurobi, warm-started
     by the heuristic. Incumbents are logged via a MIPSOL callback.
  3. The best schedule found is written out in the required schema.
"""

import argparse
import json
import random
import time

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAVE_GUROBI = True
except Exception:  # pragma: no cover
    gp = None
    GRB = None
    HAVE_GUROBI = False

from solution_logger import SolutionLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_solution_dict(seqs, p, w, d, ids):
    """Turn machine sequences (lists of job indices) into the output dict."""
    entries = []
    total = 0
    for k, seq in enumerate(seqs):
        t = 0
        prev_id = 0
        for j in seq:
            st = t
            ct = st + p[j]
            tard = max(0, ct - d[j])
            wt = w[j] * tard
            entries.append({
                "machine_id": k,
                "prev_job": prev_id,
                "job": ids[j],
                "start_time": int(st),
                "completion_time": int(ct),
                "weighted_tardiness": wt,
            })
            prev_id = ids[j]
            t = ct
            total += wt
    entries.sort(key=lambda e: e["job"])
    return {"objective_value": float(total), "schedule": entries}, total


def seq_obj(seq, p, w, d):
    """Weighted tardiness contribution of a single machine sequence."""
    t = 0
    obj = 0
    for j in seq:
        t += p[j]
        tard = t - d[j]
        if tard > 0:
            obj += w[j] * tard
    return obj


def total_obj(seqs, p, w, d):
    return sum(seq_obj(seq, p, w, d) for seq in seqs)


# ---------------------------------------------------------------------------
# Heuristic: greedy construction + local search
# ---------------------------------------------------------------------------

def greedy_construct(order, p, w, d, m):
    seqs = [[] for _ in range(m)]
    loads = [0] * m
    for j in order:
        best_k = 0
        best_cost = None
        for k in range(m):
            ct = loads[k] + p[j]
            tard = max(0, ct - d[j])
            cost = w[j] * tard
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_k = k
        seqs[best_k].append(j)
        loads[best_k] += p[j]
    return seqs


def local_search(seqs, p, w, d, m, n, deadline, rng):
    if n <= 1:
        return seqs
    job_mach = [None] * n
    for k, seq in enumerate(seqs):
        for j in seq:
            job_mach[j] = k
    best_obj = total_obj(seqs, p, w, d)
    stagnant = 0
    while time.monotonic() < deadline:
        stagnant += 1
        if stagnant > 20000:
            break
        move = rng.random()
        if move < 0.5 and n >= 2:
            # swap two random jobs
            i, j = rng.sample(range(n), 2)
            a, b = job_mach[i], job_mach[j]
            if a == b:
                seq = seqs[a]
                pi, pj = seq.index(i), seq.index(j)
                old = seq_obj(seq, p, w, d)
                seq[pi], seq[pj] = seq[pj], seq[pi]
                delta = seq_obj(seq, p, w, d) - old
                if delta < 0:
                    best_obj += delta
                    stagnant = 0
                else:
                    seq[pi], seq[pj] = seq[pj], seq[pi]
            else:
                sa, sb = seqs[a], seqs[b]
                pi, pj = sa.index(i), sb.index(j)
                old = seq_obj(sa, p, w, d) + seq_obj(sb, p, w, d)
                sa[pi], sb[pj] = j, i
                delta = (seq_obj(sa, p, w, d) + seq_obj(sb, p, w, d)) - old
                if delta < 0:
                    best_obj += delta
                    job_mach[i], job_mach[j] = b, a
                    stagnant = 0
                else:
                    sa[pi], sb[pj] = i, j
        else:
            # relocate a random job to a random machine / position
            j = rng.randrange(n)
            src = job_mach[j]
            tgt = rng.randrange(m)
            src_seq = seqs[src]
            tgt_seq = seqs[tgt]
            if src == tgt:
                old = seq_obj(src_seq, p, w, d)
                idx = src_seq.index(j)
                src_seq.pop(idx)
                pos = rng.randrange(len(src_seq) + 1)
                src_seq.insert(pos, j)
                delta = seq_obj(src_seq, p, w, d) - old
                if delta < 0:
                    best_obj += delta
                    stagnant = 0
                else:
                    src_seq.pop(pos)
                    src_seq.insert(idx, j)
            else:
                old = seq_obj(src_seq, p, w, d) + seq_obj(tgt_seq, p, w, d)
                idx = src_seq.index(j)
                src_seq.pop(idx)
                pos = rng.randrange(len(tgt_seq) + 1)
                tgt_seq.insert(pos, j)
                delta = (seq_obj(src_seq, p, w, d)
                         + seq_obj(tgt_seq, p, w, d)) - old
                if delta < 0:
                    best_obj += delta
                    job_mach[j] = tgt
                    stagnant = 0
                else:
                    tgt_seq.pop(pos)
                    src_seq.insert(idx, j)
    return seqs


def heuristic(p, w, d, n, m, deadline):
    rng = random.Random(0)
    orders = []
    orders.append(sorted(range(n), key=lambda j: (d[j], -w[j])))
    orders.append(sorted(range(n), key=lambda j: (-w[j] / max(p[j], 1), d[j])))
    orders.append(sorted(range(n), key=lambda j: (-w[j], d[j])))
    best_seqs = None
    best = None
    for order in orders:
        seqs = greedy_construct(order, p, w, d, m)
        obj = total_obj(seqs, p, w, d)
        if best is None or obj < best:
            best = obj
            best_seqs = [seq[:] for seq in seqs]
    # local search improvement
    ls_deadline = deadline
    best_seqs = local_search(best_seqs, p, w, d, m, n, ls_deadline, rng)
    return best_seqs, total_obj(best_seqs, p, w, d)


# ---------------------------------------------------------------------------
# MIP solver (Gurobi)
# ---------------------------------------------------------------------------

def solve_mip(p, w, d, n, m, H, ids, init_seqs, time_remaining, logger,
              best_tracker):
    model = gp.Model("WeightedTardiness")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", max(time_remaining, 1))

    M = float(H) + 1.0

    x = model.addVars(n, m, vtype=GRB.BINARY, name="x")
    s = model.addVars(n, lb=0.0, ub=[H - p[j] for j in range(n)], name="s")
    T = model.addVars(n, lb=0.0, name="T")

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    y = model.addVars(pairs, vtype=GRB.BINARY, name="y")

    for j in range(n):
        model.addConstr(gp.quicksum(x[j, k] for k in range(m)) == 1)
        model.addConstr(T[j] >= s[j] + p[j] - d[j])

    for (i, j) in pairs:
        yv = y[(i, j)]
        for k in range(m):
            model.addConstr(
                s[j] >= s[i] + p[i] - M * (1 - yv)
                - M * (2 - x[i, k] - x[j, k]))
            model.addConstr(
                s[i] >= s[j] + p[j] - M * yv
                - M * (2 - x[i, k] - x[j, k]))

    model.setObjective(gp.quicksum(w[j] * T[j] for j in range(n)),
                       GRB.MINIMIZE)

    # ---- warm start -------------------------------------------------------
    job_mach = [None] * n
    job_start = [0.0] * n
    for k, seq in enumerate(init_seqs):
        t = 0.0
        for j in seq:
            job_mach[j] = k
            job_start[j] = t
            t += p[j]
    for j in range(n):
        for k in range(m):
            x[j, k].Start = 1.0 if job_mach[j] == k else 0.0
        s[j].Start = job_start[j]
        T[j].Start = w[j] * max(0.0, job_start[j] + p[j] - d[j])
    for (i, j) in pairs:
        val = 0.0
        if job_mach[i] == job_mach[j] and job_mach[i] is not None:
            val = 1.0 if job_start[i] <= job_start[j] else 0.0
        y[(i, j)].Start = val

    x_flat = [x[j, k] for j in range(n) for k in range(m)]
    s_flat = [s[j] for j in range(n)]

    def reconstruct(xv, sv):
        machine_of = [None] * n
        for j in range(n):
            best_k = 0
            best_v = -1.0
            for k in range(m):
                idx = j * m + k
                if xv[idx] > best_v:
                    best_v = xv[idx]
                    best_k = k
            machine_of[j] = best_k
        seqs = [[] for _ in range(m)]
        for j in range(n):
            seqs[machine_of[j]].append(j)
        for k in range(m):
            seqs[k].sort(key=lambda j: sv[j])
        return seqs

    def callback(model_, where):
        if where == GRB.Callback.MIPSOL:
            try:
                xv = model_.cbGetSolution(x_flat)
                sv = model_.cbGetSolution(s_flat)
                seqs = reconstruct(xv, sv)
                sol_dict, obj = build_solution_dict(seqs, p, w, d, ids)
                if obj < best_tracker["obj"] - 1e-9:
                    best_tracker["obj"] = obj
                    if logger:
                        logger.log_solution(float(obj), sol_dict)
            except Exception:
                try:
                    obj = model_.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj < best_tracker["obj"]:
                        best_tracker["obj"] = obj
                        if logger:
                            logger.log(float(obj))
                except Exception:
                    pass

    model.optimize(callback)

    if model.SolCount > 0:
        xv = [x[j, k].X for j in range(n) for k in range(m)]
        sv = [s[j].X for j in range(n)]
        seqs = reconstruct(xv, sv)
        sol_dict, obj = build_solution_dict(seqs, p, w, d, ids)
        return sol_dict, obj
    return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    t0 = time.monotonic()
    deadline = t0 + args.time_limit

    logger = (SolutionLogger(args.log_path, sense="minimize")
              if args.log_path else None)

    with open(args.instance_path) as f:
        data = json.load(f)

    params = data.get("parameters", {})
    m = params.get("m", 1)
    jobs = data["jobs"]
    n = len(jobs)
    ids = [j["id"] for j in jobs]
    p = [j["processing_time"] for j in jobs]
    w = [j["weight"] for j in jobs]
    d = [j["due_date"] for j in jobs]

    H = data.get("time_horizon")
    if H is None:
        total_p = sum(p)
        p_max = max(p)
        H = (total_p - p_max) // max(m, 1) + p_max

    # ---- heuristic phase --------------------------------------------------
    pair_count = n * (n - 1) // 2
    use_mip = HAVE_GUROBI and (2 * pair_count * max(m, 1) <= 500000)

    if use_mip:
        heur_budget = min(max(args.time_limit * 0.05, 0.2), 2.0)
    else:
        heur_budget = max(args.time_limit - 0.5, 0.1)
    heur_deadline = time.monotonic() + heur_budget

    best_seqs, best_obj = heuristic(p, w, d, n, m, heur_deadline)
    best_dict, _ = build_solution_dict(best_seqs, p, w, d, ids)

    best_tracker = {"obj": best_obj}
    if logger:
        logger.log_solution(float(best_obj), best_dict)

    # ---- MIP phase --------------------------------------------------------
    remaining = deadline - time.monotonic() - 0.2
    if use_mip and remaining > 0.5:
        sol_dict, obj = solve_mip(p, w, d, n, m, H, ids, best_seqs,
                                  remaining, logger, best_tracker)
        if sol_dict is not None and obj < best_obj - 1e-9:
            best_obj = obj
            best_dict = sol_dict
            if logger:
                logger.log_solution(float(best_obj), best_dict)

    # ---- write final ------------------------------------------------------
    with open(args.solution_path, "w") as f:
        json.dump(best_dict, f)


if __name__ == "__main__":
    main()