import argparse
import json
import math
import random
import time
from copy import deepcopy

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def machine_weighted_tardiness(seq, p, d, w):
    t = 0
    obj = 0.0
    for j in seq:
        t += p[j]
        tard = t - d[j]
        if tard > 0:
            obj += w[j] * tard
    return obj


def build_starts_and_objective(seqs, p, d, w):
    starts = {}
    total = 0.0
    for seq in seqs:
        t = 0
        for j in seq:
            starts[j] = t
            t += p[j]
            tard = t - d[j]
            if tard > 0:
                total += w[j] * tard
    return starts, total


def atc_construct(n, m, p, d, w, kappa=2.5, noise=0.0, rng=None):
    if rng is None:
        rng = random.Random(0)
    avg_p = sum(p) / max(1, n)

    unscheduled = set(range(n))
    seqs = [[] for _ in range(m)]
    avail = [0] * m

    while unscheduled:
        k = min(range(m), key=lambda x: (avail[x], len(seqs[x])))
        t = avail[k]

        best_j = None
        best_score = -1e100
        for j in unscheduled:
            slack_term = max(d[j] - p[j] - t, 0)
            score = (w[j] / p[j]) * math.exp(-slack_term / (kappa * avg_p + 1e-9))
            if noise > 0.0:
                score *= (1.0 + noise * (rng.random() - 0.5))
            if score > best_score:
                best_score = score
                best_j = j

        seqs[k].append(best_j)
        avail[k] += p[best_j]
        unscheduled.remove(best_j)

    starts, obj = build_starts_and_objective(seqs, p, d, w)
    return seqs, starts, obj


def improve_intra_machine_adjacent(seqs, p, d, w, deadline):
    # Simple local improvement: adjacent swaps on each machine
    for k in range(len(seqs)):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            seq = seqs[k]
            if len(seq) <= 1:
                break
            base = machine_weighted_tardiness(seq, p, d, w)
            for i in range(len(seq) - 1):
                if time.time() >= deadline:
                    break
                cand = seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                cobj = machine_weighted_tardiness(cand, p, d, w)
                if cobj + 1e-9 < base:
                    seqs[k] = cand
                    seq = cand
                    base = cobj
                    improved = True
                    break
    return seqs


def random_relocate_search(seqs, p, d, w, deadline, rng):
    m = len(seqs)
    machine_obj = [machine_weighted_tardiness(seqs[k], p, d, w) for k in range(m)]
    total_obj = sum(machine_obj)

    all_jobs = [j for s in seqs for j in s]
    if not all_jobs:
        return seqs, total_obj

    while time.time() < deadline:
        # pick a non-empty source machine
        nonempty = [k for k in range(m) if seqs[k]]
        if not nonempty:
            break
        a = rng.choice(nonempty)
        ia = rng.randrange(len(seqs[a]))
        job = seqs[a][ia]

        src_seq_removed = seqs[a][:ia] + seqs[a][ia + 1 :]
        src_new_obj = machine_weighted_tardiness(src_seq_removed, p, d, w)

        best_delta = 0.0
        best_move = None  # (b, pos, new_obj_b)

        for b in range(m):
            if time.time() >= deadline:
                break
            base_b_seq = src_seq_removed if b == a else seqs[b]
            old_b = src_new_obj if b == a else machine_obj[b]

            L = len(base_b_seq)
            if L <= 30:
                positions = range(L + 1)
            else:
                # sample positions for speed
                positions = {0, L}
                positions.add(rng.randrange(L + 1))
                positions.add(rng.randrange(L + 1))
                positions.add((L // 2))
                positions = sorted(positions)

            for pos in positions:
                cand_b = base_b_seq[:pos] + [job] + base_b_seq[pos:]
                new_b = machine_weighted_tardiness(cand_b, p, d, w)

                if b == a:
                    delta = new_b - machine_obj[a]
                else:
                    delta = (src_new_obj - machine_obj[a]) + (new_b - machine_obj[b])

                if delta < best_delta - 1e-9:
                    best_delta = delta
                    best_move = (b, pos, new_b)

        if best_move is not None:
            b, pos, new_b = best_move
            seqs[a] = src_seq_removed
            machine_obj[a] = src_new_obj
            if b == a:
                seqs[a] = seqs[a][:pos] + [job] + seqs[a][pos:]
                machine_obj[a] = new_b
            else:
                seqs[b] = seqs[b][:pos] + [job] + seqs[b][pos:]
                machine_obj[b] = new_b
            total_obj += best_delta
        else:
            # no improving move from this random pick
            continue

    return seqs, total_obj


def build_initial_solution(n, m, p, d, w, deadline, logger=None):
    rng = random.Random(42)
    best = None
    best_obj = float("inf")

    configs = [
        (1.5, 0.00),
        (2.0, 0.00),
        (2.5, 0.00),
        (3.0, 0.03),
        (4.0, 0.05),
    ]

    for kappa, noise in configs:
        if time.time() >= deadline:
            break
        seqs, _, obj = atc_construct(n, m, p, d, w, kappa=kappa, noise=noise, rng=rng)
        if obj < best_obj:
            best_obj = obj
            best = seqs
            if logger:
                logger.log(float(best_obj))

    if best is None:
        best = [[] for _ in range(m)]

    # Local search phase
    if time.time() < deadline:
        ls_deadline = min(deadline, time.time() + max(0.1, (deadline - time.time()) * 0.9))
        best = improve_intra_machine_adjacent(best, p, d, w, ls_deadline)
    if time.time() < deadline:
        ls_deadline = min(deadline, time.time() + max(0.1, (deadline - time.time()) * 0.9))
        best, obj = random_relocate_search(best, p, d, w, ls_deadline, rng)
        if obj < best_obj:
            best_obj = obj
            if logger:
                logger.log(float(best_obj))

    starts, best_obj = build_starts_and_objective(best, p, d, w)
    return best, starts, best_obj


def solve_mip_with_gurobi(n, m, p, d, w, T_horizon, init_seqs, time_limit, logger=None):
    model = gp.Model("parallel_weighted_tardiness")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.0, time_limit)

    M = max(T_horizon, sum(p))

    x = model.addVars(n, m, vtype=GRB.BINARY, name="x")
    C = model.addVars(n, lb=0.0, vtype=GRB.CONTINUOUS, name="C")
    tard = model.addVars(n, lb=0.0, vtype=GRB.CONTINUOUS, name="T")

    y = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(m):
                y[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}_{k}")

    model.update()

    # assignment
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, k] for k in range(m)) == 1, name=f"assign_{i}")

    # pair ordering consistency per machine
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(m):
                model.addConstr(y[i, j, k] <= x[i, k])
                model.addConstr(y[i, j, k] <= x[j, k])
                model.addConstr(y[j, i, k] <= x[i, k])
                model.addConstr(y[j, i, k] <= x[j, k])
                model.addConstr(y[i, j, k] + y[j, i, k] <= 1)
                model.addConstr(y[i, j, k] + y[j, i, k] >= x[i, k] + x[j, k] - 1)

    # completion and sequencing
    for i in range(n):
        model.addConstr(C[i] >= p[i], name=f"proc_lb_{i}")
        model.addConstr(tard[i] >= C[i] - d[i], name=f"tard_def_{i}")
        model.addConstr(C[i] <= M, name=f"C_ub_{i}")

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(m):
                # if j before i on k, enforce C_i >= C_j + p_i
                model.addConstr(C[i] >= C[j] + p[i] - M * (1 - y[j, i, k]))

    model.setObjective(gp.quicksum(w[i] * tard[i] for i in range(n)), GRB.MINIMIZE)

    # warm start from heuristic
    job_machine = {}
    for k in range(m):
        for j in init_seqs[k]:
            job_machine[j] = k

    for i in range(n):
        for k in range(m):
            x[i, k].Start = 1.0 if job_machine.get(i, -1) == k else 0.0

    for k in range(m):
        seq = init_seqs[k]
        pos = {job: idx for idx, job in enumerate(seq)}
        for i in seq:
            for j in seq:
                if i == j:
                    continue
                y[i, j, k].Start = 1.0 if pos[i] < pos[j] else 0.0

    starts_init, _ = build_starts_and_objective(init_seqs, p, d, w)
    for i in range(n):
        Ci = starts_init.get(i, 0) + p[i]
        C[i].Start = float(Ci)
        tard[i].Start = float(max(0, Ci - d[i]))

    # callback logging
    if logger is not None:
        model._logger = logger
        model._best_logged = float("inf")

        def cb(mdl, where):
            if where == GRB.Callback.MIPSOL:
                obj = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj + 1e-9 < mdl._best_logged:
                    mdl._best_logged = obj
                    mdl._logger.log(float(obj))

        model.optimize(cb)
    else:
        model.optimize()

    if model.SolCount == 0:
        return None

    # reconstruct compact no-idle schedule from incumbent
    seqs = [[] for _ in range(m)]
    for i in range(n):
        assigned_k = None
        for k in range(m):
            if x[i, k].X > 0.5:
                assigned_k = k
                break
        if assigned_k is None:
            assigned_k = 0
        seqs[assigned_k].append(i)

    # order jobs per machine by completion time from MIP
    for k in range(m):
        seqs[k].sort(key=lambda j: C[j].X)

    starts, obj = build_starts_and_objective(seqs, p, d, w)
    return seqs, starts, obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    t0 = time.time()
    deadline = t0 + max(1, args.time_limit)

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    n = int(data["n"])
    m = int(data["m"])
    T_horizon = int(data.get("T", 0))

    jobs = sorted(data["jobs"], key=lambda x: x["id"])
    # index mapping
    ids = [job["id"] for job in jobs]
    id_to_idx = {jid: i for i, jid in enumerate(ids)}

    p = [int(job["p"]) for job in jobs]
    d = [int(job["d"]) for job in jobs]
    w = [int(job["w"]) for job in jobs]

    # 1) heuristic phase
    heur_deadline = min(deadline, t0 + max(0.5, args.time_limit * 0.25))
    init_seqs, init_starts, best_obj = build_initial_solution(
        n, m, p, d, w, heur_deadline, logger=logger
    )
    best_seqs = init_seqs
    best_starts = init_starts

    # 2) MIP phase if size allows and time remains
    remain = deadline - time.time()
    pair_vars = n * (n - 1) * m
    do_mip = remain > 1.0 and pair_vars <= 220000

    if do_mip:
        mip_result = solve_mip_with_gurobi(
            n=n,
            m=m,
            p=p,
            d=d,
            w=w,
            T_horizon=max(T_horizon, sum(p)),
            init_seqs=best_seqs,
            time_limit=max(0.5, deadline - time.time()),
            logger=logger,
        )
        if mip_result is not None:
            mip_seqs, mip_starts, mip_obj = mip_result
            if mip_obj + 1e-9 < best_obj:
                best_obj = mip_obj
                best_seqs = mip_seqs
                best_starts = mip_starts
                if logger:
                    logger.log(float(best_obj))

    # Build output schedule keyed by original job IDs
    schedule = {}
    for idx, st in best_starts.items():
        jid = ids[idx]
        schedule[str(jid)] = int(round(st))

    # Ensure every job is present (robustness)
    for jid in ids:
        key = str(jid)
        if key not in schedule:
            schedule[key] = 0

    out = {
        "objective_value": float(best_obj),
        "schedule": schedule,
    }

    with open(args.solution_path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()