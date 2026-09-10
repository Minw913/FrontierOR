import argparse
import json
import math
import random
import time

import numpy as np

from solution_logger import SolutionLogger


def compute_delta(p):
    """p: (m, n) processing times. Returns delta[i, j]: minimum start-to-start
    separation on machine 1 when job j immediately follows job i (no-wait)."""
    C = np.cumsum(p, axis=0)          # C[k, j] = sum_{m' <= k} p[m', j]
    B = C - p                          # B[k, j] = sum_{m' < k}  p[m', j]
    # delta[i, j] = max_k ( C[k, i] - B[k, j] )
    delta = np.max(C[:, :, None] - B[:, None, :], axis=0)
    return delta.astype(np.float64)


def evaluate(seq, delta, T, due):
    """Returns (makespan, total tardiness violation) for sequence seq (np array)."""
    n = len(seq)
    if n == 1:
        j = seq[0]
        mk = T[j]
        viol = max(0.0, mk - due[j])
        return float(mk), float(viol)
    d = delta[seq[:-1], seq[1:]]
    starts = np.empty(n)
    starts[0] = 0.0
    np.cumsum(d, out=starts[1:])
    comp = starts + T[seq]
    tard = comp - due[seq]
    viol = float(np.maximum(tard, 0.0).sum())
    return float(comp[-1]), viol


def starts_of(seq, delta, T):
    n = len(seq)
    starts = np.empty(n)
    starts[0] = 0.0
    if n > 1:
        d = delta[seq[:-1], seq[1:]]
        np.cumsum(d, out=starts[1:])
    return starts


def sol_dict(mk, seq):
    return {"objective_value": float(mk), "sequence": [int(j) + 1 for j in seq]}


def greedy_construct(start_job, delta, T, due, n):
    seq = [start_job]
    remaining = np.ones(n, dtype=bool)
    remaining[start_job] = False
    s = 0.0
    last = start_job
    for _ in range(n - 1):
        idx = np.nonzero(remaining)[0]
        cand_start = s + delta[last, idx]
        comp = cand_start + T[idx]
        feas = comp <= due[idx]
        if feas.any():
            fi = idx[feas]
            d = delta[last, fi]
            # min delta; tie-break by earliest due date
            order = np.lexsort((due[fi], d))
            nxt = fi[order[0]]
        else:
            # pick least tardy
            tard = comp - due[idx]
            nxt = idx[int(np.argmin(tard))]
        s = s + delta[last, nxt]
        seq.append(int(nxt))
        remaining[nxt] = False
        last = int(nxt)
    return np.array(seq, dtype=np.int64)


def build_initial(delta, T, due, n, rng, deadline):
    candidates = []
    # EDD
    edd = np.argsort(due, kind="stable").astype(np.int64)
    candidates.append(edd)
    # SPT by total processing time
    spt = np.argsort(T, kind="stable").astype(np.int64)
    candidates.append(spt)
    # greedy nearest-neighbor from several starts
    starts = list(range(n)) if n <= 40 else rng.sample(range(n), 40)
    for st in starts:
        if time.time() >= deadline:
            break
        candidates.append(greedy_construct(st, delta, T, due, n))
    best = None
    best_key = None
    for seq in candidates:
        mk, viol = evaluate(seq, delta, T, due)
        key = (viol, mk)
        if best_key is None or key < best_key:
            best_key = key
            best = seq
    return best, best_key


def simulated_annealing(seq0, delta, T, due, W, deadline, rng, logger,
                        best_state, fallback_state):
    n = len(seq0)
    cur = seq0.copy()
    mk, viol = evaluate(cur, delta, T, due)
    cur_cost = mk + W * viol
    if viol == 0 and mk < best_state[0] - 1e-9:
        best_state[0] = mk
        best_state[1] = cur.copy()
        if logger:
            logger.log_solution(float(mk), sol_dict(mk, cur))
    if (viol, mk) < (fallback_state[0], fallback_state[1]):
        fallback_state[0], fallback_state[1], fallback_state[2] = viol, mk, cur.copy()

    t_start = time.time()
    total = max(deadline - t_start, 1e-3)
    T0 = max(1.0, 0.05 * (cur_cost + 1.0))
    Tend = max(1e-6, 1e-4 * (cur_cost + 1.0))
    temp = T0
    it = 0
    no_improve = 0
    while True:
        it += 1
        if (it & 127) == 0:
            now = time.time()
            if now >= deadline:
                break
            frac = min(1.0, (now - t_start) / total)
            temp = T0 * (Tend / T0) ** frac
        r = rng.random()
        if r < 0.55 and n >= 2:
            i = rng.randrange(n)
            j = rng.randrange(n - 1)
            job = cur[i]
            new = np.delete(cur, i)
            new = np.insert(new, j, job)
        elif r < 0.85 and n >= 2:
            i = rng.randrange(n)
            j = rng.randrange(n)
            new = cur.copy()
            new[i], new[j] = new[j], new[i]
        else:
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i > j:
                i, j = j, i
            new = cur.copy()
            new[i:j + 1] = new[i:j + 1][::-1]
        mk2, v2 = evaluate(new, delta, T, due)
        c2 = mk2 + W * v2
        dcost = c2 - cur_cost
        if dcost <= 0 or rng.random() < math.exp(-dcost / max(temp, 1e-9)):
            cur = new
            cur_cost = c2
            if v2 == 0 and mk2 < best_state[0] - 1e-9:
                best_state[0] = mk2
                best_state[1] = new.copy()
                no_improve = 0
                if logger:
                    logger.log_solution(float(mk2), sol_dict(mk2, new))
            if (v2, mk2) < (fallback_state[0], fallback_state[1]):
                fallback_state[0], fallback_state[1] = v2, mk2
                fallback_state[2] = new.copy()
        no_improve += 1
        # restart from best if stagnating
        if no_improve > 20000 and best_state[1] is not None:
            cur = best_state[1].copy()
            mk, viol = evaluate(cur, delta, T, due)
            cur_cost = mk + W * viol
            no_improve = 0


def solve_mip(delta, T, due, n, warm_seq, warm_mk, deadline, logger, best_state):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return

    remaining = deadline - time.time()
    if remaining < 3:
        return
    if any(due[j] - T[j] < 0 for j in range(n)):
        return

    mdl = gp.Model("nowait_flowshop")
    mdl.Params.OutputFlag = 0
    mdl.Params.Seed = 0
    mdl.Params.MIPGap = 1e-4
    mdl.Params.NumericFocus = 0
    mdl.Params.Threads = 1
    mdl.Params.TimeLimit = max(1.0, remaining)

    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x[i, j] = mdl.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    s = {}
    for j in range(n):
        s[j] = mdl.addVar(lb=0.0, ub=float(due[j] - T[j]), name=f"s_{j}")
    u = {}
    for j in range(n):
        u[j] = mdl.addVar(lb=0.0, ub=n - 1, name=f"u_{j}")
    Cmax = mdl.addVar(lb=0.0, name="Cmax")

    mdl.addConstr(gp.quicksum(x.values()) == n - 1)
    for i in range(n):
        mdl.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) <= 1)
        mdl.addConstr(gp.quicksum(x[j, i] for j in range(n) if j != i) <= 1)
    for i in range(n):
        smax_i = float(due[i] - T[i])
        for j in range(n):
            if i == j:
                continue
            M = smax_i + float(delta[i, j])
            mdl.addConstr(s[j] >= s[i] + float(delta[i, j]) - M * (1 - x[i, j]))
            mdl.addConstr(u[j] >= u[i] + 1 - n * (1 - x[i, j]))
    for j in range(n):
        mdl.addConstr(Cmax >= s[j] + float(T[j]))
    mdl.setObjective(Cmax, GRB.MINIMIZE)

    # warm start
    if warm_seq is not None:
        for v in x.values():
            v.Start = 0.0
        starts = starts_of(warm_seq, delta, T)
        for k in range(n - 1):
            x[int(warm_seq[k]), int(warm_seq[k + 1])].Start = 1.0
        for k, j in enumerate(warm_seq):
            s[int(j)].Start = float(starts[k])
            u[int(j)].Start = float(k)
        Cmax.Start = float(warm_mk)

    xkeys = list(x.keys())
    xvars = [x[k] for k in xkeys]

    def extract_sequence(vals):
        succ = {}
        has_pred = set()
        for (i, j), v in zip(xkeys, vals):
            if v > 0.5:
                succ[i] = j
                has_pred.add(j)
        start_nodes = [i for i in range(n) if i not in has_pred]
        if len(start_nodes) != 1:
            return None
        seq = [start_nodes[0]]
        while seq[-1] in succ:
            seq.append(succ[seq[-1]])
            if len(seq) > n:
                return None
        if len(seq) != n:
            return None
        return np.array(seq, dtype=np.int64)

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                vals = model.cbGetSolution(xvars)
                seq = extract_sequence(vals)
                if seq is None:
                    return
                mk, viol = evaluate(seq, delta, T, due)
                if viol <= 1e-6 and mk < best_state[0] - 1e-9:
                    best_state[0] = mk
                    best_state[1] = seq.copy()
                    if logger:
                        logger.log_solution(float(mk), sol_dict(mk, seq))
            except Exception:
                pass

    try:
        mdl.optimize(cb)
    except Exception:
        return

    if mdl.SolCount > 0:
        try:
            vals = [v.X for v in xvars]
            seq = extract_sequence(vals)
            if seq is not None:
                mk, viol = evaluate(seq, delta, T, due)
                if viol <= 1e-6 and mk < best_state[0] - 1e-9:
                    best_state[0] = mk
                    best_state[1] = seq.copy()
                    if logger:
                        logger.log_solution(float(mk), sol_dict(mk, seq))
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    t0 = time.time()
    deadline = t0 + max(1, args.time_limit) - 0.8

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    n = int(data["n_jobs"])
    m = int(data["n_machines"])
    p = np.array(data["processing_times"], dtype=np.float64)  # (m, n)
    due = np.array(data["due_dates"], dtype=np.float64)

    if n == 1:
        mk = float(p[:, 0].sum())
        sol = sol_dict(mk, [0])
        if logger:
            logger.log_solution(mk, sol)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    T = p.sum(axis=0)  # total processing time per job
    delta = compute_delta(p)

    rng = random.Random(0)

    # Initial construction
    init_seq, init_key = build_initial(delta, T, due, n, rng,
                                       min(deadline, t0 + 5.0))
    init_viol, init_mk = init_key

    best_state = [float("inf"), None]           # [best makespan, best feasible seq]
    fallback_state = [init_viol, init_mk, init_seq.copy()]  # least violation

    if init_viol == 0:
        best_state[0] = init_mk
        best_state[1] = init_seq.copy()
        if logger:
            logger.log_solution(float(init_mk), sol_dict(init_mk, init_seq))

    # Penalty weight for tardiness in SA
    W = 10.0 * max(1.0, float(T.max()))

    use_mip = n <= 55
    if use_mip:
        sa_deadline = min(deadline, time.time() + max(3.0, 0.20 * args.time_limit))
    else:
        sa_deadline = deadline

    simulated_annealing(init_seq, delta, T, due, W, sa_deadline, rng, logger,
                        best_state, fallback_state)

    if use_mip and time.time() < deadline - 2:
        warm_seq = best_state[1] if best_state[1] is not None else None
        warm_mk = best_state[0] if warm_seq is not None else None
        solve_mip(delta, T, due, n, warm_seq, warm_mk, deadline, logger, best_state)

    # If still time left and no MIP (large instance), keep improving
    if time.time() < deadline - 1 and best_state[1] is not None:
        start_seq = best_state[1].copy()
        simulated_annealing(start_seq, delta, T, due, W, deadline, rng, logger,
                            best_state, fallback_state)

    if best_state[1] is not None:
        final_seq = best_state[1]
        final_mk = best_state[0]
    else:
        final_seq = fallback_state[2]
        final_mk = fallback_state[1]

    sol = sol_dict(final_mk, final_seq)
    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()