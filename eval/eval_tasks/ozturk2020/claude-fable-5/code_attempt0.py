import argparse
import json
import random
import time
import math

from solution_logger import SolutionLogger


def machine_cost(bl, rel, proc):
    """Cost (sum of job completion times) of one machine's batch sequence."""
    t = 0
    c = 0
    for b in bl:
        r = -1
        p = 0
        for j in b:
            rj = rel[j]
            if rj > r:
                r = rj
            pj = proc[j]
            if pj > p:
                p = pj
        s = t if t > r else r
        t = s + p
        c += t * len(b)
    return c


def build_solution(machines, rel, proc):
    """Project internal state onto the required solution schema."""
    batches_out = []
    job_assign = {}
    total = 0
    for mi, bl in enumerate(machines):
        t = 0
        bpos = 0
        for b in bl:
            if not b:
                continue
            bpos += 1
            r = max(rel[j] for j in b)
            p = max(proc[j] for j in b)
            s = t if t > r else r
            t = s + p
            batches_out.append({
                "batch_id": bpos,
                "machine": mi + 1,
                "start_time": s,
                "processing_time": p,
                "jobs": list(b),
            })
            for j in b:
                job_assign[str(j)] = {
                    "batch": bpos,
                    "machine": mi + 1,
                    "completion_time": t,
                }
                total += t
    return {
        "objective_value": float(total),
        "batches": batches_out,
        "job_assignments": job_assign,
    }


def initial_solution(ids, rel, proc, size, M, Cap):
    """Greedy construction: sort by release/proc, fill batches, assign to machines."""
    order = sorted(ids, key=lambda j: (rel[j], proc[j], -size[j]))
    batches = []
    cur = []
    cursz = 0
    for j in order:
        if cur and cursz + size[j] > Cap:
            batches.append(cur)
            cur = []
            cursz = 0
        cur.append(j)
        cursz += size[j]
    if cur:
        batches.append(cur)

    machines = [[] for _ in range(M)]
    avail = [0] * M
    for b in batches:
        r = max(rel[j] for j in b)
        p = max(proc[j] for j in b)
        best_m = 0
        best_f = None
        for m in range(M):
            s = avail[m] if avail[m] > r else r
            f = s + p
            if best_f is None or f < best_f:
                best_f = f
                best_m = m
        machines[best_m].append(b)
        avail[best_m] = best_f
    return machines


def deep_copy_state(machines):
    return [[list(b) for b in bl] for bl in machines]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=60)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    start_time = time.time()
    deadline = start_time + max(1, args.time_limit) - 0.5
    random.seed(0)

    with open(args.instance_path) as f:
        inst = json.load(f)

    M = inst["num_machines"]
    Cap = inst["batch_capacity"]
    jobs = inst["jobs"]
    N = inst["num_jobs"]

    rel = {}
    proc = {}
    size = {}
    ids = []
    for j in jobs:
        jid = j["job_id"]
        ids.append(jid)
        rel[jid] = j["release_date"]
        proc[jid] = j["processing_time"]
        size[jid] = j["size"]

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # ---------- initial solution ----------
    machines = initial_solution(ids, rel, proc, size, M, Cap)
    mc = [machine_cost(machines[m], rel, proc) for m in range(M)]
    total = sum(mc)

    best_total = total
    best_state = deep_copy_state(machines)
    if logger:
        logger.log_solution(float(best_total), build_solution(best_state, rel, proc))

    # ---------- simulated annealing ----------
    T0 = max(1.0, 0.5 * total / max(1, N))
    Tmin = max(1e-6, T0 * 1e-3)
    alpha = 0.9995
    T = T0
    it = 0

    while True:
        it += 1
        if (it & 63) == 0 and time.time() > deadline:
            break
        T *= alpha
        if T < Tmin:
            T = T0
            # occasionally restart from best to intensify
            if random.random() < 0.5:
                machines = deep_copy_state(best_state)
                mc = [machine_cost(machines[m], rel, proc) for m in range(M)]
                total = sum(mc)

        r = random.random()
        affected = None
        saved = None

        if r < 0.45:
            # ----- move one job -----
            cand = [m for m in range(M) if machines[m]]
            if not cand:
                continue
            ms = random.choice(cand)
            bl = machines[ms]
            bi = random.randrange(len(bl))
            b = bl[bi]
            jsel = random.choice(b)
            mt = random.randrange(M)
            tl = machines[mt]
            new_batch = (random.random() < 0.3) or (not tl)
            tb = None
            if not new_batch:
                ti = random.randrange(len(tl))
                if mt == ms and ti == bi:
                    continue
                tb = tl[ti]
                if sum(size[x] for x in tb) + size[jsel] > Cap:
                    continue
            affected = {ms, mt}
            saved = {m: [list(bb) for bb in machines[m]] for m in affected}
            b.remove(jsel)
            if not b:
                del bl[bi]
            if new_batch:
                pos = random.randrange(len(machines[mt]) + 1)
                machines[mt].insert(pos, [jsel])
            else:
                tb.append(jsel)

        elif r < 0.70:
            # ----- swap two jobs between different batches -----
            cand = [m for m in range(M) if machines[m]]
            if not cand:
                continue
            m1 = random.choice(cand)
            b1i = random.randrange(len(machines[m1]))
            m2 = random.choice(cand)
            b2i = random.randrange(len(machines[m2]))
            if m1 == m2 and b1i == b2i:
                continue
            b1 = machines[m1][b1i]
            b2 = machines[m2][b2i]
            j1 = random.choice(b1)
            j2 = random.choice(b2)
            sz1 = sum(size[x] for x in b1)
            sz2 = sum(size[x] for x in b2)
            if sz1 - size[j1] + size[j2] > Cap or sz2 - size[j2] + size[j1] > Cap:
                continue
            affected = {m1, m2}
            saved = {m: [list(bb) for bb in machines[m]] for m in affected}
            b1.remove(j1)
            b2.remove(j2)
            b1.append(j2)
            b2.append(j1)

        elif r < 0.85:
            # ----- move a whole batch -----
            cand = [m for m in range(M) if machines[m]]
            if not cand:
                continue
            ms = random.choice(cand)
            bl = machines[ms]
            bi = random.randrange(len(bl))
            mt = random.randrange(M)
            affected = {ms, mt}
            saved = {m: [list(bb) for bb in machines[m]] for m in affected}
            b = bl.pop(bi)
            pos = random.randrange(len(machines[mt]) + 1)
            machines[mt].insert(pos, b)

        else:
            # ----- swap adjacent batches on a machine -----
            cand = [m for m in range(M) if len(machines[m]) >= 2]
            if not cand:
                continue
            ms = random.choice(cand)
            bl = machines[ms]
            i = random.randrange(len(bl) - 1)
            affected = {ms}
            saved = {ms: [list(bb) for bb in bl]}
            bl[i], bl[i + 1] = bl[i + 1], bl[i]

        # ----- evaluate -----
        old_part = sum(mc[m] for m in affected)
        new_costs = {m: machine_cost(machines[m], rel, proc) for m in affected}
        new_part = sum(new_costs.values())
        delta = new_part - old_part

        if delta <= 0 or random.random() < math.exp(-delta / T):
            for m in affected:
                mc[m] = new_costs[m]
            total += delta
            if total < best_total - 1e-9:
                best_total = total
                best_state = deep_copy_state(machines)
                if logger:
                    logger.log_solution(
                        float(best_total),
                        build_solution(best_state, rel, proc),
                    )
        else:
            for m in affected:
                machines[m] = saved[m]

    # ---------- final output ----------
    sol = build_solution(best_state, rel, proc)
    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()