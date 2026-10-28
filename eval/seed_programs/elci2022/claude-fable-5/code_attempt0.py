import argparse
import json
import time
import random
import bisect

from solution_logger import SolutionLogger


# ---------------------------------------------------------------------------
# Timeline (resource profile) utilities: piecewise-constant usage function.
# points[k] .. points[k+1] has usage usage[k]; after the last point usage is 0.
# ---------------------------------------------------------------------------
def tl_add(points, usage, s, e, r):
    """Add r units of usage on [s, e)."""
    if s >= e:
        return
    i = bisect.bisect_left(points, s)
    if i == len(points) or points[i] != s:
        prev = usage[i - 1] if i > 0 else 0
        points.insert(i, s)
        usage.insert(i, prev)
    j = bisect.bisect_left(points, e)
    if j == len(points) or points[j] != e:
        prev = usage[j - 1] if j > 0 else 0
        points.insert(j, e)
        usage.insert(j, prev)
    for k in range(i, j):
        usage[k] += r


def earliest(points, usage, rel_t, p, limit):
    """Earliest t >= rel_t such that usage <= limit on all of [t, t+p)."""
    cand = rel_t
    L = len(points)
    while True:
        i = bisect.bisect_right(points, cand) - 1
        if i < 0:
            i = 0
        end = cand + p
        j = i
        bad = -1
        while j < L and points[j] < end:
            if usage[j] > limit:
                bad = j
                break
            j += 1
        if bad < 0:
            return cand
        # last segment always has usage 0, so bad + 1 < L when limit >= 0
        cand = points[bad + 1]


def greedy_schedule(order, pt_f, req_f, rel, cap):
    """List-schedule jobs in the given order at earliest feasible start times."""
    points = [0]
    usage = [0]
    starts = {}
    ms = 0
    for j in order:
        p = pt_f[j]
        r = req_f[j]
        rj = rel[j]
        if p <= 0:
            starts[j] = rj
            if rj > ms:
                ms = rj
            continue
        limit = cap - r
        if limit < 0:
            t = points[-1] if points[-1] > rj else rj
        else:
            t = earliest(points, usage, rj, p, limit)
        tl_add(points, usage, t, t + p, r)
        starts[j] = t
        if t + p > ms:
            ms = t + p
    return ms, starts


def schedule_jobs(job_list, pt_f, req_f, rel, cap):
    """Best of several priority orders."""
    if not job_list:
        return 0, {}
    o1 = sorted(job_list, key=lambda j: (rel[j], -pt_f[j], j))
    o2 = sorted(job_list, key=lambda j: (rel[j], -pt_f[j] * req_f[j], j))
    o3 = sorted(job_list, key=lambda j: (-pt_f[j], rel[j], j))
    best_ms = None
    best_starts = None
    seen = set()
    for order in (o1, o2, o3):
        key = tuple(order)
        if key in seen:
            continue
        seen.add(key)
        ms, starts = greedy_schedule(order, pt_f, req_f, rel, cap)
        if best_ms is None or ms < best_ms:
            best_ms, best_starts = ms, starts
    return best_ms, best_starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=60)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t_start = time.time()
    deadline = t_start + max(1, args.time_limit) - 0.3
    random.seed(0)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as fh:
        data = json.load(fh)

    n = data["n"]
    m = data["m"]
    S = data["num_scenarios"]
    caps = data["facility_capacity_limits"]
    req = data["capacity_requirements"]          # [m][n]
    rel = data["release_times"]                  # [n]
    scens = data["scenarios"]
    prob = [sc["probability"] for sc in scens]
    spt = [sc["processing_times"] for sc in scens]  # [s][m][n]

    # expected processing time per (facility, job)
    Ep = [[sum(prob[s] * spt[s][f][j] for s in range(S)) for j in range(n)]
          for f in range(m)]

    # feasible facilities per job
    feas = []
    for j in range(n):
        fs = [f for f in range(m) if req[f][j] <= caps[f]]
        if not fs:
            fs = [min(range(m), key=lambda f: req[f][j])]
        feas.append(fs)
    any_movable = any(len(feas[j]) > 1 for j in range(n))

    # ------------------ initial assignment: greedy load balancing -----------
    def area(f, j):
        return Ep[f][j] * req[f][j] / max(1, caps[f])

    order_jobs = sorted(range(n),
                        key=lambda j: -max(area(f, j) for f in feas[j]))
    loads = [0.0] * m
    assign = [0] * n
    for j in order_jobs:
        best_f = min(feas[j], key=lambda f: (loads[f] + area(f, j), Ep[f][j]))
        assign[j] = best_f
        loads[best_f] += area(best_f, j)

    fac_jobs = [set() for _ in range(m)]
    for j in range(n):
        fac_jobs[assign[j]].add(j)

    # ------------------ state caches ----------------------------------------
    ms_mat = [[0] * S for _ in range(m)]        # makespan per facility, scenario
    starts_cache = [[{} for _ in range(S)] for _ in range(m)]

    def recompute(f, jobs):
        jl = sorted(jobs)
        rows = []
        schs = []
        for s in range(S):
            ms, starts = schedule_jobs(jl, spt[s][f], req[f], rel, caps[f])
            rows.append(ms)
            schs.append(starts)
        return rows, schs

    def full_recompute():
        for f in range(m):
            rows, schs = recompute(f, fac_jobs[f])
            ms_mat[f] = rows
            starts_cache[f] = schs

    def calc_obj():
        tot = 0.0
        for s in range(S):
            mx = 0
            for f in range(m):
                v = ms_mat[f][s]
                if v > mx:
                    mx = v
            tot += prob[s] * mx
        return tot

    def obj_with(f, rowf, g, rowg):
        tot = 0.0
        for s in range(S):
            mx = 0
            for h in range(m):
                if h == f:
                    v = rowf[s]
                elif h == g:
                    v = rowg[s]
                else:
                    v = ms_mat[h][s]
                if v > mx:
                    mx = v
            tot += prob[s] * mx
        return tot

    def build_solution(obj):
        schedule = {}
        for s in range(S):
            d = {}
            for f in range(m):
                ptf = spt[s][f]
                for j, st in starts_cache[f][s].items():
                    d[str(j)] = {"facility": int(f),
                                 "start": int(st),
                                 "finish": int(st + ptf[j])}
            schedule[str(s)] = d
        return {
            "objective_value": float(obj),
            "assignment": {str(j): int(assign[j]) for j in range(n)},
            "schedule": schedule,
        }

    full_recompute()
    cur_obj = calc_obj()
    best_obj = cur_obj
    best_sol = build_solution(best_obj)
    if logger:
        logger.log_solution(best_obj, best_sol)

    state = {"cur_obj": cur_obj, "best_obj": best_obj, "best_sol": best_sol}

    def maybe_record():
        if state["cur_obj"] < state["best_obj"] - 1e-9:
            state["best_obj"] = state["cur_obj"]
            state["best_sol"] = build_solution(state["best_obj"])
            if logger:
                logger.log_solution(state["best_obj"], state["best_sol"])

    # ------------------ local search moves -----------------------------------
    def move_pass():
        improved = False
        jobs_order = list(range(n))
        random.shuffle(jobs_order)
        for j in jobs_order:
            if time.time() >= deadline:
                return improved
            f = assign[j]
            if len(feas[j]) <= 1:
                continue
            for g in feas[j]:
                if g == f:
                    continue
                rowf, schf = recompute(f, fac_jobs[f] - {j})
                rowg, schg = recompute(g, fac_jobs[g] | {j})
                nobj = obj_with(f, rowf, g, rowg)
                if nobj < state["cur_obj"] - 1e-9:
                    assign[j] = g
                    fac_jobs[f].discard(j)
                    fac_jobs[g].add(j)
                    ms_mat[f] = rowf
                    ms_mat[g] = rowg
                    starts_cache[f] = schf
                    starts_cache[g] = schg
                    state["cur_obj"] = nobj
                    improved = True
                    maybe_record()
                    break
        return improved

    def swap_pass(tries=200):
        improved = False
        for _ in range(tries):
            if time.time() >= deadline:
                return improved
            j = random.randrange(n)
            k = random.randrange(n)
            f = assign[j]
            g = assign[k]
            if j == k or f == g:
                continue
            if g not in feas[j] or f not in feas[k]:
                continue
            rowf, schf = recompute(f, (fac_jobs[f] - {j}) | {k})
            rowg, schg = recompute(g, (fac_jobs[g] - {k}) | {j})
            nobj = obj_with(f, rowf, g, rowg)
            if nobj < state["cur_obj"] - 1e-9:
                assign[j], assign[k] = g, f
                fac_jobs[f].discard(j); fac_jobs[f].add(k)
                fac_jobs[g].discard(k); fac_jobs[g].add(j)
                ms_mat[f] = rowf
                ms_mat[g] = rowg
                starts_cache[f] = schf
                starts_cache[g] = schg
                state["cur_obj"] = nobj
                improved = True
                maybe_record()
        return improved

    def polish(rounds=30):
        """Try random priority orders on bottleneck facility/scenario pairs."""
        improved = False
        for _ in range(rounds):
            if time.time() >= deadline:
                break
            changed = False
            for s in range(S):
                f = max(range(m), key=lambda h: ms_mat[h][s])
                jobs = list(fac_jobs[f])
                if not jobs or ms_mat[f][s] <= 0:
                    continue
                order = jobs[:]
                if random.random() < 0.5:
                    random.shuffle(order)
                else:
                    ptf = spt[s][f]
                    order.sort(key=lambda j: (rel[j],
                                              -ptf[j] * (0.5 + random.random())))
                nms, nstarts = greedy_schedule(order, spt[s][f], req[f],
                                               rel, caps[f])
                if nms < ms_mat[f][s]:
                    ms_mat[f][s] = nms
                    starts_cache[f][s] = nstarts
                    changed = True
            if changed:
                nobj = calc_obj()
                if nobj < state["cur_obj"] - 1e-9:
                    state["cur_obj"] = nobj
                    improved = True
                    maybe_record()
                else:
                    state["cur_obj"] = nobj
        return improved

    def perturb():
        movable = [j for j in range(n) if len(feas[j]) > 1]
        if not movable:
            return
        k = max(1, len(movable) // 8)
        picks = random.sample(movable, min(k, len(movable)))
        for j in picks:
            choices = [g for g in feas[j] if g != assign[j]]
            if not choices:
                continue
            g = random.choice(choices)
            fac_jobs[assign[j]].discard(j)
            assign[j] = g
            fac_jobs[g].add(j)
        full_recompute()
        state["cur_obj"] = calc_obj()
        maybe_record()

    # ------------------ main optimization loop --------------------------------
    while time.time() < deadline:
        if any_movable:
            if move_pass():
                continue
            if time.time() >= deadline:
                break
            if swap_pass(200):
                continue
            if time.time() >= deadline:
                break
            if polish(30):
                continue
            if time.time() >= deadline:
                break
            perturb()
        else:
            if not polish(50):
                # keep trying random orders while time remains (cheap)
                if time.time() >= deadline:
                    break

    with open(args.solution_path, "w") as fh:
        json.dump(state["best_sol"], fh)


if __name__ == "__main__":
    main()