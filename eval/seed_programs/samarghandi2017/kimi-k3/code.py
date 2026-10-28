#!/usr/bin/env python3
import argparse
import heapq
import json
import math
import os
import random
import time

from solution_logger import SolutionLogger

try:
    import gurobipy as gp
    from gurobipy import GRB
    _HAS_GUROBI = True
except Exception:
    gp = None
    GRB = None
    _HAS_GUROBI = False


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, required=True)
    ap.add_argument("--log_path", default=None)
    return ap.parse_args()


def iter_bits(mask):
    while mask:
        b = mask & -mask
        yield b.bit_length() - 1
        mask ^= b


def build_data(inst):
    n = int(inst["n_jobs"])
    m = int(inst["n_machines"])
    pt = inst["processing_times"]
    due = list(inst["due_dates"])

    pref = [[0] * n for _ in range(m)]
    for k in range(m):
        row = pt[k]
        for j in range(n):
            pref[k][j] = int(row[j]) + (pref[k - 1][j] if k > 0 else 0)

    if m > 0:
        proc = pref[m - 1][:]
    else:
        proc = [0] * n

    dist = [[0] * n for _ in range(n)]
    for i in range(n):
        di = dist[i]
        for j in range(n):
            if i == j:
                continue
            best = None
            for k in range(m):
                prev_j = pref[k - 1][j] if k > 0 else 0
                val = pref[k][i] - prev_j
                if best is None or val > best:
                    best = val
            di[j] = 0 if best is None else best

    min_in = [0] * n
    for j in range(n):
        vals = [dist[i][j] for i in range(n) if i != j]
        min_in[j] = min(vals) if vals else 0

    return n, m, proc, dist, due, min_in


def evaluate_sequence(seq, proc, dist, due):
    t = 0
    obj = 0
    min_slack = None
    last = len(seq) - 1
    for idx, j in enumerate(seq):
        c = t + proc[j]
        obj = c
        slack = due[j] - c
        if min_slack is None or slack < min_slack:
            min_slack = slack
        if slack < 0:
            return obj, False, min_slack
        if idx < last:
            t += dist[j][seq[idx + 1]]
    return obj, True, (0 if min_slack is None else min_slack)


def makespan_only(seq, proc, dist):
    t = 0
    obj = 0
    last = len(seq) - 1
    for idx, j in enumerate(seq):
        obj = t + proc[j]
        if idx < last:
            t += dist[j][seq[idx + 1]]
    return obj


def violation_score(seq, proc, dist, due):
    t = 0
    obj = 0
    max_v = 0
    tot_v = 0
    last = len(seq) - 1
    for idx, j in enumerate(seq):
        c = t + proc[j]
        obj = c
        v = c - due[j]
        if v > 0:
            tot_v += v
            if v > max_v:
                max_v = v
        if idx < last:
            t += dist[j][seq[idx + 1]]
    return max_v, tot_v, obj


def insertion_build(order, proc, dist, due, stop_time):
    seq = []
    checks = 0
    for job in order:
        best = None
        best_key = None
        for pos in range(len(seq) + 1):
            cand = seq[:pos] + [job] + seq[pos:]
            obj, feas, ms = evaluate_sequence(cand, proc, dist, due)
            if feas:
                key = (obj, -ms)
                if best_key is None or key < best_key:
                    best_key = key
                    best = cand
            checks += 1
            if (checks & 127) == 0 and time.monotonic() >= stop_time:
                return None
        if best is None:
            return None
        seq = best
    return seq


def beam_build(proc, dist, due, min_in, width, candidate_limit, stop_time):
    n = len(proc)
    if n == 0:
        return []
    if n == 1:
        return [0] if proc[0] <= due[0] else None

    full = (1 << n) - 1
    states = []

    for i in range(n):
        if proc[i] > due[i]:
            continue
        rem = full ^ (1 << i)
        ok = True
        sum_min = 0
        minp = None
        for r in iter_bits(rem):
            if dist[i][r] + proc[r] > due[r]:
                ok = False
                break
            sum_min += min_in[r]
            if minp is None or proc[r] < minp:
                minp = proc[r]
        if not ok:
            continue
        if rem == 0:
            return [i]
        lb = sum_min + (minp if minp is not None else 0)
        states.append((lb, 0, i, (i,), rem))

    if not states:
        return None
    states = heapq.nsmallest(width, states, key=lambda s: (s[0], s[1]))

    while states and time.monotonic() < stop_time:
        nextdict = {}
        best_final = None
        best_final_obj = None
        counter = 0

        for _score, t, last, seq, rem in states:
            if rem == 0:
                obj = t + proc[last]
                if best_final_obj is None or obj < best_final_obj:
                    best_final_obj = obj
                    best_final = seq
                continue

            if candidate_limit is not None:
                cands = list(iter_bits(rem))
                if len(cands) > candidate_limit:
                    cands = heapq.nsmallest(
                        candidate_limit,
                        cands,
                        key=lambda j: (
                            due[j] - (t + dist[last][j] + proc[j]),
                            due[j],
                            dist[last][j],
                            j,
                        ),
                    )
            else:
                cands = iter_bits(rem)

            for j in cands:
                bj = 1 << j
                tj = t + dist[last][j]
                if tj + proc[j] > due[j]:
                    continue
                rem2 = rem ^ bj
                ok = True
                sum_min = 0
                minp = None
                for r in iter_bits(rem2):
                    if tj + dist[j][r] + proc[r] > due[r]:
                        ok = False
                        break
                    sum_min += min_in[r]
                    if minp is None or proc[r] < minp:
                        minp = proc[r]
                if not ok:
                    continue

                newseq = seq + (j,)
                if rem2 == 0:
                    obj = tj + proc[j]
                    if best_final_obj is None or obj < best_final_obj:
                        best_final_obj = obj
                        best_final = newseq
                    continue

                lb = tj + sum_min + (minp if minp is not None else 0)
                key = (j, rem2)
                old = nextdict.get(key)
                if old is None or tj < old[1]:
                    nextdict[key] = (lb, tj, j, newseq, rem2)

                counter += 1
                if (counter & 511) == 0 and time.monotonic() >= stop_time:
                    break
            if time.monotonic() >= stop_time:
                break

        if best_final is not None:
            return list(best_final)
        if not nextdict:
            return None
        states = heapq.nsmallest(width, nextdict.values(), key=lambda s: (s[0], s[1]))

    return None


def exact_dp(proc, dist, due, stop_time):
    n = len(proc)
    if n == 0:
        return [], True
    full_mask = (1 << n) - 1
    size = 1 << n
    dp = [None] * size
    parent = {}

    for i in range(n):
        if proc[i] <= due[i]:
            dp[1 << i] = {i: 0}

    for mask in range(size):
        if (mask & 1023) == 0 and time.monotonic() >= stop_time:
            return None, False
        dmap = dp[mask]
        if not dmap:
            continue
        rem = full_mask ^ mask
        if rem == 0:
            continue

        for last, t in list(dmap.items()):
            dead = False
            nxt = []
            mm = rem
            while mm:
                b = mm & -mm
                j = b.bit_length() - 1
                mm ^= b
                if t + dist[last][j] + proc[j] > due[j]:
                    dead = True
                    break
                nxt.append((j, b))
            if dead:
                continue

            dl = dist[last]
            for j, b in nxt:
                tj = t + dl[j]
                nm = mask | b
                nd = dp[nm]
                if nd is None:
                    nd = {}
                    dp[nm] = nd
                old = nd.get(j)
                if old is None or tj < old:
                    nd[j] = tj
                    parent[nm * n + j] = mask * n + last

    final = dp[full_mask]
    if not final:
        return None, True

    best_last = None
    best_obj = None
    for last, t in final.items():
        obj = t + proc[last]
        if best_obj is None or obj < best_obj:
            best_obj = obj
            best_last = last

    seq = []
    mask = full_mask
    cur = best_last
    while True:
        seq.append(cur)
        if mask == (1 << cur):
            break
        enc = parent.get(mask * n + cur)
        if enc is None:
            return None, False
        mask, cur = divmod(enc, n)
    seq.reverse()
    return seq, True


def move_job(seq, i, pos):
    job = seq[i]
    rest = seq[:i] + seq[i + 1:]
    return rest[:pos] + [job] + rest[pos:]


def local_search(seq, proc, dist, due, stop_time, consider, full_threshold=130):
    if seq is None:
        return None
    obj, feas, _ = evaluate_sequence(seq, proc, dist, due)
    if not feas:
        return seq
    best = list(seq)
    best_obj = obj
    n = len(best)
    if n <= 1:
        return best

    if n <= full_threshold:
        checks = 0
        while time.monotonic() < stop_time:
            move_seq = None
            move_obj = best_obj
            for i in range(n):
                job = best[i]
                rest = best[:i] + best[i + 1:]
                for pos in range(n):
                    if pos == i:
                        continue
                    cand = rest[:pos] + [job] + rest[pos:]
                    cobj, cfeas, _ = evaluate_sequence(cand, proc, dist, due)
                    checks += 1
                    if cfeas and cobj < move_obj:
                        move_obj = cobj
                        move_seq = cand
                    if (checks & 255) == 0 and time.monotonic() >= stop_time:
                        break
                if time.monotonic() >= stop_time:
                    break
            if move_seq is None:
                break
            best = move_seq
            best_obj = move_obj
            consider(best)
        return best

    rnd = random.Random(0)
    while time.monotonic() < stop_time:
        swap_seq = None
        swap_obj = best_obj
        for i in range(n - 1):
            cand = best[:]
            cand[i], cand[i + 1] = cand[i + 1], cand[i]
            cobj, cfeas, _ = evaluate_sequence(cand, proc, dist, due)
            if cfeas and cobj < swap_obj:
                swap_obj = cobj
                swap_seq = cand
            if (i & 63) == 0 and time.monotonic() >= stop_time:
                break
        if swap_seq is not None:
            best = swap_seq
            best_obj = swap_obj
            consider(best)
            continue

        found = False
        trials = 0
        while trials < 2500 and time.monotonic() < stop_time:
            i = rnd.randrange(n)
            pos = rnd.randrange(n)
            if pos == i:
                continue
            cand = move_job(best, i, pos)
            cobj, cfeas, _ = evaluate_sequence(cand, proc, dist, due)
            if cfeas and cobj < best_obj:
                best = cand
                best_obj = cobj
                consider(best)
                found = True
                break
            trials += 1
        if not found:
            break
    return best


def sequence_from_x(vals, n):
    succ = [-1] * n
    indeg = [0] * n
    for i in range(n):
        base = i * n
        for j in range(n):
            if i != j and vals[base + j] > 0.5:
                if succ[i] != -1:
                    return None
                succ[i] = j
                indeg[j] += 1
    roots = [i for i in range(n) if indeg[i] == 0]
    if len(roots) != 1:
        return None
    seq = []
    seen = set()
    cur = roots[0]
    while cur != -1:
        if cur in seen:
            return None
        seen.add(cur)
        seq.append(cur)
        cur = succ[cur]
    return seq if len(seq) == n else None


def run_mip(proc, dist, due, min_in, best, consider, stop_time):
    if not _HAS_GUROBI:
        return "unavailable"
    n = len(proc)
    if n == 0:
        return "optimal"
    if time.monotonic() >= stop_time:
        return "no_time"

    max_t = max(due[j] - proc[j] for j in range(n))
    if max_t < 0:
        return "infeasible"

    model = gp.Model("no_wait_flow_shop_due_dates")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam("OutputFlag", 0)

    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    for i in range(n):
        x[i, i].ub = 0.0

    u = model.addVars(n, lb=0.0, ub=float(max(0, n - 1)), vtype=GRB.CONTINUOUS, name="u")
    t = model.addVars(n, lb=0.0, name="t")
    for j in range(n):
        t[j].ub = float(due[j] - proc[j])
    cmax = model.addVar(lb=0.0, name="cmax")
    model.setObjective(cmax, GRB.MINIMIZE)

    outdeg = [gp.quicksum(x[i, j] for j in range(n) if j != i) for i in range(n)]
    indeg = [gp.quicksum(x[i, j] for i in range(n) if i != j) for j in range(n)]

    for i in range(n):
        model.addConstr(outdeg[i] <= 1, name=f"out_{i}")
    for j in range(n):
        model.addConstr(indeg[j] <= 1, name=f"in_{j}")
    model.addConstr(
        gp.quicksum(x[i, j] for i in range(n) for j in range(n) if i != j) == n - 1,
        name="n_links",
    )
    for i in range(n):
        for j in range(i + 1, n):
            model.addConstr(x[i, j] + x[j, i] <= 1, name=f"anti_{i}_{j}")

    mt = float(max_t)
    for j in range(n):
        model.addConstr(u[j] >= indeg[j], name=f"u_lb_{j}")
        model.addConstr(u[j] <= float(max(0, n - 1)) * indeg[j], name=f"u_ub_{j}")
        model.addConstr(t[j] <= mt * indeg[j], name=f"root_t_{j}")
        if n > 1:
            model.addConstr(t[j] >= float(min_in[j]) * indeg[j], name=f"t_lb_{j}")
        model.addConstr(t[j] + float(proc[j]) <= float(due[j]), name=f"due_{j}")
        model.addConstr(cmax >= t[j] + float(proc[j]), name=f"cmax_{j}")

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            model.addConstr(
                u[j] >= u[i] + 1.0 - float(n) * (1.0 - x[i, j]),
                name=f"mtz_{i}_{j}",
            )
            dij = float(dist[i][j])
            bigm = mt + (dij if dij > 0.0 else 0.0)
            model.addConstr(
                t[j] >= t[i] + dij - bigm * (1.0 - x[i, j]),
                name=f"time_{i}_{j}",
            )

    if best.get("seq") is not None and best.get("obj") is not None:
        seq = best["seq"]
        starts = [0.0] * n
        tt = 0
        for pos, job in enumerate(seq):
            starts[job] = float(tt)
            if pos + 1 < n:
                tt += dist[job][seq[pos + 1]]
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[i, j].Start = 0.0
        for a, b in zip(seq, seq[1:]):
            x[a, b].Start = 1.0
        for pos, job in enumerate(seq):
            u[job].Start = float(pos)
            t[job].Start = starts[job]
        cmax.Start = float(best["obj"])

    xflat = [x[i, j] for i in range(n) for j in range(n)]

    def _callback(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                vals = model.cbGetSolution(xflat)
                seq = sequence_from_x(vals, n)
                if seq is not None:
                    consider(seq)
            except Exception:
                pass

    remaining = stop_time - time.monotonic()
    if remaining <= 0.02:
        return "no_time"
    model.setParam("TimeLimit", max(0.01, remaining))
    model.optimize(_callback)

    try:
        if model.SolCount > 0:
            vals = model.getAttr("X", xflat)
            seq = sequence_from_x(vals, n)
            if seq is not None:
                consider(seq)
    except Exception:
        pass

    status = model.Status
    if status == GRB.OPTIMAL:
        return "optimal"
    if status == GRB.TIME_LIMIT:
        return "time_limit"
    if status == GRB.INFEASIBLE:
        return "infeasible"
    if status == GRB.INF_OR_UNBD:
        return "inf_or_unbd"
    return str(status)


def main():
    args = parse_args()
    start_time = time.monotonic()
    total_limit = max(0, int(args.time_limit))
    reserve = max(0.02, min(0.50, 0.05 * total_limit)) if total_limit > 0 else 0.0
    work_deadline = start_time + max(0.0, total_limit - reserve)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    random.seed(0)

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n, m, proc, dist, due, min_in = build_data(inst)

    best = {"obj": None, "seq": None}

    def consider(seq):
        if seq is None or len(seq) != n or len(set(seq)) != n:
            return False
        obj, feas, _ = evaluate_sequence(seq, proc, dist, due)
        if not feas:
            return False
        if best["obj"] is None or obj < best["obj"]:
            best["obj"] = obj
            best["seq"] = list(seq)
            if logger:
                sol = {
                    "objective_value": float(obj),
                    "sequence": [int(j + 1) for j in seq],
                }
                logger.log_solution(float(obj), sol)
            return True
        return False

    def remaining_work():
        return work_deadline - time.monotonic()

    def fallback_sequence():
        cands = []
        if n == 0:
            return []
        orders = [
            sorted(range(n), key=lambda j: (due[j], proc[j], j)),
            sorted(range(n), key=lambda j: (due[j] - proc[j], due[j], j)),
            sorted(range(n), key=lambda j: (proc[j], due[j], j)),
            list(range(n)),
        ]
        for seq in orders:
            cands.append((violation_score(seq, proc, dist, due), seq))
        cands.sort(key=lambda z: z[0])
        return cands[0][1]

    def finalize():
        if best["seq"] is None:
            seq = fallback_sequence()
            obj = makespan_only(seq, proc, dist)
        else:
            seq = best["seq"]
            obj = best["obj"]
        sol = {
            "objective_value": float(obj),
            "sequence": [int(j + 1) for j in seq],
        }
        out_dir = os.path.dirname(os.path.abspath(args.solution_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)

    if n == 0:
        if logger:
            logger.log_solution(0.0, {"objective_value": 0.0, "sequence": []})
        finalize()
        return

    base_orders = []
    seen_orders = set()
    for keyfunc in (
        lambda j: (due[j], proc[j], j),
        lambda j: (due[j] - proc[j], due[j], j),
        lambda j: (proc[j], due[j], j),
        lambda j: (due[j], -proc[j], j),
    ):
        order = tuple(sorted(range(n), key=keyfunc))
        if order not in seen_orders:
            seen_orders.add(order)
            base_orders.append(list(order))

    for order in base_orders:
        consider(order)

    run_exact = False
    if n <= 13:
        run_exact = True
    elif n == 14 and total_limit >= 2:
        run_exact = True
    elif n == 15 and total_limit >= 8:
        run_exact = True
    elif n == 16 and total_limit >= 20:
        run_exact = True

    if run_exact and remaining_work() > 0.02:
        seq, completed = exact_dp(proc, dist, due, work_deadline)
        if completed:
            if seq is not None:
                consider(seq)
            finalize()
            return

    rem = remaining_work()
    plan_mip = False
    if _HAS_GUROBI and rem > 0.05:
        if n <= 55 and rem > 1.0:
            plan_mip = True
        elif n <= 90 and rem > 6.0:
            plan_mip = True

    if n <= 260 and remaining_work() > 0.05:
        budget = min(1.2 if plan_mip else 4.0, max(0.10, 0.18 * remaining_work()))
        stop = min(work_deadline, time.monotonic() + budget)
        for order in base_orders[:3]:
            if time.monotonic() >= stop:
                break
            seq = insertion_build(order, proc, dist, due, stop)
            if seq is not None:
                consider(seq)

    if n > 1 and remaining_work() > 0.05:
        if n <= 80:
            width, cand_limit = 48, None
        elif n <= 140:
            width, cand_limit = 24, None
        elif n <= 220:
            width, cand_limit = 10, None
        else:
            width, cand_limit = 3, 30
        if plan_mip:
            budget = min(1.8, max(0.10, 0.22 * remaining_work()))
        else:
            budget = min(8.0 if n <= 220 else 6.0, max(0.20, 0.45 * remaining_work()))
        stop = min(work_deadline, time.monotonic() + budget)
        seq = beam_build(proc, dist, due, min_in, width, cand_limit, stop)
        if seq is not None:
            consider(seq)

    if best["seq"] is not None and remaining_work() > 0.05:
        if plan_mip:
            budget = min(0.8, max(0.10, 0.12 * remaining_work()))
            stop = min(work_deadline, time.monotonic() + budget)
        elif n <= 220:
            budget = min(2.0, max(0.20, 0.30 * remaining_work()))
            stop = min(work_deadline, time.monotonic() + budget)
        else:
            stop = work_deadline
        local_search(best["seq"], proc, dist, due, stop, consider)

    mip_status = None
    if plan_mip and remaining_work() > 0.05:
        try:
            mip_status = run_mip(proc, dist, due, min_in, best, consider, work_deadline)
        except Exception:
            mip_status = "error"

    if not plan_mip and n <= 220 and remaining_work() > 0.15:
        failures = 0
        due_span = (max(due) - min(due)) if due else 1
        jitter = max(1, int(due_span // 3) + 1)
        rnd = random.Random(12345)
        while remaining_work() > 0.15 and failures < 6:
            order = sorted(
                range(n),
                key=lambda j: (
                    due[j] + rnd.randint(-jitter, jitter),
                    due[j] - proc[j] + rnd.randint(-jitter, jitter),
                    rnd.random(),
                ),
            )
            chunk = min(1.5, max(0.25, 0.30 * remaining_work()))
            ins_stop = min(work_deadline, time.monotonic() + 0.65 * chunk)
            loc_stop = min(work_deadline, time.monotonic() + chunk)
            seq = insertion_build(order, proc, dist, due, ins_stop)
            if seq is None:
                failures += 1
                continue
            seq = local_search(seq, proc, dist, due, loc_stop, consider)
            if consider(seq):
                failures = 0
            else:
                failures += 1

    if best["seq"] is not None and remaining_work() > 0.10:
        if (not plan_mip and n > 220) or mip_status in (None, "error", "infeasible", "inf_or_unbd"):
            local_search(best["seq"], proc, dist, due, work_deadline, consider)

    finalize()


if __name__ == "__main__":
    main()