import argparse
import json
import math
import random
import time

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except Exception:
    gp = None
    GRB = None
    GUROBI_AVAILABLE = False

EPS = 1e-9


# ---------------------------------------------------------------------------
# Problem data
# ---------------------------------------------------------------------------
class Problem:
    def __init__(self, inst):
        self.n = int(inst["num_jobs"])
        self.M = int(inst["num_machines"])
        jobs = sorted(inst["jobs"], key=lambda x: int(x["job_id"]))
        self.job_ids = [int(j["job_id"]) for j in jobs]
        self.L = len(jobs[0]["operations"]) if self.n else self.M
        self.N = self.n * self.L
        self.proc = [0.0] * self.N          # flat processing times, oid = j*L+k
        self.mach = [0] * self.N            # flat machine index
        self.r = [0.0] * self.n
        self.d = [0.0] * self.n
        self.w = [0.0] * self.n
        self.total_proc = [0.0] * self.n
        self.opid_mach = [[-1] * self.n for _ in range(self.M)]  # opid_mach[m][j]
        for j, job in enumerate(jobs):
            ops = job["operations"]
            self.r[j] = float(job.get("release_date", 0))
            self.d[j] = float(job.get("due_date", 0))
            self.w[j] = float(job.get("weight", 1))
            for k, op in enumerate(ops):
                oid = j * self.L + k
                m = int(op["machine"])
                pt = float(op["processing_time"])
                self.proc[oid] = pt
                self.mach[oid] = m
                self.total_proc[j] += pt
                self.opid_mach[m][j] = oid
        # canonical op list per machine (one op per job on each machine)
        self.mach_ops_ids = [
            [self.opid_mach[m][j] for j in range(self.n)] for m in range(self.M)
        ]
        sp = sum(self.proc)
        self.avg_p = sp / max(1, self.N)
        self.bigM = (max(self.r) if self.r else 0.0) + sp + 1.0
        # estimated number of disjunctive pairs for the MIP
        self.est_pairs = sum(
            len(ids) * (len(ids) - 1) // 2 for ids in self.mach_ops_ids
        )


# ---------------------------------------------------------------------------
# Decoding of machine sequences -> earliest start schedule (active schedule)
# ---------------------------------------------------------------------------
def decode(p, seqs):
    N, L, M = p.N, p.L, p.M
    proc, r, d, w = p.proc, p.r, p.d, p.w
    mach_succ = [-1] * N
    mach_pred = [-1] * N
    indeg = [0] * N
    for m in range(M):
        prev = -1
        for oid in seqs[m]:
            if prev != -1:
                mach_succ[prev] = oid
                mach_pred[oid] = prev
                indeg[oid] += 1
            k = oid - (oid // L) * L
            if k > 0:
                indeg[oid] += 1
            prev = oid
    s = [0.0] * N
    stack = [oid for oid in range(N) if indeg[oid] == 0]
    done = 0
    while stack:
        oid = stack.pop()
        done += 1
        j = oid // L
        k = oid - j * L
        if k == 0:
            est = r[j]
        else:
            est = s[oid - 1] + proc[oid - 1]
        mp = mach_pred[oid]
        if mp >= 0:
            t2 = s[mp] + proc[mp]
            if t2 > est:
                est = t2
        s[oid] = est
        if k + 1 < L:
            nxt = oid + 1
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                stack.append(nxt)
        ms = mach_succ[oid]
        if ms >= 0:
            indeg[ms] -= 1
            if indeg[ms] == 0:
                stack.append(ms)
    if done < N:
        return float("inf"), None  # cyclic sequences (safety; shouldn't occur)
    obj = 0.0
    for j in range(p.n):
        cid = j * L + (L - 1)
        tard = s[cid] + proc[cid] - d[j]
        if tard > 0.0:
            obj += w[j] * tard
    return obj, s


# ---------------------------------------------------------------------------
# ATC dispatch heuristic -> machine sequences
# ---------------------------------------------------------------------------
def atc_sequences(p, K=2.0, alpha=0.0, rng=None):
    n, M, L = p.n, p.M, p.L
    kcur = [0] * n
    e = list(p.r)
    a = [0.0] * M
    rem = list(p.total_proc)
    seqs = [[] for _ in range(M)]
    pbar = p.avg_p if p.avg_p > 0 else 1.0
    done = 0
    N = p.N
    while done < N:
        bt = float("inf")
        bm = -1
        for j in range(n):
            k = kcur[j]
            if k < L:
                mm = p.mach[j * L + k]
                tj = e[j] if e[j] > a[mm] else a[mm]
                if tj < bt:
                    bt = tj
                    bm = mm
        t = bt
        best_val = -1.0
        best_j = -1
        for j in range(n):
            k = kcur[j]
            if k < L and p.mach[j * L + k] == bm and e[j] <= t:
                oid = j * L + k
                pj = p.proc[oid]
                slack = p.d[j] - t - rem[j]
                if slack < 0.0:
                    slack = 0.0
                val = (p.w[j] / pj) * math.exp(-slack / (K * pbar))
                if alpha > 0.0 and rng is not None:
                    val *= 1.0 + alpha * rng.random()
                if val > best_val:
                    best_val = val
                    best_j = j
        if best_j < 0:  # safety fallback (should not happen)
            for j in range(n):
                k = kcur[j]
                if k < L and p.mach[j * L + k] == bm:
                    best_j = j
                    break
        j = best_j
        k = kcur[j]
        oid = j * L + k
        pj = p.proc[oid]
        st = e[j] if e[j] > a[bm] else a[bm]
        seqs[bm].append(oid)
        kcur[j] += 1
        e[j] = st + pj
        a[bm] = e[j]
        rem[j] -= pj
        done += 1
    return seqs


# ---------------------------------------------------------------------------
# Output construction / validation
# ---------------------------------------------------------------------------
def build_solution(p, s):
    schedule = []
    total = 0.0
    for j in range(p.n):
        ops_out = []
        for k in range(p.L):
            oid = j * p.L + k
            ops_out.append({"machine": p.mach[oid], "start_time": float(s[oid])})
        cid = j * p.L + (p.L - 1)
        comp = float(s[cid] + p.proc[cid])
        tard = comp - p.d[j]
        if tard < 0.0:
            tard = 0.0
        total += p.w[j] * tard
        schedule.append({
            "job_id": p.job_ids[j],
            "completion_time": comp,
            "tardiness": float(tard),
            "operations": ops_out,
        })
    return {"objective_value": float(total), "schedule": schedule}


def validate(p, sol):
    eps = 1e-4
    if abs(sol["objective_value"] - sum(
            s["tardiness"] * p.w[j] for j, s in enumerate(sol["schedule"]))) > 1e-3:
        pass  # objective recomputed anyway; don't reject on rounding
    by_machine = [[] for _ in range(p.M)]
    if len(sol["schedule"]) != p.n:
        return False
    for j, entry in enumerate(sol["schedule"]):
        ops = entry["operations"]
        if len(ops) != p.L:
            return False
        for k, op in enumerate(ops):
            oid = j * p.L + k
            if int(op["machine"]) != p.mach[oid]:
                return False
            st = float(op["start_time"])
            if k == 0 and st < p.r[j] - eps:
                return False
            if k > 0:
                prev = float(ops[k - 1]["start_time"]) + p.proc[oid - 1]
                if st < prev - eps:
                    return False
            by_machine[p.mach[oid]].append((st, st + p.proc[oid]))
        last = float(ops[-1]["start_time"]) + p.proc[j * p.L + (p.L - 1)]
        if abs(last - float(entry["completion_time"])) > 1e-3:
            return False
    for lst in by_machine:
        lst.sort()
        for i in range(len(lst) - 1):
            if lst[i + 1][0] < lst[i][1] - eps:
                return False
    return True


# ---------------------------------------------------------------------------
# Local search on machine sequences (adjacent swaps)
# ---------------------------------------------------------------------------
def local_search(p, seqs, obj, s, deadline, report):
    best_obj, best_s = obj, s
    improved = True
    while improved:
        if time.time() >= deadline:
            break
        improved = False
        for m in range(p.M):
            if time.time() >= deadline:
                return best_obj, best_s
            seq = seqs[m]
            for i in range(len(seq) - 1):
                seq[i], seq[i + 1] = seq[i + 1], seq[i]
                o, s2 = decode(p, seqs)
                if o < best_obj - EPS:
                    best_obj, best_s = o, s2
                    improved = True
                    report(o, s2)
                else:
                    seq[i], seq[i + 1] = seq[i + 1], seq[i]
    return best_obj, best_s


def iterated_local_search(p, seqs0, obj0, s0, deadline, report, rng):
    best_seqs = [seq[:] for seq in seqs0]
    best_obj, best_s = obj0, s0
    while time.time() < deadline - 0.05:
        seqs = [seq[:] for seq in best_seqs]
        n_moves = rng.randint(2, 5)
        for _ in range(n_moves):
            m = rng.randrange(p.M)
            if len(seqs[m]) >= 2:
                i = rng.randrange(len(seqs[m]) - 1)
                seqs[m][i], seqs[m][i + 1] = seqs[m][i + 1], seqs[m][i]
        o, s = decode(p, seqs)
        o, s = local_search(p, seqs, o, s, deadline, report)
        if o < best_obj - EPS:
            best_obj, best_s = o, s
            best_seqs = [seq[:] for seq in seqs]
            report(o, s, force=True)
        if time.time() >= deadline:
            break
    return best_seqs, best_obj, best_s


# ---------------------------------------------------------------------------
# Gurobi MIP (disjunctive big-M formulation), warm-started
# ---------------------------------------------------------------------------
def solve_with_gurobi(p, seqs, warm_s, time_budget, log_incumbent):
    if not GUROBI_AVAILABLE or time_budget < 1.5:
        return None
    try:
        N, n, L = p.N, p.n, p.L
        pairs = []
        for m in range(p.M):
            ids = p.mach_ops_ids[m]
            for i in range(len(ids)):
                a = ids[i]
                for j2 in range(i + 1, len(ids)):
                    pairs.append((a, ids[j2]))
        model = gp.Model("jobshop_wt_tardiness")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(1.0, float(time_budget))

        s = model.addVars(N, lb=0.0, vtype=GRB.CONTINUOUS, name="s")
        C = model.addVars(n, lb=0.0, vtype=GRB.CONTINUOUS, name="C")
        T = model.addVars(n, lb=0.0, vtype=GRB.CONTINUOUS, name="T")
        x = model.addVars(len(pairs), vtype=GRB.BINARY, name="x")

        model.setObjective(
            gp.quicksum(p.w[j] * T[j] for j in range(n)), GRB.MINIMIZE
        )

        for j in range(n):
            base = j * L
            model.addConstr(s[base] >= p.r[j])
            for k in range(1, L):
                model.addConstr(s[base + k] >= s[base + k - 1] + p.proc[base + k - 1])
            last = base + (L - 1)
            model.addConstr(C[j] == s[last] + p.proc[last])
            model.addConstr(T[j] >= C[j] - p.d[j])

        bigM = p.bigM
        for idx, (a, b) in enumerate(pairs):
            model.addConstr(s[a] + p.proc[a] <= s[b] + bigM * (1 - x[idx]))
            model.addConstr(s[b] + p.proc[b] <= s[a] + bigM * x[idx])

        # warm start
        pos = {}
        for m in range(p.M):
            for i, oid in enumerate(seqs[m]):
                pos[oid] = i
        for i in range(N):
            s[i].Start = float(warm_s[i])
        for j in range(n):
            last = j * L + (L - 1)
            cj = float(warm_s[last] + p.proc[last])
            C[j].Start = cj
            T[j].Start = max(cj - p.d[j], 0.0)
        for idx, (a, b) in enumerate(pairs):
            x[idx].Start = 1.0 if pos[a] < pos[b] else 0.0
        model.update()

        s_list = [s[i] for i in range(N)]

        def cb(mod, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    sv = list(mod.cbGetSolution(s_list))
                    sol = build_solution(p, sv)
                    log_incumbent(sol)
                except Exception:
                    pass

        model.optimize(cb)

        if model.SolCount is not None and model.SolCount > 0:
            sv = [v.X for v in s_list]
            return model.ObjVal, sv
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, required=True)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t0 = time.time()
    TL = max(1, int(args.time_limit))
    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if (args.log_path and SolutionLogger is not None)
        else None
    )

    with open(args.instance_path) as f:
        inst = json.load(f)
    p = Problem(inst)
    rng = random.Random(0)

    # logging helper (throttled for the heuristic, forced on demand)
    log_state = {"last": 0.0, "last_obj": float("inf")}

    def report(obj, s, force=False):
        if logger is None or s is None or not math.isfinite(obj):
            return
        now = time.time()
        if obj < log_state["last_obj"] - 1e-7 and (force or now - log_state["last"] > 0.2):
            log_state["last_obj"] = obj
            log_state["last"] = now
            sol = build_solution(p, s)
            logger.log_solution(sol["objective_value"], sol)

    def log_incumbent_dict(sol):
        if logger is None:
            return
        log_state["last_obj"] = sol["objective_value"]
        log_state["last"] = time.time()
        logger.log_solution(sol["objective_value"], sol)

    # ---- initial heuristic solutions (ATC with several K values) ----
    candidates = [
        atc_sequences(p, K=1.0, alpha=0.0, rng=rng),
        atc_sequences(p, K=2.0, alpha=0.0, rng=rng),
        atc_sequences(p, K=3.0, alpha=0.0, rng=rng),
    ]
    for _ in range(3):
        candidates.append(
            atc_sequences(p, K=rng.uniform(1.2, 3.2), alpha=0.15, rng=rng)
        )
    best_obj, best_s, best_seqs = float("inf"), None, None
    for seqs in candidates:
        o, s = decode(p, seqs)
        if o < best_obj:
            best_obj, best_s, best_seqs = o, s, [q[:] for q in seqs]
    report(best_obj, best_s, force=True)

    # ---- time allocation ----
    mip_planned = (
        GUROBI_AVAILABLE and p.est_pairs <= 80000 and p.N <= 3000
    )
    if mip_planned:
        heur_budget = max(0.5, min(0.25 * TL, 25.0))
    else:
        heur_budget = max(0.5, TL - 0.8)
    heur_deadline = t0 + heur_budget

    # ---- local search then iterated local search ----
    cur_obj, cur_s = local_search(
        p, best_seqs, best_obj, best_s, heur_deadline, report
    )
    if cur_obj < best_obj - EPS:
        best_obj, best_s = cur_obj, cur_s
    best_seqs, best_obj, best_s = iterated_local_search(
        p, best_seqs, best_obj, best_s, heur_deadline, report, rng
    )
    report(best_obj, best_s, force=True)

    inc_obj, inc_s = best_obj, best_s

    # ---- Gurobi MIP warm-started from the heuristic incumbent ----
    remaining = TL - (time.time() - t0) - 0.8
    if mip_planned and remaining > 1.5:
        result = solve_with_gurobi(
            p, best_seqs, best_s, remaining, log_incumbent_dict
        )
        if result is not None:
            mip_obj, mip_s = result
            if mip_obj < inc_obj - 1e-7:
                inc_obj, inc_s = mip_obj, mip_s

    # ---- finalize ----
    sol = build_solution(p, inc_s)
    if not validate(p, sol):
        sol = build_solution(p, best_s)  # fall back to verified heuristic schedule
    if logger is not None:
        logger.log_solution(sol["objective_value"], sol)

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()