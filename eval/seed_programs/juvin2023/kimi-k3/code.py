import argparse
import json
import math
import random
import time
from bisect import insort

from solution_logger import SolutionLogger

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAVE_GUROBI = True
except Exception:
    gp = None
    GRB = None
    HAVE_GUROBI = False

PAIR_LIMIT = 100000  # switch to heuristic path if disjunctive model would be too large


# ----------------------------- Problem data -----------------------------
class Problem:
    def __init__(self, inst):
        self.num_jobs = int(inst["num_jobs"])
        self.num_machines = int(inst["num_machines"])
        jobs = sorted(inst["jobs"], key=lambda j: j["job_id"])
        self.job_ids = []
        self.job_pos = {}
        self.job_base = []
        self.job_count = []
        ops = []
        for g, job in enumerate(jobs):
            jid = job["job_id"]
            self.job_ids.append(jid)
            self.job_pos[jid] = g
            ops_in = sorted(job["operations"], key=lambda o: o["operation_id"])
            self.job_base.append(len(ops))
            self.job_count.append(len(ops_in))
            for o in ops_in:
                elig = sorted(int(m) for m in o["eligible_machines"])
                pt = {int(k): int(v) for k, v in o["processing_times"].items()}
                ops.append((jid, g, o["operation_id"], elig, pt))
        self.ops = ops
        self.N = len(ops)
        self.job_id_of = [o[0] for o in ops]
        self.jidx_of = [o[1] for o in ops]
        self.op_id_of = [o[2] for o in ops]
        self.elig = [o[3] for o in ops]
        self.pt = [o[4] for o in ops]
        self.minp = [min(self.pt[i].values()) for i in range(self.N)]
        self.maxp = [max(self.pt[i].values()) for i in range(self.N)]
        self.H = max(1, sum(self.maxp))
        self.last_op = [self.job_base[g] + self.job_count[g] - 1
                        for g in range(self.num_jobs)]
        self.by_machine = [[] for _ in range(self.num_machines)]
        for i in range(self.N):
            for m in self.elig[i]:
                self.by_machine[m].append(i)


# ----------------------------- Scheduling helpers -----------------------------
def earliest_slot(busy, ready, p):
    """Earliest start >= ready such that [t, t+p) avoids busy intervals."""
    t = ready
    for (s, e) in busy:
        if s >= t + p:
            break
        if e > t:
            t = e
    return t


def decode(prob, assign, seq):
    busy = [[] for _ in range(prob.num_machines)]
    ready = [1] * prob.num_jobs
    ptr = [0] * prob.num_jobs
    starts = [0] * prob.N
    ends = [0] * prob.N
    obj = 0
    for j in seq:
        i = prob.job_base[j] + ptr[j]
        ptr[j] += 1
        m = assign[i]
        p = prob.pt[i][m]
        t = earliest_slot(busy[m], ready[j], p)
        insort(busy[m], (t, t + p))
        starts[i] = t
        ends[i] = t + p
        ready[j] = t + p
        if t + p > obj:
            obj = t + p
    return obj, starts, ends


def greedy(prob):
    busy = [[] for _ in range(prob.num_machines)]
    ready = [1] * prob.num_jobs
    ptr = [0] * prob.num_jobs
    assign = [None] * prob.N
    seq = []
    starts = [0] * prob.N
    ends = [0] * prob.N
    obj = 0
    remaining = prob.N
    while remaining > 0:
        best = None
        for j in range(prob.num_jobs):
            if ptr[j] >= prob.job_count[j]:
                continue
            i = prob.job_base[j] + ptr[j]
            r = ready[j]
            for m in prob.elig[i]:
                p = prob.pt[i][m]
                t = earliest_slot(busy[m], r, p)
                cand = (t + p, t, m, j, i, p)
                if best is None or cand < best[0]:
                    best = (cand, (t + p, t, m, j, i, p))
        if best is None:
            break
        e, t, m, j, i, p = best[1]
        assign[i] = m
        starts[i] = t
        ends[i] = e
        insort(busy[m], (t, e))
        ready[j] = e
        ptr[j] += 1
        seq.append(j)
        if e > obj:
            obj = e
        remaining -= 1
    return assign, seq, obj, starts, ends


def build_solution(prob, assign, starts, ends):
    assignment = {}
    schedule = {}
    obj = 0.0
    for i in range(prob.N):
        jid = prob.job_id_of[i]
        oid = prob.op_id_of[i]
        m = int(assign[i])
        s = int(starts[i])
        e = int(ends[i])
        assignment["(%d,%d)" % (jid, oid)] = m
        schedule["(%d,%d)" % (jid, oid)] = list(range(s, e))
        if e > obj:
            obj = float(e)
    return {
        "objective_value": float(obj),
        "assignment": assignment,
        "schedule": schedule,
    }


# ----------------------------- MIP path -----------------------------
def solve_mip(prob, greedy_res, deadline, logger):
    g_assign, g_seq, g_obj, g_starts, g_ends = greedy_res
    model = gp.Model("FJSP")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.0, deadline - time.time())
    model.Params.OutputFlag = 0

    N = prob.N
    H = prob.H
    M = H + 1

    lb_s = [1] * N
    for i in range(N):
        g = prob.jidx_of[i]
        pos = prob.op_id_of[i]
        # operation_id equals position within job (sorted earlier)
        base = prob.job_base[g]
        acc = 0
        for k in range(pos):
            acc += prob.minp[base + k]
        lb_s[i] = 1 + acc
    ub_c = H + 1

    y_idx = [(i, m) for i in range(N) for m in prob.elig[i]]
    y = model.addVars(y_idx, vtype=GRB.BINARY, name="y")
    s = model.addVars(range(N), vtype=GRB.INTEGER, lb=lb_s, ub=H, name="s")
    c = model.addVars(range(N), vtype=GRB.INTEGER,
                      lb=[lb_s[i] + prob.minp[i] for i in range(N)],
                      ub=ub_c, name="c")
    cmax = model.addVar(vtype=GRB.INTEGER, lb=1, ub=ub_c, name="cmax")

    for i in range(N):
        model.addConstr(c[i] == s[i] + gp.quicksum(prob.pt[i][m] * y[(i, m)]
                                                   for m in prob.elig[i]))
        model.addConstr(gp.quicksum(y[(i, m)] for m in prob.elig[i]) == 1)

    # precedence between consecutive operations of the same job
    for g in range(prob.num_jobs):
        base = prob.job_base[g]
        for k in range(prob.job_count[g] - 1):
            model.addConstr(s[base + k + 1] >= c[base + k])

    for g in range(prob.num_jobs):
        model.addConstr(cmax >= c[prob.last_op[g]])

    # disjunctive machine constraints
    pair_idx = []
    for m in range(prob.num_machines):
        ops_m = prob.by_machine[m]
        for a in range(len(ops_m)):
            for b in range(a + 1, len(ops_m)):
                i, k = ops_m[a], ops_m[b]
                if prob.jidx_of[i] == prob.jidx_of[k]:
                    continue
                pair_idx.append((i, k, m))
    bvars = model.addVars(pair_idx, vtype=GRB.BINARY, name="b")
    for (i, k, m) in pair_idx:
        bb = bvars[(i, k, m)]
        yi = y[(i, m)]
        yk = y[(k, m)]
        model.addConstr(s[k] >= c[i] - M * (3 - bb - yi - yk))
        model.addConstr(s[i] >= c[k] - M * (2 + bb - yi - yk))

    model.setObjective(cmax, GRB.MINIMIZE)

    # warm start with greedy solution
    for (i, m) in y_idx:
        y[(i, m)].Start = 1.0 if g_assign[i] == m else 0.0
    for i in range(N):
        s[i].Start = float(g_starts[i])

    y_flat = [y[(i, m)] for (i, m) in y_idx]
    y_pos = {}
    for pos, (i, m) in enumerate(y_idx):
        y_pos.setdefault(i, []).append((m, pos))
    s_list = [s[i] for i in range(N)]
    all_vars = y_flat + s_list

    state = {"best": math.inf}

    def extract(vals):
        assign = [None] * N
        for i in range(N):
            bestm = prob.elig[i][0]
            bestv = -1.0
            for (m, pos) in y_pos[i]:
                if vals[pos] > bestv:
                    bestv = vals[pos]
                    bestm = m
            assign[i] = bestm
        starts = [int(round(vals[len(y_flat) + i])) for i in range(N)]
        ends = [starts[i] + prob.pt[i][assign[i]] for i in range(N)]
        return build_solution(prob, assign, starts, ends)

    def callback(model, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < state["best"] - 1e-6:
                vals = model.cbGetSolution(all_vars)
                sol = extract(vals)
                logger.log_solution(sol["objective_value"], sol)
                state["best"] = min(state["best"], sol["objective_value"])

    model.optimize(callback if logger is not None else None)

    if model.SolCount > 0:
        vals = model.getAttr("X", all_vars)
        sol = extract(vals)
        return sol
    return None


# ----------------------------- Heuristic (SA) path -----------------------------
def choose_and_run(prob, budget, logger):
    t0 = time.time()
    deadline = t0 + budget

    greedy_res = greedy(prob)
    g_assign, g_seq, g_obj, g_starts, g_ends = greedy_res
    best = build_solution(prob, g_assign, g_starts, g_ends)
    if logger:
        logger.log_solution(best["objective_value"], best)

    pair_count = sum(len(l) * (len(l) - 1) // 2 for l in prob.by_machine)
    use_mip = HAVE_GUROBI and pair_count <= PAIR_LIMIT and prob.N <= 4000

    if use_mip:
        try:
            sol = solve_mip(prob, greedy_res, deadline, logger)
            if sol is not None and sol["objective_value"] < best["objective_value"]:
                best = sol
        except Exception:
            pass
        return best

    # Simulated annealing fallback for large instances
    cur_assign = list(g_assign)
    cur_seq = list(g_seq)
    best_assign = list(g_assign)
    best_seq = list(g_seq)
    best_obj = best["objective_value"]
    obj, _, _ = decode(prob, cur_assign, cur_seq)
    cur_obj = obj
    multi = [i for i in range(prob.N) if len(prob.elig[i]) > 1]
    rnd = random.Random(0)
    T0 = max(1.0, best_obj * 0.05)
    it = 0
    while time.time() < deadline:
        frac = (time.time() - t0) / max(1.0, budget)
        T = T0 * max(0.0, 1.0 - frac)
        if multi and rnd.random() < 0.5:
            i = multi[rnd.randrange(len(multi))]
            old = cur_assign[i]
            choices = [m for m in prob.elig[i] if m != old]
            new = choices[rnd.randrange(len(choices))]
            cur_assign[i] = new
            obj, s_arr, e_arr = decode(prob, cur_assign, cur_seq)
            delta = obj - cur_obj
            accept = delta <= 0 or (T > 0 and rnd.random() < math.exp(-delta / T))
            if accept:
                cur_obj = obj
                if obj < best_obj:
                    best_obj = obj
                    best_assign = list(cur_assign)
                    best_seq = list(cur_seq)
                    sol = build_solution(prob, best_assign, s_arr, e_arr)
                    if logger:
                        logger.log_solution(sol["objective_value"], sol)
            else:
                cur_assign[i] = old
        else:
            a = rnd.randrange(prob.N)
            b = rnd.randrange(prob.N)
            if a == b:
                continue
            cur_seq[a], cur_seq[b] = cur_seq[b], cur_seq[a]
            obj, s_arr, e_arr = decode(prob, cur_assign, cur_seq)
            delta = obj - cur_obj
            accept = delta <= 0 or (T > 0 and rnd.random() < math.exp(-delta / T))
            if accept:
                cur_obj = obj
                if obj < best_obj:
                    best_obj = obj
                    best_assign = list(cur_assign)
                    best_seq = list(cur_seq)
                    sol = build_solution(prob, best_assign, s_arr, e_arr)
                    if logger:
                        logger.log_solution(sol["objective_value"], sol)
            else:
                cur_seq[a], cur_seq[b] = cur_seq[b], cur_seq[a]
        it += 1

    obj, s_arr, e_arr = decode(prob, best_assign, best_seq)
    sol = build_solution(prob, best_assign, s_arr, e_arr)
    if sol["objective_value"] < best["objective_value"]:
        best = sol
    return best


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance_path", required=True)
    p.add_argument("--solution_path", required=True)
    p.add_argument("--time_limit", type=int, required=True)
    p.add_argument("--log_path", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        inst = json.load(f)
    prob = Problem(inst)
    budget = max(1, int(args.time_limit))

    try:
        best = choose_and_run(prob, budget, logger)
    except Exception:
        # ultimate fallback: greedy schedule
        g_assign, g_seq, g_obj, g_starts, g_ends = greedy(prob)
        best = build_solution(prob, g_assign, g_starts, g_ends)

    with open(args.solution_path, "w") as f:
        json.dump(best, f)


if __name__ == "__main__":
    main()