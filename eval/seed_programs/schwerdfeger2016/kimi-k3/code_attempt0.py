import argparse
import bisect
import heapq
import json
import math
import random
import time

BIG = 1 << 62


# ---------------------------------------------------------------------------
# Objective / solution helpers
# ---------------------------------------------------------------------------

def normalized_objective(loads):
    """sqrt(sum_i (C_i - avg)^2) / avg."""
    m = len(loads)
    total = 0
    for l in loads:
        total += l
    avg = total / m
    var = 0.0
    for l in loads:
        d = l - avg
        var += d * d
    return math.sqrt(var) / avg


def build_solution(assign_idx, loads):
    """Project internal solution onto the required output schema."""
    obj = normalized_objective(loads)
    m = len(loads)
    sol = {
        "objective_value": obj,
        "assignment": {str(i): sorted(assign_idx[i]) for i in range(m)},
        "machine_completion_times": {str(i): int(loads[i]) for i in range(m)},
    }
    return obj, sol


def to_index_lists(a_tl):
    return [[j for (_, j) in lst] for lst in a_tl]


def sum_sq(loads):
    s = 0
    for l in loads:
        s += l * l
    return s


# ---------------------------------------------------------------------------
# Construction heuristic: Longest Processing Time first
# ---------------------------------------------------------------------------

def lpt(p, m):
    n = len(p)
    assign = [[] for _ in range(m)]
    loads = [0] * m
    heap = [(0, i) for i in range(m)]
    heapq.heapify(heap)
    order = sorted(range(n), key=lambda j: -p[j])
    for j in order:
        ld, i = heapq.heappop(heap)
        assign[i].append(j)
        ld += p[j]
        loads[i] = ld
        heapq.heappush(heap, (ld, i))
    # internal representation: per-machine ascending-sorted list of (p_j, j)
    a_tl = [[(p[j], j) for j in reversed(assign[i])] for i in range(m)]
    return a_tl, loads


# ---------------------------------------------------------------------------
# Local improvement: pairwise balancing moves and swaps
# Moving job of size p from load A to load B (A>B) changes the sum of
# squares by 2p(p + B - A) < 0 iff p < A - B; best p is closest to (A-B)/2.
# Swapping jobs (p_j on A, p_k on B, d = p_j - p_k) changes it by
# 2d(d + B - A) < 0 iff 0 < d < A - B; best d is closest to (A-B)/2.
# ---------------------------------------------------------------------------

def try_move(a_tl, loads, H, L, diff):
    lst = a_tl[H]
    limit = bisect.bisect_left(lst, (diff, -1))  # candidates have p < diff
    if limit == 0:
        return False
    target = diff * 0.5
    pos = bisect.bisect_left(lst, (target, -1))
    best_idx = -1
    best_dev = None
    for k in (pos - 1, pos, pos + 1):
        if 0 <= k < limit:
            dev = abs(lst[k][0] - target)
            if best_dev is None or dev < best_dev:
                best_dev = dev
                best_idx = k
    if best_idx < 0:
        return False
    pj, j = lst.pop(best_idx)
    bisect.insort(a_tl[L], (pj, j))
    loads[H] -= pj
    loads[L] += pj
    return True


def try_swap(a_tl, loads, H, L, diff):
    lstH = a_tl[H]
    lstL = a_tl[L]
    if not lstH or not lstL:
        return False
    min_l = lstL[0][0]
    target = diff * 0.5
    best = None
    best_dev = None
    for idx_j in range(len(lstH)):
        pj, j = lstH[idx_j]
        if pj <= min_l:
            continue
        lo = bisect.bisect_right(lstL, (pj - diff, BIG))  # pk > pj - diff
        hi = bisect.bisect_left(lstL, (pj, -1))           # pk < pj
        if lo >= hi:
            continue
        t = pj - target
        pos = bisect.bisect_left(lstL, (t, -1))
        for kk in (pos - 1, pos, pos + 1):
            if lo <= kk < hi:
                pk, k = lstL[kk]
                d = pj - pk
                dev = d - target
                if dev < 0:
                    dev = -dev
                if best_dev is None or dev < best_dev:
                    best_dev = dev
                    best = (idx_j, kk, pj, j, pk, k)
    if best is None:
        return False
    idx_j, kk, pj, j, pk, k = best
    lstH.pop(idx_j)
    lstL.pop(kk)
    bisect.insort(lstH, (pk, k))
    bisect.insort(lstL, (pj, j))
    d = pj - pk
    loads[H] -= d
    loads[L] += d
    return True


def descent(a_tl, loads, p, m, deadline):
    T = m if m < 64 else 64
    improved = True
    while improved:
        if time.time() > deadline:
            return
        improved = False
        order = sorted(range(m), key=lambda i: loads[i])
        highs = order[m - T:]
        lows = order[:T]
        for H in reversed(highs):
            if time.time() > deadline:
                return
            for L in lows:
                if loads[H] <= loads[L]:
                    break
                diff = loads[H] - loads[L]
                if try_move(a_tl, loads, H, L, diff):
                    improved = True
                elif try_swap(a_tl, loads, H, L, diff):
                    improved = True


# ---------------------------------------------------------------------------
# Iterated local search (perturbation + descent) for large instances
# ---------------------------------------------------------------------------

def perturb(a_tl, loads, rng, strength):
    m = len(a_tl)
    for _ in range(strength):
        a = rng.randrange(m)
        tries = 0
        while not a_tl[a] and tries < 10:
            a = rng.randrange(m)
            tries += 1
        if not a_tl[a]:
            continue
        b = rng.randrange(m - 1)
        if b >= a:
            b += 1
        idx = rng.randrange(len(a_tl[a]))
        pj, j = a_tl[a].pop(idx)
        bisect.insort(a_tl[b], (pj, j))
        loads[a] -= pj
        loads[b] += pj


def ils_phase(best, best_loads, best_ss, p, m, n, deadline, logger):
    rng = random.Random(0)
    max_strength = max(3, min(30, n // 10 + 1))
    while time.time() < deadline:
        cur = [lst[:] for lst in best]
        cur_loads = best_loads[:]
        perturb(cur, cur_loads, rng, rng.randint(2, max_strength))
        descent(cur, cur_loads, p, m, deadline)
        ss = sum_sq(cur_loads)
        if ss < best_ss:
            best_ss = ss
            best = cur
            best_loads = cur_loads
            if logger:
                try:
                    obj, sol = build_solution(to_index_lists(best), best_loads)
                    logger.log_solution(obj, sol)
                except Exception:
                    pass
    return best, best_loads, best_ss


# ---------------------------------------------------------------------------
# Exact MIQP phase with Gurobi (used for small/medium instances)
# ---------------------------------------------------------------------------

def gurobi_phase(p, m, n, init_tl, init_loads, init_ss, time_budget, logger):
    import gurobipy as gp
    from gurobipy import GRB

    order = sorted(range(m, )) if False else sorted(range(m), key=lambda i: init_loads[i])
    init_idx = [[j for (_, j) in lst] for lst in init_tl]

    model = gp.Model("parallel_machine_sum_sq")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.LogToConsole = 0
    model.Params.TimeLimit = max(time_budget, 0.1)

    x = model.addVars(m, n, vtype=GRB.BINARY, name="x")
    C = model.addVars(m, lb=0.0, name="C")
    model.update()

    for j in range(n):
        model.addConstr(gp.quicksum(x[i, j] for i in range(m)) == 1)
    for i in range(m):
        model.addConstr(C[i] == gp.quicksum(p[j] * x[i, j] for j in range(n)))
    # symmetry breaking: machines are identical, order loads
    for i in range(m - 1):
        model.addConstr(C[i] <= C[i + 1])

    model.setObjective(gp.quicksum(C[i] * C[i] for i in range(m)), GRB.MINIMIZE)

    # complete warm start from the heuristic incumbent (loads sorted ascending)
    for new_i in range(m):
        old_i = order[new_i]
        js = set(init_idx[old_i])
        for j in range(n):
            x[new_i, j].Start = 1.0 if j in js else 0.0
        C[new_i].Start = float(init_loads[old_i])

    state = {"ss": init_ss, "assign": None, "loads": None, "optimal": False}
    var_list = [x[i, j] for i in range(m) for j in range(n)]

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            except Exception:
                return
            if obj < state["ss"] - 1e-6:
                vals = model.cbGetSolution(var_list)
                assign = [[] for _ in range(m)]
                loads = [0] * m
                idx = 0
                for i in range(m):
                    s = 0
                    for j in range(n):
                        if vals[idx] > 0.5:
                            assign[i].append(j)
                            s += p[j]
                        idx += 1
                    loads[i] = s
                ss = sum_sq(loads)
                if ss < state["ss"]:
                    state["ss"] = ss
                    state["assign"] = assign
                    state["loads"] = loads
                    if logger:
                        try:
                            nobj, sol = build_solution(assign, loads)
                            logger.log_solution(nobj, sol)
                        except Exception:
                            pass

    model.optimize(cb)

    if model.Status == GRB.OPTIMAL:
        state["optimal"] = True

    improved = None
    if state["assign"] is not None:
        a_tl = [sorted((p[j], j) for j in state["assign"][i]) for i in range(m)]
        improved = (a_tl, state["loads"], state["ss"])
    return improved, state["optimal"]


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    with open(args.instance_path) as f:
        inst = json.load(f)
    m = int(inst["num_machines"])
    n = int(inst["num_jobs"])
    p = [int(v) for v in inst["processing_times"]]

    start = time.time()
    deadline = start + max(0.2, float(args.time_limit) - 0.3)

    # Phase 1: fast construction + local improvement
    a_tl, loads = lpt(p, m)
    try:
        descent(a_tl, loads, p, m, deadline)
    except Exception:
        pass

    best = [lst[:] for lst in a_tl]
    best_loads = loads[:]
    best_ss = sum_sq(best_loads)

    if logger:
        try:
            obj, sol = build_solution(to_index_lists(best), best_loads)
            logger.log_solution(obj, sol)
        except Exception:
            pass

    # Phase 2: exact MIQP polishing for small/medium instances
    remaining = deadline - time.time()
    if n * m <= 8000 and remaining > 1.5:
        try:
            improved, optimal = gurobi_phase(
                p, m, n, best, best_loads, best_ss, remaining - 0.5, logger
            )
            if improved is not None:
                best, best_loads, best_ss = improved
            if optimal:
                obj, sol = build_solution(to_index_lists(best), best_loads)
                with open(args.solution_path, "w") as f:
                    json.dump(sol, f)
                return
        except Exception:
            pass

    # Phase 3: use any remaining time on iterated local search
    try:
        best, best_loads, best_ss = ils_phase(
            best, best_loads, best_ss, p, m, n, deadline, logger
        )
    except Exception:
        pass

    obj, sol = build_solution(to_index_lists(best), best_loads)
    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()