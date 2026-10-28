import argparse
import bisect
import json
import os
import sys
import time
from collections import defaultdict

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None

from solution_logger import SolutionLogger

sys.setrecursionlimit(100000)

MAX_PATTERNS = 200000
MAX_ASSIGN_VARS = 50000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance_path", required=True)
    p.add_argument("--solution_path", required=True)
    p.add_argument("--time_limit", required=True, type=int)
    p.add_argument("--log_path", required=False, default=None)
    return p.parse_args()


def read_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    c = int(data["c"])
    if c <= 0:
        raise ValueError("Bin capacity must be positive.")
    ws = data["weights"]
    ds = data["demands"]
    if len(ws) != len(ds):
        raise ValueError("weights and demands must have the same length.")
    agg = {}
    for w, d in zip(ws, ds):
        w = int(w)
        d = int(d)
        if d < 0:
            raise ValueError("Demands must be nonnegative.")
        if w <= 0 or w >= c:
            raise ValueError("Every width must be strictly between 0 and c.")
        if d:
            agg[w] = agg.get(w, 0) + d
    items = sorted(agg.items(), key=lambda x: x[0], reverse=True)
    weights = [w for w, _ in items]
    demands = [d for _, d in items]
    return c, weights, demands


def lower_bound(c, weights, demands):
    total_weight = sum(w * d for w, d in zip(weights, demands))
    total_items = sum(demands)
    if total_items == 0:
        return 0, 0, 0
    lb = (total_weight + c - 1) // c
    # weights are sorted decreasing. Any r+1 items of a prefix exceed capacity.
    pref = 0
    for w, d in zip(weights, demands):
        r = c // w
        if r <= 0:
            continue
        pref += d
        b = (pref + r - 1) // r
        if b > lb:
            lb = b
    return int(lb), int(total_weight), int(total_items)


def best_fit_decreasing(c, weights, demands):
    bins = []
    rem_caps = []  # sorted remaining capacities; exactly one entry per non-full bin
    buckets = defaultdict(list)  # remaining capacity -> list of bin indices

    for w, d in zip(weights, demands):
        for _ in range(d):
            pos = bisect.bisect_left(rem_caps, w)
            if pos == len(rem_caps):
                idx = len(bins)
                bins.append([w])
                rem = c - w
                if rem > 0:
                    ins = bisect.bisect_left(rem_caps, rem)
                    rem_caps.insert(ins, rem)
                    buckets[rem].append(idx)
            else:
                cap = rem_caps[pos]
                bucket = buckets[cap]
                idx = bucket.pop()
                # exactly one entry per bin must be removed from the sorted list
                rem_caps.pop(pos)
                if not bucket:
                    del buckets[cap]
                bins[idx].append(w)
                rem = cap - w
                if rem > 0:
                    ins = bisect.bisect_left(rem_caps, rem)
                    rem_caps.insert(ins, rem)
                    buckets[rem].append(idx)
    return bins


def bins_are_valid(bins, c, weights, demands):
    allowed = set(weights)
    counts = {w: 0 for w in weights}
    total = 0
    for b in bins:
        if not b:
            return False
        s = 0
        for w in b:
            if w not in allowed:
                return False
            s += w
            if s > c:
                return False
            counts[w] += 1
            total += 1
    if total != sum(demands):
        return False
    return all(counts[w] == d for w, d in zip(weights, demands))


def solution_dict(bins):
    nb = len(bins)
    return {
        "objective_value": float(nb),
        "num_bins": int(nb),
        "bin_assignments": [[int(x) for x in b] for b in bins],
    }


def eliminate_bins(bins, c, lb, deadline, update_best):
    bins = [list(b) for b in bins]
    while len(bins) > lb and time.monotonic() < deadline:
        loads = [sum(b) for b in bins]
        total_load = sum(loads)
        order = sorted(range(len(bins)), key=lambda i: (len(bins[i]), loads[i]))
        candidate_limit = len(order) if len(order) <= 300 else 200
        improved = False

        for src in order[:candidate_limit]:
            if time.monotonic() >= deadline:
                break
            load_src = loads[src]
            free_other = (len(bins) - 1) * c - (total_load - load_src)
            if free_other < load_src:
                continue

            caps = []
            buckets = defaultdict(list)
            max_cap = 0
            for j in range(len(bins)):
                if j == src:
                    continue
                rem = c - loads[j]
                if rem > 0:
                    caps.append(rem)
                    buckets[rem].append(j)
                    if rem > max_cap:
                        max_cap = rem

            items = sorted(bins[src], reverse=True)
            if items and items[0] > max_cap:
                continue
            caps.sort()

            placements = []
            ok = True
            for w in items:
                pos = bisect.bisect_left(caps, w)
                if pos == len(caps):
                    ok = False
                    break
                cap = caps[pos]
                bucket = buckets[cap]
                idx = bucket.pop()
                placements.append((idx, w))
                # maintain one entry per bin in the sorted capacity list
                caps.pop(pos)
                if not bucket:
                    del buckets[cap]
                rem = cap - w
                if rem > 0:
                    ins = bisect.bisect_left(caps, rem)
                    caps.insert(ins, rem)
                    buckets[rem].append(idx)

            if not ok:
                continue

            new_bins = [list(bins[j]) for j in range(len(bins)) if j != src]
            for old_idx, w in placements:
                new_idx = old_idx if old_idx < src else old_idx - 1
                new_bins[new_idx].append(w)

            if update_best(new_bins):
                bins = new_bins
                improved = True
                break

        if not improved:
            break
    return bins


def configure_model(model, lb, remaining_seconds):
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0
    model.Params.MIPGapAbs = 0.999999
    model.Params.TimeLimit = max(1, int(remaining_seconds))
    if lb > 0:
        model.Params.BestObjStop = float(lb)


def estimate_pattern_count(c, weights, demands, cap):
    est = 1
    for w, d in zip(weights, demands):
        est *= min(c // w, d) + 1
        if est > cap:
            return est
    return est


def generate_patterns(c, weights, demands, deadline, cap=MAX_PATTERNS):
    m = len(weights)
    limits = [min(c // w, d) for w, d in zip(weights, demands)]
    if not limits or any(lim <= 0 for lim in limits):
        return None
    patterns = []
    cur = [0] * m
    nodes = 0

    def rec(i, rem):
        nonlocal nodes
        nodes += 1
        if (nodes & 1023) == 0 and time.monotonic() > deadline:
            return False
        if i == m:
            if any(cur):
                patterns.append(tuple(cur))
                if len(patterns) > cap:
                    return False
            return True
        w = weights[i]
        maxk = rem // w
        if maxk > limits[i]:
            maxk = limits[i]
        for k in range(maxk + 1):
            cur[i] = k
            if not rec(i + 1, rem - k * w):
                cur[i] = 0
                return False
        cur[i] = 0
        return True

    if not rec(0, c):
        return None
    return patterns


def solve_with_patterns(c, weights, demands, lb, reference_bins, deadline, update_best):
    if gp is None:
        return False
    if deadline - time.monotonic() <= 1.0:
        return False
    patterns = generate_patterns(c, weights, demands, deadline)
    if not patterns:
        return False

    try:
        m = len(weights)
        p_count = len(patterns)
        ref_count = max(1, len(reference_bins))

        model = gp.Model("bin_packing_patterns")
        configure_model(model, lb, deadline - time.monotonic())

        x = model.addVars(p_count, vtype=GRB.INTEGER, lb=0, ub=ref_count, name="x")
        x_list = [x[i] for i in range(p_count)]
        model.setObjective(gp.quicksum(x_list), GRB.MINIMIZE)
        model.addConstr(gp.quicksum(x_list) >= lb, name="lower_bound")

        cols = [[] for _ in range(m)]
        for i, p in enumerate(patterns):
            for t, a in enumerate(p):
                if a:
                    cols[t].append((i, int(a)))
        for t in range(m):
            model.addConstr(
                gp.quicksum(a * x_list[i] for i, a in cols[t]) == demands[t],
                name="demand_%d" % t,
            )

        # MIP start from the incumbent heuristic packing.
        if reference_bins:
            index_of = {p: i for i, p in enumerate(patterns)}
            width_index = {w: t for t, w in enumerate(weights)}
            starts = defaultdict(int)
            for b in reference_bins:
                cnt = [0] * m
                for w in b:
                    cnt[width_index[w]] += 1
                i = index_of.get(tuple(cnt))
                if i is not None:
                    starts[i] += 1
            for i, k in starts.items():
                x_list[i].Start = k

        def extract(vals):
            cand = []
            for i, v in enumerate(vals):
                k = int(round(v))
                if k <= 0:
                    continue
                p = patterns[i]
                one_bin = []
                for t, a in enumerate(p):
                    if a:
                        one_bin.extend([weights[t]] * a)
                if not one_bin:
                    continue
                for _ in range(k):
                    cand.append(list(one_bin))
            return cand

        def cb(model, where):
            if where == GRB.Callback.MIPSOL:
                if time.monotonic() > deadline:
                    model.terminate()
                    return
                vals = model.cbGetSolution(x_list)
                update_best(extract(vals))
            elif where == GRB.Callback.MIP:
                if time.monotonic() > deadline:
                    model.terminate()

        model.optimize(cb)
        if model.SolCount > 0:
            vals = [v.X for v in x_list]
            update_best(extract(vals))
        return True
    except Exception:
        return False


def solve_with_assignment(c, weights, demands, lb, reference_bins, deadline, update_best):
    if gp is None:
        return False
    if deadline - time.monotonic() <= 1.0:
        return False

    ref = sorted((list(b) for b in reference_bins), key=lambda b: -sum(b))
    b_count = len(ref)
    m = len(weights)
    if b_count == 0 or b_count * (m + 1) > MAX_ASSIGN_VARS:
        return False

    try:
        model = gp.Model("bin_packing_assignment")
        configure_model(model, lb, deadline - time.monotonic())

        u = model.addVars(b_count, vtype=GRB.BINARY, name="u")
        y = model.addVars(b_count, m, vtype=GRB.INTEGER, lb=0, name="y")
        max_count = [min(demands[t], c // weights[t]) for t in range(m)]
        for t in range(m):
            mc = max_count[t]
            for j in range(b_count):
                y[j, t].ub = mc

        u_list = [u[j] for j in range(b_count)]
        y_list = [y[j, t] for j in range(b_count) for t in range(m)]
        loads = [
            gp.quicksum(weights[t] * y[j, t] for t in range(m))
            for j in range(b_count)
        ]

        for j in range(b_count):
            model.addConstr(loads[j] <= c * u[j], name="cap_%d" % j)
        for t in range(m):
            model.addConstr(
                gp.quicksum(y[j, t] for j in range(b_count)) == demands[t],
                name="demand_%d" % t,
            )
        for j in range(b_count - 1):
            model.addConstr(u[j] >= u[j + 1], name="use_order_%d" % j)
            model.addConstr(loads[j] >= loads[j + 1], name="load_order_%d" % j)
        model.addConstr(gp.quicksum(u_list) >= lb, name="lower_bound")
        model.setObjective(gp.quicksum(u_list), GRB.MINIMIZE)

        width_index = {w: t for t, w in enumerate(weights)}
        for j, b in enumerate(ref):
            u[j].Start = 1
            cnt = [0] * m
            for w in b:
                cnt[width_index[w]] += 1
            for t, a in enumerate(cnt):
                if a:
                    y[j, t].Start = a

        def extract(u_vals, y_vals):
            cand = []
            for j in range(b_count):
                one_bin = []
                base = j * m
                for t in range(m):
                    k = int(round(y_vals[base + t]))
                    if k > 0:
                        one_bin.extend([weights[t]] * k)
                if one_bin:
                    cand.append(one_bin)
            return cand

        def cb(model, where):
            if where == GRB.Callback.MIPSOL:
                if time.monotonic() > deadline:
                    model.terminate()
                    return
                u_vals = model.cbGetSolution(u_list)
                y_vals = model.cbGetSolution(y_list)
                update_best(extract(u_vals, y_vals))
            elif where == GRB.Callback.MIP:
                if time.monotonic() > deadline:
                    model.terminate()

        model.optimize(cb)
        if model.SolCount > 0:
            u_vals = [v.X for v in u_list]
            y_vals = [v.X for v in y_list]
            update_best(extract(u_vals, y_vals))
        return True
    except Exception:
        return False


def main():
    args = parse_args()
    start = time.monotonic()
    deadline = start + max(0, int(args.time_limit))

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    c, weights, demands = read_instance(args.instance_path)
    lb, _, _ = lower_bound(c, weights, demands)

    best_bins = None
    best_obj = 1 << 60

    def update_best(candidate_bins):
        nonlocal best_bins, best_obj
        if candidate_bins is None:
            return False
        obj = len(candidate_bins)
        if obj >= best_obj:
            return False
        if not bins_are_valid(candidate_bins, c, weights, demands):
            return False
        best_bins = [list(b) for b in candidate_bins]
        best_obj = obj
        if logger:
            logger.log_solution(float(best_obj), solution_dict(best_bins))
        return True

    initial = best_fit_decreasing(c, weights, demands)
    update_best(initial)

    best_bins = eliminate_bins(best_bins, c, lb, deadline, update_best)
    best_obj = len(best_bins)

    if gp is not None and best_obj > lb and deadline - time.monotonic() > 1.0:
        m = len(weights)
        assign_vars = best_obj * (m + 1)
        pat_est = estimate_pattern_count(c, weights, demands, MAX_PATTERNS)
        pattern_ok = pat_est <= MAX_PATTERNS
        attempted = False

        if pattern_ok and (pat_est <= assign_vars or assign_vars > MAX_ASSIGN_VARS):
            attempted = solve_with_patterns(
                c, weights, demands, lb, best_bins, deadline, update_best
            )
        if not attempted and assign_vars <= MAX_ASSIGN_VARS:
            attempted = solve_with_assignment(
                c, weights, demands, lb, best_bins, deadline, update_best
            )
        if not attempted and pattern_ok:
            solve_with_patterns(c, weights, demands, lb, best_bins, deadline, update_best)

    if best_bins is None:
        # Safety fallback; should not be reached because best-fit always succeeds.
        best_bins = []
        for w, d in zip(weights, demands):
            q = max(1, c // w)
            full, rem = divmod(d, q)
            for _ in range(full):
                best_bins.append([w] * q)
            if rem:
                best_bins.append([w] * rem)
        update_best(best_bins)

    out = solution_dict(best_bins)

    out_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.solution_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()