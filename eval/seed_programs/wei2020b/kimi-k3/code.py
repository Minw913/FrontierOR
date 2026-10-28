"""
Two-dimensional (weight + volume) bin packing solver.

Approach:
  1. Multi-start greedy construction heuristics (several decreasing orderings
     x several fit rules).
  2. Deterministic bin-elimination local search.
  3. Small/medium instances: a Gurobi MIP assignment formulation with symmetry
     breaking, warm-started from the heuristic solution, minimizing the number
     of bins (incumbents are captured in a callback).
     Large instances: randomized elimination combined with MIP-based
     large-neighborhood bin elimination.
Every improving incumbent is reported through solution_logger.
"""

import argparse
import json
import random
import time

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_OK = True
except Exception:
    gp = None
    GRB = None
    GUROBI_OK = False

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def read_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    items = [(int(it["w"]), int(it["v"])) for it in data["items"]]
    return items, int(data["W"]), int(data["V"])


def solution_dict_from_bins(bins):
    out_bins = []
    for items_list, tw, tv in bins:
        out_bins.append({
            "items": sorted(int(i) for i in items_list),
            "total_weight": int(tw),
            "total_volume": int(tv),
        })
    return {"objective_value": len(out_bins), "bins": out_bins}


def copy_bins(bins):
    return [[b[0][:], b[1], b[2]] for b in bins]


def validate_solution(bins, items, W, V, n):
    seen = bytearray(n)
    for items_list, tw, tv in bins:
        sw = 0
        sv = 0
        for i in items_list:
            if i < 0 or i >= n or seen[i]:
                return False
            seen[i] = 1
            sw += items[i][0]
            sv += items[i][1]
        if sw != tw or sv != tv or sw > W or sv > V:
            return False
    for b in seen:
        if not b:
            return False
    return True


# --------------------------------------------------------------------------- #
# Constructive heuristics
# --------------------------------------------------------------------------- #

def make_orders(items, W, V):
    n = len(items)
    idx = list(range(n))
    key_max = [0.0] * n
    key_sum = [0.0] * n
    key_prod = [0.0] * n
    for i, (w, v) in enumerate(items):
        a = w / W
        b = v / V
        key_max[i] = a if a >= b else b
        key_sum[i] = a + b
        key_prod[i] = a * b
    return {
        "max": sorted(idx, key=lambda i: key_max[i], reverse=True),
        "sum": sorted(idx, key=lambda i: key_sum[i], reverse=True),
        "prod": sorted(idx, key=lambda i: key_prod[i], reverse=True),
        "w": sorted(idx, key=lambda i: items[i][0], reverse=True),
        "v": sorted(idx, key=lambda i: items[i][1], reverse=True),
    }


def pack_items(items, W, V, order, fit_rule, window=0):
    """fit_rule: 0=first fit, 1=best-fit min sum slack, 2=min max slack,
    3=min min slack. `window`>0 restricts the scan to the most recent bins."""
    bins_items = []
    lw = []
    lv = []
    for i in order:
        w, v = items[i]
        chosen = -1
        nb = len(bins_items)
        lo = 0
        if window and nb > window:
            lo = nb - window
        if fit_rule == 0:
            for j in range(lo, nb):
                if lw[j] + w <= W and lv[j] + v <= V:
                    chosen = j
                    break
        else:
            best_s = None
            for j in range(lo, nb):
                rw = W - lw[j] - w
                rv = V - lv[j] - v
                if rw >= 0 and rv >= 0:
                    a = rw * V
                    b = rv * W
                    if fit_rule == 1:
                        s = a + b
                    elif fit_rule == 2:
                        s = a if a >= b else b
                    else:
                        s = a if a <= b else b
                    if best_s is None or s < best_s:
                        best_s = s
                        chosen = j
        if chosen < 0:
            bins_items.append([i])
            lw.append(w)
            lv.append(v)
        else:
            bins_items[chosen].append(i)
            lw[chosen] += w
            lv[chosen] += v
    return [[bins_items[j], lw[j], lv[j]] for j in range(len(bins_items))]


# --------------------------------------------------------------------------- #
# Bin-elimination local search
# --------------------------------------------------------------------------- #

def eliminate_once(bins, items, W, V, rng=None, deadline=None):
    """Try to empty one bin into the others. Returns (bins, success).
    Does not mutate the input; returns a fresh object on success."""
    K = len(bins)
    if K <= 1:
        return bins, False
    targets = sorted(range(K),
                     key=lambda j: (len(bins[j][0]), bins[j][1] / W + bins[j][2] / V))
    if rng is not None:
        m = min(K, 8)
        head = targets[:m]
        rng.shuffle(head)
        targets = head + targets[m:]
    for b in targets:
        if deadline is not None and time.time() > deadline:
            return bins, False
        bitems = bins[b][0]
        if rng is None:
            ord_items = sorted(bitems,
                               key=lambda i: items[i][0] / W + items[i][1] / V,
                               reverse=True)
        else:
            ord_items = bitems[:]
            rng.shuffle(ord_items)
        loads_w = [bins[j][1] for j in range(K)]
        loads_v = [bins[j][2] for j in range(K)]
        placements = []
        ok = True
        for i in ord_items:
            w, v = items[i]
            chosen = -1
            best_s = None
            for j in range(K):
                if j == b:
                    continue
                if loads_w[j] + w <= W and loads_v[j] + v <= V:
                    s = (W - loads_w[j] - w) * V + (V - loads_v[j] - v) * W
                    if rng is not None:
                        s *= 0.8 + 0.4 * rng.random()
                    if best_s is None or s < best_s:
                        best_s = s
                        chosen = j
            if chosen < 0:
                ok = False
                break
            placements.append((i, chosen))
            loads_w[chosen] += w
            loads_v[chosen] += v
        if ok:
            new_bins = []
            for j in range(K):
                if j != b:
                    new_bins.append([bins[j][0][:], bins[j][1], bins[j][2]])
            for i, j in placements:
                jj = j - 1 if j > b else j
                new_bins[jj][0].append(i)
                new_bins[jj][1] += items[i][0]
                new_bins[jj][2] += items[i][1]
            return new_bins, True
    return bins, False


def eliminate_loop(bins, items, W, V, deadline):
    bins = copy_bins(bins)
    while len(bins) > 1 and time.time() < deadline:
        new_bins, ok = eliminate_once(bins, items, W, V, None, deadline)
        if not ok:
            break
        bins = new_bins
    return bins


# --------------------------------------------------------------------------- #
# Gurobi helpers
# --------------------------------------------------------------------------- #

def _set_params(model, time_limit=None):
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = max(0.5, float(time_limit))


def mip_pack_subset(item_ids, items, W, V, num_slots, time_limit):
    """Try to pack the given items into exactly `num_slots` bins.
    Returns list of bins (lists of original indices) or None."""
    if not GUROBI_OK:
        return None
    mloc = len(item_ids)
    if mloc == 0:
        return []
    if num_slots <= 0:
        return None
    if num_slots >= mloc:
        return [[i] for i in item_ids]
    K = num_slots
    model = gp.Model("pack")
    _set_params(model, time_limit)
    x = {}
    for i in range(mloc):
        maxj = i if i < K else K - 1
        for j in range(maxj + 1):
            x[i, j] = model.addVar(vtype=GRB.BINARY)
    for i in range(mloc):
        maxj = i if i < K else K - 1
        model.addConstr(gp.quicksum(x[i, j] for j in range(maxj + 1)) == 1)
    for j in range(K):
        model.addConstr(gp.quicksum(items[item_ids[i]][0] * x[i, j]
                                    for i in range(j, mloc)) <= W)
        model.addConstr(gp.quicksum(items[item_ids[i]][1] * x[i, j]
                                    for i in range(j, mloc)) <= V)
    model.optimize()
    res = None
    if model.SolCount > 0:
        acc = [[] for _ in range(K)]
        for (i, j), var in x.items():
            if var.X > 0.5:
                acc[j].append(item_ids[i])
        res = [b for b in acc if b]
    try:
        model.dispose()
    except Exception:
        pass
    return res


def lns_attempt(bins, items, W, V, rng, deadline):
    """Try to repack the items of a small group of bins into one fewer bin."""
    K = len(bins)
    if K < 2:
        return bins, False
    order = sorted(range(K),
                   key=lambda j: (len(bins[j][0]), bins[j][1] / W + bins[j][2] / V))
    pool = order[:min(K, 6)]
    rng.shuffle(pool)
    attempts = 0
    for b in pool:
        if attempts >= 4 or time.time() >= deadline:
            break
        cand_pool = [j for j in order[:min(K, 12)] if j != b]
        rng.shuffle(cand_pool)
        for g in (2, 3):
            if g > K or attempts >= 4 or time.time() >= deadline:
                break
            others = cand_pool[:g - 1]
            if len(others) < g - 1:
                continue
            group = sorted(set([b]) | set(others))
            S = []
            for j in group:
                S.extend(bins[j][0])
            remaining = deadline - time.time()
            if remaining < 0.8:
                break
            tl = max(1.0, min(8.0, remaining / 4.0))
            try:
                res = mip_pack_subset(S, items, W, V, len(group) - 1, tl)
            except Exception:
                res = None
            attempts += 1
            if res is not None and len(res) <= len(group) - 1:
                gset = set(group)
                new_bins = [[bins[j][0][:], bins[j][1], bins[j][2]]
                            for j in range(K) if j not in gset]
                for bitems in res:
                    tw = 0
                    tv = 0
                    for i in bitems:
                        tw += items[i][0]
                        tv += items[i][1]
                    new_bins.append([list(bitems), tw, tv])
                return new_bins, True
    return bins, False


def mip_solve_global(items, W, V, init_bins, lb, time_limit, logger, state):
    """Full assignment MIP minimizing the number of bins, warm-started from
    the current best solution. Incumbents update `state` and get logged."""
    if not GUROBI_OK:
        return
    n = len(items)
    K = len(init_bins)
    if K <= lb or K < 2 or time_limit < 1.0:
        return
    order = sorted(range(n), key=lambda i: items[i][0] / W + items[i][1] / V,
                   reverse=True)
    pos = [0] * n
    for p, i in enumerate(order):
        pos[i] = p
    sw = [items[i][0] for i in order]
    sv = [items[i][1] for i in order]

    model = gp.Model("vbp")
    _set_params(model, time_limit)
    model.Params.MIPFocus = 1
    if lb > 0:
        model.Params.BestObjStop = float(lb)
    model.Params.Cutoff = float(K) - 0.5  # only strictly improving solutions

    x = {}
    for p in range(n):
        maxj = p if p < K else K - 1
        for j in range(maxj + 1):
            x[p, j] = model.addVar(vtype=GRB.BINARY)
    y = model.addVars(K, vtype=GRB.BINARY)

    model.setObjective(gp.quicksum(y[j] for j in range(K)), GRB.MINIMIZE)
    model.addConstr(gp.quicksum(y[j] for j in range(K)) >= lb)
    for p in range(n):
        maxj = p if p < K else K - 1
        model.addConstr(gp.quicksum(x[p, j] for j in range(maxj + 1)) == 1)
    for j in range(K):
        model.addConstr(gp.quicksum(sw[p] * x[p, j] for p in range(j, n))
                        <= W * y[j])
        model.addConstr(gp.quicksum(sv[p] * x[p, j] for p in range(j, n))
                        <= V * y[j])
    for j in range(K - 1):
        model.addConstr(y[j + 1] <= y[j])

    # Warm start: reorder bins by smallest item position so that the symmetry
    # breaking x[p, j] = 0 for j > p is respected.
    bin_order = sorted(range(K), key=lambda j: min(pos[i] for i in init_bins[j][0]))
    for jj, bj in enumerate(bin_order):
        y[jj].Start = 1.0
        for i in init_bins[bj][0]:
            x[pos[i], jj].Start = 1.0

    xkeys = list(x.keys())
    xvars = [x[k] for k in xkeys]

    def cb(mdl, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            obj = int(round(mdl.cbGet(GRB.Callback.MIPSOL_OBJ)))
            if obj >= state["best_obj"]:
                return
            vals = mdl.cbGetSolution(xvars)
            acc = [[] for _ in range(K)]
            for (p, j), val in zip(xkeys, vals):
                if val > 0.5:
                    acc[j].append(order[p])
            new_bins = []
            feasible = True
            for b in acc:
                if b:
                    tw = 0
                    tv = 0
                    for i in b:
                        tw += items[i][0]
                        tv += items[i][1]
                    if tw > W or tv > V:
                        feasible = False
                        break
                    new_bins.append([b, tw, tv])
            if feasible and len(new_bins) == obj:
                flat = [i for b, _, _ in new_bins for i in b]
                if len(flat) == n and len(set(flat)) == n:
                    state["best"] = new_bins
                    state["best_obj"] = obj
                    if logger is not None:
                        try:
                            logger.log_solution(obj, solution_dict_from_bins(new_bins))
                        except Exception:
                            pass
        except Exception:
            pass

    model.optimize(cb)
    try:
        model.dispose()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = None
    if args.log_path and SolutionLogger is not None:
        try:
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    t0 = time.time()
    total = max(1, int(args.time_limit))
    deadline = t0 + total
    buffer = max(0.5, min(2.0, 0.05 * total))

    items, W, V = read_instance(args.instance_path)
    n = len(items)

    state = {"best": None, "best_obj": None}

    def log_fn(bins):
        obj = len(bins)
        if state["best_obj"] is None or obj < state["best_obj"]:
            state["best"] = bins
            state["best_obj"] = obj
            if logger is not None:
                try:
                    logger.log_solution(obj, solution_dict_from_bins(bins))
                except Exception:
                    pass

    if n == 0:
        sol = {"objective_value": 0, "bins": []}
        if logger is not None:
            try:
                logger.log_solution(0, sol)
            except Exception:
                pass
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    sumw = 0
    sumv = 0
    for w, v in items:
        sumw += w
        sumv += v
    lb_w = (sumw + W - 1) // W
    lb_v = (sumv + V - 1) // V
    lb = lb_w if lb_w >= lb_v else lb_v

    # ---- Phase A: constructive heuristics ----
    try:
        orders = make_orders(items, W, V)
        combos = [
            ("max", 1), ("sum", 1), ("max", 2), ("sum", 2),
            ("prod", 1), ("max", 3), ("sum", 3), ("w", 1),
            ("v", 1), ("prod", 2), ("w", 2), ("v", 2),
            ("max", 0), ("sum", 0), ("prod", 0), ("w", 0), ("v", 0),
            ("prod", 3), ("w", 3), ("v", 3),
        ]
        window = 0
        if n > 3000:
            combos = combos[:2]
        elif n > 1200:
            combos = combos[:8]
        if n > 6000:
            window = 2500
        heur_deadline = min(deadline - buffer, t0 + max(1.0, 0.25 * total))
        best = None
        for k, (oname, frule) in enumerate(combos):
            if k >= 2 and time.time() > heur_deadline:
                break
            res = pack_items(items, W, V, orders[oname], frule, window)
            if best is None or len(res) < len(best):
                best = res
        log_fn(best)
    except Exception:
        if state["best"] is None:
            log_fn([[[i], items[i][0], items[i][1]] for i in range(n)])

    # ---- Phase B: deterministic bin elimination ----
    try:
        if state["best_obj"] > lb:
            elim_deadline = min(deadline - buffer,
                                time.time() + max(1.0, 0.10 * total))
            newb = eliminate_loop(state["best"], items, W, V, elim_deadline)
            if len(newb) < state["best_obj"]:
                log_fn(newb)
    except Exception:
        pass

    # ---- Phase C: exact / metaheuristic improvement ----
    try:
        if state["best_obj"] > lb and time.time() < deadline - buffer:
            Kcur = state["best_obj"]
            use_global = GUROBI_OK and n <= 300 and n * Kcur <= 200000
            if use_global:
                remaining = deadline - buffer - time.time()
                if remaining >= 1.0:
                    mip_solve_global(items, W, V, state["best"], lb,
                                     remaining, logger, state)
            else:
                rng = random.Random(0)
                improve_deadline = deadline - buffer
                lns_fails = 0
                outer_stall = 0
                while time.time() < improve_deadline and state["best_obj"] > lb:
                    prev = state["best_obj"]
                    burst_end = min(improve_deadline, time.time() + 2.0)
                    while time.time() < burst_end and state["best_obj"] > lb:
                        cand, ok = eliminate_once(state["best"], items, W, V,
                                                  rng, burst_end)
                        if ok and len(cand) < state["best_obj"]:
                            log_fn(cand)
                    if (GUROBI_OK and lns_fails < 3
                            and time.time() < improve_deadline - 1.0):
                        try:
                            newb, ok2 = lns_attempt(state["best"], items, W, V,
                                                    rng, improve_deadline)
                        except Exception:
                            newb, ok2 = state["best"], False
                        if ok2 and len(newb) < state["best_obj"]:
                            log_fn(newb)
                            lns_fails = 0
                        else:
                            lns_fails += 1
                    if state["best_obj"] < prev:
                        outer_stall = 0
                    else:
                        outer_stall += 1
                        if outer_stall >= 25:
                            break
    except Exception:
        pass

    # ---- Final output ----
    best = state["best"]
    if best is None or not validate_solution(best, items, W, V, n):
        best = [[[i], items[i][0], items[i][1]] for i in range(n)]
    sol = solution_dict_from_bins(best)
    if logger is not None and (state["best_obj"] is None
                               or sol["objective_value"] < state["best_obj"]):
        try:
            logger.log_solution(sol["objective_value"], sol)
        except Exception:
            pass
    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()