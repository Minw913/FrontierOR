#!/usr/bin/env python3
"""
Minimize the number of used bin-period pairs for scheduling weighted items
with lag-based precedence constraints.

Approach:
  1. Compute per-item time windows (earliest/latest period) via longest paths
     in the lag graph (source at 0, sink at T+1).
  2. Build an initial feasible solution with a constraint-propagation greedy
     insertion heuristic, repair bin-limit violations, then improve it with
     relocation and bin-elimination local search.
  3. Warm-start a Gurobi MIP (item -> (bin, period) assignment formulation)
     from the heuristic solution and log every improving incumbent.
"""

import argparse
import json
import math
import time
from collections import defaultdict

try:
    from solution_logger import SolutionLogger
except Exception:  # pragma: no cover
    SolutionLogger = None

NEG_INF = float("-inf")


# --------------------------------------------------------------------------- #
# Parsing / utilities
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance_path", required=True)
    p.add_argument("--solution_path", required=True)
    p.add_argument("--time_limit", type=int, required=True)
    p.add_argument("--log_path", default=None)
    return p.parse_args()


def parse_L(value):
    if value is None:
        return math.inf
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("infinity", "inf", "unbounded", "unrestricted", "none"):
            return math.inf
        try:
            f = float(s)
            return math.inf if math.isinf(f) else int(f)
        except ValueError:
            return math.inf
    try:
        f = float(value)
        return math.inf if math.isinf(f) else int(f)
    except Exception:
        return math.inf


def longest_paths(nodes, edges, start):
    """Longest paths from `start` in a graph with integer (possibly negative)
    arc lags. Bellman-Ford style relaxation (maximizing)."""
    dist = {n: NEG_INF for n in nodes}
    dist[start] = 0
    n = len(nodes)
    for _ in range(max(0, n - 1)):
        changed = False
        for (u, v, lag) in edges:
            du = dist.get(u, NEG_INF)
            if du != NEG_INF:
                nv = du + lag
                if nv > dist.get(v, NEG_INF):
                    dist[v] = nv
                    changed = True
        if not changed:
            break
    return dist


def compute_windows(items, arcs, source, sink, T):
    nodes = set(items)
    for (u, v, _) in arcs:
        nodes.add(u)
        nodes.add(v)
    if source is not None:
        nodes.add(source)
    if sink is not None:
        nodes.add(sink)
    d1 = longest_paths(nodes, arcs, source) if source is not None else {}
    if sink is not None:
        rev = [(v, u, lag) for (u, v, lag) in arcs]
        d2 = longest_paths(nodes, rev, sink)
    else:
        d2 = {}
    ES, LS = {}, {}
    for i in items:
        e = d1.get(i, NEG_INF)
        e = 1 if e == NEG_INF else int(e)
        lf = d2.get(i, NEG_INF)
        ls = T if lf == NEG_INF else (T + 1) - int(lf)
        e = min(max(e, 1), T)
        ls = min(max(ls, 1), T)
        if e > ls:
            e, ls = 1, T
        ES[i] = e
        LS[i] = ls
    return ES, LS


# --------------------------------------------------------------------------- #
# Heuristic schedule state
# --------------------------------------------------------------------------- #
class Schedule:
    def __init__(self, w, ES, LS, preds, succs, T, W, LcapArr):
        self.w = w
        self.ES = ES
        self.LS = LS
        self.preds = preds
        self.succs = succs
        self.T = T
        self.W = W
        self.LcapArr = LcapArr
        self.t_of = {}          # item -> period
        self.b_of = {}          # item -> bin index (0-based within period)
        self.mem = {}           # period -> list of bins, each a list of items
        self.rem = {}           # period -> list of remaining capacity per bin

    def bounds(self, i):
        lo = self.ES[i]
        hi = self.LS[i]
        t_of = self.t_of
        for (j, lag) in self.preds[i]:
            tj = t_of.get(j)
            if tj is not None and tj + lag > lo:
                lo = tj + lag
        for (j, lag) in self.succs[i]:
            tj = t_of.get(j)
            if tj is not None and tj - lag < hi:
                hi = tj - lag
        return lo, hi

    def best_insert(self, i, allow_open=True, exclude_period=None, exclude_bin=None):
        lo, hi = self.bounds(i)
        wi = self.w[i]
        best = None
        for t in range(lo, hi + 1):
            if t < 1 or t > self.T:
                continue
            if exclude_period is not None and t == exclude_period:
                continue
            rems = self.rem.get(t)
            bf = -1
            bafter = None
            if rems:
                for idx in range(len(rems)):
                    r = rems[idx]
                    if r >= wi:
                        if (exclude_bin is not None and exclude_bin[0] == t
                                and exclude_bin[1] == idx):
                            continue
                        after = r - wi
                        if bafter is None or after < bafter:
                            bafter = after
                            bf = idx
            if bf >= 0:
                cand = (0, t, bafter, bf)
            else:
                nbin = len(self.mem.get(t, ()))
                if allow_open and nbin < self.LcapArr[t]:
                    cand = (1, t, wi, nbin)
                else:
                    cand = None
            if cand is not None and (best is None or cand < best):
                best = cand
        return best

    def apply_cand(self, i, cand):
        _, t, _, b = cand
        mems = self.mem.setdefault(t, [])
        rems = self.rem.setdefault(t, [])
        if b == len(mems):
            mems.append([i])
            rems.append(self.W - self.w[i])
        else:
            mems[b].append(i)
            rems[b] -= self.w[i]
        self.t_of[i] = t
        self.b_of[i] = b

    def place_in_bin(self, i, t, b):
        self.mem[t][b].append(i)
        self.rem[t][b] -= self.w[i]
        self.t_of[i] = t
        self.b_of[i] = b

    def remove(self, i):
        t = self.t_of.pop(i)
        b = self.b_of.pop(i)
        mems = self.mem[t]
        mems[b].remove(i)
        self.rem[t][b] += self.w[i]
        if not mems[b]:
            del mems[b]
            del self.rem[t][b]
            for bb in range(b, len(mems)):
                for j in mems[bb]:
                    self.b_of[j] = bb
        if not mems:
            del self.mem[t]
            del self.rem[t]

    def reinsert(self, i, t, b):
        mems = self.mem.get(t)
        wi = self.w[i]
        if mems is not None and b < len(mems) and self.rem[t][b] >= wi:
            mems[b].append(i)
            self.rem[t][b] -= wi
        else:
            if mems is None:
                self.mem[t] = []
                self.rem[t] = []
                mems = self.mem[t]
            b = min(b, len(mems))
            mems.insert(b, [i])
            self.rem[t].insert(b, self.W - wi)
            for bb in range(b + 1, len(mems)):
                for j in mems[bb]:
                    self.b_of[j] = bb
        self.t_of[i] = t
        self.b_of[i] = b

    def total_bins(self):
        return sum(len(v) for v in self.mem.values())


def repair_L(state, deadline):
    """Move items out of periods that exceed the per-period bin limit."""
    for t in sorted(list(state.mem.keys())):
        while True:
            mems = state.mem.get(t)
            if mems is None or len(mems) <= state.LcapArr[t]:
                break
            if time.time() > deadline:
                return
            best = None
            bi = None
            for idx in range(len(mems)):
                for i in mems[idx]:
                    c = state.best_insert(i, exclude_period=t)
                    if c is not None and (best is None or c < best):
                        best = c
                        bi = i
            if bi is None:
                break
            state.remove(bi)
            state.apply_cand(bi, best)


def relocation_pass(state, items, deadline):
    improved = False
    order = sorted(items,
                   key=lambda i: (len(state.mem[state.t_of[i]][state.b_of[i]]), i))
    for i in order:
        if time.time() > deadline:
            return improved
        t0 = state.t_of[i]
        b0 = state.b_of[i]
        frees = 1 if len(state.mem[t0][b0]) == 1 else 0
        state.remove(i)
        cand = state.best_insert(i)
        if cand is not None and (cand[0] - frees) < 0:
            state.apply_cand(i, cand)
            improved = True
        else:
            state.reinsert(i, t0, b0)
    return improved


def eliminate_once(state, deadline):
    """Try to completely empty one bin-period by relocating all its items
    into existing bins (no new bin opened). Returns True on success."""
    cand_list = []
    for t, mems in list(state.mem.items()):
        for idx, members in enumerate(mems):
            if members:
                cand_list.append((len(members),
                                  sum(state.w[x] for x in members), t, idx))
    cand_list.sort()
    for _, _, t, idx in cand_list:
        if time.time() > deadline:
            return False
        mems = state.mem.get(t)
        if mems is None or idx >= len(mems):
            continue
        members = list(mems[idx])
        if not members:
            continue
        moved = []
        failed = None
        ok = True
        for i in sorted(members, key=lambda x: (-state.w[x], x)):
            state.remove(i)
            c = state.best_insert(i, allow_open=False, exclude_bin=(t, idx))
            if c is None:
                ok = False
                failed = i
                break
            state.apply_cand(i, c)
            moved.append(i)
        if ok:
            return True
        # Roll back: put the failed and already-moved items back into one bin.
        mems = state.mem.get(t)
        if mems is not None and idx < len(mems):
            rb = idx
        else:
            if mems is None:
                state.mem[t] = []
                state.rem[t] = []
                mems = state.mem[t]
            mems.append([])
            state.rem[t].append(state.W)
            rb = len(mems) - 1
        state.place_in_bin(failed, t, rb)
        for i in reversed(moved):
            state.remove(i)
            state.place_in_bin(i, t, rb)
    return False


def local_search(state, items, deadline):
    for _ in range(5):
        if time.time() > deadline:
            return
        if not relocation_pass(state, items, deadline):
            break
    rounds = 0
    while time.time() < deadline and rounds < 300:
        if not eliminate_once(state, deadline):
            break
        rounds += 1
        relocation_pass(state, items, deadline)


# --------------------------------------------------------------------------- #
# Solution construction / validation
# --------------------------------------------------------------------------- #
def build_solution(assign, w):
    """assign: item -> (bin, period). Bin ids are renumbered 1..m per period."""
    bp = defaultdict(list)
    for i, (b, t) in assign.items():
        bp[(t, b)].append(i)
    per_t = defaultdict(set)
    for (t, b) in bp:
        per_t[t].add(b)
    remap = {}
    for t, bs in per_t.items():
        for newb, oldb in enumerate(sorted(bs), start=1):
            remap[(t, oldb)] = newb
    assignments = []
    for i in sorted(assign):
        b, t = assign[i]
        assignments.append({
            "item_id": int(i),
            "bin": int(remap[(t, b)]),
            "period": int(t),
            "weight": int(w[i]),
        })
    active = []
    for (t, b) in sorted(bp, key=lambda x: (x[0], remap[x])):
        its = sorted(bp[(t, b)])
        active.append({
            "bin": int(remap[(t, b)]),
            "period": int(t),
            "items": [int(x) for x in its],
            "total_weight": int(sum(w[x] for x in its)),
        })
    num = len(bp)
    return {
        "objective_value": float(num),
        "assignments": assignments,
        "active_bin_periods": active,
        "num_active_bin_periods": int(num),
    }


def validate(assign, items, w, W, T, L, arcs):
    if set(assign.keys()) != set(items):
        return False
    bpw = defaultdict(int)
    per = defaultdict(set)
    for i, (b, t) in assign.items():
        if not (1 <= t <= T):
            return False
        bpw[(t, b)] += w[i]
        per[t].add(b)
    for v in bpw.values():
        if v > W:
            return False
    if L != math.inf:
        for t, s in per.items():
            if len(s) > L:
                return False
    tpos = {i: assign[i][1] for i in assign}
    for (u, v, lag) in arcs:
        if u in tpos and v in tpos:
            if tpos[u] + lag > tpos[v]:
                return False
    return True


# --------------------------------------------------------------------------- #
# Gurobi MIP
# --------------------------------------------------------------------------- #
def run_mip(items, w, ES, LS, arcs, T, W, LcapArr, count_can, wsum_can, Bmust,
            heur_assign, heur_bins, incumbent, logger, deadline):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return None, None

    remaining = deadline - time.time()
    if remaining < 3.0:
        return None, None

    try:
        heur_t = {i: heur_assign[i][1] for i in heur_assign}

        maxbase = 1
        for t in range(1, T + 1):
            if count_can[t]:
                base = max(heur_bins.get(t, 0),
                           (wsum_can[t] + W - 1) // W, Bmust[t])
                if base > maxbase:
                    maxbase = base

        def make_windows(radius):
            if radius is None:
                return ES, LS
            E2, L2 = {}, {}
            for i in items:
                c = heur_t.get(i, ES[i])
                a = max(ES[i], c - radius)
                b = min(LS[i], c + radius)
                if a > b:
                    a, b = ES[i], LS[i]
                E2[i] = a
                L2[i] = b
            return E2, L2

        def make_Bt(slack):
            Bt = [0] * (T + 2)
            for t in range(1, T + 1):
                cc = count_can[t]
                if cc == 0:
                    continue
                base = max(heur_bins.get(t, 0),
                           (wsum_can[t] + W - 1) // W, Bmust[t])
                b = min(cc, LcapArr[t], base + slack)
                hb = min(heur_bins.get(t, 0), LcapArr[t], cc)
                if b < hb:
                    b = hb
                Bt[t] = b
            return Bt

        chosen = None
        for slack, radius in [(max(6, maxbase // 2), None), (3, None),
                              (1, None), (0, None), (0, 25), (0, 12), (0, 6)]:
            E2, L2 = make_windows(radius)
            Bt = make_Bt(slack)
            n = 0
            for i in items:
                for t in range(E2[i], L2[i] + 1):
                    n += Bt[t]
            if n <= 500000:
                chosen = (E2, L2, Bt)
                break
        if chosen is None:
            E2, L2 = make_windows(6)
            chosen = (E2, L2, make_Bt(0))
        E2, L2, Bt = chosen

        m = gp.Model("bin_period_scheduling")
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.NumericFocus = 0
        m.Params.Threads = 1
        m.Params.LogToConsole = 0
        m.Params.TimeLimit = max(1.0, remaining - 1.0)

        can = defaultdict(list)
        for i in items:
            for t in range(E2[i], L2[i] + 1):
                can[t].append(i)

        keys = []
        item_keys = defaultdict(list)
        for i in items:
            for t in range(E2[i], L2[i] + 1):
                for k in range(1, Bt[t] + 1):
                    key = (i, t, k)
                    keys.append(key)
                    item_keys[i].append(key)
        if not keys:
            return None, None

        x = m.addVars(keys, vtype=GRB.BINARY, name="x")
        ykeys = [(t, k) for t in range(1, T + 1) for k in range(1, Bt[t] + 1)]
        y = m.addVars(ykeys, vtype=GRB.BINARY, name="y")

        pe = {}
        for i in items:
            ks = item_keys[i]
            m.addConstr(gp.quicksum(x[key] for key in ks) == 1)
            pe[i] = gp.quicksum(t * x[(ii, t, kk)] for (ii, t, kk) in ks)

        for t in range(1, T + 1):
            bt = Bt[t]
            if bt == 0:
                continue
            il = can.get(t, [])
            for k in range(1, bt + 1):
                m.addConstr(gp.quicksum(w[i] * x[i, t, k] for i in il)
                            <= W * y[t, k])
            for k in range(1, bt):
                m.addConstr(y[t, k] >= y[t, k + 1])

        for (u, v, lag) in arcs:
            if u in pe and v in pe:
                if E2[v] >= L2[u] + lag:
                    continue
                m.addConstr(pe[v] - pe[u] >= lag)

        total_w = sum(w[i] for i in items)
        m.addConstr(y.sum() >= (total_w + W - 1) // W)
        m.setObjective(y.sum(), GRB.MINIMIZE)

        # Warm start from the heuristic solution.
        for i, (b, t) in heur_assign.items():
            k = b + 1
            if E2[i] <= t <= L2[i] and k <= Bt[t]:
                x[i, t, k].Start = 1.0
        used = set()
        for i, (b, t) in heur_assign.items():
            k = b + 1
            if k <= Bt[t]:
                used.add((t, k))
        for (t, k) in used:
            y[t, k].Start = 1.0

        xvars = [x[key] for key in keys]
        cb_state = {"best": float(incumbent), "sol": None}

        def cb(model, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                except Exception:
                    obj = None
                if obj is not None and obj < cb_state["best"] - 1e-6:
                    try:
                        vals = model.cbGetSolution(xvars)
                        assign = {}
                        for key, val in zip(keys, vals):
                            if val > 0.5:
                                assign[key[0]] = (key[2], key[1])
                        sol = build_solution(assign, w)
                        cb_state["best"] = obj
                        cb_state["sol"] = sol
                        if logger is not None:
                            logger.log_solution(float(obj), sol)
                    except Exception:
                        pass
                if time.time() > deadline:
                    model.terminate()

        m.optimize(cb)

        sol, obj = None, None
        if m.SolCount > 0:
            try:
                vals = m.getAttr("X", xvars)
                assign = {}
                for key, val in zip(keys, vals):
                    if val > 0.5:
                        assign[key[0]] = (key[2], key[1])
                sol = build_solution(assign, w)
                obj = float(m.ObjVal)
            except Exception:
                sol, obj = cb_state["sol"], cb_state["best"]
        elif cb_state["sol"] is not None:
            sol, obj = cb_state["sol"], cb_state["best"]
        return sol, obj
    except Exception:
        return None, None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    start = time.time()
    deadline = start + max(1, args.time_limit)

    logger = None
    if args.log_path and SolutionLogger is not None:
        try:
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    with open(args.instance_path) as f:
        data = json.load(f)

    params = data.get("parameters", {})
    T = int(params["T"])
    W = int(params["W"])
    L = parse_L(params.get("L", "infinity"))
    source = data.get("source_item_id")
    sink = data.get("sink_item_id")

    items = []
    w = {}
    for it in data.get("items", []):
        if it.get("type") == "treatment":
            iid = int(it["item_id"])
            items.append(iid)
            w[iid] = int(it["weight"])

    arcs = []
    for a in data.get("arcs", []):
        arcs.append((int(a["from"]), int(a["to"]), int(a["lag"])))

    best_sol = {
        "objective_value": 0.0,
        "assignments": [],
        "active_bin_periods": [],
        "num_active_bin_periods": 0,
    }

    try:
        ES, LS = compute_windows(items, arcs, source, sink, T)

        preds = defaultdict(list)
        succs = defaultdict(list)
        for (u, v, lag) in arcs:
            if u in w and v in w:
                preds[v].append((u, lag))
                succs[u].append((v, lag))

        diff_c = [0] * (T + 3)
        diff_w = [0] * (T + 3)
        for i in items:
            diff_c[ES[i]] += 1
            diff_c[LS[i] + 1] -= 1
            diff_w[ES[i]] += w[i]
            diff_w[LS[i] + 1] -= w[i]
        count_can = [0] * (T + 2)
        wsum_can = [0] * (T + 2)
        c = 0
        s = 0
        for t in range(1, T + 1):
            c += diff_c[t]
            s += diff_w[t]
            count_can[t] = c
            wsum_can[t] = s

        LcapArr = [0] * (T + 2)
        for t in range(1, T + 1):
            LcapArr[t] = count_can[t] if L == math.inf else min(L, count_can[t])

        Bmust = [0] * (T + 2)
        forced = defaultdict(list)
        for i in items:
            if ES[i] == LS[i]:
                forced[ES[i]].append(w[i])
        for t, wl in forced.items():
            bins = []
            for wt in sorted(wl, reverse=True):
                placed = False
                for bi in range(len(bins)):
                    if bins[bi] >= wt:
                        bins[bi] -= wt
                        placed = True
                        break
                if not placed:
                    bins.append(W - wt)
            Bmust[t] = len(bins)

        # ---------------- Heuristic ---------------- #
        heur_deadline = min(deadline - 1.0,
                            start + max(1.0, min(12.0, 0.15 * args.time_limit)))
        state = Schedule(w, ES, LS, preds, succs, T, W, LcapArr)
        order = sorted(items, key=lambda i: (ES[i], LS[i] - ES[i], -w[i], i))
        for i in order:
            cand = state.best_insert(i)
            if cand is None:
                lo, hi = state.bounds(i)
                if lo > hi:
                    lo, hi = ES[i], LS[i]
                t = min(max(lo, 1), T)
                cand = (1, t, w[i], len(state.mem.get(t, ())))
            state.apply_cand(i, cand)
        try:
            repair_L(state, heur_deadline)
            local_search(state, items, heur_deadline)
        except Exception:
            pass

        heur_assign = {i: (state.b_of[i], state.t_of[i])
                       for i in items if i in state.t_of}
        heur_bins = {t: len(mems) for t, mems in state.mem.items()}
        sol_h = build_solution(heur_assign, w)
        obj_h = float(sol_h["num_active_bin_periods"])

        best_obj = float("inf")
        if items and validate(heur_assign, items, w, W, T, L, arcs):
            best_sol = sol_h
            best_obj = obj_h
            if logger is not None:
                try:
                    logger.log_solution(obj_h, sol_h)
                except Exception:
                    pass
        elif not items:
            best_obj = 0.0
            if logger is not None:
                try:
                    logger.log_solution(0.0, sol_h)
                except Exception:
                    pass

        # ---------------- MIP ---------------- #
        if items and time.time() < deadline - 2.0:
            inc = best_obj if best_obj < float("inf") else float("inf")
            sol_m, obj_m = run_mip(items, w, ES, LS, arcs, T, W, LcapArr,
                                   count_can, wsum_can, Bmust, heur_assign,
                                   heur_bins, inc, logger, deadline)
            if sol_m is not None and obj_m is not None:
                assign_m = {a["item_id"]: (a["bin"], a["period"])
                            for a in sol_m["assignments"]}
                if (obj_m < best_obj
                        and validate(assign_m, items, w, W, T, L, arcs)):
                    best_sol = sol_m
                    best_obj = obj_m

        if best_obj == float("inf"):
            best_sol = sol_h  # last resort
    except Exception:
        pass

    with open(args.solution_path, "w") as f:
        json.dump(best_sol, f)


if __name__ == "__main__":
    main()