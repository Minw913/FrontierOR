import argparse
import json
import time
import math
import random

random.seed(0)

from solution_logger import SolutionLogger

GUROBI_OK = False
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_OK = True
except Exception:
    GUROBI_OK = False


def popcount(x):
    return x.bit_count() if hasattr(int, "bit_count") else bin(x).count("1")


class Solver:
    EXACT_LIMIT = 11  # exact per-station sequencing (Held-Karp DP) up to this many tasks

    def __init__(self, inst):
        self.n = inst["num_tasks"]
        self.m = inst["num_stations"]
        self.p = [int(v) for v in inst["processing_times"]]
        self.Fwd = inst["forward_setup_times"]
        self.Bwd = inst["backward_setup_times"]
        prels = inst.get("precedence_relations", []) or []
        flat = [v for pr in prels for v in pr]
        shift = 1 if flat and all(v >= 1 for v in flat) else 0
        self.precs = []
        for a, b in prels:
            a -= shift
            b -= shift
            if 0 <= a < self.n and 0 <= b < self.n and a != b:
                self.precs.append((a, b))
        self.pred = [[] for _ in range(self.n)]
        self.succ = [[] for _ in range(self.n)]
        for a, b in self.precs:
            self.pred[b].append(a)
            self.succ[a].append(b)
        self.inmask = [0] * self.n
        self.outmask = [0] * self.n
        for t in range(self.n):
            for pr in self.pred[t]:
                self.inmask[t] |= 1 << pr
            for sc in self.succ[t]:
                self.outmask[t] |= 1 << sc
        self.cache = {}
        self.best_obj = math.inf
        self.best_masks = None
        self.initialized = False

    # ---------- basic helpers ----------
    def bits(self, mask):
        out = []
        while mask:
            lsb = mask & -mask
            out.append(lsb.bit_length() - 1)
            mask ^= lsb
        return out

    def seq_cost(self, seq):
        tot = sum(self.p[t] for t in seq)
        for a, b in zip(seq, seq[1:]):
            tot += self.Fwd[a][b]
        tot += self.Bwd[seq[-1]][seq[0]]
        return tot

    def seq_feasible(self, seq):
        pos = {t: i for i, t in enumerate(seq)}
        mask = 0
        for t in seq:
            mask |= 1 << t
        for t in seq:
            pre = self.inmask[t] & mask
            if pre:
                for i, u in enumerate(seq):
                    if (pre >> u) & 1 and pos[u] > pos[t]:
                        return False
        return True

    # ---------- per-station sequencing ----------
    def exact_seq(self, mask):
        tasks = self.bits(mask)
        sumP = sum(self.p[t] for t in tasks)
        best = math.inf
        bestseq = None
        for f in tasks:
            if self.inmask[f] & mask:
                continue
            fbit = 1 << f
            base = mask ^ fbit
            subs = []
            s = base
            while True:
                subs.append(s | fbit)
                if s == 0:
                    break
                s = (s - 1) & base
            subs.sort(key=popcount)
            dps = {fbit: {f: 0}}
            par = {}
            for sub in subs:
                lasts = dps.get(sub)
                if not lasts:
                    continue
                rembits = self.bits(mask ^ sub)
                for last, cost in lasts.items():
                    for nxt in rembits:
                        if (self.inmask[nxt] & mask) & ~sub:
                            continue
                        nb = 1 << nxt
                        nsub = sub | nb
                        nc = cost + self.Fwd[last][nxt]
                        nd = dps.get(nsub)
                        if nd is None:
                            dps[nsub] = {nxt: nc}
                            par[(nsub, nxt)] = last
                        elif nc < nd.get(nxt, math.inf):
                            nd[nxt] = nc
                            par[(nsub, nxt)] = last
            full = dps.get(mask)
            if full:
                for last, c in full.items():
                    tot = c + self.Bwd[last][f] + sumP
                    if tot < best:
                        # reconstruct sequence
                        key = (mask, last)
                        seq = [last]
                        cur = key
                        while cur in par:
                            prev = par[cur]
                            seq.append(prev)
                            subc, lastc = cur
                            cur = (subc & ~(1 << lastc), prev)
                        seq.reverse()
                        # seq should start with f; verify
                        best = tot
                        bestseq = seq
        if bestseq is None:
            return self.greedy_seq(mask)
        return (best, bestseq)

    def greedy_seq(self, mask):
        tasks = self.bits(mask)
        order = []
        chosen = 0
        rem = mask
        last = None
        while rem:
            cands = [t for t in self.bits(rem)
                     if ((self.inmask[t] & mask) & ~chosen) == 0 or ((self.inmask[t] & mask) & ~chosen) == 0]
            # recompute properly
            cands = []
            for t in self.bits(rem):
                if ((self.inmask[t] & mask) & ~chosen) == 0:
                    cands.append(t)
            if not cands:  # should not happen on a DAG
                cands = self.bits(rem)
            if last is None:
                t = min(cands, key=lambda x: (self.p[x], x))
            else:
                t = min(cands, key=lambda x: (self.Fwd[last][x], x))
            order.append(t)
            chosen |= 1 << t
            rem &= ~chosen if False else (rem ^ (1 << t))
            last = t
        # insertion improvement (feasibility checked cheaply)
        improved = True
        passes = 0
        while improved and passes < 25:
            improved = False
            passes += 1
            cur_cost = self.seq_cost(order)
            for i in range(len(order)):
                t = order[i]
                rest = order[:i] + order[i + 1:]
                pre_mask = self.inmask[t] & mask
                succ_mask = self.outmask[t] & mask
                lo = 0
                hi = len(rest)
                for ridx, r in enumerate(rest):
                    if (pre_mask >> r) & 1:
                        lo = max(lo, ridx + 1)
                    if (succ_mask >> r) & 1:
                        hi = min(hi, ridx)
                for j in range(lo, min(hi + 1, len(rest) + 1)):
                    cand = rest[:j] + [t] + rest[j:]
                    if cand == order:
                        continue
                    c = self.seq_cost(cand)
                    if c < cur_cost - 1e-9:
                        order = cand
                        cur_cost = c
                        improved = True
                        break
                if improved:
                    break
        return (self.seq_cost(order), order)

    def station_cost(self, mask):
        got = self.cache.get(mask)
        if got is not None:
            return got
        k = popcount(mask)
        if k <= self.EXACT_LIMIT:
            res = self.exact_seq(mask)
        else:
            res = self.greedy_seq(mask)
        self.cache[mask] = res
        return res

    # ---------- solution building / evaluation ----------
    def build_solution_dict(self, masks):
        assignment = {}
        station_sequences = {}
        worst = 0
        for k in range(self.m):
            mk = masks[k]
            cost, seq = self.station_cost(mk)
            station_sequences[str(k + 1)] = [t + 1 for t in seq]
            worst = max(worst, cost)
            if mk:
                for t in self.bits(mk):
                    assignment[str(t + 1)] = k + 1
        sol = {
            "objective_value": float(worst),
            "assignment": assignment,
            "station_sequences": station_sequences,
            "cycle_time": float(worst),
        }
        return worst, sol

    # ---------- initial construction ----------
    def topo_order(self):
        indeg = [popcount(self.inmask[t]) for t in range(self.n)]
        avail = sorted([t for t in range(self.n) if indeg[t] == 0])
        order = []
        indeg = indeg[:]
        import heapq
        heap = [(t, t) for t in avail]
        heapq.heapify(heap)
        while heap:
            _, t = heapq.heappop(heap)
            order.append(t)
            for sc in self.succ[t]:
                indeg[sc] -= 1
                if indeg[sc] == 0:
                    heapq.heappush(heap, (sc, sc))
        if len(order) != self.n:  # cycle fallback
            return list(range(self.n))
        return order

    def initial_masks(self):
        n, m = self.n, self.m
        order = self.topo_order()
        if m >= n:
            masks = [1 << order[k] for k in range(n)] + [0] * (m - n)
            # stations beyond n would be empty: merge backward
            masks = []
            for k in range(m):
                masks.append(1 << order[k])
            return masks
        total = sum(self.p)
        masks = [0] * m
        remaining_tasks = n
        remaining_stations = m
        idx = 0
        cum = 0
        for s in range(m - 1):
            quota = total * (s + 1) / m
            scount = 0
            while idx < n and (cum + self.p[order[idx]] <= quota or scount == 0) \
                    and remaining_tasks - 1 >= remaining_stations - 1:
                masks[s] |= 1 << order[idx]
                cum += self.p[order[idx]]
                remaining_tasks -= 1
                scount += 1
                idx += 1
            remaining_stations -= 1
        while idx < n:
            masks[m - 1] |= 1 << order[idx]
            idx += 1
        return masks

    # ---------- local search ----------
    def search(self, end_time, logger):
        if self.best_masks is None:
            masks = self.initial_masks()
            # fill possible empty stations defensively by moving some task
            for k in range(self.m):
                if masks[k] == 0:
                    for k2 in range(self.m):
                        if popcount(masks[k2]) > 1:
                            tb = self.bits(masks[k2])[0]
                            masks[k2] ^= 1 << tb
                            masks[k] |= 1 << tb
                            break
            self.best_masks = masks
            obj, sol = self.build_solution_dict(masks)
            self.best_obj = obj
            if logger:
                logger.log_solution(float(obj), sol)
        station_of = [0] * self.n
        masks = list(self.best_masks)
        for k in range(self.m):
            for t in self.bits(masks[k]):
                station_of[t] = k
        costs = [self.station_cost(masks[k])[0] for k in range(self.m)]
        cur_obj = max(costs) if costs else 0

        itcount = 0
        while True:
            itcount += 1
            if itcount % 8 == 0 and time.time() > end_time:
                return
            # ---------- best improving relocation move ----------
            best_move = None
            best_new_obj = cur_obj
            order_tasks = list(range(self.n))
            random.shuffle(order_tasks)
            for t in order_tasks:
                lo = 0
                hi = self.m - 1
                curs = station_of[t]
                for pr in self.pred[t]:
                    if station_of[pr] > lo:
                        lo = station_of[pr]
                for sc in self.succ[t]:
                    if station_of[sc] < hi:
                        hi = station_of[sc]
                for s in range(lo, hi + 1):
                    if s == curs:
                        continue
                    if popcount(masks[curs]) <= 1:
                        continue
                    tb = 1 << t
                    new_old = masks[curs] ^ tb
                    new_tgt = masks[s] | tb
                    c_old = self.station_cost(new_old)[0]
                    c_tgt = self.station_cost(new_tgt)[0]
                    new_obj = 0
                    for k in range(self.m):
                        if k == curs:
                            ck = c_old
                        elif k == s:
                            ck = c_tgt
                        else:
                            ck = costs[k]
                        if ck > new_obj:
                            new_obj = ck
                    if new_obj < best_new_obj - 1e-9:
                        best_new_obj = new_obj
                        best_move = (t, curs, s, new_old, new_tgt, c_old, c_tgt)
            if best_move is not None:
                t, curs, s, new_old, new_tgt, c_old, c_tgt = best_move
                masks[curs] = new_old
                masks[s] = new_tgt
                costs[curs] = c_old
                costs[s] = c_tgt
                station_of[t] = s
                cur_obj = best_new_obj
                if cur_obj < self.best_obj - 1e-9:
                    self.best_obj = cur_obj
                    self.best_masks = list(masks)
                    if logger:
                        _, sol = self.build_solution_dict(self.best_masks)
                        logger.log_solution(float(cur_obj), sol)
                continue
            # ---------- perturbation to escape local optimum ----------
            if time.time() > end_time:
                return
            perturbs = random.randint(1, 3)
            moved_any = False
            for _ in range(perturbs):
                t = random.randrange(self.n)
                curs = station_of[t]
                lo = 0
                hi = self.m - 1
                for pr in self.pred[t]:
                    lo = max(lo, station_of[pr])
                for sc in self.succ[t]:
                    hi = min(hi, station_of[sc])
                if popcount(masks[curs]) <= 1:
                    continue
                opts = [s for s in range(lo, hi + 1) if s != curs]
                if not opts:
                    continue
                s = random.choice(opts)
                tb = 1 << t
                masks[curs] ^= tb
                masks[s] |= tb
                costs[curs] = self.station_cost(masks[curs])[0]
                costs[s] = self.station_cost(masks[s])[0]
                station_of[t] = s
                moved_any = True
            cur_obj = max(costs) if costs else cur_obj
            if cur_obj < self.best_obj - 1e-9:
                self.best_obj = cur_obj
                self.best_masks = list(masks)
                if logger:
                    _, sol = self.build_solution_dict(self.best_masks)
                    logger.log_solution(float(cur_obj), sol)
            if not moved_any:
                return

    # ---------- MIP ----------
    def mip(self, deadline, logger):
        if not GUROBI_OK:
            return
        remaining = deadline - time.time() - 0.3
        if remaining < 1.0:
            return
        n, m = self.n, self.m
        try:
            model = gp.Model("ualb")
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(1.0, remaining)
            model.Params.OutputFlag = 0

            x = model.addVars(n, m, vtype=GRB.BINARY, name="x")
            ykeys = [(i, j, k) for k in range(m) for i in range(n) for j in range(n) if i != j]
            y = model.addVars(ykeys, vtype=GRB.BINARY, name="y")
            f = model.addVars(n, m, vtype=GRB.BINARY, name="f")
            l = model.addVars(n, m, vtype=GRB.BINARY, name="l")
            w = model.addVars(n, n, m, vtype=GRB.BINARY, name="w")
            u = model.addVars(n, m, vtype=GRB.CONTINUOUS, lb=0.0, ub=max(0, n - 1), name="u")
            C = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="C")

            model.setObjective(C, GRB.MINIMIZE)

            for i in range(n):
                model.addConstr(gp.quicksum(x[i, k] for k in range(m)) == 1)
            for k in range(m):
                model.addConstr(gp.quicksum(x[i, k] for i in range(n)) >= 1)
                model.addConstr(gp.quicksum(f[i, k] for i in range(n)) == 1)
                model.addConstr(gp.quicksum(l[i, k] for i in range(n)) == 1)
                for i in range(n):
                    model.addConstr(f[i, k] <= x[i, k])
                    model.addConstr(l[i, k] <= x[i, k])
                    model.addConstr(gp.quicksum(y[i, j, k] for j in range(n) if j != i)
                                    + l[i, k] == x[i, k])
                    model.addConstr(gp.quicksum(y[j, i, k] for j in range(n) if j != i)
                                    + f[i, k] == x[i, k])
                    model.addConstr(u[i, k] <= (n - 1) * x[i, k])
                    # MTZ subtour elimination
                    for j in range(n):
                        if i == j:
                            continue
                        model.addConstr(u[i, k] + 1 <= u[j, k] + n * (1 - y[i, j, k]))
                    # linearization of backward arc selection
                    for j in range(n):
                        model.addConstr(w[i, j, k] <= l[i, k])
                        model.addConstr(w[i, j, k] <= f[j, k])
                        model.addConstr(w[i, j, k] >= l[i, k] + f[j, k] - 1)

            for (a, b) in self.precs:
                model.addConstr(gp.quicksum(k * x[a, k] for k in range(m))
                                <= gp.quicksum(k * x[b, k] for k in range(m)))
                for k in range(m):
                    model.addConstr(u[a, k] + 1 <= u[b, k] + n * (2 - x[a, k] - x[b, k]))

            for k in range(m):
                expr = C >= (
                    gp.quicksum(self.p[i] * x[i, k] for i in range(n))
                    + gp.quicksum(self.Fwd[i][j] * y[i, j, k]
                                  for i in range(n) for j in range(n) if i != j)
                    + gp.quicksum(self.Bwd[i][j] * w[i, j, k]
                                  for i in range(n) for j in range(n))
                )
                model.addConstr(expr)

            solver_ref = self
            xlist = [x[i, k] for i in range(n) for k in range(m)]
            ylist = [y[i, j, k] for (i, j, k) in ykeys]
            flist = [f[i, k] for i in range(n) for k in range(m)]
            allvars = xlist + ylist + flist

            def callback(mo, where):
                if where == GRB.Callback.MIPSOL:
                    try:
                        vals = mo.cbGetSolution(allvars)
                        xv = vals[0:n * m]
                        yv = vals[n * m:n * m + len(ykeys)]
                        fv = vals[n * m + len(ykeys):]
                        masks = solver_ref._masks_from_vals(xv)
                        obj = 0
                        for k2 in range(m):
                            if masks[k2]:
                                obj = max(obj, solver_ref.station_cost(masks[k2])[0])
                        if masks and all(masks) and obj < solver_ref.best_obj - 1e-6:
                            solver_ref.best_obj = obj
                            solver_ref.best_masks = masks
                            if logger:
                                _, sol = solver_ref.build_solution_dict(masks)
                                logger.log_solution(float(obj), sol)
                    except Exception:
                        pass

            model.optimize(callback)

            if model.SolCount > 0:
                vals = model.getAttr("X", allvars)
                xv = vals[0:n * m]
                masks = self._masks_from_vals(xv)
                if all(masks):
                    obj = 0
                    for k in range(m):
                        obj = max(obj, self.station_cost(masks[k])[0])
                    if obj < self.best_obj - 1e-6:
                        self.best_obj = obj
                        self.best_masks = masks
                        if logger:
                            _, sol = self.build_solution_dict(masks)
                            logger.log_solution(float(obj), sol)
            model.dispose()
        except Exception:
            return

    def _masks_from_vals(self, xv):
        masks = [0] * self.m
        for i in range(self.n):
            for k in range(self.m):
                if xv[i * self.m + k] > 0.5:
                    masks[k] |= 1 << i
        return masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start = time.time()
    deadline = start + max(0.1, float(args.time_limit))

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as fh:
        inst = json.load(fh)

    solver = Solver(inst)

    # Decide whether a MIP attempt is worthwhile (small instances).
    use_mip = GUROBI_OK and solver.n <= 15 and (solver.n * solver.n * solver.m) <= 3200

    if use_mip:
        heur_end = start + max(0.4, (deadline - start) * 0.35)
    else:
        heur_end = deadline - 0.15

    solver.search(heur_end, logger)

    if use_mip and time.time() < deadline - 0.5:
        solver.mip(deadline, logger)

    if solver.best_masks is None:
        # extremely defensive fallback
        solver.best_masks = solver.initial_masks()

    obj, sol = solver.build_solution_dict(solver.best_masks)
    if logger and not solver.initialized:
        logger.log_solution(float(obj), sol)
    with open(args.solution_path, "w") as fh:
        json.dump(sol, fh)


if __name__ == "__main__":
    main()