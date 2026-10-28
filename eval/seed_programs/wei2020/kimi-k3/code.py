import argparse
import json
import time

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start = time.time()
    time_limit = max(1, int(args.time_limit))

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n = int(inst["n"])
    cap = int(inst["c"])
    weights = [int(w) for w in inst["weights"]]

    # ---- Optional incompatibility pairs (key name may vary across files) ----
    raw = None
    wanted = {"conflicts", "incompatible_pairs", "incompatibilities",
              "conflict_pairs", "forbidden_pairs", "incompatible", "edges"}
    for k, v in inst.items():
        if k.lower() in wanted and v:
            raw = v
            break

    conflicts = []
    if raw:
        pairs = [(int(p[0]), int(p[1])) for p in raw]
        flat = [v for p in pairs for v in p]
        if flat and min(flat) >= 1 and max(flat) == n:  # looks 1-based
            pairs = [(a - 1, b - 1) for a, b in pairs]
        seen = set()
        for a, b in pairs:
            if a == b or not (0 <= a < n and 0 <= b < n):
                continue
            if a > b:
                a, b = b, a
            if (a, b) not in seen:
                seen.add((a, b))
                conflicts.append((a, b))

    adj = [set() for _ in range(n)]
    for a, b in conflicts:
        adj[a].add(b)
        adj[b].add(a)
    degree = [len(s) for s in adj]

    def write_solution(bins):
        sol = {"objective_value": float(len(bins)),
               "bins": [sorted(b) for b in bins]}
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)

    if n == 0:
        write_solution([])
        return

    # ------------------------- Heuristic construction -------------------------
    def construct(order):
        bins, loads, bsets = [], [], []
        for i in order:
            wi = weights[i]
            ai = adj[i]
            best = -1
            best_rem = cap + 1
            for b in range(len(bins)):
                if loads[b] + wi <= cap and ai.isdisjoint(bsets[b]):
                    rem = cap - loads[b] - wi
                    if rem < best_rem:
                        best_rem = rem
                        best = b
            if best < 0:
                bins.append([i])
                loads.append(wi)
                bsets.append({i})
            else:
                bins[best].append(i)
                loads[best] += wi
                bsets[best].add(i)
        return bins, loads, bsets

    orders = [
        sorted(range(n), key=lambda i: (-weights[i], -degree[i])),
        sorted(range(n), key=lambda i: (-degree[i], -weights[i])),
        sorted(range(n), key=lambda i: -weights[i]),
    ]
    best = None
    for order in orders:
        b, l, s = construct(order)
        if best is None or len(b) < len(best[0]):
            best = (b, l, s)
    bins, loads, bsets = best

    # ------------------------- Heuristic bin reduction -------------------------
    def reduce_bins(bins, loads, bsets, deadline):
        while time.time() < deadline:
            changed = False
            for b in sorted(range(len(bins)), key=lambda z: loads[z]):
                if time.time() >= deadline:
                    break
                if not bins[b]:
                    continue
                items = sorted(bins[b], key=lambda i: -weights[i])
                moves = []
                ok = True
                for i in items:
                    wi = weights[i]
                    ai = adj[i]
                    tgt = -1
                    tgt_rem = cap + 1
                    for b2 in range(len(bins)):
                        if b2 == b or not bins[b2]:
                            continue
                        if loads[b2] + wi <= cap and ai.isdisjoint(bsets[b2]):
                            rem = cap - loads[b2] - wi
                            if rem < tgt_rem:
                                tgt_rem = rem
                                tgt = b2
                    if tgt < 0:
                        ok = False
                        break
                    loads[b] -= wi
                    bsets[b].discard(i)
                    loads[tgt] += wi
                    bsets[tgt].add(i)
                    bins[tgt].append(i)
                    moves.append((i, tgt))
                if ok:
                    bins[b] = []
                    bsets[b] = set()
                    loads[b] = 0
                    changed = True
                else:  # rollback
                    for i, b2 in reversed(moves):
                        wi = weights[i]
                        loads[b2] -= wi
                        bsets[b2].discard(i)
                        bins[b2].remove(i)
                        loads[b] += wi
                        bsets[b].add(i)
            if changed:
                nb, nl, ns = [], [], []
                for b in range(len(bins)):
                    if bins[b]:
                        nb.append(bins[b])
                        nl.append(loads[b])
                        ns.append(bsets[b])
                bins, loads, bsets = nb, nl, ns
            else:
                break
        return bins

    # Will a MIP model be a reasonable size?
    UB0 = len(bins)
    est_vars = sum(min(i, UB0 - 1) + 1 for i in range(n))
    est_conf = len(conflicts) * UB0
    mip_viable = est_vars <= 500000 and est_conf <= 800000

    frac = 0.15 if mip_viable else 0.45
    heur_deadline = start + min(frac * time_limit, 10.0)
    bins = reduce_bins(bins, loads, bsets, heur_deadline)

    # Relabel bins by smallest contained item index (valid symmetry-broken domain).
    bins = [sorted(b) for b in sorted(bins, key=lambda bl: min(bl))]
    best_bins = [list(b) for b in bins]
    best_obj = len(best_bins)

    if logger:
        logger.log_solution(float(best_obj),
                            {"objective_value": float(best_obj), "bins": best_bins})

    # ------------------------- Simple lower bound -------------------------
    total_w = sum(weights)
    lb = (total_w + cap - 1) // cap
    if conflicts and n <= 5000:
        clique = []
        for v in sorted(range(n), key=lambda i: -degree[i]):
            if all(v in adj[u] for u in clique):
                clique.append(v)
        lb = max(lb, len(clique))

    # ------------------------- MIP improvement -------------------------
    remaining = time_limit - (time.time() - start)
    if best_obj > 1 and best_obj > lb and mip_viable and remaining > 3.0:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            UB = best_obj
            m = gp.Model("bin_packing_conflicts")
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.OutputFlag = 0
            m.Params.TimeLimit = max(1.0, remaining - 1.5)

            keys = [(i, b) for i in range(n) for b in range(min(i, UB - 1) + 1)]
            x = m.addVars(keys, vtype=GRB.BINARY, name="x")
            y = m.addVars(UB, vtype=GRB.BINARY, name="y")

            m.setObjective(y.sum(), GRB.MINIMIZE)
            m.addConstrs((x.sum(i, "*") == 1 for i in range(n)), name="assign")
            m.addConstrs((gp.quicksum(weights[i] * x[i, b] for i in range(b, n)) <= cap * y[b]
                          for b in range(UB)), name="capacity")
            m.addConstrs((x[a, bb] + x[c_, bb] <= 1
                          for (a, c_) in conflicts
                          for bb in range(min(a, UB - 1) + 1)), name="conflict")
            m.addConstrs((y[b] >= y[b + 1] for b in range(UB - 1)), name="symmetry")
            if lb >= 2:
                m.addConstr(y.sum() >= lb, name="lower_bound")

            assign = {}
            for bidx, items in enumerate(best_bins):
                for i in items:
                    assign[i] = bidx
            for (i, b) in keys:
                x[i, b].Start = 1.0 if assign[i] == b else 0.0
            for b in range(UB):
                y[b].Start = 1.0

            incumbent = [best_obj]

            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    try:
                        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                        k = int(round(obj))
                        if k < incumbent[0]:
                            xv = model.cbGetSolution(x)
                            bk = {}
                            for (i, b), v in xv.items():
                                if v > 0.5:
                                    bk.setdefault(b, []).append(i)
                            sol_bins = [sorted(v) for _, v in sorted(bk.items())]
                            incumbent[0] = k
                            if logger:
                                logger.log_solution(float(k),
                                                    {"objective_value": float(k),
                                                     "bins": sol_bins})
                    except Exception:
                        pass

            m.optimize(callback)

            if m.SolCount > 0:
                k = int(round(m.ObjVal))
                if k < best_obj:
                    xv = m.getAttr("X", x)
                    bk = {}
                    for (i, b), v in xv.items():
                        if v > 0.5:
                            bk.setdefault(b, []).append(i)
                    best_bins = [sorted(v) for _, v in sorted(bk.items())]
                    best_obj = k
        except Exception:
            pass

    write_solution(best_bins)


if __name__ == "__main__":
    main()