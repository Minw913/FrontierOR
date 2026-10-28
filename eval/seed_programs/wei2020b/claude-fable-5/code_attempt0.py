import argparse
import json
import math
import random
import time

from solution_logger import SolutionLogger


def read_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    n = data["n"]
    W = data["W"]
    V = data["V"]
    items = [(int(it["w"]), int(it["v"])) for it in data["items"]]
    return n, W, V, items


def build_solution_dict(bins, items):
    out_bins = []
    for lst in bins:
        tw = sum(items[i][0] for i in lst)
        tv = sum(items[i][1] for i in lst)
        out_bins.append({
            "items": [int(i) for i in lst],
            "total_weight": int(tw),
            "total_volume": int(tv),
        })
    return {"objective_value": len(bins), "bins": out_bins}


def pack(order, items, W, V, mode="ff"):
    """Pack items in given order. mode: 'ff' first fit, 'bf' best fit."""
    bins = []  # [rw, rv, item_list]
    for i in order:
        w, v = items[i]
        best = -1
        if mode == "ff":
            for b in range(len(bins)):
                if w <= bins[b][0] and v <= bins[b][1]:
                    best = b
                    break
        else:
            best_score = None
            for b in range(len(bins)):
                rw, rv = bins[b][0], bins[b][1]
                if w <= rw and v <= rv:
                    s = (rw - w) / W + (rv - v) / V
                    if best_score is None or s < best_score:
                        best_score = s
                        best = b
        if best == -1:
            bins.append([W - w, V - v, [i]])
        else:
            bins[best][0] -= w
            bins[best][1] -= v
            bins[best][2].append(i)
    return [b[2] for b in bins]


def try_eliminate_bins(bins, items, W, V, deadline):
    """Try to empty the least-filled bins by moving their items elsewhere."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # compute residuals
        res = []
        for lst in bins:
            tw = sum(items[i][0] for i in lst)
            tv = sum(items[i][1] for i in lst)
            res.append((tw, tv))
        # order bins by fill ascending (candidates to eliminate)
        order = sorted(range(len(bins)),
                       key=lambda b: res[b][0] / W + res[b][1] / V)
        for b in order:
            if time.time() >= deadline:
                break
            # try to relocate all items of bin b into other bins
            others = [[W - res[k][0], V - res[k][1], k] for k in range(len(bins)) if k != b]
            moves = []
            ok = True
            for i in sorted(bins[b], key=lambda i: -(items[i][0] / W + items[i][1] / V)):
                w, v = items[i]
                placed = False
                for o in others:
                    if w <= o[0] and v <= o[1]:
                        o[0] -= w
                        o[1] -= v
                        moves.append((i, o[2]))
                        placed = True
                        break
                if not placed:
                    ok = False
                    break
            if ok:
                for i, k in moves:
                    bins[k].append(i)
                bins.pop(b)
                improved = True
                break
    return bins


def heuristic_solve(items, W, V, n, deadline, logger):
    keys = [
        lambda i: -(items[i][0] / W + items[i][1] / V),
        lambda i: -max(items[i][0] / W, items[i][1] / V),
        lambda i: -items[i][0],
        lambda i: -items[i][1],
        lambda i: -(items[i][0] / W * items[i][1] / V),
    ]
    best = None
    for key in keys:
        order = sorted(range(n), key=key)
        for mode in ("ff", "bf"):
            if time.time() >= deadline:
                break
            bins = pack(order, items, W, V, mode)
            if best is None or len(bins) < len(best):
                best = bins
                if logger:
                    logger.log_solution(len(best), build_solution_dict(best, items))
        if time.time() >= deadline:
            break

    lb = max(int(math.ceil(sum(w for w, _ in items) / W)),
             int(math.ceil(sum(v for _, v in items) / V)), 1)

    # randomized restarts
    rng = random.Random(0)
    base = sorted(range(n), key=keys[0])
    while time.time() < deadline and len(best) > lb:
        order = base[:]
        # small perturbation: swap random nearby pairs
        for _ in range(max(1, n // 10)):
            a = rng.randrange(n)
            b = min(n - 1, a + rng.randrange(1, 6))
            order[a], order[b] = order[b], order[a]
        bins = pack(order, items, W, V, "bf")
        if len(bins) < len(best):
            best = bins
            if logger:
                logger.log_solution(len(best), build_solution_dict(best, items))

    # elimination local search
    if time.time() < deadline and len(best) > lb:
        before = len(best)
        best = try_eliminate_bins(best, items, W, V, deadline)
        if len(best) < before and logger:
            logger.log_solution(len(best), build_solution_dict(best, items))

    return best, lb


def mip_solve(items, W, V, n, ub_bins, lb, time_limit, logger):
    """Assignment MIP with warm start. Returns improved bins or None."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None

    B = len(ub_bins)
    if B <= lb:
        return None
    # limit model size
    if n * B > 400000:
        return None

    model = gp.Model("vbp")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1.0, time_limit)

    x = model.addVars(n, B, vtype=GRB.BINARY, name="x")
    y = model.addVars(B, vtype=GRB.BINARY, name="y")

    for i in range(n):
        model.addConstr(gp.quicksum(x[i, b] for b in range(B)) == 1)
    for b in range(B):
        model.addConstr(gp.quicksum(items[i][0] * x[i, b] for i in range(n)) <= W * y[b])
        model.addConstr(gp.quicksum(items[i][1] * x[i, b] for i in range(n)) <= V * y[b])
    for b in range(B - 1):
        model.addConstr(y[b] >= y[b + 1])

    model.setObjective(gp.quicksum(y[b] for b in range(B)), GRB.MINIMIZE)
    model.addConstr(gp.quicksum(y[b] for b in range(B)) >= lb)

    # warm start
    for b, lst in enumerate(ub_bins):
        y[b].Start = 1
        for i in lst:
            x[i, b].Start = 1

    state = {"best_obj": B, "best_bins": None}

    def cb(m, where):
        if where == GRB.Callback.MIPSOL:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            obj_i = int(round(obj))
            if obj_i < state["best_obj"]:
                xv = m.cbGetSolution(x)
                sol_bins = [[] for _ in range(B)]
                for i in range(n):
                    for b in range(B):
                        if xv[i, b] > 0.5:
                            sol_bins[b].append(i)
                            break
                sol_bins = [lst for lst in sol_bins if lst]
                state["best_obj"] = len(sol_bins)
                state["best_bins"] = sol_bins
                if logger:
                    logger.log_solution(len(sol_bins),
                                        build_solution_dict(sol_bins, items))

    model.optimize(cb)

    if model.SolCount > 0:
        obj = int(round(model.ObjVal))
        if obj < B:
            sol_bins = [[] for _ in range(B)]
            for i in range(n):
                for b in range(B):
                    if x[i, b].X > 0.5:
                        sol_bins[b].append(i)
                        break
            sol_bins = [lst for lst in sol_bins if lst]
            return sol_bins
    if state["best_bins"] is not None and len(state["best_bins"]) < B:
        return state["best_bins"]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    n, W, V, items = read_instance(args.instance_path)
    total_deadline = start + max(1, args.time_limit) - 1.0

    # heuristic phase: small fraction of the time budget
    heur_deadline = min(total_deadline, start + max(2.0, 0.15 * args.time_limit))
    best_bins, lb = heuristic_solve(items, W, V, n, heur_deadline, logger)

    # MIP phase if not proven optimal
    if len(best_bins) > lb:
        remaining = total_deadline - time.time()
        if remaining > 2.0:
            improved = mip_solve(items, W, V, n, best_bins, lb, remaining, logger)
            if improved is not None and len(improved) < len(best_bins):
                best_bins = improved
                if logger:
                    logger.log_solution(len(best_bins),
                                        build_solution_dict(best_bins, items))

    solution = build_solution_dict(best_bins, items)
    if logger:
        logger.log_solution(solution["objective_value"], solution)
    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()