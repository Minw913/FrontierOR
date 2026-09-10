import argparse
import json
import math
import time
import sys

from solution_logger import SolutionLogger


def read_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    W = int(data["parameters"]["W"])
    items = data["items"]
    types = []
    for it in items:
        types.append((int(it["item_type"]), int(it["weight"]), int(it["demand"])))
    return W, types


def ffd_pack(W, types):
    """First-fit decreasing packing by type. Returns list of dicts {type: count}."""
    order = sorted(types, key=lambda t: -t[1])
    bins = []       # list of dicts type->count
    residual = []   # residual capacity per bin
    for (tid, w, d) in order:
        if d <= 0:
            continue
        rem = d
        # try existing bins first
        for i in range(len(bins)):
            if rem == 0:
                break
            if residual[i] >= w:
                k = min(rem, residual[i] // w)
                if k > 0:
                    bins[i][tid] = bins[i].get(tid, 0) + k
                    residual[i] -= k * w
                    rem -= k
        # open new bins
        while rem > 0:
            cap_per_bin = W // w
            k = min(rem, cap_per_bin)
            bins.append({tid: k})
            residual.append(W - k * w)
            rem -= k
    return bins


def lower_bound(W, types):
    total = sum(w * d for (_, w, d) in types)
    lb1 = (total + W - 1) // W
    # Martello-Toth L2 bound
    best = lb1
    weights = sorted(set(w for (_, w, d) in types if d > 0))
    cand = [a for a in weights if a * 2 <= W]
    cand.append(0)
    for alpha in set(cand):
        N1 = 0
        N2 = 0
        free2 = 0
        S3 = 0
        for (_, w, d) in types:
            if d <= 0:
                continue
            if w > W - alpha:
                N1 += d
            elif 2 * w > W:
                N2 += d
                free2 += d * (W - w)
            elif w >= alpha:
                S3 += d * w
        extra = max(0, (S3 - free2 + W - 1) // W) if S3 > free2 else 0
        L = N1 + N2 + extra
        if L > best:
            best = L
    return best


def bins_to_solution(bins_dicts):
    out_bins = []
    for b in bins_dicts:
        lst = []
        for tid in sorted(b.keys()):
            lst.extend([tid] * b[tid])
        out_bins.append(lst)
    return {
        "objective_value": float(len(out_bins)),
        "num_bins": len(out_bins),
        "bins": out_bins,
    }


def solve_mip(W, types, ub_bins, lb, start_bins, time_left, logger):
    """Assignment-based MIP with symmetry breaking. Returns list of bin dicts or None."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return None

    m_types = len(types)
    B = ub_bins

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model("binpack", env=env)
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1.0, time_left)

    tids = [t[0] for t in types]
    ws = [t[1] for t in types]
    ds = [t[2] for t in types]

    x = {}
    for i in range(m_types):
        ub_x = min(ds[i], W // ws[i])
        for b in range(B):
            x[i, b] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=ub_x, name=f"x_{i}_{b}")
    y = {}
    for b in range(B):
        if b < lb:
            y[b] = model.addVar(vtype=GRB.BINARY, lb=1, ub=1, name=f"y_{b}")
        else:
            y[b] = model.addVar(vtype=GRB.BINARY, name=f"y_{b}")

    model.setObjective(gp.quicksum(y[b] for b in range(B)), GRB.MINIMIZE)

    for b in range(B):
        model.addConstr(gp.quicksum(ws[i] * x[i, b] for i in range(m_types)) <= W * y[b])
    for i in range(m_types):
        model.addConstr(gp.quicksum(x[i, b] for b in range(B)) == ds[i])
    # symmetry breaking
    for b in range(B - 1):
        model.addConstr(y[b] >= y[b + 1])

    # MIP start from FFD
    tid_to_idx = {tid: i for i, tid in enumerate(tids)}
    for b in range(B):
        y[b].Start = 1 if b < len(start_bins) else 0
        for i in range(m_types):
            x[i, b].Start = 0
    for b, bd in enumerate(start_bins):
        for tid, cnt in bd.items():
            x[tid_to_idx[tid], b].Start = cnt

    best_obj = [len(start_bins)]

    def callback(mdl, where):
        if where == GRB.Callback.MIPSOL:
            obj = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
            nb = int(round(obj))
            if nb < best_obj[0]:
                best_obj[0] = nb
                try:
                    xv = mdl.cbGetSolution([x[i, b] for i in range(m_types) for b in range(B)])
                    bins_d = []
                    for b in range(B):
                        bd = {}
                        for i in range(m_types):
                            v = int(round(xv[i * B + b]))
                            if v > 0:
                                bd[tids[i]] = v
                        if bd:
                            bins_d.append(bd)
                    if logger:
                        logger.log_solution(float(len(bins_d)), bins_to_solution(bins_d))
                except Exception:
                    if logger:
                        logger.log(float(nb))

    model.optimize(callback)

    if model.SolCount > 0:
        bins_d = []
        for b in range(B):
            bd = {}
            for i in range(m_types):
                v = int(round(x[i, b].X))
                if v > 0:
                    bd[tids[i]] = v
            if bd:
                bins_d.append(bd)
        # verify feasibility of demands (should hold by constraints)
        return bins_d
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    W, types = read_instance(args.instance_path)

    # Heuristic solution
    ffd_bins = ffd_pack(W, types)
    best_bins = ffd_bins
    sol = bins_to_solution(best_bins)
    if logger:
        logger.log_solution(sol["objective_value"], sol)

    lb = lower_bound(W, types)
    ub = len(best_bins)

    # Try exact MIP if there's a gap and problem size is tractable
    elapsed = time.time() - start_time
    time_left = args.time_limit - elapsed - 2.0
    m_types = len(types)
    if ub > lb and time_left > 3 and m_types * ub <= 300000 and ub <= 3000:
        mip_bins = solve_mip(W, types, ub, lb, ffd_bins, time_left, logger)
        if mip_bins is not None and len(mip_bins) < len(best_bins):
            best_bins = mip_bins
            sol = bins_to_solution(best_bins)
            if logger:
                logger.log_solution(sol["objective_value"], sol)

    sol = bins_to_solution(best_bins)
    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()