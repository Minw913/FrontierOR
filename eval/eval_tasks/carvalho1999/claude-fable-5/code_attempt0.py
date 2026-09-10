import argparse
import json
import math
import time
import bisect
from collections import Counter


def ffd_pack(items, W):
    """First-Fit Decreasing using a max segment tree for O(n log n)."""
    order = sorted(items, reverse=True)
    n = len(order)
    if n == 0:
        return []
    M = 1
    while M < n:
        M *= 2
    tree = [0] * (2 * M)
    for i in range(n):
        tree[M + i] = W
    for i in range(M - 1, 0, -1):
        tree[i] = max(tree[2 * i], tree[2 * i + 1])
    contents = [[] for _ in range(n)]
    for s in order:
        node = 1
        # find leftmost bin with remaining >= s (always exists)
        while node < M:
            node *= 2
            if tree[node] < s:
                node += 1
        idx = node - M
        contents[idx].append(s)
        tree[node] -= s
        node //= 2
        while node:
            tree[node] = max(tree[2 * node], tree[2 * node + 1])
            node //= 2
    return [c for c in contents if c]


def lower_bound(items, W):
    """L1 (area) and L2 (Martello-Toth) lower bounds."""
    if not items:
        return 0
    total = sum(items)
    lb = (total + W - 1) // W
    ss = sorted(items)
    n = len(ss)
    pre = [0]
    for s in ss:
        pre.append(pre[-1] + s)
    half = W // 2  # s > W/2 <=> s > half (integers)
    j = bisect.bisect_right(ss, half)  # first idx with s > W/2
    alphas = sorted(set(s for s in ss if 2 * s <= W))
    alphas.append(0)
    for alpha in alphas:
        i1 = bisect.bisect_right(ss, W - alpha)  # first idx with s > W-alpha
        n1 = n - i1
        jj = min(j, i1)
        n2 = i1 - jj
        sumJ2 = pre[i1] - pre[jj]
        a = bisect.bisect_left(ss, alpha) if alpha > 0 else 0
        a = min(a, jj)
        sumJ3 = pre[jj] - pre[a]
        free = n2 * W - sumJ2
        extra = sumJ3 - free
        add = (extra + W - 1) // W if extra > 0 else 0
        L = n1 + n2 + add
        if L > lb:
            lb = L
    return lb


def build_solution(bins):
    ba = []
    for b in bins:
        ba.append({"items": list(b), "total_size": sum(b)})
    return {
        "objective_value": len(bins),
        "bin_assignments": ba,
        "num_bins": len(bins),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    t0 = time.time()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)
    W = int(inst["bin_capacity"])
    items = [int(s) for s in inst["items"]]

    if len(items) == 0:
        sol = build_solution([])
        if logger:
            logger.log_solution(0, sol)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f, indent=2)
        return

    # Items that occupy a full bin by themselves
    fixed_bins = [[s] for s in items if s == W]
    rem_items = [s for s in items if s < W]

    ffd_rem = ffd_pack(rem_items, W)
    best_bins = fixed_bins + ffd_rem
    best_sol = build_solution(best_bins)
    if logger:
        logger.log_solution(best_sol["objective_value"], best_sol)

    lb_rem = lower_bound(rem_items, W)
    lb_total = len(fixed_bins) + lb_rem

    # If FFD is provably optimal, done.
    if len(best_bins) <= lb_total or len(ffd_rem) == 0:
        with open(args.solution_path, "w") as f:
            json.dump(best_sol, f, indent=2)
        return

    # ---- MIP refinement with Gurobi ----
    cnt = Counter(rem_items)
    types = sorted(cnt.keys(), reverse=True)
    T = len(types)
    B = len(ffd_rem)  # upper bound from FFD

    elapsed = time.time() - t0
    remaining = args.time_limit - elapsed - 1.0
    if remaining <= 1.0 or T * B > 300000:
        with open(args.solution_path, "w") as f:
            json.dump(best_sol, f, indent=2)
        return

    best_container = {"obj": len(best_bins), "sol": best_sol}

    try:
        import gurobipy as gp
        from gurobipy import GRB

        m = gp.Model("binpack")
        m.Params.OutputFlag = 0
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.NumericFocus = 0
        m.Params.Threads = 1
        m.Params.TimeLimit = max(1.0, remaining)

        z = m.addVars(B, vtype=GRB.BINARY, name="z")
        y = {}
        for ti, s in enumerate(types):
            ub = min(cnt[s], W // s)
            for b in range(B):
                y[(ti, b)] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=ub,
                                      name=f"y_{ti}_{b}")
        # demand
        for ti, s in enumerate(types):
            m.addConstr(gp.quicksum(y[(ti, b)] for b in range(B)) == cnt[s])
        # capacity
        for b in range(B):
            m.addConstr(gp.quicksum(types[ti] * y[(ti, b)] for ti in range(T))
                        <= W * z[b])
        # symmetry breaking
        for b in range(B - 1):
            m.addConstr(z[b] >= z[b + 1])
        # valid lower bound
        m.addConstr(gp.quicksum(z[b] for b in range(B)) >= lb_rem)

        m.setObjective(gp.quicksum(z[b] for b in range(B)), GRB.MINIMIZE)

        # warm start from FFD
        tindex = {s: i for i, s in enumerate(types)}
        for b in range(B):
            z[b].Start = 1.0
            c = Counter(ffd_rem[b])
            for s, k in c.items():
                y[(tindex[s], b)].Start = float(k)

        yvars = [y[(ti, b)] for ti in range(T) for b in range(B)]

        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                vals = model.cbGetSolution(yvars)
                bins_local = []
                for b in range(B):
                    cur = []
                    for ti in range(T):
                        k = int(round(vals[ti * B + b]))
                        if k > 0:
                            cur.extend([types[ti]] * k)
                    if cur:
                        bins_local.append(cur)
                total_bins = fixed_bins + bins_local
                obj = len(total_bins)
                if obj < best_container["obj"]:
                    sol = build_solution(total_bins)
                    best_container["obj"] = obj
                    best_container["sol"] = sol
                    if logger:
                        logger.log_solution(obj, sol)

        m.optimize(callback)

        # extract final solution if available and better
        if m.SolCount > 0:
            bins_local = []
            for b in range(B):
                cur = []
                for ti in range(T):
                    k = int(round(y[(ti, b)].X))
                    if k > 0:
                        cur.extend([types[ti]] * k)
                if cur:
                    bins_local.append(cur)
            total_bins = fixed_bins + bins_local
            if len(total_bins) < best_container["obj"]:
                sol = build_solution(total_bins)
                best_container["obj"] = len(total_bins)
                best_container["sol"] = sol
                if logger:
                    logger.log_solution(len(total_bins), sol)
    except Exception:
        pass

    with open(args.solution_path, "w") as f:
        json.dump(best_container["sol"], f, indent=2)


if __name__ == "__main__":
    main()