import argparse
import json
import math
import time

from solution_logger import SolutionLogger


def read_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    n = int(data["n"])
    c = int(data["c"])
    weights = [int(w) for w in data["weights"]]
    conflicts = []
    for key in ("conflicts", "incompatible_pairs", "incompatibilities",
                "pairs", "conflict_pairs", "incompatible"):
        if key in data and data[key]:
            for p in data[key]:
                a, b = int(p[0]), int(p[1])
                if a != b:
                    conflicts.append((a, b))
            break
    return n, c, weights, conflicts


def build_adj(n, conflicts):
    adj = [set() for _ in range(n)]
    for a, b in conflicts:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def first_fit(order, weights, c, adj):
    bins_items = []
    bins_load = []
    bins_set = []
    for i in order:
        w = weights[i]
        placed = False
        for j in range(len(bins_items)):
            if bins_load[j] + w <= c and not (adj[i] & bins_set[j]):
                bins_items[j].append(i)
                bins_load[j] += w
                bins_set[j].add(i)
                placed = True
                break
        if not placed:
            bins_items.append([i])
            bins_load.append(w)
            bins_set.append({i})
    return bins_items


def best_fit(order, weights, c, adj):
    bins_items = []
    bins_load = []
    bins_set = []
    for i in order:
        w = weights[i]
        best_j = -1
        best_resid = None
        for j in range(len(bins_items)):
            if bins_load[j] + w <= c and not (adj[i] & bins_set[j]):
                resid = c - bins_load[j] - w
                if best_resid is None or resid < best_resid:
                    best_resid = resid
                    best_j = j
        if best_j >= 0:
            bins_items[best_j].append(i)
            bins_load[best_j] += w
            bins_set[best_j].add(i)
        else:
            bins_items.append([i])
            bins_load.append(w)
            bins_set.append({i})
    return bins_items


def greedy_clique_lb(n, adj):
    # greedy clique on conflict graph -> lower bound on number of bins
    order = sorted(range(n), key=lambda v: -len(adj[v]))
    best = 1 if n > 0 else 0
    for start in order[:50]:
        clique = [start]
        cset = {start}
        cand = sorted(adj[start], key=lambda v: -len(adj[v]))
        for v in cand:
            if cset <= adj[v]:
                clique.append(v)
                cset.add(v)
        if len(clique) > best:
            best = len(clique)
    return best


def make_solution(bins_items):
    used = [sorted(b) for b in bins_items if b]
    return {
        "objective_value": float(len(used)),
        "bins": used,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    n, c, weights, conflicts = read_instance(args.instance_path)
    adj = build_adj(n, conflicts)

    # --- Heuristics ---
    order_desc = sorted(range(n), key=lambda i: (-weights[i], i))
    sol_ffd = first_fit(order_desc, weights, c, adj)
    sol_bfd = best_fit(order_desc, weights, c, adj)
    best_bins = sol_ffd if len(sol_ffd) <= len(sol_bfd) else sol_bfd
    ub = len(best_bins)

    if logger:
        logger.log_solution(float(ub), make_solution(best_bins))

    # --- Lower bound ---
    lb = max(math.ceil(sum(weights) / c), 1 if n > 0 else 0)
    if conflicts:
        lb = max(lb, greedy_clique_lb(n, adj))

    # write heuristic solution early as a safeguard
    with open(args.solution_path, "w") as f:
        json.dump(make_solution(best_bins), f)

    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed - 2.0

    # --- MIP improvement (only if worthwhile and tractable) ---
    if ub > lb and remaining > 3 and n * ub <= 400000:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            m = gp.Model("bpp")
            m.Params.OutputFlag = 0
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.TimeLimit = max(1.0, remaining)

            B = ub
            # items sorted descending by weight; item at position p may only
            # occupy bins 0..p (symmetry reduction)
            pos = {order_desc[p]: p for p in range(n)}

            x = {}
            for i in range(n):
                p = pos[i]
                for j in range(min(p + 1, B)):
                    x[i, j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
            y = [m.addVar(vtype=GRB.BINARY, name=f"y_{j}") for j in range(B)]

            m.setObjective(gp.quicksum(y), GRB.MINIMIZE)

            for i in range(n):
                p = pos[i]
                m.addConstr(gp.quicksum(x[i, j] for j in range(min(p + 1, B))) == 1)

            for j in range(B):
                m.addConstr(
                    gp.quicksum(weights[i] * x[i, j] for i in range(n) if (i, j) in x)
                    <= c * y[j]
                )

            for j in range(B - 1):
                m.addConstr(y[j] >= y[j + 1])

            for a, b in conflicts:
                for j in range(B):
                    if (a, j) in x and (b, j) in x:
                        m.addConstr(x[a, j] + x[b, j] <= 1)

            m.addConstr(gp.quicksum(y) >= lb)

            # warm start from heuristic
            assign = {}
            for j, items in enumerate(best_bins):
                for i in items:
                    assign[i] = j
            for (i, j), var in x.items():
                var.Start = 1.0 if assign.get(i) == j else 0.0
            for j in range(B):
                y[j].Start = 1.0

            xkeys = list(x.keys())
            xvars = [x[k] for k in xkeys]

            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    vals = model.cbGetSolution(xvars)
                    bins_tmp = [[] for _ in range(B)]
                    for (i, j), v in zip(xkeys, vals):
                        if v > 0.5:
                            bins_tmp[j].append(i)
                    sol = make_solution(bins_tmp)
                    if logger:
                        logger.log_solution(sol["objective_value"], sol)

            m.optimize(callback)

            if m.SolCount > 0 and m.ObjVal < ub - 0.5:
                bins_tmp = [[] for _ in range(B)]
                for (i, j), var in x.items():
                    if var.X > 0.5:
                        bins_tmp[j].append(i)
                new_bins = [b for b in bins_tmp if b]
                if len(new_bins) < len(best_bins):
                    best_bins = new_bins
        except Exception:
            pass

    final_sol = make_solution(best_bins)
    if logger:
        logger.log_solution(final_sol["objective_value"], final_sol)
    with open(args.solution_path, "w") as f:
        json.dump(final_sol, f)


if __name__ == "__main__":
    main()