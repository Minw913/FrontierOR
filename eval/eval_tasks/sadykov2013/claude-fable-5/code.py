import json
import argparse
import time
import random
import math
import sys

from solution_logger import SolutionLogger


def read_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    n = data["num_items"]
    cap = data["bin_capacity"]
    items = data["items"]
    ids = [it["id"] for it in items]
    weights = [it["weight"] for it in items]
    id2idx = {ids[i]: i for i in range(n)}
    adj = [set() for _ in range(n)]
    for e in data.get("conflict_edges", []):
        a, b = e[0], e[1]
        if a == b:
            continue
        if a in id2idx and b in id2idx:
            ia, ib = id2idx[a], id2idx[b]
            adj[ia].add(ib)
            adj[ib].add(ia)
    return n, cap, ids, weights, adj


def first_fit(order, weights, adj, cap):
    """First-fit packing respecting conflicts. Returns list of [item_list, weight, item_set]."""
    bins = []
    for i in order:
        w = weights[i]
        ai = adj[i]
        placed = False
        for b in bins:
            if b[1] + w <= cap and not (ai & b[2]):
                b[0].append(i)
                b[1] += w
                b[2].add(i)
                placed = True
                break
        if not placed:
            bins.append([[i], w, {i}])
    return bins


def best_fit(order, weights, adj, cap):
    bins = []
    for i in order:
        w = weights[i]
        ai = adj[i]
        best_b = None
        best_slack = None
        for b in bins:
            if b[1] + w <= cap and not (ai & b[2]):
                slack = cap - b[1] - w
                if best_slack is None or slack < best_slack:
                    best_slack = slack
                    best_b = b
        if best_b is None:
            bins.append([[i], w, {i}])
        else:
            best_b[0].append(i)
            best_b[1] += w
            best_b[2].add(i)
    return bins


def greedy_clique_lb(adj, n, tries=20, rng=None):
    """Greedy max-clique lower bound on conflict graph."""
    if n == 0:
        return 0
    if rng is None:
        rng = random.Random(0)
    deg_order = sorted(range(n), key=lambda v: -len(adj[v]))
    best = 1
    seeds = deg_order[:tries]
    for s in seeds:
        clique = [s]
        cand = set(adj[s])
        while cand:
            # pick vertex in cand with max connections within cand
            v = max(cand, key=lambda u: len(adj[u] & cand))
            clique.append(v)
            cand &= adj[v]
        if len(clique) > best:
            best = len(clique)
    return best


def build_solution(bins_list, ids, weights):
    sol = {"objective_value": len(bins_list), "bins": {}}
    for k, items in enumerate(bins_list):
        sol["bins"][str(k)] = {
            "items": [ids[i] for i in items],
            "total_weight": int(sum(weights[i] for i in items)),
        }
    return sol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start = time.time()
    deadline = start + max(1, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    n, cap, ids, weights, adj = read_instance(args.instance_path)

    if n == 0:
        sol = {"objective_value": 0, "bins": {}}
        with open(args.solution_path, "w") as f:
            json.dump(sol, f, indent=2)
        if logger:
            logger.log_solution(0, sol)
        return

    total_w = sum(weights)
    lb_weight = max(1, math.ceil(total_w / cap)) if cap > 0 else n
    lb_clique = greedy_clique_lb(adj, n)
    LB = max(lb_weight, lb_clique)

    rng = random.Random(0)

    # -------- Phase 1: constructive heuristics --------
    best_bins = None  # list of item-index lists

    def record(bins_struct):
        nonlocal best_bins
        blist = [b[0] for b in bins_struct]
        if best_bins is None or len(blist) < len(best_bins):
            best_bins = blist
            sol = build_solution(best_bins, ids, weights)
            if logger:
                logger.log_solution(len(best_bins), sol)
            return True
        return False

    # deterministic orderings
    order_wd = sorted(range(n), key=lambda i: (-weights[i], -len(adj[i])))
    record(first_fit(order_wd, weights, adj, cap))
    order_dd = sorted(range(n), key=lambda i: (-len(adj[i]), -weights[i]))
    record(first_fit(order_dd, weights, adj, cap))
    record(best_fit(order_wd, weights, adj, cap))
    # combined score
    maxw = max(weights) if weights else 1
    maxd = max((len(a) for a in adj), default=1) or 1
    order_c = sorted(range(n),
                     key=lambda i: -(weights[i] / max(1, maxw) + len(adj[i]) / max(1, maxd)))
    record(first_fit(order_c, weights, adj, cap))

    # decide whether MIP is tractable
    num_edges = sum(len(a) for a in adj) // 2
    use_mip = True
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        use_mip = False

    if use_mip:
        # sort positions by weight desc for representative formulation
        pos_order = sorted(range(n), key=lambda i: -weights[i])
        pos_of = {itm: p for p, itm in enumerate(pos_order)}
        nvars = n * (n + 1) // 2
        conf_cons = 0
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    conf_cons += min(pos_of[i], pos_of[j]) + 1
        if nvars > 180000 or conf_cons > 600000 or n > 600:
            use_mip = False

    # randomized restart budget
    if use_mip:
        heur_end = min(deadline, start + max(2.0, 0.15 * args.time_limit))
    else:
        heur_end = deadline

    it = 0
    while time.time() < heur_end and len(best_bins) > LB:
        it += 1
        # perturbed weight-descending order
        noise = 0.05 + 0.5 * rng.random()
        order = sorted(range(n),
                       key=lambda i: -(weights[i] * (1.0 + noise * rng.random())
                                       + 0.01 * len(adj[i]) * rng.random()))
        if it % 3 == 0:
            bs = best_fit(order, weights, adj, cap)
        else:
            bs = first_fit(order, weights, adj, cap)
        record(bs)
        if it % 50 == 0 and time.time() >= heur_end:
            break

    # -------- Phase 2: exact MIP (representative formulation) --------
    if use_mip and len(best_bins) > LB and time.time() < deadline - 1:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            w_sorted = [weights[pos_order[p]] for p in range(n)]
            adj_pos = [set() for _ in range(n)]
            for i in range(n):
                for j in adj[i]:
                    adj_pos[pos_of[i]].add(pos_of[j])

            m = gp.Model("bppc")
            m.Params.OutputFlag = 0
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.TimeLimit = max(1.0, deadline - time.time() - 1.0)

            # x[i][b]: item at position i assigned to bin represented by position b (b <= i)
            x = {}
            for i in range(n):
                for b in range(i + 1):
                    if b < i and b in adj_pos[i]:
                        continue  # conflicting with representative -> impossible
                    x[(i, b)] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{b}")
            m.update()

            # each item assigned exactly once
            for i in range(n):
                m.addConstr(gp.quicksum(x[(i, b)] for b in range(i + 1) if (i, b) in x) == 1)

            # capacity and linking
            for b in range(n):
                if (b, b) not in x:
                    continue
                terms = [(w_sorted[i], x[(i, b)]) for i in range(b, n) if (i, b) in x]
                m.addConstr(gp.quicksum(c * v for c, v in terms) <= cap * x[(b, b)])
                for i in range(b + 1, n):
                    if (i, b) in x:
                        m.addConstr(x[(i, b)] <= x[(b, b)])

            # conflict constraints
            for i in range(n):
                for j in adj_pos[i]:
                    if j <= i:
                        continue
                    for b in range(i):  # b < i < j
                        if (i, b) in x and (j, b) in x:
                            m.addConstr(x[(i, b)] + x[(j, b)] <= x[(b, b)])

            obj = gp.quicksum(x[(b, b)] for b in range(n) if (b, b) in x)
            m.setObjective(obj, GRB.MINIMIZE)
            m.addConstr(obj >= LB)

            # warm start from best heuristic solution
            for key in x:
                x[key].Start = 0.0
            for items_in_bin in best_bins:
                positions = [pos_of[i] for i in items_in_bin]
                rep = min(positions)
                ok = True
                for p in positions:
                    if (p, rep) not in x:
                        ok = False
                        break
                if ok:
                    for p in positions:
                        x[(p, rep)].Start = 1.0

            var_keys = list(x.keys())
            var_list = [x[k] for k in var_keys]

            state = {"best": len(best_bins)}

            def cb(model, where):
                if where == GRB.Callback.MIPSOL:
                    objv = int(round(model.cbGet(GRB.Callback.MIPSOL_OBJ)))
                    if objv < state["best"]:
                        vals = model.cbGetSolution(var_list)
                        binmap = {}
                        for k, v in zip(var_keys, vals):
                            if v > 0.5:
                                i, b = k
                                binmap.setdefault(b, []).append(pos_order[i])
                        blist = list(binmap.values())
                        if len(blist) < state["best"]:
                            state["best"] = len(blist)
                            state["bins"] = blist
                            if logger:
                                logger.log_solution(len(blist),
                                                    build_solution(blist, ids, weights))

            m.optimize(cb)

            if m.SolCount > 0:
                objv = int(round(m.ObjVal))
                if objv < len(best_bins):
                    binmap = {}
                    for k in var_keys:
                        if x[k].X > 0.5:
                            i, b = k
                            binmap.setdefault(b, []).append(pos_order[i])
                    blist = list(binmap.values())
                    if len(blist) < len(best_bins):
                        best_bins = blist
                        if logger:
                            logger.log_solution(len(best_bins),
                                                build_solution(best_bins, ids, weights))
            if "bins" in state and len(state["bins"]) < len(best_bins):
                best_bins = state["bins"]
        except Exception:
            pass  # fall back to heuristic solution

    # -------- Final output --------
    sol = build_solution(best_bins, ids, weights)
    if logger:
        logger.log_solution(len(best_bins), sol)
    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()