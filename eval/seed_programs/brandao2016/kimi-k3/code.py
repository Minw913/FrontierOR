import argparse
import json
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Bin packing with item-type demands")
    parser.add_argument("--instance_path", required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", required=True,
                        help="Path where the solution JSON is written.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", default=None,
                        help="Optional path to a JSONL file for incumbent logging.")
    return parser.parse_args()


def ffd(types, weights, demands, W, order):
    """Aggregated first-fit decreasing. Returns (bin_counts, bin_remaining)."""
    bin_counts = []
    bin_rem = []
    for t in order:
        w = weights[t]
        copies = demands[t]
        for i in range(len(bin_rem)):
            if copies == 0:
                break
            if bin_rem[i] >= w:
                k = bin_rem[i] // w
                take = copies if copies < k else k
                bin_counts[i][t] = bin_counts[i].get(t, 0) + take
                bin_rem[i] -= take * w
                copies -= take
        per_full = W // w
        while copies > 0:
            take = copies if copies < per_full else per_full
            bin_counts.append({t: take})
            bin_rem.append(W - take * w)
            copies -= take
    return bin_counts, bin_rem


def bfd(types, weights, demands, W, order):
    """Aggregated best-fit decreasing. Returns (bin_counts, bin_remaining)."""
    bin_counts = []
    bin_rem = []
    for t in order:
        w = weights[t]
        copies = demands[t]
        while copies > 0:
            best_i = -1
            best_rem = None
            for i in range(len(bin_rem)):
                r = bin_rem[i]
                if r >= w and (best_rem is None or r < best_rem):
                    best_rem = r
                    best_i = i
            if best_i < 0:
                per_full = W // w
                take = copies if copies < per_full else per_full
                bin_counts.append({t: take})
                bin_rem.append(W - take * w)
                copies -= take
            else:
                k = bin_rem[best_i] // w
                take = copies if copies < k else k
                bin_counts[best_i][t] = bin_counts[best_i].get(t, 0) + take
                bin_rem[best_i] -= take * w
                copies -= take
    return bin_counts, bin_rem


def try_eliminate(bin_counts, bin_rem, weights, W, deadline):
    """Repeatedly try to empty the least-loaded bin into the other bins."""
    while time.time() < deadline:
        improved = False
        order = sorted(range(len(bin_counts)), key=lambda i: W - bin_rem[i])
        for bi in order:
            if time.time() >= deadline:
                return
            items = sorted(bin_counts[bi].items(), key=lambda kv: -weights[kv[0]])
            used = [0] * len(bin_counts)
            placement = []
            feasible = True
            for t, c in items:
                w = weights[t]
                left = c
                for j in range(len(bin_counts)):
                    if j == bi:
                        continue
                    cap = bin_rem[j] - used[j]
                    if cap >= w:
                        k = cap // w
                        take = left if left < k else k
                        used[j] += take * w
                        placement.append((j, t, take))
                        left -= take
                        if left == 0:
                            break
                if left > 0:
                    feasible = False
                    break
            if feasible:
                for j, t, take in placement:
                    bin_counts[j][t] = bin_counts[j].get(t, 0) + take
                    bin_rem[j] -= take * weights[t]
                bin_counts.pop(bi)
                bin_rem.pop(bi)
                improved = True
                break
        if not improved:
            return


def counts_to_solution(bin_counts):
    bins = []
    for cnt in bin_counts:
        bl = []
        for t in sorted(cnt):
            bl.extend([t] * cnt[t])
        bins.append(bl)
    return {
        "objective_value": float(len(bins)),
        "num_bins": len(bins),
        "bins": bins,
    }


def gurobi_solve(types, weights, demands, W, bin_counts, time_budget,
                 logger, current_best):
    """Aggregated assignment MIP over at most len(bin_counts) bins."""
    import gurobipy as gp
    from gurobipy import GRB

    B = len(bin_counts)
    loads0 = [sum(weights[t] * c for t, c in cnt.items())
              for cnt in bin_counts]
    orderb = sorted(range(B), key=lambda b: -loads0[b])
    sorted_counts = [bin_counts[b] for b in orderb]

    model = gp.Model("bin_packing")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = max(time_budget, 0.1)

    z = {}
    for t in types:
        ub = min(demands[t], W // weights[t])
        for b in range(B):
            z[(t, b)] = model.addVar(lb=0, ub=ub, vtype=GRB.INTEGER,
                                     name="z_%d_%d" % (t, b))
    y = [model.addVar(vtype=GRB.BINARY, name="y_%d" % b) for b in range(B)]

    load_exprs = []
    for b in range(B):
        le = gp.quicksum(weights[t] * z[(t, b)] for t in types)
        load_exprs.append(le)
        model.addConstr(le <= W * y[b])
    for t in types:
        model.addConstr(gp.quicksum(z[(t, b)] for b in range(B)) == demands[t])
    # Symmetry breaking: bins sorted by non-increasing load.
    for b in range(B - 1):
        model.addConstr(y[b] >= y[b + 1])
        model.addConstr(load_exprs[b] >= load_exprs[b + 1])

    model.setObjective(gp.quicksum(y), GRB.MINIMIZE)

    # MIP start from the incumbent heuristic solution.
    for b in range(B):
        y[b].Start = 1.0
    for b, cnt in enumerate(sorted_counts):
        for t, c in cnt.items():
            z[(t, b)].Start = float(c)

    def make_solution(zvals):
        out_counts = []
        for b in range(B):
            cnt = {}
            for t in types:
                v = zvals.get((t, b), 0.0)
                iv = int(round(v))
                if iv > 0:
                    cnt[t] = iv
            if cnt:
                out_counts.append(cnt)
        return counts_to_solution(out_counts)

    state = {"best": current_best}

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            except gp.GurobiError:
                return
            nb = int(round(obj))
            if nb < state["best"]:
                try:
                    zvals = model.cbGetSolution(z)
                    sol = make_solution(zvals)
                except gp.GurobiError:
                    return
                if sol["num_bins"] < state["best"]:
                    state["best"] = sol["num_bins"]
                    if logger:
                        logger.log_solution(sol["objective_value"], sol)

    model.optimize(callback)

    if model.SolCount > 0:
        try:
            zvals = model.getAttr("X", z)
            return make_solution(zvals)
        except gp.GurobiError:
            return None
    return None


def main():
    args = parse_args()
    start = time.time()
    deadline = start + max(args.time_limit, 0)

    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path) as f:
        data = json.load(f)

    W = int(data["parameters"]["W"])
    types, weights, demands = [], {}, {}
    for it in data.get("items", []):
        t = int(it["item_type"])
        w = int(it["weight"])
        d = int(it["demand"])
        if d > 0 and w > 0:
            types.append(t)
            weights[t] = w
            demands[t] = d

    if not types:
        best_sol = {"objective_value": 0.0, "num_bins": 0, "bins": []}
        if logger:
            logger.log_solution(0.0, best_sol)
        with open(args.solution_path, "w") as f:
            json.dump(best_sol, f)
        return

    order = sorted(types, key=lambda t: -weights[t])

    # ---- Constructive heuristics ------------------------------------------
    ffd_counts, ffd_rem = ffd(types, weights, demands, W, order)
    bfd_counts, bfd_rem = bfd(types, weights, demands, W, order)
    if len(bfd_counts) < len(ffd_counts):
        bin_counts, bin_rem = bfd_counts, bfd_rem
    else:
        bin_counts, bin_rem = ffd_counts, ffd_rem

    # ---- Cheap local improvement: bin elimination -------------------------
    elim_budget = min(5.0, 0.2 * max(deadline - time.time(), 0.0))
    if elim_budget > 0.05:
        try_eliminate(bin_counts, bin_rem, weights, W,
                      time.time() + elim_budget)

    best_sol = counts_to_solution(bin_counts)
    if logger:
        logger.log_solution(best_sol["objective_value"], best_sol)

    # ---- Lower bound -------------------------------------------------------
    total_weight = sum(weights[t] * demands[t] for t in types)
    lb = -(-total_weight // W)

    # ---- Gurobi improvement pass ------------------------------------------
    remaining = deadline - time.time() - 0.5
    if (best_sol["num_bins"] > lb and remaining > 0.5
            and len(types) * best_sol["num_bins"] <= 300000
            and best_sol["num_bins"] <= 5000):
        try:
            result = gurobi_solve(types, weights, demands, W, bin_counts,
                                  remaining, logger, best_sol["num_bins"])
            if result is not None and result["num_bins"] < best_sol["num_bins"]:
                best_sol = result
                if logger:
                    logger.log_solution(best_sol["objective_value"], best_sol)
        except Exception:
            pass

    with open(args.solution_path, "w") as f:
        json.dump(best_sol, f)


if __name__ == "__main__":
    main()