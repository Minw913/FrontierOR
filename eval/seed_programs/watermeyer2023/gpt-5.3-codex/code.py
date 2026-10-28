import argparse
import json
import time
import bisect
from math import inf

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def compute_longest_from_source(num_nodes, arcs, source):
    """
    Compute longest-path distances from a fixed source for constraints:
        s[v] >= s[u] + w
    using Bellman-Ford on negated weights for shortest paths.
    Returns dist where dist[i] is longest known from source to i, or -inf if unreachable.
    If a positive cycle is reachable from source, returns None.
    """
    # shortest on weights -w
    dist = [inf] * num_nodes
    dist[source] = 0.0

    for _ in range(num_nodes - 1):
        updated = False
        for u, v, w in arcs:
            # shortest update on edge u->v with weight -w
            if dist[u] < inf and dist[v] > dist[u] - w:
                dist[v] = dist[u] - w
                updated = True
        if not updated:
            break

    # cycle check (positive cycle in original reachable from source)
    for u, v, w in arcs:
        if dist[u] < inf and dist[v] > dist[u] - w - 1e-12:
            return None

    longest = [-inf] * num_nodes
    for i in range(num_nodes):
        if dist[i] < inf:
            longest[i] = -dist[i]
    return longest


def compute_longest_to_sink(num_nodes, arcs, sink):
    """
    Compute longest path from each node to sink for constraints s[v] >= s[u] + w.
    Do longest-from-source on reversed arcs starting at sink.
    """
    rev = [(v, u, w) for (u, v, w) in arcs]
    return compute_longest_from_source(num_nodes, rev, sink)


class IncumbentLoggerCallback:
    def __init__(self, logger):
        self.logger = logger
        self.best = float("inf")

    def __call__(self, model, where):
        if self.logger is None:
            return
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < self.best:
                self.best = obj
                self.logger.log(int(round(obj)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    t0 = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # ---------- Read instance ----------
    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n_real = inst["num_real_activities"]
    n_nodes = inst["num_nodes"]  # includes 0 and n+1
    deadline = inst["deadline"]
    p = inst["processing_times"]
    temporal = inst["temporal_constraints"]
    resources = inst["resources"]

    start_node = 0
    end_node = n_real + 1

    arcs = []
    max_abs_delta = 0
    for c in temporal:
        u = int(c["from"])
        v = int(c["to"])
        d = int(c["delta"])
        arcs.append((u, v, d))
        max_abs_delta = max(max_abs_delta, abs(d))

    sum_p = sum(max(0, int(x)) for x in p)
    # safe fallback horizon width for potentially unconnected nodes
    H = int(deadline + (n_nodes + 2) * max_abs_delta + sum_p + 5)

    # ---------- Build rough time bounds ----------
    lb = [-H] * n_nodes
    ub = [deadline] * n_nodes

    lb_from_start = compute_longest_from_source(n_nodes, arcs, start_node)
    if lb_from_start is not None:
        for i in range(n_nodes):
            if lb_from_start[i] > -inf:
                lb[i] = max(lb[i], int(lb_from_start[i]))

    to_end = compute_longest_to_sink(n_nodes, arcs, end_node)
    if to_end is not None:
        for i in range(n_nodes):
            if to_end[i] > -inf:
                ub[i] = min(ub[i], int(deadline - to_end[i]))

    # hard-fix start and end range
    lb[start_node] = 0
    ub[start_node] = 0
    ub[end_node] = min(ub[end_node], deadline)

    # ensure consistent integer bounds fallback
    for i in range(n_nodes):
        if lb[i] > ub[i]:
            # keep as-is; model will detect infeasible. But avoid empty domains where x may be needed.
            pass

    # ---------- Decide activities needing start-time binaries ----------
    # only activities with potential nonzero partially-renewable consumption need x(i,t)
    needs_x = [False] * n_nodes
    resource_periods_sorted = {}
    for r in resources:
        rid = int(r["resource_id"])
        periods = sorted(set(int(tt) for tt in r["time_periods"]))
        resource_periods_sorted[rid] = periods
        dem = r["demands"]
        for i in range(n_nodes):
            if p[i] > 0 and dem[i] > 0 and len(periods) > 0:
                needs_x[i] = True

    # dummies with zero duration never consume anything
    needs_x[start_node] = False
    needs_x[end_node] = False

    # ---------- Build MIP ----------
    model = gp.Model("prcpsp_max")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1

    elapsed = time.time() - t0
    remaining = max(1, int(args.time_limit - elapsed))
    model.Params.TimeLimit = remaining

    # Vars: integer start times
    s = {}
    for i in range(n_nodes):
        # if inconsistent bounds, let gurobi carry infeasibility with a tiny valid interval
        lbi = lb[i]
        ubi = ub[i]
        if lbi > ubi:
            lbi, ubi = ubi, lbi  # avoid API errors; constraints will still force infeasibility if needed
        s[i] = model.addVar(vtype=GRB.INTEGER, lb=lbi, ub=ubi, name=f"s_{i}")

    model.addConstr(s[start_node] == 0, name="start_at_zero")
    model.addConstr(s[end_node] <= deadline, name="deadline")

    # temporal constraints
    for k, c in enumerate(temporal):
        u = int(c["from"])
        v = int(c["to"])
        d = int(c["delta"])
        model.addConstr(s[v] - s[u] >= d, name=f"temp_{k}")

    # start-time binary expansion only where needed for resource counting
    x = {}
    for i in range(n_nodes):
        if not needs_x[i]:
            continue
        lbi = int(lb[i])
        ubi = int(ub[i])
        if lbi > ubi:
            continue
        dom = range(lbi, ubi + 1)
        # if domain singleton, no binary needed; s fixed by equality
        if lbi == ubi:
            model.addConstr(s[i] == lbi, name=f"fix_s_{i}")
            continue
        for t in dom:
            x[(i, t)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{t}")
        model.addConstr(gp.quicksum(x[(i, t)] for t in dom) == 1, name=f"choose_{i}")
        model.addConstr(s[i] == gp.quicksum(t * x[(i, t)] for t in dom), name=f"link_{i}")

    # resource constraints
    def overlap_count(sorted_periods, start_t, dur):
        # count q in sorted_periods such that start_t < q <= start_t + dur
        if dur <= 0 or not sorted_periods:
            return 0
        lo = bisect.bisect_right(sorted_periods, start_t)
        hi = bisect.bisect_right(sorted_periods, start_t + dur)
        return max(0, hi - lo)

    for r in resources:
        rid = int(r["resource_id"])
        periods = resource_periods_sorted[rid]
        cap = int(r["capacity"])
        dem = r["demands"]

        expr = gp.LinExpr()
        const_term = 0

        for i in range(n_nodes):
            di = int(dem[i])
            if di <= 0 or p[i] <= 0 or len(periods) == 0:
                continue

            lbi = int(lb[i])
            ubi = int(ub[i])
            if lbi > ubi:
                continue

            if needs_x[i] and lbi < ubi and (i, lbi) in x:
                for t in range(lbi, ubi + 1):
                    var = x.get((i, t), None)
                    if var is None:
                        continue
                    cnt = overlap_count(periods, t, p[i])
                    if cnt:
                        expr.addTerms(di * cnt, var)
            else:
                # deterministic (either no x needed, or singleton domain)
                # take current fixed/implicit start:
                t_fix = lbi if lbi == ubi else None
                if t_fix is not None:
                    cnt = overlap_count(periods, t_fix, p[i])
                    if cnt:
                        const_term += di * cnt
                else:
                    # no x but non-singleton: this should only happen if needs_x=False (zero demand cases),
                    # so we can ignore safely.
                    pass

        model.addConstr(expr + const_term <= cap, name=f"res_{rid}")

    # objective: minimize project end start time
    model.setObjective(s[end_node], GRB.MINIMIZE)

    # ---------- Optimize ----------
    cb = IncumbentLoggerCallback(logger)
    try:
        model.optimize(cb)
    except gp.GurobiError:
        model.optimize()

    # ---------- Extract solution ----------
    start_times = [0] * n_nodes
    objective_value = 0

    if model.SolCount > 0:
        for i in range(n_nodes):
            val = s[i].X
            start_times[i] = int(round(val))
        objective_value = int(round(start_times[end_node]))
    else:
        # fallback (may be infeasible to original model, but guarantees schema output)
        for i in range(n_nodes):
            start_times[i] = int(lb[i] if lb[i] <= ub[i] else ub[i])
        start_times[start_node] = 0
        if end_node < n_nodes:
            start_times[end_node] = min(start_times[end_node], deadline)
        objective_value = int(start_times[end_node])

    sol = {
        "objective_value": int(objective_value),
        "start_times": [int(v) for v in start_times],
    }

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()