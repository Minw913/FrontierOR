import argparse
import json
import time
import bisect
import sys

from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB

NEG_INF = float("-inf")


def compute_time_windows(n_nodes, arcs, deadline, proc):
    # ES: longest path from node 0 with arc weights delta
    es = [NEG_INF] * n_nodes
    es[0] = 0
    for _ in range(n_nodes):
        changed = False
        for (i, j, d) in arcs:
            if es[i] != NEG_INF and es[i] + d > es[j]:
                es[j] = es[i] + d
                changed = True
        if not changed:
            break
    for j in range(n_nodes):
        if es[j] == NEG_INF:
            es[j] = 0
        es[j] = max(0, int(es[j]))

    # L[j]: longest path from j to node n_nodes-1
    end = n_nodes - 1
    L = [NEG_INF] * n_nodes
    L[end] = 0
    for _ in range(n_nodes):
        changed = False
        for (i, j, d) in arcs:
            if L[j] != NEG_INF and d + L[j] > L[i]:
                L[i] = d + L[j]
                changed = True
        if not changed:
            break
    ls = [0] * n_nodes
    for j in range(n_nodes):
        if L[j] == NEG_INF:
            ls[j] = deadline - proc[j]
        else:
            ls[j] = deadline - int(L[j])
        ls[j] = min(ls[j], deadline - proc[j])
    ls[0] = 0
    return es, ls


def resource_usage(t, pj, periods_sorted):
    # number of periods p in sorted list with t < p <= t + pj
    if pj <= 0:
        return 0
    lo = bisect.bisect_right(periods_sorted, t)
    hi = bisect.bisect_right(periods_sorted, t + pj)
    return hi - lo


def check_resource_feasible(starts, proc, resources_data, n_nodes):
    for r in resources_data:
        periods = r["_sorted_periods"]
        dem = r["demands"]
        cap = r["capacity"]
        total = 0
        for j in range(n_nodes):
            if dem[j] > 0 and proc[j] > 0:
                total += dem[j] * resource_usage(starts[j], proc[j], periods)
        if total > cap:
            return False
    return True


def main():
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n_nodes = inst["num_nodes"]
    deadline = int(inst["deadline"])
    proc = [int(p) for p in inst["processing_times"]]
    end = n_nodes - 1

    arcs = []
    for tc in inst["temporal_constraints"]:
        arcs.append((int(tc["from"]), int(tc["to"]), int(tc["delta"])))

    resources_data = inst.get("resources", []) or []
    for r in resources_data:
        r["_sorted_periods"] = sorted(set(int(p) for p in r["time_periods"]))

    es, ls = compute_time_windows(n_nodes, arcs, deadline, proc)

    best_starts = None
    best_obj = None

    # Heuristic: earliest-start schedule (temporal-optimal); check resources
    es_sched = list(es)
    temporal_ok = all(es_sched[j] - es_sched[i] >= d for (i, j, d) in arcs)
    if temporal_ok and es_sched[0] == 0 and es_sched[end] <= deadline:
        if check_resource_feasible(es_sched, proc, resources_data, n_nodes):
            best_starts = list(es_sched)
            best_obj = es_sched[end]
            if logger:
                logger.log_solution(best_obj, {
                    "objective_value": int(best_obj),
                    "start_times": [int(s) for s in best_starts],
                })

    # Build time-indexed MIP
    infeasible_window = any(es[j] > ls[j] for j in range(n_nodes))

    model = gp.Model("prcpsp")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    remaining = args.time_limit - (time.time() - t0) - 2.0
    if remaining < 5:
        remaining = max(1.0, args.time_limit - (time.time() - t0) - 0.5)
    model.Params.TimeLimit = max(1.0, remaining)

    x = {}          # (j, t) -> var
    times_of = {}   # j -> list of t
    if not infeasible_window:
        for j in range(n_nodes):
            tlist = list(range(es[j], ls[j] + 1))
            times_of[j] = tlist
            for t in tlist:
                x[(j, t)] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{t}")

        for j in range(n_nodes):
            model.addConstr(gp.quicksum(x[(j, t)] for t in times_of[j]) == 1)

        # start time expressions
        S = {j: gp.quicksum(t * x[(j, t)] for t in times_of[j]) for j in range(n_nodes)}

        # decide precedence formulation size
        disagg_count = 0
        for (i, j, d) in arcs:
            lo = es[j]
            hi = min(ls[j], ls[i] + d - 1)
            if hi >= lo:
                disagg_count += hi - lo + 1
        use_disagg = disagg_count <= 400000

        for (i, j, d) in arcs:
            if use_disagg:
                hi = min(ls[j], ls[i] + d - 1)
                lo = es[j]
                for t in range(lo, hi + 1):
                    lhs = gp.quicksum(x[(j, s)] for s in range(es[j], t + 1))
                    ub_i = min(ls[i], t - d)
                    if ub_i < es[i]:
                        model.addConstr(lhs <= 0)
                    else:
                        rhs = gp.quicksum(x[(i, s)] for s in range(es[i], ub_i + 1))
                        model.addConstr(lhs <= rhs)
            else:
                model.addConstr(S[j] - S[i] >= d)

        # resource constraints
        for r in resources_data:
            periods = r["_sorted_periods"]
            dem = r["demands"]
            cap = int(r["capacity"])
            expr = gp.LinExpr()
            nonzero = False
            for j in range(n_nodes):
                dj = int(dem[j]) if j < len(dem) else 0
                if dj <= 0 or proc[j] <= 0 or not periods:
                    continue
                for t in times_of[j]:
                    u = resource_usage(t, proc[j], periods)
                    if u > 0:
                        expr.addTerms(dj * u, x[(j, t)])
                        nonzero = True
            if nonzero:
                model.addConstr(expr <= cap)

        model.setObjective(S[end], GRB.MINIMIZE)

        # warm start
        if best_starts is not None:
            for j in range(n_nodes):
                for t in times_of[j]:
                    x[(j, t)].Start = 1.0 if t == best_starts[j] else 0.0

        # callback for incumbent logging
        var_index = []  # list of (j, t, var)
        for (j, t), v in x.items():
            var_index.append((j, t, v))

        state = {"best_obj": best_obj, "best_starts": best_starts}

        def cb(m, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    vals = m.cbGetSolution([v for (_, _, v) in var_index])
                except gp.GurobiError:
                    return
                starts = [None] * n_nodes
                for idx, (j, t, _) in enumerate(var_index):
                    if vals[idx] > 0.5:
                        starts[j] = t
                if any(s is None for s in starts):
                    return
                obj = starts[end]
                if state["best_obj"] is None or obj < state["best_obj"]:
                    state["best_obj"] = obj
                    state["best_starts"] = starts
                    if logger:
                        logger.log_solution(int(obj), {
                            "objective_value": int(obj),
                            "start_times": [int(s) for s in starts],
                        })

        try:
            model.optimize(cb)
        except gp.GurobiError:
            pass

        if model.SolCount > 0:
            starts = [None] * n_nodes
            for (j, t), v in x.items():
                if v.X > 0.5:
                    starts[j] = t
            if all(s is not None for s in starts):
                obj = starts[end]
                if state["best_obj"] is None or obj <= state["best_obj"]:
                    state["best_obj"] = obj
                    state["best_starts"] = starts
                    if logger:
                        logger.log_solution(int(obj), {
                            "objective_value": int(obj),
                            "start_times": [int(s) for s in starts],
                        })

        best_obj = state["best_obj"]
        best_starts = state["best_starts"]

    # Fallback: output ES schedule even if resource-infeasible (best effort)
    if best_starts is None:
        best_starts = [min(max(es[j], 0), deadline) for j in range(n_nodes)]
        best_starts[0] = 0
        best_obj = best_starts[end]
        if logger:
            logger.log_solution(int(best_obj), {
                "objective_value": int(best_obj),
                "start_times": [int(s) for s in best_starts],
            })

    solution = {
        "objective_value": int(best_obj),
        "start_times": [int(s) for s in best_starts],
    }
    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()