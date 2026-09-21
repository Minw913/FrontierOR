import argparse
import json
import math
import time
import sys

from solution_logger import SolutionLogger


def load_instance(path):
    with open(path, "r") as f:
        return json.load(f)


def greedy_patterns(types_sorted, W, demands):
    """Greedy 'fill bins with largest items first' heuristic, batched by
    identical patterns. Returns list of (pattern, multiplicity) where a
    pattern is a list of (type_id, width, count) in the fixed sorted order."""
    rem = dict(demands)
    patterns = []
    while True:
        cap = W
        pat = []
        for tid, w, d in types_sorted:
            r = rem[tid]
            if r > 0 and w <= cap:
                k = min(r, cap // w)
                if k > 0:
                    pat.append((tid, w, k))
                    cap -= k * w
        if not pat:
            break
        mult = min(rem[tid] // k for tid, _, k in pat)
        if mult < 1:
            mult = 1
        for tid, _, k in pat:
            rem[tid] -= k * mult
        patterns.append((pat, mult))
    return patterns


def patterns_to_arcflow(patterns, W):
    """Convert batched bin patterns into a sparse arc-flow solution dict."""
    sol = {}
    z = 0
    for pat, mult in patterns:
        pos = 0
        for tid, w, k in pat:
            for _ in range(k):
                key = "f_{}_{}_{}".format(pos, pos + w, tid)
                sol[key] = sol.get(key, 0) + mult
                pos += w
        if pos < W:
            key = "f_{}_{}_loss".format(pos, W)
            sol[key] = sol.get(key, 0) + mult
        z += mult
    sol["z"] = z
    return z, sol


def build_graph(types_sorted, W, max_arcs=2500000):
    """Build symmetry-reduced arc-flow graph (Valerio de Carvalho style):
    within a bin items appear in non-increasing width order."""
    reach = {0}
    arc_set = set()
    for tid, w, d in types_sorted:
        if w > W:
            continue
        max_k = min(d, W // w)
        if max_k <= 0:
            continue
        base = sorted(reach)
        new_nodes = set()
        for u in base:
            v = u
            for _ in range(max_k):
                if v + w > W:
                    break
                arc_set.add((v, v + w, tid))
                v += w
                new_nodes.add(v)
            if len(arc_set) > max_arcs:
                return None, None
        reach |= new_nodes
    return reach, arc_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t0 = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    inst = load_instance(args.instance_path)
    W = int(inst["bin_capacity"])
    raw_items = inst.get("items", [])
    items = []
    for it in raw_items:
        tid = int(it["type_id"])
        w = int(it["width"])
        d = int(it["demand"])
        if d > 0 and w > 0 and w <= W:
            items.append((tid, w, d))

    if not items:
        out = {"objective_value": 0, "model_variables": {"z": 0}}
        if logger:
            logger.log_solution(0, out["model_variables"])
        with open(args.solution_path, "w") as f:
            json.dump(out, f)
        return

    # Fixed order: non-increasing width, tie-break by type_id (used consistently
    # for both the greedy heuristic and graph construction).
    types_sorted = sorted(items, key=lambda x: (-x[1], x[0]))
    demands = {tid: d for tid, w, d in types_sorted}
    widths = {tid: w for tid, w, d in types_sorted}

    # ---- Greedy initial solution ----
    patterns = greedy_patterns(types_sorted, W, demands)
    greedy_obj, greedy_sol = patterns_to_arcflow(patterns, W)

    best_obj = greedy_obj
    best_sol = greedy_sol
    if logger:
        logger.log_solution(best_obj, best_sol)

    total_width = sum(w * d for tid, w, d in types_sorted)
    lb = (total_width + W - 1) // W

    def finish():
        out = {"objective_value": int(best_obj), "model_variables": best_sol}
        with open(args.solution_path, "w") as f:
            json.dump(out, f)

    if best_obj <= lb:
        finish()
        return

    remaining = args.time_limit - (time.time() - t0) - 2.0
    if remaining < 3.0:
        finish()
        return

    # ---- Arc-flow MIP with Gurobi ----
    try:
        reach, arc_set = build_graph(types_sorted, W)
        if arc_set is None:
            finish()
            return

        import gurobipy as gp
        from gurobipy import GRB

        model = gp.Model("arcflow_csp")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        remaining = args.time_limit - (time.time() - t0) - 2.0
        if remaining < 3.0:
            finish()
            return
        model.Params.TimeLimit = max(1.0, remaining)

        arcs = sorted(arc_set)
        x = {}
        out_arcs = {}
        in_arcs = {}
        type_arcs = {}
        for a in arcs:
            u, v, tid = a
            var = model.addVar(vtype=GRB.INTEGER, lb=0.0, ub=float(demands[tid]),
                               name="x_{}_{}_{}".format(u, v, tid))
            x[a] = var
            out_arcs.setdefault(u, []).append(var)
            in_arcs.setdefault(v, []).append(var)
            type_arcs.setdefault(tid, []).append(var)

        loss = {}
        internal_nodes = [u for u in reach if 0 < u < W]
        for u in internal_nodes:
            loss[u] = model.addVar(vtype=GRB.INTEGER, lb=0.0, ub=float(greedy_obj),
                                   name="loss_{}".format(u))

        zvar = model.addVar(vtype=GRB.INTEGER, lb=float(lb), ub=float(greedy_obj), name="z")

        # Flow conservation at source
        model.addConstr(gp.quicksum(out_arcs.get(0, [])) == zvar)
        # Conservation at internal nodes
        for u in internal_nodes:
            inflow = gp.quicksum(in_arcs.get(u, []))
            outflow = gp.quicksum(out_arcs.get(u, []))
            model.addConstr(inflow == outflow + loss[u])
        # Demand coverage
        for tid, w, d in types_sorted:
            model.addConstr(gp.quicksum(type_arcs.get(tid, [])) >= d)

        model.setObjective(zvar, GRB.MINIMIZE)

        # MIP start from greedy solution
        start_flows = {}
        start_loss = {}
        for pat, mult in patterns:
            pos = 0
            for tid, w, k in pat:
                for _ in range(k):
                    a = (pos, pos + w, tid)
                    start_flows[a] = start_flows.get(a, 0) + mult
                    pos += w
            if pos < W:
                start_loss[pos] = start_loss.get(pos, 0) + mult
        for a, var in x.items():
            var.Start = float(start_flows.get(a, 0))
        for u, var in loss.items():
            var.Start = float(start_loss.get(u, 0))
        zvar.Start = float(greedy_obj)

        # Callback for incumbent logging
        model._x = x
        model._loss = loss
        model._z = zvar
        state = {"best": best_obj}

        def cb(m, where):
            if where == GRB.Callback.MIPSOL:
                obj = int(round(m.cbGet(GRB.Callback.MIPSOL_OBJ)))
                if obj < state["best"]:
                    state["best"] = obj
                    sol = {}
                    xvars = list(m._x.items())
                    vals = m.cbGetSolution([v for _, v in xvars])
                    for (a, _), val in zip(xvars, vals):
                        iv = int(round(val))
                        if iv > 0:
                            sol["f_{}_{}_{}".format(a[0], a[1], a[2])] = iv
                    lvars = list(m._loss.items())
                    lvals = m.cbGetSolution([v for _, v in lvars])
                    for (u, _), val in zip(lvars, lvals):
                        iv = int(round(val))
                        if iv > 0:
                            sol["f_{}_{}_loss".format(u, W)] = iv
                    sol["z"] = int(round(m.cbGetSolution(m._z)))
                    if logger:
                        logger.log_solution(obj, sol)

        model.optimize(cb)

        if model.SolCount > 0:
            obj = int(round(model.ObjVal))
            if obj < best_obj:
                sol = {}
                for a, var in x.items():
                    iv = int(round(var.X))
                    if iv > 0:
                        sol["f_{}_{}_{}".format(a[0], a[1], a[2])] = iv
                for u, var in loss.items():
                    iv = int(round(var.X))
                    if iv > 0:
                        sol["f_{}_{}_loss".format(u, W)] = iv
                sol["z"] = int(round(zvar.X))
                best_obj = obj
                best_sol = sol
                if logger and state["best"] > obj:
                    logger.log_solution(best_obj, best_sol)
    except Exception:
        # Fall back to the greedy solution on any failure
        pass

    finish()


if __name__ == "__main__":
    main()