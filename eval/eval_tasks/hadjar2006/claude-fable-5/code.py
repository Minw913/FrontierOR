import argparse
import json
import time
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    t_start = time.time()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    tasks = data["tasks"]
    depots = data["depots"]
    pairs = data["compatible_pairs"]

    task_ids = [t["id"] for t in tasks]
    task_by_id = {t["id"]: t for t in tasks}
    depot_ids = [d["id"] for d in depots]
    depot_cap = {d["id"]: d["capacity"] for d in depots}
    nd = len(depot_ids)
    nt = len(task_ids)

    d2t_cost = {}
    d2t_tt = {}
    for e in data["depot_to_task_costs"]:
        d2t_cost[(e["depot_id"], e["task_id"])] = e["cost"]
        d2t_tt[(e["depot_id"], e["task_id"])] = e["travel_time"]
    t2d_cost = {}
    t2d_tt = {}
    for e in data["task_to_depot_costs"]:
        t2d_cost[(e["task_id"], e["depot_id"])] = e["cost"]
        t2d_tt[(e["task_id"], e["depot_id"])] = e["travel_time"]

    # ------- optional arc pruning to keep the model tractable -------
    MAX_X = 2500000
    all_pairs = [(p["task_i"], p["task_j"], p["cost"]) for p in pairs]
    if nd * len(all_pairs) > MAX_X and nt > 0:
        K = max(2, MAX_X // (max(1, nd) * 2 * nt))
        by_src = {}
        by_dst = {}
        for idx, (i, j, c) in enumerate(all_pairs):
            by_src.setdefault(i, []).append((c, idx))
            by_dst.setdefault(j, []).append((c, idx))
        keep_idx = set()
        for i, lst in by_src.items():
            lst.sort()
            for c, idx in lst[:K]:
                keep_idx.add(idx)
        for j, lst in by_dst.items():
            lst.sort()
            for c, idx in lst[:K]:
                keep_idx.add(idx)
        kept_pairs = [all_pairs[idx] for idx in sorted(keep_idx)]
    else:
        kept_pairs = all_pairs

    pair_cost = {}
    succs = {i: [] for i in task_ids}
    preds = {i: [] for i in task_ids}
    for (i, j, c) in kept_pairs:
        pair_cost[(i, j)] = c
        succs[i].append(j)
        preds[j].append(i)

    # ------------------ greedy warm-start heuristic ------------------
    def greedy():
        order = sorted(task_ids, key=lambda t: (task_by_id[t]["start_time"],
                                                task_by_id[t]["end_time"]))
        cap = dict(depot_cap)
        vehicles = []  # each: [depot, list_of_tasks]
        for t in order:
            best_ext = None
            best_ext_cost = float("inf")
            for vi, veh in enumerate(vehicles):
                last = veh[1][-1]
                c = pair_cost.get((last, t))
                if c is not None and c < best_ext_cost:
                    best_ext_cost = c
                    best_ext = vi
            best_new_d = None
            best_new_cost = float("inf")
            for d in depot_ids:
                if cap.get(d, 0) > 0:
                    c = d2t_cost.get((d, t))
                    if c is not None and c < best_new_cost:
                        best_new_cost = c
                        best_new_d = d
            if best_ext is not None and (best_new_d is None or best_ext_cost <= best_new_cost):
                vehicles[best_ext][1].append(t)
            elif best_new_d is not None:
                cap[best_new_d] -= 1
                vehicles.append([best_new_d, [t]])
            else:
                return None  # infeasible greedy
        routes = []
        total = 0.0
        for d, seq in vehicles:
            c = d2t_cost.get((d, seq[0]), 0.0)
            for a, b in zip(seq[:-1], seq[1:]):
                c += pair_cost[(a, b)]
            c += t2d_cost.get((seq[-1], d), 0.0)
            routes.append({"depot": d, "tasks": list(seq), "cost": c})
            total += c
        return total, routes

    greedy_sol = greedy()
    best_obj = float("inf")
    best_routes = []
    if greedy_sol is not None:
        best_obj, best_routes = greedy_sol
        if logger:
            logger.log_solution(best_obj, {"objective_value": best_obj,
                                           "routes": best_routes})

    def write_solution(obj, routes):
        sol = {"objective_value": obj, "routes": routes}
        with open(args.solution_path, "w") as f:
            json.dump(sol, f, indent=1)

    # write an early fallback so a file always exists
    write_solution(best_obj if best_routes else 0.0, best_routes)

    # ------------------------- MIP model -----------------------------
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return

    remaining = args.time_limit - (time.time() - t_start) - 3.0
    if remaining <= 1.0:
        return

    try:
        m = gp.Model("mdvsp")
        m.Params.OutputFlag = 0
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.NumericFocus = 0
        m.Params.Threads = 1
        m.Params.TimeLimit = max(1.0, remaining)

        # variables
        xkeys = []
        xobj = []
        for d in depot_ids:
            for (i, j, c) in kept_pairs:
                xkeys.append((d, i, j))
                xobj.append(c)
        pokeys = []
        poobj = []
        for d in depot_ids:
            for t in task_ids:
                if (d, t) in d2t_cost:
                    pokeys.append((d, t))
                    poobj.append(d2t_cost[(d, t)])
        pikeys = []
        piobj = []
        for t in task_ids:
            for d in depot_ids:
                if (t, d) in t2d_cost:
                    pikeys.append((t, d))
                    piobj.append(t2d_cost[(t, d)])

        xv = m.addVars(range(len(xkeys)), vtype=GRB.BINARY)
        pov = m.addVars(range(len(pokeys)), vtype=GRB.BINARY)
        piv = m.addVars(range(len(pikeys)), vtype=GRB.BINARY)
        xlist = [xv[k] for k in range(len(xkeys))]
        polist = [pov[k] for k in range(len(pokeys))]
        pilist = [piv[k] for k in range(len(pikeys))]
        if xlist:
            m.setAttr("Obj", xlist, xobj)
        if polist:
            m.setAttr("Obj", polist, poobj)
        if pilist:
            m.setAttr("Obj", pilist, piobj)

        xidx = {k: v for k, v in zip(xkeys, xlist)}
        poidx = {k: v for k, v in zip(pokeys, polist)}
        piidx = {k: v for k, v in zip(pikeys, pilist)}

        # incoming / outgoing var lists per (depot, task)
        in_terms = {(d, t): [] for d in depot_ids for t in task_ids}
        out_terms = {(d, t): [] for d in depot_ids for t in task_ids}
        for k, var in zip(xkeys, xlist):
            d, i, j = k
            out_terms[(d, i)].append(var)
            in_terms[(d, j)].append(var)
        for k, var in zip(pokeys, polist):
            d, t = k
            in_terms[(d, t)].append(var)
        for k, var in zip(pikeys, pilist):
            t, d = k
            out_terms[(d, t)].append(var)

        # coverage: exactly one inflow per task (over all depots)
        for t in task_ids:
            expr = gp.LinExpr()
            for d in depot_ids:
                for v in in_terms[(d, t)]:
                    expr.addTerms(1.0, v)
            m.addConstr(expr == 1)

        # flow conservation per depot per task
        for d in depot_ids:
            for t in task_ids:
                ein = gp.LinExpr()
                for v in in_terms[(d, t)]:
                    ein.addTerms(1.0, v)
                eout = gp.LinExpr()
                for v in out_terms[(d, t)]:
                    eout.addTerms(1.0, v)
                m.addConstr(ein == eout)

        # depot capacity
        for d in depot_ids:
            expr = gp.LinExpr()
            for k, var in zip(pokeys, polist):
                if k[0] == d:
                    expr.addTerms(1.0, var)
            m.addConstr(expr <= depot_cap[d])

        # warm start from greedy
        if greedy_sol is not None:
            allv = xlist + polist + pilist
            m.setAttr("Start", allv, [0.0] * len(allv))
            ok = True
            for r in best_routes:
                d = r["depot"]
                seq = r["tasks"]
                if (d, seq[0]) in poidx and (seq[-1], d) in piidx:
                    poidx[(d, seq[0])].Start = 1.0
                    piidx[(seq[-1], d)].Start = 1.0
                    for a, b in zip(seq[:-1], seq[1:]):
                        if (d, a, b) in xidx:
                            xidx[(d, a, b)].Start = 1.0
                        else:
                            ok = False
                else:
                    ok = False
            if not ok:
                # incomplete start; let gurobi try to repair or ignore
                pass

        def extract(xvals, povals, pivals):
            succ = {}
            for k, v in zip(xkeys, xvals):
                if v > 0.5:
                    succ[(k[0], k[1])] = k[2]
            routes = []
            total = 0.0
            for k, v in zip(pokeys, povals):
                if v > 0.5:
                    d, i = k
                    seq = [i]
                    cur = i
                    while (d, cur) in succ:
                        cur = succ.pop((d, cur))
                        seq.append(cur)
                    c = d2t_cost.get((d, seq[0]), 0.0)
                    for a, b in zip(seq[:-1], seq[1:]):
                        c += pair_cost.get((a, b), 0.0)
                    c += t2d_cost.get((seq[-1], d), 0.0)
                    routes.append({"depot": d, "tasks": seq, "cost": c})
                    total += c
            return total, routes

        state = {"best": best_obj}

        def cb(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj < state["best"] - 1e-6:
                    try:
                        xvals = model.cbGetSolution(xlist) if xlist else []
                        povals = model.cbGetSolution(polist) if polist else []
                        pivals = model.cbGetSolution(pilist) if pilist else []
                        tot, routes = extract(xvals, povals, pivals)
                        state["best"] = tot
                        if logger:
                            logger.log_solution(tot, {"objective_value": tot,
                                                      "routes": routes})
                    except Exception:
                        if logger:
                            logger.log(obj)
                        state["best"] = obj

        # refresh time limit right before solving
        remaining = args.time_limit - (time.time() - t_start) - 3.0
        m.Params.TimeLimit = max(1.0, remaining)
        m.optimize(cb)

        if m.SolCount > 0:
            xvals = m.getAttr("X", xlist) if xlist else []
            povals = m.getAttr("X", polist) if polist else []
            pivals = m.getAttr("X", pilist) if pilist else []
            tot, routes = extract(xvals, povals, pivals)
            if tot < best_obj - 1e-9 or not best_routes:
                best_obj = tot
                best_routes = routes
                if logger:
                    logger.log_solution(best_obj, {"objective_value": best_obj,
                                                   "routes": best_routes})
    except Exception:
        pass

    if best_routes or best_obj < float("inf"):
        write_solution(best_obj, best_routes)


if __name__ == "__main__":
    main()