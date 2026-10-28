import argparse
import json
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance_path", required=True,
                   help="Path to the problem instance JSON file.")
    p.add_argument("--solution_path", required=True,
                   help="Path where the final solution JSON will be written.")
    p.add_argument("--time_limit", type=int, required=True,
                   help="Maximum runtime in seconds.")
    p.add_argument("--log_path", default=None,
                   help="Optional path to a JSONL file for logging incumbents.")
    return p.parse_args()


def make_solution_dict(selection, groups):
    """Build a schema-conforming solution dict from a list of chosen item indices."""
    selected = []
    total = 0.0
    for g, grp in enumerate(groups):
        i = selection[g]
        item = grp["items"][i]
        total += item["profit"]
        selected.append({
            "group_id": grp.get("group_id", g),
            "item_id": item.get("item_id", i),
            "profit": item["profit"],
            "weights": list(item["weights"]),
        })
    return float(total), {"objective_value": float(total),
                          "selected_items": selected}


def greedy_solution(groups, capacities, m, deadline):
    """
    Construct a (hopefully feasible) solution:
      1) pick the min normalized-weight item per group,
      2) repair resource violations by overload-reducing swaps,
      3) improve profit by feasibility-preserving swaps.
    """
    n = len(groups)
    sel = [0] * n
    totals = [0.0] * m
    inv_cap = [1.0 / c if c > 0 else 1.0 for c in capacities]

    # Step 1: cheapest (normalized weight) item per group.
    for g, grp in enumerate(groups):
        best_i, best_key = 0, None
        for i, it in enumerate(grp["items"]):
            w = it["weights"]
            nw = 0.0
            for r in range(m):
                nw += w[r] * inv_cap[r]
            key = (nw, -it["profit"], i)
            if best_key is None or key < best_key:
                best_key, best_i = key, i
        sel[g] = best_i
        w = grp["items"][best_i]["weights"]
        for r in range(m):
            totals[r] += w[r]

    def total_overload():
        return sum(max(0.0, totals[r] - capacities[r]) for r in range(m))

    # Step 2: repair infeasibility.
    ov = total_overload()
    attempts = 0
    while ov > 1e-9 and attempts < 300 and time.time() < deadline:
        attempts += 1
        best = None
        for g in range(n):
            cur = groups[g]["items"][sel[g]]
            cw = cur["weights"]
            for i, it in enumerate(groups[g]["items"]):
                if i == sel[g]:
                    continue
                iw = it["weights"]
                new_ov = 0.0
                for r in range(m):
                    v = totals[r] - cw[r] + iw[r] - capacities[r]
                    if v > 0:
                        new_ov += v
                if new_ov < ov - 1e-9:
                    loss = cur["profit"] - it["profit"]
                    score = (new_ov, loss)
                    if best is None or score < best[0]:
                        best = (score, g, i, new_ov)
        if best is None:
            break
        _, g, i, ov = best
        cur = groups[g]["items"][sel[g]]
        it = groups[g]["items"][i]
        for r in range(m):
            totals[r] += it["weights"][r] - cur["weights"][r]
        sel[g] = i

    # Step 3: profit-improving feasible swaps (best improvement per group).
    improved = True
    passes = 0
    while improved and passes < 60 and time.time() < deadline:
        improved = False
        passes += 1
        for g in range(n):
            if time.time() >= deadline:
                break
            cur_i = sel[g]
            cur = groups[g]["items"][cur_i]
            cw = cur["weights"]
            best_gain = 1e-9
            best_j = -1
            for i, it in enumerate(groups[g]["items"]):
                gain = it["profit"] - cur["profit"]
                if gain <= best_gain:
                    continue
                iw = it["weights"]
                ok = True
                for r in range(m):
                    if totals[r] - cw[r] + iw[r] > capacities[r] + 1e-9:
                        ok = False
                        break
                if ok:
                    best_gain = gain
                    best_j = i
            if best_j >= 0:
                it = groups[g]["items"][best_j]
                for r in range(m):
                    totals[r] += it["weights"][r] - cw[r]
                sel[g] = best_j
                improved = True
    return sel


def main():
    args = parse_args()
    t0 = time.time()
    overall_deadline = t0 + max(0, args.time_limit)

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    groups = inst["groups"]
    capacities = inst["capacities"]
    m = int(inst.get("m_resources", len(capacities)))

    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="maximize")

    # ---- Initial heuristic solution (also serves as fallback / MIP start) ----
    heur_budget = max(0.2, min(0.25 * args.time_limit, args.time_limit - 1.0))
    heur_deadline = min(t0 + heur_budget, overall_deadline - 0.3)
    selection = greedy_solution(groups, capacities, m, heur_deadline)
    best_obj, best_dict = make_solution_dict(selection, groups)
    if logger:
        logger.log_solution(best_obj, best_dict)

    # ---- Exact MIP solve with Gurobi (multiple-choice multidimensional KP) ----
    remaining = overall_deadline - time.time() - 0.5
    if remaining > 0.1:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            model = gp.Model("multiple_choice_multidimensional_knapsack")
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(0.1, remaining)
            model.Params.LogToConsole = 0

            var_grid = []
            all_vars = []
            for g, grp in enumerate(groups):
                row = []
                for item in grp["items"]:
                    v = model.addVar(vtype=GRB.BINARY,
                                     obj=float(item["profit"]))
                    row.append(v)
                    all_vars.append(v)
                var_grid.append(row)
            model.ModelSense = GRB.MAXIMIZE

            # Exactly one item per group.
            for g in range(len(groups)):
                model.addConstr(gp.quicksum(var_grid[g]) == 1.0)

            # Shared resource capacities.
            for r in range(m):
                expr = gp.LinExpr()
                for g, grp in enumerate(groups):
                    row = var_grid[g]
                    for i, item in enumerate(grp["items"]):
                        w = item["weights"][r]
                        if w:
                            expr.addTerm(float(w), row[i])
                model.addConstr(expr <= float(capacities[r]))

            # Provide heuristic solution as a MIP start.
            for g in range(len(groups)):
                for i, v in enumerate(var_grid[g]):
                    v.Start = 1.0 if i == selection[g] else 0.0

            state = {"best": best_obj}

            def callback(mdl, where):
                if where == GRB.Callback.MIPSOL:
                    try:
                        obj = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
                    except Exception:
                        return
                    if obj > state["best"] + 1e-6:
                        state["best"] = obj
                        if logger:
                            try:
                                vals = mdl.cbGetSolution(all_vars)
                                cur_sel = [0] * len(groups)
                                idx = 0
                                for g in range(len(groups)):
                                    k = len(var_grid[g])
                                    bj, bv = 0, -1.0
                                    for j in range(k):
                                        val = vals[idx + j]
                                        if val > bv:
                                            bv, bj = val, j
                                    cur_sel[g] = bj
                                    idx += k
                                o, d = make_solution_dict(cur_sel, groups)
                                logger.log_solution(o, d)
                            except Exception:
                                logger.log(obj)

            model.optimize(callback)

            if model.SolCount > 0:
                vals = model.getAttr("X", all_vars)
                cur_sel = [0] * len(groups)
                idx = 0
                for g in range(len(groups)):
                    k = len(var_grid[g])
                    bj, bv = 0, -1.0
                    for j in range(k):
                        val = vals[idx + j]
                        if val > bv:
                            bv, bj = val, j
                    cur_sel[g] = bj
                    idx += k
                obj, d = make_solution_dict(cur_sel, groups)
                if obj >= best_obj - 1e-9:
                    best_obj, best_dict = obj, d
        except Exception:
            # Gurobi unavailable or failed: keep the heuristic solution.
            pass

    with open(args.solution_path, "w") as f:
        json.dump(best_dict, f)


if __name__ == "__main__":
    main()