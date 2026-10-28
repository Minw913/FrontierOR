import argparse
import json
import math
import time
from typing import List, Tuple, Dict, Optional

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_solution_dict_from_usage(usage: Dict[int, int]) -> Dict:
    model_vars = {}
    obj = 0
    for p, v in usage.items():
        if v and v != 0:
            iv = int(round(v))
            if abs(v - iv) <= 1e-9:
                v_out = iv
            else:
                v_out = float(v)
            model_vars[f"z[{p}]"] = v_out
            obj += iv if isinstance(v_out, int) else float(v_out)
    return {"objective_value": int(round(obj)), "model_variables": model_vars}


def pricing_unbounded_knapsack(widths: List[int], duals: List[float], capacity: int, eps: float = 1e-9):
    m = len(widths)
    dp = [0.0] * (capacity + 1)
    prev_item = [-1] * (capacity + 1)
    prev_cap = [-1] * (capacity + 1)

    for c in range(1, capacity + 1):
        best = dp[c]
        bi = -1
        bc = -1
        for i in range(m):
            w = widths[i]
            if w <= c:
                val = dp[c - w] + duals[i]
                if val > best + eps:
                    best = val
                    bi = i
                    bc = c - w
        dp[c] = best
        prev_item[c] = bi
        prev_cap[c] = bc

    best_cap = max(range(capacity + 1), key=lambda c: dp[c])
    best_val = dp[best_cap]

    if best_val <= 1.0 + 1e-7:
        return None, best_val

    pattern = [0] * m
    c = best_cap
    while c > 0 and prev_item[c] != -1:
        i = prev_item[c]
        pattern[i] += 1
        c = prev_cap[c]

    if sum(pattern) == 0:
        return None, best_val

    return tuple(pattern), best_val


def solve(instance: Dict, time_limit: int, logger: Optional[SolutionLogger] = None) -> Dict:
    start = time.time()

    W = int(instance["bin_capacity"])
    items = instance["items"]

    # Keep only positive-demand types
    filtered = [it for it in items if int(it["demand"]) > 0]
    if not filtered:
        return {"objective_value": 0, "model_variables": {}}

    # Sort by width descending (often helps)
    filtered.sort(key=lambda x: int(x["width"]), reverse=True)

    type_ids = [int(it["type_id"]) for it in filtered]
    widths = [int(it["width"]) for it in filtered]
    demands = [int(it["demand"]) for it in filtered]
    m = len(filtered)

    # Feasibility check
    for i in range(m):
        if widths[i] > W and demands[i] > 0:
            return {"objective_value": -1, "model_variables": {}}

    # Initial patterns: "single type, as many as fit"
    patterns: List[Tuple[int, ...]] = []
    pattern_set = set()
    single_type_pattern_index = {}

    for i in range(m):
        cnt = W // widths[i]
        if cnt <= 0:
            continue
        pat = [0] * m
        pat[i] = cnt
        tpat = tuple(pat)
        if tpat not in pattern_set:
            single_type_pattern_index[i] = len(patterns)
            patterns.append(tpat)
            pattern_set.add(tpat)

    # Fallback feasible incumbent from single-type patterns
    incumbent_solution = None
    incumbent_obj = float("inf")

    if len(single_type_pattern_index) == m:
        usage = {}
        for i in range(m):
            p = single_type_pattern_index[i]
            c = patterns[p][i]
            k = int(math.ceil(demands[i] / c))
            usage[p] = usage.get(p, 0) + k
        sol = build_solution_dict_from_usage(usage)
        incumbent_solution = sol
        incumbent_obj = sol["objective_value"]
        if logger:
            logger.log_solution(sol["objective_value"], sol)

    def remaining_time():
        return time_limit - (time.time() - start)

    # ---------------------------
    # Column generation (LP RMP)
    # ---------------------------
    if remaining_time() > 0.5 and patterns:
        lp = gp.Model("rmp_lp")
        lp.Params.OutputFlag = 0
        lp.Params.Seed = 0
        lp.Params.NumericFocus = 0
        lp.Params.Threads = 1
        lp.Params.MIPGap = 1e-4
        lp.Params.Method = 1  # dual simplex (good for reopt)

        cons = []
        for i in range(m):
            cons.append(lp.addConstr(0.0 >= demands[i], name=f"demand[{i}]"))

        z_lp = []
        for p_idx, pat in enumerate(patterns):
            col = gp.Column(list(pat), cons)
            var = lp.addVar(lb=0.0, vtype=GRB.CONTINUOUS, obj=1.0, column=col, name=f"z[{p_idx}]")
            z_lp.append(var)

        lp.ModelSense = GRB.MINIMIZE
        lp.update()

        max_cg_iters = 10000
        for _ in range(max_cg_iters):
            rt = remaining_time()
            if rt <= 0.5:
                break
            lp.Params.TimeLimit = max(0.1, rt)
            lp.optimize()
            if lp.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
                break
            if lp.Status == GRB.TIME_LIMIT:
                break

            duals = [cons[i].Pi for i in range(m)]
            new_pat, best_val = pricing_unbounded_knapsack(widths, duals, W)
            if new_pat is None:
                break
            if new_pat in pattern_set:
                break

            pattern_set.add(new_pat)
            patterns.append(new_pat)
            p_idx = len(patterns) - 1

            col = gp.Column(list(new_pat), cons)
            var = lp.addVar(lb=0.0, vtype=GRB.CONTINUOUS, obj=1.0, column=col, name=f"z[{p_idx}]")
            z_lp.append(var)
            lp.update()

    # ---------------------------
    # Final integer master
    # ---------------------------
    if remaining_time() <= 0.2 or not patterns:
        if incumbent_solution is None:
            return {"objective_value": -1, "model_variables": {}}
        return incumbent_solution

    mip = gp.Model("master_ip")
    mip.Params.OutputFlag = 0
    mip.Params.Seed = 0
    mip.Params.NumericFocus = 0
    mip.Params.Threads = 1
    mip.Params.MIPGap = 1e-4
    mip.Params.TimeLimit = max(0.1, remaining_time())

    cons_ip = []
    for i in range(m):
        cons_ip.append(mip.addConstr(0.0 >= demands[i], name=f"demand[{i}]"))

    z = []
    for p_idx, pat in enumerate(patterns):
        col = gp.Column(list(pat), cons_ip)
        var = mip.addVar(lb=0.0, vtype=GRB.INTEGER, obj=1.0, column=col, name=f"z[{p_idx}]")
        z.append(var)

    mip.ModelSense = GRB.MINIMIZE
    mip.update()

    # Warm start from fallback incumbent usage if available
    if incumbent_solution is not None:
        for var in z:
            var.Start = 0.0
        for name, val in incumbent_solution["model_variables"].items():
            if name.startswith("z[") and name.endswith("]"):
                idx = int(name[2:].strip("[]"))
                if 0 <= idx < len(z):
                    z[idx].Start = float(val)

    mip._vars = z
    mip._logger = logger
    mip._best_obj = incumbent_obj
    mip._best_solution = incumbent_solution

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-7 < model._best_obj:
                vals = model.cbGetSolution(model._vars)
                model_vars = {}
                for j, v in enumerate(vals):
                    iv = int(round(v))
                    if iv > 0:
                        model_vars[f"z[{j}]"] = iv
                sol = {"objective_value": int(round(obj)), "model_variables": model_vars}
                model._best_obj = obj
                model._best_solution = sol
                if model._logger:
                    model._logger.log_solution(sol["objective_value"], sol)

    mip.optimize(cb)

    # Extract final
    final_sol = None
    if mip.SolCount > 0:
        model_vars = {}
        for j, var in enumerate(z):
            v = var.X
            iv = int(round(v))
            if iv > 0:
                model_vars[f"z[{j}]"] = iv
        final_obj = int(round(mip.ObjVal))
        final_sol = {"objective_value": final_obj, "model_variables": model_vars}

    # Choose best between callback incumbent and final extracted
    best = None
    if final_sol is not None:
        best = final_sol
    if mip._best_solution is not None:
        if best is None or mip._best_solution["objective_value"] < best["objective_value"]:
            best = mip._best_solution

    if best is None:
        best = incumbent_solution if incumbent_solution is not None else {"objective_value": -1, "model_variables": {}}

    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    solution = solve(instance, args.time_limit, logger)

    # Ensure schema fields exist
    out = {
        "objective_value": int(solution.get("objective_value", -1)),
        "model_variables": solution.get("model_variables", {})
    }

    with open(args.solution_path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()