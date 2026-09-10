import argparse
import json
import math
import time
import sys

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def load_instance(path):
    with open(path, "r") as f:
        return json.load(f)


def build_coverage(inst):
    n_f = inst["num_facilities"]
    n_c = inst["num_customers"]
    cust_to_fac = {}
    fac_to_cust = {}
    c2f = inst.get("customer_to_facilities_coverage")
    f2c = inst.get("facility_to_customers_coverage")
    if c2f is not None:
        for i in range(n_c):
            cust_to_fac[i] = list(c2f.get(str(i), []))
    if f2c is not None:
        for j in range(n_f):
            fac_to_cust[j] = list(f2c.get(str(j), []))
    if c2f is None or f2c is None:
        # compute from coordinates
        R = inst["radius_of_coverage"]
        fc = inst["facility_coordinates"]
        cc = inst["customer_coordinates"]
        cust_to_fac = {i: [] for i in range(n_c)}
        fac_to_cust = {j: [] for j in range(n_f)}
        R2 = R * R + 1e-9
        for i in range(n_c):
            xi, yi = cc[i]
            for j in range(n_f):
                xj, yj = fc[j]
                dx = xi - xj
                dy = yi - yj
                if dx * dx + dy * dy <= R2:
                    cust_to_fac[i].append(j)
                    fac_to_cust[j].append(i)
    return cust_to_fac, fac_to_cust


def covered_demand_of(open_set, cust_to_fac, demands):
    tot = 0.0
    for i, facs in cust_to_fac.items():
        for j in facs:
            if j in open_set:
                tot += demands[i]
                break
    return tot


def greedy_psclp(n_f, costs, demands, cust_to_fac, fac_to_cust, D):
    open_set = set()
    covered = set()
    cov_dem = 0.0
    while cov_dem < D - 1e-9:
        best_j, best_ratio, best_gain = -1, -1.0, 0.0
        for j in range(n_f):
            if j in open_set:
                continue
            gain = sum(demands[i] for i in fac_to_cust.get(j, []) if i not in covered)
            if gain <= 0:
                continue
            ratio = gain / max(costs[j], 1e-9)
            if ratio > best_ratio:
                best_ratio, best_j, best_gain = ratio, j, gain
        if best_j < 0:
            break
        open_set.add(best_j)
        for i in fac_to_cust.get(best_j, []):
            if i not in covered:
                covered.add(i)
                cov_dem += demands[i]
    return open_set, cov_dem


def greedy_mclp(n_f, costs, demands, cust_to_fac, fac_to_cust, B):
    open_set = set()
    covered = set()
    cov_dem = 0.0
    spent = 0.0
    while True:
        best_j, best_ratio = -1, -1.0
        for j in range(n_f):
            if j in open_set:
                continue
            if spent + costs[j] > B + 1e-9:
                continue
            gain = sum(demands[i] for i in fac_to_cust.get(j, []) if i not in covered)
            if gain <= 0:
                continue
            ratio = gain / max(costs[j], 1e-9)
            if ratio > best_ratio:
                best_ratio, best_j = ratio, j
        if best_j < 0:
            break
        open_set.add(best_j)
        spent += costs[best_j]
        for i in fac_to_cust.get(best_j, []):
            if i not in covered:
                covered.add(i)
                cov_dem += demands[i]
    return open_set, cov_dem


def make_solution_dict(primary, psclp_res, mclp_res):
    if primary == "PSCLP":
        obj = psclp_res["objective_value"]
    else:
        obj = mclp_res["objective_value"]
    return {
        "objective_value": float(obj),
        "primary_problem_type": primary,
        "results": {
            "PSCLP": {
                "objective_value": float(psclp_res["objective_value"]),
                "open_facilities": sorted(int(j) for j in psclp_res["open_facilities"]),
                "covered_demand": float(psclp_res["covered_demand"]),
            },
            "MCLP": {
                "objective_value": float(mclp_res["objective_value"]),
                "open_facilities": sorted(int(j) for j in mclp_res["open_facilities"]),
                "covered_demand": float(mclp_res["covered_demand"]),
            },
        },
    }


def solve_psclp(inst, cust_to_fac, time_limit, start_open, callback_ctx=None):
    n_f = inst["num_facilities"]
    n_c = inst["num_customers"]
    costs = inst["facility_cost"]
    demands = inst["customer_demands"]
    D = inst["covering_demand_D"]

    m = gp.Model("PSCLP")
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = max(1.0, time_limit)

    x = m.addVars(n_f, vtype=GRB.BINARY, name="x")
    z = m.addVars(n_c, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="z")
    for i in range(n_c):
        facs = cust_to_fac.get(i, [])
        if facs:
            m.addConstr(z[i] <= gp.quicksum(x[j] for j in facs))
        else:
            m.addConstr(z[i] == 0)
    m.addConstr(gp.quicksum(demands[i] * z[i] for i in range(n_c)) >= D)
    m.setObjective(gp.quicksum(costs[j] * x[j] for j in range(n_f)), GRB.MINIMIZE)

    if start_open is not None:
        for j in range(n_f):
            x[j].Start = 1.0 if j in start_open else 0.0

    cb = None
    if callback_ctx is not None:
        logger, other_res, primary = callback_ctx

        def cb(model, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    xv = model.cbGetSolution([x[j] for j in range(n_f)])
                    open_set = set(j for j in range(n_f) if xv[j] > 0.5)
                    cd = covered_demand_of(open_set, cust_to_fac, demands)
                    obj = sum(costs[j] for j in open_set)
                    res = {"objective_value": obj, "open_facilities": open_set,
                           "covered_demand": cd}
                    sol = make_solution_dict(primary, res, other_res)
                    logger.log_solution(float(obj), sol)
                except Exception:
                    pass

    if cb is not None:
        m.optimize(cb)
    else:
        m.optimize()

    if m.SolCount > 0:
        open_set = set(j for j in range(n_f) if x[j].X > 0.5)
        cd = covered_demand_of(open_set, cust_to_fac, demands)
        obj = sum(costs[j] for j in open_set)
        return {"objective_value": obj, "open_facilities": open_set, "covered_demand": cd}
    # fallback to start
    if start_open is not None:
        cd = covered_demand_of(start_open, cust_to_fac, demands)
        return {"objective_value": sum(costs[j] for j in start_open),
                "open_facilities": start_open, "covered_demand": cd}
    return {"objective_value": 0.0, "open_facilities": set(), "covered_demand": 0.0}


def solve_mclp(inst, cust_to_fac, time_limit, start_open, callback_ctx=None):
    n_f = inst["num_facilities"]
    n_c = inst["num_customers"]
    costs = inst["facility_cost"]
    demands = inst["customer_demands"]
    B = inst["budget_B"]

    m = gp.Model("MCLP")
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = max(1.0, time_limit)

    x = m.addVars(n_f, vtype=GRB.BINARY, name="x")
    z = m.addVars(n_c, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="z")
    for i in range(n_c):
        facs = cust_to_fac.get(i, [])
        if facs:
            m.addConstr(z[i] <= gp.quicksum(x[j] for j in facs))
        else:
            m.addConstr(z[i] == 0)
    m.addConstr(gp.quicksum(costs[j] * x[j] for j in range(n_f)) <= B)
    m.setObjective(gp.quicksum(demands[i] * z[i] for i in range(n_c)), GRB.MAXIMIZE)

    if start_open is not None:
        for j in range(n_f):
            x[j].Start = 1.0 if j in start_open else 0.0

    cb = None
    if callback_ctx is not None:
        logger, other_res, primary = callback_ctx

        def cb(model, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    xv = model.cbGetSolution([x[j] for j in range(n_f)])
                    open_set = set(j for j in range(n_f) if xv[j] > 0.5)
                    cd = covered_demand_of(open_set, cust_to_fac, demands)
                    res = {"objective_value": cd, "open_facilities": open_set,
                           "covered_demand": cd}
                    sol = make_solution_dict(primary, other_res, res)
                    logger.log_solution(float(cd), sol)
                except Exception:
                    pass

    if cb is not None:
        m.optimize(cb)
    else:
        m.optimize()

    if m.SolCount > 0:
        open_set = set(j for j in range(n_f) if x[j].X > 0.5)
        cd = covered_demand_of(open_set, cust_to_fac, demands)
        return {"objective_value": cd, "open_facilities": open_set, "covered_demand": cd}
    if start_open is not None:
        cd = covered_demand_of(start_open, cust_to_fac, demands)
        return {"objective_value": cd, "open_facilities": start_open, "covered_demand": cd}
    return {"objective_value": 0.0, "open_facilities": set(), "covered_demand": 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    t_start = time.time()
    inst = load_instance(args.instance_path)

    # Determine primary problem type
    primary = "PSCLP"
    pt = inst.get("problem_types") or inst.get("problem_type")
    if isinstance(pt, list) and len(pt) > 0:
        primary = str(pt[0])
    elif isinstance(pt, str):
        primary = pt
    if primary not in ("PSCLP", "MCLP"):
        primary = "PSCLP"

    sense = "minimize" if primary == "PSCLP" else "maximize"
    logger = SolutionLogger(args.log_path, sense=sense) if args.log_path else None

    cust_to_fac, fac_to_cust = build_coverage(inst)
    n_f = inst["num_facilities"]
    costs = inst["facility_cost"]
    demands = inst["customer_demands"]
    D = inst["covering_demand_D"]
    B = inst["budget_B"]

    # Greedy initial solutions
    ps_open, ps_cd = greedy_psclp(n_f, costs, demands, cust_to_fac, fac_to_cust, D)
    mc_open, mc_cd = greedy_mclp(n_f, costs, demands, cust_to_fac, fac_to_cust, B)

    psclp_res = {"objective_value": sum(costs[j] for j in ps_open),
                 "open_facilities": ps_open, "covered_demand": ps_cd}
    mclp_res = {"objective_value": mc_cd, "open_facilities": mc_open,
                "covered_demand": mc_cd}

    # Log initial greedy solution (only if feasible for primary)
    if logger:
        feasible_primary = (primary == "MCLP") or (ps_cd >= D - 1e-6)
        if feasible_primary:
            sol0 = make_solution_dict(primary, psclp_res, mclp_res)
            logger.log_solution(sol0["objective_value"], sol0)

    # Time budget: primary gets ~65% of remaining, secondary the rest
    elapsed = time.time() - t_start
    remaining = max(2.0, args.time_limit - elapsed - 2.0)
    t_primary = remaining * 0.65
    t_secondary = remaining * 0.35

    if primary == "PSCLP":
        psclp_res = solve_psclp(inst, cust_to_fac, t_primary, ps_open,
                                callback_ctx=(logger, mclp_res, primary) if logger else None)
        elapsed = time.time() - t_start
        rem = max(1.0, args.time_limit - elapsed - 1.0)
        mclp_res = solve_mclp(inst, cust_to_fac, min(t_secondary, rem), mc_open)
    else:
        mclp_res = solve_mclp(inst, cust_to_fac, t_primary, mc_open,
                              callback_ctx=(logger, psclp_res, primary) if logger else None)
        elapsed = time.time() - t_start
        rem = max(1.0, args.time_limit - elapsed - 1.0)
        psclp_res = solve_psclp(inst, cust_to_fac, min(t_secondary, rem), ps_open)

    final_sol = make_solution_dict(primary, psclp_res, mclp_res)

    if logger:
        logger.log_solution(final_sol["objective_value"], final_sol)

    with open(args.solution_path, "w") as f:
        json.dump(final_sol, f, indent=2)


if __name__ == "__main__":
    main()