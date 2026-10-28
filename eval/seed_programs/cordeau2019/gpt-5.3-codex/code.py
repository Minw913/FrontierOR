import argparse
import json
import math
import time
from typing import List, Dict, Tuple, Set

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Solve PSCLP and MCLP from JSON instance.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Optional JSONL path for incumbent logging")
    return parser.parse_args()


def load_instance(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def euclidean_leq(p, q, r):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy <= r * r + 1e-12


def build_coverage(instance: dict) -> Tuple[List[List[int]], List[List[int]]]:
    n = int(instance["num_facilities"])
    m = int(instance["num_customers"])

    f2c_raw = instance.get("facility_to_customers_coverage", None)
    c2f_raw = instance.get("customer_to_facilities_coverage", None)

    # If valid coverage provided, use it.
    if isinstance(f2c_raw, dict) and isinstance(c2f_raw, dict) and len(f2c_raw) > 0 and len(c2f_raw) > 0:
        fac_to_cust = [[] for _ in range(n)]
        cust_to_fac = [[] for _ in range(m)]

        for k, v in f2c_raw.items():
            i = int(k)
            fac_to_cust[i] = [int(x) for x in v]

        for k, v in c2f_raw.items():
            j = int(k)
            cust_to_fac[j] = [int(x) for x in v]

        return fac_to_cust, cust_to_fac

    # Else derive from coordinates + radius.
    fcoords = instance["facility_coordinates"]
    ccoords = instance["customer_coordinates"]
    r = float(instance["radius_of_coverage"])

    fac_to_cust = [[] for _ in range(n)]
    cust_to_fac = [[] for _ in range(m)]

    for i in range(n):
        pi = fcoords[i]
        for j in range(m):
            if euclidean_leq(pi, ccoords[j], r):
                fac_to_cust[i].append(j)
                cust_to_fac[j].append(i)

    return fac_to_cust, cust_to_fac


def evaluate_open_set(open_set: List[int], fac_to_cust: List[List[int]], demands: List[float], costs: List[float]):
    covered = set()
    total_cost = 0.0
    for i in open_set:
        total_cost += costs[i]
        for j in fac_to_cust[i]:
            covered.add(j)
    covered_demand = float(sum(demands[j] for j in covered))
    return total_cost, covered_demand, covered


def greedy_psclp(n, m, D, costs, demands, fac_to_cust):
    if D <= 0:
        return []

    opened = set()
    covered = set()
    covered_d = 0.0

    while covered_d + 1e-9 < D:
        best_i = None
        best_ratio = -1.0
        best_gain = 0.0
        best_cost = None

        for i in range(n):
            if i in opened:
                continue
            gain = 0.0
            for j in fac_to_cust[i]:
                if j not in covered:
                    gain += demands[j]
            if gain <= 1e-12:
                continue

            c = costs[i]
            ratio = float("inf") if c <= 1e-12 else gain / c

            if (
                ratio > best_ratio + 1e-12
                or (abs(ratio - best_ratio) <= 1e-12 and gain > best_gain + 1e-12)
                or (abs(ratio - best_ratio) <= 1e-12 and abs(gain - best_gain) <= 1e-12 and (best_cost is None or c < best_cost))
            ):
                best_ratio = ratio
                best_gain = gain
                best_cost = c
                best_i = i

        if best_i is None:
            break

        opened.add(best_i)
        for j in fac_to_cust[best_i]:
            covered.add(j)
        covered_d = float(sum(demands[j] for j in covered))

    # Prune redundant facilities if feasible
    open_list = sorted(opened)
    _, cov_d, _ = evaluate_open_set(open_list, fac_to_cust, demands, costs)
    if cov_d + 1e-9 >= D:
        changed = True
        while changed:
            changed = False
            # try removing expensive facilities first
            open_list = sorted(open_list, key=lambda i: costs[i], reverse=True)
            for i in list(open_list):
                trial = [x for x in open_list if x != i]
                _, trial_cov_d, _ = evaluate_open_set(trial, fac_to_cust, demands, costs)
                if trial_cov_d + 1e-9 >= D:
                    open_list = trial
                    changed = True
                    break
        open_list = sorted(open_list)

    return open_list


def greedy_mclp(n, B, costs, demands, fac_to_cust):
    opened = set()
    covered = set()
    spent = 0.0

    while True:
        best_i = None
        best_ratio = -1.0
        best_gain = 0.0
        best_cost = None

        for i in range(n):
            if i in opened:
                continue
            c = costs[i]
            if spent + c > B + 1e-9:
                continue

            gain = 0.0
            for j in fac_to_cust[i]:
                if j not in covered:
                    gain += demands[j]
            if gain <= 1e-12:
                continue

            ratio = float("inf") if c <= 1e-12 else gain / c
            if (
                ratio > best_ratio + 1e-12
                or (abs(ratio - best_ratio) <= 1e-12 and gain > best_gain + 1e-12)
                or (abs(ratio - best_ratio) <= 1e-12 and abs(gain - best_gain) <= 1e-12 and (best_cost is None or c < best_cost))
            ):
                best_ratio = ratio
                best_gain = gain
                best_cost = c
                best_i = i

        if best_i is None:
            break

        opened.add(best_i)
        spent += costs[best_i]
        for j in fac_to_cust[best_i]:
            covered.add(j)

    return sorted(opened)


def build_full_solution(primary_type: str, results_psclp: dict, results_mclp: dict):
    if primary_type == "PSCLP":
        obj = float(results_psclp["objective_value"])
    else:
        obj = float(results_mclp["objective_value"])

    return {
        "objective_value": obj,
        "primary_problem_type": primary_type,
        "results": {
            "PSCLP": {
                "objective_value": float(results_psclp["objective_value"]),
                "open_facilities": [int(i) for i in results_psclp["open_facilities"]],
                "covered_demand": float(results_psclp["covered_demand"]),
            },
            "MCLP": {
                "objective_value": float(results_mclp["objective_value"]),
                "open_facilities": [int(i) for i in results_mclp["open_facilities"]],
                "covered_demand": float(results_mclp["covered_demand"]),
            },
        },
    }


def solve_psclp(
    n, m, D, costs, demands, cust_to_fac, fac_to_cust,
    time_limit, warm_open, logger, primary_type, incumbent_results
):
    model = gp.Model("PSCLP")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.0, float(time_limit))

    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    y = {}
    for j in range(m):
        ub = 1.0 if len(cust_to_fac[j]) > 0 else 0.0
        y[j] = model.addVar(vtype=GRB.BINARY, ub=ub, name=f"y_{j}")

    for j in range(m):
        if len(cust_to_fac[j]) > 0:
            model.addConstr(y[j] <= gp.quicksum(x[i] for i in cust_to_fac[j]), name=f"link_{j}")

    model.addConstr(gp.quicksum(demands[j] * y[j] for j in range(m)) >= D, name="demand_req")
    model.setObjective(gp.quicksum(costs[i] * x[i] for i in range(n)), GRB.MINIMIZE)

    # Warm start
    warm_set = set(warm_open)
    for i in range(n):
        x[i].Start = 1.0 if i in warm_set else 0.0
    _, _, warm_cov = evaluate_open_set(warm_open, fac_to_cust, demands, costs)
    for j in range(m):
        y[j].Start = 1.0 if j in warm_cov else 0.0

    best_cb_obj = [float("inf")]

    def callback(model_cb, where):
        if where == GRB.Callback.MIPSOL and logger is not None and primary_type == "PSCLP":
            xv = model_cb.cbGetSolution([x[i] for i in range(n)])
            open_list = [i for i in range(n) if xv[i] > 0.5]
            total_cost, covered_demand, _ = evaluate_open_set(open_list, fac_to_cust, demands, costs)
            if total_cost + 1e-9 < best_cb_obj[0]:
                best_cb_obj[0] = total_cost
                incumbent_results["PSCLP"] = {
                    "objective_value": float(total_cost),
                    "open_facilities": sorted(open_list),
                    "covered_demand": float(covered_demand),
                }
                sol = build_full_solution(primary_type, incumbent_results["PSCLP"], incumbent_results["MCLP"])
                logger.log_solution(float(total_cost), sol)

    if logger is not None and primary_type == "PSCLP":
        model.optimize(callback)
    else:
        model.optimize()

    if model.SolCount > 0:
        open_list = sorted([i for i in range(n) if x[i].X > 0.5])
        total_cost, covered_demand, _ = evaluate_open_set(open_list, fac_to_cust, demands, costs)
        return {
            "objective_value": float(total_cost),
            "open_facilities": open_list,
            "covered_demand": float(covered_demand),
        }

    # fallback to warm start evaluation
    total_cost, covered_demand, _ = evaluate_open_set(warm_open, fac_to_cust, demands, costs)
    if covered_demand + 1e-9 >= D:
        return {
            "objective_value": float(total_cost),
            "open_facilities": sorted(warm_open),
            "covered_demand": float(covered_demand),
        }

    # no feasible found
    return {
        "objective_value": float(1e18),
        "open_facilities": [],
        "covered_demand": 0.0,
    }


def solve_mclp(
    n, m, B, costs, demands, cust_to_fac, fac_to_cust,
    time_limit, warm_open, logger, primary_type, incumbent_results
):
    model = gp.Model("MCLP")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.0, float(time_limit))

    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    y = {}
    for j in range(m):
        ub = 1.0 if len(cust_to_fac[j]) > 0 else 0.0
        y[j] = model.addVar(vtype=GRB.BINARY, ub=ub, name=f"y_{j}")

    for j in range(m):
        if len(cust_to_fac[j]) > 0:
            model.addConstr(y[j] <= gp.quicksum(x[i] for i in cust_to_fac[j]), name=f"link_{j}")

    model.addConstr(gp.quicksum(costs[i] * x[i] for i in range(n)) <= B, name="budget")
    model.setObjective(gp.quicksum(demands[j] * y[j] for j in range(m)), GRB.MAXIMIZE)

    # Warm start
    warm_set = set(warm_open)
    for i in range(n):
        x[i].Start = 1.0 if i in warm_set else 0.0
    _, _, warm_cov = evaluate_open_set(warm_open, fac_to_cust, demands, costs)
    for j in range(m):
        y[j].Start = 1.0 if j in warm_cov else 0.0

    best_cb_obj = [-float("inf")]

    def callback(model_cb, where):
        if where == GRB.Callback.MIPSOL and logger is not None and primary_type == "MCLP":
            xv = model_cb.cbGetSolution([x[i] for i in range(n)])
            open_list = [i for i in range(n) if xv[i] > 0.5]
            _, covered_demand, _ = evaluate_open_set(open_list, fac_to_cust, demands, costs)
            if covered_demand > best_cb_obj[0] + 1e-9:
                best_cb_obj[0] = covered_demand
                incumbent_results["MCLP"] = {
                    "objective_value": float(covered_demand),
                    "open_facilities": sorted(open_list),
                    "covered_demand": float(covered_demand),
                }
                sol = build_full_solution(primary_type, incumbent_results["PSCLP"], incumbent_results["MCLP"])
                logger.log_solution(float(covered_demand), sol)

    if logger is not None and primary_type == "MCLP":
        model.optimize(callback)
    else:
        model.optimize()

    if model.SolCount > 0:
        open_list = sorted([i for i in range(n) if x[i].X > 0.5])
        _, covered_demand, _ = evaluate_open_set(open_list, fac_to_cust, demands, costs)
        return {
            "objective_value": float(covered_demand),
            "open_facilities": open_list,
            "covered_demand": float(covered_demand),
        }

    # fallback
    _, covered_demand, _ = evaluate_open_set(warm_open, fac_to_cust, demands, costs)
    return {
        "objective_value": float(covered_demand),
        "open_facilities": sorted(warm_open),
        "covered_demand": float(covered_demand),
    }


def main():
    args = parse_args()
    t0 = time.time()

    instance = load_instance(args.instance_path)

    n = int(instance["num_facilities"])
    m = int(instance["num_customers"])
    costs = [float(c) for c in instance["facility_cost"]]
    demands = [float(d) for d in instance["customer_demands"]]

    total_demand = float(instance.get("total_demand_D_bar", sum(demands)))
    if "covering_demand_D" in instance and instance["covering_demand_D"] is not None:
        D = float(instance["covering_demand_D"])
    else:
        frac = float(instance.get("covering_demand_fraction", 0.0))
        D = frac * total_demand

    B = float(instance.get("budget_B", 0.0))

    # determine primary problem type
    primary_type = None
    if "primary_problem_type" in instance:
        primary_type = str(instance["primary_problem_type"])
    elif "problem_type" in instance:
        primary_type = str(instance["problem_type"])
    elif "problem_types" in instance and isinstance(instance["problem_types"], list) and len(instance["problem_types"]) > 0:
        primary_type = str(instance["problem_types"][0])
    if primary_type not in {"PSCLP", "MCLP"}:
        primary_type = "PSCLP"

    fac_to_cust, cust_to_fac = build_coverage(instance)

    # Heuristic incumbents
    warm_ps = greedy_psclp(n, m, D, costs, demands, fac_to_cust)
    ps_cost, ps_covd, _ = evaluate_open_set(warm_ps, fac_to_cust, demands, costs)
    if ps_covd + 1e-9 < D:
        # mark as very bad if not meeting threshold
        heur_ps_result = {
            "objective_value": float(1e18),
            "open_facilities": [],
            "covered_demand": float(ps_covd),
        }
    else:
        heur_ps_result = {
            "objective_value": float(ps_cost),
            "open_facilities": warm_ps,
            "covered_demand": float(ps_covd),
        }

    warm_mc = greedy_mclp(n, B, costs, demands, fac_to_cust)
    _, mc_covd, _ = evaluate_open_set(warm_mc, fac_to_cust, demands, costs)
    heur_mc_result = {
        "objective_value": float(mc_covd),
        "open_facilities": warm_mc,
        "covered_demand": float(mc_covd),
    }

    incumbent_results = {
        "PSCLP": heur_ps_result.copy(),
        "MCLP": heur_mc_result.copy(),
    }

    logger = None
    if args.log_path:
        logger = SolutionLogger(args.log_path, sense=("minimize" if primary_type == "PSCLP" else "maximize"))

    def remaining_time():
        return max(0.0, args.time_limit - (time.time() - t0))

    # Solve primary first (for logging consistency), then secondary
    other_type = "MCLP" if primary_type == "PSCLP" else "PSCLP"
    order = [primary_type, other_type]

    for idx, ptype in enumerate(order):
        rem = remaining_time()
        if rem <= 1e-6:
            break

        # Keep some time for second problem if first one
        if idx == 0 and len(order) == 2:
            alloc = max(0.1, rem * 0.6)
        else:
            alloc = rem

        if ptype == "PSCLP":
            res = solve_psclp(
                n=n, m=m, D=D, costs=costs, demands=demands,
                cust_to_fac=cust_to_fac, fac_to_cust=fac_to_cust,
                time_limit=alloc, warm_open=warm_ps,
                logger=logger, primary_type=primary_type, incumbent_results=incumbent_results
            )
            incumbent_results["PSCLP"] = res
        else:
            res = solve_mclp(
                n=n, m=m, B=B, costs=costs, demands=demands,
                cust_to_fac=cust_to_fac, fac_to_cust=fac_to_cust,
                time_limit=alloc, warm_open=warm_mc,
                logger=logger, primary_type=primary_type, incumbent_results=incumbent_results
            )
            incumbent_results["MCLP"] = res

    final_solution = build_full_solution(
        primary_type=primary_type,
        results_psclp=incumbent_results["PSCLP"],
        results_mclp=incumbent_results["MCLP"],
    )

    # Final log snapshot
    if logger is not None:
        logger.log_solution(float(final_solution["objective_value"]), final_solution)

    with open(args.solution_path, "w") as f:
        json.dump(final_solution, f, indent=2)


if __name__ == "__main__":
    main()