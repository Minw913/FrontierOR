#!/usr/bin/env python3
"""
Solver for the Partial Set Covering Location Problem (PSCLP) and the
Maximal Covering Location Problem (MCLP).

Approach:
  1. Read the instance, build / read coverage sets.
  2. Compute fast greedy (lazy priority-queue) feasible solutions that act
     as incumbents and as MIP starts.
  3. Solve both variants with Gurobi (exact MIP formulations), splitting the
     available time. Incumbents found during the search are tracked and the
     best one for the primary variant is reported to the SolutionLogger.
  4. Always write a final solution JSON following solution_schema.json.
"""

import argparse
import heapq
import json
import os
import time

try:
    from solution_logger import SolutionLogger
except Exception:  # pragma: no cover - fallback if module unavailable
    SolutionLogger = None


# --------------------------------------------------------------------------- #
# Instance handling
# --------------------------------------------------------------------------- #
def load_instance(path):
    with open(path, "r") as f:
        return json.load(f)


def build_coverage(inst):
    """Return (facility_to_customers, customer_to_facilities) as lists of lists."""
    nF = int(inst["num_facilities"])
    nC = int(inst["num_customers"])
    f2c = inst.get("facility_to_customers_coverage")
    c2f = inst.get("customer_to_facilities_coverage")
    if f2c is not None and c2f is not None:
        F2C = [list(map(int, f2c.get(str(i), []))) for i in range(nF)]
        C2F = [list(map(int, c2f.get(str(j), []))) for j in range(nC)]
        return F2C, C2F

    # Derive coverage from coordinates.
    R = float(inst["radius_of_coverage"])
    r2 = R * R + 1e-9
    fac = inst["facility_coordinates"]
    cus = inst["customer_coordinates"]
    F2C = [[] for _ in range(nF)]
    C2F = [[] for _ in range(nC)]
    for i in range(nF):
        xi, yi = float(fac[i][0]), float(fac[i][1])
        for j in range(nC):
            xj, yj = float(cus[j][0]), float(cus[j][1])
            dx = xi - xj
            dy = yi - yj
            if dx * dx + dy * dy <= r2:
                F2C[i].append(j)
                C2F[j].append(i)
    return F2C, C2F


# --------------------------------------------------------------------------- #
# Greedy heuristics (lazy priority queue implementations)
# --------------------------------------------------------------------------- #
def greedy_psclp(cost, demand, F2C, D):
    """Greedy min-cost partial cover: repeatedly add the facility with the
    best (newly covered demand)/(cost) ratio until the threshold is met."""
    covered = [False] * len(demand)
    covered_d = 0.0
    chosen = []
    heap = []
    for i in range(len(cost)):
        nd = sum(demand[j] for j in F2C[i])
        if nd <= 0.0:
            continue
        ratio = nd / cost[i] if cost[i] > 0.0 else float("inf")
        heapq.heappush(heap, (-ratio, i))
    while covered_d < D - 1e-9 and heap:
        _, i = heapq.heappop(heap)
        nd = sum(demand[j] for j in F2C[i] if not covered[j])
        if nd <= 0.0:
            continue
        ratio = nd / cost[i] if cost[i] > 0.0 else float("inf")
        if heap and ratio < -heap[0][0] - 1e-12:
            heapq.heappush(heap, (-ratio, i))
            continue
        chosen.append(i)
        for j in F2C[i]:
            if not covered[j]:
                covered[j] = True
                covered_d += demand[j]
    return chosen


def greedy_mclp(cost, demand, F2C, B):
    """Greedy budgeted max coverage by (newly covered demand)/(cost) ratio."""
    covered = [False] * len(demand)
    chosen = []
    spent = 0.0
    heap = []
    for i in range(len(cost)):
        if cost[i] > B + 1e-9:
            continue
        nd = sum(demand[j] for j in F2C[i])
        if nd <= 0.0:
            continue
        ratio = nd / cost[i] if cost[i] > 0.0 else float("inf")
        heapq.heappush(heap, (-ratio, i))
    while heap:
        _, i = heapq.heappop(heap)
        if spent + cost[i] > B + 1e-9:
            continue
        nd = sum(demand[j] for j in F2C[i] if not covered[j])
        if nd <= 0.0:
            continue
        ratio = nd / cost[i] if cost[i] > 0.0 else float("inf")
        if heap and ratio < -heap[0][0] - 1e-12:
            heapq.heappush(heap, (-ratio, i))
            continue
        chosen.append(i)
        spent += cost[i]
        for j in F2C[i]:
            covered[j] = True
    return chosen


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance_path", required=True)
    p.add_argument("--solution_path", required=True)
    p.add_argument("--time_limit", type=int, required=True)
    p.add_argument("--log_path", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    deadline = t0 + max(0, args.time_limit)

    inst = load_instance(args.instance_path)
    nF = int(inst["num_facilities"])
    nC = int(inst["num_customers"])
    cost = [float(c) for c in inst["facility_cost"]]
    demand = [float(d) for d in inst["customer_demands"]]
    F2C, C2F = build_coverage(inst)

    total_demand = sum(demand)
    D = inst.get("covering_demand_D")
    if D is None:
        D = float(inst.get("covering_demand_fraction", 1.0)) * total_demand
    D = float(D)
    B = float(inst.get("budget_B", 0.0))

    # Determine the primary variant.
    ptypes = inst.get("problem_types", inst.get("problem_type", ["PSCLP"]))
    if isinstance(ptypes, str):
        ptypes = [ptypes]
    primary = ptypes[0] if ptypes else "PSCLP"
    if primary not in ("PSCLP", "MCLP"):
        primary = "PSCLP"
    secondary = "MCLP" if primary == "PSCLP" else "PSCLP"

    sense = "minimize" if primary == "PSCLP" else "maximize"
    logger = None
    if args.log_path and SolutionLogger is not None:
        try:
            logger = SolutionLogger(args.log_path, sense=sense)
        except Exception:
            logger = None

    # ----------------------------------------------------------------- #
    # Solution bookkeeping
    # ----------------------------------------------------------------- #
    def build_result(variant, open_list):
        open_sorted = sorted(set(int(i) for i in open_list))
        covered = set()
        for i in open_sorted:
            covered.update(F2C[i])
        cd = float(sum(demand[j] for j in covered))
        oc = float(sum(cost[i] for i in open_sorted))
        obj = oc if variant == "PSCLP" else cd
        return {
            "objective_value": float(obj),
            "open_facilities": open_sorted,
            "covered_demand": float(cd),
        }

    def make_solution():
        prim_obj = best[primary]["objective_value"]
        return {
            "objective_value": float(prim_obj),
            "primary_problem_type": primary,
            "results": {
                "PSCLP": best["PSCLP"],
                "MCLP": best["MCLP"],
            },
        }

    def log_primary():
        if logger:
            try:
                sol = make_solution()
                logger.log_solution(sol["objective_value"], sol)
            except Exception:
                pass

    def try_update(variant, open_list):
        """Update incumbent if feasible and better; log if primary improved."""
        res = build_result(variant, open_list)
        cur = best[variant]
        if variant == "PSCLP":
            if res["covered_demand"] < D - 1e-6:
                return False
            if res["objective_value"] < cur["objective_value"] - 1e-9:
                best["PSCLP"] = res
                if primary == "PSCLP":
                    log_primary()
                return True
        else:
            ocost = sum(cost[i] for i in res["open_facilities"])
            if ocost > B + 1e-6:
                return False
            if res["objective_value"] > cur["objective_value"] + 1e-9:
                best["MCLP"] = res
                if primary == "MCLP":
                    log_primary()
                return True
        return False

    # ----------------------------------------------------------------- #
    # Initial incumbents
    # ----------------------------------------------------------------- #
    best = {
        # All-open is the fallback feasible solution for PSCLP.
        "PSCLP": build_result("PSCLP", list(range(nF))),
        # Empty solution is always feasible for MCLP.
        "MCLP": build_result("MCLP", []),
    }

    # Greedy PSCLP improvement.
    try:
        g = greedy_psclp(cost, demand, F2C, D)
        try_update("PSCLP", g)
    except Exception:
        pass

    # Greedy MCLP improvement; also compare against the best single facility
    # (standard trick for budgeted maximum coverage).
    try:
        g = greedy_mclp(cost, demand, F2C, B)
        try_update("MCLP", g)
        best_single, best_single_val = -1, -1.0
        for i in range(nF):
            if cost[i] <= B + 1e-9:
                v = sum(demand[j] for j in F2C[i])
                if v > best_single_val:
                    best_single_val, best_single = v, i
        if best_single >= 0:
            try_update("MCLP", [best_single])
    except Exception:
        pass

    log_primary()  # record the initial incumbent

    # ----------------------------------------------------------------- #
    # Gurobi exact solves
    # ----------------------------------------------------------------- #
    def gurobi_solve(variant, tl):
        if tl <= 0.05:
            return
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except Exception:
            return
        try:
            m = gp.Model(variant)
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.TimeLimit = float(max(0.1, tl))
            m.Params.OutputFlag = 0

            x = m.addVars(nF, vtype=GRB.BINARY, name="x")
            y = m.addVars(nC, vtype=GRB.BINARY, name="y")

            for j in range(nC):
                if not C2F[j]:
                    y[j].UB = 0.0

            if variant == "PSCLP":
                m.setObjective(
                    gp.quicksum(cost[i] * x[i] for i in range(nF)), GRB.MINIMIZE
                )
                m.addConstr(
                    gp.quicksum(demand[j] * y[j] for j in range(nC)) >= D,
                    name="demand_requirement",
                )
            else:
                m.setObjective(
                    gp.quicksum(demand[j] * y[j] for j in range(nC)), GRB.MAXIMIZE
                )
                m.addConstr(
                    gp.quicksum(cost[i] * x[i] for i in range(nF)) <= B,
                    name="budget",
                )

            for j in range(nC):
                if C2F[j]:
                    m.addConstr(
                        y[j] <= gp.quicksum(x[i] for i in C2F[j]),
                        name="link_%d" % j,
                    )

            # MIP start from current incumbent.
            start_set = set(best[variant]["open_facilities"])
            for i in range(nF):
                x[i].Start = 1.0 if i in start_set else 0.0

            xlist = [x[i] for i in range(nF)]

            def cb(model, where):
                if where == GRB.Callback.MIPSOL:
                    try:
                        xv = model.cbGetSolution(xlist)
                    except Exception:
                        return
                    open_f = [i for i, v in enumerate(xv) if v > 0.5]
                    try_update(variant, open_f)

            m.optimize(cb)

            if m.SolCount > 0:
                try:
                    xv = m.getAttr("X", xlist)
                    open_f = [i for i, v in enumerate(xv) if v > 0.5]
                    try_update(variant, open_f)
                except Exception:
                    pass
            try:
                m.dispose()
            except Exception:
                pass
        except Exception:
            return

    reserve = min(1.0, max(0.2, 0.02 * max(1, args.time_limit)))

    try:
        rem = deadline - time.time() - reserve
        if rem > 0.5:
            first_tl = max(0.5, rem * 0.5)
            gurobi_solve(primary, first_tl)
        rem = deadline - time.time() - reserve
        if rem > 0.2:
            gurobi_solve(secondary, rem)
    except Exception:
        pass

    # ----------------------------------------------------------------- #
    # Final output
    # ----------------------------------------------------------------- #
    solution = make_solution()
    try:
        out_dir = os.path.dirname(args.solution_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass
    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)

    if logger:
        try:
            logger.log_solution(solution["objective_value"], solution)
        except Exception:
            pass


if __name__ == "__main__":
    main()