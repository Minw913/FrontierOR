#!/usr/bin/env python3
import argparse
import heapq
import json
import time

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def make_functions(allocation, opening, m):
    n = len(allocation)

    def eval_y(y):
        denom = [0.0] * m
        fixed = 0.0
        for i in range(n):
            if y[i]:
                fixed += opening[i]
                row = allocation[i]
                for j in range(m):
                    c = row[j]
                    if c > 0.0:
                        denom[j] += 1.0 / c
        obj = fixed
        for d in denom:
            if d <= 0.0:
                return float("inf"), denom
            obj += 1.0 / d
        return obj, denom

    def solution_from_y(y):
        obj, denom = eval_y(y)
        x = []
        for i in range(n):
            if not y[i]:
                x.append([0.0] * m)
            else:
                row = allocation[i]
                xi = [0.0] * m
                for j in range(m):
                    c = row[j]
                    if c > 0.0 and denom[j] > 0.0:
                        xi[j] = (1.0 / c) / denom[j]
                x.append(xi)
        sol = {
            "objective_value": float(obj),
            "variant": "quadratic",
            "y": [1 if v else 0 for v in y],
            "x": x,
        }
        return obj, sol, denom

    return eval_y, solution_from_y


def greedy_close(allocation, opening, denom0, end_time):
    # Descent by closing facilities while a single closure improves the exact
    # fixed-open quadratic objective. Used only as a warm start/incumbent.
    n = len(allocation)
    m = len(denom0)
    if n <= 1:
        return [1] * n, sum(opening) + sum(1.0 / d for d in denom0)

    denom = denom0[:]
    inv_denom = [1.0 / d for d in denom]
    is_open = [True] * n
    open_count = n
    tol = 1e-10

    def close_improvement(i):
        # cost decrease from closing i, at current denominators
        row = allocation[i]
        penalty = 0.0
        for j in range(m):
            c = row[j]
            if c > 0.0:
                a = 1.0 / c
                dm = denom[j] - a
                if dm <= 1e-14:
                    return float("-inf"), None
                penalty += (1.0 / dm) - inv_denom[j]
        return opening[i] - penalty, row

    heap = []
    for i in range(n):
        if (i & 255) == 0 and time.monotonic() > end_time:
            return [1] * n, sum(opening) + sum(inv_denom)
        imp, _ = close_improvement(i)
        if imp > tol:
            heap.append((-imp, i))
    heapq.heapify(heap)

    cost = sum(opening) + sum(inv_denom)
    while heap and open_count > 1 and time.monotonic() < end_time:
        _, i = heapq.heappop(heap)
        if not is_open[i]:
            continue
        imp, row = close_improvement(i)
        if imp <= tol:
            continue
        if heap and imp < -heap[0][0] - tol:
            heapq.heappush(heap, (-imp, i))
            continue

        is_open[i] = False
        open_count -= 1
        cost -= imp
        for j in range(m):
            c = row[j]
            if c > 0.0:
                denom[j] -= 1.0 / c
                inv_denom[j] = 1.0 / denom[j]

    return [1 if v else 0 for v in is_open], cost


def write_solution(path, sol):
    with open(path, "w") as f:
        json.dump(sol, f, separators=(",", ":"))


def main():
    args = parse_args()
    start_time = time.monotonic()
    time_limit = max(0, int(args.time_limit))
    deadline = start_time + time_limit
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    n = int(data["num_facilities"])
    m = int(data["num_customers"])
    opening = [float(v) for v in data["opening_costs"]]
    allocation = data["allocation_costs"]

    if m == 0:
        sol = {
            "objective_value": 0.0,
            "variant": "quadratic",
            "y": [0] * n,
            "x": [[] for _ in range(n)],
        }
        if logger:
            logger.log_solution(0.0, sol)
        write_solution(args.solution_path, sol)
        return

    if n == 0:
        sol = {"objective_value": 0.0, "variant": "quadratic", "y": [], "x": []}
        if logger:
            logger.log_solution(0.0, sol)
        write_solution(args.solution_path, sol)
        return

    eval_y, solution_from_y = make_functions(allocation, opening, m)

    # One pass: all-open denominators and best single-facility solution.
    denom_all = [0.0] * m
    single_costs = opening[:]
    for i in range(n):
        row = allocation[i]
        sc = single_costs[i]
        for j in range(m):
            c = float(row[j])
            if c > 0.0:
                denom_all[j] += 1.0 / c
            sc += c
        single_costs[i] = sc

    finite_all = all(d > 0.0 for d in denom_all)
    if finite_all:
        all_cost = sum(opening) + sum(1.0 / d for d in denom_all)
    else:
        all_cost = float("inf")

    best_single = min(range(n), key=lambda i: single_costs[i])
    y_single = [0] * n
    y_single[best_single] = 1

    if single_costs[best_single] < all_cost:
        best_y = y_single[:]
        best_obj = single_costs[best_single]
    else:
        best_y = [1] * n
        best_obj = all_cost

    # Cheap descent from all-open, strictly time-capped; exact solver gets the rest.
    remaining = deadline - time.monotonic()
    if finite_all and remaining > 0.05:
        heuristic_budget = min(3.0, remaining * 0.10)
        if n * m > 2_000_000:
            heuristic_budget = min(heuristic_budget, 0.5)
        gy, gobj = greedy_close(allocation, opening, denom_all, min(deadline, time.monotonic() + heuristic_budget))
        if gobj < best_obj - 1e-9 * max(1.0, abs(best_obj)):
            best_y, best_obj = gy, gobj
        if single_costs[best_single] < best_obj - 1e-9 * max(1.0, abs(best_obj)):
            best_y, best_obj = y_single[:], single_costs[best_single]

    best_obj, best_sol, best_denom = solution_from_y(best_y)
    best_y = best_sol["y"][:]
    if logger:
        logger.log_solution(best_obj, best_sol)
    last_logged = best_obj

    # Exact compact mixed-integer conic formulation.
    # For fixed y, customer j has s_j = sum_i y_i / c_ij and optimal quadratic
    # allocation cost 1/s_j.  With p=(s+t), q=(s-t), r=2, the cone
    # q^2+r^2 <= p^2 enforces s*t >= 1, so t=1/s at optimum.
    remaining = deadline - time.monotonic()
    if remaining > 0.25:
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except Exception:
            gp = None
            GRB = None

        if gp is not None:
            try:
                model = gp.Model("separable_convex_quadratic_UFL")
                model.Params.Seed = 0
                model.Params.MIPGap = 1e-4
                model.Params.NumericFocus = 0
                model.Params.Threads = 1
                model.Params.OutputFlag = 0

                yv = model.addVars(n, vtype=GRB.BINARY, name="y")
                pv = model.addVars(m, lb=0.0, name="p")
                qv = model.addVars(m, lb=-GRB.INFINITY, name="q")
                rv = model.addVars(m, lb=2.0, ub=2.0, name="r")

                model.setObjective(
                    gp.quicksum(opening[i] * yv[i] for i in range(n))
                    + 0.5 * gp.quicksum(pv[j] - qv[j] for j in range(m)),
                    GRB.MINIMIZE,
                )

                exprs = [pv[j] + qv[j] for j in range(m)]
                for i in range(n):
                    row = allocation[i]
                    yi = yv[i]
                    for j in range(m):
                        c = float(row[j])
                        if c > 0.0:
                            exprs[j].addTerm((-2.0) / c, yi)
                for j in range(m):
                    model.addConstr(exprs[j] == 0.0, name="balance_%d" % j)
                    model.addConstr(
                        qv[j] * qv[j] + rv[j] * rv[j] <= pv[j] * pv[j],
                        name="cone_%d" % j,
                    )

                # Feasible MIP start from the heuristic incumbent.
                for i in range(n):
                    yv[i].Start = float(best_y[i])
                for j in range(m):
                    d = best_denom[j]
                    if d > 1e-12:
                        pv[j].Start = d + 1.0 / d
                        qv[j].Start = d - 1.0 / d
                    else:
                        pv[j].Start = 2.0
                        qv[j].Start = 0.0
                    rv[j].Start = 2.0

                yvars = [yv[i] for i in range(n)]

                def incumbent_callback(cb_model, where):
                    nonlocal best_obj, best_y, last_logged
                    if where != GRB.Callback.MIPSOL:
                        return
                    if time.monotonic() > deadline:
                        cb_model.terminate()
                        return
                    try:
                        vals = cb_model.cbGetSolution(yvars)
                        yb = [1 if vals[i] > 0.5 else 0 for i in range(n)]
                        if not any(yb):
                            return
                        obj, sol, _ = solution_from_y(yb)
                    except Exception:
                        return
                    if obj < best_obj - 1e-8 * max(1.0, abs(best_obj)):
                        best_obj = obj
                        best_y = yb
                        if logger:
                            logger.log_solution(obj, sol)
                            last_logged = obj

                solve_time = deadline - time.monotonic() - 0.20
                if solve_time > 0.0:
                    model.Params.TimeLimit = solve_time
                    model.optimize(incumbent_callback)

                if model.SolCount > 0:
                    yb = [1 if yv[i].X > 0.5 else 0 for i in range(n)]
                    if any(yb):
                        obj, sol, _ = solution_from_y(yb)
                        if obj < best_obj - 1e-8 * max(1.0, abs(best_obj)):
                            best_obj = obj
                            best_y = yb
                            if logger and obj < last_logged - 1e-8 * max(1.0, abs(last_logged)):
                                logger.log_solution(obj, sol)
                                last_logged = obj
            except Exception:
                pass

    final_obj, final_sol, _ = solution_from_y(best_y)
    if logger and final_obj < last_logged - 1e-8 * max(1.0, abs(last_logged)):
        logger.log_solution(final_obj, final_sol)
    write_solution(args.solution_path, final_sol)


if __name__ == "__main__":
    main()