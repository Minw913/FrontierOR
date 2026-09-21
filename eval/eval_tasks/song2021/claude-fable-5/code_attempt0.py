import argparse
import json
import math
import sys

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def read_instance(path):
    with open(path, "r") as f:
        return json.load(f)


def build_solution_dict(assignments, orders, capacities, sigma, H):
    """assignments: dict order_id -> (s, f, {t: intensity}) ; returns (obj, solution_dict)"""
    loads = [0.0] * (H + 2)
    total_tard_cost = 0.0
    orders_solution = []
    for od in orders:
        oid = od["order_id"]
        s, f, xs = assignments[oid]
        p = od["work_content_p"]
        d = od["due_date_d"]
        w = od["tardiness_cost_w"]
        tard = max(0, f - d)
        total_tard_cost += w * tard
        intens = {}
        for t, v in sorted(xs.items()):
            if v < 0:
                v = 0.0
            intens[str(t)] = float(v)
            loads[t] += p * v
        orders_solution.append({
            "order_id": oid,
            "chosen_interval": [int(s), int(f)],
            "intensities": intens
        })
    nonreg = {}
    total_over = 0.0
    for t in range(1, H + 1):
        cap = capacities[t - 1]
        over = loads[t] - cap
        if over > 1e-9:
            nonreg[str(t)] = float(over)
            total_over += over
    obj = total_tard_cost + sigma * total_over
    sol = {
        "objective_value": float(obj),
        "orders_solution": orders_solution,
        "nonregular_capacity_by_period": nonreg
    }
    return obj, sol


def greedy_fallback(orders, H):
    """Assign each order earliest feasible interval of minimal length, uniform intensity."""
    assignments = {}
    for od in orders:
        r = max(1, int(od["release_date_r"]))
        UB = od["intensity_upper_bound_UB"]
        LB = od["intensity_lower_bound_LB"]
        Lmin = max(int(od["lower_exec_interval_length_l"]),
                   int(math.ceil(1.0 / UB - 1e-9)))
        Lmax = min(int(od["upper_exec_interval_length_l_bar"]),
                   int(math.floor(1.0 / LB + 1e-9)))
        if Lmax < Lmin:
            Lmax = Lmin
        # pick minimal length fitting in horizon
        L = Lmin
        s = r
        if s + L - 1 > H:
            s = max(1, H - L + 1)
        f = s + L - 1
        val = 1.0 / L
        xs = {t: val for t in range(s, f + 1)}
        assignments[od["order_id"]] = (s, f, xs)
    return assignments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if (args.log_path and SolutionLogger) else None

    inst = read_instance(args.instance_path)
    H = int(inst["planning_horizon_H"])
    n = int(inst["num_orders_n"])
    sigma = float(inst["unit_cost_nonregular_capacity_sigma"])
    capacities = inst["capacities"]
    orders = inst["orders"]

    import time as _time
    t_start = _time.time()

    # Fallback solution (guaranteed available)
    fb_assign = greedy_fallback(orders, H)
    fb_obj, fb_sol = build_solution_dict(fb_assign, orders, capacities, sigma, H)
    best_obj = fb_obj
    best_sol = fb_sol
    if logger:
        logger.log_solution(fb_obj, fb_sol)

    # Build MIP
    model = gp.Model("order_scheduling")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    y = {}          # (oid, s, f) -> binary var
    x = {}          # (oid, t) -> continuous var
    order_intervals = {}  # oid -> list of (s, f)
    order_periods = {}    # oid -> set of periods
    cover = {}      # (oid, t) -> list of interval keys covering t

    for od in orders:
        oid = od["order_id"]
        r = max(1, int(od["release_date_r"]))
        UB = od["intensity_upper_bound_UB"]
        LB = od["intensity_lower_bound_LB"]
        Lmin = max(int(od["lower_exec_interval_length_l"]),
                   int(math.ceil(1.0 / UB - 1e-9)), 1)
        Lmax = min(int(od["upper_exec_interval_length_l_bar"]),
                   int(math.floor(1.0 / LB + 1e-9)))
        if Lmax < Lmin:
            Lmax = Lmin
        ivs = []
        for s in range(r, H + 1):
            for L in range(Lmin, Lmax + 1):
                f = s + L - 1
                if f > H:
                    break
                ivs.append((s, f))
        if not ivs:
            # relax: allow interval clipped to horizon end
            L = min(Lmin, H)
            s = max(1, H - L + 1)
            ivs = [(s, H)]
        order_intervals[oid] = ivs
        periods = set()
        for (s, f) in ivs:
            for t in range(s, f + 1):
                periods.add(t)
        order_periods[oid] = periods

        d = od["due_date_d"]
        w = od["tardiness_cost_w"]
        for (s, f) in ivs:
            tard = max(0, f - d)
            y[(oid, s, f)] = model.addVar(vtype=GRB.BINARY, obj=w * tard,
                                          name="y_%d_%d_%d" % (oid, s, f))
        for t in periods:
            x[(oid, t)] = model.addVar(lb=0.0, ub=UB, vtype=GRB.CONTINUOUS,
                                       name="x_%d_%d" % (oid, t))
            cover[(oid, t)] = []
        for (s, f) in ivs:
            for t in range(s, f + 1):
                cover[(oid, t)].append((oid, s, f))

    O = {}
    for t in range(1, H + 1):
        O[t] = model.addVar(lb=0.0, obj=sigma, vtype=GRB.CONTINUOUS, name="O_%d" % t)

    model.ModelSense = GRB.MINIMIZE
    model.update()

    for od in orders:
        oid = od["order_id"]
        ivs = order_intervals[oid]
        model.addConstr(gp.quicksum(y[(oid, s, f)] for (s, f) in ivs) == 1)
        model.addConstr(gp.quicksum(x[(oid, t)] for t in order_periods[oid]) == 1)
        UB = od["intensity_upper_bound_UB"]
        LB = od["intensity_lower_bound_LB"]
        for t in order_periods[oid]:
            cov = gp.quicksum(y[k] for k in cover[(oid, t)])
            model.addConstr(x[(oid, t)] <= UB * cov)
            model.addConstr(x[(oid, t)] >= LB * cov)

    p_of = {od["order_id"]: od["work_content_p"] for od in orders}
    for t in range(1, H + 1):
        terms = [p_of[oid] * x[(oid, t)] for od_ in [None] for oid in
                 [o["order_id"] for o in orders] if (oid, t) in x]
        if terms:
            model.addConstr(O[t] >= gp.quicksum(terms) - capacities[t - 1])

    # Callback: log every incumbent
    y_keys = list(y.keys())
    y_vars = [y[k] for k in y_keys]
    x_keys = list(x.keys())
    x_vars = [x[k] for k in x_keys]

    state = {"best": best_obj, "best_sol": best_sol}

    def cb(m, where):
        if where == GRB.Callback.MIPSOL:
            try:
                yv = m.cbGetSolution(y_vars)
                xv = m.cbGetSolution(x_vars)
                assignments = {}
                chosen = {}
                for k, v in zip(y_keys, yv):
                    if v > 0.5:
                        chosen[k[0]] = (k[1], k[2])
                xvals = {}
                for k, v in zip(x_keys, xv):
                    xvals[k] = v
                for od in orders:
                    oid = od["order_id"]
                    if oid not in chosen:
                        return
                    s, f = chosen[oid]
                    xs = {}
                    for t in range(s, f + 1):
                        v = xvals.get((oid, t), 0.0)
                        if v < 0:
                            v = 0.0
                        xs[t] = v
                    tot = sum(xs.values())
                    if tot > 1e-9:
                        xs = {t: v / tot for t, v in xs.items()}
                    assignments[oid] = (s, f, xs)
                obj, sol = build_solution_dict(assignments, orders, capacities, sigma, H)
                if obj < state["best"] - 1e-9:
                    state["best"] = obj
                    state["best_sol"] = sol
                    if logger:
                        logger.log_solution(obj, sol)
            except Exception:
                pass

    elapsed = _time.time() - t_start
    remaining = max(1.0, args.time_limit - elapsed - 2.0)
    model.Params.TimeLimit = remaining

    try:
        model.optimize(cb)
    except Exception:
        pass

    # Extract final solution from model if available
    if model.SolCount > 0:
        try:
            assignments = {}
            chosen = {}
            for k in y_keys:
                if y[k].X > 0.5:
                    chosen[k[0]] = (k[1], k[2])
            ok = True
            for od in orders:
                oid = od["order_id"]
                if oid not in chosen:
                    ok = False
                    break
                s, f = chosen[oid]
                xs = {}
                for t in range(s, f + 1):
                    v = x[(oid, t)].X if (oid, t) in x else 0.0
                    if v < 0:
                        v = 0.0
                    xs[t] = v
                tot = sum(xs.values())
                if tot > 1e-9:
                    xs = {t: v / tot for t, v in xs.items()}
                assignments[oid] = (s, f, xs)
            if ok:
                obj, sol = build_solution_dict(assignments, orders, capacities, sigma, H)
                if obj < state["best"] - 1e-9:
                    state["best"] = obj
                    state["best_sol"] = sol
                    if logger:
                        logger.log_solution(obj, sol)
        except Exception:
            pass

    final_sol = state["best_sol"]
    with open(args.solution_path, "w") as f:
        json.dump(final_sol, f, indent=2)


if __name__ == "__main__":
    main()