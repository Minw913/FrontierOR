import argparse
import json
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance_path", required=True)
    p.add_argument("--solution_path", required=True)
    p.add_argument("--time_limit", required=True, type=int)
    p.add_argument("--log_path", default=None)
    return p.parse_args()


def read_initial_inventory(data, n):
    for name in (
        "initial_inventory",
        "initial_inventories",
        "initial_inventory_levels",
        "starting_inventory",
    ):
        if name in data:
            v = data[name]
            if isinstance(v, dict):
                return [
                    float(v.get(f"item_{i + 1}", v.get(str(i + 1), 0.0)))
                    for i in range(n)
                ]
            return [float(v[i]) for i in range(n)]
    return [0.0] * n


def solve_instance(data, time_limit, start_time, logger):
    t0 = time.monotonic()
    overall_deadline = start_time + max(0, int(time_limit))

    n = int(data["num_items"])
    nt = int(data["num_periods"])
    nm = int(data["num_machines"])

    caps = {}
    for i in range(n):
        caps[i] = sorted(set(int(m) - 1 for m in data["machine_capabilities"][i]))
    items_on_machine = {m: [i for i in range(n) if m in caps[i]] for m in range(nm)}

    demand = [
        [float(data["demand"][i][t]) for t in range(nt)] for i in range(n)
    ]
    capacity = [
        [float(data["capacity"][m][t]) for t in range(nt)] for m in range(nm)
    ]
    holding = [float(x) for x in data["holding_cost"]]
    max_machines = [
        int(x) for x in data.get("max_machines_per_item_per_period", [nm] * n)
    ]
    initial = read_initial_inventory(data, n)

    def keyed_param(name):
        raw = data[name]
        out = {}
        for i in range(n):
            for m in caps[i]:
                out[(i, m)] = float(raw[f"item_{i + 1}_machine_{m + 1}"])
        return out

    prod_cost = keyed_param("production_cost")
    prod_time = keyed_param("production_time")
    setup_cost = keyed_param("setup_cost")
    setup_time = keyed_param("setup_time")

    x_index = [(i, m, t) for i in range(n) for m in caps[i] for t in range(nt)]
    inv_index = [(i, t) for i in range(n) for t in range(nt)]

    remaining_demand = [[0.0] * nt for _ in range(n)]
    for i in range(n):
        run = 0.0
        for t in range(nt - 1, -1, -1):
            run += demand[i][t]
            remaining_demand[i][t] = run

    total_demand = [sum(demand[i]) for i in range(n)]
    initial_excess = [max(0.0, initial[i] - total_demand[i]) for i in range(n)]
    future_after = [[0.0] * nt for _ in range(n)]
    for i in range(n):
        run = 0.0
        for t in range(nt - 1, -1, -1):
            future_after[i][t] = run
            run += demand[i][t]

    big_m = {}
    eps = 1e-12
    for i, m, t in x_index:
        bm = remaining_demand[i][t]
        pt = prod_time[(i, m)]
        if pt > eps:
            cu = (capacity[m][t] - setup_time[(i, m)]) / pt
            if cu < bm:
                bm = cu
        if bm < 0.0:
            bm = 0.0
        big_m[(i, m, t)] = bm

    inv_ub = {
        (i, t): future_after[i][t] + initial_excess[i] for (i, t) in inv_index
    }

    zero_tol = 1e-9

    def make_solution(x_vals, y_vals, inv_vals):
        production = {}
        setups = {}
        inventory = {}
        obj = 0.0
        for k, (i, m, t) in enumerate(x_index):
            xv = float(x_vals[k])
            if abs(xv) < zero_tol:
                xv = 0.0
            yb = 1 if float(y_vals[k]) > 0.5 else 0
            production[f"X_item{i + 1}_machine{m + 1}_period{t + 1}"] = xv
            setups[f"Y_item{i + 1}_machine{m + 1}_period{t + 1}"] = yb
            obj += prod_cost[(i, m)] * xv + setup_cost[(i, m)] * yb
        for k, (i, t) in enumerate(inv_index):
            iv = float(inv_vals[k])
            if abs(iv) < zero_tol:
                iv = 0.0
            inventory[f"I_item{i + 1}_period{t + 1}"] = iv
            obj += holding[i] * iv
        sol = {
            "objective_value": float(obj),
            "production": production,
            "inventory": inventory,
            "setups": setups,
        }
        return float(obj), sol

    def construct_greedy(deadline):
        net = [row[:] for row in demand]
        for i in range(n):
            inv0 = initial[i]
            for t in range(nt):
                if inv0 <= 0.0:
                    break
                use = inv0 if inv0 < net[i][t] else net[i][t]
                net[i][t] -= use
                inv0 -= use

        remcap = [row[:] for row in capacity]
        x = {}
        y = set()
        used = [[set() for _ in range(nt)] for __ in range(n)]

        min_pt = []
        total_net = []
        for i in range(n):
            min_pt.append(min((prod_time[(i, m)] for m in caps[i]), default=0.0))
            total_net.append(sum(net[i]))
        item_order = sorted(
            range(n),
            key=lambda i: (len(caps[i]), -total_net[i] * min_pt[i], -total_net[i]),
        )

        tol = 1e-9
        for t in range(nt - 1, -1, -1):
            if time.monotonic() > deadline:
                return None
            for i in item_order:
                qty = net[i][t]
                if qty <= tol:
                    continue
                remaining = qty
                for tau in range(t, -1, -1):
                    if remaining <= tol:
                        break
                    if time.monotonic() > deadline:
                        return None
                    used_set = used[i][tau]
                    while remaining > tol:
                        best = None
                        for m in list(used_set):
                            pt = prod_time[(i, m)]
                            if pt > eps:
                                qmax = remcap[m][tau] / pt
                                if qmax <= tol:
                                    continue
                            else:
                                qmax = remaining
                            qpos = remaining if remaining < qmax else qmax
                            cand = (prod_cost[(i, m)], pt, m, qpos, False)
                            if best is None or cand < best:
                                best = cand
                        if len(used_set) < max_machines[i]:
                            for m in caps[i]:
                                if m in used_set:
                                    continue
                                st = setup_time[(i, m)]
                                if st > remcap[m][tau] + tol:
                                    continue
                                avail = remcap[m][tau] - st
                                pt = prod_time[(i, m)]
                                if pt > eps:
                                    qmax = avail / pt
                                    if qmax <= tol:
                                        continue
                                else:
                                    qmax = remaining
                                qpos = remaining if remaining < qmax else qmax
                                if qpos <= tol:
                                    continue
                                score = prod_cost[(i, m)] + setup_cost[(i, m)] / qpos
                                cand = (score, pt, m, qpos, True)
                                if best is None or cand < best:
                                    best = cand
                        if best is None:
                            break
                        _, _, m, qpos, is_new = best
                        if qpos <= tol:
                            break
                        key = (i, m, tau)
                        if is_new:
                            if len(used_set) >= max_machines[i]:
                                break
                            st = setup_time[(i, m)]
                            if st > remcap[m][tau] + tol:
                                break
                            remcap[m][tau] -= st
                            used_set.add(m)
                            y.add(key)
                        pt = prod_time[(i, m)]
                        if pt > eps:
                            qmax = remcap[m][tau] / pt
                            q = remaining if remaining < qmax else qmax
                            if q <= tol:
                                break
                            remcap[m][tau] -= pt * q
                        else:
                            q = remaining
                        if q <= tol:
                            break
                        x[key] = x.get(key, 0.0) + q
                        remaining -= q
                if remaining > 1e-7:
                    return None

        prod = [[0.0] * nt for _ in range(n)]
        for (i, m, t), q in x.items():
            if q < -1e-8:
                return None
            prod[i][t] += q

        inv = [[0.0] * nt for _ in range(n)]
        for i in range(n):
            val = initial[i]
            for t in range(nt):
                val += prod[i][t] - demand[i][t]
                if val < -1e-7:
                    return None
                if val < 0.0:
                    val = 0.0
                if val > inv_ub[(i, t)] + 1e-7:
                    return None
                inv[i][t] = val

        for m in range(nm):
            for t in range(nt):
                load = 0.0
                for i in items_on_machine[m]:
                    key = (i, m, t)
                    q = x.get(key, 0.0)
                    has = key in y or q > tol
                    if q > tol and key not in y:
                        return None
                    if has:
                        load += setup_time[(i, m)] + prod_time[(i, m)] * q
                if load > capacity[m][t] + 1e-7:
                    return None

        for i in range(n):
            for t in range(nt):
                cnt = 0
                for m in caps[i]:
                    if (i, m, t) in y or x.get((i, m, t), 0.0) > tol:
                        cnt += 1
                if cnt > max_machines[i]:
                    return None

        x_vals = [x.get(idx, 0.0) for idx in x_index]
        y_vals = [
            1.0 if (idx in y or x.get(idx, 0.0) > tol) else 0.0 for idx in x_index
        ]
        inv_vals = [inv[i][t] for (i, t) in inv_index]
        obj, sol = make_solution(x_vals, y_vals, inv_vals)
        return obj, sol, x_vals, y_vals, inv_vals

    def fallback_solution():
        x_vals = [0.0] * len(x_index)
        y_vals = [0.0] * len(x_index)
        inv_vals = []
        for i, t in inv_index:
            val = initial[i] - sum(demand[i][: t + 1])
            inv_vals.append(val if val > 0.0 else 0.0)
        return make_solution(x_vals, y_vals, inv_vals)

    greedy_res = None
    if time_limit > 0:
        greedy_budget = min(1.0, 0.2 * float(time_limit))
        if greedy_budget > 0.0:
            greedy_deadline = min(overall_deadline, time.monotonic() + greedy_budget)
            greedy_res = construct_greedy(greedy_deadline)

    last_logged = [float("inf")]
    best_obj = None
    best_sol = None
    if greedy_res is not None:
        best_obj, best_sol = greedy_res[0], greedy_res[1]
        if logger:
            logger.log_solution(best_obj, best_sol)
        last_logged[0] = best_obj

    if time.monotonic() >= overall_deadline:
        if best_sol is None:
            best_obj, best_sol = fallback_solution()
        return best_sol

    model = gp.Model("multi_item_multi_machine_lot_sizing")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.LogToConsole = 0

    x_var = model.addVars(x_index, lb=0.0, vtype=GRB.CONTINUOUS, name="x")
    y_var = model.addVars(x_index, vtype=GRB.BINARY, name="y")
    inv_var = model.addVars(inv_index, lb=0.0, vtype=GRB.CONTINUOUS, name="inv")
    for (i, t) in inv_index:
        inv_var[i, t].ub = inv_ub[(i, t)]

    model.setObjective(
        gp.quicksum(
            prod_cost[(i, m)] * x_var[i, m, t]
            + setup_cost[(i, m)] * y_var[i, m, t]
            for (i, m, t) in x_index
        )
        + gp.quicksum(holding[i] * inv_var[i, t] for (i, t) in inv_index),
        GRB.MINIMIZE,
    )

    for m in range(nm):
        for t in range(nt):
            model.addConstr(
                gp.quicksum(
                    prod_time[(i, m)] * x_var[i, m, t]
                    + setup_time[(i, m)] * y_var[i, m, t]
                    for i in items_on_machine[m]
                )
                <= capacity[m][t],
                name=f"cap_{m + 1}_{t + 1}",
            )

    for i in range(n):
        for t in range(nt):
            prev = initial[i] if t == 0 else inv_var[i, t - 1]
            model.addConstr(
                gp.quicksum(x_var[i, m, t] for m in caps[i])
                + prev
                - inv_var[i, t]
                == demand[i][t],
                name=f"bal_{i + 1}_{t + 1}",
            )
            if max_machines[i] < len(caps[i]):
                model.addConstr(
                    gp.quicksum(y_var[i, m, t] for m in caps[i]) <= max_machines[i],
                    name=f"maxm_{i + 1}_{t + 1}",
                )

    for idx in x_index:
        model.addConstr(
            x_var[idx] <= big_m[idx] * y_var[idx],
            name=f"link_{idx[0] + 1}_{idx[1] + 1}_{idx[2] + 1}",
        )

    x_vars = [x_var[idx] for idx in x_index]
    y_vars = [y_var[idx] for idx in x_index]
    inv_vars = [inv_var[idx] for idx in inv_index]

    if greedy_res is not None:
        _, _, gx, gy, ginv = greedy_res
        for v, val in zip(x_vars, gx):
            v.Start = float(val)
        for v, val in zip(y_vars, gy):
            v.Start = 1.0 if val > 0.5 else 0.0
        for v, val in zip(inv_vars, ginv):
            v.Start = float(val)

    remaining = overall_deadline - time.monotonic()
    if remaining <= 0.0:
        if best_sol is None:
            best_obj, best_sol = fallback_solution()
        return best_sol
    model.Params.TimeLimit = remaining

    if logger:
        def incumbent_callback(m, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    xv = m.cbGetSolution(x_vars)
                    yv = m.cbGetSolution(y_vars)
                    iv = m.cbGetSolution(inv_vars)
                    obj, sol = make_solution(xv, yv, iv)
                    if obj < last_logged[0] - 1e-9:
                        logger.log_solution(obj, sol)
                        last_logged[0] = obj
                except Exception:
                    pass

        model.optimize(incumbent_callback)
    else:
        model.optimize()

    if model.SolCount > 0:
        xv = model.getAttr(GRB.Attr.X, x_vars)
        yv = model.getAttr(GRB.Attr.X, y_vars)
        iv = model.getAttr(GRB.Attr.X, inv_vars)
        obj, sol = make_solution(xv, yv, iv)
        if best_sol is None or obj < best_obj - 1e-9:
            best_obj, best_sol = obj, sol
            if logger and obj < last_logged[0] - 1e-9:
                try:
                    logger.log_solution(obj, sol)
                    last_logged[0] = obj
                except Exception:
                    pass

    if best_sol is None:
        best_obj, best_sol = fallback_solution()
    return best_sol


def main():
    args = parse_args()
    start_time = time.monotonic()
    with open(args.instance_path, "r") as f:
        data = json.load(f)

    logger = (
        SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    )
    solution = solve_instance(data, args.time_limit, start_time, logger)

    out_dir = os.path.dirname(args.solution_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()