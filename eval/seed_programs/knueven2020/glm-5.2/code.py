import argparse
import json
import math
import time

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def prepare_generator(gen):
    g = dict(gen)
    segs = sorted(
        gen["production_cost_segments"],
        key=lambda s: float(s["output_mw_start"]),
    )
    g["segs"] = segs

    finite_startup = [
        t for t in gen["startup_costs"] if t["max_downtime_hours"] is not None
    ]
    finite_startup.sort(key=lambda t: t["max_downtime_hours"])
    null_startup = [
        t for t in gen["startup_costs"] if t["max_downtime_hours"] is None
    ]

    if null_startup:
        final_startup_cost = float(null_startup[0]["cost"])
    else:
        if finite_startup:
            final_startup_cost = float(finite_startup[-1]["cost"])
            finite_startup = finite_startup[:-1]
        else:
            final_startup_cost = 0.0

    g["thresholds"] = [int(t["max_downtime_hours"]) for t in finite_startup]
    g["tier_costs"] = (
        [float(t["cost"]) for t in finite_startup] + [final_startup_cost]
    )
    return g


def production_cost_at_power(g, p):
    if p <= 0.0:
        return 0.0
    c = 0.0
    for seg in g["segs"]:
        start = float(seg["output_mw_start"])
        end = float(seg["output_mw_end"])
        mc = float(seg["marginal_cost_per_mwh"])
        if p > start:
            c += mc * (min(p, end) - start)
    return c


def startup_cost_for_downtime(g, downtime_hours):
    for th, cost in zip(g["thresholds"], g["tier_costs"][:-1]):
        if downtime_hours <= th + 1e-6:
            return cost
    return g["tier_costs"][-1]


def compute_objective(instance, schedule, gen_info, T, H):
    total = 0.0
    for gen in instance["generators"]:
        gid = gen["id"]
        g = gen_info[gid]
        s = schedule[gid]
        u = s["u"]
        p = s["p"]
        init_status = int(g["initial_status"])
        d0 = -init_status if init_status < 0 else 0
        u0 = 1 if init_status > 0 else 0

        for t in range(T):
            if u[t] == 1:
                total += float(g["no_load_cost"])
                total += H * production_cost_at_power(g, float(p[t]))

        for t in range(T):
            start = False
            downtime_periods = 0
            if t == 0:
                if u[0] == 1 and u0 == 0:
                    downtime_periods = d0 if init_status < 0 else 0
                    start = True
            else:
                if u[t] == 1 and u[t - 1] == 0:
                    count = 0
                    j = t - 1
                    while j >= 0 and u[j] == 0:
                        count += 1
                        j -= 1
                    if j < 0 and init_status < 0:
                        count += -init_status
                    downtime_periods = count
                    start = True
            if start:
                total += startup_cost_for_downtime(g, downtime_periods * H)
    return total


def add_production_cost(model, g, p, u, T, H):
    segs = g["segs"]
    if not segs:
        return gp.LinExpr(0)

    contiguous = True
    for i in range(len(segs) - 1):
        if abs(
            float(segs[i]["output_mw_end"])
            - float(segs[i + 1]["output_mw_start"])
        ) > 1e-6:
            contiguous = False
            break

    nondecreasing = True
    for i in range(len(segs) - 1):
        if (
            float(segs[i + 1]["marginal_cost_per_mwh"])
            < float(segs[i]["marginal_cost_per_mwh"]) - 1e-9
        ):
            nondecreasing = False
            break

    first_start = float(segs[0]["output_mw_start"])
    pmin = float(g["p_min"])

    if (
        contiguous
        and nondecreasing
        and first_start >= -1e-6
        and first_start <= pmin + 1e-6
    ):
        base = max(0.0, first_start)
        widths = [
            max(0.0, float(seg["output_mw_end"]) - float(seg["output_mw_start"]))
            for seg in segs
        ]
        K = len(segs)
        deltas = model.addVars(T, K, lb=0.0)

        for k in range(K):
            if widths[k] > 0:
                model.addConstrs(
                    (deltas[t, k] <= widths[k] for t in range(T))
                )

        model.addConstrs(
            (
                p[t] == base * u[t] + gp.quicksum(deltas[t, k] for k in range(K))
                for t in range(T)
            )
        )

        return H * gp.quicksum(
            float(segs[k]["marginal_cost_per_mwh"]) * deltas[t, k]
            for t in range(T)
            for k in range(K)
        )

    xs = [0.0]
    pmax = float(g["p_max"])
    for seg in segs:
        s = max(0.0, float(seg["output_mw_start"]))
        e = max(0.0, float(seg["output_mw_end"]))
        if s <= pmax + 1e-6:
            xs.append(s)
        if e <= pmax + 1e-6:
            xs.append(e)
    if pmin > 0:
        xs.append(pmin)
    xs.append(pmax)
    xs.sort()

    unique = []
    for x in xs:
        if x < 0:
            x = 0.0
        if not unique or x > unique[-1] + 1e-9:
            unique.append(x)

    ys = [H * production_cost_at_power(g, x) for x in unique]
    for t in range(T):
        model.setPWLObj(p[t], unique, ys)
    return gp.LinExpr(0)


def add_startup_cost(model, g, u, v, T, H):
    thresholds = g["thresholds"]
    K = len(thresholds)
    costs = g["tier_costs"]

    if K == 0:
        return costs[0] * gp.quicksum(v[t] for t in range(T))

    n_list = [int(th // H) + 1 for th in thresholds]
    dt = model.addVars(T, K, lb=0.0, ub=1.0)
    z = model.addVars(T, K + 1, lb=0.0, ub=1.0)
    init_status = int(g["initial_status"])

    for t in range(T):
        for i in range(K):
            ni = n_list[i]
            if ni <= 0:
                dt[t, i].lb = 1.0
                continue

            fixed_on = False
            if t < ni:
                if init_status > 0:
                    fixed_on = True
                else:
                    d0 = -init_status if init_status < 0 else 0
                    if d0 < ni - t:
                        fixed_on = True

            if fixed_on:
                dt[t, i].ub = 0.0
            else:
                var_js = range(max(0, t - ni), t)
                if len(var_js) == 0:
                    dt[t, i].lb = 1.0
                else:
                    model.addConstrs(
                        (dt[t, i] <= 1 - u[j] for j in var_js)
                    )
                    model.addConstr(
                        dt[t, i] >= 1 - gp.quicksum(u[j] for j in var_js)
                    )

        model.addConstr(
            gp.quicksum(z[t, i] for i in range(K + 1)) == v[t]
        )

        model.addConstr(z[t, 0] <= v[t])
        model.addConstr(z[t, 0] <= 1 - dt[t, 0])
        model.addConstr(z[t, 0] >= v[t] - dt[t, 0])

        for i in range(1, K):
            model.addConstr(z[t, i] <= v[t])
            model.addConstr(z[t, i] <= dt[t, i - 1])
            model.addConstr(z[t, i] <= 1 - dt[t, i])
            model.addConstr(
                z[t, i] >= dt[t, i - 1] - dt[t, i] - (1 - v[t])
            )

        model.addConstr(z[t, K] <= v[t])
        model.addConstr(z[t, K] <= dt[t, K - 1])
        model.addConstr(z[t, K] >= v[t] + dt[t, K - 1] - 1)

    return gp.quicksum(
        costs[i] * z[t, i] for t in range(T) for i in range(K + 1)
    )


def clean_power(pval, on, pmin, pmax):
    if on == 0:
        return 0.0
    p = float(pval)
    if p < 0.0:
        p = 0.0
    if p > pmax:
        p = pmax
    if p < pmin:
        p = pmin
    return p


def main():
    args = parse_args()
    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    start_time = time.time()
    T = len(instance["demand"])
    H = int(instance.get("time_period_hours", 1))
    if H <= 0:
        H = 1

    gen_info = {}
    gen_ids = []
    for gen in instance["generators"]:
        g = prepare_generator(gen)
        gen_info[gen["id"]] = g
        gen_ids.append(gen["id"])

    model = gp.Model("uc")
    model.Params.OutputFlag = 0

    u_vars = {}
    v_vars = {}
    w_vars = {}
    p_vars = {}

    for gid in gen_ids:
        g = gen_info[gid]
        u_vars[gid] = model.addVars(T, vtype=GRB.BINARY)
        v_vars[gid] = model.addVars(T, vtype=GRB.BINARY)
        w_vars[gid] = model.addVars(T, vtype=GRB.BINARY)
        p_vars[gid] = model.addVars(T, lb=0.0, ub=float(g["p_max"]))

    obj_expr = gp.LinExpr(0)

    for gid in gen_ids:
        g = gen_info[gid]
        u = u_vars[gid]
        v = v_vars[gid]
        w = w_vars[gid]
        p = p_vars[gid]

        Pmin = float(g["p_min"])
        Pmax = float(g["p_max"])
        RU = float(g["ramp_up_limit"])
        RD = float(g["ramp_down_limit"])
        SU = float(g["startup_ramp_limit"])
        SD = float(g["shutdown_ramp_limit"])
        UT = int(g["min_up_time"])
        DT = int(g["min_down_time"])
        init_status = int(g["initial_status"])
        init_power = float(g["initial_power"])
        u0 = 1 if init_status > 0 else 0

        if init_status > 0 and UT > init_status:
            R = UT - init_status
            for j in range(min(R, T)):
                u[j].lb = 1.0

        if init_status <= 0 and DT > 0:
            d0 = -init_status if init_status < 0 else 0
            if DT > d0:
                R = DT - d0
                for j in range(min(R, T)):
                    u[j].ub = 0.0

        model.addConstr(u[0] - u0 == v[0] - w[0])
        model.addConstr(v[0] <= 1 - u0)
        model.addConstr(w[0] <= u0)
        model.addConstr(v[0] <= u[0])
        model.addConstr(w[0] <= 1 - u[0])
        model.addConstr(v[0] + w[0] <= 1)

        if T > 1:
            model.addConstrs(
                (u[t] - u[t - 1] == v[t] - w[t] for t in range(1, T))
            )
            model.addConstrs(
                (v[t] <= 1 - u[t - 1] for t in range(1, T))
            )
            model.addConstrs(
                (w[t] <= u[t - 1] for t in range(1, T))
            )
            model.addConstrs((v[t] <= u[t] for t in range(1, T)))
            model.addConstrs(
                (w[t] <= 1 - u[t] for t in range(1, T))
            )
            model.addConstrs((v[t] + w[t] <= 1 for t in range(1, T)))

        if UT > 0:
            model.addConstrs(
                (
                    u[j] >= v[t]
                    for t in range(T)
                    for j in range(t, min(t + UT, T))
                )
            )
        if DT > 0:
            model.addConstrs(
                (
                    u[j] <= 1 - w[t]
                    for t in range(T)
                    for j in range(t, min(t + DT, T))
                )
            )

        model.addConstrs((p[t] <= Pmax * u[t] for t in range(T)))
        model.addConstrs((p[t] >= Pmin * u[t] for t in range(T)))

        model.addConstr(
            p[0] <= init_power + RU * u0 + SU * (1 - u0) + Pmax * (1 - u[0])
        )
        model.addConstr(p[0] >= init_power - RD - Pmax * (1 - u[0]))
        model.addConstr(init_power <= SD + Pmax * u[0])

        if T > 1:
            model.addConstrs(
                (
                    p[t] <= p[t - 1] + RU * u[t - 1] + SU * (1 - u[t - 1])
                    + Pmax * (1 - u[t])
                    for t in range(1, T)
                )
            )
            model.addConstrs(
                (
                    p[t] >= p[t - 1] - RD - Pmax * (1 - u[t])
                    for t in range(1, T)
                )
            )
            model.addConstrs(
                (p[t - 1] <= SD + Pmax * u[t] for t in range(1, T))
            )

        obj_expr += add_production_cost(model, g, p, u, T, H)
        obj_expr += float(g["no_load_cost"]) * gp.quicksum(
            u[t] for t in range(T)
        )
        obj_expr += add_startup_cost(model, g, u, v, T, H)

    demand = instance["demand"]
    reserve = instance["reserve_requirement"]
    for t in range(T):
        model.addConstr(
            gp.quicksum(p_vars[gid][t] for gid in gen_ids) == float(demand[t])
        )
        model.addConstr(
            gp.quicksum(
                float(gen_info[gid]["p_max"]) * u_vars[gid][t]
                for gid in gen_ids
            )
            >= float(demand[t]) + float(reserve[t])
        )

    model.setObjective(obj_expr, GRB.MINIMIZE)

    best = {"obj": float("inf"), "sol": None, "logged": float("inf")}

    def build_solution_callback(model):
        sol = {}
        for gid in gen_ids:
            g = gen_info[gid]
            uvals = model.cbGetSolution([u_vars[gid][t] for t in range(T)])
            pvals = model.cbGetSolution([p_vars[gid][t] for t in range(T)])
            u_list = [1 if v > 0.5 else 0 for v in uvals]
            p_list = [
                clean_power(pvals[t], u_list[t], float(g["p_min"]), float(g["p_max"]))
                for t in range(T)
            ]
            pb = [
                float(g["p_max"]) if u_list[t] == 1 else 0.0
                for t in range(T)
            ]
            sol[gid] = {"u": u_list, "p": p_list, "pb": pb}
        return sol

    def build_solution_final(model):
        sol = {}
        for gid in gen_ids:
            g = gen_info[gid]
            uvals = [u_vars[gid][t].X for t in range(T)]
            pvals = [p_vars[gid][t].X for t in range(T)]
            u_list = [1 if v > 0.5 else 0 for v in uvals]
            p_list = [
                clean_power(pvals[t], u_list[t], float(g["p_min"]), float(g["p_max"]))
                for t in range(T)
            ]
            pb = [
                float(g["p_max"]) if u_list[t] == 1 else 0.0
                for t in range(T)
            ]
            sol[gid] = {"u": u_list, "p": p_list, "pb": pb}
        return sol

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best["obj"] - 1e-6:
                best["obj"] = obj
                sol = build_solution_callback(model)
                best["sol"] = sol
                if logger is not None and obj < best["logged"] - 1e-6:
                    logger.log_solution(obj, sol)
                    best["logged"] = obj

    elapsed = time.time() - start_time
    tl = args.time_limit - elapsed - 1.0
    if tl < 1.0:
        tl = 1.0

    model.Params.TimeLimit = tl
    model.Params.Threads = 1
    model.Params.MIPGap = 1e-4
    model.Params.Seed = 0
    model.Params.NumericFocus = 0

    model.optimize(callback)

    candidates = []
    if model.SolCount > 0:
        candidates.append(build_solution_final(model))
    if best["sol"] is not None:
        candidates.append(best["sol"])

    sol = None
    if candidates:
        sol = min(
            candidates,
            key=lambda s: compute_objective(instance, s, gen_info, T, H),
        )

    if sol is None:
        sol = {}
        for gid in gen_ids:
            sol[gid] = {
                "u": [0] * T,
                "p": [0.0] * T,
                "pb": [0.0] * T,
            }
        obj = 0.0
    else:
        obj = compute_objective(instance, sol, gen_info, T, H)
        if logger is not None and obj < best["logged"] - 1e-6:
            logger.log_solution(obj, sol)
            best["logged"] = obj

    output = {
        "objective_value": float(obj),
        "schedule": sol,
    }
    with open(args.solution_path, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()