import argparse
import json
import time
import math

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def load_instance(path):
    with open(path, "r") as f:
        return json.load(f)


def sorted_tiers(gen):
    tiers = list(gen["startup_costs"])
    tiers.sort(key=lambda t: (t["max_downtime_hours"] is None,
                              t["max_downtime_hours"] if t["max_downtime_hours"] is not None else 0))
    return tiers


def startup_cost_for_downtime(gen, downtime_periods, tph):
    hours = downtime_periods * tph
    tiers = sorted_tiers(gen)
    for tier in tiers:
        if tier["max_downtime_hours"] is None or hours <= tier["max_downtime_hours"]:
            return tier["cost"]
    return tiers[-1]["cost"]


def production_cost(gen, pval):
    c = 0.0
    for seg in gen["production_cost_segments"]:
        lo = seg["output_mw_start"]
        hi = seg["output_mw_end"]
        c += seg["marginal_cost_per_mwh"] * max(0.0, min(pval, hi) - lo)
    return c


def compute_true_objective(data, schedule):
    T = data["num_time_periods"]
    tph = data.get("time_period_hours", 1)
    total = 0.0
    for gen in data["generators"]:
        gid = gen["id"]
        u = schedule[gid]["u"]
        p = schedule[gid]["p"]
        init = gen["initial_status"]
        prev_on = init > 0
        off_count = -init if init < 0 else 0
        for t in range(T):
            if u[t] == 1:
                if not prev_on:
                    total += startup_cost_for_downtime(gen, off_count, tph)
                total += gen["no_load_cost"]
                total += production_cost(gen, p[t])
                prev_on = True
                off_count = 0
            else:
                off_count = off_count + 1 if not prev_on else 1
                prev_on = False
    return total


def build_model(data):
    G = data["num_generators"]
    T = data["num_time_periods"]
    tph = data.get("time_period_hours", 1)
    gens = data["generators"]
    demand = data["demand"]
    reserve = data["reserve_requirement"]

    m = gp.Model("uc")
    m.Params.OutputFlag = 0

    u = {}
    v = {}
    w = {}
    p = {}
    pb = {}
    x = {}
    delta = {}

    obj = gp.LinExpr()

    for gi, gen in enumerate(gens):
        pmin = gen["p_min"]
        pmax = gen["p_max"]
        RU = gen["ramp_up_limit"]
        RD = gen["ramp_down_limit"]
        SU = gen["startup_ramp_limit"]
        SD = gen["shutdown_ramp_limit"]
        UT = gen["min_up_time"]
        DT = gen["min_down_time"]
        init = gen["initial_status"]
        p0 = gen["initial_power"] if init > 0 else 0.0
        u_init = 1 if init > 0 else 0
        segs = gen["production_cost_segments"]
        start1 = segs[0]["output_mw_start"] if segs else 0.0

        for t in range(T):
            u[gi, t] = m.addVar(vtype=GRB.BINARY, name=f"u_{gi}_{t}")
            v[gi, t] = m.addVar(lb=0.0, ub=1.0, name=f"v_{gi}_{t}")
            w[gi, t] = m.addVar(lb=0.0, ub=1.0, name=f"w_{gi}_{t}")
            p[gi, t] = m.addVar(lb=0.0, ub=pmax, name=f"p_{gi}_{t}")
            pb[gi, t] = m.addVar(lb=0.0, ub=pmax, name=f"pb_{gi}_{t}")
            obj += gen["no_load_cost"] * u[gi, t]
            for k, seg in enumerate(segs):
                ln = max(0.0, seg["output_mw_end"] - seg["output_mw_start"])
                x[gi, t, k] = m.addVar(lb=0.0, ub=ln, name=f"x_{gi}_{t}_{k}")
                obj += seg["marginal_cost_per_mwh"] * x[gi, t, k]

        # logic
        m.addConstr(u[gi, 0] - u_init == v[gi, 0] - w[gi, 0])
        for t in range(1, T):
            m.addConstr(u[gi, t] - u[gi, t - 1] == v[gi, t] - w[gi, t])

        # initial min up/down enforcement
        if init > 0:
            rem = max(0, UT - init)
            for t in range(min(rem, T)):
                m.addConstr(u[gi, t] == 1)
        else:
            k_off = -init
            rem = max(0, DT - k_off)
            for t in range(min(rem, T)):
                m.addConstr(u[gi, t] == 0)

        # min up / min down
        for t in range(T):
            lo = max(0, t - UT + 1)
            m.addConstr(gp.quicksum(v[gi, i] for i in range(lo, t + 1)) <= u[gi, t])
            lo = max(0, t - DT + 1)
            m.addConstr(gp.quicksum(w[gi, i] for i in range(lo, t + 1)) <= 1 - u[gi, t])

        # power bounds and segment decomposition
        for t in range(T):
            m.addConstr(p[gi, t] >= pmin * u[gi, t])
            m.addConstr(p[gi, t] <= pb[gi, t])
            m.addConstr(pb[gi, t] <= pmax * u[gi, t])
            m.addConstr(p[gi, t] == start1 * u[gi, t] +
                        gp.quicksum(x[gi, t, k] for k in range(len(segs))))

        # ramping
        m.addConstr(pb[gi, 0] <= p0 + RU * u_init + SU * v[gi, 0])
        m.addConstr(p0 - p[gi, 0] <= RD * u[gi, 0] + SD * w[gi, 0])
        for t in range(1, T):
            m.addConstr(pb[gi, t] <= p[gi, t - 1] + RU * u[gi, t - 1] + SU * v[gi, t])
            m.addConstr(p[gi, t - 1] - p[gi, t] <= RD * u[gi, t] + SD * w[gi, t])
        # shutdown ramp cap
        if SD < pmax:
            for t in range(T - 1):
                m.addConstr(pb[gi, t] <= pmax * u[gi, t] - (pmax - SD) * w[gi, t + 1])

        # startup cost tiers
        tiers = sorted_tiers(gen)
        S = len(tiers)
        # period windows for each tier
        windows = []
        prev_hi = DT - 1
        for s, tier in enumerate(tiers):
            lo_s = prev_hi + 1
            if tier["max_downtime_hours"] is None:
                hi_s = None
            else:
                hi_s = int(math.floor(tier["max_downtime_hours"] / tph))
                prev_hi = max(prev_hi, hi_s)
            windows.append((lo_s, hi_s))
        k_off_init = -init if init < 0 else 0
        for t in range(T):
            dvars = []
            for s, tier in enumerate(tiers):
                lo_s, hi_s = windows[s]
                if hi_s is not None and hi_s < lo_s:
                    continue  # tier never applicable
                dv = m.addVar(lb=0.0, ub=1.0, name=f"d_{gi}_{t}_{s}")
                delta[gi, t, s] = dv
                dvars.append(dv)
                obj += tier["cost"] * dv
                if hi_s is not None:
                    expr = gp.LinExpr()
                    for i in range(lo_s, hi_s + 1):
                        if 0 <= t - i:
                            expr += w[gi, t - i]
                    if u_init == 0 and lo_s <= t + k_off_init <= hi_s:
                        expr += 1.0
                    m.addConstr(dv <= expr)
            m.addConstr(gp.quicksum(dvars) == v[gi, t])

    # system constraints
    for t in range(T):
        m.addConstr(gp.quicksum(p[gi, t] for gi in range(G)) == demand[t])
        m.addConstr(gp.quicksum(pb[gi, t] for gi in range(G)) >= demand[t] + reserve[t])

    m.setObjective(obj, GRB.MINIMIZE)
    return m, u, p, pb


def extract_schedule(data, getval, u, p, pb):
    T = data["num_time_periods"]
    schedule = {}
    for gi, gen in enumerate(data["generators"]):
        uu = []
        ppv = []
        pbv = []
        for t in range(T):
            uval = 1 if getval(u[gi, t]) > 0.5 else 0
            pv = getval(p[gi, t])
            pbval = getval(pb[gi, t])
            if uval == 0:
                pv = 0.0
                pbval = 0.0
            else:
                pv = max(0.0, pv)
                pbval = max(pv, pbval)
            uu.append(uval)
            ppv.append(float(pv))
            pbv.append(float(pbval))
        schedule[gen["id"]] = {"u": uu, "p": ppv, "pb": pbv}
    return schedule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start = time.time()
    data = load_instance(args.instance_path)

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    model, u, p, pb = build_model(data)

    elapsed = time.time() - start
    remaining = max(5.0, args.time_limit - elapsed - 3.0)

    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = remaining

    best = {"obj": float("inf"), "schedule": None}

    def callback(mdl, where):
        if where == GRB.Callback.MIPSOL:
            try:
                sched = extract_schedule(data, lambda var: mdl.cbGetSolution(var), u, p, pb)
                obj = compute_true_objective(data, sched)
                if obj < best["obj"]:
                    best["obj"] = obj
                    best["schedule"] = sched
                    if logger:
                        logger.log_solution(obj, {"objective_value": obj, "schedule": sched})
            except Exception:
                pass

    model.optimize(callback)

    schedule = None
    objective = None
    if model.SolCount > 0:
        sched = extract_schedule(data, lambda var: var.X, u, p, pb)
        obj = compute_true_objective(data, sched)
        if obj < best["obj"]:
            best["obj"] = obj
            best["schedule"] = sched
            if logger:
                logger.log_solution(obj, {"objective_value": obj, "schedule": sched})
    if best["schedule"] is not None:
        schedule = best["schedule"]
        objective = best["obj"]
    else:
        # fallback: trivial (may be infeasible) schedule - all generators off
        T = data["num_time_periods"]
        schedule = {}
        for gen in data["generators"]:
            schedule[gen["id"]] = {"u": [0] * T, "p": [0.0] * T, "pb": [0.0] * T}
        objective = 0.0

    solution = {"objective_value": float(objective), "schedule": schedule}
    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()