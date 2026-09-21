import argparse
import json
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()
    t_start = time.time()

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    with open(args.instance_path) as fh:
        inst = json.load(fh)

    T = int(inst["num_time_periods"])
    Delta = int(inst["maximum_displacement_delta"])
    flights = inst["flights"]
    scenarios = inst["scenarios"]
    envelopes = inst["capacity_envelopes"]
    air_conn = inst.get("aircraft_connections", []) or []
    pax_conn = inst.get("passenger_connections", []) or []

    rho = 0.5
    for key in ("rho", "rho_weight", "weight_rho"):
        if key in inst:
            rho = float(inst[key])
            break

    nF = len(flights)
    nS = len(scenarios)
    id2idx = {fl["id"]: i for i, fl in enumerate(flights)}

    # detect period indexing base (0-indexed or 1-indexed)
    allp = []
    for fl in flights:
        allp.append(int(fl["dep_period"]))
        allp.append(int(fl["arr_period"]))
    base = 0 if (allp and min(allp) <= 0) else 1
    Pmin = base
    Pmax = base + T - 1
    periods = list(range(Pmin, Pmax + 1))

    # ---------- flight windows ----------
    sd_lo = [0] * nF
    sd_hi = [0] * nF
    od_lo = [0] * nF
    od_hi = [0] * nF
    oa_lo = [0] * nF
    oa_hi = [0] * nF
    for i, fl in enumerate(flights):
        d = int(fl["dep_period"])
        lo = max(Pmin, d - Delta)
        hi = min(Pmax, d + Delta)
        if hi < lo:
            hi = lo
        sd_lo[i], sd_hi[i] = lo, hi
        olo = lo
        ohi = min(Pmax, hi + int(fl["max_dep_delay"]))
        if ohi < olo:
            ohi = olo
        od_lo[i], od_hi[i] = olo, ohi
        alo = olo + int(fl["delta_min"])
        alo = min(alo, Pmax)
        ahi = min(Pmax,
                  ohi + int(fl["delta_max"]),
                  hi + int(fl["delta_sch"]) + int(fl["max_arr_delay"]))
        if ahi < alo:
            ahi = alo
        oa_lo[i], oa_hi[i] = alo, ahi

    # ---------- capacity pre-processing ----------
    dep_at = defaultdict(list)  # (airport, t) -> list of flight indices whose operated dep window contains t
    arr_at = defaultdict(list)
    for i, fl in enumerate(flights):
        ap = fl["dep_airport"]
        for t in range(od_lo[i], od_hi[i] + 1):
            dep_at[(ap, t)].append(i)
        ap = fl["arr_airport"]
        for t in range(oa_lo[i], oa_hi[i] + 1):
            arr_at[(ap, t)].append(i)

    # active segments per (airport,t,cond) given the max possible counts
    active_keys = set(list(dep_at.keys()) + list(arr_at.keys()))
    active_segs = {}
    for (ap, t) in active_keys:
        nD = len(dep_at.get((ap, t), []))
        nA = len(arr_at.get((ap, t), []))
        env_ap = envelopes.get(ap, {})
        for cond in ("VMC", "IMC"):
            segs = []
            for seg in env_ap.get(cond, []):
                a = float(seg["a"])
                b = float(seg["b"])
                Q = float(seg["Q"])
                if a * nD + b * nA > Q + 1e-9:
                    segs.append((a, b, Q))
            if segs:
                active_segs[(ap, t, cond)] = segs

    # ---------- build model ----------
    m = gp.Model("saap")
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1

    all_vars = []
    z_slice = [None] * nF          # (start, length)
    xd_slice = {}                  # (f,s) -> (start, length)
    xa_slice = {}

    # first-stage scheduled-departure indicators
    zvars = []
    for i in range(nF):
        n = sd_hi[i] - sd_lo[i] + 1
        vs = [m.addVar(vtype=GRB.BINARY, name=f"z_{i}_{k}") for k in range(n)]
        vs[0].LB = 1.0
        for k in range(n - 1):
            m.addConstr(vs[k] >= vs[k + 1])
        z_slice[i] = (len(all_vars), n)
        all_vars.extend(vs)
        zvars.append(vs)

    sched_dep_expr = [sd_lo[i] - 1 + gp.quicksum(zvars[i]) for i in range(nF)]

    # first-stage connections (aircraft + passenger)
    for c in list(air_conn) + list(pax_conn):
        fi = id2idx.get(c["flight_i"])
        fj = id2idx.get(c["flight_j"])
        if fi is None or fj is None:
            continue
        mct = int(c["min_connection_time"])
        dsch_i = int(flights[fi]["delta_sch"])
        m.addConstr(sched_dep_expr[fj] - (sched_dep_expr[fi] + dsch_i) >= mct)

    # objective: displacement part
    obj = gp.LinExpr()
    for i, fl in enumerate(flights):
        d = int(fl["dep_period"])
        for k in range(sd_hi[i] - sd_lo[i] + 1):
            t = sd_lo[i] + k
            if t <= d:
                obj += rho * (1.0 - zvars[i][k])
            else:
                obj += rho * zvars[i][k]

    # second stage
    xdvars = {}
    xavars = {}
    op_dep_expr = {}
    op_arr_expr = {}
    for si, sc in enumerate(scenarios):
        prob = float(sc["probability"])
        for i, fl in enumerate(flights):
            nd = od_hi[i] - od_lo[i] + 1
            xd = [m.addVar(vtype=GRB.BINARY, name=f"x_{si}_{i}_{k}") for k in range(nd)]
            xd[0].LB = 1.0
            for k in range(nd - 1):
                m.addConstr(xd[k] >= xd[k + 1])
            # operated dep >= scheduled dep (pointwise dominance)
            nz = sd_hi[i] - sd_lo[i] + 1
            for k in range(min(nz, nd)):
                m.addConstr(xd[k] >= zvars[i][k])
            xd_slice[(i, si)] = (len(all_vars), nd)
            all_vars.extend(xd)
            xdvars[(i, si)] = xd

            na = oa_hi[i] - oa_lo[i] + 1
            xa = [m.addVar(vtype=GRB.BINARY, name=f"y_{si}_{i}_{k}") for k in range(na)]
            xa[0].LB = 1.0
            for k in range(na - 1):
                m.addConstr(xa[k] >= xa[k + 1])
            xa_slice[(i, si)] = (len(all_vars), na)
            all_vars.extend(xa)
            xavars[(i, si)] = xa

            odep = od_lo[i] - 1 + gp.quicksum(xd)
            oarr = oa_lo[i] - 1 + gp.quicksum(xa)
            op_dep_expr[(i, si)] = odep
            op_arr_expr[(i, si)] = oarr

            # departure delay bound (v_dep = odep - sched_dep; >=0 implied by dominance)
            mdd = int(fl["max_dep_delay"])
            m.addConstr(odep - sched_dep_expr[i] <= mdd)

            # en-route duration limits
            m.addConstr(oarr - odep >= int(fl["delta_min"]))
            m.addConstr(oarr - odep <= int(fl["delta_max"]))

            # arrival delay variable
            mad = int(fl["max_arr_delay"])
            va = m.addVar(lb=0.0, ub=mad, vtype=GRB.CONTINUOUS, name=f"va_{si}_{i}")
            m.addConstr(va >= oarr - (sched_dep_expr[i] + int(fl["delta_sch"])))

            cd = float(fl["cost_dep_delay"])
            ca = float(fl["cost_arr_delay"])
            obj += (1.0 - rho) * prob * (cd * (odep - sched_dep_expr[i]) + ca * va)

        # aircraft connections in this scenario
        for c in air_conn:
            fi = id2idx.get(c["flight_i"])
            fj = id2idx.get(c["flight_j"])
            if fi is None or fj is None:
                continue
            mct = int(c["min_connection_time"])
            m.addConstr(op_dep_expr[(fj, si)] - op_arr_expr[(fi, si)] >= mct)

        # capacity constraints
        oc = sc["operating_conditions"]
        for (ap, t) in active_keys:
            cond_list = oc.get(ap)
            if cond_list is None:
                continue
            cond = cond_list[t - Pmin]
            segs = active_segs.get((ap, t, cond))
            if not segs:
                continue
            # departures at (ap,t)
            dterms = gp.LinExpr()
            for i in dep_at.get((ap, t), []):
                k = t - od_lo[i]
                xd = xdvars[(i, si)]
                dterms += xd[k]
                if t < od_hi[i]:
                    dterms += -xd[k + 1]
            aterms = gp.LinExpr()
            for i in arr_at.get((ap, t), []):
                k = t - oa_lo[i]
                xa = xavars[(i, si)]
                aterms += xa[k]
                if t < oa_hi[i]:
                    aterms += -xa[k + 1]
            for (a, b, Q) in segs:
                m.addConstr(a * dterms + b * aterms <= Q)

    m.setObjective(obj, GRB.MINIMIZE)

    # ---------- solution extraction helpers ----------
    def count_ones(vals, st, ln):
        c = 0
        for v in vals[st:st + ln]:
            if v > 0.5:
                c += 1
        return c

    def extract_times(vals):
        sdep = [0] * nF
        for i in range(nF):
            st, ln = z_slice[i]
            sdep[i] = sd_lo[i] - 1 + count_ones(vals, st, ln)
        odep = {}
        oarr = {}
        for si in range(nS):
            for i in range(nF):
                st, ln = xd_slice[(i, si)]
                odep[(i, si)] = od_lo[i] - 1 + count_ones(vals, st, ln)
                st, ln = xa_slice[(i, si)]
                oarr[(i, si)] = oa_lo[i] - 1 + count_ones(vals, st, ln)
        return sdep, odep, oarr

    def make_solution(sdep, odep, oarr):
        disp = 0.0
        schedule = {}
        for i, fl in enumerate(flights):
            disp += abs(sdep[i] - int(fl["dep_period"]))
            schedule[str(fl["id"])] = {
                "scheduled_dep": int(sdep[i]),
                "scheduled_arr": int(sdep[i] + int(fl["delta_sch"])),
            }
        exp_delay = 0.0
        scen_sols = []
        for si, sc in enumerate(scenarios):
            xdep_d = {}
            xarr_d = {}
            vdep_d = {}
            varr_d = {}
            dcost = 0.0
            for i, fl in enumerate(flights):
                od = odep[(i, si)]
                oa = oarr[(i, si)]
                vd = max(0, od - sdep[i])
                va = max(0, oa - (sdep[i] + int(fl["delta_sch"])))
                dcost += float(fl["cost_dep_delay"]) * vd + float(fl["cost_arr_delay"]) * va
                fid = fl["id"]
                vdep_d[str(fid)] = int(vd)
                varr_d[str(fid)] = int(va)
                for t in periods:
                    xdep_d[f"{fid}_{t}"] = 1 if t <= od else 0
                    xarr_d[f"{fid}_{t}"] = 1 if t <= oa else 0
            exp_delay += float(sc["probability"]) * dcost
            scen_sols.append({
                "scenario_id": sc["id"],
                "x_dep": xdep_d,
                "x_arr": xarr_d,
                "v_dep": vdep_d,
                "v_arr": varr_d,
            })
        objective = rho * disp + (1.0 - rho) * exp_delay
        sol = {
            "objective_value": objective,
            "schedule": schedule,
            "scenario_solutions": scen_sols,
        }
        return objective, sol

    # ---------- callback for incumbent logging ----------
    m._best = float("inf")
    m._allvars = all_vars

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                cur = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if cur >= model._best - 1e-9:
                    return
                model._best = cur
                if logger is None:
                    return
                vals = model.cbGetSolution(model._allvars)
                sdep, odep, oarr = extract_times(vals)
                objective, sol = make_solution(sdep, odep, oarr)
                logger.log_solution(objective, sol)
            except Exception:
                pass

    # ---------- solve ----------
    reserve = max(2.0, min(15.0, 0.05 * args.time_limit))
    remaining = args.time_limit - (time.time() - t_start) - reserve
    if remaining < 1.0:
        remaining = 1.0
    m.Params.TimeLimit = remaining

    try:
        m.optimize(cb)
    except Exception:
        pass

    # ---------- write final solution ----------
    if m.SolCount > 0:
        vals = m.getAttr("X", all_vars)
        sdep, odep, oarr = extract_times(vals)
        objective, sol = make_solution(sdep, odep, oarr)
    else:
        # fallback: requested schedule, no delays (may violate capacity, best effort)
        sdep = [int(fl["dep_period"]) for fl in flights]
        odep = {}
        oarr = {}
        for si in range(nS):
            for i, fl in enumerate(flights):
                odep[(i, si)] = sdep[i]
                oarr[(i, si)] = sdep[i] + int(fl["delta_sch"])
        objective, sol = make_solution(sdep, odep, oarr)

    if logger is not None:
        try:
            logger.log_solution(objective, sol)
        except Exception:
            pass

    with open(args.solution_path, "w") as fh:
        json.dump(sol, fh)


if __name__ == "__main__":
    main()