import argparse
import json
import sys
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


class _Abort(Exception):
    """Internal exception used to bail out to a fallback solution."""
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    t_start = time.time()
    time_limit = max(1, int(args.time_limit))
    reserve = max(2.0, min(30.0, 0.05 * float(time_limit)))

    with open(args.instance_path, "r") as fh:
        inst = json.load(fh)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # ---------------- Instance data ----------------
    T = int(inst["num_time_periods"])
    Delta = int(inst["maximum_displacement_delta"])
    airports = list(inst.get("airports", []))
    flights = list(inst.get("flights", []))
    aircraft_conns = list(inst.get("aircraft_connections", []))
    passenger_conns = list(inst.get("passenger_connections", []))
    envelopes = inst.get("capacity_envelopes", {})
    scenarios = list(inst.get("scenarios", []))
    rho = float(inst.get("rho", 0.5))

    nF = len(flights)
    nS = len(scenarios)

    fids = [f["id"] for f in flights]
    fid_str = [str(v) for v in fids]
    idx_of = {fids[i]: i for i in range(nF)}
    req_dep = [int(f["dep_period"]) for f in flights]
    delta = [int(f["delta_sch"]) for f in flights]
    dmin = [int(f["delta_min"]) for f in flights]
    dmax = [int(f["delta_max"]) for f in flights]
    mdd = [int(f["max_dep_delay"]) for f in flights]
    maa = [int(f["max_arr_delay"]) for f in flights]
    cdep = [float(f["cost_dep_delay"]) for f in flights]
    carr = [float(f["cost_arr_delay"]) for f in flights]
    dep_ap = [f["dep_airport"] for f in flights]
    arr_ap = [f["arr_airport"] for f in flights]

    # Detect whether periods are 0-based or 1-based.
    if nF:
        minp = min(min(f["dep_period"], f["arr_period"]) for f in flights)
    else:
        minp = 1
    base = 0 if minp == 0 else 1
    last = base + T - 1
    periods = list(range(base, last + 1))

    # Scheduling windows (scheduled departure must stay within +-Delta).
    lds = [max(base, req_dep[i] - Delta) for i in range(nF)]
    uds = [min(last, req_dep[i] + Delta) for i in range(nF)]
    for i in range(nF):
        if uds[i] < lds[i]:
            uds[i] = lds[i]
    # Operated windows (departures may be delayed; arrivals may shift both ways
    # within en-route-duration and delay limits).
    ldo = list(lds)
    udo = [min(last, uds[i] + mdd[i]) for i in range(nF)]
    lao = [max(base, lds[i] + dmin[i]) for i in range(nF)]
    uao = [min(last, uds[i] + delta[i] + maa[i], udo[i] + dmax[i]) for i in range(nF)]
    for i in range(nF):
        if uao[i] < base:
            uao[i] = base
        if lao[i] > uao[i]:
            lao[i] = uao[i]

    # Assignment-binary index sets (equivalent to the non-increasing indicator
    # sequences: y[i,t]=1 iff the flight operates in period t; the indicator
    # drop at t equals y[i,t], and sum of indicators equals the chosen period).
    dep_idx = [(i, t) for i in range(nF) for t in range(ldo[i], udo[i] + 1)]
    arr_idx = [(i, t) for i in range(nF) for t in range(lao[i], uao[i] + 1)]
    dep_span = []
    off = 0
    for i in range(nF):
        w = udo[i] - ldo[i] + 1
        dep_span.append((off, off + w))
        off += w
    arr_span = []
    off = 0
    for i in range(nF):
        w = uao[i] - lao[i] + 1
        arr_span.append((off, off + w))
        off += w

    # Per-airport/period coverage lists for capacity constraints.
    dep_at = {a: [] for a in airports}
    arr_at = {a: [] for a in airports}
    for i in range(nF):
        if dep_ap[i] in dep_at:
            dep_at[dep_ap[i]].append(i)
        if arr_ap[i] in arr_at:
            arr_at[arr_ap[i]].append(i)
    dep_cover = {}
    arr_cover = {}
    for a in airports:
        dc = {t: [] for t in periods}
        for i in dep_at[a]:
            for t in range(ldo[i], udo[i] + 1):
                dc[t].append(i)
        dep_cover[a] = dc
        acov = {t: [] for t in periods}
        for i in arr_at[a]:
            for t in range(lao[i], uao[i] + 1):
                acov[t].append(i)
        arr_cover[a] = acov

    # Precompute output key strings and indicator value patterns.
    keys = [[f"{fid_str[i]}_{t}" for t in periods] for i in range(nF)]
    vals_cache = [[1] * m + [0] * (T - m) for m in range(T + 1)]

    probs = [float(s.get("probability", 1.0)) for s in scenarios]
    scen_ids = [s.get("id", si) for si, s in enumerate(scenarios)]

    def fallback_solution():
        sched = {}
        for i in range(nF):
            sched[fid_str[i]] = {
                "scheduled_dep": req_dep[i],
                "scheduled_arr": req_dep[i] + delta[i],
            }
        scen_sols = []
        for si in range(nS):
            xdep = {}
            xarr = {}
            vd = {}
            va = {}
            for i in range(nF):
                md = max(0, min(T, req_dep[i] - base + 1))
                ma = max(0, min(T, req_dep[i] + delta[i] - base + 1))
                xdep.update(dict(zip(keys[i], vals_cache[md])))
                xarr.update(dict(zip(keys[i], vals_cache[ma])))
                vd[fid_str[i]] = 0
                va[fid_str[i]] = 0
            scen_sols.append({
                "scenario_id": scen_ids[si],
                "x_dep": xdep,
                "x_arr": xarr,
                "v_dep": vd,
                "v_arr": va,
            })
        return 0.0, {
            "objective_value": 0.0,
            "schedule": sched,
            "scenario_solutions": scen_sols,
        }

    final_obj = None
    final_sol = None
    best = {"val": float("inf")}

    try:
        model = gp.Model("atm_sched")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.LogToConsole = 0

        # ---------------- First stage ----------------
        # td[i] = scheduled departure period (integer). Its value equals the sum
        # of the non-increasing scheduled-departure indicators. Scheduled
        # arrival period is td[i] + delta[i].
        td = model.addVars(range(nF), lb=lds, ub=uds, vtype=GRB.INTEGER, name="td")
        dabs = model.addVars(range(nF), lb=0.0, name="dabs")
        model.addConstrs((dabs[i] >= td[i] - req_dep[i] for i in range(nF)), name="disp_p")
        model.addConstrs((dabs[i] >= req_dep[i] - td[i] for i in range(nF)), name="disp_n")
        stage1_conns = aircraft_conns + passenger_conns
        model.addConstrs(
            (
                td[idx_of[c["flight_j"]]] - td[idx_of[c["flight_i"]]]
                >= int(c["min_connection_time"]) + delta[idx_of[c["flight_i"]]]
                for c in stage1_conns
            ),
            name="sched_conn",
        )
        for i in range(nF):
            td[i].Start = float(req_dep[i])

        # ---------------- Second stage ----------------
        yd_flat = []
        ya_flat = []
        vdep_all = []
        varr_all = []

        for si in range(nS):
            if time.time() - t_start > time_limit - reserve:
                raise _Abort()
            scen = scenarios[si]
            yd = model.addVars(dep_idx, vtype=GRB.BINARY, name=f"yd_{si}")
            ya = model.addVars(arr_idx, vtype=GRB.BINARY, name=f"ya_{si}")
            vdep = model.addVars(range(nF), lb=0.0, ub=[float(v) for v in mdd],
                                 name=f"vdep_{si}")
            varr = model.addVars(range(nF), lb=0.0, ub=[float(v) for v in maa],
                                 name=f"varr_{si}")
            model.addConstrs((yd.sum(i, "*") == 1 for i in range(nF)),
                             name=f"one_dep_{si}")
            model.addConstrs((ya.sum(i, "*") == 1 for i in range(nF)),
                             name=f"one_arr_{si}")

            # Operated departure/arrival periods as linear expressions.
            ed = [None] * nF
            ea = [None] * nF
            for i in range(nF):
                rng = range(ldo[i], udo[i] + 1)
                ed[i] = gp.LinExpr(list(rng), [yd[i, t] for t in rng])
                rng = range(lao[i], uao[i] + 1)
                ea[i] = gp.LinExpr(list(rng), [ya[i, t] for t in rng])

            # Departure delay equals the difference of indicator sums; it is
            # nonnegative and bounded above by max_dep_delay (via ub).
            model.addConstrs((vdep[i] == ed[i] - td[i] for i in range(nF)),
                             name=f"vdep_def_{si}")
            # Arrival delay is at least the difference of indicator sums.
            model.addConstrs((varr[i] >= ea[i] - td[i] - delta[i] for i in range(nF)),
                             name=f"varr_def_{si}")
            # En-route duration limits.
            model.addConstrs((ea[i] - ed[i] >= dmin[i] for i in range(nF)),
                             name=f"enr_min_{si}")
            model.addConstrs((ea[i] - ed[i] <= dmax[i] for i in range(nF)),
                             name=f"enr_max_{si}")
            # Aircraft-only operated connections.
            model.addConstrs(
                (
                    ed[idx_of[c["flight_j"]]] - ea[idx_of[c["flight_i"]]]
                    >= int(c["min_connection_time"])
                    for c in aircraft_conns
                ),
                name=f"op_conn_{si}",
            )

            # Airport capacity envelopes under the prevailing condition.
            oc = scen.get("operating_conditions", {})
            for a in airports:
                conds = oc.get(a)
                env_a = envelopes.get(a)
                if conds is None or not env_a:
                    continue
                if time.time() - t_start > time_limit - reserve:
                    raise _Abort()
                dcov = dep_cover[a]
                acov = arr_cover[a]
                for t in periods:
                    cond = conds[t - base]
                    segs = env_a.get(cond)
                    if segs is None:
                        segs = env_a.get(str(cond).upper(), [])
                    if not segs:
                        continue
                    di = dcov[t]
                    ai = acov[t]
                    if not di and not ai:
                        continue
                    for seg in segs:
                        acoef = seg.get("a", 0)
                        bcoef = seg.get("b", 0)
                        Q = seg.get("Q", 0)
                        coeffs = []
                        vlist = []
                        if acoef:
                            for i in di:
                                coeffs.append(acoef)
                                vlist.append(yd[i, t])
                        if bcoef:
                            for i in ai:
                                coeffs.append(bcoef)
                                vlist.append(ya[i, t])
                        if vlist:
                            model.addConstr(gp.LinExpr(coeffs, vlist) <= Q)

            yd_flat.append([yd[k] for k in dep_idx])
            ya_flat.append([ya[k] for k in arr_idx])
            vdep_all.append(vdep)
            varr_all.append(varr)

        # ---------------- Objective ----------------
        obj = gp.LinExpr()
        if rho > 0.0:
            obj.addTerms([rho] * nF, [dabs[i] for i in range(nF)])
        w1 = 1.0 - rho
        if w1 > 0.0:
            for si in range(nS):
                w = w1 * probs[si]
                if w == 0.0:
                    continue
                vdep = vdep_all[si]
                varr = varr_all[si]
                obj.addTerms([w * cdep[i] for i in range(nF)],
                             [vdep[i] for i in range(nF)])
                obj.addTerms([w * carr[i] for i in range(nF)],
                             [varr[i] for i in range(nF)])
        model.setObjective(obj, GRB.MINIMIZE)

        td_list = [td[i] for i in range(nF)]

        def assemble(getter):
            """Build the complete output solution dict from variable values."""
            td_vals = getter(td_list)
            tau_d = [0] * nF
            disp = 0.0
            sched = {}
            for i in range(nF):
                v = int(round(td_vals[i]))
                if v < lds[i]:
                    v = lds[i]
                elif v > uds[i]:
                    v = uds[i]
                tau_d[i] = v
                disp += abs(v - req_dep[i])
                sched[fid_str[i]] = {"scheduled_dep": v,
                                     "scheduled_arr": v + delta[i]}
            scen_sols = []
            exp_delay = 0.0
            for si in range(nS):
                ydv = getter(yd_flat[si])
                yav = getter(ya_flat[si])
                xdep = {}
                xarr = {}
                vd = {}
                va = {}
                scen_cost = 0.0
                for i in range(nF):
                    s0, s1 = dep_span[i]
                    bt = ldo[i]
                    acc = 0.0
                    for k in range(s0, s1):
                        val = ydv[k]
                        if val > 1e-6:
                            acc += (bt + (k - s0)) * val
                    thd = int(round(acc))
                    s0, s1 = arr_span[i]
                    bt = lao[i]
                    acc = 0.0
                    for k in range(s0, s1):
                        val = yav[k]
                        if val > 1e-6:
                            acc += (bt + (k - s0)) * val
                    tha = int(round(acc))
                    vdep_i = thd - tau_d[i]
                    if vdep_i < 0:
                        vdep_i = 0
                    varr_i = tha - tau_d[i] - delta[i]
                    if varr_i < 0:
                        varr_i = 0
                    md = thd - base + 1
                    if md < 0:
                        md = 0
                    elif md > T:
                        md = T
                    ma = tha - base + 1
                    if ma < 0:
                        ma = 0
                    elif ma > T:
                        ma = T
                    xdep.update(dict(zip(keys[i], vals_cache[md])))
                    xarr.update(dict(zip(keys[i], vals_cache[ma])))
                    vd[fid_str[i]] = vdep_i
                    va[fid_str[i]] = varr_i
                    scen_cost += cdep[i] * vdep_i + carr[i] * varr_i
                exp_delay += probs[si] * scen_cost
                scen_sols.append({
                    "scenario_id": scen_ids[si],
                    "x_dep": xdep,
                    "x_arr": xarr,
                    "v_dep": vd,
                    "v_arr": va,
                })
            obj_val = rho * disp + (1.0 - rho) * exp_delay
            return obj_val, {
                "objective_value": float(obj_val),
                "schedule": sched,
                "scenario_solutions": scen_sols,
            }

        def cb(m, where):
            if where == GRB.Callback.MIPSOL and logger is not None:
                try:
                    obj_val, sol = assemble(m.cbGetSolution)
                except Exception:
                    return
                if obj_val < best["val"] - 1e-6:
                    best["val"] = obj_val
                    logger.log_solution(obj_val, sol)

        remaining = time_limit - (time.time() - t_start) - reserve
        if remaining <= 0.0:
            raise _Abort()
        model.Params.TimeLimit = float(remaining)
        if logger is not None:
            model.optimize(cb)
        else:
            model.optimize()

        if model.SolCount > 0:
            final_obj, final_sol = assemble(lambda vs: [v.X for v in vs])
            if logger is not None and final_obj < best["val"] - 1e-6:
                best["val"] = final_obj
                logger.log_solution(final_obj, final_sol)
    except _Abort:
        pass
    except Exception as exc:  # pragma: no cover - robustness fallback
        print(f"Warning: optimization aborted with exception: {exc}",
              file=sys.stderr)

    if final_sol is None:
        final_obj, final_sol = fallback_solution()
        if logger is not None:
            logger.log_solution(final_obj, final_sol)

    with open(args.solution_path, "w") as fh:
        json.dump(final_sol, fh)


if __name__ == "__main__":
    main()