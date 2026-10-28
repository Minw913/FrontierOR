import argparse
import json
import time
import math

from solution_logger import SolutionLogger

import gurobipy as gp
from gurobipy import GRB


def fetch_by_float(d, val):
    """Look up a dict keyed by stringified floats using a numeric value."""
    if d is None:
        raise KeyError(val)
    if val in d:
        return d[val]
    sval = str(val)
    if sval in d:
        return d[sval]
    try:
        f = float(val)
    except (TypeError, ValueError):
        f = None
    for k, v in d.items():
        try:
            if f is not None and abs(float(k) - f) < 1e-6:
                return v
        except (TypeError, ValueError):
            continue
    raise KeyError(f"key {val} not found in {list(d.keys())[:5]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, required=True)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t_start = time.time()

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    cp_ = inst["cost_parameters"]
    fuel_price = float(cp_["fuel_price_usd_per_tonne"])
    ch_cost = float(cp_["handling_cost_usd_per_hour"])
    cd_cost = float(cp_["delay_cost_usd_per_hour"])
    cw_cost = float(cp_["waiting_cost_usd_per_hour"])
    clft = float(cp_["lft_penalty_usd"])
    beta = float(inst["handling_time_deviation_factor_beta_per_meter"])
    horizon = float(inst["planning_horizon_hours"])

    ports = {p["id"]: float(p["quay_length_meters"]) for p in inst["ports"]}
    dist = inst["distances_nautical_miles"]
    speeds = [float(s) for s in inst["speeds_knots"]]
    nspeed = len(speeds)
    ttnm_map = inst["travel_time_hours_per_nm_by_speed"]
    ttnm = [float(fetch_by_float(ttnm_map, s)) for s in speeds]

    ext_by_port = {}
    for grp in inst.get("external_ships", []):
        lst = []
        for es in grp["ships"]:
            x0 = float(es["berthing_position"])
            x1 = x0 + float(es["length_meters"])
            lst.append((x0, x1, float(es["start_time_hours"]), float(es["end_time_hours"])))
        ext_by_port[grp["port_id"]] = lst

    shiprecs = []
    for sh in inst["ships"]:
        route = list(sh["route"])
        rec = {"id": sh["id"], "L": float(sh["length_meters"]), "route": route}
        pv_list = []
        for pv in sh["port_visits"]:
            pv_list.append({
                "port": pv["port_id"],
                "idx": int(pv["port_call_index"]),
                "ideal": float(pv["ideal_berthing_position"]),
                "hmin": float(pv["min_handling_time_hours"]),
                "hmax": float(pv["max_handling_time_hours"]),
                "est": float(pv["earliest_start_time_hours"]),
                "eft": float(pv["expected_finish_time_hours"]),
                "lft": float(pv["latest_finish_time_hours"]),
            })
        rec["visits"] = pv_list
        fc_map = sh["fuel_consumption_per_nm_by_speed"]
        fcs = [float(fetch_by_float(fc_map, s)) for s in speeds]
        legs = []
        for l in range(len(route) - 1):
            key = f"{route[l]}_{route[l+1]}"
            if key in dist:
                d_ = float(dist[key])
            else:
                key2 = f"{route[l+1]}_{route[l]}"
                if key2 in dist:
                    d_ = float(dist[key2])
                else:
                    raise KeyError(f"No distance for {key}")
            legs.append({
                "dist": d_,
                "tt": [d_ * ttnm[i] for i in range(nspeed)],
                "fuelcost": [fuel_price * d_ * fcs[i] for i in range(nspeed)],
            })
        rec["legs"] = legs
        span = sum(v["hmax"] for v in pv_list) + sum(max(lg["tt"]) for lg in legs)
        rec["span"] = span
        shiprecs.append(rec)

    total_span = sum(r["span"] for r in shiprecs) if shiprecs else 1.0
    Mt = horizon + total_span + 100.0

    # ---------------- heuristic (fallback / initial incumbent / MIP start) ----------------
    def clamp(v, lo, hi):
        return min(max(v, lo), hi)

    def run_heuristic():
        placed = {p: list(lst) for p, lst in ext_by_port.items()}
        valmap = {}
        if not shiprecs:
            return valmap
        order = sorted(range(len(shiprecs)),
                       key=lambda i: min(v["est"] for v in shiprecs[i]["visits"]))
        for si in order:
            rec = shiprecs[si]
            L = rec["L"]
            prev_end = None
            for k, pv in enumerate(rec["visits"]):
                port = pv["port"]
                Q = ports[port]
                portRects = placed.get(port, [])
                candp = {clamp(pv["ideal"], 0, Q - L), 0.0, max(0.0, Q - L)}
                for r in portRects:
                    candp.add(clamp(r[1], 0, Q - L))
                    candp.add(clamp(r[0] - L, 0, Q - L))
                candp = [c for c in candp if -1e-9 <= c <= max(0.0, Q - L) + 1e-9]
                best = None
                sp_iter = range(nspeed) if k > 0 else [None]
                for spi in sp_iter:
                    if k > 0:
                        leg = rec["legs"][k - 1]
                        arr = prev_end + leg["tt"][spi]
                        legcost = leg["fuelcost"][spi]
                    else:
                        arr = None
                        legcost = 0.0
                    t0 = pv["est"] if k == 0 else max(arr, pv["est"])
                    for pos in candp:
                        dev = abs(pos - pv["ideal"])
                        h = pv["hmin"] * (1.0 + beta * dev)
                        t = t0
                        while True:
                            ends = []
                            for (x0, x1, s, e) in portRects:
                                if x0 < pos + L and x1 > pos:
                                    if s < t + h and e > t:
                                        ends.append(e)
                            if not ends:
                                break
                            t = max(ends)
                        ev = t + h
                        wait = 0.0 if k == 0 else max(0.0, t - arr)
                        delay = max(0.0, ev - pv["eft"])
                        lftv = max(0.0, ev - pv["lft"])
                        cost = cw_cost * wait + ch_cost * h + cd_cost * delay + clft * lftv + legcost
                        if best is None or cost < best[0] - 1e-9:
                            best = (cost, pos, t, h, dev, wait, delay, lftv,
                                    spi, (t if k == 0 else arr))
                (cost, pos, t, h, dev, wait, delay, lftv, spi, arrv) = best
                placed.setdefault(port, []).append((pos, pos + L, t, t + h))
                prev_end = t + h
                valmap[("X", si, k)] = pos
                valmap[("T", si, k)] = t
                valmap[("A", si, k)] = arrv
                valmap[("D", si, k)] = dev
                valmap[("H", si, k)] = h
                valmap[("E", si, k)] = t + h
                valmap[("W", si, k)] = wait
                valmap[("DE", si, k)] = delay
                valmap[("LV", si, k)] = lftv
                if k > 0:
                    for i in range(nspeed):
                        valmap[("Z", si, k - 1, i)] = 1.0 if i == spi else 0.0
        return valmap

    # ---------------- solution extraction ----------------
    def extract(getter, objval=None, compute_obj=False):
        out_ships = []
        total = 0.0
        for si, rec in enumerate(shiprecs):
            pv_out = []
            for k, pv in enumerate(rec["visits"]):
                xv = float(getter(("X", si, k)))
                tv = float(getter(("T", si, k)))
                av = float(getter(("A", si, k)))
                dev = abs(xv - pv["ideal"])
                h = pv["hmin"] * (1.0 + beta * dev)
                ev = tv + h
                delay = max(0.0, ev - pv["eft"])
                lftv = max(0.0, ev - pv["lft"])
                wait = max(0.0, tv - av)
                pv_out.append({
                    "port_call_index": pv["idx"],
                    "port_id": pv["port"],
                    "berthing_position": xv,
                    "start_time": tv,
                    "handling_time": h,
                    "arrival_time": av,
                    "delay": delay,
                    "lft_violation": lftv,
                    "position_deviation": dev,
                })
                if compute_obj:
                    total += cw_cost * wait + ch_cost * h + cd_cost * delay + clft * lftv
            sp_out = []
            for l in range(len(rec["legs"])):
                if nspeed > 0:
                    vals = [getter(("Z", si, l, i)) for i in range(nspeed)]
                    bi = max(range(nspeed), key=lambda i: vals[i])
                else:
                    bi = 0
                sp_out.append({"from_call": l + 1, "to_call": l + 2,
                               "speed_knots": speeds[bi] if nspeed > 0 else 0.0})
                if compute_obj and nspeed > 0:
                    total += rec["legs"][l]["fuelcost"][bi]
            out_ships.append({"ship_id": rec["id"],
                              "port_visits": pv_out,
                              "speed_selections": sp_out})
        obj = total if compute_obj else objval
        return {"objective_value": float(obj), "ships": out_ships}

    heuristic_valmap = run_heuristic()
    if logger and heuristic_valmap is not None:
        getter0 = lambda key: heuristic_valmap.get(key, 0.0)
        hsol = extract(getter0, compute_obj=True)
        logger.log_solution(hsol["objective_value"], hsol)

    final_sol = None
    # ---------------- MIP ----------------
    try:
        model = gp.Model("BASP")
        model.setParam("Seed", 0)
        model.setParam("MIPGap", 1e-4)
        model.setParam("NumericFocus", 0)
        model.setParam("Threads", 1)
        remaining = args.time_limit - (time.time() - t_start)
        model.setParam("TimeLimit", max(0.01, remaining - 1.0))

        V = {}
        obj = gp.LinExpr()

        for si, rec in enumerate(shiprecs):
            for k, pv in enumerate(rec["visits"]):
                Q = ports[pv["port"]]
                L = rec["L"]
                V[("X", si, k)] = model.addVar(lb=0.0, ub=max(0.0, Q - L), name=f"x_{si}_{k}")
                V[("T", si, k)] = model.addVar(lb=pv["est"], ub=Mt, name=f"t_{si}_{k}")
                V[("A", si, k)] = model.addVar(lb=0.0, ub=Mt, name=f"a_{si}_{k}")
                V[("D", si, k)] = model.addVar(lb=0.0, ub=Q, name=f"d_{si}_{k}")
                V[("H", si, k)] = model.addVar(lb=0.0, ub=Mt, name=f"h_{si}_{k}")
                V[("E", si, k)] = model.addVar(lb=0.0, ub=Mt, name=f"e_{si}_{k}")
                V[("W", si, k)] = model.addVar(lb=0.0, ub=Mt, name=f"w_{si}_{k}")
                V[("DE", si, k)] = model.addVar(lb=0.0, ub=Mt, name=f"de_{si}_{k}")
                V[("LV", si, k)] = model.addVar(lb=0.0, ub=Mt, name=f"lv_{si}_{k}")
                ideal = pv["ideal"]
                model.addConstr(V[("D", si, k)] >= V[("X", si, k)] - ideal)
                model.addConstr(V[("D", si, k)] >= ideal - V[("X", si, k)])
                model.addConstr(V[("H", si, k)] == pv["hmin"] * (1.0 + beta * V[("D", si, k)]))
                model.addConstr(V[("E", si, k)] == V[("T", si, k)] + V[("H", si, k)])
                if k == 0:
                    model.addConstr(V[("T", si, k)] >= V[("A", si, k)])
                else:
                    leg = rec["legs"][k - 1]
                    zs = [model.addVar(vtype=GRB.BINARY, name=f"z_{si}_{k-1}_{i}")
                          for i in range(nspeed)]
                    for i in range(nspeed):
                        V[("Z", si, k - 1, i)] = zs[i]
                    model.addConstr(gp.quicksum(zs) == 1.0, name=f"onespeed_{si}_{k-1}")
                    tte = gp.quicksum(leg["tt"][i] * zs[i] for i in range(nspeed))
                    model.addConstr(V[("A", si, k)] == V[("E", si, k - 1)] + tte)
                    model.addConstr(V[("T", si, k)] >= V[("A", si, k)])
                    for i in range(nspeed):
                        obj += leg["fuelcost"][i] * zs[i]
                model.addConstr(V[("W", si, k)] >= V[("T", si, k)] - V[("A", si, k)])
                model.addConstr(V[("DE", si, k)] >= V[("E", si, k)] - pv["eft"])
                model.addConstr(V[("LV", si, k)] >= V[("E", si, k)] - pv["lft"])
                obj += (cw_cost * V[("W", si, k)] + ch_cost * V[("H", si, k)]
                        + cd_cost * V[("DE", si, k)] + clft * V[("LV", si, k)])

        # non-overlap constraints
        by_port = {}
        for si, rec in enumerate(shiprecs):
            for k, pv in enumerate(rec["visits"]):
                by_port.setdefault(pv["port"], []).append((si, k))

        for port, keys in by_port.items():
            Q = ports[port]
            Mq = Q
            # optimized vs optimized
            for a_i in range(len(keys)):
                for b_i in range(a_i + 1, len(keys)):
                    si, k = keys[a_i]
                    sj, l = keys[b_i]
                    Li = shiprecs[si]["L"]
                    Lj = shiprecs[sj]["L"]
                    b1 = model.addVar(vtype=GRB.BINARY)
                    b2 = model.addVar(vtype=GRB.BINARY)
                    b3 = model.addVar(vtype=GRB.BINARY)
                    b4 = model.addVar(vtype=GRB.BINARY)
                    model.addConstr(V[("X", si, k)] + Li <= V[("X", sj, l)] + Mq * (1 - b1))
                    model.addConstr(V[("X", sj, l)] + Lj <= V[("X", si, k)] + Mq * (1 - b2))
                    model.addConstr(V[("E", si, k)] <= V[("T", sj, l)] + Mt * (1 - b3))
                    model.addConstr(V[("E", sj, l)] <= V[("T", si, k)] + Mt * (1 - b4))
                    model.addConstr(b1 + b2 + b3 + b4 >= 1)
            # optimized vs external
            exts = ext_by_port.get(port, [])
            for (si, k) in keys:
                Ls = shiprecs[si]["L"]
                for (x0, x1, s, e) in exts:
                    b1 = model.addVar(vtype=GRB.BINARY)
                    b2 = model.addVar(vtype=GRB.BINARY)
                    b3 = model.addVar(vtype=GRB.BINARY)
                    b4 = model.addVar(vtype=GRB.BINARY)
                    model.addConstr(V[("X", si, k)] + Ls <= x0 + Mq * (1 - b1))
                    model.addConstr(x1 <= V[("X", si, k)] + Mq * (1 - b2))
                    model.addConstr(V[("E", si, k)] <= s + Mt * (1 - b3))
                    model.addConstr(e <= V[("T", si, k)] + Mt * (1 - b4))
                    model.addConstr(b1 + b2 + b3 + b4 >= 1)

        # heuristic MIP start
        for key, val in (heuristic_valmap or {}).items():
            if key in V:
                V[key].Start = val

        model.setObjective(obj, GRB.MINIMIZE)

        best_logged = [float("inf")]

        def cb(m, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    objv = m.cbGet(GRB.Callback.MIPSOL_OBJ)
                except Exception:
                    return
                if objv < best_logged[0] - 1e-6:
                    best_logged[0] = objv
                    if logger:
                        getter = lambda key: m.cbGet(V[key])
                        sol = extract(getter, objval=objv)
                        logger.log_solution(objv, sol)

        if logger:
            model.optimize(cb)
        else:
            model.optimize()

        if model.SolCount > 0:
            getter = lambda key: V[key].X
            final_sol = extract(getter, objval=model.ObjVal)
    except Exception:
        final_sol = None

    if final_sol is None:
        if heuristic_valmap is not None:
            getter = lambda key: heuristic_valmap.get(key, 0.0)
            final_sol = extract(getter, compute_obj=True)
        else:
            final_sol = {"objective_value": 0.0, "ships": []}

    with open(args.solution_path, "w") as f:
        json.dump(final_sol, f, indent=2)


if __name__ == "__main__":
    main()