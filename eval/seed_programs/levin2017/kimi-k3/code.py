import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger

EPS = 1e-6


def solve_instance(inst, time_limit, start_time):
    """Build and solve the LTM-SAV linear program. Returns (solution_dict, objective)."""
    net = inst["network"]
    tpar = inst["time_parameters"]
    demdata = inst["demand"]
    fleet = inst["fleet"]

    T = int(tpar["time_horizon_T"])
    dt_hours = float(tpar["time_step_duration_seconds"]) / 3600.0
    Tf = max(T - 1, 0)  # last time step index at which flows/pickups can occur

    node_type = {}
    for n in net["nodes"]:
        node_type[n["id"]] = n.get("type", "junction")

    links = net["links"]
    nlinks = len(links)
    l_from, l_to, l_kind = [], [], []
    l_fft, l_ctt, l_cap, l_jam = [], [], [], []
    for L in links:
        frm, to = L["from"], L["to"]
        rtype = L.get("type", "road")
        if rtype == "centroid_connector":
            if node_type.get(frm) == "centroid":
                kind = "oc"   # outgoing centroid connector (centroid -> junction)
            else:
                kind = "ic"   # incoming centroid connector (junction -> centroid)
        else:
            kind = "road"
        l_from.append(frm)
        l_to.append(to)
        l_kind.append(kind)
        l_fft.append(int(L.get("free_flow_travel_time_steps", 1)))
        l_ctt.append(int(L.get("congested_travel_time_steps", 1)))
        l_cap.append(float(L.get("capacity_vph", 0.0)) * dt_hours)
        l_jam.append(float(L.get("jam_density_vehicles", 1e9)))

    pred = {}
    succ = {}
    for idx in range(nlinks):
        pred.setdefault(l_to[idx], []).append(idx)
        succ.setdefault(l_from[idx], []).append(idx)

    centroids = [n["id"] for n in net["nodes"] if node_type[n["id"]] == "centroid"]
    junctions = [n["id"] for n in net["nodes"] if node_type[n["id"]] == "junction"]

    # ---------------- demand ----------------
    od_demand = {}   # (o, d) -> list over t
    for od in demdata.get("od_pairs", []):
        o, d = od["origin"], od["destination"]
        key = (o, d)
        if key not in od_demand:
            od_demand[key] = [0.0] * (T + 1)
        for dt_val in od["departure_times"]:
            t = int(dt_val)
            if t < 0:
                t = 0
            if t > T - 1:
                t = T - 1
            od_demand[key][t] += 1.0

    od_pairs = sorted(od_demand.keys())
    dest_set = sorted(set([od[0] for od in od_pairs] + [od[1] for od in od_pairs]))

    init_park = {}
    for c, v in fleet.get("initial_distribution", {}).items():
        init_park[c] = float(v)
    total_init = sum(init_park.values())

    # applicable destination set per link
    app_dests = []
    for idx in range(nlinks):
        if l_kind[idx] == "ic":
            app_dests.append([l_to[idx]] if l_to[idx] in dest_set else [])
        else:
            app_dests.append(list(dest_set))

    remaining = time_limit - (time.time() - start_time)
    if remaining < 1:
        remaining = 1

    m = gp.Model("sav_ltm")
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = remaining
    m.ModelSense = GRB.MINIMIZE

    # ---------------- variables ----------------
    # NU / ND with objective coefficients +1 / -1 per time step
    NU, ND = {}, {}
    for idx in range(nlinks):
        NU[idx] = {}
        ND[idx] = {}
        for d in app_dests[idx]:
            NU[idx][d] = [m.addVar(lb=0.0, obj=1.0) for _ in range(T + 1)]
            ND[idx][d] = [m.addVar(lb=0.0, obj=-1.0) for _ in range(T + 1)]

    # turning flows at junctions
    yt = {}          # (pred_link, succ_link) -> {dest: [vars per t]}
    yt_meta = {}     # (pred_link, succ_link) -> (from_node, via_node, to_node)
    for v in junctions:
        for pidx in pred.get(v, []):
            for sidx in succ.get(v, []):
                if l_kind[sidx] == "ic":
                    dlist = [l_to[sidx]] if l_to[sidx] in dest_set else []
                else:
                    dlist = dest_set
                if not dlist:
                    continue
                key = (pidx, sidx)
                yt[key] = {}
                yt_meta[key] = (l_from[pidx], v, l_to[sidx])
                for d in dlist:
                    yt[key][d] = [m.addVar(lb=0.0) for _ in range(T)]

    # centroid departures
    yc = {}          # outgoing connector link -> {dest: [vars]}
    for idx in range(nlinks):
        if l_kind[idx] == "oc":
            yc[idx] = {}
            for d in dest_set:
                yc[idx][d] = [m.addVar(lb=0.0) for _ in range(T)]

    # parked vehicles
    pvar = {}
    for c in centroids:
        pvar[c] = [m.addVar(lb=0.0) for _ in range(T + 1)]

    # pickups and waiting
    evar, ovar = {}, {}
    for od in od_pairs:
        evar[od] = [m.addVar(lb=0.0) for _ in range(T)]
        ovar[od] = [m.addVar(lb=0.0, obj=1.0) for _ in range(T + 1)]

    # ---------------- constraints ----------------
    zero_const = m.addVar(lb=0.0, ub=0.0)  # constant zero helper

    def nu_term(idx, d, t):
        if t < 0:
            return zero_const
        return NU[idx][d][t]

    # per-link flow dynamics, disaggregated by destination
    for idx in range(nlinks):
        kind = l_kind[idx]
        tail, head = l_from[idx], l_to[idx]
        preds = pred.get(tail, [])
        succs = succ.get(head, [])
        for d in app_dests[idx]:
            NUD = NU[idx][d]
            NDD = ND[idx][d]
            if kind == "road":
                k = l_fft[idx]
                for t in range(T):
                    inflow = [yt[(pr, idx)][d][t] for pr in preds if (pr, idx) in yt and d in yt[(pr, idx)]]
                    outflow = [yt[(idx, s)][d][t] for s in succs if (idx, s) in yt and d in yt[(idx, s)]]
                    if inflow:
                        m.addConstr(NUD[t + 1] - NUD[t] == gp.quicksum(inflow))
                    else:
                        m.addConstr(NUD[t + 1] - NUD[t] == 0.0)
                    if outflow:
                        osum = gp.quicksum(outflow)
                        m.addConstr(NDD[t + 1] - NDD[t] == osum)
                        src = t - k + 1
                        # sending flow bound
                        m.addConstr(osum <= nu_term(idx, d, src) - NDD[t])
                    else:
                        m.addConstr(NDD[t + 1] - NDD[t] == 0.0)
            elif kind == "oc":
                ycd = yc[idx].get(d, [])
                for t in range(T):
                    if ycd:
                        m.addConstr(NUD[t + 1] - NUD[t] == ycd[t])
                    else:
                        m.addConstr(NUD[t + 1] - NUD[t] == 0.0)
                    outflow = [yt[(idx, s)][d][t] for s in succs if (idx, s) in yt and d in yt[(idx, s)]]
                    if outflow:
                        osum = gp.quicksum(outflow)
                        m.addConstr(NDD[t + 1] - NDD[t] == osum)
                        m.addConstr(osum <= NUD[t] - NDD[t])  # FFT = 1
                    else:
                        m.addConstr(NDD[t + 1] - NDD[t] == 0.0)
            else:  # incoming centroid connector
                for t in range(T):
                    inflow = [yt[(pr, idx)][d][t] for pr in preds if (pr, idx) in yt and d in yt[(pr, idx)]]
                    if inflow:
                        m.addConstr(NUD[t + 1] - NUD[t] == gp.quicksum(inflow))
                    else:
                        m.addConstr(NUD[t + 1] - NUD[t] == 0.0)
                    m.addConstr(NDD[t + 1] == NUD[t])

    # road-link aggregate constraints (all destinations): capacity + receiving
    for idx in range(nlinks):
        if l_kind[idx] != "road":
            continue
        capv = l_cap[idx]
        jamv = l_jam[idx]
        w = l_ctt[idx]
        tail, head = l_from[idx], l_to[idx]
        preds = pred.get(tail, [])
        succs = succ.get(head, [])
        for t in range(T):
            out_all = []
            in_all = []
            for s in succs:
                if (idx, s) in yt:
                    for d in yt[(idx, s)]:
                        out_all.append(yt[(idx, s)][d][t])
            for pr in preds:
                if (pr, idx) in yt:
                    for d in yt[(pr, idx)]:
                        in_all.append(yt[(pr, idx)][d][t])
            if out_all:
                m.addConstr(gp.quicksum(out_all) <= capv)
            if in_all:
                in_sum = gp.quicksum(in_all)
                m.addConstr(in_sum <= capv)
                src = t - w + 1
                # congested-wave receiving bound
                recv_terms = []
                for d in app_dests[idx]:
                    recv_terms.append(nu_term(idx, d, src) - ND[idx][d][t])
                m.addConstr(in_sum <= gp.quicksum(recv_terms) + jamv)

    # ---------------- parking ----------------
    for c in centroids:
        ic_links = [idx for idx in pred.get(c, []) if l_kind[idx] == "ic" and c in app_dests[idx]]
        oc_links = [idx for idx in succ.get(c, []) if l_kind[idx] == "oc"]
        init_v = init_park.get(c, 0.0)
        m.addConstr(pvar[c][0] == init_v)
        for t in range(T):
            arrivals = [NU[idx][c][t] - ND[idx][c][t] for idx in ic_links]
            deps = []
            for idx in oc_links:
                for d in yc[idx]:
                    deps.append(yc[idx][d][t])
            ar_expr = gp.quicksum(arrivals) if arrivals else 0.0
            de_expr = gp.quicksum(deps) if deps else 0.0
            m.addConstr(pvar[c][t + 1] == pvar[c][t] + ar_expr - de_expr)
            if deps:
                m.addConstr(de_expr <= pvar[c][t])
    if centroids:
        m.addConstr(gp.quicksum(pvar[c][T] for c in centroids) == total_init)

    # ---------------- demand / waiting ----------------
    for od in od_pairs:
        dem_arr = od_demand[od]
        o, d = od
        depart_links = [idx for idx in succ.get(o, []) if l_kind[idx] == "oc" and d in yc.get(idx, {})]
        e = evar[od]
        om = ovar[od]
        m.addConstr(om[0] == dem_arr[0] if len(dem_arr) > 0 else 0.0)
        for t in range(T):
            nxt_dem = dem_arr[t + 1] if t + 1 <= T else 0.0
            m.addConstr(om[t + 1] == om[t] + nxt_dem - e[t])
            m.addConstr(e[t] <= om[t])
            if depart_links:
                m.addConstr(e[t] <= gp.quicksum(yc[idx][d][t] for idx in depart_links))
            else:
                m.addConstr(e[t] == 0.0)
        m.addConstr(om[T] == 0.0)

    m.optimize()

    # ---------------- extraction ----------------
    feasible = False
    try:
        if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL, GRB.ITERATION_LIMIT,
                        GRB.NODE_LIMIT, GRB.SOLUTION_LIMIT):
            _ = m.ObjVal  # will raise if no solution available
            feasible = True
    except Exception:
        feasible = False

    sol = {"objective_value": 0.0, "y": {}, "y_centroid": {}, "N_U": {}, "N_D": {},
           "p": {}, "e": {}, "omega": {}}

    if feasible:
        try:
            obj = float(m.ObjVal)
        except Exception:
            obj = 0.0
        sol["objective_value"] = obj

        for (pidx, sidx), dd in yt.items():
            frm, via, to = yt_meta[(pidx, sidx)]
            for d, lst in dd.items():
                for t, v in enumerate(lst):
                    x = v.X
                    if x > EPS:
                        sol["y"][f"{frm}|{via}|{to}|{d}|{t}"] = x

        for idx, dd in yc.items():
            frm, to = l_from[idx], l_to[idx]
            for d, lst in dd.items():
                for t, v in enumerate(lst):
                    x = v.X
                    if x > EPS:
                        sol["y_centroid"][f"{frm}|{to}|{d}|{t}"] = x

        for idx in range(nlinks):
            frm, to = l_from[idx], l_to[idx]
            for d in NU[idx]:
                for t, v in enumerate(NU[idx][d]):
                    x = v.X
                    if x > EPS:
                        sol["N_U"][f"{frm}|{to}|{d}|{t}"] = x
                for t, v in enumerate(ND[idx][d]):
                    x = v.X
                    if x > EPS:
                        sol["N_D"][f"{frm}|{to}|{d}|{t}"] = x

        for c in centroids:
            for t, v in enumerate(pvar[c]):
                x = v.X
                if x > EPS:
                    sol["p"][f"{c}|{t}"] = x

        for od in od_pairs:
            o, d = od
            for t, v in enumerate(evar[od]):
                x = v.X
                if x > EPS:
                    sol["e"][f"{o}|{d}|{t}"] = x
            for t, v in enumerate(ovar[od]):
                x = v.X
                if x > EPS:
                    sol["omega"][f"{o}|{d}|{t}"] = x
    else:
        # Fallback: a structured but possibly infeasible empty solution.
        for c in centroids:
            if init_park.get(c, 0.0) > EPS:
                sol["p"][f"{c}|0"] = init_park[c]
        dem_total = sum(sum(a) for a in od_demand.values())
        sol["objective_value"] = float(dem_total * (T + 1))

    m.dispose()
    return sol, sol["objective_value"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    sol, obj = solve_instance(inst, args.time_limit, start_time)

    if logger:
        logger.log_solution(obj, sol)

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()