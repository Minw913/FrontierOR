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
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    start_time = time.time()

    with open(args.instance_path, "r") as fh:
        inst = json.load(fh)

    T = int(inst["planning_horizon"])
    vessels = inst.get("vessels", [])
    supply_ports = inst.get("supply_ports", [])
    demand_ports = inst.get("demand_ports", [])
    travel_times = inst.get("travel_times", {}) or {}
    draft_limits = inst.get("draft_limits", {}) or {}

    supply_ids = [p["port_id"] for p in supply_ports]
    demand_ids = [p["port_id"] for p in demand_ports]
    port_ids = supply_ids + demand_ids
    is_supply = {pid: True for pid in supply_ids}
    for pid in demand_ids:
        is_supply[pid] = False

    pinfo = {p["port_id"]: p for p in (supply_ports + demand_ports)}
    prod = {p["port_id"]: list(p.get("production_rate", [0] * T)) for p in supply_ports}
    cons = {p["port_id"]: list(p.get("consumption_rate", [0] * T)) for p in demand_ports}
    store_cap = {pid: list(pinfo[pid]["storage_capacity"]) for pid in port_ids}
    init_inv = {pid: float(pinfo[pid]["initial_inventory"]) for pid in port_ids}
    Fmin = {pid: float(pinfo[pid]["F_min"]) for pid in port_ids}
    Fmax = {pid: float(pinfo[pid]["F_max"]) for pid in port_ids}
    pcost = {p["port_id"]: float(p["unit_procurement_cost"]) for p in supply_ports}
    prev_rev = {p["port_id"]: float(p["unit_supply_revenue"]) for p in demand_ports}

    vessel_ids = [v["id"] for v in vessels]
    init_load = {v["id"]: float(v.get("initial_load", 0)) for v in vessels}

    def get_draft(vid, pid, cap):
        val = draft_limits.get("{}->{}".format(vid, pid))
        if val is None:
            return cap
        return min(cap, float(val))

    # ------------------------------------------------------------------
    # Build the per-vessel time-expanded networks.
    # ------------------------------------------------------------------
    vessel_nodes = {}   # vid -> list of (port_id, time)
    vessel_arcs = {}    # vid -> list of arc dicts
    for v in vessels:
        vid = v["id"]
        a = max(1, int(v["availability_start"]))
        b = min(T, int(v["availability_end"]))
        if b < a:
            b = a
        cap = float(v["capacity"])
        dc = float(v["demurrage_cost_per_day"])
        tc = float(v["transport_cost_per_day"])

        nodes = [(pid, t) for t in range(a, b + 1) for pid in port_ids]
        arcs = []

        # source arcs
        for pid in port_ids:
            arcs.append({"tail": ("src",), "head": (pid, a), "type": "source",
                         "cost": 0.0, "U": get_draft(vid, pid, cap)})
        # sink arcs
        for pid in port_ids:
            arcs.append({"tail": (pid, b), "head": ("snk",), "type": "sink",
                         "cost": 0.0, "U": cap})
        # demurrage arcs
        for t in range(a, b):
            for pid in port_ids:
                arcs.append({"tail": (pid, t), "head": (pid, t + 1), "type": "demurrage",
                             "cost": dc, "U": get_draft(vid, pid, cap)})
        # travel arcs
        for t in range(a, b + 1):
            for p in port_ids:
                for q_ in port_ids:
                    if p == q_:
                        continue
                    tt = travel_times.get("{}->{}".format(p, q_))
                    if tt is None:
                        continue
                    tt = int(tt)
                    t2 = t + tt
                    if t2 > b:
                        continue
                    arcs.append({"tail": (p, t), "head": (q_, t2), "type": "travel",
                                 "cost": tc * tt, "U": get_draft(vid, q_, cap)})

        vessel_nodes[vid] = nodes
        vessel_arcs[vid] = arcs

    # Flattened key lists (aligned with the variable lists used later).
    all_keys = []   # (vid, arc_index)
    all_meta = []   # arc dict
    for vid in vessel_ids:
        for i, a in enumerate(vessel_arcs[vid]):
            all_keys.append((vid, i))
            all_meta.append(a)

    node_keys = []  # (vid, port_id, t)
    for vid in vessel_ids:
        for (pid, t) in vessel_nodes[vid]:
            node_keys.append((vid, pid, t))

    def fmt_node(node, vid):
        if node[0] == "src":
            return "(src, {})".format(vid)
        if node[0] == "snk":
            return "(snk, {})".format(vid)
        return "({}, {})".format(node[0], node[1])

    # ------------------------------------------------------------------
    # Fallback solution (used only if no incumbent is found).
    # ------------------------------------------------------------------
    def build_fallback():
        dem_cost = 0.0
        routes = {}
        for v in vessels:
            vid = v["id"]
            a = max(1, int(v["availability_start"]))
            b = min(T, int(v["availability_end"]))
            if b < a:
                b = a
            pid = port_ids[0]
            il = float(v.get("initial_load", 0))
            dc = float(v["demurrage_cost_per_day"])
            route = [{"tail": "(src, {})".format(vid), "head": "({}, {})".format(pid, a),
                      "type": "source", "cost": 0.0, "product_flow": il}]
            for t in range(a, b):
                dem_cost += dc
                route.append({"tail": "({}, {})".format(pid, t),
                              "head": "({}, {})".format(pid, t + 1),
                              "type": "demurrage", "cost": dc, "product_flow": il})
            route.append({"tail": "({}, {})".format(pid, b), "head": "(snk, {})".format(vid),
                          "type": "sink", "cost": 0.0, "product_flow": il})
            routes[vid] = route
        inventories = {}
        for pid in port_ids:
            entries = {}
            prevI = init_inv[pid]
            rate = prod[pid] if is_supply[pid] else cons[pid]
            for t in range(1, T + 1):
                r = rate[t - 1] if t - 1 < len(rate) else 0.0
                cur = prevI + r if is_supply[pid] else prevI - r
                entries[str(t)] = round(cur, 6)
                prevI = cur
            inventories[pid] = entries
        total = dem_cost
        return {
            "objective_value": round(total, 6),
            "vessel_routes": routes,
            "port_inventories": inventories,
            "vessel_operations": {vid: [] for vid in vessel_ids},
            "cost_breakdown": {
                "transport_cost": 0.0,
                "demurrage_cost": round(dem_cost, 6),
                "procurement_cost": 0.0,
                "supply_revenue": 0.0,
                "total_cost": round(total, 6),
            },
        }

    solution = None

    try:
        m = gp.Model("mirp")
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.NumericFocus = 0
        m.Params.Threads = 1
        m.Params.LogToConsole = 0

        x = m.addVars(all_keys, vtype=GRB.BINARY, name="x")
        f = m.addVars(all_keys, lb=0.0, name="f")
        y = m.addVars(node_keys, vtype=GRB.BINARY, name="y")
        qv = m.addVars(node_keys, lb=0.0, name="q")

        xvars = [x[k] for k in all_keys]
        fvars = [f[k] for k in all_keys]
        qvars = [qv[k] for k in node_keys]
        m.setAttr("UB", fvars, [a["U"] for a in all_meta])
        m.setAttr("UB", qvars, [Fmax[nk[1]] for nk in node_keys])

        I = m.addVars([(pid, t) for pid in port_ids for t in range(1, T + 1)],
                      lb=0.0, name="I")
        for pid in port_ids:
            caps = store_cap[pid]
            for t in range(1, T + 1):
                if t - 1 < len(caps):
                    I[(pid, t)].ub = float(caps[t - 1])

        # Objective
        obj = gp.quicksum(all_meta[i]["cost"] * x[all_keys[i]]
                          for i in range(len(all_keys)) if all_meta[i]["cost"] != 0.0)
        obj += gp.quicksum(pcost[nk[1]] * qv[nk] for nk in node_keys if is_supply[nk[1]])
        obj -= gp.quicksum(prev_rev[nk[1]] * qv[nk] for nk in node_keys if not is_supply[nk[1]])
        m.setObjective(obj, GRB.MINIMIZE)

        # Arc incidence
        in_map = {vid: defaultdict(list) for vid in vessel_ids}
        out_map = {vid: defaultdict(list) for vid in vessel_ids}
        for idx, (vid, i) in enumerate(all_keys):
            a = all_meta[idx]
            out_map[vid][a["tail"]].append(idx)
            in_map[vid][a["head"]].append(idx)

        # Path (route) conservation + cargo balance
        for vid in vessel_ids:
            m.addConstr(gp.quicksum(x[all_keys[idx]] for idx in out_map[vid][("src",)]) == 1.0)
            m.addConstr(gp.quicksum(x[all_keys[idx]] for idx in in_map[vid][("snk",)]) == 1.0)
            for (pid, t) in vessel_nodes[vid]:
                node = (pid, t)
                m.addConstr(
                    gp.quicksum(x[all_keys[idx]] for idx in in_map[vid][node])
                    == gp.quicksum(x[all_keys[idx]] for idx in out_map[vid][node])
                )
                bal = (gp.quicksum(f[all_keys[idx]] for idx in in_map[vid][node])
                       - gp.quicksum(f[all_keys[idx]] for idx in out_map[vid][node]))
                if is_supply[pid]:
                    m.addConstr(bal + qv[(vid, pid, t)] == 0.0)
                else:
                    m.addConstr(bal - qv[(vid, pid, t)] == 0.0)

        # Arc cargo capacity (incl. draft at the port being entered) and initial load
        for idx, (vid, i) in enumerate(all_keys):
            a = all_meta[idx]
            k = all_keys[idx]
            m.addConstr(f[k] <= a["U"] * x[k])
            if a["type"] == "source":
                m.addConstr(f[k] == init_load[vid] * x[k])

        # Transfer quantity bounds (all-or-nothing operations)
        for nk in node_keys:
            pid = nk[1]
            m.addConstr(qv[nk] >= Fmin[pid] * y[nk])
            m.addConstr(qv[nk] <= Fmax[pid] * y[nk])

        # First loading operation must begin no later than loading_window_end
        lw_constrs = []
        for v in vessels:
            vid = v["id"]
            Lraw = v.get("loading_window_end", None)
            if Lraw is None:
                continue
            L = int(Lraw)
            b = min(T, int(v["availability_end"]))
            supply_ns = [(pid, t) for (pid, t) in vessel_nodes[vid] if is_supply[pid]]
            if not supply_ns or L >= b:
                continue
            early = [(vid, pid, t) for (pid, t) in supply_ns if t <= L]
            esum = gp.quicksum(y[k] for k in early)
            for (pid, t) in supply_ns:
                lw_constrs.append(m.addConstr(y[(vid, pid, t)] <= esum))

        # Port inventory balance (couples the vessels)
        ops_at = defaultdict(list)
        for nk in node_keys:
            ops_at[(nk[1], nk[2])].append(nk)
        for pid in supply_ids:
            rate = prod[pid]
            for t in range(1, T + 1):
                prevI = init_inv[pid] if t == 1 else I[(pid, t - 1)]
                r = rate[t - 1] if t - 1 < len(rate) else 0.0
                loads = gp.quicksum(qv[k] for k in ops_at.get((pid, t), []))
                m.addConstr(I[(pid, t)] == prevI + r - loads)
        for pid in demand_ids:
            rate = cons[pid]
            for t in range(1, T + 1):
                prevI = init_inv[pid] if t == 1 else I[(pid, t - 1)]
                r = rate[t - 1] if t - 1 < len(rate) else 0.0
                dis = gp.quicksum(qv[k] for k in ops_at.get((pid, t), []))
                m.addConstr(I[(pid, t)] == prevI - r + dis)

        # ------------------------------------------------------------------
        # Assemble a solution dictionary from raw variable values.
        # ------------------------------------------------------------------
        def assemble(xvals, fvals, qvals):
            transport_cost = 0.0
            demurrage_cost = 0.0
            procurement_cost = 0.0
            supply_revenue = 0.0
            load_at = defaultdict(float)
            dis_at = defaultdict(float)
            ops_map = {vid: [] for vid in vessel_ids}
            for idx, (vid, pid, t) in enumerate(node_keys):
                val = qvals[idx]
                if val > 1e-6:
                    if is_supply[pid]:
                        load_at[(pid, t)] += val
                        procurement_cost += val * pcost[pid]
                        ops_map[vid].append({"port": pid, "time": int(t),
                                             "quantity": round(val, 6), "type": "load"})
                    else:
                        dis_at[(pid, t)] += val
                        supply_revenue += val * prev_rev[pid]
                        ops_map[vid].append({"port": pid, "time": int(t),
                                             "quantity": round(-val, 6), "type": "discharge"})
            sel = defaultdict(list)
            for idx, (vid, i) in enumerate(all_keys):
                if xvals[idx] > 0.5:
                    sel[vid].append(idx)
            routes = {}
            for vid in vessel_ids:
                by_tail = {}
                for idx in sel.get(vid, []):
                    by_tail[all_meta[idx]["tail"]] = idx
                route = []
                node = ("src",)
                guard = 0
                max_steps = len(sel.get(vid, [])) + 5
                while guard < max_steps:
                    guard += 1
                    idx = by_tail.get(node)
                    if idx is None:
                        break
                    a = all_meta[idx]
                    if a["type"] == "travel":
                        transport_cost += a["cost"]
                    elif a["type"] == "demurrage":
                        demurrage_cost += a["cost"]
                    route.append({
                        "tail": fmt_node(a["tail"], vid),
                        "head": fmt_node(a["head"], vid),
                        "type": a["type"],
                        "cost": round(a["cost"], 6),
                        "product_flow": round(max(0.0, fvals[idx]), 6),
                    })
                    node = a["head"]
                    if a["type"] == "sink":
                        break
                routes[vid] = route
            inventories = {}
            for pid in port_ids:
                entries = {}
                prevI = init_inv[pid]
                rate = prod[pid] if is_supply[pid] else cons[pid]
                for t in range(1, T + 1):
                    r = rate[t - 1] if t - 1 < len(rate) else 0.0
                    if is_supply[pid]:
                        cur = prevI + r - load_at.get((pid, t), 0.0)
                    else:
                        cur = prevI - r + dis_at.get((pid, t), 0.0)
                    entries[str(t)] = round(cur, 6)
                    prevI = cur
                inventories[pid] = entries
            total = transport_cost + demurrage_cost + procurement_cost - supply_revenue
            return {
                "objective_value": round(total, 6),
                "vessel_routes": routes,
                "port_inventories": inventories,
                "vessel_operations": {
                    vid: sorted(ops_map[vid], key=lambda d: (d["time"], d["port"]))
                    for vid in vessel_ids
                },
                "cost_breakdown": {
                    "transport_cost": round(transport_cost, 6),
                    "demurrage_cost": round(demurrage_cost, 6),
                    "procurement_cost": round(procurement_cost, 6),
                    "supply_revenue": round(supply_revenue, 6),
                    "total_cost": round(total, 6),
                },
            }

        def log_callback(model, where):
            if where == GRB.Callback.MIPSOL and logger is not None:
                try:
                    objval = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    xv = model.cbGetSolution(xvars)
                    fv = model.cbGetSolution(fvars)
                    qq = model.cbGetSolution(qvars)
                    sol = assemble(xv, fv, qq)
                    logger.log_solution(objval, sol)
                except Exception:
                    pass

        remaining = args.time_limit - (time.time() - start_time)
        m.Params.TimeLimit = max(0.5, remaining - 1.0)

        if logger is not None:
            m.optimize(log_callback)
        else:
            m.optimize()

        # Safeguard: if the loading-window interpretation made the model
        # infeasible, retry without those constraints using remaining time.
        if m.Status == GRB.INFEASIBLE and lw_constrs:
            for c in lw_constrs:
                m.remove(c)
            m.update()
            remaining = args.time_limit - (time.time() - start_time)
            if remaining > 1.0:
                m.Params.TimeLimit = max(0.5, remaining - 1.0)
                if logger is not None:
                    m.optimize(log_callback)
                else:
                    m.optimize()

        if m.SolCount > 0:
            xv = [var.X for var in xvars]
            fv = [var.X for var in fvars]
            qq = [var.X for var in qvars]
            solution = assemble(xv, fv, qq)
    except Exception:
        solution = None

    if solution is None:
        solution = build_fallback()

    with open(args.solution_path, "w") as fh:
        json.dump(solution, fh)

    if logger is not None:
        try:
            logger.log_solution(solution["objective_value"], solution)
        except Exception:
            pass


if __name__ == "__main__":
    main()