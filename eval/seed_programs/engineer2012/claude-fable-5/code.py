import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None

SLACK_PEN = 1.0e6


def node_str(n, vid):
    if n == ("src",):
        return f"(src, {vid})"
    if n == ("snk",):
        return f"(snk, {vid})"
    return f"({n[0]}, {n[1]})"


def fallback_solution(data):
    """Trivial solution: every vessel idles at the first port, no operations."""
    T = data["planning_horizon"]
    supply = data["supply_ports"]
    demand = data["demand_ports"]
    ports = [p["port_id"] for p in supply] + [p["port_id"] for p in demand]
    p0 = ports[0]

    vessel_routes = {}
    vessel_ops = {}
    dem_cost = 0.0
    for v in data["vessels"]:
        vid = v["id"]
        t0 = max(0, int(v["availability_start"]))
        t1 = min(T, int(v["availability_end"]))
        if t1 < t0:
            t1 = t0
        init = float(v["initial_load"])
        dc = float(v["demurrage_cost_per_day"])
        route = [{"tail": node_str(("src",), vid), "head": node_str((p0, t0), vid),
                  "type": "source", "cost": 0.0, "product_flow": init}]
        for t in range(t0, t1):
            route.append({"tail": node_str((p0, t), vid), "head": node_str((p0, t + 1), vid),
                          "type": "demurrage", "cost": dc, "product_flow": init})
            dem_cost += dc
        route.append({"tail": node_str((p0, t1), vid), "head": node_str(("snk",), vid),
                      "type": "sink", "cost": 0.0, "product_flow": init})
        vessel_routes[vid] = route
        vessel_ops[vid] = []

    port_inv = {}
    for p in supply:
        inv = {}
        cur = float(p["initial_inventory"])
        for t in range(1, T + 1):
            cur = cur + p["production_rate"][t - 1]
            inv[str(t)] = cur
        port_inv[p["port_id"]] = inv
    for p in demand:
        inv = {}
        cur = float(p["initial_inventory"])
        for t in range(1, T + 1):
            cur = cur - p["consumption_rate"][t - 1]
            inv[str(t)] = cur
        port_inv[p["port_id"]] = inv

    total = dem_cost
    return {
        "objective_value": total,
        "vessel_routes": vessel_routes,
        "port_inventories": port_inv,
        "vessel_operations": vessel_ops,
        "cost_breakdown": {
            "transport_cost": 0.0,
            "demurrage_cost": dem_cost,
            "procurement_cost": 0.0,
            "supply_revenue": 0.0,
            "total_cost": total,
        },
    }


def solve(data, time_limit, logger):
    T = data["planning_horizon"]
    supply = data["supply_ports"]
    demand = data["demand_ports"]
    ports = [p["port_id"] for p in supply] + [p["port_id"] for p in demand]
    is_supply = {}
    for p in supply:
        is_supply[p["port_id"]] = True
    for p in demand:
        is_supply[p["port_id"]] = False
    pinfo = {p["port_id"]: p for p in supply + demand}
    tt = data["travel_times"]
    draftd = data["draft_limits"]
    vessels = data["vessels"]
    vids = [v["id"] for v in vessels]

    m = gp.Model("mirp")
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = max(1.0, time_limit)

    obj = gp.LinExpr()
    arcs = {}          # vid -> list of arc dicts
    in_arcs = {}       # (vid,node) -> list of arcs
    out_arcs = {}
    vtimes = {}        # vid -> (t0,t1)

    SRC = ("src",)
    SNK = ("snk",)

    for v in vessels:
        vid = v["id"]
        cap = float(v["capacity"])
        t0 = max(0, int(v["availability_start"]))
        t1 = min(T, int(v["availability_end"]))
        if t1 < t0:
            t1 = t0
        vtimes[vid] = (t0, t1)
        alist = []
        arcs[vid] = alist

        def add_arc(tail, head, typ, cost, ub):
            ub = max(0.0, float(ub))
            x = m.addVar(vtype=GRB.BINARY)
            l = m.addVar(lb=0.0, ub=ub)
            m.addConstr(l <= ub * x)
            a = {"tail": tail, "head": head, "type": typ, "cost": float(cost), "x": x, "l": l}
            alist.append(a)
            in_arcs.setdefault((vid, head), []).append(a)
            out_arcs.setdefault((vid, tail), []).append(a)
            obj.add(x, float(cost))
            return a

        init_load = float(v["initial_load"])
        # source arcs
        for p in ports:
            a = add_arc(SRC, (p, t0), "source", 0.0, max(cap, init_load))
            m.addConstr(a["l"] == init_load * a["x"])
        # sink arcs
        for p in ports:
            add_arc((p, t1), SNK, "sink", 0.0, cap)
        # demurrage arcs
        dc = float(v["demurrage_cost_per_day"])
        for p in ports:
            for t in range(t0, t1):
                add_arc((p, t), (p, t + 1), "demurrage", dc, cap)
        # travel arcs
        tc = float(v["transport_cost_per_day"])
        for p in ports:
            for q in ports:
                if p == q:
                    continue
                tau = tt.get(f"{p}->{q}")
                if tau is None:
                    continue
                tau = max(1, int(tau))
                ubq = min(cap, float(draftd.get(f"{vid}->{q}", cap)))
                for t in range(t0, t1 - tau + 1):
                    add_arc((p, t), (q, t + tau), "travel", tc * tau, ubq)

        # flow conservation
        m.addConstr(gp.quicksum(a["x"] for a in out_arcs.get((vid, SRC), [])) == 1)
        m.addConstr(gp.quicksum(a["x"] for a in in_arcs.get((vid, SNK), [])) == 1)

    # operation variables
    qov = {}  # (vid,p,t) -> (o, q)
    for v in vessels:
        vid = v["id"]
        t0, t1 = vtimes[vid]
        for p in ports:
            info = pinfo[p]
            fmin = float(info["F_min"])
            fmax = float(info["F_max"])
            for t in range(max(t0, 1), min(t1, T) + 1):
                o = m.addVar(vtype=GRB.BINARY)
                q = m.addVar(lb=0.0, ub=fmax)
                m.addConstr(q >= fmin * o)
                m.addConstr(q <= fmax * o)
                m.addConstr(o <= gp.quicksum(a["x"] for a in in_arcs.get((vid, (p, t)), [])))
                qov[(vid, p, t)] = (o, q)
                if is_supply[p]:
                    obj.add(q, float(info["unit_procurement_cost"]))
                else:
                    obj.add(q, -float(info["unit_supply_revenue"]))

    # vessel flow conservation + cargo conservation at nodes
    for v in vessels:
        vid = v["id"]
        t0, t1 = vtimes[vid]
        for p in ports:
            for t in range(t0, t1 + 1):
                node = (p, t)
                ins = in_arcs.get((vid, node), [])
                outs = out_arcs.get((vid, node), [])
                m.addConstr(gp.quicksum(a["x"] for a in ins) ==
                            gp.quicksum(a["x"] for a in outs))
                delta = gp.LinExpr()
                key = (vid, p, t)
                if key in qov:
                    if is_supply[p]:
                        delta.add(qov[key][1], 1.0)
                    else:
                        delta.add(qov[key][1], -1.0)
                m.addConstr(gp.quicksum(a["l"] for a in ins) + delta ==
                            gp.quicksum(a["l"] for a in outs))

    # port inventories (with penalized slack for robustness)
    svar = {}
    for p in ports:
        info = pinfo[p]
        rates = info["production_rate"] if is_supply[p] else info["consumption_rate"]
        caps = info["storage_capacity"]
        prev = float(info["initial_inventory"])
        for t in range(1, T + 1):
            s = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
            opsum = gp.quicksum(qov[(vid, p, t)][1] for vid in vids if (vid, p, t) in qov)
            if is_supply[p]:
                m.addConstr(s == prev + float(rates[t - 1]) - opsum)
            else:
                m.addConstr(s == prev - float(rates[t - 1]) + opsum)
            u = m.addVar(lb=0.0)
            w = m.addVar(lb=0.0)
            m.addConstr(s + u >= 0.0)
            m.addConstr(s - w <= float(caps[t - 1]))
            obj.add(u, SLACK_PEN)
            obj.add(w, SLACK_PEN)
            svar[(p, t)] = s
            prev = s

    m.setObjective(obj, GRB.MINIMIZE)

    best = [float("inf")]

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if val < best[0] - 1e-6:
                best[0] = val
                if logger:
                    logger.log(val)

    m.optimize(cb)

    if m.SolCount == 0:
        return None

    # ---------- extraction ----------
    vessel_routes = {}
    vessel_ops = {}
    transport_cost = 0.0
    demurrage_cost = 0.0
    proc_cost = 0.0
    revenue = 0.0

    for v in vessels:
        vid = v["id"]
        chosen = [a for a in arcs[vid] if a["x"].X > 0.5]
        by_tail = {}
        for a in chosen:
            by_tail[a["tail"]] = a
        route = []
        cur = SRC
        guard = 0
        while cur != SNK and guard < len(chosen) + 2:
            a = by_tail.get(cur)
            if a is None:
                break
            route.append({
                "tail": node_str(a["tail"], vid),
                "head": node_str(a["head"], vid),
                "type": a["type"],
                "cost": a["cost"],
                "product_flow": float(a["l"].X),
            })
            if a["type"] == "travel":
                transport_cost += a["cost"]
            elif a["type"] == "demurrage":
                demurrage_cost += a["cost"]
            cur = a["head"]
            guard += 1
        vessel_routes[vid] = route

        ops = []
        t0, t1 = vtimes[vid]
        for p in ports:
            for t in range(max(t0, 1), min(t1, T) + 1):
                key = (vid, p, t)
                if key not in qov:
                    continue
                o, q = qov[key]
                if o.X > 0.5 and q.X > 1e-6:
                    qty = float(q.X)
                    if is_supply[p]:
                        ops.append({"port": p, "time": t, "quantity": qty, "type": "load"})
                        proc_cost += float(pinfo[p]["unit_procurement_cost"]) * qty
                    else:
                        ops.append({"port": p, "time": t, "quantity": -qty, "type": "discharge"})
                        revenue += float(pinfo[p]["unit_supply_revenue"]) * qty
        ops.sort(key=lambda e: e["time"])
        vessel_ops[vid] = ops

    port_inv = {}
    for p in ports:
        inv = {}
        for t in range(1, T + 1):
            inv[str(t)] = float(svar[(p, t)].X)
        port_inv[p] = inv

    total = transport_cost + demurrage_cost + proc_cost - revenue
    sol = {
        "objective_value": total,
        "vessel_routes": vessel_routes,
        "port_inventories": port_inv,
        "vessel_operations": vessel_ops,
        "cost_breakdown": {
            "transport_cost": transport_cost,
            "demurrage_cost": demurrage_cost,
            "procurement_cost": proc_cost,
            "supply_revenue": revenue,
            "total_cost": total,
        },
    }
    return sol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t_start = time.time()
    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path) as f:
        data = json.load(f)

    budget = args.time_limit - (time.time() - t_start) - 3.0
    sol = None
    try:
        if budget > 1.0:
            sol = solve(data, budget, logger)
    except Exception:
        sol = None

    if sol is None:
        sol = fallback_solution(data)

    if logger:
        logger.log_solution(sol["objective_value"], sol)

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()