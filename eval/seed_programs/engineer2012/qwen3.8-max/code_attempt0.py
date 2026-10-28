import argparse
import json
import math
import time
from collections import defaultdict

try:
    from solution_logger import SolutionLogger
except Exception:
    class SolutionLogger:
        def __init__(self, *args, **kwargs):
            pass

        def log(self, objective_value):
            pass

        def log_solution(self, objective_value, solution):
            pass


def clean_number(x):
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return 0.0
        if abs(x) < 1e-9:
            return 0.0
        return round(x, 6)
    except Exception:
        return 0.0


def _series_to_list(raw, T, default=0.0):
    if T <= 0:
        return []
    if raw is None:
        return [float(default)] * T
    if isinstance(raw, dict):
        out = []
        for t in range(1, T + 1):
            val = raw.get(str(t), raw.get(t, default))
            try:
                out.append(float(val))
            except Exception:
                out.append(float(default))
        return out
    if isinstance(raw, list):
        out = []
        for x in raw[:T]:
            try:
                out.append(float(x))
            except Exception:
                out.append(float(default))
        while len(out) < T:
            out.append(float(default))
        return out
    try:
        return [float(raw)] * T
    except Exception:
        return [float(default)] * T


def parse_instance(path):
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        data = {}

    try:
        T = int(data.get("planning_horizon", 0) or 0)
    except Exception:
        T = 0
    if T < 0:
        T = 0

    vessels = []
    for i, vd in enumerate(data.get("vessels", []) or []):
        vid = str(vd.get("vessel_id", vd.get("id", f"V{i}")))
        try:
            capacity = float(vd.get("capacity", 0.0) or 0.0)
        except Exception:
            capacity = 0.0
        try:
            availability_start = int(vd.get("availability_start", 1) or 1)
        except Exception:
            availability_start = 1
        try:
            availability_end = int(vd.get("availability_end", T) or T)
        except Exception:
            availability_end = T
        try:
            loading_window_end = int(vd.get("loading_window_end", availability_end))
        except Exception:
            loading_window_end = availability_end
        try:
            transport_cost = float(vd.get("transport_cost_per_day", 0.0) or 0.0)
        except Exception:
            transport_cost = 0.0
        try:
            demurrage_cost = float(vd.get("demurrage_cost_per_day", 0.0) or 0.0)
        except Exception:
            demurrage_cost = 0.0
        try:
            initial_load = float(vd.get("initial_load", 0.0) or 0.0)
        except Exception:
            initial_load = 0.0

        op_start = max(1, availability_start)
        op_end = min(T, availability_end)

        vessels.append(
            {
                "id": vid,
                "capacity": capacity,
                "availability_start": availability_start,
                "availability_end": availability_end,
                "loading_window_end": loading_window_end,
                "transport_cost": transport_cost,
                "demurrage_cost": demurrage_cost,
                "initial_load": initial_load,
                "op_start": op_start,
                "op_end": op_end,
            }
        )

    ports = {}
    supply_ids = []
    demand_ids = []

    for i, pd in enumerate(data.get("supply_ports", []) or []):
        pid = str(pd.get("port_id", pd.get("id", f"S{i}")))
        try:
            initial = float(pd.get("initial_inventory", 0.0) or 0.0)
        except Exception:
            initial = 0.0
        production = _series_to_list(pd.get("production_rate", []), T, 0.0)
        storage = _series_to_list(pd.get("storage_capacity", []), T, 1e12)
        storage = [max(0.0, x) for x in storage]
        try:
            f_min = float(pd.get("F_min", 0.0) or 0.0)
        except Exception:
            f_min = 0.0
        try:
            f_max = float(pd.get("F_max", 0.0) or 0.0)
        except Exception:
            f_max = 0.0
        if f_min < 0:
            f_min = 0.0
        if f_max < 0:
            f_max = 0.0
        if f_min > f_max:
            f_min = f_max
        try:
            unit_cost = float(pd.get("unit_procurement_cost", 0.0) or 0.0)
        except Exception:
            unit_cost = 0.0
        try:
            x = float(pd.get("x", 0.0) or 0.0)
        except Exception:
            x = 0.0
        try:
            y = float(pd.get("y", 0.0) or 0.0)
        except Exception:
            y = 0.0

        ports[pid] = {
            "id": pid,
            "type": "supply",
            "initial": initial,
            "rate": production,
            "storage": storage,
            "F_min": f_min,
            "F_max": f_max,
            "unit_cost": unit_cost,
            "unit_rev": 0.0,
            "x": x,
            "y": y,
        }
        supply_ids.append(pid)

    for i, pd in enumerate(data.get("demand_ports", []) or []):
        pid = str(pd.get("port_id", pd.get("id", f"D{i}")))
        try:
            initial = float(pd.get("initial_inventory", 0.0) or 0.0)
        except Exception:
            initial = 0.0
        consumption = _series_to_list(pd.get("consumption_rate", []), T, 0.0)
        storage = _series_to_list(pd.get("storage_capacity", []), T, 1e12)
        storage = [max(0.0, x) for x in storage]
        try:
            f_min = float(pd.get("F_min", 0.0) or 0.0)
        except Exception:
            f_min = 0.0
        try:
            f_max = float(pd.get("F_max", 0.0) or 0.0)
        except Exception:
            f_max = 0.0
        if f_min < 0:
            f_min = 0.0
        if f_max < 0:
            f_max = 0.0
        if f_min > f_max:
            f_min = f_max
        try:
            unit_rev = float(pd.get("unit_supply_revenue", 0.0) or 0.0)
        except Exception:
            unit_rev = 0.0
        try:
            x = float(pd.get("x", 0.0) or 0.0)
        except Exception:
            x = 0.0
        try:
            y = float(pd.get("y", 0.0) or 0.0)
        except Exception:
            y = 0.0

        ports[pid] = {
            "id": pid,
            "type": "demand",
            "initial": initial,
            "rate": consumption,
            "storage": storage,
            "F_min": f_min,
            "F_max": f_max,
            "unit_cost": 0.0,
            "unit_rev": unit_rev,
            "x": x,
            "y": y,
        }
        demand_ids.append(pid)

    travel = {}
    for k, v in (data.get("travel_times", {}) or {}).items():
        ks = str(k)
        if "->" not in ks:
            continue
        o, d = ks.split("->", 1)
        o = o.strip()
        d = d.strip()
        try:
            tau = int(math.ceil(float(v)))
        except Exception:
            tau = 1
        travel[(o, d)] = max(0, tau)

    draft = {}
    for k, v in (data.get("draft_limits", {}) or {}).items():
        ks = str(k)
        if "->" not in ks:
            continue
        vid, pid = ks.split("->", 1)
        vid = vid.strip()
        pid = pid.strip()
        try:
            draft[(vid, pid)] = float(v)
        except Exception:
            pass

    return {
        "T": T,
        "vessels": vessels,
        "ports": ports,
        "port_ids": list(ports.keys()),
        "supply_ids": supply_ids,
        "demand_ids": demand_ids,
        "travel": travel,
        "draft": draft,
    }


def get_travel_time(inst, i, j):
    if i == j:
        return 0
    tt = inst.get("travel", {})
    if (i, j) in tt:
        return max(1, int(tt[(i, j)]))
    if (j, i) in tt:
        return max(1, int(tt[(j, i)]))

    p1 = inst["ports"].get(i)
    p2 = inst["ports"].get(j)
    if p1 is not None and p2 is not None:
        try:
            dx = float(p1.get("x", 0.0)) - float(p2.get("x", 0.0))
            dy = float(p1.get("y", 0.0)) - float(p2.get("y", 0.0))
            dist = math.hypot(dx, dy)
            if dist > 0:
                return max(1, int(math.ceil(dist / 10.0)))
        except Exception:
            pass
    return 1


def get_draft_limit(inst, vessel, port):
    cap = float(vessel.get("capacity", 0.0))
    d = inst.get("draft", {}).get((vessel.get("id"), port), cap)
    try:
        d = float(d)
    except Exception:
        d = cap
    if d < 0:
        d = 0.0
    return d


def node_to_str(key):
    try:
        a, b = key
        if a in ("src", "snk"):
            return f"({a},{b})"
        try:
            b = int(b)
        except Exception:
            pass
        return f"({a},{b})"
    except Exception:
        return str(key)


def recompute_inventories(inst, operations):
    T = inst["T"]
    load = defaultdict(float)
    discharge = defaultdict(float)

    for vid, vops in (operations or {}).items():
        for op in vops or []:
            p = op.get("port")
            try:
                t = int(op.get("time", 0))
            except Exception:
                t = 0
            try:
                qty = float(op.get("quantity", 0.0))
            except Exception:
                qty = 0.0
            if t < 1 or t > T or p not in inst["ports"]:
                continue
            if op.get("type") == "load":
                load[(p, t)] += qty
            else:
                discharge[(p, t)] += -qty

    out = {}
    for p, info in inst["ports"].items():
        inv = float(info["initial"])
        lst = []
        for t in range(1, T + 1):
            rate = float(info["rate"][t - 1])
            if info["type"] == "supply":
                inv = inv + rate - load.get((p, t), 0.0)
            else:
                inv = inv - rate + discharge.get((p, t), 0.0)
            lst.append(float(inv))
        out[p] = lst
    return out


def add_movement_to_route(inst, vessel, route, p1, t1, p2, t2, flow):
    trans = 0.0
    dem = 0.0
    try:
        t1 = int(t1)
        t2 = int(t2)
    except Exception:
        return trans, dem
    if t2 < t1:
        t2 = t1
    flow = max(0.0, float(flow))

    if p1 == p2:
        for tt in range(t1, t2):
            cost = float(vessel.get("demurrage_cost", 0.0))
            route.append(
                {
                    "tail": node_to_str((p1, tt)),
                    "head": node_to_str((p1, tt + 1)),
                    "type": "demurrage",
                    "cost": clean_number(cost),
                    "product_flow": clean_number(flow),
                }
            )
            dem += cost
        return trans, dem

    if t2 <= t1:
        tau = 0
        arrive = t2
    else:
        tau = get_travel_time(inst, p1, p2)
        if t1 + tau > t2:
            tau = max(0, t2 - t1)
        arrive = t1 + tau

    cost = float(vessel.get("transport_cost", 0.0)) * float(tau)
    route.append(
        {
            "tail": node_to_str((p1, t1)),
            "head": node_to_str((p2, arrive)),
            "type": "travel",
            "cost": clean_number(cost),
            "product_flow": clean_number(flow),
        }
    )
    trans += cost

    if arrive < t2:
        for tt in range(arrive, t2):
            dcost = float(vessel.get("demurrage_cost", 0.0))
            route.append(
                {
                    "tail": node_to_str((p2, tt)),
                    "head": node_to_str((p2, tt + 1)),
                    "type": "demurrage",
                    "cost": clean_number(dcost),
                    "product_flow": clean_number(flow),
                }
            )
            dem += dcost

    return trans, dem


def construct_route_from_ops(inst, vessel, ops_list):
    route = []
    trans = 0.0
    dem = 0.0
    vid = vessel["id"]
    initial_load = float(vessel.get("initial_load", 0.0))

    ops = []
    for op in ops_list or []:
        try:
            qty = float(op.get("quantity", 0.0))
        except Exception:
            qty = 0.0
        if abs(qty) > 1e-7:
            op = dict(op)
            op["quantity"] = qty
            ops.append(op)
    ops.sort(key=lambda x: (int(x.get("time", 0)), str(x.get("port", ""))))

    if not ops:
        if vessel["op_start"] <= vessel["op_end"] and inst["port_ids"]:
            if initial_load > 1e-6 and inst["demand_ids"]:
                candidates = inst["demand_ids"]
            elif inst["supply_ids"]:
                candidates = inst["supply_ids"]
            else:
                candidates = inst["port_ids"]
            if not candidates:
                candidates = inst["port_ids"]
            p = max(
                candidates,
                key=lambda q: min(float(vessel["capacity"]), get_draft_limit(inst, vessel, q)),
            )
            t = int(vessel["op_start"])
            route.append(
                {
                    "tail": f"(src,{vid})",
                    "head": node_to_str((p, t)),
                    "type": "source",
                    "cost": 0.0,
                    "product_flow": clean_number(max(0.0, initial_load)),
                }
            )
            route.append(
                {
                    "tail": node_to_str((p, t)),
                    "head": f"(snk,{vid})",
                    "type": "sink",
                    "cost": 0.0,
                    "product_flow": clean_number(max(0.0, initial_load)),
                }
            )
        return route, trans, dem

    cargo = initial_load
    cur_p = None
    cur_t = None

    for idx, op in enumerate(ops):
        p = op.get("port")
        try:
            t = int(op.get("time", 0))
        except Exception:
            t = 1

        if idx == 0:
            route.append(
                {
                    "tail": f"(src,{vid})",
                    "head": node_to_str((p, t)),
                    "type": "source",
                    "cost": 0.0,
                    "product_flow": clean_number(max(0.0, cargo)),
                }
            )
        else:
            mt, md = add_movement_to_route(inst, vessel, route, cur_p, cur_t, p, t, cargo)
            trans += mt
            dem += md

        cur_p = p
        cur_t = t
        cargo += float(op.get("quantity", 0.0))
        if cargo < 0.0 and cargo > -1e-6:
            cargo = 0.0

    route.append(
        {
            "tail": node_to_str((cur_p, cur_t)),
            "head": f"(snk,{vid})",
            "type": "sink",
            "cost": 0.0,
            "product_flow": clean_number(max(0.0, cargo)),
        }
    )

    return route, trans, dem


def finalize_solution(inst, sol):
    if not isinstance(sol, dict):
        sol = {}

    sol.setdefault("vessel_routes", {})
    sol.setdefault("vessel_operations", {})
    sol.setdefault("port_inventories", {})
    sol.setdefault("cost_breakdown", {})
    sol.setdefault("objective_value", 0.0)

    # Normalize operations.
    new_ops = {v["id"]: [] for v in inst["vessels"]}
    for vid, vops in sol.get("vessel_operations", {}).items():
        if vid not in new_ops:
            continue
        for op in vops or []:
            p = op.get("port")
            if p not in inst["ports"]:
                continue
            try:
                t = int(op.get("time", 0))
            except Exception:
                continue
            if t < 1 or t > inst["T"]:
                continue
            try:
                qty = float(op.get("quantity", 0.0))
            except Exception:
                qty = 0.0
            typ = "load" if op.get("type") == "load" else "discharge"

            if typ == "load":
                if qty <= 1e-8:
                    continue
                new_qty = qty
            else:
                amt = -qty if qty < 0 else qty
                if amt <= 1e-8:
                    continue
                new_qty = -amt

            new_ops[vid].append(
                {
                    "port": p,
                    "time": t,
                    "quantity": clean_number(new_qty),
                    "type": typ,
                }
            )
    sol["vessel_operations"] = new_ops

    # Ensure routes exist for all vessels.
    new_routes = {}
    for v in inst["vessels"]:
        vid = v["id"]
        raw_route = sol.get("vessel_routes", {}).get(vid, None)
        if raw_route is None:
            raw_route, _, _ = construct_route_from_ops(inst, v, new_ops.get(vid, []))
        cleaned = []
        for arc in raw_route or []:
            if not isinstance(arc, dict):
                continue
            typ = arc.get("type", "travel")
            if typ not in ("travel", "demurrage", "source", "sink"):
                typ = "travel"
            try:
                cost = float(arc.get("cost", 0.0))
            except Exception:
                cost = 0.0
            try:
                flow = float(arc.get("product_flow", 0.0))
            except Exception:
                flow = 0.0
            cleaned.append(
                {
                    "tail": str(arc.get("tail", "")),
                    "head": str(arc.get("head", "")),
                    "type": typ,
                    "cost": clean_number(cost),
                    "product_flow": clean_number(max(0.0, flow)),
                }
            )
        new_routes[vid] = cleaned
    sol["vessel_routes"] = new_routes

    # Recompute inventories from operations.
    calc_inv = recompute_inventories(inst, new_ops)
    calc_ok = True
    for p, info in inst["ports"].items():
        vals = calc_inv.get(p, [])
        if len(vals) != inst["T"]:
            calc_ok = False
            break
        for idx, val in enumerate(vals):
            if val < -1e-5 or val > float(info["storage"][idx]) + 1e-5:
                calc_ok = False
                break
        if not calc_ok:
            break

    if calc_ok:
        sol["port_inventories"] = {
            p: [clean_number(x) for x in calc_inv[p]] for p in inst["port_ids"]
        }
    else:
        existing = sol.get("port_inventories", {})
        fixed = {}
        for p in inst["port_ids"]:
            vals = existing.get(p, [])
            if len(vals) != inst["T"]:
                vals = calc_inv.get(p, [0.0] * inst["T"])
                if len(vals) != inst["T"]:
                    vals = [0.0] * inst["T"]
            fixed[p] = [clean_number(x) for x in vals[: inst["T"]]]
        sol["port_inventories"] = fixed

    # Recompute costs from final routes and operations.
    transport_cost = 0.0
    demurrage_cost = 0.0
    procurement_cost = 0.0
    supply_revenue = 0.0

    for vid, route in sol["vessel_routes"].items():
        for arc in route:
            typ = arc.get("type")
            try:
                cost = float(arc.get("cost", 0.0))
            except Exception:
                cost = 0.0
            if typ == "travel":
                transport_cost += cost
            elif typ == "demurrage":
                demurrage_cost += cost

    for vid, vops in sol["vessel_operations"].items():
        for op in vops:
            info = inst["ports"].get(op.get("port"))
            if info is None:
                continue
            try:
                qty = float(op.get("quantity", 0.0))
            except Exception:
                qty = 0.0
            if op.get("type") == "load":
                procurement_cost += float(info.get("unit_cost", 0.0)) * qty
            else:
                supply_revenue += float(info.get("unit_rev", 0.0)) * (-qty)

    total = transport_cost + demurrage_cost + procurement_cost - supply_revenue
    sol["cost_breakdown"] = {
        "transport_cost": clean_number(transport_cost),
        "demurrage_cost": clean_number(demurrage_cost),
        "procurement_cost": clean_number(procurement_cost),
        "supply_revenue": clean_number(supply_revenue),
    }
    sol["objective_value"] = clean_number(total)
    return sol


def validate_solution(inst, sol):
    if not isinstance(sol, dict):
        return False
    T = inst["T"]
    tol = 1e-3

    ops = sol.get("vessel_operations", {})
    for vid, vops in ops.items():
        for op in vops or []:
            p = op.get("port")
            info = inst["ports"].get(p)
            if info is None:
                return False
            try:
                t = int(op.get("time", 0))
            except Exception:
                return False
            if t < 1 or t > T:
                return False
            try:
                qty = float(op.get("quantity", 0.0))
            except Exception:
                return False

            if op.get("type") == "load":
                amt = qty
                if amt < -tol:
                    return False
                if info["type"] != "supply":
                    return False
            else:
                amt = -qty
                if amt < -tol:
                    return False
                if info["type"] != "demand":
                    return False

            if amt > float(info["F_max"]) + tol:
                return False
            if amt > tol and amt < float(info["F_min"]) - tol:
                return False

    calc = recompute_inventories(inst, ops)
    inv_out = sol.get("port_inventories", {})
    for p, info in inst["ports"].items():
        vals = inv_out.get(p, [])
        if len(vals) != T:
            return False
        cvals = calc.get(p, [])
        for idx in range(T):
            try:
                v = float(vals[idx])
                cv = float(cvals[idx])
            except Exception:
                return False
            if abs(v - cv) > 1e-2:
                return False
            if v < -tol or v > float(info["storage"][idx]) + tol:
                return False

    cb = sol.get("cost_breakdown", {})
    try:
        total = (
            float(cb.get("transport_cost", 0.0))
            + float(cb.get("demurrage_cost", 0.0))
            + float(cb.get("procurement_cost", 0.0))
            - float(cb.get("supply_revenue", 0.0))
        )
        obj = float(sol.get("objective_value", total))
    except Exception:
        return False
    if abs(total - obj) > 1e-2:
        return False

    return True


def build_default_solution(inst):
    ops = {v["id"]: [] for v in inst["vessels"]}
    routes = {}
    for v in inst["vessels"]:
        route, _, _ = construct_route_from_ops(inst, v, [])
        routes[v["id"]] = route

    sol = {
        "vessel_routes": routes,
        "vessel_operations": ops,
        "port_inventories": {p: [] for p in inst["port_ids"]},
        "cost_breakdown": {
            "transport_cost": 0.0,
            "demurrage_cost": 0.0,
            "procurement_cost": 0.0,
            "supply_revenue": 0.0,
        },
        "objective_value": 0.0,
    }
    return finalize_solution(inst, sol)


def build_heuristic_solution(inst):
    T = inst["T"]
    vessels = inst["vessels"]
    ops = {v["id"]: [] for v in vessels}

    if T <= 0 or not inst["port_ids"]:
        sol = {
            "vessel_routes": {v["id"]: [] for v in vessels},
            "vessel_operations": ops,
            "port_inventories": {p: [] for p in inst["port_ids"]},
            "cost_breakdown": {
                "transport_cost": 0.0,
                "demurrage_cost": 0.0,
                "procurement_cost": 0.0,
                "supply_revenue": 0.0,
            },
            "objective_value": 0.0,
        }
        sol = finalize_solution(inst, sol)
        return sol, validate_solution(inst, sol)

    # Base inventory projection with no operations.
    h_inv = {}
    for p, info in inst["ports"].items():
        arr = [float(info["initial"])] + [0.0] * T
        for t in range(1, T + 1):
            rate = float(info["rate"][t - 1])
            if info["type"] == "supply":
                arr[t] = arr[t - 1] + rate
            else:
                arr[t] = arr[t - 1] - rate
        h_inv[p] = arr

    def demand_can_accept(v, from_p, t_load, qty, end):
        horizon = min(int(end), T)
        for q in inst["demand_ids"]:
            info = inst["ports"][q]
            fmax = float(info["F_max"])
            fmin = float(info["F_min"])
            if fmax <= 1e-9:
                continue
            if fmin > 1e-9 and qty + 1e-9 < fmin:
                continue
            tau = get_travel_time(inst, from_p, q)
            earliest = int(t_load) + tau
            if earliest > horizon:
                continue
            eff = min(float(v["capacity"]), get_draft_limit(inst, v, q))
            if qty > eff + 1e-6:
                continue

            for t in range(earliest, horizon + 1):
                slack = 1e100
                for s in range(t, T + 1):
                    val = float(info["storage"][s - 1]) - h_inv[q][s]
                    if val < slack:
                        slack = val
                min_req = fmin if fmin > 1e-9 else 1e-6
                if slack + 1e-6 >= min_req:
                    return True
        return False

    def find_discharge(v, cur_p, cur_t, end, cargo):
        best = None
        horizon = min(int(end), T)
        for q in inst["demand_ids"]:
            info = inst["ports"][q]
            fmax = float(info["F_max"])
            fmin = float(info["F_min"])
            if fmax <= 1e-9:
                continue
            if cargo + 1e-9 < fmin:
                continue

            if cur_p is None:
                earliest = int(cur_t)
            elif q == cur_p:
                earliest = int(cur_t) + 1
            else:
                tau = get_travel_time(inst, cur_p, q)
                eff = min(float(v["capacity"]), get_draft_limit(inst, v, q))
                if cargo > eff + 1e-6:
                    continue
                earliest = int(cur_t) + tau

            if earliest > horizon:
                continue

            min_inv = 0.0
            if earliest <= horizon:
                min_inv = min(h_inv[q][earliest : horizon + 1])
            first_neg = horizon + 1
            for s in range(earliest, horizon + 1):
                if h_inv[q][s] < -1e-6:
                    first_neg = s
                    break

            for t in range(earliest, horizon + 1):
                slack = 1e100
                for s in range(t, T + 1):
                    val = float(info["storage"][s - 1]) - h_inv[q][s]
                    if val < slack:
                        slack = val
                if slack < -1e-6:
                    continue
                max_qty = min(cargo, fmax, max(0.0, slack))
                if max_qty + 1e-6 < fmin:
                    continue
                qty = max_qty
                if qty < fmin:
                    qty = fmin
                if qty <= 1e-6:
                    continue

                late_penalty = max(0, t - first_neg)
                score = (late_penalty, min_inv, t, -float(info.get("unit_rev", 0.0)))
                if best is None or score < best[0]:
                    best = (score, q, t, qty)
                break

        if best is None:
            return None
        return best[1], best[2], best[3]

    def find_load(v, cur_p, cur_t, end, first_load_done):
        horizon = min(int(end), T)

        def search(limit_window):
            best_local = None
            for p in inst["supply_ids"]:
                info = inst["ports"][p]
                fmax = float(info["F_max"])
                fmin = float(info["F_min"])
                if fmax <= 1e-9:
                    continue

                if cur_p is None:
                    earliest = int(cur_t)
                elif p == cur_p:
                    earliest = int(cur_t) + 1
                else:
                    earliest = int(cur_t) + get_travel_time(inst, cur_p, p)

                if earliest > horizon:
                    continue

                max_t = horizon
                if limit_window:
                    lwe = v.get("loading_window_end")
                    if lwe is not None:
                        try:
                            lwe = int(lwe)
                        except Exception:
                            lwe = horizon
                        if lwe < max_t:
                            max_t = lwe

                if earliest > max_t:
                    continue

                for t in range(earliest, max_t + 1):
                    allowed = 1e100
                    for s in range(t, T + 1):
                        if h_inv[p][s] < allowed:
                            allowed = h_inv[p][s]
                    if allowed + 1e-6 < fmin:
                        continue

                    qty = min(float(v["capacity"]), fmax, max(0.0, allowed))
                    if qty + 1e-6 < fmin or qty <= 1e-6:
                        continue
                    if not demand_can_accept(v, p, t, qty, end):
                        continue

                    max_over = -1e100
                    max_inv = -1e100
                    for s in range(t, horizon + 1):
                        over = h_inv[p][s] - float(info["storage"][s - 1])
                        if over > max_over:
                            max_over = over
                        if h_inv[p][s] > max_inv:
                            max_inv = h_inv[p][s]
                    if max_over == -1e100:
                        max_over = 0.0
                        max_inv = 0.0

                    score = (-max_over, -max_inv, t, float(info.get("unit_cost", 0.0)))
                    if best_local is None or score < best_local[0]:
                        best_local = (score, p, t, qty)
                    break

            if best_local is None:
                return None
            return best_local[1], best_local[2], best_local[3]

        if not first_load_done:
            res = search(True)
            if res is not None:
                return res
        return search(False)

    for v in vessels:
        start = int(v["op_start"])
        end = int(v["op_end"])
        if start > end or start > T or end < 1:
            continue
        start = max(1, start)
        end = min(T, end)

        cur_t = start
        cur_p = None
        cargo = float(v["initial_load"])
        first_load_done = False
        iterations = 0
        max_iter = 4 * T + 20

        while cur_t <= end and iterations < max_iter:
            iterations += 1

            if cargo > 1e-6:
                act = find_discharge(v, cur_p, cur_t, end, cargo)
                if act is None:
                    break
                q, t, qty = act
                t = int(t)
                qty = float(qty)
                for s in range(t, T + 1):
                    h_inv[q][s] += qty
                ops[v["id"]].append(
                    {"port": q, "time": t, "quantity": -qty, "type": "discharge"}
                )
                cargo -= qty
                if cargo < 1e-6:
                    cargo = 0.0
                cur_p = q
                cur_t = t
            else:
                act = find_load(v, cur_p, cur_t, end, first_load_done)
                if act is None:
                    break
                p, t, qty = act
                t = int(t)
                qty = float(qty)
                for s in range(t, T + 1):
                    h_inv[p][s] -= qty
                ops[v["id"]].append(
                    {"port": p, "time": t, "quantity": qty, "type": "load"}
                )
                cargo += qty
                cur_p = p
                cur_t = t
                first_load_done = True

    routes = {}
    for v in vessels:
        route, _, _ = construct_route_from_ops(inst, v, ops[v["id"]])
        routes[v["id"]] = route

    port_inventories = {
        p: [float(h_inv[p][t]) for t in range(1, T + 1)] for p in inst["port_ids"]
    }

    sol = {
        "vessel_routes": routes,
        "vessel_operations": ops,
        "port_inventories": port_inventories,
        "cost_breakdown": {
            "transport_cost": 0.0,
            "demurrage_cost": 0.0,
            "procurement_cost": 0.0,
            "supply_revenue": 0.0,
        },
        "objective_value": 0.0,
    }
    sol = finalize_solution(inst, sol)
    return sol, validate_solution(inst, sol)


def estimate_mip_size(inst):
    P = len(inst["port_ids"])
    S = len(inst["supply_ids"])
    D = len(inst["demand_ids"])
    full_travel = 0
    total_times = 0

    for v in inst["vessels"]:
        tv = max(0, int(v["op_end"]) - int(v["op_start"]) + 1)
        total_times += tv
        full_travel += tv * P * max(0, P - 1)

    restrict = full_travel > 100000
    if restrict:
        allowed_pairs = 2 * S * D if S and D else P * max(0, P - 1)
    else:
        allowed_pairs = P * max(0, P - 1)

    est = 0
    for v in inst["vessels"]:
        tv = max(0, int(v["op_end"]) - int(v["op_start"]) + 1)
        est += tv * (P + P + P + allowed_pairs)
    est += total_times * P * 2
    return est, restrict


def build_mip(inst, time_left):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return None, None

    if time_left <= 1.0:
        return None, None

    T = inst["T"]
    port_ids = inst["port_ids"]
    if T <= 0 or not port_ids or not inst["vessels"]:
        return None, None

    try:
        model = gp.Model("mirp")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = max(1.0, float(time_left) - 1.0)

        all_pairs = []
        restricted_pairs = []
        S = inst["supply_ids"]
        D = inst["demand_ids"]
        ports = inst["ports"]

        for i in port_ids:
            for j in port_ids:
                if i == j:
                    continue
                tau = get_travel_time(inst, i, j)
                all_pairs.append((i, j, tau))
                if not S or not D:
                    restricted_pairs.append((i, j, tau))
                else:
                    ti = ports[i]["type"]
                    tj = ports[j]["type"]
                    if (ti == "supply" and tj == "demand") or (
                        ti == "demand" and tj == "supply"
                    ):
                        restricted_pairs.append((i, j, tau))

        full_est = 0
        for v in inst["vessels"]:
            tv = max(0, int(v["op_end"]) - int(v["op_start"]) + 1)
            full_est += tv * len(port_ids) * max(0, len(port_ids) - 1)
        restrict = full_est > 100000
        pairs = restricted_pairs if restrict else all_pairs

        obj = gp.LinExpr()
        arcs_by_v = {}
        load_vars = {}
        discharge_vars = {}
        inv_vars = {}
        load_by_pt = defaultdict(list)
        discharge_by_pt = defaultdict(list)

        for v in inst["vessels"]:
            vid = v["id"]
            cap = float(v["capacity"])
            init = float(v["initial_load"])
            start = int(v["op_start"])
            end = int(v["op_end"])

            arcs = []
            in_arcs = defaultdict(list)
            out_arcs = defaultdict(list)

            if start > end or start > T or end < 1:
                arcs_by_v[vid] = []
                continue

            times = list(range(start, end + 1))
            times_set = set(times)

            if restrict:
                if init > 1e-6 and D:
                    cand = D
                elif S:
                    cand = S
                else:
                    cand = port_ids
            else:
                cand = port_ids
            if not cand:
                cand = port_ids

            feas = []
            for p in cand:
                eff = min(cap, get_draft_limit(inst, v, p))
                if init <= eff + 1e-6:
                    feas.append(p)
            if not feas:
                feas = list(cand)
            if not feas:
                feas = list(port_ids)

            # Source arcs.
            for t in times:
                for p in feas:
                    eff = min(cap, get_draft_limit(inst, v, p))
                    arc_cap = max(init, eff, 0.0)
                    x = model.addVar(vtype=GRB.BINARY)
                    f = model.addVar(lb=0.0, ub=arc_cap)
                    model.addConstr(f == init * x)
                    if arc_cap > 0:
                        model.addConstr(f <= arc_cap * x)
                    arc = {
                        "tail_key": ("src", vid),
                        "head_key": (p, int(t)),
                        "tail_str": f"(src,{vid})",
                        "head_str": f"({p},{int(t)})",
                        "type": "source",
                        "cost": 0.0,
                        "cap": arc_cap,
                        "x": x,
                        "f": f,
                    }
                    arcs.append(arc)
                    in_arcs[(p, int(t))].append(arc)

            # Demurrage arcs.
            for idx, t in enumerate(times[:-1]):
                t2 = t + 1
                for p in port_ids:
                    eff = min(cap, get_draft_limit(inst, v, p))
                    arc_cap = max(0.0, eff)
                    x = model.addVar(vtype=GRB.BINARY)
                    f = model.addVar(lb=0.0, ub=arc_cap)
                    if arc_cap > 0:
                        model.addConstr(f <= arc_cap * x)
                    else:
                        model.addConstr(f == 0.0)
                    cost = float(v["demurrage_cost"])
                    obj += cost * x
                    arc = {
                        "tail_key": (p, int(t)),
                        "head_key": (p, int(t2)),
                        "tail_str": f"({p},{int(t)})",
                        "head_str": f"({p},{int(t2)})",
                        "type": "demurrage",
                        "cost": cost,
                        "cap": arc_cap,
                        "x": x,
                        "f": f,
                    }
                    arcs.append(arc)
                    out_arcs[(p, int(t))].append(arc)
                    in_arcs[(p, int(t2))].append(arc)

            # Travel arcs.
            for t in times:
                for i, j, tau in pairs:
                    t2 = t + int(tau)
                    if t2 not in times_set:
                        continue
                    eff_i = min(cap, get_draft_limit(inst, v, i))
                    eff_j = min(cap, get_draft_limit(inst, v, j))
                    arc_cap = max(0.0, min(eff_i, eff_j))
                    x = model.addVar(vtype=GRB.BINARY)
                    f = model.addVar(lb=0.0, ub=arc_cap)
                    if arc_cap > 0:
                        model.addConstr(f <= arc_cap * x)
                    else:
                        model.addConstr(f == 0.0)
                    cost = float(v["transport_cost"]) * float(tau)
                    obj += cost * x
                    arc = {
                        "tail_key": (i, int(t)),
                        "head_key": (j, int(t2)),
                        "tail_str": f"({i},{int(t)})",
                        "head_str": f"({j},{int(t2)})",
                        "type": "travel",
                        "cost": cost,
                        "cap": arc_cap,
                        "x": x,
                        "f": f,
                    }
                    arcs.append(arc)
                    out_arcs[(i, int(t))].append(arc)
                    in_arcs[(j, int(t2))].append(arc)

            # Sink arcs.
            for t in times:
                for p in port_ids:
                    eff = min(cap, get_draft_limit(inst, v, p))
                    arc_cap = max(0.0, eff)
                    x = model.addVar(vtype=GRB.BINARY)
                    f = model.addVar(lb=0.0, ub=arc_cap)
                    if arc_cap > 0:
                        model.addConstr(f <= arc_cap * x)
                    else:
                        model.addConstr(f == 0.0)
                    arc = {
                        "tail_key": (p, int(t)),
                        "head_key": ("snk", vid),
                        "tail_str": f"({p},{int(t)})",
                        "head_str": f"(snk,{vid})",
                        "type": "sink",
                        "cost": 0.0,
                        "cap": arc_cap,
                        "x": x,
                        "f": f,
                    }
                    arcs.append(arc)
                    out_arcs[(p, int(t))].append(arc)

            arcs_by_v[vid] = arcs
            source_arcs = [a for a in arcs if a["type"] == "source"]
            sink_arcs = [a for a in arcs if a["type"] == "sink"]

            if source_arcs and sink_arcs:
                model.addConstr(gp.quicksum(a["x"] for a in source_arcs) == 1.0)
                model.addConstr(gp.quicksum(a["x"] for a in sink_arcs) == 1.0)

                nodes = set(in_arcs.keys()) | set(out_arcs.keys())
                for node in nodes:
                    ins = in_arcs.get(node, [])
                    outs = out_arcs.get(node, [])
                    if ins or outs:
                        model.addConstr(
                            gp.quicksum(a["x"] for a in ins)
                            == gp.quicksum(a["x"] for a in outs)
                        )

                for node in nodes:
                    p, t = node
                    ins = in_arcs.get(node, [])
                    outs = out_arcs.get(node, [])
                    if not ins and not outs:
                        continue

                    visit = gp.quicksum(a["x"] for a in ins)
                    inflow = gp.quicksum(a["f"] for a in ins)
                    outflow = gp.quicksum(a["f"] for a in outs)
                    info = ports.get(p)
                    if info is None:
                        model.addConstr(inflow == outflow)
                        continue

                    fmax = float(info["F_max"])
                    fmin = float(info["F_min"])

                    if info["type"] == "supply" and fmax > 1e-9:
                        L = model.addVar(lb=0.0, ub=fmax)
                        load_vars[(vid, p, int(t))] = (L, None)
                        load_by_pt[(p, int(t))].append(L)

                        if fmin > 1e-9:
                            z = model.addVar(vtype=GRB.BINARY)
                            model.addConstr(L <= fmax * z)
                            model.addConstr(L >= fmin * z)
                            model.addConstr(z <= visit)
                            load_vars[(vid, p, int(t))] = (L, z)
                        else:
                            model.addConstr(L <= fmax * visit)

                        obj += float(info["unit_cost"]) * L
                        model.addConstr(inflow + L == outflow)

                    elif info["type"] == "demand" and fmax > 1e-9:
                        D = model.addVar(lb=0.0, ub=fmax)
                        discharge_vars[(vid, p, int(t))] = (D, None)
                        discharge_by_pt[(p, int(t))].append(D)

                        if fmin > 1e-9:
                            z = model.addVar(vtype=GRB.BINARY)
                            model.addConstr(D <= fmax * z)
                            model.addConstr(D >= fmin * z)
                            model.addConstr(z <= visit)
                            discharge_vars[(vid, p, int(t))] = (D, z)
                        else:
                            model.addConstr(D <= fmax * visit)

                        obj -= float(info["unit_rev"]) * D
                        model.addConstr(inflow - D == outflow)
                    else:
                        model.addConstr(inflow == outflow)
            else:
                for a in arcs:
                    model.addConstr(a["x"] == 0.0)

        # Inventory constraints.
        for p, info in ports.items():
            prev = float(info["initial"])
            for t in range(1, T + 1):
                cap_ub = max(0.0, float(info["storage"][t - 1]))
                inv = model.addVar(lb=0.0, ub=cap_ub)
                inv_vars[(p, t)] = inv

                if info["type"] == "supply":
                    loads = load_by_pt.get((p, t), [])
                    model.addConstr(
                        inv == prev + float(info["rate"][t - 1]) - gp.quicksum(loads)
                    )
                else:
                    disch = discharge_by_pt.get((p, t), [])
                    model.addConstr(
                        inv == prev - float(info["rate"][t - 1]) + gp.quicksum(disch)
                    )
                prev = inv

        model.setObjective(obj, GRB.MINIMIZE)
        model.update()

        structs = {
            "arcs": arcs_by_v,
            "load_vars": load_vars,
            "discharge_vars": discharge_vars,
            "inv_vars": inv_vars,
        }
        return model, structs
    except Exception:
        return None, None


def extract_mip_solution(inst, model, structs):
    arcs_by_v = structs.get("arcs", {})
    load_vars = structs.get("load_vars", {})
    discharge_vars = structs.get("discharge_vars", {})
    inv_vars = structs.get("inv_vars", {})
    T = inst["T"]

    ops = {v["id"]: [] for v in inst["vessels"]}

    for (vid, p, t), (var, z) in load_vars.items():
        try:
            val = float(var.X)
        except Exception:
            val = 0.0
        if val > 1e-6:
            ops.setdefault(vid, []).append(
                {"port": p, "time": int(t), "quantity": val, "type": "load"}
            )

    for (vid, p, t), (var, z) in discharge_vars.items():
        try:
            val = float(var.X)
        except Exception:
            val = 0.0
        if val > 1e-6:
            ops.setdefault(vid, []).append(
                {"port": p, "time": int(t), "quantity": -val, "type": "discharge"}
            )

    for vid in ops:
        ops[vid].sort(key=lambda x: (int(x.get("time", 0)), str(x.get("port", ""))))

    inv_out = {p: [0.0] * T for p in inst["port_ids"]}
    for (p, t), var in inv_vars.items():
        try:
            val = float(var.X)
        except Exception:
            val = 0.0
        if 1 <= int(t) <= T:
            inv_out[p][int(t) - 1] = val

    routes = {}
    for v in inst["vessels"]:
        vid = v["id"]
        arc_list = arcs_by_v.get(vid, [])
        selected = []
        for a in arc_list:
            try:
                xv = float(a["x"].X)
            except Exception:
                xv = 0.0
            if xv > 0.5:
                selected.append(a)

        out_map = {}
        for a in selected:
            out_map[a["tail_key"]] = a

        route = []
        cur = ("src", vid)
        seen = 0
        complete = False
        while cur in out_map and seen <= len(selected) + 1:
            a = out_map[cur]
            try:
                flow = float(a["f"].X)
            except Exception:
                flow = 0.0
            route.append(
                {
                    "tail": a["tail_str"],
                    "head": a["head_str"],
                    "type": a["type"],
                    "cost": float(a["cost"]),
                    "product_flow": max(0.0, flow),
                }
            )
            cur = a["head_key"]
            seen += 1
            if cur == ("snk", vid):
                complete = True
                break

        if not route or not complete:
            route, _, _ = construct_route_from_ops(inst, v, ops.get(vid, []))

        routes[vid] = route

    sol = {
        "vessel_routes": routes,
        "vessel_operations": ops,
        "port_inventories": inv_out,
        "cost_breakdown": {
            "transport_cost": 0.0,
            "demurrage_cost": 0.0,
            "procurement_cost": 0.0,
            "supply_revenue": 0.0,
        },
        "objective_value": 0.0,
    }
    return finalize_solution(inst, sol)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-path", default="instance.json")
    parser.add_argument("--solution-path", default="solution.json")
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--log-path", default="log.jsonl")
    args, _ = parser.parse_known_args()

    start_time = time.time()
    inst = parse_instance(args.instance_path)

    logger = None
    if args.log_path:
        try:
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            try:
                logger = SolutionLogger(args.log_path)
            except Exception:
                logger = SolutionLogger()

    heuristic_sol, heuristic_feasible = build_heuristic_solution(inst)
    best_sol = None
    fallback_sol = heuristic_sol

    if heuristic_feasible:
        best_sol = heuristic_sol
        if logger is not None:
            try:
                logger.log_solution(heuristic_sol["objective_value"], heuristic_sol)
            except Exception:
                pass
    else:
        default_sol = build_default_solution(inst)
        fallback_sol = default_sol
        if validate_solution(inst, default_sol):
            best_sol = default_sol
            if logger is not None:
                try:
                    logger.log_solution(default_sol["objective_value"], default_sol)
                except Exception:
                    pass

    remaining = float(args.time_limit) - (time.time() - start_time)
    if remaining > 5.0:
        est, _ = estimate_mip_size(inst)
        max_est = 600000 if args.time_limit > 300 else 300000
        if est <= max_est:
            try:
                model, structs = build_mip(inst, remaining)
                if model is not None:
                    try:
                        from gurobipy import GRB

                        if logger is not None:

                            def cb(model, where):
                                if where == GRB.Callback.MIPSOL:
                                    try:
                                        obj = float(
                                            model.cbGet(GRB.Callback.MIPSOL_OBJ)
                                        )
                                        logger.log(obj)
                                    except Exception:
                                        pass

                            model.optimize(cb)
                        else:
                            model.optimize()
                    except Exception:
                        model.optimize()

                    if model.SolCount > 0:
                        mip_sol = extract_mip_solution(inst, model, structs)
                        mip_feasible = validate_solution(inst, mip_sol)
                        if mip_feasible:
                            if logger is not None:
                                try:
                                    logger.log_solution(
                                        mip_sol["objective_value"], mip_sol
                                    )
                                except Exception:
                                    pass
                            if (
                                best_sol is None
                                or mip_sol["objective_value"]
                                < best_sol["objective_value"]
                                - 1e-6
                            ):
                                best_sol = mip_sol
                        else:
                            if best_sol is None:
                                best_sol = mip_sol
            except Exception:
                pass

    if best_sol is None:
        best_sol = fallback_sol if fallback_sol is not None else build_default_solution(inst)

    best_sol = finalize_solution(inst, best_sol)

    if logger is not None:
        try:
            if validate_solution(inst, best_sol):
                logger.log_solution(best_sol["objective_value"], best_sol)
            else:
                logger.log(best_sol.get("objective_value", 0.0))
        except Exception:
            pass

    with open(args.solution_path, "w") as f:
        json.dump(best_sol, f, indent=2)


if __name__ == "__main__":
    main()