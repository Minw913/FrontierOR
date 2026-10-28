import argparse
import json
import math
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def parse_instance(path):
    with open(path, "r") as f:
        return json.load(f)


def build_and_solve(instance, time_limit, logger=None):
    T = int(instance["planning_horizon"])
    vessels = instance["vessels"]
    supply_ports = instance["supply_ports"]
    demand_ports = instance["demand_ports"]
    travel_times = instance["travel_times"]
    draft_limits = instance["draft_limits"]

    supply_ids = [p["port_id"] for p in supply_ports]
    demand_ids = [p["port_id"] for p in demand_ports]
    all_ports = supply_ids + demand_ids

    supply_by_id = {p["port_id"]: p for p in supply_ports}
    demand_by_id = {p["port_id"]: p for p in demand_ports}

    # Travel time lookup
    def get_tt(a, b):
        key = f"{a}->{b}"
        return travel_times.get(key, None)

    # Draft lookup
    def get_draft_cap(v_id, port_id, vessel_cap):
        key = f"{v_id}->{port_id}"
        dl = draft_limits.get(key, vessel_cap)
        return float(min(vessel_cap, dl))

    model = gp.Model("maritime_inventory_routing")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = int(time_limit)
    model.Params.OutputFlag = 0

    # Data containers
    arcs = defaultdict(list)  # arcs[v_id] = list of arc dicts
    in_arcs = defaultdict(lambda: defaultdict(list))   # in_arcs[v_id][(p,t)] -> [a_idx]
    out_arcs = defaultdict(lambda: defaultdict(list))  # out_arcs[v_id][(p,t)] -> [a_idx]
    source_arc_idx = defaultdict(list)
    sink_arc_idx = defaultdict(list)

    # Build time-expanded network per vessel
    for v in vessels:
        v_id = v["id"]
        cap = float(v["capacity"])
        a_start = int(clamp(v["availability_start"], 0, T))
        a_end = int(clamp(v["availability_end"], 0, T))
        if a_end < a_start:
            a_end = a_start

        tc = float(v["transport_cost_per_day"])
        dc = float(v["demurrage_cost_per_day"])

        # Source arcs: virtual source -> (p, a_start)
        for p in all_ports:
            arc = {
                "type": "source",
                "tail": ("src", v_id),
                "head": (p, a_start),
                "tail_port": None,
                "tail_t": None,
                "head_port": p,
                "head_t": a_start,
                "cost": 0.0,
                "is_source": True,
                "is_sink": False,
                "is_wait": False,
                "is_travel": False,
            }
            a_idx = len(arcs[v_id])
            arcs[v_id].append(arc)
            source_arc_idx[v_id].append(a_idx)
            in_arcs[v_id][(p, a_start)].append(a_idx)

        # Wait arcs
        for t in range(a_start, a_end):
            for p in all_ports:
                arc = {
                    "type": "demurrage",
                    "tail": (p, t),
                    "head": (p, t + 1),
                    "tail_port": p,
                    "tail_t": t,
                    "head_port": p,
                    "head_t": t + 1,
                    "cost": dc,
                    "is_source": False,
                    "is_sink": False,
                    "is_wait": True,
                    "is_travel": False,
                }
                a_idx = len(arcs[v_id])
                arcs[v_id].append(arc)
                out_arcs[v_id][(p, t)].append(a_idx)
                in_arcs[v_id][(p, t + 1)].append(a_idx)

        # Travel arcs
        for t in range(a_start, a_end + 1):
            for p in all_ports:
                for q in all_ports:
                    if p == q:
                        continue
                    tau = get_tt(p, q)
                    if tau is None:
                        continue
                    tau = int(tau)
                    if tau <= 0:
                        continue
                    t2 = t + tau
                    if t2 <= a_end:
                        arc = {
                            "type": "travel",
                            "tail": (p, t),
                            "head": (q, t2),
                            "tail_port": p,
                            "tail_t": t,
                            "head_port": q,
                            "head_t": t2,
                            "cost": tc * tau,
                            "is_source": False,
                            "is_sink": False,
                            "is_wait": False,
                            "is_travel": True,
                        }
                        a_idx = len(arcs[v_id])
                        arcs[v_id].append(arc)
                        out_arcs[v_id][(p, t)].append(a_idx)
                        in_arcs[v_id][(q, t2)].append(a_idx)

        # Sink arcs: (p,t) -> virtual sink
        for t in range(a_start, a_end + 1):
            for p in all_ports:
                arc = {
                    "type": "sink",
                    "tail": (p, t),
                    "head": ("snk", v_id),
                    "tail_port": p,
                    "tail_t": t,
                    "head_port": None,
                    "head_t": None,
                    "cost": 0.0,
                    "is_source": False,
                    "is_sink": True,
                    "is_wait": False,
                    "is_travel": False,
                }
                a_idx = len(arcs[v_id])
                arcs[v_id].append(arc)
                sink_arc_idx[v_id].append(a_idx)
                out_arcs[v_id][(p, t)].append(a_idx)

    # Decision variables
    x = {}  # arc usage
    f = {}  # onboard cargo on arc
    y = {}  # visit indicator at (port,time)
    z = {}  # operation on/off
    load = {}
    discharge = {}
    inv = {}

    # Arc variables and bounds
    for v in vessels:
        v_id = v["id"]
        cap = float(v["capacity"])
        init_load = float(v["initial_load"])

        for a_idx, a in enumerate(arcs[v_id]):
            x[(v_id, a_idx)] = model.addVar(vtype=GRB.BINARY, name=f"x_{v_id}_{a_idx}")
            f[(v_id, a_idx)] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"f_{v_id}_{a_idx}")

            # Cargo upper bound by entering-port draft/cap when head is a real port
            if a["is_sink"]:
                ub = cap
            else:
                head_port = a["head_port"]
                ub = get_draft_cap(v_id, head_port, cap)
            model.addConstr(f[(v_id, a_idx)] <= ub * x[(v_id, a_idx)],
                            name=f"fub_{v_id}_{a_idx}")

            # Initial load fixed on source arc
            if a["is_source"]:
                model.addConstr(f[(v_id, a_idx)] == init_load * x[(v_id, a_idx)],
                                name=f"finit_{v_id}_{a_idx}")

    # Routing flow constraints and visit indicators
    for v in vessels:
        v_id = v["id"]
        a_start = int(clamp(v["availability_start"], 0, T))
        a_end = int(clamp(v["availability_end"], 0, T))
        if a_end < a_start:
            a_end = a_start

        # Exactly one source and one sink arc
        model.addConstr(gp.quicksum(x[(v_id, a)] for a in source_arc_idx[v_id]) == 1,
                        name=f"src_one_{v_id}")
        model.addConstr(gp.quicksum(x[(v_id, a)] for a in sink_arc_idx[v_id]) == 1,
                        name=f"snk_one_{v_id}")

        # Node conservation and y definition
        for t in range(a_start, a_end + 1):
            for p in all_ports:
                ins = in_arcs[v_id][(p, t)]
                outs = out_arcs[v_id][(p, t)]
                inflow = gp.quicksum(x[(v_id, a)] for a in ins) if ins else 0
                outflow = gp.quicksum(x[(v_id, a)] for a in outs) if outs else 0
                model.addConstr(inflow == outflow, name=f"route_bal_{v_id}_{p}_{t}")

                y[(v_id, p, t)] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY,
                                               name=f"y_{v_id}_{p}_{t}")
                model.addConstr(y[(v_id, p, t)] == inflow, name=f"ydef_{v_id}_{p}_{t}")

    # Operations variables and constraints
    for v in vessels:
        v_id = v["id"]
        a_start = int(clamp(v["availability_start"], 0, T))
        a_end = int(clamp(v["availability_end"], 0, T))
        if a_end < a_start:
            a_end = a_start

        # Supply loads
        for p in supply_ports:
            pid = p["port_id"]
            fmin = float(p["F_min"])
            fmax = float(p["F_max"])
            for t in range(1, T + 1):
                if not (a_start <= t <= a_end):
                    continue
                z[(v_id, pid, t, "load")] = model.addVar(vtype=GRB.BINARY,
                                                         name=f"zL_{v_id}_{pid}_{t}")
                load[(v_id, pid, t)] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                                    name=f"L_{v_id}_{pid}_{t}")
                model.addConstr(load[(v_id, pid, t)] <= fmax * z[(v_id, pid, t, "load")],
                                name=f"Lmax_{v_id}_{pid}_{t}")
                model.addConstr(load[(v_id, pid, t)] >= fmin * z[(v_id, pid, t, "load")],
                                name=f"Lmin_{v_id}_{pid}_{t}")
                model.addConstr(z[(v_id, pid, t, "load")] <= y[(v_id, pid, t)],
                                name=f"Ly_{v_id}_{pid}_{t}")

        # Demand discharges
        for p in demand_ports:
            pid = p["port_id"]
            fmin = float(p["F_min"])
            fmax = float(p["F_max"])
            for t in range(1, T + 1):
                if not (a_start <= t <= a_end):
                    continue
                z[(v_id, pid, t, "discharge")] = model.addVar(vtype=GRB.BINARY,
                                                              name=f"zD_{v_id}_{pid}_{t}")
                discharge[(v_id, pid, t)] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                                         name=f"D_{v_id}_{pid}_{t}")
                model.addConstr(discharge[(v_id, pid, t)] <= fmax * z[(v_id, pid, t, "discharge")],
                                name=f"Dmax_{v_id}_{pid}_{t}")
                model.addConstr(discharge[(v_id, pid, t)] >= fmin * z[(v_id, pid, t, "discharge")],
                                name=f"Dmin_{v_id}_{pid}_{t}")
                model.addConstr(z[(v_id, pid, t, "discharge")] <= y[(v_id, pid, t)],
                                name=f"Dy_{v_id}_{pid}_{t}")

    # Cargo balance on vessel at each node
    for v in vessels:
        v_id = v["id"]
        a_start = int(clamp(v["availability_start"], 0, T))
        a_end = int(clamp(v["availability_end"], 0, T))
        if a_end < a_start:
            a_end = a_start

        for t in range(a_start, a_end + 1):
            for p in all_ports:
                ins = in_arcs[v_id][(p, t)]
                outs = out_arcs[v_id][(p, t)]
                fin = gp.quicksum(f[(v_id, a)] for a in ins) if ins else 0
                fout = gp.quicksum(f[(v_id, a)] for a in outs) if outs else 0

                lvar = load.get((v_id, p, t), 0.0)
                dvar = discharge.get((v_id, p, t), 0.0)

                model.addConstr(fout == fin + lvar - dvar, name=f"cargo_bal_{v_id}_{p}_{t}")

    # Port inventory dynamics
    # Supply ports
    for p in supply_ports:
        pid = p["port_id"]
        init_inv = float(p["initial_inventory"])
        prod = p["production_rate"]
        stor = p["storage_capacity"]

        for t in range(1, T + 1):
            inv[(pid, t)] = model.addVar(lb=0.0, ub=float(stor[t - 1]), vtype=GRB.CONTINUOUS,
                                         name=f"Inv_{pid}_{t}")
            total_load = gp.quicksum(load[(v["id"], pid, t)] for v in vessels if (v["id"], pid, t) in load)
            if t == 1:
                model.addConstr(inv[(pid, t)] == init_inv + float(prod[t - 1]) - total_load,
                                name=f"inv_sup_{pid}_{t}")
            else:
                model.addConstr(inv[(pid, t)] == inv[(pid, t - 1)] + float(prod[t - 1]) - total_load,
                                name=f"inv_sup_{pid}_{t}")

    # Demand ports
    for p in demand_ports:
        pid = p["port_id"]
        init_inv = float(p["initial_inventory"])
        cons = p["consumption_rate"]
        stor = p["storage_capacity"]

        for t in range(1, T + 1):
            inv[(pid, t)] = model.addVar(lb=0.0, ub=float(stor[t - 1]), vtype=GRB.CONTINUOUS,
                                         name=f"Inv_{pid}_{t}")
            total_dis = gp.quicksum(discharge[(v["id"], pid, t)] for v in vessels if (v["id"], pid, t) in discharge)
            if t == 1:
                model.addConstr(inv[(pid, t)] == init_inv - float(cons[t - 1]) + total_dis,
                                name=f"inv_dem_{pid}_{t}")
            else:
                model.addConstr(inv[(pid, t)] == inv[(pid, t - 1)] - float(cons[t - 1]) + total_dis,
                                name=f"inv_dem_{pid}_{t}")

    # Objective components
    transport_expr = gp.LinExpr()
    demurrage_expr = gp.LinExpr()
    procurement_expr = gp.LinExpr()
    revenue_expr = gp.LinExpr()

    for v in vessels:
        v_id = v["id"]
        for a_idx, a in enumerate(arcs[v_id]):
            if a["is_travel"]:
                transport_expr += a["cost"] * x[(v_id, a_idx)]
            elif a["is_wait"]:
                demurrage_expr += a["cost"] * x[(v_id, a_idx)]

    for p in supply_ports:
        pid = p["port_id"]
        c = float(p["unit_procurement_cost"])
        for v in vessels:
            v_id = v["id"]
            for t in range(1, T + 1):
                if (v_id, pid, t) in load:
                    procurement_expr += c * load[(v_id, pid, t)]

    for p in demand_ports:
        pid = p["port_id"]
        r = float(p["unit_supply_revenue"])
        for v in vessels:
            v_id = v["id"]
            for t in range(1, T + 1):
                if (v_id, pid, t) in discharge:
                    revenue_expr += r * discharge[(v_id, pid, t)]

    model.setObjective(transport_expr + demurrage_expr + procurement_expr - revenue_expr, GRB.MINIMIZE)

    # Incumbent logging callback
    best_cb_obj = {"val": float("inf")}

    def cb(m, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < best_cb_obj["val"]:
                best_cb_obj["val"] = obj
                logger.log(float(obj))

    model.optimize(cb)

    return model, arcs, x, f, load, discharge, inv, vessels, all_ports, T


def build_solution(instance, model, arcs, x, f, load, discharge, inv, vessels, all_ports, T):
    feasible = model.SolCount > 0
    if not feasible:
        # Fallback empty solution
        vessel_routes = {v["id"]: [] for v in vessels}
        vessel_ops = {v["id"]: [] for v in vessels}
        port_inventories = {}
        for p in all_ports:
            port_inventories[p] = {str(t): 0.0 for t in range(1, T + 1)}
        sol = {
            "objective_value": 1e18,
            "vessel_routes": vessel_routes,
            "port_inventories": port_inventories,
            "vessel_operations": vessel_ops,
            "cost_breakdown": {
                "transport_cost": 0.0,
                "demurrage_cost": 0.0,
                "procurement_cost": 0.0,
                "supply_revenue": 0.0,
                "total_cost": 1e18,
            },
        }
        return sol

    # Costs
    transport_cost = 0.0
    demurrage_cost = 0.0

    vessel_routes = {}
    for v in vessels:
        v_id = v["id"]
        chosen = []
        for a_idx, a in enumerate(arcs[v_id]):
            xv = x[(v_id, a_idx)].X
            if xv > 0.5:
                flow = f[(v_id, a_idx)].X
                if a["type"] == "source":
                    tail = f"(src,{v_id})"
                    head = f"({a['head_port']},{a['head_t']})"
                elif a["type"] == "sink":
                    tail = f"({a['tail_port']},{a['tail_t']})"
                    head = f"(snk,{v_id})"
                else:
                    tail = f"({a['tail_port']},{a['tail_t']})"
                    head = f"({a['head_port']},{a['head_t']})"

                cost = float(a["cost"])
                if a["type"] == "travel":
                    transport_cost += cost
                elif a["type"] == "demurrage":
                    demurrage_cost += cost

                chosen.append({
                    "tail": tail,
                    "head": head,
                    "type": a["type"],
                    "cost": cost,
                    "product_flow": float(flow),
                    "_sort_t": -1 if a["type"] == "source" else (10**9 if a["type"] == "sink" else a["tail_t"]),
                })

        chosen.sort(key=lambda d: (d["_sort_t"], d["tail"], d["head"]))
        for d in chosen:
            d.pop("_sort_t", None)
        vessel_routes[v_id] = chosen

    # Operations
    vessel_operations = {v["id"]: [] for v in vessels}
    procurement_cost = 0.0
    supply_revenue = 0.0

    supply_map = {p["port_id"]: p for p in instance["supply_ports"]}
    demand_map = {p["port_id"]: p for p in instance["demand_ports"]}

    for (v_id, p, t), var in load.items():
        q = var.X
        if q > 1e-6:
            vessel_operations[v_id].append({
                "port": p,
                "time": int(t),
                "quantity": float(q),
                "type": "load",
            })
            procurement_cost += float(q) * float(supply_map[p]["unit_procurement_cost"])

    for (v_id, p, t), var in discharge.items():
        q = var.X
        if q > 1e-6:
            vessel_operations[v_id].append({
                "port": p,
                "time": int(t),
                "quantity": float(-q),  # negative for discharge
                "type": "discharge",
            })
            supply_revenue += float(q) * float(demand_map[p]["unit_supply_revenue"])

    for v in vessels:
        vessel_operations[v["id"]].sort(key=lambda r: (r["time"], r["port"]))

    # Port inventories
    port_inventories = {}
    for p in all_ports:
        port_inventories[p] = {}
        for t in range(1, T + 1):
            val = inv[(p, t)].X if (p, t) in inv else 0.0
            port_inventories[p][str(t)] = float(val)

    total_cost = transport_cost + demurrage_cost + procurement_cost - supply_revenue
    objective_value = float(model.ObjVal)

    sol = {
        "objective_value": objective_value,
        "vessel_routes": vessel_routes,
        "port_inventories": port_inventories,
        "vessel_operations": vessel_operations,
        "cost_breakdown": {
            "transport_cost": float(transport_cost),
            "demurrage_cost": float(demurrage_cost),
            "procurement_cost": float(procurement_cost),
            "supply_revenue": float(supply_revenue),
            "total_cost": float(total_cost),
        },
    }
    return sol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = parse_instance(args.instance_path)
    model, arcs, x, f, load, discharge, inv, vessels, all_ports, T = build_and_solve(
        instance, args.time_limit, logger=logger
    )
    solution = build_solution(instance, model, arcs, x, f, load, discharge, inv, vessels, all_ports, T)

    if logger is not None:
        logger.log_solution(solution["objective_value"], solution)

    with open(args.solution_path, "w") as f_out:
        json.dump(solution, f_out, indent=2)


if __name__ == "__main__":
    main()