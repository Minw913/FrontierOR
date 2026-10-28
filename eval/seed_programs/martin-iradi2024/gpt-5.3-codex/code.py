import argparse
import json
from itertools import combinations
from typing import Dict, List, Tuple, Any

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def to_float(x):
    return float(x)


def get_dict_value_by_float_key(d: Dict[str, Any], key_float: float):
    # Robust lookup for JSON dicts keyed by speed strings
    k1 = str(key_float)
    if k1 in d:
        return d[k1]
    k2 = f"{key_float:.1f}"
    if k2 in d:
        return d[k2]
    k3 = f"{key_float:.2f}"
    if k3 in d:
        return d[k3]
    for k, v in d.items():
        try:
            if abs(float(k) - key_float) <= 1e-9:
                return v
        except Exception:
            pass
    raise KeyError(f"Could not find key for float {key_float} in dict keys={list(d.keys())[:5]}...")


def get_distance(distances: Dict[str, float], o: str, d: str) -> float:
    k = f"{o}_{d}"
    if k in distances:
        return float(distances[k])
    raise KeyError(f"Distance key missing: {k}")


def solve(instance: Dict[str, Any], time_limit: int, logger: SolutionLogger = None):
    planning_horizon = float(instance["planning_horizon_hours"])
    costs = instance["cost_parameters"]
    fuel_price = float(costs["fuel_price_usd_per_tonne"])
    handling_cost = float(costs["handling_cost_usd_per_hour"])
    delay_cost = float(costs["delay_cost_usd_per_hour"])
    waiting_cost = float(costs["waiting_cost_usd_per_hour"])
    lft_penalty = float(costs["lft_penalty_usd"])

    beta = float(instance["handling_time_deviation_factor_beta_per_meter"])

    ports = instance["ports"]
    quay_len = {p["id"]: float(p["quay_length_meters"]) for p in ports}
    max_quay = max(quay_len.values()) if quay_len else 0.0

    distances = {k: float(v) for k, v in instance["distances_nautical_miles"].items()}
    speeds = [float(s) for s in instance["speeds_knots"]]
    tt_per_nm = {
        float(s): float(get_dict_value_by_float_key(instance["travel_time_hours_per_nm_by_speed"], float(s)))
        for s in speeds
    }

    ships_data = instance["ships"]

    external_by_port = {}
    for ep in instance.get("external_ships", []):
        pid = ep["port_id"]
        external_by_port.setdefault(pid, [])
        for es in ep.get("ships", []):
            external_by_port[pid].append(
                {
                    "id": es["id"],
                    "pos": float(es["berthing_position"]),
                    "len": float(es["length_meters"]),
                    "start": float(es["start_time_hours"]),
                    "ht": float(es["handling_time_hours"]),
                    "end": float(es["end_time_hours"]),
                }
            )

    # Build visit structures
    visits = {}  # (ship_id, call_idx) -> dict
    ship_calls_order = {}  # ship_id -> sorted list of call_idx
    ship_length = {}
    ship_fuel_by_speed = {}

    for sh in ships_data:
        sid = int(sh["id"])
        ship_length[sid] = float(sh["length_meters"])
        ship_fuel_by_speed[sid] = {
            sp: float(get_dict_value_by_float_key(sh["fuel_consumption_per_nm_by_speed"], sp))
            for sp in speeds
        }
        pvs = sorted(sh["port_visits"], key=lambda x: int(x["port_call_index"]))
        call_list = []
        for pv in pvs:
            c = int(pv["port_call_index"])
            call_list.append(c)
            visits[(sid, c)] = {
                "port_id": pv["port_id"],
                "ideal": float(pv["ideal_berthing_position"]),
                "min_h": float(pv["min_handling_time_hours"]),
                "max_h": float(pv["max_handling_time_hours"]),
                "earliest": float(pv["earliest_start_time_hours"]),
                "expected_finish": float(pv["expected_finish_time_hours"]),
                "latest_finish": float(pv["latest_finish_time_hours"]),
            }
        ship_calls_order[sid] = call_list

    # Time big-M
    TMAX = planning_horizon + 10000.0
    M_time = TMAX + 10000.0
    M_pos = max_quay + max(ship_length.values()) + 1000.0 if ship_length else 10000.0

    model = gp.Model("ship_berth_speed_integrated")
    model.Params.TimeLimit = int(time_limit)
    model.Params.Threads = 1
    model.Params.OutputFlag = 0

    # Decision variables
    x = {}
    s = {}
    a = {}
    h = {}
    dev = {}
    delay = {}
    lftv = {}
    wait = {}

    for (sid, c), vd in visits.items():
        pid = vd["port_id"]
        L = ship_length[sid]
        x[(sid, c)] = model.addVar(lb=0.0, ub=max(0.0, quay_len[pid] - L), vtype=GRB.CONTINUOUS, name=f"x_{sid}_{c}")
        s[(sid, c)] = model.addVar(lb=0.0, ub=TMAX, vtype=GRB.CONTINUOUS, name=f"s_{sid}_{c}")
        a[(sid, c)] = model.addVar(lb=0.0, ub=TMAX, vtype=GRB.CONTINUOUS, name=f"a_{sid}_{c}")
        h[(sid, c)] = model.addVar(lb=0.0, ub=TMAX, vtype=GRB.CONTINUOUS, name=f"h_{sid}_{c}")
        dev[(sid, c)] = model.addVar(lb=0.0, ub=max_quay + 1000.0, vtype=GRB.CONTINUOUS, name=f"dev_{sid}_{c}")
        delay[(sid, c)] = model.addVar(lb=0.0, ub=TMAX, vtype=GRB.CONTINUOUS, name=f"delay_{sid}_{c}")
        lftv[(sid, c)] = model.addVar(lb=0.0, ub=TMAX, vtype=GRB.CONTINUOUS, name=f"lft_{sid}_{c}")
        wait[(sid, c)] = model.addVar(lb=0.0, ub=TMAX, vtype=GRB.CONTINUOUS, name=f"wait_{sid}_{c}")

    # Speed selection binaries per leg
    y = {}  # (sid, leg_pos, k_idx)
    leg_info = {}  # (sid, leg_pos) -> (from_call, to_call, dist)
    for sid, calls in ship_calls_order.items():
        for lp in range(len(calls) - 1):
            c1, c2 = calls[lp], calls[lp + 1]
            p1 = visits[(sid, c1)]["port_id"]
            p2 = visits[(sid, c2)]["port_id"]
            dist = get_distance(distances, p1, p2)
            leg_info[(sid, lp)] = (c1, c2, dist)
            for k, sp in enumerate(speeds):
                y[(sid, lp, k)] = model.addVar(vtype=GRB.BINARY, name=f"y_{sid}_{lp}_{k}")

    model.update()

    # Core constraints
    for (sid, c), vd in visits.items():
        ideal = vd["ideal"]
        min_h = vd["min_h"]

        model.addConstr(dev[(sid, c)] >= x[(sid, c)] - ideal, name=f"devp_{sid}_{c}")
        model.addConstr(dev[(sid, c)] >= ideal - x[(sid, c)], name=f"devm_{sid}_{c}")
        model.addConstr(h[(sid, c)] == min_h * (1.0 + beta * dev[(sid, c)]), name=f"ht_{sid}_{c}")

        model.addConstr(s[(sid, c)] >= vd["earliest"], name=f"earliest_{sid}_{c}")
        model.addConstr(wait[(sid, c)] == s[(sid, c)] - a[(sid, c)], name=f"waitdef_{sid}_{c}")
        model.addConstr(s[(sid, c)] >= a[(sid, c)], name=f"start_after_arr_{sid}_{c}")

        end_expr = s[(sid, c)] + h[(sid, c)]
        model.addConstr(delay[(sid, c)] >= end_expr - vd["expected_finish"], name=f"delaydef_{sid}_{c}")
        model.addConstr(lftv[(sid, c)] >= end_expr - vd["latest_finish"], name=f"lftdef_{sid}_{c}")

    # Arrival propagation
    for sid, calls in ship_calls_order.items():
        if not calls:
            continue
        first = calls[0]
        model.addConstr(a[(sid, first)] <= s[(sid, first)], name=f"first_arr_{sid}")

        for lp in range(len(calls) - 1):
            c1, c2, dist = leg_info[(sid, lp)]
            model.addConstr(gp.quicksum(y[(sid, lp, k)] for k in range(len(speeds))) == 1, name=f"speed1_{sid}_{lp}")
            travel_time_expr = gp.quicksum(dist * tt_per_nm[speeds[k]] * y[(sid, lp, k)] for k in range(len(speeds)))
            model.addConstr(
                a[(sid, c2)] == s[(sid, c1)] + h[(sid, c1)] + travel_time_expr,
                name=f"arrprop_{sid}_{lp}"
            )

    # Non-overlap optimized vs optimized, per port
    visits_by_port = {}
    for key, vd in visits.items():
        visits_by_port.setdefault(vd["port_id"], []).append(key)

    for pid, vlist in visits_by_port.items():
        for (u_sid, u_c), (v_sid, v_c) in combinations(vlist, 2):
            bL = model.addVar(vtype=GRB.BINARY, name=f"noovL_{u_sid}_{u_c}_{v_sid}_{v_c}")
            bR = model.addVar(vtype=GRB.BINARY, name=f"noovR_{u_sid}_{u_c}_{v_sid}_{v_c}")
            bT = model.addVar(vtype=GRB.BINARY, name=f"noovT_{u_sid}_{u_c}_{v_sid}_{v_c}")
            bB = model.addVar(vtype=GRB.BINARY, name=f"noovB_{u_sid}_{u_c}_{v_sid}_{v_c}")

            model.addConstr(
                x[(u_sid, u_c)] + ship_length[u_sid] <= x[(v_sid, v_c)] + M_pos * (1 - bL),
                name=f"spaceL_{u_sid}_{u_c}_{v_sid}_{v_c}"
            )
            model.addConstr(
                x[(v_sid, v_c)] + ship_length[v_sid] <= x[(u_sid, u_c)] + M_pos * (1 - bR),
                name=f"spaceR_{u_sid}_{u_c}_{v_sid}_{v_c}"
            )
            model.addConstr(
                s[(u_sid, u_c)] + h[(u_sid, u_c)] <= s[(v_sid, v_c)] + M_time * (1 - bT),
                name=f"timeT_{u_sid}_{u_c}_{v_sid}_{v_c}"
            )
            model.addConstr(
                s[(v_sid, v_c)] + h[(v_sid, v_c)] <= s[(u_sid, u_c)] + M_time * (1 - bB),
                name=f"timeB_{u_sid}_{u_c}_{v_sid}_{v_c}"
            )
            model.addConstr(bL + bR + bT + bB >= 1, name=f"disj_{u_sid}_{u_c}_{v_sid}_{v_c}")

    # Non-overlap optimized vs external
    for pid, vlist in visits_by_port.items():
        ex_list = external_by_port.get(pid, [])
        for (sid, c) in vlist:
            for ex in ex_list:
                epos = ex["pos"]
                elen = ex["len"]
                estart = ex["start"]
                eend = ex["end"]

                bL = model.addVar(vtype=GRB.BINARY, name=f"exL_{sid}_{c}_{pid}_{ex['id']}")
                bR = model.addVar(vtype=GRB.BINARY, name=f"exR_{sid}_{c}_{pid}_{ex['id']}")
                bT = model.addVar(vtype=GRB.BINARY, name=f"exT_{sid}_{c}_{pid}_{ex['id']}")
                bB = model.addVar(vtype=GRB.BINARY, name=f"exB_{sid}_{c}_{pid}_{ex['id']}")

                model.addConstr(
                    x[(sid, c)] + ship_length[sid] <= epos + M_pos * (1 - bL),
                    name=f"exSpaceL_{sid}_{c}_{ex['id']}"
                )
                model.addConstr(
                    epos + elen <= x[(sid, c)] + M_pos * (1 - bR),
                    name=f"exSpaceR_{sid}_{c}_{ex['id']}"
                )
                model.addConstr(
                    s[(sid, c)] + h[(sid, c)] <= estart + M_time * (1 - bT),
                    name=f"exTimeT_{sid}_{c}_{ex['id']}"
                )
                model.addConstr(
                    eend <= s[(sid, c)] + M_time * (1 - bB),
                    name=f"exTimeB_{sid}_{c}_{ex['id']}"
                )
                model.addConstr(bL + bR + bT + bB >= 1, name=f"exDisj_{sid}_{c}_{ex['id']}")

    # Objective
    obj = gp.LinExpr()
    for key in visits.keys():
        obj += waiting_cost * wait[key]
        obj += handling_cost * h[key]
        obj += delay_cost * delay[key]
        obj += lft_penalty * lftv[key]

    for (sid, lp), (_, _, dist) in leg_info.items():
        for k, sp in enumerate(speeds):
            fuel_nm = ship_fuel_by_speed[sid][sp]
            obj += fuel_price * fuel_nm * dist * y[(sid, lp, k)]

    model.setObjective(obj, GRB.MINIMIZE)

    # Incumbent logging callback
    best_inc = [float("inf")]

    def cb(m, where):
        if logger is None:
            return
        if where == GRB.Callback.MIPSOL:
            val = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if val + 1e-8 < best_inc[0]:
                best_inc[0] = val
                logger.log(float(val))

    model.optimize(cb)

    if model.SolCount == 0:
        # Fallback output if no feasible solution found
        return {"objective_value": 1e30, "ships": []}

    # Build output
    out = {
        "objective_value": float(model.ObjVal),
        "ships": []
    }

    for sh in ships_data:
        sid = int(sh["id"])
        calls = ship_calls_order[sid]
        ship_out = {
            "ship_id": sid,
            "port_visits": [],
            "speed_selections": []
        }

        for c in calls:
            vd = visits[(sid, c)]
            start_val = float(s[(sid, c)].X)
            arr_val = float(a[(sid, c)].X)
            hand_val = float(h[(sid, c)].X)
            end_val = start_val + hand_val
            del_val = max(0.0, end_val - vd["expected_finish"])
            lft_val = max(0.0, end_val - vd["latest_finish"])
            pos_val = float(x[(sid, c)].X)
            dev_val = abs(pos_val - vd["ideal"])

            ship_out["port_visits"].append({
                "port_call_index": int(c),
                "port_id": vd["port_id"],
                "berthing_position": pos_val,
                "start_time": start_val,
                "handling_time": hand_val,
                "arrival_time": arr_val,
                "delay": del_val,
                "lft_violation": lft_val,
                "position_deviation": dev_val
            })

        for lp in range(len(calls) - 1):
            c1, c2, _ = leg_info[(sid, lp)]
            chosen_k = max(range(len(speeds)), key=lambda k: y[(sid, lp, k)].X)
            ship_out["speed_selections"].append({
                "from_call": int(c1),
                "to_call": int(c2),
                "speed_knots": float(speeds[chosen_k])
            })

        out["ships"].append(ship_out)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    solution = solve(instance, args.time_limit, logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()