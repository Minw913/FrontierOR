import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def bid_profitability_threshold(generator, bid, gas_price_lb, gas_price_ub):
    """Read an explicit threshold when available, otherwise use the schema fallback."""
    possible_keys = (
        "profitability_threshold",
        "gas_price_threshold",
        "max_gas_price",
        "max_gas_price_mmBtu",
        "rho",
    )
    for key in possible_keys:
        if key in bid and bid[key] is not None:
            return max(0.0, min(gas_price_ub, float(bid[key])))

    fraction = generator.get("max_gas_price_fraction")
    if fraction is None:
        return gas_price_ub

    # In instances without a separately stored per-bid threshold, the fraction
    # converts the electricity bid price to an allowable gas price.
    threshold = float(fraction) * float(bid["price"])
    return max(0.0, min(gas_price_ub, threshold))


def main():
    args = parse_args()
    start_time = time.monotonic()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("released_problem_variant") != "integrated_primal_ucgna":
        raise ValueError("Unsupported released_problem_variant")

    beta = float(data["objective_weight_beta"])
    T = int(data["time_periods"])
    periods = range(1, T + 1)

    eta_p = float(data.get("stress_parameters", {}).get("eta_p", 1.0))
    eta_g = float(data.get("stress_parameters", {}).get("eta_g", 1.0))

    buses = {int(x["id"]): x for x in data["electricity_network"]["buses"]}
    lines = {int(x["id"]): x for x in data["electricity_network"]["lines"]}
    generators = {int(x["id"]): x for x in data["generators"]["generators"]}
    junctions = {int(x["id"]): x for x in data["gas_network"]["junctions"]}
    connections = {int(x["id"]): x for x in data["gas_network"]["connections"]}
    zones = {int(x["id"]): x for x in data["gas_network"]["pricing_zones"]}

    gas_price_lb = float(data["gas_network"]["min_gas_price_mmBtu"])
    gas_price_ub = float(data["gas_network"]["max_gas_price_mmBtu"])

    gens_at_bus = {b: [] for b in buses}
    gfpps_at_junction = {j: [] for j in junctions}
    for gid, g in generators.items():
        gens_at_bus[int(g["bus"])].append(gid)
        if bool(g["is_gfpp"]):
            gfpps_at_junction[int(g["gas_junction"])].append(gid)

    lines_out = {b: [] for b in buses}
    lines_in = {b: [] for b in buses}
    for lid, line in lines.items():
        lines_out[int(line["from_bus"])].append(lid)
        lines_in[int(line["to_bus"])].append(lid)

    conns_out = {j: [] for j in junctions}
    conns_in = {j: [] for j in junctions}
    for cid, conn in connections.items():
        conns_out[int(conn["from_junction"])].append(cid)
        conns_in[int(conn["to_junction"])].append(cid)

    model = gp.Model("integrated_primal_ucgna")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.NonConvex = 2
    model.Params.MIPFocus = 1

    # Electricity commitment and dispatch variables.
    o = {}
    vp = {}
    vm = {}
    r = {}
    p = {}
    w = {}
    se = {}
    phimax = {}

    for gid, g in generators.items():
        initial_status = int(g["initial_status"])
        initial_gen = float(g["initial_gen"])

        o[gid, 0] = model.addVar(
            lb=initial_status, ub=initial_status, vtype=GRB.CONTINUOUS,
            name=f"o_{gid}_0"
        )
        p[gid, 0] = model.addVar(
            lb=initial_gen, ub=initial_gen, vtype=GRB.CONTINUOUS,
            name=f"p_{gid}_0"
        )

        max_startup_cost = max(
            [float(x[1]) for x in g.get("startup_cost_params", [])] + [0.0]
        )

        for t in periods:
            o[gid, t] = model.addVar(vtype=GRB.BINARY, name=f"o_{gid}_{t}")
            vp[gid, t] = model.addVar(vtype=GRB.BINARY, name=f"vp_{gid}_{t}")
            vm[gid, t] = model.addVar(vtype=GRB.BINARY, name=f"vm_{gid}_{t}")
            r[gid, t] = model.addVar(
                lb=0.0,
                ub=max_startup_cost if max_startup_cost > 0.0 else 0.0,
                name=f"r_{gid}_{t}",
            )
            p[gid, t] = model.addVar(lb=0.0, name=f"p_{gid}_{t}")

            for bid in g["bids"]:
                bid_id = int(bid["id"])
                w[gid, bid_id, t] = model.addVar(
                    vtype=GRB.BINARY, name=f"w_{gid}_{bid_id}_{t}"
                )
                se[gid, bid_id, t] = model.addVar(
                    lb=0.0,
                    ub=float(bid["max_amount"]),
                    name=f"se_{gid}_{bid_id}_{t}",
                )

            if bool(g["is_gfpp"]):
                phimax[gid, t] = model.addVar(
                    lb=0.0, ub=gas_price_ub, name=f"phimax_{gid}_{t}"
                )

    # Electricity network variables.
    fline = {}
    theta = {}
    for lid, line in lines.items():
        limit = float(line["thermal_limit"])
        for t in periods:
            fline[lid, t] = model.addVar(
                lb=-limit, ub=limit, name=f"f_{lid}_{t}"
            )

    for bid, bus in buses.items():
        lb = float(bus["voltage_angle_lb"])
        ub = float(bus["voltage_angle_ub"])
        for t in periods:
            theta[bid, t] = model.addVar(
                lb=lb, ub=ub, name=f"theta_{bid}_{t}"
            )

    # Gas network variables.
    sg = {}
    sgs = {}
    pisq = {}
    phig = {}
    lg = {}
    qg = {}
    gamma = {}
    psi = {}

    for jid, junction in junctions.items():
        pressure_lb = float(junction["pressure_lb_squared"])
        pressure_ub = float(junction["pressure_ub_squared"])
        demand_profile = junction["gas_demand_profile"]

        for t in periods:
            demand = eta_g * float(demand_profile[t - 1])
            sg[jid, t] = model.addVar(lb=0.0, name=f"sg_{jid}_{t}")
            pisq[jid, t] = model.addVar(
                lb=pressure_lb, ub=pressure_ub, name=f"pisq_{jid}_{t}"
            )
            lg[jid, t] = model.addVar(
                lb=0.0, ub=demand, name=f"lg_{jid}_{t}"
            )
            qg[jid, t] = model.addVar(
                lb=0.0, ub=demand, name=f"qg_{jid}_{t}"
            )
            gamma[jid, t] = model.addVar(lb=0.0, name=f"gamma_{jid}_{t}")

            if bool(junction["is_source"]):
                for interval in junction.get("supply_intervals", []):
                    sid = int(interval["id"])
                    capacity = max(
                        0.0,
                        float(interval["interval_ub"])
                        - float(interval["interval_lb"]),
                    )
                    sgs[jid, sid, t] = model.addVar(
                        lb=0.0, ub=capacity, name=f"sgs_{jid}_{sid}_{t}"
                    )

    for cid in connections:
        for t in periods:
            phig[cid, t] = model.addVar(lb=0.0, name=f"phig_{cid}_{t}")

    # The minimum zonal price weakly relaxes all bid-validity constraints and
    # carries no direct objective penalty, so it is always an optimal choice.
    for zid in zones:
        for t in periods:
            psi[zid, t] = model.addVar(
                lb=gas_price_lb,
                ub=gas_price_lb,
                name=f"psi_{zid}_{t}",
            )

    model.update()

    # Initial commitment restrictions and transitions.
    for gid, g in generators.items():
        initial_status = int(g["initial_status"])
        min_up = int(g["min_up_time"])
        min_down = int(g["min_down_time"])

        if initial_status == 1:
            remaining = max(1, min_up - int(g.get("initial_active_periods", 0)))
        else:
            remaining = max(1, min_down - int(g.get("initial_inactive_periods", 0)))

        for t in range(1, min(T, remaining) + 1):
            model.addConstr(o[gid, t] == initial_status)

        for t in periods:
            model.addConstr(
                vp[gid, t] - vm[gid, t] == o[gid, t] - o[gid, t - 1]
            )
            model.addConstr(vp[gid, t] + vm[gid, t] <= 1)

        # Startup costs from lookback tiers.
        for t in periods:
            for tier in g.get("startup_cost_params", []):
                lookback = int(tier[0])
                cost = float(tier[1])
                history_terms = []
                history_constant = 0.0

                for tau in range(t - lookback, t):
                    if tau >= 0:
                        history_terms.append(o[gid, tau])
                    else:
                        if initial_status == 1:
                            active = int(g.get("initial_active_periods", 0))
                            past_status = 1 if tau >= -active else 0
                        else:
                            inactive = int(g.get("initial_inactive_periods", 0))
                            past_status = 0 if tau >= -inactive else 1
                        history_constant += past_status

                model.addConstr(
                    r[gid, t] >= cost * (
                        o[gid, t]
                        - gp.quicksum(history_terms)
                        - history_constant
                    )
                )

            if r[gid, t].UB > 0:
                model.addConstr(r[gid, t] <= r[gid, t].UB * vp[gid, t])

        # Minimum up-time windows.
        if min_up > 0:
            for start in range(1, T - min_up + 2):
                end = start + min_up - 1
                model.addConstr(
                    gp.quicksum(vp[gid, tau] for tau in range(start, end + 1))
                    <= o[gid, end]
                )

        # Released formulation's minimum down-time startup-window rule.
        if min_down > 0:
            for start in range(1, T - min_down + 2):
                end = start + min_down - 1
                model.addConstr(
                    gp.quicksum(vp[gid, tau] for tau in range(start, end + 1))
                    <= 1 - o[gid, start - 1]
                )

    # Generator output, bid, and ramping constraints.
    for gid, g in generators.items():
        pmin = float(g["min_power"])
        pmax = float(g["max_power"])
        ramp_up = float(g["ramp_up"])
        ramp_down = float(g["ramp_down"])
        bid_list = list(g["bids"])

        for t in periods:
            model.addConstr(
                p[gid, t]
                == gp.quicksum(
                    se[gid, int(bid["id"]), t] for bid in bid_list
                )
            )
            model.addConstr(p[gid, t] >= pmin * o[gid, t])
            model.addConstr(p[gid, t] <= pmax * o[gid, t])

            model.addConstr(
                p[gid, t] - p[gid, t - 1]
                <= ramp_up * o[gid, t - 1] + pmax * vp[gid, t]
            )
            model.addConstr(
                p[gid, t - 1] - p[gid, t]
                <= ramp_down * o[gid, t] + pmin * vm[gid, t]
            )

            if bool(g["is_gfpp"]):
                for idx, bid in enumerate(bid_list):
                    bid_id = int(bid["id"])
                    max_amount = float(bid["max_amount"])
                    model.addConstr(w[gid, bid_id, t] <= o[gid, t])
                    model.addConstr(
                        se[gid, bid_id, t]
                        <= max_amount * w[gid, bid_id, t]
                    )

                    if idx + 1 < len(bid_list):
                        next_id = int(bid_list[idx + 1]["id"])
                        model.addConstr(
                            w[gid, next_id, t] <= w[gid, bid_id, t]
                        )
                        model.addConstr(
                            se[gid, bid_id, t]
                            >= max_amount * w[gid, next_id, t]
                        )

                threshold_expr = gp.LinExpr()
                for idx, bid in enumerate(bid_list):
                    bid_id = int(bid["id"])
                    threshold = bid_profitability_threshold(
                        g, bid, gas_price_lb, gas_price_ub
                    )
                    if idx + 1 < len(bid_list):
                        next_id = int(bid_list[idx + 1]["id"])
                        threshold_expr += threshold * (
                            w[gid, bid_id, t] - w[gid, next_id, t]
                        )
                    else:
                        threshold_expr += threshold * w[gid, bid_id, t]

                model.addConstr(phimax[gid, t] == threshold_expr)

                gas_junction = int(g["gas_junction"])
                zone_id = next(
                    zid for zid, zone in zones.items()
                    if gas_junction in [int(x) for x in zone["junctions"]]
                )
                # Exact binary-continuous implication:
                # o = 1 => phimax >= psi.
                model.addConstr(
                    phimax[gid, t]
                    >= psi[zone_id, t] - gas_price_ub * (1 - o[gid, t])
                )
            else:
                # These schema fields are present for every generator, but bid
                # selection has operational meaning only for GFPPs.
                for bid in bid_list:
                    model.addConstr(w[gid, int(bid["id"]), t] == 0)

    # Power balance and DC transmission.
    for bus_id, bus in buses.items():
        for t in periods:
            demand = eta_p * float(bus["demand_profile"][t - 1])
            generation = gp.quicksum(
                p[gid, t] for gid in gens_at_bus[bus_id]
            )
            net_outflow = (
                gp.quicksum(fline[lid, t] for lid in lines_out[bus_id])
                - gp.quicksum(fline[lid, t] for lid in lines_in[bus_id])
            )
            model.addConstr(generation - demand == net_outflow)

    for lid, line in lines.items():
        from_bus = int(line["from_bus"])
        to_bus = int(line["to_bus"])
        susceptance = float(line["susceptance"])
        angle_limit = float(line["angle_diff_limit"])

        for t in periods:
            angle_difference = theta[from_bus, t] - theta[to_bus, t]
            model.addConstr(fline[lid, t] == susceptance * angle_difference)
            model.addConstr(angle_difference <= angle_limit)
            model.addConstr(angle_difference >= -angle_limit)

    # Fix one angle per connected formulation to remove the global shift.
    if buses:
        reference_bus = min(buses)
        for t in periods:
            ref_lb = float(buses[reference_bus]["voltage_angle_lb"])
            ref_ub = float(buses[reference_bus]["voltage_angle_ub"])
            if ref_lb <= 0.0 <= ref_ub:
                model.addConstr(theta[reference_bus, t] == 0.0)

    # Gas demand, supply, balance, and GFPP consumption.
    for jid, junction in junctions.items():
        for t in periods:
            demand = eta_g * float(junction["gas_demand_profile"][t - 1])
            model.addConstr(lg[jid, t] + qg[jid, t] == demand)

            if bool(junction["is_source"]):
                model.addConstr(
                    sg[jid, t]
                    == gp.quicksum(
                        sgs[jid, int(interval["id"]), t]
                        for interval in junction.get("supply_intervals", [])
                    )
                )
            else:
                model.addConstr(sg[jid, t] == 0.0)

            net_outflow = (
                gp.quicksum(phig[cid, t] for cid in conns_out[jid])
                - gp.quicksum(phig[cid, t] for cid in conns_in[jid])
            )
            model.addConstr(
                sg[jid, t] - lg[jid, t] - gamma[jid, t] == net_outflow
            )

            gfpp_list = gfpps_at_junction[jid]
            if gfpp_list:
                heat_expr = gp.QuadExpr()
                for gid in gfpp_list:
                    coeff = generators[gid]["heat_rate_coefficients"]
                    h2 = float(coeff["H_u2"] or 0.0)
                    h1 = float(coeff["H_u1"] or 0.0)
                    h0 = float(coeff["H_u0"] or 0.0)
                    heat_expr += h2 * p[gid, t] * p[gid, t]
                    heat_expr += h1 * p[gid, t]
                    heat_expr += h0 * o[gid, t]
                model.addQConstr(gamma[jid, t] >= heat_expr)
            else:
                model.addConstr(gamma[jid, t] == 0.0)

    # Gas pressure and connection constraints.
    for cid, conn in connections.items():
        from_j = int(conn["from_junction"])
        to_j = int(conn["to_junction"])
        conn_type = str(conn.get("type", "")).lower()

        for t in periods:
            control_lb = conn.get("control_ratio_lb")
            control_ub = conn.get("control_ratio_ub")

            if control_lb is not None and control_ub is not None:
                lb2 = float(control_lb) ** 2
                ub2 = float(control_ub) ** 2
                model.addConstr(pisq[to_j, t] >= lb2 * pisq[from_j, t])
                model.addConstr(pisq[to_j, t] <= ub2 * pisq[from_j, t])
            elif conn_type == "compressor":
                ratio_lb = float(conn["compression_ratio_lb"])
                ratio_ub = float(conn["compression_ratio_ub"])
                model.addConstr(
                    pisq[to_j, t] >= ratio_lb ** 2 * pisq[from_j, t]
                )
                model.addConstr(
                    pisq[to_j, t] <= ratio_ub ** 2 * pisq[from_j, t]
                )
            else:
                factor = float(conn["weymouth_factor"])
                model.addQConstr(
                    factor * phig[cid, t] * phig[cid, t]
                    <= pisq[from_j, t] - pisq[to_j, t]
                )

    electricity_cost = gp.LinExpr()
    gas_cost = gp.LinExpr()

    for gid, g in generators.items():
        no_load = float(g["no_load_cost"])
        for t in periods:
            electricity_cost += no_load * o[gid, t] + r[gid, t]
            for bid in g["bids"]:
                bid_id = int(bid["id"])
                electricity_cost += (
                    float(bid["price"]) * se[gid, bid_id, t]
                )

    for jid, junction in junctions.items():
        shedding_cost = float(junction["demand_shedding_cost"])
        for t in periods:
            gas_cost += shedding_cost * qg[jid, t]
            if bool(junction["is_source"]):
                for interval in junction.get("supply_intervals", []):
                    sid = int(interval["id"])
                    gas_cost += float(interval["slope"]) * sgs[jid, sid, t]

    model.setObjective(beta * electricity_cost + (1.0 - beta) * gas_cost, GRB.MINIMIZE)

    all_variable_maps = {
        "o": o, "vp": vp, "vm": vm, "w": w, "r": r,
        "phimax": phimax, "p": p, "se": se, "f": fline,
        "theta": theta, "sg": sg, "sgs": sgs, "pisq": pisq,
        "phig": phig, "lg": lg, "qg": qg, "gamma": gamma, "psi": psi,
    }

    def clean_value(value, binary=False):
        value = float(value)
        if abs(value) < 1e-9:
            return 0.0
        if binary:
            if abs(value - 1.0) < 1e-6:
                return 1.0
            if abs(value) < 1e-6:
                return 0.0
        return value

    def build_solution(get_value):
        primary = {}

        binary_prefixes = {"o", "vp", "vm", "w"}
        for prefix, variable_map in all_variable_maps.items():
            for index, variable in variable_map.items():
                if not isinstance(index, tuple):
                    index = (index,)
                key = prefix + "_" + "_".join(str(x) for x in index)
                primary[key] = clean_value(
                    get_value(variable), prefix in binary_prefixes
                )

        e_cost = 0.0
        for gid, g in generators.items():
            for t in periods:
                e_cost += float(g["no_load_cost"]) * primary[f"o_{gid}_{t}"]
                e_cost += primary[f"r_{gid}_{t}"]
                for bid in g["bids"]:
                    bid_id = int(bid["id"])
                    e_cost += (
                        float(bid["price"])
                        * primary[f"se_{gid}_{bid_id}_{t}"]
                    )

        g_cost = 0.0
        for jid, junction in junctions.items():
            for t in periods:
                g_cost += (
                    float(junction["demand_shedding_cost"])
                    * primary[f"qg_{jid}_{t}"]
                )
                if bool(junction["is_source"]):
                    for interval in junction.get("supply_intervals", []):
                        sid = int(interval["id"])
                        g_cost += (
                            float(interval["slope"])
                            * primary[f"sgs_{jid}_{sid}_{t}"]
                        )

        objective_value = beta * e_cost + (1.0 - beta) * g_cost
        solution = {
            "objective_value": float(objective_value),
            "released_problem_variant": "integrated_primal_ucgna",
            "objective_weight_beta": beta,
            "primary_variables": primary,
        }
        return objective_value, solution

    best_logged = math.inf

    def incumbent_callback(cb_model, where):
        nonlocal best_logged
        if where != GRB.Callback.MIPSOL:
            return
        try:
            incumbent_obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if incumbent_obj >= best_logged - 1e-7:
                return

            if logger:
                objective_value, solution = build_solution(
                    lambda variable: cb_model.cbGetSolution(variable)
                )
                if objective_value < best_logged - 1e-7:
                    logger.log_solution(objective_value, solution)
                    best_logged = objective_value
            else:
                best_logged = incumbent_obj
        except Exception:
            # Logging must not interrupt optimization.
            pass

    elapsed = time.monotonic() - start_time
    remaining = max(0.01, float(args.time_limit) - elapsed)
    model.Params.TimeLimit = remaining

    model.optimize(incumbent_callback)

    if model.SolCount <= 0:
        raise RuntimeError(
            f"No feasible solution found within the time limit; solver status={model.Status}"
        )

    objective_value, final_solution = build_solution(lambda variable: variable.X)

    if logger and objective_value < best_logged - 1e-7:
        logger.log_solution(objective_value, final_solution)

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(output_dir, exist_ok=True)
    temp_path = args.solution_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, separators=(",", ":"), allow_nan=False)
    os.replace(temp_path, args.solution_path)


if __name__ == "__main__":
    main()