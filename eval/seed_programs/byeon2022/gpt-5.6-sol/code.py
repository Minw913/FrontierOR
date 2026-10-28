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


def bid_threshold(generator, bid, price_lb, price_ub):
    for key in (
        "profitability_threshold",
        "gas_price_threshold",
        "max_gas_price",
        "max_gas_price_mmBtu",
        "bid_profitability_threshold",
        "rho",
    ):
        if bid.get(key) is not None:
            return min(price_ub, max(0.0, float(bid[key])))

    fraction = generator.get("max_gas_price_fraction")
    if fraction is None:
        return price_ub
    return min(price_ub, max(price_lb, float(fraction) * float(bid["price"])))


def main():
    args = parse_args()
    start_time = time.monotonic()
    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path
        else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as stream:
        data = json.load(stream)

    if data.get("released_problem_variant") != "integrated_primal_ucgna":
        raise ValueError("Unsupported released_problem_variant")

    beta = float(data["objective_weight_beta"])
    T = int(data["time_periods"])
    periods = range(1, T + 1)

    stress = data.get("stress_parameters", {})
    eta_p = float(stress.get("eta_p", 1.0))
    eta_g = float(stress.get("eta_g", 1.0))

    buses = {
        int(x["id"]): x for x in data["electricity_network"]["buses"]
    }
    lines = {
        int(x["id"]): x for x in data["electricity_network"]["lines"]
    }
    generators = {
        int(x["id"]): x for x in data["generators"]["generators"]
    }
    junctions = {
        int(x["id"]): x for x in data["gas_network"]["junctions"]
    }
    connections = {
        int(x["id"]): x for x in data["gas_network"]["connections"]
    }
    zones = {
        int(x["id"]): x for x in data["gas_network"]["pricing_zones"]
    }

    gas_price_lb = float(data["gas_network"]["min_gas_price_mmBtu"])
    gas_price_ub = float(data["gas_network"]["max_gas_price_mmBtu"])

    generators_at_bus = {b: [] for b in buses}
    gfpps_at_junction = {j: [] for j in junctions}
    for g, generator in generators.items():
        generators_at_bus[int(generator["bus"])].append(g)
        if bool(generator["is_gfpp"]):
            gfpps_at_junction[int(generator["gas_junction"])].append(g)

    lines_out = {b: [] for b in buses}
    lines_in = {b: [] for b in buses}
    for line_id, line in lines.items():
        lines_out[int(line["from_bus"])].append(line_id)
        lines_in[int(line["to_bus"])].append(line_id)

    gas_out = {j: [] for j in junctions}
    gas_in = {j: [] for j in junctions}
    for connection_id, connection in connections.items():
        gas_out[int(connection["from_junction"])].append(connection_id)
        gas_in[int(connection["to_junction"])].append(connection_id)

    zone_of_junction = {}
    for zone_id, zone in zones.items():
        for junction_id in zone["junctions"]:
            zone_of_junction[int(junction_id)] = zone_id

    # Use Gurobi's default environment. The execution environment may provide
    # a managed/preconfigured license through the default environment; creating
    # and starting a separate Env can instead select a stale local license.
    model = gp.Model("integrated_primal_ucgna")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.NonConvex = 2
    model.Params.MIPFocus = 1

    o = {}
    vp = {}
    vm = {}
    w = {}
    r = {}
    phimax = {}
    p = {}
    se = {}
    power_flow = {}
    theta = {}
    sg = {}
    sgs = {}
    pisq = {}
    phig = {}
    lg = {}
    qg = {}
    gamma = {}
    psi = {}

    # Generator variables.
    for g, generator in generators.items():
        initial_status = int(generator["initial_status"])
        initial_gen = float(generator["initial_gen"])

        o[g, 0] = model.addVar(
            lb=initial_status,
            ub=initial_status,
            name=f"o_{g}_0",
        )
        p[g, 0] = model.addVar(
            lb=initial_gen,
            ub=initial_gen,
            name=f"p_{g}_0",
        )

        startup_cost_ub = max(
            [float(x[1]) for x in generator.get("startup_cost_params", [])]
            + [0.0]
        )

        for t in periods:
            o[g, t] = model.addVar(vtype=GRB.BINARY, name=f"o_{g}_{t}")
            vp[g, t] = model.addVar(vtype=GRB.BINARY, name=f"vp_{g}_{t}")
            vm[g, t] = model.addVar(vtype=GRB.BINARY, name=f"vm_{g}_{t}")
            r[g, t] = model.addVar(
                lb=0.0,
                ub=startup_cost_ub,
                name=f"r_{g}_{t}",
            )
            p[g, t] = model.addVar(
                lb=0.0,
                ub=float(generator["max_power"]),
                name=f"p_{g}_{t}",
            )

            for bid in generator["bids"]:
                bid_id = int(bid["id"])
                w[g, bid_id, t] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"w_{g}_{bid_id}_{t}",
                )
                se[g, bid_id, t] = model.addVar(
                    lb=0.0,
                    ub=float(bid["max_amount"]),
                    name=f"se_{g}_{bid_id}_{t}",
                )

            if bool(generator["is_gfpp"]):
                phimax[g, t] = model.addVar(
                    lb=0.0,
                    ub=gas_price_ub,
                    name=f"phimax_{g}_{t}",
                )

    # Electricity network variables.
    for line_id, line in lines.items():
        limit = float(line["thermal_limit"])
        for t in periods:
            power_flow[line_id, t] = model.addVar(
                lb=-limit,
                ub=limit,
                name=f"f_{line_id}_{t}",
            )

    for bus_id, bus in buses.items():
        for t in periods:
            theta[bus_id, t] = model.addVar(
                lb=float(bus["voltage_angle_lb"]),
                ub=float(bus["voltage_angle_ub"]),
                name=f"theta_{bus_id}_{t}",
            )

    # Gas variables.
    for junction_id, junction in junctions.items():
        for t in periods:
            demand = eta_g * float(junction["gas_demand_profile"][t - 1])

            sg[junction_id, t] = model.addVar(
                lb=0.0, name=f"sg_{junction_id}_{t}"
            )
            pisq[junction_id, t] = model.addVar(
                lb=float(junction["pressure_lb_squared"]),
                ub=float(junction["pressure_ub_squared"]),
                name=f"pisq_{junction_id}_{t}",
            )
            lg[junction_id, t] = model.addVar(
                lb=0.0,
                ub=demand,
                name=f"lg_{junction_id}_{t}",
            )
            qg[junction_id, t] = model.addVar(
                lb=0.0,
                ub=demand,
                name=f"qg_{junction_id}_{t}",
            )
            gamma[junction_id, t] = model.addVar(
                lb=0.0,
                name=f"gamma_{junction_id}_{t}",
            )

            if bool(junction["is_source"]):
                for interval in junction.get("supply_intervals", []):
                    interval_id = int(interval["id"])
                    capacity = max(
                        0.0,
                        float(interval["interval_ub"])
                        - float(interval["interval_lb"]),
                    )
                    sgs[junction_id, interval_id, t] = model.addVar(
                        lb=0.0,
                        ub=capacity,
                        name=f"sgs_{junction_id}_{interval_id}_{t}",
                    )

    for connection_id in connections:
        for t in periods:
            phig[connection_id, t] = model.addVar(
                lb=0.0,
                name=f"phig_{connection_id}_{t}",
            )

    # Gas prices carry no direct objective cost. Fixing them to the global
    # lower bound is optimal and makes all bid-validity rules least restrictive.
    for zone_id in zones:
        for t in periods:
            psi[zone_id, t] = model.addVar(
                lb=gas_price_lb,
                ub=gas_price_lb,
                name=f"psi_{zone_id}_{t}",
            )

    model.update()

    # Commitment constraints.
    for g, generator in generators.items():
        initial_status = int(generator["initial_status"])
        minimum_up = int(generator["min_up_time"])
        minimum_down = int(generator["min_down_time"])

        if initial_status == 1:
            required_initial_periods = max(
                1,
                minimum_up - int(generator.get("initial_active_periods", 0)),
            )
        else:
            required_initial_periods = max(
                1,
                minimum_down - int(generator.get("initial_inactive_periods", 0)),
            )

        for t in range(1, min(T, required_initial_periods) + 1):
            model.addConstr(o[g, t] == initial_status)

        for t in periods:
            model.addConstr(
                vp[g, t] - vm[g, t] == o[g, t] - o[g, t - 1]
            )
            model.addConstr(vp[g, t] + vm[g, t] <= 1)

            if startup_cost_ub > 0.0:
                model.addConstr(r[g, t] <= startup_cost_ub * vp[g, t])

            for tier in generator.get("startup_cost_params", []):
                lookback = int(tier[0])
                cost = float(tier[1])
                historical_online = gp.LinExpr()
                historical_constant = 0.0

                for tau in range(t - lookback, t):
                    if tau >= 0:
                        historical_online += o[g, tau]
                    elif initial_status == 1:
                        active = int(generator.get("initial_active_periods", 0))
                        if tau >= -active:
                            historical_constant += 1.0
                    else:
                        inactive = int(
                            generator.get("initial_inactive_periods", 0)
                        )
                        if tau < -inactive:
                            historical_constant += 1.0

                model.addConstr(
                    r[g, t]
                    >= cost
                    * (
                        o[g, t]
                        - historical_online
                        - historical_constant
                    )
                )

        if minimum_up > 0:
            for start in range(1, T - minimum_up + 2):
                end = start + minimum_up - 1
                model.addConstr(
                    gp.quicksum(vp[g, tau] for tau in range(start, end + 1))
                    <= o[g, end]
                )

        if minimum_down > 0:
            for start in range(1, T - minimum_down + 2):
                end = start + minimum_down - 1
                model.addConstr(
                    gp.quicksum(vp[g, tau] for tau in range(start, end + 1))
                    <= 1 - o[g, start - 1]
                )

    # Generator dispatch, bid selection, and ramping.
    for g, generator in generators.items():
        minimum_power = float(generator["min_power"])
        maximum_power = float(generator["max_power"])
        ramp_up = float(generator["ramp_up"])
        ramp_down = float(generator["ramp_down"])
        bids = list(generator["bids"])

        for t in periods:
            model.addConstr(
                p[g, t]
                == gp.quicksum(se[g, int(bid["id"]), t] for bid in bids)
            )
            model.addConstr(p[g, t] >= minimum_power * o[g, t])
            model.addConstr(p[g, t] <= maximum_power * o[g, t])

            model.addConstr(
                p[g, t] - p[g, t - 1]
                <= ramp_up * o[g, t - 1] + maximum_power * vp[g, t]
            )
            model.addConstr(
                p[g, t - 1] - p[g, t]
                <= ramp_down * o[g, t] + minimum_power * vm[g, t]
            )

            if bool(generator["is_gfpp"]):
                threshold_expr = gp.LinExpr()

                for bid_index, bid in enumerate(bids):
                    bid_id = int(bid["id"])
                    amount = float(bid["max_amount"])
                    threshold = bid_threshold(
                        generator,
                        bid,
                        gas_price_lb,
                        gas_price_ub,
                    )

                    model.addConstr(w[g, bid_id, t] <= o[g, t])
                    model.addConstr(
                        se[g, bid_id, t] <= amount * w[g, bid_id, t]
                    )

                    if bid_index + 1 < len(bids):
                        next_bid_id = int(bids[bid_index + 1]["id"])
                        model.addConstr(
                            w[g, next_bid_id, t] <= w[g, bid_id, t]
                        )
                        model.addConstr(
                            se[g, bid_id, t]
                            >= amount * w[g, next_bid_id, t]
                        )
                        threshold_expr += threshold * (
                            w[g, bid_id, t] - w[g, next_bid_id, t]
                        )
                    else:
                        threshold_expr += threshold * w[g, bid_id, t]

                model.addConstr(phimax[g, t] == threshold_expr)

                junction_id = int(generator["gas_junction"])
                zone_id = zone_of_junction[junction_id]
                model.addConstr(
                    phimax[g, t]
                    >= psi[zone_id, t]
                    - gas_price_ub * (1 - o[g, t])
                )
            else:
                for bid in bids:
                    model.addConstr(w[g, int(bid["id"]), t] == 0)

    # Electricity balance.
    for bus_id, bus in buses.items():
        for t in periods:
            demand = eta_p * float(bus["demand_profile"][t - 1])
            model.addConstr(
                gp.quicksum(p[g, t] for g in generators_at_bus[bus_id])
                - demand
                == gp.quicksum(
                    power_flow[line_id, t]
                    for line_id in lines_out[bus_id]
                )
                - gp.quicksum(
                    power_flow[line_id, t]
                    for line_id in lines_in[bus_id]
                )
            )

    # DC power-flow constraints.
    for line_id, line in lines.items():
        from_bus = int(line["from_bus"])
        to_bus = int(line["to_bus"])
        susceptance = float(line["susceptance"])
        angle_limit = float(line["angle_diff_limit"])

        for t in periods:
            difference = theta[from_bus, t] - theta[to_bus, t]
            model.addConstr(
                power_flow[line_id, t] == susceptance * difference
            )
            model.addConstr(difference <= angle_limit)
            model.addConstr(difference >= -angle_limit)

    # Fix one reference angle if zero is within the reference bus bounds.
    if buses:
        reference_bus = min(buses)
        reference = buses[reference_bus]
        if (
            float(reference["voltage_angle_lb"]) <= 0.0
            <= float(reference["voltage_angle_ub"])
        ):
            for t in periods:
                model.addConstr(theta[reference_bus, t] == 0.0)

    # Gas balance, supply, demand, and heat-rate constraints.
    for junction_id, junction in junctions.items():
        for t in periods:
            demand = eta_g * float(junction["gas_demand_profile"][t - 1])

            model.addConstr(
                lg[junction_id, t] + qg[junction_id, t] == demand
            )

            if bool(junction["is_source"]):
                model.addConstr(
                    sg[junction_id, t]
                    == gp.quicksum(
                        sgs[junction_id, int(interval["id"]), t]
                        for interval in junction.get("supply_intervals", [])
                    )
                )
            else:
                model.addConstr(sg[junction_id, t] == 0.0)

            model.addConstr(
                sg[junction_id, t]
                - lg[junction_id, t]
                - gamma[junction_id, t]
                == gp.quicksum(
                    phig[c, t] for c in gas_out[junction_id]
                )
                - gp.quicksum(
                    phig[c, t] for c in gas_in[junction_id]
                )
            )

            gfpp_ids = gfpps_at_junction[junction_id]
            if gfpp_ids:
                heat = gp.QuadExpr()
                for g in gfpp_ids:
                    coefficients = generators[g]["heat_rate_coefficients"]
                    h2 = float(coefficients.get("H_u2") or 0.0)
                    h1 = float(coefficients.get("H_u1") or 0.0)
                    h0 = float(coefficients.get("H_u0") or 0.0)
                    heat += h2 * p[g, t] * p[g, t]
                    heat += h1 * p[g, t]
                    heat += h0 * o[g, t]

                model.addQConstr(gamma[junction_id, t] >= heat)
            else:
                model.addConstr(gamma[junction_id, t] == 0.0)

    # Gas connection pressure constraints.
    for connection_id, connection in connections.items():
        from_junction = int(connection["from_junction"])
        to_junction = int(connection["to_junction"])
        connection_type = str(connection.get("type", "")).lower()

        for t in periods:
            control_lb = connection.get("control_ratio_lb")
            control_ub = connection.get("control_ratio_ub")

            if control_lb is not None and control_ub is not None:
                model.addConstr(
                    pisq[to_junction, t]
                    >= float(control_lb) ** 2 * pisq[from_junction, t]
                )
                model.addConstr(
                    pisq[to_junction, t]
                    <= float(control_ub) ** 2 * pisq[from_junction, t]
                )
            elif connection_type == "compressor":
                ratio_lb = float(connection["compression_ratio_lb"])
                ratio_ub = float(connection["compression_ratio_ub"])
                model.addConstr(
                    pisq[to_junction, t]
                    >= ratio_lb ** 2 * pisq[from_junction, t]
                )
                model.addConstr(
                    pisq[to_junction, t]
                    <= ratio_ub ** 2 * pisq[from_junction, t]
                )
            else:
                factor = float(connection["weymouth_factor"])
                model.addQConstr(
                    factor * phig[connection_id, t] * phig[connection_id, t]
                    <= pisq[from_junction, t] - pisq[to_junction, t]
                )

    electricity_cost = gp.LinExpr()
    gas_cost = gp.LinExpr()

    for g, generator in generators.items():
        for t in periods:
            electricity_cost += (
                float(generator["no_load_cost"]) * o[g, t] + r[g, t]
            )
            for bid in generator["bids"]:
                bid_id = int(bid["id"])
                electricity_cost += (
                    float(bid["price"]) * se[g, bid_id, t]
                )

    for junction_id, junction in junctions.items():
        for t in periods:
            gas_cost += (
                float(junction["demand_shedding_cost"])
                * qg[junction_id, t]
            )
            if bool(junction["is_source"]):
                for interval in junction.get("supply_intervals", []):
                    interval_id = int(interval["id"])
                    gas_cost += (
                        float(interval["slope"])
                        * sgs[junction_id, interval_id, t]
                    )

    model.setObjective(
        beta * electricity_cost + (1.0 - beta) * gas_cost,
        GRB.MINIMIZE,
    )

    variable_maps = {
        "o": o,
        "vp": vp,
        "vm": vm,
        "w": w,
        "r": r,
        "phimax": phimax,
        "p": p,
        "se": se,
        "f": power_flow,
        "theta": theta,
        "sg": sg,
        "sgs": sgs,
        "pisq": pisq,
        "phig": phig,
        "lg": lg,
        "qg": qg,
        "gamma": gamma,
        "psi": psi,
    }

    def clean_value(value, binary=False):
        value = float(value)
        if binary:
            return 1.0 if value >= 0.5 else 0.0
        return 0.0 if abs(value) <= 1e-9 else value

    def construct_solution(value_getter):
        primary = {}
        binary_prefixes = {"o", "vp", "vm", "w"}

        for prefix, variables in variable_maps.items():
            for indices, variable in variables.items():
                index_tuple = indices if isinstance(indices, tuple) else (indices,)
                key = prefix + "_" + "_".join(map(str, index_tuple))
                primary[key] = clean_value(
                    value_getter(variable),
                    prefix in binary_prefixes,
                )

        electricity_value = 0.0
        for g, generator in generators.items():
            for t in periods:
                electricity_value += (
                    float(generator["no_load_cost"])
                    * primary[f"o_{g}_{t}"]
                )
                electricity_value += primary[f"r_{g}_{t}"]
                for bid in generator["bids"]:
                    bid_id = int(bid["id"])
                    electricity_value += (
                        float(bid["price"])
                        * primary[f"se_{g}_{bid_id}_{t}"]
                    )

        gas_value = 0.0
        for junction_id, junction in junctions.items():
            for t in periods:
                gas_value += (
                    float(junction["demand_shedding_cost"])
                    * primary[f"qg_{junction_id}_{t}"]
                )
                if bool(junction["is_source"]):
                    for interval in junction.get("supply_intervals", []):
                        interval_id = int(interval["id"])
                        gas_value += (
                            float(interval["slope"])
                            * primary[
                                f"sgs_{junction_id}_{interval_id}_{t}"
                            ]
                        )

        objective = beta * electricity_value + (1.0 - beta) * gas_value
        solution = {
            "objective_value": float(objective),
            "released_problem_variant": "integrated_primal_ucgna",
            "objective_weight_beta": beta,
            "primary_variables": primary,
        }
        return float(objective), solution

    best_logged = math.inf

    def callback(callback_model, where):
        nonlocal best_logged
        if where != GRB.Callback.MIPSOL:
            return

        try:
            incumbent = callback_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if incumbent >= best_logged - 1e-7:
                return

            if logger:
                objective, solution = construct_solution(
                    lambda variable: callback_model.cbGetSolution(variable)
                )
                if objective < best_logged - 1e-7:
                    logger.log_solution(objective, solution)
                    best_logged = objective
            else:
                best_logged = incumbent
        except Exception:
            # Logging errors must not interrupt optimization.
            pass

    remaining = max(
        0.01,
        float(args.time_limit) - (time.monotonic() - start_time),
    )
    model.Params.TimeLimit = remaining
    model.optimize(callback)

    if model.SolCount <= 0:
        status = model.Status
        model.dispose()
        raise RuntimeError(
            f"No feasible solution was found; Gurobi status is {status}"
        )

    objective, solution = construct_solution(lambda variable: variable.X)

    if logger and objective < best_logged - 1e-7:
        logger.log_solution(objective, solution)

    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(output_directory, exist_ok=True)

    temporary_path = args.solution_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as stream:
        json.dump(
            solution,
            stream,
            separators=(",", ":"),
            allow_nan=False,
        )
    os.replace(temporary_path, args.solution_path)

    model.dispose()


if __name__ == "__main__":
    main()