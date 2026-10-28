import argparse
import json
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


def composite_key(node_id, day_id, scenario_id, hour, asset_id):
    return f"{node_id},{day_id},{scenario_id},{hour},{asset_id}"


def clean_continuous(value):
    value = float(value)
    if abs(value) < 1e-8:
        return 0.0
    return value


def main():
    args = parse_args()
    start_time = time.monotonic()
    deadline = start_time + max(1, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    system = data["system"]
    scenario_parameters = data["scenario_parameters"]
    global_parameters = data["global_parameters"]

    buses = list(system["buses"])
    generators = list(system.get("generators", []))
    lines = list(system.get("transmission_lines", []))
    storage_units = list(system.get("storage", []))
    tree_nodes = list(data["scenario_tree"])
    representative_days = list(data["representative_days"])
    lower_level_scenarios = data["lower_level_scenarios"]

    hours_per_day = int(scenario_parameters["hours_per_day"])
    hours = range(hours_per_day)

    gen_by_id = {str(g["id"]): g for g in generators}
    line_by_id = {str(line["id"]): line for line in lines}
    storage_by_id = {str(st["id"]): st for st in storage_units}
    node_by_id = {int(node["id"]): node for node in tree_nodes}
    day_by_id = {int(day["id"]): day for day in representative_days}

    generator_ids = list(gen_by_id)
    line_ids = list(line_by_id)
    storage_ids = list(storage_by_id)
    node_ids = [int(node["id"]) for node in tree_nodes]
    day_ids = [int(day["id"]) for day in representative_days]

    candidate_generator_ids = [
        gid for gid, g in gen_by_id.items() if not bool(g["existing"])
    ]
    candidate_line_ids = [
        lid for lid, line in line_by_id.items() if not bool(line["existing"])
    ]
    candidate_storage_ids = [
        sid for sid, st in storage_by_id.items() if not bool(st["existing"])
    ]

    generators_at_bus = {bus: [] for bus in buses}
    storage_at_bus = {bus: [] for bus in buses}
    outgoing_lines = {bus: [] for bus in buses}
    incoming_lines = {bus: [] for bus in buses}

    for gid, g in gen_by_id.items():
        generators_at_bus.setdefault(g["bus"], []).append(gid)
    for sid, st in storage_by_id.items():
        storage_at_bus.setdefault(st["bus"], []).append(sid)
    for lid, line in line_by_id.items():
        outgoing_lines.setdefault(line["from_bus"], []).append(lid)
        incoming_lines.setdefault(line["to_bus"], []).append(lid)

    peak_load_by_bus = {bus: 0.0 for bus in buses}
    for load in system.get("loads", []):
        peak_load_by_bus[load["bus"]] = (
            peak_load_by_bus.get(load["bus"], 0.0)
            + float(load["annual_peak_load_MW"])
        )

    roots = [nid for nid in node_ids if node_by_id[nid].get("parent_id") is None]
    if not roots:
        raise ValueError("The scenario tree has no root node.")

    # Build root-to-node paths and infer the scenario-dependent demand and
    # capital-cost multipliers from the ordering of children.
    ancestor_nodes = {}
    demand_multiplier_at_node = {}
    cost_multiplier_at_node = {}

    growth_high = float(scenario_parameters.get("demand_growth_high_percent", 0.0))
    growth_low = float(scenario_parameters.get("demand_growth_low_percent", 0.0))
    cost_variation = float(scenario_parameters.get("cost_variation_percent", 0.0))
    period_years = int(scenario_parameters.get("investment_period_years", 1))

    def edge_factors(parent_id, child_id):
        children = [int(x) for x in node_by_id[parent_id].get("children", [])]
        if child_id not in children:
            children = sorted(
                int(n["id"])
                for n in tree_nodes
                if n.get("parent_id") is not None
                and int(n["parent_id"]) == parent_id
            )

        count = len(children)
        index = children.index(child_id) if child_id in children else 0

        if count <= 1:
            growth = 0.5 * (growth_high + growth_low)
            capital_factor = 1.0
        elif count == 2:
            if index == 0:
                growth = growth_high
                capital_factor = 1.0 + cost_variation
            else:
                growth = growth_low
                capital_factor = 1.0 - cost_variation
        else:
            # For a Cartesian high/low branching scheme, the first half is
            # treated as high demand and alternating children as high/low cost.
            growth = growth_high if index < (count + 1) // 2 else growth_low
            capital_factor = (
                1.0 + cost_variation if index % 2 == 0
                else 1.0 - cost_variation
            )

        demand_factor = (1.0 + growth) ** period_years
        return demand_factor, max(0.0, capital_factor)

    def compute_node_factors(node_id):
        if node_id in ancestor_nodes:
            return

        node = node_by_id[node_id]
        parent = node.get("parent_id")

        if parent is None:
            ancestor_nodes[node_id] = [node_id]
            demand_multiplier_at_node[node_id] = float(
                node.get("demand_multiplier", 1.0)
            )
            cost_multiplier_at_node[node_id] = float(
                node.get("capital_cost_multiplier", 1.0)
            )
            return

        parent = int(parent)
        compute_node_factors(parent)
        demand_edge, cost_edge = edge_factors(parent, node_id)

        ancestor_nodes[node_id] = ancestor_nodes[parent] + [node_id]
        demand_multiplier_at_node[node_id] = (
            float(node["demand_multiplier"])
            if "demand_multiplier" in node
            else demand_multiplier_at_node[parent] * demand_edge
        )
        cost_multiplier_at_node[node_id] = (
            float(node["capital_cost_multiplier"])
            if "capital_cost_multiplier" in node
            else cost_multiplier_at_node[parent] * cost_edge
        )

    for nid in node_ids:
        compute_node_factors(nid)

    operational_scenarios = {}
    for did in day_ids:
        scenarios = lower_level_scenarios.get(str(did))
        if scenarios is None:
            scenarios = lower_level_scenarios.get(did)
        if scenarios is None:
            raise ValueError(f"No lower-level scenarios found for day {did}.")
        operational_scenarios[did] = scenarios

    model = gp.Model("generation_transmission_storage_expansion")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0
    model.Params.Presolve = 2
    model.Params.Heuristics = 0.20
    model.Params.MIPFocus = 1

    # Investment decisions.
    build_gen = {}
    build_line = {}
    build_storage = {}

    for nid in node_ids:
        for gid in candidate_generator_ids:
            build_gen[nid, gid] = model.addVar(vtype=GRB.BINARY)
        for lid in candidate_line_ids:
            build_line[nid, lid] = model.addVar(vtype=GRB.BINARY)
        for sid in candidate_storage_ids:
            build_storage[nid, sid] = model.addVar(vtype=GRB.BINARY)

    def available_generator(node_id, generator_id):
        return gp.quicksum(
            build_gen[a, generator_id] for a in ancestor_nodes[node_id]
        )

    def available_line(node_id, line_id):
        return gp.quicksum(
            build_line[a, line_id] for a in ancestor_nodes[node_id]
        )

    def available_storage(node_id, storage_id):
        return gp.quicksum(
            build_storage[a, storage_id] for a in ancestor_nodes[node_id]
        )

    # Each candidate can be constructed at most once along any realized path.
    for nid in node_ids:
        for gid in candidate_generator_ids:
            model.addConstr(available_generator(nid, gid) <= 1)
        for lid in candidate_line_ids:
            model.addConstr(available_line(nid, lid) <= 1)
        for sid in candidate_storage_ids:
            model.addConstr(available_storage(nid, sid) <= 1)

    alpha = {}
    gamma = {}
    p_gen = {}
    s_res = {}
    f_fwd = {}
    f_bwd = {}
    u_stor = {}
    v_stor = {}
    r_stor = {}

    operation_indices = []

    for nid in node_ids:
        for did in day_ids:
            for scenario in operational_scenarios[did]:
                scenario_id = int(scenario["scenario_id"])
                for h in hours:
                    prefix = (nid, did, scenario_id, h)
                    operation_indices.append(prefix)

                    for gid in generator_ids:
                        idx = prefix + (gid,)
                        alpha[idx] = model.addVar(vtype=GRB.BINARY)
                        gamma[idx] = model.addVar(vtype=GRB.BINARY)
                        p_gen[idx] = model.addVar(lb=0.0)
                        s_res[idx] = model.addVar(lb=0.0)

                    for lid in line_ids:
                        idx = prefix + (lid,)
                        f_fwd[idx] = model.addVar(lb=0.0)
                        f_bwd[idx] = model.addVar(lb=0.0)

                    for sid in storage_ids:
                        idx = prefix + (sid,)
                        u_stor[idx] = model.addVar(lb=0.0)
                        v_stor[idx] = model.addVar(lb=0.0)
                        r_stor[idx] = model.addVar(lb=0.0)

    model.update()

    def realized_demand(node_id, day_id, scenario, hour, bus):
        day = day_by_id[day_id]
        profile = float(day["load_profile"][hour])
        random_multiplier = float(scenario["demand_multipliers"][hour])
        return (
            peak_load_by_bus.get(bus, 0.0)
            * demand_multiplier_at_node[node_id]
            * profile
            * random_multiplier
        )

    def availability_factor(generator, day_id, scenario, hour):
        tech = str(generator.get("type", "")).lower()
        day = day_by_id[day_id]

        if "solar" in tech:
            value = (
                float(day["solar_profile"][hour])
                * float(scenario["solar_multipliers"][hour])
            )
            return max(0.0, min(1.0, value))

        if "wind" in tech:
            value = float(scenario["wind_power_factors"][hour])
            return max(0.0, min(1.0, value))

        return 1.0

    # Resource adequacy. The schema supplies one system-wide reserve margin,
    # so all buses are treated as one reserve-margin region.
    reserve_margin = float(global_parameters["reserve_margin_requirement"])
    derating_factors = global_parameters.get("derating_factors", {})

    def derating(generator):
        tech = str(generator.get("type", "")).lower()
        if tech in derating_factors:
            return float(derating_factors[tech])
        if "wind" in tech:
            return float(derating_factors.get("wind", 1.0))
        if "solar" in tech:
            return float(derating_factors.get("solar", 1.0))
        return float(derating_factors.get("fossil", 1.0))

    existing_adequacy = sum(
        derating(g) * float(g["Pmax_MW"])
        for g in generators
        if bool(g["existing"])
    )
    base_system_peak = sum(peak_load_by_bus.values())

    for nid in node_ids:
        candidate_adequacy = gp.quicksum(
            derating(gen_by_id[gid])
            * float(gen_by_id[gid]["Pmax_MW"])
            * available_generator(nid, gid)
            for gid in candidate_generator_ids
        )
        required_capacity = (
            (1.0 + reserve_margin)
            * base_system_peak
            * demand_multiplier_at_node[nid]
        )
        model.addConstr(existing_adequacy + candidate_adequacy >= required_capacity)

    spinning_reserve_percent = float(
        global_parameters["spinning_reserve_percent"]
    )

    # Operational constraints.
    for nid in node_ids:
        for did in day_ids:
            for scenario in operational_scenarios[did]:
                scenario_id = int(scenario["scenario_id"])

                for gid, g in gen_by_id.items():
                    pmin = float(g["Pmin_MW"])
                    pmax = float(g["Pmax_MW"])
                    ramp = float(g["ramp_MW_per_h"])
                    min_up = max(1, int(g.get("min_on_h", 1)))
                    min_down = max(1, int(g.get("min_off_h", 1)))

                    for h in hours:
                        idx = (nid, did, scenario_id, h, gid)
                        a = alpha[idx]
                        startup = gamma[idx]
                        power = p_gen[idx]
                        reserve = s_res[idx]
                        factor = availability_factor(g, did, scenario, h)

                        if not bool(g["existing"]):
                            model.addConstr(a <= available_generator(nid, gid))

                        model.addConstr(power >= pmin * a)
                        model.addConstr(power + reserve <= pmax * factor * a)

                        if h == 0:
                            model.addConstr(startup == a)
                            start_expression = a
                        else:
                            previous_idx = (nid, did, scenario_id, h - 1, gid)
                            previous_a = alpha[previous_idx]
                            model.addConstr(startup >= a - previous_a)
                            model.addConstr(startup <= a)
                            model.addConstr(startup <= 1 - previous_a)
                            start_expression = a - previous_a

                            previous_power = p_gen[previous_idx]
                            model.addConstr(power - previous_power <= ramp)
                            model.addConstr(previous_power - power <= ramp)

                        up_end = min(hours_per_day, h + min_up)
                        for k in range(h, up_end):
                            model.addConstr(
                                alpha[nid, did, scenario_id, k, gid]
                                >= start_expression
                            )

                        if h > 0:
                            previous_a = alpha[
                                nid, did, scenario_id, h - 1, gid
                            ]
                            shutdown_expression = previous_a - a
                            down_end = min(hours_per_day, h + min_down)
                            for k in range(h, down_end):
                                model.addConstr(
                                    alpha[nid, did, scenario_id, k, gid]
                                    <= 1 - shutdown_expression
                                )

                for sid, st in storage_by_id.items():
                    capacity = float(st["capacity_MW"])
                    efficiency = float(
                        st.get(
                            "efficiency",
                            global_parameters.get("storage_efficiency", 1.0),
                        )
                    )

                    for h in hours:
                        idx = (nid, did, scenario_id, h, sid)
                        charge = u_stor[idx]
                        discharge = v_stor[idx]
                        state = r_stor[idx]

                        if bool(st["existing"]):
                            cap_expression = capacity
                        else:
                            cap_expression = (
                                capacity * available_storage(nid, sid)
                            )

                        model.addConstr(charge <= cap_expression)
                        model.addConstr(discharge <= cap_expression)
                        model.addConstr(state <= cap_expression)
                        model.addConstr(discharge <= state)

                        if h == 0:
                            model.addConstr(state == 0)
                        else:
                            previous_idx = (
                                nid,
                                did,
                                scenario_id,
                                h - 1,
                                sid,
                            )
                            model.addConstr(
                                state
                                == r_stor[previous_idx]
                                + efficiency * u_stor[previous_idx]
                                - v_stor[previous_idx]
                            )

                for lid, line in line_by_id.items():
                    capacity = float(line["flow_limit_MW"])
                    if bool(line["existing"]):
                        cap_expression = capacity
                    else:
                        cap_expression = capacity * available_line(nid, lid)

                    for h in hours:
                        idx = (nid, did, scenario_id, h, lid)
                        model.addConstr(
                            f_fwd[idx] + f_bwd[idx] <= cap_expression
                        )

                for h in hours:
                    for bus in buses:
                        demand = realized_demand(
                            nid, did, scenario, h, bus
                        )

                        reserve_expression = gp.quicksum(
                            s_res[nid, did, scenario_id, h, gid]
                            for gid in generators_at_bus.get(bus, [])
                        )
                        model.addConstr(
                            reserve_expression
                            >= spinning_reserve_percent * demand
                        )

                        generation_expression = gp.quicksum(
                            p_gen[nid, did, scenario_id, h, gid]
                            for gid in generators_at_bus.get(bus, [])
                        )
                        storage_expression = gp.quicksum(
                            v_stor[nid, did, scenario_id, h, sid]
                            - u_stor[nid, did, scenario_id, h, sid]
                            for sid in storage_at_bus.get(bus, [])
                        )

                        exports = gp.LinExpr()
                        imports_after_losses = gp.LinExpr()

                        for lid in outgoing_lines.get(bus, []):
                            line = line_by_id[lid]
                            loss = float(line["loss_factor"])
                            exports += f_fwd[
                                nid, did, scenario_id, h, lid
                            ]
                            imports_after_losses += (
                                1.0 - loss
                            ) * f_bwd[nid, did, scenario_id, h, lid]

                        for lid in incoming_lines.get(bus, []):
                            line = line_by_id[lid]
                            loss = float(line["loss_factor"])
                            exports += f_bwd[
                                nid, did, scenario_id, h, lid
                            ]
                            imports_after_losses += (
                                1.0 - loss
                            ) * f_fwd[nid, did, scenario_id, h, lid]

                        model.addConstr(
                            generation_expression
                            + storage_expression
                            + imports_after_losses
                            - exports
                            == demand
                        )

    # Expected investment and operating cost.
    objective = gp.LinExpr()

    for nid in node_ids:
        probability = float(node_by_id[nid]["probability"])
        capital_factor = cost_multiplier_at_node[nid]

        for gid in candidate_generator_ids:
            g = gen_by_id[gid]
            unit_cost = float(g["capital_cost_per_kW"])
            investment_cost = (
                unit_cost * 1000.0 * float(g["Pmax_MW"]) * capital_factor
            )
            objective += probability * investment_cost * build_gen[nid, gid]

        for lid in candidate_line_ids:
            line = line_by_id[lid]
            unit_cost = float(line["capital_cost_per_kW"])
            investment_cost = (
                unit_cost
                * 1000.0
                * float(line["flow_limit_MW"])
                * capital_factor
            )
            objective += probability * investment_cost * build_line[nid, lid]

        for sid in candidate_storage_ids:
            st = storage_by_id[sid]
            unit_cost = float(st["capital_cost_per_kW"])
            investment_cost = (
                unit_cost
                * 1000.0
                * float(st["capacity_MW"])
                * capital_factor
            )
            objective += probability * investment_cost * build_storage[nid, sid]

        for did in day_ids:
            scenarios = operational_scenarios[did]
            scenario_weight = 1.0 / max(1, len(scenarios))

            for scenario in scenarios:
                scenario_id = int(scenario["scenario_id"])
                weight = probability * scenario_weight

                for h in hours:
                    for gid, g in gen_by_id.items():
                        idx = (nid, did, scenario_id, h, gid)
                        pmax = float(g["Pmax_MW"])
                        marginal_cost = (
                            float(g["b_cost"])
                            + float(g["c_cost"]) * pmax
                        )
                        objective += weight * (
                            float(g["startup_cost"]) * gamma[idx]
                            + float(g["a_cost"]) * alpha[idx]
                            + marginal_cost * p_gen[idx]
                        )

    model.setObjective(objective, GRB.MINIMIZE)

    output_records = {
        "alpha": [
            (composite_key(*idx), var) for idx, var in alpha.items()
        ],
        "gamma": [
            (composite_key(*idx), var) for idx, var in gamma.items()
        ],
        "p_gen": [
            (composite_key(*idx), var) for idx, var in p_gen.items()
        ],
        "s_res": [
            (composite_key(*idx), var) for idx, var in s_res.items()
        ],
        "f_fwd": [
            (composite_key(*idx), var) for idx, var in f_fwd.items()
        ],
        "f_bwd": [
            (composite_key(*idx), var) for idx, var in f_bwd.items()
        ],
        "u_stor": [
            (composite_key(*idx), var) for idx, var in u_stor.items()
        ],
        "v_stor": [
            (composite_key(*idx), var) for idx, var in v_stor.items()
        ],
        "r_stor": [
            (composite_key(*idx), var) for idx, var in r_stor.items()
        ],
    }

    binary_output_fields = {"alpha", "gamma"}

    def create_solution(get_values, objective_value):
        solution = {"objective_value": float(objective_value)}

        for field, records in output_records.items():
            variables = [var for _, var in records]
            values = get_values(variables)
            field_output = {}

            for (key, _), value in zip(records, values):
                if field in binary_output_fields:
                    field_output[key] = int(float(value) >= 0.5)
                else:
                    field_output[key] = clean_continuous(value)

            solution[field] = field_output

        investments = {}
        for nid in node_ids:
            generator_values = (
                get_values(
                    [build_gen[nid, gid] for gid in candidate_generator_ids]
                )
                if candidate_generator_ids
                else []
            )
            line_values = (
                get_values(
                    [build_line[nid, lid] for lid in candidate_line_ids]
                )
                if candidate_line_ids
                else []
            )
            storage_values = (
                get_values(
                    [build_storage[nid, sid] for sid in candidate_storage_ids]
                )
                if candidate_storage_ids
                else []
            )

            investments[str(nid)] = {
                "generators": {
                    gid: int(float(value) >= 0.5)
                    for gid, value in zip(
                        candidate_generator_ids, generator_values
                    )
                },
                "lines": {
                    lid: int(float(value) >= 0.5)
                    for lid, value in zip(candidate_line_ids, line_values)
                },
                "storage": {
                    sid: int(float(value) >= 0.5)
                    for sid, value in zip(
                        candidate_storage_ids, storage_values
                    )
                },
            }

        solution["investments"] = investments
        return solution

    callback_state = {"best_logged": float("inf")}

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL or logger is None:
            return

        try:
            incumbent_objective = float(
                cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            )
            tolerance = 1e-7 * max(1.0, abs(incumbent_objective))

            if (
                incumbent_objective
                >= callback_state["best_logged"] - tolerance
            ):
                return

            solution = create_solution(
                lambda variables: cb_model.cbGetSolution(variables),
                incumbent_objective,
            )
            logger.log_solution(incumbent_objective, solution)
            callback_state["best_logged"] = incumbent_objective
        except Exception:
            # Logging must not terminate the optimization.
            try:
                logger.log(incumbent_objective)
                callback_state["best_logged"] = incumbent_objective
            except Exception:
                pass

    remaining = deadline - time.monotonic()
    model.Params.TimeLimit = max(0.01, remaining)
    model.optimize(incumbent_callback if logger is not None else None)

    if model.SolCount <= 0:
        raise RuntimeError(
            f"No feasible solution was found. Gurobi status: {model.Status}"
        )

    final_solution = create_solution(
        lambda variables: model.getAttr("X", variables),
        model.ObjVal,
    )

    if logger is not None:
        tolerance = 1e-7 * max(1.0, abs(model.ObjVal))
        if model.ObjVal < callback_state["best_logged"] - tolerance:
            logger.log_solution(model.ObjVal, final_solution)

    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()