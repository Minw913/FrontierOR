import argparse
import json
import math
import time
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger


def to_int_keyed_dict(d):
    return {int(k): v for k, v in d.items()}


def safe_get_series(dct, key, n):
    if key in dct:
        return dct[key]
    sk = str(key)
    if sk in dct:
        return dct[sk]
    return [0.0] * n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    pp = inst["problem_parameters"]
    Z = pp["num_zones"]
    W = pp["num_weeks"]
    H = pp["hours_per_subperiod"]
    T = pp["total_hours"]

    subperiod_weights = inst["subperiod_weights"]
    assert len(subperiod_weights) == W

    generators = inst["generators"]
    gen_params = inst["generator_parameters"]
    existing_capacity = to_int_keyed_dict(inst["existing_capacity"])
    demand_raw = to_int_keyed_dict(inst["demand"])
    availability_raw = inst["availability"]
    hydro_inflow_raw = inst.get("hydro_inflow", {})

    transmission_lines = inst["transmission_lines"]
    consumer_segments = inst["consumer_segments"]
    policy = inst["policy"]

    # Indices and mappings
    cluster_ids = [g["cluster_id"] for g in generators]
    gen_by_id = {g["cluster_id"]: g for g in generators}
    zones = sorted(list({g["zone"] for g in generators} | set(demand_raw.keys())))
    lines = [l["line_id"] for l in transmission_lines]
    line_by_id = {l["line_id"]: l for l in transmission_lines}

    uc_ids = [i for i in cluster_ids if gen_by_id[i]["is_UC"]]
    storage_ids = [i for i in cluster_ids if gen_by_id[i]["is_storage"]]
    hydro_ids = [i for i in cluster_ids if gen_by_id[i]["is_hydro"]]
    non_uc_ids = [i for i in cluster_ids if not gen_by_id[i]["is_UC"]]
    rps_ids = [i for i in cluster_ids if gen_by_id[i]["is_RPS_qualifying"]]

    # Precompute params per cluster
    p = {}
    for i in cluster_ids:
        g = gen_by_id[i]
        rt = g["resource_type"]
        gp = gen_params[rt]
        p[i] = {
            "zone": g["zone"],
            "rt": rt,
            "is_UC": g["is_UC"],
            "is_storage": g["is_storage"],
            "is_hydro": g["is_hydro"],
            "is_RPS": g["is_RPS_qualifying"],
            "can_retire": g["can_retire"],
            "unit_MW": float(g["capacity_size_MW"]),
            "min_out": float(gp.get("min_output_frac", 0.0)),
            "ru": float(gp.get("ramp_up_rate_pct_per_hr", 1e6)),
            "rd": float(gp.get("ramp_down_rate_pct_per_hr", 1e6)),
            "mut": int(gp.get("min_up_time_hr", 0)),
            "mdt": int(gp.get("min_down_time_hr", 0)),
            "var_cost": float(gp.get("variable_cost_per_MWh", 0.0)),
            "fix_om": float(gp.get("fixed_OM_cost_per_MW_yr", 0.0)),
            "inv_cost": float(gp.get("investment_cost_per_MW_yr", 0.0)),
            "startup_cost": float(gp.get("start_up_cost", 0.0)),
            "co2": float(gp.get("CO2_emission_tons_per_MWh", 0.0)),
            "max_new_MW": float(gp.get("max_capacity_MW", 1e12)),
        }
        if g["is_storage"]:
            bp = gen_params["battery"] if rt == "battery" else gp
            p[i].update({
                "e_unit_MWh": float(bp.get("energy_capacity_size_MWh", 0.0)),
                "eta_c": float(bp.get("charging_efficiency", 1.0)),
                "eta_d": float(bp.get("discharging_efficiency", 1.0)),
                "self_dis": float(bp.get("self_discharge_rate_pct", 0.0)),
                "min_dur": float(bp.get("min_duration_MWh_per_MW", 0.0)),
                "max_dur": float(bp.get("max_duration_MWh_per_MW", 1e12)),
                "e_inv_cost": float(bp.get("energy_investment_cost_per_MWh_yr", 0.0)),
                "e_fix_om": float(bp.get("energy_fixed_OM_cost_per_MWh_yr", 0.0)),
                "max_new_E": float(bp.get("max_energy_capacity_MWh", 1e12)),
            })
        if g["is_hydro"]:
            hp = gen_params["hydro"] if rt == "hydro" else gp
            p[i].update({
                "dur": float(hp.get("duration_MWh_per_MW", 0.0)),
                "e_inv_cost_h": float(hp.get("energy_investment_cost_per_MWh_yr", 0.0)),
                "e_fix_om_h": float(hp.get("energy_fixed_OM_cost_per_MWh_yr", 0.0)),
            })

    demand = {}
    for z in zones:
        dz = demand_raw.get(z, demand_raw.get(int(z), demand_raw.get(str(z), [0.0] * T)))
        demand[z] = dz

    availability = {i: safe_get_series(availability_raw, i, T) for i in cluster_ids}
    hydro_inflow = {i: safe_get_series(hydro_inflow_raw, i, T) for i in hydro_ids}

    # Sets by zone
    gens_in_zone = {z: [i for i in cluster_ids if p[i]["zone"] == z] for z in zones}
    stor_in_zone = {z: [i for i in storage_ids if p[i]["zone"] == z] for z in zones}
    segs_in_zone = {z: [s for s in consumer_segments if s["zone"] == z] for z in zones}

    in_lines = {z: [] for z in zones}
    out_lines = {z: [] for z in zones}
    for l in transmission_lines:
        out_lines[l["from_zone"]].append(l["line_id"])
        in_lines[l["to_zone"]].append(l["line_id"])

    # Time helpers
    week_of_t = [t // H for t in range(T)]
    hour_of_t = [t % H for t in range(T)]

    def prev_in_week(t):
        w = week_of_t[t]
        h = hour_of_t[t]
        return w * H + (h - 1 if h > 0 else H - 1)

    # Build model
    m = Model("power_planning")
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = args.time_limit

    # Investment variables
    y_P_new, y_P_ret, y_P = {}, {}, {}
    y_E_new, y_E_ret, y_E = {}, {}, {}
    y_F_new, y_F = {}, {}

    for i in cluster_ids:
        inc = p[i]["unit_MW"] if p[i]["is_UC"] else 1.0
        max_new_units = math.floor(max(0.0, p[i]["max_new_MW"]) / inc + 1e-9)
        ex = float(existing_capacity.get(i, 0.0))
        max_ret_units = math.floor(max(0.0, ex) / inc + 1e-9)

        y_P_new[i] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=max_new_units)
        ret_ub = max_ret_units if p[i]["can_retire"] else 0
        y_P_ret[i] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=ret_ub)
        y_P[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0)

        m.addConstr(y_P[i] == ex + inc * (y_P_new[i] - y_P_ret[i]))

    for i in storage_ids:
        incE = p[i]["e_unit_MWh"] if p[i]["is_UC"] else 1.0
        exP = float(existing_capacity.get(i, 0.0))
        ratio = p[i]["e_unit_MWh"] / max(1e-9, p[i]["unit_MW"])
        exE = exP * ratio

        max_new_e_units = math.floor(max(0.0, p[i]["max_new_E"]) / incE + 1e-9)
        max_ret_e_units = math.floor(max(0.0, exE) / incE + 1e-9)

        y_E_new[i] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=max_new_e_units)
        ret_ub = max_ret_e_units if p[i]["can_retire"] else 0
        y_E_ret[i] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=ret_ub)
        y_E[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0)

        m.addConstr(y_E[i] == exE + incE * (y_E_new[i] - y_E_ret[i]))
        m.addConstr(y_E[i] >= p[i]["min_dur"] * y_P[i])
        m.addConstr(y_E[i] <= p[i]["max_dur"] * y_P[i])

    for l in lines:
        dat = line_by_id[l]
        ex = float(dat["existing_capacity_MW"])
        max_cap = float(dat["max_capacity_MW"])
        ub_new = max(0.0, max_cap - ex)
        y_F_new[l] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=math.floor(ub_new + 1e-9))
        y_F[l] = m.addVar(vtype=GRB.CONTINUOUS, lb=0)
        m.addConstr(y_F[l] == ex + y_F_new[l])
        m.addConstr(y_F[l] <= max_cap)

    # Operations
    x_inj = {(i, t): m.addVar(lb=0) for i in cluster_ids for t in range(T)}
    x_wdw, x_soc = {}, {}
    for i in storage_ids:
        for t in range(T):
            x_wdw[(i, t)] = m.addVar(lb=0)
            x_soc[(i, t)] = m.addVar(lb=0)

    x_lvl, x_spill = {}, {}
    for i in hydro_ids:
        for t in range(T):
            x_lvl[(i, t)] = m.addVar(lb=0)
            x_spill[(i, t)] = m.addVar(lb=0)

    x_flow = {(l, t): m.addVar(lb=-GRB.INFINITY) for l in lines for t in range(T)}

    x_commit, x_start, x_shut = {}, {}, {}
    for i in uc_ids:
        for t in range(T):
            x_commit[(i, t)] = m.addVar(lb=0)
            x_start[(i, t)] = m.addVar(lb=0)
            x_shut[(i, t)] = m.addVar(lb=0)

    x_nse = {}
    for seg in consumer_segments:
        s = seg["segment_id"]
        z = seg["zone"]
        frac = float(seg["max_nse_fraction"])
        for t in range(T):
            ub = frac * float(demand[z][t])
            x_nse[(s, z, t)] = m.addVar(lb=0, ub=ub)

    x_RPS = {w: m.addVar(lb=0) for w in range(W)}
    x_CO2 = {w: m.addVar(lb=0) for w in range(W)}

    m.update()

    # Core constraints
    for i in cluster_ids:
        min_out = p[i]["min_out"]
        for t in range(T):
            av = float(availability[i][t])

            if p[i]["is_UC"]:
                u = p[i]["unit_MW"]
                m.addConstr(u * x_commit[(i, t)] <= y_P[i])
                m.addConstr(u * x_start[(i, t)] <= y_P[i])
                m.addConstr(u * x_shut[(i, t)] <= y_P[i])

                m.addConstr(x_inj[(i, t)] <= u * av * x_commit[(i, t)])
                m.addConstr(x_inj[(i, t)] >= u * min_out * x_commit[(i, t)])

                tp = prev_in_week(t)
                m.addConstr(
                    x_commit[(i, t)] - x_commit[(i, tp)] ==
                    x_start[(i, t)] - x_shut[(i, t)]
                )
            else:
                m.addConstr(x_inj[(i, t)] <= av * y_P[i])
                if (not p[i]["is_storage"]) and (not p[i]["is_hydro"]) and min_out > 0:
                    m.addConstr(x_inj[(i, t)] >= min_out * y_P[i])

    # Storage dynamics and limits
    for i in storage_ids:
        eta_c = max(1e-9, p[i]["eta_c"])
        eta_d = max(1e-9, p[i]["eta_d"])
        sd = p[i]["self_dis"]
        for t in range(T):
            tp = prev_in_week(t)
            av = float(availability[i][t])

            m.addConstr(x_soc[(i, t)] <= y_E[i])
            m.addConstr(x_wdw[(i, t)] <= av * y_P[i])
            m.addConstr(x_inj[(i, t)] + x_wdw[(i, t)] <= y_P[i])

            m.addConstr(
                x_soc[(i, t)] ==
                x_soc[(i, tp)] + eta_c * x_wdw[(i, t)] - (1.0 / eta_d) * x_inj[(i, t)] - sd * x_soc[(i, tp)]
            )
            m.addConstr((1.0 / eta_d) * x_inj[(i, t)] <= x_soc[(i, tp)])
            m.addConstr(eta_c * x_wdw[(i, t)] <= y_E[i] - x_soc[(i, tp)])

    # Hydro dynamics
    for i in hydro_ids:
        dur = p[i]["dur"]
        min_out = p[i]["min_out"]
        infl = hydro_inflow[i]
        for t in range(T):
            tp = prev_in_week(t)
            m.addConstr(x_lvl[(i, t)] <= dur * y_P[i])
            m.addConstr(
                x_lvl[(i, t)] ==
                x_lvl[(i, tp)] + float(infl[t]) * y_P[i] - x_inj[(i, t)] - x_spill[(i, t)]
            )
            m.addConstr(x_inj[(i, t)] + x_spill[(i, t)] >= min_out * y_P[i])

    # Ramping
    for i in non_uc_ids:
        ru, rd = p[i]["ru"], p[i]["rd"]
        for t in range(T):
            tp = prev_in_week(t)
            m.addConstr(x_inj[(i, t)] - x_inj[(i, tp)] <= ru * y_P[i])
            m.addConstr(x_inj[(i, tp)] - x_inj[(i, t)] <= rd * y_P[i])

    for i in uc_ids:
        u = p[i]["unit_MW"]
        ru, rd = p[i]["ru"], p[i]["rd"]
        min_out = p[i]["min_out"]
        for t in range(T):
            tp = prev_in_week(t)
            av_t = float(availability[i][t])
            av_p = float(availability[i][tp])

            alpha_up = min(av_t, max(min_out, ru))
            alpha_dn = min(av_p, max(min_out, rd))

            m.addConstr(
                x_inj[(i, t)] - x_inj[(i, tp)] <=
                ru * u * (x_commit[(i, tp)] - x_shut[(i, t)]) +
                u * alpha_up * x_start[(i, t)] -
                u * min_out * x_shut[(i, t)]
            )
            m.addConstr(
                x_inj[(i, tp)] - x_inj[(i, t)] <=
                rd * u * (x_commit[(i, t)] - x_start[(i, t)]) +
                u * alpha_dn * x_shut[(i, t)] -
                u * min_out * x_start[(i, t)]
            )

    # Min up/down with wrap within week
    for i in uc_ids:
        mut = p[i]["mut"]
        mdt = p[i]["mdt"]
        u = p[i]["unit_MW"]

        for t in range(T):
            w = week_of_t[t]
            h = hour_of_t[t]

            if mut > 0:
                idxs = [w * H + ((h - k) % H) for k in range(mut)]
                m.addConstr(x_commit[(i, t)] >= quicksum(x_start[(i, tt)] for tt in idxs))
            if mdt > 0:
                idxs = [w * H + ((h - k) % H) for k in range(mdt)]
                m.addConstr(
                    x_commit[(i, t)] + quicksum(x_shut[(i, tt)] for tt in idxs) <= y_P[i] / u
                )

    # Transmission bounds
    for l in lines:
        for t in range(T):
            m.addConstr(x_flow[(l, t)] <= y_F[l])
            m.addConstr(x_flow[(l, t)] >= -y_F[l])

    # Power balance
    for z in zones:
        for t in range(T):
            gen_sum = quicksum(x_inj[(i, t)] for i in gens_in_zone[z])
            wdw_sum = quicksum(x_wdw[(i, t)] for i in stor_in_zone[z]) if stor_in_zone[z] else 0.0
            imports = quicksum(x_flow[(l, t)] for l in in_lines[z]) - quicksum(x_flow[(l, t)] for l in out_lines[z])
            nse_sum = quicksum(x_nse[(seg["segment_id"], z, t)] for seg in segs_in_zone[z]) if segs_in_zone[z] else 0.0
            m.addConstr(gen_sum - wdw_sum + imports + nse_sum == float(demand[z][t]))

    # Policy constraints
    hourly_weight = [subperiod_weights[week_of_t[t]] / H for t in range(T)]
    total_weighted_demand = sum(hourly_weight[t] * float(demand[z][t]) for z in zones for t in range(T))

    weighted_rps_gen = quicksum(hourly_weight[t] * x_inj[(i, t)] for i in rps_ids for t in range(T))
    weighted_emis = quicksum(
        hourly_weight[t] * p[i]["co2"] * x_inj[(i, t)] for i in cluster_ids for t in range(T)
    ) + quicksum(
        hourly_weight[t] * p[i]["co2"] * x_wdw[(i, t)] for i in storage_ids for t in range(T)
    )

    ptype = policy["type"].upper()
    if ptype == "RPS":
        m.addConstr(weighted_rps_gen + quicksum(x_RPS[w] for w in range(W)) >= float(policy["RPS_share"]) * total_weighted_demand)
        for w in range(W):
            m.addConstr(x_CO2[w] == 0)
    elif ptype == "CO2":
        m.addConstr(weighted_emis - quicksum(x_CO2[w] for w in range(W)) <= float(policy["CO2_cap_tons_per_MWh"]) * total_weighted_demand)
        for w in range(W):
            m.addConstr(x_RPS[w] == 0)
    else:  # REF
        for w in range(W):
            m.addConstr(x_RPS[w] == 0)
            m.addConstr(x_CO2[w] == 0)

    # Objective
    fixed_cost = 0
    for i in cluster_ids:
        inc = p[i]["unit_MW"] if p[i]["is_UC"] else 1.0
        new_MW_expr = inc * y_P_new[i]
        fixed_cost += p[i]["inv_cost"] * new_MW_expr
        fixed_cost += p[i]["fix_om"] * y_P[i]

        if p[i]["is_storage"]:
            incE = p[i]["e_unit_MWh"] if p[i]["is_UC"] else 1.0
            new_E_expr = incE * y_E_new[i]
            fixed_cost += p[i]["e_inv_cost"] * new_E_expr
            fixed_cost += p[i]["e_fix_om"] * y_E[i]

        if p[i]["is_hydro"]:
            fixed_cost += p[i]["e_inv_cost_h"] * p[i]["dur"] * new_MW_expr
            fixed_cost += p[i]["e_fix_om_h"] * p[i]["dur"] * y_P[i]

    fixed_cost += quicksum(float(line_by_id[l]["investment_cost_per_MW_yr"]) * y_F_new[l] for l in lines)

    var_cost = quicksum(hourly_weight[t] * p[i]["var_cost"] * x_inj[(i, t)] for i in cluster_ids for t in range(T))
    var_cost += quicksum(hourly_weight[t] * p[i]["var_cost"] * x_wdw[(i, t)] for i in storage_ids for t in range(T))

    nse_cost = quicksum(
        hourly_weight[t] * float(seg["cost_per_MWh"]) * x_nse[(seg["segment_id"], seg["zone"], t)]
        for seg in consumer_segments for t in range(T)
    )

    startup_cost = quicksum(
        hourly_weight[t] * p[i]["startup_cost"] * x_start[(i, t)] for i in uc_ids for t in range(T)
    )

    policy_cost = float(policy.get("RPS_noncompliance_cost_per_MWh", 0.0)) * quicksum(x_RPS[w] for w in range(W)) + \
                  float(policy.get("CO2_noncompliance_cost_per_ton", 0.0)) * quicksum(x_CO2[w] for w in range(W))

    m.setObjective(fixed_cost + var_cost + nse_cost + startup_cost + policy_cost, GRB.MINIMIZE)

    # Callback for incumbent logging
    def cb(model, where):
        if logger is None:
            return
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            logger.log(obj)

    m.optimize(cb if logger else None)

    # Build output
    sol = {
        "objective_value": None,
        "model_variables": {},
        "investments": {},
        "storage_energy_investments": {},
        "transmission_investments": {},
        "model_variable_name_patterns": {
            "y_P_new[i], y_P_ret[i], y_P[i]": "generator capacity variables",
            "y_E_new[i], y_E_ret[i], y_E[i]": "storage-energy variables",
            "y_F_new[l], y_F[l]": "transmission variables",
            "x_inj[i,t], x_wdw[i,t], x_soc[i,t]": "generation/storage operations",
            "x_nse[s,z,t]": "non-served energy",
            "x_lvl[i,t], x_spill[i,t]": "hydro operations",
            "x_flow[l,t]": "transmission flow",
            "x_commit[i,t], x_start[i,t], x_shut[i,t]": "unit commitment",
            "x_RPS[w], x_CO2[w]": "policy slack variables"
        }
    }

    if m.SolCount > 0:
        sol["objective_value"] = float(m.ObjVal)

        tol = 1e-9
        mv = {}

        def add_mv(name, val):
            if abs(val) > tol:
                mv[name] = float(val)

        # Investments
        for i in cluster_ids:
            inc = p[i]["unit_MW"] if p[i]["is_UC"] else 1.0
            v_new = y_P_new[i].X
            v_ret = y_P_ret[i].X
            v_tot = y_P[i].X

            add_mv(f"y_P_new[{i}]", v_new)
            add_mv(f"y_P_ret[{i}]", v_ret)
            add_mv(f"y_P[{i}]", v_tot)

            sol["investments"][str(i)] = {
                "resource_type": p[i]["rt"],
                "zone": p[i]["zone"],
                "new_units": float(v_new),
                "new_MW": float(inc * v_new),
                "retired_units": float(v_ret),
                "retired_MW": float(inc * v_ret),
                "total_capacity_MW": float(v_tot),
            }

        for i in storage_ids:
            incE = p[i]["e_unit_MWh"] if p[i]["is_UC"] else 1.0
            v_new = y_E_new[i].X
            v_tot = y_E[i].X
            add_mv(f"y_E_new[{i}]", v_new)
            add_mv(f"y_E_ret[{i}]", y_E_ret[i].X)
            add_mv(f"y_E[{i}]", v_tot)

            sol["storage_energy_investments"][str(i)] = {
                "new_units": float(v_new),
                "new_MWh": float(incE * v_new),
                "total_energy_MWh": float(v_tot),
            }

        for l in lines:
            v_new = y_F_new[l].X
            v_tot = y_F[l].X
            add_mv(f"y_F_new[{l}]", v_new)
            add_mv(f"y_F[{l}]", v_tot)
            sol["transmission_investments"][str(l)] = {
                "new_MW": float(v_new),
                "total_capacity_MW": float(v_tot),
            }

        # Operations
        for i in cluster_ids:
            for t in range(T):
                add_mv(f"x_inj[{i},{t}]", x_inj[(i, t)].X)

        for i in storage_ids:
            for t in range(T):
                add_mv(f"x_wdw[{i},{t}]", x_wdw[(i, t)].X)
                add_mv(f"x_soc[{i},{t}]", x_soc[(i, t)].X)

        for seg in consumer_segments:
            s, z = seg["segment_id"], seg["zone"]
            for t in range(T):
                add_mv(f"x_nse[{s},{z},{t}]", x_nse[(s, z, t)].X)

        for i in hydro_ids:
            for t in range(T):
                add_mv(f"x_lvl[{i},{t}]", x_lvl[(i, t)].X)
                add_mv(f"x_spill[{i},{t}]", x_spill[(i, t)].X)

        for l in lines:
            for t in range(T):
                add_mv(f"x_flow[{l},{t}]", x_flow[(l, t)].X)

        for i in uc_ids:
            for t in range(T):
                add_mv(f"x_commit[{i},{t}]", x_commit[(i, t)].X)
                add_mv(f"x_start[{i},{t}]", x_start[(i, t)].X)
                add_mv(f"x_shut[{i},{t}]", x_shut[(i, t)].X)

        for w in range(W):
            add_mv(f"x_RPS[{w}]", x_RPS[w].X)
            add_mv(f"x_CO2[{w}]", x_CO2[w].X)

        sol["model_variables"] = mv

        if logger:
            logger.log_solution(sol["objective_value"], sol)
    else:
        # No feasible solution found
        sol["objective_value"] = 1e30
        sol["model_variables"] = {}
        for i in cluster_ids:
            sol["investments"][str(i)] = {
                "resource_type": p[i]["rt"],
                "zone": p[i]["zone"],
                "new_units": 0.0,
                "new_MW": 0.0,
                "retired_units": 0.0,
                "retired_MW": 0.0,
                "total_capacity_MW": float(existing_capacity.get(i, 0.0)),
            }
        for i in storage_ids:
            ratio = p[i]["e_unit_MWh"] / max(1e-9, p[i]["unit_MW"])
            exE = float(existing_capacity.get(i, 0.0)) * ratio
            sol["storage_energy_investments"][str(i)] = {
                "new_units": 0.0,
                "new_MWh": 0.0,
                "total_energy_MWh": exE,
            }
        for l in lines:
            ex = float(line_by_id[l]["existing_capacity_MW"])
            sol["transmission_investments"][str(l)] = {"new_MW": 0.0, "total_capacity_MW": ex}

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()