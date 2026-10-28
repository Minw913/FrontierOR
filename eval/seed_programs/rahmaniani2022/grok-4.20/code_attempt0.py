import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    # Extract key parameters
    pp = data['problem_parameters']
    num_zones = pp['num_zones']
    num_weeks = pp['num_weeks']
    hours_per_subperiod = pp['hours_per_subperiod']
    total_hours = pp['total_hours']
    num_gens = pp['num_generator_clusters']
    num_UC = pp['num_UC_generators']

    policy = data['policy']
    policy_type = policy['type']
    RPS_share = policy.get('RPS_share', 0.0)
    CO2_cap_tons_per_MWh = policy.get('CO2_cap_tons_per_MWh', 0.0)
    RPS_penalty = policy.get('RPS_noncompliance_cost_per_MWh', 0.0)
    CO2_penalty = policy.get('CO2_noncompliance_cost_per_ton', 0.0)

    subperiod_weights = data['subperiod_weights']
    hourly_weights = [w / hours_per_subperiod for w in subperiod_weights for _ in range(hours_per_subperiod)]

    generators = data['generators']
    gen_params = data['generator_parameters']
    existing_capacity = data['existing_capacity']
    demand = data['demand']
    consumer_segments = data['consumer_segments']
    transmission_lines = data['transmission_lines']
    availability = data['availability']
    hydro_inflow = data.get('hydro_inflow', {})

    # Map cluster_id to index
    gen_by_id = {g['cluster_id']: g for g in generators}
    gen_list = list(range(num_gens))  # indices 0 to num_gens-1
    id_to_idx = {g['cluster_id']: i for i, g in enumerate(generators)}

    # Precompute some constants
    is_UC = [gen_by_id[i]['is_UC'] for i in range(num_gens)] if False else [g['is_UC'] for g in generators]
    is_storage = [g['is_storage'] for g in generators]
    is_hydro = [g['is_hydro'] for g in generators]
    is_VRE = [g['is_VRE'] for g in generators]
    is_RPS_qual = [g['is_RPS_qualifying'] for g in generators]
    can_retire = [g['can_retire'] for g in generators]
    capacity_size = [g['capacity_size_MW'] for g in generators]
    zone_of = [g['zone'] for g in generators]
    resource_type_of = [g['resource_type'] for g in generators]

    # Get parameters per generator
    var_cost = []
    fixed_OM = []
    invest_cost = []
    CO2_factor = []
    min_output_frac = []
    ramp_up = []
    ramp_down = []
    min_up_time = []
    min_down_time = []
    start_up_cost = []
    max_cap_MW = []

    for g in generators:
        rt = g['resource_type']
        p = gen_params[rt]
        var_cost.append(p['variable_cost_per_MWh'])
        fixed_OM.append(p['fixed_OM_cost_per_MW_yr'])
        invest_cost.append(p['investment_cost_per_MW_yr'])
        CO2_factor.append(p['CO2_emission_tons_per_MWh'])
        min_output_frac.append(p.get('min_output_frac', 0.0))
        ramp_up.append(p.get('ramp_up_rate_pct_per_hr', 1.0))
        ramp_down.append(p.get('ramp_down_rate_pct_per_hr', 1.0))
        min_up_time.append(p.get('min_up_time_hr', 1))
        min_down_time.append(p.get('min_down_time_hr', 1))
        start_up_cost.append(p.get('start_up_cost', 0.0))
        max_cap_MW.append(p.get('max_capacity_MW', 1e6))

    # Storage parameters
    charge_eff = []
    discharge_eff = []
    self_discharge = []
    min_duration = []
    max_duration = []
    energy_invest_cost = []
    energy_fixed_OM = []
    max_energy_MWh = []
    energy_size_MWh = []

    for g in generators:
        rt = g['resource_type']
        if rt == 'battery':
            p = gen_params['battery']
            charge_eff.append(p['charging_efficiency'])
            discharge_eff.append(p['discharging_efficiency'])
            self_discharge.append(p['self_discharge_rate_pct'])
            min_duration.append(p['min_duration_MWh_per_MW'])
            max_duration.append(p['max_duration_MWh_per_MW'])
            energy_invest_cost.append(p['energy_investment_cost_per_MWh_yr'])
            energy_fixed_OM.append(p['energy_fixed_OM_cost_per_MWh_yr'])
            max_energy_MWh.append(p.get('max_energy_capacity_MWh', 1e6))
            energy_size_MWh.append(p['energy_capacity_size_MWh'])
        else:
            charge_eff.append(1.0)
            discharge_eff.append(1.0)
            self_discharge.append(0.0)
            min_duration.append(0.0)
            max_duration.append(0.0)
            energy_invest_cost.append(0.0)
            energy_fixed_OM.append(0.0)
            max_energy_MWh.append(0.0)
            energy_size_MWh.append(1.0)

    # Hydro parameters
    hydro_duration = []
    hydro_min_output_frac = []
    for g in generators:
        rt = g['resource_type']
        if rt == 'hydro':
            p = gen_params['hydro']
            hydro_duration.append(p['duration_MWh_per_MW'])
            hydro_min_output_frac.append(p['min_output_frac'])
        else:
            hydro_duration.append(0.0)
            hydro_min_output_frac.append(0.0)

    # Existing capacity
    exist_cap = [existing_capacity[str(g['cluster_id'])] for g in generators]

    # Demand
    demand_list = []
    for z in range(num_zones):
        demand_list.append(demand[str(z)])

    # Consumer segments
    nse_segments = consumer_segments
    num_segments = len(nse_segments)
    nse_max_frac = [s['max_nse_fraction'] for s in nse_segments]
    nse_cost = [s['cost_per_MWh'] for s in nse_segments]
    nse_zone = [s['zone'] for s in nse_segments]

    # Transmission
    num_lines = len(transmission_lines)
    from_zone = [l['from_zone'] for l in transmission_lines]
    to_zone = [l['to_zone'] for l in transmission_lines]
    exist_trans = [l['existing_capacity_MW'] for l in transmission_lines]
    max_trans = [l['max_capacity_MW'] for l in transmission_lines]
    trans_invest_cost = [l['investment_cost_per_MW_yr'] for l in transmission_lines]

    # Availability and inflow
    avail = [availability[str(g['cluster_id'])] for g in generators]
    inflow = [hydro_inflow.get(str(g['cluster_id']), [0.0]*total_hours) for g in generators]

    # Create Gurobi model
    model = gp.Model("PowerGridPlanning")
    model.setParam(GRB.Param.Seed, 0)
    model.setParam(GRB.Param.MIPGap, 1e-4)
    model.setParam(GRB.Param.NumericFocus, 0)
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.TimeLimit, args.time_limit - 5)  # reserve some time for output

    # ==================== VARIABLES ====================

    # Investment variables
    y_P_new = model.addVars(num_gens, vtype=GRB.INTEGER, lb=0, name="y_P_new")
    y_P_ret = model.addVars(num_gens, vtype=GRB.INTEGER, lb=0, name="y_P_ret")
    y_P = model.addVars(num_gens, vtype=GRB.CONTINUOUS, lb=0, name="y_P")

    y_E_new = model.addVars(num_gens, vtype=GRB.INTEGER, lb=0, name="y_E_new")  # only for storage
    y_E = model.addVars(num_gens, vtype=GRB.CONTINUOUS, lb=0, name="y_E")

    y_F_new = model.addVars(num_lines, vtype=GRB.INTEGER, lb=0, name="y_F_new")
    y_F = model.addVars(num_lines, vtype=GRB.CONTINUOUS, lb=0, name="y_F")

    # Operational variables
    x_inj = model.addVars(num_gens, total_hours, lb=0, name="x_inj")  # generation / discharge
    x_wdw = model.addVars(num_gens, total_hours, lb=0, name="x_wdw")  # charge / withdrawal
    x_soc = model.addVars(num_gens, total_hours, lb=0, name="x_soc")
    x_lvl = model.addVars(num_gens, total_hours, lb=0, name="x_lvl")
    x_spill = model.addVars(num_gens, total_hours, lb=0, name="x_spill")
    x_nse = model.addVars(num_segments, num_zones, total_hours, lb=0, name="x_nse")
    x_flow = model.addVars(num_lines, total_hours, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x_flow")

    # UC variables
    x_commit = model.addVars(num_gens, total_hours, lb=0, name="x_commit")
    x_start = model.addVars(num_gens, total_hours, lb=0, name="x_start")
    x_shut = model.addVars(num_gens, total_hours, lb=0, name="x_shut")

    # Policy slacks
    x_RPS = model.addVars(num_weeks, lb=0, name="x_RPS")
    x_CO2 = model.addVars(num_weeks, lb=0, name="x_CO2")

    # ==================== INVESTMENT CONSTRAINTS ====================

    for i in range(num_gens):
        # Capacity balance
        model.addConstr(y_P[i] == exist_cap[i] + capacity_size[i] * y_P_new[i] - y_P_ret[i])

        # Bounds
        model.addConstr(y_P[i] <= max_cap_MW[i])
        if not can_retire[i]:
            model.addConstr(y_P_ret[i] == 0)
        model.addConstr(y_P_ret[i] <= exist_cap[i])

        if is_storage[i]:
            model.addConstr(y_E[i] == energy_size_MWh[i] * y_E_new[i] if is_UC[i] else y_E_new[i])
            model.addConstr(y_E[i] >= min_duration[i] * y_P[i])
            model.addConstr(y_E[i] <= max_duration[i] * y_P[i])
            model.addConstr(y_E[i] <= max_energy_MWh[i])
        else:
            model.addConstr(y_E[i] == 0)

    for l in range(num_lines):
        model.addConstr(y_F[l] == exist_trans[l] + y_F_new[l])
        model.addConstr(y_F[l] <= max_trans[l])

    # ==================== OPERATIONAL CONSTRAINTS ====================

    M = 1e6  # big-M

    for t in range(total_hours):
        w = t // hours_per_subperiod
        h = t % hours_per_subperiod
        next_t = t + 1 if h < hours_per_subperiod - 1 else w * hours_per_subperiod

        for i in range(num_gens):
            cap = y_P[i]
            av = avail[i][t]

            # Generation limits
            if is_UC[i]:
                model.addConstr(x_inj[i, t] <= cap * av * x_commit[i, t] / (cap / capacity_size[i] if capacity_size[i] > 0 else 1))
                model.addConstr(x_inj[i, t] >= min_output_frac[i] * cap * x_commit[i, t] / (cap / capacity_size[i] if capacity_size[i] > 0 else 1))
                model.addConstr(x_commit[i, t] <= 1.0)
                model.addConstr(x_start[i, t] <= 1.0)
                model.addConstr(x_shut[i, t] <= 1.0)
            else:
                model.addConstr(x_inj[i, t] <= av * cap)
                if min_output_frac[i] > 0 and not is_storage[i] and not is_hydro[i]:
                    model.addConstr(x_inj[i, t] >= min_output_frac[i] * cap)

            if is_storage[i]:
                # Storage power limit
                model.addConstr(x_inj[i, t] + x_wdw[i, t] <= cap)
                model.addConstr(x_wdw[i, t] <= av * cap)

                # SOC dynamics (with wrap-around per subperiod)
                soc_prev = x_soc[i, t-1] if h > 0 else x_soc[i, w*hours_per_subperiod + hours_per_subperiod - 1]
                model.addConstr(
                    x_soc[i, t] == (1 - self_discharge[i]) * soc_prev +
                    charge_eff[i] * x_wdw[i, t] - x_inj[i, t] / discharge_eff[i]
                )
                model.addConstr(x_soc[i, t] <= y_E[i])

                # Charging and discharging limits
                model.addConstr(charge_eff[i] * x_wdw[i, t] <= y_E[i] - (1 - self_discharge[i]) * soc_prev)
                model.addConstr(x_inj[i, t] / discharge_eff[i] <= soc_prev)

            elif is_hydro[i]:
                lvl_prev = x_lvl[i, t-1] if h > 0 else x_lvl[i, w*hours_per_subperiod + hours_per_subperiod - 1]
                inflow_val = inflow[i][t] * cap
                model.addConstr(
                    x_lvl[i, t] == lvl_prev + inflow_val - x_inj[i, t] - x_spill[i, t]
                )
                model.addConstr(x_lvl[i, t] <= hydro_duration[i] * cap)
                model.addConstr(x_inj[i, t] + x_spill[i, t] >= hydro_min_output_frac[i] * cap)
            else:
                model.addConstr(x_wdw[i, t] == 0)
                model.addConstr(x_soc[i, t] == 0)
                if not is_hydro[i]:
                    model.addConstr(x_lvl[i, t] == 0)
                    model.addConstr(x_spill[i, t] == 0)

            # Ramping
            if not is_storage[i] and not is_hydro[i]:
                prev_gen = x_inj[i, t-1] if h > 0 else x_inj[i, w*hours_per_subperiod + hours_per_subperiod - 1]
                if is_UC[i]:
                    # Simplified UC ramping (approximate linear version)
                    model.addConstr(x_inj[i, t] - prev_gen <= ramp_up[i] * cap + M * (x_start[i, t]))
                    model.addConstr(prev_gen - x_inj[i, t] <= ramp_down[i] * cap + M * (x_shut[i, t]))
                else:
                    model.addConstr(x_inj[i, t] - prev_gen <= ramp_up[i] * cap)
                    model.addConstr(prev_gen - x_inj[i, t] <= ramp_down[i] * cap)

        # UC logic
        for i in range(num_gens):
            if is_UC[i]:
                prev_commit = x_commit[i, t-1] if h > 0 else x_commit[i, w*hours_per_subperiod + hours_per_subperiod - 1]
                model.addConstr(x_commit[i, t] == prev_commit + x_start[i, t] - x_shut[i, t])

                # Min up / down time (simple formulation with wrap)
                if min_up_time[i] > 1:
                    for k in range(1, min_up_time[i]):
                        prev_t = (t - k) % hours_per_subperiod + w * hours_per_subperiod
                        model.addConstr(x_commit[i, t] >= x_start[i, prev_t])

                if min_down_time[i] > 1:
                    for k in range(1, min_down_time[i]):
                        prev_t = (t - k) % hours_per_subperiod + w * hours_per_subperiod
                        model.addConstr(x_commit[i, t] + x_shut[i, prev_t] <= 1.0)

        # Transmission flow limits
        for l in range(num_lines):
            model.addConstr(x_flow[l, t] <= y_F[l])
            model.addConstr(x_flow[l, t] >= -y_F[l])

        # Power balance
        for z in range(num_zones):
            gen_total = gp.quicksum(x_inj[i, t] for i in range(num_gens) if zone_of[i] == z)
            wdw_total = gp.quicksum(x_wdw[i, t] for i in range(num_gens) if zone_of[i] == z and is_storage[i])
            nse_total = gp.quicksum(x_nse[s, z, t] for s in range(num_segments) if nse_zone[s] == z)
            inflow_trans = gp.quicksum(x_flow[l, t] for l in range(num_lines) if to_zone[l] == z)
            outflow_trans = gp.quicksum(x_flow[l, t] for l in range(num_lines) if from_zone[l] == z)

            model.addConstr(
                gen_total - wdw_total + inflow_trans - outflow_trans + nse_total == demand[str(z)][t]
            )

        # NSE limits
        for s in range(num_segments):
            z = nse_zone[s]
            model.addConstr(x_nse[s, z, t] <= nse_max_frac[s] * demand[str(z)][t])

    # ==================== POLICY CONSTRAINTS ====================

    total_demand_weighted = gp.quicksum(hourly_weights[t] * demand[str(z)][t] for z in range(num_zones) for t in range(total_hours))

    if policy_type == "RPS" or policy_type == "REF":
        rps_gen = gp.quicksum(
            hourly_weights[t] * x_inj[i, t]
            for i in range(num_gens)
            for t in range(total_hours)
            if is_RPS_qual[i]
        )
        model.addConstr(rps_gen + gp.quicksum(x_RPS[w] * subperiod_weights[w] * hours_per_subperiod for w in range(num_weeks)) >= RPS_share * total_demand_weighted)

    if policy_type == "CO2" or policy_type == "REF":
        co2_emissions = gp.quicksum(
            hourly_weights[t] * CO2_factor[i] * (x_inj[i, t] + x_wdw[i, t] if is_storage[i] else x_inj[i, t])
            for i in range(num_gens)
            for t in range(total_hours)
        )
        model.addConstr(co2_emissions - gp.quicksum(x_CO2[w] * subperiod_weights[w] * hours_per_subperiod for w in range(num_weeks)) <= CO2_cap_tons_per_MWh * total_demand_weighted)

    # ==================== OBJECTIVE ====================

    # Fixed costs
    fixed_cost = gp.quicksum(
        invest_cost[i] * capacity_size[i] * y_P_new[i] +
        fixed_OM[i] * y_P[i]
        for i in range(num_gens)
    )

    storage_energy_cost = gp.quicksum(
        energy_invest_cost[i] * y_E[i] + energy_fixed_OM[i] * y_E[i]
        for i in range(num_gens) if is_storage[i]
    )

    hydro_energy_cost = gp.quicksum(
        gen_params['hydro']['energy_investment_cost_per_MWh_yr'] * hydro_duration[i] * y_P[i] +
        gen_params['hydro']['energy_fixed_OM_cost_per_MWh_yr'] * hydro_duration[i] * y_P[i]
        for i in range(num_gens) if is_hydro[i]
    )

    trans_cost = gp.quicksum(trans_invest_cost[l] * y_F_new[l] for l in range(num_lines))

    # Variable costs
    var_cost_total = gp.quicksum(
        hourly_weights[t] * var_cost[i] * (x_inj[i, t] + x_wdw[i, t] if is_storage[i] else x_inj[i, t])
        for i in range(num_gens) for t in range(total_hours)
    )

    # Unserved energy
    nse_cost_total = gp.quicksum(
        hourly_weights[t] * nse_cost[s] * x_nse[s, nse_zone[s], t]
        for s in range(num_segments) for t in range(total_hours)
    )

    # Startup costs
    startup_cost_total = gp.quicksum(
        hourly_weights[t] * start_up_cost[i] * x_start[i, t] * capacity_size[i]
        for i in range(num_gens) if is_UC[i] for t in range(total_hours)
    )

    # Policy noncompliance
    policy_cost = gp.quicksum(
        RPS_penalty * x_RPS[w] * subperiod_weights[w] * hours_per_subperiod +
        CO2_penalty * x_CO2[w] * subperiod_weights[w] * hours_per_subperiod
        for w in range(num_weeks)
    )

    total_cost = (fixed_cost + storage_energy_cost + hydro_energy_cost + trans_cost +
                  var_cost_total + nse_cost_total + startup_cost_total + policy_cost)

    model.setObjective(total_cost, GRB.MINIMIZE)

    # Callback for logging incumbents
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                # We cannot easily get full solution in callback without heavy overhead
                # So we log the objective only
                logger.log(obj)

    # Optimize
    model.optimize(callback)

    # ==================== EXTRACT SOLUTION ====================

    if model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT]:
        # Try to get best feasible if any
        if model.SolCount == 0:
            print("No feasible solution found!")
            sys.exit(1)

    obj_val = model.ObjVal

    # Build solution dictionary
    solution = {
        "objective_value": float(obj_val),
        "model_variables": {},
        "investments": {},
        "storage_energy_investments": {},
        "transmission_investments": {}
    }

    # Fill investments
    for i, g in enumerate(generators):
        cid = str(g['cluster_id'])
        new_units = y_P_new[i].X if y_P_new[i].X is not None else 0.0
        ret_units = y_P_ret[i].X if y_P_ret[i].X is not None else 0.0
        total_cap = y_P[i].X if y_P[i].X is not None else exist_cap[i]

        solution["investments"][cid] = {
            "resource_type": resource_type_of[i],
            "zone": int(zone_of[i]),
            "new_units": float(new_units),
            "new_MW": float(new_units * capacity_size[i]),
            "retired_units": float(ret_units),
            "retired_MW": float(ret_units * capacity_size[i]),
            "total_capacity_MW": float(total_cap)
        }

        if is_storage[i]:
            new_e_units = y_E_new[i].X if y_E_new[i].X is not None else 0.0
            total_e = y_E[i].X if y_E[i].X is not None else 0.0
            solution["storage_energy_investments"][cid] = {
                "new_units": float(new_e_units),
                "new_MWh": float(new_e_units if not is_UC[i] else new_e_units * energy_size_MWh[i]),
                "total_energy_MWh": float(total_e)
            }

    for l_idx, line in enumerate(transmission_lines):
        lid = str(line['line_id'])
        new_mw = y_F_new[l_idx].X if y_F_new[l_idx].X is not None else 0.0
        total_mw = y_F[l_idx].X if y_F[l_idx].X is not None else exist_trans[l_idx]
        solution["transmission_investments"][lid] = {
            "new_MW": float(new_mw),
            "total_capacity_MW": float(total_mw)
        }

    # Fill model_variables (sparse - only non-zero)
    mv = solution["model_variables"]

    def add_if_nonzero(name, val):
        if abs(val) > 1e-6:
            mv[name] = float(val)

    for i in range(num_gens):
        add_if_nonzero(f"y_P_new[{i}]", y_P_new[i].X)
        add_if_nonzero(f"y_P_ret[{i}]", y_P_ret[i].X)
        add_if_nonzero(f"y_P[{i}]", y_P[i].X)
        if is_storage[i]:
            add_if_nonzero(f"y_E_new[{i}]", y_E_new[i].X)
            add_if_nonzero(f"y_E[{i}]", y_E[i].X)

    for l in range(num_lines):
        add_if_nonzero(f"y_F_new[{l}]", y_F_new[l].X)
        add_if_nonzero(f"y_F[{l}]", y_F[l].X)

    for i in range(num_gens):
        for t in range(total_hours):
            add_if_nonzero(f"x_inj[{i},{t}]", x_inj[i, t].X)
            if is_storage[i]:
                add_if_nonzero(f"x_wdw[{i},{t}]", x_wdw[i, t].X)
                add_if_nonzero(f"x_soc[{i},{t}]", x_soc[i, t].X)
            if is_hydro[i]:
                add_if_nonzero(f"x_lvl[{i},{t}]", x_lvl[i, t].X)
                add_if_nonzero(f"x_spill[{i},{t}]", x_spill[i, t].X)
            if is_UC[i]:
                add_if_nonzero(f"x_commit[{i},{t}]", x_commit[i, t].X)
                add_if_nonzero(f"x_start[{i},{t}]", x_start[i, t].X)
                add_if_nonzero(f"x_shut[{i},{t}]", x_shut[i, t].X)

    for s in range(num_segments):
        for z in range(num_zones):
            for t in range(total_hours):
                add_if_nonzero(f"x_nse[{s},{z},{t}]", x_nse[s, z, t].X)

    for l in range(num_lines):
        for t in range(total_hours):
            add_if_nonzero(f"x_flow[{l},{t}]", x_flow[l, t].X)

    for w in range(num_weeks):
        add_if_nonzero(f"x_RPS[{w}]", x_RPS[w].X)
        add_if_nonzero(f"x_CO2[{w}]", x_CO2[w].X)

    # Log final solution
    if logger:
        logger.log_solution(obj_val, solution)

    # Write output
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    print(f"Solution written to {args.solution_path} with objective {obj_val}")

if __name__ == "__main__":
    main()