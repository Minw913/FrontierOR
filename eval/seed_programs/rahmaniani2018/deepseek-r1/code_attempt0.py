import gurobipy as gp
from gurobipy import GRB
import json
import argparse
import time
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    dim = instance['problem_dimensions']
    subperiods = instance['subperiods']
    num_subperiods = subperiods['num_subperiods']
    hours_per_subperiod = subperiods['hours_per_subperiod']
    total_timesteps = num_subperiods * hours_per_subperiod
    subperiod_weights = subperiods['subperiod_weights']
    circular_indexing = subperiods['circular_indexing']

    hourly_weights = []
    for w in range(num_subperiods):
        weight_w = subperiod_weights[w] / hours_per_subperiod
        hourly_weights.extend([weight_w] * hours_per_subperiod)

    total_weighted_demand = 0.0
    for z, demand_list in instance['demand'].items():
        for t, d in enumerate(demand_list):
            total_weighted_demand += hourly_weights[t] * d

    model = gp.Model("Electricity_System_Planning")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    generators_by_id = {g['id']: g for g in instance['generators']}
    availability_by_generator = {}
    for gen_id, prof in instance['availability_profiles'].items():
        gen_id_int = int(gen_id)
        if prof['type'] == 'constant':
            availability_by_generator[gen_id_int] = [prof['value']] * total_timesteps
        else:
            availability_by_generator[gen_id_int] = prof['values']

    lines_by_id = {l['id']: l for l in instance['transmission_lines']}
    segments = instance['nse_segments']

    yPnew = {}
    yPret = {}
    yEnew = {}
    yEret = {}
    for gen in instance['generators']:
        i = gen['id']
        capacity_size = gen['capacity_size_MW']
        existing_capacity_MW = gen['existing_capacity_MW']
        max_capacity_MW = gen['max_capacity_MW']
        initial_units = existing_capacity_MW / capacity_size
        max_new_units = (max_capacity_MW - existing_capacity_MW) / capacity_size
        yPnew[i] = model.addVar(lb=0, ub=max_new_units, vtype=GRB.INTEGER, name=f"yPnew_{i}")
        if gen['can_retire']:
            yPret[i] = model.addVar(lb=0, ub=initial_units, vtype=GRB.INTEGER, name=f"yPret_{i}")
        else:
            yPret[i] = 0
        if gen['is_storage']:
            storage_capacity_size = gen['storage_capacity_size_MWh']
            existing_storage_MWh = gen['existing_storage_capacity_MWh']
            max_storage_MWh = gen['max_storage_capacity_MWh']
            initial_units_E = existing_storage_MWh / storage_capacity_size
            max_new_units_E = (max_storage_MWh - existing_storage_MWh) / storage_capacity_size
            yEnew[i] = model.addVar(lb=0, ub=max_new_units_E, vtype=GRB.INTEGER, name=f"yEnew_{i}")
            if gen['can_retire']:
                yEret[i] = model.addVar(lb=0, ub=initial_units_E, vtype=GRB.INTEGER, name=f"yEret_{i}")
            else:
                yEret[i] = 0

    yFnew = {}
    for line in instance['transmission_lines']:
        l = line['id']
        existing_cap = line['existing_capacity_MW']
        max_new = line['max_new_capacity_MW']
        yFnew[l] = model.addVar(lb=0, ub=max_new, vtype=GRB.INTEGER, name=f"yFnew_{l}")

    xinj = {}
    xwdw = {}
    xsoc = {}
    xcommit = {}
    xstart = {}
    xshut = {}
    xflow = {}
    xnse = {}
    xRPS = {}
    xCO2 = {}

    if instance['policy']['RPS_enabled']:
        for w in range(num_subperiods):
            xRPS[w] = model.addVar(lb=0, name=f"xRPS_{w}")
    if instance['policy']['CO2_cap_enabled']:
        for w in range(num_subperiods):
            xCO2[w] = model.addVar(lb=0, name=f"xCO2_{w}")

    for gen in instance['generators']:
        i = gen['id']
        capacity_size = gen['capacity_size_MW']
        existing_capacity_MW = gen['existing_capacity_MW']
        initial_units = existing_capacity_MW / capacity_size
        total_P_MW = (initial_units - (yPret[i] if type(yPret[i]) != int else yPret[i]) + yPnew[i]) * capacity_size
        for t in range(total_timesteps):
            xinj[i, t] = model.addVar(lb=0, name=f"xinj_{i}_{t}")
            if gen['is_storage']:
                xwdw[i, t] = model.addVar(lb=0, name=f"xwdw_{i}_{t}")
                xsoc[i, t] = model.addVar(lb=0, name=f"xsoc_{i}_{t}")
            if gen['is_UC']:
                total_units = total_P_MW / capacity_size
                xcommit[i, t] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=total_units, name=f"xcommit_{i}_{t}")
                xstart[i, t] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=total_units, name=f"xstart_{i}_{t}")
                xshut[i, t] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=total_units, name=f"xshut_{i}_{t}")

    for line in instance['transmission_lines']:
        l = line['id']
        total_cap = line['existing_capacity_MW'] + yFnew[l]
        for t in range(total_timesteps):
            xflow[l, t] = model.addVar(lb=-total_cap, ub=total_cap, name=f"xflow_{l}_{t}")

    for seg in segments:
        s = seg['segment']
        max_frac = seg['max_frac']
        for z_str, demand_list in instance['demand'].items():
            z = int(z_str)
            for t in range(total_timesteps):
                max_nse = max_frac * demand_list[t]
                xnse[s, z, t] = model.addVar(lb=0, ub=max_nse, name=f"xnse_{s}_{z}_{t}")

    generators_by_zone = {}
    for gen in instance['generators']:
        z = gen['zone']
        if z not in generators_by_zone:
            generators_by_zone[z] = []
        generators_by_zone[z].append(gen['id'])

    outgoing_lines = {}
    incoming_lines = {}
    for line in instance['transmission_lines']:
        fz = line['from_zone']
        tz = line['to_zone']
        if fz not in outgoing_lines:
            outgoing_lines[fz] = []
        outgoing_lines[fz].append(line['id'])
        if tz not in incoming_lines:
            incoming_lines[tz] = []
        incoming_lines[tz].append(line['id'])

    for t in range(total_timesteps):
        for z_str, demand_list in instance['demand'].items():
            z = int(z_str)
            demand_t = demand_list[t]
            gen_sum = gp.quicksum(xinj[i, t] for i in generators_by_zone.get(z, []))
            storage_withdrawal_sum = gp.quicksum(xwdw[i, t] for i in generators_by_zone.get(z, []) if generators_by_id[i]['is_storage'])
            flow_out = gp.quicksum(xflow[l, t] for l in outgoing_lines.get(z, []))
            flow_in = gp.quicksum(xflow[l, t] for l in incoming_lines.get(z, []))
            nse_sum = gp.quicksum(xnse[s, z, t] for s in [seg['segment'] for seg in segments])
            model.addConstr(gen_sum - storage_withdrawal_sum - flow_out + flow_in + nse_sum == demand_t, f"power_balance_z{z}_t{t}")

    for gen in instance['generators']:
        i = gen['id']
        capacity_size = gen['capacity_size_MW']
        existing_capacity_MW = gen['existing_capacity_MW']
        initial_units = existing_capacity_MW / capacity_size
        total_P_MW = (initial_units - (yPret[i] if type(yPret[i]) != int else yPret[i]) + yPnew[i]) * capacity_size
        min_output_frac = gen['min_output_frac']
        availability = availability_by_generator[i]
        if gen['is_storage']:
            storage_capacity_size = gen['storage_capacity_size_MWh']
            existing_storage_MWh = gen['existing_storage_capacity_MWh']
            initial_units_E = existing_storage_MWh / storage_capacity_size
            total_E_MWh = (initial_units_E - (yEret[i] if type(yEret[i]) != int else yEret[i]) + yEnew[i]) * storage_capacity_size
        for t in range(total_timesteps):
            model.addConstr(xinj[i, t] <= availability[t] * total_P_MW, f"gen_cap_{i}_{t}")
            if not gen['is_UC'] and not gen['is_storage'] and not gen['is_hydro'] and min_output_frac > 0:
                model.addConstr(xinj[i, t] >= min_output_frac * total_P_MW, f"gen_min_{i}_{t}")
            if gen['is_storage']:
                model.addConstr(xwdw[i, t] <= availability[t] * total_P_MW, f"storage_wdw_cap_{i}_{t}")
                model.addConstr(xinj[i, t] + xwdw[i, t] <= total_P_MW, f"storage_power_cap_{i}_{t}")
                if t == 0:
                    if circular_indexing:
                        prev_t = hours_per_subperiod - 1
                    else:
                        prev_t = None
                else:
                    if circular_indexing and (t % hours_per_subperiod == 0):
                        prev_t = t + hours_per_subperiod - 1
                    else:
                        prev_t = t - 1
                if prev_t is not None:
                    model.addConstr(xsoc[i, t] == xsoc[i, prev_t] * (1 - gen['self_discharge_rate']) + gen['charge_efficiency'] * xwdw[i, t] - xinj[i, t] / gen['discharge_efficiency'], f"soc_balance_{i}_{t}")
                else:
                    model.addConstr(xsoc[i, t] == gen['charge_efficiency'] * xwdw[i, t] - xinj[i, t] / gen['discharge_efficiency'], f"soc_balance_{i}_{t}")
                model.addConstr(xsoc[i, t] <= total_E_MWh, f"soc_upper_{i}_{t}")
                if prev_t is not None:
                    room = total_E_MWh - xsoc[i, prev_t]
                else:
                    room = total_E_MWh
                model.addConstr(gen['charge_efficiency'] * xwdw[i, t] <= room, f"storage_charge_room_{i}_{t}")
                if prev_t is not None:
                    model.addConstr(xinj[i, t] / gen['discharge_efficiency'] <= xsoc[i, prev_t], f"storage_discharge_room_{i}_{t}")
                else:
                    model.addConstr(xinj[i, t] / gen['discharge_efficiency'] <= 0, f"storage_discharge_room_{i}_{t}")
            if gen['is_hydro'] and min_output_frac > 0:
                model.addConstr(xinj[i, t] >= min_output_frac * total_P_MW, f"hydro_min_{i}_{t}")
            if gen['is_UC']:
                model.addConstr(xinj[i, t] <= availability[t] * (xcommit[i, t] * capacity_size), f"uc_cap_upper_{i}_{t}")
                model.addConstr(xinj[i, t] >= min_output_frac * (xcommit[i, t] * capacity_size), f"uc_min_{i}_{t}")
                if t == 0:
                    if circular_indexing:
                        prev_t = hours_per_subperiod - 1
                    else:
                        prev_t = None
                else:
                    if circular_indexing and (t % hours_per_subperiod == 0):
                        prev_t = t + hours_per_subperiod - 1
                    else:
                        prev_t = t - 1
                if prev_t is not None:
                    model.addConstr(xcommit[i, t] == xcommit[i, prev_t] + xstart[i, t] - xshut[i, t], f"commit_change_{i}_{t}")
                else:
                    model.addConstr(xcommit[i, t] == xstart[i, t] - xshut[i, t], f"commit_change_{i}_{t}")
                min_up = gen['min_up_time_hr']
                min_down = gen['min_down_time_hr']
                total_units = total_P_MW / capacity_size
                if min_up > 0:
                    if t >= min_up - 1:
                        model.addConstr(gp.quicksum(xstart[i, tau] for tau in range(t - min_up + 1, t + 1)) <= xcommit[i, t], f"min_up_{i}_{t}")
                    else:
                        if circular_indexing:
                            tau_list = list(range(t - min_up + 1, t + 1))
                            adjusted_tau_list = [tau % total_timesteps for tau in tau_list]
                            model.addConstr(gp.quicksum(xstart[i, tau] for tau in adjusted_tau_list) <= xcommit[i, t], f"min_up_{i}_{t}")
                if min_down > 0:
                    if t >= min_down - 1:
                        model.addConstr(gp.quicksum(xshut[i, tau] for tau in range(t - min_down + 1, t + 1)) <= total_units - xcommit[i, t], f"min_down_{i}_{t}")
                    else:
                        if circular_indexing:
                            tau_list = list(range(t - min_down + 1, t + 1))
                            adjusted_tau_list = [tau % total_timesteps for tau in tau_list]
                            model.addConstr(gp.quicksum(xshut[i, tau] for tau in adjusted_tau_list) <= total_units - xcommit[i, t], f"min_down_{i}_{t}")
                ramp_up_frac = gen['ramp_up_frac_per_hr']
                ramp_dn_frac = gen['ramp_dn_frac_per_hr']
                if t == 0:
                    if circular_indexing:
                        prev_t = hours_per_subperiod - 1
                    else:
                        prev_t = None
                else:
                    if circular_indexing and (t % hours_per_subperiod == 0):
                        prev_t = t + hours_per_subperiod - 1
                    else:
                        prev_t = t - 1
                if prev_t is not None:
                    ramp_up_limit_new = min(availability[t], max(min_output_frac, ramp_up_frac))
                    ramp_dn_limit_new = min(availability[t], max(min_output_frac, ramp_dn_frac))
                    model.addConstr(xinj[i, t] - xinj[i, prev_t] <= capacity_size * (ramp_up_frac * (xcommit[i, prev_t] - xshut[i, t]) + ramp_up_limit_new * xstart[i, t] - min_output_frac * xshut[i, t]), f"ramp_up_{i}_{t}")
                    model.addConstr(xinj[i, prev_t] - xinj[i, t] <= capacity_size * (ramp_dn_frac * (xcommit[i, prev_t] - xstart[i, t]) + ramp_dn_limit_new * xshut[i, t] - min_output_frac * xstart[i, t]), f"ramp_dn_{i}_{t}")

    for line in instance['transmission_lines']:
        l = line['id']
        total_cap = line['existing_capacity_MW'] + yFnew[l]
        for t in range(total_timesteps):
            model.addConstr(xflow[l, t] >= -total_cap, f"flow_lb_{l}_{t}")
            model.addConstr(xflow[l, t] <= total_cap, f"flow_ub_{l}_{t}")

    for gen in instance['generators']:
        i = gen['id']
        if gen['is_storage']:
            capacity_size = gen['capacity_size_MW']
            existing_capacity_MW = gen['existing_capacity_MW']
            initial_units = existing_capacity_MW / capacity_size
            total_P_MW = (initial_units - (yPret[i] if type(yPret[i]) != int else yPret[i]) + yPnew[i]) * capacity_size
            storage_capacity_size = gen['storage_capacity_size_MWh']
            existing_storage_MWh = gen['existing_storage_capacity_MWh']
            initial_units_E = existing_storage_MWh / storage_capacity_size
            total_E_MWh = (initial_units_E - (yEret[i] if type(yEret[i]) != int else yEret[i]) + yEnew[i]) * storage_capacity_size
            model.addConstr(total_E_MWh >= gen['min_duration_MWh_per_MW'] * total_P_MW, f"storage_min_duration_{i}")
            model.addConstr(total_E_MWh <= gen['max_duration_MWh_per_MW'] * total_P_MW, f"storage_max_duration_{i}")

    if instance['policy']['RPS_enabled']:
        qualifying_resources = [g['id'] for g in instance['generators'] if g['is_RPS_qualifying']]
        lhs = 0.0
        for w in range(num_subperiods):
            weight_w = subperiod_weights[w] / hours_per_subperiod
            for t0 in range(w * hours_per_subperiod, (w + 1) * hours_per_subperiod):
                lhs += weight_w * gp.quicksum(xinj[i, t0] for i in qualifying_resources)
        lhs += gp.quicksum(xRPS[w] for w in range(num_subperiods))
        model.addConstr(lhs >= instance['policy']['RPS_share'] * total_weighted_demand, "RPS_constraint")

    if instance['policy']['CO2_cap_enabled']:
        emissions_sum = 0.0
        for w in range(num_subperiods):
            weight_w = subperiod_weights[w] / hours_per_subperiod
            for t0 in range(w * hours_per_subperiod, (w + 1) * hours_per_subperiod):
                for gen in instance['generators']:
                    i = gen['id']
                    co2_rate = gen['co2_tons_per_MWh']
                    if gen['is_storage']:
                        emissions_sum += weight_w * co2_rate * (xinj[i, t0] + xwdw[i, t0])
                    else:
                        emissions_sum += weight_w * co2_rate * xinj[i, t0]
        model.addConstr(emissions_sum - gp.quicksum(xCO2[w] for w in range(num_subperiods)) <= 0.05 * total_weighted_demand, "CO2_constraint")

    fixed_cost = 0.0
    for gen in instance['generators']:
        i = gen['id']
        capacity_size = gen['capacity_size_MW']
        existing_capacity_MW = gen['existing_capacity_MW']
        initial_units = existing_capacity_MW / capacity_size
        total_P_MW = (initial_units - (yPret[i] if type(yPret[i]) != int else yPret[i]) + yPnew[i]) * capacity_size
        fixed_cost += gen['inv_cost_per_MW_yr'] * (yPnew[i] * capacity_size)
        fixed_cost += gen['fom_cost_per_MW_yr'] * total_P_MW
        if gen['is_hydro']:
            duration = gen['duration_MWh_per_MW']
            fixed_cost += gen['inv_cost_energy_per_MWh_yr'] * (duration * (yPnew[i] * capacity_size))
            fixed_cost += gen['fom_cost_energy_per_MWh_yr'] * (duration * total_P_MW)
        if gen['is_storage']:
            storage_capacity_size = gen['storage_capacity_size_MWh']
            existing_storage_MWh = gen['existing_storage_capacity_MWh']
            initial_units_E = existing_storage_MWh / storage_capacity_size
            total_E_MWh = (initial_units_E - (yEret[i] if type(yEret[i]) != int else yEret[i]) + yEnew[i]) * storage_capacity_size
            fixed_cost += gen['inv_cost_energy_per_MWh_yr'] * (yEnew[i] * storage_capacity_size)
            fixed_cost += gen['fom_cost_energy_per_MWh_yr'] * total_E_MWh
    for line in instance['transmission_lines']:
        l = line['id']
        fixed_cost += line['inv_cost_per_MW_yr'] * yFnew[l]

    var_cost = 0.0
    for t in range(total_timesteps):
        weight_t = hourly_weights[t]
        for gen in instance['generators']:
            i = gen['id']
            var_cost += weight_t * gen['var_cost_per_MWh'] * xinj[i, t]
            if gen['is_storage']:
                var_cost += weight_t * gen['var_cost_per_MWh'] * xwdw[i, t]
        for seg in segments:
            s = seg['segment']
            cost_per_MWh = seg['cost_per_MWh']
            for z_str in instance['demand']:
                z = int(z_str)
                var_cost += weight_t * cost_per_MWh * xnse[s, z, t]
        for gen in instance['generators']:
            i = gen['id']
            if gen['is_UC']:
                var_cost += weight_t * gen['start_cost_per_unit'] * xstart[i, t]

    policy_cost = 0.0
    if instance['policy']['RPS_enabled']:
        policy_cost += instance['policy']['RPS_noncompliance_cost_per_MWh'] * gp.quicksum(xRPS[w] for w in range(num_subperiods))
    if instance['policy']['CO2_cap_enabled']:
        policy_cost += instance['policy']['CO2_noncompliance_cost_per_ton'] * gp.quicksum(xCO2[w] for w in range(num_subperiods))

    total_cost = fixed_cost + var_cost + policy_cost
    model.setObjective(total_cost)

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            if logger:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                sol = {}
                for v in model.getVars():
                    if model.cbGetSolution(v) > 1e-6:
                        sol[v.VarName] = model.cbGetSolution(v)
                logger.log_solution(obj, sol)

    model.optimize(callback)

    solution = {}
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        solution = {
            "instance_path": args.instance_path,
            "solution_path": args.solution_path,
            "time_limit_seconds": args.time_limit,
            "solver_status": "Optimal" if model.status == GRB.OPTIMAL else "TimeLimit",
            "objective_value": model.ObjVal,
            "model_variables": {}
        }
        for v in model.getVars():
            if v.X > 1e-6:
                solution["model_variables"][v.VarName] = v.X
    else:
        solution = {
            "instance_path": args.instance_path,
            "solution_path": args.solution_path,
            "time_limit_seconds": args.time_limit,
            "solver_status": "Infeasible" if model.status == GRB.INFEASIBLE else "Error",
            "objective_value": None,
            "model_variables": {}
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()