import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


ZERO_TOL = 1e-9


def numeric_value(value, zone=None, zone_position=None):
    """Read a scalar that may optionally be represented by zone."""
    if isinstance(value, dict):
        if zone is not None:
            if str(zone) in value:
                return float(value[str(zone)])
            if zone in value:
                return float(value[zone])
        return float(next(iter(value.values())))
    if isinstance(value, list):
        if zone_position is not None and zone_position < len(value):
            return float(value[zone_position])
        return float(value[0])
    return float(value)


def get_profile(profile_collection, resource_id, total_timesteps, default=1.0):
    profile = profile_collection.get(str(resource_id))
    if profile is None:
        profile = profile_collection.get(resource_id)

    if profile is None:
        return [float(default)] * total_timesteps

    if isinstance(profile, (int, float)):
        return [float(profile)] * total_timesteps

    if isinstance(profile, list):
        values = [float(x) for x in profile]
    elif profile.get("type") == "constant":
        values = [float(profile.get("value", default))] * total_timesteps
    else:
        values = [float(x) for x in profile.get("values", [])]

    if not values:
        return [float(default)] * total_timesteps
    if len(values) < total_timesteps:
        values.extend([values[-1]] * (total_timesteps - len(values)))
    return values[:total_timesteps]


def safe_float(record, key, default=0.0):
    value = record.get(key, default)
    return float(default if value is None else value)


def integer_upper_bound(quantity, unit_size):
    if unit_size <= 0:
        return 0
    return max(0, int(math.floor((quantity + 1e-7) / unit_size)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dimensions = data["problem_dimensions"]
    subperiod_data = data["subperiods"]
    policy = data["policy"]

    generators = data.get("generators", [])
    lines = data.get("transmission_lines", [])
    nse_segments = data.get("nse_segments", [])
    demand_data = data.get("demand", {})

    num_subperiods = int(
        subperiod_data.get("num_subperiods", dimensions.get("num_weeks", 1))
    )
    hours_per_subperiod = int(
        subperiod_data.get(
            "hours_per_subperiod", dimensions.get("hours_per_week", 168)
        )
    )
    total_timesteps = int(
        dimensions.get(
            "total_timesteps", num_subperiods * hours_per_subperiod
        )
    )
    circular = bool(subperiod_data.get("circular_indexing", True))

    subperiod_weights = [
        float(x) for x in subperiod_data.get(
            "subperiod_weights", [1.0] * num_subperiods
        )
    ]
    if len(subperiod_weights) < num_subperiods:
        subperiod_weights.extend(
            [1.0] * (num_subperiods - len(subperiod_weights))
        )

    # Zone IDs are taken from demand whenever possible, then augmented from
    # generators and transmission endpoints.
    zones = []
    for key in demand_data:
        try:
            zones.append(int(key))
        except (TypeError, ValueError):
            zones.append(key)
    for g in generators:
        if g["zone"] not in zones:
            zones.append(g["zone"])
    for line in lines:
        if line["from_zone"] not in zones:
            zones.append(line["from_zone"])
        if line["to_zone"] not in zones:
            zones.append(line["to_zone"])

    if not zones:
        num_zones = int(dimensions.get("num_zones", 0))
        zones = list(range(1, num_zones + 1))

    try:
        zones = sorted(zones)
    except TypeError:
        pass

    zone_position = {z: p for p, z in enumerate(zones)}

    demand = {}
    for z in zones:
        values = demand_data.get(str(z), demand_data.get(z, []))
        values = [float(x) for x in values]
        if len(values) < total_timesteps:
            values.extend([0.0] * (total_timesteps - len(values)))
        demand[z] = values[:total_timesteps]

    availability_profiles = data.get("availability_profiles", {})
    hydro_profile_collection = data.get(
        "hydro_inflow_profiles",
        data.get("hydro_inflows", availability_profiles),
    )

    gen_by_id = {int(g["id"]): g for g in generators}
    line_by_id = {int(line["id"]): line for line in lines}
    gen_ids = list(gen_by_id)
    line_ids = list(line_by_id)

    storage_ids = [i for i in gen_ids if bool(gen_by_id[i].get("is_storage"))]
    hydro_ids = [i for i in gen_ids if bool(gen_by_id[i].get("is_hydro"))]
    uc_ids = [i for i in gen_ids if bool(gen_by_id[i].get("is_UC"))]
    regular_ids = [
        i for i in gen_ids
        if i not in storage_ids and i not in hydro_ids and i not in uc_ids
    ]

    availability = {
        i: get_profile(availability_profiles, i, total_timesteps, default=1.0)
        for i in gen_ids
    }
    hydro_inflow = {
        i: get_profile(hydro_profile_collection, i, total_timesteps, default=0.0)
        for i in hydro_ids
    }

    week_of_t = [
        min(num_subperiods - 1, t // hours_per_subperiod)
        for t in range(total_timesteps)
    ]
    hour_weights = [
        subperiod_weights[week_of_t[t]] / float(hours_per_subperiod)
        for t in range(total_timesteps)
    ]

    week_hours = []
    for w in range(num_subperiods):
        first = w * hours_per_subperiod
        last = min(total_timesteps, first + hours_per_subperiod)
        week_hours.append(list(range(first, last)))

    model = gp.Model("capacity_expansion")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.MIPFocus = 1

    # Projected model variables and metadata used to construct the witness.
    projected = {}
    integer_projected_names = set()

    yPnew = {}
    yPret = {}
    yP = {}

    yEnew = {}
    yEret = {}
    yE = {}

    yFnew = {}

    xinj = {}
    xwdw = {}
    xsoc = {}
    xhydro = {}
    xspill = {}
    xflow = {}
    xnse = {}
    xcommit = {}
    xstart = {}
    xshut = {}
    xRPS = {}
    xCO2 = {}

    # Capacity investment variables.
    for i in gen_ids:
        g = gen_by_id[i]
        size = safe_float(g, "capacity_size_MW", 1.0)
        if size <= 0:
            size = 1.0

        existing = safe_float(g, "existing_capacity_MW", 0.0)
        max_total = safe_float(g, "max_capacity_MW", existing)
        max_total = max(0.0, max_total)

        # max_capacity_MW is interpreted according to the schema as a maximum
        # total capacity. New construction potential is capacity above existing.
        new_potential = max(0.0, max_total - existing)
        new_ub = integer_upper_bound(new_potential, size)
        retire_ub = (
            integer_upper_bound(existing, size)
            if bool(g.get("can_retire", True))
            else 0
        )

        yPnew[i] = model.addVar(
            vtype=GRB.INTEGER, lb=0, ub=new_ub, name=f"yPnew_{i}"
        )
        yPret[i] = model.addVar(
            vtype=GRB.INTEGER, lb=0, ub=retire_ub, name=f"yPret_{i}"
        )
        yP[i] = model.addVar(
            vtype=GRB.CONTINUOUS, lb=0, ub=max_total, name=f"yP_{i}"
        )

        projected[f"yPnew_{i}"] = yPnew[i]
        projected[f"yPret_{i}"] = yPret[i]
        projected[f"yP_{i}"] = yP[i]
        integer_projected_names.add(f"yPnew_{i}")
        integer_projected_names.add(f"yPret_{i}")

        model.addConstr(
            yP[i] == existing + size * (yPnew[i] - yPret[i]),
            name=f"power_capacity_accounting_{i}",
        )

        if i in storage_ids:
            e_size = safe_float(g, "storage_capacity_size_MWh", 1.0)
            if e_size <= 0:
                e_size = 1.0

            e_existing = safe_float(g, "existing_storage_capacity_MWh", 0.0)
            e_max = safe_float(g, "max_storage_capacity_MWh", e_existing)
            e_max = max(0.0, e_max)
            e_new_potential = max(0.0, e_max - e_existing)

            e_new_ub = integer_upper_bound(e_new_potential, e_size)
            e_ret_ub = (
                integer_upper_bound(e_existing, e_size)
                if bool(g.get("can_retire", True))
                else 0
            )

            yEnew[i] = model.addVar(
                vtype=GRB.INTEGER, lb=0, ub=e_new_ub, name=f"yEnew_{i}"
            )
            yEret[i] = model.addVar(
                vtype=GRB.INTEGER, lb=0, ub=e_ret_ub, name=f"yEret_{i}"
            )
            yE[i] = model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, ub=e_max, name=f"yE_{i}"
            )

            projected[f"yEnew_{i}"] = yEnew[i]
            projected[f"yEret_{i}"] = yEret[i]
            projected[f"yE_{i}"] = yE[i]
            integer_projected_names.add(f"yEnew_{i}")
            integer_projected_names.add(f"yEret_{i}")

            model.addConstr(
                yE[i] == e_existing + e_size * (yEnew[i] - yEret[i]),
                name=f"energy_capacity_accounting_{i}",
            )

            min_duration = safe_float(g, "min_duration_MWh_per_MW", 0.0)
            max_duration = safe_float(g, "max_duration_MWh_per_MW", GRB.INFINITY)

            model.addConstr(
                yE[i] >= min_duration * yP[i],
                name=f"minimum_duration_{i}",
            )
            if math.isfinite(max_duration):
                model.addConstr(
                    yE[i] <= max_duration * yP[i],
                    name=f"maximum_duration_{i}",
                )

    for l in line_ids:
        line = line_by_id[l]
        max_new = max(0.0, safe_float(line, "max_new_capacity_MW", 0.0))
        yFnew[l] = model.addVar(
            vtype=GRB.INTEGER,
            lb=0,
            ub=math.floor(max_new + 1e-7),
            name=f"yFnew_{l}",
        )
        projected[f"yFnew_{l}"] = yFnew[l]
        integer_projected_names.add(f"yFnew_{l}")

    # Operational variables.
    for i in gen_ids:
        for t in range(total_timesteps):
            external_t = t + 1
            name = f"xinj_{i}_{external_t}"
            xinj[i, t] = model.addVar(lb=0, name=name)
            projected[name] = xinj[i, t]

    for i in storage_ids:
        for t in range(total_timesteps):
            external_t = t + 1

            wdw_name = f"xwdw_{i}_{external_t}"
            soc_name = f"xsoc_{i}_{external_t}"

            xwdw[i, t] = model.addVar(lb=0, name=wdw_name)
            xsoc[i, t] = model.addVar(lb=0, name=soc_name)

            projected[wdw_name] = xwdw[i, t]
            projected[soc_name] = xsoc[i, t]

    for i in hydro_ids:
        for t in range(total_timesteps):
            external_t = t + 1
            hydro_name = f"xhydro_{i}_{external_t}"
            xhydro[i, t] = model.addVar(lb=0, name=hydro_name)
            xspill[i, t] = model.addVar(lb=0, name=f"_spill_{i}_{external_t}")
            projected[hydro_name] = xhydro[i, t]

    for l in line_ids:
        for t in range(total_timesteps):
            external_t = t + 1
            name = f"xflow_{l}_{external_t}"
            xflow[l, t] = model.addVar(lb=-GRB.INFINITY, name=name)
            projected[name] = xflow[l, t]

    segment_ids = []
    segment_by_id = {}
    for position, segment in enumerate(nse_segments):
        s = int(segment.get("segment", position + 1))
        segment_ids.append(s)
        segment_by_id[s] = segment

    for s in segment_ids:
        for z in zones:
            for t in range(total_timesteps):
                external_t = t + 1
                name = f"xnse_{s}_{z}_{external_t}"
                xnse[s, z, t] = model.addVar(lb=0, name=name)
                projected[name] = xnse[s, z, t]

    for i in uc_ids:
        for t in range(total_timesteps):
            external_t = t + 1
            commit_name = f"xcommit_{i}_{external_t}"
            start_name = f"xstart_{i}_{external_t}"
            shut_name = f"xshut_{i}_{external_t}"

            xcommit[i, t] = model.addVar(
                vtype=GRB.INTEGER, lb=0, name=commit_name
            )
            xstart[i, t] = model.addVar(
                vtype=GRB.INTEGER, lb=0, name=start_name
            )
            xshut[i, t] = model.addVar(
                vtype=GRB.INTEGER, lb=0, name=shut_name
            )

            projected[commit_name] = xcommit[i, t]
            projected[start_name] = xstart[i, t]
            projected[shut_name] = xshut[i, t]

            integer_projected_names.update(
                {commit_name, start_name, shut_name}
            )

    if bool(policy.get("RPS_enabled", False)):
        for w in range(num_subperiods):
            name = f"xRPS_{w + 1}"
            xRPS[w] = model.addVar(lb=0, name=name)
            projected[name] = xRPS[w]

    if bool(policy.get("CO2_cap_enabled", False)):
        for w in range(num_subperiods):
            name = f"xCO2_{w + 1}"
            xCO2[w] = model.addVar(lb=0, name=name)
            projected[name] = xCO2[w]

    model.update()

    resources_in_zone = {
        z: [i for i in gen_ids if gen_by_id[i]["zone"] == z] for z in zones
    }
    storage_in_zone = {
        z: [i for i in storage_ids if gen_by_id[i]["zone"] == z] for z in zones
    }
    outgoing_lines = {
        z: [l for l in line_ids if line_by_id[l]["from_zone"] == z]
        for z in zones
    }
    incoming_lines = {
        z: [l for l in line_ids if line_by_id[l]["to_zone"] == z]
        for z in zones
    }

    # Zonal power balances.
    for z in zones:
        for t in range(total_timesteps):
            model.addConstr(
                gp.quicksum(xinj[i, t] for i in resources_in_zone[z])
                - gp.quicksum(xwdw[i, t] for i in storage_in_zone[z])
                - gp.quicksum(xflow[l, t] for l in outgoing_lines[z])
                + gp.quicksum(xflow[l, t] for l in incoming_lines[z])
                + gp.quicksum(xnse[s, z, t] for s in segment_ids)
                == demand[z][t],
                name=f"power_balance_{z}_{t + 1}",
            )

    # Non-served energy segment bounds.
    for s in segment_ids:
        segment = segment_by_id[s]
        max_frac_data = segment.get("max_frac", 0.0)
        for z in zones:
            max_frac = numeric_value(
                max_frac_data, z, zone_position.get(z)
            )
            for t in range(total_timesteps):
                model.addConstr(
                    xnse[s, z, t] <= max_frac * demand[z][t],
                    name=f"nse_bound_{s}_{z}_{t + 1}",
                )

    # Transmission capacity.
    for l in line_ids:
        line = line_by_id[l]
        existing = safe_float(line, "existing_capacity_MW", 0.0)
        total_capacity = existing + yFnew[l]
        for t in range(total_timesteps):
            model.addConstr(
                xflow[l, t] <= total_capacity,
                name=f"flow_upper_{l}_{t + 1}",
            )
            model.addConstr(
                xflow[l, t] >= -total_capacity,
                name=f"flow_lower_{l}_{t + 1}",
            )

    # Availability and minimum generation.
    for i in gen_ids:
        g = gen_by_id[i]
        min_output = safe_float(g, "min_output_frac", 0.0)

        if i not in uc_ids:
            for t in range(total_timesteps):
                model.addConstr(
                    xinj[i, t] <= availability[i][t] * yP[i],
                    name=f"generation_availability_{i}_{t + 1}",
                )

        if i in regular_ids:
            for t in range(total_timesteps):
                model.addConstr(
                    xinj[i, t] >= min_output * yP[i],
                    name=f"minimum_generation_{i}_{t + 1}",
                )

        if i in hydro_ids:
            duration = safe_float(g, "duration_MWh_per_MW", 0.0)
            for t in range(total_timesteps):
                model.addConstr(
                    xhydro[i, t] <= duration * yP[i],
                    name=f"hydro_reservoir_capacity_{i}_{t + 1}",
                )
                model.addConstr(
                    xinj[i, t] + xspill[i, t] >= min_output * yP[i],
                    name=f"hydro_minimum_release_{i}_{t + 1}",
                )

    # Storage operation and circular state equations.
    for i in storage_ids:
        g = gen_by_id[i]
        eta_charge = safe_float(g, "charge_efficiency", 1.0)
        eta_discharge = safe_float(g, "discharge_efficiency", 1.0)
        eta_discharge = max(eta_discharge, 1e-8)
        self_discharge = safe_float(g, "self_discharge_rate", 0.0)

        for t in range(total_timesteps):
            avail = availability[i][t]
            model.addConstr(
                xwdw[i, t] <= avail * yP[i],
                name=f"storage_charge_availability_{i}_{t + 1}",
            )
            model.addConstr(
                xinj[i, t] + xwdw[i, t] <= yP[i],
                name=f"storage_power_capacity_{i}_{t + 1}",
            )
            model.addConstr(
                eta_charge * xwdw[i, t] <= yE[i] - xsoc[i, t],
                name=f"storage_charging_room_{i}_{t + 1}",
            )
            model.addConstr(
                xinj[i, t] / eta_discharge <= xsoc[i, t],
                name=f"storage_discharge_inventory_{i}_{t + 1}",
            )
            model.addConstr(
                xsoc[i, t] <= yE[i],
                name=f"storage_energy_capacity_{i}_{t + 1}",
            )

        for w, hours in enumerate(week_hours):
            if not hours:
                continue
            for local_pos, t in enumerate(hours):
                if local_pos == 0:
                    if not circular:
                        continue
                    prev = hours[-1]
                else:
                    prev = hours[local_pos - 1]

                model.addConstr(
                    xsoc[i, t]
                    == (1.0 - self_discharge) * xsoc[i, prev]
                    + eta_charge * xwdw[i, t]
                    - xinj[i, t] / eta_discharge,
                    name=f"storage_balance_{i}_{t + 1}",
                )

    # Hydropower reservoir equations.
    for i in hydro_ids:
        for w, hours in enumerate(week_hours):
            if not hours:
                continue
            for local_pos, t in enumerate(hours):
                if local_pos == 0:
                    if not circular:
                        continue
                    prev = hours[-1]
                else:
                    prev = hours[local_pos - 1]

                model.addConstr(
                    xhydro[i, t]
                    == xhydro[i, prev]
                    + hydro_inflow[i][t] * yP[i]
                    - xinj[i, t]
                    - xspill[i, t],
                    name=f"hydro_balance_{i}_{t + 1}",
                )

    # Non-UC ramping, excluding storage resources lacking ramp parameters.
    for i in gen_ids:
        if i in uc_ids or i in storage_ids:
            continue

        g = gen_by_id[i]
        if g.get("ramp_up_frac_per_hr") is None:
            continue

        ramp_up = safe_float(g, "ramp_up_frac_per_hr", 1.0)
        ramp_down = safe_float(g, "ramp_dn_frac_per_hr", 1.0)

        for w, hours in enumerate(week_hours):
            if not hours:
                continue
            for local_pos, t in enumerate(hours):
                if local_pos == 0:
                    if not circular:
                        continue
                    prev = hours[-1]
                else:
                    prev = hours[local_pos - 1]

                model.addConstr(
                    xinj[i, t] - xinj[i, prev] <= ramp_up * yP[i],
                    name=f"ramp_up_{i}_{t + 1}",
                )
                model.addConstr(
                    xinj[i, prev] - xinj[i, t] <= ramp_down * yP[i],
                    name=f"ramp_down_{i}_{t + 1}",
                )

    # Unit commitment.
    for i in uc_ids:
        g = gen_by_id[i]
        size = safe_float(g, "capacity_size_MW", 1.0)
        size = max(size, 1e-8)

        min_output = safe_float(g, "min_output_frac", 0.0)
        ramp_up = safe_float(g, "ramp_up_frac_per_hr", 1.0)
        ramp_down = safe_float(g, "ramp_dn_frac_per_hr", 1.0)
        min_up = max(0, int(g.get("min_up_time_hr", 0) or 0))
        min_down = max(0, int(g.get("min_down_time_hr", 0) or 0))

        for t in range(total_timesteps):
            available_units = yP[i] / size

            model.addConstr(
                xcommit[i, t] <= available_units,
                name=f"commit_capacity_{i}_{t + 1}",
            )
            model.addConstr(
                xstart[i, t] <= available_units,
                name=f"startup_capacity_{i}_{t + 1}",
            )
            model.addConstr(
                xshut[i, t] <= available_units,
                name=f"shutdown_capacity_{i}_{t + 1}",
            )

            model.addConstr(
                xinj[i, t] >= size * min_output * xcommit[i, t],
                name=f"uc_minimum_generation_{i}_{t + 1}",
            )
            model.addConstr(
                xinj[i, t]
                <= size * availability[i][t] * xcommit[i, t],
                name=f"uc_maximum_generation_{i}_{t + 1}",
            )

        for w, hours in enumerate(week_hours):
            if not hours:
                continue
            period_length = len(hours)

            for local_pos, t in enumerate(hours):
                if local_pos == 0:
                    if not circular:
                        continue
                    prev = hours[-1]
                else:
                    prev = hours[local_pos - 1]

                model.addConstr(
                    xcommit[i, t] - xcommit[i, prev]
                    == xstart[i, t] - xshut[i, t],
                    name=f"commitment_transition_{i}_{t + 1}",
                )

                startup_cap = min(
                    availability[i][t], max(min_output, ramp_up)
                )
                shutdown_cap = min(
                    availability[i][t], max(min_output, ramp_down)
                )

                model.addConstr(
                    xinj[i, t] - xinj[i, prev]
                    <= size * ramp_up * (xcommit[i, t] - xstart[i, t])
                    + size * startup_cap * xstart[i, t]
                    - size * min_output * xshut[i, t],
                    name=f"uc_ramp_up_{i}_{t + 1}",
                )
                model.addConstr(
                    xinj[i, prev] - xinj[i, t]
                    <= size * ramp_down * (xcommit[i, t] - xstart[i, t])
                    + size * shutdown_cap * xshut[i, t]
                    - size * min_output * xstart[i, t],
                    name=f"uc_ramp_down_{i}_{t + 1}",
                )

                if min_up > 0:
                    lookback = min(min_up, period_length)
                    startup_window = [
                        hours[(local_pos - k) % period_length]
                        for k in range(lookback)
                    ]
                    model.addConstr(
                        xcommit[i, t]
                        >= gp.quicksum(xstart[i, q] for q in startup_window),
                        name=f"minimum_up_time_{i}_{t + 1}",
                    )

                if min_down > 0:
                    lookback = min(min_down, period_length)
                    shutdown_window = [
                        hours[(local_pos - k) % period_length]
                        for k in range(lookback)
                    ]
                    model.addConstr(
                        xcommit[i, t]
                        + gp.quicksum(xshut[i, q] for q in shutdown_window)
                        <= yP[i] / size,
                        name=f"minimum_down_time_{i}_{t + 1}",
                    )

    total_weighted_demand = sum(
        hour_weights[t] * demand[z][t]
        for z in zones
        for t in range(total_timesteps)
    )

    if xRPS:
        rps_generation = gp.quicksum(
            hour_weights[t] * xinj[i, t]
            for i in gen_ids
            if bool(gen_by_id[i].get("is_RPS_qualifying", False))
            for t in range(total_timesteps)
        )
        required_share = safe_float(policy, "RPS_share", 0.0)
        model.addConstr(
            rps_generation + gp.quicksum(xRPS.values())
            >= required_share * total_weighted_demand,
            name="RPS_requirement",
        )

    if xCO2:
        emissions = gp.LinExpr()
        for i in gen_ids:
            emission_rate = safe_float(gen_by_id[i], "co2_tons_per_MWh", 0.0)
            if emission_rate == 0:
                continue
            for t in range(total_timesteps):
                emissions += (
                    hour_weights[t] * emission_rate * xinj[i, t]
                )
                if i in storage_ids:
                    emissions += (
                        hour_weights[t] * emission_rate * xwdw[i, t]
                    )

        cap_rate_value = policy.get("CO2_cap_tons_per_MWh")
        cap_rate = 0.05 if cap_rate_value is None else float(cap_rate_value)
        model.addConstr(
            emissions - gp.quicksum(xCO2.values())
            <= cap_rate * total_weighted_demand,
            name="CO2_cap",
        )

    # Objective.
    objective = gp.LinExpr()

    for i in gen_ids:
        g = gen_by_id[i]
        size = safe_float(g, "capacity_size_MW", 1.0)

        objective += (
            safe_float(g, "inv_cost_per_MW_yr", 0.0)
            * size
            * yPnew[i]
        )
        objective += safe_float(g, "fom_cost_per_MW_yr", 0.0) * yP[i]

        if i in hydro_ids:
            duration = safe_float(g, "duration_MWh_per_MW", 0.0)
            objective += (
                safe_float(g, "inv_cost_energy_per_MWh_yr", 0.0)
                * duration
                * size
                * yPnew[i]
            )
            objective += (
                safe_float(g, "fom_cost_energy_per_MWh_yr", 0.0)
                * duration
                * yP[i]
            )

        if i in storage_ids:
            e_size = safe_float(g, "storage_capacity_size_MWh", 1.0)
            objective += (
                safe_float(g, "inv_cost_energy_per_MWh_yr", 0.0)
                * e_size
                * yEnew[i]
            )
            objective += (
                safe_float(g, "fom_cost_energy_per_MWh_yr", 0.0)
                * yE[i]
            )

    for l in line_ids:
        objective += (
            safe_float(line_by_id[l], "inv_cost_per_MW_yr", 0.0)
            * yFnew[l]
        )

    for i in gen_ids:
        variable_cost = safe_float(gen_by_id[i], "var_cost_per_MWh", 0.0)
        for t in range(total_timesteps):
            objective += hour_weights[t] * variable_cost * xinj[i, t]
            if i in storage_ids:
                objective += hour_weights[t] * variable_cost * xwdw[i, t]

    for s in segment_ids:
        segment = segment_by_id[s]
        cost_data = segment.get("cost_per_MWh", 0.0)
        for z in zones:
            cost = numeric_value(cost_data, z, zone_position.get(z))
            for t in range(total_timesteps):
                objective += hour_weights[t] * cost * xnse[s, z, t]

    for i in uc_ids:
        start_cost = safe_float(gen_by_id[i], "start_cost_per_unit", 0.0)
        for t in range(total_timesteps):
            objective += hour_weights[t] * start_cost * xstart[i, t]

    if xRPS:
        penalty = safe_float(policy, "RPS_noncompliance_cost_per_MWh", 0.0)
        objective += penalty * gp.quicksum(xRPS.values())

    if xCO2:
        penalty = safe_float(policy, "CO2_noncompliance_cost_per_ton", 0.0)
        objective += penalty * gp.quicksum(xCO2.values())

    model.setObjective(objective, GRB.MINIMIZE)

    projected_names = list(projected)
    projected_vars = [projected[name] for name in projected_names]

    def clean_value(name, value):
        value = float(value)
        if name in integer_projected_names:
            nearest = round(value)
            if abs(value - nearest) <= 1e-5:
                value = float(nearest)
        if abs(value) <= ZERO_TOL:
            return 0.0
        return value

    def construct_solution(objective_value, value_lookup):
        values = {
            name: clean_value(name, value_lookup(projected[name]))
            for name in projected_names
        }

        sparse_variables = {
            name: value
            for name, value in values.items()
            if abs(value) > ZERO_TOL
        }

        investment_decisions = []
        for i in gen_ids:
            g = gen_by_id[i]
            entry = {
                "generator_id": int(i),
                "resource_type": str(g.get("resource_type", "")),
                "zone": g["zone"],
                "y_P_new": values[f"yPnew_{i}"],
                "y_P_ret": values[f"yPret_{i}"],
                "y_P_total_MW": values[f"yP_{i}"],
            }
            if i in storage_ids:
                entry.update(
                    {
                        "y_E_new": values[f"yEnew_{i}"],
                        "y_E_ret": values[f"yEret_{i}"],
                        "y_E_total_MWh": values[f"yE_{i}"],
                    }
                )
            investment_decisions.append(entry)

        transmission_decisions = []
        for l in line_ids:
            new_capacity = values[f"yFnew_{l}"]
            existing = safe_float(
                line_by_id[l], "existing_capacity_MW", 0.0
            )
            transmission_decisions.append(
                {
                    "line_id": int(l),
                    "y_F_new_MW": new_capacity,
                    "total_capacity_MW": existing + new_capacity,
                }
            )

        return {
            "objective_value": float(objective_value),
            "model_variables": sparse_variables,
            "investment_decisions": investment_decisions,
            "transmission_decisions": transmission_decisions,
        }

    last_logged_objective = [math.inf]

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL or logger is None:
            return
        try:
            objective_value = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
            if objective_value >= last_logged_objective[0] - 1e-7:
                return

            incumbent_values = cb_model.cbGetSolution(projected_vars)
            incumbent_map = {
                var: incumbent_values[k]
                for k, var in enumerate(projected_vars)
            }
            solution = construct_solution(
                objective_value, lambda var: incumbent_map[var]
            )
            logger.log_solution(objective_value, solution)
            last_logged_objective[0] = objective_value
        except Exception:
            # Logging must never interrupt the optimization.
            try:
                objective_value = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
                logger.log(objective_value)
                last_logged_objective[0] = min(
                    last_logged_objective[0], objective_value
                )
            except Exception:
                pass

    remaining_time = max(0.01, args.time_limit - (time.monotonic() - start_time))
    model.Params.TimeLimit = remaining_time
    model.optimize(incumbent_callback if logger is not None else None)

    if model.SolCount <= 0:
        # A schema-valid diagnostic witness is still written, though normal
        # benchmark instances are expected to admit a feasible solution.
        solution = {
            "objective_value": float("inf"),
            "model_variables": {},
            "investment_decisions": [],
            "transmission_decisions": [],
        }
    else:
        objective_value = float(model.ObjVal)
        solution = construct_solution(objective_value, lambda var: var.X)

        if logger is not None and objective_value < last_logged_objective[0] - 1e-7:
            logger.log_solution(objective_value, solution)

    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(solution, f, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()