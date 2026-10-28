import argparse
import json
import math
import sys
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def solve(instance_path, solution_path, time_limit, log_path=None):
    with open(instance_path, 'r') as f:
        inst = json.load(f)

    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    beta = inst["objective_weight_beta"]
    T = inst["time_periods"]
    periods = list(range(1, T + 1))  # 1..T

    # Electricity network
    enet = inst["electricity_network"]
    buses = {b["id"]: b for b in enet["buses"]}
    lines = {l["id"]: l for l in enet["lines"]}

    # Generators
    gdata = inst["generators"]
    gens = {g["id"]: g for g in gdata["generators"]]
    gfpps = {g["id"]: g for g in gdata["generators"] if g["is_gfpp"]}

    # Gas network
    gnet = inst["gas_network"]
    junctions = {j["id"]: j for j in gnet["junctions"]}
    connections = {c["id"]: c for c in gnet["connections"]}
    pricing_zones = {z["id"]: z for z in gnet["pricing_zones"]}
    max_gas_price = gnet["max_gas_price_mmBtu"]
    min_gas_price = gnet["min_gas_price_mmBtu"]

    # Build bus-to-generators map
    bus_gens = {}
    for g in gens.values():
        bus_gens.setdefault(g["bus"], []).append(g["id"])

    # Build junction-to-gfpp map
    junc_gfpps = {}
    for g in gfpps.values():
        junc_gfpps.setdefault(g["gas_junction"], []).append(g["id"])

    # Build junction-to-zone map
    junc_zone = {}
    for z in pricing_zones.values():
        for j in z["junctions"]:
            junc_zone[j] = z["id"]

    # Build bus adjacency for lines
    bus_lines_from = {}  # lines where bus is from_bus
    bus_lines_to = {}    # lines where bus is to_bus
    for lid, l in lines.items():
        bus_lines_from.setdefault(l["from_bus"], []).append(lid)
        bus_lines_to.setdefault(l["to_bus"], []).append(lid)

    # Build junction adjacency for connections
    junc_conn_from = {}
    junc_conn_to = {}
    for cid, c in connections.items():
        junc_conn_from.setdefault(c["from_junction"], []).append(cid)
        junc_conn_to.setdefault(c["to_junction"], []).append(cid)

    # Create model
    m = gp.Model("UCGNA")
    m.setParam("Seed", 0)
    m.setParam("MIPGap", 1e-4)
    m.setParam("NumericFocus", 0)
    m.setParam("Threads", 1)
    m.setParam("TimeLimit", max(time_limit - 5, 1))

    # ===================== VARIABLES =====================

    # Generator commitment variables
    # o[g][t] for t=0..T (t=0 is initial)
    o = {}
    vp = {}  # startup
    vm = {}  # shutdown
    r_var = {}  # startup cost
    p = {}  # power output, t=0..T
    phimax = {}  # max gas price for gfpps

    for gid, g in gens.items():
        # t=0: fixed initial status
        o[gid, 0] = m.addVar(lb=g["initial_status"], ub=g["initial_status"], vtype=GRB.BINARY, name=f"o_{gid}_0")
        p[gid, 0] = m.addVar(lb=g["initial_gen"], ub=g["initial_gen"], name=f"p_{gid}_0")

        for t in periods:
            o[gid, t] = m.addVar(vtype=GRB.BINARY, name=f"o_{gid}_{t}")
            vp[gid, t] = m.addVar(vtype=GRB.BINARY, name=f"vp_{gid}_{t}")
            vm[gid, t] = m.addVar(vtype=GRB.BINARY, name=f"vm_{gid}_{t}")
            r_var[gid, t] = m.addVar(lb=0, name=f"r_{gid}_{t}")
            p[gid, t] = m.addVar(lb=0, ub=g["max_power"], name=f"p_{gid}_{t}")

    # Bid selection and generation
    w = {}
    se = {}
    for gid, g in gens.items():
        for bid in g["bids"]:
            bid_id = bid["id"]
            for t in periods:
                w[gid, bid_id, t] = m.addVar(vtype=GRB.BINARY, name=f"w_{gid}_{bid_id}_{t}")
                se[gid, bid_id, t] = m.addVar(lb=0, ub=bid["max_amount"], name=f"se_{gid}_{bid_id}_{t}")

    # GFPP phimax
    for gid, g in gfpps.items():
        for t in periods:
            phimax[gid, t] = m.addVar(lb=0, ub=max_gas_price, name=f"phimax_{gid}_{t}")

    # Line flows
    f_var = {}
    for lid, l in lines.items():
        for t in periods:
            f_var[lid, t] = m.addVar(lb=-l["thermal_limit"], ub=l["thermal_limit"], name=f"f_{lid}_{t}")

    # Voltage angles
    theta = {}
    for bid_val, b in buses.items():
        for t in periods:
            theta[bid_val, t] = m.addVar(lb=b["voltage_angle_lb"], ub=b["voltage_angle_ub"], name=f"theta_{bid_val}_{t}")

    # Gas supply
    sg = {}
    sgs = {}
    for jid, j in junctions.items():
        for t in periods:
            sg[jid, t] = m.addVar(lb=0, name=f"sg_{jid}_{t}")
            if j["is_source"]:
                for si in j["supply_intervals"]:
                    sgs[jid, si["id"], t] = m.addVar(lb=0, ub=si["interval_ub"], name=f"sgs_{jid}_{si['id']}_{t}")

    # Squared pressure
    pisq = {}
    for jid, j in junctions.items():
        for t in periods:
            pisq[jid, t] = m.addVar(lb=j["pressure_lb_squared"], ub=j["pressure_ub_squared"], name=f"pisq_{jid}_{t}")

    # Gas flows on connections
    phig = {}
    for cid, c in connections.items():
        for t in periods:
            phig[cid, t] = m.addVar(lb=0, name=f"phig_{cid}_{t}")

    # Satisfied and shed gas demand
    lg = {}
    qg = {}
    for jid, j in junctions.items():
        for t in periods:
            lg[jid, t] = m.addVar(lb=0, name=f"lg_{jid}_{t}")
            qg[jid, t] = m.addVar(lb=0, name=f"qg_{jid}_{t}")

    # GFPP gas consumption at junction
    gamma = {}
    # Identify junctions that have GFPPs
    gfpp_junctions = set()
    for g in gfpps.values():
        gfpp_junctions.add(g["gas_junction"])
    for jid in gfpp_junctions:
        for t in periods:
            gamma[jid, t] = m.addVar(lb=0, name=f"gamma_{jid}_{t}")

    # Zonal gas prices
    psi = {}
    for zid in pricing_zones:
        for t in periods:
            psi[zid, t] = m.addVar(lb=min_gas_price, ub=max_gas_price, name=f"psi_{zid}_{t}")

    m.update()

    # ===================== CONSTRAINTS =====================

    # --- Generator commitment constraints ---
    for gid, g in gens.items():
        # Initial status enforcement
        # If initial_status = 1 and initial_active_periods < min_up_time,
        # must stay on for (min_up_time - initial_active_periods) more periods
        if g["initial_status"] == 1 and g["initial_active_periods"] < g["min_up_time"]:
            remaining = g["min_up_time"] - g["initial_active_periods"]
            for t in range(1, min(remaining + 1, T + 1)):
                m.addConstr(o[gid, t] == 1, name=f"init_on_{gid}_{t}")

        # If initial_status = 0 and initial_inactive_periods < min_down_time,
        # must stay off for (min_down_time - initial_inactive_periods) more periods
        if g["initial_status"] == 0 and g["initial_inactive_periods"] < g["min_down_time"]:
            remaining = g["min_down_time"] - g["initial_inactive_periods"]
            for t in range(1, min(remaining + 1, T + 1)):
                m.addConstr(o[gid, t] == 0, name=f"init_off_{gid}_{t}")

        for t in periods:
            # Startup/shutdown logic: vp - vm = o[t] - o[t-1]
            m.addConstr(vp[gid, t] - vm[gid, t] == o[gid, t] - o[gid, t - 1],
                        name=f"startup_logic_{gid}_{t}")
            # Can't start up and shut down simultaneously
            m.addConstr(vp[gid, t] + vm[gid, t] <= 1, name=f"no_simult_{gid}_{t}")

        # Minimum up-time constraints
        mut = g["min_up_time"]
        for t in periods:
            start = max(1, t - mut + 1)
            if t >= mut:
                m.addConstr(gp.quicksum(vp[gid, s] for s in range(start, t + 1)) <= o[gid, t],
                            name=f"min_up_{gid}_{t}")

        # Minimum down-time constraints
        mdt = g["min_down_time"]
        for t in periods:
            start = max(1, t - mdt + 1)
            if t >= mdt:
                m.addConstr(gp.quicksum(vp[gid, s] for s in range(start, t + 1)) <= 1 - o[gid, t - mdt],
                            name=f"min_down_{gid}_{t}")

        # Startup cost tiers
        # startup_cost_params is a list of [lookback_window, cost] pairs
        scp = g["startup_cost_params"]
        for t in periods:
            for tier_idx, (lookback, cost) in enumerate(scp):
                lookback = int(lookback)
                # Check if generator was off for the lookback window before t
                # The startup cost must be >= cost * vp[t] if off for the window
                # Simpler: r[g,t] >= cost * vp[g,t] for each tier
                # Actually the correct formulation: r[g,t] >= cost_k * (vp[g,t] - sum_{s in lookback window} vp[g,s])
                # But the standard formulation is:
                # r[g,t] >= cost_k * (o[g,t] - sum_{s=t-lookback..t-1} o[g,s])
                # Let me use the standard stepped startup cost formulation
                # For tier k with lookback L_k and cost C_k:
                # r[g,t] >= C_k * (vp[g,t] - sum_{s=t-L_k..t-1} vp[g,s])
                # But actually looking more carefully at the problem statement:
                # "The startup cost in each period must be at least as large as the applicable startup cost tier,
                # determined by comparing the current on status against the on/off history over the lookback window."
                # This means: if the generator starts up at t, find the tier where it has been off
                # for at least the lookback period. The startup cost is then the associated cost.
                # Standard formulation: r[g,t] >= C_k * (o[g,t] - sum_{j=t-L_k}^{t-1} o[g,j])
                sum_prev = gp.LinExpr()
                for s in range(max(0, t - lookback), t):
                    sum_prev += o[gid, s]
                m.addConstr(r_var[gid, t] >= cost * (o[gid, t] - sum_prev),
                            name=f"startup_cost_{gid}_{t}_{tier_idx}")

        # Power output bounds
        for t in periods:
            m.addConstr(p[gid, t] >= g["min_power"] * o[gid, t], name=f"pmin_{gid}_{t}")
            m.addConstr(p[gid, t] <= g["max_power"] * o[gid, t], name=f"pmax_{gid}_{t}")

        # Ramp constraints
        for t in periods:
            # Ramp up: p[t] - p[t-1] <= ramp_up * o[t-1] + max_power * vp[t]
            m.addConstr(p[gid, t] - p[gid, t - 1] <= g["ramp_up"] * o[gid, t - 1] + g["max_power"] * vp[gid, t],
                        name=f"ramp_up_{gid}_{t}")
            # Ramp down: p[t-1] - p[t] <= ramp_down * o[t] + min_power * vm[t]
            m.addConstr(p[gid, t - 1] - p[gid, t] <= g["ramp_down"] * o[gid, t] + g["max_power"] * vm[gid, t],
                        name=f"ramp_down_{gid}_{t}")

        # Bid constraints
        bids = g["bids"]
        num_bids = len(bids)

        for t in periods:
            # Total output = sum of bid amounts
            m.addConstr(p[gid, t] == gp.quicksum(se[gid, bid["id"], t] for bid in bids),
                        name=f"p_sum_bids_{gid}_{t}")

            for i, bid in enumerate(bids):
                bid_id = bid["id"]
                # Bid generation bounded by selection
                m.addConstr(se[gid, bid_id, t] <= bid["max_amount"] * w[gid, bid_id, t],
                            name=f"se_ub_{gid}_{bid_id}_{t}")

                # Bid can only be selected if generator is on
                m.addConstr(w[gid, bid_id, t] <= o[gid, t],
                            name=f"bid_on_{gid}_{bid_id}_{t}")

            # Sequential bid activation: w[i+1] can be selected only if bid i is fully utilized
            for i in range(num_bids - 1):
                bid_curr = bids[i]
                bid_next = bids[i + 1]
                # Next bid selected => current bid fully utilized
                m.addConstr(se[gid, bid_curr["id"], t] >= bid_curr["max_amount"] * w[gid, bid_next["id"], t],
                            name=f"seq_bid_{gid}_{bid_curr['id']}_{bid_next['id']}_{t}")
                # Also: next bid selected only if current is selected
                m.addConstr(w[gid, bid_next["id"], t] <= w[gid, bid_curr["id"], t],
                            name=f"seq_sel_{gid}_{bid_curr['id']}_{bid_next['id']}_{t}")

    # --- GFPP-specific constraints ---
    for gid, g in gfpps.items():
        bids = g["bids"]
        num_bids = len(bids)

        # Compute per-bid profitability thresholds
        # max_gas_price_fraction * max_gas_price for the overall generator
        # but each bid has a specific threshold
        # Actually looking at the schema: there's no per-bid profitability threshold field directly...
        # Let me re-read: "a per-bid profitability threshold representing the maximum gas price at which 
        # that particular bid remains profitable"
        # But in the schema, bids only have id, price, max_amount. The max_gas_price_fraction is per-generator.
        # So the per-bid threshold must be derived. For a GFPP with bid price p_b and heat rate,
        # the profitability threshold for bid b is: price_b / (heat_rate at marginal).
        # Actually, looking more carefully at the description:
        # "a maximum allowable fraction of the gas price" and "a per-bid profitability threshold"
        # The per-bid threshold = bid_price / marginal_heat_rate? 
        # Actually, I think the per-bid profitability threshold is computed as:
        # For bid b: threshold_b = bid_price_b / H_u1 (if linear heat rate)
        # But with quadratic: it's more complex.
        
        # Wait - re-reading the problem: "Each GFPP additionally has... a maximum allowable fraction of 
        # the gas price at which its bids remain profitable, and a per-bid profitability threshold 
        # representing the maximum gas price at which that particular bid remains profitable."
        # 
        # The schema shows max_gas_price_fraction per generator but no per-bid threshold field.
        # I think the per-bid profitability threshold = bid_price / marginal_heat_rate_at_bid_cumulative_output
        # Or perhaps it's simply max_gas_price_fraction * max_gas_price for all bids?
        # 
        # Actually, let me look at the phimax constraint description more carefully:
        # "for each bid except the last, the bid's profitability threshold is multiplied by the difference 
        # between that bid's selection indicator and the next bid's selection indicator, and the last bid's 
        # threshold is multiplied by its own selection indicator"
        #
        # The profitability threshold per bid seems to be: bid_price / marginal_heat_rate
        # For a quadratic heat rate H(p) = H2*p^2 + H1*p + H0, marginal = 2*H2*p + H1
        # At cumulative output up to bid b: p_b = sum of max_amounts up to b
        # threshold_b = bid_price_b / (2*H2*p_b + H1)
        # But this is the only interpretation that makes sense with the data.
        
        H2 = g["heat_rate_coefficients"]["H_u2"] or 0
        H1 = g["heat_rate_coefficients"]["H_u1"] or 0
        H0 = g["heat_rate_coefficients"]["H_u0"] or 0
        
        # Compute cumulative output at each bid level
        cum_output = 0
        bid_thresholds = []
        for i, bid in enumerate(bids):
            cum_output += bid["max_amount"]
            # Marginal heat rate at cumulative output
            marginal_hr = 2 * H2 * cum_output + H1
            if marginal_hr > 1e-10:
                threshold = bid["price"] / marginal_hr
            else:
                threshold = max_gas_price
            bid_thresholds.append(min(threshold, max_gas_price))

        for t in periods:
            # phimax = sum over bids of threshold_b * indicator
            expr = gp.LinExpr()
            for i in range(num_bids - 1):
                expr += bid_thresholds[i] * (w[gid, bids[i]["id"], t] - w[gid, bids[i + 1]["id"], t])
            expr += bid_thresholds[num_bids - 1] * w[gid, bids[num_bids - 1]["id"], t]
            m.addConstr(phimax[gid, t] == expr, name=f"phimax_def_{gid}_{t}")

            # Gas price constraint: phimax >= psi * o[g,t]
            # This is: phimax[g,t] >= psi[zone, t] * o[g,t]
            # Linearize: phimax[g,t] >= psi[zone,t] - max_gas_price * (1 - o[g,t])
            zone_id = junc_zone[g["gas_junction"]]
            # Create auxiliary variable aux = psi * o
            # aux <= psi, aux <= max_gas_price * o, aux >= psi - max_gas_price*(1-o), aux >= 0
            # Then phimax >= aux
            # Simpler: phimax >= psi - max_gas_price*(1 - o)
            m.addConstr(phimax[gid, t] >= psi[zone_id, t] - max_gas_price * (1 - o[gid, t]),
                        name=f"phimax_lb_{gid}_{t}")

    # --- Power balance at each bus ---
    for bid_val, b in buses.items():
        for t in periods:
            gen_sum = gp.quicksum(p[gid, t] for gid in bus_gens.get(bid_val, []))
            flow_out = gp.quicksum(f_var[lid, t] for lid in bus_lines_from.get(bid_val, []))
            flow_in = gp.quicksum(f_var[lid, t] for lid in bus_lines_to.get(bid_val, []))
            m.addConstr(gen_sum - b["demand_profile"][t - 1] == flow_out - flow_in,
                        name=f"power_balance_{bid_val}_{t}")

    # --- Line flow constraints ---
    for lid, l in lines.items():
        for t in periods:
            m.addConstr(f_var[lid, t] == l["susceptance"] * (theta[l["from_bus"], t] - theta[l["to_bus"], t]),
                        name=f"flow_def_{lid}_{t}")
            # Angle difference limit
            m.addConstr(theta[l["from_bus"], t] - theta[l["to_bus"], t] <= l["angle_diff_limit"],
                        name=f"angle_diff_ub_{lid}_{t}")
            m.addConstr(theta[l["from_bus"], t] - theta[l["to_bus"], t] >= -l["angle_diff_limit"],
                        name=f"angle_diff_lb_{lid}_{t}")

    # --- Gas network constraints ---

    # Gas balance at each junction
    for jid, j in junctions.items():
        for t in periods:
            lhs = sg[jid, t] - lg[jid, t]
            if jid in gfpp_junctions:
                lhs -= gamma[jid, t]
            flow_out = gp.quicksum(phig[cid, t] for cid in junc_conn_from.get(jid, []))
            flow_in = gp.quicksum(phig[cid, t] for cid in junc_conn_to.get(jid, []))
            m.addConstr(lhs == flow_out - flow_in, name=f"gas_balance_{jid}_{t}")

    # Source junction supply
    for jid, j in junctions.items():
        for t in periods:
            if j["is_source"]:
                m.addConstr(sg[jid, t] == gp.quicksum(sgs[jid, si["id"], t] for si in j["supply_intervals"]),
                            name=f"supply_sum_{jid}_{t}")
                for si in j["supply_intervals"]:
                    m.addConstr(sgs[jid, si["id"], t] >= si["interval_lb"],
                                name=f"supply_int_lb_{jid}_{si['id']}_{t}")
                    m.addConstr(sgs[jid, si["id"], t] <= si["interval_ub"],
                                name=f"supply_int_ub_{jid}_{si['id']}_{t}")
            else:
                m.addConstr(sg[jid, t] == 0, name=f"no_supply_{jid}_{t}")

    # Demand satisfaction
    for jid, j in junctions.items():
        for t in periods:
            demand = j["gas_demand_profile"][t - 1]
            m.addConstr(lg[jid, t] + qg[jid, t] == demand, name=f"demand_sat_{jid}_{t}")
            m.addConstr(qg[jid, t] <= demand, name=f"shed_ub_{jid}_{t}")
            m.addConstr(lg[jid, t] <= demand, name=f"lg_ub_{jid}_{t}")

    # Pipeline Weymouth constraint (convex relaxation)
    # pisq[from] - pisq[to] >= W * phig^2
    for cid, c in connections.items():
        if c["type"] == "pipeline":
            W = c["weymouth_factor"]
            for t in periods:
                m.addConstr(pisq[c["from_junction"], t] - pisq[c["to_junction"], t] >= W * phig[cid, t] * phig[cid, t],
                            name=f"weymouth_{cid}_{t}")

    # Compressor constraints
    for cid, c in connections.items():
        if c["type"] == "compressor":
            cr_lb = c["compression_ratio_lb"]
            cr_ub = c["compression_ratio_ub"]
            for t in periods:
                m.addConstr(pisq[c["to_junction"], t] >= cr_lb ** 2 * pisq[c["from_junction"], t],
                            name=f"comp_lb_{cid}_{t}")
                m.addConstr(pisq[c["to_junction"], t] <= cr_ub ** 2 * pisq[c["from_junction"], t],
                            name=f"comp_ub_{cid}_{t}")

    # Control valve constraints
    for cid, c in connections.items():
        if c["type"] == "control_valve":
            cv_lb = c["control_ratio_lb"]
            cv_ub = c["control_ratio_ub"]
            for t in periods:
                m.addConstr(pisq[c["to_junction"], t] >= cv_lb ** 2 * pisq[c["from_junction"], t],
                            name=f"cv_lb_{cid}_{t}")
                m.addConstr(pisq[c["to_junction"], t] <= cv_ub ** 2 * pisq[c["from_junction"], t],
                            name=f"cv_ub_{cid}_{t}")

    # GFPP gas consumption: gamma[j,t] >= sum over gfpps at j of (H2*p^2 + H1*p + H0*o)
    for jid in gfpp_junctions:
        for t in periods:
            consumption_expr = gp.QuadExpr()
            for gid in junc_gfpps[jid]:
                g = gens[gid]
                H2 = g["heat_rate_coefficients"]["H_u2"] or 0
                H1 = g["heat_rate_coefficients"]["H_u1"] or 0
                H0 = g["heat_rate_coefficients"]["H_u0"] or 0
                consumption_expr += H2 * p[gid, t] * p[gid, t] + H1 * p[gid, t] + H0 * o[gid, t]
            m.addConstr(gamma[jid, t] >= consumption_expr, name=f"gas_consumption_{jid}_{t}")

    # ===================== OBJECTIVE =====================

    # Electricity cost: sum over generators and periods of (no_load_cost * o + r + sum_bids(price * se))
    elec_cost = gp.LinExpr()
    for gid, g in gens.items():
        for t in periods:
            elec_cost += g["no_load_cost"] * o[gid, t]
            elec_cost += r_var[gid, t]
            for bid in g["bids"]:
                elec_cost += bid["price"] * se[gid, bid["id"], t]

    # Gas cost: sum over junctions and periods of (shedding_cost * qg + supply_interval_cost)
    gas_cost = gp.LinExpr()
    for jid, j in junctions.items():
        for t in periods:
            gas_cost += j["demand_shedding_cost"] * qg[jid, t]
            if j["is_source"]:
                for si in j["supply_intervals"]:
                    gas_cost += si["slope"] * sgs[jid, si["id"], t]

    m.setObjective(beta * elec_cost + (1 - beta) * gas_cost, GRB.MINIMIZE)

    # ===================== CALLBACK for logging =====================
    best_obj = [float('inf')]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0] - 1e-6:
                best_obj[0] = obj_val
                if logger:
                    logger.log(obj_val)

    m.optimize(callback)

    # ===================== EXTRACT SOLUTION =====================
    sol = {
        "objective_value": None,
        "released_problem_variant": "integrated_primal_ucgna",
        "objective_weight_beta": beta,
        "primary_variables": {}
    }

    if m.SolCount > 0:
        sol["objective_value"] = m.ObjVal
        pv = sol["primary_variables"]

        for gid, g in gens.items():
            # o for t=0..T
            for t in range(0, T + 1):
                pv[f"o_{gid}_{t}"] = round(o[gid, t].X)

            # p for t=0..T
            for t in range(0, T + 1):
                pv[f"p_{gid}_{t}"] = p[gid, t].X

            for t in periods:
                pv[f"vp_{gid}_{t}"] = round(vp[gid, t].X)
                pv[f"vm_{gid}_{t}"] = round(vm[gid, t].X)
                pv[f"r_{gid}_{t}"] = r_var[gid, t].X

                for bid in g["bids"]:
                    bid_id = bid["id"]
                    pv[f"w_{gid}_{bid_id}_{t}"] = round(w[gid, bid_id, t].X)
                    pv[f"se_{gid}_{bid_id}_{t}"] = se[gid, bid_id, t].X

            if g["is_gfpp"]:
                for t in periods:
                    pv[f"phimax_{gid}_{t}"] = phimax[gid, t].X

        for lid in lines:
            for t in periods:
                pv[f"f_{lid}_{t}"] = f_var[lid, t].X

        for bid_val in buses:
            for t in periods:
                pv[f"theta_{bid_val}_{t}"] = theta[bid_val, t].X

        for jid, j in junctions.items():
            for t in periods:
                pv[f"sg_{jid}_{t}"] = sg[jid, t].X
                if j["is_source"]:
                    for si in j["supply_intervals"]:
                        pv[f"sgs_{jid}_{si['id']}_{t}"] = sgs[jid, si["id"], t].X
                pv[f"pisq_{jid}_{t}"] = pisq[jid, t].X
                pv[f"lg_{jid}_{t}"] = lg[jid, t].X
                pv[f"qg_{jid}_{t}"] = qg[jid, t].X

        for cid in connections:
            for t in periods:
                pv[f"phig_{cid}_{t}"] = phig[cid, t].X

        for jid in gfpp_junctions:
            for t in periods:
                pv[f"gamma_{jid}_{t}"] = gamma[jid, t].X

        # Also add gamma for junctions that have no GFPPs but might be expected
        # Actually only junctions with GFPPs need gamma
        for jid in junctions:
            if jid not in gfpp_junctions:
                for t in periods:
                    pv[f"gamma_{jid}_{t}"] = 0.0

        for zid in pricing_zones:
            for t in periods:
                pv[f"psi_{zid}_{t}"] = psi[zid, t].X

        # Also ensure sgs for non-source junctions
        for jid, j in junctions.items():
            if not j["is_source"]:
                # No supply intervals, but check if we need empty entries
                pass

        if logger:
            logger.log_solution(sol["objective_value"], sol)
    else:
        sol["objective_value"] = float('inf')
        sol["primary_variables"] = {}

    with open(solution_path, 'w') as f:
        json.dump(sol, f, indent=2)

    return sol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == "__main__":
    main()