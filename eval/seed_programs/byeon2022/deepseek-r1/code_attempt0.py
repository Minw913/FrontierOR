import sys
import json
import math
import gurobipy as gp
from gurobipy import GRB

def main():
    # Parse command-line arguments
    if len(sys.argv) != 4:
        print("Usage: python ucgna.py <instance_path> <solution_path> <time_limit>")
        sys.exit(1)
    
    instance_path = sys.argv[1]
    solution_path = sys.argv[2]
    time_limit = float(sys.argv[3])
    
    # Read instance JSON
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    # Extract stress parameters
    eta_p = instance['stress_parameters']['eta_p']
    eta_g = instance['stress_parameters']['eta_g']
    
    # Preprocess electricity network
    elec_net = instance['electricity_network']
    buses = {bus['id']: bus for bus in elec_net['buses']}
    lines = {line['id']: line for line in elec_net['lines']}
    generators = {gen['id']: gen for gen in elec_net['generators']}
    T = instance['time_periods']  # Planning horizon periods (1..T)
    
    # Scale electricity demands
    for bus in elec_net['buses']:
        bus['demand_profile'] = [d * eta_p for d in bus['demand_profile']]
    
    # Preprocess gas network
    gas_net = instance['gas_network']
    junctions = {junc['id']: junc for junc in gas_net['junctions']}
    connections = {conn['id']: conn for conn in gas_net['connections']}
    pricing_zones = gas_net['pricing_zones']
    min_gas_price = gas_net['min_gas_price_mmBtu']
    max_gas_price = gas_net['max_gas_price_mmBtu']
    
    # Scale gas demands
    for junc in gas_net['junctions']:
        if 'gas_demand_profile' in junc:
            junc['gas_demand_profile'] = [d * eta_g for d in junc['gas_demand_profile']]
    
    # Create junction to pricing zone mapping
    junction_to_zone = {}
    for zone in pricing_zones:
        for junc_id in zone['junctions']:
            junction_to_zone[junc_id] = zone['id']
    
    # Precompute GFPP thresholds
    gfpp_thresholds = {}
    for gen in generators.values():
        if gen['max_gas_price_fraction'] is not None:
            gfpp_thresholds[gen['id']] = gen['max_gas_price_fraction'] * max_gas_price
    
    # Initialize model
    model = gp.Model("UCGNA")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    
    # Define variables for periods 1..T
    # Electricity variables
    o = {}  # o[g,t] for generator g, period t (1..T)
    vp = {}  # vp[g,t] for generator g, period t (1..T)
    vm = {}  # vm[g,t] for generator g, period t (1..T)
    w = {}   # w[g,b,t] for generator g, bid b, period t (1..T)
    p = {}   # p[g,t] for generator g, period t (1..T)
    se = {}  # se[g,b,t] for generator g, bid b, period t (1..T)
    r = {}   # r[g,t] for generator g, period t (1..T) - startup cost
    f = {}   # f[l,t] for line l, period t (1..T)
    theta = {}  # theta[i,t] for bus i, period t (1..T)
    
    # Gas variables
    sg = {}      # sg[j,t] for junction j, period t (1..T)
    sgs = {}     # sgs[j,interval,t] for source junction j, interval, period t (1..T)
    lg = {}      # lg[j,t] for junction j, period t (1..T)
    qg = {}      # qg[j,t] for junction j, period t (1..T)
    gamma = {}   # gamma[j,t] for junction j, period t (1..T)
    pisq = {}    # pisq[j,t] for junction j, period t (1..T)
    phig = {}    # phig[c,t] for connection c, period t (1..T)
    psi = {}     # psi[z,t] for pricing zone z, period t (1..T)
    phimax = {}  # phimax[g,t] for GFPP g, period t (1..T)
    
    # Add variables to model
    # Electricity
    for gen_id, gen in generators.items():
        for t in range(1, T+1):
            o[gen_id, t] = model.addVar(vtype=GRB.BINARY, name=f"o_{gen_id}_{t}")
            vp[gen_id, t] = model.addVar(vtype=GRB.BINARY, name=f"vp_{gen_id}_{t}")
            vm[gen_id, t] = model.addVar(vtype=GRB.BINARY, name=f"vm_{gen_id}_{t}")
            p[gen_id, t] = model.addVar(lb=0, ub=gen['max_power'], name=f"p_{gen_id}_{t}")
            r[gen_id, t] = model.addVar(lb=0, name=f"r_{gen_id}_{t}")
            
            # Bid variables
            bids = gen.get('bids', [])
            for bid in bids:
                bid_id = bid['id']
                w[gen_id, bid_id, t] = model.addVar(vtype=GRB.BINARY, name=f"w_{gen_id}_{bid_id}_{t}")
                se[gen_id, bid_id, t] = model.addVar(lb=0, ub=bid['max_amount'], name=f"se_{gen_id}_{bid_id}_{t}")
    
    # Transmission variables
    for line in lines.values():
        for t in range(1, T+1):
            f[line['id'], t] = model.addVar(lb=-line['max_flow'], ub=line['max_flow'], name=f"f_{line['id']}_{t}")
    
    # Voltage angle variables (reference bus fixed to 0)
    ref_bus = elec_net['reference_bus_id']
    for bus_id in buses:
        for t in range(1, T+1):
            if bus_id == ref_bus:
                theta[bus_id, t] = model.addVar(lb=0, ub=0, name=f"theta_{bus_id}_{t}")
            else:
                theta[bus_id, t] = model.addVar(lb=-GRB.INFINITY, name=f"theta_{bus_id}_{t}")
    
    # Gas supply variables
    for junc_id, junc in junctions.items():
        if junc.get('is_source', False):
            supply_intervals = junc['supply_intervals']
            for interval in supply_intervals:
                interval_id = interval['id']
                for t in range(1, T+1):
                    sgs[junc_id, interval_id, t] = model.addVar(
                        lb=interval['interval_lb'], ub=interval['interval_ub'], 
                        name=f"sgs_{junc_id}_{interval_id}_{t}"
                    )
            for t in range(1, T+1):
                sg[junc_id, t] = model.addVar(name=f"sg_{junc_id}_{t}")
        else:
            for t in range(1, T+1):
                sg[junc_id, t] = model.addVar(lb=0, ub=0, name=f"sg_{junc_id}_{t}")  # Non-sources have 0 supply
    
    # Gas demand and consumption variables
    for junc_id in junctions:
        for t in range(1, T+1):
            lg[junc_id, t] = model.addVar(lb=0, name=f"lg_{junc_id}_{t}")
            qg[junc_id, t] = model.addVar(lb=0, name=f"qg_{junc_id}_{t}")
            gamma[junc_id, t] = model.addVar(lb=0, name=f"gamma_{junc_id}_{t}")
            pisq[junc_id, t] = model.addVar(
                lb=junctions[junc_id]['min_pressure_psi']**2,
                ub=junctions[junc_id]['max_pressure_psi']**2,
                name=f"pisq_{junc_id}_{t}"
            )
    
    # Gas flow variables
    for conn in connections.values():
        for t in range(1, T+1):
            phig[conn['id'], t] = model.addVar(lb=0, name=f"phig_{conn['id']}_{t}")
    
    # Gas price variables
    for zone in pricing_zones:
        zone_id = zone['id']
        for t in range(1, T+1):
            psi[zone_id, t] = model.addVar(lb=min_gas_price, ub=max_gas_price, name=f"psi_{zone_id}_{t}")
    
    # GFPP max gas price variables
    for gen_id, gen in generators.items():
        if gen.get('max_gas_price_fraction') is not None:
            for t in range(1, T+1):
                phimax[gen_id, t] = model.addVar(lb=0, name=f"phimax_{gen_id}_{t}")
    
    model.update()
    
    # Add constraints
    # Electricity constraints
    # Generator status evolution
    for gen_id, gen in generators.items():
        # Fix initial status for period 0 (not optimized)
        initial_status = gen['initial_status']
        # Status evolution for t=1..T
        for t in range(1, T+1):
            if t == 1:
                model.addConstr(o[gen_id, t] == initial_status + vp[gen_id, t] - vm[gen_id, t], 
                               f"status_evolution_{gen_id}_{t}")
            else:
                model.addConstr(o[gen_id, t] == o[gen_id, t-1] + vp[gen_id, t] - vm[gen_id, t], 
                               f"status_evolution_{gen_id}_{t}")
            model.addConstr(vp[gen_id, t] + vm[gen_id, t] <= 1, f"start_shutdown_{gen_id}_{t}")
    
    # Minimum up and down time constraints
    for gen_id, gen in generators.items():
        min_up = gen['min_up_time']
        min_down = gen['min_down_time']
        initial_active = gen['initial_active_periods']
        initial_inactive = gen['initial_inactive_periods']
        
        # Constraints for initial periods
        for t in range(1, min_up - initial_active + 1):
            if t <= T:
                model.addConstr(o[gen_id, t] == 1, f"min_up_initial_{gen_id}_{t}")
        
        for t in range(1, min_down - initial_inactive + 1):
            if t <= T:
                model.addConstr(o[gen_id, t] == 0, f"min_down_initial_{gen_id}_{t}")
        
        # General min up/down constraints
        for t in range(min_up - initial_active + 1, T+1):
            model.addConstr(sum(vp[gen_id, tau] for tau in range(t-min_up+1, t+1)) <= o[gen_id, t], 
                           f"min_up_{gen_id}_{t}")
        
        for t in range(min_down - initial_inactive + 1, T+1):
            model.addConstr(sum(vm[gen_id, tau] for tau in range(t-min_down+1, t+1)) <= 1 - o[gen_id, t], 
                           f"min_down_{gen_id}_{t}")
    
    # Ramp constraints
    for gen_id, gen in generators.items():
        ramp_up = gen['ramp_up_rate']
        ramp_down = gen['ramp_down_rate']
        initial_gen = gen['initial_gen']
        
        # Ramp from period 0 to 1
        model.addConstr(p[gen_id, 1] - initial_gen <= ramp_up, f"ramp_up_{gen_id}_1")
        model.addConstr(initial_gen - p[gen_id, 1] <= ramp_down, f"ramp_down_{gen_id}_1")
        
        # Ramp for subsequent periods
        for t in range(2, T+1):
            model.addConstr(p[gen_id, t] - p[gen_id, t-1] <= ramp_up, f"ramp_up_{gen_id}_{t}")
            model.addConstr(p[gen_id, t-1] - p[gen_id, t] <= ramp_down, f"ramp_down_{gen_id}_{t}")
    
    # Bid constraints
    for gen_id, gen in generators.items():
        bids = gen.get('bids', [])
        if not bids:
            continue
        
        # Sort bids by id
        bids = sorted(bids, key=lambda x: x['id'])
        bid_ids = [bid['id'] for bid in bids]
        
        for t in range(1, T+1):
            # Bid selection order: w[g,b,t] >= w[g,b+1,t]
            for i in range(len(bid_ids) - 1):
                model.addConstr(w[gen_id, bid_ids[i], t] >= w[gen_id, bid_ids[i+1], t], 
                               f"bid_order_{gen_id}_{bid_ids[i]}_{t}")
            
            # Bid can only be selected if generator is on
            for bid_id in bid_ids:
                model.addConstr(w[gen_id, bid_id, t] <= o[gen_id, t], 
                               f"bid_on_{gen_id}_{bid_id}_{t}")
            
            # Generation from bids
            model.addConstr(p[gen_id, t] == sum(se[gen_id, bid_id, t] for bid_id in bid_ids), 
                           f"gen_bid_sum_{gen_id}_{t}")
            
            # Bid amount constraints
            for bid in bids:
                bid_id = bid['id']
                model.addConstr(se[gen_id, bid_id, t] <= bid['max_amount'] * w[gen_id, bid_id, t], 
                               f"bid_ub_{gen_id}_{bid_id}_{t}")
                model.addConstr(se[gen_id, bid_id, t] >= 0, 
                               f"bid_lb_{gen_id}_{bid_id}_{t}")
            
            # Bid activation: next bid can only be selected if current bid is fully utilized
            for i in range(len(bid_ids) - 1):
                model.addConstr(se[gen_id, bid_ids[i], t] >= bids[i]['max_amount'] * w[gen_id, bid_ids[i+1], t], 
                               f"bid_activation_{gen_id}_{bid_ids[i]}_{t}")
    
    # Generator output bounds
    for gen_id, gen in generators.items():
        min_power = gen['min_power']
        for t in range(1, T+1):
            model.addConstr(p[gen_id, t] >= min_power * o[gen_id, t], f"gen_min_{gen_id}_{t}")
            model.addConstr(p[gen_id, t] <= gen['max_power'] * o[gen_id, t], f"gen_max_{gen_id}_{t}")
    
    # Startup cost constraints
    for gen_id, gen in generators.items():
        startup_costs = gen.get('startup_cost_params', [])
        if not startup_costs:
            continue
        
        # Sort startup costs by lookback window
        startup_costs = sorted(startup_costs, key=lambda x: x['lookback_window'])
        max_lookback = max(cost['lookback_window'] for cost in startup_costs)
        
        for t in range(1, T+1):
            # Constraint: r[g,t] >= cost * (vp[g,t] - sum_{tau in lookback windows} vp[g,t-tau])
            # We use indicator constraints or big-M
            for cost_param in startup_costs:
                lookback = cost_param['lookback_window']
                cost = cost_param['cost']
                # If t - lookback < 1, then we consider initial state
                if t - lookback < 1:
                    # Initial state: if generator was off for at least lookback periods before t
                    # We assume initial_inactive_periods >= lookback
                    if gen['initial_inactive_periods'] >= lookback:
                        model.addConstr(r[gen_id, t] >= cost * vp[gen_id, t], 
                                       f"startup_cost_{gen_id}_{t}_{lookback}")
                else:
                    # For t - lookback >= 1
                    # The generator should not have started up in the last 'lookback' periods
                    expr = vp[gen_id, t]
                    for tau in range(1, lookback+1):
                        if t - tau >= 1:
                            expr -= vp[gen_id, t-tau]
                    model.addConstr(r[gen_id, t] >= cost * expr, 
                                   f"startup_cost_{gen_id}_{t}_{lookback}")
    
    # Power balance at each bus
    for bus_id, bus in buses.items():
        for t in range(1, T+1):
            # Sum of generation at bus
            gen_sum = sum(p[gen_id, t] for gen_id, gen in generators.items() if gen['bus'] == bus_id)
            # Sum of flows into bus (lines where to_bus == bus_id) minus flows out (lines where from_bus == bus_id)
            flow_in = sum(f[line_id, t] for line_id, line in lines.items() if line['to_bus'] == bus_id)
            flow_out = sum(f[line_id, t] for line_id, line in lines.items() if line['from_bus'] == bus_id)
            net_flow = flow_in - flow_out
            demand = bus['demand_profile'][t-1]  # t-1 because demand_profile is 0-indexed for period 1
            model.addConstr(gen_sum - net_flow == demand, f"power_balance_{bus_id}_{t}")
    
    # Line flow constraints (angle difference)
    for line_id, line in lines.items():
        from_bus = line['from_bus']
        to_bus = line['to_bus']
        susceptance = line['susceptance_pu']
        for t in range(1, T+1):
            model.addConstr(f[line_id, t] == susceptance * (theta[from_bus, t] - theta[to_bus, t]), 
                           f"line_flow_{line_id}_{t}")
    
    # Gas network constraints
    # Gas supply at source junctions
    for junc_id, junc in junctions.items():
        if junc.get('is_source', False):
            supply_intervals = junc['supply_intervals']
            for t in range(1, T+1):
                model.addConstr(sg[junc_id, t] == sum(sgs[junc_id, interval['id'], t] for interval in supply_intervals), 
                               f"source_supply_{junc_id}_{t}")
    
    # Gas demand satisfaction
    for junc_id, junc in junctions.items():
        if 'gas_demand_profile' in junc:
            demand_profile = junc['gas_demand_profile']
            for t in range(1, T+1):
                demand = demand_profile[t-1]  # t-1 because demand_profile is 0-indexed for period 1
                model.addConstr(lg[junc_id, t] + qg[junc_id, t] == demand, 
                               f"gas_demand_{junc_id}_{t}")
                model.addConstr(lg[junc_id, t] <= demand, 
                               f"gas_demand_ub_{junc_id}_{t}")
        else:
            for t in range(1, T+1):
                model.addConstr(lg[junc_id, t] == 0, f"no_gas_demand_{junc_id}_{t}")
                model.addConstr(qg[junc_id, t] == 0, f"no_gas_shed_{junc_id}_{t}")
    
    # Gas consumption at GFPP junctions
    for gen_id, gen in generators.items():
        if gen.get('max_gas_price_fraction') is not None:
            # GFPP: gas consumption = heat rate function
            junction_id = gen['gas_junction']
            H_u2 = gen['heat_rate_coeffs']['H_u2']
            H_u1 = gen['heat_rate_coeffs']['H_u1']
            H_u0 = gen['heat_rate_coeffs']['H_u0']
            for t in range(1, T+1):
                # Quadratic heat rate: gamma = H_u2 * p^2 + H_u1 * p + H_u0 * o
                # We add as quadratic constraint
                expr = H_u2 * p[gen_id, t] * p[gen_id, t] + H_u1 * p[gen_id, t] + H_u0 * o[gen_id, t]
                model.addConstr(gamma[junction_id, t] == expr, f"heat_rate_{gen_id}_{t}")
    
    # Gas flow conservation at junctions
    for junc_id, junc in junctions.items():
        for t in range(1, T+1):
            # Sum of incoming flows (connections where to_junction == junc_id)
            incoming = sum(phig[conn_id, t] for conn_id, conn in connections.items() if conn['to_junction'] == junc_id)
            # Sum of outgoing flows (connections where from_junction == junc_id)
            outgoing = sum(phig[conn_id, t] for conn_id, conn in connections.items() if conn['from_junction'] == junc_id)
            # Conservation: supply - demand - consumption = incoming - outgoing
            model.addConstr(sg[junc_id, t] - lg[junc_id, t] - gamma[junc_id, t] == incoming - outgoing, 
                           f"gas_conservation_{junc_id}_{t}")
    
    # Weymouth equation for pipelines
    for conn_id, conn in connections.items():
        if conn['type'] == 'pipeline':
            from_junc = conn['from_junction']
            to_junc = conn['to_junction']
            C = conn['weymouth_coeff']
            for t in range(1, T+1):
                # phig^2 = C^2 * (pisq_from - pisq_to)
                model.addConstr(phig[conn_id, t] * phig[conn_id, t] == 
                               C * C * (pisq[from_junc, t] - pisq[to_junc, t]), 
                               f"weymouth_{conn_id}_{t}")
    
    # Compressor constraints
    for conn_id, conn in connections.items():
        if conn['type'] == 'compressor':
            from_junc = conn['from_junction']
            to_junc = conn['to_junction']
            max_ratio = conn['max_compression_ratio']
            for t in range(1, T+1):
                # Pressure ratio constraint: pisq_to <= (max_ratio^2) * pisq_from
                model.addConstr(pisq[to_junc, t] <= max_ratio * max_ratio * pisq[from_junc, t], 
                               f"compressor_ratio_{conn_id}_{t}")
                # Flow direction: phig >= 0 (already enforced by lb)
    
    # GFPP bid validity constraints
    for gen_id, gen in generators.items():
        if gen.get('max_gas_price_fraction') is not None:
            junction_id = gen['gas_junction']
            zone_id = junction_to_zone[junction_id]
            threshold = gfpp_thresholds[gen_id]
            for t in range(1, T+1):
                # phimax[g,t] = threshold * o[g,t]
                model.addConstr(phimax[gen_id, t] == threshold * o[gen_id, t], 
                               f"phimax_{gen_id}_{t}")
                # Constraint: psi[z,t] <= threshold * o[g,t] + max_gas_price * (1 - o[g,t])
                model.addConstr(psi[zone_id, t] <= threshold * o[gen_id, t] + max_gas_price * (1 - o[gen_id, t]), 
                               f"bid_validity_{gen_id}_{t}")
    
    # Set objective
    # Electricity cost: sum over generators and bids of bid_price * se[g,b,t]
    electricity_cost = 0
    for gen_id, gen in generators.items():
        bids = gen.get('bids', [])
        for bid in bids:
            bid_id = bid['id']
            price = bid['bid_price']
            for t in range(1, T+1):
                electricity_cost += price * se[gen_id, bid_id, t]
    
    # Startup cost
    startup_cost = sum(r[gen_id, t] for gen_id in generators for t in range(1, T+1))
    
    # Gas cost: sum over pricing zones and periods of psi[z,t] * (sg[j,t] for j in zone)
    gas_cost = 0
    for zone in pricing_zones:
        zone_id = zone['id']
        for t in range(1, T+1):
            # Sum of sg[j,t] for junctions j in this zone
            sg_sum = sum(sg[junc_id, t] for junc_id in zone['junctions'])
            gas_cost += psi[zone_id, t] * sg_sum
    
    # Shed gas cost: penalty for shed gas
    shed_gas_cost = 0
    for junc_id in junctions:
        shed_penalty = junctions[junc_id].get('gas_shed_penalty', 1e6)  # Default high penalty
        for t in range(1, T+1):
            shed_gas_cost += shed_penalty * qg[junc_id, t]
    
    # Total objective: weighted sum
    alpha = instance['objective_weights']['electricity_cost']
    beta = instance['objective_weights']['gas_cost']
    total_obj = alpha * electricity_cost + beta * gas_cost + startup_cost + shed_gas_cost
    model.setObjective(total_obj, GRB.MINIMIZE)
    
    # Optimize
    model.optimize()
    
    # Prepare solution dictionary
    solution = {
        "instance_path": instance_path,
        "solution_path": solution_path,
        "time_limit": time_limit,
        "solver_status": model.Status,
        "objective_value": model.ObjVal,
        "primary_variables": {}
    }
    
    # Extract solution for period 0 (fixed initial state)
    # Electricity
    for gen_id, gen in generators.items():
        # Status
        solution['primary_variables'][f"o_{gen_id}_0"] = gen['initial_status']
        # Generation
        solution['primary_variables'][f"p_{gen_id}_0"] = gen['initial_gen']
        # Bids (precomputed based on initial_gen)
        bids = sorted(gen.get('bids', []), key=lambda x: x['id'])
        max_amounts = [bid['max_amount'] for bid in bids]
        initial_gen = gen['initial_gen']
        total_capacity = sum(max_amounts)
        if initial_gen > total_capacity:
            initial_gen = total_capacity
        w0 = [0] * len(bids)
        se0 = [0] * len(bids)
        rem = initial_gen
        for i in range(len(bids)):
            if rem <= 0:
                break
            if i < len(bids) - 1:
                if rem >= max_amounts[i]:
                    w0[i] = 1
                    se0[i] = max_amounts[i]
                    rem -= max_amounts[i]
                else:
                    w0[i] = 1
                    se0[i] = rem
                    rem = 0
                    break
            else:
                if rem > 0:
                    w0[i] = 1
                    se0[i] = rem
                    rem = 0
        for i, bid in enumerate(bids):
            bid_id = bid['id']
            solution['primary_variables'][f"w_{gen_id}_{bid_id}_0"] = w0[i]
            solution['primary_variables'][f"se_{gen_id}_{bid_id}_0"] = se0[i]
    
    # Gas network period 0 (dummy values)
    for junc_id in junctions:
        solution['primary_variables'][f"lg_{junc_id}_0"] = 0
        solution['primary_variables'][f"qg_{junc_id}_0"] = 0
        solution['primary_variables'][f"sg_{junc_id}_0"] = 0
        solution['primary_variables'][f"pisq_{junc_id}_0"] = (
            (junctions[junc_id]['min_pressure_psi'] + junctions[junc_id]['max_pressure_psi']) / 2
        ) ** 2
        # Gamma: sum of GFPP consumption at junction
        gamma0 = 0
        for gen in generators.values():
            if gen.get('gas_junction') == junc_id and gen.get('max_gas_price_fraction') is not None:
                H_u2 = gen['heat_rate_coeffs']['H_u2']
                H_u1 = gen['heat_rate_coeffs']['H_u1']
                H_u0 = gen['heat_rate_coeffs']['H_u0']
                p0 = gen['initial_gen']
                o0 = gen['initial_status']
                gamma0 += H_u2 * p0 * p0 + H_u1 * p0 + H_u0 * o0
        solution['primary_variables'][f"gamma_{junc_id}_0"] = gamma0
    
    for conn in connections.values():
        solution['primary_variables'][f"phig_{conn['id']}_0"] = 0
    
    for zone in pricing_zones:
        solution['primary_variables'][f"psi_{zone['id']}_0"] = min_gas_price
    
    for gen_id, gen in generators.items():
        if gen.get('max_gas_price_fraction') is not None:
            solution['primary_variables'][f"phimax_{gen_id}_0"] = gfpp_thresholds[gen_id] * gen['initial_status']
    
    # Extract solution for periods 1..T
    for gen_id in generators:
        for t in range(1, T+1):
            solution['primary_variables'][f"o_{gen_id}_{t}"] = o[gen_id, t].X
            solution['primary_variables'][f"p_{gen_id}_{t}"] = p[gen_id, t].X
            solution['primary_variables'][f"vp_{gen_id}_{t}"] = vp[gen_id, t].X
            solution['primary_variables'][f"vm_{gen_id}_{t}"] = vm[gen_id, t].X
            solution['primary_variables'][f"r_{gen_id}_{t}"] = r[gen_id, t].X
            bids = generators[gen_id].get('bids', [])
            for bid in bids:
                bid_id = bid['id']
                solution['primary_variables'][f"w_{gen_id}_{bid_id}_{t}"] = w[gen_id, bid_id, t].X
                solution['primary_variables'][f"se_{gen_id}_{bid_id}_{t}"] = se[gen_id, bid_id, t].X
            if gen_id in gfpp_thresholds:
                solution['primary_variables'][f"phimax_{gen_id}_{t}"] = phimax[gen_id, t].X
    
    for line_id in lines:
        for t in range(1, T+1):
            solution['primary_variables'][f"f_{line_id}_{t}"] = f[line_id, t].X
    
    for bus_id in buses:
        for t in range(1, T+1):
            solution['primary_variables'][f"theta_{bus_id}_{t}"] = theta[bus_id, t].X
    
    for junc_id in junctions:
        for t in range(1, T+1):
            solution['primary_variables'][f"sg_{junc_id}_{t}"] = sg[junc_id, t].X
            solution['primary_variables'][f"lg_{junc_id}_{t}"] = lg[junc_id, t].X
            solution['primary_variables'][f"qg_{junc_id}_{t}"] = qg[junc_id, t].X
            solution['primary_variables'][f"gamma_{junc_id}_{t}"] = gamma[junc_id, t].X
            solution['primary_variables'][f"pisq_{junc_id}_{t}"] = pisq[junc_id, t].X
            if junctions[junc_id].get('is_source', False):
                for interval in junctions[junc_id]['supply_intervals']:
                    interval_id = interval['id']
                    solution['primary_variables'][f"sgs_{junc_id}_{interval_id}_{t}"] = sgs[junc_id, interval_id, t].X
    
    for conn in connections.values():
        for t in range(1, T+1):
            solution['primary_variables'][f"phig_{conn['id']}_{t}"] = phig[conn['id'], t].X
    
    for zone in pricing_zones:
        for t in range(1, T+1):
            solution['primary_variables'][f"psi_{zone['id']}_{t}"] = psi[zone['id'], t].X
    
    # Write solution to JSON
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()