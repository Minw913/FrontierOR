import argparse
import json
import math
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_instance(instance_data, time_limit, log_path):
    # Initialize the logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Extract instance data
    T = instance_data['time_periods']
    beta = instance_data['objective_weight_beta']
    
    # Electric network data
    buses = {b['id']: b for b in instance_data['electricity_network']['buses']}
    lines = {l['id']: l for l in instance_data['electricity_network']['lines']}
    
    # Generator data
    generators = {}
    gen_by_bus = {}
    gfpps = []
    for g in instance_data['generators']['generators']:
        generators[g['id']] = g
        if g['bus'] not in gen_by_bus:
            gen_by_bus[g['bus']] = []
        gen_by_bus[g['bus']].append(g['id'])
        
        if g['is_gfpp']:
            gfpps.append(g['id'])
    
    # Gas network data
    junctions = {j['id']: j for j in instance_data['gas_network']['junctions']}
    connections = {c['id']: c for c in instance_data['gas_network']['connections']}
    pricing_zones = {z['id']: z for z in instance_data['gas_network']['pricing_zones']}
    
    # Create model
    model = gp.Model("UCGNA")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    
    # Decision variables
    # Generator on/off status (including pre-horizon)
    o = {}
    for g_id in generators:
        for t in range(T + 1):  # t=0 is pre-horizon
            o[g_id, t] = model.addVar(vtype=GRB.BINARY, name=f"o_{g_id}_{t}")
    
    # Startup/shutdown variables
    vp = {}  # startup
    vm = {}  # shutdown
    for g_id in generators:
        for t in range(1, T + 1):
            vp[g_id, t] = model.addVar(vtype=GRB.BINARY, name=f"vp_{g_id}_{t}")
            vm[g_id, t] = model.addVar(vtype=GRB.BINARY, name=f"vm_{g_id}_{t}")
    
    # Bid selection variables
    w = {}
    for g_id in generators:
        g = generators[g_id]
        for b_idx in range(g['num_bids']):
            for t in range(1, T + 1):
                w[g_id, b_idx, t] = model.addVar(vtype=GRB.BINARY, name=f"w_{g_id}_{b_idx}_{t}")
    
    # Startup costs
    r = {}
    for g_id in generators:
        for t in range(1, T + 1):
            r[g_id, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"r_{g_id}_{t}")
    
    # Power output (including pre-horizon)
    p = {}
    for g_id in generators:
        for t in range(T + 1):  # t=0 is pre-horizon
            p[g_id, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"p_{g_id}_{t}")
    
    # Bid-level generation
    se = {}
    for g_id in generators:
        g = generators[g_id]
        for b_idx in range(g['num_bids']):
            for t in range(1, T + 1):
                se[g_id, b_idx, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"se_{g_id}_{b_idx}_{t}")
    
    # Power flows
    f = {}
    for l_id in lines:
        for t in range(1, T + 1):
            f[l_id, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=-lines[l_id]['thermal_limit'], 
                                      ub=lines[l_id]['thermal_limit'], name=f"f_{l_id}_{t}")
    
    # Voltage angles
    theta = {}
    for b_id in buses:
        for t in range(1, T + 1):
            theta[b_id, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=buses[b_id]['voltage_angle_lb'],
                                          ub=buses[b_id]['voltage_angle_ub'], name=f"theta_{b_id}_{t}")
    
    # Gas variables
    sg = {}  # total gas supply
    sgs = {}  # supply from intervals
    pisq = {}  # squared pressure
    phig = {}  # gas flow
    lg = {}  # satisfied demand
    qg = {}  # shed demand
    gamma = {}  # gas consumption by GFPPs
    psi = {}  # zonal gas price
    
    for j_id in junctions:
        for t in range(1, T + 1):
            sg[j_id, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"sg_{j_id}_{t}")
            lg[j_id, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"lg_{j_id}_{t}")
            qg[j_id, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"qg_{j_id}_{t}")
            gamma[j_id, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"gamma_{j_id}_{t}")
            pisq[j_id, t] = model.addVar(vtype=GRB.CONTINUOUS, 
                                         lb=junctions[j_id]['pressure_lb_squared'],
                                         ub=junctions[j_id]['pressure_ub_squared'], name=f"pisq_{j_id}_{t}")
            
            # Supply from intervals
            for i_idx, interval in enumerate(junctions[j_id]['supply_intervals']):
                sgs[j_id, i_idx, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"sgs_{j_id}_{i_idx}_{t}")
    
    for conn_id in connections:
        for t in range(1, T + 1):
            phig[conn_id, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"phig_{conn_id}_{t}")
    
    # Zonal gas prices
    for z_id in pricing_zones:
        for t in range(1, T + 1):
            psi[z_id, t] = model.addVar(vtype=GRB.CONTINUOUS, 
                                        lb=instance_data['gas_network']['min_gas_price_mmBtu'],
                                        ub=instance_data['gas_network']['max_gas_price_mmBtu'], name=f"psi_{z_id}_{t}")
    
    # Max allowable gas price for GFPPs
    phimax = {}
    for g_id in gfpps:
        for t in range(1, T + 1):
            phimax[g_id, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, 
                                           ub=instance_data['gas_network']['max_gas_price_mmBtu'], 
                                           name=f"phimax_{g_id}_{t}")
    
    # Constraints
    
    # Initial conditions
    for g_id in generators:
        g = generators[g_id]
        model.addConstr(o[g_id, 0] == g['initial_status'], name=f"init_status_{g_id}")
        model.addConstr(p[g_id, 0] == g['initial_gen'], name=f"init_output_{g_id}")
    
    # Startup/shutdown logic
    for g_id in generators:
        for t in range(1, T + 1):
            model.addConstr(vp[g_id, t] - vm[g_id, t] == o[g_id, t] - o[g_id, t-1], 
                            name=f"startup_shutdown_{g_id}_{t}")
    
    # Minimum up/down time
    for g_id in generators:
        g = generators[g_id]
        # Minimum up time
        if g['min_up_time'] > 0:
            for t in range(g['min_up_time'], T + 1):
                model.addConstr(gp.quicksum(vp[g_id, k] for k in range(t - g['min_up_time'] + 1, t + 1)) <= o[g_id, t],
                                name=f"min_up_time_{g_id}_{t}")
        
        # Minimum down time
        if g['min_down_time'] > 0:
            for t in range(g['min_down_time'], T + 1):
                model.addConstr(gp.quicksum(vp[g_id, k] for k in range(t - g['min_down_time'] + 1, t + 1)) <= 1 - o[g_id, t-1],
                                name=f"min_down_time_{g_id}_{t}")
    
    # Startup cost constraints
    for g_id in generators:
        g = generators[g_id]
        for t in range(1, T + 1):
            # Apply each startup cost tier constraint
            for tier_idx, (lookback, cost) in enumerate(g['startup_cost_params']):
                if t >= lookback:
                    model.addConstr(r[g_id, t] >= cost * vp[g_id, t] * o[g_id, t-lookback],
                                    name=f"startup_cost_tier_{g_id}_{t}_{tier_idx}")
                else:
                    # For early periods, we use the initial state
                    if t > g['initial_inactive_periods']:
                        model.addConstr(r[g_id, t] >= cost * vp[g_id, t],
                                        name=f"startup_cost_tier_{g_id}_{t}_{tier_idx}")
                    else:
                        model.addConstr(r[g_id, t] >= 0, name=f"startup_cost_min_{g_id}_{t}")
    
    # Power balance at buses
    for b_id in buses:
        for t in range(1, T + 1):
            gen_output = gp.quicksum(p[g_id, t] for g_id in gen_by_bus.get(b_id, []))
            net_flow = gp.LinExpr()
            for l_id, line in lines.items():
                if line['from_bus'] == b_id:
                    net_flow += f[l_id, t]
                elif line['to_bus'] == b_id:
                    net_flow -= f[l_id, t]
            
            model.addConstr(gen_output - buses[b_id]['demand_profile'][t-1] == net_flow,
                            name=f"power_balance_{b_id}_{t}")
    
    # Line flow equations
    for l_id, line in lines.items():
        for t in range(1, T + 1):
            model.addConstr(f[l_id, t] == line['susceptance'] * (theta[line['from_bus'], t] - theta[line['to_bus'], t]),
                            name=f"line_flow_{l_id}_{t}")
    
    # Line flow and angle limits
    for l_id, line in lines.items():
        for t in range(1, T + 1):
            model.addConstr(f[l_id, t] <= line['thermal_limit'], name=f"flow_ub_{l_id}_{t}")
            model.addConstr(f[l_id, t] >= -line['thermal_limit'], name=f"flow_lb_{l_id}_{t}")
            model.addConstr(theta[line['from_bus'], t] - theta[line['to_bus'], t] <= line['angle_diff_limit'],
                            name=f"angle_diff_ub_{l_id}_{t}")
            model.addConstr(theta[line['from_bus'], t] - theta[line['to_bus'], t] >= -line['angle_diff_limit'],
                            name=f"angle_diff_lb_{l_id}_{t}")
    
    # Generator output constraints
    for g_id in generators:
        g = generators[g_id]
        for t in range(1, T + 1):
            # Power output equals sum of bid outputs
            model.addConstr(p[g_id, t] == gp.quicksum(se[g_id, b_idx, t] for b_idx in range(g['num_bids'])),
                            name=f"gen_output_sum_{g_id}_{t}")
            
            # Min/max output bounds
            model.addConstr(p[g_id, t] >= g['min_power'] * o[g_id, t], name=f"min_output_{g_id}_{t}")
            model.addConstr(p[g_id, t] <= g['max_power'] * o[g_id, t], name=f"max_output_{g_id}_{t}")
            
            # Ramp constraints
            if t > 1:
                model.addConstr(p[g_id, t] - p[g_id, t-1] <= g['ramp_up'] * o[g_id, t-1] + g['max_power'] * vp[g_id, t],
                                name=f"ramp_up_{g_id}_{t}")
                model.addConstr(p[g_id, t-1] - p[g_id, t] <= g['ramp_down'] * o[g_id, t] + g['min_power'] * vm[g_id, t],
                                name=f"ramp_down_{g_id}_{t}")
            else:
                # First period ramp constraints relative to initial state
                model.addConstr(p[g_id, 1] - g['initial_gen'] <= g['ramp_up'] * g['initial_status'] + g['max_power'] * vp[g_id, 1],
                                name=f"ramp_up_init_{g_id}_1")
                model.addConstr(g['initial_gen'] - p[g_id, 1] <= g['ramp_down'] * o[g_id, 1] + g['min_power'] * vm[g_id, 1],
                                name=f"ramp_down_init_{g_id}_1")
    
    # Bid selection constraints
    for g_id in generators:
        g = generators[g_id]
        for t in range(1, T + 1):
            for b_idx in range(g['num_bids']):
                bid = g['bids'][b_idx]
                
                # Bound bid output by max amount if selected
                model.addConstr(se[g_id, b_idx, t] <= bid['max_amount'] * w[g_id, b_idx, t],
                                name=f"bid_output_bound_{g_id}_{b_idx}_{t}")
                
                # Sequential bidding: next bid can only be selected if current is fully used
                if b_idx < g['num_bids'] - 1:
                    model.addConstr(w[g_id, b_idx+1, t] <= w[g_id, b_idx, t],
                                    name=f"sequential_bid_{g_id}_{b_idx}_{t}")
    
    # Gas network constraints
    for j_id in junctions:
        for t in range(1, T + 1):
            # Supply-demand balance
            total_supply = sg[j_id, t]
            total_consumption = gamma[j_id, t] + lg[j_id, t]
            
            # Net gas flow calculation
            net_flow = gp.LinExpr()
            for conn_id, conn in connections.items():
                if conn['from_junction'] == j_id:
                    net_flow += phig[conn_id, t]
                elif conn['to_junction'] == j_id:
                    net_flow -= phig[conn_id, t]
            
            # Gas balance equation
            model.addConstr(total_supply - total_consumption == net_flow,
                            name=f"gas_balance_{j_id}_{t}")
            
            # Demand satisfaction
            model.addConstr(lg[j_id, t] + qg[j_id, t] == junctions[j_id]['gas_demand_profile'][t-1],
                            name=f"demand_satisfaction_{j_id}_{t}")
            
            if junctions[j_id]['is_source']:
                # Supply from intervals
                model.addConstr(sg[j_id, t] == gp.quicksum(sgs[j_id, i_idx, t] 
                                                           for i_idx in range(len(junctions[j_id]['supply_intervals']))),
                                name=f"total_supply_{j_id}_{t}")
                
                # Interval limits
                for i_idx, interval in enumerate(junctions[j_id]['supply_intervals']):
                    model.addConstr(sgs[j_id, i_idx, t] >= interval['interval_lb'],
                                    name=f"supply_interval_lb_{j_id}_{i_idx}_{t}")
                    model.addConstr(sgs[j_id, i_idx, t] <= interval['interval_ub'],
                                    name=f"supply_interval_ub_{j_id}_{i_idx}_{t}")
            else:
                # Non-source junctions have zero supply
                model.addConstr(sg[j_id, t] == 0, name=f"zero_supply_{j_id}_{t}")
    
    # Gas flow equations
    for conn_id, conn in connections.items():
        for t in range(1, T + 1):
            if conn['type'] == 'pipeline':
                # Weymouth equation (relaxed as inequality)
                model.addConstr(pisq[conn['from_junction'], t] - pisq[conn['to_junction'], t] >= 
                                conn['weymouth_factor'] * phig[conn_id, t] * phig[conn_id, t],
                                name=f"weymouth_{conn_id}_{t}")
            elif conn['type'] == 'compressor':
                # Compression ratio constraints
                model.addConstr(pisq[conn['to_junction'], t] >= 
                                conn['compression_ratio_lb'] * conn['compression_ratio_lb'] * pisq[conn['from_junction'], t],
                                name=f"comp_ratio_lb_{conn_id}_{t}")
                model.addConstr(pisq[conn['to_junction'], t] <= 
                                conn['compression_ratio_ub'] * conn['compression_ratio_ub'] * pisq[conn['from_junction'], t],
                                name=f"comp_ratio_ub_{conn_id}_{t}")
            # Handle control valves if present
            elif conn.get('control_ratio_lb') is not None and conn['control_ratio_lb'] is not None:
                # Control valve ratio constraints
                model.addConstr(pisq[conn['to_junction'], t] >= 
                                conn['control_ratio_lb'] * conn['control_ratio_lb'] * pisq[conn['from_junction'], t],
                                name=f"valve_ratio_lb_{conn_id}_{t}")
                model.addConstr(pisq[conn['to_junction'], t] <= 
                                conn['control_ratio_ub'] * conn['control_ratio_ub'] * pisq[conn['from_junction'], t],
                                name=f"valve_ratio_ub_{conn_id}_{t}")
    
    # Heat rate constraints for GFPPs
    for g_id in gfpps:
        g = generators[g_id]
        j_id = g['gas_junction']
        for t in range(1, T + 1):
            # Quadratic heat rate: H_u2*p^2 + H_u1*p + H_u0*o
            # To linearize p^2, we use McCormick envelopes or piecewise linear approximations
            # Here we'll use a relaxed approach for the quadratic term
            # We'll create an auxiliary variable for p^2 and add McCormick envelopes
            p_sq = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"p_sq_{g_id}_{t}")
            
            # McCormick envelopes for p^2
            p_max = g['max_power']
            p_min = g['min_power'] if g['initial_status'] == 1 else 0
            
            # Envelope constraints for p^2
            model.addConstr(p_sq >= p_min * p[g_id, t] + p[g_id, t] * p_min - p_min * p_min, 
                           name=f"mc1_{g_id}_{t}")
            model.addConstr(p_sq >= p_max * p[g_id, t] + p[g_id, t] * p_max - p_max * p_max, 
                           name=f"mc2_{g_id}_{t}")
            model.addConstr(p_sq <= p_min * p[g_id, t] + p[g_id, t] * p_max - p_min * p_max, 
                           name=f"mc3_{g_id}_{t}")
            
            # Heat rate constraint using auxiliary variable
            heat_rate_expr = gp.LinExpr()
            if g['heat_rate_coefficients']['H_u2'] is not None:
                heat_rate_expr.addTerms([g['heat_rate_coefficients']['H_u2']], [p_sq])
            if g['heat_rate_coefficients']['H_u1'] is not None:
                heat_rate_expr.addTerms([g['heat_rate_coefficients']['H_u1']], [p[g_id, t]])
            if g['heat_rate_coefficients']['H_u0'] is not None:
                heat_rate_expr.addTerms([g['heat_rate_coefficients']['H_u0']], [o[g_id, t]])
            
            model.addConstr(gamma[j_id, t] >= heat_rate_expr,
                            name=f"heat_rate_{g_id}_{t}")
    
    # Pricing zone assignment
    # Map junctions to zones
    junction_to_zone = {}
    for z_id, zone in pricing_zones.items():
        for j_id in zone['junctions']:
            junction_to_zone[j_id] = z_id
    
    # GFPP profitability constraints
    for g_id in gfpps:
        g = generators[g_id]
        j_id = g['gas_junction']
        z_id = junction_to_zone[j_id]
        
        for t in range(1, T + 1):
            # Calculate max allowable gas price based on bid selection
            # For each bid except the last, the threshold is multiplied by (w_current - w_next)
            # For the last bid, it's multiplied by w_last
            profit_expr = gp.LinExpr()
            for b_idx, bid in enumerate(g['bids']):
                # Get the max gas price fraction from the generator data
                max_gas_price_threshold = g['max_gas_price_fraction'] if g['max_gas_price_fraction'] else instance_data['gas_network']['max_gas_price_mmBtu']
                
                if b_idx < len(g['bids']) - 1:
                    # Current bid - next bid
                    profit_expr.addTerms([max_gas_price_threshold], 
                                         [w[g_id, b_idx, t] - w[g_id, b_idx+1, t]])
                else:
                    # Last bid
                    profit_expr.addTerms([max_gas_price_threshold], 
                                         [w[g_id, b_idx, t]])
            
            model.addConstr(phimax[g_id, t] == profit_expr, name=f"phimax_calc_{g_id}_{t}")
            
            # The max allowable gas price must be >= zonal gas price when generator is on
            model.addConstr(phimax[g_id, t] >= psi[z_id, t] * o[g_id, t], 
                            name=f"profitability_{g_id}_{t}")
    
    # Objective function
    elec_cost = gp.LinExpr()
    gas_cost = gp.LinExpr()
    
    # Electricity costs
    for g_id in generators:
        g = generators[g_id]
        for t in range(1, T + 1):
            # No load cost
            elec_cost.addTerms([g['no_load_cost']], [o[g_id, t]])
            # Startup cost
            elec_cost.addTerms([1], [r[g_id, t]])
            # Bid costs
            for b_idx, bid in enumerate(g['bids']):
                elec_cost.addTerms([bid['price']], [se[g_id, b_idx, t]])
    
    # Gas costs
    for j_id in junctions:
        for t in range(1, T + 1):
            # Demand shedding cost
            gas_cost.addTerms([junctions[j_id]['demand_shedding_cost']], [qg[j_id, t]])
            # Supply interval costs
            for i_idx, interval in enumerate(junctions[j_id]['supply_intervals']):
                gas_cost.addTerms([interval['slope']], [sgs[j_id, i_idx, t]])
    
    # Combine costs
    total_cost = beta * elec_cost + (1 - beta) * gas_cost
    
    model.setObjective(total_cost, GRB.MINIMIZE)
    
    # Optimize the model
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                # Get current solution values
                sol_dict = {}
                # Add primary variables to solution dict
                for var in model.getVars():
                    if var.x is not None:
                        sol_dict[var.varName] = var.x
                
                solution_dict = {"primary_variables": sol_dict}
                logger.log_solution(obj_val, solution_dict)
    
    model.optimize(callback)
    
    # Prepare solution dictionary
    solution_dict = {}
    
    # Check if we have a feasible solution
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
        # Extract solution values
        sol_vars = {}
        for var in model.getVars():
            sol_vars[var.varName] = var.x
        
        # Build primary variables dict
        primary_vars = {}
        for var_name, value in sol_vars.items():
            primary_vars[var_name] = value
        
        solution_dict = {
            "objective_value": model.ObjVal,
            "released_problem_variant": "integrated_primal_ucgna",
            "objective_weight_beta": beta,
            "primary_variables": primary_vars
        }
    else:
        solution_dict = {
            "objective_value": float('inf'),
            "released_problem_variant": "integrated_primal_ucgna",
            "objective_weight_beta": beta,
            "primary_variables": {}
        }
    
    return solution_dict


def main():
    parser = argparse.ArgumentParser(description='Solve UCGNA problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to save solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log file')
    
    args = parser.parse_args()
    
    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Solve the instance
    solution = solve_instance(instance_data, args.time_limit, args.log_path)
    
    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()