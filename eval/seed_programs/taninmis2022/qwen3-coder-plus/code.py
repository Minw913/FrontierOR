import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import math
from datetime import datetime
from solution_logger import SolutionLogger


def solve_power_system_expansion(instance_path, solution_path, time_limit, log_path):
    # Read instance data
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    # Initialize logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Extract system parameters
    system = data['system']
    scenario_params = data['scenario_parameters']
    scenario_tree = {node['id']: node for node in data['scenario_tree']}
    representative_days = {day['id']: day for day in data['representative_days']}
    lower_level_scenarios = data['lower_level_scenarios']
    
    # Create mapping dictionaries
    gen_by_id = {gen['id']: gen for gen in system['generators']}
    line_by_id = {line['id']: line for line in system['transmission_lines']}
    storage_by_id = {stor['id']: stor for stor in system['storage']}
    load_by_bus = {load['bus']: load for load in system['loads']}
    
    # Extract relevant constants
    num_stages = scenario_params['num_stages']
    num_representative_days = scenario_params['num_representative_days']
    hours_per_day = scenario_params['hours_per_day']
    reserve_margin_req = data['global_parameters']['reserve_margin_requirement']
    spinning_reserve_pct = data['global_parameters']['spinning_reserve_percent']
    
    # Identify buses, generators, lines, and storage units
    buses = system['buses']
    generators = system['generators']
    transmission_lines = system['transmission_lines']
    storage_units = system['storage']
    
    # Candidate items
    candidate_gens = [g for g in generators if not g['existing']]
    candidate_lines = [l for l in transmission_lines if not l['existing']]
    candidate_storage = [s for s in storage_units if not s['existing']]
    
    # Existing items
    existing_gens = [g for g in generators if g['existing']]
    existing_lines = [l for l in transmission_lines if l['existing']]
    existing_storage = [s for s in storage_units if s['existing']]
    
    # Create indices for easier access
    gen_ids = [g['id'] for g in generators]
    line_ids = [l['id'] for l in transmission_lines]
    storage_ids = [s['id'] for s in storage_units]
    bus_to_idx = {bus: i for i, bus in enumerate(buses)}
    
    # Group generators by bus
    gens_by_bus = {bus: [] for bus in buses}
    for g in generators:
        gens_by_bus[g['bus']].append(g['id'])
    
    # Group loads by bus
    loads_by_bus = {load['bus']: load for load in system['loads']}
    
    # Create sets of IDs for candidates
    candidate_gen_ids = [g['id'] for g in candidate_gens]
    candidate_line_ids = [l['id'] for l in candidate_lines]
    candidate_storage_ids = [s['id'] for s in candidate_storage]
    
    # Build line connections
    lines_from_bus = {bus: [] for bus in buses}
    lines_to_bus = {bus: [] for bus in buses}
    for line in transmission_lines:
        lines_from_bus[line['from_bus']].append(line['id'])
        lines_to_bus[line['to_bus']].append(line['id'])
    
    # Create model
    model = gp.Model("PowerSystemExpansion")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    
    # Decision variables
    # Investment variables
    z_gen = {}  # Binary: whether to build candidate generator
    z_line = {}  # Binary: whether to build candidate line
    z_storage = {}  # Binary: whether to build candidate storage
    
    for node_id in scenario_tree:
        for gen_id in candidate_gen_ids:
            z_gen[(node_id, gen_id)] = model.addVar(vtype=GRB.BINARY, name=f"z_gen_{node_id}_{gen_id}")
        
        for line_id in candidate_line_ids:
            z_line[(node_id, line_id)] = model.addVar(vtype=GRB.BINARY, name=f"z_line_{node_id}_{line_id}")
            
        for storage_id in candidate_storage_ids:
            z_storage[(node_id, storage_id)] = model.addVar(vtype=GRB.BINARY, name=f"z_storage_{node_id}_{storage_id}")
    
    # Available capacities
    available_gen_cap = {}
    available_line_cap = {}
    available_storage_cap = {}
    
    for node_id in scenario_tree:
        for gen_id in candidate_gen_ids:
            available_gen_cap[(node_id, gen_id)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, 
                ub=gen_by_id[gen_id]['Pmax_MW'], name=f"available_gen_cap_{node_id}_{gen_id}")
        
        for line_id in candidate_line_ids:
            available_line_cap[(node_id, line_id)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0,
                ub=line_by_id[line_id]['flow_limit_MW'], name=f"available_line_cap_{node_id}_{line_id}")
        
        for storage_id in candidate_storage_ids:
            available_storage_cap[(node_id, storage_id)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0,
                ub=storage_by_id[storage_id]['capacity_MW'], name=f"available_storage_cap_{node_id}_{storage_id}")
    
    # Operational variables
    alpha = {}  # Generator commitment
    gamma = {}  # Generator startup
    p_gen = {}  # Generator dispatch
    s_res = {}  # Spinning reserve
    f_fwd = {}  # Forward flow
    f_bwd = {}  # Backward flow
    u_stor = {}  # Storage charging
    v_stor = {}  # Storage discharging
    r_stor = {}  # Storage state of charge
    
    # Create operational variables for each scenario node, day, scenario, hour, and component
    for node_id in scenario_tree:
        for day_id in representative_days:
            for scen_id, scen_data in enumerate(lower_level_scenarios[str(day_id)]):
                for h in range(hours_per_day):
                    # Generator variables
                    for gen_id in gen_ids:
                        alpha[(node_id, day_id, scen_id, h, gen_id)] = model.addVar(vtype=GRB.BINARY, 
                            name=f"alpha_{node_id}_{day_id}_{scen_id}_{h}_{gen_id}")
                        gamma[(node_id, day_id, scen_id, h, gen_id)] = model.addVar(vtype=GRB.BINARY, 
                            name=f"gamma_{node_id}_{day_id}_{scen_id}_{h}_{gen_id}")
                        p_gen[(node_id, day_id, scen_id, h, gen_id)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0,
                            ub=gen_by_id[gen_id]['Pmax_MW'], name=f"p_gen_{node_id}_{day_id}_{scen_id}_{h}_{gen_id}")
                        s_res[(node_id, day_id, scen_id, h, gen_id)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0,
                            ub=gen_by_id[gen_id]['Pmax_MW'], name=f"s_res_{node_id}_{day_id}_{scen_id}_{h}_{gen_id}")
                    
                    # Transmission flow variables
                    for line_id in line_ids:
                        f_fwd[(node_id, day_id, scen_id, h, line_id)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0,
                            name=f"f_fwd_{node_id}_{day_id}_{scen_id}_{h}_{line_id}")
                        f_bwd[(node_id, day_id, scen_id, h, line_id)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0,
                            name=f"f_bwd_{node_id}_{day_id}_{scen_id}_{h}_{line_id}")
                    
                    # Storage variables
                    for storage_id in storage_ids:
                        u_stor[(node_id, day_id, scen_id, h, storage_id)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0,
                            ub=storage_by_id[storage_id]['capacity_MW'], name=f"u_stor_{node_id}_{day_id}_{scen_id}_{h}_{storage_id}")
                        v_stor[(node_id, day_id, scen_id, h, storage_id)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0,
                            ub=storage_by_id[storage_id]['capacity_MW'], name=f"v_stor_{node_id}_{day_id}_{scen_id}_{h}_{storage_id}")
                        r_stor[(node_id, day_id, scen_id, h, storage_id)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0,
                            name=f"r_stor_{node_id}_{day_id}_{scen_id}_{h}_{storage_id}")
    
    # Objective function: minimize total expected cost
    investment_cost = 0
    operating_cost = 0
    
    # Add investment costs
    for node_id in scenario_tree:
        node = scenario_tree[node_id]
        prob = node['probability']
        
        for gen_id in candidate_gen_ids:
            gen = gen_by_id[gen_id]
            investment_cost += prob * z_gen[(node_id, gen_id)] * gen['capital_cost_per_kW'] * gen['Pmax_MW'] * 1000
        
        for line_id in candidate_line_ids:
            line = line_by_id[line_id]
            investment_cost += prob * z_line[(node_id, line_id)] * line['capital_cost_per_kW'] * line['flow_limit_MW'] * 1000
            
        for storage_id in candidate_storage_ids:
            storage = storage_by_id[storage_id]
            investment_cost += prob * z_storage[(node_id, storage_id)] * storage['capital_cost_per_kW'] * storage['capacity_MW'] * 1000
    
    # Add operating costs
    for node_id in scenario_tree:
        node = scenario_tree[node_id]
        prob = node['probability']
        
        for day_id in representative_days:
            for scen_id, scen_data in enumerate(lower_level_scenarios[str(day_id)]):
                scen_prob = prob / num_representative_days / len(lower_level_scenarios[str(day_id)])
                
                for h in range(hours_per_day):
                    for gen_id in gen_ids:
                        gen = gen_by_id[gen_id]
                        
                        # Startup costs
                        operating_cost += scen_prob * gamma[(node_id, day_id, scen_id, h, gen_id)] * gen['startup_cost']
                        
                        # Generation costs (linearized as per benchmark)
                        operating_cost += scen_prob * (
                            gen['a_cost'] * alpha[(node_id, day_id, scen_id, h, gen_id)] +
                            (gen['b_cost'] + gen['c_cost'] * gen['Pmax_MW']) * p_gen[(node_id, day_id, scen_id, h, gen_id)]
                        )
    
    model.setObjective(investment_cost + operating_cost, GRB.MINIMIZE)
    
    # Constraints
    
    # Investment availability constraints
    for node_id in scenario_tree:
        node = scenario_tree[node_id]
        
        # For each candidate generator
        for gen_id in candidate_gen_ids:
            # Available capacity = capacity at parent + capacity built at this node
            if node['parent_id'] is not None:
                model.addConstr(
                    available_gen_cap[(node_id, gen_id)] == 
                    available_gen_cap[(node['parent_id'], gen_id)] + 
                    z_gen[(node_id, gen_id)] * gen_by_id[gen_id]['Pmax_MW']
                )
            else:
                model.addConstr(
                    available_gen_cap[(node_id, gen_id)] == 
                    z_gen[(node_id, gen_id)] * gen_by_id[gen_id]['Pmax_MW']
                )
        
        # For each candidate line
        for line_id in candidate_line_ids:
            if node['parent_id'] is not None:
                model.addConstr(
                    available_line_cap[(node_id, line_id)] == 
                    available_line_cap[(node['parent_id'], line_id)] + 
                    z_line[(node_id, line_id)] * line_by_id[line_id]['flow_limit_MW']
                )
            else:
                model.addConstr(
                    available_line_cap[(node_id, line_id)] == 
                    z_line[(node_id, line_id)] * line_by_id[line_id]['flow_limit_MW']
                )
        
        # For each candidate storage
        for storage_id in candidate_storage_ids:
            if node['parent_id'] is not None:
                model.addConstr(
                    available_storage_cap[(node_id, storage_id)] == 
                    available_storage_cap[(node['parent_id'], storage_id)] + 
                    z_storage[(node_id, storage_id)] * storage_by_id[storage_id]['capacity_MW']
                )
            else:
                model.addConstr(
                    available_storage_cap[(node_id, storage_id)] == 
                    z_storage[(node_id, storage_id)] * storage_by_id[storage_id]['capacity_MW']
                )
    
    # Generator commitment constraints
    for node_id in scenario_tree:
        for day_id in representative_days:
            for scen_id, scen_data in enumerate(lower_level_scenarios[str(day_id)]):
                for h in range(hours_per_day):
                    for gen_id in gen_ids:
                        gen = gen_by_id[gen_id]
                        
                        # For candidate generators, commitment cannot exceed available capacity ratio
                        if gen['existing']:
                            # For existing generators, max output is Pmax_MW
                            model.addConstr(
                                p_gen[(node_id, day_id, scen_id, h, gen_id)] + 
                                s_res[(node_id, day_id, scen_id, h, gen_id)] <= 
                                gen['Pmax_MW'] * alpha[(node_id, day_id, scen_id, h, gen_id)]
                            )
                        else:
                            # For candidate generators, max output is available capacity
                            model.addConstr(
                                p_gen[(node_id, day_id, scen_id, h, gen_id)] + 
                                s_res[(node_id, day_id, scen_id, h, gen_id)] <= 
                                available_gen_cap[(node_id, gen_id)]
                            )
                            
                            # Commitment is bounded by available capacity ratio
                            model.addConstr(
                                alpha[(node_id, day_id, scen_id, h, gen_id)] <= 
                                available_gen_cap[(node_id, gen_id)] / gen['Pmax_MW']
                            )
                        
                        # Minimum power output when committed
                        model.addConstr(
                            p_gen[(node_id, day_id, scen_id, h, gen_id)] >= 
                            gen['Pmin_MW'] * alpha[(node_id, day_id, scen_id, h, gen_id)]
                        )
    
    # Minimum up/down time constraints
    for node_id in scenario_tree:
        for day_id in representative_days:
            for scen_id, scen_data in enumerate(lower_level_scenarios[str(day_id)]):
                for gen_id in gen_ids:
                    gen = gen_by_id[gen_id]
                    min_on_time = gen['min_on_h']
                    min_off_time = gen['min_off_h']
                    
                    for h in range(hours_per_day):
                        # Startup indicator constraint
                        if h == 0:
                            # Assume initial state is off
                            model.addConstr(
                                gamma[(node_id, day_id, scen_id, h, gen_id)] >= 
                                alpha[(node_id, day_id, scen_id, h, gen_id)]
                            )
                        else:
                            model.addConstr(
                                gamma[(node_id, day_id, scen_id, h, gen_id)] >= 
                                alpha[(node_id, day_id, scen_id, h, gen_id)] - 
                                alpha[(node_id, day_id, scen_id, h-1, gen_id)]
                            )
                        
                        # Minimum up time constraint
                        # If started up in hour h, must stay on for at least min_on_time hours
                        for k in range(1, min_on_time):
                            if h + k < hours_per_day:
                                model.addConstr(
                                    alpha[(node_id, day_id, scen_id, h+k, gen_id)] >= 
                                    gamma[(node_id, day_id, scen_id, h, gen_id)]
                                )
                        
                        # Minimum down time constraint
                        # If shut down in hour h, must stay off for at least min_off_time hours
                        if h > 0:
                            shutdown = alpha[(node_id, day_id, scen_id, h-1, gen_id)] - alpha[(node_id, day_id, scen_id, h, gen_id)]
                            for k in range(1, min_off_time):
                                if h + k < hours_per_day:
                                    model.addConstr(
                                        1 - alpha[(node_id, day_id, scen_id, h+k, gen_id)] >= 
                                        shutdown
                                    )
    
    # Ramping constraints
    for node_id in scenario_tree:
        for day_id in representative_days:
            for scen_id, scen_data in enumerate(lower_level_scenarios[str(day_id)]):
                for gen_id in gen_ids:
                    gen = gen_by_id[gen_id]
                    for h in range(1, hours_per_day):  # Start from hour 1 to compare with previous
                        # Ramp up constraint
                        model.addConstr(
                            p_gen[(node_id, day_id, scen_id, h, gen_id)] - 
                            p_gen[(node_id, day_id, scen_id, h-1, gen_id)] <= 
                            gen['ramp_MW_per_h']
                        )
                        
                        # Ramp down constraint
                        model.addConstr(
                            p_gen[(node_id, day_id, scen_id, h-1, gen_id)] - 
                            p_gen[(node_id, day_id, scen_id, h, gen_id)] <= 
                            gen['ramp_MW_per_h']
                        )
    
    # Storage dynamics
    for node_id in scenario_tree:
        for day_id in representative_days:
            for scen_id, scen_data in enumerate(lower_level_scenarios[str(day_id)]):
                for storage_id in storage_ids:
                    storage = storage_by_id[storage_id]
                    
                    # Initial state of charge is 0 at beginning of each day
                    model.addConstr(
                        r_stor[(node_id, day_id, scen_id, 0, storage_id)] == 
                        storage['efficiency'] * u_stor[(node_id, day_id, scen_id, 0, storage_id)] - 
                        v_stor[(node_id, day_id, scen_id, 0, storage_id)]
                    )
                    
                    # Subsequent hours
                    for h in range(1, hours_per_day):
                        model.addConstr(
                            r_stor[(node_id, day_id, scen_id, h, storage_id)] == 
                            r_stor[(node_id, day_id, scen_id, h-1, storage_id)] + 
                            storage['efficiency'] * u_stor[(node_id, day_id, scen_id, h, storage_id)] - 
                            v_stor[(node_id, day_id, scen_id, h, storage_id)]
                        )
                    
                    # Storage capacity constraints
                    for h in range(hours_per_day):
                        if storage['existing']:
                            # For existing storage, capacity is fixed
                            model.addConstr(
                                r_stor[(node_id, day_id, scen_id, h, storage_id)] <= 
                                storage['capacity_MW']
                            )
                        else:
                            # For candidate storage, capacity depends on investment
                            model.addConstr(
                                r_stor[(node_id, day_id, scen_id, h, storage_id)] <= 
                                available_storage_cap[(node_id, storage_id)]
                            )
                        
                        # Discharge cannot exceed stored energy
                        model.addConstr(
                            v_stor[(node_id, day_id, scen_id, h, storage_id)] <= 
                            r_stor[(node_id, day_id, scen_id, h, storage_id)]
                        )
    
    # Power flow constraints
    for node_id in scenario_tree:
        for day_id in representative_days:
            for scen_id, scen_data in enumerate(lower_level_scenarios[str(day_id)]):
                for h in range(hours_per_day):
                    # Calculate actual demand for this hour
                    day_data = representative_days[day_id]
                    bus_loads = {}
                    for bus in buses:
                        if bus in loads_by_bus:
                            # Apply demand multiplier from scenario
                            base_demand = loads_by_bus[bus]['annual_peak_load_MW'] * day_data['load_profile'][h]
                            actual_demand = base_demand * scen_data['demand_multipliers'][h]
                            bus_loads[bus] = actual_demand
                        else:
                            bus_loads[bus] = 0  # No load at this bus
    
                    # Energy balance at each bus
                    for bus in buses:
                        # Power generated at this bus
                        gen_power = gp.quicksum(
                            p_gen[(node_id, day_id, scen_id, h, gen_id)] 
                            for gen_id in gens_by_bus[bus]
                        )
                        
                        # Net storage at this bus
                        stor_net = gp.quicksum(
                            v_stor[(node_id, day_id, scen_id, h, storage_id)] - 
                            u_stor[(node_id, day_id, scen_id, h, storage_id)]
                            for storage_id in [s['id'] for s in storage_units if s['bus'] == bus]
                        )
                        
                        # Power flowing out of this bus (subtracting losses)
                        flow_out = gp.quicksum(
                            f_fwd[(node_id, day_id, scen_id, h, line_id)] * (1 - line_by_id[line_id]['loss_factor'])
                            for line_id in lines_from_bus[bus]
                        )
                        
                        # Power flowing into this bus
                        flow_in = gp.quicksum(
                            f_bwd[(node_id, day_id, scen_id, h, line_id)]
                            for line_id in lines_to_bus[bus]
                        )
                        
                        # Energy balance equation
                        model.addConstr(
                            gen_power + stor_net + flow_in == 
                            bus_loads[bus] + flow_out
                        )
    
    # Transmission capacity constraints
    for node_id in scenario_tree:
        for day_id in representative_days:
            for scen_id, scen_data in enumerate(lower_level_scenarios[str(day_id)]):
                for h in range(hours_per_day):
                    for line_id in line_ids:
                        line = line_by_id[line_id]
                        if line['existing']:
                            model.addConstr(
                                f_fwd[(node_id, day_id, scen_id, h, line_id)] + 
                                f_bwd[(node_id, day_id, scen_id, h, line_id)] <= 
                                line['flow_limit_MW']
                            )
                        else:
                            model.addConstr(
                                f_fwd[(node_id, day_id, scen_id, h, line_id)] + 
                                f_bwd[(node_id, day_id, scen_id, h, line_id)] <= 
                                available_line_cap[(node_id, line_id)]
                            )
    
    # Spinning reserve constraints
    for node_id in scenario_tree:
        for day_id in representative_days:
            for scen_id, scen_data in enumerate(lower_level_scenarios[str(day_id)]):
                for h in range(hours_per_day):
                    for bus in buses:
                        # Calculate demand at this bus and hour
                        day_data = representative_days[day_id]
                        if bus in loads_by_bus:
                            base_demand = loads_by_bus[bus]['annual_peak_load_MW'] * day_data['load_profile'][h]
                            actual_demand = base_demand * scen_data['demand_multipliers'][h]
                        else:
                            actual_demand = 0
                        
                        # Sum of spinning reserves at this bus
                        total_spin_res = gp.quicksum(
                            s_res[(node_id, day_id, scen_id, h, gen_id)]
                            for gen_id in gens_by_bus[bus]
                        )
                        
                        # Constraint: spinning reserve must meet requirement
                        model.addConstr(total_spin_res >= spinning_reserve_pct * actual_demand)
    
    # Reserve margin requirements (grouping by region not specified in schema, assuming all buses form one region)
    for node_id in scenario_tree:
        # Calculate total derated capacity in the system
        total_derated_cap = gp.LinExpr()
        
        # Add existing generator derated capacity
        for gen in existing_gens:
            derating_factor = data['global_parameters']['derating_factors'].get(gen['type'], 1.0)
            total_derated_cap += gen['Pmax_MW'] * derating_factor
        
        # Add candidate generator derated capacity
        for gen_id in candidate_gen_ids:
            gen = gen_by_id[gen_id]
            derating_factor = data['global_parameters']['derating_factors'].get(gen['type'], 1.0)
            total_derated_cap += available_gen_cap[(node_id, gen_id)] * derating_factor
        
        # Calculate total peak demand at this node
        total_peak_demand = 0
        for bus_load in system['loads']:
            # Apply demand growth based on scenario path
            # For simplicity, assuming no growth for this basic implementation
            total_peak_demand += bus_load['annual_peak_load_MW']
        
        # Reserve margin constraint
        model.addConstr(total_derated_cap >= (1 + reserve_margin_req) * total_peak_demand)
    
    # Solve the model
    model.optimize()
    
    # Check if solution was found
    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible")
        model.computeIIS()
        model.write("infeasible.ilp")
        return
    elif model.status == GRB.TIME_LIMIT:
        print("Optimization terminated due to time limit")
    elif model.status != GRB.OPTIMAL and model.status != GRB.SUBOPTIMAL:
        print(f"Optimization ended with status {model.status}")
        return
    
    # Prepare solution dictionary
    solution = {
        "objective_value": model.objVal,
        "alpha": {},
        "gamma": {},
        "p_gen": {},
        "s_res": {},
        "f_fwd": {},
        "f_bwd": {},
        "u_stor": {},
        "v_stor": {},
        "r_stor": {},
        "investments": {}
    }
    
    # Fill in investment decisions
    for node_id in scenario_tree:
        solution["investments"][str(node_id)] = {
            "generators": {},
            "lines": {},
            "storage": {}
        }
        
        for gen_id in candidate_gen_ids:
            solution["investments"][str(node_id)]["generators"][gen_id] = int(z_gen[(node_id, gen_id)].x)
        
        for line_id in candidate_line_ids:
            solution["investments"][str(node_id)]["lines"][line_id] = int(z_line[(node_id, line_id)].x)
            
        for storage_id in candidate_storage_ids:
            solution["investments"][str(node_id)]["storage"][storage_id] = int(z_storage[(node_id, storage_id)].x)
    
    # Fill in operational variables
    for node_id in scenario_tree:
        for day_id in representative_days:
            for scen_id, scen_data in enumerate(lower_level_scenarios[str(day_id)]):
                for h in range(hours_per_day):
                    for gen_id in gen_ids:
                        key = f"{node_id},{day_id},{scen_id},{h},{gen_id}"
                        solution["alpha"][key] = alpha[(node_id, day_id, scen_id, h, gen_id)].x
                        solution["gamma"][key] = gamma[(node_id, day_id, scen_id, h, gen_id)].x
                        solution["p_gen"][key] = p_gen[(node_id, day_id, scen_id, h, gen_id)].x
                        solution["s_res"][key] = s_res[(node_id, day_id, scen_id, h, gen_id)].x
                    
                    for line_id in line_ids:
                        key = f"{node_id},{day_id},{scen_id},{h},{line_id}"
                        solution["f_fwd"][key] = f_fwd[(node_id, day_id, scen_id, h, line_id)].x
                        solution["f_bwd"][key] = f_bwd[(node_id, day_id, scen_id, h, line_id)].x
                    
                    for storage_id in storage_ids:
                        key = f"{node_id},{day_id},{scen_id},{h},{storage_id}"
                        solution["u_stor"][key] = u_stor[(node_id, day_id, scen_id, h, storage_id)].x
                        solution["v_stor"][key] = v_stor[(node_id, day_id, scen_id, h, storage_id)].x
                        solution["r_stor"][key] = r_stor[(node_id, day_id, scen_id, h, storage_id)].x
    
    # Log solution if logger exists
    if logger:
        logger.log_solution(model.objVal, solution)
    
    # Write solution to file
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


def main():
    parser = argparse.ArgumentParser(description='Solve power system expansion problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to output solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log file (optional)')
    
    args = parser.parse_args()
    
    solve_power_system_expansion(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == "__main__":
    main()