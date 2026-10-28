import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        return json.load(file)

def build_model(instance_data):
    generator = instance_data['generator']
    scenario_tree = instance_data['scenario_tree']
    nodes = scenario_tree['nodes']
    
    model = Model('generator_scheduling')
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    
    # Extract relevant data
    C_upper_MW = generator['C_upper_MW']
    C_lower_MW = generator['C_lower_MW']
    V_plus_MW_per_h = generator['V_plus_MW_per_h']
    V_minus_MW_per_h = generator['V_minus_MW_per_h']
    U_bar_startup_cost_dollars = generator['U_bar_startup_cost_dollars']
    U_lower_shutdown_cost_dollars = generator['U_lower_shutdown_cost_dollars']
    fuel_a_dollars_per_h = generator['fuel_a_dollars_per_h']
    fuel_b_dollars_per_MWh = generator['fuel_b_dollars_per_MWh']
    L_min_up_time = scenario_tree['L_min_up_time']
    ell_min_down_time = scenario_tree['ell_min_down_time']
    initial_generator_status_y0 = scenario_tree['initial_generator_status_y0']
    initial_generation_x0_MW = scenario_tree['initial_generation_x0_MW']
    
    # Decision variables
    x = {}  # Generation level
    y = {}  # On/off status
    u = {}  # Startup indicator
    for node in nodes:
        node_id = node['id']
        x[node_id] = model.addVar(lb=0, ub=C_upper_MW, vtype=GRB.CONTINUOUS, name=f'x_{node_id}')
        y[node_id] = model.addVar(vtype=GRB.BINARY, name=f'y_{node_id}')
        if node['parent_id'] is not None:
            u[node_id] = model.addVar(vtype=GRB.BINARY, name=f'u_{node_id}')
    
    # Objective function
    model.setObjective(quicksum(node['probability'] * (node['electricity_price_dollars_per_MWh'] * x[node['id']] 
                                                       - U_bar_startup_cost_dollars * u.get(node['id'], 0) 
                                                       - U_lower_shutdown_cost_dollars * (y.get(node['parent_id'], 0) - y[node['id']] + u.get(node['id'], 0)) 
                                                       - (fuel_a_dollars_per_h + fuel_b_dollars_per_MWh * x[node['id']])) 
                                for node in nodes if 'probability' in node), GRB.MAXIMIZE)
    
    # Constraints
    for node in nodes:
        node_id = node['id']
        time = node['time']
        parent_id = node['parent_id']
        children_ids = node['children_ids']
        
        # Generation level and on/off status
        model.addConstr(x[node_id] <= C_upper_MW * y[node_id], name=f'upper_gen_limit_{node_id}')
        model.addConstr(x[node_id] >= C_lower_MW * y[node_id], name=f'lower_gen_limit_{node_id}')
        
        if parent_id is not None:
            # Startup and shutdown logic
            model.addConstr(u[node_id] >= y[node_id] - y[parent_id], name=f'startup_logic_{node_id}')
            model.addConstr(u[node_id] <= y[node_id], name=f'startup_upper_bound_{node_id}')
            model.addConstr(u[node_id] <= 1 - y[parent_id], name=f'startup_lower_bound_{node_id}')
            
            # Ramp-up and ramp-down limits
            model.addConstr(x[node_id] - x[parent_id] <= V_plus_MW_per_h + C_upper_MW * (1 - y[parent_id]), name=f'ramp_up_{node_id}')
            model.addConstr(x[parent_id] - x[node_id] <= V_minus_MW_per_h + C_upper_MW * (1 - y[node_id]), name=f'ramp_down_{node_id}')
        
        # Minimum up and down times
        if L_min_up_time > 1:
            for child_id in children_ids:
                child_node = next(n for n in nodes if n['id'] == child_id)
                if time + 1 <= child_node['time'] <= time + L_min_up_time - 1:
                    model.addConstr(y[child_id] >= y[node_id], name=f'min_up_time_{node_id}_{child_id}')
        if ell_min_down_time > 1:
            for child_id in children_ids:
                child_node = next(n for n in nodes if n['id'] == child_id)
                if time + 1 <= child_node['time'] <= time + ell_min_down_time - 1:
                    model.addConstr(y[child_id] <= 1 - u.get(node_id, 0), name=f'min_down_time_{node_id}_{child_id}')
        
        # Initial conditions for root node
        if node_id == nodes[0]['id']:
            model.addConstr(y[node_id] == initial_generator_status_y0, name='initial_status')
            if initial_generator_status_y0 == 1:
                model.addConstr(x[node_id] == initial_generation_x0_MW, name='initial_generation')
            else:
                model.addConstr(x[node_id] == 0, name='initial_generation_off')
    
    return model, x, y, u

def solve_instance(instance_data, time_limit):
    model, x, y, u = build_model(instance_data)
    model.Params.TimeLimit = time_limit
    model.optimize()
    return model, x, y, u

def format_solution(model, x, y, u):
    solution = {'objective_value': model.ObjVal if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT else None}
    solution['x'] = {node_id: x[node_id].X for node_id in x}
    solution['y'] = {node_id: y[node_id].X for node_id in y}
    if u:
        solution['u'] = {node_id: u[node_id].X for node_id in u}
    solution['sparse_defaults'] = {'x': 0, 'y': 0, 'u': 0}
    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()
    
    instance_data = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense='maximize') if args.log_path else None
    
    model, x, y, u = solve_instance(instance_data, args.time_limit)
    
    if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
        solution = format_solution(model, x, y, u)
        with open(args.solution_path, 'w') as file:
            json.dump(solution, file, indent=2)
        if logger:
            logger.log_solution(model.ObjVal, solution)

if __name__ == '__main__':
    main()