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
    
    # Decision variables
    x = {}  # Generation level
    y = {}  # On/off status
    u = {}  # Startup indicator
    for node in nodes:
        node_id = node['id']
        x[node_id] = model.addVar(lb=0, ub=generator['C_upper_MW'], vtype=GRB.CONTINUOUS, name=f'x_{node_id}')
        y[node_id] = model.addVar(vtype=GRB.BINARY, name=f'y_{node_id}')
        if node['parent_id'] is not None:
            u[node_id] = model.addVar(vtype=GRB.BINARY, name=f'u_{node_id}')
    
    # Objective function
    model.setObjective(quicksum(node['probability'] * (node['electricity_price_dollars_per_MWh'] * x[node['id']] 
                                                       - startup_shutdown_cost(model, generator, node['id'], u, y) 
                                                       - fuel_cost(model, generator, x[node['id']])) 
                                for node in nodes), GRB.MAXIMIZE)
    
    # Constraints
    for node in nodes:
        node_id = node['id']
        time = node['time']
        parent_id = node['parent_id']
        children_ids = node['children_ids']
        
        # Generation level and on/off status
        model.addConstr(x[node_id] <= generator['C_upper_MW'] * y[node_id], name=f'upper_gen_limit_{node_id}')
        model.addConstr(x[node_id] >= generator['C_lower_MW'] * y[node_id], name=f'lower_gen_limit_{node_id}')
        
        if parent_id is not None:
            # Startup and shutdown logic
            model.addConstr(u[node_id] >= y[node_id] - y[parent_id], name=f'startup_logic_{node_id}')
            model.addConstr(u[node_id] <= y[node_id], name=f'startup_upper_bound_{node_id}')
            model.addConstr(u[node_id] <= 1 - y[parent_id], name=f'startup_lower_bound_{node_id}')
            
            # Ramp-up and ramp-down limits
            if y[parent_id] == 1:
                model.addConstr(x[node_id] - x[parent_id] <= generator['V_plus_MW_per_h'], name=f'ramp_up_{node_id}')
                model.addConstr(x[parent_id] - x[node_id] <= generator['V_minus_MW_per_h'] + generator['C_upper_MW'] * (1 - y[node_id]), name=f'ramp_down_{node_id}')
            else:
                model.addConstr(x[node_id] <= generator['C_upper_MW'], name=f'ramp_up_from_off_{node_id}')
        
        # Minimum up and down times
        for child_id in children_ids:
            child_node = next(n for n in nodes if n['id'] == child_id)
            if time + 1 <= child_node['time']:
                model.addConstr(y[child_id] >= y[node_id] - (1 - u[node_id]), name=f'min_up_time_{node_id}_{child_id}')
            if time + 1 <= child_node['time'] + scenario_tree['ell_min_down_time'] - 1:
                model.addConstr(y[child_id] <= 1 - u[node_id], name=f'min_down_time_after_startup_{node_id}_{child_id}')
        
        # Initial conditions
        if node_id == nodes[0]['id']:
            model.addConstr(y[node_id] == scenario_tree['initial_generator_status_y0'], name='initial_status')
            model.addConstr(x[node_id] == scenario_tree['initial_generation_x0_MW'] * y[node_id], name='initial_generation')
    
    return model, x, y, u

def startup_shutdown_cost(model, generator, node_id, u, y):
    parent_id = next((n['parent_id'] for n in model._nodes if n['id'] == node_id), None)
    if parent_id is not None:
        startup_cost = generator['U_bar_startup_cost_dollars'] * u[node_id]
        shutdown_cost = generator['U_lower_shutdown_cost_dollars'] * (y[parent_id] - y[node_id] + u[node_id])
        return startup_cost + shutdown_cost
    else:
        return 0

def fuel_cost(model, generator, generation_level):
    # Simplified fuel cost calculation based on provided data schema
    return generator['fuel_a_dollars_per_h'] * generation_level + generator['fuel_b_dollars_per_MWh'] * generation_level

def solve_instance(instance_data, time_limit):
    model, x, y, u = build_model(instance_data)
    model._nodes = instance_data['scenario_tree']['nodes']
    model.Params.TimeLimit = time_limit
    model.optimize()
    return model, x, y, u

def format_solution(model, x, y, u):
    solution = {'objective_value': model.ObjVal}
    solution['x'] = {node_id: x[node_id].X for node_id in x}
    solution['y'] = {node_id: y[node_id].X for node_id in y}
    solution['u'] = {node_id: u[node_id].X for node_id in u}
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