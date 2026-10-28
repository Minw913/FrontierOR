import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    return parser.parse_args()

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def model_instance(instance):
    num_nodes = instance['num_nodes']
    num_arcs = instance['num_arcs']
    num_commodities = instance['num_commodities']
    arcs = instance['arcs']
    fixed_costs = instance['fixed_costs']
    variable_costs = instance['variable_costs']
    commodities = instance['commodities']

    # Create a Gurobi model
    model = gp.Model("network_design")

    # Arc installation variables
    arc_installed = model.addVars([f'{arc[0]}_{arc[1]}' for arc in arcs], vtype=GRB.BINARY, name='arc_installed')

    # Commodity flow variables
    flow = model.addVars([f'{commodity_index}_{arc[0]}_{arc[1]}' for commodity_index in range(num_commodities) for arc in arcs], vtype=GRB.CONTINUOUS, name='flow', lb=0, ub=1)

    # Objective function: minimize total cost
    model.setObjective(
        gp.quicksum(fixed_costs[f'{arc[0]}_{arc[1]}'] * arc_installed[f'{arc[0]}_{arc[1]}'] for arc in arcs) +
        gp.quicksum(commodities[commodity_index]['demand'] * variable_costs[f'{arc[0]}_{arc[1]}'][commodity_index] * flow[f'{commodity_index}_{arc[0]}_{arc[1]}'] for commodity_index in range(num_commodities) for arc in arcs),
        GRB.MINIMIZE
    )

    # Constraints: flow conservation for each commodity at each node
    for commodity_index, commodity in enumerate(commodities):
        for node in range(num_nodes):
            inflow = gp.quicksum(flow[f'{commodity_index}_{arc[0]}_{arc[1]}'] for arc in arcs if arc[1] == node)
            outflow = gp.quicksum(flow[f'{commodity_index}_{arc[0]}_{arc[1]}'] for arc in arcs if arc[0] == node)
            if node == commodity['origin']:
                model.addConstr(inflow - outflow == -1, f'flow_conservation_{commodity_index}_{node}_origin')
            elif node == commodity['destination']:
                model.addConstr(inflow - outflow == 1, f'flow_conservation_{commodity_index}_{node}_destination')
            else:
                model.addConstr(inflow - outflow == 0, f'flow_conservation_{commodity_index}_{node}')

    # Constraints: flow can only occur on installed arcs
    for commodity_index in range(num_commodities):
        for arc in arcs:
            arc_key = f'{arc[0]}_{arc[1]}'
            model.addConstr(flow[f'{commodity_index}_{arc[0]}_{arc[1]}'] <= arc_installed[arc_key], f'arc_capacity_{commodity_index}_{arc_key}')

    return model, arc_installed, flow, num_commodities, arcs, commodities

def solve_model(model, time_limit):
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0  # Suppress Gurobi logging to console
    model.optimize()

def extract_solution(model, arc_installed, flow, num_commodities, arcs, commodities):
    try:
        objective_value = model.ObjVal
    except gp.GurobiError:
        # If no solution is available (e.g., due to timeout without finding a feasible solution)
        objective_value = float('inf')  # or handle appropriately

    open_arcs = {f'{arc[0]}_{arc[1]}': int(arc_installed[f'{arc[0]}_{arc[1]}'].X) for arc in arcs}

    routings = {}
    for commodity_index in range(num_commodities):
        commodity_routing = {}
        for arc in arcs:
            arc_key = f'{arc[0]}_{arc[1]}'
            flow_value = flow[f'{commodity_index}_{arc[0]}_{arc[1]}'].X if model.SolCount > 0 else 0
            commodity_routing[arc_key] = flow_value
        routings[str(commodity_index)] = commodity_routing

    return objective_value, open_arcs, routings

def write_solution(solution_path, objective_value, open_arcs, routings):
    solution = {
        "objective_value": objective_value,
        "open_arcs": open_arcs,
        "routings": routings
    }
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

def main():
    args = parse_args()
    instance = read_instance(args.instance_path)
    model, arc_installed, flow, num_commodities, arcs, commodities = model_instance(instance)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Set up a callback to log incumbents
    def log_incumbent(model, where):
        if where == GRB.Callback.MIPSOL:
            objective_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(objective_value)

    model.optimize(log_incumbent)
    solve_model(model, args.time_limit)

    objective_value, open_arcs, routings = extract_solution(model, arc_installed, flow, num_commodities, arcs, commodities)
    write_solution(args.solution_path, objective_value, open_arcs, routings)

if __name__ == "__main__":
    main()