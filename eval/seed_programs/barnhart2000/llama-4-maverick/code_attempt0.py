import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def build_model(instance):
    num_nodes = instance['network']['num_nodes']
    num_commodities = instance['commodities']['num_commodities']
    nodes = instance['network']['nodes']
    arcs = instance['network']['arcs']
    commodities = instance['commodities']['commodity_list']

    model = Model('network_routing')
    model.modelSense = GRB.MINIMIZE

    # Create flow variables for each commodity on each arc
    flow_vars = {}
    for commodity in commodities:
        commodity_id = commodity['commodity_id']
        flow_vars[commodity_id] = {}
        for arc in arcs + [{'arc_id': f"artificial_{commodity_id}", 'from_node': commodity['origin'], 'to_node': commodity['destination'], 'capacity': GRB.INFINITY, 'cost': commodity['artificial_arc_cost']}]:
            arc_id = arc['arc_id']
            flow_vars[commodity_id][arc_id] = model.addVar(vtype=GRB.BINARY, name=f"flow_{commodity_id}_{arc_id}")

    # Objective: minimize total cost (revenue lost from rejected commodities)
    model.setObjective(quicksum(commodity['demand'] * flow_vars[commodity['commodity_id']][f"artificial_{commodity['commodity_id']}"] * commodity['artificial_arc_cost'] / commodity['artificial_arc_cost'] for commodity in commodities), GRB.MINIMIZE)

    # Constraints: flow conservation for each commodity at each node
    for commodity in commodities:
        commodity_id = commodity['commodity_id']
        origin = commodity['origin']
        destination = commodity['destination']
        for node in nodes:
            inflow = quicksum(flow_vars[commodity_id][arc['arc_id']] for arc in arcs + [{'arc_id': f"artificial_{commodity_id}", 'from_node': origin, 'to_node': destination}] if arc['to_node'] == node)
            outflow = quicksum(flow_vars[commodity_id][arc['arc_id']] for arc in arcs + [{'arc_id': f"artificial_{commodity_id}", 'from_node': origin, 'to_node': destination}] if arc['from_node'] == node)
            if node == origin:
                model.addConstr(outflow - inflow == 1, name=f"flow_conservation_{commodity_id}_{node}_origin")
            elif node == destination:
                model.addConstr(inflow - outflow == 1, name=f"flow_conservation_{commodity_id}_{node}_destination")
            else:
                model.addConstr(inflow == outflow, name=f"flow_conservation_{commodity_id}_{node}")

    # Constraints: exactly one path per commodity
    for commodity in commodities:
        commodity_id = commodity['commodity_id']
        model.addConstr(quicksum(flow_vars[commodity_id][arc['arc_id']] for arc in arcs + [{'arc_id': f"artificial_{commodity_id}"}]) == 1, name=f"single_path_{commodity_id}")

    # Constraints: capacity constraints on physical arcs
    for arc in arcs:
        arc_id = arc['arc_id']
        model.addConstr(quicksum(commodity['demand'] * flow_vars[commodity['commodity_id']][arc_id] for commodity in commodities) <= arc['capacity'], name=f"capacity_{arc_id}")

    return model, flow_vars, commodities, arcs

def solve_model(model, time_limit, logger):
    model.setParam('TimeLimit', time_limit)
    model.optimize()
    if logger:
        for obj_value in model.getAttr('IncumbentObj'):
            if obj_value != GRB.INFINITY:
                logger.log(obj_value)
    return model

def extract_solution(model, flow_vars, commodities, arcs):
    solution = {'commodities': []}
    objective_value = model.getObjective().getValue()
    solution['objective_value'] = objective_value

    for commodity in commodities:
        commodity_id = commodity['commodity_id']
        commodity_solution = {'commodity_id': commodity_id, 'rejected': False, 'path_arcs': []}
        for arc_id, var in flow_vars[commodity_id].items():
            if var.x > 0.5:
                if arc_id.startswith('artificial'):
                    commodity_solution['rejected'] = True
                else:
                    arc = next(arc for arc in arcs if arc['arc_id'] == arc_id)
                    commodity_solution['path_arcs'].append({'from': arc['from_node'], 'to': arc['to_node'], 'arc_id': arc_id})
        solution['commodities'].append(commodity_solution)

    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', required=False)
    args = parser.parse_args()

    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None

    instance = read_instance(args.instance_path)
    model, flow_vars, commodities, arcs = build_model(instance)
    model = solve_model(model, args.time_limit, logger)
    solution = extract_solution(model, flow_vars, commodities, arcs)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()