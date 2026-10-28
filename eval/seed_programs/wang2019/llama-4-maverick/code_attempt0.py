import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def build_model(instance):
    num_nodes = instance['num_nodes']
    num_arcs = instance['num_arcs']
    num_commodities = instance['num_commodities']
    num_fleet_types = instance['num_fleet_types']
    nodes = instance['nodes']
    arcs = instance['arcs']
    commodities = instance['commodities']
    fleet_types = instance['fleet_types']

    arc_dict = {(arc['from'], arc['to']): arc['id'] for arc in arcs}
    arc_data = {(arc['from'], arc['to']): {'flow_cost': arc['flow_cost']} for arc in arcs}

    model = gp.Model('freight_transportation')
    model.modelSense = GRB.MINIMIZE

    # Flow variables
    flows = {}
    for commodity in commodities:
        for arc in arcs:
            from_node = arc['from']
            to_node = arc['to']
            flows[(commodity['id'], from_node, to_node)] = model.addVar(lb=0, name=f'x_{commodity["id"]}_{from_node}_{to_node}')

    # Vehicle variables
    vehicles = {}
    for fleet_type in fleet_types:
        for arc in arcs:
            from_node = arc['from']
            to_node = arc['to']
            vehicles[(fleet_type['id'], from_node, to_node)] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f'y_{fleet_type["id"]}_{from_node}_{to_node}')

    # Objective function
    model.setObjective(
        gp.quicksum(flows[(commodity['id'], arc['from'], arc['to'])] * arc_data[(arc['from'], arc['to'])]['flow_cost'] for commodity in commodities for arc in arcs) +
        gp.quicksum(vehicles[(fleet_type['id'], arc['from'], arc['to'])] * fleet_type['fixed_cost'] for fleet_type in fleet_types for arc in arcs)
    )

    # Flow balance constraints for commodities
    for commodity in commodities:
        for node in nodes:
            inflow = gp.quicksum(flows[(commodity['id'], arc['from'], node)] for arc in arcs if arc['to'] == node)
            outflow = gp.quicksum(flows[(commodity['id'], node, arc['to'])] for arc in arcs if arc['from'] == node)
            demand = commodity['demand'] if node == commodity['origin'] else -commodity['demand'] if node == commodity['destination'] else 0
            model.addConstr(inflow - outflow == demand, name=f'flow_balance_{commodity["id"]}_{node}')

    # Capacity constraints
    for arc in arcs:
        from_node = arc['from']
        to_node = arc['to']
        total_flow = gp.quicksum(flows[(commodity['id'], from_node, to_node)] for commodity in commodities)
        total_capacity = gp.quicksum(vehicles[(fleet_type['id'], from_node, to_node)] * fleet_type['capacity'] for fleet_type in fleet_types)
        model.addConstr(total_flow <= total_capacity, name=f'capacity_{from_node}_{to_node}')

    # Vehicle balance constraints
    for fleet_type in fleet_types:
        for node in nodes:
            inflow = gp.quicksum(vehicles[(fleet_type['id'], arc['from'], node)] for arc in arcs if arc['to'] == node)
            outflow = gp.quicksum(vehicles[(fleet_type['id'], node, arc['to'])] for arc in arcs if arc['from'] == node)
            model.addConstr(inflow == outflow, name=f'vehicle_balance_{fleet_type["id"]}_{node}')

    return model, flows, vehicles

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    model, flows, vehicles = build_model(instance)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model.Params.TimeLimit = args.time_limit
    model.Params.LogToConsole = 1

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        solution = {
            'objective_value': model.ObjVal,
            'flows': {f'x_{commodity}_{from}_{to}': flows[(commodity, from_, to_)].x for (commodity, from_, to_), var in flows.items()},
            'vehicles': {f'y_{fleet_type}_{from}_{to}': vehicles[(fleet_type, from_, to_)].x for (fleet_type, from_, to_), var in vehicles.items()}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)

        if logger and model.status == GRB.TIME_LIMIT:
            logger.log(model.ObjVal)

    if logger:
        logger.close()

if __name__ == '__main__':
    main()