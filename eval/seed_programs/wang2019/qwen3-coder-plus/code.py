import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
def solve_instance(instance_data, time_limit, logger):
    # Extract instance data
    num_nodes = instance_data['num_nodes']
    num_arcs = instance_data['num_arcs']
    num_commodities = instance_data['num_commodities']
    num_fleet_types = instance_data['num_fleet_types']
    
    nodes = instance_data['nodes']
    arcs = instance_data['arcs']
    commodities = instance_data['commodities']
    fleet_types = instance_data['fleet_types']
    
    # Create index mappings for easier access
    arc_dict = {arc['id']: {'from': arc['from'], 'to': arc['to']} for arc in arcs}
    commodity_dict = {comm['id']: {'origin': comm['origin'], 'destination': comm['destination'], 'demand': comm['demand']} 
                      for comm in commodities}
    fleet_dict = {ftype['id']: {'capacity': ftype['capacity'], 'fixed_cost': ftype['fixed_cost']} 
                 for ftype in fleet_types}
    
    # Create dictionaries for faster lookup by from/to pair
    arc_costs = {(arc['from'], arc['to']): arc['flow_cost'] for arc in arcs}
    arc_ids_map = {(arc['from'], arc['to']): arc['id'] for arc in arcs}
    
    # Build network structure - adjacency lists for in/outgoing arcs
    outgoing_arcs = {node: [] for node in nodes}
    incoming_arcs = {node: [] for node in nodes}
    
    for arc in arcs:
        fr = arc['from']
        to = arc['to']
        outgoing_arcs[fr].append((fr, to))
        incoming_arcs[to].append((fr, to))
    
    # Start the Gurobi model
    model = gp.Model("Freight Transportation")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    
    # Suppress console output
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)
    
    # Decision Variables
    # Flow variables: x[c_id, from, to] - flow of commodity c on arc (from, to)
    flows = {}
    for comm in commodities:
        for arc in arcs:
            c_id = comm['id']
            fr = arc['from']
            to = arc['to']
            flows[c_id, fr, to] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"x_{c_id}_{fr}_{to}")
    
    # Vehicle assignment variables: y[f_id, from, to] - number of vehicles of type f on arc (from, to)
    vehicles = {}
    for ftype in fleet_types:
        for arc in arcs:
            f_id = ftype['id']
            fr = arc['from']
            to = arc['to']
            vehicles[f_id, fr, to] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"y_{f_id}_{fr}_{to}")
    
    # Constraints: flow conservation for each commodity at each node
    for comm in commodities:
        c_id = comm['id']  # Fixed: it was comm['origin']
        origin_node = commodity_dict[c_id]['origin']
        dest_node = commodity_dict[c_id]['destination']
        demand = commodity_dict[c_id]['demand']
        
        for node in nodes:
            flow_balance_expr = gp.LinExpr()
            
            # Incoming flow (into this node)
            for fr, to in incoming_arcs[node]:
                flow_balance_expr.add(flows[c_id, fr, to])
                
            # Outgoing flow (out of this node)
            for fr, to in outgoing_arcs[node]:
                flow_balance_expr.add(-flows[c_id, fr, to])
                
            # Right-hand side depends on whether node is origin, destination, or intermediate
            if node == origin_node:  # Origin node, flow out is positive at full demand
                rhs = -demand
            elif node == dest_node:  # Dest node, flow in is positive at full demand
                rhs = demand
            else:  # Intermediate nodes, net flow is 0
                rhs = 0
            
            model.addConstr(flow_balance_expr == rhs, name=f"flow_balance_comm_{c_id}_node_{node}")
    
    # Constraints: capacity constraint for each arc
    for fr, to in arc_ids_map.keys():
        total_flow_on_arc = gp.LinExpr()
        
        # Sum all commodity flows on this arc
        for comm in commodities:
            c_id = comm['id']
            total_flow_on_arc.add(flows[c_id, fr, to])
        
        # Sum total capacity provided by all vehicle types on this arc
        total_capacity = gp.LinExpr()
        for ftype in fleet_types:
            f_id = ftype['id']
            total_capacity.add(vehicles[f_id, fr, to] * ftype['capacity'])
        
        model.addConstr(total_flow_on_arc <= total_capacity, name=f"capacity_{fr}_{to}")
    
    # Constraints: vehicle flow balance for each vehicle type at each node
    for ftype in fleet_types:
        f_id = ftype['id']
        
        for node in nodes:
            vehicle_balance_expr = gp.LinExpr()
            
            # Incoming vehicles (into this node)
            for fr, to in incoming_arcs[node]:
                vehicle_balance_expr.add(vehicles[f_id, fr, to])
                
            # Outgoing vehicles (from this node)
            for fr, to in outgoing_arcs[node]:
                vehicle_balance_expr.add(-vehicles[f_id, fr, to])
                
            model.addConstr(vehicle_balance_expr == 0, name=f"vehicle_balance_fleet_{f_id}_node_{node}")
    
    # Objective function
    total_flow_cost = gp.quicksum(
        flows[comm['id'], fr, to] * arc_costs[fr, to]
        for comm in commodities
        for fr, to in arc_ids_map.keys()
    )
    
    total_vehicle_cost = gp.quicksum(
        vehicles[ftype['id'], fr, to] * ftype['fixed_cost']
        for ftype in fleet_types
        for fr, to in arc_ids_map.keys()
    )
    
    model.setObjective(total_flow_cost + total_vehicle_cost, GRB.MINIMIZE)
    
    # Set callback for logging if logger is provided
    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            logger.log(obj)
    
    if logger:
        model.optimize(mycallback)
    else:
        model.optimize()
    
    # Prepare solution
    obj_val = model.objVal if model.solCount > 0 else None
    
    solution = {"objective_value": obj_val, "flows": {}, "vehicles": {}}
    
    if model.solCount > 0:
        # Fill in flows
        for (c_id, fr, to), var in flows.items():
            flow_value = var.x
            if flow_value > 1e-8:  # Only include non-trivial values to save space
                solution["flows"][f"x_{c_id}_{fr}_{to}"] = flow_value
        
        # Fill in vehicles
        for (f_id, fr, to), var in vehicles.items():
            vehicle_value = var.x
            solution["vehicles"][f"y_{f_id}_{fr}_{to}"] = int(round(vehicle_value))
    
    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    
    args = parser.parse_args()
    
    # Initialize the logger if log_path is provided
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Solve the instance
    solution = solve_instance(instance_data, args.time_limit, logger)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()