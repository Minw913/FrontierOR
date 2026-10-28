import gurobipy as gp
import json
import argparse
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser(description='Solve the telecommunications network routing problem.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the input JSON instance file.')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file.')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds.')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log intermediate solutions (optional).')
    args = parser.parse_args()

    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    network = instance['network']
    commodities = instance['commodities']
    nodes = network['nodes']
    arcs = network['arcs']
    commodity_list = commodities['commodity_list']
    
    # Precompute node mappings for physical arcs
    outgoing = {node: [] for node in nodes}
    incoming = {node: [] for node in nodes}
    arc_map = {}
    for arc in arcs:
        a_id = arc['arc_id']
        from_node = arc['from_node']
        to_node = arc['to_node']
        outgoing[from_node].append(a_id)
        incoming[to_node].append(a_id)
        arc_map[a_id] = arc
    
    # Initialize Gurobi model
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', 1)
    
    # Create variables
    x = {}  # Physical arc variables: (commodity_id, arc_id) -> binary
    y = {}  # Artificial arc variables: commodity_id -> binary
    
    for comm in commodity_list:
        k_id = comm['commodity_id']
        y[k_id] = model.addVar(vtype=gp.GRB.BINARY, name=f"y_{k_id}")
        for arc in arcs:
            a_id = arc['arc_id']
            x[(k_id, a_id)] = model.addVar(vtype=gp.GRB.BINARY, name=f"x_{k_id}_{a_id}")
    
    # Set initial solution: reject all commodities
    for comm in commodity_list:
        k_id = comm['commodity_id']
        y[k_id].Start = 1
        for arc in arcs:
            a_id = arc['arc_id']
            x[(k_id, a_id)].Start = 0
    
    # Add flow conservation constraints
    for comm in commodity_list:
        k_id = comm['commodity_id']
        origin = comm['origin']
        dest = comm['destination']
        for node in nodes:
            expr = 0.0
            # Physical outgoing arcs
            for a_id in outgoing.get(node, []):
                expr += x[(k_id, a_id)]
            # Physical incoming arcs
            for a_id in incoming.get(node, []):
                expr -= x[(k_id, a_id)]
            # Artificial arc
            if node == origin:
                expr += y[k_id]
            if node == dest:
                expr -= y[k_id]
            # Set constraint based on node type
            if node == origin:
                model.addConstr(expr == 1, name=f"flow_cons_{k_id}_{node}")
            elif node == dest:
                model.addConstr(expr == -1, name=f"flow_cons_{k_id}_{node}")
            else:
                model.addConstr(expr == 0, name=f"flow_cons_{k_id}_{node}")
    
    # Add capacity constraints for physical arcs
    for arc in arcs:
        a_id = arc['arc_id']
        cap = arc['capacity']
        expr = 0.0
        for comm in commodity_list:
            k_id = comm['commodity_id']
            demand = comm['demand']
            expr += demand * x[(k_id, a_id)]
        model.addConstr(expr <= cap, name=f"cap_{a_id}")
    
    # Set objective: minimize total cost of rejected commodities
    obj = 0.0
    for comm in commodity_list:
        k_id = comm['commodity_id']
        cost_k = comm['artificial_arc_cost']
        obj += cost_k * y[k_id]
    model.setObjective(obj, gp.GRB.MINIMIZE)
    
    # Set time limit
    model.setParam('TimeLimit', args.time_limit)
    
    # Initialize solution logger if provided
    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None
    
    # Optimize with callback for logging
    if logger:
        model._best_obj = None
        def callback(model, where):
            if where == gp.GRB.Callback.MIPSOL:
                obj_val = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
                if model._best_obj is None or obj_val < model._best_obj:
                    model._best_obj = obj_val
                    logger.log(obj_val)
        model.optimize(callback)
    else:
        model.optimize()
    
    # Prepare solution output
    solution = {}
    solution_commodities = []
    
    # Check if a solution was found; use initial solution otherwise
    if model.SolCount > 0:
        obj_val = model.ObjVal
    else:
        obj_val = sum(comm['artificial_arc_cost'] for comm in commodity_list)
    
    solution['objective_value'] = obj_val
    
    # Extract solution for each commodity
    for comm in commodity_list:
        k_id = comm['commodity_id']
        origin = comm['origin']
        dest = comm['destination']
        sol_comm = {"commodity_id": k_id}
        
        # Check if commodity is rejected
        if model.SolCount > 0 and y[k_id].X > 0.5:
            sol_comm["rejected"] = True
            sol_comm["path_arcs"] = [{"from": origin, "to": dest, "arc_id": -k_id}]
        else:
            sol_comm["rejected"] = False
            path_arcs = []
            # If solution found, extract used physical arcs; otherwise, use initial (rejected) state
            if model.SolCount > 0:
                for arc in arcs:
                    a_id = arc['arc_id']
                    if x[(k_id, a_id)].X > 0.5:
                        path_arcs.append({"from": arc['from_node'], "to": arc['to_node'], "arc_id": a_id})
            sol_comm["path_arcs"] = path_arcs
        solution_commodities.append(sol_comm)
    
    solution['commodities'] = solution_commodities
    
    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()