import argparse
import json
import time
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB

def solve(instance_path, solution_path, time_limit, log_path=None):
    from solution_logger import SolutionLogger
    
    start_time = time.time()
    
    # Read instance
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    network = instance['network']
    commodities_data = instance['commodities']
    
    num_nodes = network['num_nodes']
    nodes = network['nodes']
    arcs = network['arcs']
    num_arcs = network['num_arcs']
    commodity_list = commodities_data['commodity_list']
    num_commodities = commodities_data['num_commodities']
    
    # Build arc data structures
    arc_by_id = {}
    for arc in arcs:
        arc_by_id[arc['arc_id']] = arc
    
    # Build adjacency for path finding
    # outgoing[node] = list of (to_node, arc_id, capacity)
    outgoing = defaultdict(list)
    for arc in arcs:
        outgoing[arc['from_node']].append((arc['to_node'], arc['arc_id']))
    
    # Initialize logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # MIP formulation:
    # Binary variable x[k, a] = 1 if commodity k uses arc a
    # Binary variable r[k] = 1 if commodity k is rejected
    # Minimize sum of r[k] * artificial_arc_cost[k]
    # Subject to:
    # - Flow conservation for each commodity at each node
    # - Capacity constraints on each arc
    
    model = gp.Model("unsplittable_mcf")
    model.setParam('TimeLimit', max(1, time_limit - (time.time() - start_time) - 1))
    model.setParam('Threads', 1)
    model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
    
    # For each commodity, find reachable arcs via BFS/DFS to reduce variables
    # x[k][a] binary variables
    x = {}
    r = {}
    
    # Pre-compute which arcs are relevant for each commodity
    # (arcs on some path from origin to destination)
    def get_relevant_arcs(origin, destination):
        """Find arcs that could be on a path from origin to destination."""
        # Forward reachability from origin
        forward_reachable = set()
        stack = [origin]
        while stack:
            node = stack.pop()
            if node in forward_reachable:
                continue
            forward_reachable.add(node)
            for to_node, arc_id in outgoing[node]:
                if to_node not in forward_reachable:
                    stack.append(to_node)
        
        # Backward reachability from destination
        # Build reverse adjacency
        incoming_map = defaultdict(list)
        for arc in arcs:
            incoming_map[arc['to_node']].append((arc['from_node'], arc['arc_id']))
        
        backward_reachable = set()
        stack = [destination]
        while stack:
            node = stack.pop()
            if node in backward_reachable:
                continue
            backward_reachable.add(node)
            for from_node, arc_id in incoming_map[node]:
                if from_node not in backward_reachable:
                    stack.append(from_node)
        
        relevant_nodes = forward_reachable & backward_reachable
        relevant_arcs = []
        for arc in arcs:
            if arc['from_node'] in relevant_nodes and arc['to_node'] in relevant_nodes:
                relevant_arcs.append(arc['arc_id'])
        
        return relevant_arcs, relevant_nodes
    
    # Build incoming arcs per node
    incoming_arcs_by_node = defaultdict(list)  # node -> list of arc_ids
    outgoing_arcs_by_node = defaultdict(list)  # node -> list of arc_ids
    for arc in arcs:
        incoming_arcs_by_node[arc['to_node']].append(arc['arc_id'])
        outgoing_arcs_by_node[arc['from_node']].append(arc['arc_id'])
    
    commodity_relevant_arcs = {}
    commodity_relevant_nodes = {}
    
    for comm in commodity_list:
        k = comm['commodity_id']
        origin = comm['origin']
        destination = comm['destination']
        
        rel_arcs, rel_nodes = get_relevant_arcs(origin, destination)
        commodity_relevant_arcs[k] = set(rel_arcs)
        commodity_relevant_nodes[k] = rel_nodes
        
        # Create rejection variable
        r[k] = model.addVar(vtype=GRB.BINARY, name=f"r_{k}", obj=comm['artificial_arc_cost'])
        
        # Create arc variables for this commodity
        for a_id in rel_arcs:
            x[k, a_id] = model.addVar(vtype=GRB.BINARY, name=f"x_{k}_{a_id}")
    
    model.update()
    
    # Flow conservation constraints
    for comm in commodity_list:
        k = comm['commodity_id']
        origin = comm['origin']
        destination = comm['destination']
        rel_nodes = commodity_relevant_nodes[k]
        rel_arcs = commodity_relevant_arcs[k]
        
        for node in rel_nodes:
            # Compute flow balance
            out_flow = gp.LinExpr()
            in_flow = gp.LinExpr()
            
            for a_id in outgoing_arcs_by_node[node]:
                if a_id in rel_arcs:
                    out_flow += x[k, a_id]
            
            for a_id in incoming_arcs_by_node[node]:
                if a_id in rel_arcs:
                    in_flow += x[k, a_id]
            
            if node == origin and node == destination:
                # Self-loop commodity: just reject
                model.addConstr(out_flow - in_flow == 0, name=f"flow_{k}_{node}")
            elif node == origin:
                # out - in = 1 - r[k]
                model.addConstr(out_flow - in_flow == 1 - r[k], name=f"flow_{k}_{node}")
            elif node == destination:
                # in - out = 1 - r[k]
                model.addConstr(in_flow - out_flow == 1 - r[k], name=f"flow_{k}_{node}")
            else:
                # in = out
                model.addConstr(in_flow - out_flow == 0, name=f"flow_{k}_{node}")
    
    # Handle self-loop commodities (origin == destination): must reject or no routing needed
    for comm in commodity_list:
        k = comm['commodity_id']
        if comm['origin'] == comm['destination']:
            # Commodity is trivially routed (no arcs needed), not rejected
            model.addConstr(r[k] == 0, name=f"selfloop_{k}")
    
    # Capacity constraints on each physical arc
    for arc in arcs:
        a_id = arc['arc_id']
        cap = arc['capacity']
        
        flow_expr = gp.LinExpr()
        has_terms = False
        for comm in commodity_list:
            k = comm['commodity_id']
            if (k, a_id) in x:
                flow_expr += comm['demand'] * x[k, a_id]
                has_terms = True
        
        if has_terms:
            model.addConstr(flow_expr <= cap, name=f"cap_{a_id}")
    
    model.setParam('OutputFlag', 1)
    model.setAttr('ModelSense', GRB.MINIMIZE)
    
    # Callback for logging
    best_obj = [float('inf')]
    best_solution = [None]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0] - 1e-6:
                best_obj[0] = obj_val
                if logger:
                    logger.log(obj_val)
    
    model.optimize(callback)
    
    # Extract solution
    solution_commodities = []
    total_cost = 0.0
    
    if model.SolCount > 0:
        total_cost = model.ObjVal
        
        for comm in commodity_list:
            k = comm['commodity_id']
            
            rejected = r[k].X > 0.5
            
            path_arcs = []
            if not rejected:
                # Extract the path
                rel_arcs = commodity_relevant_arcs[k]
                used_arcs = []
                for a_id in rel_arcs:
                    if x[k, a_id].X > 0.5:
                        used_arcs.append(a_id)
                
                # Order the arcs into a path
                if used_arcs:
                    arc_from_node = {}
                    for a_id in used_arcs:
                        arc = arc_by_id[a_id]
                        arc_from_node[arc['from_node']] = a_id
                    
                    current = comm['origin']
                    while current != comm['destination']:
                        a_id = arc_from_node[current]
                        arc = arc_by_id[a_id]
                        path_arcs.append({
                            "from": arc['from_node'],
                            "to": arc['to_node'],
                            "arc_id": a_id
                        })
                        current = arc['to_node']
            
            solution_commodities.append({
                "commodity_id": k,
                "rejected": rejected,
                "path_arcs": path_arcs
            })
    else:
        # No solution found - reject everything
        for comm in commodity_list:
            total_cost += comm['artificial_arc_cost']
            solution_commodities.append({
                "commodity_id": comm['commodity_id'],
                "rejected": True,
                "path_arcs": []
            })
        if logger:
            logger.log(total_cost)
    
    solution = {
        "objective_value": total_cost,
        "commodities": solution_commodities
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)

if __name__ == '__main__':
    main()