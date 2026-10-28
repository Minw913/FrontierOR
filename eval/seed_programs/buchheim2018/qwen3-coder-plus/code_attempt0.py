import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_logistics_planner(instance_path, solution_path, time_limit, log_path):
    # Read instance data
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    grid_size_k = data['grid_size_k']
    num_nodes = data['num_nodes']
    num_arcs = data['num_arcs']
    source_node = data['source_node']
    target_node = data['target_node']
    nodes = data['nodes']
    arcs = data['arcs']
    linear_costs = data['linear_costs']
    quadratic_costs = data['quadratic_costs']

    # Create Gurobi model
    model = gp.Model('LogisticsPlanner')
    model.setParam('TimeLimit', time_limit)
    model.setParam('NonConvex', 2)  # Needed for quadratic terms
    
    # Variables - binary variables for which arcs to use
    x = model.addVars(num_arcs, vtype=GRB.BINARY, name='x')
    
    # Flow conservation constraints
    # For each node except source and target
    for i, node_data in enumerate(nodes):
        node_id = node_data['id']
        
        if node_id == source_node:
            continue
        if node_id == target_node:
            continue
            
        # Incoming arcs
        incoming_arc_indices = []
        for j, arc in enumerate(arcs):
            if arc['to_node'] == node_id:
                incoming_arc_indices.append(j)
        
        # Outgoing arcs
        outgoing_arc_indices = []
        for j, arc in enumerate(arcs):
            if arc['from_node'] == node_id:
                outgoing_arc_indices.append(j)
        
        # Flow conservation: incoming_flow = outgoing_flow
        model.addConstr(
            gp.quicksum(x[j] for j in incoming_arc_indices) == 
            gp.quicksum(x[j] for j in outgoing_arc_indices),
            name=f'flow_conservation_{node_id}'
        )
    
    # Exactly one arc leaves the source
    source_outgoing_indices = []
    for j, arc in enumerate(arcs):
        if arc['from_node'] == source_node:
            source_outgoing_indices.append(j)
            
    model.addConstr(gp.quicksum(x[j] for j in source_outgoing_indices) == 1, name="source_leaving")
    
    # Exactly one arc enters the target
    target_incoming_indices = []
    for j, arc in enumerate(arcs):
        if arc['to_node'] == target_node:
            target_incoming_indices.append(j)
            
    model.addConstr(gp.quicksum(x[j] for j in target_incoming_indices) == 1, name="target_entering")
    
    # Objective function
    # Linear term: sum of linear costs over selected arcs
    linear_part = gp.quicksum(linear_costs[i] * x[i] for i in range(num_arcs))
    
    # Quadratic term: sum of quadratic costs for all pairs of selected arcs
    quadratic_part = gp.quicksum(
        quadratic_costs[i][j] * x[i] * x[j]
        for i in range(num_arcs)
        for j in range(num_arcs)
    )
    
    total_cost = linear_part + quadratic_part
    
    model.setObjective(total_cost, GRB.MINIMIZE)
    
    # Log best solution callback
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            if logger:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj_val)

    model.optimize(callback)
    
    # Extract the solution
    solution_arcs = []
    for i in range(num_arcs):
        if x[i].X > 0.5:  # Selected arcs
            arc_info = {
                "id": arcs[i]['id'],
                "from_node": arcs[i]['from_node'],
                "to_node": arcs[i]['to_node']
            }
            solution_arcs.append(arc_info)
    
    # Calculate objective value
    selected_arc_ids = [arc['id'] for arc in solution_arcs]
    selected_arc_indices = [i for i in range(num_arcs) if x[i].X > 0.5]
    
    # Compute objective value explicitly to ensure precision
    linear_sum = sum(linear_costs[i] for i in selected_arc_indices)
    
    quadratic_sum = 0
    for i in selected_arc_indices:
        for j in selected_arc_indices:
            quadratic_sum += quadratic_costs[i][j]
    
    objective_value = linear_sum + quadratic_sum

    # Create solution output
    solution_output = {
        "objective_value": float(objective_value),
        "solution_arcs": solution_arcs
    }
    
    # Write solution to file
    with open(solution_path, 'w') as f:
        json.dump(solution_output, f)
    
    return solution_output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solve Logistics Planner Problem')
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON file containing the problem instance')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path where the final solution JSON file should be written')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, default=None,
                        help='Path to a JSONL file for logging intermediate solutions')
    
    args = parser.parse_args()
    
    solve_logistics_planner(
        args.instance_path,
        args.solution_path,
        args.time_limit,
        args.log_path
    )