import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_network_flow(instance_data, time_limit, log_path):
    # Extract instance data
    num_nodes = instance_data['num_nodes']
    num_arcs = instance_data['num_arcs']
    num_commodities = instance_data['num_commodities']
    arcs = instance_data['arcs']  # [[from_node, to_node], ...]
    
    # Map edge to index
    arc_to_idx = {}
    for idx, (from_node, to_node) in enumerate(arcs):
        arc_to_idx[f"{from_node}_{to_node}"] = idx
        
    commoditiy_list = instance_data['commodities']
    fixed_costs = instance_data['fixed_costs']
    variable_costs = instance_data['variable_costs']  # dict of list of costs per commodity
    
    # Create model
    model = gp.Model("network_flow")
    model.Params.OutputFlag = 0  # Suppress Gurobi output
    model.setParam('TimeLimit', time_limit)
    
    # Variables:
    # y_ij binary indicating if arc (i,j) is opened
    y = model.addVars(num_arcs, vtype=GRB.BINARY, name='y')
    
    # x_ijk continuous: fraction of commodity k's demand routed on arc (i,j)
    x = model.addVars(num_arcs, num_commodities, lb=0.0, name='x')
    
    # Flow conservation constraints
    for k in range(num_commodities):
        origin = commoditiy_list[k]['origin']
        destination = commoditiy_list[k]['destination']
        
        for i in range(num_nodes):
            flow_in = gp.quicksum(
                x[arc_to_idx[f"{from_node}_{to_node}"], k] 
                for arc_idx, (from_node, to_node) in enumerate(arcs) 
                if to_node == i
            )
            
            flow_out = gp.quicksum(
                x[arc_to_idx[f"{from_node}_{to_node}"], k] 
                for arc_idx, (from_node, to_node) in enumerate(arcs) 
                if from_node == i
            )
            
            if i == origin:
                rhs = 1  # produce one unit of flow
            elif i == destination:
                rhs = -1  # consume one unit of flow
            else:
                rhs = 0  # no production or consumption
            
            model.addConstr(flow_in - flow_out == rhs, f'flow_cons_{k}_{i}')
    
    # If a commodity sends flow on an arc, then the arc must be opened
    for arc_idx, (from_node, to_node) in enumerate(arcs):
        for k in range(num_commodities):
            # x[arc_idx, k] <= y[arc_idx]
            model.addConstr(x[arc_idx, k] <= y[arc_idx], f'flow_through_opened_{arc_idx}_{k}')
    
    # Objective function
    # Fixed + Variable costs
    fixed_cost_expr = gp.quicksum(fixed_costs[f"{from_node}_{to_node}"] * y[arc_idx] 
                                  for arc_idx, (from_node, to_node) in enumerate(arcs))
    
    var_cost_expr = gp.LinExpr()
    for arc_idx, (from_node, to_node) in enumerate(arcs):
        arc_str = f"{from_node}_{to_node}"
        for k in range(num_commodities):
            demand = commoditiy_list[k]['demand']
            cost_per_unit = variable_costs[arc_str][k]
            var_cost_expr.add(x[arc_idx, k] * demand * cost_per_unit)
          
    model.setObjective(fixed_cost_expr + var_cost_expr, GRB.MINIMIZE)
    
    # Initialize solution logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    # Optimize
    solution_found = False
    best_objective = float('inf')
    
    def callback(model, where):
        nonlocal solution_found, best_objective
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger and obj < best_objective:
                best_objective = obj
                logger.log(obj)
    
    model.optimize(callback)

    # Prepare solution
    if model.SolCount > 0:
        # Extract the solution
        final_solution = {
            "objective_value": model.ObjVal,
            "open_arcs": {},
            "routings": {}
        }

        # Get open arcs
        for arc_idx, (from_node, to_node) in enumerate(arcs):
            arc_str = f"{from_node}_{to_node}"
            if y[arc_idx].X > 0.5:
                final_solution["open_arcs"][arc_str] = 1
            else:
                final_solution["open_arcs"][arc_str] = 0
                
        # Get routings
        for k in range(num_commodities):
            k_dict = {}
            for arc_idx, (from_node, to_node) in enumerate(arcs):
                arc_str = f"{from_node}_{to_node}"
                flow_frac = x[arc_idx, k].X
                if abs(flow_frac) > 1e-8:  # Only include meaningful flows
                    k_dict[arc_str] = flow_frac
            final_solution["routings"][str(k)] = k_dict

    else:
        # If no solution was found within time limit
        if hasattr(model, 'ObjBound') and model.ObjBound != float('inf'):
            obj_val = model.ObjBound
        else:
            # Approximate best known or return default
            obj_val = float('inf')

        final_solution = {
            "objective_value": model.ObjVal if model.solCount > 0 else float('inf'),
            "open_arcs": {},
            "routings": {}
        }

        # For the case when a feasible solution exists but not accessed, fill it
        if model.Solcount > 0:
            final_solution["objective_value"] = model.ObjVal
            for arc_idx, (from_node, to_node) in enumerate(arcs):
                arc_str = f"{from_node}_{to_node}"
                if y[arc_idx].X > 0.5:
                    final_solution["open_arcs"][arc_str] = 1
                else:
                    final_solution["open_arcs"][arc_str] = 0
                    
            for k in range(num_commodities):
                k_dict = {}
                for arc_idx, (from_node, to_node) in enumerate(arcs):
                    arc_str = f"{from_node}_{to_node}"
                    flow_frac = x[arc_idx, k].X
                    if abs(flow_frac) > 1e-9:
                        k_dict[arc_str] = flow_frac
                final_solution["routings"][str(k)] = k_dict
        else:
            # Default empty solution when nothing is returned
            for arc in arcs:
                arc_str = f"{arc[0]}_{arc[1]}"
                final_solution["open_arcs"][arc_str] = 0
            for k in range(num_commodities):
                final_solution["routings"][str(k)] = {}

    # Ensure final objective is set correctly in case no solution was found
    if model.SolCount > 0:
        final_solution["objective_value"] = model.ObjVal
    else:
        # Fallback to bound if available
        final_solution["objective_value"] = model.ObjVal if model.Status == GRB.OPTIMAL else float('inf')

    return final_solution

def main():
    parser = argparse.ArgumentParser(description='Solve Network Flow Optimization Problem')
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, required=False,
                        help='Path to the log file for intermediate solutions')
    
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    solution = solve_network_flow(instance_data, args.time_limit, args.log_path)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()