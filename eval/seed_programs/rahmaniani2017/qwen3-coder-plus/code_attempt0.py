import argparse
import json
import math
import gurobipy as gp
from gurobipy import GRB

def solve_hub_location_routing(instance_data, time_limit):
    """Solve the hub location and routing problem using a MIP formulation."""
    
    # Extract instance data
    num_nodes = instance_data['num_nodes']
    num_commodities = instance_data['num_commodities']
    tau = instance_data['parameters']['tau'] if 'tau' in instance_data['parameters'] else 0.8
    # In the example provided, chi and delta both equal 1 (symmetric case)
    chi = instance_data['parameters'].get('chi', 1.0)
    delta = instance_data['parameters'].get('delta', 1.0)
    
    nodes = instance_data['nodes']
    distances = instance_data['distances']
    commodities = instance_data['commodities']
    
    # Build map of commodity -> index for tracking
    # Each commodity has origin-destination-flow
    
    fixed_costs = [node['fixed_cost'] for node in nodes]
    
    # Create a mapping to handle same-origin/destination differently if needed
    # But for this version we assume each commodity is uniquely indexed based on order
    
    # Decision Variables Setup
    # y_i: binary variable indicating if hub at node i is opened
    # x_{odij}: continuous variable indicating fraction of flow od shipped via hubs i,j
    
    # Create Gurobi model
    model = gp.Model("hub_and_spoke")
    model.setParam(GRB.Param.TimeLimit, time_limit)
    model.setParam(GRB.Param.LogToConsole, 0)  # Disable default logging
    
    # Binary variable: hub opened at node i
    y = model.addVars(num_nodes, vtype=GRB.BINARY, name="y")
    
    # Variable for commodity assignments: x[commodity_idx][(i,j)] 
    # represents flow assignment to hub pair (i,j)
    x = {}
    for k in range(num_commodities):
        x[k] = {}
        commodity = commodities[k]
        origin = commodity['origin']
        dest = commodity['destination']
        flow = commodity['flow']
        
        # For each commodity, we enumerate possible hub pairs (i, j) with i <= j
        # For single hub, we have i == j
        for i in range(num_nodes):
            for j in range(i, num_nodes):  # Since distance matrix is symmetric and problem is symmetric, i <= j is sufficient
                # Cost calculation for routing commodity k via hubs (i, j)
                # Path: origin -> i -> j -> dest
                # Cost = flow * [chi*distance(origin,i) + tau*distance(i,j) + delta*distance(j,dest)]
                
                if i == j:
                    # Single hub case
                    hub_cost = flow * (chi * distances[origin][i] + delta * distances[i][dest])
                else:
                    # Two hubs case
                    hub_cost = flow * (
                        chi * distances[origin][i] + 
                        tau * distances[i][j] + 
                        delta * distances[j][dest]
                    )
                
                x[k][(i,j)] = model.addVar(lb=0.0, ub=1.0, obj=hub_cost, name=f"x_{k}_{i}_{j}")

    # Constraints

    # Constraint: Each commodity must be assigned to exactly one hub pair
    for k in range(num_commodities):
        model.addConstr(gp.quicksum(x[k][pair] for pair in x[k]) == 1, f"assign_comp_{k}")

    # Constraint: Hub must be open if used for a commodity
    # If commodity k uses any path with hub i, hub i must be open
    for k in range(num_commodities):
        for i in range(num_nodes):
            # Sum over all pairs that include hub i
            hub_pairs_using_i = []
            for pair in x[k]:
                if i in pair:
                    hub_pairs_using_i.append(x[k][pair])
            
            # If any assignment in this commodity involves hub i, y[i] must be 1,
            # and the total flow using hub i cannot exceed 1 if y[i]=1
            model.addConstr(gp.quicksum(hub_pairs_using_i) <= y[i], f"use_hub_ik_{k}_{i}")

    # Set hub opening costs as additional terms in objective function
    hub_opening_cost_var = model.addVar(obj=1.0, lb=0.0, name="hub_setup_cost")
    model.addConstr(
        hub_opening_cost_var == gp.quicksum(fixed_costs[i] * y[i] for i in range(num_nodes)),
        "hub_setup_def"
    )

    model.optimize()
    
    # Process the result
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
        open_hubs = [i for i in range(num_nodes) if y[i].X > 0.5]
        
        assignments = []
        for k in range(num_commodities):
            commodity = commodities[k]
            origin = commodity['origin']
            dest = commodity['destination']
            flow = commodity['flow']
            
            # Find assignments with non-zero values
            for (i, j), var in x[k].items():
                if var.X > 0.001:  # Threshold for considering assignment active
                    hub_edge = [i, j] if i != j else [i]  # Represent as list
                    # Calculate actual cost for this route:
                    if i == j:
                        path_cost = flow * (chi * distances[origin][i] + delta * distances[i][dest])
                    else:
                        path_cost = flow * (
                            chi * distances[origin][i] + 
                            tau * distances[i][j] + 
                            delta * distances[j][dest]
                        )
                    
                    assignments.append({
                        "commodity": k,
                        "origin": origin,
                        "destination": dest,
                        "flow": flow,
                        "hub_edge": hub_edge,
                        "cost": path_cost
                    })
                    break  # Since each commodity is assigned exactly once, break after finding assignment
        
        # The total cost includes routing and hub opening costs
        objective_value = model.ObjVal
        return {
            "objective_value": objective_value,
            "open_hubs": open_hubs,
            "assignments": assignments
        }

    else:
        raise Exception("Model did not find a solution. Status: {}".format(model.Status))

def main():
    parser = argparse.ArgumentParser(description='Solve hub location routing problem.')
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON instance file')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path to save the JSON solution')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Time limit for solving in seconds')
    parser.add_argument('--log_path', type=str, required=False,
                        help='Path to save logs (optional)')
    
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Initialize logger if path provided
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None

    # Solve instance
    solution = solve_hub_location_routing(instance_data, args.time_limit)
    
    # Log final objective if using logger
    if logger:
        logger.log(solution["objective_value"])

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()