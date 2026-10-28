import argparse
import json
import math
import sys

# Try importing gurobipy first, and if it fails, implement a fallback
try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False
    # Define placeholder classes to avoid immediate failure
    class DummyModel:
        def __init__(self):
            pass
        
        def addVar(self, **kwargs):
            return DummyVar()
        
        def addVars(self, *args, **kwargs):
            n = args[0]
            result = {}
            for i in range(n):
                result[i] = DummyVar()
            return result
            
        def setObjective(self, obj, sense):
            pass
            
        def addConstr(self, constr, name=None):
            pass
            
        def optimize(self):
            pass
            
    class DummyVar:
        X = 0
        Obj = 0
    
    class DummyGRB:
        BINARY = "B"
        MINIMIZE = "MIN"
        OPTIMAL = 2
        TIME_LIMIT = 9
        SUBOPTIMAL = 6
    
    gp = DummyModel()
    GRB = DummyGRB()


def solve_with_gurobi(instance_data, time_limit, logger=None):
    """Solve the hub location and routing problem using a MIP formulation in Gurobi."""
    # Extract instance data
    num_nodes = instance_data['num_nodes']
    num_commodities = instance_data['num_commodities']
    tau = instance_data['parameters']['tau'] 
    chi = instance_data['parameters'].get('chi', 1.0)
    delta = instance_data['parameters'].get('delta', 1.0)
    TC = instance_data['parameters'].get('TC', 1.0)  # transportation scaling factor
    
    nodes = instance_data['nodes']
    distances = instance_data['distances']
    commodities = instance_data['commodities']
    
    fixed_costs = [node['fixed_cost'] for node in nodes]
    
    # Create Gurobi model
    model = gp.Model("hub_and_spoke")
    model.setParam("Threads", 1)
    model.setParam(GRB.Param.TimeLimit, time_limit)
    model.setParam(GRB.Param.LogToConsole, 0)  # Disable default logging
    
    # Binary variable: hub opened at node i
    y = model.addVars(num_nodes, vtype=GRB.BINARY, name="y")
    
    # Variables for commodity assignments: z_{k,i,j} for commodity k routed via unordered pair i, j
    z = {}
    costs = {}  # Stores calculated cost for each assignment
    hub_pairs_for_commodity = {}

    for k in range(num_commodities):
        z[k] = {}
        costs[k] = {}
        hub_pairs_for_commodity[k] = []
        commodity = commodities[k]
        origin = commodity['origin']
        dest = commodity['destination']
        flow = commodity['flow']
        
        # Enumerate possible hub pairs (i, j) with i <= j for symmetry
        for i in range(num_nodes):
            for j in range(i, num_nodes):  # ensures i <= j
                if i == j:
                    # Single hub: origin -> i -> dest
                    cost = flow * (chi * distances[origin][i] + delta * distances[i][dest])
                else:
                    # Two hubs: origin -> i -> j -> dest
                    cost = flow * (chi * distances[origin][i] + tau * distances[i][j] + delta * distances[j][dest])
                
                z[k][(i,j)] = model.addVar(vtype=GRB.BINARY, name=f"z_{k}_{i}_{j}")
                costs[k][(i,j)] = cost
                hub_pairs_for_commodity[k].append((i, j))

    # Objective: minimize sum of hub opening costs and transportation costs
    total_hub_cost = gp.quicksum(fixed_costs[i] * y[i] for i in range(num_nodes))
    
    total_transport_cost = 0
    for k in range(num_commodities):
        total_transport_cost_expr = gp.LinExpr()
        for (i, j) in hub_pairs_for_commodity[k]:
            total_transport_cost_expr.addTerms(costs[k][(i,j)], z[k][(i,j)])
        total_transport_cost = total_transport_cost_expr
    
    # Note: in Gurobi, we can add expressions directly to the objective function
    model.setObjective(total_hub_cost + total_transport_cost, GRB.MINIMIZE)
    
    # Constraints
    # 1. Each commodity must be assigned to exactly one hub pair
    for k in range(num_commodities):
        model.addConstr(gp.quicksum(z[k][pair] for pair in hub_pairs_for_commodity[k]) == 1, 
                       f"assign_comp_{k}")

    # 2. Hubs must be open if used in an assignment
    for k in range(num_commodities):
        for (i, j) in hub_pairs_for_commodity[k]:
            # If commodity k is routed through hub i, then hub i must be open
            model.addConstr(z[k][(i,j)] <= y[i], f"req_hub_i_for_comp_{k}_{i}_{j}")
            if i != j:  # Only constrain the second hub if it's different
                model.addConstr(z[k][(i,j)] <= y[j], f"req_hub_j_for_comp_{k}_{i}_{j}")

    # Optimize
    model.optimize()
    
    # Process results
    obj_val = model.ObjVal

    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and model.SolCount > 0:
        open_hubs = [i for i in range(num_nodes) if y[i].X > 0.5]
        
        assignments = []
        for k in range(num_commodities):
            commodity = commodities[k]
            origin = commodity['origin']
            dest = commodity['destination']
            flow = commodity['flow']
            
            # Find the assigned hub pair
            assigned_pair = None
            for (i, j) in hub_pairs_for_commodity[k]:
                if z[k][(i,j)].X > 0.99:  # Avoid numerical issues
                    assigned_pair = (i, j)
                    break
            
            if assigned_pair:
                i, j = assigned_pair
                hub_edge = [i, j] if i != j else [i]
                
                # Calculate cost based on assignment
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
                    "hub_edge": hub_edge,
                    "cost": path_cost
                })
        
        return {
            "objective_value": obj_val,
            "open_hubs": open_hubs,
            "assignments": assignments
        }
    
    else:
        # Return any feasible base solution
        if model.SolCount > 0:
            # At least one solution exists, even if suboptimal
            best_known_obj = model.ObjVal
        else:
            # Fallback to heuristic if no solution found within time limit
            return solve_with_heuristic(instance_data, time_limit, logger)
        
        # Since there's no solution available, return heuristic result
        return solve_with_heuristic(instance_data, time_limit, logger)


def solve_with_heuristic(instance_data, time_limit, logger=None):
    """Fallback heuristic solver if needed."""
    from datetime import datetime, timedelta
    import random
    
    start_time = datetime.now()
    
    num_nodes = instance_data['num_nodes']
    num_commodities = instance_data['num_commodities']
    tau = instance_data['parameters']['tau'] 
    chi = instance_data['parameters'].get('chi', 1.0)
    delta = instance_data['parameters'].get('delta', 1.0)
    
    nodes = instance_data['nodes']
    distances = instance_data['distances']
    commodities = instance_data['commodities']
    
    fixed_costs = [node['fixed_cost'] for node in nodes]
    

    # Greedy construction method
    # Start with empty hub set
    hubs_opened = set()
    assignments = []

    # Sort commodities by flow volume in descending order (prioritize high flow routes)
    sorted_comms = sorted(enumerate(commodities), key=lambda x: x[1]['flow'], reverse=True)

    # For each commodity, find the best assignment given currently opened hubs,
    # potentially adding new hubs if it reduces total cost
    for orig_k, commodity in sorted_comms:
        origin = commodity['origin']
        dest = commodity['destination']
        flow = commodity['flow']

        # Option 1: Route through existing hubs
        best_single_hub_cost = float('inf')
        best_single_hub = None
        for h in hubs_opened:
            cost = flow * (chi * distances[origin][h] + delta * distances[h][dest])
            if cost < best_single_hub_cost:
                best_single_hub_cost = cost
                best_single_hub = h
                
        best_two_hub_cost = float('inf')
        best_two_hub_pair = None
        for h1 in hubs_opened:
            for h2 in hubs_opened:
                if h1 == h2:
                    continue
                cost = flow * (chi * distances[origin][h1] + tau * distances[h1][h2] + delta * distances[h2][dest])
                if cost < best_two_hub_cost:
                    best_two_hub_cost = cost
                    best_two_hub_pair = (h1, h2)

        # Option 2: Try opening new hubs
        new_single_hub_cost = float('inf')
        new_single_best_hub = None
        for h_new in range(num_nodes):
            cost = flow * (chi * distances[origin][h_new] + delta * distances[h_new][dest])
            if cost < new_single_hub_cost:
                new_single_hub_cost = cost
                new_single_best_hub = h_new
        
        # Try all pairs of new hubs
        new_double_hub_cost = float('inf')
        new_double_hub_pair = None
        for h1_idx in range(num_nodes):
            for h2_idx in range(h1_idx+1, num_nodes):  # Different hubs
                h1 = h1_idx
                h2 = h2_idx
                base_cost = flow * (chi * distances[origin][h1] + tau * distances[h1][h2] + delta * distances[h2][dest])
                
                # Determine how many of these hubs are not yet open
                hubs_to_add = []
                for hub in [h1, h2]:
                    if hub not in hubs_opened:
                        hubs_to_add.append(fixed_costs[hub])
                
                if len(hubs_to_add) == 0:  # Both already open - this is like existing hubs option
                    adjusted_cost = base_cost
                elif len(hubs_to_add) == 1:  # One new hub
                    adjusted_cost = base_cost + hubs_to_add[0]
                else:  # 2 new hubs
                    adjusted_cost = base_cost + sum(hubs_to_add)
                
                if adjusted_cost < new_double_hub_cost:
                    new_double_hub_cost = adjusted_cost
                    new_double_hub_pair = (h1, h2)
        
        # Determine if new hubs need to be added based on overall improvement
        # Actually just pick the best assignment option for now
        # Compare best_single_hub_cost, best_two_hub_cost, new_single_hub_cost with corresponding fixed additions
        
        # For existing hubs scenario, we consider the base routing costs
        existing_best_cost = min(best_single_hub_cost, best_two_hub_cost)
        
        # For new scenario, we calculate actual total increase to global objective
        # New single: add fixed cost of hub if not there, then add routing cost
        new_single_actual = new_single_hub_cost
        if new_single_best_hub not in hubs_opened:
            new_single_actual += fixed_costs[new_single_best_hub] 
            
        new_double_actual = new_double_hub_cost  # Already includes fixed costs if we account for them later
        
        # Pick best among available options (we'll recompute accurately if a new hub is selected)
        # For simplicity, let's choose the route that minimizes local cost impact
        route_single_hub = None
        route_cost = float('inf')
        route_type = None  # 'single' or 'double'
        
        if best_single_hub is not None:
            route_single_hub = [best_single_hub]
            route_cost = best_single_hub_cost
            route_type = 'single exist'
        if best_two_hub_pair is not None and best_two_hub_cost < route_cost:
            route_single_hub = [best_two_hub_pair[0], best_two_hub_pair[1]]
            route_cost = best_two_hub_cost
            route_type = 'double exist'
        if new_single_best_hub is not None and new_single_actual < route_cost:
            # Add new hub if needed
            if new_single_best_hub not in hubs_opened:
                hubs_opened.add(new_single_best_hub)
            route_single_hub = [new_single_best_hub]
            route_cost = flow * (chi * distances[origin][new_single_best_hub] + delta * distances[new_single_best_hub][dest])
            route_type = 'single new'
        if new_double_hub_pair is not None and new_double_actual < route_cost:
            # Maybe add the needed hubs - but actually we should just evaluate and see what happens
            # This is getting complex with the fixed costs so let's use simpler logic
            # Reconsider opening only the most impactful hubs first
            hubs_needed = []
            for h in [new_double_hub_pair[0], new_double_hub_pair[1]]:
                if h not in hubs_opened:
                    hubs_needed.append(h)
            
            # Add the needed hubs first
            for h in hubs_needed:
                hubs_opened.add(h)
            
            route_single_hub = [new_double_hub_pair[0], new_double_hub_pair[1]]
            route_cost = flow * (
                   chi * distances[origin][new_double_hub_pair[0]] + 
                   tau * distances[new_double_hub_pair[0]][new_double_hub_pair[1]] + 
                   delta * distances[new_double_hub_pair[1]][dest]
            )
            route_type = 'double new'

        hub_edge = route_single_hub
        
        assignments.append({
            "commodity": orig_k,
            "origin": origin,
            "destination": dest,
            "hub_edge": hub_edge,
            "cost": route_cost
        })

    # Calculate final objective value
    # Opening cost
    total_fix = sum(fixed_costs[i] for i in hubs_opened)
    # Transportation cost
    total_trans = sum(ass['cost'] for ass in assignments)
    final_obj = total_fix + total_trans
    
    return {
        "objective_value": final_obj,
        "open_hubs": list(hubs_opened),
        "assignments": assignments
    }


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

    if HAS_GUROBI:
        # Use Gurobi solver
        try:
            solution = solve_with_gurobi(instance_data, args.time_limit, logger)
        except Exception as e:
            # Fallback to heuristic in case of error
            solution = solve_with_heuristic(instance_data, args.time_limit, logger)
    else:
        # Use heuristic fallback
        solution = solve_with_heuristic(instance_data, args.time_limit, logger)

    if logger:
        logger.log(solution["objective_value"])

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()