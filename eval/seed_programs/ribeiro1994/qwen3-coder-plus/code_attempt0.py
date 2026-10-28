import argparse
import json
import math
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def solve_transit_scheduling(instance_path, time_limit, log_path=None):
    # Read the instance data
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    # Initialize logger if provided
    logger = None
    if log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(log_path, sense="minimize")
        
    n_trips = data['parameters']['n_trips']
    m_depots = data['parameters']['m_depots']
    vehicle_capacities = data['vehicle_capacities']
    
    # Dictionary for easy lookup
    depot_to_trip_cost_dict = {(item['depot'], item['trip']): item['cost'] 
                                for item in data['depot_to_trip_costs']}
    trip_to_depot_cost_dict = {(item['trip'], item['depot']): item['cost'] 
                                for item in data['trip_to_depot_costs']}
    compatible_arc_costs = {(item['from_trip'], item['to_trip']): item['cost'] 
                             for item in data['compatible_arcs']}
    
    # Model creation
    model = gp.Model('transit_scheduling')
    model.setParam('TimeLimit', time_limit)
    model.setParam('LogToConsole', 0)  # Suppress Gurobi output
    
    # Variables
    # x[i,j,k]: binary variable indicating if trip i -> j via depot k
    # y[k,i]: binary variable indicating if trip i starts from depot k
    # z[i,k]: binary variable indicating if trip i ends at depot k
    
    # Decision variables for trip-trip connections
    x_vars = {}
    for arc_data in data['compatible_arcs']:
        from_trip = arc_data['from_trip']
        to_trip = arc_data['to_trip']
        x_vars[(from_trip, to_trip)] = model.addVar(vtype=GRB.BINARY, name=f"x_{from_trip}_{to_trip}")
    
    # Decision variables for depot -> trip connections
    y_vars = {}
    for cost_data in data['depot_to_trip_costs']:
        depot = cost_data['depot']
        trip = cost_data['trip']
        y_vars[(depot, trip)] = model.addVar(vtype=GRB.BINARY, name=f"y_{depot}_{trip}")
    
    # Decision variables for trip -> depot connections
    z_vars = {}
    for cost_data in data['trip_to_depot_costs']:
        trip = cost_data['trip']
        depot = cost_data['depot']
        z_vars[(trip, depot)] = model.addVar(vtype=GRB.BINARY, name=f"z_{trip}_{depot}")
    
    # Variables to count number of vehicles used per depot
    v_vars = {}  # How many vehicles are used at each depot
    for d in range(1, m_depots + 1):
        max_vehicles = vehicle_capacities[d - 1]
        v_vars[d] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_vehicles, name=f"v_{d}")
    
    # Constraints
    # Every trip must be entered exactly once
    for i in range(1, n_trips + 1):
        entering_trips = gp.LinExpr()
        for j in range(1, n_trips + 1):
            if i != j and (j, i) in x_vars:
                entering_trips += x_vars[(j, i)]
        
        # Add departure from depot
        for k in range(1, m_depots + 1):
            if (k, i) in y_vars:
                entering_trips += y_vars[(k, i)]
                
        model.addConstr(entering_trips == 1, name=f"enter_trip_{i}")
    
    # Every trip must be left exactly once
    for i in range(1, n_trips + 1):
        leaving_trips = gp.LinExpr()
        for j in range(1, n_trips + 1):
            if i != j and (i, j) in x_vars:
                leaving_trips += x_vars[(i, j)]
        
        # Add return to depot
        for k in range(1, m_depots + 1):
            if (i, k) in z_vars:
                leaving_trips += z_vars[(i, k)]
                
        model.addConstr(leaving_trips == 1, name=f"leave_trip_{i}")

    # Flow conservation: trips leaving depot must equal trips returning to depot
    for k in range(1, m_depots + 1):
        leaving_from_depot = gp.LinExpr()
        for i in range(1, n_trips + 1):
            if (k, i) in y_vars:
                leaving_from_depot += y_vars[(k, i)]
        
        returning_to_depot = gp.LinExpr()
        for i in range(1, n_trips + 1):
            if (i, k) in z_vars:
                returning_to_depot += z_vars[(i, k)]
        
        model.addConstr(returning_to_depot == leaving_from_depot, name=f"depot_flow_{k}")
        
        # Vars v_vars[k] must be equal to the flow through depot k
        model.addConstr(v_vars[k] >= leaving_from_depot, name=f"vehicle_count_lb_{k}")
    
    # Vehicle availability constraints
    for k in range(1, m_depots + 1):
        leaving_from_depot = gp.LinExpr()
        for i in range(1, n_trips + 1):
            if (k, i) in y_vars:
                leaving_from_depot += y_vars[(k, i)]
        
        model.addConstr(leaving_from_depot <= vehicle_capacities[k - 1], 
                        name=f"depot_capacity_{k}")
    
    # Objective function
    obj_expr = gp.LinExpr()
    
    # Add costs for trip to trip transitions
    for (from_trip, to_trip), var in x_vars.items():
        if (from_trip, to_trip) in compatible_arc_costs:
            obj_expr.addTerms(compatible_arc_costs[(from_trip, to_trip)], var)
    
    # Add costs for depot to trip assignments
    for (depot, trip), var in y_vars.items():
        if (depot, trip) in depot_to_trip_cost_dict:
            obj_expr.addTerms(depot_to_trip_cost_dict[(depot, trip)], var)
    
    # Add costs for trip to depot assignments
    for (trip, depot), var in z_vars.items():
        if (trip, depot) in trip_to_depot_cost_dict:
            obj_expr.addTerms(trip_to_depot_cost_dict[(trip, depot)], var)
    
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    # Optimize
    model.optimize()
    
    # Extract the solution
    if model.SolCount > 0:
        objective_value = model.objVal
        
        # Reconstruct routes
        routes = []
        
        # Create adjacency list of the solution graph
        next_trip = {}
        prev_trip = {}
        trip_depot_out = {}  # Maps trip to depot it starts from
        trip_depot_in = {}   # Maps trip to depot it ends at
        
        for (from_trip, to_trip), var in x_vars.items():
            if var.x > 0.9:  # Using threshold for binary check
                next_trip[from_trip] = to_trip
                prev_trip[to_trip] = from_trip
        
        for (depot, trip), var in y_vars.items():
            if var.x > 0.9:
                # This trip starts directly after departing from depot
                if trip not in prev_trip:
                    # Found a start trip for a route
                    trip_depot_out[trip] = depot
        
        for (trip, depot), var in z_vars.items():
            if var.x > 0.9:
                # This trip ends before returning to depot
                if trip not in next_trip:
                    # Found an end trip for a route
                    trip_depot_in[trip] = depot
        
        # Identify start trips (those without a predecessor)
        all_trips = set(range(1, n_trips + 1))
        visited = set()
        
        for trip in all_trips:
            if trip not in prev_trip and trip not in visited:
                # Starting point of a new route
                current_trip = trip
                route_trips = []
                
                # Follow chain of trips
                while current_trip is not None:
                    route_trips.append(current_trip)
                    visited.add(current_trip)
                    
                    next_trip_id = next_trip.get(current_trip)
                    if next_trip_id is None:
                        # End of the route - should have a return to depot
                        break
                    else:
                        current_trip = next_trip_id
                
                # Get the depot that the route started from and ended at
                start_trip = route_trips[0]
                end_trip = route_trips[-1]
                
                if start_trip in trip_depot_out and end_trip in trip_depot_in:
                    depot_id = trip_depot_out[start_trip]
                    if trip_depot_in[end_trip] == depot_id:  # Ensure it returns to the same depot
                        routes.append({
                            "depot": depot_id,
                            "trips": route_trips
                        })
        
        # Count number of vehicles used (sum of vehicle vars)
        num_vehicles = int(sum(v_vars[d].x for d in v_vars.keys()))
        
        # Log solution objective value if improvement occurred
        if logger and model.status != GRB.INF_OR_UNBD:
            logger.log(objective_value)
        
        # Create final solution dict
        solution = {
            "objective_value": objective_value,
            "routes": routes,
            "num_vehicles": num_vehicles
        }
        
        return solution
    else:
        # Infeasible or no solution
        return None

def main():
    parser = argparse.ArgumentParser(description='Solve Transit Scheduling Problem')
    parser.add_argument('--instance_path', required=True, help='Path to instance JSON file')
    parser.add_argument('--solution_path', required=True, help='Path to output solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum allowed runtime in seconds')
    parser.add_argument('--log_path', type=str, help='Path to logging JSONL file (optional)')
    
    args = parser.parse_args()
    
    solution = solve_transit_scheduling(
        args.instance_path,
        args.time_limit,
        args.log_path
    )
    
    if solution is not None:
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
    else:
        print("No solution found!")

if __name__ == "__main__":
    main()