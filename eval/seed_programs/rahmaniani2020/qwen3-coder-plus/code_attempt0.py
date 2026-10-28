import argparse
import json
import math
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def compute_routing_costs(facility_coords, customer_coords):
    """Compute Euclidean distances / routing costs between facilities and customers."""
    num_facilities = len(facility_coords[0])
    num_customers = len(customer_coords[0])
    
    costs = []
    for i in range(num_facilities):
        row = []
        for j in range(num_customers):
            dist = math.sqrt(
                (facility_coords[0][i] - customer_coords[0][j])**2 +
                (facility_coords[1][i] - customer_coords[1][j])**2
            )
            row.append(dist)
        costs.append(row)
    
    return costs


def solve_stochastic_facility_location(instance_data, time_limit, log_path):
    # Extract data
    num_facilities = instance_data["num_facilities"]
    num_customers = instance_data["num_customers"]
    num_scenarios = instance_data["num_scenarios"]
    
    facilities = instance_data["facilities"]
    customers = instance_data["customers"]
    scenarios = instance_data["scenarios"]
    
    fixed_costs = facilities["fixed_costs"]
    capacities = facilities["capacities"]
    
    # Use provided routing_costs unless we need to compute them
    routing_costs = instance_data.get("routing_costs")
    if routing_costs is None:
        facility_coords = (facilities["x_coordinates"], facilities["y_coordinates"])
        customer_coords = (customers["x_coordinates"], customers["y_coordinates"])
        routing_costs = compute_routing_costs(facility_coords, customer_coords)
    
    probabilities = scenarios["probabilities"]
    demands = scenarios["demands"]
    
    # Create model
    model = gp.Model("stochastic_facility_location")
    
    # Suppress output if not needed (but may want to see progress)
    model.setParam('OutputFlag', 1)
    model.setParam('TimeLimit', time_limit)
    
    # Define variables
    # y[i] - binary, whether facility i is opened
    y = model.addVars(num_facilities, vtype=GRB.BINARY, name="open_facility")
    
    # x[i,j,k] - continuous, amount shipped from facility i to customer j in scenario k
    x = {}
    for k in range(num_scenarios):
        for i in range(num_facilities):
            for j in range(num_customers):
                x[i, j, k] = model.addVar(lb=0.0, name=f"flow_{i}_{j}_{k}")
    
    # Set objective
    fixed_cost_expr = gp.quicksum(fixed_costs[i] * y[i] for i in range(num_facilities))
    routing_cost_expr = 0
    
    for k in range(num_scilities):
        prob = probabilities[k]
        scenario_routing_cost = gp.quicksum(
            routing_costs[i][j] * x[i, j, k] 
            for i in range(num_facilities) 
            for j in range(num_customers)
        )
        routing_cost_expr += prob * scenario_routing_cost
    
    model.setObjective(fixed_cost_expr + routing_cost_expr, GRB.MINIMIZE)
    
    # Add constraints
    # Demand constraint: for each scenario and each customer, total flow must meet demand
    for k in range(num_scenarios):
        for j in range(num_customers):
            model.addConstr(
                gp.quicksum(x[i, j, k] for i in range(num_facilities)) >= demands[k][j],
                name=f"demand_s{k}_c{j}"
            )
    
    # Capacity constraint: for each scenario and each facility, total outflow <= capacity if open
    for k in range(num_scenarios):
        for i in range(num_facilities):
            model.addConstr(
                gp.quicksum(x[i, j, k] for j in range(num_customers)) <= capacities[i] * y[i],
                name=f"capacity_s{k}_f{i}"
            )
    
    # Overall capacity requirement: total capacity of open facilities >= max total demand in any scenario
    max_total_demand = max(sum(demands[s]) for s in range(num_scenarios))
    model.addConstr(
        gp.quicksum(capacities[i] * y[i] for i in range(num_facilities)) >= max_total_demand,
        name="total_capacity_requirement"
    )
    
    # Set up logger for incumbent solutions
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Optimization callback to log solutions
    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)
    
    # Optimize the model
    model.optimize(mycallback)
    
    # Get results
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        y_values = [int(y[i].X) for i in range(num_facilities)]
        
        # Return solution according to the specified schema
        solution = {
            "objective_value": model.objVal,
            "y": y_values
        }
        
        return solution
    else:
        # Handle other statuses - return empty or error solution if infeasible
        raise Exception(f"Optimization ended with status {model.status}")


def main():
    parser = argparse.ArgumentParser(description='Solve Stochastic Facility Location Problem')
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON instance file')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path where the solution JSON file will be written')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, default=None,
                        help='Path to log the intermediate solutions (optional)')
    
    args = parser.parse_args()
    
    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Solve the problem
    solution = solve_stochastic_facility_location(instance_data, args.time_limit, args.log_path)
    
    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()