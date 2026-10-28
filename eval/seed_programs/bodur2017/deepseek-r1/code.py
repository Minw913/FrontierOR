import argparse
import json
import math
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser(description='Solve Facility Location Problem with Demand Uncertainty')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the JSON instance file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log intermediate solutions (JSONL)')
    args = parser.parse_args()

    # Initialize solution logger if log_path is provided
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Load instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    facilities = data['facilities']
    customers = data['customers']
    scenarios = data['scenarios']
    num_facilities = data['num_facilities']
    num_customers = data['num_customers']
    num_scenarios = data['num_scenarios']
    transportation_costs = data['transportation_costs']
    
    # Precompute maximum total demand across scenarios
    max_total_demand = 0
    for scenario in scenarios:
        total_demand = sum(scenario['demands'])
        if total_demand > max_total_demand:
            max_total_demand = total_demand
    
    # Check feasibility: total available capacity must be >= max_total_demand
    total_capacity = sum(f['capacity'] for f in facilities)
    if total_capacity < max_total_demand:
        # Infeasible instance: output empty solution
        solution_dict = {
            "objective_value": 0.0,
            "open_facilities": [],
            "x": {str(fac['id']): 0 for fac in facilities},
            "y": {}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution_dict, f)
        if logger:
            logger.log_solution(0.0, solution_dict)
        return

    # Create Gurobi model
    model = gp.Model()
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)

    # Add decision variables
    x = model.addVars(num_facilities, vtype=GRB.BINARY, name="x")
    y = model.addVars(num_facilities, num_customers, num_scenarios, vtype=GRB.CONTINUOUS, name="y")

    # Set objective: minimize total expected cost
    fixed_cost = gp.quicksum(facilities[i]['opening_cost'] * x[i] for i in range(num_facilities))
    trans_cost = gp.quicksum(
        scenarios[s]['probability'] * transportation_costs[i][j] * y[i, j, s]
        for i in range(num_facilities)
        for j in range(num_customers)
        for s in range(num_scenarios)
    )
    model.setObjective(fixed_cost + trans_cost, GRB.MINIMIZE)

    # Add constraints
    # Aggregate capacity constraint
    model.addConstr(
        gp.quicksum(facilities[i]['capacity'] * x[i] for i in range(num_facilities)) >= max_total_demand,
        "agg_capacity"
    )

    # Demand satisfaction constraints
    for s in range(num_scenarios):
        for j in range(num_customers):
            model.addConstr(
                gp.quicksum(y[i, j, s] for i in range(num_facilities)) >= scenarios[s]['demands'][j],
                f"demand_s{s}_c{j}"
            )

    # Capacity constraints per facility and scenario
    for i in range(num_facilities):
        cap_i = facilities[i]['capacity']
        for s in range(num_scenarios):
            model.addConstr(
                gp.quicksum(y[i, j, s] for j in range(num_customers)) <= cap_i * x[i],
                f"cap_f{i}_s{s}"
            )

    # Callback function for logging intermediate solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    # Optimize model
    if logger:
        model.optimize(callback)
    else:
        model.optimize()

    # Process solution
    solution_dict = {}
    if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
            # Extract facility opening decisions
            open_facilities = [facilities[i]['id'] for i in range(num_facilities) if x[i].X > 0.5]
            x_dict = {str(facilities[i]['id']): int(round(x[i].X)) for i in range(num_facilities)}
            
            # Extract flow decisions (only non-zero flows)
            y_dict = {}
            for s in range(num_scenarios):
                scenario_id = str(scenarios[s]['id'])
                y_dict_scenario = {}
                for i in range(num_facilities):
                    if x[i].X > 0.5:
                        facility_id = str(facilities[i]['id'])
                        customer_dict = {}
                        for j in range(num_customers):
                            flow_val = y[i, j, s].X
                            if flow_val > 1e-6:  # Tolerance for non-zero flows
                                customer_id = str(customers[j]['id'])
                                customer_dict[customer_id] = flow_val
                        if customer_dict:
                            y_dict_scenario[facility_id] = customer_dict
                y_dict[scenario_id] = y_dict_scenario
            
            solution_dict = {
                "objective_value": model.ObjVal,
                "open_facilities": open_facilities,
                "x": x_dict,
                "y": y_dict
            }
        else:
            # No solution found within time limit
            solution_dict = {
                "objective_value": 0.0,
                "open_facilities": [],
                "x": {str(fac['id']): 0 for fac in facilities},
                "y": {}
            }
    else:
        # Model status is not optimal or time limit (e.g., infeasible)
        solution_dict = {
            "objective_value": 0.0,
            "open_facilities": [],
            "x": {str(fac['id']): 0 for fac in facilities},
            "y": {}
        }

    # Write final solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f)
    
    # Log final solution if logger is active
    if logger:
        logger.log_solution(solution_dict['objective_value'], solution_dict)

if __name__ == '__main__':
    main()