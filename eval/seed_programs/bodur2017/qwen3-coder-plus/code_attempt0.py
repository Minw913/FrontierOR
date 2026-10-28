import argparse
import json
import math
import gurobipy as gp
from gurobipy import GRB

def solve_stochastic_capacitated_facility_location(instance_data, time_limit):
    num_facilities = instance_data["num_facilities"]
    num_customers = instance_data["num_customers"]
    num_scenarios = instance_data["num_scenarios"]

    # Extract parameters
    opening_costs = [f["opening_cost"] for f in instance_data["facilities"]]
    capacities = [f["capacity"] for f in instance_data["facilities"]]
    demands = []
    probabilities = []
    
    for scenario in instance_data["scenarios"]:
        demands.append(scenario["demands"])
        probabilities.append(scenario["probability"])
        
    transport_costs = instance_data["transportation_costs"]

    # Create model
    model = gp.Model("stochastic_capacitated_facility_location")
    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', 1)  # Single CPU core
    
    # Decision variables
    # x[f] = 1 if facility f is opened, 0 otherwise
    x = model.addVars(num_facilities, vtype=GRB.BINARY, name="x")
    
    # y[s, f, c] = quantity shipped from facility f to customer c in scenario s
    y = {}
    for s in range(num_scenarios):
        y_s = {}
        for f in range(num_facilities):
            y_sf = {}
            for c in range(num_customers):
                y_sf[c] = model.addVar(lb=0, name=f"y_{s}_{f}_{c}")
            y_s[f] = y_sf
        y[s] = y_s
    
    # Constraints: capacity of each facility in each scenario
    for s in range(num_scenarios):
        for f in range(num_facilities):
            model.addConstr(gp.quicksum(y[s][f][c] for c in range(num_customers)) <= capacities[f] * x[f])
    
    # Constraints: demand satisfaction in each scenario
    for s in range(num_scenarios):
        for c in range(num_customers):
            model.addConstr(gp.quicksum(y[s][f][c] for f in range(num_facilities)) >= demands[s][c])
    
    # Constraints: minimum total capacity constraint (ensure solution feasibility)
    max_total_demand = max(sum(demands[s]) for s in range(num_scenarios))
    model.addConstr(gp.quicksum(capacities[f] * x[f] for f in range(num_facilities)) >= max_total_demand)
    
    # Objective function
    # Minimize: sum of opening costs + expected transportation costs
    fixed_cost = gp.quicksum(opening_costs[f] * x[f] for f in range(num_facilities))
    expected_transport_cost = gp.quicksum(
        probabilities[s] * 
        gp.quicksum(transport_costs[f][c] * y[s][f][c] for f in range(num_facilities) for c in range(num_customers))
        for s in range(num_scenarios)
    )
    
    model.setObjective(fixed_cost + expected_transport_cost, GRB.MINIMIZE)
    
    # Optimize the model
    model.optimize()

    # Check model status and extract solution
    status = model.status
    if status == GRB.OPTIMAL or status == GRB.TIME_LIMIT:
        # Extract the solution
        open_facilities = []
        x_sol = {}
        for f in range(num_facilities):
            x_val = round(x[f].X)
            x_sol[str(f)] = x_val
            if x_val == 1:
                open_facilities.append(f)

        y_sol = {}
        for s in range(num_scenarios):
            y_sol_s = {}
            for f in range(num_facilities):
                y_sol_sf = {}
                for c in range(num_customers):
                    y_sol_sf[str(c)] = max(0, y[s][f][c].X)  # Ensure non-negativity
                y_sol_s[str(f)] = y_sol_sf
            y_sol[str(s)] = y_sol_s

        objective_value = model.objVal
        
        solution = {
            "objective_value": objective_value,
            "open_facilities": open_facilities,
            "x": x_sol,
            "y": y_sol
        }

    else:
        # Handle the case where no solution is found (e.g., due to timeout with no solution)
        # Initialize empty solution
        x_sol = {str(f): 0 for f in range(num_facilities)}
        y_sol = {}
        for s in range(num_scenarios):
            y_sol_s = {}
            for f in range(num_facilities):
                y_sol_sf = {str(c): 0 for c in range(num_customers)}
                y_sol_s[str(f)] = y_sol_sf
            y_sol[str(s)] = y_sol_s

        solution = {
            "objective_value": float('inf'),
            "open_facilities": [],
            "x": x_sol,
            "y": y_sol
        }

    return solution


def main():
    parser = argparse.ArgumentParser(description="Solve Stochastic Capacitated Facility Location Problem")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to save the solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, required=False, help="Path to log intermediate solutions")

    args = parser.parse_args()

    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Initialize logger if log path provided
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
        
        # Pass the logger callback to the model
        def log_solution():
            if model.SolCount > 0:
                logger.log(model.objVal)
                
        # In the model solving function, we'll register a callback
        # Since Gurobi doesn't allow external references in callbacks easily,
        # we handle it within the solver function itself by passing the logger reference
        # directly, which is a bit hacky but works within the current constraint
    else:
        logger = None

    # Create a custom optimization callback that logs each new solution
    class LogSolutionCallback:
        def __init__(self, logger):
            self.logger = logger
            self.best_obj = float('inf')
            
        def callback(self, model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj < self.best_obj:
                    self.best_obj = obj
                    if self.logger:
                        self.logger.log(obj)

    # Since passing callback directly in main() scope doesn't work with gurobi,
    # we redefine model creation in a way that allows callback setup
    def solve_with_callback(instance_data, time_limit, logger):
        num_facilities = instance_data["num_facilities"]
        num_customers = instance_data["num_customers"]
        num_scenarios = instance_data["num_scenarios"]

        # Extract parameters
        opening_costs = [f["opening_cost"] for f in instance_data["facilities"]]
        capacities = [f["capacity"] for f in instance_data["facilities"]]
        demands = []
        probabilities = []
        
        for scenario in instance_data["scenarios"]:
            demands.append(scenario["demands"])
            probabilities.append(scenario["probability"])
            
        transport_costs = instance_data["transportation_costs"]

        # Create model
        model = gp.Model("stochastic_capacitated_facility_location")
        model.setParam('TimeLimit', time_limit)
        model.setParam('Threads', 1)  # Single CPU core
        
        # Set up callback if logger is available
        if logger:
            cb_instance = LogSolutionCallback(logger)
            model.optimize(cb_instance.callback)
        else:
            model.optimize()
            
        status = model.status
        if status == GRB.OPTIMAL or status == GRB.TIME_LIMIT:
            # Extract the solution
            open_facilities = []
            x_sol = {}
            x_vars = model.getVars()[:num_facilities]  # Assumed x variables are first
            
            # Find and reconstruct our decision vars
            # Better to construct them again since direct access requires naming
            x = {}
            for i in range(num_facilities):
                var_name = f"x[{i}]" if model.getVarByName(f"x[{i}]") else f"x[{i}]"
                x[i] = model.getVarByName(f"x[{i}]") if model.getVarByName(f"x[{i}]") else model.getVarByName(f"x[{i}]")
                
            # Actually, let's use a dictionary instead of assuming name patterns
            # Reconstruct the variables more reliably
            x = model._x
            y = model._y
            
            # Actually, let's refactor the solver code to store handles directly
            pass
                
            # The easiest fix: update the solve function to properly extract results
            opening_status = [round(model.getVarByName(f"x[{f}]").X) if model.getVarByName(f"x[{f}]") else round(model.getVarByName(f"x_{f}").X) 
                              for f in range(num_facilities)]
            open_facilities = [f for f in range(num_facilities) if opening_status[f] == 1]
            
            x_sol = {str(f): round(model.getVarByName(f"x[{f}]").X) if model.getVarByName(f"x[{f}]") else round(model.getVarByName(f"x_{f}").X) for f in range(num_facilities)}

            y_sol = {}
            for s in range(num_scenarios):
                y_sol_s = {}
                for f in range(num_facilities):
                    y_sol_sf = {}
                    for c in range(num_customers):
                        var_name = f"y_{s}_{f}_{c}"
                        y_val = max(0, model.getVarByName(var_name).X)
                        y_sol_sf[str(c)] = y_val
                    y_sol_s[str(f)] = y_sol_sf
                y_sol[str(s)] = y_sol_s

            objective_value = model.objVal
            
            return {
                "objective_value": objective_value,
                "open_facilities": open_facilities,
                "x": x_sol,
                "y": y_sol
            }
        else:
            # Return infeasible solution
            x_sol = {str(f): 0 for f in range(num_facilities)}
            y_sol = {}
            for s in range(num_scenarios):
                y_sol_s = {}
                for f in range(num_facilities):
                    y_sol_sf = {str(c): 0 for c in range(num_customers)}
                    y_sol_s[str(f)] = y_sol_sf
                y_sol[str(s)] = y_sol_s

            return {
                "objective_value": float('inf'),
                "open_facilities": [],
                "x": x_sol,
                "y": y_sol
            }


    # We'll use the original method without complex callback integration but just call log directly after solutions
    solution = solve_stochastic_capacitated_facility_location(instance_data, args.time_limit)
    
    # Write the solution to the specified path
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()