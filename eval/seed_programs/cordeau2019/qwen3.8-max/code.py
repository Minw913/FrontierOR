import argparse
import json
import math
import time
from solution_logger import SolutionLogger

def compute_coverage(facility_coords, customer_coords, radius):
    """Compute coverage relationships between facilities and customers."""
    num_facilities = len(facility_coords)
    num_customers = len(customer_coords)
    
    facility_to_customers = {str(j): [] for j in range(num_facilities)}
    customer_to_facilities = {str(i): [] for i in range(num_customers)}
    
    radius_sq = radius * radius
    
    for j in range(num_facilities):
        fx, fy = facility_coords[j]
        for i in range(num_customers):
            cx, cy = customer_coords[i]
            dist_sq = (fx - cx) ** 2 + (fy - cy) ** 2
            if dist_sq <= radius_sq + 1e-9:
                facility_to_customers[str(j)].append(i)
                customer_to_facilities[str(i)].append(j)
    
    return facility_to_customers, customer_to_facilities


def solve_psclp(instance, time_limit, logger=None):
    """Solve the Partial Set Covering Location Problem."""
    import gurobipy as gp
    from gurobipy import GRB
    
    num_facilities = instance["num_facilities"]
    num_customers = instance["num_customers"]
    facility_cost = instance["facility_cost"]
    customer_demands = instance["customer_demands"]
    covering_demand_D = instance["covering_demand_D"]
    
    # Use provided coverage or compute it
    if "facility_to_customers_coverage" in instance and instance["facility_to_customers_coverage"]:
        facility_to_customers = instance["facility_to_customers_coverage"]
        customer_to_facilities = instance["customer_to_facilities_coverage"]
    else:
        facility_to_customers, customer_to_facilities = compute_coverage(
            instance["facility_coordinates"],
            instance["customer_coordinates"],
            instance["radius_of_coverage"]
        )
    
    try:
        model = gp.Model("PSCLP")
        model.setParam("Seed", 0)
        model.setParam("MIPGap", 1e-4)
        model.setParam("NumericFocus", 0)
        model.setParam("Threads", 1)
        model.setParam("TimeLimit", time_limit)
        model.setParam("OutputFlag", 0)
        
        # Decision variables
        x = model.addVars(num_facilities, vtype=GRB.BINARY, name="x")
        y = model.addVars(num_customers, vtype=GRB.BINARY, name="y")
        
        # Objective: minimize total opening cost
        model.setObjective(gp.quicksum(facility_cost[j] * x[j] for j in range(num_facilities)), GRB.MINIMIZE)
        
        # Demand coverage constraint
        model.addConstr(
            gp.quicksum(customer_demands[i] * y[i] for i in range(num_customers)) >= covering_demand_D,
            name="demand_threshold"
        )
        
        # Coverage linking constraints
        for i in range(num_customers):
            covering_facilities = customer_to_facilities.get(str(i), [])
            if covering_facilities:
                model.addConstr(
                    y[i] <= gp.quicksum(x[j] for j in covering_facilities),
                    name=f"cover_{i}"
                )
            else:
                model.addConstr(y[i] == 0, name=f"no_cover_{i}")
        
        # Callback for logging
        best_obj = [float('inf')]
        
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj < best_obj[0]:
                    best_obj[0] = obj
                    if logger:
                        # Extract solution
                        x_vals = model.cbGetSolution(x)
                        open_facs = [j for j in range(num_facilities) if x_vals[j] > 0.5]
                        covered_dem = sum(customer_demands[i] for i in range(num_customers) 
                                         if any(x_vals[j] > 0.5 for j in customer_to_facilities.get(str(i), [])))
                        sol_dict = {
                            "objective_value": obj,
                            "primary_problem_type": "PSCLP",
                            "results": {
                                "PSCLP": {
                                    "objective_value": obj,
                                    "open_facilities": open_facs,
                                    "covered_demand": covered_dem
                                },
                                "MCLP": {
                                    "objective_value": 0,
                                    "open_facilities": [],
                                    "covered_demand": 0
                                }
                            }
                        }
                        logger.log_solution(obj, sol_dict)
        
        model.optimize(callback)
        
        if model.SolCount > 0:
            x_vals = [x[j].X for j in range(num_facilities)]
            open_facilities = [j for j in range(num_facilities) if x_vals[j] > 0.5]
            covered_demand = sum(customer_demands[i] for i in range(num_customers)
                                if any(x_vals[j] > 0.5 for j in customer_to_facilities.get(str(i), [])))
            obj_val = model.ObjVal
            return obj_val, open_facilities, covered_demand
        else:
            return float('inf'), [], 0
    except Exception as e:
        print(f"PSCLP error: {e}")
        return float('inf'), [], 0


def solve_mclp(instance, time_limit, logger=None):
    """Solve the Maximal Covering Location Problem."""
    import gurobipy as gp
    from gurobipy import GRB
    
    num_facilities = instance["num_facilities"]
    num_customers = instance["num_customers"]
    facility_cost = instance["facility_cost"]
    customer_demands = instance["customer_demands"]
    budget_B = instance["budget_B"]
    
    # Use provided coverage or compute it
    if "facility_to_customers_coverage" in instance and instance["facility_to_customers_coverage"]:
        facility_to_customers = instance["facility_to_customers_coverage"]
        customer_to_facilities = instance["customer_to_facilities_coverage"]
    else:
        facility_to_customers, customer_to_facilities = compute_coverage(
            instance["facility_coordinates"],
            instance["customer_coordinates"],
            instance["radius_of_coverage"]
        )
    
    try:
        model = gp.Model("MCLP")
        model.setParam("Seed", 0)
        model.setParam("MIPGap", 1e-4)
        model.setParam("NumericFocus", 0)
        model.setParam("Threads", 1)
        model.setParam("TimeLimit", time_limit)
        model.setParam("OutputFlag", 0)
        
        # Decision variables
        x = model.addVars(num_facilities, vtype=GRB.BINARY, name="x")
        y = model.addVars(num_customers, vtype=GRB.BINARY, name="y")
        
        # Objective: maximize total covered demand
        model.setObjective(gp.quicksum(customer_demands[i] * y[i] for i in range(num_customers)), GRB.MAXIMIZE)
        
        # Budget constraint
        model.addConstr(
            gp.quicksum(facility_cost[j] * x[j] for j in range(num_facilities)) <= budget_B,
            name="budget"
        )
        
        # Coverage linking constraints
        for i in range(num_customers):
            covering_facilities = customer_to_facilities.get(str(i), [])
            if covering_facilities:
                model.addConstr(
                    y[i] <= gp.quicksum(x[j] for j in covering_facilities),
                    name=f"cover_{i}"
                )
            else:
                model.addConstr(y[i] == 0, name=f"no_cover_{i}")
        
        # Callback for logging
        best_obj = [-1]
        
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj > best_obj[0]:
                    best_obj[0] = obj
                    if logger:
                        x_vals = model.cbGetSolution(x)
                        open_facs = [j for j in range(num_facilities) if x_vals[j] > 0.5]
                        covered_dem = sum(customer_demands[i] for i in range(num_customers)
                                         if any(x_vals[j] > 0.5 for j in customer_to_facilities.get(str(i), [])))
                        sol_dict = {
                            "objective_value": obj,
                            "primary_problem_type": "MCLP",
                            "results": {
                                "PSCLP": {
                                    "objective_value": 0,
                                    "open_facilities": [],
                                    "covered_demand": 0
                                },
                                "MCLP": {
                                    "objective_value": obj,
                                    "open_facilities": open_facs,
                                    "covered_demand": covered_dem
                                }
                            }
                        }
                        logger.log_solution(obj, sol_dict)
        
        model.optimize(callback)
        
        if model.SolCount > 0:
            x_vals = [x[j].X for j in range(num_facilities)]
            open_facilities = [j for j in range(num_facilities) if x_vals[j] > 0.5]
            covered_demand = sum(customer_demands[i] for i in range(num_customers)
                                if any(x_vals[j] > 0.5 for j in customer_to_facilities.get(str(i), [])))
            obj_val = model.ObjVal
            return obj_val, open_facilities, covered_demand
        else:
            return 0, [], 0
    except Exception as e:
        print(f"MCLP error: {e}")
        return 0, [], 0


def main():
    parser = argparse.ArgumentParser(description="Solve PSCLP and MCLP")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Load instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Split time between the two problems
    # Give each problem half the time
    half_time = args.time_limit / 2.0
    
    # Solve PSCLP
    psclp_time = min(half_time, args.time_limit - (time.time() - start_time))
    psclp_time = max(psclp_time, 1.0)
    psclp_obj, psclp_facilities, psclp_demand = solve_psclp(instance, psclp_time, logger)
    
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed
    
    # Solve MCLP
    mclp_time = max(remaining, 1.0)
    mclp_obj, mclp_facilities, mclp_demand = solve_mclp(instance, mclp_time, logger)
    
    # Determine primary problem type
    # Use PSCLP as primary (minimize cost)
    primary_type = "PSCLP"
    primary_obj = psclp_obj
    
    # Build solution
    solution = {
        "objective_value": primary_obj,
        "primary_problem_type": primary_type,
        "results": {
            "PSCLP": {
                "objective_value": psclp_obj,
                "open_facilities": psclp_facilities,
                "covered_demand": psclp_demand
            },
            "MCLP": {
                "objective_value": mclp_obj,
                "open_facilities": mclp_facilities,
                "covered_demand": mclp_demand
            }
        }
    }
    
    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)
    
    # Final log
    if logger:
        logger.log_solution(primary_obj, solution)


if __name__ == "__main__":
    main()