import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser(description="Airline Crew Scheduling Optimization")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write the solution JSON.")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path for logging intermediate solutions.")
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Load instance data
    with open(args.instance_path, "r") as f:
        data = json.load(f)
        
    dimensions = data["dimensions"]
    num_rows = dimensions["num_rows"]
    num_cols = dimensions["num_cols"]
    
    costs = data["cost_vector"]
    A_cols = data["constraint_matrix_A"]["columns"]
    
    has_base_constraints = data.get("has_base_constraints", False)
    
    # Initialize Gurobi model
    m = gp.Model("CrewScheduling")
    
    # Enforce constraints and single CPU core limit
    m.setParam("TimeLimit", args.time_limit)
    m.setParam("Threads", 1)
    
    # Create decision variables
    # x[j] = 1 if candidate rotation j is selected, 0 otherwise
    x = m.addVars(num_cols, vtype=GRB.BINARY, obj=costs, name="x")
    
    # Transpose rotation to flight legs coverage matrix to easily iterate over row covers
    row_covers = [[] for _ in range(num_rows)]
    for j, col in enumerate(A_cols):
        for i in col:
            row_covers[i].append(j)
            
    # Ensure exact coverage constraint for each flight leg: each must be assigned exactly once
    for i in range(num_rows):
        m.addConstr(gp.quicksum(x[j] for j in row_covers[i]) == 1, name=f"cov_{i}")
        
    # Enforce base workload utilization constraints if any exist in the instance
    if has_base_constraints:
        base_data = data["base_constraints"]
        num_bases = base_data["num_bases"]
        D_rows = base_data["D_matrix"]["rows"]
        lower_bounds = base_data["lower_bounds_d1"]
        upper_bounds = base_data["upper_bounds_d2"]
        
        for b in range(num_bases):
            # Only add terms with strictly positive weight
            expr = gp.quicksum(weight * x[j] for j, weight in enumerate(D_rows[b]) if weight != 0)
            m.addConstr(expr >= lower_bounds[b], name=f"base_lb_{b}")
            m.addConstr(expr <= upper_bounds[b], name=f"base_ub_{b}")
            
    # Callback to log new best solutions during search
    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)
                
    # Optimize
    m.optimize(cb)
    
    # Save output if a feasible solution was found
    if m.solCount > 0:
        selected_rotations = []
        variable_values = {}
        
        for j in range(num_cols):
            # Read threshold to address numerical imperfections optimally
            val = x[j].X
            is_selected = val > 0.5
            variable_values[str(j)] = 1.0 if is_selected else 0.0
            if is_selected:
                selected_rotations.append(j)
                
        solution = {
            "objective_value": m.objVal,
            "selected_rotations": selected_rotations,
            "variable_values": variable_values
        }
        
        with open(args.solution_path, "w") as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()