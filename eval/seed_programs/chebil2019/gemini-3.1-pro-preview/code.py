import json
import argparse
import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser(description="Optimal solver for the Setup Knapsack Problem.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write the solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds.")
    parser.add_argument("--log_path", type=str, required=False, help="Path for logging the solutions.")
    
    args = parser.parse_args()
    
    # Initialize SolutionLogger if requested
    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="maximize")
        except ImportError:
            pass

    # Read instance data
    with open(args.instance_path, "r") as f:
        data = json.load(f)
        
    C = data["knapsack_capacity"]
    
    # Configure Gurobi Environment and Model
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    
    model = gp.Model("SetupKnapsack", env=env)
    model.setParam('Threads', 1)  # Strictly restrained to a single CPU core
    if args.time_limit:
        model.setParam('TimeLimit', args.time_limit)
        
    y = {}
    x = {}
    capacity_expr = gp.LinExpr()
    
    # Variable definitions and constraints constructions
    for fam in data["families"]:
        f_id = fam["family_id"]
        sc = fam["setup_cost"]
        sq = fam["setup_capacity"]
        
        # If the setup capacity exceeds the knapsack capacity alone, it forms an invalid state
        if sq > C:
            continue
            
        y[f_id] = model.addVar(vtype=GRB.BINARY, obj=-sc, name=f"y_{f_id}")
        capacity_expr += sq * y[f_id]
        
        items_expr = gp.LinExpr()
        sum_w = 0
        valid_items_count = 0
        
        for it in fam["items"]:
            i_id = it["item_id"]
            p = it["profit"]
            w = it["weight"]
            
            # If the item's weight plus the setup capacity exceeds the total capacity, it can never be selected
            if sq + w > C:
                continue
                
            x[(f_id, i_id)] = model.addVar(vtype=GRB.BINARY, obj=p, name=f"x_{f_id}_{i_id}")
            # Logical dependency: item can only be selected if its community/family is activated
            model.addConstr(x[(f_id, i_id)] <= y[f_id], name=f"dep_{f_id}_{i_id}")
            
            capacity_expr += w * x[(f_id, i_id)]
            items_expr += w * x[(f_id, i_id)]
            sum_w += w
            valid_items_count += 1
            
        if valid_items_count > 0:
            # Strengthening LP relaxation with a surrogate bounding constraint
            max_w = min(C - sq, sum_w)
            model.addConstr(items_expr <= max_w * y[f_id], name=f"str_{f_id}")
        else:
            # If no valid items exist, activating family strictly drops bounds (setup_cost) and wastes space
            model.addConstr(y[f_id] == 0, name=f"noitems_{f_id}")
            
    # Global capacity adherence
    model.addConstr(capacity_expr <= C, name="capacity")
    model.ModelSense = GRB.MAXIMIZE
    
    # Optimization callback for solution logging 
    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)
                
    # Launch optimization
    model.optimize(cb)
    
    # Process solution layout mapping and evaluate objective value accurately
    families_selected = []
    items_selected = []
    obj_val = 0.0
    
    if model.SolCount > 0:
        for fam in data["families"]:
            f_id = fam["family_id"]
            
            # Check effectively chosen items mapped under this family scope
            f_items_selected = []
            if f_id in y and y[f_id].X > 0.5:
                for it in fam["items"]:
                    i_id = it["item_id"]
                    if (f_id, i_id) in x and x[(f_id, i_id)].X > 0.5:
                        f_items_selected.append(it)
                        
            # Project values into final output format (omitting functionally hollow families out of edge guarantees)
            if f_items_selected:
                families_selected.append(f_id)
                obj_val -= fam["setup_cost"]
                for it in f_items_selected:
                    items_selected.append({"family": f_id, "item": it["item_id"]})
                    obj_val += it["profit"]
                    
    # Generate formal output payload structured across solution_schema
    sol = {
        "objective_value": float(obj_val),
        "families_selected": families_selected,
        "items_selected": items_selected
    }
    
    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)

if __name__ == "__main__":
    main()