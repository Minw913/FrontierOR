import argparse
import json
import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser(description="Solve Knapsack with Setups Optimization Problem")
    parser.add_argument('--instance_path', type=str, required=True, help="Path to the instance JSON file")
    parser.add_argument('--solution_path', type=str, required=True, help="Path where the output solution JSON file will be written")
    parser.add_argument('--time_limit', type=int, required=True, help="Maximum runtime in seconds")
    parser.add_argument('--log_path', type=str, default=None, help="Path to the JSONL logging file")
    args = parser.parse_args()

    # Load instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    # Setup the logger
    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="maximize")
        except ImportError:
            pass

    # Callback class for logging intermediate better solutions
    class GurobiCallback:
        def __init__(self, logger):
            self.logger = logger
            
        def __call__(self, model, where):
            if where == GRB.Callback.MIPSOL:
                # Get the objective value of the newly found integer feasible solution
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if self.logger:
                    self.logger.log(obj)
    
    # Initialize Gurobi environment and model
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    model = gp.Model("KnapsackSetup", env=env)
    
    # Execution constraints given by requirements
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    
    y = {}
    x = {}

    obj_expr = gp.LinExpr()
    cap_expr = gp.LinExpr()

    # Model variables and coefficients
    for c in data['classes']:
        cid = c['class_id']
        
        # Decision variable for setting up the class
        y[cid] = model.addVar(vtype=GRB.BINARY, name=f"y_{cid}")
        y[cid].Start = 0  # Initial feasible solution baseline
        
        # Add setup cost (negative value) to objective and capacity consumption to constraint
        obj_expr.addTerms(c['f_i'], y[cid])
        cap_expr.addTerms(c['d_i'], y[cid])
        
        for item in c['items']:
            iid = item['item_id']
            
            # Decision variable for selecting an item in the class
            x[cid, iid] = model.addVar(vtype=GRB.BINARY, name=f"x_{cid}_{iid}")
            x[cid, iid].Start = 0  # Initial feasible solution baseline
            
            # Add item profit to objective and element weight to capacity constraint
            obj_expr.addTerms(item['c_ij'], x[cid, iid])
            cap_expr.addTerms(item['a_ij'], x[cid, iid])

    # Set objective: maximize profit
    model.setObjective(obj_expr, GRB.MAXIMIZE)
    
    # Knapsack overall capacity constraint
    model.addConstr(cap_expr <= data['b'], name="capacity")

    # Dependency combinations linking item choices to corresponding setup variable
    for c in data['classes']:
        cid = c['class_id']
        for item in c['items']:
            iid = item['item_id']
            model.addConstr(x[cid, iid] <= y[cid], name=f"link_{cid}_{iid}")

    # Optimize the model while registering callback
    cb = GurobiCallback(logger)
    model.optimize(cb)

    # Process findings and write to solution dictionary format
    if model.SolCount > 0:
        obj_val = model.ObjVal
        sol_classes = []
        for c in data['classes']:
            cid = c['class_id']
            y_val = int(round(y[cid].X))
            items_sol = []
            for item in c['items']:
                iid = item['item_id']
                x_val = int(round(x[cid, iid].X))
                items_sol.append({
                    "item_id": iid,
                    "x_ij": x_val
                })
            sol_classes.append({
                "class_id": cid,
                "y_i": y_val,
                "items": items_sol
            })
    else:
        # Fallback empty solution if no optimal paths evaluated in bounds
        obj_val = 0.0
        sol_classes = []
        for c in data['classes']:
            cid = c['class_id']
            items_sol = []
            for item in c['items']:
                iid = item['item_id']
                items_sol.append({
                    "item_id": iid,
                    "x_ij": 0
                })
            sol_classes.append({
                "class_id": cid,
                "y_i": 0,
                "items": items_sol
            })
    
    # Prepare mapping to exactly adhere to JSON schema defined layout specifications
    output_data = {
        "classes": sol_classes,
        "objective_value": obj_val
    }

    # Stream out the resultant formatted object
    with open(args.solution_path, 'w') as f:
        json.dump(output_data, f, indent=2)

if __name__ == '__main__':
    main()