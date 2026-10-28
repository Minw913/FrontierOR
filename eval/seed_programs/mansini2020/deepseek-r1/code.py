import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser(description='Solve the multi-dimensional multi-choice knapsack problem.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file.')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file.')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds.')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log intermediate solutions.')
    args = parser.parse_args()

    # Read instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    n_groups = data['parameters']['n_groups']
    m_resources = data['parameters']['m_resources']
    capacities = data['capacities']
    groups = data['groups']

    # Initialize model
    model = gp.Model()
    model.setParam("Threads", 1)
    model.setParam('OutputFlag', 0)  # Suppress solver output
    model.setParam('TimeLimit', args.time_limit)

    # Initialize solution logger if log_path is provided
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    # Create data structures
    variables = {}  # Maps (group_id, item_id) to variable
    obj = gp.LinExpr()
    res_expr = [gp.LinExpr() for _ in range(m_resources)]

    # Add variables and build constraints
    for group in groups:
        group_id = group['group_id']
        items = group['items']
        group_vars = []
        for item in items:
            item_id = item['item_id']
            var = model.addVar(vtype=GRB.BINARY, name=f"g{group_id}_i{item_id}")
            variables[(group_id, item_id)] = var
            group_vars.append(var)
            
            # Add to objective
            obj += item['profit'] * var
            
            # Add to resource expressions
            for r, weight in enumerate(item['weights']):
                res_expr[r] += weight * var
        
        # Exactly one item per group
        model.addConstr(sum(group_vars) == 1, name=f"group_{group_id}")

    # Add resource constraints
    for r in range(m_resources):
        model.addConstr(res_expr[r] <= capacities[r], name=f"res_{r}")

    model.setObjective(obj, GRB.MAXIMIZE)

    # Define callback for logging intermediate solutions
    def callback_function(model, where):
        if where == GRB.Callback.MIPSOL:
            if logger:
                objval = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(objval)

    # Solve the model
    if logger:
        model.optimize(callback_function)
    else:
        model.optimize()

    # Prepare solution
    solution = {}
    if model.SolCount > 0:
        # Extract best solution found
        selected_items = []
        total_profit = model.ObjVal
        
        # Find selected items per group
        for group in groups:
            group_id = group['group_id']
            for item in group['items']:
                item_id = item['item_id']
                var = variables[(group_id, item_id)]
                if var.X > 0.5:  # Tolerance for binary variable
                    selected_items.append({
                        'group_id': group_id,
                        'item_id': item_id,
                        'profit': item['profit'],
                        'weights': item['weights']
                    })
                    break  # Only one item selected per group
                    
        solution['objective_value'] = float(total_profit)
        solution['selected_items'] = selected_items
    else:
        # No feasible solution found
        solution['objective_value'] = 0.0
        solution['selected_items'] = []

    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()