import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_single_machine_lot_sizing(instance_path, time_limit, log_path):
    # Read instance data
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    num_items = data['num_items']
    num_periods = data['num_periods']
    demand = data['demand']      # [num_items][num_periods]
    setup_cost = data['setup_cost']  # [num_items]
    holding_cost = data['holding_cost']  # [num_items]
    variable_cost = data['variable_cost']  # [num_items]
    setup_time = data['setup_time']  # [num_items]
    variable_time = data['variable_time']  # [num_items]
    capacity = data['capacity']  # [num_periods]
    initial_inventory_cost = data['initial_inventory_cost']  # [num_items]

    # Calculate total demand for each item to determine maximum production possible
    total_demand_per_item = [sum(demand[i][t] for t in range(num_periods)) for i in range(num_items)]

    # Create model
    model = gp.Model('single_machine_lot_sizing')
    model.setParam("Threads", 1)
    model.setParam(GRB.Param.TimeLimit, time_limit)
    
    # Silence Gurobi unless we have debugging needs
    model.setParam(GRB.Param.OutputFlag, 1)  

    # Add variables
    # Initial inventory (one-time decision at beginning)
    x_init = []
    for i in range(num_items):
        var = model.addVar(lb=0.0, name=f"x_init_{i}")
        x_init.append(var)

    # Production quantities
    x_prod = []
    for i in range(num_items):
        item_prods = []
        for t in range(num_periods):
            var = model.addVar(lb=0.0, ub=GRB.INFINITY, name=f"x_prod_{i}_{t}")
            item_prods.append(var)
        x_prod.append(item_prods)

    # Setup decisions (binary)
    y_setup = []
    for i in range(num_items):
        item_setups = []
        for t in range(num_periods):
            var = model.addVar(vtype=GRB.BINARY, name=f"y_setup_{i}_{t}")
            item_setups.append(var)
        y_setup.append(item_setups)

    # Inventory levels (including initial inventory as period 0)
    s_inv = []
    for i in range(num_items):
        item_invs = []
        for t in range(num_periods + 1):  # t=0 is initial, then t=1..T are actual periods
            var = model.addVar(lb=0.0, name=f"s_inv_{i}_{t}")
            item_invs.append(var)
        s_inv.append(item_invs)

    model.update()

    # Demand balance constraints (flow conservation)
    for i in range(num_items):
        for t in range(num_periods):
            # Inventory at start of period + production = demand + inventory at end
            # Initial inventory is s_inv[i][0], s_inv[i][1] is end of period 0 (before first real period)
            model.addConstr(
                s_inv[i][t] + x_prod[i][t] == demand[i][t] + s_inv[i][t + 1],
                name=f"demand_balance_{i}_{t}"
            )

    # Link production to setup (can't produce without setup in that period)
    for i in range(num_items):
        for t in range(num_periods):
            # Production upper bound based on capacity constraints later, but at least need setup
            model.addConstr(x_prod[i][t] <= total_demand_per_item[i] * y_setup[i][t], 
                            name=f"prod_setup_link_{i}_{t}")

    # Capacity constraints for each period
    for t in range(num_periods):
        capacity_usage_expr = gp.LinExpr()
        # Add setup times for items being produced
        for i in range(num_items):
            capacity_usage_expr += setup_time[i] * y_setup[i][t]
        
        # Add variable production times
        for i in range(num_items):
            capacity_usage_expr += variable_time[i] * x_prod[i][t]
        
        model.addConstr(capacity_usage_expr <= capacity[t], name=f"capacity_{t}")

    # Set initial inventory relationship
    for i in range(num_items):
        model.addConstr(s_inv[i][0] == x_init[i], name=f"initial_inventory_{i}")

    # No inventory left at the end
    for i in range(num_items):
        model.addConstr(s_inv[i][num_periods] == 0, name=f"terminal_inventory_{i}")

    # Define objective function
    # Initial inventory cost
    obj_expr = gp.LinExpr()
    for i in range(num_items):
        obj_expr.addTerms(initial_inventory_cost[i], x_init[i])

    # Setup costs
    for i in range(num_items):
        for t in range(num_periods):
            obj_expr.addTerms(setup_cost[i], y_setup[i][t])
    
    # Variable production costs
    for i in range(num_items):
        for t in range(num_periods):
            obj_expr.addTerms(variable_cost[i], x_prod[i][t])

    # Holding costs
    for i in range(num_items):
        for t in range(1, num_periods + 1):  # Start from 1 since s_inv[i][0] is not carrying cost
            obj_expr.addTerms(holding_cost[i], s_inv[i][t])

    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Callback to log improved solutions (if logging path provided)
    logger = None
    if log_path:
        logger = SolutionLogger(log_path, sense="minimize")
        
    def sol_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    # Optimize
    model.optimize(sol_callback)

    # Prepare solution output
    # Check if a solution exists
    if model.solCount == 0:
        raise ValueError("No feasible solution found")

    # Extract solution values
    solution_x_init = [x_init[i].x for i in range(num_items)]
    solution_production = [[x_prod[i][t].x for t in range(num_periods)] for i in range(num_items)]
    solution_setup = [[int(y_setup[i][t].x) for t in range(num_periods)] for i in range(num_items)]
    
    # For inventory, we need the stored values for all periods including initial
    solution_inventory = [[s_inv[i][t].x for t in range(num_periods + 1)] for i in range(num_items)]

    # Calculate objective value manually to output exactly what's used in the model
    obj_val = model.objVal

    return {
        "objective_value": obj_val,
        "production": solution_production,
        "setup": solution_setup,
        "inventory": solution_inventory
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON file containing the problem instance.')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path where the final solution JSON file must be written.')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum runtime in seconds.')
    parser.add_argument('--log_path', type=str, required=False,
                        help='Path to a JSONL file for logging intermediate solutions.')

    args = parser.parse_args()

    # Solve and get solution
    solution = solve_single_machine_lot_sizing(
        instance_path=args.instance_path,
        time_limit=args.time_limit,
        log_path=args.log_path
    )

    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()