import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)


def create_sets_and_mappings(instance):
    # Sets
    items = list(range(instance['num_items']))
    periods = list(range(instance['num_periods']))
    machines = list(range(instance['num_machines']))
    
    # Machine capability mapping (convert from 1-indexed to 0-indexed)
    machine_capabilities = {i: [m-1 for m in cap_list] for i, cap_list in enumerate(instance['machine_capabilities'])}
    
    # Item capabilities (reverse mapping)
    item_machines = {i: [] for i in items}
    for i in items:
        item_machines[i] = machine_capabilities[i]
    
    # Compute total remaining demands
    total_remaining_demand = {}
    for i in items:
        cumulative = 0
        for t in reversed(periods):
            cumulative += instance['demand'][i][t]
            total_remaining_demand[(i, t)] = cumulative
    
    return items, periods, machines, item_machines, total_remaining_demand


def solve_instance(instance, time_limit, logger=None):
    try:
        items, periods, machines, item_machines, total_remaining_demand = create_sets_and_mappings(instance)

        # Check if any item doesn't have accessible machine
        for i_idx, cap_list in enumerate(instance['machine_capabilities']):
            if not cap_list:  # No machine can produce this item
                print(f"No available machines for item {i_idx + 1}, stopping.")
                return {"objective_value": float('inf'), "production": {}, "inventory": {}, "setups": {}}

        # Create model
        model = gp.Model("CapacitatedLotSizing")
        model.setParam("Threads", 1)
        model.setParam(GRB.Param.TimeLimit, time_limit)
        model.setParam(GRB.Param.OutputFlag, 1)  # Enable console output
        
        # Decision Variables
        X = {}  # Production quantities
        I = {}  # Inventory levels
        Y = {}  # Setup decisions
        
        # Production variables - create only for valid combinations
        for i in items:
            for m in item_machines[i]:
                for t in periods:
                    var_name = f"X_item{i}_machine{m}_period{t}"
                    X[(i, m, t)] = model.addVar(name=var_name, 
                                               vtype=GRB.CONTINUOUS, lb=0)
        
        # Inventory variables
        for i in items:
            for t in periods:
                var_name = f"I_item{i}_period{t}"
                I[(i, t)] = model.addVar(name=var_name, 
                                         vtype=GRB.CONTINUOUS, lb=0)
        
        # Setup variables - create only for valid combinations
        for i in items:
            for m in item_machines[i]:
                for t in periods:
                    var_name = f"Y_item{i}_machine{m}_period{t}"
                    Y[(i, m, t)] = model.addVar(name=var_name,
                                                vtype=GRB.BINARY)
        
        # Initial inventories - from instance data if available
        initial_inventory = {}
        if 'initial_inventory' in instance:
            initial_inventory = {i: instance['initial_inventory'][i] for i in items}
        else:
            initial_inventory = {i: 0 for i in items}
        
        # Objective function
        obj_expr = gp.LinExpr()
        
        # Setup cost
        for i in items:
            for m in item_machines[i]:
                for t in periods:
                    setup_cost_key = f"item_{i}_machine_{m}"
                    obj_expr.addTerms(instance['setup_cost'][setup_cost_key], Y[(i, m, t)])
                    
        # Production cost
        for i in items:
            for m in item_machines[i]:
                for t in periods:
                    prod_cost_key = f"item_{i}_machine_{m}"
                    obj_expr.addTerms(instance['production_cost'][prod_cost_key], X[(i, m, t)])
        
        # Holding cost
        for i in items:
            for t in periods:
                obj_expr.addTerms(instance['holding_cost'][i], I[(i, t)])
        
        model.setObjective(obj_expr, GRB.MINIMIZE)
        
        # Constraints
        
        # Flow balance constraints (flow conservation)
        for i in items:
            for t in periods:
                lhs = gp.LinExpr()
                
                # Initial inventory handling for period 0
                if t == 0:
                    lhs.addConstant(initial_inventory[i])
                
                # Add productions in period t
                for m in item_machines[i]:
                    lhs.addTerms(1.0, X[(i, m, t)])
                
                # Subtract demand in period t
                lhs.addConstant(-instance['demand'][i][t])
                
                # Add inventory from last period (for period > 0)
                if t > 0:
                    lhs.addTerms(1.0, I[(i, t-1)])
                
                # Subtract current period's inventory
                lhs.addTerms(-1.0, I[(i, t)])
                
                model.addConstr(lhs == 0, name=f"flow_balance_{i}_{t}")
        
        # Setup-production linking constraints (if production, then setup)
        for i in items:
            for m in item_machines[i]:
                for t in periods:
                    # Determine max production possible based on capacity and remaining demand
                    setup_time_key = f"item_{i}_machine_{m}"
                    setup_time = instance['setup_time'][setup_time_key]
                    
                    cap_available = instance['capacity'][m][t]
                    
                    # Cap on number of units based on residual capacity after setup time
                    prod_time_key = f"item_{i}_machine_{m}"
                    prod_time = instance['production_time'][prod_time_key]
                    
                    # Time available for production after setup
                    time_for_production = max(0, cap_available - setup_time)  # Ensure non-negative
                    
                    # Theoretical max units that can be produced in remaining time
                    max_units_by_capacity = time_for_production / prod_time if prod_time > 0 else float('inf')
                    max_units_by_demand = total_remaining_demand[(i, t)]
                    
                    # The actual upper bound for this machine-item-period is limited by demand also
                    max_possible_prod = min(max_units_by_capacity, max_units_by_demand)
                    
                    # Link production to setup: X[i,m,t] <= max_possible_prod * Y[i,m,t]
                    model.addConstr(X[(i, m, t)] <= max_possible_prod * Y[(i, m, t)], 
                                    name=f"linking_{i}_{m}_{t}")
                    
        # Capacity constraints per machine per period
        for m in machines:
            for t in periods:
                lhs = gp.LinExpr()
                
                # Add production times
                for i in items:
                    if m in item_machines[i]:
                        prod_time_key = f"item_{i}_machine_{m}"
                        lhs.addTerms(instance['production_time'][prod_time_key], X[(i, m, t)])
                        
                # Add setup times
                for i in items:
                    if m in item_machines[i]:
                        setup_time_key = f"item_{i}_machine_{m}"
                        lhs.addTerms(instance['setup_time'][setup_time_key], Y[(i, m, t)])
                
                model.addConstr(lhs <= instance['capacity'][m][t],
                                name=f"capacity_{m}_{t}")
        
        # Limit on the number of machines per item per period (setup count limit)
        for i in items:
            for t in periods:
                lhs = gp.LinExpr()
                for m in item_machines[i]:
                    lhs.addTerms(1.0, Y[(i, m, t)])
                
                model.addConstr(lhs <= instance['max_machines_per_item_per_period'][i],
                                name=f"machine_limit_{i}_{t}")

        # Define callback inside the function to access local variables
        def mycallback(model, where):
            if where == GRB.Callback.MIPSol:
                if logger:
                    obj = model.cbGet(GRB.Callback.MIPSolObj)
                    logger.log(obj)

        # Solve the model
        model.optimize(mycallback)
        
        # Prepare the result dictionary
        solution = {
            "objective_value": float(model.ObjVal) if model.SolCount > 0 else float('inf'),
            "production": {},
            "inventory": {},
            "setups": {}
        }
        
        if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
            for i in items:
                for m in item_machines[i]:
                    for t in periods:
                        var_name = f"X_item{i}_machine{m}_period{t}"
                        val = X[(i, m, t)].x if X[(i, m, t)].x >= 1e-9 else 0.0  # Avoid tiny floats
                        solution["production"][var_name] = val
                        
            for i in items:
                for t in periods:
                    var_name = f"I_item{i}_period{t}"
                    val = I[(i, t)].x if I[(i, t)].x >= 1e-9 else 0.0
                    solution["inventory"][var_name] = val
                    
            for i in items:
                for m in item_machines[i]:
                    for t in periods:
                        var_name = f"Y_item{i}_machine{m}_period{t}"
                        val = int(round(Y[(i, m, t)].x))
                        solution["setups"][var_name] = val
        
        return solution

    except gp.GurobiError as e:
        print(f"Gurobi error: {e}")
        return {"objective_value": float('inf'), "production": {}, "inventory": {}, "setups": {}}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {"objective_value": float('inf'), "production": {}, "inventory": {}, "setups": {}}


def main():
    parser = argparse.ArgumentParser(description="Solve capacitated lot sizing problem.")
    parser.add_argument('--instance_path', required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', required=True, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to the JSONL log file')

    args = parser.parse_args()

    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance
    instance = read_instance(args.instance_path)

    # Solve the instance
    solution = solve_instance(instance, args.time_limit, logger)

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()