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


def solve_instance(instance, time_limit):
    items, periods, machines, item_machines, total_remaining_demand = create_sets_and_mappings(instance)

    # Create model
    model = gp.Model("CapacitatedLotSizing")
    model.setParam(GRB.Param.TimeLimit, time_limit)
    model.setParam(GRB.Param.OutputFlag, 1)  # Enable console output
    
    # Decision Variables
    X = {}  # Production quantities
    I = {}  # Inventory levels
    Y = {}  # Setup decisions
    
    # Production variables
    for i in items:
        for t in periods:
            for m in item_machines[i]:
                X[(i, m, t)] = model.addVar(name=f"X_item{i}_machine{m}_period{t}", 
                                           vtype=GRB.CONTINUOUS, lb=0)
    
    # Inventory variables
    for i in items:
        for t in periods:
            I[(i, t)] = model.addVar(name=f"I_item{i}_period{t}", 
                                     vtype=GRB.CONTINUOUS, lb=0)
    
    # Setup variables
    for i in items:
        for t in periods:
            for m in item_machines[i]:
                Y[(i, m, t)] = model.addVar(name=f"Y_item{i}_machine{m}_period{t}",
                                            vtype=GRB.BINARY)
    
    # Objective function
    obj_expr = gp.LinExpr()
    
    # Setup cost
    for i in items:
        for t in periods:
            for m in item_machines[i]:
                setup_cost_key = f"item_{i}_machine_{m}"
                obj_expr.addTerms(instance['setup_cost'][setup_cost_key], Y[(i, m, t)])
                
    # Production cost
    for i in items:
        for t in periods:
            for m in item_machines[i]:
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
                lhs.addConstant(instance.get('initial_inventory', [0]*len(items))[i])
            
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
        for t in periods:
            for m in item_machines[i]:
                # Production is bounded by setup and total remaining demand
                max_possible_prod = min(
                    total_remaining_demand[(i, t)],
                    (instance['capacity'][m][t] - instance['setup_time'][f"item_{i}_machine_{m}"]) / 
                    instance['production_time'][f"item_{i}_machine_{m}"]
                )
                
                # X[i,m,t] <= max_possible_prod * Y[i,m,t]
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
    
    # Solve the model
    model.optimize()
    
    # Extract solution values
    solution_production = {}
    solution_inventory = {}
    solution_setups = {}
    
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
        for i in items:
            for t in periods:
                for m in item_machines[i]:
                    var_name = f"X_item{i}_machine{m}_period{t}"
                    solution_production[var_name] = X[(i, m, t)].x
                    
        for i in items:
            for t in periods:
                var_name = f"I_item{i}_period{t}"
                solution_inventory[var_name] = I[(i, t)].x
                
        for i in items:
            for t in periods:
                for m in item_machines[i]:
                    var_name = f"Y_item{i}_machine{m}_period{t}"
                    solution_setups[var_name] = Y[(i, m, t)].x
    
    return {
        "objective_value": model.ObjVal if model.SolCount > 0 else float('inf'),
        "production": solution_production,
        "inventory": solution_inventory,
        "setups": solution_setups
    }


def log_solution_callback(model, where, logger):
    if where == GRB.Callback.MIPSol:
        if logger:
            obj = model.cbGet(GRB.Callback.MIPSolObj)
            logger.log(obj)


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
    solution = solve_instance(instance, args.time_limit)

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()