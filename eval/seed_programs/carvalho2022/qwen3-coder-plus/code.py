import argparse
import json
import math
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def create_model(data):
    n = data['dimensions']['n']  # number of items
    m = data['dimensions']['m']  # number of machines
    p = data['dimensions']['p']  # number of periods
    
    # parameters 
    demands = data['demands']
    inventory_costs = data['inventory_costs']
    processing_time = data['processing_time']
    setup_times = data['setup_times']
    setup_costs = data['setup_costs']
    
    machine_capacities = data['machine_capacities']
    max_production = data['max_production']  # cumulative max production
    max_setups_per_item = data['max_setups_per_item']  # [item][machine][period]
    minimum_lot_sizes = data['minimum_lot_sizes']
    machine_eligibility = data['machine_eligibility']  # [item][machine]
    
    # Build model
    model = gp.Model('ParallelMachineScheduling')
    model.setParam("Threads", 1)
    
    # Define variables - production allocation decisions:
    # x[item][machine][prod_period][demand_period] = units of item i produced on machine j in production period k to meet demand in period l
    x = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for l in range(k, p):  # items can't meet future demand
                    x[i, j, k, l] = model.addVar(vtype=GRB.CONTINUOUS, name=f"x_{i}_{j}_{k}_{l}", lb=0)
                    
    # Define binary setup variables: y[from_item][to_item][machine][period] = number of times setup from i to i_prime happens
    y = {}
    for i in range(n):
        for i_prime in range(n):
            for j in range(m):
                for t in range(p):
                    y[i, i_prime, j, t] = model.addVar(vtype=GRB.INTEGER, name=f"y_{i}_{i_prime}_{j}_{t}", lb=0)
    
    # Define binary carryover variables: z[item][machine][period] = whether machine j starts period t configured for item i
    z = {}
    for i in range(n):
        for j in range(m):
            for t in range(p):
                z[i, j, t] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}_{t}")
    
    # Demand satisfaction constraints - each demand must be fully served
    for i in range(n):
        for l in range(p):
            model.addConstr(
                gp.quicksum(x[i, j, k, l] for j in range(m) for k in range(l + 1)) == demands[i][l],
                name=f"demand_satisfaction_{i}_{l}"
            )
    
    # Machine capacity constraints
    for j in range(m):
        for t in range(p):
            production_time = gp.quicksum(processing_time * x[i, j, t, l]
                                         for i in range(n) for l in range(t, p))
            setup_time = gp.quicksum(setup_times[i][i_prime][j] * y[i, i_prime, j, t]
                                     for i in range(n) for i_prime in range(n))
            
            model.addConstr(
                production_time + setup_time <= machine_capacities[j][t],
                name=f"capacity_{j}_{t}"
            )
    
    # Link production with setups - production only when setup
    for i in range(n):
        for j in range(m):
            for t in range(p):
                # Total production of item i on machine j in period t
                total_prod_i_j_t = gp.quicksum(x[i, j, t, l] for l in range(t, p))
                
                # Count setups of item i on machine j in period t
                setup_count = gp.quicksum(y[i_prime, i, j, t] for i_prime in range(n))  # setups INTO item i
                
                # The setup count is limited by max setups
                model.addConstr(setup_count <= max_setups_per_item[i][j][t],
                               name=f"max_setup_count_{i}_{j}_{t}")
                
                # Only produce if machine is eligible for this item
                if machine_eligibility[i][j] == 0:
                    # No production allowed if not eligible
                    model.addConstr(total_prod_i_j_t == 0, name=f"no_prod_if_not_eligible_prod_{i}_{j}_{t}")
                    continue
                
                # If setup_count > 0 (set up fresh - not carryover) and production > 0, enforce minimum lot size
                # We need a big-M constraint to capture: if setup_count > 0 and production > 0, then production >= min_lot
                # This is tricky to model directly; use indicator-based logic
                
                # Create a binary variable to denote if there has been a setup to item i (not from same item)
                b_setup_done = model.addVar(vtype=GRB.BINARY, name=f"b_setup_done_{i}_{j}_{t}")
                
                # Indicator: if any setup to i (fresh), then b_setup_done = 1
                model.addConstr(b_setup_done >= (setup_count - z[i, j, t]) / 
                    max(1, max_setups_per_item[i][j][t]), name=f"setup_done_indicator_{i}_{j}_{t}")
                
                # If both production > 0 AND fresh setup > 0, enforce lot size constraint
                model.addConstr(total_prod_i_j_t >= minimum_lot_sizes[i] * b_setup_done,
                               name=f"min_lot_size_enforcement_{i}_{j}_{t}")
    
    # Machine configuration constraints: exactly one item per machine per period
    for j in range(m):
        for t in range(p):
            model.addConstr(gp.quicksum(z[i, j, t] for i in range(n)) == 1, name=f"one_config_{j}_{t}")
    
    # Setup flow balance equations to maintain configuration continuity
    # Final setup state = initial state + (setups_in - setups_out)
    w_vals = {}  # For tracking final configurations
    for j in range(m):
        for t in range(p):
            for i in range(n):
                # Compute final state after period t
                final_config_var = model.addVar(vtype=GRB.BINARY, name=f"w_final_{i}_{j}_{t}")
                # Final = Initial + In - Out
                setups_in = gp.quicksum(y[i_prime, i, j, t] for i_prime in range(n))
                setups_out = gp.quicksum(y[i, i_prime, j, t] for i_prime in range(n) if i != i_prime)
                
                if t == 0:
                    # Initial state comes from z at start
                    model.addConstr(final_config_var == z[i, j, t] - setups_out + setups_in,
                                   name=f"final_config_balance_{i}_{j}_{t}")
                else:
                    # Final state of t-1 feeds into initial state of t
                    model.addConstr(final_config_var == z[i, j, t] - setups_out + setups_in,
                                   name=f"final_config_balance_{i}_{j}_{t}")
    
    # Link carryover state with final state from previous
    for i in range(n):
        for j in range(m):
            for t in range(1, p):  # From period 1 onwards
                # z[i, j, t] should match final_config_var from period t-1 where we're the final state
                model.addConstr(z[i, j, t] <= gp.quicksum(y[i_prime, i, j, t-1] for i_prime in range(n)) + z[i, j, t-1],
                               name=f"carryover_link_{i}_{j}_{t}_a")
                model.addConstr(z[i, j, t] >= (gp.quicksum(y[i_prime, i, j, t-1] for i_prime in range(n)) + z[i, j, t-1]) - 1,
                               name=f"carryover_link_{i}_{j}_{t}_b")

    # Calculate objective function
    ## Inventory Holding Costs
    inventory_cost_expr = 0
    for i in range(n):
        for l in range(p):
            # Units to satisfy demand in period l made in period k where k < l
            satisfied_by_prod_before_l = gp.quicksum(
                x[i, j, k, l] * inventory_costs[i] * (l - k)
                for j in range(m) for k in range(l)
            )
            inventory_cost_expr += satisfied_by_prod_before_l
            
    ## Setup Costs
    setup_cost_expr = 0
    for i in range(n):
        for i_prime in range(n):
            for j in range(m):
                for t in range(p):
                    setup_cost_expr += setup_costs[i][i_prime][j] * y[i, i_prime, j, t]
    
    obj_expr = inventory_cost_expr + setup_cost_expr
    
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    # Respect machine eligibility
    for i in range(n):
        for j in range(m):
            if machine_eligibility[i][j] == 0:
                for t in range(p):
                    # If not eligible, no setups to/from and no production
                    for i_prime in range(n):
                        model.addConstr(y[i_prime, i, j, t] == 0, name=f"no_setup_in_if_not_eligible_{i}_{j}_{t}_{i_prime}")
                        if i_prime != i:
                            model.addConstr(y[i, i_prime, j, t] == 0, name=f"no_setup_out_if_not_eligible_{i}_{j}_{t}_{i_prime}")
                    # No production
                    total_prod = gp.quicksum(x[i, j, t, l] for l in range(t, p))
                    model.addConstr(total_prod == 0, name=f"no_prod_if_not_eligible_{i}_{j}_{t}")
                    # No carryover configuration (z)
                    model.addConstr(z[i, j, t] == 0, name=f"no_config_if_not_eligible_{i}_{j}_{t}")

    return model, x, y, z, n, m, p


def solve_instance(instance_path, time_limit, log_path=None):
    # Load instance data
    with open(instance_path, "r") as f:
        data = json.load(f)
    
    # Initialize logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Build the optimization model
    model, x, y, z, n, m, p = create_model(data)
    
    # Set solver parameters
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPGap', 1e-4)
    # Heuristics might help find initial good solutions
    model.setParam('Heuristics', 0.05)
    
    # Callback for logging improvements
    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)
    
    # Optimize
    model.optimize(mycallback)
    
    # Prepare solution
    solution = {
        "objective_value": model.objVal if model.solCount > 0 else float('inf'),
        "production": {},
        "setups": {},
        "carryover": {}
    }
    
    if model.solCount > 0:
        # Extract production quantities
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for l in range(k, p):
                        val = x.get((i, j, k, l), None)
                        if val is not None:
                            val = val.X
                            if val > 1e-6:  # Non-zero production
                                var_name = f"x_{i}_{j}_{k}_{l}"
                                solution["production"][var_name] = round(val)
        
        # Extract setup counts  
        for i in range(n):
            for i_prime in range(n):
                for j in range(m):
                    for t in range(p):
                        val = y.get((i, i_prime, j, t), None)
                        if val is not None:
                            val = val.X
                            if val > 1e-6:  # Non-zero setups
                                var_name = f"y_{i}_{i_prime}_{j}_{t}"
                                solution["setups"][var_name] = round(val)
        
        # Extract carryover states
        for i in range(n):
            for j in range(m):
                for t in range(p):
                    val = z.get((i, j, t), None)
                    if val is not None:
                        val = val.X
                        var_name = f"z_{i}_{j}_{t}"
                        solution["carryover"][var_name] = round(val)
    
    return solution


def main():
    parser = argparse.ArgumentParser(description='Solve Parallel Machine Scheduling Problem')
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON file containing the problem instance")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path where the final solution JSON file must be written")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None,
                        help="Path to a JSONL file for logging intermediate solutions (optional)")
    
    args = parser.parse_args()
    
    # Solve the instance
    solution = solve_instance(args.instance_path, args.time_limit, args.log_path)
    
    # Write solution to file
    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()