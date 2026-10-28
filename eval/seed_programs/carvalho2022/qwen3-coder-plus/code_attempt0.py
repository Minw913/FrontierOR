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
                model.addConstr(total_prod_i_j_t <= max_setups_per_item[i][j][t] * max(max(demands[i][t:] + [1])),
                               name=f"machine_eligibility_{i}_{j}_{t}")
                
                # Produce must respect setup and lot size constraints
                # If not set up, produce 0; otherwise must be > min lot size when setup occurs
                # This requires linking with minimum lot size: if we setup the item from cold start, enforce min lot
                
                # Minimum lot size constraint: if the machine wasn't already configured (carried over) and we're setting up to i,
                # then if there's any production, it must be at least minimum lot size
                if t == 0:
                    # First period: not carry-in from prior, so any setup directly implies start-up
                    model.addConstr(total_prod_i_j_t >= minimum_lot_sizes[i] * setup_count,
                                   name=f"min_lot_size_first_0_{i}_{j}_{t}")
                else:
                    # Later period: need to consider carryover
                    # Production = 0 if no carryover and no setup
                    # If there's no setup to i and no carryover of i, production must be 0
                    model.addConstr(total_prod_i_j_t <= gp.quicksum(x[i, j, t, l] for l in range(t, p)) *
                                   (z[i, j, t] + setup_count),
                                   name=f"production_requires_setup_or_carryover_{i}_{j}_{t}")
                    
                    # The actual min lot size enforcement when there's a changeover to i (not carried)
                    model.addConstr(total_prod_i_j_t >= minimum_lot_sizes[i] * (setup_count - z[i, j, t]),
                                   name=f"min_lot_size_changeover_0_{i}_{j}_{t}")
    
    # Machine configuration constraints: exactly one item per machine per period
    for j in range(m):
        for t in range(p):
            model.addConstr(gp.quicksum(z[i, j, t] for i in range(n)) == 1, name=f"one_config_{j}_{t}")
    
    # Setup consistency: incoming setup flow balances current config with outgoing
    for i in range(n):
        for j in range(m):
            for t in range(p):
                # At end of period t, what is the final setup?
                incoming_from_carry = z[i, j, t]  # carried in state
                setups_into_i = gp.quicksum(y[i_prime, i, j, t] for i_prime in range(n))
                setups_out_of_i = gp.quicksum(y[i, i_prime, j, t] for i_prime in range(n) if i != i_prime)
                
                # If we start as i and don't change out, end up as i
                # If we start not-i but setup to i, end up as i
                # If we start as i and setup to j, end up as j
                # We need an auxiliary variable to track end state
                # Let w[i,j,t] = 1 if machine j ends period t configured for item i
                w = {}  # For simplicity, we'll linearize directly
                if t < p - 1:  # Not last period
                    # End of period t config is start of period t+1 config
                    # What matters for carry-over: machine must end up as some configuration in period t
                    pass  # Constraints below handle flow balance
    
    # Setup flow balance equations to maintain configuration continuity
    for i in range(n):
        for j in range(m):
            for t in range(1, p):  # From period 1 onwards
                setups_into_i = gp.quicksum(y[i_prime, i, j, t-1] for i_prime in range(n))  # Incoming to i in prev period
                setups_out_of_i = gp.quicksum(y[i, i_prime, j, t-1] for i_prime in range(n) if i != i_prime)  # Outgoing from i in prev period
                
                # Carry-over from t-1 to t of item i setup means final state for item i in period t-1 was active
                # Final setup of item i on machine j at end of period t-1
                # Final state = start state + incoming setups - outgoing setups
                # But this isn't linearizable easily; instead, define constraints linking z across periods
                # z[i][j][t] = z[i][j][t-1] - outgoing_setups_to_other[i][j][t-1] + incoming_setups_to_i[t-1]
                
                model.addConstr(
                    z[i, j, t] == 
                    # Carried over (was configured and didn't switch in prior period)
                    z[i, j, t-1] - setups_out_of_i +
                    # Switched to i in prior period 
                    setups_into_i,
                    name=f"carryover_consistency_{i}_{j}_{t}"
                )
    
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
    
    # Also need to respect machine eligibility
    for i in range(n):
        for j in range(m):
            if machine_eligibility[i][j] == 0:
                for t in range(p):
                    # If not eligible, no production and no setups to/from
                    total_setups_to_i = gp.quicksum(y[i_prime, i, j, t] for i_prime in range(n))
                    total_setups_from_i = gp.quicksum(y[i, i_prime, j, t] for i_prime in range(n))
                    total_prod = gp.quicksum(x[i, j, t, l] for l in range(t, p))
                    
                    # No production allowed if not eligible
                    model.addConstr(total_prod == 0, name=f"no_prod_if_not_eligible_prod_{i}_{j}_{t}")
                    # No setups to this item if not eligible  
                    model.addConstr(total_setups_to_i == 0, name=f"no_setup_if_not_eligible_in_{i}_{j}_{t}")
                    # No setups from this item if not eligible (needed for flow)
                    model.addConstr(total_setups_from_i == 0, name=f"no_setup_if_not_eligible_out_{i}_{j}_{t}")
                    # Configuration not allowed if not eligible
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
                        var_name = f"x_{i}_{j}_{k}_{l}"
                        if var_name in [v.varName for v in model.getVars()]:
                            val = model.getVarByName(var_name).X
                            if val > 1e-6:  # Non-zero production
                                solution["production"][var_name] = round(val)
        
        # Extract setup counts  
        for i in range(n):
            for ip in range(n):
                for j in range(m):
                    for t in range(p):
                        var_name = f"y_{i}_{ip}_{j}_{t}"
                        if var_name in [v.varName for v in model.getVars()]:
                            val = model.getVarByName(var_name).X
                            if val > 1e-6:  # Non-zero setups
                                solution["setups"][var_name] = round(val)
        
        # Extract carryover states
        for i in range(n):
            for j in range(m):
                for t in range(p):
                    var_name = f"z_{i}_{j}_{t}"
                    if var_name in [v.varName for v in model.getVars()]:
                        val = model.getVarByName(var_name).X
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