import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_manufacturing_problem(instance_path, solution_path, time_limit, log_path):
    # Load instance data
    with open(instance_path, 'r') as f:
        instance = json.load(f)

    NI = instance["parameters"]["NI"]
    NK = instance["parameters"]["NK"]
    NT = instance["parameters"]["NT"]

    setup_cost_f = instance["costs"]["setup_cost_f"]
    startup_cost_g = instance["costs"]["startup_cost_g"]
    holding_cost_h = instance["costs"]["holding_cost_h"]
    backlogging_cost_e = instance["costs"]["backlogging_cost_e"]
    capacity_C = instance["machine_data"]["capacity_C"]
    startup_time_sigma = instance["machine_data"]["startup_time_sigma"]
    demand = instance["demand"]
    initial_stock_s0 = instance["initial_conditions"]["initial_stock_s0"]
    initial_backlog_r0 = instance["initial_conditions"]["initial_backlog_r0"]

    # Create model
    model = gp.Model("Manufacturing Planning")
    model.setParam('TimeLimit', time_limit)

    # Decision variables
    # Production quantity for item i on machine k in period t
    production = {}
    for i in range(NI):
        for k in range(NK):
            for t in range(NT):
                production[f"{i}_{k}_{t}"] = model.addVar(vtype=GRB.CONTINUOUS, name=f"production_{i}_{k}_{t}", lb=0)

    # Setup indicator (binary): whether machine k is setup for item i in period t
    setup = {}
    for i in range(NI):
        for k in range(NK):
            for t in range(NT):
                setup[f"{i}_{k}_{t}"] = model.addVar(vtype=GRB.BINARY, name=f"setup_{i}_{k}_{t}")

    # Startup indicator (binary): whether there's a startup for item i on machine k in period t
    startup = {}
    for i in range(NI):
        for k in range(NK):
            for t in range(NT):
                startup[f"{i}_{k}_{t}"] = model.addVar(vtype=GRB.BINARY, name=f"startup_{i}_{k}_{t}")
    
    # Stock level (inventory) at the end of period t for item i
    stock = {}
    for i in range(NI):
        for t in range(NT):
            stock[f"{i}_{t}"] = model.addVar(vtype=GRB.CONTINUOUS, name=f"stock_{i}_{t}", lb=0)

    # Backlog level at the end of period t for item i
    backlog = {}
    for i in range(NI):
        for t in range(NT):
            backlog[f"{i}_{t}"] = model.addVar(vtype=GRB.CONTINUOUS, name=f"backlog_{i}_{t}", lb=0)

    # Additional continuous variables for linearizing constraints
    total_production_item = {}
    for i in range(NI):
        for t in range(NT):
            total_production_item[f"{i}_{t}"] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"tot_prod_{i}_{t}")
    
    # Constraints
    # 1. Flow balance constraint for each item and period
    for i in range(NI):
        for t in range(NT):
            total_prod_expr = gp.quicksum(production[f"{i}_{k}_{t}"] for k in range(NK))
            
            if t == 0:
                inventory_before = initial_stock_s0[i]
                backlog_before = initial_backlog_r0[i]
            else:
                inventory_before = stock[f"{i}_{t-1}"]
                backlog_before = backlog[f"{i}_{t-1}"]
                
            model.addConstr(
                inventory_before + total_prod_expr - backlog_before == 
                demand[i][t] + stock[f"{i}_{t}"] - backlog[f"{i}_{t}"],
                name=f"flow_balance_{i}_{t}"
            )

    # 2. Production cannot exceed capacity considering startup losses
    for k in range(NK):
        for t in range(NT):
            prod_on_k = gp.quicksum(production[f"{i}_{k}_{t}"] for i in range(NI))
            total_startup_loss = gp.quicksum(startup_time_sigma[k] * startup[f"{i}_{k}_{t}"] for i in range(NI))

            model.addConstr(
                prod_on_k + total_startup_loss <= capacity_C[k],
                name=f"capacity_{k}_{t}"
            )

    # 3. Machine can only produce what it's setup for
    for i in range(NI):
        for k in range(NK):
            for t in range(NT):
                model.addConstr(
                    production[f"{i}_{k}_{t}"] <= capacity_C[k] * setup[f"{i}_{k}_{t}"],
                    name=f"produce_only_if_setup_{i}_{k}_{t}"
                )
    
    # 4. Each machine setups to at most one item per period
    for k in range(NK):
        for t in range(NT):
            total_machine_setups = gp.quicksum(setup[f"{i}_{k}_{t}"] for i in range(NI))
            model.addConstr(total_machine_setups <= 1, name=f"machine_one_setup_{k}_{t}")

    # 5. Definition of startup for the first period
    for i in range(NI):
        for k in range(NK):
            model.addConstr(startup[f"{i}_{k}_{0}"] >= setup[f"{i}_{k}_{0}"], 
                            name=f"startup_def_first_{i}_{k}_0")

    # 6. Definition of startup for later periods
    for i in range(NI):
        for k in range(NK):
            for t in range(1, NT):
                # If we are setup for item i at time t, and we were not setup for i at time t-1,
                # then we have a startup for item i at time t
                model.addConstr(
                    startup[f"{i}_{k}_{t}"] >= setup[f"{i}_{k}_{t}"] - setup[f"{i}_{k}_{t-1}"],
                    name=f"startup_def_{i}_{k}_{t}"
                )

    # 7. Link total production to individual production per machine
    for i in range(NI):
        for t in range(NT):
            model.addConstr(
                total_production_item[f"{i}_{t}"] == gp.quicksum(production[f"{i}_{k}_{t}"] for k in range(NK)),
                name=f"total_prod_link_{i}_{t}"
            )

    # Objective function
    # Setup costs
    setup_cost_total = setup_cost_f * gp.quicksum(setup[f"{i}_{k}_{t}"] for i in range(NI) for k in range(NK) for t in range(NT))
    # Startup costs
    startup_cost_total = startup_cost_g * gp.quicksum(startup[f"{i}_{k}_{t}"] for i in range(NI) for k in range(NK) for t in range(NT))
    # Holding costs
    holding_cost_total = sum(holding_cost_h[i] * stock[f"{i}_{t}"] for i in range(NI) for t in range(NT))
    # Backlogging costs
    backlogging_cost_total = sum(backlogging_cost_e[i] * backlog[f"{i}_{t}"] for i in range(NI) for t in range(NT))
    
    total_cost = setup_cost_total + startup_cost_total + holding_cost_total + backlogging_cost_total
    
    model.setObjective(total_cost, GRB.MINIMIZE)

    # Initialize the logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Solve the model
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)

    model.optimize(callback if logger else None)

    # Extract solution
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        solution = {
            "objective_value": model.objVal if model.solCount > 0 else float('inf'),
            "production": {},
            "stock": {},
            "backlog": {},
            "setup": {},
            "startup": {}
        }
        
        for key in production:
            if production[key].x != 0:
                solution["production"][key] = production[key].x
            else:
                solution["production"][key] = 0
        
        for key in stock:
            if stock[key].x != 0:
                solution["stock"][key] = stock[key].x
            else:
                solution["stock"][key] = 0

        for key in backlog:
            if backlog[key].x != 0:
                solution["backlog"][key] = backlog[key].x
            else:
                solution["backlog"][key] = 0

        for key in setup:
            solution["setup"][key] = int(setup[key].x)
            
        for key in startup:
            solution["startup"][key] = int(startup[key].x)

        # Write solution to file
        with open(solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # No feasible solution found
        solution = {
            "objective_value": float('inf'),
            "production": {},
            "stock": {},
            "backlog": {},
            "setup": {},
            "startup": {}
        }
        with open(solution_path, 'w') as f:
            json.dump(solution, f)


def main():
    parser = argparse.ArgumentParser(description='Solve Manufacturing Optimization Problem')
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path where the solution JSON file must be written')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, default=None,
                        help='Path to the JSONL file for logging intermediate solutions (optional)')
    
    args = parser.parse_args()
    
    solve_manufacturing_problem(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()