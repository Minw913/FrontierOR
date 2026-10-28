import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_supply_chain(instance_path, solution_path, time_limit, log_path):
    # Load instance data
    with open(instance_path, 'r') as f:
        data = json.load(f)

    num_items = data['dimensions']['num_items']
    num_periods = data['dimensions']['num_periods']
    num_plants = data['dimensions']['num_plants']
    num_customers = data['dimensions']['num_customers']

    plant_capabilities = data['plant_capabilities']
    customer_item_demands = data['customer_item_demands']
    demand = data['demand']
    setup_cost = data['setup_cost']
    production_cost = data['production_cost']
    holding_cost = data['holding_cost']
    setup_time = data['setup_time']
    production_time = data['production_time']
    capacity = data['capacity']
    plant_opening_cost = data['plant_opening_cost']
    transportation_cost = data['transportation_cost']

    # Calculate total demand per item per period for constraint calculations
    item_total_demand = [[sum(demand[i][j][t] for j in range(num_customers)) 
                         for t in range(num_periods)] 
                        for i in range(num_items)]

    # Initialize logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    # Create model
    model = gp.Model("SupplyChain")
    model.setParam("Threads", 1)
    model.setParam('TimeLimit', time_limit)
    
    # Callback to log objective values during the optimization process
    def callback(model, where):
        if where == GRB.Callback.MIP:
            obj_best = model.cbGet(GRB.Callback.MIP_OBJBST)
            if logger:
                logger.log(obj_best)

    # Variable creation
    # Z[plant] - whether plant p is open
    Z = {}
    for p in range(num_plants):
        Z[p] = model.addVar(vtype=GRB.BINARY, name=f"Z_{p}")

    # U[item, plant] - whether item i is assigned to plant p
    U = {}
    for i in range(num_items):
        for p in plant_capabilities[i]:
            U[(i, p)] = model.addVar(vtype=GRB.BINARY, name=f"U_{i}_{p}")
        # Add dummy vars for non-capable plants (should remain 0)
        for p in range(num_plants):
            if p not in plant_capabilities[i]:
                U[(i, p)] = 0

    # Y[item, plant, period] - whether there's a setup for item i at plant p in period t
    Y = {}
    for i in range(num_items):
        for p in plant_capabilities[i]:
            for t in range(num_periods):
                Y[(i, p, t)] = model.addVar(vtype=GRB.BINARY, name=f"Y_{i}_{p}_{t}")
    
    # X[item, plant, period] - amount of item i produced at plant p in period t
    X = {}
    for i in range(num_items):
        for p in plant_capabilities[i]:
            for t in range(num_periods):
                X[(i, p, t)] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"X_{i}_{p}_{t}")

    # I[item, plant, period] - inventory of item i at plant p at end of period t
    I = {}
    for i in range(num_items):
        for p in plant_capabilities[i]:
            for t in range(num_periods):
                I[(i, p, t)] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"I_{i}_{p}_{t}")

    # T[item, customer, plant, period] - amount of item i shipped from plant p to customer j in period t
    T = {}
    for i in range(num_items):
        for j in range(num_customers):
            if i in customer_item_demands[j]:  # Only if customer requests item i
                for p in plant_capabilities[i]:
                    for t in range(num_periods):
                        T[(i, j, p, t)] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"T_{i}_{j}_{p}_{t}")
            else:
                # Dummy vars for items customer doesn't request
                for p in plant_capabilities[i]:
                    for t in range(num_periods):
                        T[(i, j, p, t)] = 0

    model.update()

    # Constraints
    
    # Each item is produced by exactly one plant
    for i in range(num_items):
        model.addConstr(gp.quicksum(U[(i, p)] for p in plant_capabilities[i]) == 1, f"one_assignment_{i}")

    # A plant must be open to produce anything
    for i in range(num_items):
        for p in plant_capabilities[i]:
            model.addConstr(U[(i, p)] <= Z[p], f"PlantOpen_{i}_{p}")

    # Production is only allowed if both item is assigned to the plant and a setup occurs
    for i in range(num_items):
        for p in plant_capabilities[i]:
            for t in range(num_periods):
                model.addConstr(X[(i, p, t)] <= item_total_demand[i][t] * Y[(i, p, t)], 
                                f"ProductionSetupLink_{i}_{p}_{t}")
                
                # Also link production to item assignment
                model.addConstr(X[(i, p, t)] <= item_total_demand[i][t] * U[(i, p)], 
                                f"ProductionAssignmentLink_{i}_{p}_{t}")

    # Capacity constraints (time capacity per period per plant)
    for p in range(num_plants):
        for t in range(num_periods):
            time_expr = gp.LinExpr()
            for i in range(num_items):
                if p in plant_capabilities[i]:
                    time_expr += Y[(i, p, t)] * setup_time[i][p][t]
                    time_expr += X[(i, p, t)] * production_time[i][p][t]
            
            model.addConstr(time_expr <= capacity[p][t] * Z[p], 
                            f"Capacity_{p}_{t}")

    # Transportation equals production allocation among customers
    for i in range(num_items):
        for p in plant_capabilities[i]:
            for j in range(num_customers):
                if i in customer_item_demands[j]:
                    for t in range(num_periods):
                        # Make sure transportation from this plant does not exceed what was produced at this plant
                        model.addConstr(T[(i, j, p, t)] <= X[(i, p, t)], 
                                        f"TransportProduction_{i}_{j}_{p}_{t}")

    # Inventory flow conservation
    for i in range(num_items):
        for p in plant_capabilities[i]:
            for t in range(num_periods):
                inflow = X[(i, p, t)]
                
                outflow = gp.quicksum(T[(i, j, p, t)] for j in range(num_customers) 
                                     if (i, j, p, t) in T)  # Filter to avoid dummy vars
                
                if t == 0:
                    inventory_in = 0
                else:
                    inventory_in = I[(i, p, t-1)]
                    
                inventory_out = I[(i, p, t)]
                
                model.addConstr(inventory_in + inflow == outflow + inventory_out, 
                                f"InventoryFlow_{i}_{p}_{t}")

    # Demand satisfaction constraints
    for i in range(num_items):
        for j in range(num_customers):
            if i in customer_item_demands[j]:
                for t in range(num_periods):
                    model.addConstr(
                        gp.quicksum(T[(i, j, p, t)] for p in plant_capabilities[i]) == demand[i][j][t],
                        f"DemandSatisfaction_{i}_{j}_{t}"
                    )

    # Cumulative production meets or exceeds cumulative demand
    for i in range(num_items):
        for p in plant_capabilities[i]:
            for t in range(num_periods):
                cum_prod = gp.quicksum(X[(i, p, tau)] for tau in range(t+1))
                
                cum_demand = sum(sum(demand[i][j][tau] for j in range(num_customers)) 
                                for tau in range(t+1))
                
                # This constraint ensures that cumulative production can't exceed cumulative demand 
                # times whether the item is assigned to this plant
                model.addConstr(cum_prod <= cum_demand * U[(i, p)], 
                                f"CumulativeProdLimit_{i}_{p}_{t}")

    # Objective function components
    total_setup_cost = gp.quicksum(setup_cost[i][p][t] * Y[(i, p, t)] 
                                  for i in range(num_items) 
                                  for p in plant_capabilities[i] 
                                  for t in range(num_periods))

    total_production_cost = gp.quicksum(production_cost[i][p][t] * X[(i, p, t)] 
                                       for i in range(num_items) 
                                       for p in plant_capabilities[i] 
                                       for t in range(num_periods))

    total_holding_cost = gp.quicksum(holding_cost[i][p][t] * I[(i, p, t)] 
                                    for i in range(num_items) 
                                    for p in plant_capabilities[i] 
                                    for t in range(num_periods))

    total_transportation_cost = gp.quicksum(transportation_cost[i][j][p][t] * T[(i, j, p, t)] 
                                           for i in range(num_items) 
                                           for j in range(num_customers) 
                                           if i in customer_item_demands[j]
                                           for p in plant_capabilities[i] 
                                           for t in range(num_periods))

    total_plant_opening_cost = gp.quicksum(plant_opening_cost[p] * Z[p] 
                                          for p in range(num_plants))

    # Set objective
    model.setObjective(total_setup_cost + 
                       total_production_cost + 
                       total_holding_cost + 
                       total_transportation_cost + 
                       total_plant_opening_cost,
                       GRB.MINIMIZE)

    # Solve
    model.optimize(callback)
    
    # Prepare solution
    solution = {
        "objective_value": float(model.objVal) if model.solCount > 0 else float('inf'),
        "Z": {},
        "U": {},
        "Y": {}
    }
    
    # Fill in solution values
    if model.solCount > 0:
        for p in Z:
            solution["Z"][str(p)] = int(Z[p].x)
        
        for key in U:
            if isinstance(U[key], gp.Var):  # Only for actual variables, not dummy 0
                i, p = key
                solution["U"][f"{i}_{p}"] = int(U[key].x)
        
        for key in Y:
            i, p, t = key
            solution["Y"][f"{i}_{p}_{t}"] = int(Y[key].x)
    
    # Write solution
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


def main():
    parser = argparse.ArgumentParser(description='Solve Supply Chain Optimization Problem')
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the input instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path to save the output solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, required=False,
                        help='Path to save the solution log')

    args = parser.parse_args()

    solve_supply_chain(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()