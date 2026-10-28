import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_stochastic_three_level_supply_chain(instance_data, time_limit, log_path):
    # Parse input data
    num_periods = instance_data["num_periods"]
    num_warehouses = instance_data["num_warehouses"]
    num_retailers = instance_data["num_retailers"]
    num_scenarios = instance_data["num_scenarios"]
    initial_inventory = instance_data["initial_inventory"]
    initial_setups_imposed = instance_data["initial_setups_imposed"]
    retailer_to_warehouse_idx = instance_data["retailer_to_warehouse"]
    
    scenario_probabilities = instance_data["scenario_probabilities"]
    holding_cost_plant = instance_data["holding_cost_plant"]
    holding_cost_warehouse = instance_data["holding_cost_warehouse"]
    holding_costs_retailers = instance_data["holding_costs_retailers"]
    
    setup_costs_plant = instance_data["setup_costs_plant"]
    setup_costs_warehouses = instance_data["setup_costs_warehouses"]  # [warehouse][period]
    setup_costs_retailers = instance_data["setup_costs_retailers"]  # [retailer][period]
    
    demands = instance_data["demands"]  # [scenario][retailer][period]

    # Create model
    model = gp.Model("stochastic_three_level_supply_chain")
    model.setParam("Threads", 1)
    model.setParam('OutputFlag', 0)  # Suppress Gurobi output in script
    
    # Variables for tracking flows and inventory per demand
    # q_plant_per_demand[retailer][prod_period][demand_period][scenario]
    q_plant_per_demand = {}
    for r in range(num_retailers):
        q_plant_per_demand[r] = {}
        for t_prod in range(num_periods):
            q_plant_per_demand[r][t_prod] = {}
            for d in range(num_periods):
                q_plant_per_demand[r][t_prod][d] = {}
                for s in range(num_scenarios):
                    q_plant_per_demand[r][t_prod][d][s] = model.addVar(
                        vtype=GRB.CONTINUOUS,
                        name=f'q_plant_{r}_{t_prod}_{d}_{s}',
                        lb=0
                    )
    
    # i_plant_per_demand[retailer][at_period][demand_period][scenario]
    i_plant_per_demand = {}
    for r in range(num_retailers):
        i_plant_per_demand[r] = {}
        for t_curr in range(num_periods):
            i_plant_per_demand[r][t_curr] = {}
            for d in range(num_periods):
                i_plant_per_demand[r][t_curr][d] = {}
                for s in range(num_scenarios):
                    i_plant_per_demand[r][t_curr][d][s] = model.addVar(
                        vtype=GRB.CONTINUOUS,
                        name=f'i_plant_{r}_{t_curr}_{d}_{s}',
                        lb=0
                    )

    # q_warehouse_per_demand
    q_warehouse_per_demand = {}
    for r in range(num_retailers):
        q_warehouse_per_demand[r] = {}
        for t_ord in range(num_periods):
            q_warehouse_per_demand[r][t_ord] = {}
            for d in range(num_periods):
                q_warehouse_per_demand[r][t_ord][d] = {}
                for s in range(num_scenarios):
                    q_warehouse_per_demand[r][t_ord][d][s] = model.addVar(
                        vtype=GRB.CONTINUOUS,
                        name=f'q_warehouse_{r}_{t_ord}_{d}_{s}',
                        lb=0
                    )
                    
    # i_warehouse_per_demand
    i_warehouse_per_demand = {}
    for r in range(num_retailers):
        i_warehouse_per_demand[r] = {}
        for t_curr in range(num_periods):
            i_warehouse_per_demand[r][t_curr] = {}
            for d in range(num_periods):
                i_warehouse_per_demand[r][t_curr][d] = {}
                for s in range(num_scenarios):
                    i_warehouse_per_demand[r][t_curr][d][s] = model.addVar(
                        vtype=GRB.CONTINUOUS,
                        name=f'i_warehouse_{r}_{t_curr}_{d}_{s}',
                        lb=0
                    ) 

    # q_retailer_per_demand
    q_retailer_per_demand = {}
    for r in range(num_retailers):
        q_retailer_per_demand[r] = {}
        for t_ord in range(num_periods):
            q_retailer_per_demand[r][t_ord] = {}
            for d in range(num_periods):
                q_retailer_per_demand[r][t_ord][d] = {}
                for s in range(num_scenarios):
                    q_retailer_per_demand[r][t_ord][d][s] = model.addVar(
                        vtype=GRB.CONTINUOUS,
                        name=f'q_retailer_{r}_{t_ord}_{d}_{s}',
                        lb=0
                    )
                    
    # i_retailer_per_demand
    i_retailer_per_demand = {}
    for r in range(num_retailers):
        i_retailer_per_demand[r] = {}
        for t_curr in range(num_periods):
            i_retailer_per_demand[r][t_curr] = {}
            for d in range(num_periods):
                i_retailer_per_demand[r][t_curr][d] = {}
                for s in range(num_scenarios):
                    i_retailer_per_demand[r][t_curr][d][s] = model.addVar(
                        vtype=GRB.CONTINUOUS,
                        name=f'i_retailer_{r}_{t_curr}_{d}_{s}',
                        lb=0
                    )
    
    # Binary setup variables
    y_plant = []  # y[period]
    for t in range(num_periods):
        y_plant.append(model.addVar(vtype=GRB.BINARY, name=f'y_plant_{t}'))

    y_warehouse = []  # y[warehouse][period]
    for w in range(num_warehouses):
        y_warehouse.append([])
        for t in range(num_periods):
            y_warehouse[w].append(model.addVar(vtype=GRB.BINARY, name=f'y_warehouse_{w}_{t}'))

    y_retailer = []  # y[retailer][period]
    for r in range(num_retailers):
        y_retailer.append([])
        for t in range(num_periods):
            y_retailer[r].append(model.addVar(vtype=GRB.BINARY, name=f'y_retailer_{r}_{t}'))

    # Inventory conservation constraints at plant level
    for r in range(num_retailers):
        for d in range(num_periods):  # demand period
            for s in range(num_scenarios):
                for t in range(num_periods):  # current period
                    lhs_items = []
                    # Add delivery to warehouse in period t for retailer-period-demand r,d,s
                    if retailer_to_warehouse_idx[r] == retailer_to_warehouse_idx[r]:  # This condition will always be True for self-checking
                        # Adding delivery term (we'll handle mapping outside of constraint definition properly)
                        for w_idx in range(num_warehouses):
                            if retailer_to_warehouse_idx[r] == w_idx:
                                lhs_items.append(q_warehouse_per_demand[r][t][d][s])
                    
                    lhs_items.append(i_plant_per_demand[r][t][d][s])  # inventory end of period t
                    
                    rhs_items = []
                    # Initial inventory at period 0
                    if t == 0:
                        if initial_inventory > 0:
                            raise ValueError("Initial inventory should be zero in this problem")
                    else:
                        rhs_items.append(i_plant_per_demand[r][t-1][d][s])

                    # Production during period t -> goes toward satisfying demand r,d under scenario s
                    rhs_items.append(q_plant_per_demand[r][t][d][s])

                    model.addConstr(gp.quicksum(lhs_items) == gp.quicksum(rhs_items),
                                   name=f'plant_inv_cons_{r}_{d}_{t}_{s}')

    # Inventory conservation at warehouse level
    for w_idx in range(num_warehouses):
        retailers_at_warehouse = [r for r in range(num_retailers) if retailer_to_warehouse_idx[r] == w_idx]
        for r in retailers_at_warehouse:
            for d in range(num_periods):
                for s in range(num_scenarios):
                    for t in range(num_periods):
                        lhs_items = [
                            q_warehouse_per_demand[r][t][d][s],  # Shipped to retailer in period t
                            i_warehouse_per_demand[r][t][d][s]  # Inventory remaining at end of period t
                        ]
                        
                        rhs_items = []
                        if t > 0:
                            rhs_items.append(i_warehouse_per_demand[r][t-1][d][s])
                        else:
                            pass  # Assume zero starting inventory
                        
                        # Received from plant
                        rhs_items.append(q_warehouse_per_demand[r][t][d][s])

                        model.addConstr(gp.quicksum(lhs_items) == gp.quicksum(rhs_items),
                                       name=f'warehouse_inv_cons_{r}_{d}_{t}_{s}')


    # Correctly defined inventory conservation at retailer level
    for r in range(num_retailers):
        for d in range(num_periods):
            for s in range(num_scenarios):
                for t in range(num_periods):
                    lhs_items = [
                        i_retailer_per_demand[r][t][d][s]  # End inventory for this demand
                    ]
                    
                    if t == d:  # Demand realization period: consume the demand amount
                        lhs_items.append(demands[s][r][d])  # Added demand consumption term to the left side

                    rhs_items = []

                    if t > 0:
                        rhs_items.append(i_retailer_per_demand[r][t-1][d][s])

                    # Arrival from warehouse
                    rhs_items.append(q_retailer_per_demand[r][t][d][s])

                    model.addConstr(gp.quicksum(lhs_items) == gp.quicksum(rhs_items),
                                   name=f'retailer_inv_cons_{r}_{d}_{t}_{s}')


    # Constraint links: flows to setup decisions
    demand_upper_bound = max(max(sum(demands[s][r][t] for r in range(num_retailers) for t in range(num_periods)) for s in range(num_scenarios)),
                             1)

    for r in range(num_retailers):
        for t_prod in range(num_periods):
            for d in range(num_periods):
                for s in range(num_scenarios):
                    # Production cannot exceed setup activation
                    demand_value = demands[s][r][d]
                    model.addConstr(
                        q_plant_per_demand[r][t_prod][d][s] <= 
                        demand_value * y_plant[t_prod],
                        name=f'plant_setup_link_{r}_{t_prod}_{d}_{s}'
                    )
                    
    for r in range(num_retailers):
        for t_ord in range(num_periods):
            for d in range(num_periods):
                for s in range(num_scenarios):
                    w_idx = retailer_to_warehouse_idx[r]
                    demand_value = demands[s][r][d]
                    # Orders at warehouse cannot exceed setup activation
                    model.addConstr(
                        q_warehouse_per_demand[r][t_ord][d][s] <= 
                        demand_value * y_warehouse[w_idx][t_ord],
                        name=f'warehouse_setup_link_{r}_{t_ord}_{d}_{s}'
                    )
                    
    for r in range(num_retailers):
        for t_ord in range(num_periods):
            for d in range(num_periods):
                for s in range(num_scenarios):
                    demand_value = demands[s][r][d]
                    # Orders at retailer cannot exceed setup activation
                    model.addConstr(
                        q_retailer_per_demand[r][t_ord][d][s] <= 
                        demand_value * y_retailer[r][t_ord],
                        name=f'retailer_setup_link_{r}_{t_ord}_{d}_{s}'
                    )


    # Initial setup requirement (if applicable)
    if initial_setups_imposed:
        for t in [0]:
            model.addConstr(y_plant[t] == 1, name=f'init_plant_setup_t{t}')
            for w in range(num_warehouses):
                model.addConstr(y_warehouse[w][t] == 1, name=f'init_warehouse_{w}_setup_t{t}')
            for r in range(num_retailers):
                model.addConstr(y_retailer[r][t] == 1, name=f'init_retailer_{r}_setup_t{t}')


    # Objective function: minimize expected total cost = setup costs + expected inventory costs
    setup_cost = 0
    for t in range(num_periods):
        setup_cost += setup_costs_plant[t] * y_plant[t]
        
    for w in range(num_warehouses):
        for t in range(num_periods):
            setup_cost += setup_costs_warehouses[w][t] * y_warehouse[w][t]
            
    for r in range(num_retailers):
        for t in range(num_periods):
            setup_cost += setup_costs_retailers[r][t] * y_retailer[r][t]
    
    inventory_cost = 0
    for s in range(num_scenarios):
        prob = scenario_probabilities[s]
        
        # Plant inventory holding costs
        for r in range(num_retailers):
            for d in range(num_periods):
                for t in range(num_periods):  # Inventory held from t-1 to t, for items destined for demand d
                    inventory_cost += prob * holding_cost_plant * i_plant_per_demand[r][t][d][s]
        
        # Warehouse inventory holding costs
        for r in range(num_retailers):
            for d in range(num_periods):
                for t in range(num_periods):
                    inventory_cost += prob * holding_cost_warehouse * i_warehouse_per_demand[r][t][d][s]
        
        # Retailer inventory holding costs
        for r in range(num_retailers):
            for d in range(num_periods):
                for t in range(num_periods):
                    inventory_cost += prob * holding_costs_retailers[r] * i_retailer_per_demand[r][t][d][s]
                    
    model.setObjective(setup_cost + inventory_cost, GRB.MINIMIZE)

    # Set time limit and callbacks for logging
    model.setParam('TimeLimit', time_limit)
    
    if log_path:
        logger = SolutionLogger(log_path, sense="minimize")
        
        def mycallback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj)
        
        model.Params.DisplayInterval = 1
        model.optimize(mycallback)
    else:
        model.optimize()

    # Prepare solution dictionary
    solution = {}
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        solution["objective_value"] = model.objVal
        
        setups = {}
        productions = {}
        deliveries_w = {}
        deliveries_r = {}
        inv_p = {}
        inv_w = {}
        inv_r = {}
        
        # Extract binary setup variables
        for p in range(num_periods):
            setups[f'y_0_{p}'] = round(y_plant[p].X)
            
        for w in range(num_warehouses):
            for p in range(num_periods):
                setups[f'y_{w+1}_{p}'] = round(y_warehouse[w][p].X)
                
        for r in range(num_retailers):
            for p in range(num_periods):
                setups[f'y_{r+num_warehouses+1}_{p}'] = round(y_retailer[r][p].X)
    
        solution["setup_variables"] = setups

        # Calculate total production at plant per period/scenario
        for t in range(num_periods):
            for s in range(num_scenarios):
                prod_sum = sum(q_plant_per_demand[r][t][d][s].X 
                              for r in range(num_retailers) 
                              for d in range(num_periods))
                productions[f'{t}_{s}'] = round(prod_sum, 3)
        
        # Calculate total delivery from warehouses per period/scenario 
        for w in range(num_warehouses):
            for t in range(num_periods):
                for s in range(num_scenarios):
                    del_sum = sum(q_warehouse_per_demand[r][t][d][s].X 
                                 for r in range(num_retailers) 
                                 if retailer_to_warehouse_idx[r] == w 
                                 for d in range(num_periods))
                    deliveries_w[f'{w}_{t}_{s}'] = round(del_sum, 3)
                 
        # Calculate total delivery to retailers per period/scenario
        for r in range(num_retailers):
            for t in range(num_periods):
                for s in range(num_scenarios):
                    del_sum = sum(q_retailer_per_demand[r][t][d][s].X 
                                 for d in range(num_periods))
                    deliveries_r[f'{r}_{t}_{s}'] = round(del_sum, 3)
    
        # Aggregate inventory per period/scenario 
        # Plant total inventory
        for t in range(num_periods):
            for s in range(num_scenarios):
                inv_sum = sum(i_plant_per_demand[r][t][d][s].X 
                             for r in range(num_retailers) 
                             for d in range(num_periods))
                inv_p[f'{t}_{s}'] = round(inv_sum, 3)
                
        # Warehouse total inventory
        for w in range(num_warehouses):
            for t in range(num_periods):
                for s in range(num_scenarios):
                    inv_sum = sum(i_warehouse_per_demand[r][t][d][s].X 
                                 for r in range(num_retailers) 
                                 if retailer_to_warehouse_idx[r] == w 
                                 for d in range(num_periods))
                    inv_w[f'{w}_{t}_{s}'] = round(inv_sum, 3)
        
        # Retailer total inventory
        for r in range(num_retailers):
            for t in range(num_periods):
                for s in range(num_scenarios):
                    inv_sum = sum(i_retailer_per_demand[r][t][d][s].X 
                                 for d in range(num_periods))
                    inv_r[f'{r}_{t}_{s}'] = round(inv_sum, 3)

        solution["production_plant"] = productions
        solution["delivery_warehouse"] = deliveries_w
        solution["delivery_retailer"] = deliveries_r
        solution["inventory_plant"] = inv_p
        solution["inventory_warehouse"] = inv_w
        solution["inventory_retailer"] = inv_r
    else:
        # Return empty solution for infeasible or error case
        solution["objective_value"] = float('inf')
        solution["setup_variables"] = {}
        solution["production_plant"] = {}
        solution["delivery_warehouse"] = {}
        solution["delivery_retailer"] = {}
        solution["inventory_plant"] = {}
        solution["inventory_warehouse"] = {}
        solution["inventory_retailer"] = {}

    return solution


def main():
    parser = argparse.ArgumentParser(description='Solve stochastic three-level supply chain problem.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the input instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to the output solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to the logging output JSONL file (optional)')
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    solution = solve_stochastic_three_level_supply_chain(instance_data, args.time_limit, args.log_path)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()