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
    
    # Set time limit and callbacks for logging
    model.setParam('TimeLimit', time_limit)
    
    if log_path:
        logger = SolutionLogger(log_path, sense="minimize")
        
        def mycallback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj)
        
        model.Params.LogToConsole = 1
    
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
        
        # Extract relevant variables after optimization
        # Setup variables: y[facility][period] where facility=0 is plant, 1..W are warehouses, W+1..W+R are retailers
        for p in range(num_periods):
            setups[f'y_0_{p}'] = round(y_plant[p].X)
            
        for w in range(num_warehouses):
            for p in range(num_periods):
                setups[f'y_{w+1}_{p}'] = round(y_warehouse[w][p].X)
                
        for r in range(num_retailers):
            for p in range(num_periods):
                setups[f'y_{r+num_warehouses+1}_{p}'] = round(y_retailer[r][p].X)
    
        solution["setup_variables"] = setups
    
        # Aggregate total production at plant for each period/scenario
        for t in range(num_periods):
            for s in range(num_scenarios):
                prod_sum = sum(q_plant_per_demand[r][t][d][s] for r in range(num_retailers) for d in range(num_periods))
                productions[f'{t}_{s}'] = round(prod_sum.X)
    
        # Aggregate delivery quantities
        for w in range(num_warehouses):
            for t in range(num_periods):
                for s in range(num_scenarios):
                    del_sum = sum(q_warehouse_per_demand[r][t][d][s] for r in range(num_retailers) 
                                if retailer_to_warehouse_idx[r] == w for d in range(num_periods))
                    deliveries_w[f'{w}_{t}_{s}'] = round(del_sum.X)
                    
        for r in range(num_retailers):
            for t in range(num_periods):
                for s in range(num_scenarios):
                    del_sum = sum(q_retailer_per_demand[r][t][d][s] for d in range(num_periods))
                    deliveries_r[f'{r}_{t}_{s}'] = round(del_sum.X)
    
        # Calculate and store inventory levels
        # Plant inventory
        for t in range(num_periods):
            for s in range(num_scenarios):
                inv_sum = sum(i_plant_per_demand[r][t][d][s] for r in range(num_retailers) for d in range(num_periods))  
                inv_p[f'{t}_{s}'] = round(inv_sum.X)
    
        # Warehouse inventory
        for w in range(num_warehouses):
            for t in range(num_periods):
                for s in range(num_scenarios):
                    inv_sum = sum(i_warehouse_per_demand[r][t][d][s] for r in range(num_retailers) 
                                 if retailer_to_warehouse_idx[r] == w for d in range(num_periods))
                    inv_w[f'{w}_{t}_{s}'] = round(inv_sum.X)
    
        # Retailer inventory
        for r in range(num_retailers):
            for t in range(num_periods):
                for s in range(num_scenarios):
                    inv_sum = sum(i_retailer_per_demand[r][t][d][s] for d in range(num_periods))
                    inv_r[f'{r}_{t}_{s}'] = round(inv_sum.X)
    
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