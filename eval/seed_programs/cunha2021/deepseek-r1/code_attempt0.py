import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    num_retailers = instance['num_retailers']
    num_warehouses = instance['num_warehouses']
    num_periods = instance['num_periods']
    demands = instance['demands']
    fixed_costs = instance['fixed_costs']
    holding_costs = instance['holding_costs']
    assignment = instance['retailer_warehouse_assignment']
    
    # Precompute remaining demands
    # For retailers
    retailer_remaining = []
    for r in range(num_retailers):
        rem = [0] * num_periods
        total = 0
        for t in range(num_periods-1, -1, -1):
            total += demands[r][t]
            rem[t] = total
        retailer_remaining.append(rem)
    
    # For warehouses: group retailers by warehouse
    retailers_per_warehouse = [[] for _ in range(num_warehouses)]
    for r in range(num_retailers):
        w = assignment[r]
        retailers_per_warehouse[w].append(r)
    
    warehouse_remaining = []
    for w in range(num_warehouses):
        rem = [0] * num_periods
        for t in range(num_periods):
            total = 0
            for r in retailers_per_warehouse[w]:
                total += retailer_remaining[r][t]
            rem[t] = total
        warehouse_remaining.append(rem)
    
    # For plant
    plant_remaining = [0] * num_periods
    for t in range(num_periods):
        total = 0
        for w in range(num_warehouses):
            total += warehouse_remaining[w][t]
        plant_remaining[t] = total
    
    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Create model
    model = gp.Model('supply_chain')
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    
    # Decision variables
    # Plant variables
    x_plant = model.addVars(num_periods, lb=0.0, name="x_plant")
    s_plant = model.addVars(num_periods, lb=0.0, name="s_plant")
    y_plant = model.addVars(num_periods, vtype=GRB.BINARY, name="y_plant")
    
    # Warehouse variables
    x_warehouses = {}
    s_warehouses = {}
    y_warehouses = {}
    for w in range(num_warehouses):
        x_warehouses[w] = model.addVars(num_periods, lb=0.0, name=f"x_warehouses_{w}")
        s_warehouses[w] = model.addVars(num_periods, lb=0.0, name=f"s_warehouses_{w}")
        y_warehouses[w] = model.addVars(num_periods, vtype=GRB.BINARY, name=f"y_warehouses_{w}")
    
    # Retailer variables
    x_retailers = {}
    s_retailers = {}
    y_retailers = {}
    for r in range(num_retailers):
        x_retailers[r] = model.addVars(num_periods, lb=0.0, name=f"x_retailers_{r}")
        s_retailers[r] = model.addVars(num_periods, lb=0.0, name=f"s_retailers_{r}")
        y_retailers[r] = model.addVars(num_periods, vtype=GRB.BINARY, name=f"y_retailers_{r}")
    
    # Constraints
    # Plant flow balance
    model.addConstr(s_plant[0] == x_plant[0] - gp.quicksum(x_warehouses[w][0] for w in range(num_warehouses)))
    for t in range(1, num_periods):
        model.addConstr(s_plant[t-1] + x_plant[t] == 
                        gp.quicksum(x_warehouses[w][t] for w in range(num_warehouses)) + s_plant[t])
    
    # Warehouse flow balance
    for w in range(num_warehouses):
        retailers_in_w = retailers_per_warehouse[w]
        model.addConstr(0 + x_warehouses[w][0] == 
                        gp.quicksum(x_retailers[r][0] for r in retailers_in_w) + s_warehouses[w][0])
        for t in range(1, num_periods):
            model.addConstr(s_warehouses[w][t-1] + x_warehouses[w][t] == 
                            gp.quicksum(x_retailers[r][t] for r in retailers_in_w) + s_warehouses[w][t])
    
    # Retailer flow balance
    for r in range(num_retailers):
        model.addConstr(0 + x_retailers[r][0] == demands[r][0] + s_retailers[r][0])
        for t in range(1, num_periods):
            model.addConstr(s_retailers[r][t-1] + x_retailers[r][t] == demands[r][t] + s_retailers[r][t])
    
    # Setup forcing constraints
    for t in range(num_periods):
        model.addConstr(x_plant[t] <= plant_remaining[t] * y_plant[t])
    
    for w in range(num_warehouses):
        for t in range(num_periods):
            model.addConstr(x_warehouses[w][t] <= warehouse_remaining[w][t] * y_warehouses[w][t])
    
    for r in range(num_retailers):
        for t in range(num_periods):
            model.addConstr(x_retailers[r][t] <= retailer_remaining[r][t] * y_retailers[r][t])
    
    # Objective function
    obj = gp.quicksum(
        fixed_costs['plant'][t] * y_plant[t] + 
        holding_costs['plant'] * s_plant[t] 
        for t in range(num_periods)
    )
    
    for w in range(num_warehouses):
        obj += gp.quicksum(
            fixed_costs['warehouses'][w][t] * y_warehouses[w][t] + 
            holding_costs['warehouses'] * s_warehouses[w][t]
            for t in range(num_periods)
        )
    
    for r in range(num_retailers):
        obj += gp.quicksum(
            fixed_costs['retailers'][r][t] * y_retailers[r][t] + 
            holding_costs['retailers'][r] * s_retailers[r][t]
            for t in range(num_periods)
        )
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Optimize with callback for logging
    if logger:
        model._logger = logger
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                model._logger.log(obj_val)
        model.optimize(callback)
    else:
        model.optimize()
    
    # Extract solution or use trivial solution if no solution found
    if model.SolCount > 0:
        solution = {
            'objective_value': model.ObjVal,
            'x_plant': [x_plant[t].X for t in range(num_periods)],
            's_plant': [s_plant[t].X for t in range(num_periods)],
            'y_plant': [int(round(y_plant[t].X)) for t in range(num_periods)],
            'x_warehouses': [],
            's_warehouses': [],
            'y_warehouses': [],
            'x_retailers': [],
            's_retailers': [],
            'y_retailers': []
        }
        
        for w in range(num_warehouses):
            solution['x_warehouses'].append([x_warehouses[w][t].X for t in range(num_periods)])
            solution['s_warehouses'].append([s_warehouses[w][t].X for t in range(num_periods)])
            solution['y_warehouses'].append([int(round(y_warehouses[w][t].X)) for t in range(num_periods)])
        
        for r in range(num_retailers):
            solution['x_retailers'].append([x_retailers[r][t].X for t in range(num_periods)])
            solution['s_retailers'].append([s_retailers[r][t].X for t in range(num_periods)])
            solution['y_retailers'].append([int(round(y_retailers[r][t].X)) for t in range(num_periods)])
    else:
        # Construct trivial solution
        solution = {
            'x_plant': [0.0] * num_periods,
            's_plant': [0.0] * num_periods,
            'y_plant': [0] * num_periods,
            'x_warehouses': [[0.0] * num_periods for _ in range(num_warehouses)],
            's_warehouses': [[0.0] * num_periods for _ in range(num_warehouses)],
            'y_warehouses': [[0] * num_periods for _ in range(num_warehouses)],
            'x_retailers': [[0.0] * num_periods for _ in range(num_retailers)],
            's_retailers': [[0.0] * num_periods for _ in range(num_retailers)],
            'y_retailers': [[0] * num_periods for _ in range(num_retailers)],
            'objective_value': 0.0
        }
        
        # Calculate trivial solution costs
        total_cost = 0
        
        # Plant: produce everything in period 0
        solution['x_plant'][0] = plant_remaining[0]
        solution['y_plant'][0] = 1
        total_cost += fixed_costs['plant'][0]
        
        # Warehouses: receive everything in period 0
        for w in range(num_warehouses):
            solution['x_warehouses'][w][0] = warehouse_remaining[w][0]
            solution['y_warehouses'][w][0] = 1
            total_cost += fixed_costs['warehouses'][w][0]
        
        # Retailers: receive everything in period 0 and hold inventory
        for r in range(num_retailers):
            solution['x_retailers'][r][0] = retailer_remaining[r][0]
            solution['y_retailers'][r][0] = 1
            total_cost += fixed_costs['retailers'][r][0]
            
            # Compute inventory levels
            inv = retailer_remaining[r][0] - demands[r][0]
            solution['s_retailers'][r][0] = inv
            total_cost += holding_costs['retailers'][r] * inv
            
            for t in range(1, num_periods):
                inv -= demands[r][t]
                solution['s_retailers'][r][t] = inv
                total_cost += holding_costs['retailers'][r] * inv
        
        solution['objective_value'] = total_cost
    
    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()