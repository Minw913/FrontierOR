import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve(instance, time_limit, logger):
    num_retailers = instance['num_retailers']
    num_warehouses = instance['num_warehouses']
    num_periods = instance['num_periods']
    retailer_warehouse_assignment = instance['retailer_warehouse_assignment']
    warehouse_retailer_sets = instance['warehouse_retailer_sets']
    demands = instance['demands']
    fixed_costs = instance['fixed_costs']
    holding_costs = instance['holding_costs']

    model = Model('supply_chain')
    model.setParam("Threads", 1)
    model.Params.LogToConsole = 0
    model.Params.TimeLimit = time_limit

    # Variables
    x_plant = model.addVars(num_periods, vtype=GRB.CONTINUOUS, name='x_plant')
    s_plant = model.addVars(num_periods, vtype=GRB.CONTINUOUS, name='s_plant')
    y_plant = model.addVars(num_periods, vtype=GRB.BINARY, name='y_plant')
    x_warehouses = model.addVars(num_warehouses, num_periods, vtype=GRB.CONTINUOUS, name='x_warehouses')
    s_warehouses = model.addVars(num_warehouses, num_periods, vtype=GRB.CONTINUOUS, name='s_warehouses')
    y_warehouses = model.addVars(num_warehouses, num_periods, vtype=GRB.BINARY, name='y_warehouses')
    x_retailers = model.addVars(num_retailers, num_periods, vtype=GRB.CONTINUOUS, name='x_retailers')
    s_retailers = model.addVars(num_retailers, num_periods, vtype=GRB.CONTINUOUS, name='s_retailers')
    y_retailers = model.addVars(num_retailers, num_periods, vtype=GRB.BINARY, name='y_retailers')

    # Objective
    model.setObjective(
        quicksum(fixed_costs['plant'][t] * y_plant[t] for t in range(num_periods)) +
        quicksum(fixed_costs['warehouses'][w][t] * y_warehouses[w, t] for w in range(num_warehouses) for t in range(num_periods)) +
        quicksum(fixed_costs['retailers'][r][t] * y_retailers[r, t] for r in range(num_retailers) for t in range(num_periods)) +
        holding_costs['plant'] * quicksum(s_plant[t] for t in range(num_periods)) +
        holding_costs['warehouses'] * quicksum(s_warehouses[w, t] for w in range(num_warehouses) for t in range(num_periods)) +
        quicksum(holding_costs['retailers'][r] * s_retailers[r, t] for r in range(num_retailers) for t in range(num_periods)),
        GRB.MINIMIZE
    )

    # Constraints
    for t in range(num_periods):
        # Plant production and inventory balance
        if t == 0:
            model.addConstr(s_plant[t] == x_plant[t] - quicksum(x_warehouses[w, t] for w in range(num_warehouses)))
        else:
            model.addConstr(s_plant[t] == s_plant[t-1] + x_plant[t] - quicksum(x_warehouses[w, t] for w in range(num_warehouses)))
        model.addConstr(x_plant[t] <= y_plant[t] * sum(demands[r][tau] for r in range(num_retailers) for tau in range(t, num_periods)))
        
        # Warehouse inventory balance and shipment
        for w in range(num_warehouses):
            if t == 0:
                model.addConstr(s_warehouses[w, t] == x_warehouses[w, t] - quicksum(x_retailers[r, t] for r in warehouse_retailer_sets[str(w)]))
            else:
                model.addConstr(s_warehouses[w, t] == s_warehouses[w, t-1] + x_warehouses[w, t] - quicksum(x_retailers[r, t] for r in warehouse_retailer_sets[str(w)]))
            model.addConstr(x_warehouses[w, t] <= y_warehouses[w, t] * sum(demands[r][tau] for r in warehouse_retailer_sets[str(w)] for tau in range(t, num_periods)))

        # Retailer inventory balance and demand satisfaction
        for r in range(num_retailers):
            w = retailer_warehouse_assignment[r]
            if t == 0:
                model.addConstr(s_retailers[r, t] == x_retailers[r, t] - demands[r][t])
            else:
                model.addConstr(s_retailers[r, t] == s_retailers[r, t-1] + x_retailers[r, t] - demands[r][t])
            model.addConstr(x_retailers[r, t] <= y_retailers[r, t] * sum(demands[r][tau] for tau in range(t, num_periods)))

    model.optimize()
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.ObjVal
        solution = {
            'objective_value': objective_value,
            'x_plant': [x_plant[t].X for t in range(num_periods)],
            's_plant': [s_plant[t].X for t in range(num_periods)],
            'y_plant': [y_plant[t].X for t in range(num_periods)],
            'x_warehouses': [[x_warehouses[w, t].X for t in range(num_periods)] for w in range(num_warehouses)],
            's_warehouses': [[s_warehouses[w, t].X for t in range(num_periods)] for w in range(num_warehouses)],
            'y_warehouses': [[y_warehouses[w, t].X for t in range(num_periods)] for w in range(num_warehouses)],
            'x_retailers': [[x_retailers[r, t].X for t in range(num_periods)] for r in range(num_retailers)],
            's_retailers': [[s_retailers[r, t].X for t in range(num_periods)] for r in range(num_retailers)],
            'y_retailers': [[y_retailers[r, t].X for t in range(num_periods)] for r in range(num_retailers)]
        }
        if logger:
            logger.log(objective_value)
        return solution
    else:
        return None

def write_solution(solution_path, solution):
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)
    solution = solve(instance, args.time_limit, logger)
    if solution is not None:
        write_solution(args.solution_path, solution)

if __name__ == '__main__':
    main()