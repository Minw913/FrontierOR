import json
import argparse
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def build_model(instance):
    n, m, p = instance['dimensions']['n'], instance['dimensions']['m'], instance['dimensions']['p']
    demands = instance['demands']
    inventory_costs = instance['inventory_costs']
    processing_time = instance['processing_time']
    setup_times = instance['setup_times']
    setup_costs = instance['setup_costs']
    machine_capacities = instance['machine_capacities']
    max_production = instance['max_production']
    max_setups_per_item = instance['max_setups_per_item']
    minimum_lot_sizes = instance['minimum_lot_sizes']
    machine_eligibility = instance['machine_eligibility']

    model = Model("production_planning")
    model.setParam("Threads", 1)
    model.modelSense = GRB.MINIMIZE

    # Production variables: x_{item}_{machine}_{production_period}_{demand_period}
    x = {}
    for item in range(n):
        for machine in range(m):
            for production_period in range(p):
                for demand_period in range(production_period, p):
                    x[f"x_{item}_{machine}_{production_period}_{demand_period}"] = model.addVar(vtype=GRB.INTEGER, name=f"x_{item}_{machine}_{production_period}_{demand_period}")

    # Setup variables: y_{from_item}_{to_item}_{machine}_{period}
    y = {}
    for from_item in range(n):
        for to_item in range(n):
            for machine in range(m):
                for period in range(p):
                    y[f"y_{from_item}_{to_item}_{machine}_{period}"] = model.addVar(vtype=GRB.INTEGER, name=f"y_{from_item}_{to_item}_{machine}_{period}")

    # Carryover variables: z_{item}_{machine}_{period}
    z = {}
    for item in range(n):
        for machine in range(m):
            for period in range(p):
                z[f"z_{item}_{machine}_{period}"] = model.addVar(vtype=GRB.BINARY, name=f"z_{item}_{machine}_{period}")

    # Objective function: minimize total cost
    model.setObjective(
        quicksum(inventory_costs[item] * (demand_period - production_period) * x[f"x_{item}_{machine}_{production_period}_{demand_period}"] 
                 for item in range(n) for machine in range(m) for production_period in range(p) for demand_period in range(production_period, p)) +
        quicksum(setup_costs[from_item][to_item][machine] * y[f"y_{from_item}_{to_item}_{machine}_{period}"] 
                 for from_item in range(n) for to_item in range(n) for machine in range(m) for period in range(p))
    )

    # Demand satisfaction constraints
    for item in range(n):
        for demand_period in range(p):
            model.addConstr(quicksum(x[f"x_{item}_{machine}_{production_period}_{demand_period}"] 
                                     for machine in range(m) for production_period in range(demand_period + 1)) == demands[item][demand_period])

    # Machine capacity constraints
    for machine in range(m):
        for period in range(p):
            model.addConstr(quicksum(processing_time * x[f"x_{item}_{machine}_{production_period}_{demand_period}"] 
                                     for item in range(n) for production_period in range(period, p) for demand_period in range(production_period, p)) +
                            quicksum(setup_times[from_item][to_item][machine] * y[f"y_{from_item}_{to_item}_{machine}_{period}"] 
                                     for from_item in range(n) for to_item in range(n)) <= machine_capacities[machine][period])

    # Maximum production constraints
    for item in range(n):
        for production_period in range(p):
            model.addConstr(quicksum(x[f"x_{item}_{machine}_{production_period}_{demand_period}"] 
                                     for machine in range(m) for demand_period in range(production_period, p)) <= max_production[item][production_period])

    # Setup and carryover constraints
    for item in range(n):
        for machine in range(m):
            for period in range(p):
                model.addConstr(quicksum(y[f"y_{from_item}_{item}_{machine}_{period}"] for from_item in range(n)) + z.get(f"z_{item}_{machine}_{period-1}", 0) == 
                                quicksum(y[f"y_{item}_{to_item}_{machine}_{period}"] for to_item in range(n)) + z[f"z_{item}_{machine}_{period}"])

    # Maximum setups per item constraint
    for item in range(n):
        for machine in range(m):
            for period in range(p):
                model.addConstr(quicksum(y[f"y_{from_item}_{item}_{machine}_{period}"] for from_item in range(n)) <= max_setups_per_item[item][machine][period])

    # Minimum lot size constraints for new setups
    for item in range(n):
        for machine in range(m):
            for period in range(p):
                for from_item in range(n):
                    if from_item != item:
                        model.addConstr(quicksum(x[f"x_{item}_{machine}_{period}_{demand_period}"] for demand_period in range(period, p)) >= 
                                        minimum_lot_sizes[item] * y[f"y_{from_item}_{item}_{machine}_{period}"])

    # Eligibility constraints
    for item in range(n):
        for machine in range(m):
            if machine_eligibility[item][machine] == 0:
                for period in range(p):
                    model.addConstr(z[f"z_{item}_{machine}_{period}"] == 0)
                    for demand_period in range(period, p):
                        model.addConstr(x[f"x_{item}_{machine}_{period}_{demand_period}"] == 0)
                    for to_item in range(n):
                        model.addConstr(y[f"y_{item}_{to_item}_{machine}_{period}"] == 0)
                    for from_item in range(n):
                        model.addConstr(y[f"y_{from_item}_{item}_{machine}_{period}"] == 0)

    return model, x, y, z

def extract_solution(model, x, y, z):
    solution = {}
    solution['objective_value'] = model.ObjVal
    solution['production'] = {var.VarName: var.X for var in model.getVars() if var.VarName.startswith('x_') and var.X > 0}
    solution['setups'] = {var.VarName: var.X for var in model.getVars() if var.VarName.startswith('y_') and var.X > 0}
    solution['carryover'] = {var.VarName: var.X for var in model.getVars() if var.VarName.startswith('z_') and var.X > 0}
    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    model, x, y, z = build_model(instance)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model.setParam('TimeLimit', args.time_limit)
    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        solution = extract_solution(model, x, y, z)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
        if logger:
            logger.log(solution['objective_value'])
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()