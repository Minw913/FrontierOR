import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def solve(instance):
    num_customers = instance['num_customers']
    num_plants = instance['num_plants']
    demands = instance['customers']['demands']
    fixed_costs = instance['plants']['fixed_costs']
    capacities = instance['plants']['capacities']
    shipping_costs = instance['shipping_costs']

    model = gp.Model('FacilityLocation')
    model.setParam("Threads", 1)

    # Variables: open_plants[i] = 1 if plant i is open, 0 otherwise
    open_plants = model.addVars(num_plants, vtype=GRB.BINARY, name='open_plants')

    # Variables: assignments[c, p] = fraction of customer c's demand served by plant p
    assignments = model.addVars(num_customers, num_plants, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='assignments')

    # Objective: minimize total fixed costs + total shipping costs
    model.setObjective(
        gp.quicksum(fixed_costs[p] * open_plants[p] for p in range(num_plants)) +
        gp.quicksum(shipping_costs[c][p] * assignments[c, p] for c in range(num_customers) for p in range(num_plants)),
        GRB.MINIMIZE
    )

    # Constraints: each customer's demand must be fully met
    for c in range(num_customers):
        model.addConstr(gp.quicksum(assignments[c, p] for p in range(num_plants)) == 1, f'demand_met_{c}')

    # Constraints: total demand assigned to a plant cannot exceed its capacity
    for p in range(num_plants):
        model.addConstr(gp.quicksum(demands[c] * assignments[c, p] for c in range(num_customers)) <= capacities[p] * open_plants[p], f'capacity_{p}')

    return model, open_plants, assignments

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', required=True, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', required=True, type=int, help='Maximum runtime in seconds')
    parser.add_argument('--log_path', help='Path to log intermediate solutions (optional)')
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    model, open_plants, assignments = solve(instance)

    logger = SolutionLogger(args.log_path, sense='minimize') if args.log_path else None

    model.Params.TimeLimit = args.time_limit
    model.Params.LogToConsole = 0  # Disable Gurobi console output

    def log_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    model.optimize(log_callback)

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        open_plants_sol = [p for p in range(instance['num_plants']) if open_plants[p].x > 0.5]
        assignments_sol = {f'{c}': {f'{p}': assignments[c, p].x for p in range(instance['num_plants'])} for c in range(instance['num_customers'])}
        solution = {
            'objective_value': model.objVal,
            'open_plants': open_plants_sol,
            'assignments': assignments_sol
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        print(f"Model status: {model.status}")

if __name__ == '__main__':
    main()