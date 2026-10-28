import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser(description='Capacitated Facility Location Problem Solver')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log intermediate solutions (JSONL format)')
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    num_customers = data['num_customers']
    num_plants = data['num_plants']
    demands = data['customers']['demands']
    fixed_costs = data['plants']['fixed_costs']
    capacities = data['plants']['capacities']
    shipping_costs = data['shipping_costs']

    model = gp.Model('CapacitatedFacilityLocation')
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', 0)  # Suppress Gurobi output

    y = model.addVars(num_plants, vtype=GRB.BINARY, name='y')
    x = model.addVars(num_customers, num_plants, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='x')

    total_fixed = gp.quicksum(fixed_costs[j] * y[j] for j in range(num_plants))
    total_shipping = gp.quicksum(shipping_costs[i][j] * x[i, j] for i in range(num_customers) for j in range(num_plants))
    model.setObjective(total_fixed + total_shipping, GRB.MINIMIZE)

    for i in range(num_customers):
        model.addConstr(gp.quicksum(x[i, j] for j in range(num_plants)) == 1, name=f'demand_{i}')

    for j in range(num_plants):
        model.addConstr(gp.quicksum(demands[i] * x[i, j] for i in range(num_customers)) <= capacities[j] * y[j], name=f'cap_{j}')

    if logger:
        model._logger = logger
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                model._logger.log(obj)
        model.optimize(callback)
    else:
        model.optimize()

    if model.SolCount > 0:
        open_plants = [j for j in range(num_plants) if y[j].X > 0.5]
        assignments = {}
        for i in range(num_customers):
            cust_assign = {}
            for j in range(num_plants):
                val = x[i, j].X
                if val > 1e-6:
                    cust_assign[str(j)] = val
            assignments[str(i)] = cust_assign

        solution = {
            "objective_value": model.ObjVal,
            "open_plants": open_plants,
            "assignments": assignments
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
    else:
        raise RuntimeError("No feasible solution found within the time limit.")

if __name__ == '__main__':
    main()