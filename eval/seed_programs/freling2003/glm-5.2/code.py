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

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    num_fac = data['parameters']['num_facilities']
    num_cust = data['parameters']['num_customers']
    num_per = data['parameters']['num_periods']

    seasonal = data['seasonal_factors']
    demands = data['demands']
    holding_costs = data['holding_costs']      # [num_fac][num_per]
    capacities = data['capacities']              # [num_fac][num_per]
    trans_costs = data['transportation_costs']   # [num_fac][num_cust][num_per]

    # Demand per customer per period
    D = [[demands[c] * seasonal[t] for t in range(num_per)] for c in range(num_cust)]

    # Build MIP model
    model = gp.Model('logistics')
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', 0)

    # Decision variables
    x = model.addVars(num_fac, num_cust, vtype=GRB.BINARY, name='x')
    I = model.addVars(num_fac, num_per, vtype=GRB.CONTINUOUS, lb=0, name='I')

    # Each customer assigned to exactly one facility
    for c in range(num_cust):
        model.addConstr(gp.quicksum(x[f, c] for f in range(num_fac)) == 1)

    # Inventory balance / capacity constraints
    for f in range(num_fac):
        for t in range(num_per):
            prev_inv = I[f, t - 1] if t > 0 else 0
            model.addConstr(
                gp.quicksum(D[c][t] * x[f, c] for c in range(num_cust)) + I[f, t]
                <= capacities[f][t] + prev_inv
            )

    # Objective: supply cost + holding cost
    obj = gp.LinExpr()
    for f in range(num_fac):
        for c in range(num_cust):
            for t in range(num_per):
                obj += trans_costs[f][c][t] * D[c][t] * x[f, c]
    for f in range(num_fac):
        for t in range(num_per):
            obj += holding_costs[f][t] * I[f, t]

    model.setObjective(obj, GRB.MINIMIZE)

    # Callback to log incumbents
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            assignment = {}
            for c in range(num_cust):
                for f in range(num_fac):
                    val = model.cbGetSolution(x[f, c])
                    if val > 0.5:
                        assignment[str(c)] = f
                        break
            inventory = {}
            for f in range(num_fac):
                inventory[str(f)] = [model.cbGetSolution(I[f, t]) for t in range(num_per)]
            sol_dict = {
                "objective_value": obj_val,
                "assignment": assignment,
                "inventory": inventory
            }
            if logger:
                logger.log_solution(obj_val, sol_dict)

    model.optimize(callback)

    # Extract final solution
    if model.SolCount > 0:
        obj_val = model.ObjVal
        assignment = {}
        for c in range(num_cust):
            for f in range(num_fac):
                if x[f, c].X > 0.5:
                    assignment[str(c)] = f
                    break
        inventory = {}
        for f in range(num_fac):
            inventory[str(f)] = [I[f, t].X for t in range(num_per)]
        solution = {
            "objective_value": obj_val,
            "assignment": assignment,
            "inventory": inventory
        }
    else:
        # Fallback (should not happen for feasible instances)
        solution = {
            "objective_value": 0.0,
            "assignment": {str(c): 0 for c in range(num_cust)},
            "inventory": {str(f): [0.0] * num_per for f in range(num_fac)}
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    if logger:
        logger.log_solution(solution["objective_value"], solution)


if __name__ == '__main__':
    main()