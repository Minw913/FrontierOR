import json
import argparse
import time
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

    knapsack_capacity = data['problem_parameters']['knapsack_capacity']
    uncertainty_budget = data['problem_parameters']['uncertainty_budget']
    items = data['items']
    n = len(items)

    # Precompute indices for easy access
    item_ids = [item['id'] for item in items]
    w = [item['weight'] for item in items]
    p = [item['nominal_profit'] for item in items]
    d = [item['max_degradation'] for item in items]
    o = [item['outsource_penalty'] for item in items]
    r = [item['repair_capacity'] for item in items]

    # The model is a min-max problem. We use the dual of the inner maximization.
    # Inner max: sum y_i * d_i  s.t. sum y_i <= uncertainty_budget, 0 <= y_i <= x_i
    # Dual: min  Gamma * lambda + sum x_i * mu_i
    #      s.t. lambda + mu_i >= d_i   for all i
    #           lambda, mu >= 0
    #
    # Overall objective becomes:
    # sum x_i * (o_i - p_i) + Gamma * lambda + sum x_i * mu_i
    # s.t. lambda + mu_i >= d_i for all i
    #      capacity constraints with continuous y_prod, y_repair in [0, x_i]

    model = gp.Model("RobustKnapsackLPRecourse")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)

    # First-stage binary variables
    x = model.addVars(n, vtype=GRB.BINARY, name="x")

    # Dual variables for adversary
    lam = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="lambda")
    mu = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0.0, name="mu")

    # Second-stage continuous variables (LP relaxation)
    y_prod = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="y_prod")
    y_repair = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="y_repair")

    # Objective: sum x*(o-p) + Gamma*lam + sum x*mu
    obj = gp.quicksum(x[i] * (o[i] - p[i]) for i in range(n)) + \
          uncertainty_budget * lam + \
          gp.quicksum(x[i] * mu[i] for i in range(n))
    model.setObjective(obj, GRB.MINIMIZE)

    # Dual feasibility constraints
    for i in range(n):
        model.addConstr(lam + mu[i] >= d[i], name=f"dual_feas_{i}")

    # y_prod and y_repair can only be positive if x_i == 1
    for i in range(n):
        model.addConstr(y_prod[i] <= x[i], name=f"prod_bound_{i}")
        model.addConstr(y_repair[i] <= y_prod[i], name=f"repair_bound_{i}")  # repair <= produced
        model.addConstr(y_repair[i] <= x[i], name=f"repair_x_bound_{i}")

    # Capacity constraint (LP relaxation)
    prod_capacity = gp.quicksum(y_prod[i] * w[i] for i in range(n))
    repair_capacity = gp.quicksum(y_repair[i] * r[i] for i in range(n))
    model.addConstr(prod_capacity + repair_capacity <= knapsack_capacity,
                    name="capacity")

    # Callback to log incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            x_sol = model.cbGetSolution(x)
            solution_dict = {"x": {str(item_ids[i]): 1 if x_sol[i] > 0.5 else 0 for i in range(n)}}
            if logger:
                logger.log_solution(obj_val, solution_dict)

    model.optimize(callback)

    # Extract solution
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED):
        obj_value = model.objVal
        x_vals = model.getAttr('X', x)
        solution_x = {str(item_ids[i]): 1 if x_vals[i] > 0.5 else 0 for i in range(n)}
        
        output = {
            "objective_value": float(obj_value),
            "x": solution_x
        }
        
        with open(args.solution_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        if logger:
            logger.log_solution(obj_value, output)
    else:
        # Fallback: return all zeros if no solution found
        output = {
            "objective_value": 0.0,
            "x": {str(item_ids[i]): 0 for i in range(n)}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()