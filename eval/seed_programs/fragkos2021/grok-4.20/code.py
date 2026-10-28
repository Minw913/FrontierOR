import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import numpy as np

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

    n = data['num_facilities']
    m = data['num_customers']
    nodes = data['nodes']
    demand = np.array([nodes[i]['demand'] for i in range(m)])
    fixed_cost = np.array([nodes[i]['fixed_setup_cost'] for i in range(n)])
    trans_cost = np.array(data['transportation_cost_matrix'])
    penalty = np.array(data['penalty_cost'])

    # Precompute for each customer the sorted facilities by increasing cost
    sorted_fac = []
    for j in range(m):
        idx = np.argsort(trans_cost[j])
        sorted_fac.append(idx.tolist())

    model = gp.Model("Stochastic_FL_Independent")
    model.setParam('TimeLimit', max(1, args.time_limit - 5))
    model.setParam('MIPGap', 1e-4)
    model.setParam('Threads', 1)
    model.setParam('LogToConsole', 0)
    model.setParam('Presolve', 2)
    model.setParam('Cuts', 2)

    y = model.addVars(n, vtype=GRB.BINARY, name="y")
    prob_reach = {}
    prob_serve = {}
    exp_cost = model.addVars(m, lb=0.0, vtype=GRB.CONTINUOUS, name="exp_cost")

    for j in range(m):
        ordered = sorted_fac[j]
        prob_reach[j, 0] = 1.0
        for k_idx in range(n):
            prob_reach[j, k_idx + 1] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"preach_{j}_{k_idx+1}")

        for k_idx in range(n):
            i = ordered[k_idx]
            surv = nodes[i]['survival_probability']
            z_var = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"z_{j}_{k_idx}")
            model.addConstr(z_var <= y[i])
            model.addConstr(z_var <= prob_reach[j, k_idx])
            model.addConstr(z_var >= prob_reach[j, k_idx] - (1 - y[i]))
            prob_serve[j, i] = surv * z_var
            model.addConstr(prob_reach[j, k_idx + 1] == prob_reach[j, k_idx] - surv * z_var)

    for j in range(m):
        ordered = sorted_fac[j]
        ec = penalty[j] * prob_reach[j, n]
        for k_idx in range(n):
            i = ordered[k_idx]
            ec += trans_cost[j, i] * prob_serve[j, i]
        model.addConstr(exp_cost[j] == ec)

    total_fixed = gp.quicksum(fixed_cost[i] * y[i] for i in range(n))
    model.setObjective(total_fixed + gp.quicksum(demand[j] * exp_cost[j] for j in range(m)), GRB.MINIMIZE)

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger and obj_val < float('inf'):
                logger.log(obj_val)

    model.optimize(callback)

    if model.SolCount == 0:
        facility_open = [0] * n
        obj_val = sum(demand[j] * penalty[j] for j in range(m))
    else:
        facility_open = [int(round(y[i].X)) for i in range(n)]
        obj_val = model.objVal

    solution = {
        "objective_value": float(obj_val),
        "facility_locations": facility_open
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    # No close() call - just let the context end

if __name__ == "__main__":
    main()