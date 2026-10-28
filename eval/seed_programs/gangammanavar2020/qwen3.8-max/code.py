import json
import argparse
import time
import numpy as np
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

    start_time = time.time()

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    N = instance['N']
    holding_costs = instance['second_stage']['holding_costs']
    penalty_costs = instance['second_stage']['penalty_costs']
    trans_costs_mean = instance['second_stage']['transshipment_costs_mean']
    trans_costs_std = instance['second_stage']['transshipment_costs_std']
    random_cost_links = instance['second_stage']['random_cost_links']
    random_cost_std_fraction = instance['second_stage'].get('random_cost_std_fraction', 0.0)
    mean_demands = instance['second_stage']['mean_demands']

    num_scenarios = instance['second_stage']['demand_distribution']['num_scenarios']
    scenarios = instance['second_stage']['demand_distribution']['scenarios']
    probabilities = instance['second_stage']['demand_distribution']['probabilities']

    lower_bounds = instance['first_stage']['variable_lower_bounds']

    # Get seed from instance if available
    seed = instance.get('seed', instance.get('random_seed', 0))

    # Generate cost realizations for each scenario
    rng = np.random.default_rng(seed)

    # Precompute cost realizations
    cost_realizations = []
    for k in range(num_scenarios):
        costs_k = [row[:] for row in trans_costs_mean]
        for link in random_cost_links:
            i, j = link
            std_val = trans_costs_std[i][j]
            if std_val > 0:
                cost_val = rng.normal(trans_costs_mean[i][j], std_val)
                cost_val = max(cost_val, 0.0)
                costs_k[i][j] = cost_val
        cost_realizations.append(costs_k)

    # Build LP model
    model = gp.Model("distribution_network")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', max(args.time_limit - 5, 10))
    model.setParam('OutputFlag', 0)

    # First-stage variables: order-up-to levels
    s = model.addVars(N, lb=0.0, name="s")
    for i in range(N):
        if lower_bounds[i] > 0:
            s[i].LB = lower_bounds[i]

    # Second-stage variables for each scenario
    f_vars = {}
    e_vars = {}
    q_vars = {}
    r_vars = {}
    t_vars = {}

    for k in range(num_scenarios):
        for i in range(N):
            f_vars[k, i] = model.addVar(lb=0.0, name=f"f_{k}_{i}")
            e_vars[k, i] = model.addVar(lb=0.0, name=f"e_{k}_{i}")
            q_vars[k, i] = model.addVar(lb=0.0, name=f"q_{k}_{i}")
            r_vars[k, i] = model.addVar(lb=0.0, name=f"r_{k}_{i}")
        for i in range(N):
            for j in range(N):
                if i != j:
                    t_vars[k, i, j] = model.addVar(lb=0.0, name=f"t_{k}_{i}_{j}")

    model.update()

    # Constraints
    for k in range(num_scenarios):
        for i in range(N):
            # Constraint 1: f[i] + sum_j(t[i][j]) + e[i] = s[i]
            model.addConstr(
                f_vars[k, i] + gp.quicksum(t_vars[k, i, j] for j in range(N) if j != i) + e_vars[k, i] == s[i],
                name=f"stock_bal_{k}_{i}"
            )

            # Constraint 2: f[i] + sum_j(t[j][i]) + r[i] = d[k][i]
            model.addConstr(
                f_vars[k, i] + gp.quicksum(t_vars[k, j, i] for j in range(N) if j != i) + r_vars[k, i] == scenarios[k][i],
                name=f"demand_bal_{k}_{i}"
            )

            # Constraint 3: e[i] + q[i] = s[i]
            model.addConstr(
                e_vars[k, i] + q_vars[k, i] == s[i],
                name=f"replenish_bal_{k}_{i}"
            )

    # Objective: minimize expected total cost
    obj = gp.LinExpr()
    for k in range(num_scenarios):
        prob_k = probabilities[k]
        for i in range(N):
            obj += prob_k * holding_costs[i] * e_vars[k, i]
            obj += prob_k * penalty_costs[i] * r_vars[k, i]
        for i in range(N):
            for j in range(N):
                if i != j and cost_realizations[k][i][j] > 0:
                    obj += prob_k * cost_realizations[k][i][j] * t_vars[k, i, j]

    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    # Extract solution
    if model.SolCount > 0:
        obj_val = model.ObjVal
        s_vals = [s[i].X for i in range(N)]

        scenario_solutions = []
        for k in range(num_scenarios):
            t_matrix = [[0.0] * N for _ in range(N)]
            for i in range(N):
                for j in range(N):
                    if i != j:
                        t_matrix[i][j] = max(t_vars[k, i, j].X, 0.0)

            sol_k = {
                'e': [max(e_vars[k, i].X, 0.0) for i in range(N)],
                'f': [max(f_vars[k, i].X, 0.0) for i in range(N)],
                'q': [max(q_vars[k, i].X, 0.0) for i in range(N)],
                'r': [max(r_vars[k, i].X, 0.0) for i in range(N)],
                't': t_matrix
            }
            scenario_solutions.append(sol_k)

        solution = {
            'objective_value': obj_val,
            's': s_vals,
            'scenarios': scenario_solutions
        }

        if logger:
            logger.log_solution(obj_val, solution)

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)

    else:
        # Fallback: heuristic solution using mean demands
        s_vals = [max(mean_demands[i], lower_bounds[i]) for i in range(N)]

        scenario_solutions = []
        total_cost = 0.0
        for k in range(num_scenarios):
            e_list = []
            f_list = []
            q_list = []
            r_list = []
            t_matrix = [[0.0] * N for _ in range(N)]

            for i in range(N):
                d_i = scenarios[k][i]
                s_i = s_vals[i]
                if s_i >= d_i:
                    f_list.append(d_i)
                    e_list.append(s_i - d_i)
                    q_list.append(0.0)
                    r_list.append(0.0)
                else:
                    f_list.append(s_i)
                    e_list.append(0.0)
                    q_list.append(0.0)
                    r_list.append(d_i - s_i)

            # Compute cost for this scenario
            scen_cost = 0.0
            for i in range(N):
                scen_cost += holding_costs[i] * e_list[i]
                scen_cost += penalty_costs[i] * r_list[i]
            total_cost += probabilities[k] * scen_cost

            sol_k = {
                'e': e_list,
                'f': f_list,
                'q': q_list,
                'r': r_list,
                't': t_matrix
            }
            scenario_solutions.append(sol_k)

        solution = {
            'objective_value': total_cost,
            's': s_vals,
            'scenarios': scenario_solutions
        }

        if logger:
            logger.log_solution(total_cost, solution)

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)


if __name__ == '__main__':
    main()