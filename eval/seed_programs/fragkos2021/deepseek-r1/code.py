import json
import random
import argparse
import gurobipy as grb
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    random.seed(0)

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n = data['num_facilities']
    nodes = data['nodes']
    demand = [0.0] * n
    fixed_setup_cost = [0.0] * n
    survival_probability = [0.0] * n
    transportation_cost_matrix = data['transportation_cost_matrix']
    penalty_cost = data['penalty_cost']

    for node in nodes:
        i = node['id']
        demand[i] = node['demand']
        fixed_setup_cost[i] = node['fixed_setup_cost']
        survival_probability[i] = node['survival_probability']

    if n <= 50:
        N = 100
    elif n <= 100:
        N = 50
    elif n <= 200:
        N = 20
    else:
        N = 10

    available = []
    for _ in range(N):
        scenario = []
        for j in range(n):
            if random.random() < survival_probability[j]:
                scenario.append(1)
            else:
                scenario.append(0)
        available.append(scenario)

    model = grb.Model()
    x = model.addVars(n, vtype=grb.GRB.BINARY, name="x")
    y = {}
    z = {}

    for s in range(N):
        for i in range(n):
            z[(i, s)] = model.addVar(lb=0, ub=1, name=f"z_{i}_{s}")
            for j in range(n):
                if available[s][j]:
                    y[(i, j, s)] = model.addVar(lb=0, ub=1, name=f"y_{i}_{j}_{s}")

    for s in range(N):
        for i in range(n):
            model.addConstr(
                grb.quicksum(y.get((i, j, s), 0) for j in range(n) if available[s][j]) + z[(i, s)] == 1,
                name=f"assign_{i}_{s}"
            )
        for j in range(n):
            if available[s][j]:
                for i in range(n):
                    model.addConstr(y.get((i, j, s), 0) <= x[j], name=f"open_{i}_{j}_{s}")

    setup_cost = grb.quicksum(fixed_setup_cost[j] * x[j] for j in range(n))
    scenario_cost = 0
    for s in range(N):
        for i in range(n):
            term = grb.quicksum(transportation_cost_matrix[i][j] * y.get((i, j, s), 0) for j in range(n) if available[s][j])
            scenario_cost += demand[i] * (term + penalty_cost[i] * z[(i, s)])
    scenario_cost = scenario_cost / N
    model.setObjective(setup_cost + scenario_cost, grb.GRB.MINIMIZE)

    model.setParam('OutputFlag', 0)
    model.setParam('Seed', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)

    logger = None
    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="minimize")

    def callback(model, where):
        if where == grb.GRB.Callback.MIPSOL:
            obj = model.cbGet(grb.GRB.Callback.MIPSOL_OBJ)
            x_vals = model.cbGetSolution(model._vars)
            solution_dict = {
                "objective_value": obj,
                "facility_locations": [int(round(x_val)) for x_val in x_vals]
            }
            if logger:
                logger.log_solution(obj, solution_dict)

    model._vars = x
    model.optimize(callback)

    if model.SolCount > 0:
        x_vals = [int(round(x[j].X)) for j in range(n)]
    else:
        x_vals = [0] * n

    total_setup_cost = sum(fixed_setup_cost[j] * x_vals[j] for j in range(n))
    total_expected_cost = 0.0
    for i in range(n):
        sorted_indices = sorted(range(n), key=lambda j: transportation_cost_matrix[i][j])
        prod = 1.0
        cost_i = 0.0
        for j in sorted_indices:
            if x_vals[j] == 1:
                cost_i += transportation_cost_matrix[i][j] * (prod * survival_probability[j])
                prod *= (1 - survival_probability[j])
        cost_i += penalty_cost[i] * prod
        total_expected_cost += demand[i] * cost_i
    objective_value = total_setup_cost + total_expected_cost

    solution_dict = {
        "objective_value": objective_value,
        "facility_locations": x_vals
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f)

if __name__ == "__main__":
    main()