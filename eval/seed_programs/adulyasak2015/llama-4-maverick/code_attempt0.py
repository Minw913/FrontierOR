import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        instance = json.load(file)
    return instance

def solve_instance(instance, time_limit, logger):
    n = instance['n']
    T = instance['T']
    m = instance['m']
    Q = instance['Q']
    C = instance['C']
    f = instance['f']
    u = instance['u']
    epsilon = instance['epsilon']
    alpha_hat = instance['alpha_hat']
    coordinates = instance['coordinates']
    transportation_costs = instance['transportation_costs']
    h = instance['h']
    L = instance['L']
    I0 = instance['I0']
    d_bar = instance['d_bar']
    sigma = instance['sigma']
    n_scenarios = instance['n_scenarios']
    scenario_probabilities = instance['scenario_probabilities']
    demand_scenarios = instance['demand_scenarios']

    model = gp.Model('Production_Distribution_Planning')
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit

    # First-stage variables
    y = model.addVars(T, vtype=GRB.BINARY, name='y')
    z = model.addVars(range(n+1), range(m), T, vtype=GRB.BINARY, name='z')
    x = model.addVars(range(n+1), range(n+1), range(m), T, vtype=GRB.INTEGER, name='x')

    # Second-stage variables
    p = model.addVars(T, range(n_scenarios), vtype=GRB.CONTINUOUS, name='p')
    I = model.addVars(range(n+1), T, range(n_scenarios), vtype=GRB.CONTINUOUS, name='I')
    e = model.addVars(range(1, n+1), T, range(n_scenarios), vtype=GRB.CONTINUOUS, name='e')
    q = model.addVars(range(1, n+1), range(m), T, range(n_scenarios), vtype=GRB.CONTINUOUS, name='q')

    # Objective function
    model.setObjective(
        gp.quicksum(f * y[t] for t in range(T)) +
        gp.quicksum(transportation_costs[i][j] * x[i, j, k, t] for i in range(n+1) for j in range(n+1) for k in range(m) for t in range(T)) +
        gp.quicksum(scenario_probabilities[omega] * u * p[t, omega] for t in range(T) for omega in range(n_scenarios)) +
        gp.quicksum(scenario_probabilities[omega] * h[i] * I[i, t, omega] for i in range(n+1) for t in range(T) for omega in range(n_scenarios)) +
        gp.quicksum(scenario_probabilities[omega] * sigma[i-1] * e[i, t, omega] for i in range(1, n+1) for t in range(T) for omega in range(n_scenarios)),
        GRB.MINIMIZE
    )

    # Constraints
    for t in range(T):
        model.addConstr(gp.quicksum(p[t, omega] for omega in range(n_scenarios)) <= C * y[t])
        for omega in range(n_scenarios):
            model.addConstr(p[t, omega] <= C * y[t])

    for i in range(n+1):
        for t in range(T):
            for omega in range(n_scenarios):
                if t == 0:
                    model.addConstr(I[i, t, omega] == I0[i] + (p[t, omega] if i == 0 else 0) - (gp.quicksum(q[i, k, t, omega] for k in range(m)) if i > 0 else gp.quicksum(q[j, k, t, omega] for j in range(1, n+1) for k in range(m))) - (e[i, t, omega] if i > 0 else 0))
                else:
                    model.addConstr(I[i, t, omega] == I[i, t-1, omega] + (p[t, omega] if i == 0 else 0) - (gp.quicksum(q[i, k, t, omega] for k in range(m)) if i > 0 else gp.quicksum(q[j, k, t, omega] for j in range(1, n+1) for k in range(m))) - (e[i, t, omega] if i > 0 else 0))
                model.addConstr(I[i, t, omega] <= L[i])

    for i in range(1, n+1):
        for t in range(T):
            for omega in range(n_scenarios):
                demand = demand_scenarios[omega][i-1][t]
                model.addConstr(e[i, t, omega] >= demand - I[i, t-1, omega] - gp.quicksum(q[i, k, t, omega] for k in range(m)))
                model.addConstr(e[i, t, omega] >= 0)

    for k in range(m):
        for t in range(T):
            model.addConstr(gp.quicksum(q[i, k, t, omega] for i in range(1, n+1) for omega in range(n_scenarios)) <= Q * z[0, k, t])

    for i in range(1, n+1):
        for k in range(m):
            for t in range(T):
                for omega in range(n_scenarios):
                    model.addConstr(q[i, k, t, omega] <= Q * z[i, k, t])
                    model.addConstr(q[i, k, t, omega] <= L[i] - I[i, t-1, omega])

    # Routing constraints
    for k in range(m):
        for t in range(T):
            model.addConstr(x[0, 0, k, t] == 0)
            for i in range(1, n+1):
                model.addConstr(x[0, i, k, t] == x[i, 0, k, t])
                model.addConstr(z[i, k, t] <= x[0, i, k, t])
                model.addConstr(gp.quicksum(x[j, i, k, t] for j in range(n+1)) == z[i, k, t])
                model.addConstr(gp.quicksum(x[i, j, k, t] for j in range(n+1)) == z[i, k, t])
            model.addConstr(gp.quicksum(z[i, k, t] for i in range(1, n+1)) <= n)
            model.addConstr(gp.quicksum(x[0, i, k, t] for i in range(1, n+1)) <= 2)

    # Subtour elimination constraints
    for k in range(m):
        for t in range(T):
            for S in range(2, n+1):
                for subset in itertools.combinations(range(1, n+1), S):
                    model.addConstr(gp.quicksum(x[i, j, k, t] for i in subset for j in subset if i != j) <= len(subset) - 1)

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.ObjVal
        solution = {
            'objective_value': objective_value,
            'model_variables': {},
            'y': {t+1: y[t].X for t in range(T)},
            'z': {(node, vehicle, period): z[node, vehicle, period].X for node in range(n+1) for vehicle in range(m) for period in range(T)},
            'x': {(node_i, node_j, vehicle, period): x[node_i, node_j, vehicle, period].X for node_i in range(n+1) for node_j in range(n+1) for vehicle in range(m) for period in range(T)},
        }
        for var in model.getVars():
            if var.VarName.startswith(('p_', 'I_', 'e_', 'q_')):
                solution['model_variables'][var.VarName] = var.X

        if logger:
            logger.log_solution(objective_value, solution)

        return solution
    else:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense='minimize') if args.log_path else None

    solution = solve_instance(instance, args.time_limit, logger)

    if solution is not None:
        with open(args.solution_path, 'w') as file:
            json.dump(solution, file, indent=4)

if __name__ == '__main__':
    import itertools  # noqa: F401
    main()