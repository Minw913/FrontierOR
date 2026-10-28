import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve_instance(instance, time_limit):
    N = instance['N']
    S = instance['S']
    k = instance['k']
    beta = instance['beta']
    gamma = instance['gamma']
    mu_bar = instance['mu_bar']
    mu = instance['mu']
    p_s = instance['p_s']  # Use the provided probability
    scenarios = instance['scenarios']

    model = gp.Model()
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)  # Silence Gurobi output

    x = model.addVars(N, lb=0, ub=1, name='x')
    z = model.addVars(N, vtype=GRB.BINARY, name='z')
    a = model.addVar(lb=-GRB.INFINITY, name='a')
    v = model.addVar(lb=0, name='v')
    y = model.addVars(S, lb=0, name='y')  # Auxiliary variables for CVaR

    # Objective function
    regularization = 1 / (2 * gamma) * sum(x[i] * x[i] for i in range(N))
    model.setObjective(regularization + a + v, GRB.MINIMIZE)

    # Constraints
    model.addConstr(sum(x[i] for i in range(N)) == 1, name='budget')
    model.addConstr(sum(mu[i] * x[i] for i in range(N)) >= mu_bar, name='return_requirement')
    model.addConstr(sum(z[i] for i in range(N)) <= k, name='cardinality_limit')
    for i in range(N):
        model.addConstr(x[i] <= z[i], name=f'link_x_z_{i}')

    for s in range(S):
        scenario_loss = sum(-scenarios[s][i] * x[i] for i in range(N))
        model.addConstr(y[s] >= scenario_loss - a, name=f'cvar_constr1_{s}')
        model.addConstr(y[s] >= 0, name=f'cvar_constr2_{s}')
    model.addConstr(v >= sum(p_s * y[s] for s in range(S)) / (1 - beta), name='cvar_constr3')

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.getObjective().getValue()
        x_sol = [x[i].X for i in range(N)]
        z_sol = [int(z[i].X) for i in range(N)]
        a_sol = a.X
        v_sol = v.X

        return {
            'objective_value': objective_value,
            'x': x_sol,
            'z': z_sol,
            'a': a_sol,
            'v': v_sol
        }
    else:
        raise Exception(f"Gurobi failed to solve the model. Status: {model.status}")

def main():
    parser = argparse.ArgumentParser(description='Solve portfolio optimization problem.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file.')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to the solution JSON file.')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds.')
    parser.add_argument('--log_path', type=str, help='Path to the log JSONL file.')
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)
    solution = solve_instance(instance, args.time_limit)

    if logger:
        logger.log(solution['objective_value'])

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == "__main__":
    main()