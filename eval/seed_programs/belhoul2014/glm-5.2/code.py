import argparse
import json
from gurobipy import Model, GRB, quicksum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Load instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    p = instance['p']
    cost_matrices = instance['cost_matrices']  # [p][n][n]
    reference_point = instance['reference_point']  # [p]
    search_direction_lambda = instance['search_direction_lambda']  # [p]

    # Initialize logger
    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except ImportError:
            pass

    # Build MILP
    model = Model('min_max_assignment')
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)

    # Decision variables: x[i,j] = 1 if task i assigned to agent j
    x = {}
    for i in range(n):
        for j in range(n):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')

    # Auxiliary variable: worst-case weighted deviation
    z = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='z')

    model.setObjective(z, GRB.MINIMIZE)

    # Assignment constraints: each task to exactly one agent
    for i in range(n):
        model.addConstr(quicksum(x[i, j] for j in range(n)) == 1)

    # Each agent gets exactly one task
    for j in range(n):
        model.addConstr(quicksum(x[i, j] for i in range(n)) == 1)

    # Worst-case deviation constraints
    for k in range(p):
        total_cost_k = quicksum(
            cost_matrices[k][i][j] * x[i, j]
            for i in range(n) for j in range(n)
        )
        model.addConstr(
            z >= search_direction_lambda[k] * (total_cost_k - reference_point[k])
        )

    # Callback for logging incumbents
    if logger:
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                xs = model.cbGetSolution([x[i, j] for i in range(n) for j in range(n)])
                assignment = []
                assignment_matrix = [[0] * n for _ in range(n)]
                idx = 0
                for i in range(n):
                    for j in range(n):
                        if xs[idx] > 0.5:
                            assignment.append(j)
                            assignment_matrix[i][j] = 1
                        idx += 1
                solution_dict = {
                    "objective_value": float(obj),
                    "assignment": assignment,
                    "assignment_matrix": assignment_matrix
                }
                logger.log_solution(float(obj), solution_dict)

        model.optimize(callback)
    else:
        model.optimize()

    # Extract solution
    if model.SolCount > 0:
        assignment = []
        assignment_matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if x[i, j].X > 0.5:
                    assignment.append(j)
                    assignment_matrix[i][j] = 1
                    break

        obj_val = max(
            search_direction_lambda[k] * (
                sum(cost_matrices[k][i][assignment[i]] for i in range(n))
                - reference_point[k]
            )
            for k in range(p)
        )
    else:
        # Fallback: identity assignment
        assignment = list(range(n))
        assignment_matrix = [[1 if j == i else 0 for j in range(n)] for i in range(n)]
        obj_val = max(
            search_direction_lambda[k] * (
                sum(cost_matrices[k][i][i] for i in range(n))
                - reference_point[k]
            )
            for k in range(p)
        )
        if logger:
            logger.log_solution(
                float(obj_val),
                {
                    "objective_value": float(obj_val),
                    "assignment": assignment,
                    "assignment_matrix": assignment_matrix
                }
            )

    solution_dict = {
        "objective_value": float(obj_val),
        "assignment": assignment,
        "assignment_matrix": assignment_matrix
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f, indent=2)


if __name__ == '__main__':
    main()