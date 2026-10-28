import argparse
import json
import sys

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("Gurobi (gurobipy) is not available.", file=sys.stderr)
    sys.exit(1)

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser(description="Solve crew rotation set partitioning with base constraints.")
    parser.add_argument('--instance_path', required=True, help="Path to the instance JSON file.")
    parser.add_argument('--solution_path', required=True, help="Path to the output solution JSON file.")
    parser.add_argument('--time_limit', type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument('--log_path', default=None, help="Optional path for incumbent logging.")
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Load instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_rows = instance['dimensions']['num_rows']
    num_cols = instance['dimensions']['num_cols']
    cost = instance['cost_vector']
    columns = instance['constraint_matrix_A']['columns']

    # Create Gurobi environment and model
    env = gp.Env(empty=True)
    env.setParam('Seed', 0)
    env.setParam('Threads', 1)
    env.setParam('OutputFlag', 0)
    env.start()

    model = gp.Model('crew_rotations', env=env)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('TimeLimit', args.time_limit)

    # Decision variables: x[j] = 1 if rotation j is selected
    x = model.addVars(range(num_cols), vtype=GRB.BINARY, name='x')
    model.setObjective(gp.quicksum(cost[j] * x[j] for j in range(num_cols)), GRB.MINIMIZE)

    # Exact coverage constraints: each flight leg covered exactly once
    row_vars = [[] for _ in range(num_rows)]
    for j, col in enumerate(columns):
        for i in col:
            if 0 <= i < num_rows:
                row_vars[i].append(x[j])
    for i in range(num_rows):
        if row_vars[i]:
            model.addConstr(gp.quicksum(row_vars[i]) == 1, name=f'cover_{i}')
        else:
            # Uncovered row -> model infeasible, add a trivially violated constraint
            model.addConstr(gp.LinExpr() == 1, name=f'cover_{i}_empty')

    # Base constraints (if any)
    if instance.get('has_base_constraints', False):
        bc = instance['base_constraints']
        num_bases = bc['num_bases']
        D_rows = bc['D_matrix']['rows']
        lower = bc['lower_bounds_d1']
        upper = bc['upper_bounds_d2']
        for b in range(num_bases):
            d_row = D_rows[b]
            expr = gp.quicksum(d_row[j] * x[j] for j in range(num_cols))
            if lower[b] > 0:
                model.addConstr(expr >= lower[b], name=f'base_low_{b}')
            if upper[b] < float('inf'):
                model.addConstr(expr <= upper[b], name=f'base_up_{b}')

    # Store variables for callback access
    model._vars = [x[j] for j in range(num_cols)]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                vals = model.cbGet(GRB.Callback.MIPSOL_SOL)
                selected = []
                var_vals = {}
                for j, val in enumerate(vals):
                    if val > 0.5:
                        selected.append(j)
                        var_vals[str(j)] = 1.0
                    else:
                        var_vals[str(j)] = 0.0
                solution = {
                    "objective_value": float(obj),
                    "selected_rotations": selected,
                    "variable_values": var_vals,
                }
                if logger:
                    logger.log_solution(float(obj), solution)
            except Exception:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if logger:
                    logger.log(float(obj))

    model.optimize(callback)

    # Extract best solution found
    if model.SolCount > 0:
        obj = model.ObjVal
        vals = model.getAttr('X', model._vars)
        selected = [j for j in range(num_cols) if vals[j] > 0.5]
        var_vals = {str(j): (1.0 if vals[j] > 0.5 else 0.0) for j in range(num_cols)}
        solution = {
            "objective_value": float(obj),
            "selected_rotations": selected,
            "variable_values": var_vals,
        }
        # Log the final incumbent (logger will ignore if not improved)
        if logger:
            logger.log_solution(float(obj), solution)
    else:
        # No feasible solution found within the time limit
        solution = {
            "objective_value": 0.0,
            "selected_rotations": [],
            "variable_values": {str(j): 0.0 for j in range(num_cols)},
        }
        if logger:
            logger.log_solution(0.0, solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()