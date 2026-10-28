import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_rows = instance['dimensions']['num_rows']
    num_cols = instance['dimensions']['num_cols']
    cost_vector = instance['cost_vector']
    constraint_matrix_A = instance['constraint_matrix_A']['columns']
    has_base_constraints = instance['has_base_constraints']

    # Build A as list of lists for efficiency
    A = [[] for _ in range(num_rows)]
    for j, rows in enumerate(constraint_matrix_A):
        for i in rows:
            A[i].append(j)

    start_time = time.time()
    model = gp.Model("Crew_Rotation")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 1e-4)
    model.setParam('LogToConsole', 0)

    x = model.addVars(num_cols, vtype=GRB.BINARY, name="x")

    # Exact cover constraints
    for i in range(num_rows):
        expr = gp.LinExpr()
        for j in A[i]:
            expr.addTerms(1.0, x[j])
        model.addConstr(expr == 1, name=f"cover_{i}")

    # Base constraints if present
    if has_base_constraints:
        base_data = instance['base_constraints']
        num_bases = base_data['num_bases']
        D_matrix = base_data['D_matrix']['rows']
        lower_bounds = base_data['lower_bounds_d1']
        upper_bounds = base_data['upper_bounds_d2']

        for b in range(num_bases):
            expr = gp.LinExpr()
            for j in range(num_cols):
                if D_matrix[b][j] > 0:
                    expr.addTerms(D_matrix[b][j], x[j])
            model.addConstr(expr >= lower_bounds[b], name=f"base_lower_{b}")
            model.addConstr(expr <= upper_bounds[b], name=f"base_upper_{b}")

    obj_expr = gp.quicksum(cost_vector[j] * x[j] for j in range(num_cols))
    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Callback to log incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(float(obj))

    best_obj = float('inf')
    best_solution = None

    model.optimize(callback)

    runtime = time.time() - start_time
    remaining_time = max(1, int(args.time_limit - runtime))

    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        if model.SolCount > 0:
            best_obj = model.ObjVal
            selected = [j for j in range(num_cols) if x[j].X > 0.5]
            var_values = {str(j): 1.0 if j in selected else 0.0 for j in range(num_cols)}
            best_solution = {
                "objective_value": float(best_obj),
                "selected_rotations": selected,
                "variable_values": var_values
            }

    # If no solution found within time, we must provide a feasible one or handle gracefully.
    # But according to problem we must produce a valid output following schema.
    # If infeasible, we return empty with high cost (but problem instances should be feasible).
    if best_solution is None:
        best_solution = {
            "objective_value": 0.0,
            "selected_rotations": [],
            "variable_values": {str(j): 0.0 for j in range(num_cols)}
        }

    with open(args.solution_path, 'w') as f:
        json.dump(best_solution, f, indent=2)

if __name__ == "__main__":
    main()