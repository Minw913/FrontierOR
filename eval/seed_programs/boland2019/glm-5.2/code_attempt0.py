import argparse
import json
import time
import sys
import gurobipy as gp
from gurobipy import GRB

sys.setrecursionlimit(100000)

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None


def filter_dominated(solutions):
    if not solutions:
        return []
    pts = sorted(solutions, key=lambda s: (s['z1'], s['z2']))
    filtered = []
    for s in pts:
        if filtered and s['z2'] >= filtered[-1]['z2']:
            continue
        filtered.append(s)
    return filtered


def write_output(path, objective_value, solutions):
    pareto_front = [[s['z1'], s['z2']] for s in solutions]
    output = {
        'objective_value': float(objective_value) if objective_value is not None else 0.0,
        'pareto_front': pareto_front,
        'solutions': solutions
    }
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    start_time = time.time()

    with open(args.instance_path) as f:
        data = json.load(f)

    m = data['num_facilities']
    n = data['num_customers']
    demands = data['demands']
    capacities = data['capacities']
    fc1 = data['fixed_costs_obj1']
    fc2 = data['fixed_costs_obj2']
    ac1 = data['assignment_costs_obj1']
    ac2 = data['assignment_costs_obj2']

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    # Build persistent Gurobi model
    model = gp.Model("flp")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    y = model.addVars(m, vtype=GRB.BINARY, name="y")
    x = model.addVars(m, n, vtype=GRB.BINARY, name="x")

    for i in range(n):
        model.addConstr(gp.quicksum(x[j, i] for j in range(m)) == 1)
    for j in range(m):
        for i in range(n):
            model.addConstr(x[j, i] <= y[j])
    for j in range(m):
        model.addConstr(
            gp.quicksum(demands[i] * x[j, i] for i in range(n)) <= capacities[j] * y[j]
        )

    z1_expr = (
        gp.quicksum(fc1[j] * y[j] for j in range(m))
        + gp.quicksum(ac1[j][i] * x[j, i] for j in range(m) for i in range(n))
    )
    z2_expr = (
        gp.quicksum(fc2[j] * y[j] for j in range(m))
        + gp.quicksum(ac2[j][i] * x[j, i] for j in range(m) for i in range(n))
    )

    z1_var = model.addVar(vtype=GRB.INTEGER, lb=0, name="z1")
    z2_var = model.addVar(vtype=GRB.INTEGER, lb=0, name="z2")
    model.addConstr(z1_var == z1_expr)
    model.addConstr(z2_var == z2_expr)

    extra_constraints = []

    def clear_extras():
        for c in extra_constraints:
            model.remove(c)
        extra_constraints.clear()

    def add_bounds(z1_lo=None, z1_hi=None, z2_lo=None, z2_hi=None):
        if z1_lo is not None:
            extra_constraints.append(model.addConstr(z1_var >= z1_lo))
        if z1_hi is not None:
            extra_constraints.append(model.addConstr(z1_var <= z1_hi))
        if z2_lo is not None:
            extra_constraints.append(model.addConstr(z2_var >= z2_lo))
        if z2_hi is not None:
            extra_constraints.append(model.addConstr(z2_var <= z2_hi))

    def get_solution():
        if model.SolCount == 0:
            return None
        open_facilities = [j for j in range(m) if y[j].X > 0.5]
        assignments = {}
        for i in range(n):
            for j in range(m):
                if x[j, i].X > 0.5:
                    assignments[str(i)] = j
                    break
        z1_val = int(round(z1_var.X))
        z2_val = int(round(z2_var.X))
        return {
            'z1': z1_val,
            'z2': z2_val,
            'open_facilities': open_facilities,
            'assignments': assignments,
        }

    def time_left():
        return args.time_limit - (time.time() - start_time)

    best_weighted_obj = None
    best_weighted_sol = None
    pareto_solutions = []

    # --- 1. Weighted-sum solve (primary objective) ---
    clear_extras()
    model.setObjective(0.5 * z1_var + 0.5 * z2_var, GRB.MINIMIZE)
    tl_remaining = time_left()
    model.Params.TimeLimit = max(1.0, min(tl_remaining - 2, tl_remaining * 0.5))
    model.optimize()
    sol = get_solution()
    if sol is not None:
        obj_val = 0.5 * sol['z1'] + 0.5 * sol['z2']
        best_weighted_obj = obj_val
        best_weighted_sol = sol
        pareto_solutions.append(sol)
        if logger:
            logger.log_solution(obj_val, sol)

    if time_left() < 3:
        final = filter_dominated(pareto_solutions)
        write_output(args.solution_path, best_weighted_obj, final)
        return

    # --- 2. Find bounds via min-z1 and min-z2 ---
    # min z1
    clear_extras()
    model.setObjective(z1_var, GRB.MINIMIZE)
    tl_remaining = time_left()
    model.Params.TimeLimit = max(1.0, min(tl_remaining - 2, tl_remaining * 0.3))
    model.optimize()
    sol_minz1 = get_solution()

    if time_left() < 3:
        final = filter_dominated(pareto_solutions)
        write_output(args.solution_path, best_weighted_obj, final)
        return

    # min z2
    clear_extras()
    model.setObjective(z2_var, GRB.MINIMIZE)
    tl_remaining = time_left()
    model.Params.TimeLimit = max(1.0, min(tl_remaining - 2, tl_remaining * 0.3))
    model.optimize()
    sol_minz2 = get_solution()

    if sol_minz1 is None or sol_minz2 is None:
        final = filter_dominated(pareto_solutions)
        write_output(args.solution_path, best_weighted_obj, final)
        return

    z1_min = sol_minz1['z1']
    z2_hi_bound = sol_minz1['z2']  # valid upper bound on z2 of non-dominated points
    z1_hi_bound = sol_minz2['z1']  # valid upper bound on z1 of non-dominated points
    z2_min = sol_minz2['z2']

    # These might themselves be non-dominated; include and let filter handle
    pareto_solutions.append(sol_minz1)
    pareto_solutions.append(sol_minz2)

    # --- 3. Recursive box algorithm (epsilon-constraint style) ---
    def explore(z1_lo, z1_hi, z2_hi):
        tl_remaining = time_left()
        if tl_remaining < 2:
            return
        if z1_lo > z1_hi:
            return
        clear_extras()
        add_bounds(z1_lo=z1_lo, z1_hi=z1_hi, z2_hi=z2_hi)
        model.setObjective(z2_var, GRB.MINIMIZE)
        tl = max(1.0, min(tl_remaining - 1, tl_remaining * 0.15))
        model.Params.TimeLimit = tl
        model.optimize()
        sol = get_solution()
        if sol is None:
            return
        z1_star = sol['z1']
        z2_star = sol['z2']
        pareto_solutions.append(sol)
        obj_val = 0.5 * z1_star + 0.5 * z2_star
        if best_weighted_obj is None or obj_val < best_weighted_obj - 1e-9:
            best_weighted_obj = obj_val
            best_weighted_sol = sol
            if logger:
                logger.log_solution(obj_val, sol)
        # If the solve was not optimal, stop recursing to avoid incorrect partitioning
        if model.Status != GRB.OPTIMAL:
            return
        explore(z1_lo, z1_star - 1, z2_hi)
        explore(z1_star + 1, z1_hi, z2_star - 1)

    explore(z1_min, z1_hi_bound, z2_hi_bound)

    # --- 4. Filter dominated and write output ---
    final = filter_dominated(pareto_solutions)
    write_output(args.solution_path, best_weighted_obj, final)


if __name__ == '__main__':
    main()