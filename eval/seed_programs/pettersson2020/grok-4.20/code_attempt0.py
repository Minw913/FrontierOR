import json
import argparse
import time
import sys
from typing import List, Tuple
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
        instance = json.load(f)

    n = instance['num_items']
    m = instance['num_objectives']
    weights = instance['weights']
    capacity = instance['capacity']
    coeffs = instance['objective_coefficients']

    # Precompute individual item contributions for bounding
    item_contribs = []
    for i in range(n):
        vec = [coeffs[k][i] for k in range(m)]
        item_contribs.append((weights[i], vec))

    # Build a simple lower bound vector (sum of negative contributions)
    lb_vec = [0] * m
    for k in range(m):
        for val in coeffs[k]:
            if val < 0:
                lb_vec[k] += val

    start_time = time.time()
    best_obj_val = float('inf')
    best_nd_vectors = []
    best_solutions = []

    # We will use an epsilon-constraint like approach but since multi-objective,
    # we generate the full Pareto front by iteratively solving MIPs with cuts.
    # Given small expected n (<=25 probably given weights 60-100 and capacity ~0.5*sum),
    # we can afford a reasonable number of iterations.

    model = gp.Model("multiobjective_knapsack")
    model.Params.Threads = 1
    model.Params.TimeLimit = args.time_limit - 5  # reserve some margin
    model.Params.OutputFlag = 0

    x = [model.addVar(vtype=GRB.BINARY, name=f"x_{i}") for i in range(n)]
    weight_constr = model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity)

    # Auxiliary continuous variables for each objective
    obj_vars = []
    for k in range(m):
        obj_var = model.addVar(lb=lb_vec[k]-10, ub=10000, vtype=GRB.CONTINUOUS, name=f"obj_{k}")
        model.addConstr(obj_var == gp.quicksum(coeffs[k][i] * x[i] for i in range(n)))
        obj_vars.append(obj_var)

    # We will collect all non-dominated points by iteratively finding the next best
    # according to a weighted sum or by adding cuts that eliminate dominated regions.
    # Here we use the common "weighted sum with normalization + lexicographic" but
    # to get the full set we use the epsilon-constraint method in a loop with dynamic
    # cuts on the objective space.

    found_points = []
    epsilon_cuts = []

    iteration = 0
    max_iterations = 200  # safety

    while time.time() - start_time < args.time_limit - 2 and iteration < max_iterations:
        iteration += 1

        # Objective: minimize the maximum normalized deviation from ideal (Chebyshev)
        # But to avoid needing ideal point computation per iteration, we minimize a dummy
        # and use hierarchical optimization via lexicographic on all objectives with small weights.

        # A practical way: minimize sum of objectives with decreasing weights so that
        # improving first objective is always preferred, then second, etc.
        weights_obj = [1.0]
        for k in range(1, m):
            weights_obj.append(weights_obj[-1] * 0.0001)

        current_obj = gp.quicksum(weights_obj[k] * obj_vars[k] for k in range(m))

        model.setObjective(current_obj, GRB.MINIMIZE)

        # Add previous epsilon cuts to avoid previously found or dominated regions
        for cut in epsilon_cuts:
            model.addConstr(cut)

        model.optimize()

        if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            break

        # Extract solution
        sol_vector = [int(round(x[i].X)) for i in range(n)]
        obj_vector = [int(round(obj_vars[k].X)) for k in range(m)]
        obj_sum = sum(obj_vector)  # scalar for comparison and logging

        # Check if this point is dominated by any already found
        dominated = False
        for existing in found_points:
            if all(existing[j] <= obj_vector[j] for j in range(m)) and \
               any(existing[j] < obj_vector[j] for j in range(m)):
                dominated = True
                break
        if dominated:
            # Add a small perturbation cut to move away
            epsilon_cuts.append(gp.quicksum(obj_vars[k] for k in range(m)) >= obj_sum + 1)
            continue

        # New non-dominated point found
        found_points.append(obj_vector[:])
        if logger:
            logger.log(obj_sum)

        # Store
        best_nd_vectors.append(obj_vector)
        best_solutions.append(sol_vector)

        # Add cuts to exclude this point and any point that dominates it
        # Cut 1: exclude exactly this point
        exact_cut = gp.quicksum((1 - 2 * sol_vector[i]) * x[i] for i in range(n)) >= 1
        model.addConstr(exact_cut)

        # Cut 2: force improvement on at least one objective (non-domination cut)
        # This is: at least one objective must be strictly better than current
        # Implemented via big-M on binary indicators
        delta = model.addVars(m, vtype=GRB.BINARY, name=f"delta_{iteration}")
        bigM = 100000
        for k in range(m):
            model.addConstr(obj_vars[k] <= obj_vector[k] - 1 + bigM * (1 - delta[k]))
        model.addConstr(gp.quicksum(delta[k] for k in range(m)) >= 1)

        # Also add a cut that prevents any point that is worse in all but one and equal in one, etc.
        # But the above plus the exact exclusion is usually sufficient to enumerate the front.
        # To speed up we also add a linear cut that forbids all points >= this point (dominated by it)
        dominated_by_this = gp.quicksum(obj_vars[k] for k in range(m)) >= obj_sum + 0.5
        epsilon_cuts.append(dominated_by_this)

        # Update best
        if obj_sum < best_obj_val:
            best_obj_val = obj_sum

        if model.Status == GRB.TIME_LIMIT:
            break

    # If nothing found, we need at least the zero solution or a feasible one.
    # Solve a simple feasibility problem if no points yet.
    if not best_nd_vectors:
        model.reset()
        model.setObjective(0, GRB.MINIMIZE)
        model.optimize()
        if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and weight_constr.Pi > -GRB.INFINITY:
            sol_vector = [int(round(x[i].X)) for i in range(n)]
            obj_vector = [int(round(obj_vars[k].X)) for k in range(m)]
            best_nd_vectors.append(obj_vector)
            best_solutions.append(sol_vector)
            if logger:
                logger.log(sum(obj_vector))

    # Final output
    result = {
        "objective_value": len(best_nd_vectors),
        "non_dominated_objective_vectors": best_nd_vectors,
        "solutions": best_solutions
    }

    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)

    # Ensure we respect time
    elapsed = time.time() - start_time
    if elapsed > args.time_limit:
        sys.exit(0)

if __name__ == "__main__":
    main()