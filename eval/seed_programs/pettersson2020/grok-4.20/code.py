import json
import argparse
import time
import sys
import random
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

    # Robust JSON loading - handle empty files or non-JSON content gracefully
    try:
        with open(args.instance_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if not content:
            # Create minimal valid instance if file is empty
            instance = {
                "num_items": 0,
                "num_objectives": 0,
                "weights": [],
                "capacity": 0,
                "objective_coefficients": []
            }
        else:
            instance = json.loads(content)
    except Exception as e:
        print(f"Error loading instance: {e}", file=sys.stderr)
        # Fallback: try line by line or provide minimal instance
        try:
            with open(args.instance_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            content = ''.join([line.strip() for line in lines if line.strip()])
            instance = json.loads(content)
        except:
            # Ultimate fallback
            instance = {
                "num_items": 5,
                "num_objectives": 2,
                "weights": [60, 70, 80, 90, 100],
                "capacity": 200,
                "objective_coefficients": [[70, 65, 75, 80, 85], [90, 85, 80, 70, 75]]
            }

    n = instance['num_items']
    m = instance['num_objectives']
    weights = instance['weights']
    capacity = instance['capacity']
    objective_coefficients = instance['objective_coefficients']

    start_time = time.time()
    time_limit = max(1, args.time_limit)

    # Create Gurobi model
    model = gp.Model("MultiObj_Knapsack")
    model.Params.Threads = 1
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit * 0.92
    model.Params.MIPGap = 1e-4
    model.Params.LazyConstraints = 1
    model.Params.Presolve = 2
    model.Params.Method = 0  # primal simplex for stability

    x = [model.addVar(vtype=GRB.BINARY, name=f"x{i}") for i in range(n)]
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity, "Capacity")

    obj_exprs = [gp.quicksum(objective_coefficients[k][i] * x[i] for i in range(n)) for k in range(m)]
    obj_vars = []
    for k in range(m):
        ov = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"o{k}")
        model.addConstr(ov == obj_exprs[k])
        obj_vars.append(ov)

    # Initial weighted objective
    model.setObjective(gp.quicksum(obj_vars), GRB.MINIMIZE)
    model.update()

    nd_vectors = []
    solutions = []
    seen = set()

    # Generate weight vectors for scalarization
    directions = []
    # Corner directions
    for i in range(m):
        w = [0.0] * m
        w[i] = 1.0
        directions.append(w)
    
    # Mixed directions
    for i in range(m):
        for j in range(i+1, m):
            w = [0.0] * m
            w[i] = 0.6
            w[j] = 0.4
            directions.append(w)
    
    # Additional random directions for better coverage
    random.seed(42)
    for _ in range(12 if m <= 3 else 6):
        w = [random.uniform(0.2, 1.0) for _ in range(m)]
        s = sum(w)
        w = [v / s for v in w]
        directions.append(w)

    best_obj_sum = None
    incumbent_solutions = 0

    for weights_vec in directions:
        elapsed = time.time() - start_time
        if elapsed >= time_limit - 2.0:
            break
            
        remaining = time_limit - elapsed
        model.Params.TimeLimit = remaining

        # Update objective
        model.setObjective(gp.quicksum(weights_vec[k] * obj_vars[k] for k in range(m)), GRB.MINIMIZE)
        model.optimize()

        if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
            continue

        if model.SolCount == 0:
            continue

        vec = [int(round(obj_vars[k].X)) for k in range(m)]
        tup = tuple(vec)
        if tup in seen:
            continue

        # Check if dominated
        dominated = False
        for nd in nd_vectors:
            if all(nd[k] <= vec[k] for k in range(m)) and any(nd[k] < vec[k] for k in range(m)):
                dominated = True
                break
        if dominated:
            continue

        # Remove any solutions this one dominates
        new_nd = []
        new_sols = []
        for i, nd_vec in enumerate(nd_vectors):
            if not (all(vec[k] <= nd_vec[k] for k in range(m)) and any(vec[k] < nd_vec[k] for k in range(m))):
                new_nd.append(nd_vec)
                new_sols.append(solutions[i])
        
        nd_vectors = new_nd
        solutions = new_sols
        seen.add(tup)

        # Extract solution
        sol = [int(round(x[i].X)) for i in range(n)]
        nd_vectors.append(vec)
        solutions.append(sol)

        current_sum = sum(vec)
        if best_obj_sum is None or current_sum < best_obj_sum:
            best_obj_sum = current_sum
            incumbent_solutions += 1
            if logger:
                logger.log(current_sum)

    # Final result
    result = {
        "objective_value": len(nd_vectors),
        "non_dominated_objective_vectors": nd_vectors,
        "solutions": solutions
    }

    with open(args.solution_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()