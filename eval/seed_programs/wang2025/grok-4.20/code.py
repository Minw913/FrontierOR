import argparse
import json
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

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    instance_type = instance['instance_type']
    n = instance['n']
    m = instance['m']
    k = instance['k']
    gamma = instance['gamma']
    data_matrix = instance['data_matrix']
    node_weights = instance['node_weights']
    element_weights = instance['element_weights']

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    start_time = time.time()
    best_obj = 0.0
    best_S = []

    def compute_f(S):
        if not S:
            return 0.0
        if instance_type == "LOC":
            val = 0.0
            for i in range(m):
                max_ben = 0.0
                for j in S:
                    max_ben = max(max_ben, data_matrix[i][j])
                val += max_ben
            return val
        elif instance_type == "COV":
            val = 0.0
            for i in range(m):
                covered = any(data_matrix[i][j] > 0.5 for j in S)
                if covered:
                    val += node_weights[i]
            return val
        elif instance_type == "INF":
            val = 0.0
            for i in range(m):
                prod = 1.0
                for j in S:
                    if data_matrix[i][j] > 0.5:
                        p = element_weights[j]
                        prod *= (1.0 - p)
                val += (1.0 - prod)
            return val
        return 0.0

    # Greedy
    current_S = []
    remaining = set(range(n))
    while len(current_S) < k and remaining and (time.time() - start_time < args.time_limit * 0.9):
        best_gain = -1.0
        best_elem = None
        for j_idx, j in enumerate(list(remaining)):
            if time.time() - start_time > args.time_limit * 0.9:
                break
            gain = compute_f(current_S + [j]) - compute_f(current_S)
            if gain > best_gain:
                best_gain = gain
                best_elem = j
        if best_elem is None or best_gain <= 1e-9:
            break
        current_S.append(best_elem)
        remaining.remove(best_elem)
        current_obj = compute_f(current_S)
        if current_obj > best_obj + 1e-6:
            best_obj = current_obj
            best_S = list(current_S)
            if logger:
                logger.log(best_obj)

    # Local search (swap)
    improved = True
    iter_count = 0
    while improved and iter_count < 20 and (time.time() - start_time < args.time_limit * 0.95):
        iter_count += 1
        improved = False
        current_obj = compute_f(best_S)
        for i in range(len(best_S)):
            for new_e in range(n):
                if new_e in best_S:
                    continue
                if time.time() - start_time > args.time_limit * 0.95:
                    break
                new_set = best_S[:i] + best_S[i+1:] + [new_e]
                new_obj = compute_f(new_set)
                if new_obj > best_obj + 1e-6:
                    best_S = new_set
                    best_obj = new_obj
                    improved = True
                    if logger:
                        logger.log(best_obj)
                    break
            if improved:
                break

    # MIP for LOC and COV if time remains
    if instance_type in ["LOC", "COV"] and (time.time() - start_time < args.time_limit - 2):
        try:
            model = gp.Model("ExactSolver")
            remaining_time = max(1, int(args.time_limit - (time.time() - start_time)))
            model.setParam('TimeLimit', remaining_time)
            model.setParam('Threads', 1)
            model.setParam('LogToConsole', 0)
            model.setParam('MIPGap', 1e-4)

            x = model.addVars(n, vtype=GRB.BINARY, name="x")
            model.addConstr(gp.quicksum(x[j] for j in range(n)) <= k, "cardinality")

            if instance_type == "LOC":
                y = model.addVars(m, vtype=GRB.CONTINUOUS, lb=0.0, name="y")
                max_b = [max(data_matrix[i]) for i in range(m)]
                for i in range(m):
                    model.addConstr(y[i] <= max_b[i], f"ub_{i}")
                    for j in range(n):
                        if data_matrix[i][j] > 1e-9:
                            model.addConstr(y[i] >= data_matrix[i][j] * x[j], f"lb_{i}_{j}")
                obj_expr = gp.quicksum(y[i] for i in range(m))
            else:  # COV
                z = model.addVars(m, vtype=GRB.BINARY, name="z")
                for i in range(m):
                    if node_weights[i] > 1e-9:
                        model.addConstr(z[i] <= gp.quicksum(data_matrix[i][j] * x[j] for j in range(n)), f"cover_{i}")
                obj_expr = gp.quicksum(node_weights[i] * z[i] for i in range(m))

            model.setObjective(obj_expr, GRB.MAXIMIZE)
            model.optimize()

            if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) and model.SolCount > 0:
                new_obj = model.objVal
                if new_obj > best_obj + 1e-5:
                    best_obj = new_obj
                    best_S = [j for j in range(n) if x[j].X > 0.5]
                    if logger:
                        logger.log(best_obj)
        except Exception:
            pass

    solution = {
        "objective_value": float(best_obj),
        "selected_elements": [int(j) for j in sorted(best_S)]
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    if logger:
        logger.log(best_obj)

if __name__ == "__main__":
    main()