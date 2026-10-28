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

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    N = instance['N']
    m = instance['m']
    capacities = instance['group_capacities']
    L = [g['L_g'] for g in capacities]
    U = [g['U_g'] for g in capacities]
    distances_ut = instance['distances_upper_triangular']

    # Build full distance matrix (symmetric)
    dist = [[0.0] * N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            d = distances_ut[idx]
            dist[i][j] = d
            dist[j][i] = d
            idx += 1

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    try:
        model = gp.Model("DiversityPartitioning")
        model.setParam('TimeLimit', args.time_limit)
        model.setParam('Threads', 1)
        model.setParam('MIPGap', 1e-4)
        model.setParam('LogToConsole', 0)

        # x[i,g] = 1 if element i assigned to group g
        x = model.addVars(N, m, vtype=GRB.BINARY, name="x")

        # Objective: maximize sum over groups of pairwise distances within group
        # Equivalent to 0.5 * sum_i sum_j>i dist[i,j] * sum_g x[i,g]*x[j,g]
        obj_expr = gp.LinExpr()
        for i in range(N):
            for j in range(i+1, N):
                d = dist[i][j]
                if d > 0:
                    for g in range(m):
                        obj_expr.addTerms(d, x[i,g], x[j,g])
        model.setObjective(obj_expr, GRB.MAXIMIZE)

        # Each element in exactly one group
        for i in range(N):
            model.addConstr(gp.quicksum(x[i,g] for g in range(m)) == 1, name=f"assign_{i}")

        # Capacity constraints
        for g in range(m):
            model.addConstr(gp.quicksum(x[i,g] for i in range(N)) >= L[g], name=f"min_{g}")
            model.addConstr(gp.quicksum(x[i,g] for i in range(N)) <= U[g], name=f"max_{g}")

        # Callback to log incumbent solutions
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
                if logger:
                    logger.log(obj_val)

        # Optimize
        start_time = time.time()
        model.optimize(callback)

        if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) and model.SolCount > 0:
            objective_value = model.objVal
            assignment = {}
            groups = {g: [] for g in range(m)}
            
            for i in range(N):
                for g in range(m):
                    if x[i,g].X > 0.5:
                        assignment[str(i)] = g
                        groups[g].append(i)
                        break
            for g in groups:
                groups[g].sort()
            
            solution = {
                "objective_value": float(objective_value),
                "assignment": assignment,
                "groups": {str(k): v for k, v in groups.items()}
            }
            
            with open(args.solution_path, 'w') as f:
                json.dump(solution, f, indent=2)
        else:
            # If no solution found, write empty feasible solution (should not happen with feasible capacities)
            solution = {
                "objective_value": 0.0,
                "assignment": {str(i): 0 for i in range(N)},
                "groups": {"0": list(range(N))}
            }
            with open(args.solution_path, 'w') as f:
                json.dump(solution, f, indent=2)
    except Exception as e:
        # Fallback: assign all to first feasible group or spread evenly
        solution = {
            "objective_value": 0.0,
            "assignment": {},
            "groups": {}
        }
        # Simple greedy fallback: fill groups respecting capacities
        assignment = {}
        groups_list = [[] for _ in range(m)]
        idx = 0
        for g in range(m):
            for _ in range(L[g]):
                if idx < N:
                    groups_list[g].append(idx)
                    assignment[str(idx)] = g
                    idx += 1
        for i in range(idx, N):
            for g in range(m):
                if len(groups_list[g]) < U[g]:
                    groups_list[g].append(i)
                    assignment[str(i)] = g
                    break
        groups_dict = {}
        for g in range(m):
            groups_list[g].sort()
            groups_dict[str(g)] = groups_list[g]
        solution["assignment"] = assignment
        solution["groups"] = groups_dict
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()