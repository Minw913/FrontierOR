import gurobipy as grb
import json
import time
import argparse
from solution_logger import SolutionLogger

def compute_cost(A, B, C, assignment):
    s_A = 0.0
    s_B = 0.0
    s_C = 0.0
    n = len(assignment)
    for i in range(n):
        j = assignment[i]
        s_A += A[i][j]
        s_B += B[i][j]
        s_C += C[i][j]
    return s_A * s_B + s_C

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    n = data['n']
    A = data['A']
    B = data['B']
    C = data['C']

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start_time = time.time()

    # Heuristic: generate trivial solution and linear assignment solution
    cand0 = list(range(n))
    cost0 = compute_cost(A, B, C, cand0)
    best_candidate = cand0
    best_cost = cost0

    time_for_heuristic = min(1.0, args.time_limit)
    if time_for_heuristic > 0:
        D = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(A[i][j] + B[i][j] + C[i][j])
            D.append(row)
        
        try:
            model_heur = grb.Model()
            model_heur.Params.LogToConsole = 0
            model_heur.Params.OutputFlag = 0
            model_heur.Params.TimeLimit = time_for_heuristic
            x_heur = model_heur.addVars(n, n, vtype=grb.GRB.BINARY)
            for i in range(n):
                model_heur.addConstr(grb.quicksum(x_heur[i, j] for j in range(n)) == 1)
            for j in range(n):
                model_heur.addConstr(grb.quicksum(x_heur[i, j] for i in range(n)) == 1)
            obj_heur = grb.quicksum(D[i][j] * x_heur[i, j] for i in range(n) for j in range(n))
            model_heur.setObjective(obj_heur, grb.GRB.MINIMIZE)
            model_heur.optimize()
            
            if model_heur.SolCount > 0:
                cand1 = [0] * n
                for i in range(n):
                    for j in range(n):
                        if abs(x_heur[i, j].X) > 0.5:
                            cand1[i] = j
                cost1 = compute_cost(A, B, C, cand1)
                if cost1 < best_cost:
                    best_candidate = cand1
                    best_cost = cost1
        except grb.GurobiError as e:
            pass

    if logger:
        logger.log(best_cost)

    elapsed = time.time() - start_time
    remaining_time = max(0, args.time_limit - elapsed)
    if remaining_time > 0:
        try:
            model = grb.Model()
            model.Params.LogToConsole = 0
            model.Params.OutputFlag = 0
            model.Params.TimeLimit = remaining_time
            model.Params.NonConvex = 2

            x = model.addVars(n, n, vtype=grb.GRB.BINARY, name="x")
            for i in range(n):
                model.addConstr(grb.quicksum(x[i, j] for j in range(n)) == 1)
            for j in range(n):
                model.addConstr(grb.quicksum(x[i, j] for i in range(n)) == 1)

            s_A = grb.quicksum(A[i][j] * x[i, j] for i in range(n) for j in range(n))
            s_B = grb.quicksum(B[i][j] * x[i, j] for i in range(n) for j in range(n))
            s_C = grb.quicksum(C[i][j] * x[i, j] for i in range(n) for j in range(n))

            model.setObjective(s_A * s_B + s_C, grb.GRB.MINIMIZE)

            for i in range(n):
                j_val = best_candidate[i]
                x[i, j_val].Start = 1

            if logger:
                model._logger = logger
                def callback(model, where):
                    if where == grb.GRB.Callback.MIPSOL:
                        objval = model.cbGet(grb.GRB.Callback.MIPSOL_OBJ)
                        model._logger.log(objval)
                model.optimize(callback)
            else:
                model.optimize()

            if model.SolCount > 0:
                assignment = [0] * n
                for i in range(n):
                    for j in range(n):
                        if abs(x[i, j].X) > 0.5:
                            assignment[i] = j
                objective_value = model.ObjVal
            else:
                assignment = best_candidate
                objective_value = best_cost
        except grb.GurobiError as e:
            assignment = best_candidate
            objective_value = best_cost
    else:
        assignment = best_candidate
        objective_value = best_cost

    solution = {
        "objective_value": float(objective_value),
        "assignment": assignment
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()