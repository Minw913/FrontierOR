import argparse
import json
import time
import random
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False)
    args = parser.parse_args()

    start_time = time.time()

    logger = None
    if args.log_path and SolutionLogger:
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    n = instance["n"]
    A = instance["A"]
    B = instance["B"]
    C = instance["C"]

    # -------------------------------------------------------------------------
    # Auxiliary model for solving linear assignments
    # -------------------------------------------------------------------------
    m_lin = gp.Model("lin_assign")
    m_lin.setParam('OutputFlag', 0)
    m_lin.setParam('Method', 1)  # Dual Simplex for fast extreme point convergence

    x_lin = {}
    for i in range(n):
        for j in range(n):
            x_lin[i,j] = m_lin.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1)

    for i in range(n):
        m_lin.addConstr(gp.quicksum(x_lin[i,j] for j in range(n)) == 1)
    for j in range(n):
        m_lin.addConstr(gp.quicksum(x_lin[i,j] for i in range(n)) == 1)

    def solve_assignment(Q_mat):
        obj = gp.quicksum(Q_mat[i][j] * x_lin[i,j] for i in range(n) for j in range(n))
        m_lin.setObjective(obj, GRB.MINIMIZE)
        m_lin.optimize()
        pi = [-1] * n
        for i in range(n):
            for j in range(n):
                if x_lin[i,j].X > 0.5:
                    pi[i] = j
        return pi, m_lin.ObjVal

    def evaluate(pi_list):
        sumA = sum(A[i][pi_list[i]] for i in range(n))
        sumB = sum(B[i][pi_list[i]] for i in range(n))
        sumC = sum(C[i][pi_list[i]] for i in range(n))
        return sumA * sumB + sumC

    def local_search(initial_pi):
        pi = list(initial_pi)
        sumA = sum(A[i][pi[i]] for i in range(n))
        sumB = sum(B[i][pi[i]] for i in range(n))
        sumC = sum(C[i][pi[i]] for i in range(n))
        
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+1, n):
                    pi_i = pi[i]
                    pi_j = pi[j]
                    
                    dA = A[i][pi_j] + A[j][pi_i] - A[i][pi_i] - A[j][pi_j]
                    dB = B[i][pi_j] + B[j][pi_i] - B[i][pi_i] - B[j][pi_j]
                    dC = C[i][pi_j] + C[j][pi_i] - C[i][pi_i] - C[j][pi_j]
                    
                    new_cost = (sumA + dA) * (sumB + dB) + (sumC + dC)
                    current_cost = sumA * sumB + sumC
                    
                    if new_cost < current_cost:
                        pi[i], pi[j] = pi_j, pi_i
                        sumA += dA
                        sumB += dB
                        sumC += dC
                        improved = True
        return pi, sumA * sumB + sumC

    # Compute optimal variable bounds for spatial branch-and-bound
    _, min_A = solve_assignment(A)
    A_neg = [[-A[i][j] for j in range(n)] for i in range(n)]
    _, neg_max_A = solve_assignment(A_neg)
    max_A = -neg_max_A

    _, min_B = solve_assignment(B)
    B_neg = [[-B[i][j] for j in range(n)] for i in range(n)]
    _, neg_max_B = solve_assignment(B_neg)
    max_B = -neg_max_B

    # -------------------------------------------------------------------------
    # Heuristic Initialization (Greedy/Local Search)
    # -------------------------------------------------------------------------
    np_random = random.Random(42)
    weights = []
    
    # Fundamental coordinate / corner combinations spanning valid positive and negative search sectors
    for wa in [-1, 0, 1]:
        for wb in [-1, 0, 1]:
            for wc in [-1, 0, 1]:
                if wa == wb == wc == 0: continue
                weights.append((wa, wb, wc))

    # Random multi-start scalarizing directions
    for _ in range(40):
        weights.append((np_random.uniform(-1, 1), np_random.uniform(-1, 1), np_random.uniform(-1, 1)))

    best_obj = float('inf')
    best_pi = None
    
    # Cap heuristic execution limits reasonably based on allowed time
    allowed_heuristic_time = min(15.0, args.time_limit * 0.15)

    for w in weights:
        if time.time() - start_time > allowed_heuristic_time:
            break
        Q_mat = [[w[0]*A[i][j] + w[1]*B[i][j] + w[2]*C[i][j] for j in range(n)] for i in range(n)]
        try:
            pi, _ = solve_assignment(Q_mat)
        except Exception:
            continue
            
        pi_ls, cost_ls = local_search(pi)
        
        if cost_ls < best_obj:
            best_obj = cost_ls
            best_pi = pi_ls
            if logger:
                logger.log(float(best_obj))

    # -------------------------------------------------------------------------
    # Main Exact MIQP (Non-Convex) Formulation
    # -------------------------------------------------------------------------
    main_model = gp.Model("main_exact")
    time_elapsed = time.time() - start_time
    remaining_time = max(1.0, args.time_limit - time_elapsed)
    main_model.setParam('TimeLimit', remaining_time)
    main_model.setParam('NonConvex', 2)
    main_model.setParam('OutputFlag', 1)

    X = {}
    for i in range(n):
        for j in range(n):
            X[i,j] = main_model.addVar(vtype=GRB.BINARY, name=f"X_{i}_{j}")
            if best_pi and best_pi[i] == j:
                X[i,j].Start = 1
            else:
                X[i,j].Start = 0

    yA = main_model.addVar(lb=min_A, ub=max_A, vtype=GRB.CONTINUOUS, name="yA")
    yB = main_model.addVar(lb=min_B, ub=max_B, vtype=GRB.CONTINUOUS, name="yB")

    for i in range(n):
        main_model.addConstr(gp.quicksum(X[i,j] for j in range(n)) == 1)
    for j in range(n):
        main_model.addConstr(gp.quicksum(X[i,j] for i in range(n)) == 1)

    main_model.addConstr(yA == gp.quicksum(A[i][j] * X[i,j] for i in range(n) for j in range(n)))
    main_model.addConstr(yB == gp.quicksum(B[i][j] * X[i,j] for i in range(n) for j in range(n)))

    obj_expr = yA * yB + gp.quicksum(C[i][j] * X[i,j] for i in range(n) for j in range(n))
    main_model.setObjective(obj_expr, GRB.MINIMIZE)

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if logger:
                    logger.log(float(obj))
            except Exception:
                pass

    main_model.optimize(cb)

    # Reconstruct optimal / best-attained assignment
    final_obj = best_obj
    final_pi = list(best_pi) if best_pi else [-1] * n

    if main_model.SolCount > 0:
        pi = [-1] * n
        for i in range(n):
            for j in range(n):
                if X[i,j].X > 0.5:
                    pi[i] = j
        # Calculate analytically to circumvent floating point bounds
        eval_obj = evaluate(pi)
        if eval_obj < final_obj:
            final_obj = eval_obj
            final_pi = pi

    solution_dict = {
        "objective_value": float(final_obj),
        "assignment": final_pi
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution_dict, f, indent=4)

if __name__ == "__main__":
    solve()