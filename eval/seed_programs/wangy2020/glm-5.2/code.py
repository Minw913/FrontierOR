import argparse
import json
import time
import random
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        instance = json.load(f)
    n = instance['n']
    A = instance['A']
    B = instance['B']
    C = instance['C']

    if n == 0:
        sol = {"objective_value": 0.0, "assignment": []}
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f)
        if logger:
            logger.log_solution(0.0, sol)
        return

    start_time = time.time()

    def compute_obj(perm):
        a = sum(A[i][perm[i]] for i in range(n))
        b = sum(B[i][perm[i]] for i in range(n))
        c = sum(C[i][perm[i]] for i in range(n))
        return a * b + c, a, b, c

    def two_opt(perm, max_iter=100000):
        a = sum(A[i][perm[i]] for i in range(n))
        b = sum(B[i][perm[i]] for i in range(n))
        c = sum(C[i][perm[i]] for i in range(n))
        obj = a * b + c
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            for i in range(n):
                pi = perm[i]
                for j in range(i + 1, n):
                    pj = perm[j]
                    dA = A[i][pj] + A[j][pi] - A[i][pi] - A[j][pj]
                    dB = B[i][pj] + B[j][pi] - B[i][pi] - B[j][pj]
                    dC = C[i][pj] + C[j][pi] - C[i][pi] - C[j][pj]
                    delta = a * dB + b * dA + dA * dB + dC
                    if delta < -1e-9:
                        perm[i], perm[j] = pj, pi
                        a += dA
                        b += dB
                        c += dC
                        obj += delta
                        improved = True
                        break
                if improved:
                    break
        return perm, obj, a, b, c

    initial_perms = []

    try:
        from scipy.optimize import linear_sum_assignment
        have_scipy = True
    except ImportError:
        have_scipy = False

    if have_scipy:
        import numpy as np
        def add_lap(mat):
            arr = np.array(mat)
            row_ind, col_ind = linear_sum_assignment(arr)
            perm = [0] * n
            for r, c in zip(row_ind, col_ind):
                perm[r] = int(c)
            initial_perms.append(perm)

        add_lap(C)
        add_lap(A)
        add_lap(B)
        add_lap([[A[i][j] + B[i][j] for j in range(n)] for i in range(n)])
        add_lap([[A[i][j] + C[i][j] for j in range(n)] for i in range(n)])
        add_lap([[B[i][j] + C[i][j] for j in range(n)] for i in range(n)])
        add_lap([[A[i][j] + B[i][j] + C[i][j] for j in range(n)] for i in range(n)])
    else:
        def greedy_lap(mat):
            perm = [-1] * n
            used = set()
            for i in range(n):
                min_val = float('inf')
                min_j = -1
                for j in range(n):
                    if j not in used and mat[i][j] < min_val:
                        min_val = mat[i][j]
                        min_j = j
                perm[i] = min_j
                used.add(min_j)
            return perm
        initial_perms.append(greedy_lap(C))
        initial_perms.append(greedy_lap(A))
        initial_perms.append(greedy_lap(B))
        initial_perms.append(greedy_lap([[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]))
        initial_perms.append(greedy_lap([[A[i][j] + C[i][j] for j in range(n)] for i in range(n)]))
        initial_perms.append(greedy_lap([[B[i][j] + C[i][j] for j in range(n)] for i in range(n)]))
        initial_perms.append(greedy_lap([[A[i][j] + B[i][j] + C[i][j] for j in range(n)] for i in range(n)]))

    for _ in range(5):
        perm = list(range(n))
        random.shuffle(perm)
        initial_perms.append(perm)

    best_perm = None
    best_obj = float('inf')
    best_abc = None

    for perm in initial_perms:
        perm2 = [int(p) for p in perm]
        perm2, obj, a, b, c = two_opt(perm2)
        if obj < best_obj:
            best_obj = float(obj)
            best_perm = perm2[:]
            best_abc = (float(a), float(b), float(c))
            if logger:
                logger.log_solution(best_obj, {"objective_value": best_obj, "assignment": best_perm})

    def run_ils(time_budget, best_perm, best_obj, best_abc):
        start = time.time()
        while time.time() - start < time_budget:
            new_perm = best_perm[:]
            num_swaps = random.randint(1, 3)
            for _ in range(num_swaps):
                i, j = random.sample(range(n), 2)
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            new_perm, new_obj, na, nb, nc = two_opt(new_perm)
            if new_obj < best_obj - 1e-9:
                best_obj = float(new_obj)
                best_perm = [int(p) for p in new_perm]
                best_abc = (float(na), float(nb), float(nc))
                if logger:
                    logger.log_solution(best_obj, {"objective_value": best_obj, "assignment": best_perm})
        return best_perm, best_obj, best_abc

    ils_time = min(args.time_limit * 0.4, max(1.0, args.time_limit - 5))
    if ils_time > 0:
        best_perm, best_obj, best_abc = run_ils(ils_time, best_perm, best_obj, best_abc)

    use_gurobi = (n <= 250)
    if use_gurobi:
        remaining_time = max(1.0, args.time_limit - (time.time() - start_time) - 2)
        if remaining_time > 2:
            try:
                import gurobipy as gp
                m = gp.Model()
                m.Params.OutputFlag = 0
                m.Params.Seed = 0
                m.Params.MIPGap = 1e-4
                m.Params.NumericFocus = 0
                m.Params.Threads = 1
                m.Params.TimeLimit = remaining_time
                m.Params.NonConvex = 2

                x = m.addVars(n, n, vtype=gp.GRB.BINARY, name="x")
                for i in range(n):
                    m.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1)
                for j in range(n):
                    m.addConstr(gp.quicksum(x[i, j] for i in range(n)) == 1)

                a_min = sum(min(row) for row in A)
                a_max = sum(max(row) for row in A)
                b_min = sum(min(row) for row in B)
                b_max = sum(max(row) for row in B)
                a = m.addVar(lb=a_min, ub=a_max, name="a")
                b = m.addVar(lb=b_min, ub=b_max, name="b")
                m.addConstr(a == gp.quicksum(A[i][j] * x[i, j] for i in range(n) for j in range(n)))
                m.addConstr(b == gp.quicksum(B[i][j] * x[i, j] for i in range(n) for j in range(n)))

                obj = a * b + gp.quicksum(C[i][j] * x[i, j] for i in range(n) for j in range(n))
                m.setObjective(obj, gp.GRB.MINIMIZE)

                if best_perm is not None:
                    for i in range(n):
                        for j in range(n):
                            x[i, j].Start = 1.0 if best_perm[i] == j else 0.0
                    a.Start = best_abc[0]
                    b.Start = best_abc[1]

                m.optimize()

                if m.SolCount > 0:
                    perm = []
                    for i in range(n):
                        for j in range(n):
                            if x[i, j].X > 0.5:
                                perm.append(int(j))
                                break
                    obj_val, _, _, _ = compute_obj(perm)
                    if obj_val < best_obj - 1e-9:
                        best_obj = float(obj_val)
                        best_perm = perm[:]
                        if logger:
                            logger.log_solution(best_obj, {"objective_value": best_obj, "assignment": best_perm})
            except Exception as e:
                sys.stderr.write(f"Gurobi failed: {e}\n")

    if not use_gurobi:
        remaining = max(0.0, args.time_limit - (time.time() - start_time) - 1)
        if remaining > 1:
            best_perm, best_obj, best_abc = run_ils(remaining, best_perm, best_obj, best_abc)

    sol = {
        "objective_value": float(best_obj),
        "assignment": [int(p) for p in best_perm]
    }
    with open(args.solution_path, 'w') as f:
        json.dump(sol, f)
    if logger:
        logger.log_solution(float(best_obj), sol)

if __name__ == "__main__":
    main()