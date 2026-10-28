import json
import argparse
import time
import random
import numpy as np
from itertools import permutations
from scipy.optimize import linear_sum_assignment


def compute_objective(A, B, C, assignment, n):
    """Compute the objective: (sum A) * (sum B) + sum C for the given assignment."""
    sa = sum(A[i][assignment[i]] for i in range(n))
    sb = sum(B[i][assignment[i]] for i in range(n))
    sc = sum(C[i][assignment[i]] for i in range(n))
    return sa * sb + sc


def local_search(A, B, C, assignment, n, deadline):
    """2-opt local search: try all pairwise swaps until no improvement or deadline."""
    assignment = list(assignment)
    sa = sum(A[i][assignment[i]] for i in range(n))
    sb = sum(B[i][assignment[i]] for i in range(n))
    sc = sum(C[i][assignment[i]] for i in range(n))

    best_obj = sa * sb + sc
    best_assignment = list(assignment)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                pi, pj = assignment[i], assignment[j]

                da = A[i][pj] + A[j][pi] - A[i][pi] - A[j][pj]
                db = B[i][pj] + B[j][pi] - B[i][pi] - B[j][pj]
                dc = C[i][pj] + C[j][pi] - C[i][pi] - C[j][pj]

                new_sa = sa + da
                new_sb = sb + db
                new_sc = sc + dc
                new_obj = new_sa * new_sb + new_sc

                if new_obj < best_obj:
                    best_obj = new_obj
                    assignment[i], assignment[j] = pj, pi
                    sa, sb, sc = new_sa, new_sb, new_sc
                    best_assignment = list(assignment)
                    improved = True

    return best_assignment, best_obj


def solve_with_gurobi(A, B, C, n, time_limit, warm_start=None):
    """Solve using Gurobi non-convex MIQP formulation."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None, None

    try:
        model = gp.Model()
        model.Params.NonConvex = 2
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = time_limit
        model.Params.OutputFlag = 0

        x = {}
        for i in range(n):
            for j in range(n):
                x[i, j] = model.addVar(vtype=GRB.BINARY)

        # Compute tight bounds for auxiliary variables
        A_np = np.array(A, dtype=np.float64)
        B_np = np.array(B, dtype=np.float64)
        C_np = np.array(C, dtype=np.float64)

        _, col_a_min = linear_sum_assignment(A_np)
        sa_lb = sum(A[i][col_a_min[i]] for i in range(n))
        _, col_a_max = linear_sum_assignment(-A_np)
        sa_ub = sum(A[i][col_a_max[i]] for i in range(n))

        _, col_b_min = linear_sum_assignment(B_np)
        sb_lb = sum(B[i][col_b_min[i]] for i in range(n))
        _, col_b_max = linear_sum_assignment(-B_np)
        sb_ub = sum(B[i][col_b_max[i]] for i in range(n))

        _, col_c_min = linear_sum_assignment(C_np)
        sc_lb = sum(C[i][col_c_min[i]] for i in range(n))
        _, col_c_max = linear_sum_assignment(-C_np)
        sc_ub = sum(C[i][col_c_max[i]] for i in range(n))

        sa_var = model.addVar(lb=sa_lb, ub=sa_ub)
        sb_var = model.addVar(lb=sb_lb, ub=sb_ub)
        sc_var = model.addVar(lb=sc_lb, ub=sc_ub)

        # Assignment constraints
        for i in range(n):
            model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1)
        for j in range(n):
            model.addConstr(gp.quicksum(x[i, j] for i in range(n)) == 1)

        # Linking constraints
        model.addConstr(sa_var == gp.quicksum(A[i][j] * x[i, j] for i in range(n) for j in range(n)))
        model.addConstr(sb_var == gp.quicksum(B[i][j] * x[i, j] for i in range(n) for j in range(n)))
        model.addConstr(sc_var == gp.quicksum(C[i][j] * x[i, j] for i in range(n) for j in range(n)))

        # Quadratic objective: sa * sb + sc
        model.setObjective(sa_var * sb_var + sc_var, GRB.MINIMIZE)

        # Warm start from heuristic solution
        if warm_start is not None:
            for i in range(n):
                for j in range(n):
                    x[i, j].Start = 1.0 if j == warm_start[i] else 0.0

        model.optimize()

        if model.SolCount > 0:
            assignment = [0] * n
            for i in range(n):
                for j in range(n):
                    if x[i, j].X > 0.5:
                        assignment[i] = j
                        break
            obj_val = compute_objective(A, B, C, assignment, n)
            return assignment, obj_val
    except Exception:
        pass

    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()
    deadline = start_time + args.time_limit

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    A = instance['A']
    B = instance['B']
    C = instance['C']

    best_assignment = None
    best_obj = float('inf')

    def update_best(assignment, obj):
        nonlocal best_assignment, best_obj
        if obj < best_obj:
            best_obj = obj
            best_assignment = list(assignment)
            if logger:
                sol = {"objective_value": float(best_obj), "assignment": best_assignment}
                logger.log_solution(best_obj, sol)

    if n <= 10:
        # Exact enumeration for small instances
        for perm in permutations(range(n)):
            sa = sum(A[i][perm[i]] for i in range(n))
            sb = sum(B[i][perm[i]] for i in range(n))
            sc = sum(C[i][perm[i]] for i in range(n))
            obj = sa * sb + sc
            if obj < best_obj:
                best_obj = obj
                best_assignment = list(perm)
                if logger:
                    sol = {"objective_value": float(best_obj), "assignment": best_assignment}
                    logger.log_solution(best_obj, sol)
    else:
        # Heuristic approach for larger instances
        A_np = np.array(A, dtype=np.float64)
        B_np = np.array(B, dtype=np.float64)
        C_np = np.array(C, dtype=np.float64)

        # Generate diverse initial solutions using Hungarian algorithm
        cost_matrices = []
        cost_matrices.append(C_np)
        cost_matrices.append(A_np + B_np)
        cost_matrices.append(A_np)
        cost_matrices.append(B_np)
        cost_matrices.append(A_np * B_np)

        for alpha in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]:
            for beta in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]:
                for gamma in [0.0, 0.5, 1.0]:
                    if alpha == 0 and beta == 0 and gamma == 0:
                        continue
                    cost_matrices.append(alpha * A_np + beta * B_np + gamma * C_np)

        initial_solutions = []
        for cm in cost_matrices:
            if time.time() >= deadline:
                break
            try:
                _, col_ind = linear_sum_assignment(cm)
                assignment = [int(x) for x in col_ind]
                obj = compute_objective(A, B, C, assignment, n)
                initial_solutions.append((obj, assignment))
                update_best(assignment, obj)
            except Exception:
                pass

        # Sort by objective and deduplicate
        initial_solutions.sort(key=lambda x: x[0])
        seen = set()
        unique_solutions = []
        for obj, assignment in initial_solutions:
            key = tuple(assignment)
            if key not in seen:
                seen.add(key)
                unique_solutions.append((obj, assignment))

        # Local search from top initial solutions
        for idx in range(min(10, len(unique_solutions))):
            if time.time() >= deadline - 2:
                break
            _, init_assign = unique_solutions[idx]
            res_assign, res_obj = local_search(A, B, C, init_assign, n, deadline)
            update_best(res_assign, res_obj)

        # Iterated Local Search for diversification
        random.seed(42)
        no_improve = 0
        max_no_improve = 100
        while time.time() < deadline - 2 and no_improve < max_no_improve:
            perturbed = list(best_assignment)
            num_swaps = random.randint(2, max(2, n // 5))
            for _ in range(num_swaps):
                i, j = random.sample(range(n), 2)
                perturbed[i], perturbed[j] = perturbed[j], perturbed[i]

            res_assign, res_obj = local_search(A, B, C, perturbed, n, deadline)
            if res_obj < best_obj:
                no_improve = 0
                update_best(res_assign, res_obj)
            else:
                no_improve += 1

        # Try Gurobi MIQP if time permits and problem size is manageable
        remaining = deadline - time.time()
        if remaining > 5 and n <= 50:
            g_assign, g_obj = solve_with_gurobi(A, B, C, n, remaining, best_assignment)
            if g_assign is not None:
                update_best(g_assign, g_obj)

    # Write final solution
    solution = {
        "objective_value": float(best_obj),
        "assignment": best_assignment
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()