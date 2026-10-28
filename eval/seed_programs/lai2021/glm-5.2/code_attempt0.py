import argparse
import json
import random
import sys
import time

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = None
    GRB = None

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None


def load_instance(path):
    with open(path, 'r') as f:
        data = json.load(f)
    N = data["N"]
    m = data["m"]
    caps = data["group_capacities"]
    L = [int(c["L_g"]) for c in caps]
    U = [int(c["U_g"]) for c in caps]
    upper = data["distances_upper_triangular"]
    D = [[0.0] * N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            val = float(upper[idx])
            D[i][j] = val
            D[j][i] = val
            idx += 1
    return N, m, L, U, D


def random_feasible_assignment(N, m, L, U):
    elems = list(range(N))
    random.shuffle(elems)
    assignment = [-1] * N
    sizes = [0] * m
    e = 0
    for g in range(m):
        for _ in range(L[g]):
            assignment[elems[e]] = g
            sizes[g] += 1
            e += 1
    while e < N:
        g = random.randint(0, m - 1)
        while sizes[g] >= U[g]:
            g = (g + 1) % m
        assignment[elems[e]] = g
        sizes[g] += 1
        e += 1
    return assignment


def local_search(assignment, D, N, m, L, U, start_time, time_limit):
    groups = [[] for _ in range(m)]
    for i, g in enumerate(assignment):
        groups[g].append(i)

    dist_sum = [[0.0] * m for _ in range(N)]
    for i in range(N):
        for g in range(m):
            s = 0.0
            for j in groups[g]:
                s += D[i][j]
            dist_sum[i][g] = s

    group_obj = [0.0] * m
    for g in range(m):
        s = 0.0
        for i in groups[g]:
            s += dist_sum[i][g]
        group_obj[g] = s / 2.0
    current_obj = sum(group_obj)

    def update_dist_sum_for_element(i):
        for g in range(m):
            s = 0.0
            for k in groups[g]:
                s += D[i][k]
            dist_sum[i][g] = s

    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False

        # Move elements
        for i in range(N):
            if i % 100 == 0 and time.time() - start_time > time_limit:
                return assignment, current_obj, groups
            g_curr = assignment[i]
            if len(groups[g_curr]) <= L[g_curr]:
                continue
            best_delta = 0.0
            best_g = -1
            for g_new in range(m):
                if g_new == g_curr:
                    continue
                if len(groups[g_new]) >= U[g_new]:
                    continue
                delta = dist_sum[i][g_new] - dist_sum[i][g_curr]
                if delta > best_delta + 1e-9:
                    best_delta = delta
                    best_g = g_new
            if best_g != -1:
                groups[g_curr].remove(i)
                groups[best_g].append(i)
                assignment[i] = best_g
                group_obj[g_curr] -= dist_sum[i][g_curr]
                group_obj[best_g] += dist_sum[i][best_g]
                current_obj += best_delta
                for k in range(N):
                    if k == i:
                        continue
                    dist_sum[k][g_curr] -= D[k][i]
                    dist_sum[k][best_g] += D[k][i]
                update_dist_sum_for_element(i)
                improved = True

        # Swap elements between different groups
        for i in range(N):
            if i % 50 == 0 and time.time() - start_time > time_limit:
                return assignment, current_obj, groups
            gi = assignment[i]
            for j in range(i + 1, N):
                gj = assignment[j]
                if gi == gj:
                    continue
                delta = (dist_sum[i][gj] - D[i][j]) + (dist_sum[j][gi] - D[i][j]) - dist_sum[i][gi] - dist_sum[j][gj]
                if delta > 1e-9:
                    groups[gi].remove(i)
                    groups[gi].append(j)
                    groups[gj].remove(j)
                    groups[gj].append(i)
                    assignment[i] = gj
                    assignment[j] = gi
                    group_obj[gi] += (dist_sum[j][gi] - D[i][j]) - dist_sum[i][gi]
                    group_obj[gj] += (dist_sum[i][gj] - D[i][j]) - dist_sum[j][gj]
                    current_obj += delta
                    for k in range(N):
                        if k == i or k == j:
                            continue
                        dist_sum[k][gi] += D[k][j] - D[k][i]
                        dist_sum[k][gj] += D[k][i] - D[k][j]
                    update_dist_sum_for_element(i)
                    update_dist_sum_for_element(j)
                    improved = True

    return assignment, current_obj, groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    if SolutionLogger is not None and args.log_path:
        logger = SolutionLogger(args.log_path, sense="maximize")
    else:
        logger = None

    random.seed(0)

    N, m, L, U, D = load_instance(args.instance_path)
    start_time = time.time()
    time_limit = float(args.time_limit)

    best_obj = -1.0
    best_assignment = None

    # Heuristic phase
    heuristic_time = min(10.0, time_limit * 0.2) if gp is not None and N <= 300 else time_limit
    while time.time() - start_time < heuristic_time:
        assign = random_feasible_assignment(N, m, L, U)
        assign, obj, groups = local_search(assign, D, N, m, L, U, start_time, heuristic_time)
        if obj > best_obj + 1e-9:
            best_obj = obj
            best_assignment = assign[:]
            if logger:
                groups_dict = {g: [] for g in range(m)}
                for i, g in enumerate(assign):
                    groups_dict[g].append(i)
                for g in groups_dict:
                    groups_dict[g].sort()
                sol_dict = {
                    "objective_value": float(obj),
                    "assignment": {str(i): int(assign[i]) for i in range(N)},
                    "groups": {str(g): groups_dict[g] for g in range(m)}
                }
                logger.log_solution(float(obj), sol_dict)

    # MIP phase with Gurobi
    if gp is not None and GRB is not None and N <= 300:
        remaining = time_limit - (time.time() - start_time)
        if remaining > 1.0:
            model = gp.Model("max_diversity")
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.NonConvex = 2
            model.Params.TimeLimit = remaining

            y = model.addVars(N, m, vtype=GRB.BINARY, name="y")
            model.addConstrs((y.sum(i, '*') == 1 for i in range(N)), name="assign")
            for g in range(m):
                model.addConstr(L[g] <= gp.quicksum(y[i, g] for i in range(N)), name=f"lower_{g}")
                model.addConstr(gp.quicksum(y[i, g] for i in range(N)) <= U[g], name=f"upper_{g}")

            obj = gp.QuadExpr()
            for i in range(N):
                for j in range(i + 1, N):
                    d = D[i][j]
                    for g in range(m):
                        obj += d * y[i, g] * y[j, g]
            model.setObjective(obj, GRB.MAXIMIZE)

            if best_assignment is not None:
                for i in range(N):
                    for g in range(m):
                        y[i, g].Start = 1 if best_assignment[i] == g else 0

            model._vars = y
            model._N = N
            model._m = m
            best_logged = [best_obj]

            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj_val = model.cbGetSolution(GRB.Callback.MIPSOL_OBJ)
                    if obj_val > best_logged[0] + 1e-9:
                        best_logged[0] = obj_val
                        sol = model.cbGetSolution(model._vars)
                        assignment = {}
                        for i in range(model._N):
                            for g in range(model._m):
                                if sol[i, g] > 0.5:
                                    assignment[i] = g
                                    break
                        groups_dict = {g: [] for g in range(model._m)}
                        for i, g in assignment.items():
                            groups_dict[g].append(i)
                        for g in groups_dict:
                            groups_dict[g].sort()
                        sol_dict = {
                            "objective_value": float(obj_val),
                            "assignment": {str(i): g for i, g in assignment.items()},
                            "groups": {str(g): groups_dict[g] for g in range(model._m)}
                        }
                        if logger:
                            logger.log_solution(float(obj_val), sol_dict)

            model.optimize(callback)
            if model.SolCount > 0:
                obj_val = model.ObjVal
                if obj_val > best_obj + 1e-9:
                    best_obj = obj_val
                    assign = [-1] * N
                    for i in range(N):
                        for g in range(m):
                            if y[i, g].X > 0.5:
                                assign[i] = g
                                break
                    best_assignment = assign

    # Fallback if nothing found
    if best_assignment is None:
        best_assignment = random_feasible_assignment(N, m, L, U)

    # Final solution formatting
    groups_dict = {g: [] for g in range(m)}
    for i, g in enumerate(best_assignment):
        groups_dict[g].append(i)
    for g in groups_dict:
        groups_dict[g].sort()

    final_obj = 0.0
    for g in range(m):
        elems = groups_dict[g]
        for i in range(len(elems)):
            for j in range(i + 1, len(elems)):
                final_obj += D[elems[i]][elems[j]]

    solution = {
        "objective_value": float(final_obj),
        "assignment": {str(i): int(best_assignment[i]) for i in range(N)},
        "groups": {str(g): groups_dict[g] for g in range(m)}
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()