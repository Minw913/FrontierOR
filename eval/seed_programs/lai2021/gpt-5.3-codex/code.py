import argparse
import json
import time
import random
from typing import List, Tuple

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_distance_matrix(N: int, upper: List[float]) -> List[List[float]]:
    d = [[0.0] * N for _ in range(N)]
    idx = 0
    for i in range(N - 1):
        row = d[i]
        for j in range(i + 1, N):
            v = float(upper[idx])
            idx += 1
            row[j] = v
            d[j][i] = v
    return d


def objective_from_groups(groups: List[set], dist: List[List[float]]) -> float:
    obj = 0.0
    for members in groups:
        mem = list(members)
        k = len(mem)
        for a in range(k - 1):
            i = mem[a]
            di = dist[i]
            for b in range(a + 1, k):
                obj += di[mem[b]]
    return obj


def random_feasible_assignment(
    N: int, m: int, L: List[int], U: List[int], rng: random.Random
) -> Tuple[List[int], List[set]]:
    elements = list(range(N))
    rng.shuffle(elements)

    assign = [-1] * N
    groups = [set() for _ in range(m)]
    ptr = 0

    for g in range(m):
        for _ in range(L[g]):
            i = elements[ptr]
            ptr += 1
            assign[i] = g
            groups[g].add(i)

    for t in range(ptr, N):
        i = elements[t]
        candidates = [g for g in range(m) if len(groups[g]) < U[g]]
        g = rng.choice(candidates)
        assign[i] = g
        groups[g].add(i)

    return assign, groups


def build_gain(
    N: int, m: int, groups: List[set], dist: List[List[float]]
) -> List[List[float]]:
    gain = [[0.0] * m for _ in range(N)]
    for g in range(m):
        mem = list(groups[g])
        for i in mem:
            s = 0.0
            di = dist[i]
            for j in mem:
                s += di[j]
            gain[i][g] = s
        # For i not in group g, compute sum to group members
        # done by full loop for simplicity
        for i in range(N):
            if i in groups[g]:
                continue
            di = dist[i]
            s = 0.0
            for j in mem:
                s += di[j]
            gain[i][g] = s
    return gain


def local_search(
    assign: List[int],
    groups: List[set],
    dist: List[List[float]],
    L: List[int],
    U: List[int],
    time_end: float,
) -> Tuple[List[int], List[set], float]:
    N = len(assign)
    m = len(groups)
    sizes = [len(groups[g]) for g in range(m)]
    gain = build_gain(N, m, groups, dist)
    obj = objective_from_groups(groups, dist)

    while time.time() < time_end:
        best_delta = 1e-12
        best_op = None

        # Best single move
        for i in range(N):
            a = assign[i]
            if sizes[a] <= L[a]:
                continue
            gi = gain[i]
            base = gi[a]
            for b in range(m):
                if b == a or sizes[b] >= U[b]:
                    continue
                delta = gi[b] - base
                if delta > best_delta:
                    best_delta = delta
                    best_op = ("move", i, a, b)

        # Best swap
        # Keep full search for quality; may be expensive for very large N.
        for i in range(N - 1):
            a = assign[i]
            gi = gain[i]
            for j in range(i + 1, N):
                b = assign[j]
                if a == b:
                    continue
                dij = dist[i][j]
                delta = (gi[b] - dij) + (gain[j][a] - dij) - gi[a] - gain[j][b]
                if delta > best_delta:
                    best_delta = delta
                    best_op = ("swap", i, j, a, b)

        if best_op is None:
            break

        if best_op[0] == "move":
            _, i, a, b = best_op
            groups[a].remove(i)
            groups[b].add(i)
            assign[i] = b
            sizes[a] -= 1
            sizes[b] += 1

            for k in range(N):
                if k == i:
                    continue
                dik = dist[k][i]
                gain[k][a] -= dik
                gain[k][b] += dik

            obj += best_delta

        else:
            _, i, j, a, b = best_op
            dij = dist[i][j]

            groups[a].remove(i)
            groups[b].remove(j)
            groups[a].add(j)
            groups[b].add(i)
            assign[i], assign[j] = b, a

            for k in range(N):
                if k == i or k == j:
                    continue
                dki = dist[k][i]
                dkj = dist[k][j]
                gain[k][a] += -dki + dkj
                gain[k][b] += -dkj + dki

            gain[i][a] += dij
            gain[i][b] -= dij
            gain[j][a] -= dij
            gain[j][b] += dij

            obj += best_delta

    return assign, groups, obj


def solve_with_gurobi(
    N: int,
    m: int,
    L: List[int],
    U: List[int],
    dist: List[List[float]],
    init_assign: List[int],
    time_limit: float,
    logger: SolutionLogger = None,
    incumbent_start: float = None,
):
    model = gp.Model("max_diversity_partition")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.0, time_limit)
    model.Params.NonConvex = 2
    model.Params.MIPFocus = 1

    x = model.addVars(N, m, vtype=GRB.BINARY, name="x")

    for i in range(N):
        model.addConstr(gp.quicksum(x[i, g] for g in range(m)) == 1)

    for g in range(m):
        model.addConstr(gp.quicksum(x[i, g] for i in range(N)) >= L[g])
        model.addConstr(gp.quicksum(x[i, g] for i in range(N)) <= U[g])

    obj = gp.QuadExpr()
    for g in range(m):
        for i in range(N - 1):
            xi = x[i, g]
            di = dist[i]
            for j in range(i + 1, N):
                dij = di[j]
                if dij != 0.0:
                    obj += dij * xi * x[j, g]
    model.setObjective(obj, GRB.MAXIMIZE)

    if init_assign is not None:
        for i in range(N):
            gi = init_assign[i]
            for g in range(m):
                x[i, g].Start = 1.0 if g == gi else 0.0

    best_logged = [-1e100 if incumbent_start is None else float(incumbent_start)]

    def cb(model_, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            val = model_.cbGet(GRB.Callback.MIPSOL_OBJ)
            if val > best_logged[0] + 1e-9:
                best_logged[0] = float(val)
                logger.log(float(val))

    model.optimize(cb)

    if model.SolCount > 0:
        assign = [-1] * N
        for i in range(N):
            for g in range(m):
                if x[i, g].X > 0.5:
                    assign[i] = g
                    break
            if assign[i] < 0:
                assign[i] = max(range(m), key=lambda gg: x[i, gg].X)

        groups = [set() for _ in range(m)]
        for i, g in enumerate(assign):
            groups[g].add(i)
        obj_val = objective_from_groups(groups, dist)
        return assign, groups, obj_val

    return None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    start = time.time()
    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    N = int(inst["N"])
    m = int(inst["m"])
    caps = inst["group_capacities"]
    L = [int(c["L_g"]) for c in caps]
    U = [int(c["U_g"]) for c in caps]
    upper = inst["distances_upper_triangular"]

    # Feasibility quick check
    if sum(L) > N or sum(U) < N:
        sol = {"objective_value": 0.0, "assignment": {}, "groups": {str(g): [] for g in range(m)}}
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    dist = build_distance_matrix(N, upper)

    # Heuristic phase
    rng = random.Random(0)
    best_assign, best_groups = random_feasible_assignment(N, m, L, U, rng)
    best_obj = objective_from_groups(best_groups, dist)

    heuristic_budget = min(max(1.0, 0.25 * args.time_limit), 8.0)
    h_end = min(start + heuristic_budget, start + args.time_limit - 0.1)

    # A few restarts + local search
    tries = 0
    while time.time() < h_end and tries < 20:
        tries += 1
        a, g = random_feasible_assignment(N, m, L, U, rng)
        a, g, o = local_search(a, g, dist, L, U, h_end)
        if o > best_obj + 1e-9:
            best_obj = o
            best_assign = a[:]
            best_groups = [set(s) for s in g]

    if logger:
        logger.log(float(best_obj))

    # Exact/improvement phase with remaining time
    remaining = args.time_limit - (time.time() - start)
    if remaining > 0.05:
        a2, g2, o2 = solve_with_gurobi(
            N, m, L, U, dist, best_assign, remaining, logger=logger, incumbent_start=best_obj
        )
        if a2 is not None and o2 is not None and o2 > best_obj + 1e-9:
            best_assign, best_groups, best_obj = a2, g2, o2

    assignment_out = {str(i): int(best_assign[i]) for i in range(N)}
    groups_out = {str(g): sorted(int(i) for i in best_groups[g]) for g in range(m)}

    sol = {
        "objective_value": float(best_obj),
        "assignment": assignment_out,
        "groups": groups_out,
    }

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()