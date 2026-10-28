import argparse
import json
import time
import random
from typing import List, Tuple

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def evaluate_perm(perm: List[int], A, B, C) -> Tuple[float, float, float, float]:
    sa = 0.0
    sb = 0.0
    sc = 0.0
    for i, j in enumerate(perm):
        sa += A[i][j]
        sb += B[i][j]
        sc += C[i][j]
    obj = sa * sb + sc
    return sa, sb, sc, obj


def solve_linear_assignment(cost, time_limit: float) -> List[int]:
    n = len(cost)
    m = gp.Model("lap")
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    if time_limit is not None and time_limit > 0:
        m.Params.TimeLimit = time_limit

    x = m.addVars(n, n, vtype=GRB.BINARY, name="x")
    m.addConstrs((gp.quicksum(x[i, j] for j in range(n)) == 1 for i in range(n)), name="row")
    m.addConstrs((gp.quicksum(x[i, j] for i in range(n)) == 1 for j in range(n)), name="col")
    m.setObjective(gp.quicksum(cost[i][j] * x[i, j] for i in range(n) for j in range(n)), GRB.MINIMIZE)

    m.optimize()

    if m.SolCount == 0:
        # fallback random permutation
        perm = list(range(n))
        random.shuffle(perm)
        return perm

    perm = [-1] * n
    used = set()
    for i in range(n):
        best_j = max(range(n), key=lambda j: x[i, j].X)
        if best_j not in used:
            perm[i] = best_j
            used.add(best_j)

    # repair if any duplicates/missing due numerical edge case
    missing = [j for j in range(n) if j not in used]
    for i in range(n):
        if perm[i] == -1:
            perm[i] = missing.pop()

    return perm


def local_search_swap(perm, A, B, C, end_time):
    n = len(perm)
    sa, sb, sc, best_obj = evaluate_perm(perm, A, B, C)

    improved = True
    while improved and time.time() < end_time:
        improved = False
        for i in range(n - 1):
            if time.time() >= end_time:
                break
            j1 = perm[i]
            for k in range(i + 1, n):
                j2 = perm[k]

                dA = A[i][j2] + A[k][j1] - A[i][j1] - A[k][j2]
                dB = B[i][j2] + B[k][j1] - B[i][j1] - B[k][j2]
                dC = C[i][j2] + C[k][j1] - C[i][j1] - C[k][j2]

                new_obj = (sa + dA) * (sb + dB) + (sc + dC)
                if new_obj + 1e-12 < best_obj:
                    # first improvement
                    perm[i], perm[k] = perm[k], perm[i]
                    sa += dA
                    sb += dB
                    sc += dC
                    best_obj = new_obj
                    improved = True
                    break
            if improved:
                break

    return perm, (sa, sb, sc, best_obj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    def remaining():
        return args.time_limit - (time.time() - start_time)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n = inst["n"]
    A = inst["A"]
    B = inst["B"]
    C = inst["C"]

    random.seed(0)

    # ---------------- Heuristic phase ----------------
    best_perm = list(range(n))
    random.shuffle(best_perm)
    _, _, _, best_obj = evaluate_perm(best_perm, A, B, C)
    if logger:
        logger.log(float(best_obj))

    # Keep heuristic modest to leave time for MIQCP
    heur_budget = min(max(0.5, 0.25 * args.time_limit), max(0.0, args.time_limit - 0.5))
    heur_end = time.time() + max(0.0, heur_budget)

    # Initial LAP on C
    if remaining() > 0.1 and time.time() < heur_end:
        tl = max(0.05, min(remaining() * 0.2, heur_end - time.time()))
        perm_c = solve_linear_assignment(C, tl)
        _, _, _, obj_c = evaluate_perm(perm_c, A, B, C)
        if obj_c < best_obj:
            best_perm, best_obj = perm_c, obj_c
            if logger:
                logger.log(float(best_obj))

        # A few linearization rounds
        curr = perm_c[:]
        for _ in range(2):
            if time.time() >= heur_end or remaining() <= 0.1:
                break
            sa, sb, _, _ = evaluate_perm(curr, A, B, C)
            lin_cost = [[sb * A[i][j] + sa * B[i][j] + C[i][j] for j in range(n)] for i in range(n)]
            tl = max(0.05, min(remaining() * 0.15, heur_end - time.time()))
            nxt = solve_linear_assignment(lin_cost, tl)
            _, _, _, obj_nxt = evaluate_perm(nxt, A, B, C)
            curr = nxt
            if obj_nxt < best_obj:
                best_perm, best_obj = nxt, obj_nxt
                if logger:
                    logger.log(float(best_obj))

    # Local search
    if time.time() < heur_end:
        perm_ls = best_perm[:]
        perm_ls, (_, _, _, obj_ls) = local_search_swap(perm_ls, A, B, C, heur_end)
        if obj_ls < best_obj:
            best_perm, best_obj = perm_ls, obj_ls
            if logger:
                logger.log(float(best_obj))

    # ---------------- Exact MIQCP phase ----------------
    final_perm = best_perm[:]
    final_obj = best_obj

    if remaining() > 0.05:
        m = gp.Model("bilinear_assignment")
        m.Params.OutputFlag = 0
        m.Params.Threads = 1
        m.Params.NonConvex = 2
        m.Params.TimeLimit = max(0.01, remaining())

        x = m.addVars(n, n, vtype=GRB.BINARY, name="x")
        m.addConstrs((gp.quicksum(x[i, j] for j in range(n)) == 1 for i in range(n)), name="row")
        m.addConstrs((gp.quicksum(x[i, j] for i in range(n)) == 1 for j in range(n)), name="col")

        sa_min = sum(min(A[i]) for i in range(n))
        sa_max = sum(max(A[i]) for i in range(n))
        sb_min = sum(min(B[i]) for i in range(n))
        sb_max = sum(max(B[i]) for i in range(n))
        sc_min = sum(min(C[i]) for i in range(n))
        sc_max = sum(max(C[i]) for i in range(n))

        SA = m.addVar(lb=sa_min, ub=sa_max, vtype=GRB.CONTINUOUS, name="SA")
        SB = m.addVar(lb=sb_min, ub=sb_max, vtype=GRB.CONTINUOUS, name="SB")
        SC = m.addVar(lb=sc_min, ub=sc_max, vtype=GRB.CONTINUOUS, name="SC")

        z_lb = min(sa_min * sb_min, sa_min * sb_max, sa_max * sb_min, sa_max * sb_max)
        z_ub = max(sa_min * sb_min, sa_min * sb_max, sa_max * sb_min, sa_max * sb_max)
        Z = m.addVar(lb=z_lb, ub=z_ub, vtype=GRB.CONTINUOUS, name="Z")

        m.addConstr(SA == gp.quicksum(A[i][j] * x[i, j] for i in range(n) for j in range(n)), name="defSA")
        m.addConstr(SB == gp.quicksum(B[i][j] * x[i, j] for i in range(n) for j in range(n)), name="defSB")
        m.addConstr(SC == gp.quicksum(C[i][j] * x[i, j] for i in range(n) for j in range(n)), name="defSC")
        m.addQConstr(Z == SA * SB, name="bilin")

        m.setObjective(Z + SC, GRB.MINIMIZE)

        # MIP start from heuristic
        for i in range(n):
            for j in range(n):
                x[i, j].Start = 1.0 if best_perm[i] == j else 0.0

        cb_best = [float(best_obj)]

        def cb(model, where):
            if where == GRB.Callback.MIPSOL and logger is not None:
                try:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj + 1e-9 < cb_best[0]:
                        cb_best[0] = obj
                        logger.log(float(obj))
                except gp.GurobiError:
                    pass

        m.optimize(cb)

        if m.SolCount > 0:
            perm = [-1] * n
            used = set()
            for i in range(n):
                chosen = None
                for j in range(n):
                    if x[i, j].X > 0.5:
                        chosen = j
                        break
                if chosen is None:
                    chosen = max(range(n), key=lambda j: x[i, j].X)
                perm[i] = chosen
                used.add(chosen)

            # repair if needed
            missing = [j for j in range(n) if j not in used]
            if missing:
                seen = {}
                for i, j in enumerate(perm):
                    seen.setdefault(j, []).append(i)
                for j, rows in seen.items():
                    if len(rows) > 1:
                        for rr in rows[1:]:
                            perm[rr] = missing.pop()

            _, _, _, obj_perm = evaluate_perm(perm, A, B, C)
            if obj_perm < final_obj:
                final_obj = obj_perm
                final_perm = perm
                if logger:
                    logger.log(float(final_obj))

    out = {
        "objective_value": float(final_obj),
        "assignment": [int(v) for v in final_perm],
    }

    with open(args.solution_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()