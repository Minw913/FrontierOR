import argparse
import json
import math
import time
from typing import List, Tuple, Set, Optional

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    return parser.parse_args()


def load_instance(path: str):
    with open(path, "r") as f:
        return json.load(f)


def now():
    return time.time()


def build_precomputed(instance):
    n = instance["num_nodes"]
    commodities = instance["commodities"]
    cnum = instance["num_commodities"]
    dist = instance["distances"]
    params = instance["parameters"]
    tau = float(params["tau"])
    chi = float(params["chi"])
    delta = float(params["delta"])

    fixed = [float(node["fixed_cost"]) for node in instance["nodes"]]

    origins = [int(commodities[c]["origin"]) for c in range(cnum)]
    dests = [int(commodities[c]["destination"]) for c in range(cnum)]
    flows = [float(commodities[c]["flow"]) for c in range(cnum)]

    # A[c][i] = flow*chi*d[o][i]
    # B[c][j] = flow*delta*d[j][d]
    # FT[c] = flow*tau
    A = [[0.0] * n for _ in range(cnum)]
    B = [[0.0] * n for _ in range(cnum)]
    FT = [0.0] * cnum

    for c in range(cnum):
        o = origins[c]
        d = dests[c]
        f = flows[c]
        FT[c] = f * tau
        row_o = dist[o]
        for i in range(n):
            A[c][i] = f * chi * row_o[i]
        for j in range(n):
            B[c][j] = f * delta * dist[j][d]

    return n, cnum, dist, fixed, origins, dests, flows, A, B, FT


def cost_cij(c: int, i: int, j: int, A, B, FT, dist):
    return A[c][i] + FT[c] * dist[i][j] + B[c][j]


def evaluate_set(
    open_set: List[int], cnum: int, A, B, FT, dist, fixed
) -> Tuple[float, List[float], List[Tuple[int, int]]]:
    fixed_cost = sum(fixed[i] for i in open_set)
    best_costs = [0.0] * cnum
    best_pairs = [(open_set[0], open_set[0])] * cnum

    if len(open_set) == 1:
        h = open_set[0]
        total_trans = 0.0
        for c in range(cnum):
            v = A[c][h] + B[c][h]  # dist[h][h] = 0
            best_costs[c] = v
            best_pairs[c] = (h, h)
            total_trans += v
        return fixed_cost + total_trans, best_costs, best_pairs

    total_trans = 0.0
    for c in range(cnum):
        best_v = float("inf")
        best_p = (open_set[0], open_set[0])
        for i in open_set:
            ai = A[c][i]
            row_i = dist[i]
            ft = FT[c]
            for j in open_set:
                v = ai + ft * row_i[j] + B[c][j]
                if v < best_v:
                    best_v = v
                    best_p = (i, j)
        best_costs[c] = best_v
        best_pairs[c] = best_p
        total_trans += best_v
    return fixed_cost + total_trans, best_costs, best_pairs


def evaluate_add(
    h: int,
    open_set: Set[int],
    current_obj: float,
    current_best_costs: List[float],
    current_best_pairs: List[Tuple[int, int]],
    cnum: int,
    A,
    B,
    FT,
    dist,
    fixed,
):
    S = list(open_set)
    new_costs = [0.0] * cnum
    new_pairs = [None] * cnum
    delta_trans = 0.0

    for c in range(cnum):
        best_h = A[c][h] + B[c][h]
        best_hp = (h, h)

        ah = A[c][h]
        bh = B[c][h]
        ftt = FT[c]
        for j in S:
            # (h, j)
            v1 = ah + ftt * dist[h][j] + B[c][j]
            if v1 < best_h:
                best_h = v1
                best_hp = (h, j)
            # (j, h)
            v2 = A[c][j] + ftt * dist[j][h] + bh
            if v2 < best_h:
                best_h = v2
                best_hp = (j, h)

        if best_h < current_best_costs[c]:
            new_costs[c] = best_h
            new_pairs[c] = best_hp
            delta_trans += best_h - current_best_costs[c]
        else:
            new_costs[c] = current_best_costs[c]
            new_pairs[c] = current_best_pairs[c]

    new_obj = current_obj + fixed[h] + delta_trans
    return new_obj, new_costs, new_pairs


def solve_heuristic(instance, time_limit, start_time, logger=None):
    n, cnum, dist, fixed, origins, dests, flows, A, B, FT = build_precomputed(instance)

    # Initial single-hub solution
    best_single_obj = float("inf")
    best_h = 0
    for h in range(n):
        s = fixed[h]
        for c in range(cnum):
            s += A[c][h] + B[c][h]
        if s < best_single_obj:
            best_single_obj = s
            best_h = h

    open_set = {best_h}
    current_obj = best_single_obj
    current_best_costs = [A[c][best_h] + B[c][best_h] for c in range(cnum)]
    current_best_pairs = [(best_h, best_h) for _ in range(cnum)]

    if logger:
        logger.log(current_obj)

    eps = 1e-9
    improved = True
    while improved and (now() - start_time) < time_limit - 0.05:
        improved = False
        best_move_obj = current_obj
        best_move_type = None
        best_move_h = None
        best_move_costs = None
        best_move_pairs = None

        # ADD move
        closed = [i for i in range(n) if i not in open_set]
        if len(closed) > 80:
            closed = sorted(closed, key=lambda x: fixed[x])[:80]

        for h in closed:
            if (now() - start_time) >= time_limit - 0.05:
                break
            obj_h, costs_h, pairs_h = evaluate_add(
                h,
                open_set,
                current_obj,
                current_best_costs,
                current_best_pairs,
                cnum,
                A,
                B,
                FT,
                dist,
                fixed,
            )
            if obj_h + eps < best_move_obj:
                best_move_obj = obj_h
                best_move_type = "add"
                best_move_h = h
                best_move_costs = costs_h
                best_move_pairs = pairs_h

        # DROP move
        if len(open_set) > 1 and (now() - start_time) < time_limit - 0.05:
            for h in list(open_set):
                if (now() - start_time) >= time_limit - 0.05:
                    break
                S2 = list(open_set - {h})
                obj_h, costs_h, pairs_h = evaluate_set(S2, cnum, A, B, FT, dist, fixed)
                if obj_h + eps < best_move_obj:
                    best_move_obj = obj_h
                    best_move_type = "drop"
                    best_move_h = h
                    best_move_costs = costs_h
                    best_move_pairs = pairs_h

        if best_move_type is not None and best_move_obj + eps < current_obj:
            if best_move_type == "add":
                open_set.add(best_move_h)
            else:
                open_set.remove(best_move_h)
            current_obj = best_move_obj
            current_best_costs = best_move_costs
            current_best_pairs = best_move_pairs
            improved = True
            if logger:
                logger.log(current_obj)

    return {
        "objective": current_obj,
        "open_set": sorted(open_set),
        "best_costs": current_best_costs,
        "best_pairs": current_best_pairs,
        "origins": origins,
        "dests": dests,
        "n": n,
        "cnum": cnum,
        "dist": dist,
        "fixed": fixed,
        "A": A,
        "B": B,
        "FT": FT,
    }


def run_mip_if_small(
    data,
    heuristic_sol,
    time_limit,
    start_time,
    logger=None,
):
    n = data["n"]
    cnum = data["cnum"]
    dist = data["dist"]
    fixed = data["fixed"]
    A = data["A"]
    B = data["B"]
    FT = data["FT"]

    remaining = time_limit - (now() - start_time)
    if remaining <= 1.0:
        return heuristic_sol

    # Size guard
    z_count = cnum * n * n
    if z_count > 250000:
        return heuristic_sol

    model = gp.Model("hub_location")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.5, remaining - 0.1)

    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    z = model.addVars(cnum, n, n, vtype=GRB.BINARY, name="z")

    model.setObjective(
        gp.quicksum(fixed[i] * x[i] for i in range(n))
        + gp.quicksum(
            (A[c][i] + FT[c] * dist[i][j] + B[c][j]) * z[c, i, j]
            for c in range(cnum)
            for i in range(n)
            for j in range(n)
        ),
        GRB.MINIMIZE,
    )

    for c in range(cnum):
        model.addConstr(gp.quicksum(z[c, i, j] for i in range(n) for j in range(n)) == 1)

    for c in range(cnum):
        for i in range(n):
            for j in range(n):
                model.addConstr(z[c, i, j] <= x[i])
                model.addConstr(z[c, i, j] <= x[j])

    # Warm start from heuristic
    h_open = set(heuristic_sol["open_set"])
    h_pairs = heuristic_sol["best_pairs"]
    for i in range(n):
        x[i].Start = 1.0 if i in h_open else 0.0
    for c in range(cnum):
        i, j = h_pairs[c]
        z[c, i, j].Start = 1.0

    best_logged = [heuristic_sol["objective"]]

    def cb(m, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < best_logged[0]:
                best_logged[0] = obj
                logger.log(obj)

    model.optimize(cb)

    if model.SolCount == 0:
        return heuristic_sol

    open_set = [i for i in range(n) if x[i].X > 0.5]
    if not open_set:
        return heuristic_sol

    best_pairs = [(open_set[0], open_set[0])] * cnum
    best_costs = [0.0] * cnum

    for c in range(cnum):
        sel = None
        for i in range(n):
            found = False
            for j in range(n):
                if z[c, i, j].X > 0.5:
                    sel = (i, j)
                    found = True
                    break
            if found:
                break
        if sel is None:
            # Fallback: take max z value
            maxv = -1.0
            arg = (0, 0)
            for i in range(n):
                for j in range(n):
                    v = z[c, i, j].X
                    if v > maxv:
                        maxv = v
                        arg = (i, j)
            sel = arg
        i, j = sel
        best_pairs[c] = (i, j)
        best_costs[c] = A[c][i] + FT[c] * dist[i][j] + B[c][j]

    obj = sum(fixed[i] for i in open_set) + sum(best_costs)
    if obj + 1e-9 < heuristic_sol["objective"]:
        if logger:
            logger.log(obj)
        out = dict(heuristic_sol)
        out["objective"] = obj
        out["open_set"] = sorted(open_set)
        out["best_pairs"] = best_pairs
        out["best_costs"] = best_costs
        return out
    return heuristic_sol


def build_output(instance, sol):
    commodities = instance["commodities"]
    assignments = []
    for c, (i, j) in enumerate(sol["best_pairs"]):
        if i == j:
            hub_edge = [int(i)]
        else:
            a, b = (int(i), int(j))
            hub_edge = [a, b] if a < b else [b, a]
        assignments.append(
            {
                "commodity": c,
                "origin": int(commodities[c]["origin"]),
                "destination": int(commodities[c]["destination"]),
                "hub_edge": hub_edge,
                "cost": float(sol["best_costs"][c]),
            }
        )

    return {
        "objective_value": float(sol["objective"]),
        "open_hubs": [int(i) for i in sorted(sol["open_set"])],
        "assignments": assignments,
    }


def main():
    args = parse_args()
    start_time = now()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = load_instance(args.instance_path)

    heuristic_sol = solve_heuristic(instance, args.time_limit, start_time, logger=logger)

    # Optional exact polishing for small instances
    polished_sol = run_mip_if_small(
        heuristic_sol, heuristic_sol, args.time_limit, start_time, logger=logger
    )

    solution = build_output(instance, polished_sol)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()