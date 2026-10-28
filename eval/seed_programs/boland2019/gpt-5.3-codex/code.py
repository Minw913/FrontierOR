import argparse
import json
import time
from typing import Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def read_instance(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def compute_abs_bounds(data: dict) -> Tuple[int, int]:
    n_f = data["num_facilities"]
    n_c = data["num_customers"]

    fc1 = data["fixed_costs_obj1"]
    fc2 = data["fixed_costs_obj2"]
    ac1 = data["assignment_costs_obj1"]
    ac2 = data["assignment_costs_obj2"]

    b1_fixed = sum(abs(v) for v in fc1)
    b2_fixed = sum(abs(v) for v in fc2)

    b1_assign = 0
    b2_assign = 0
    for j in range(n_c):
        b1_assign += max(abs(ac1[i][j]) for i in range(n_f))
        b2_assign += max(abs(ac2[i][j]) for i in range(n_f))

    return b1_fixed + b1_assign, b2_fixed + b2_assign


def extract_solution(
    n_f: int, n_c: int, x: Dict[Tuple[int, int], gp.Var], y: Dict[int, gp.Var], z1_val: int, z2_val: int
) -> dict:
    open_facilities = [i for i in range(n_f) if y[i].X > 0.5]
    assignments = {}
    for j in range(n_c):
        assigned_i = None
        for i in range(n_f):
            if x[i, j].X > 0.5:
                assigned_i = i
                break
        if assigned_i is None:
            assigned_i = 0
        assignments[str(j)] = assigned_i

    return {
        "z1": int(z1_val),
        "z2": int(z2_val),
        "open_facilities": open_facilities,
        "assignments": assignments,
    }


def filter_nondominated(solutions: List[dict]) -> List[dict]:
    # Keep best solution per (z1, z2) first
    pair_to_sol = {}
    for s in solutions:
        pair_to_sol[(s["z1"], s["z2"])] = s
    uniq = list(pair_to_sol.values())

    # O(n^2) nondominance filter
    nd = []
    for i, a in enumerate(uniq):
        dominated = False
        for j, b in enumerate(uniq):
            if i == j:
                continue
            if (b["z1"] <= a["z1"] and b["z2"] <= a["z2"]) and (b["z1"] < a["z1"] or b["z2"] < a["z2"]):
                dominated = True
                break
        if not dominated:
            nd.append(a)

    nd.sort(key=lambda s: (s["z1"], s["z2"]))
    return nd


def build_output(nondom: List[dict]) -> dict:
    if not nondom:
        return {"objective_value": float("inf"), "pareto_front": [], "solutions": []}

    best = min(nondom, key=lambda s: 0.5 * s["z1"] + 0.5 * s["z2"])
    obj_val = 0.5 * best["z1"] + 0.5 * best["z2"]
    pareto = [[int(s["z1"]), int(s["z2"])] for s in nondom]

    return {
        "objective_value": float(obj_val),
        "pareto_front": pareto,
        "solutions": nondom,
    }


def solve(instance: dict, time_limit: int, logger: SolutionLogger = None) -> dict:
    start = time.time()

    n_f = instance["num_facilities"]
    n_c = instance["num_customers"]
    demands = instance["demands"]
    caps = instance["capacities"]
    fc1 = instance["fixed_costs_obj1"]
    fc2 = instance["fixed_costs_obj2"]
    ac1 = instance["assignment_costs_obj1"]
    ac2 = instance["assignment_costs_obj2"]

    bz1, _ = compute_abs_bounds(instance)
    # big-M for lexicographic (min z2 first, then z1)
    BIG = max(1, 2 * bz1 + 1)

    m = gp.Model("biobj_cflp")
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.OutputFlag = 0

    y = {i: m.addVar(vtype=GRB.BINARY, name=f"y_{i}") for i in range(n_f)}
    x = {(i, j): m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}") for i in range(n_f) for j in range(n_c)}

    # assignment exactly one
    for j in range(n_c):
        m.addConstr(gp.quicksum(x[i, j] for i in range(n_f)) == 1, name=f"assign_{j}")

    # capacities + linking
    for i in range(n_f):
        m.addConstr(
            gp.quicksum(demands[j] * x[i, j] for j in range(n_c)) <= caps[i] * y[i],
            name=f"cap_{i}",
        )
        for j in range(n_c):
            m.addConstr(x[i, j] <= y[i], name=f"link_{i}_{j}")

    z1_expr = gp.quicksum(fc1[i] * y[i] for i in range(n_f)) + gp.quicksum(
        ac1[i][j] * x[i, j] for i in range(n_f) for j in range(n_c)
    )
    z2_expr = gp.quicksum(fc2[i] * y[i] for i in range(n_f)) + gp.quicksum(
        ac2[i][j] * x[i, j] for i in range(n_f) for j in range(n_c)
    )

    # Epsilon constraint on z1, start loose
    eps_constr = m.addConstr(z1_expr <= bz1, name="eps_z1")

    # lexicographic by scalarization: minimize z2 * BIG + z1
    m.setObjective(z2_expr * BIG + z1_expr, GRB.MINIMIZE)

    collected = []
    seen_pairs = set()
    best_combined = float("inf")

    while True:
        elapsed = time.time() - start
        remain = time_limit - elapsed
        if remain <= 0:
            break

        m.Params.TimeLimit = max(0.01, remain)
        m.optimize()

        status = m.Status
        if status == GRB.OPTIMAL and m.SolCount > 0:
            z1_val = int(round(z1_expr.getValue()))
            z2_val = int(round(z2_expr.getValue()))
            pair = (z1_val, z2_val)

            if pair not in seen_pairs:
                seen_pairs.add(pair)
                sol = extract_solution(n_f, n_c, x, y, z1_val, z2_val)
                collected.append(sol)

                comb = 0.5 * z1_val + 0.5 * z2_val
                if comb < best_combined:
                    best_combined = comb
                    if logger:
                        temp_nd = filter_nondominated(collected)
                        logger.log_solution(comb, build_output(temp_nd))

            # tighten epsilon: force strictly smaller z1
            next_eps = z1_val - 1
            eps_constr.RHS = next_eps
            if next_eps < -bz1 - 5:
                break
        else:
            # Infeasible means no more points for tighter epsilon, or time limit/non-optimal -> stop.
            break

    nondom = filter_nondominated(collected)
    return build_output(nondom)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)
    result = solve(instance, args.time_limit, logger=logger)

    with open(args.solution_path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()